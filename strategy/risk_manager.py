"""
Risk management engine enforcing all rules from the strategy:

  風控規則:
  - 每日最多 1 筆         → max_trades_per_day = 1
  - 1 筆虧損當晚停手      → stop_after_daily_loss = 1
  - 連續 2 晚虧損停手 3 天 → consecutive_loss_days_limit = 2, pause = 3 days
  - 月度盈利 +10% 即停手   → monthly_profit_target_pct = 0.10

  策略 A 保守版 (生存模式):
  - 資金 < 21 USDT 強制切換
  - 連續虧損 2 筆後強制切換
  - FOMC / major-news week → manual override
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DailyRecord:
    date: str           # ISO date "YYYY-MM-DD"
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_usdt: float = 0.0


@dataclass
class BotState:
    # ── Persistence ───────────────────────────────────────────────────────────
    capital_usdt: float = 30.0
    monthly_start_capital: float = 30.0
    monthly_pnl_usdt: float = 0.0
    monthly_trades: int = 0

    # Pause / halt state
    paused_until: Optional[str] = None          # ISO date
    survival_mode: bool = False

    # Daily tracking
    today: str = ""                             # ISO date
    daily_trades: int = 0
    daily_losses: int = 0
    daily_pnl_usdt: float = 0.0

    # Consecutive loss nights
    consecutive_loss_nights: int = 0

    # History (last 30 days)
    history: list[DailyRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["history"] = [asdict(r) for r in self.history]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BotState":
        history = [DailyRecord(**r) for r in d.pop("history", [])]
        obj = cls(**d)
        obj.history = history
        return obj


class RiskManager:
    def __init__(self, config: dict):
        self.risk_cfg = config["risk"]
        self.trading_cfg = config["trading"]
        self.state_file = config["state"]["file"]
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.state = self._load_state()
        self._maybe_reset_daily()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_state(self) -> BotState:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return BotState.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"State file corrupt, resetting: {e}")
        state = BotState(
            capital_usdt=self.trading_cfg["capital_usdt"],
            monthly_start_capital=self.trading_cfg["capital_usdt"],
        )
        return state

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)
        logger.debug("State saved")

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _maybe_reset_daily(self):
        today_str = date.today().isoformat()
        if self.state.today == today_str:
            return

        # Archive yesterday before resetting
        if self.state.today:
            rec = DailyRecord(
                date=self.state.today,
                trades=self.state.daily_trades,
                wins=self.state.daily_trades - self.state.daily_losses,
                losses=self.state.daily_losses,
                pnl_usdt=self.state.daily_pnl_usdt,
            )
            self.state.history.append(rec)
            # Keep only 30 days
            if len(self.state.history) > 30:
                self.state.history = self.state.history[-30:]

            # Check consecutive loss nights
            if self.state.daily_losses > 0 and self.state.daily_pnl_usdt < 0:
                self.state.consecutive_loss_nights += 1
                logger.info(
                    f"Losing night #{self.state.consecutive_loss_nights} recorded"
                )
            else:
                self.state.consecutive_loss_nights = 0

            # Enforce pause if consecutive loss limit hit
            limit = self.risk_cfg["consecutive_loss_days_limit"]
            pause_days = self.risk_cfg["consecutive_loss_pause_days"]
            if self.state.consecutive_loss_nights >= limit:
                pause_until = (date.today() + timedelta(days=pause_days)).isoformat()
                self.state.paused_until = pause_until
                self.state.consecutive_loss_nights = 0
                logger.warning(
                    f"PAUSED: {limit} consecutive losing nights → pause until {pause_until}"
                )

        # Reset daily counters
        self.state.today = today_str
        self.state.daily_trades = 0
        self.state.daily_losses = 0
        self.state.daily_pnl_usdt = 0.0
        self.save_state()
        logger.info(f"Daily counters reset for {today_str}")

    # ── Guard checks ──────────────────────────────────────────────────────────

    def can_trade(self) -> tuple[bool, str]:
        """Return (allowed, reason). Call before every potential entry."""
        self._maybe_reset_daily()

        # 1. Pause / cooldown period active
        if self.state.paused_until:
            pause_date = date.fromisoformat(self.state.paused_until)
            if date.today() < pause_date:
                return False, f"Paused until {self.state.paused_until} (consecutive losses)"
            else:
                self.state.paused_until = None
                self.save_state()

        # 2. Monthly profit target reached
        monthly_target = self.risk_cfg["monthly_profit_target_pct"]
        monthly_gain_pct = self.state.monthly_pnl_usdt / self.state.monthly_start_capital
        if monthly_gain_pct >= monthly_target:
            return False, f"Monthly profit target reached ({monthly_gain_pct:.1%} >= {monthly_target:.0%})"

        # 3. Max trades per day
        max_daily = self.risk_cfg["max_trades_per_day"]
        if self.state.daily_trades >= max_daily:
            return False, f"Daily trade limit reached ({self.state.daily_trades}/{max_daily})"

        # 4. Stop after N losses today
        stop_after_loss = self.risk_cfg["stop_after_daily_loss"]
        if self.state.daily_losses >= stop_after_loss:
            return False, f"Daily loss limit reached ({self.state.daily_losses} losses)"

        return True, "OK"

    # ── Survival mode check ───────────────────────────────────────────────────

    def check_survival_mode(self):
        """Evaluate whether survival mode should be activated/deactivated."""
        threshold = self.risk_cfg["survival_mode"]["capital_threshold_usdt"]
        if self.state.capital_usdt < threshold and not self.state.survival_mode:
            self.state.survival_mode = True
            logger.warning(
                f"SURVIVAL MODE ON: capital {self.state.capital_usdt:.2f} < {threshold} USDT"
            )
            self.save_state()

        # Also activate after 2 consecutive losses (per image: 連續虧損 2 筆後強制切換)
        if self.state.daily_losses >= 2 and not self.state.survival_mode:
            self.state.survival_mode = True
            logger.warning("SURVIVAL MODE ON: 2 consecutive intraday losses")
            self.save_state()

    # ── Trade recording ───────────────────────────────────────────────────────

    def record_trade(self, pnl_usdt: float):
        """Call after each trade closes with realized PnL (negative = loss)."""
        self._maybe_reset_daily()
        self.state.daily_trades += 1
        self.state.monthly_trades += 1
        self.state.daily_pnl_usdt += pnl_usdt
        self.state.monthly_pnl_usdt += pnl_usdt
        self.state.capital_usdt += pnl_usdt

        if pnl_usdt < 0:
            self.state.daily_losses += 1
            logger.info(f"Trade LOSS: {pnl_usdt:.4f} USDT | daily_losses={self.state.daily_losses}")
        else:
            logger.info(f"Trade WIN:  {pnl_usdt:.4f} USDT")

        self.check_survival_mode()
        self.save_state()

    def monthly_reset(self):
        """Call at the start of each new month."""
        self.state.monthly_start_capital = self.state.capital_usdt
        self.state.monthly_pnl_usdt = 0.0
        self.state.monthly_trades = 0
        self.state.survival_mode = False           # reset to normal at month start
        self.save_state()
        logger.info("Monthly stats reset")

    # ── Status summary ────────────────────────────────────────────────────────

    def status_summary(self) -> str:
        s = self.state
        monthly_pct = s.monthly_pnl_usdt / s.monthly_start_capital * 100 if s.monthly_start_capital else 0
        lines = [
            f"=== Bot Status [{s.today}] ===",
            f"Capital:        {s.capital_usdt:.4f} USDT",
            f"Monthly PnL:    {s.monthly_pnl_usdt:+.4f} USDT ({monthly_pct:+.2f}%)",
            f"Monthly trades: {s.monthly_trades}",
            f"Daily trades:   {s.daily_trades}/{self.risk_cfg['max_trades_per_day']}",
            f"Daily losses:   {s.daily_losses}",
            f"Consec. losses: {s.consecutive_loss_nights} nights",
            f"Survival mode:  {'ON' if s.survival_mode else 'off'}",
            f"Paused until:   {s.paused_until or 'N/A'}",
        ]
        return "\n".join(lines)
