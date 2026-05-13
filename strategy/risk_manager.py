"""
Risk management engine — SMC 30U 紐約盤保守版 v5.3

風控規則：
  每日最多 1 筆
  1 筆虧損當晚停手
  連續 2 晚虧損 → 停手 3 天
  連續 3 晚虧損 → 停手 7 天
  月度 +10% → 當月停手

資金預警（8.2）：
  ≤ 21U → 停手 24H（黃線）
  ≤ 15U → 停手 3 天（橘線）
  ≤  9U → 停手 7 天（紅線）
  =  0U → 當月停手（黑線）

月度獲利提醒：
  +20% (36U) → 提示提取超額
  +50% (45U) → 提示提取 15U 至冷錢包
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

from utils.logger import get_logger
from utils.notifier import notify_risk_event

logger = get_logger(__name__)


@dataclass
class DailyRecord:
    date: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_usdt: float = 0.0


@dataclass
class BotState:
    capital_usdt: float = 30.0
    monthly_start_capital: float = 30.0
    monthly_pnl_usdt: float = 0.0
    monthly_trades: int = 0

    paused_until: Optional[str] = None           # ISO date
    paused_until_datetime: Optional[str] = None  # ISO datetime (hour-level)
    survival_mode: bool = False

    today: str = ""
    daily_trades: int = 0
    daily_losses: int = 0
    daily_pnl_usdt: float = 0.0

    consecutive_loss_nights: int = 0
    history: list[DailyRecord] = field(default_factory=list)

    # Monthly profit-level alerts (reset each month)
    profit_alert_20pct: bool = False
    profit_alert_50pct: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["history"] = [asdict(r) for r in self.history]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BotState":
        history = [DailyRecord(**r) for r in d.pop("history", [])]
        # Tolerate old state files missing newer fields
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        d = {k: v for k, v in d.items() if k in valid}
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

    # ── Persistence ───────────────────────────────────────────────

    def _load_state(self) -> BotState:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    return BotState.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"State file corrupt, resetting: {e}")
        return BotState(
            capital_usdt=self.trading_cfg["capital_usdt"],
            monthly_start_capital=self.trading_cfg["capital_usdt"],
        )

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)

    # ── Daily reset ───────────────────────────────────────────────

    def _maybe_reset_daily(self):
        today_str = date.today().isoformat()
        if self.state.today == today_str:
            return

        if self.state.today:
            rec = DailyRecord(
                date=self.state.today,
                trades=self.state.daily_trades,
                wins=self.state.daily_trades - self.state.daily_losses,
                losses=self.state.daily_losses,
                pnl_usdt=self.state.daily_pnl_usdt,
            )
            self.state.history.append(rec)
            if len(self.state.history) > 30:
                self.state.history = self.state.history[-30:]

            if self.state.daily_losses > 0 and self.state.daily_pnl_usdt < 0:
                self.state.consecutive_loss_nights += 1
                self._apply_loss_night_rules()
            else:
                self.state.consecutive_loss_nights = 0

        self.state.today = today_str
        self.state.daily_trades = 0
        self.state.daily_losses = 0
        self.state.daily_pnl_usdt = 0.0
        self.save_state()

    def _apply_loss_night_rules(self):
        nights = self.state.consecutive_loss_nights
        hard_limit = self.risk_cfg.get("hard_loss_days_limit", 3)
        hard_pause = self.risk_cfg.get("hard_loss_pause_days", 7)
        soft_limit = self.risk_cfg["consecutive_loss_days_limit"]
        soft_pause = self.risk_cfg["consecutive_loss_pause_days"]

        if nights >= hard_limit:
            pause_until = (date.today() + timedelta(days=hard_pause)).isoformat()
            self.state.paused_until = pause_until
            self.state.consecutive_loss_nights = 0
            msg = f"連續 {nights} 晚虧損 → 停手 {hard_pause} 天（至 {pause_until}）"
            logger.warning(msg)
            notify_risk_event(f"⛔ 風控停手：{msg}")
        elif nights >= soft_limit:
            pause_until = (date.today() + timedelta(days=soft_pause)).isoformat()
            self.state.paused_until = pause_until
            self.state.consecutive_loss_nights = 0
            msg = f"連續 {nights} 晚虧損 → 停手 {soft_pause} 天（至 {pause_until}）"
            logger.warning(msg)
            notify_risk_event(f"⛔ 風控停手：{msg}")

    # ── Drawdown guard（8.2）──────────────────────────────────────

    def _check_drawdown(self):
        cap = self.state.capital_usdt
        dl = self.risk_cfg.get("drawdown_levels", {})

        red_usdt    = dl.get("red_usdt", 9.0)
        orange_usdt = dl.get("orange_usdt", 15.0)
        yellow_usdt = dl.get("yellow_usdt", 21.0)

        # Black line — capital at or below zero
        if cap <= 0:
            msg = "⚫ 黑線：資金歸零 → 當月停手，執行四週回測協議"
            logger.critical(msg)
            notify_risk_event(msg)
            return

        if cap <= red_usdt:
            pause_days = dl.get("red_pause_days", 7)
            until = (date.today() + timedelta(days=pause_days)).isoformat()
            if not self.state.paused_until or self.state.paused_until < until:
                self.state.paused_until = until
                msg = f"🔴 紅線：資金 {cap:.2f}U ≤ {red_usdt}U → 停手 {pause_days} 天"
                logger.warning(msg)
                notify_risk_event(msg)

        elif cap <= orange_usdt:
            pause_days = dl.get("orange_pause_days", 3)
            until = (date.today() + timedelta(days=pause_days)).isoformat()
            if not self.state.paused_until or self.state.paused_until < until:
                self.state.paused_until = until
                msg = f"🟠 橘線：資金 {cap:.2f}U ≤ {orange_usdt}U → 停手 {pause_days} 天"
                logger.warning(msg)
                notify_risk_event(msg)

        elif cap <= yellow_usdt:
            pause_hours = dl.get("yellow_pause_hours", 24)
            until_dt = (datetime.now() + timedelta(hours=pause_hours)).isoformat()
            if not self.state.paused_until_datetime:
                self.state.paused_until_datetime = until_dt
                msg = f"🟡 黃線：資金 {cap:.2f}U ≤ {yellow_usdt}U → 停手 {pause_hours}H"
                logger.warning(msg)
                notify_risk_event(msg)

        # Survival mode
        threshold = self.risk_cfg["survival_mode"]["capital_threshold_usdt"]
        if cap < threshold and not self.state.survival_mode:
            self.state.survival_mode = True
            notify_risk_event(f"⚠️ 生存模式啟動：資金 {cap:.2f}U < {threshold}U")

    # ── Monthly profit alerts（8.3）───────────────────────────

    def _check_monthly_targets(self):
        cap   = self.state.capital_usdt
        start = self.state.monthly_start_capital
        if start <= 0:
            return
        gain = (cap - start) / start

        if gain >= 0.50 and not self.state.profit_alert_50pct:
            self.state.profit_alert_50pct = True
            notify_risk_event(
                f"💰 月度資金達 {cap:.2f}U (+{gain:.0%}) → 建議提取 15U 至冷錢包"
            )
        elif gain >= 0.20 and not self.state.profit_alert_20pct:
            self.state.profit_alert_20pct = True
            notify_risk_event(
                f"✅ 月度資金達 {cap:.2f}U (+{gain:.0%}) → 建議提取超額部分"
            )

    # ── can_trade ────────────────────────────────────────────────

    def can_trade(self) -> tuple[bool, str]:
        self._maybe_reset_daily()

        # Hour-level pause (yellow line)
        if self.state.paused_until_datetime:
            until_dt = datetime.fromisoformat(self.state.paused_until_datetime)
            if datetime.now() < until_dt:
                return False, f"黃線停手中（至 {until_dt.strftime('%m/%d %H:%M')}）"
            else:
                self.state.paused_until_datetime = None
                self.save_state()

        # Day-level pause
        if self.state.paused_until:
            if date.today() < date.fromisoformat(self.state.paused_until):
                return False, f"風控停手中（至 {self.state.paused_until}）"
            else:
                self.state.paused_until = None
                self.save_state()

        # Monthly target
        monthly_gain = self.state.monthly_pnl_usdt / self.state.monthly_start_capital
        if monthly_gain >= self.risk_cfg["monthly_profit_target_pct"]:
            return False, f"月度目標達成（{monthly_gain:.1%}）→ 當月停手"

        # Daily trade limit
        if self.state.daily_trades >= self.risk_cfg["max_trades_per_day"]:
            return False, f"已達每日 {self.risk_cfg['max_trades_per_day']} 筆上限"

        # Daily loss limit
        if self.state.daily_losses >= self.risk_cfg["stop_after_daily_loss"]:
            return False, f"今日已虧損 {self.state.daily_losses} 筆 → 當晚停手"

        return True, "OK"

    # ── Record trade ──────────────────────────────────────────────

    def record_trade(self, pnl_usdt: float):
        self._maybe_reset_daily()
        self.state.daily_trades += 1
        self.state.monthly_trades += 1
        self.state.daily_pnl_usdt += pnl_usdt
        self.state.monthly_pnl_usdt += pnl_usdt
        self.state.capital_usdt += pnl_usdt

        if pnl_usdt < 0:
            self.state.daily_losses += 1

        self._check_drawdown()
        self._check_monthly_targets()
        self.save_state()

        result = "WIN ✅" if pnl_usdt >= 0 else "LOSS ❌"
        logger.info(
            f"Trade {result} {pnl_usdt:+.4f} USDT | "
            f"Capital={self.state.capital_usdt:.4f} | "
            f"Monthly={self.state.monthly_pnl_usdt:+.4f}"
        )

    def monthly_reset(self):
        self.state.monthly_start_capital = self.state.capital_usdt
        self.state.monthly_pnl_usdt = 0.0
        self.state.monthly_trades = 0
        self.state.survival_mode = False
        self.state.profit_alert_20pct = False
        self.state.profit_alert_50pct = False
        self.save_state()
        logger.info("Monthly stats reset")

    def status_summary(self) -> str:
        s = self.state
        monthly_pct = (
            s.monthly_pnl_usdt / s.monthly_start_capital * 100
            if s.monthly_start_capital else 0
        )
        return "\n".join([
            f"=== Bot Status [{s.today}] ===",
            f"Capital:        {s.capital_usdt:.4f} USDT",
            f"Monthly PnL:    {s.monthly_pnl_usdt:+.4f} USDT ({monthly_pct:+.2f}%)",
            f"Monthly trades: {s.monthly_trades}",
            f"Daily trades:   {s.daily_trades}/{self.risk_cfg['max_trades_per_day']}",
            f"Daily losses:   {s.daily_losses}",
            f"Consec. nights: {s.consecutive_loss_nights}",
            f"Survival mode:  {'ON' if s.survival_mode else 'off'}",
            f"Paused until:   {s.paused_until or s.paused_until_datetime or 'N/A'}",
        ])
