"""
Position sizing following the strategy's 倉位計算 (position calculation) rules:

  資金: 30 USDT | 槓桿: 1x | 止損: 1.0% | 止盈: 2.5%
  風險金額 = 資金 × 1%  = 0.30 USDT
  名義倉位 = 資金        = 30 USDT  (1x leverage → notional = capital)
  保證金   = 名義倉位   = 30 USDT
  預期盈利 = 資金 × 2.5% - 手續費
  真實盈虧比 ≈ 2.28:1 ✅
"""
from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionPlan:
    symbol: str
    direction: str                  # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit: float

    # Sizing
    leverage: int
    capital_usdt: float             # total allocated capital
    risk_usdt: float                # 1% of capital
    notional_usdt: float            # contract value (capital × leverage)
    qty: float                      # base asset quantity

    # Expected outcomes (after fees)
    expected_profit_net: float
    expected_loss_net: float
    true_rr: float

    def summary(self) -> str:
        return (
            f"\n{'─'*42}\n"
            f"  {self.direction.upper()}  {self.symbol}\n"
            f"  Entry:      {self.entry_price:.4f}\n"
            f"  Stop Loss:  {self.stop_loss:.4f}  ({abs(self.entry_price - self.stop_loss)/self.entry_price*100:.2f}%)\n"
            f"  Take Profit:{self.take_profit:.4f}  ({abs(self.entry_price - self.take_profit)/self.entry_price*100:.2f}%)\n"
            f"  Qty:        {self.qty:.6f}\n"
            f"  Notional:   {self.notional_usdt:.2f} USDT  ({self.leverage}x)\n"
            f"  Risk:       {self.risk_usdt:.4f} USDT\n"
            f"  Net profit: +{self.expected_profit_net:.4f} USDT\n"
            f"  Net loss:   -{self.expected_loss_net:.4f} USDT\n"
            f"  True R:R    {self.true_rr:.2f}:1\n"
            f"{'─'*42}"
        )


class PositionCalculator:
    def __init__(self, config: dict):
        cfg = config["trading"]
        self.capital = cfg["capital_usdt"]
        self.leverage = cfg["leverage"]
        self.risk_pct = cfg["risk_per_trade"]       # 0.01
        self.sl_pct = cfg["stop_loss_pct"]          # 0.01
        self.tp_pct = cfg["take_profit_pct"]        # 0.025
        self.maker_fee = cfg["maker_fee"]           # 0.0002
        self.taker_fee = cfg["taker_fee"]           # 0.0005

    def calculate(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        current_capital: float | None = None,
    ) -> PositionPlan:
        """
        Compute position size and expected PnL.
        current_capital: use live capital if provided, else config default.
        """
        capital = current_capital if current_capital is not None else self.capital

        # ── Risk-based sizing ─────────────────────────────────────────────────
        risk_usdt = capital * self.risk_pct          # e.g. 30 × 1% = 0.30 USDT

        # With 1x leverage, the notional position equals capital
        notional_usdt = capital * self.leverage       # 30 × 1 = 30 USDT
        qty = notional_usdt / entry_price             # units of base asset

        # ── Stop / target prices ──────────────────────────────────────────────
        if direction == "long":
            sl = entry_price * (1 - self.sl_pct)
            tp = entry_price * (1 + self.tp_pct)
        else:
            sl = entry_price * (1 + self.sl_pct)
            tp = entry_price * (1 - self.tp_pct)

        # ── Fee calculation (entry + exit, assume maker for limit orders) ─────
        entry_fee = notional_usdt * self.maker_fee
        exit_fee = notional_usdt * self.maker_fee
        total_fee = entry_fee + exit_fee

        # ── Expected PnL ──────────────────────────────────────────────────────
        gross_profit = notional_usdt * self.tp_pct
        gross_loss = notional_usdt * self.sl_pct
        net_profit = gross_profit - total_fee
        net_loss = gross_loss + total_fee            # loss + fees = larger outflow
        true_rr = net_profit / net_loss if net_loss > 0 else 0.0

        plan = PositionPlan(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            leverage=self.leverage,
            capital_usdt=capital,
            risk_usdt=risk_usdt,
            notional_usdt=notional_usdt,
            qty=qty,
            expected_profit_net=net_profit,
            expected_loss_net=net_loss,
            true_rr=true_rr,
        )
        logger.info(plan.summary())
        return plan

    def update_capital(self, new_capital: float):
        self.capital = new_capital
