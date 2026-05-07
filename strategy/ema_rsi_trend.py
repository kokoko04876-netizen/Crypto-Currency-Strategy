"""
EMA Trend + RSI + ATR strategy.

Entry conditions (LONG):
  1. EMA-fast > EMA-medium > EMA-slow  (uptrend aligned)
  2. Price pulls back to within EMA-medium band (pullback entry)
  3. RSI in momentum zone [rsi_long_min, rsi_long_max]  (not overbought)
  4. Volume spike: current volume > volume_spike_factor × 20-bar average
  5. Stop-loss  = entry - atr_sl_multiplier  × ATR
  6. Take-profit = entry + atr_tp_multiplier × ATR

Entry conditions (SHORT): mirror of the above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Enums shared with ict_signals (redeclared locally to stay independent) ────

class SignalGrade(Enum):
    A = "A"
    B = "B"
    C = "C"
    NONE = "NONE"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class EMAState:
    fast: float
    medium: float
    slow: float
    aligned_up: bool    # fast > medium > slow
    aligned_down: bool  # fast < medium < slow


@dataclass
class RSIState:
    value: float
    in_long_zone: bool   # [rsi_long_min, rsi_long_max]
    in_short_zone: bool  # [rsi_short_min, rsi_short_max]


@dataclass
class ATRResult:
    value: float
    stop_loss: float
    take_profit: float


@dataclass
class VolumeState:
    current: float
    average: float
    is_spike: bool


@dataclass
class EMARSISignalResult:
    grade: SignalGrade
    direction: Optional[Direction]
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    ema: Optional[EMAState] = None
    rsi: Optional[RSIState] = None
    atr: Optional[ATRResult] = None
    volume: Optional[VolumeState] = None
    conditions_met: list = field(default_factory=list)
    reason: str = ""


# ── Indicator calculations ────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def compute_ema_state(df: pd.DataFrame, cfg: dict) -> EMAState:
    fast = _ema(df["close"], cfg["ema_fast"]).iloc[-1]
    medium = _ema(df["close"], cfg["ema_medium"]).iloc[-1]
    slow = _ema(df["close"], cfg["ema_slow"]).iloc[-1]
    return EMAState(
        fast=fast,
        medium=medium,
        slow=slow,
        aligned_up=fast > medium > slow,
        aligned_down=fast < medium < slow,
    )


def compute_rsi_state(df: pd.DataFrame, cfg: dict) -> RSIState:
    rsi_val = _rsi(df["close"], cfg["rsi_period"]).iloc[-1]
    return RSIState(
        value=rsi_val,
        in_long_zone=cfg["rsi_long_min"] <= rsi_val <= cfg["rsi_long_max"],
        in_short_zone=cfg["rsi_short_min"] <= rsi_val <= cfg["rsi_short_max"],
    )


def compute_atr(df: pd.DataFrame, cfg: dict, direction: Direction, entry: float) -> ATRResult:
    atr_val = _atr(df, cfg["atr_period"]).iloc[-1]
    sl_dist = cfg["atr_sl_multiplier"] * atr_val
    tp_dist = cfg["atr_tp_multiplier"] * atr_val
    if direction == Direction.LONG:
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
    return ATRResult(value=atr_val, stop_loss=sl, take_profit=tp)


def compute_volume_state(df: pd.DataFrame, cfg: dict) -> VolumeState:
    avg = df["volume"].iloc[-cfg["volume_ma_period"] - 1:-1].mean()
    current = df["volume"].iloc[-1]
    return VolumeState(
        current=current,
        average=avg,
        is_spike=current >= cfg["volume_spike_factor"] * avg,
    )


# ── Pullback-to-EMA check ─────────────────────────────────────────────────────

def _price_near_medium_ema(price: float, ema_medium: float, atr_val: float) -> bool:
    """Price within 0.5 × ATR of the medium EMA (pullback entry zone)."""
    return abs(price - ema_medium) <= 0.5 * atr_val


# ── Main signal evaluator ─────────────────────────────────────────────────────

class EMARSISignalDetector:
    """
    EMA Trend + RSI momentum + ATR-based risk levels.

    Requires config['ema_rsi_strategy'] block (see config.yaml).
    """

    def __init__(self, config: dict):
        self.cfg = config["ema_rsi_strategy"]

    def evaluate(self, df_htf: pd.DataFrame) -> EMARSISignalResult:
        """Evaluate signal on the provided HTF DataFrame."""
        if len(df_htf) < max(self.cfg["ema_slow"], self.cfg["atr_period"]) + 5:
            return EMARSISignalResult(
                grade=SignalGrade.NONE,
                direction=None,
                reason="Insufficient candle history",
            )

        current_price = df_htf["close"].iloc[-1]
        ema = compute_ema_state(df_htf, self.cfg)
        rsi = compute_rsi_state(df_htf, self.cfg)
        vol = compute_volume_state(df_htf, self.cfg)

        logger.debug(
            f"EMA fast={ema.fast:.4f} med={ema.medium:.4f} slow={ema.slow:.4f} | "
            f"RSI={rsi.value:.1f} | vol_spike={vol.is_spike}"
        )

        for direction in [Direction.LONG, Direction.SHORT]:
            result = self._check_direction(
                direction, current_price, ema, rsi, vol, df_htf
            )
            if result.grade == SignalGrade.A:
                logger.info(
                    f"EMA-RSI A-GRADE {direction.value.upper()} | "
                    f"entry={result.entry_price:.4f} SL={result.stop_loss:.4f} "
                    f"TP={result.take_profit:.4f}"
                )
                return result

        return EMARSISignalResult(
            grade=SignalGrade.NONE,
            direction=None,
            reason="No EMA-RSI signal conditions met",
        )

    def _check_direction(
        self,
        direction: Direction,
        price: float,
        ema: EMAState,
        rsi: RSIState,
        vol: VolumeState,
        df: pd.DataFrame,
    ) -> EMARSISignalResult:
        conditions: list[str] = []

        # 1. EMA alignment
        if direction == Direction.LONG and ema.aligned_up:
            conditions.append("EMA_UP")
        elif direction == Direction.SHORT and ema.aligned_down:
            conditions.append("EMA_DOWN")
        else:
            return EMARSISignalResult(
                grade=SignalGrade.NONE,
                direction=direction,
                conditions_met=conditions,
                reason=f"EMA not aligned for {direction.value}",
            )

        # 2. Price near medium EMA (pullback zone)
        atr_rough = _atr(df, self.cfg["atr_period"]).iloc[-1]
        if _price_near_medium_ema(price, ema.medium, atr_rough):
            conditions.append("PULLBACK")
        else:
            return EMARSISignalResult(
                grade=SignalGrade.B,
                direction=direction,
                conditions_met=conditions,
                reason="EMA aligned but no pullback to EMA-medium",
            )

        # 3. RSI in momentum zone
        if direction == Direction.LONG and rsi.in_long_zone:
            conditions.append("RSI_LONG")
        elif direction == Direction.SHORT and rsi.in_short_zone:
            conditions.append("RSI_SHORT")
        else:
            return EMARSISignalResult(
                grade=SignalGrade.B,
                direction=direction,
                conditions_met=conditions,
                reason=f"RSI {rsi.value:.1f} not in required zone",
            )

        # 4. Volume spike confirmation
        if vol.is_spike:
            conditions.append("VOL_SPIKE")
        else:
            return EMARSISignalResult(
                grade=SignalGrade.B,
                direction=direction,
                conditions_met=conditions,
                reason=f"No volume spike ({vol.current:.0f} < {vol.spike_threshold:.0f})",
            )

        # ── A-Grade: all 4 conditions met ─────────────────────────────────────
        atr_result = compute_atr(df, self.cfg, direction, price)

        return EMARSISignalResult(
            grade=SignalGrade.A,
            direction=direction,
            entry_price=price,
            stop_loss=atr_result.stop_loss,
            take_profit=atr_result.take_profit,
            ema=ema,
            rsi=rsi,
            atr=atr_result,
            volume=vol,
            conditions_met=conditions,
            reason="A-grade: EMA+Pullback+RSI+Volume",
        )
