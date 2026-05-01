"""
ICT (Inner Circle Trader) signal detection.

A-Grade signal requires ALL three HTF conditions:
  1. Order Block (OB)  – price returning to a mitigation zone
  2. Fair Value Gap (FVG) – imbalance overlapping the OB
  3. OTE (Optimal Trade Entry) – 0.62–0.79 Fibonacci retracement

LTF entry trigger (one of):
  - CISD  (Change in State of Delivery)
  - Turtle Soup (false breakout / stop-hunt)

Session context (one of):
  - London Reversal
  - NY Open setup
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

class SignalGrade(Enum):
    A = "A"
    B = "B"
    C = "C"
    NONE = "NONE"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class OrderBlock:
    direction: Direction          # demand (bullish) or supply (bearish) OB
    high: float
    low: float
    index: int                    # candle index where OB formed
    mitigated: bool = False


@dataclass
class FairValueGap:
    direction: Direction
    top: float
    bottom: float
    index: int


@dataclass
class OTEZone:
    direction: Direction
    entry_ideal: float            # 0.705 level
    zone_low: float               # 0.62 retracement
    zone_high: float              # 0.79 retracement
    swing_high: float
    swing_low: float


@dataclass
class LTFTrigger:
    kind: str                     # "cisd" or "turtle_soup"
    direction: Direction
    trigger_price: float
    index: int


@dataclass
class SignalResult:
    grade: SignalGrade
    direction: Optional[Direction]
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    ob: Optional[OrderBlock] = None
    fvg: Optional[FairValueGap] = None
    ote: Optional[OTEZone] = None
    ltf_trigger: Optional[LTFTrigger] = None
    conditions_met: list = field(default_factory=list)
    reason: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_bullish(candle: pd.Series) -> bool:
    return candle["close"] > candle["open"]


def _is_bearish(candle: pd.Series) -> bool:
    return candle["close"] < candle["open"]


def _candle_body_size(candle: pd.Series) -> float:
    return abs(candle["close"] - candle["open"])


def _find_swing_high(df: pd.DataFrame, lookback: int = 20) -> tuple[float, int]:
    window = df.iloc[-lookback:]
    idx = window["high"].idxmax()
    return window.loc[idx, "high"], window.index.get_loc(idx)


def _find_swing_low(df: pd.DataFrame, lookback: int = 20) -> tuple[float, int]:
    window = df.iloc[-lookback:]
    idx = window["low"].idxmin()
    return window.loc[idx, "low"], window.index.get_loc(idx)


# ── Order Block detection ─────────────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, cfg: dict) -> list[OrderBlock]:
    """
    Bullish OB: last bearish candle before a sharp bullish impulse.
    Bearish OB: last bullish candle before a sharp bearish impulse.
    """
    min_impulse = cfg.get("min_impulse_pct", 0.003)
    max_age = cfg.get("max_ob_age_candles", 50)
    obs: list[OrderBlock] = []
    start = max(0, len(df) - max_age)

    for i in range(start, len(df) - 2):
        c0 = df.iloc[i]
        c1 = df.iloc[i + 1]
        c2 = df.iloc[i + 2] if i + 2 < len(df) else None

        # Bullish OB: bearish candle followed by impulse up
        if _is_bearish(c0) and _is_bullish(c1):
            impulse = (c1["close"] - c1["open"]) / c1["open"]
            if impulse >= min_impulse:
                obs.append(OrderBlock(
                    direction=Direction.LONG,
                    high=c0["high"],
                    low=c0["low"],
                    index=i,
                ))

        # Bearish OB: bullish candle followed by impulse down
        if _is_bullish(c0) and _is_bearish(c1):
            impulse = (c1["open"] - c1["close"]) / c1["open"]
            if impulse >= min_impulse:
                obs.append(OrderBlock(
                    direction=Direction.SHORT,
                    high=c0["high"],
                    low=c0["low"],
                    index=i,
                ))

    # Mark OBs that have been mitigated (price traded through them)
    current_price = df.iloc[-1]["close"]
    for ob in obs:
        if ob.direction == Direction.LONG and current_price < ob.low:
            ob.mitigated = True
        elif ob.direction == Direction.SHORT and current_price > ob.high:
            ob.mitigated = True

    active = [ob for ob in obs if not ob.mitigated]
    logger.debug(f"Detected {len(active)} active Order Blocks")
    return active


# ── Fair Value Gap detection ──────────────────────────────────────────────────

def detect_fvgs(df: pd.DataFrame, cfg: dict) -> list[FairValueGap]:
    """
    Bullish FVG: df[i-2].high < df[i].low  (gap above candle i-2, below candle i)
    Bearish FVG: df[i-2].low  > df[i].high (gap below candle i-2, above candle i)
    """
    min_gap = cfg.get("min_gap_pct", 0.001)
    fvgs: list[FairValueGap] = []

    for i in range(2, len(df)):
        c_prev2 = df.iloc[i - 2]
        c_curr = df.iloc[i]

        # Bullish FVG
        if c_prev2["high"] < c_curr["low"]:
            gap_pct = (c_curr["low"] - c_prev2["high"]) / c_prev2["high"]
            if gap_pct >= min_gap:
                fvgs.append(FairValueGap(
                    direction=Direction.LONG,
                    top=c_curr["low"],
                    bottom=c_prev2["high"],
                    index=i,
                ))

        # Bearish FVG
        elif c_prev2["low"] > c_curr["high"]:
            gap_pct = (c_prev2["low"] - c_curr["high"]) / c_curr["high"]
            if gap_pct >= min_gap:
                fvgs.append(FairValueGap(
                    direction=Direction.SHORT,
                    top=c_prev2["low"],
                    bottom=c_curr["high"],
                    index=i,
                ))

    logger.debug(f"Detected {len(fvgs)} Fair Value Gaps")
    return fvgs


# ── OTE (Optimal Trade Entry) ─────────────────────────────────────────────────

def detect_ote(df: pd.DataFrame, cfg: dict, direction: Direction) -> Optional[OTEZone]:
    """
    Compute OTE zone from the most recent significant swing.
    For LONG:  swing from recent low → swing high, OTE retracement zone.
    For SHORT: swing from recent high → swing low, OTE retracement zone.
    """
    fib_level = cfg.get("fib_level", 0.705)
    fib_low = cfg.get("fib_low", 0.62)
    fib_high = cfg.get("fib_high", 0.79)
    lookback = 30

    swing_high, _ = _find_swing_high(df, lookback)
    swing_low, _ = _find_swing_low(df, lookback)
    rng = swing_high - swing_low

    if rng <= 0:
        return None

    if direction == Direction.LONG:
        # Price retraced from swing_high down toward swing_low
        ote_ideal = swing_high - fib_level * rng
        zone_high = swing_high - fib_low * rng
        zone_low = swing_high - fib_high * rng
    else:
        # Price retraced from swing_low up toward swing_high
        ote_ideal = swing_low + fib_level * rng
        zone_low = swing_low + fib_low * rng
        zone_high = swing_low + fib_high * rng

    return OTEZone(
        direction=direction,
        entry_ideal=ote_ideal,
        zone_low=zone_low,
        zone_high=zone_high,
        swing_high=swing_high,
        swing_low=swing_low,
    )


def price_in_ote(current_price: float, ote: OTEZone) -> bool:
    return ote.zone_low <= current_price <= ote.zone_high


# ── LTF Triggers ─────────────────────────────────────────────────────────────

def detect_cisd(df_ltf: pd.DataFrame, direction: Direction) -> Optional[LTFTrigger]:
    """
    CISD – Change in State of Delivery:
    LONG:  price breaks above the most recent lower high on the LTF (structure shift up)
    SHORT: price breaks below the most recent higher low (structure shift down)
    """
    if len(df_ltf) < 5:
        return None

    recent = df_ltf.iloc[-10:]

    if direction == Direction.LONG:
        # Find the most recent lower high (swing high that is lower than the one before)
        highs = recent["high"].values
        for i in range(len(highs) - 2, 0, -1):
            if highs[i] < highs[i - 1]:            # lower high found
                level = highs[i]
                if recent["close"].iloc[-1] > level:
                    return LTFTrigger("cisd", Direction.LONG, level, len(df_ltf) - 1)
    else:
        lows = recent["low"].values
        for i in range(len(lows) - 2, 0, -1):
            if lows[i] > lows[i - 1]:              # higher low found
                level = lows[i]
                if recent["close"].iloc[-1] < level:
                    return LTFTrigger("cisd", Direction.SHORT, level, len(df_ltf) - 1)

    return None


def detect_turtle_soup(df_ltf: pd.DataFrame, direction: Direction, lookback: int = 20) -> Optional[LTFTrigger]:
    """
    Turtle Soup – false breakout / stop hunt:
    LONG:  price briefly breaks below the N-bar low, then closes back above (stop hunt).
    SHORT: price briefly breaks above the N-bar high, then closes back below.
    """
    if len(df_ltf) < lookback + 2:
        return None

    window = df_ltf.iloc[-(lookback + 2):-2]
    last = df_ltf.iloc[-2]
    current = df_ltf.iloc[-1]

    if direction == Direction.LONG:
        prev_low = window["low"].min()
        if last["low"] < prev_low and current["close"] > prev_low:
            return LTFTrigger("turtle_soup", Direction.LONG, prev_low, len(df_ltf) - 1)
    else:
        prev_high = window["high"].max()
        if last["high"] > prev_high and current["close"] < prev_high:
            return LTFTrigger("turtle_soup", Direction.SHORT, prev_high, len(df_ltf) - 1)

    return None


def get_ltf_trigger(df_ltf: pd.DataFrame, direction: Direction) -> Optional[LTFTrigger]:
    cisd = detect_cisd(df_ltf, direction)
    if cisd:
        logger.debug(f"LTF trigger: CISD {direction.value} @ {cisd.trigger_price:.4f}")
        return cisd
    ts = detect_turtle_soup(df_ltf, direction)
    if ts:
        logger.debug(f"LTF trigger: Turtle Soup {direction.value} @ {ts.trigger_price:.4f}")
        return ts
    return None


# ── Session context ───────────────────────────────────────────────────────────

def is_london_reversal_context(df_htf: pd.DataFrame) -> bool:
    """
    London Reversal: price reversed direction during London open (around 15:00-17:00 UTC+8).
    Approximated by checking if the HTF 1H candle shows a significant reversal from the
    Asian session direction.
    """
    if len(df_htf) < 6:
        return False
    # Asian session avg (last 4–6 candles before London open ~ midnight to morning)
    asian = df_htf.iloc[-6:-2]
    asian_direction = "up" if asian["close"].iloc[-1] > asian["open"].iloc[0] else "down"
    # London candle should reverse
    london_candle = df_htf.iloc[-2]
    london_reversal = (
        asian_direction == "up" and _is_bearish(london_candle)
    ) or (
        asian_direction == "down" and _is_bullish(london_candle)
    )
    return london_reversal


def is_ny_open_context(df_htf: pd.DataFrame) -> bool:
    """
    NY Open setup: significant move on the candle aligned with NY open.
    Approximated by checking if the most recent closed HTF candle has a strong body.
    """
    if len(df_htf) < 3:
        return False
    recent = df_htf.iloc[-2]
    body = _candle_body_size(recent)
    avg_body = df_htf.iloc[-20:]["close"].sub(df_htf.iloc[-20:]["open"]).abs().mean()
    return body > avg_body * 1.5


# ── Main signal evaluator ─────────────────────────────────────────────────────

class ICTSignalDetector:
    def __init__(self, config: dict):
        self.sig_cfg = config["signals"]
        self.sl_pct = config["trading"]["stop_loss_pct"]
        self.tp_pct = config["trading"]["take_profit_pct"]
        self.min_conditions = self.sig_cfg.get("min_conditions_for_a_grade", 3)

    def evaluate(self, df_htf: pd.DataFrame, df_ltf: pd.DataFrame) -> SignalResult:
        """
        Run full ICT analysis and return a SignalResult.
        Only returns A-grade (all 3 conditions + session + LTF trigger).
        """
        current_price = df_htf.iloc[-1]["close"]
        logger.debug(f"Evaluating signal @ price={current_price:.4f}")

        # ── Detect structures ─────────────────────────────────────────────────
        obs = detect_order_blocks(df_htf, self.sig_cfg["ob"])
        fvgs = detect_fvgs(df_htf, self.sig_cfg["fvg"])

        # ── Try LONG then SHORT ───────────────────────────────────────────────
        for direction in [Direction.LONG, Direction.SHORT]:
            result = self._check_direction(
                direction, current_price, obs, fvgs, df_htf, df_ltf
            )
            if result.grade == SignalGrade.A:
                logger.info(
                    f"A-GRADE {direction.value.upper()} signal | "
                    f"entry={result.entry_price:.4f} SL={result.stop_loss:.4f} TP={result.take_profit:.4f}"
                )
                return result

        return SignalResult(
            grade=SignalGrade.NONE,
            direction=None,
            reason="No A-grade conditions met",
        )

    def _check_direction(
        self,
        direction: Direction,
        current_price: float,
        obs: list[OrderBlock],
        fvgs: list[FairValueGap],
        df_htf: pd.DataFrame,
        df_ltf: pd.DataFrame,
    ) -> SignalResult:
        conditions: list[str] = []
        matched_ob: Optional[OrderBlock] = None
        matched_fvg: Optional[FairValueGap] = None
        matched_ote: Optional[OTEZone] = None

        # 1. Order Block – price must be inside an active OB
        for ob in obs:
            if ob.direction == direction and ob.low <= current_price <= ob.high:
                matched_ob = ob
                conditions.append("OB")
                break

        # 2. Fair Value Gap overlapping with OB zone
        for fvg in fvgs:
            if fvg.direction == direction:
                if matched_ob:
                    # FVG overlaps with OB
                    overlap = (
                        fvg.bottom <= matched_ob.high and fvg.top >= matched_ob.low
                    )
                    if overlap and fvg.bottom <= current_price <= fvg.top:
                        matched_fvg = fvg
                        conditions.append("FVG")
                        break
                else:
                    if fvg.bottom <= current_price <= fvg.top:
                        matched_fvg = fvg
                        conditions.append("FVG")
                        break

        # 3. OTE – price within Fibonacci retracement zone
        ote = detect_ote(df_htf, self.sig_cfg["ote"], direction)
        if ote and price_in_ote(current_price, ote):
            matched_ote = ote
            conditions.append("OTE")

        # Grade based on HTF conditions
        n = len(conditions)
        if n < self.min_conditions:
            grade = SignalGrade.B if n == 2 else SignalGrade.C
            return SignalResult(
                grade=grade, direction=direction,
                conditions_met=conditions,
                reason=f"Only {n}/3 HTF conditions: {conditions}",
            )

        # 4. Session context (London Reversal or NY Open)
        session_ok = is_london_reversal_context(df_htf) or is_ny_open_context(df_htf)
        if not session_ok:
            return SignalResult(
                grade=SignalGrade.B, direction=direction,
                conditions_met=conditions,
                reason="HTF conditions met but no session context (London/NY)",
            )

        # 5. LTF trigger (CISD or Turtle Soup)
        ltf_trigger = get_ltf_trigger(df_ltf, direction)
        if not ltf_trigger:
            return SignalResult(
                grade=SignalGrade.B, direction=direction,
                conditions_met=conditions,
                reason="HTF+session OK but no LTF trigger (CISD/TurtleSoup)",
            )

        # ── A-Grade confirmed ─────────────────────────────────────────────────
        entry = current_price
        if direction == Direction.LONG:
            sl = entry * (1 - self.sl_pct)
            tp = entry * (1 + self.tp_pct)
        else:
            sl = entry * (1 + self.sl_pct)
            tp = entry * (1 - self.tp_pct)

        return SignalResult(
            grade=SignalGrade.A,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            ob=matched_ob,
            fvg=matched_fvg,
            ote=matched_ote,
            ltf_trigger=ltf_trigger,
            conditions_met=conditions + ["Session", f"LTF:{ltf_trigger.kind}"],
            reason="A-grade: OB+FVG+OTE+Session+LTF",
        )
