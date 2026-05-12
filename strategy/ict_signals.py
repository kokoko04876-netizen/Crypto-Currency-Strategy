"""
ICT Signal Detector — Full 7-Element A-Grade System

Element 5.1  HTF Bias          : 4H BOS + MA7/14/28 + Daily HH/HL or LH/LL
Element 5.2  Premium/Discount  : 50% midline, ±5% strong zones
Element 5.3  MTF POI 4×Resonance: OB + FVG + OTE(0.618-0.79) + Volume Profile POC
Element 5.4  Liquidity         : SSL/BSL sweep + DOL direction
Element 5.5  LTF Triggers      : ≥2 of CISD / MSS / Turtle Soup / LTF FVG
Element 5.6  Time Window       : 22:00–00:00 Taiwan (handled by SessionManager)
Element 5.7  News & SMT        : FOMC/CPI ±4H block + BTC vs ETH divergence

Grade:  A = 4 resonances (5.3) + all other elements pass
        B = 3 resonances  → skip
        C = ≤2 resonances → skip
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Enums & data classes ──────────────────────────────────────────────────────

class SignalGrade(Enum):
    A = "A"
    B = "B"
    C = "C"
    NONE = "NONE"


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class HTFBias:
    direction: Direction
    bos_confirmed: bool
    ma_aligned: bool
    consecutive_candles: int
    daily_aligned: bool
    daily_choch: bool              # True = CHoCH present (conflict)
    score: int                     # 0-4


@dataclass
class PremiumDiscount:
    midline: float
    swing_high: float
    swing_low: float
    zone: str                      # "discount", "premium", "neutral"
    pct_from_mid: float


@dataclass
class OrderBlock:
    direction: Direction
    high: float
    low: float
    index: int
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
    entry_ideal: float
    zone_low: float
    zone_high: float
    swing_high: float
    swing_low: float


@dataclass
class VolumeProfile:
    poc: float                     # Point of Control
    vah: float                     # Value Area High
    val: float                     # Value Area Low
    price_in_value_area: bool


@dataclass
class LiquidityState:
    ssl_swept: bool                # Stop-side liquidity (lows) swept
    bsl_swept: bool                # Buy-side liquidity (highs) swept
    dol_direction: Optional[Direction]   # Draw on Liquidity target direction
    ssl_level: float
    bsl_level: float


@dataclass
class LTFTrigger:
    kind: str                      # cisd | mss | turtle_soup | ltf_fvg
    direction: Direction
    trigger_price: float
    index: int


@dataclass
class SMTResult:
    divergence_found: bool
    btc_direction: Optional[Direction]
    eth_direction: Optional[Direction]
    grade_boost: bool              # A → A+ if True


@dataclass
class SignalResult:
    grade: SignalGrade
    direction: Optional[Direction]
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    htf_bias: Optional[HTFBias] = None
    pd_zone: Optional[PremiumDiscount] = None
    ob: Optional[OrderBlock] = None
    fvg: Optional[FairValueGap] = None
    ote: Optional[OTEZone] = None
    volume_profile: Optional[VolumeProfile] = None
    liquidity: Optional[LiquidityState] = None
    ltf_triggers: list = field(default_factory=list)
    smt: Optional[SMTResult] = None
    resonances: int = 0
    conditions_met: list = field(default_factory=list)
    reason: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_bullish(c: pd.Series) -> bool:
    return float(c["close"]) > float(c["open"])

def _is_bearish(c: pd.Series) -> bool:
    return float(c["close"]) < float(c["open"])

def _body(c: pd.Series) -> float:
    return abs(float(c["close"]) - float(c["open"]))

def _swing_high(df: pd.DataFrame, lookback: int = 30) -> tuple[float, int]:
    w = df.iloc[-lookback:]
    i = w["high"].idxmax()
    return float(w.loc[i, "high"]), int(w.index.get_loc(i))

def _swing_low(df: pd.DataFrame, lookback: int = 30) -> tuple[float, int]:
    w = df.iloc[-lookback:]
    i = w["low"].idxmin()
    return float(w.loc[i, "low"]), int(w.index.get_loc(i))


# ── 5.1 HTF Bias ─────────────────────────────────────────────────────────────

def detect_htf_bias(df_4h: pd.DataFrame, df_daily: pd.DataFrame, cfg: dict) -> Optional[HTFBias]:
    ma_periods = cfg.get("ma_periods", [7, 14, 28])
    min_consec = cfg.get("min_consecutive_candles", 2)

    ma_vals = {}
    for p in ma_periods:
        if len(df_4h) >= p:
            ma_vals[p] = float(df_4h["close"].rolling(p).mean().iloc[-1])

    ma_bullish = ma_aligned = False
    if len(ma_vals) == len(ma_periods):
        ma_aligned_bull = ma_vals[ma_periods[0]] > ma_vals[ma_periods[1]] > ma_vals[ma_periods[2]]
        ma_aligned_bear = ma_vals[ma_periods[0]] < ma_vals[ma_periods[1]] < ma_vals[ma_periods[2]]
        ma_aligned = ma_aligned_bull or ma_aligned_bear
        ma_bullish = ma_aligned_bull

    recent = df_4h.iloc[-min_consec:]
    all_bull = all(_is_bullish(recent.iloc[i]) for i in range(len(recent)))
    all_bear = all(_is_bearish(recent.iloc[i]) for i in range(len(recent)))
    consec_dir = Direction.LONG if all_bull else (Direction.SHORT if all_bear else None)
    consec_ok = consec_dir is not None

    lookback = cfg.get("lookback", 50)
    bos_direction = None
    if len(df_4h) >= 3:
        prev_high, _ = _swing_high(df_4h.iloc[:-3], min(lookback, len(df_4h) - 3))
        prev_low, _ = _swing_low(df_4h.iloc[:-3], min(lookback, len(df_4h) - 3))
        last_close = float(df_4h.iloc[-1]["close"])
        if last_close > prev_high:
            bos_direction = Direction.LONG
        elif last_close < prev_low:
            bos_direction = Direction.SHORT

    bos_confirmed = bos_direction is not None

    daily_direction = None
    daily_choch = False
    if len(df_daily) >= 5:
        highs = df_daily["high"].iloc[-5:].values
        lows = df_daily["low"].iloc[-5:].values
        hh = highs[-1] > highs[-2] > highs[-3]
        hl = lows[-1] > lows[-2]
        ll = lows[-1] < lows[-2] < lows[-3]
        lh = highs[-1] < highs[-2]
        if hh and hl:
            daily_direction = Direction.LONG
        elif ll and lh:
            daily_direction = Direction.SHORT
        if daily_direction == Direction.LONG and lows[-1] < lows[-3]:
            daily_choch = True
        elif daily_direction == Direction.SHORT and highs[-1] > highs[-3]:
            daily_choch = True

    votes_long = sum([
        bos_direction == Direction.LONG,
        consec_dir == Direction.LONG,
        ma_bullish,
        daily_direction == Direction.LONG,
    ])
    votes_short = sum([
        bos_direction == Direction.SHORT,
        consec_dir == Direction.SHORT,
        not ma_bullish and ma_aligned,
        daily_direction == Direction.SHORT,
    ])
    direction = Direction.LONG if votes_long > votes_short else Direction.SHORT
    score = max(votes_long, votes_short)
    daily_aligned = daily_direction == direction if daily_direction else False

    return HTFBias(
        direction=direction,
        bos_confirmed=bos_confirmed,
        ma_aligned=ma_aligned,
        consecutive_candles=min_consec if consec_ok else 0,
        daily_aligned=daily_aligned,
        daily_choch=daily_choch,
        score=score,
    )


# ── 5.2 Premium / Discount zone ──────────────────────────────────────────────

def calculate_premium_discount(df: pd.DataFrame, cfg: dict, current_price: float) -> PremiumDiscount:
    lookback = cfg.get("lookback_candles", 50)
    strong_pct = cfg.get("strong_zone_pct", 0.05)
    swing_h, _ = _swing_high(df, lookback)
    swing_l, _ = _swing_low(df, lookback)
    midline = (swing_h + swing_l) / 2
    pct_from_mid = (current_price - midline) / midline

    if pct_from_mid <= -strong_pct:
        zone = "discount"
    elif pct_from_mid >= strong_pct:
        zone = "premium"
    else:
        zone = "neutral"

    logger.debug(f"PD zone: mid={midline:.2f} price={current_price:.2f} ({pct_from_mid:+.2%}) → {zone}")
    return PremiumDiscount(midline=midline, swing_high=swing_h, swing_low=swing_l, zone=zone, pct_from_mid=pct_from_mid)


# ── 5.3 OB / FVG / OTE / Volume Profile ──────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, cfg: dict) -> list[OrderBlock]:
    min_impulse = cfg.get("min_impulse_pct", 0.003)
    max_age = cfg.get("max_ob_age_candles", 50)
    obs: list[OrderBlock] = []
    start = max(0, len(df) - max_age)
    current_price = float(df.iloc[-1]["close"])

    for i in range(start, len(df) - 1):
        c0, c1 = df.iloc[i], df.iloc[i + 1]
        if _is_bearish(c0) and _is_bullish(c1):
            if (float(c1["close"]) - float(c1["open"])) / float(c1["open"]) >= min_impulse:
                obs.append(OrderBlock(Direction.LONG, float(c0["high"]), float(c0["low"]), i))
        elif _is_bullish(c0) and _is_bearish(c1):
            if (float(c1["open"]) - float(c1["close"])) / float(c1["open"]) >= min_impulse:
                obs.append(OrderBlock(Direction.SHORT, float(c0["high"]), float(c0["low"]), i))

    for ob in obs:
        if ob.direction == Direction.LONG and current_price < ob.low:
            ob.mitigated = True
        elif ob.direction == Direction.SHORT and current_price > ob.high:
            ob.mitigated = True

    return [ob for ob in obs if not ob.mitigated]


def detect_fvgs(df: pd.DataFrame, cfg: dict) -> list[FairValueGap]:
    min_gap = cfg.get("min_gap_pct", 0.001)
    fvgs: list[FairValueGap] = []
    for i in range(2, len(df)):
        c_prev2 = df.iloc[i - 2]
        c_curr = df.iloc[i]
        if float(c_prev2["high"]) < float(c_curr["low"]):
            gap_pct = (float(c_curr["low"]) - float(c_prev2["high"])) / float(c_prev2["high"])
            if gap_pct >= min_gap:
                fvgs.append(FairValueGap(Direction.LONG, float(c_curr["low"]), float(c_prev2["high"]), i))
        elif float(c_prev2["low"]) > float(c_curr["high"]):
            gap_pct = (float(c_prev2["low"]) - float(c_curr["high"])) / float(c_curr["high"])
            if gap_pct >= min_gap:
                fvgs.append(FairValueGap(Direction.SHORT, float(c_prev2["low"]), float(c_curr["high"]), i))
    return fvgs


def detect_ote(df: pd.DataFrame, cfg: dict, direction: Direction) -> Optional[OTEZone]:
    fib_level = cfg.get("fib_level", 0.705)
    fib_low = cfg.get("fib_low", 0.618)
    fib_high = cfg.get("fib_high", 0.79)
    swing_h, _ = _swing_high(df, 30)
    swing_l, _ = _swing_low(df, 30)
    rng = swing_h - swing_l
    if rng <= 0:
        return None
    if direction == Direction.LONG:
        ote_ideal = swing_h - fib_level * rng
        zone_high = swing_h - fib_low * rng
        zone_low = swing_h - fib_high * rng
    else:
        ote_ideal = swing_l + fib_level * rng
        zone_low = swing_l + fib_low * rng
        zone_high = swing_l + fib_high * rng
    return OTEZone(direction, ote_ideal, zone_low, zone_high, swing_h, swing_l)


def detect_volume_profile(df: pd.DataFrame, cfg: dict, current_price: float) -> VolumeProfile:
    bins = cfg.get("bins", 50)
    va_pct = cfg.get("value_area_pct", 0.70)
    prices = (df["high"] + df["low"] + df["close"]) / 3.0
    price_min, price_max = float(prices.min()), float(prices.max())
    if price_max == price_min:
        return VolumeProfile(price_min, price_min, price_min, True)

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_bins = np.zeros(bins)
    for idx in range(len(df)):
        tp = float(prices.iloc[idx])
        vol = float(df["volume"].iloc[idx])
        b = min(int((tp - price_min) / (price_max - price_min) * bins), bins - 1)
        vol_bins[b] += vol

    poc_idx = int(np.argmax(vol_bins))
    poc = float(bin_centers[poc_idx])
    total_vol = float(vol_bins.sum())
    target = total_vol * va_pct
    lo, hi = poc_idx, poc_idx
    accumulated = float(vol_bins[poc_idx])
    while accumulated < target and (lo > 0 or hi < bins - 1):
        add_lo = float(vol_bins[lo - 1]) if lo > 0 else 0.0
        add_hi = float(vol_bins[hi + 1]) if hi < bins - 1 else 0.0
        if add_lo >= add_hi and lo > 0:
            lo -= 1
            accumulated += add_lo
        elif hi < bins - 1:
            hi += 1
            accumulated += add_hi
        else:
            lo -= 1
            accumulated += add_lo

    val = float(bin_centers[lo])
    vah = float(bin_centers[hi])
    logger.debug(f"Volume Profile: POC={poc:.2f} VAH={vah:.2f} VAL={val:.2f}")
    return VolumeProfile(poc, vah, val, val <= current_price <= vah)


# ── 5.4 Liquidity structure ───────────────────────────────────────────────────

def detect_liquidity(df: pd.DataFrame, cfg: dict, direction: Direction) -> LiquidityState:
    lookback = cfg.get("lookback_candles", 30)
    tol = cfg.get("equal_level_tolerance_pct", 0.001)
    ret_candles = cfg.get("sweep_return_candles", 3)
    window = df.iloc[-lookback:]
    ssl_level, _ = _swing_low(window, lookback)
    bsl_level, _ = _swing_high(window, lookback)
    recent_lows = window["low"].iloc[-ret_candles:].values
    recent_highs = window["high"].iloc[-ret_candles:].values
    recent_close = float(window["close"].iloc[-1])
    ssl_swept = any(l < ssl_level * (1 - tol) for l in recent_lows) and recent_close > ssl_level
    bsl_swept = any(h > bsl_level * (1 + tol) for h in recent_highs) and recent_close < bsl_level
    dol = Direction.LONG if ssl_swept else (Direction.SHORT if bsl_swept else None)
    logger.debug(f"Liquidity: SSL={ssl_level:.2f} swept={ssl_swept} | BSL={bsl_level:.2f} swept={bsl_swept} | DOL={dol}")
    return LiquidityState(ssl_swept, bsl_swept, dol, ssl_level, bsl_level)


# ── 5.5 LTF Triggers ─────────────────────────────────────────────────────────

def detect_cisd(df_ltf: pd.DataFrame, direction: Direction) -> Optional[LTFTrigger]:
    if len(df_ltf) < 10:
        return None
    recent = df_ltf.iloc[-10:]
    if direction == Direction.LONG:
        highs = recent["high"].values
        for i in range(len(highs) - 2, 0, -1):
            if highs[i] < highs[i - 1] and float(recent["close"].iloc[-1]) > highs[i]:
                return LTFTrigger("cisd", Direction.LONG, highs[i], len(df_ltf) - 1)
    else:
        lows = recent["low"].values
        for i in range(len(lows) - 2, 0, -1):
            if lows[i] > lows[i - 1] and float(recent["close"].iloc[-1]) < lows[i]:
                return LTFTrigger("cisd", Direction.SHORT, lows[i], len(df_ltf) - 1)
    return None


def detect_mss(df_ltf: pd.DataFrame, direction: Direction) -> Optional[LTFTrigger]:
    if len(df_ltf) < 6:
        return None
    recent = df_ltf.iloc[-15:]
    close = float(recent["close"].iloc[-1])
    if direction == Direction.LONG:
        highs = recent["high"].values
        for i in range(len(highs) - 2, 1, -1):
            if highs[i] < highs[i - 1] and highs[i] < highs[i - 2] and close > highs[i]:
                return LTFTrigger("mss", Direction.LONG, highs[i], len(df_ltf) - 1)
    else:
        lows = recent["low"].values
        for i in range(len(lows) - 2, 1, -1):
            if lows[i] > lows[i - 1] and lows[i] > lows[i - 2] and close < lows[i]:
                return LTFTrigger("mss", Direction.SHORT, lows[i], len(df_ltf) - 1)
    return None


def detect_turtle_soup(df_ltf: pd.DataFrame, direction: Direction) -> Optional[LTFTrigger]:
    if len(df_ltf) < 22:
        return None
    window = df_ltf.iloc[-22:-2]
    last = df_ltf.iloc[-2]
    current = df_ltf.iloc[-1]
    if direction == Direction.LONG:
        prev_low = float(window["low"].min())
        if float(last["low"]) < prev_low and float(current["close"]) > prev_low:
            return LTFTrigger("turtle_soup", Direction.LONG, prev_low, len(df_ltf) - 1)
    else:
        prev_high = float(window["high"].max())
        if float(last["high"]) > prev_high and float(current["close"]) < prev_high:
            return LTFTrigger("turtle_soup", Direction.SHORT, prev_high, len(df_ltf) - 1)
    return None


def detect_ltf_fvg(df_ltf: pd.DataFrame, direction: Direction, cfg: dict) -> Optional[LTFTrigger]:
    min_gap = cfg.get("ltf_fvg_min_gap_pct", 0.0005)
    if len(df_ltf) < 3:
        return None
    c0, c1, c2 = df_ltf.iloc[-3], df_ltf.iloc[-2], df_ltf.iloc[-1]
    if direction == Direction.LONG:
        if float(c0["high"]) < float(c2["low"]):
            gap_pct = (float(c2["low"]) - float(c0["high"])) / float(c0["high"])
            if gap_pct >= min_gap:
                return LTFTrigger("ltf_fvg", Direction.LONG, (float(c0["high"]) + float(c2["low"])) / 2, len(df_ltf) - 1)
    else:
        if float(c0["low"]) > float(c2["high"]):
            gap_pct = (float(c0["low"]) - float(c2["high"])) / float(c2["high"])
            if gap_pct >= min_gap:
                return LTFTrigger("ltf_fvg", Direction.SHORT, (float(c0["low"]) + float(c2["high"])) / 2, len(df_ltf) - 1)
    return None


def collect_ltf_triggers(df_ltf: pd.DataFrame, direction: Direction, cfg: dict) -> list[LTFTrigger]:
    triggers: list[LTFTrigger] = []
    for fn in [
        lambda: detect_cisd(df_ltf, direction),
        lambda: detect_mss(df_ltf, direction),
        lambda: detect_turtle_soup(df_ltf, direction),
        lambda: detect_ltf_fvg(df_ltf, direction, cfg),
    ]:
        t = fn()
        if t:
            triggers.append(t)
            logger.debug(f"LTF trigger: {t.kind} @ {t.trigger_price:.4f}")
    return triggers


# ── 5.7 SMT Divergence ───────────────────────────────────────────────────────

def detect_smt(df_btc: pd.DataFrame, df_eth: pd.DataFrame, cfg: dict) -> SMTResult:
    lookback = cfg.get("lookback_candles", 10)
    if len(df_btc) < lookback or len(df_eth) < lookback:
        return SMTResult(False, None, None, False)
    btc = df_btc.iloc[-lookback:]
    eth = df_eth.iloc[-lookback:]
    btc_lo_new = float(btc["low"].iloc[-1]) < float(btc["low"].iloc[-2])
    btc_hi_new = float(btc["high"].iloc[-1]) > float(btc["high"].iloc[-2])
    eth_lo_new = float(eth["low"].iloc[-1]) < float(eth["low"].iloc[-2])
    eth_hi_new = float(eth["high"].iloc[-1]) > float(eth["high"].iloc[-2])
    bullish_div = btc_lo_new and not eth_lo_new
    bearish_div = btc_hi_new and not eth_hi_new
    divergence = bullish_div or bearish_div
    if divergence:
        logger.info(f"SMT divergence: bullish={bullish_div} bearish={bearish_div}")
    return SMTResult(
        divergence_found=divergence,
        btc_direction=Direction.LONG if not btc_lo_new else Direction.SHORT,
        eth_direction=Direction.LONG if not eth_lo_new else Direction.SHORT,
        grade_boost=divergence,
    )


# ── Main evaluator ────────────────────────────────────────────────────────────

class ICTSignalDetector:
    def __init__(self, config: dict):
        self.cfg = config
        self.sig_cfg = config["signals"]
        self.ltf_cfg = config["ltf_triggers"]
        self.sl_pct = config["trading"]["stop_loss_pct"]
        self.tp_pct = config["trading"]["take_profit_pct"]
        self.a_grade_res = self.sig_cfg.get("a_grade_resonances", 4)
        self.b_grade_res = self.sig_cfg.get("b_grade_resonances", 3)

    def evaluate(
        self,
        df_htf: pd.DataFrame,
        df_ltf: pd.DataFrame,
        df_4h: pd.DataFrame,
        df_daily: pd.DataFrame,
        df_eth: Optional[pd.DataFrame] = None,
    ) -> SignalResult:
        current_price = float(df_htf.iloc[-1]["close"])
        logger.debug(f"Evaluating signal @ {current_price:.4f}")

        # 5.1 HTF Bias
        htf_bias = detect_htf_bias(df_4h, df_daily, self.cfg.get("htf_bias", {}))
        if htf_bias.daily_choch:
            return SignalResult(grade=SignalGrade.NONE, direction=None, reason="Daily CHoCH – structural conflict, skip")
        if htf_bias.score < 2:
            return SignalResult(grade=SignalGrade.NONE, direction=None, reason=f"HTF Bias weak (score={htf_bias.score}/4)")

        bias_direction = htf_bias.direction

        # 5.2 Premium / Discount
        pd_zone = calculate_premium_discount(df_htf, self.cfg.get("premium_discount", {}), current_price)
        if pd_zone.zone == "discount" and bias_direction == Direction.SHORT:
            return SignalResult(grade=SignalGrade.NONE, direction=None, reason="PD conflict: discount but bias SHORT")
        if pd_zone.zone == "premium" and bias_direction == Direction.LONG:
            return SignalResult(grade=SignalGrade.NONE, direction=None, reason="PD conflict: premium but bias LONG")

        # 5.3 4× Resonance
        obs = detect_order_blocks(df_htf, self.sig_cfg["ob"])
        fvgs = detect_fvgs(df_htf, self.sig_cfg["fvg"])
        ote = detect_ote(df_htf, self.sig_cfg["ote"], bias_direction)
        vp = detect_volume_profile(df_htf, self.sig_cfg["volume_profile"], current_price)

        resonances = 0
        matched_ob: Optional[OrderBlock] = None
        matched_fvg: Optional[FairValueGap] = None
        conditions: list[str] = []

        for ob in obs:
            if ob.direction == bias_direction and ob.low <= current_price <= ob.high:
                matched_ob = ob; resonances += 1; conditions.append("OB"); break

        for fvg in fvgs:
            if fvg.direction == bias_direction and fvg.bottom <= current_price <= fvg.top:
                matched_fvg = fvg; resonances += 1; conditions.append("FVG"); break

        if ote and ote.zone_low <= current_price <= ote.zone_high:
            resonances += 1; conditions.append("OTE")

        if abs(current_price - vp.poc) / vp.poc <= 0.005 or vp.price_in_value_area:
            resonances += 1; conditions.append("VP-POC")

        if resonances < self.b_grade_res:
            return SignalResult(grade=SignalGrade.C, direction=bias_direction, resonances=resonances,
                                conditions_met=conditions, reason=f"C-grade: {resonances} resonances")
        if resonances < self.a_grade_res:
            return SignalResult(grade=SignalGrade.B, direction=bias_direction, resonances=resonances,
                                conditions_met=conditions, reason=f"B-grade: {resonances}/{self.a_grade_res} — skip")

        # 5.4 Liquidity
        liq = detect_liquidity(df_htf, self.cfg.get("liquidity", {}), bias_direction)
        liq_ok = (
            (bias_direction == Direction.LONG and liq.ssl_swept and liq.dol_direction == Direction.LONG)
            or
            (bias_direction == Direction.SHORT and liq.bsl_swept and liq.dol_direction == Direction.SHORT)
        )
        if not liq_ok:
            return SignalResult(grade=SignalGrade.B, direction=bias_direction, resonances=resonances,
                                conditions_met=conditions, reason="4 resonances but liquidity not swept",
                                ob=matched_ob, fvg=matched_fvg, ote=ote, volume_profile=vp, liquidity=liq)

        # 5.5 LTF Triggers
        ltf_triggers = collect_ltf_triggers(df_ltf, bias_direction, self.ltf_cfg)
        min_triggers = self.ltf_cfg.get("min_triggers_required", 2)
        if len(ltf_triggers) < min_triggers:
            return SignalResult(grade=SignalGrade.B, direction=bias_direction, resonances=resonances,
                                conditions_met=conditions, reason=f"Only {len(ltf_triggers)}/{min_triggers} LTF triggers",
                                ob=matched_ob, fvg=matched_fvg, ote=ote, volume_profile=vp, liquidity=liq)

        # 5.7 SMT
        smt_result = None
        if df_eth is not None and self.cfg.get("smt", {}).get("enabled", False):
            smt_result = detect_smt(df_htf, df_eth, self.cfg.get("smt", {}))

        # A-Grade
        entry = current_price
        sl = entry * (1 - self.sl_pct) if bias_direction == Direction.LONG else entry * (1 + self.sl_pct)
        tp = entry * (1 + self.tp_pct) if bias_direction == Direction.LONG else entry * (1 - self.tp_pct)

        all_conditions = conditions + ["Liq-OK", f"LTF:{','.join(t.kind for t in ltf_triggers)}"]
        if smt_result and smt_result.grade_boost:
            all_conditions.append("SMT-A+")

        grade_label = "A+ (SMT)" if (smt_result and smt_result.grade_boost) else "A"
        logger.info(f"{grade_label} {bias_direction.value.upper()} | res={resonances} | entry={entry:.4f} SL={sl:.4f} TP={tp:.4f}")

        return SignalResult(
            grade=SignalGrade.A, direction=bias_direction,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            htf_bias=htf_bias, pd_zone=pd_zone, ob=matched_ob, fvg=matched_fvg,
            ote=ote, volume_profile=vp, liquidity=liq, ltf_triggers=ltf_triggers,
            smt=smt_result, resonances=resonances, conditions_met=all_conditions,
            reason=f"{grade_label}: {resonances} resonances + liq + {len(ltf_triggers)} LTF",
        )
