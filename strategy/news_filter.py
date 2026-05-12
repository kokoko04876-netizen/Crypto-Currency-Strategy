"""
News filter — blocks trading within ±4H of FOMC / CPI events.

Events stored in data/news_events.json (user-maintained):
[
  {"date": "2026-05-07", "time": "02:00", "event": "FOMC", "impact": "high"},
  {"date": "2026-05-13", "time": "20:30", "event": "CPI",  "impact": "high"}
]
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from utils.logger import get_logger

logger = get_logger(__name__)

_HIGH_IMPACT = {"fomc", "cpi", "nfp", "pce", "fed"}


def _load_events(path: str) -> list[dict]:
    if not os.path.exists(path):
        logger.warning(f"News events file not found: {path} — no news filter active")
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load news events: {e}")
        return []


def is_blocked_by_news(cfg: dict, tz: ZoneInfo) -> tuple[bool, str]:
    block_hours = cfg.get("block_window_hours", 4)
    events_file = cfg.get("events_file", "data/news_events.json")
    events = _load_events(events_file)
    now = datetime.now(tz)

    for ev in events:
        event_name = ev.get("event", "").lower()
        impact = ev.get("impact", "").lower()
        if impact != "high" and not any(k in event_name for k in _HIGH_IMPACT):
            continue
        try:
            ev_dt = datetime.strptime(f"{ev['date']} {ev['time']}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        except Exception:
            continue
        delta = abs((now - ev_dt).total_seconds()) / 3600
        if delta <= block_hours:
            reason = f"News block: {ev.get('event')} at {ev['date']} {ev['time']} (±{block_hours}H, now {delta:.1f}H away)"
            logger.warning(reason)
            return True, reason

    return False, "OK"


def create_sample_events_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    sample = [
        {"date": "2026-06-11", "time": "02:00", "event": "FOMC", "impact": "high"},
        {"date": "2026-06-13", "time": "20:30", "event": "CPI",  "impact": "high"},
    ]
    with open(path, "w") as f:
        json.dump(sample, f, indent=2)
    logger.info(f"Sample news events file created: {path}")
