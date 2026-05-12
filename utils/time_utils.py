"""
Trading session time utilities.
All times in config are Taiwan time (Asia/Taipei, UTC+8).
Silver Bullet: 22:00–00:00 (crosses midnight).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from utils.logger import get_logger

logger = get_logger(__name__)


def _parse_time(t: str) -> time:
    h, m = t.split(":")
    return time(int(h), int(m))


def _between(now: time, start: time, end: time) -> bool:
    """Handle ranges that cross midnight (e.g. 22:00–00:00)."""
    if start == end:
        return False
    if end == time(0, 0):
        return now >= start
    if start < end:
        return start <= now < end
    return now >= start or now < end


class SessionManager:
    def __init__(self, config: dict):
        sess = config["sessions"]
        self.tz = ZoneInfo(sess.get("timezone", "Asia/Taipei"))
        self.pre_market_start  = _parse_time(sess["pre_market"]["start"])
        self.pre_market_end    = _parse_time(sess["pre_market"]["end"])
        self.mid_session_start = _parse_time(sess["mid_session"]["start"])
        self.mid_session_end   = _parse_time(sess["mid_session"]["end"])
        self.sb_start          = _parse_time(sess["silver_bullet"]["start"])
        self.sb_end            = _parse_time(sess["silver_bullet"]["end"])
        self.force_close_time  = _parse_time(sess["force_close"])

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def current_time(self) -> time:
        return self.now().time()

    def is_pre_market(self) -> bool:
        return _between(self.current_time(), self.pre_market_start, self.pre_market_end)

    def is_mid_session(self) -> bool:
        return _between(self.current_time(), self.mid_session_start, self.mid_session_end)

    def is_silver_bullet(self) -> bool:
        return _between(self.current_time(), self.sb_start, self.sb_end)

    def is_force_close(self) -> bool:
        now = self.current_time()
        fc = self.force_close_time
        return now.hour == fc.hour and now.minute == fc.minute

    def is_after_force_close(self) -> bool:
        return self.current_time() >= self.force_close_time

    def current_session(self) -> str:
        if self.is_pre_market(): return "pre_market"
        if self.is_mid_session(): return "mid_session"
        if self.is_silver_bullet(): return "silver_bullet"
        return "off_hours"

    def seconds_until_silver_bullet(self) -> float:
        if self.is_silver_bullet():
            return 0.0
        now_dt = self.now()
        target = now_dt.replace(hour=self.sb_start.hour, minute=self.sb_start.minute, second=0, microsecond=0)
        if target <= now_dt:
            target += timedelta(days=1)
        return (target - now_dt).total_seconds()

    def log_status(self):
        logger.info(f"Session: {self.current_session()} | Local: {self.now().strftime('%H:%M:%S %Z')}")
