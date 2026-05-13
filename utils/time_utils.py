"""
Trading session time utilities.
All times in config are Taiwan time (Asia/Taipei, UTC+8).

Silver Bullet = ICT 10:00–12:00 New York local time (DST-aware):
  Summer (EDT, UTC-4): 22:00–00:00 Taiwan
  Winter (EST, UTC-5): 23:00–01:00 Taiwan
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from utils.logger import get_logger

logger = get_logger(__name__)

_NY_TZ       = ZoneInfo("America/New_York")
_SB_NY_START = time(10, 0)
_SB_NY_END   = time(12, 0)


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
        self.pre_market_start   = _parse_time(sess["pre_market"]["start"])
        self.pre_market_end     = _parse_time(sess["pre_market"]["end"])
        self.mid_session_start  = _parse_time(sess["mid_session"]["start"])
        self.mid_session_end    = _parse_time(sess["mid_session"]["end"])
        self.force_close_time   = _parse_time(sess["force_close"])
        self.no_new_entry_after = _parse_time(sess.get("no_new_entry_after", "02:00"))

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def now_ny(self) -> datetime:
        return datetime.now(_NY_TZ)

    def current_time(self) -> time:
        return self.now().time()

    def is_pre_market(self) -> bool:
        return _between(self.current_time(), self.pre_market_start, self.pre_market_end)

    def is_mid_session(self) -> bool:
        return _between(self.current_time(), self.mid_session_start, self.mid_session_end)

    def is_silver_bullet(self) -> bool:
        """True when NY local time is 10:00–12:00 — handles EDT/EST automatically."""
        return _SB_NY_START <= self.now_ny().time() < _SB_NY_END

    def is_force_close(self) -> bool:
        now = self.current_time()
        fc = self.force_close_time
        return now.hour == fc.hour and now.minute == fc.minute

    def is_after_force_close(self) -> bool:
        return self.current_time() >= self.force_close_time

    def is_after_no_new_entry(self) -> bool:
        """No new positions after 02:00 Taiwan time (safety net)."""
        return self.current_time() >= self.no_new_entry_after

    def current_session(self) -> str:
        if self.is_pre_market(): return "pre_market"
        if self.is_mid_session(): return "mid_session"
        if self.is_silver_bullet(): return "silver_bullet"
        return "off_hours"

    def seconds_until_silver_bullet(self) -> float:
        if self.is_silver_bullet():
            return 0.0
        now_ny = self.now_ny()
        target = now_ny.replace(hour=_SB_NY_START.hour, minute=0, second=0, microsecond=0)
        if target <= now_ny:
            target += timedelta(days=1)
        return (target - now_ny).total_seconds()

    def log_status(self):
        ny_str = self.now_ny().strftime('%H:%M %Z')
        logger.info(
            f"Session: {self.current_session()} | "
            f"TW: {self.now().strftime('%H:%M:%S %Z')} | NY: {ny_str}"
        )
