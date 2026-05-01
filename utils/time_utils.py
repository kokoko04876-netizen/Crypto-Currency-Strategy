"""
Trading session time utilities.
All times in the config are expressed in Taiwan time (Asia/Taipei, UTC+8).
"""
from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from utils.logger import get_logger

logger = get_logger(__name__)


def _now_local(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))


def _parse_time(t: str) -> time:
    """Parse "HH:MM" string to time object."""
    h, m = t.split(":")
    return time(int(h), int(m))


class SessionManager:
    def __init__(self, config: dict):
        sess = config["sessions"]
        self.tz = ZoneInfo(sess.get("timezone", "Asia/Taipei"))
        self.pre_market_start = _parse_time(sess["pre_market"]["start"])
        self.pre_market_end = _parse_time(sess["pre_market"]["end"])
        self.macro_start = _parse_time(sess["macro_window"]["start"])
        self.macro_end = _parse_time(sess["macro_window"]["end"])
        self.sb_start = _parse_time(sess["silver_bullet"]["start"])
        self.sb_end = _parse_time(sess["silver_bullet"]["end"])
        self.force_close_time = _parse_time(sess["force_close"])

    def now(self) -> datetime:
        return datetime.now(self.tz)

    def current_time(self) -> time:
        return self.now().time()

    def _between(self, t: time, start: time, end: time) -> bool:
        if start <= end:
            return start <= t < end
        # crosses midnight
        return t >= start or t < end

    def is_pre_market(self) -> bool:
        return self._between(self.current_time(), self.pre_market_start, self.pre_market_end)

    def is_macro_window(self) -> bool:
        return self._between(self.current_time(), self.macro_start, self.macro_end)

    def is_silver_bullet(self) -> bool:
        return self._between(self.current_time(), self.sb_start, self.sb_end)

    def is_force_close(self) -> bool:
        """True within 1 minute of the force-close time."""
        now = self.current_time()
        fc = self.force_close_time
        # within the same minute
        return now.hour == fc.hour and now.minute == fc.minute

    def is_after_force_close(self) -> bool:
        now = self.current_time()
        fc = self.force_close_time
        if fc.hour == 0 and fc.minute == 0:
            return False
        return now >= fc

    def current_session(self) -> str:
        if self.is_pre_market():
            return "pre_market"
        if self.is_macro_window():
            return "macro_window"
        if self.is_silver_bullet():
            return "silver_bullet"
        return "off_hours"

    def seconds_until_silver_bullet(self) -> float:
        """Seconds until Silver Bullet window opens. 0 if already in window."""
        if self.is_silver_bullet():
            return 0.0
        now = self.now()
        target = now.replace(
            hour=self.sb_start.hour, minute=self.sb_start.second, second=0, microsecond=0
        )
        if target <= now:
            from datetime import timedelta
            target += timedelta(days=1)
        return (target - now).total_seconds()

    def log_status(self):
        session = self.current_session()
        logger.info(f"Session: {session} | Local time: {self.now().strftime('%H:%M:%S %Z')}")
