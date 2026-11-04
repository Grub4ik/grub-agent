"""Read the current local date and time from the system clock."""

from __future__ import annotations

from datetime import datetime
from typing import Any


CURRENT_TIME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def get_current_time() -> dict[str, str]:
    """Return local date/time, timezone name, and UTC offset as an ISO 8601 value."""
    now = datetime.now().astimezone()
    return {
        "datetime": now.isoformat(sep=" ", timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "timezone": now.tzname() or "local time zone",
        "utc_offset": now.strftime("%z"),
    }
