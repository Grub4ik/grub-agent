"""Read-only operating system information tool."""

from __future__ import annotations

import platform


SYSTEM_INFO_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def get_system_info() -> dict[str, str]:
    return {
        "os": platform.system(),
        "version": platform.release(),
        "architecture": platform.machine(),
    }
