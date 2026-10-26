"""Tool implementations registered explicitly by the Grub entry point."""

from .calculate_sum import CALCULATE_SUM_SCHEMA, calculate_sum
from .system_info import SYSTEM_INFO_SCHEMA, get_system_info
from .smartled import (
    SMARTLED_SET_COLOR_SCHEMA,
    SmartLEDAPIClient,
    register_smartled_tool,
    smartled_set_color,
)

__all__ = [
    "CALCULATE_SUM_SCHEMA",
    "SYSTEM_INFO_SCHEMA",
    "SMARTLED_SET_COLOR_SCHEMA",
    "SmartLEDAPIClient",
    "register_smartled_tool",
    "calculate_sum",
    "get_system_info",
]
