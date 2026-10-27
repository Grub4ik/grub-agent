"""Tool implementations registered explicitly by the Grub entry point."""

from .calculate_sum import CALCULATE_SUM_SCHEMA, calculate_sum
from .discover_smartled import (
    SMARTLED_DISCOVERY_SCHEMA,
    SmartLEDDiscoveryError,
    discover_smartled,
)
from .system_info import SYSTEM_INFO_SCHEMA, get_system_info

__all__ = [
    "CALCULATE_SUM_SCHEMA",
    "SMARTLED_DISCOVERY_SCHEMA",
    "SYSTEM_INFO_SCHEMA",
    "SmartLEDDiscoveryError",
    "calculate_sum",
    "discover_smartled",
    "get_system_info",
]
