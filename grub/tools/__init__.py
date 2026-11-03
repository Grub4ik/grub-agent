"""Tool implementations registered explicitly by the Grub entry point."""

from .calculate_sum import CALCULATE_SUM_SCHEMA, calculate_sum
from .discover_smartled import (
    SMARTLED_DISCOVERY_SCHEMA,
    SmartLEDDiscoveryError,
    discover_smartled,
)
from .system_info import SYSTEM_INFO_SCHEMA, get_system_info
from .memory import (
    MEMORY_FORGET_SCHEMA,
    MEMORY_LIST_SCHEMA,
    MEMORY_SAVE_SCHEMA,
    MEMORY_SEARCH_SCHEMA,
    forget_user_fact,
    list_user_facts,
    save_user_fact,
    search_user_facts,
)
from .smartled import (
    SMARTLED_MUSIC_SCHEMA,
    SMARTLED_RAINBOW_SCHEMA,
    SMARTLED_SPECTRUM_SCHEMA,
    SMARTLED_STATIC_SCHEMA,
    SMARTLED_STOP_AUDIO_SCHEMA,
    SmartLEDCommandError,
    SmartLEDTools,
)
from .youtube import YOUTUBE_PLAY_SCHEMA, play_youtube_video

__all__ = [
    "CALCULATE_SUM_SCHEMA",
    "SMARTLED_DISCOVERY_SCHEMA",
    "SMARTLED_MUSIC_SCHEMA",
    "SMARTLED_RAINBOW_SCHEMA",
    "SMARTLED_SPECTRUM_SCHEMA",
    "SMARTLED_STATIC_SCHEMA",
    "SMARTLED_STOP_AUDIO_SCHEMA",
    "SYSTEM_INFO_SCHEMA",
    "YOUTUBE_PLAY_SCHEMA",
    "MEMORY_FORGET_SCHEMA",
    "MEMORY_LIST_SCHEMA",
    "MEMORY_SAVE_SCHEMA",
    "MEMORY_SEARCH_SCHEMA",
    "SmartLEDCommandError",
    "SmartLEDDiscoveryError",
    "SmartLEDTools",
    "calculate_sum",
    "discover_smartled",
    "get_system_info",
    "play_youtube_video",
    "forget_user_fact",
    "list_user_facts",
    "save_user_fact",
    "search_user_facts",
]
