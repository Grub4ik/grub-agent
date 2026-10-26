"""Small integer addition tool used to exercise argument validation."""

from __future__ import annotations


CALCULATE_SUM_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "integer", "minimum": -1_000_000_000, "maximum": 1_000_000_000},
        "b": {"type": "integer", "minimum": -1_000_000_000, "maximum": 1_000_000_000},
    },
    "required": ["a", "b"],
    "additionalProperties": False,
}


def calculate_sum(a: int, b: int) -> dict[str, int]:
    return {"result": a + b}
