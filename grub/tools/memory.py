"""Tool contracts for explicit and automatic management of saved user facts."""

from __future__ import annotations

from typing import Any

from ..memory import FactMemory


MEMORY_SAVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fact": {"type": "string", "minLength": 1, "maxLength": 1000},
        "category": {
            "type": "string",
            "enum": ["general", "preference", "project", "possession", "person"],
        },
    },
    "required": ["fact"],
    "additionalProperties": False,
}

MEMORY_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 500}},
    "required": ["query"],
    "additionalProperties": False,
}

MEMORY_LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

MEMORY_FORGET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"fact_id": {"type": "string", "minLength": 1, "maxLength": 64}},
    "required": ["fact_id"],
    "additionalProperties": False,
}


def save_user_fact(memory: FactMemory, fact: str, category: str = "general") -> dict[str, Any]:
    """Store one concise, user-provided fact in persistent semantic memory."""
    return memory.save_fact(fact, category)


def search_user_facts(memory: FactMemory, query: str) -> dict[str, Any]:
    """Find saved facts semantically related to a natural-language query."""
    return {"facts": memory.search_facts(query)}


def list_user_facts(memory: FactMemory) -> dict[str, Any]:
    """Return the currently saved facts so the user can review them."""
    return {"facts": memory.list_facts()}


def forget_user_fact(memory: FactMemory, fact_id: str) -> dict[str, Any]:
    """Delete one saved fact by its ID."""
    return memory.forget_fact(fact_id)
