"""Explicit registry of available tools and their JSON Schema contracts."""

from __future__ import annotations

import re
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .models import Permission, ToolDefinition


class ToolRegistrationError(ValueError):
    """Raised when a tool definition is invalid or duplicated."""


class ToolRegistry:
    PERMISSIONS = {"read", "write", "restricted"}

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
        permission: Permission = "read",
    ) -> ToolDefinition:
        if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", name):
            raise ToolRegistrationError("Tool names must be 1-64 letters, digits, or underscores.")
        if name in self._tools:
            raise ToolRegistrationError(f"Tool {name!r} is already registered.")
        if permission not in self.PERMISSIONS:
            raise ToolRegistrationError(f"Unknown permission {permission!r}.")
        if not description.strip():
            raise ToolRegistrationError("Tool description cannot be empty.")
        if not callable(handler):
            raise ToolRegistrationError("Tool handler must be callable.")
        try:
            Draft202012Validator.check_schema(parameters)
        except Exception as exc:
            raise ToolRegistrationError(f"Invalid JSON Schema for {name!r}: {exc}") from exc

        definition = ToolDefinition(
            name=name,
            description=description.strip(),
            parameters=parameters,
            handler=handler,
            permission=permission,
        )
        self._tools[name] = definition
        return definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.as_ollama_schema() for tool in self._tools.values()]
