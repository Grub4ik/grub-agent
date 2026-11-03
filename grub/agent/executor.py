"""Validates permissions and arguments before dispatching a tool handler."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from .models import ToolCall, ToolResult
from .registry import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        allowed_permissions: Iterable[str] = ("read",),
        allowed_tool_permissions: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self.registry = registry
        self.allowed_permissions = frozenset(allowed_permissions)
        self.allowed_tool_permissions = {
            permission: frozenset(tool_names)
            for permission, tool_names in (allowed_tool_permissions or {}).items()
        }

    def execute(self, call: ToolCall | dict[str, Any]) -> ToolResult:
        if isinstance(call, dict):
            call = ToolCall.from_ollama(call)
        if not isinstance(call, ToolCall):
            return ToolResult("<invalid>", False, error="Invalid tool call.")

        definition = self.registry.get(call.name)
        if definition is None:
            return ToolResult(call.name, False, error="Tool not found.")
        tool_permissions = self.allowed_tool_permissions.get(definition.permission, frozenset())
        if definition.permission not in self.allowed_permissions and call.name not in tool_permissions:
            return ToolResult(
                call.name,
                False,
                error=f"Permission {definition.permission!r} is denied.",
            )

        validator = Draft202012Validator(definition.parameters)
        errors = sorted(
            validator.iter_errors(call.arguments),
            key=lambda item: tuple(str(part) for part in item.path),
        )
        if errors:
            first = errors[0]
            location = ".".join(str(part) for part in first.path) or "arguments"
            return ToolResult(
                call.name,
                False,
                error=f"Invalid arguments at {location}: {first.message}",
            )

        try:
            data = definition.handler(**call.arguments)
            # A tool result must be serializable before it can be returned to Ollama.
            json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            return ToolResult(
                call.name,
                False,
                error=f"Tool execution failed: {type(exc).__name__}: {exc}",
            )
        return ToolResult(call.name, True, data=data)
