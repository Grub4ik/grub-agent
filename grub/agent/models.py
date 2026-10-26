"""Data structures shared by the registry, executor, and agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


Permission = Literal["read", "write", "restricted"]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any] = field(repr=False, compare=False)
    permission: Permission = "read"

    def as_ollama_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Any

    @classmethod
    def from_ollama(cls, value: Any) -> "ToolCall":
        if not isinstance(value, dict):
            return cls(name="<invalid>", arguments=None)
        function = value.get("function")
        if not isinstance(function, dict):
            return cls(name="<invalid>", arguments=None)
        name = function.get("name")
        arguments = function.get("arguments")
        return cls(name=name if isinstance(name, str) else "<invalid>", arguments=arguments)


@dataclass(frozen=True)
class ToolResult:
    tool: str
    success: bool
    data: Any = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "success": self.success,
            "data": self.data,
            "error": self.error,
        }


@dataclass(frozen=True)
class AgentResponse:
    text: str | None
    llm_seconds: float
    tool_calls: int = 0
    tool_results: tuple[ToolResult, ...] = ()
    complete: bool = True
