"""Bounded, sequential agent loop using Ollama's native tool-calling API."""

from __future__ import annotations

import json
import threading
from typing import Any

from ..llm import OllamaClient
from .executor import ToolExecutor
from .models import AgentResponse, ToolCall, ToolResult
from .registry import ToolRegistry


class AgentCore:
    MAX_TOOL_ROUNDS = 5
    MAX_TOOL_CALLS = 8

    def __init__(
        self,
        llm: OllamaClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        history_turns: int = 10,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        if history_turns < 1:
            raise ValueError("history_turns must be at least 1.")
        if max_tool_rounds < 1 or max_tool_calls < 1:
            raise ValueError("Tool-call limits must be positive.")
        self.llm = llm
        self.registry = registry
        self.executor = executor
        self.history_turns = history_turns
        self.max_tool_rounds = max_tool_rounds
        self.max_tool_calls = max_tool_calls
        self._history: list[list[dict[str, Any]]] = []
        self._lock = threading.Lock()

    def run(self, user_text: str) -> AgentResponse:
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("Empty user text cannot be sent to the agent.")

        with self._lock:
            prior_messages = [message for turn in self._history for message in turn]
            messages: list[dict[str, Any]] = [*prior_messages, {"role": "user", "content": user_text}]
            turn_start = len(prior_messages)
            tool_results: list[ToolResult] = []
            unresolved_errors: dict[str, str] = {}
            llm_seconds = 0.0
            calls_used = 0
            rounds_used = 0

            while True:
                raw_message, elapsed = self.llm.chat_messages(messages, tools=self.registry.schemas())
                llm_seconds += elapsed
                assistant_message = self._assistant_message(raw_message)
                messages.append(assistant_message)
                raw_calls = assistant_message.get("tool_calls") or []

                if not raw_calls:
                    text = str(assistant_message.get("content") or "").strip() or None
                    if unresolved_errors:
                        text = self._failure_summary(unresolved_errors)
                        messages.append({"role": "assistant", "content": text})
                    self._save_turn(messages[turn_start:])
                    return AgentResponse(
                        text=text,
                        llm_seconds=llm_seconds,
                        tool_calls=calls_used,
                        tool_results=tuple(tool_results),
                    )

                if rounds_used >= self.max_tool_rounds:
                    for raw_call in raw_calls:
                        call = ToolCall.from_ollama(raw_call)
                        result = ToolResult(
                            call.name,
                            False,
                            error="Maximum tool-call rounds reached; this call was not executed.",
                        )
                        tool_results.append(result)
                        unresolved_errors[call.name] = result.error or "Tool call limit reached."
                        messages.append(self._tool_message(result))
                    text = self._limit_summary("rounds", rounds_used)
                    messages.append({"role": "assistant", "content": text})
                    self._save_turn(messages[turn_start:])
                    return AgentResponse(text, llm_seconds, calls_used, tuple(tool_results), complete=False)

                rounds_used += 1
                call_limit_reached = False
                for raw_call in raw_calls:
                    call = ToolCall.from_ollama(raw_call)
                    if calls_used >= self.max_tool_calls:
                        result = ToolResult(
                            call.name,
                            False,
                            error="Maximum tool-call count reached; this call was not executed.",
                        )
                        call_limit_reached = True
                    else:
                        calls_used += 1
                        result = self.executor.execute(call)

                    tool_results.append(result)
                    if result.success:
                        unresolved_errors.pop(result.tool, None)
                    else:
                        unresolved_errors[result.tool] = result.error or "Tool failed."
                    messages.append(self._tool_message(result))

                if call_limit_reached:
                    text = self._limit_summary("calls", calls_used)
                    messages.append({"role": "assistant", "content": text})
                    self._save_turn(messages[turn_start:])
                    return AgentResponse(text, llm_seconds, calls_used, tuple(tool_results), complete=False)

    def _save_turn(self, messages: list[dict[str, Any]]) -> None:
        self._history.append(messages)
        self._history = self._history[-self.history_turns :]

    @staticmethod
    def _assistant_message(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"role": "assistant", "content": "", "tool_calls": []}
        message: dict[str, Any] = {"role": "assistant", "content": str(value.get("content") or "")}
        tool_calls = value.get("tool_calls")
        if isinstance(tool_calls, list):
            message["tool_calls"] = tool_calls
        return message

    @staticmethod
    def _tool_message(result: ToolResult) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_name": result.tool,
            "content": json.dumps(result.as_dict(), ensure_ascii=False),
        }

    @staticmethod
    def _failure_summary(errors: dict[str, str]) -> str:
        details = "; ".join(f"{name}: {error}" for name, error in errors.items())
        return f"Не удалось подтвердить результат инструмента. {details}"

    @staticmethod
    def _limit_summary(kind: str, used: int) -> str:
        label = "раундов" if kind == "rounds" else "вызовов"
        return f"Не удалось завершить запрос: достигнут лимит инструментов ({used} {label})."
