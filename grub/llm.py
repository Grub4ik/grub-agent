"""Small persistent-connection client for Ollama's local chat API."""

from __future__ import annotations

import http.client
import json
import time
from urllib.parse import urlsplit
from typing import Any


class OllamaError(RuntimeError):
    """Raised for Ollama connectivity, model, and response errors."""


class OllamaClient:
    def __init__(
        self,
        model: str = "gemma4:12b",
        base_url: str = "http://localhost:11434",
        system_prompt: str = "",
        history_turns: int = 10,
        timeout: float = 120.0,
        keep_alive: str = "10m",
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Ollama URL должен начинаться с http:// или https://")
        if history_turns < 1:
            raise ValueError("history_turns должен быть не меньше 1")

        self.model = model
        self.system_prompt = system_prompt
        self.history_turns = history_turns
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._prefix = parsed.path.rstrip("/")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self._connection = connection_class(parsed.hostname, port, timeout=timeout)
        self._history: list[dict[str, Any]] = []

    def check_ready(self) -> None:
        try:
            result = self._request("GET", "/api/tags")
        except OllamaError as exc:
            raise OllamaError(
                f"Ollama недоступен по адресу /api/tags: {exc}. Запустите Ollama и проверьте GRUB_OLLAMA_URL."
            ) from exc

        model_items = result.get("models", [])
        if not isinstance(model_items, list):
            raise OllamaError("Ollama вернула некорректный список моделей.")
        available = {
            item.get("name", "")
            for item in model_items
            if isinstance(item, dict)
        }
        if self.model not in available:
            raise OllamaError(
                f"Модель {self.model!r} не найдена в Ollama. Выполните: ollama pull {self.model}"
            )

    def chat(self, user_text: str) -> tuple[str, float]:
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("Пустой текст нельзя отправлять в Ollama.")

        messages: list[dict[str, Any]] = [*self._history, {"role": "user", "content": user_text}]
        message, elapsed = self.chat_messages(messages)
        answer = str(message.get("content") or "").strip()
        if not answer:
            raise OllamaError("Ollama вернула пустой ответ.")

        self._history.extend(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ]
        )
        self._history = self._history[-self.history_turns * 2 :]
        return answer, elapsed

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], float]:
        """Send full Ollama chat history, including native tool messages."""
        request_messages: list[dict[str, Any]] = []
        if self.system_prompt:
            request_messages.append({"role": "system", "content": self.system_prompt})
        request_messages.extend(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
        }
        if tools is not None:
            payload["tools"] = tools

        started = time.perf_counter()
        result = self._request("POST", "/api/chat", payload)
        try:
            elapsed = float(result.get("total_duration") or 0) / 1_000_000_000
        except (TypeError, ValueError):
            elapsed = 0.0
        if elapsed <= 0:
            elapsed = time.perf_counter() - started
        message = result.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama вернула ответ без корректного message.")
        return message, elapsed

    def close(self) -> None:
        self._connection.close()

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            self._connection.request(method, self._prefix + path, body=body, headers=headers)
            response = self._connection.getresponse()
            content = response.read()
            if response.status >= 400:
                detail = content.decode("utf-8", errors="replace")
                raise OllamaError(f"HTTP {response.status}: {detail}")
            try:
                value = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OllamaError("Ollama вернула некорректный JSON.") from exc
            if not isinstance(value, dict):
                raise OllamaError("Ollama вернула неожиданный формат ответа.")
            return value
        except OllamaError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            self._connection.close()
            if isinstance(self._connection, http.client.HTTPSConnection):
                cls = http.client.HTTPSConnection
            else:
                cls = http.client.HTTPConnection
            parsed_port = self._connection.port
            self._connection = cls(self._connection.host, parsed_port, timeout=self.timeout)
            raise OllamaError(str(exc)) from exc
