"""HTTP client and explicit tool schema for the local SmartLED API."""

from __future__ import annotations

import ipaddress
import json
import math
from functools import partial
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from grub.agent.registry import ToolRegistry


SMARTLED_SET_COLOR_SCHEMA = {
    "type": "object",
    "properties": {
        "r": {"type": "integer", "minimum": 0, "maximum": 255},
        "g": {"type": "integer", "minimum": 0, "maximum": 255},
        "b": {"type": "integer", "minimum": 0, "maximum": 255},
    },
    "required": ["r", "g", "b"],
    "additionalProperties": False,
}


class SmartLEDAPIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout: float = 4.0) -> None:
        self.base_url = self._validated_local_url(base_url)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("SmartLED API timeout must be a finite positive number.")
        self.timeout = timeout

    @staticmethod
    def _validated_local_url(base_url: str) -> str:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("SmartLED API URL must be a local HTTP URL.")
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("SmartLED API URL must contain only the local server origin.")
        try:
            _ = parsed.port
        except ValueError as error:
            raise ValueError("SmartLED API URL has an invalid port.") from error
        hostname = parsed.hostname.lower()
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = hostname == "localhost"
        if not is_loopback:
            raise ValueError("SmartLED API must be hosted on this computer (localhost).")
        return base_url.rstrip("/")

    def set_color(self, red: int, green: int, blue: int) -> dict[str, str]:
        payload = json.dumps({
            "mode": "static",
            "color": {"r": red, "g": green, "b": blue},
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/v1/light",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="PUT",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return self._convert_response(response.status, response.read())
        except HTTPError as error:
            try:
                body = error.read()
            finally:
                error.close()
            return self._convert_response(error.code, body)
        except (URLError, TimeoutError, OSError):
            return {
                "status": "failed",
                "message": "Не удалось связаться с локальным SmartLED API. Проверьте, что сервер запущен.",
            }

    @staticmethod
    def _convert_response(http_status: int, body: bytes) -> dict[str, str]:
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = {}
        if (
            http_status == 200
            and isinstance(data, dict)
            and data.get("success") is True
            and data.get("status") == "confirmed"
        ):
            return {"status": "confirmed", "message": "Статический цвет подсветки применён."}
        if http_status == 202 and isinstance(data, dict) and data.get("status") == "sent_unconfirmed":
            return {
                "status": "sent_unconfirmed",
                "message": "Команда отправлена, но контроллер не подтвердил применение цвета.",
            }
        if http_status == 422:
            message = "Локальный SmartLED API отклонил цвет или формат запроса."
        elif http_status == 503:
            message = "В API не настроен адрес SmartLED."
        elif http_status == 504:
            message = "SmartLED не ответил вовремя."
        elif http_status == 502:
            message = "Контроллер недоступен или отклонил команду."
        else:
            message = "Не удалось изменить цвет: локальный SmartLED API вернул ошибку."
        return {"status": "failed", "message": message}


def smartled_set_color(
    r: int,
    g: int,
    b: int,
    *,
    client: SmartLEDAPIClient,
) -> dict[str, str]:
    """Set a static RGB color through the separately running local SmartLED API."""
    return client.set_color(r, g, b)


def register_smartled_tool(registry: ToolRegistry, client: SmartLEDAPIClient) -> None:
    registry.register(
        name="smartled_set_color",
        description=(
            "Set a static RGB color for the room's SmartLED strip. Choose r, g, and b "
            "from the user's requested color. This changes the device. Inspect the result: "
            "only status 'confirmed' means it was applied; for 'sent_unconfirmed' say it was "
            "sent but not confirmed, and for 'failed' explain that it did not succeed. "
            "After a confirmed change, reply briefly in the user's language without reading JSON."
        ),
        parameters=SMARTLED_SET_COLOR_SCHEMA,
        handler=partial(smartled_set_color, client=client),
        permission="write",
    )
