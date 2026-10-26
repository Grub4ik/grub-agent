from __future__ import annotations

from functools import partial
from io import BytesIO
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from grub.agent import ToolCall, ToolExecutor, ToolRegistry
from grub.tools.smartled import (
    SMARTLED_SET_COLOR_SCHEMA,
    SmartLEDAPIClient,
    register_smartled_tool,
    smartled_set_color,
)


class FakeResponse:
    def __init__(self, status: int, body: dict):
        self.status = status
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self._body


class SmartLEDClientTests(unittest.TestCase):
    def setUp(self):
        self.client = SmartLEDAPIClient()

    def test_confirmed_http_response_becomes_confirmed_tool_result(self):
        response = FakeResponse(200, {"success": True, "status": "confirmed"})
        with patch("grub.tools.smartled.urlopen", return_value=response) as open_url:
            result = self.client.set_color(128, 0, 255)

        request = open_url.call_args.args[0]
        self.assertEqual(request.method, "PUT")
        self.assertEqual(request.full_url, "http://127.0.0.1:8765/api/v1/light")
        self.assertEqual(json.loads(request.data), {
            "mode": "static", "color": {"r": 128, "g": 0, "b": 255}
        })
        self.assertEqual(result["status"], "confirmed")

    def test_unconfirmed_http_202_is_not_reported_as_applied(self):
        with patch("grub.tools.smartled.urlopen", return_value=FakeResponse(
            202, {"success": False, "status": "sent_unconfirmed"}
        )):
            result = self.client.set_color(1, 2, 3)
        self.assertEqual(result["status"], "sent_unconfirmed")

    def test_http_error_is_converted_to_safe_failed_result(self):
        error = HTTPError(
            "http://127.0.0.1:8765/api/v1/light", 502, "Bad gateway", {},
            BytesIO(b'{"success":false,"status":"failed","message":"device down"}'),
        )
        with patch("grub.tools.smartled.urlopen", side_effect=error):
            result = self.client.set_color(1, 2, 3)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["message"], "Контроллер недоступен или отклонил команду.")

    def test_unavailable_api_is_converted_to_failed_result(self):
        with patch("grub.tools.smartled.urlopen", side_effect=URLError("offline")):
            result = self.client.set_color(1, 2, 3)
        self.assertEqual(result["status"], "failed")
        self.assertIn("локальным SmartLED API", result["message"])

    def test_api_url_must_be_local_and_cannot_be_model_supplied(self):
        for url in ("https://127.0.0.1", "http://192.168.1.8:8765", "http://localhost/admin"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                SmartLEDAPIClient(url)


class SmartLEDToolTests(unittest.TestCase):
    def make_registry(self, client=None):
        registry = ToolRegistry()
        register_smartled_tool(registry, client or SmartLEDAPIClient())
        registry.register(
            "another_write",
            "A separate write operation.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            lambda: {"done": True},
            permission="write",
        )
        return registry

    def test_tool_schema_requires_three_bounded_integer_channels(self):
        registry = self.make_registry()
        schema = registry.get("smartled_set_color").parameters
        self.assertEqual(set(schema["required"]), {"r", "g", "b"})
        self.assertFalse(schema["additionalProperties"])
        executor = ToolExecutor(registry, allowed_write_tools={"smartled_set_color"})
        for arguments in (
            {"r": 256, "g": 0, "b": 0},
            {"r": -1, "g": 0, "b": 0},
            {"r": True, "g": 0, "b": 0},
            {"r": 1, "g": 2.5, "b": 3},
            {"r": 1, "g": 2},
            {"r": 1, "g": 2, "b": 3, "extra": 4},
        ):
            result = executor.execute(ToolCall("smartled_set_color", arguments))
            self.assertFalse(result.success, arguments)

    def test_write_tool_is_denied_by_default(self):
        result = ToolExecutor(self.make_registry()).execute(
            ToolCall("smartled_set_color", {"r": 1, "g": 2, "b": 3})
        )
        self.assertFalse(result.success)
        self.assertIn("denied", result.error)

    def test_specific_write_allowance_does_not_allow_other_write_tools(self):
        executor = ToolExecutor(self.make_registry(), allowed_write_tools={"smartled_set_color"})
        allowed = executor.execute(ToolCall("smartled_set_color", {"r": 1, "g": 2, "b": 3}))
        denied = executor.execute(ToolCall("another_write", {}))
        self.assertTrue(allowed.success)
        self.assertFalse(denied.success)

    def test_http_error_flows_through_tool_executor_as_failed_action(self):
        client = SmartLEDAPIClient()
        error = HTTPError(
            "http://127.0.0.1:8765/api/v1/light", 503, "Unavailable", {},
            BytesIO(b'{"success":false,"status":"failed","message":"not configured"}'),
        )
        with patch("grub.tools.smartled.urlopen", side_effect=error):
            result = ToolExecutor(
                self.make_registry(client), allowed_write_tools={"smartled_set_color"}
            ).execute(ToolCall("smartled_set_color", {"r": 1, "g": 2, "b": 3}))
        self.assertTrue(result.success)  # Tool ran; the device action did not.
        self.assertEqual(result.data["status"], "failed")
        self.assertEqual(result.data["message"], "В API не настроен адрес SmartLED.")


if __name__ == "__main__":
    unittest.main()
