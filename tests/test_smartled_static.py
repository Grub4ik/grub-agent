from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from grub.agent import AgentCore, ToolCall, ToolExecutor, ToolRegistry
from grub.tools.smartled import (
    SMARTLED_RAINBOW_SCHEMA,
    SMARTLED_STATIC_SCHEMA,
    SmartLEDCommandError,
    SmartLEDTools,
)


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def chat_messages(self, messages, tools=None):
        self.requests.append((messages, tools))
        return self.responses.pop(0), 0.01


def tool_call(name: str, arguments: dict):
    return {"function": {"name": name, "arguments": arguments}}


def make_agent(smartled: SmartLEDTools, responses):
    registry = ToolRegistry()
    registry.register(
        "discover_smartled",
        "Find the device.",
        {"type": "object", "properties": {}, "additionalProperties": False},
        smartled.discover,
        "read",
    )
    registry.register(
        "smartled_set_static",
        "Send the fixed static command; execution may not be confirmed.",
        SMARTLED_STATIC_SCHEMA,
        smartled.set_static,
        "write",
    )
    registry.register(
        "smartled_set_rainbow",
        "Set the discovered device to rainbow mode.",
        SMARTLED_RAINBOW_SCHEMA,
        smartled.set_rainbow,
        "write",
    )
    llm = ScriptedLLM(responses)
    executor = ToolExecutor(
        registry,
        allowed_tool_permissions={"write": {"smartled_set_static", "smartled_set_rainbow"}},
    )
    return AgentCore(llm, registry, executor), llm


class SmartLEDStaticTests(unittest.TestCase):
    def setUp(self):
        self.smartled = SmartLEDTools()

    def discover(self, ip="192.168.0.42"):
        with patch(
            "grub.tools.smartled.discover_smartled",
            return_value={"status": "found", "ip": ip, "port": 4210},
        ):
            return self.smartled.discover()

    def test_discovery_result_is_saved_and_used_on_the_next_agent_turn(self):
        agent, _llm = make_agent(
            self.smartled,
            [
                {"role": "assistant", "content": "", "tool_calls": [tool_call("discover_smartled", {})]},
                {"role": "assistant", "content": "Нашёл устройство."},
                {"role": "assistant", "content": "", "tool_calls": [tool_call("smartled_set_static", {"r": 255, "g": 0, "b": 0})]},
                {"role": "assistant", "content": "Команда отправлена, но подтверждения нет."},
            ],
        )
        completed = subprocess.CompletedProcess(
            args=["nc"], returncode=0, stdout=b"", stderr=b""
        )

        with patch(
            "grub.tools.smartled.discover_smartled",
            return_value={"status": "found", "ip": "192.168.0.42", "port": 4210},
        ), patch("grub.tools.smartled.subprocess.run", return_value=completed) as run:
            first_response = agent.run("Найди подсветку")
            second_response = agent.run("Переключи её в статический режим")

        self.assertEqual(first_response.tool_results[0].data["ip"], "192.168.0.42")
        self.assertEqual(second_response.tool_results[0].data["status"], "sent_unconfirmed")
        self.assertEqual(run.call_args.args[0][-2:], ["192.168.0.42", "4210"])

    def test_static_tool_schema_accepts_only_rgb_channels(self):
        self.assertEqual(
            SMARTLED_STATIC_SCHEMA,
            {
                "type": "object",
                "properties": {
                    "r": {"type": "integer", "minimum": 0, "maximum": 255},
                    "g": {"type": "integer", "minimum": 0, "maximum": 255},
                    "b": {"type": "integer", "minimum": 0, "maximum": 255},
                },
                "required": ["r", "g", "b"],
                "additionalProperties": False,
            },
        )

    def test_model_cannot_supply_an_ip_or_other_tool_arguments(self):
        registry = ToolRegistry()
        registry.register(
            "smartled_set_static",
            "Set the discovered device to the requested static RGB color.",
            SMARTLED_STATIC_SCHEMA,
            self.smartled.set_static,
            "write",
        )
        executor = ToolExecutor(
            registry,
            allowed_tool_permissions={"write": {"smartled_set_static"}},
        )

        result = executor.execute(ToolCall("smartled_set_static", {"r": 255, "g": 0, "b": 0, "ip": "127.0.0.1"}))

        self.assertFalse(result.success)
        self.assertIn("Invalid arguments", result.error)

    def test_model_rgb_arguments_are_required_and_bounded(self):
        registry = ToolRegistry()
        registry.register("smartled_set_static", "Set static RGB.", SMARTLED_STATIC_SCHEMA,
                          self.smartled.set_static, "write")
        executor = ToolExecutor(registry, allowed_tool_permissions={"write": {"smartled_set_static"}})
        for arguments in (
            {},
            {"r": -1, "g": 0, "b": 0},
            {"r": 256, "g": 0, "b": 0},
            {"r": True, "g": 0, "b": 0},
            {"r": 1.5, "g": 0, "b": 0},
        ):
            with self.subTest(arguments=arguments):
                result = executor.execute(ToolCall("smartled_set_static", arguments))
                self.assertFalse(result.success)
                self.assertIn("Invalid arguments", result.error)

    def test_red_color_is_transmitted_as_rgb_255_0_0(self):
        self.discover()
        completed = subprocess.CompletedProcess(args=["nc"], returncode=0, stdout=b"", stderr=b"")
        with patch("grub.tools.smartled.subprocess.run", return_value=completed) as run:
            result = self.smartled.set_static(255, 0, 0)
        self.assertEqual(run.call_args.kwargs["input"], b"MODE STATIC 255 0 0\n")
        self.assertEqual(result["rgb"], {"r": 255, "g": 0, "b": 0})
        self.assertEqual(result["status"], "sent_unconfirmed")

    def test_rainbow_interval_is_transmitted_exactly(self):
        self.discover()
        completed = subprocess.CompletedProcess(args=["nc"], returncode=0, stdout=b"", stderr=b"")
        with patch("grub.tools.smartled.subprocess.run", return_value=completed) as run:
            result = self.smartled.set_rainbow(1000)
        self.assertEqual(run.call_args.args[0][-2:], ["192.168.0.42", "4210"])
        self.assertEqual(run.call_args.kwargs["input"], b"MODE RAINBOW INTERVAL 1000\n")
        self.assertEqual(result["mode"], "rainbow")
        self.assertEqual(result["interval"], 1000)
        self.assertEqual(result["status"], "sent_unconfirmed")
        self.assertFalse(result["execution_confirmed"])

    def test_rainbow_schema_requires_and_bounds_interval(self):
        self.assertEqual(
            SMARTLED_RAINBOW_SCHEMA,
            {
                "type": "object",
                "properties": {"interval": {"type": "integer", "minimum": 100, "maximum": 60000}},
                "required": ["interval"],
                "additionalProperties": False,
            },
        )
        registry = ToolRegistry()
        registry.register("smartled_set_rainbow", "Set rainbow.", SMARTLED_RAINBOW_SCHEMA,
                          self.smartled.set_rainbow, "write")
        executor = ToolExecutor(registry, allowed_tool_permissions={"write": {"smartled_set_rainbow"}})
        for arguments in (
            {},
            {"interval": 99},
            {"interval": 60001},
            {"interval": True},
            {"interval": 1000, "command": "x"},
        ):
            with self.subTest(arguments=arguments):
                result = executor.execute(ToolCall("smartled_set_rainbow", arguments))
                self.assertFalse(result.success)
                self.assertIn("Invalid arguments", result.error)

    def test_rainbow_accepts_interval_boundaries(self):
        self.discover()
        completed = subprocess.CompletedProcess(args=["nc"], returncode=0, stdout=b"", stderr=b"")
        for interval in (100, 60000):
            with self.subTest(interval=interval), patch(
                "grub.tools.smartled.subprocess.run", return_value=completed
            ) as run:
                result = self.smartled.set_rainbow(interval)
                self.assertEqual(run.call_args.kwargs["input"], f"MODE RAINBOW INTERVAL {interval}\n".encode())
                self.assertEqual(result["interval"], interval)

    def test_no_discovered_ip_is_a_clear_error_and_does_not_start_nc(self):
        with patch("grub.tools.smartled.subprocess.run") as run:
            with self.assertRaisesRegex(SmartLEDCommandError, "not been discovered"):
                self.smartled.set_static(255, 0, 0)
        run.assert_not_called()

    def test_not_discovered_state_becomes_structured_executor_error(self):
        registry = ToolRegistry()
        registry.register(
            "smartled_set_static",
            "Set the discovered device to the requested static RGB color.",
            SMARTLED_STATIC_SCHEMA,
            self.smartled.set_static,
            "write",
        )
        executor = ToolExecutor(
            registry,
            allowed_tool_permissions={"write": {"smartled_set_static"}},
        )

        result = executor.execute(ToolCall("smartled_set_static", {"r": 255, "g": 0, "b": 0}))

        self.assertFalse(result.success)
        self.assertIn("not been discovered", result.error)

    def test_not_found_discovery_clears_previous_saved_address(self):
        self.discover()
        with patch(
            "grub.tools.smartled.discover_smartled",
            return_value={"status": "not_found", "ip": None},
        ):
            result = self.smartled.discover()
        self.assertEqual(result["status"], "not_found")
        with self.assertRaises(SmartLEDCommandError):
            self.smartled.set_static(255, 0, 0)

    def test_invalid_discovery_address_is_not_saved(self):
        for ip in ("10.0.0.4", "192.168.0.0", "192.168.0.255"):
            with self.subTest(ip=ip):
                with patch(
                    "grub.tools.smartled.discover_smartled",
                    return_value={"status": "found", "ip": ip},
                ):
                    with self.assertRaisesRegex(SmartLEDCommandError, "unusable host address"):
                        self.smartled.discover()
                with self.assertRaises(SmartLEDCommandError):
                    self.smartled.set_static(255, 0, 0)

    def test_exact_nc_arguments_and_bytes_and_unconfirmed_status(self):
        self.discover()
        completed = subprocess.CompletedProcess(
            args=["nc"], returncode=0, stdout=b"OK\n", stderr=b""
        )
        with patch("grub.tools.smartled.subprocess.run", return_value=completed) as run:
            result = self.smartled.set_static(255, 0, 0)

        args, kwargs = run.call_args
        self.assertEqual(
            args[0],
            ["nc", "-4", "-N", "-w", "2", "192.168.0.42", "4210"],
        )
        self.assertEqual(kwargs["input"], b"MODE STATIC 255 0 0\n")
        self.assertEqual(result["rgb"], {"r": 255, "g": 0, "b": 0})
        self.assertIs(kwargs["shell"], False)
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.PIPE)
        self.assertEqual(kwargs["timeout"], 4.0)
        self.assertEqual(result["status"], "sent_unconfirmed")
        self.assertFalse(result["execution_confirmed"])
        self.assertEqual(result["response"], "OK")

    def test_unavailable_device_is_reported_as_error(self):
        self.discover()
        completed = subprocess.CompletedProcess(
            args=["nc"], returncode=1, stdout=b"", stderr=b"Connection refused"
        )
        with patch("grub.tools.smartled.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(SmartLEDCommandError, "may have reached"):
                self.smartled.set_static(255, 0, 0)

    def test_netcat_timeout_reports_unknown_delivery(self):
        self.discover()
        timeout = subprocess.TimeoutExpired(cmd=["nc"], timeout=4.0)
        with patch("grub.tools.smartled.subprocess.run", side_effect=timeout) as run:
            with self.assertRaisesRegex(SmartLEDCommandError, "delivery and execution are unknown"):
                self.smartled.set_static(255, 0, 0)
        self.assertEqual(run.call_args.kwargs["timeout"], 4.0)

    def test_missing_nc_is_reported_without_claiming_execution(self):
        self.discover()
        with patch(
            "grub.tools.smartled.subprocess.run",
            side_effect=FileNotFoundError("nc not found"),
        ):
            with self.assertRaisesRegex(SmartLEDCommandError, "Could not run Netcat"):
                self.smartled.set_static(255, 0, 0)

    def test_zero_exit_is_not_misrepresented_as_device_confirmation(self):
        self.discover()
        completed = subprocess.CompletedProcess(args=["nc"], returncode=0, stdout=b"", stderr=b"")
        with patch("grub.tools.smartled.subprocess.run", return_value=completed):
            result = self.smartled.set_static(255, 0, 0)
        self.assertEqual(result["status"], "sent_unconfirmed")
        self.assertFalse(result["execution_confirmed"])
        self.assertFalse(result["response_received"])


if __name__ == "__main__":
    unittest.main()
