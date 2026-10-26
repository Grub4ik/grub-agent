from __future__ import annotations

import unittest

from grub.agent import ToolCall, ToolExecutor, ToolRegistry
from grub.agent.registry import ToolRegistrationError
from grub.tools.calculate_sum import CALCULATE_SUM_SCHEMA, calculate_sum
from grub.tools.system_info import get_system_info


class ToolSystemTests(unittest.TestCase):
    def make_registry(self, handler=calculate_sum, permission="read"):
        registry = ToolRegistry()
        registry.register(
            "calculate_sum",
            "Add two integers.",
            CALCULATE_SUM_SCHEMA,
            handler,
            permission,
        )
        return registry

    def test_registration_exposes_schema_for_ollama(self):
        registry = self.make_registry()
        schema = registry.schemas()[0]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "calculate_sum")
        self.assertEqual(schema["function"]["parameters"], CALCULATE_SUM_SCHEMA)

    def test_duplicate_registration_is_rejected(self):
        registry = self.make_registry()
        with self.assertRaises(ToolRegistrationError):
            registry.register("calculate_sum", "duplicate", CALCULATE_SUM_SCHEMA, calculate_sum)

    def test_successful_execution(self):
        result = ToolExecutor(self.make_registry()).execute(ToolCall("calculate_sum", {"a": 12, "b": 30}))
        self.assertEqual(result.as_dict(), {
            "tool": "calculate_sum",
            "success": True,
            "data": {"result": 42},
            "error": None,
        })

    def test_unknown_tool_returns_structured_error(self):
        result = ToolExecutor(self.make_registry()).execute(ToolCall("no_such_tool", {}))
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Tool not found.")

    def test_invalid_arguments_are_rejected(self):
        executor = ToolExecutor(self.make_registry())
        missing = executor.execute(ToolCall("calculate_sum", {"a": 12}))
        extra = executor.execute(ToolCall("calculate_sum", {"a": 12, "b": 30, "c": 1}))
        wrong_type = executor.execute(ToolCall("calculate_sum", {"a": True, "b": 30}))
        out_of_range = executor.execute(ToolCall("calculate_sum", {"a": 1_000_000_001, "b": 0}))
        for result in (missing, extra, wrong_type, out_of_range):
            self.assertFalse(result.success)
            self.assertIn("Invalid arguments", result.error)

    def test_non_read_permission_is_denied_by_default(self):
        for permission in ("write", "restricted"):
            executor = ToolExecutor(self.make_registry(permission=permission))
            result = executor.execute(ToolCall("calculate_sum", {"a": 1, "b": 2}))
            self.assertFalse(result.success)
            self.assertIn("denied", result.error)

    def test_system_info_returns_the_documented_fields(self):
        value = get_system_info()
        self.assertEqual(set(value), {"os", "version", "architecture"})
        self.assertTrue(value["os"])
        self.assertTrue(value["architecture"])

    def test_handler_exception_becomes_tool_error(self):
        def fail(**_kwargs):
            raise RuntimeError("test failure")

        result = ToolExecutor(self.make_registry(handler=fail)).execute(
            ToolCall("calculate_sum", {"a": 1, "b": 2})
        )
        self.assertFalse(result.success)
        self.assertIn("test failure", result.error)


if __name__ == "__main__":
    unittest.main()
