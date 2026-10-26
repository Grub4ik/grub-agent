from __future__ import annotations

import json
import unittest

from grub.agent import AgentCore, ToolExecutor, ToolRegistry
from grub.tools.calculate_sum import CALCULATE_SUM_SCHEMA, calculate_sum
from grub.tools.system_info import SYSTEM_INFO_SCHEMA, get_system_info


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        "get_system_info",
        "Read system information.",
        SYSTEM_INFO_SCHEMA,
        get_system_info,
        "read",
    )
    registry.register(
        "calculate_sum",
        "Add two integers.",
        CALCULATE_SUM_SCHEMA,
        calculate_sum,
        "read",
    )
    return registry


class ScriptedLLM:
    def __init__(self, messages):
        self.responses = list(messages)
        self.requests = []

    def chat_messages(self, messages, tools=None):
        self.requests.append((json.loads(json.dumps(messages)), tools))
        return self.responses.pop(0), 0.025


def tool_call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


class AgentCoreTests(unittest.TestCase):
    def make_agent(self, responses, **limits):
        llm = ScriptedLLM(responses)
        registry = make_registry()
        agent = AgentCore(llm, registry, ToolExecutor(registry), **limits)
        return agent, llm

    def test_plain_reply_uses_no_tools(self):
        agent, llm = self.make_agent([{"role": "assistant", "content": "Привет!"}])
        response = agent.run("Привет")
        self.assertEqual(response.text, "Привет!")
        self.assertEqual(response.tool_calls, 0)
        self.assertEqual(llm.requests[0][1][0]["function"]["name"], "get_system_info")

    def test_single_tool_call_and_native_tool_result_message(self):
        agent, llm = self.make_agent([
            {"role": "assistant", "content": "", "tool_calls": [tool_call("calculate_sum", {"a": 125, "b": 347})]},
            {"role": "assistant", "content": "125 плюс 347 будет 472."},
        ])
        response = agent.run("Сколько 125 плюс 347?")
        self.assertEqual(response.text, "125 плюс 347 будет 472.")
        self.assertEqual(response.tool_calls, 1)
        tool_messages = [item for item in llm.requests[1][0] if item["role"] == "tool"]
        self.assertEqual(tool_messages[0]["tool_name"], "calculate_sum")
        self.assertEqual(json.loads(tool_messages[0]["content"])["data"], {"result": 472})
        assistant_call_messages = [item for item in llm.requests[1][0] if item.get("tool_calls")]
        self.assertEqual(len(assistant_call_messages), 1)

    def test_multiple_sequential_tool_rounds(self):
        agent, llm = self.make_agent([
            {"role": "assistant", "content": "", "tool_calls": [tool_call("get_system_info", {})]},
            {"role": "assistant", "content": "", "tool_calls": [tool_call("calculate_sum", {"a": 3, "b": 4})]},
            {"role": "assistant", "content": "Система определена, сумма равна 7."},
        ])
        response = agent.run("Проверь ОС и сложи 3 и 4.")
        self.assertEqual(response.tool_calls, 2)
        self.assertEqual([result.tool for result in response.tool_results], ["get_system_info", "calculate_sum"])
        self.assertEqual(response.text, "Система определена, сумма равна 7.")
        self.assertEqual(len(llm.requests), 3)

    def test_failed_tool_result_is_returned_to_model_for_correction(self):
        agent, llm = self.make_agent([
            {"role": "assistant", "content": "", "tool_calls": [tool_call("calculate_sum", {"a": "one", "b": 2})]},
            {"role": "assistant", "content": "", "tool_calls": [tool_call("calculate_sum", {"a": 1, "b": 2})]},
            {"role": "assistant", "content": "Сумма равна 3."},
        ])
        response = agent.run("Сложи числа.")
        self.assertEqual(response.text, "Сумма равна 3.")
        self.assertFalse(response.tool_results[0].success)
        self.assertTrue(response.tool_results[1].success)
        tool_messages = [item for item in llm.requests[1][0] if item["role"] == "tool"]
        self.assertFalse(json.loads(tool_messages[0]["content"])["success"])

    def test_round_limit_returns_incomplete_result(self):
        repeated = {"role": "assistant", "content": "", "tool_calls": [tool_call("get_system_info", {})]}
        agent, llm = self.make_agent([repeated] * 4, max_tool_rounds=2)
        response = agent.run("Сделай много запросов инструментов")
        self.assertFalse(response.complete)
        self.assertIn("лимит", response.text.lower())
        self.assertEqual(response.tool_calls, 2)
        self.assertEqual(len(llm.requests), 3)
        self.assertFalse(response.tool_results[-1].success)

    def test_total_tool_call_limit_is_enforced(self):
        three_calls = {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("get_system_info", {}) for _ in range(3)],
        }
        agent, _ = self.make_agent([three_calls], max_tool_calls=2)
        response = agent.run("Запроси сведения три раза")
        self.assertFalse(response.complete)
        self.assertEqual(response.tool_calls, 2)
        self.assertEqual(len(response.tool_results), 3)
        self.assertIn("count reached", response.tool_results[-1].error)


if __name__ == "__main__":
    unittest.main()
