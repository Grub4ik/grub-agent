from __future__ import annotations

import unittest
from unittest.mock import patch

from grub.llm import OllamaClient


class OllamaToolTransportTests(unittest.TestCase):
    def test_chat_messages_sends_native_tools_and_tool_history(self):
        client = OllamaClient(system_prompt="system", model="gemma4:12b")
        self.addCleanup(client.close)
        messages = [
            {"role": "user", "content": "sum"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "calculate_sum", "arguments": {"a": 2, "b": 3}}}],
            },
            {"role": "tool", "tool_name": "calculate_sum", "content": "{\"success\": true}"},
        ]
        tools = [{"type": "function", "function": {"name": "calculate_sum"}}]
        with patch.object(
            client,
            "_request",
            return_value={"message": {"role": "assistant", "content": "5"}, "total_duration": 50_000_000},
        ) as request:
            message, elapsed = client.chat_messages(messages, tools=tools)

        payload = request.call_args.args[2]
        self.assertEqual(message["content"], "5")
        self.assertEqual(elapsed, 0.05)
        self.assertEqual(payload["tools"], tools)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][-1]["role"], "tool")
        self.assertEqual(payload["messages"][-1]["tool_name"], "calculate_sum")


if __name__ == "__main__":
    unittest.main()
