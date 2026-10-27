from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock

from grub.agent.models import AgentResponse, ToolResult
from grub.assistant import VoiceAssistant
from grub.audio import Recording
from grub.tts import SynthesizedAudio


class AssistantAgentIntegrationTests(unittest.TestCase):
    def make_assistant(self, text):
        audio = Mock()
        stt = Mock()
        stt.transcribe.return_value = ("user request", 0.01)
        llm = Mock()
        agent = Mock()
        agent.run.return_value = AgentResponse(
            text=text,
            llm_seconds=0.05,
            tool_calls=1,
            tool_results=(ToolResult("calculate_sum", True, {"result": 42}),),
        )
        feedback = Mock()
        tts = Mock()
        tts.synthesize.return_value = SynthesizedAudio([0.1] * 480, 48_000, 0.01)
        playback = Mock()

        def play(_samples, _rate, on_started=None):
            if on_started:
                on_started()
            return True

        playback.play.side_effect = play
        return VoiceAssistant(audio, stt, llm, agent, feedback, tts, playback), agent, tts

    def test_only_final_agent_text_is_notified_and_spoken(self):
        assistant, agent, tts = self.make_assistant("The sum is 42.")
        with redirect_stdout(io.StringIO()):
            assistant.process_recording(Recording([0.0], 1.0))
        agent.run.assert_called_once_with("user request")
        tts.synthesize.assert_called_once_with("The sum is 42.")

    def test_missing_final_text_does_not_speak(self):
        assistant, _agent, tts = self.make_assistant(None)
        with redirect_stdout(io.StringIO()):
            assistant.process_recording(Recording([0.0], 1.0))
        tts.synthesize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
