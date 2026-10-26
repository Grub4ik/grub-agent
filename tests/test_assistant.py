from __future__ import annotations

import io
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock

from grub.assistant import AssistantState, VoiceAssistant
from grub.audio import Recording


class AssistantTests(unittest.TestCase):
    def make_assistant(self, audio=None, stt=None, llm=None, feedback=None):
        assistant = VoiceAssistant(
            audio or Mock(),
            stt or Mock(),
            llm or Mock(),
            feedback or Mock(),
        )
        assistant._keyboard = SimpleNamespace(Key=SimpleNamespace(f8="F8"))
        return assistant

    def test_f8_callbacks_ignore_other_keys_and_auto_repeat(self) -> None:
        assistant = self.make_assistant()

        assistant._on_key_press("A")
        assistant._on_key_press("F8")
        assistant._on_key_press("F8")
        self.assertEqual(assistant._events.qsize(), 1)
        self.assertEqual(assistant._events.get_nowait(), ("pressed", None))

        assistant._set_state(AssistantState.LISTENING)
        assistant._on_key_release("F8")
        self.assertEqual(assistant._events.get_nowait(), ("released", None))

    def test_recording_state_prevents_duplicate_capture(self) -> None:
        audio = Mock()
        feedback = Mock()
        assistant = self.make_assistant(audio=audio, feedback=feedback)

        with redirect_stdout(io.StringIO()):
            assistant._begin_recording()
            assistant._begin_recording()

        self.assertEqual(assistant.state, AssistantState.LISTENING)
        audio.start_recording.assert_called_once_with()
        feedback.recording_started.assert_called_once_with()

    def test_f8_is_ignored_while_request_worker_is_busy(self) -> None:
        transcribe_started = threading.Event()
        allow_transcription = threading.Event()
        audio = Mock()
        audio.stop_recording.return_value = Recording(object(), 1.0)
        stt = Mock()

        def blocking_transcribe(_samples):
            transcribe_started.set()
            allow_transcription.wait(timeout=2)
            return "hello", 0.01

        stt.transcribe.side_effect = blocking_transcribe
        llm = Mock()
        llm.chat.return_value = ("Hi", 0.02)
        assistant = self.make_assistant(audio, stt, llm)
        assistant._set_state(AssistantState.LISTENING)

        with redirect_stdout(io.StringIO()):
            assistant._finish_recording()
            self.assertTrue(transcribe_started.wait(timeout=1))
            self.assertEqual(assistant.state, AssistantState.PROCESSING)

            assistant._on_key_press("F8")
            assistant._on_key_release("F8")
            self.assertTrue(assistant._events.empty())

            allow_transcription.set()
            event, error = assistant._events.get(timeout=2)
            self.assertEqual(event, "processing_done")
            assistant._finish_processing(error)
            assistant._worker.join(timeout=1)
        self.assertEqual(assistant.state, AssistantState.IDLE)
        llm.chat.assert_called_once_with("hello")


if __name__ == "__main__":
    unittest.main()
