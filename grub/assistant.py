"""Coordinates push-to-talk recording, transcription, and Ollama replies."""

from __future__ import annotations

import sys
import time

from .audio import AudioCapture, AudioError, HotkeyUnavailable, Recording
from .llm import OllamaClient, OllamaError
from .stt import STTError, WhisperSTT


SYSTEM_PROMPT = (
    "You are Grub, a local AI voice assistant running on Linux. You are helpful, "
    "technically competent, concise, and conversational. Respond in the user's language. "
    "Your responses will eventually be spoken aloud, so prefer clear and natural sentences. "
    "Do not claim to have performed actions you cannot actually perform."
)


class VoiceAssistant:
    def __init__(
        self,
        audio: AudioCapture,
        stt: WhisperSTT,
        llm: OllamaClient,
    ) -> None:
        self.audio = audio
        self.stt = stt
        self.llm = llm

    def run(self) -> None:
        self.llm.check_ready()
        print(f"[OLLAMA] Connected. Model: {self.llm.model}")
        self.audio.check_microphone()
        print("[AUDIO] Microphone ready.")
        model_load_time = self.stt.load()
        print(f"[STT] Whisper loaded in {model_load_time:.2f}s.")

        use_f8 = self.audio.hotkey_available()
        if use_f8:
            print("[GRUB] Ready. Hold F8 to speak. Press Ctrl+C to exit.")
        else:
            print("[KEY] Global F8 is unavailable; using terminal ENTER mode.")
            if not sys.stdin.isatty():
                raise AudioError("F8 недоступна, а резервный режим требует интерактивного терминала.")

        while True:
            try:
                recording = self.audio.record_with_f8() if use_f8 else self.audio.record_with_enter()
            except HotkeyUnavailable as exc:
                print(f"[KEY] {exc}. Переключаюсь на ENTER.")
                use_f8 = False
                continue
            except AudioError as exc:
                print(f"[AUDIO] {exc}", file=sys.stderr)
                if use_f8:
                    continue
                # In terminal mode, allow a fresh attempt after a short recording/device error.
                continue
            self.process_recording(recording)

    def process_recording(self, recording: Recording) -> None:
        request_started = time.perf_counter()
        try:
            text, stt_time = self.stt.transcribe(recording.samples)
        except STTError as exc:
            print(f"[STT] {exc}", file=sys.stderr)
            return

        if not text.strip():
            print("[STT] Речь не распознана; пустой запрос пропущен.")
            return

        print(f"[STT] {text}")
        print(f"[STT] Time: {stt_time:.2f}s")
        try:
            answer, llm_time = self.llm.chat(text)
        except (OllamaError, ValueError) as exc:
            print(f"[LLM] {exc}", file=sys.stderr)
            return

        total_time = time.perf_counter() - request_started
        print(f"[GRUB] {answer}")
        print(f"[LLM] Time: {llm_time:.2f}s")
        print(f"[GRUB] Total processing time: {total_time:.2f}s")
        print("[GRUB] Ready.")
