#!/usr/bin/env python3
"""Run Grub Agent from the project root."""

from __future__ import annotations

import os
import sys

from grub.audio import AudioCapture, AudioError
from grub.assistant import SYSTEM_PROMPT, VoiceAssistant
from grub.feedback import FeedbackManager
from grub.llm import OllamaClient, OllamaError
from grub.stt import STTError, WhisperSTT, configure_cuda_library_path


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def main() -> int:
    try:
        configure_cuda_library_path()

        audio_device_value = os.environ.get("GRUB_AUDIO_DEVICE", "").strip()
        try:
            audio_device = int(audio_device_value) if audio_device_value else None
        except ValueError as exc:
            raise AudioError("GRUB_AUDIO_DEVICE должен быть номером аудиоустройства.") from exc

        feedback_device_value = os.environ.get("GRUB_FEEDBACK_DEVICE", "").strip()
        try:
            feedback_device = int(feedback_device_value) if feedback_device_value else None
        except ValueError as exc:
            raise AudioError("GRUB_FEEDBACK_DEVICE must be an output device number.") from exc

        audio = AudioCapture(device=audio_device)
        stt = WhisperSTT(
            model_name=os.environ.get("GRUB_STT_MODEL", "small"),
            device=os.environ.get("GRUB_STT_DEVICE", "cuda"),
            compute_type=os.environ.get("GRUB_STT_COMPUTE_TYPE", "float16"),
            language=os.environ.get("GRUB_STT_LANGUAGE", "ru"),
        )
        llm = OllamaClient(
            model=os.environ.get("GRUB_MODEL", "gemma4:12b"),
            base_url=os.environ.get("GRUB_OLLAMA_URL", "http://localhost:11434"),
            system_prompt=SYSTEM_PROMPT,
            history_turns=_int_env("GRUB_HISTORY_TURNS", 10, 1, 50),
            timeout=float(os.environ.get("GRUB_OLLAMA_TIMEOUT", "120")),
            keep_alive=os.environ.get("GRUB_OLLAMA_KEEP_ALIVE", "10m"),
        )
        feedback = FeedbackManager(
            volume=float(os.environ.get("GRUB_BEEP_VOLUME", "0.12")),
            output_device=feedback_device,
        )
        try:
            VoiceAssistant(audio, stt, llm, feedback).run()
        finally:
            llm.close()
        return 0
    except KeyboardInterrupt:
        print("\n[GRUB] Завершение работы.")
        return 0
    except (AudioError, OllamaError, STTError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
