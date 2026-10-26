#!/usr/bin/env python3
"""Run Grub Agent from the project root."""

from __future__ import annotations

import argparse
import os
import sys

from grub.agent import AgentCore, ToolExecutor, ToolRegistry
from grub.audio import AudioCapture, AudioError
from grub.assistant import SYSTEM_PROMPT, VoiceAssistant
from grub.feedback import FeedbackManager
from grub.llm import OllamaClient, OllamaError
from grub.playback import AudioPlayback
from grub.stt import STTError, WhisperSTT, configure_cuda_library_path
from grub.tools import (
    CALCULATE_SUM_SCHEMA,
    SYSTEM_INFO_SCHEMA,
    calculate_sum,
    get_system_info,
    SmartLEDAPIClient,
    register_smartled_tool,
)
from grub.tts import SUPPORTED_VOICES, SpeechSynthesizer


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _enabled_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Grub voice assistant")
    parser.add_argument("--no-tts", action="store_true", help="disable speech synthesis")
    parser.add_argument(
        "--voice",
        choices=SUPPORTED_VOICES,
        default=os.environ.get("GRUB_TTS_VOICE", "aidar"),
    )
    parser.add_argument(
        "--playback-device",
        type=int,
        default=None,
        help="sounddevice output device index",
    )
    parser.add_argument("--tts-volume", type=float, default=None, help="speech volume from 0.0 to 1.0")
    return parser


def main() -> int:
    try:
        args = _parser().parse_args()
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

        playback_device = args.playback_device
        if playback_device is None:
            playback_device_value = os.environ.get("GRUB_PLAYBACK_DEVICE", "").strip()
            try:
                playback_device = int(playback_device_value) if playback_device_value else None
            except ValueError as exc:
                raise AudioError("GRUB_PLAYBACK_DEVICE must be an output device number.") from exc

        tts_volume = args.tts_volume
        if tts_volume is None:
            tts_volume = float(os.environ.get("GRUB_TTS_VOLUME", "1.0"))
        if not 0.0 <= tts_volume <= 1.0:
            raise ValueError("TTS volume must be between 0.0 and 1.0.")

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
        registry = ToolRegistry()
        smartled_client = SmartLEDAPIClient(
            base_url=os.environ.get("GRUB_SMARTLED_API_URL", "http://127.0.0.1:8765"),
            timeout=float(os.environ.get("GRUB_SMARTLED_API_TIMEOUT", "4")),
        )
        registry.register(
            name="get_system_info",
            description="Get the operating system name, release, and architecture.",
            parameters=SYSTEM_INFO_SCHEMA,
            handler=get_system_info,
            permission="read",
        )
        registry.register(
            name="calculate_sum",
            description="Add two integers and return their sum.",
            parameters=CALCULATE_SUM_SCHEMA,
            handler=calculate_sum,
            permission="read",
        )
        register_smartled_tool(registry, smartled_client)
        allowed_write_tools = (
            {"smartled_set_color"}
            if _enabled_env("GRUB_ALLOW_SMARTLED_CONTROL")
            else set()
        )
        agent = AgentCore(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry, allowed_write_tools=allowed_write_tools),
            history_turns=_int_env("GRUB_HISTORY_TURNS", 10, 1, 50),
        )
        feedback = FeedbackManager(
            volume=float(os.environ.get("GRUB_BEEP_VOLUME", "0.12")),
            output_device=feedback_device,
        )
        tts = None if args.no_tts else SpeechSynthesizer(voice=args.voice, device="cpu")
        playback = AudioPlayback(device=playback_device, volume=tts_volume)
        try:
            VoiceAssistant(audio, stt, llm, agent, feedback, tts, playback).run()
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
