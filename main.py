#!/usr/bin/env python3
"""Run Grub Agent from the project root."""

from __future__ import annotations

import argparse
from functools import partial
import os
import sys

from grub.agent import AgentCore, ToolExecutor, ToolRegistry
from grub.audio import AudioCapture, AudioError
from grub.assistant import SYSTEM_PROMPT, VoiceAssistant
from grub.feedback import FeedbackManager
from grub.llm import OllamaClient, OllamaError
from grub.memory import FactMemory, MemoryError
from grub.playback import AudioPlayback
from grub.stt import STTError, WhisperSTT, configure_cuda_library_path
from grub.tools import (
    CALCULATE_SUM_SCHEMA,
    SMARTLED_DISCOVERY_SCHEMA,
    SMARTLED_MUSIC_SCHEMA,
    SMARTLED_RAINBOW_SCHEMA,
    SMARTLED_SPECTRUM_SCHEMA,
    SMARTLED_STATIC_SCHEMA,
    SMARTLED_STOP_AUDIO_SCHEMA,
    SYSTEM_INFO_SCHEMA,
    MEMORY_FORGET_SCHEMA,
    MEMORY_LIST_SCHEMA,
    MEMORY_SAVE_SCHEMA,
    MEMORY_SEARCH_SCHEMA,
    SmartLEDTools,
    calculate_sum,
    forget_user_fact,
    get_system_info,
    list_user_facts,
    save_user_fact,
    search_user_facts,
)
from grub.tts import (
    SUPPORTED_DEVICES,
    SUPPORTED_ENGINES,
    SUPPORTED_LANGUAGES,
    SUPPORTED_VOICES,
    SpeechSynthesizer,
)


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _choice_env(name: str, choices: tuple[str, ...], default: str | None) -> str | None:
    value = os.environ.get(name)
    return value if value in choices else default


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Grub voice assistant")
    parser.add_argument("--no-tts", action="store_true", help="disable speech synthesis")
    parser.add_argument(
        "--voice",
        choices=SUPPORTED_VOICES,
        default=_choice_env("GRUB_TTS_VOICE", SUPPORTED_VOICES, None),
        help="Supertonic voice (default M1)",
    )
    parser.add_argument(
        "--tts-engine",
        choices=SUPPORTED_ENGINES,
        default=_choice_env("GRUB_TTS_ENGINE", SUPPORTED_ENGINES, "supertonic"),
        help="speech engine: supertonic",
    )
    parser.add_argument(
        "--tts-device",
        choices=SUPPORTED_DEVICES,
        default="cpu",
        help="TTS device; Supertonic uses CPU ONNX Runtime",
    )
    parser.add_argument(
        "--tts-language",
        choices=SUPPORTED_LANGUAGES,
        default=_choice_env("GRUB_TTS_LANGUAGE", SUPPORTED_LANGUAGES, "na"),
        help="Supertonic text language: na (mixed/unknown), ru, or en",
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
        smartled = SmartLEDTools()
        try:
            memory = FactMemory(
                path=os.environ.get("GRUB_MEMORY_PATH") or None,
                ollama_url=os.environ.get("GRUB_OLLAMA_URL", "http://localhost:11434"),
                embedding_model=os.environ.get("GRUB_EMBEDDING_MODEL", "embeddinggemma"),
                timeout=float(os.environ.get("GRUB_EMBEDDING_TIMEOUT", "20")),
                max_distance=float(os.environ.get("GRUB_MEMORY_MAX_DISTANCE", "0.55")),
            )
        except (MemoryError, OSError, ValueError) as exc:
            memory = None
            print(f"[MEMORY] Disabled; install/configure Chroma to enable saved facts: {exc}", file=sys.stderr)
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
        registry.register(
            name="discover_smartled",
            description="Discover a SmartLED device on the local network and return its IP address.",
            parameters=SMARTLED_DISCOVERY_SCHEMA,
            handler=smartled.discover,
            permission="read",
        )
        registry.register(
            name="smartled_set_static",
            description=(
                "Set the discovered SmartLED to static RGB color using integer channels r, g, b from 0 to 255. "
                "Do not provide an IP, port, or command. "
                "A successful Netcat send is not confirmation that the device applied it."
            ),
            parameters=SMARTLED_STATIC_SCHEMA,
            handler=smartled.set_static,
            permission="write",
        )
        registry.register(
            name="smartled_set_rainbow",
            description=(
                "Set the discovered SmartLED to rainbow mode. Provide interval in milliseconds, an integer from 100 to 60000. "
                "Do not provide an IP, port, or command. A successful send does not confirm device execution."
            ),
            parameters=SMARTLED_RAINBOW_SCHEMA,
            handler=smartled.set_rainbow,
            permission="write",
        )
        registry.register(
            name="smartled_start_music",
            description=(
                "Start continuous SmartLED Music mode driven by system output audio. "
                "Optional RGB base color, sensitivity, noise threshold, attack/release, brightness range, and normalization settings are supported."
            ),
            parameters=SMARTLED_MUSIC_SCHEMA,
            handler=smartled.start_music,
            permission="write",
        )
        registry.register(
            name="smartled_start_spectrum",
            description=(
                "Start continuous SmartLED Spectrum mode driven by system output audio FFT. "
                "Optional sensitivity (0-100) and noise threshold (0-30 percent) are supported."
            ),
            parameters=SMARTLED_SPECTRUM_SCHEMA,
            handler=smartled.start_spectrum,
            permission="write",
        )
        registry.register(
            name="smartled_stop_audio_mode",
            description="Stop the currently running SmartLED Music or Spectrum mode. Takes no arguments.",
            parameters=SMARTLED_STOP_AUDIO_SCHEMA,
            handler=smartled.stop_audio_mode,
            permission="write",
        )
        if memory is not None:
            registry.register(
                name="memory_save_fact",
                description=(
                    "Save a concise, stable fact explicitly shared by the user about their possessions, projects, "
                    "or preferences. Do not save temporary or sensitive information without an explicit request."
                ),
                parameters=MEMORY_SAVE_SCHEMA,
                handler=partial(save_user_fact, memory),
                permission="write",
            )
            registry.register(
                name="memory_search_facts",
                description="Search the user's saved facts by meaning when one may help answer the current request.",
                parameters=MEMORY_SEARCH_SCHEMA,
                handler=partial(search_user_facts, memory),
                permission="read",
            )
            registry.register(
                name="memory_list_facts",
                description="List saved user facts when the user asks what Grub remembers.",
                parameters=MEMORY_LIST_SCHEMA,
                handler=partial(list_user_facts, memory),
                permission="read",
            )
            registry.register(
                name="memory_forget_fact",
                description="Forget one saved fact only when the user explicitly asks, using its exact ID.",
                parameters=MEMORY_FORGET_SCHEMA,
                handler=partial(forget_user_fact, memory),
                permission="write",
            )
        agent = AgentCore(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(
                registry,
                allowed_tool_permissions={"write": {
                    "smartled_set_static",
                    "smartled_set_rainbow",
                    "smartled_start_music",
                    "smartled_start_spectrum",
                    "smartled_stop_audio_mode",
                    "memory_save_fact",
                    "memory_forget_fact",
                }},
            ),
            history_turns=_int_env("GRUB_HISTORY_TURNS", 10, 1, 50),
        )
        feedback = FeedbackManager(
            volume=float(os.environ.get("GRUB_BEEP_VOLUME", "0.12")),
            output_device=feedback_device,
        )
        tts = None if args.no_tts else SpeechSynthesizer(
            engine=args.tts_engine,
            voice=args.voice,
            device=args.tts_device,
            language=args.tts_language,
        )
        playback = AudioPlayback(device=playback_device, volume=tts_volume)
        try:
            VoiceAssistant(audio, stt, llm, agent, feedback, tts, playback).run()
        finally:
            smartled.close()
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
