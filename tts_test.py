#!/usr/bin/env python3
"""Standalone TTS voice audition without Ollama, Whisper, or a microphone."""

from __future__ import annotations

import argparse
import sys

from grub.playback import AudioPlayback, PlaybackError
from grub.tts import SUPPORTED_VOICES, SpeechSynthesizer, TTSError


TEST_PHRASE = (
    "Я нашёл SmartLED в Local Network. IP-адрес 192.168.0.42, порт 4210. "
    "Используется Python 3.12."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize and play a local Supertonic 3 phrase")
    parser.add_argument("--voice", choices=SUPPORTED_VOICES, default="M1")
    parser.add_argument("--language", choices=("na", "ru", "en"), default="na")
    parser.add_argument(
        "--compare-languages",
        action="store_true",
        help="synthesize and play the phrase once with lang=na and once with lang=ru",
    )
    parser.add_argument("--text", default=TEST_PHRASE, help="text to synthesize")
    parser.add_argument("--playback-device", type=int, default=None)
    parser.add_argument("--volume", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 <= args.volume <= 1.0:
        parser.error("--volume must be between 0.0 and 1.0")

    text = args.text
    if not text:
        print("Enter a non-empty phrase.", file=sys.stderr)
        return 2

    try:
        synthesizer = SpeechSynthesizer(voice=args.voice, language=args.language)
        loaded = synthesizer.load()
        print(f"{synthesizer.engine} {synthesizer.MODEL_ID} loaded in {loaded:.2f}s.")
        languages = ("na", "ru") if args.compare_languages else (args.language,)
        player = AudioPlayback(device=args.playback_device, volume=args.volume)
        for language in languages:
            synthesizer.language = language
            audio = synthesizer.synthesize(text)
            rtf = audio.synthesis_seconds / audio.duration if audio.duration else float("inf")
            print(
                f"lang={language}: synthesized in {audio.synthesis_seconds:.3f}s; "
                f"audio {audio.duration:.3f}s at {audio.sample_rate} Hz; RTF {rtf:.3f}."
            )
            player.play(audio.samples, audio.sample_rate)
            print(f"Played Supertonic 3 voice {synthesizer.voice} (lang={language}).")
    except (TTSError, PlaybackError, ValueError) as exc:
        print(f"TTS audition failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
