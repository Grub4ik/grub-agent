#!/usr/bin/env python3
"""Standalone Silero voice audition without Ollama, Whisper, or a microphone."""

from __future__ import annotations

import argparse
import sys

from grub.playback import AudioPlayback, PlaybackError
from grub.tts import SUPPORTED_VOICES, SpeechSynthesizer, TTSError


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize and play a Silero TTS phrase")
    parser.add_argument("--voice", choices=SUPPORTED_VOICES, default="aidar")
    parser.add_argument("--text", help="text to synthesize; prompts interactively when omitted")
    parser.add_argument("--playback-device", type=int, default=None)
    parser.add_argument("--volume", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 <= args.volume <= 1.0:
        parser.error("--volume must be between 0.0 and 1.0")

    text = args.text
    if text is None:
        try:
            text = input("Phrase to speak: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
    if not text:
        print("Enter a non-empty phrase.", file=sys.stderr)
        return 2

    synthesizer = SpeechSynthesizer(voice=args.voice, device="cpu")
    try:
        loaded = synthesizer.load()
        print(f"Silero {synthesizer.MODEL_ID} loaded in {loaded:.2f}s.")
        audio = synthesizer.synthesize(text)
        print(f"Synthesized in {audio.synthesis_seconds:.2f}s; audio duration {audio.duration:.2f}s.")
        AudioPlayback(device=args.playback_device, volume=args.volume).play(
            audio.samples,
            audio.sample_rate,
        )
        print(f"Played voice {args.voice}.")
    except (TTSError, PlaybackError) as exc:
        print(f"TTS audition failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
