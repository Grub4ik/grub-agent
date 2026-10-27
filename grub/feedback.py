"""Short audio cues for push-to-talk recording."""

from __future__ import annotations

import importlib
import math
import sys


class FeedbackManager:
    SAMPLE_RATE = 22_050

    def __init__(self, volume: float = 0.12, output_device: int | None = None) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))
        self.output_device = output_device
        self._sound_disabled = False
        self._sound_warning_shown = False
        self._sd = None
        self._np = None

    def recording_started(self) -> None:
        self._play_tone(0.12, 520.0, 880.0)

    def recording_stopped(self) -> None:
        self._play_tone(0.14, 820.0, 480.0)

    def _play_tone(self, duration: float, start_hz: float, end_hz: float) -> None:
        if self._sound_disabled or self.volume <= 0:
            return
        try:
            if self._sd is None:
                self._sd = importlib.import_module("sounddevice")
                self._np = importlib.import_module("numpy")
            tone = self._make_tone(duration, start_hz, end_hz)
            # Cues are short and run on the assistant coordinator, never the
            # pynput callback. Blocking here ensures the start cue finishes before
            # microphone capture opens, so it is excluded from STT audio.
            self._sd.play(
                tone,
                samplerate=self.SAMPLE_RATE,
                device=self.output_device,
                blocking=True,
            )
        except Exception as exc:
            self._sound_disabled = True
            if not self._sound_warning_shown:
                print(f"[AUDIO] Sound cues disabled: {exc}", file=sys.stderr)
                self._sound_warning_shown = True

    def _make_tone(self, duration: float, start_hz: float, end_hz: float):
        np = self._np
        count = int(self.SAMPLE_RATE * duration)
        time_axis = np.arange(count, dtype=np.float32) / self.SAMPLE_RATE
        phase = 2.0 * math.pi * (
            start_hz * time_axis
            + ((end_hz - start_hz) / (2.0 * duration)) * time_axis**2
        )
        fade_count = max(1, int(self.SAMPLE_RATE * 0.018))
        envelope = np.ones(count, dtype=np.float32)
        envelope[:fade_count] = np.linspace(0.0, 1.0, fade_count, endpoint=False)
        envelope[-fade_count:] = np.linspace(1.0, 0.0, fade_count, endpoint=True)
        return (self.volume * envelope * np.sin(phase)).astype(np.float32, copy=False)
