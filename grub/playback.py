"""Interruptible, single-stream playback for synthesized assistant speech."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable


class PlaybackError(RuntimeError):
    """Raised when the selected system output device cannot play audio."""


class AudioPlayback:
    def __init__(
        self,
        device: int | None = None,
        volume: float = 1.0,
        block_frames: int = 1024,
    ) -> None:
        self.device = device
        self.volume = max(0.0, min(1.0, float(volume)))
        self.block_frames = max(128, int(block_frames))
        self._lock = threading.Lock()
        self._active = False
        self._stop_event: threading.Event | None = None

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._active

    def stop(self) -> None:
        """Request a stop; the output worker closes its stream between small blocks."""
        with self._lock:
            stop_event = self._stop_event
        if stop_event is not None:
            stop_event.set()

    def play(
        self,
        samples,
        sample_rate: int,
        on_started: Callable[[], None] | None = None,
    ) -> bool:
        """Play mono float audio; return False if stopped before the end."""
        try:
            sd = importlib.import_module("sounddevice")
            np = importlib.import_module("numpy")
        except ImportError as exc:
            raise PlaybackError(f"Missing audio dependency {exc.name}.") from exc

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise PlaybackError("Cannot play empty audio.")
        audio = np.clip(audio * self.volume, -1.0, 1.0)
        fade_frames = min(max(1, int(sample_rate * 0.005)), audio.size // 2)
        if fade_frames:
            audio = audio.copy()
            audio[:fade_frames] *= np.linspace(0.0, 1.0, fade_frames, dtype=np.float32)
            audio[-fade_frames:] *= np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)
        stop_event = threading.Event()
        with self._lock:
            if self._active:
                raise PlaybackError("Another response is already playing.")
            self._active = True
            self._stop_event = stop_event

        try:
            if on_started is not None:
                on_started()
            with sd.OutputStream(
                device=self.device,
                samplerate=int(sample_rate),
                channels=1,
                dtype="float32",
                blocksize=self.block_frames,
            ) as stream:
                for start in range(0, audio.size, self.block_frames):
                    if stop_event.is_set():
                        ramp_frames = max(1, int(sample_rate * 0.005))
                        ramp = np.linspace(
                            float(audio[max(0, start - 1)]),
                            0.0,
                            ramp_frames,
                            dtype=np.float32,
                        )
                        stream.write(ramp.reshape(-1, 1))
                        return False
                    block = audio[start : start + self.block_frames]
                    stream.write(block.reshape(-1, 1))
            return not stop_event.is_set()
        except Exception as exc:
            if isinstance(exc, PlaybackError):
                raise
            raise PlaybackError(f"Could not play assistant audio: {exc}") from exc
        finally:
            with self._lock:
                self._active = False
                self._stop_event = None
