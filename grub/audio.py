"""Microphone capture for global push-to-talk and terminal fallback."""

from __future__ import annotations

import importlib
import sys
import threading
import time
from dataclasses import dataclass
from math import gcd
from typing import Callable


TARGET_SAMPLE_RATE = 16_000
MIN_AUDIO_SECONDS = 0.35


class AudioError(RuntimeError):
    """Raised when microphone capture cannot continue."""


@dataclass(frozen=True)
class Recording:
    samples: object
    duration: float


class AudioCapture:
    def __init__(self, device: int | None = None) -> None:
        try:
            self._sd = importlib.import_module("sounddevice")
            self._np = importlib.import_module("numpy")
        except ImportError as exc:
            raise AudioError(
                f"Missing {exc.name}. Install dependencies from requirements.txt."
            ) from exc

        self._requested_device = device
        self.device: int | None = None
        self.sample_rate = TARGET_SAMPLE_RATE
        self._stream = None
        self._chunks: list[object] = []
        self._recording_lock = threading.Lock()
        self._started_at = 0.0

    @property
    def recording(self) -> bool:
        with self._recording_lock:
            return self._stream is not None

    def check_microphone(self) -> None:
        try:
            devices = self._sd.query_devices()
            inputs = [
                index
                for index, info in enumerate(devices)
                if int(info.get("max_input_channels", 0)) > 0
            ]
            if not inputs:
                raise AudioError("No microphone input device was found.")

            if self._requested_device is not None:
                if self._requested_device not in inputs:
                    raise AudioError(
                        f"Audio device {self._requested_device} was not found or has no input."
                    )
                selected = self._requested_device
            else:
                try:
                    default_device = self._sd.default.device[0]
                except (AttributeError, TypeError, IndexError):
                    default_device = None
                selected = default_device if default_device in inputs else inputs[0]

            info = self._sd.query_devices(selected, "input")
            self.device = selected
            self.sample_rate = int(round(float(info["default_samplerate"])))
            # Opening the stream checks PortAudio access without requiring root.
            with self._sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            ):
                pass
        except AudioError:
            raise
        except Exception as exc:
            raise AudioError(f"Could not open microphone: {exc}") from exc

    def start_recording(self) -> None:
        if self.device is None:
            raise AudioError("Microphone has not been initialized.")

        with self._recording_lock:
            if self._stream is not None:
                raise AudioError("A recording is already in progress.")
            self._chunks = []
            stream = None
            try:
                stream = self._sd.InputStream(
                    device=self.device,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=self._input_callback,
                )
                self._stream = stream
                self._started_at = time.perf_counter()
                stream.start()
            except Exception as exc:
                self._stream = None
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
                raise AudioError(f"Could not start microphone recording: {exc}") from exc
        print("[REC] Recording...")

    def stop_recording(self) -> Recording:
        with self._recording_lock:
            stream = self._stream
            if stream is None:
                raise AudioError("There is no active recording to stop.")
            self._stream = None
            chunks = self._chunks
            self._chunks = []
            started_at = self._started_at

        try:
            stream.stop()
        except Exception as exc:
            raise AudioError(f"Could not stop microphone recording: {exc}") from exc
        finally:
            stream.close()

        duration = time.perf_counter() - started_at
        audio = (
            self._np.concatenate(chunks).astype("float32", copy=False)
            if chunks
            else self._np.empty(0, dtype="float32")
        )
        print(f"[REC] Stopped ({duration:.2f}s).")
        if duration < MIN_AUDIO_SECONDS or audio.size == 0:
            raise AudioError("Recording was too short; request ignored.")

        if self.sample_rate != TARGET_SAMPLE_RATE:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise AudioError("Install scipy from requirements.txt to resample audio.") from exc
            divisor = gcd(self.sample_rate, TARGET_SAMPLE_RATE)
            audio = resample_poly(
                audio,
                TARGET_SAMPLE_RATE // divisor,
                self.sample_rate // divisor,
            ).astype("float32", copy=False)

        return Recording(audio, duration)

    def cancel_recording(self) -> None:
        with self._recording_lock:
            stream = self._stream
            self._stream = None
            self._chunks = []
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()

    def record_with_enter(
        self,
        on_start: Callable[[], None] | None = None,
        on_stop: Callable[[], None] | None = None,
    ) -> Recording:
        if not sys.stdin.isatty():
            raise AudioError("Terminal fallback requires an interactive terminal.")
        input("[GRUB] Press ENTER to start recording (Ctrl+C to quit). ")
        stop_event = threading.Event()
        input_thread: threading.Thread | None = None

        def wait_for_enter() -> None:
            try:
                input("[REC] Recording. Press ENTER to stop. ")
            except EOFError:
                pass
            finally:
                stop_event.set()

        def start_stop_key_reader() -> None:
            nonlocal input_thread
            input_thread = threading.Thread(target=wait_for_enter, daemon=True)
            input_thread.start()

        try:
            recording = self._record_until(
                stop_event,
                before_start=on_start,
                on_started=start_stop_key_reader,
            )
        finally:
            if on_stop is not None:
                on_stop()
        if input_thread is not None:
            input_thread.join(timeout=0.2)
        return recording

    def _input_callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            print(f"\n[REC] Audio stream warning: {status}", file=sys.stderr)
        with self._recording_lock:
            if self._stream is not None:
                self._chunks.append(indata[:, 0].copy())

    def _record_until(
        self,
        stop_event: threading.Event,
        before_start: Callable[[], None] | None = None,
        on_started: Callable[[], None] | None = None,
    ) -> Recording:
        if before_start is not None:
            before_start()
        self.start_recording()
        try:
            if on_started is not None:
                on_started()
            while not stop_event.wait(timeout=0.05):
                pass
            return self.stop_recording()
        except BaseException:
            self.cancel_recording()
            raise
