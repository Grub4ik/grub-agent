"""Headless, realtime SmartLED Music and Spectrum modes.

The analysis formulas and UDP frame format follow LEDClient's
``AudioAnalyzer``, ``SpectrumAnalyzer``, and ``DynamicUdpSender``. Qt is
deliberately not imported: Grub runs tools in a regular Python worker thread.
"""

from __future__ import annotations

from dataclasses import dataclass
import colorsys
import importlib
import math
import socket
import struct
import threading
import time
from typing import Any

import numpy as np


SMARTLED_MUSIC_SCHEMA = {
    "type": "object",
    "properties": {
        "r": {"type": "integer", "minimum": 0, "maximum": 255, "default": 255},
        "g": {"type": "integer", "minimum": 0, "maximum": 255, "default": 255},
        "b": {"type": "integer", "minimum": 0, "maximum": 255, "default": 255},
        "sensitivity": {"type": "integer", "minimum": 0, "maximum": 100, "default": 60},
        "noise_threshold": {"type": "integer", "minimum": 0, "maximum": 100, "default": 1},
        "attack_ms": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 8},
        "release_ms": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 20},
        "minimum_brightness": {"type": "integer", "minimum": 0, "maximum": 100, "default": 0},
        "maximum_brightness": {"type": "integer", "minimum": 0, "maximum": 100, "default": 100},
        "adaptive_normalization": {"type": "boolean", "default": True},
    },
    "additionalProperties": False,
}
SMARTLED_SPECTRUM_SCHEMA = {
    "type": "object",
    "properties": {
        "sensitivity": {"type": "integer", "minimum": 0, "maximum": 100, "default": 70},
        "noise_threshold": {"type": "integer", "minimum": 0, "maximum": 30, "default": 1},
    },
    "additionalProperties": False,
}
SMARTLED_STOP_AUDIO_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

TCP_PORT = 4210
UDP_PORT = 4211
SAMPLE_RATE = 48_000
BLOCK_SIZE = 1024
UDP_FPS = 30
TCP_TIMEOUT_SECONDS = 2.0
STARTUP_TIMEOUT_SECONDS = 5.0
STOP_TIMEOUT_SECONDS = 3.0
MAX_ACK_BYTES = 256


@dataclass(frozen=True)
class MusicSettings:
    sensitivity: int = 60
    noise_threshold: int = 1
    attack_ms: int = 8
    release_ms: int = 20
    minimum_brightness: int = 0
    maximum_brightness: int = 100
    adaptive_normalization: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.sensitivity <= 100:
            raise ValueError("Music sensitivity must be between 0 and 100.")
        if not 0 <= self.noise_threshold <= 100:
            raise ValueError("Music noise threshold must be between 0 and 100.")
        if self.attack_ms < 1 or self.release_ms < 1:
            raise ValueError("Music attack and release must be at least 1 ms.")
        if not 0 <= self.minimum_brightness <= self.maximum_brightness <= 100:
            raise ValueError("Brightness must satisfy 0 <= minimum <= maximum <= 100.")


@dataclass(frozen=True)
class SpectrumSettings:
    sensitivity: int = 70
    noise_threshold: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.sensitivity <= 100:
            raise ValueError("Spectrum sensitivity must be between 0 and 100.")
        if not 0 <= self.noise_threshold <= 30:
            raise ValueError("Spectrum noise threshold must be between 0 and 30.")


@dataclass(frozen=True)
class SpectrumFrame:
    red: int
    green: int
    blue: int
    brightness: float
    bass: float
    mid: float
    treble: float
    hue_degrees: float


class AudioAnalyzer:
    """LEDClient's RMS brightness analysis for Music mode."""

    FIXED_REFERENCE_LEVEL = 0.25
    ADAPTIVE_HEADROOM = 2.0
    REFERENCE_RISE_SECONDS = 1.5
    REFERENCE_FALL_SECONDS = 0.8

    def __init__(self, settings: MusicSettings, sample_rate: int = SAMPLE_RATE) -> None:
        self.settings = settings
        self.sample_rate = sample_rate
        self.reference_level: float | None = None
        self.envelope = 0.0

    def process(self, samples: Any) -> float:
        frames = int(samples.shape[0]) if hasattr(samples, "shape") else len(samples)
        values = samples.flat if hasattr(samples, "flat") else (v for frame in samples for v in (frame if isinstance(frame, (list, tuple)) else (frame,)))
        square_sum = 0.0
        sample_count = 0
        for sample in values:
            value = float(sample)
            square_sum += value * value
            sample_count += 1
        if sample_count == 0:
            minimum = self.settings.minimum_brightness / 100.0
            maximum = self.settings.maximum_brightness / 100.0
            return minimum + self.envelope * (maximum - minimum)

        rms = math.sqrt(square_sum / sample_count)
        threshold = self.settings.noise_threshold / 100.0
        level = 0.0 if rms < threshold else rms
        delta = max(frames / self.sample_rate, 1e-6)
        if self.settings.adaptive_normalization:
            if self.reference_level is None:
                self.reference_level = max(level, threshold, 0.02)
            else:
                tau = self.REFERENCE_RISE_SECONDS if level > self.reference_level else self.REFERENCE_FALL_SECONDS
                alpha = 1.0 - math.exp(-delta / tau)
                self.reference_level += alpha * (level - self.reference_level)
            reference = max(self.reference_level, threshold, 1e-4) * self.ADAPTIVE_HEADROOM
        else:
            reference = self.FIXED_REFERENCE_LEVEL
        normalized = min(max(level / max(reference, 1e-6), 0.0), 1.0)
        contrast = 1.0 + 3.0 * (self.settings.sensitivity / 100.0)
        target = min(max(0.5 + (normalized - 0.5) * contrast, 0.0), 1.0)
        envelope_ms = self.settings.attack_ms if target > self.envelope else self.settings.release_ms
        alpha = 1.0 - math.exp(-delta / (envelope_ms / 1000.0))
        self.envelope = min(max(self.envelope + alpha * (target - self.envelope), 0.0), 1.0)
        minimum = self.settings.minimum_brightness / 100.0
        maximum = self.settings.maximum_brightness / 100.0
        return minimum + self.envelope * (maximum - minimum)

class SpectrumAnalyzer:
    """LEDClient's FFT bands, hue selection, and smoothed brightness."""

    FFT_SIZE = 2048
    EDGES = tuple(40.0 * (2 ** index) for index in range(9))
    BANDS = tuple(zip(EDGES[:-1], EDGES[1:]))
    HUES = tuple(index * 45.0 for index in range(8))

    def __init__(self, settings: SpectrumSettings, sample_rate: int = SAMPLE_RATE) -> None:
        self.settings = settings
        self.sample_rate = sample_rate
        self.window = np.hanning(self.FFT_SIZE).astype(np.float32)
        self.window_sum = float(np.sum(self.window))
        frequencies = np.fft.rfftfreq(self.FFT_SIZE, d=1.0 / sample_rate)
        self.masks = tuple((frequencies >= low) & (frequencies < high) for low, high in self.BANDS)
        centers = np.sqrt([low * high for low, high in self.BANDS])
        reference_center = float(np.sqrt(centers[0] * centers[-1]))
        self.gains = np.sqrt(centers / reference_center)
        self.samples = np.zeros((self.FFT_SIZE, 1), dtype=np.float32)
        self.reference: float | None = None
        self.brightness = 0.0
        self.band_envelopes = np.zeros(len(self.BANDS), dtype=np.float64)
        self.hue: float | None = None
        self.gate_open = False

    def process(self, samples: Any) -> SpectrumFrame:
        array = np.asarray(samples, dtype=np.float32)
        if array.size == 0:
            return self._frame()
        audio = array.reshape(-1, 1) if array.ndim == 1 else array
        if audio.ndim != 2:
            raise ValueError("Spectrum audio must have shape (frames, channels).")
        frames = int(audio.shape[0])
        delta = max(frames / self.sample_rate, 1e-6)
        # Keep channels separate until after the FFT. Averaging stereo waveforms
        # first can cancel anti-phase music, while averaging their power cannot.
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if self.samples.shape[1] != audio.shape[1]:
            self.samples = np.zeros((self.FFT_SIZE, audio.shape[1]), dtype=np.float32)
        if frames >= self.FFT_SIZE:
            self.samples[:] = audio[-self.FFT_SIZE:]
        else:
            self.samples[:-frames] = self.samples[frames:]
            self.samples[-frames:] = audio

        threshold = self.settings.noise_threshold / 100.0
        if threshold <= 0:
            self.gate_open = rms > 1e-5
        elif self.gate_open:
            self.gate_open = rms >= threshold * 0.8
        else:
            self.gate_open = rms >= threshold * 1.2

        if self.gate_open:
            centered = self.samples - np.mean(self.samples, axis=0, keepdims=True)
            windowed = centered * self.window[:, np.newaxis]
            magnitude = np.abs(np.fft.rfft(windowed, axis=0)) * (2.0 / self.window_sum)
            energy = np.array([
                float(np.sqrt(np.mean(np.square(magnitude[mask, :]))))
                for mask in self.masks
            ]) * self.gains
        else:
            energy = np.zeros(len(self.BANDS), dtype=np.float64)
        total = float(np.sum(energy))
        targets = energy / total if total > 1e-9 else np.zeros(len(self.BANDS), dtype=np.float64)
        tau = np.where(targets > self.band_envelopes, 0.05, 0.22)
        self.band_envelopes += (1.0 - np.exp(-delta / tau)) * (targets - self.band_envelopes)

        weights = np.square(np.maximum(self.band_envelopes, 0.0))
        weight_total = float(np.sum(weights))
        if weight_total > 1e-12:
            weights /= weight_total
            angles = np.deg2rad(self.HUES)
            x = float(np.dot(weights, np.cos(angles)))
            y = float(np.dot(weights, np.sin(angles)))
            if math.hypot(x, y) > 0.12:
                target_hue = math.degrees(math.atan2(y, x)) % 360.0
                if self.hue is None:
                    self.hue = target_hue
                else:
                    difference = (target_hue - self.hue + 180.0) % 360.0 - 180.0
                    self.hue = (self.hue + (1.0 - math.exp(-delta / 0.06)) * difference) % 360.0

        level = rms if self.gate_open else 0.0
        if self.reference is None:
            self.reference = max(level, threshold, 0.02)
        else:
            tau_ref = 1.5 if level > self.reference else 0.8
            self.reference += (1.0 - math.exp(-delta / tau_ref)) * (level - self.reference)
        normalized = min(max(rms / max(self.reference * 2.0, 1e-6), 0.0), 1.0) if self.gate_open else 0.0
        contrast = 1.0 + 5.0 * (self.settings.sensitivity / 100.0)
        target_brightness = min(max(0.5 + (normalized - 0.5) * contrast, 0.0), 1.0)
        tau_brightness = 0.06 if target_brightness > self.brightness else 0.24
        self.brightness += (1.0 - math.exp(-delta / tau_brightness)) * (target_brightness - self.brightness)
        return self._frame()

    def _frame(self) -> SpectrumFrame:
        hue = self.hue or 0.0
        red, green, blue = colorsys.hsv_to_rgb((hue % 360.0) / 360.0, 1.0, min(max(self.brightness, 0.0), 1.0))
        return SpectrumFrame(
            round(red * 255), round(green * 255), round(blue * 255),
            min(max(self.brightness, 0.0), 1.0),
            min(max(float(np.sum(self.band_envelopes[:2])), 0.0), 1.0),
            min(max(float(np.sum(self.band_envelopes[2:6])), 0.0), 1.0),
            min(max(float(np.sum(self.band_envelopes[6:])), 0.0), 1.0), hue % 360.0,
        )


@dataclass
class _Session:
    mode: str
    address: str
    tcp: socket.socket | None
    stop: threading.Event
    ready: threading.Event
    go: threading.Event
    finished: threading.Event
    errors: list[str]
    thread: threading.Thread | None = None


class SmartLEDAudioModes:
    """Manage one mutually exclusive background audio-reactive stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: _Session | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._session is not None and self._session.thread is not None and self._session.thread.is_alive()

    def start_music(
        self, address: str, r: int = 255, g: int = 255, b: int = 255,
        sensitivity: int = 60, noise_threshold: int = 1, attack_ms: int = 8,
        release_ms: int = 20, minimum_brightness: int = 0,
        maximum_brightness: int = 100, adaptive_normalization: bool = True,
    ) -> dict[str, Any]:
        color = (r, g, b)
        if any(type(channel) is not int or not 0 <= channel <= 255 for channel in color):
            raise ValueError("Music base RGB channels must be integers from 0 to 255.")
        settings = MusicSettings(sensitivity, noise_threshold, attack_ms, release_ms,
                                 minimum_brightness, maximum_brightness, adaptive_normalization)
        return self._start(address, "music", settings, color)

    def start_spectrum(
        self, address: str, sensitivity: int = 70, noise_threshold: int = 1,
    ) -> dict[str, Any]:
        settings = SpectrumSettings(sensitivity, noise_threshold)
        return self._start(address, "spectrum", settings, None)

    def _start(self, address: str, mode: str, settings: Any, color: tuple[int, int, int] | None) -> dict[str, Any]:
        with self._lock:
            current = self._session
            if current is not None and current.thread is not None and current.thread.is_alive():
                raise RuntimeError(f"SmartLED {current.mode} audio mode is already running; stop it first.")
            self._session = None

        # Open loopback capture first. If sound capture is unavailable, Grub
        # fails without switching the controller into DYNAMIC mode.
        session = _Session(
            mode, address, None, threading.Event(), threading.Event(),
            threading.Event(), threading.Event(), [],
        )
        session.thread = threading.Thread(
            target=self._run, args=(session, settings, color),
            name=f"smartled-{mode}", daemon=True,
        )
        with self._lock:
            self._session = session
            session.thread.start()
        if not session.ready.wait(STARTUP_TIMEOUT_SECONDS):
            session.stop.set()
            session.go.set()
            session.thread.join(STOP_TIMEOUT_SECONDS)
            self._clear_session(session)
            raise RuntimeError("System audio capture did not become ready in time.")
        if session.errors:
            session.thread.join(STOP_TIMEOUT_SECONDS)
            self._clear_session(session)
            raise RuntimeError(session.errors[0])
        try:
            tcp = socket.create_connection((address, TCP_PORT), timeout=TCP_TIMEOUT_SECONDS)
            tcp.settimeout(TCP_TIMEOUT_SECONDS)
            tcp.sendall(b"MODE DYNAMIC\n")
            response = _read_line(tcp, MAX_ACK_BYTES)
            if response != "OK":
                raise RuntimeError(f"SmartLED rejected MODE DYNAMIC: {response or 'empty response'}")
            session.tcp = tcp
            session.go.set()
        except Exception:
            session.stop.set()
            session.go.set()
            if "tcp" in locals():
                tcp.close()
            session.thread.join(STOP_TIMEOUT_SECONDS)
            self._clear_session(session)
            raise
        return {
            "status": "running", "mode": mode, "ip": address,
            "udp_port": UDP_PORT, "fps": UDP_FPS,
            "details": {"base_rgb": color} if color is not None else {
                "sensitivity": settings.sensitivity,
                "noise_threshold": settings.noise_threshold,
            },
        }

    def stop(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            if session is None:
                return {"status": "already_stopped", "mode": None}
            session.stop.set()
            session.go.set()
        assert session.thread is not None
        session.thread.join(STOP_TIMEOUT_SECONDS)
        if session.thread.is_alive():
            raise RuntimeError("Audio mode stop was requested, but its capture thread has not exited yet.")
        self._clear_session(session)
        result: dict[str, Any] = {"status": "stopped", "mode": session.mode}
        if session.errors:
            result["error"] = session.errors[0]
        result["device_note"] = "UDP updates stopped; the device may keep the last received color until another mode command."
        return result

    def close(self) -> None:
        try:
            self.stop()
        except RuntimeError:
            pass

    def _clear_session(self, session: _Session) -> None:
        with self._lock:
            if self._session is session:
                self._session = None

    @staticmethod
    def _run(session: _Session, settings: Any, color: tuple[int, int, int] | None) -> None:
        udp: socket.socket | None = None
        try:
            soundcard = importlib.import_module("soundcard")
            speaker = soundcard.default_speaker()
            if speaker is None:
                raise RuntimeError("No default speaker is available for system loopback capture.")
            loopback = soundcard.get_microphone(str(speaker.name), include_loopback=True)
            analyzer = AudioAnalyzer(settings) if session.mode == "music" else SpectrumAnalyzer(settings)
            udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            with loopback.recorder(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE) as recorder:
                session.ready.set()
                while not session.go.wait(0.1):
                    if session.stop.is_set():
                        return
                sequence = 0
                next_frame = time.perf_counter()
                rgb = (0, 0, 0)
                while not session.stop.is_set():
                    samples = recorder.record(numframes=BLOCK_SIZE)
                    if session.stop.is_set():
                        break
                    if session.mode == "music":
                        brightness = analyzer.process(samples)
                        rgb = tuple(round(channel * brightness) for channel in color)  # type: ignore[arg-type]
                    else:
                        frame = analyzer.process(samples)
                        rgb = (frame.red, frame.green, frame.blue)
                    now = time.perf_counter()
                    if now >= next_frame:
                        packet = struct.pack("!HBBB", sequence, *rgb)
                        sent = udp.sendto(packet, (session.address, UDP_PORT))
                        if sent != len(packet):
                            raise RuntimeError("UDP frame was not fully sent.")
                        sequence = (sequence + 1) & 0xFFFF
                        next_frame += 1.0 / UDP_FPS
                        if next_frame < now:
                            next_frame = now + 1.0 / UDP_FPS
        except Exception as exc:
            session.errors.append(f"{type(exc).__name__}: {exc}")
            session.ready.set()
        finally:
            if udp is not None:
                udp.close()
            if session.tcp is not None:
                session.tcp.close()
            session.finished.set()


def _read_line(connection: socket.socket, limit: int) -> str:
    response = bytearray()
    while len(response) < limit:
        chunk = connection.recv(1)
        if not chunk:
            break
        if chunk == b"\n":
            return response.rstrip(b"\r").decode("utf-8", errors="replace")
        response.extend(chunk)
    if len(response) >= limit:
        raise RuntimeError("SmartLED response exceeded the acknowledgement limit.")
    return response.decode("utf-8", errors="replace")
