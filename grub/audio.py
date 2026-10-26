"""Microphone capture and push-to-talk controls."""

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


class HotkeyUnavailable(AudioError):
    """Raised when the current desktop session cannot listen for F8."""


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
                f"Не найдена библиотека {exc.name}. Установите зависимости из requirements.txt."
            ) from exc

        self._requested_device = device
        self.device: int | None = None
        self.sample_rate = TARGET_SAMPLE_RATE

    def check_microphone(self) -> None:
        try:
            devices = self._sd.query_devices()
            inputs = [
                index
                for index, info in enumerate(devices)
                if int(info.get("max_input_channels", 0)) > 0
            ]
            if not inputs:
                raise AudioError("Не найден микрофон с входными аудиоканалами.")

            if self._requested_device is not None:
                if self._requested_device not in inputs:
                    raise AudioError(
                        f"Аудиоустройство {self._requested_device} не найдено или не имеет входа."
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
            # Opening the stream here checks that PortAudio can actually access it.
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
            raise AudioError(f"Не удалось открыть микрофон: {exc}") from exc

    def hotkey_available(self) -> bool:
        listener = None
        try:
            keyboard = importlib.import_module("pynput.keyboard")
            listener = keyboard.Listener(on_press=lambda _key: None)
            listener.start()
            listener.wait()
            if not listener.is_alive():
                raise RuntimeError("обработчик клавиш завершился при запуске")
            listener.stop()
            listener.join(timeout=1)
            return True
        except Exception:
            if listener is not None:
                try:
                    listener.stop()
                    listener.join(timeout=1)
                except Exception:
                    pass
            return False

    def record_with_f8(self) -> Recording:
        try:
            keyboard = importlib.import_module("pynput.keyboard")
        except Exception as exc:
            raise HotkeyUnavailable(f"pynput недоступен: {exc}") from exc

        pressed = threading.Event()
        released = threading.Event()
        released.set()

        def on_press(key) -> None:
            if key == keyboard.Key.f8 and not pressed.is_set():
                released.clear()
                pressed.set()

        def on_release(key) -> None:
            if key == keyboard.Key.f8 and pressed.is_set():
                released.set()

        listener = None
        try:
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            listener.wait()
        except Exception as exc:
            if listener is not None:
                try:
                    listener.stop()
                    listener.join(timeout=1)
                except Exception:
                    pass
            raise HotkeyUnavailable(f"Не удалось слушать клавишу F8: {exc}") from exc

        print("[GRUB] Готов. Удерживайте F8, чтобы говорить.")
        try:
            while not pressed.wait(timeout=0.1):
                if not listener.is_alive():
                    raise HotkeyUnavailable("обработчик клавиш завершился")

            def listener_still_running() -> None:
                if not listener.is_alive() and not released.is_set():
                    raise HotkeyUnavailable("обработчик клавиш завершился во время записи")

            return self._record_until(released, listener_still_running)
        finally:
            listener.stop()
            listener.join(timeout=1)

    def record_with_enter(self) -> Recording:
        if not sys.stdin.isatty():
            raise AudioError("Резервный режим ENTER требует интерактивного терминала.")
        input("[GRUB] Нажмите ENTER, чтобы начать запись (Ctrl+C — выход). ")
        stop_recording = threading.Event()
        input_thread: threading.Thread | None = None

        def wait_for_enter() -> None:
            try:
                input("[REC] Запись. Нажмите ENTER, чтобы остановить. ")
            except EOFError:
                pass
            finally:
                stop_recording.set()

        def start_stop_key_reader() -> None:
            nonlocal input_thread
            input_thread = threading.Thread(target=wait_for_enter, daemon=True)
            input_thread.start()

        recording = self._record_until(stop_recording, on_started=start_stop_key_reader)
        if input_thread is not None:
            input_thread.join(timeout=0.2)
        return recording

    def _record_until(
        self,
        stop_event: threading.Event,
        monitor: Callable[[], None] | None = None,
        on_started: Callable[[], None] | None = None,
    ) -> Recording:
        if self.device is None:
            raise AudioError("Микрофон не инициализирован.")

        chunks: list[object] = []
        lock = threading.Lock()

        def callback(indata, _frames, _time_info, status) -> None:
            if status:
                print(f"\n[REC] Предупреждение аудиопотока: {status}", file=sys.stderr)
            with lock:
                chunks.append(indata[:, 0].copy())

        try:
            with self._sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=callback,
            ):
                started_at = time.perf_counter()
                print("[REC] Запись началась.")
                if on_started is not None:
                    on_started()
                while not stop_event.wait(timeout=0.05):
                    if monitor is not None:
                        monitor()
            duration = time.perf_counter() - started_at
        except (AudioError, HotkeyUnavailable):
            raise
        except Exception as exc:
            raise AudioError(f"Ошибка записи с микрофона: {exc}") from exc

        with lock:
            audio = (
                self._np.concatenate(chunks).astype("float32", copy=False)
                if chunks
                else self._np.empty(0, dtype="float32")
            )
        print(f"[REC] Запись остановлена ({duration:.2f} с).")
        if duration < MIN_AUDIO_SECONDS or audio.size == 0:
            raise AudioError("Запись слишком короткая; запрос пропущен.")

        if self.sample_rate != TARGET_SAMPLE_RATE:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise AudioError("Для аудиоресемплинга установите scipy из requirements.txt.") from exc
            divisor = gcd(self.sample_rate, TARGET_SAMPLE_RATE)
            audio = resample_poly(
                audio,
                TARGET_SAMPLE_RATE // divisor,
                self.sample_rate // divisor,
            ).astype("float32", copy=False)

        return Recording(audio, duration)
