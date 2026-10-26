"""Coordinates global push-to-talk, transcription, and Ollama replies."""

from __future__ import annotations

import importlib
import queue
import sys
import threading
import time
from enum import Enum

from .audio import AudioCapture, AudioError, Recording
from .feedback import FeedbackManager
from .llm import OllamaClient, OllamaError
from .stt import STTError, WhisperSTT


SYSTEM_PROMPT = (
    "You are Grub, a local AI voice assistant running on Linux. You are helpful, "
    "technically competent, concise, and conversational. Respond in the user's language. "
    "Your responses will eventually be spoken aloud, so prefer clear and natural sentences. "
    "Do not claim to have performed actions you cannot actually perform."
)


class AssistantState(Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    ERROR = "ERROR"


class VoiceAssistant:
    def __init__(
        self,
        audio: AudioCapture,
        stt: WhisperSTT,
        llm: OllamaClient,
        feedback: FeedbackManager | None = None,
    ) -> None:
        self.audio = audio
        self.stt = stt
        self.llm = llm
        self.feedback = feedback or FeedbackManager()

        self._state = AssistantState.IDLE
        self._state_lock = threading.Lock()
        self._f8_down = False
        self._accepted_f8_press = False
        self._events: queue.Queue[tuple[str, object | None]] = queue.Queue()
        self._keyboard = None
        self._listener = None
        self._worker: threading.Thread | None = None

    @property
    def state(self) -> AssistantState:
        with self._state_lock:
            return self._state

    def run(self) -> None:
        self.llm.check_ready()
        print(f"[OLLAMA] Connected. Model: {self.llm.model}")
        self.audio.check_microphone()
        print("[AUDIO] Microphone ready.")
        model_load_time = self.stt.load()
        print(f"[STT] Whisper loaded in {model_load_time:.2f}s.")

        try:
            self._start_listener()
        except Exception as exc:
            print(
                f"[KEY] Global F8 is unavailable ({exc}); falling back to terminal ENTER mode.",
                file=sys.stderr,
            )
            self._stop_listener()
            if not sys.stdin.isatty():
                raise AudioError(
                    "Global F8 is unavailable and terminal fallback needs an interactive terminal."
                ) from exc
            self._run_terminal_fallback()
            return

        print("[GRUB] Ready. Hold F8 to speak from any window. Press Ctrl+C to exit.")
        try:
            listener_alive = self._run_global_loop()
            self._stop_listener()
            if not listener_alive:
                print("[KEY] Global keyboard listener stopped; switching to ENTER mode.")
                if not sys.stdin.isatty():
                    raise AudioError("F8 listener stopped and terminal fallback needs a TTY.")
                self._run_terminal_fallback()
        finally:
            if self.audio.recording:
                try:
                    self.audio.cancel_recording()
                except Exception as exc:
                    print(f"[AUDIO] Could not close recording stream: {exc}", file=sys.stderr)
            self._stop_listener()
            if self._worker is not None and self._worker.is_alive():
                self._worker.join()

    def _start_listener(self) -> None:
        self._keyboard = importlib.import_module("pynput.keyboard")
        self._listener = self._keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._listener.start()
        self._listener.wait()
        if not self._listener.is_alive():
            raise RuntimeError("keyboard listener exited during startup")

    def _stop_listener(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.stop()
                listener.join(timeout=1)
            except Exception:
                pass

    def _on_key_press(self, key) -> None:
        """pynput callback: only filter F8, update flags, and enqueue an event."""
        if key != self._keyboard.Key.f8:
            return
        with self._state_lock:
            if self._f8_down:
                return  # Ignore key repeat while F8 is held.
            self._f8_down = True
            if self._state is AssistantState.IDLE:
                self._accepted_f8_press = True
                self._events.put(("pressed", None))

    def _on_key_release(self, key) -> None:
        """pynput callback: never records audio or runs STT/LLM work."""
        if key != self._keyboard.Key.f8:
            return
        with self._state_lock:
            if not self._f8_down:
                return
            self._f8_down = False
            if self._accepted_f8_press:
                self._accepted_f8_press = False
                self._events.put(("released", None))

    def _run_global_loop(self) -> bool:
        while True:
            try:
                event, payload = self._events.get(timeout=0.2)
            except queue.Empty:
                if self._listener is None or not self._listener.is_alive():
                    self._abort_recording("keyboard listener exited")
                    self._wait_for_worker()
                    return False
                continue

            if event == "pressed":
                self._begin_recording()
            elif event == "released":
                self._finish_recording()
            elif event == "processing_done":
                self._finish_processing(payload)

    def _begin_recording(self) -> None:
        if self.state is not AssistantState.IDLE:
            return
        self._set_state(AssistantState.LISTENING)
        try:
            self.feedback.recording_started()
            self.audio.start_recording()
            print("[STATE] LISTENING — speak after the start tone.")
        except Exception as exc:
            self._set_state(AssistantState.ERROR)
            print(f"[ERROR] Could not start recording: {exc}", file=sys.stderr)

    def _finish_recording(self) -> None:
        current = self.state
        if current is AssistantState.ERROR:
            self._recover_to_idle()
            return
        if current is not AssistantState.LISTENING:
            return

        recording = None
        error = None
        try:
            recording = self.audio.stop_recording()
        except Exception as exc:
            error = exc
        finally:
            self.feedback.recording_stopped()

        if error is not None:
            self._set_state(AssistantState.ERROR)
            print(f"[ERROR] Could not finish recording: {error}", file=sys.stderr)
            self._recover_to_idle()
            return
        self._set_state(AssistantState.PROCESSING)
        self._start_processing(recording)

    def _start_processing(self, recording: Recording) -> None:
        def run_request() -> None:
            error = None
            try:
                self.process_recording(recording)
            except Exception as exc:
                error = exc
            finally:
                self._events.put(("processing_done", error))

        self._worker = threading.Thread(
            target=run_request,
            name="grub-request",
            daemon=True,
        )
        self._worker.start()

    def _finish_processing(self, error: object | None) -> None:
        if error is not None:
            self._set_state(AssistantState.ERROR)
            print(f"[ERROR] Request failed: {error}", file=sys.stderr)
            self._recover_to_idle()
            return
        self._set_state(AssistantState.IDLE)
        print("[GRUB] Ready. Hold F8 to speak.")

    def _abort_recording(self, reason: str) -> None:
        was_listening = self.state is AssistantState.LISTENING
        if self.audio.recording:
            try:
                self.audio.cancel_recording()
            except Exception as exc:
                print(f"[AUDIO] Could not close recording stream: {exc}", file=sys.stderr)
        if was_listening:
            self.feedback.recording_stopped()
            self._set_state(AssistantState.ERROR)
            print(f"[KEY] Recording stopped because {reason}.", file=sys.stderr)
            self._recover_to_idle()

    def _wait_for_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if worker.is_alive():
            worker.join()
        while True:
            try:
                event, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if event == "processing_done":
                self._finish_processing(payload)

    def _run_terminal_fallback(self) -> None:
        print("[KEY] Terminal fallback: ENTER starts/stops recording; Ctrl+C exits.")
        while True:
            try:
                recording = self.audio.record_with_enter(
                    on_start=self._fallback_recording_started,
                    on_stop=self.feedback.recording_stopped,
                )
            except AudioError as exc:
                self._set_state(AssistantState.ERROR)
                print(f"[AUDIO] {exc}", file=sys.stderr)
                self._recover_to_idle()
                continue

            self._set_state(AssistantState.PROCESSING)
            try:
                self.process_recording(recording)
            except Exception as exc:
                self._set_state(AssistantState.ERROR)
                print(f"[ERROR] Request failed: {exc}", file=sys.stderr)
                self._recover_to_idle()
            else:
                self._set_state(AssistantState.IDLE)
                print("[GRUB] Ready. Press ENTER when you want to speak.")

    def _fallback_recording_started(self) -> None:
        self.feedback.recording_started()
        self._set_state(AssistantState.LISTENING)
        print("[STATE] LISTENING — speak after the start tone.")

    def _recover_to_idle(self) -> None:
        self._set_state(AssistantState.IDLE)
        print("[GRUB] Ready.")

    def _set_state(self, state: AssistantState) -> None:
        with self._state_lock:
            self._state = state

    def process_recording(self, recording: Recording) -> None:
        request_started = time.perf_counter()
        text, stt_time = self.stt.transcribe(recording.samples)
        if not text.strip():
            print("[STT] No speech recognized; empty request ignored.")
            return

        print(f"[STT] {text}")
        print(f"[STT] Time: {stt_time:.2f}s")
        answer, llm_time = self.llm.chat(text)
        total_time = time.perf_counter() - request_started
        print(f"[GRUB] {answer}")
        print(f"[LLM] Time: {llm_time:.2f}s")
        print(f"[GRUB] Total processing time: {total_time:.2f}s")
        self.feedback.notify(answer)
