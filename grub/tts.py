"""Local Supertonic 3 text-to-speech using ONNX Runtime on the CPU."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass


SUPERTONIC_MODEL_ID = "supertonic-3"
SUPPORTED_ENGINES = ("supertonic",)
SUPPORTED_VOICES = ("M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5")
SUPPORTED_DEVICES = ("cpu",)
SUPPORTED_LANGUAGES = ("na", "ru", "en")
SAMPLE_RATE = 44_100
_AUDIO_GAP_SECONDS = 0.12
_MAX_CHUNK_CHARS = 300


class TTSError(RuntimeError):
    """Raised when a configured TTS engine cannot load or synthesize speech."""


@dataclass(frozen=True)
class SynthesizedAudio:
    samples: object
    sample_rate: int
    synthesis_seconds: float

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def clean_for_speech(text: str) -> str:
    """Remove Markdown and control/symbol characters while retaining words."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~]{1,3}", "", text)
    text = "".join(
        char
        if char.isprintable() and unicodedata.category(char)[0] not in {"C", "S"}
        else " "
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split long text near sentence boundaries, then at whitespace if needed."""
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            boundary = sentence.rfind(" ", 0, max_chars + 1)
            if boundary <= 0:
                boundary = max_chars
            part, sentence = sentence[:boundary].strip(), sentence[boundary:].strip()
            if current:
                chunks.append(current)
                current = ""
            if part:
                chunks.append(part)
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


class SpeechSynthesizer:
    """Load Supertonic 3 once and synthesize locally through CPU ONNX Runtime."""

    def __init__(
        self,
        voice: str | None = None,
        device: str = "cpu",
        engine: str = "supertonic",
        language: str = "na",
    ) -> None:
        if engine not in SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported TTS engine {engine!r}; choose from {SUPPORTED_ENGINES}.")
        if device not in SUPPORTED_DEVICES:
            raise ValueError(f"Unsupported TTS device {device!r}; choose from {SUPPORTED_DEVICES}.")

        voice = voice or "M1"
        if voice not in SUPPORTED_VOICES:
            raise ValueError(
                f"Voice {voice!r} is not supported; choose from {SUPPORTED_VOICES}."
            )
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported TTS language {language!r}; choose from {SUPPORTED_LANGUAGES}.")

        self.engine = engine
        self.voice = voice
        self.device = device
        self.language = language
        self.MODEL_ID = SUPERTONIC_MODEL_ID
        self._model = None
        self._voice_style = None
        self._sample_rate = SAMPLE_RATE

    def load(self) -> float:
        """Load the selected model once and return its wall-clock load time."""
        if self._model is not None:
            return 0.0
        started = time.perf_counter()
        try:
            from supertonic import TTS
        except ImportError as exc:
            raise TTSError("Supertonic is not installed. Install supertonic from requirements.txt.") from exc
        try:
            model = TTS(model=self.MODEL_ID, auto_download=True)
            voice_style = model.get_voice_style(voice_name=self.voice)
            sample_rate = int(getattr(model, "sample_rate", SAMPLE_RATE))
        except Exception as exc:
            raise TTSError(f"Could not load {self.engine} TTS model {self.MODEL_ID}: {exc}") from exc

        self._model = model
        self._voice_style = voice_style
        self._sample_rate = sample_rate
        return time.perf_counter() - started

    def synthesize(self, text: str) -> SynthesizedAudio:
        """Synthesize the cleaned answer and return mono float samples."""
        if self._model is None:
            raise TTSError("TTS model has not been loaded.")
        chunks = split_text(clean_for_speech(text))
        if not chunks:
            raise TTSError("There is no text to synthesize.")

        np = __import__("numpy")
        started = time.perf_counter()
        outputs = []
        try:
            for index, chunk in enumerate(chunks):
                audio, _duration = self._model.synthesize(
                    text=chunk,
                    voice_style=self._voice_style,
                    lang=self.language,
                    total_steps=8,
                )
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    outputs.append(audio)
                if index + 1 < len(chunks):
                    outputs.append(np.zeros(int(self._sample_rate * _AUDIO_GAP_SECONDS), dtype=np.float32))
        except Exception as exc:
            raise TTSError(f"{self.engine} speech synthesis failed: {exc}") from exc

        if not outputs:
            raise TTSError(f"{self.engine} generated empty audio.")
        samples = np.concatenate(outputs)
        return SynthesizedAudio(
            samples=samples,
            sample_rate=self._sample_rate,
            synthesis_seconds=time.perf_counter() - started,
        )
