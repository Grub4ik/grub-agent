"""Silero v5_6 Russian speech synthesis, loaded lazily and kept in memory."""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass


SUPPORTED_VOICES = ("aidar", "eugene", "baya", "kseniya", "xenia")


class TTSError(RuntimeError):
    """Raised when Silero cannot load or synthesize speech."""


@dataclass(frozen=True)
class SynthesizedAudio:
    samples: object
    sample_rate: int
    synthesis_seconds: float

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def clean_for_speech(text: str) -> str:
    """Remove Markdown formatting while keeping readable words and punctuation."""
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


def split_text(text: str, max_chars: int = 450) -> list[str]:
    """Split long speech near sentence boundaries, then at whitespace if needed."""
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
    LANGUAGE = "ru"
    MODEL_ID = "v5_6_ru"
    SAMPLE_RATE = 48_000

    def __init__(self, voice: str = "aidar", device: str = "cpu") -> None:
        if voice not in SUPPORTED_VOICES:
            supported = ", ".join(SUPPORTED_VOICES)
            raise ValueError(f"Unsupported Silero voice {voice!r}; choose from {supported}.")
        self.voice = voice
        self.device = device
        self._model = None
        self._torch = None

    def load(self) -> float:
        if self._model is not None:
            return 0.0
        try:
            import torch
        except ImportError as exc:
            raise TTSError(
                "Silero TTS requires PyTorch. Install a PyTorch build matching the existing CUDA setup; "
                "see the README before installing it."
            ) from exc

        started = time.perf_counter()
        try:
            torch.set_num_threads(4)
            model, _example_text = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language=self.LANGUAGE,
                speaker=self.MODEL_ID,
                trust_repo=True,
            )
            model.to(torch.device(self.device))
        except Exception as exc:
            raise TTSError(f"Could not load Silero {self.MODEL_ID}: {exc}") from exc
        self._torch = torch
        self._model = model
        return time.perf_counter() - started

    def synthesize(self, text: str) -> SynthesizedAudio:
        if self._model is None:
            raise TTSError("Silero TTS has not been loaded.")
        clean_text = clean_for_speech(text)
        chunks = split_text(clean_text)
        if not chunks:
            raise TTSError("There is no text to synthesize.")

        np = __import__("numpy")
        started = time.perf_counter()
        outputs = []
        try:
            for index, chunk in enumerate(chunks):
                audio = self._model.apply_tts(
                    text=chunk,
                    speaker=self.voice,
                    sample_rate=self.SAMPLE_RATE,
                )
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    outputs.append(audio)
                if index + 1 < len(chunks):
                    outputs.append(np.zeros(int(self.SAMPLE_RATE * 0.12), dtype=np.float32))
        except Exception as exc:
            raise TTSError(f"Silero synthesis failed: {exc}") from exc
        if not outputs:
            raise TTSError("Silero generated empty audio.")
        samples = np.concatenate(outputs)
        return SynthesizedAudio(samples, self.SAMPLE_RATE, time.perf_counter() - started)
