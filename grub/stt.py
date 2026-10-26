"""Speech-to-text adapter built on the existing VoiceSTT faster-whisper setup."""

from __future__ import annotations

import importlib.util
import os
import sys
import time


class STTError(RuntimeError):
    """Raised when Whisper cannot load or transcribe audio."""


def configure_cuda_library_path() -> None:
    """Expose pip-installed CUDA libraries to CTranslate2 before native imports."""
    if os.environ.get("GRUB_CUDA_PATH_READY"):
        return

    library_dirs: list[str] = []
    for package in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            for package_dir in spec.submodule_search_locations:
                library_dir = os.path.join(package_dir, "lib")
                if os.path.isdir(library_dir):
                    library_dirs.append(library_dir)

    existing = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if path]
    additions = [path for path in library_dirs if path not in existing]
    if additions:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = os.pathsep.join(additions + existing)
        env["GRUB_CUDA_PATH_READY"] = "1"
        # glibc reads LD_LIBRARY_PATH at process start; restart before importing CTranslate2.
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


class WhisperSTT:
    def __init__(
        self,
        model_name: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "ru",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def load(self) -> float:
        if self._model is not None:
            return 0.0

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTError(
                "Не установлен faster-whisper. Выполните: python -m pip install -r requirements.txt"
            ) from exc

        started_at = time.perf_counter()
        try:
            # Reuse a cached model without network access. On first use, faster-whisper
            # downloads the selected model and stores it in its normal local cache.
            from faster_whisper.utils import download_model

            try:
                model_source = download_model(self.model_name, local_files_only=True)
            except Exception:
                model_source = self.model_name
            self._model = WhisperModel(
                model_source,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            details = str(exc)
            lowered = details.lower()
            if any(term in lowered for term in ("cuda", "cudnn", "cublas", "out of memory", "memory")):
                suggestion = (
                    "Проверьте драйвер NVIDIA и свободную VRAM. Если LLM и Whisper не помещаются "
                    "одновременно, задайте GRUB_STT_DEVICE=cpu (распознавание будет медленнее)."
                )
                raise STTError(f"Ошибка CUDA при загрузке Whisper: {details}\n{suggestion}") from exc
            raise STTError(f"Не удалось загрузить Whisper {self.model_name}: {details}") from exc
        return time.perf_counter() - started_at

    def transcribe(self, audio) -> tuple[str, float]:
        if self._model is None:
            raise STTError("Whisper ещё не загружен.")
        if audio is None or getattr(audio, "size", 0) == 0:
            return "", 0.0

        started_at = time.perf_counter()
        try:
            language = None if self.language.lower() == "auto" else self.language
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                condition_on_previous_text=False,
            )
            # faster-whisper performs much of its work while the generator is consumed.
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            details = str(exc)
            if any(term in details.lower() for term in ("cuda", "cudnn", "cublas", "out of memory")):
                raise STTError(f"Ошибка CUDA при распознавании: {details}") from exc
            raise STTError(f"Ошибка распознавания речи: {details}") from exc
        return text, time.perf_counter() - started_at
