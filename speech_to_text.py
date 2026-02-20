"""
speech_to_text.py
=================
Speech-to-text backends with a unified interface.

Supported backends
------------------
* faster-whisper  – OpenAI Whisper running via CTranslate2 (fast, local, offline)
* vosk            – Lightweight offline ASR (lower accuracy, very fast, no GPU needed)

Both backends expose the same `transcribe(audio_bytes) -> str` method so the
rest of the application is backend-agnostic.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class STTBackend(ABC):
    """Abstract Speech-to-Text backend."""

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        """
        Convert raw PCM int16 mono audio bytes → plain text string.

        Parameters
        ----------
        audio_bytes : bytes
            Raw 16-bit PCM, 16 000 Hz, mono.
        language : str | None
            BCP-47 language code of the spoken audio (e.g. "en", "fr").
            Pass None to auto-detect.

        Returns
        -------
        str
            Transcribed text, or empty string on failure.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Run any one-time model loading / warm-up."""


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------

class FasterWhisperSTT(STTBackend):
    """
    Speech recognition using faster-whisper (CTranslate2-optimised Whisper).

    Models available: tiny, base, small, medium, large-v2, large-v3
    Download happens automatically on first use.

    GPU acceleration is used automatically when a CUDA device is available.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        cache_dir: str | Path | None = None,
    ):
        """
        Parameters
        ----------
        model_size : str
            Whisper model variant. Smaller = faster; larger = more accurate.
            Recommended: "base" for real-time use on CPU.
        device : str
            "auto" selects CUDA if available, else CPU.
        compute_type : str
            Quantisation type. "auto" selects int8 for CPU, float16 for GPU.
        cache_dir : path, optional
            Where to store downloaded model weights.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._model = None

    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Load the Whisper model into memory."""
        try:
            from faster_whisper import WhisperModel  # type: ignore

            # Resolve device
            device = self.device
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            # Resolve compute type
            compute_type = self.compute_type
            if compute_type == "auto":
                compute_type = "float16" if device == "cuda" else "int8"

            logger.info(
                "Loading faster-whisper model '%s' on %s (%s) …",
                self.model_size, device, compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                download_root=self.cache_dir,
            )
            logger.info("faster-whisper ready ✓")

        except ImportError:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            )

    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        if self._model is None:
            self.warmup()

        if not audio_bytes:
            return ""

        # Convert int16 PCM bytes → float32 numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_np /= 32768.0  # normalise to [-1, 1]

        try:
            segments, info = self._model.transcribe(
                audio_np,
                language=language,
                beam_size=5,
                vad_filter=True,          # built-in VAD to skip silence
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            text = " ".join(seg.text for seg in segments).strip()
            logger.info("STT [%s]: %s", info.language, text)
            return text
        except Exception as exc:
            logger.error("Transcription error: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Vosk backend
# ---------------------------------------------------------------------------

class VoskSTT(STTBackend):
    """
    Lightweight offline speech recognition using Vosk.

    Requires a downloaded Vosk model for each language.
    Models: https://alphacephei.com/vosk/models
    """

    def __init__(self, model_path: str | Path, sample_rate: int = 16_000):
        """
        Parameters
        ----------
        model_path : str | Path
            Path to the extracted Vosk model directory.
        sample_rate : int
            Must match the recording sample rate (default 16 000).
        """
        self.model_path = str(model_path)
        self.sample_rate = sample_rate
        self._recognizer = None

    # ------------------------------------------------------------------

    def warmup(self) -> None:
        try:
            import vosk  # type: ignore
            import json

            if not Path(self.model_path).exists():
                raise FileNotFoundError(
                    f"Vosk model not found at '{self.model_path}'. "
                    "Download a model from https://alphacephei.com/vosk/models "
                    "and extract it to that path."
                )
            model = vosk.Model(self.model_path)
            self._recognizer = vosk.KaldiRecognizer(model, self.sample_rate)
            self._json = json
            logger.info("Vosk model loaded from %s ✓", self.model_path)
        except ImportError:
            raise RuntimeError("vosk is not installed. Run: pip install vosk")

    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        if self._recognizer is None:
            self.warmup()

        if not audio_bytes:
            return ""

        try:
            self._recognizer.AcceptWaveform(audio_bytes)
            result = self._json.loads(self._recognizer.FinalResult())
            text = result.get("text", "").strip()
            logger.info("Vosk STT: %s", text)
            return text
        except Exception as exc:
            logger.error("Vosk transcription error: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BackendName = Literal["faster-whisper", "vosk"]


def create_stt_backend(
    backend: BackendName = "faster-whisper",
    **kwargs,
) -> STTBackend:
    """
    Factory function – returns the requested STT backend.

    Parameters
    ----------
    backend : str
        "faster-whisper" or "vosk".
    **kwargs
        Passed directly to the backend constructor.
    """
    if backend == "faster-whisper":
        return FasterWhisperSTT(**kwargs)
    elif backend == "vosk":
        return VoskSTT(**kwargs)
    else:
        raise ValueError(f"Unknown STT backend: '{backend}'. Choose 'faster-whisper' or 'vosk'.")
