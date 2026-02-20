"""
text_to_speech.py
=================
Text-to-speech backends with a unified interface.

Supported backends
------------------
* coqui    – Coqui TTS (high quality, neural, supports many languages)
* pyttsx3  – System TTS engine (no models needed, lower quality, fast)

Both expose `synthesize(text, language) -> Path` which writes a WAV file
and returns its path. The caller is responsible for playback.
"""

from __future__ import annotations

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Default output directory for synthesised audio
OUTPUT_DIR = Path("output_audio")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TTSBackend(ABC):

    @abstractmethod
    def synthesize(self, text: str, language: str = "en", output_path: Path | None = None) -> Path:
        """
        Convert `text` to speech and save as a WAV file.

        Parameters
        ----------
        text : str
            Text to synthesise.
        language : str
            ISO-639-1 language code (e.g. "en", "fr", "de").
        output_path : Path | None
            Where to write the WAV file.  A temp file is used when None.

        Returns
        -------
        Path
            Path to the generated WAV file.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Perform one-time initialisation (model loading, etc.)."""


# ---------------------------------------------------------------------------
# Coqui TTS backend
# ---------------------------------------------------------------------------

class CoquiTTS(TTSBackend):
    """
    High-quality neural TTS using Coqui TTS.

    Recommended models (downloaded automatically):
        tts_models/multilingual/multi-dataset/xtts_v2  ← best quality
        tts_models/en/ljspeech/tacotron2-DDC            ← English only, lighter

    GPU is used automatically when available.
    """

    # Map ISO codes → Coqui language strings for multilingual models
    LANG_MAP: dict[str, str] = {
        "en": "en", "fr": "fr", "de": "de", "es": "es",
        "it": "it", "pt": "pt", "pl": "pl", "tr": "tr",
        "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
        "zh": "zh-cn", "ja": "ja", "ko": "ko", "hu": "hu",
        "hi": "hi",
    }

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        speaker_wav: str | None = None,
        gpu: bool | None = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            Coqui TTS model identifier. List all with: `tts --list_models`.
        speaker_wav : str | None
            Path to a reference WAV for voice cloning (XTTS only).
        gpu : bool | None
            Force GPU on/off. None = auto-detect.
        """
        self.model_name = model_name
        self.speaker_wav = speaker_wav
        self.gpu = gpu
        self._tts = None

    # ------------------------------------------------------------------

    def warmup(self) -> None:
        try:
            from TTS.api import TTS  # type: ignore

            use_gpu = self.gpu
            if use_gpu is None:
                try:
                    import torch
                    use_gpu = torch.cuda.is_available()
                except ImportError:
                    use_gpu = False

            logger.info(
                "Loading Coqui TTS model '%s' (gpu=%s) …",
                self.model_name, use_gpu,
            )
            self._tts = TTS(model_name=self.model_name, gpu=use_gpu)
            logger.info("Coqui TTS ready ✓")
        except ImportError:
            raise RuntimeError(
                "TTS (Coqui) is not installed. "
                "Run: pip install TTS"
            )

    # ------------------------------------------------------------------

    def synthesize(self, text: str, language: str = "en", output_path: Path | None = None) -> Path:
        if self._tts is None:
            self.warmup()

        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / f"tts_{language}_{abs(hash(text))}.wav"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Map language code
        lang_str = self.LANG_MAP.get(language, language)

        try:
            # Check if the model is multilingual
            is_multi = hasattr(self._tts, "is_multi_lingual") and self._tts.is_multi_lingual
            is_multi_speaker = hasattr(self._tts, "is_multi_speaker") and self._tts.is_multi_speaker

            kwargs: dict = {"text": text, "file_path": str(output_path)}

            if is_multi:
                kwargs["language"] = lang_str
            if is_multi_speaker:
                if self.speaker_wav:
                    kwargs["speaker_wav"] = self.speaker_wav
                else:
                    # Use first available speaker when no reference provided
                    kwargs["speaker"] = self._tts.speakers[0] if self._tts.speakers else None

            self._tts.tts_to_file(**kwargs)
            logger.info("TTS saved to %s", output_path)
            return output_path

        except Exception as exc:
            logger.error("Coqui TTS synthesis error: %s", exc)
            # Fall back to silence
            return _write_silent_wav(output_path)


# ---------------------------------------------------------------------------
# pyttsx3 backend
# ---------------------------------------------------------------------------

class Pyttsx3TTS(TTSBackend):
    """
    System TTS using pyttsx3 (wraps espeak / SAPI / NSSpeechSynthesizer).

    Pros : No model download, zero config, works offline everywhere.
    Cons : Robot-like voice quality; limited language support.
    """

    # Map ISO code → pyttsx3 voice search string
    VOICE_HINTS: dict[str, str] = {
        "en": "english", "fr": "french", "de": "german",
        "es": "spanish", "it": "italian", "pt": "portuguese",
        "ru": "russian", "nl": "dutch", "pl": "polish",
        "ar": "arabic", "zh": "chinese", "ja": "japan",
    }

    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Parameters
        ----------
        rate : int
            Speaking rate in words per minute.
        volume : float
            Volume in [0.0, 1.0].
        """
        self.rate = rate
        self.volume = volume
        self._engine = None

    # ------------------------------------------------------------------

    def warmup(self) -> None:
        try:
            import pyttsx3  # type: ignore
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            self._pyttsx3 = pyttsx3
            logger.info("pyttsx3 TTS ready ✓")
        except ImportError:
            raise RuntimeError(
                "pyttsx3 is not installed. "
                "Run: pip install pyttsx3"
            )

    # ------------------------------------------------------------------

    def _set_voice(self, language: str) -> None:
        """Try to select a voice matching the requested language."""
        hint = self.VOICE_HINTS.get(language, language)
        voices = self._engine.getProperty("voices")
        for voice in voices:
            name_lower = voice.name.lower()
            id_lower = voice.id.lower()
            if hint.lower() in name_lower or hint.lower() in id_lower:
                self._engine.setProperty("voice", voice.id)
                logger.debug("pyttsx3 voice: %s", voice.name)
                return
        # Keep default voice if no match
        logger.debug(
            "No pyttsx3 voice found for '%s'; using default.", language
        )

    def synthesize(self, text: str, language: str = "en", output_path: Path | None = None) -> Path:
        if self._engine is None:
            self.warmup()

        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / f"tts_{language}_{abs(hash(text))}.wav"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._set_voice(language)
            self._engine.save_to_file(text, str(output_path))
            self._engine.runAndWait()
            logger.info("pyttsx3 TTS saved to %s", output_path)
            return output_path
        except Exception as exc:
            logger.error("pyttsx3 synthesis error: %s", exc)
            return _write_silent_wav(output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_silent_wav(path: Path) -> Path:
    """Write a 0.5-second silent WAV as a fallback."""
    import wave, struct, math
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00" * 22050)  # 0.5 s silence
    return path


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BackendName = Literal["coqui", "pyttsx3"]


def create_tts_backend(
    backend: BackendName = "pyttsx3",
    **kwargs,
) -> TTSBackend:
    """
    Factory function for TTS backends.

    Parameters
    ----------
    backend : str
        "coqui" for high-quality neural TTS, or "pyttsx3" for system TTS.
    """
    if backend == "coqui":
        return CoquiTTS(**kwargs)
    elif backend == "pyttsx3":
        return Pyttsx3TTS(**kwargs)
    else:
        raise ValueError(
            f"Unknown TTS backend: '{backend}'. Choose 'coqui' or 'pyttsx3'."
        )
