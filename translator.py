"""
translator.py
=============
Text translation backends with a unified interface.

Supported backends
------------------
* argos   – Argos Translate (fully offline, no API key needed)
* marian  – MarianMT via HuggingFace Transformers (offline after model download)

Both expose the same `translate(text, src_lang, tgt_lang) -> str` method.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------------

# Human-readable name → BCP-47 / ISO-639-1 code
LANGUAGES: dict[str, str] = {
    "Afrikaans": "af",
    "Arabic": "ar",
    "Azerbaijani": "az",
    "Catalan": "ca",
    "Chinese (Simplified)": "zh",
    "Czech": "cs",
    "Danish": "da",
    "Dutch": "nl",
    "English": "en",
    "Esperanto": "eo",
    "Finnish": "fi",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hebrew": "he",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Irish": "ga",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Persian": "fa",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Slovak": "sk",
    "Spanish": "es",
    "Swedish": "sv",
    "Thai": "th",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Vietnamese": "vi",
}

# Reverse lookup: code → name
CODE_TO_NAME: dict[str, str] = {v: k for k, v in LANGUAGES.items()}


def get_language_name(code: str) -> str:
    return CODE_TO_NAME.get(code, code.upper())


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TranslationBackend(ABC):

    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate `text` from `src_lang` to `tgt_lang`.

        Parameters
        ----------
        text : str
            Source text.
        src_lang : str
            Source language ISO code, e.g. "en".
        tgt_lang : str
            Target language ISO code, e.g. "fr".

        Returns
        -------
        str
            Translated text, or original text on failure.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Perform any one-time initialisation (model loading, package setup)."""

    @abstractmethod
    def supported_pairs(self) -> list[tuple[str, str]]:
        """Return a list of (src_lang, tgt_lang) pairs this backend supports."""


# ---------------------------------------------------------------------------
# Argos Translate backend
# ---------------------------------------------------------------------------

class ArgosTranslator(TranslationBackend):
    """
    Translation using Argos Translate – 100% offline, no API key.

    Language packages are downloaded once and cached locally.
    Pairs with no direct package are bridged via English (en → X → en → Y).
    """

    def __init__(self, cache_dir: str | None = None):
        """
        Parameters
        ----------
        cache_dir : str | None
            Custom directory for Argos package storage.
            Defaults to Argos' own ~/.argos-translate.
        """
        self._cache_dir = cache_dir
        self._installed: set[tuple[str, str]] = set()
        self._ready = False

    # ------------------------------------------------------------------

    def warmup(self) -> None:
        try:
            import argostranslate.package as pkg
            import argostranslate.translate as tr
            self._pkg = pkg
            self._tr = tr
            self._ready = True
            logger.info("Argos Translate initialised ✓")
        except ImportError:
            raise RuntimeError(
                "argostranslate is not installed. "
                "Run: pip install argostranslate"
            )

    def _ensure_package(self, src: str, tgt: str) -> bool:
        """Download and install the translation package if missing."""
        if (src, tgt) in self._installed:
            return True
        try:
            self._pkg.update_package_index()
            available = self._pkg.get_available_packages()
            package = next(
                (p for p in available if p.from_code == src and p.to_code == tgt),
                None,
            )
            if package is None:
                logger.error("No Argos package for %s → %s", src, tgt)
                return False
            logger.info("Downloading Argos package %s → %s …", src, tgt)
            self._pkg.install_from_path(package.download())
            self._installed.add((src, tgt))
            logger.info("Package %s → %s installed ✓", src, tgt)
            return True
        except Exception as exc:
            logger.error("Failed to install Argos package: %s", exc)
            return False

    # ------------------------------------------------------------------

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not self._ready:
            self.warmup()
        if not text.strip():
            return text
        if src_lang == tgt_lang:
            return text

        # Try direct pair first; fall back to bridging via English
        pairs_to_try = [(src_lang, tgt_lang)]
        if src_lang != "en" and tgt_lang != "en":
            pairs_to_try = [(src_lang, "en"), ("en", tgt_lang)]

        if len(pairs_to_try) == 1:
            if not self._ensure_package(src_lang, tgt_lang):
                logger.warning("Using identity fallback for translation.")
                return text
            try:
                result = self._tr.translate(text, src_lang, tgt_lang)
                logger.info("Translated [%s→%s]: %s", src_lang, tgt_lang, result)
                return result
            except Exception as exc:
                logger.error("Argos translate error: %s", exc)
                return text
        else:
            # Bridge: src → en → tgt
            try:
                self._ensure_package(src_lang, "en")
                intermediate = self._tr.translate(text, src_lang, "en")
                self._ensure_package("en", tgt_lang)
                result = self._tr.translate(intermediate, "en", tgt_lang)
                logger.info(
                    "Bridged [%s→en→%s]: %s", src_lang, tgt_lang, result
                )
                return result
            except Exception as exc:
                logger.error("Argos bridge translate error: %s", exc)
                return text

    def supported_pairs(self) -> list[tuple[str, str]]:
        if not self._ready:
            return []
        try:
            installed = self._pkg.get_installed_packages()
            return [(p.from_code, p.to_code) for p in installed]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# MarianMT (HuggingFace) backend
# ---------------------------------------------------------------------------

class MarianMTTranslator(TranslationBackend):
    """
    Translation using Helsinki-NLP MarianMT models from HuggingFace.

    Models are downloaded once and cached in ~/.cache/huggingface.
    GPU is used automatically when available.
    """

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = cache_dir
        self._models: dict[str, tuple] = {}  # key → (tokenizer, model)
        self._device = "cpu"

    # ------------------------------------------------------------------

    def warmup(self) -> None:
        try:
            from transformers import MarianMTModel, MarianTokenizer  # type: ignore
            self._MarianMTModel = MarianMTModel
            self._MarianTokenizer = MarianTokenizer

            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self._device = "cpu"

            logger.info("MarianMT backend ready (device: %s) ✓", self._device)
        except ImportError:
            raise RuntimeError(
                "transformers is not installed. "
                "Run: pip install transformers sentencepiece sacremoses"
            )

    def _get_model(self, src: str, tgt: str):
        key = f"{src}-{tgt}"
        if key in self._models:
            return self._models[key]

        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        logger.info("Loading MarianMT model %s …", model_name)
        try:
            import torch
            tokenizer = self._MarianTokenizer.from_pretrained(
                model_name, cache_dir=self._cache_dir
            )
            model = self._MarianMTModel.from_pretrained(
                model_name, cache_dir=self._cache_dir
            ).to(self._device)
            model.eval()
            self._models[key] = (tokenizer, model)
            logger.info("Loaded %s ✓", model_name)
            return tokenizer, model
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_name, exc)
            return None, None

    # ------------------------------------------------------------------

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not hasattr(self, "_MarianMTModel"):
            self.warmup()
        if not text.strip() or src_lang == tgt_lang:
            return text

        import torch

        # Try direct pair; bridge via English if unavailable
        tokenizer, model = self._get_model(src_lang, tgt_lang)
        if tokenizer is None and src_lang != "en" and tgt_lang != "en":
            logger.info("Direct model unavailable; bridging via English")
            intermediate = self.translate(text, src_lang, "en")
            return self.translate(intermediate, "en", tgt_lang)

        if tokenizer is None:
            return text

        try:
            inputs = tokenizer([text], return_tensors="pt", padding=True).to(self._device)
            with torch.no_grad():
                translated = model.generate(**inputs)
            result = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            logger.info("MarianMT [%s→%s]: %s", src_lang, tgt_lang, result)
            return result
        except Exception as exc:
            logger.error("MarianMT translate error: %s", exc)
            return text

    def supported_pairs(self) -> list[tuple[str, str]]:
        return list(self._models.keys())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

BackendName = Literal["argos", "marian"]


def create_translator(
    backend: BackendName = "argos",
    **kwargs,
) -> TranslationBackend:
    """
    Factory function for translation backends.

    Parameters
    ----------
    backend : str
        "argos" or "marian".
    """
    if backend == "argos":
        return ArgosTranslator(**kwargs)
    elif backend == "marian":
        return MarianMTTranslator(**kwargs)
    else:
        raise ValueError(f"Unknown translation backend: '{backend}'. Choose 'argos' or 'marian'.")
