"""Language detection service.

This module provides simple language detection utilities intended for
PDF-only workflows. It optionally uses ``langdetect`` when available
and falls back to lightweight heuristics otherwise. The utilities are
designed to fail closed (returning "Unknown") on errors rather than
raising exceptions to callers.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import TypedDict

# Optional dependency detection without importing at module import time
LANGDETECT_AVAILABLE: bool = importlib.util.find_spec("langdetect") is not None

# Removed legacy PDF engine dependency as part of Dolphin OCR migration
FITZ_AVAILABLE = False

DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)


LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
}


class LanguagePattern(TypedDict):
    """Heuristic language pattern configuration for a language.

    Keys:
        words: Common function words for the language (lowercase).
        chars: Distinctive characters/diacritics typical for the language.
        word_weight: Weight applied to word matches in scoring.
        char_weight: Weight applied to character matches in scoring.
    """

    words: list[str]
    chars: list[str]
    word_weight: float
    char_weight: float


LANGUAGE_PATTERNS: dict[str, LanguagePattern] = {
    "German": {
        "words": [
            "der",
            "die",
            "das",
            "und",
            "ist",
            "ein",
            "eine",
            "mit",
            "von",
            "zu",
        ],
        "chars": ["ä", "ö", "ü", "ß"],
        "word_weight": 1.0,
        "char_weight": 2.0,
    },
    "English": {
        "words": [
            "the",
            "and",
            "is",
            "a",
            "an",
            "with",
            "of",
            "to",
            "in",
            "for",
        ],
        "chars": [],
        "word_weight": 1.0,
        "char_weight": 0.0,
    },
    "Spanish": {
        "words": [
            "el",
            "la",
            "y",
            "es",
            "un",
            "una",
            "con",
            "de",
            "en",
            "para",
        ],
        "chars": ["ñ", "á", "é", "í", "ó", "ú"],
        "word_weight": 1.0,
        "char_weight": 2.0,
    },
    "French": {
        "words": [
            "le",
            "la",
            "et",
            "est",
            "un",
            "une",
            "avec",
            "de",
            "en",
            "pour",
        ],
        "chars": ["à", "é", "è", "ê", "ç", "ô"],
        "word_weight": 1.0,
        "char_weight": 2.0,
    },
}


class LanguageDetector:
    """Detects language of document content."""

    def __init__(self) -> None:
        """Initialize detector with module-level language mappings."""
        self.language_map = LANGUAGE_MAP
        # Caches for optional langdetect dependency
        self._langdetect_initialized = False
        self._langdetect_mod = None
        self._langdetect_exception = None
        self._detect_func = None

    def _ensure_langdetect(self) -> None:
        """Lazily import and cache langdetect modules/functions.

        This avoids repeated dynamic imports on every call while
        remaining resilient when the optional dependency is missing.
        """
        if self._langdetect_initialized:
            return
        self._langdetect_initialized = True
        if not LANGDETECT_AVAILABLE:
            return
        try:
            mod = importlib.import_module("langdetect")
            exc_mod = importlib.import_module(
                "langdetect.lang_detect_exception"
            )
            fallback_exc = type("FallbackLangDetectException", (RuntimeError,), {})
            exc = getattr(exc_mod, "LangDetectException", fallback_exc)
            detect_func = getattr(mod, "detect", None)
            self._langdetect_mod = mod
            self._langdetect_exception = exc
            self._detect_func = detect_func
        except (ModuleNotFoundError, ImportError, AttributeError) as err:
            logger.debug("langdetect import failed: %s", err)

    def detect_language(self, file_path: str) -> str:
        """Detect the language of a document from its file path.

        Extracts sample text from the document and uses language
        detection algorithms to identify the language. Project scope is
        PDF-only.

        Args:
            file_path: Path to the document file to analyze. Supported
                format is ``.pdf``.

        Returns:
            str: Detected language name (e.g., "German", "English") or
            "Unknown" if the language cannot be determined or if an
            error occurs. Language codes are mapped to full language
            names using the internal ``language_map`` dictionary.

        Note:
            This method uses the langdetect library if available,
            otherwise falls back to simple heuristic-based detection.
            Errors are handled internally and result in "Unknown"
            rather than propagating to the caller.
        """
        # Extract sample text from document
        sample_text = self._extract_sample_text(file_path)

        if not sample_text or len(sample_text.strip()) < 50:
            return "Unknown"

        if LANGDETECT_AVAILABLE:
            self._ensure_langdetect()
            detect_func = self._detect_func
            lang_exc = self._langdetect_exception
            if detect_func is None or lang_exc is None:
                return self._simple_language_detection(sample_text)
            try:
                detected_code = detect_func(sample_text)
            except (lang_exc, ValueError) as e:  # type: ignore[misc]
                logger.warning("Language detection error: %s", e)
                return "Unknown"
            return self.language_map.get(detected_code, detected_code.upper())

        # Fallback to simple heuristics
        return self._simple_language_detection(sample_text)

    def _extract_sample_text(self, file_path: str, _max_chars: int = 2000) -> str:
        """Extract sample text for language detection from PDF files.

        The project is PDF-only. For ``.pdf`` files, language detection
        relies on OCR upstream, so this method returns an empty string.
        Other file types are unsupported and also return an empty string.

        Args:
            file_path: Path to the document file to extract text from
            _max_chars: Reserved for future extensibility. Intentionally
                unused in the current PDF-only pipeline because OCR text is
                produced upstream and this function returns an empty string.

        Returns:
            str: Extracted text sample or empty string if unsupported or if
            an error occurs during text extraction
        """
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".pdf":
                # OCR-first pipeline: PDF text must be pre-extracted upstream
                # Provide explicit guidance for operators on where to look
                logger.info(
                    (
                        "PDF text not available locally; expecting upstream OCR. "
                        "Verify OCR service/pipeline (env: OCR_SERVICE/OCR_PIPELINE) "
                        "is configured and producing text for: %s"
                    ),
                    file_path,
                )
                return ""

            return ""

        except (OSError, UnicodeDecodeError) as e:
            logger.error("Text extraction error: %s", e)
            return ""

    def _simple_language_detection(self, text: str) -> str:
        """Simple language detection based on character patterns.

        Args:
            text: Input text to analyze

        Returns:
            str: Detected language or 'Unknown' if not confident
        """
        # Early return for insufficient input
        if not text or len(text.strip()) < 10:
            return "Unknown"

        text = text.lower()
        words = text.split()
        word_count = len(words)

        if word_count == 0:
            return "Unknown"

        # Use precomputed language patterns defined at module scope
        language_patterns = LANGUAGE_PATTERNS

        # Calculate normalized scores
        scores: dict[str, float] = {}
        for lang, patterns in language_patterns.items():
            # Count matching words and characters
            pattern_words: list[str] = patterns["words"]
            pattern_chars: list[str] = patterns["chars"]
            word_weight: float = patterns["word_weight"]
            char_weight: float = patterns["char_weight"]

            word_matches = sum(1 for token in pattern_words if token in words)
            char_matches = sum(1 for ch in pattern_chars if ch in text)

            # Calculate normalized scores (per 100 words)
            word_score = (word_matches * word_weight) / word_count * 100
            char_score = (char_matches * char_weight) / word_count * 100

            # Combine scores with weights
            scores[lang] = (word_score * 0.7) + (char_score * 0.3)

        # Get language with highest score if above threshold
        if scores:
            best_lang = max(scores.items(), key=lambda kv: kv[1])[0]
            # Only return result if score is above minimum threshold
            # At least 0.5 matches per 100 words
            if scores[best_lang] > 0.5:
                return best_lang

        return "Unknown"

    def detect_language_from_text(self, text: str) -> str:
        """Detect language from provided text using a two-path approach.

        This function validates the input, then uses the best available
        detection method.

        1. Validation: Require at least 10 non-whitespace characters,
           otherwise return "Unknown" without attempting detection.
        2. Detection: Prefer ``langdetect`` if installed; fall back to a
           simple heuristic based on common words and characters.

        Args:
            text: Text to analyze for language detection. Should contain
                at least 10 characters for reliable detection.

        Returns:
            str: Detected language name (e.g., "German", "English") or
            "Unknown" if the language cannot be determined.

        Note:
            This function is resilient to missing dependencies and errors.
        """
        if not text or len(text.strip()) < 10:
            return "Unknown"

        if LANGDETECT_AVAILABLE:
            self._ensure_langdetect()
            detect_func = self._detect_func
            lang_exc = self._langdetect_exception
            if detect_func is None or lang_exc is None:
                return self._simple_language_detection(text)
            try:
                detected_code = detect_func(text)
            except (lang_exc, ValueError) as e:  # type: ignore[misc]
                logger.warning("Language detection error: %s", e)
                return "Unknown"
            return self.language_map.get(detected_code, detected_code.upper())

        # Fallback to simple heuristics
        return self._simple_language_detection(text)

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages.

        Returns:
            list[str]: Human-readable names (e.g., ["English", "German"]).
        """
        return list(self.language_map.values())
