"""Language detection service."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from langdetect import detect

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# Optional heavy imports at module level to avoid repeated import overhead
try:
    import fitz  # PyMuPDF

    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from docx import Document as DocxDocument

    DOCX_AVAILABLE = True
except ImportError:
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

LANGUAGE_PATTERNS: dict[str, dict[str, object]] = {
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

    def detect_language(self, file_path: str) -> str:
        """Detect the language of a document from its file path.

        Extracts sample text from the document and uses language detection
        algorithms to identify the language. Supports multiple file formats
        including PDF, DOCX, and plain text files.

        Args:
            file_path: Path to the document file to analyze. Supported formats
                      include .pdf, .docx, and .txt files.

        Returns:
            str: Detected language name (e.g., "German", "English") or "Unknown"
                 if the language cannot be determined or if an error occurs.
                 Language codes are mapped to full language names using the
                 internal language_map dictionary.

        Note:
            This method uses the langdetect library if available, otherwise
            falls back to simple heuristic-based detection. All exceptions
            are caught internally and result in "Unknown" being returned
            rather than propagating errors to the caller.
        """
        try:
            # Extract sample text from document
            sample_text = self._extract_sample_text(file_path)

            if not sample_text or len(sample_text.strip()) < 50:
                return "Unknown"

            if LANGDETECT_AVAILABLE:
                # Use langdetect library
                detected_code = detect(sample_text)
                return self.language_map.get(detected_code, detected_code.upper())
            else:
                # Fallback to simple heuristics
                return self._simple_language_detection(sample_text)

        except Exception as e:
            logger.warning(f"Language detection error: {e}")
            return "Unknown"

    def _extract_sample_text(self, file_path: str, max_chars: int = 2000) -> str:
        """Extract sample text for language detection from supported file formats.

        Supports the following file types:
        - TXT: Plain text files (direct file reading)
        - PDF: Portable Document Format files (using PyMuPDF/fitz library)
        - DOCX: Microsoft Word documents (using python-docx library)

        For unsupported file types or when required libraries are not available,
        returns an empty string. PDF and DOCX support depends on the availability
        of their respective libraries (PyMuPDF and python-docx).

        Args:
            file_path: Path to the document file to extract text from
            max_chars: Maximum number of characters to extract (default: 2000)

        Returns:
            str: Extracted text sample or empty string if file type is unsupported
                 or if an error occurs during text extraction
        """
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".txt":
                with open(file_path, encoding="utf-8") as f:
                    return f.read(max_chars)

            elif file_ext == ".pdf" and FITZ_AVAILABLE:
                doc = fitz.open(file_path)
                try:
                    text = ""
                    for page_num in range(min(3, len(doc))):  # First 3 pages
                        page = doc[page_num]
                        text += page.get_text()
                        if len(text) > max_chars:
                            break
                    return text[:max_chars]
                finally:
                    doc.close()

            elif file_ext == ".docx" and DOCX_AVAILABLE:
                doc = DocxDocument(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + " "
                    if len(text) > max_chars:
                        break
                return text[:max_chars]

            return ""

        except Exception as e:
            logger.error(f"Text extraction error: {e}")
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
        scores = {}
        for lang, patterns in language_patterns.items():
            # Count matching words and characters
            word_matches = sum(1 for word in patterns["words"] if word in words)
            char_matches = sum(1 for char in patterns["chars"] if char in text)

            # Calculate normalized scores (per 100 words)
            word_score = (word_matches * patterns["word_weight"]) / word_count * 100
            char_score = (char_matches * patterns["char_weight"]) / word_count * 100

            # Combine scores with weights
            scores[lang] = (word_score * 0.7) + (char_score * 0.3)

        # Get language with highest score if above threshold
        if scores:
            best_lang = max(scores, key=scores.get)
            # Only return result if score is above minimum threshold
            if scores[best_lang] > 0.5:  # At least 0.5 matches per 100 words
                return best_lang

        return "Unknown"

    def detect_language_from_text(self, text: str) -> str:
        """Detect language from provided text using a two-path approach.

        This function implements a robust language detection strategy that first
        validates the input text and then uses the best available detection method:

        1. Text Validation: Checks if the input text is valid and contains at least
           10 characters after stripping whitespace. Returns "Unknown" immediately
           if the text is too short or empty, as reliable language detection
           requires sufficient text content.

        2. Two-Path Detection:
           - Primary: Uses the langdetect library if available for accurate
             statistical language detection based on character n-grams
           - Fallback: Uses simple heuristic-based detection when langdetect
             is not available, analyzing character patterns and common words

        The function gracefully handles missing dependencies and always returns
        a result, falling back to "Unknown" for any errors or edge cases.

        Args:
            text: Input text to analyze for language detection. Should contain
                  at least 10 characters for reliable detection.

        Returns:
            str: Detected language name (e.g., "German", "English") or "Unknown"
                 if the language cannot be determined, text is too short, or an
                 error occurs. Language codes from langdetect are mapped to full
                 language names using the internal language_map dictionary.

        Note:
            This method requires at least 10 characters of text for detection.
            Shorter texts will immediately return "Unknown" without attempting
            detection, as statistical language detection is unreliable on very
            short text samples.
        """
        try:
            if not text or len(text.strip()) < 10:
                return "Unknown"

            if LANGDETECT_AVAILABLE:
                # Use langdetect library
                detected_code = detect(text)
                return self.language_map.get(detected_code, detected_code.upper())
            else:
                # Fallback to simple heuristics
                return self._simple_language_detection(text)

        except Exception as e:
            logger.warning(f"Language detection error: {e}")
            return "Unknown"

    def get_supported_languages(self) -> list[str]:
        """Get list of supported languages.

        Returns:
            list[str]: List of supported language names (e.g., ["English", "German",
                      "French", ...]). These are the full language names that can be
                      returned by the language detection methods.
        """
        return list(self.language_map.values())
