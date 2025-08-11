"""Standardized error codes and exception classes for Dolphin OCR system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

# Known sensitive keys that should be redacted when serializing error context
_SENSITIVE_KEYS = {
    "token",
    "authorization",
    "password",
    "secret",
    "api_key",
    "hf_token",
}


def _redact_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Return a redacted copy of context to avoid leaking secrets.

    Any key whose lowercased name appears in _SENSITIVE_KEYS will have its
    value replaced with "***REDACTED***". Keys are left untouched otherwise.
    """
    redacted: dict[str, Any] = {}
    if not ctx:
        return redacted
    for key, value in ctx.items():
        if isinstance(key, str) and key.lower() in _SENSITIVE_KEYS:
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted


# Canonical mapping of error codes to default messages. All exceptions fall
# back to this mapping if no explicit message is provided.
CODE_TO_MESSAGE: dict[str, str] = {
    # API and Authentication Errors
    "DOLPHIN_001": "Rate limit exceeded - HuggingFace API quota reached",
    "DOLPHIN_002": "Service unavailable - Dolphin OCR service down",
    "DOLPHIN_003": "Authentication failure - Invalid HuggingFace token",
    # Processing Errors
    "DOLPHIN_004": "Processing timeout - Document too complex or large",
    "DOLPHIN_005": "Invalid document format - Unsupported file type",
    "DOLPHIN_006": "OCR processing failed - Unable to extract text",
    "DOLPHIN_007": "Layout analysis failed - Complex document structure",
    # Translation Errors
    "DOLPHIN_008": ("Translation service error - Lingo.dev API failure"),
    "DOLPHIN_009": ("Layout preservation failed - Unable to maintain formatting"),
    "DOLPHIN_010": ("Document reconstruction failed - Output generation error"),
    # System Errors
    "DOLPHIN_011": ("Memory exhaustion - Document too large for processing"),
    "DOLPHIN_012": ("Storage error - Unable to save processed document"),
    "DOLPHIN_013": "Configuration error - Invalid system settings",
    "DOLPHIN_014": "Encrypted PDFs not supported - please provide unlocked PDF",
}


def get_error_message(code: str, override: str | None = None) -> str:
    """Return the message for an error code, allowing optional override."""
    if override is not None:
        return override
    return CODE_TO_MESSAGE.get(code, code)


class DolphinError(Exception):
    """Base exception for Dolphin OCR errors.

    Subclasses should set ``error_code`` to one of the DOLPHIN_* codes.
    Message defaults to the canonical mapping but can be overridden.
    """

    error_code: ClassVar[str] = ""

    def __init__(
        self, message: str | None = None, *, context: dict[str, Any] | None = None
    ) -> None:
        """Create a new error.

        Parameters
        - message: Optional explicit message; falls back to mapping.
        - context: Optional structured context to aid debugging.
        """
        if not self.error_code:
            raise ValueError(f"Error code not defined for {self.__class__.__name__}")
        resolved_message = get_error_message(self.error_code, message)
        super().__init__(resolved_message)
        self.context = context.copy() if context else {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error with a redacted context."""
        return {
            "error_code": self.error_code,
            "message": str(self),
            "context": _redact_context(self.context),
        }


class ApiRateLimitError(DolphinError):
    """Rate limit exceeded - HuggingFace API quota reached."""

    error_code = "DOLPHIN_001"


class ServiceUnavailableError(DolphinError):
    """Service unavailable - Dolphin OCR service down."""

    error_code = "DOLPHIN_002"


class AuthenticationError(DolphinError):
    """Authentication failure - Invalid HuggingFace token."""

    error_code = "DOLPHIN_003"


class ProcessingTimeoutError(DolphinError):
    """Processing timeout - Document too complex or large."""

    error_code = "DOLPHIN_004"


class InvalidDocumentFormatError(DolphinError):
    """Invalid document format - Unsupported file type."""

    error_code = "DOLPHIN_005"


class OcrProcessingError(DolphinError):
    """OCR processing failed - Unable to extract text."""

    error_code = "DOLPHIN_006"


class LayoutAnalysisError(DolphinError):
    """Layout analysis failed - Complex document structure."""

    error_code = "DOLPHIN_007"


class TranslationServiceError(DolphinError):
    """Translation service error - Lingo.dev API failure."""

    error_code = "DOLPHIN_008"


class LayoutPreservationError(DolphinError):
    """Layout preservation failed - Unable to maintain formatting."""

    error_code = "DOLPHIN_009"


class DocumentReconstructionError(DolphinError):
    """Document reconstruction failed - Output generation error."""

    error_code = "DOLPHIN_010"


class MemoryExhaustionError(DolphinError):
    """Memory exhaustion - Document too large for processing."""

    error_code = "DOLPHIN_011"


class StorageError(DolphinError):
    """Storage error - Unable to save processed document."""

    error_code = "DOLPHIN_012"


class ConfigurationError(DolphinError):
    """Configuration error - Invalid system settings."""

    error_code = "DOLPHIN_013"


class EncryptedPdfError(DolphinError):
    """Encrypted PDFs not supported - please provide unlocked PDF."""

    error_code = "DOLPHIN_014"


@dataclass
class ErrorResponse:
    """Standard error response shape used across services and APIs."""

    error_code: str
    message: str
    recoverable: bool = False
    retry_after: int | None = None
    estimated_recovery_time: int | None = None
    context: dict[str, Any] | None = None
