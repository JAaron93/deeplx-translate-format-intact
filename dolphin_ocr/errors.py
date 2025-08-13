"""Standardized error codes and exception classes for Dolphin OCR system."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
import logging
import time

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
    "DOLPHIN_008": "Translation service error - Lingo.dev API failure",
    "DOLPHIN_009": "Layout preservation failed - Unable to maintain formatting",
    "DOLPHIN_010": "Document reconstruction failed - Output generation error",
    # System Errors
    "DOLPHIN_011": "Memory exhaustion - Document too large for processing",
    "DOLPHIN_012": "Storage error - Unable to save processed document",
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
        self,
        message: str | None = None,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Create a new error.

        Parameters
        - message: Optional explicit message; falls back to mapping.
        - context: Optional structured context to aid debugging.
        """
        if not self.error_code:
            raise ValueError(
                f"Error code not defined for {self.__class__.__name__}"
            )
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


class ErrorHandlingStrategy:
    """Basic error handling and recovery mapping to standardized codes.

    Provides helpers to translate exceptions into ErrorResponse with
    recoverability hints and logs the event with context.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("dolphin_ocr.errors")

    def _log(
        self, code: str, message: str, context: dict[str, Any] | None
    ) -> None:
        payload = {
            "ts": int(time.time()),
            "error_code": code,
            "message": message,
            "context": _redact_context(context or {}),
        }
        # Structured logging for downstream analysis
        self.logger.error("error: %s", payload)

    def handle(
        self, error: Exception, *, context: dict[str, Any] | None = None
    ) -> ErrorResponse:
        """Map an Exception to an ErrorResponse and log it."""
        # Specific typed exceptions first
        if isinstance(error, ApiRateLimitError):
            return self._handle_rate_limit(error, context)
        if isinstance(error, ServiceUnavailableError):
            return self._handle_service_unavailable(error, context)
        if isinstance(error, AuthenticationError):
            return self._handle_auth_failure(error, context)
        if isinstance(error, ProcessingTimeoutError):
            return self._handle_processing_timeout(error, context)
        if isinstance(error, InvalidDocumentFormatError):
            return self._handle_invalid_format(error, context)
        if isinstance(error, EncryptedPdfError):
            return self._handle_encrypted_pdf(error, context)

        # Fallback for DolphinError subclasses
        if isinstance(error, DolphinError):
            code = error.error_code
            msg = str(error)
            self._log(code, msg, getattr(error, "context", None))
            return ErrorResponse(
                error_code=code,
                message=msg,
                recoverable=False,
                context=_redact_context(getattr(error, "context", {}) or {}),
            )

        # Generic fallback: service unavailable
        msg = str(error)
        self._log("DOLPHIN_002", msg, context)
        return ErrorResponse(
            error_code="DOLPHIN_002",
            message=get_error_message("DOLPHIN_002", msg),
            recoverable=True,
            estimated_recovery_time=300,
            context=_redact_context(context or {}),
        )

    # Specific handlers
    def _handle_rate_limit(
        self, error: Exception, context: dict[str, Any] | None
    ) -> ErrorResponse:
        code = "DOLPHIN_001"
        msg = get_error_message(code, str(error))
        self._log(code, msg, context)
        return ErrorResponse(
            error_code=code,
            message=msg,
            recoverable=True,
            retry_after=60,
            context=_redact_context(context or {}),
        )

    def _handle_service_unavailable(
        self, error: Exception, context: dict[str, Any] | None
    ) -> ErrorResponse:
        code = "DOLPHIN_002"
        msg = get_error_message(code, str(error))
        self._log(code, msg, context)
        return ErrorResponse(
            error_code=code,
            message=msg,
            recoverable=True,
            estimated_recovery_time=300,
            context=_redact_context(context or {}),
        )

    def _handle_auth_failure(
        self, error: Exception, context: dict[str, Any] | None
    ) -> ErrorResponse:
        code = "DOLPHIN_003"
        msg = get_error_message(code, str(error))
        self._log(code, msg, context)
        return ErrorResponse(
            error_code=code,
            message=msg,
            recoverable=False,
            context=_redact_context(context or {}),
        )

    def _handle_processing_timeout(
        self, error: Exception, context: dict[str, Any] | None
    ) -> ErrorResponse:
        code = "DOLPHIN_004"
        msg = get_error_message(code, str(error))
        self._log(code, msg, context)
        return ErrorResponse(
            error_code=code,
            message=msg,
            recoverable=True,
            context=_redact_context(context or {}),
        )

    def _handle_invalid_format(
        self, error: Exception, context: dict[str, Any] | None
    ) -> ErrorResponse:
        code = "DOLPHIN_005"
        msg = get_error_message(code, str(error))
        self._log(code, msg, context)
        return ErrorResponse(
            error_code=code,
            message=msg,
            recoverable=False,
            context=_redact_context(context or {}),
        )

    def _handle_encrypted_pdf(
        self, error: Exception, context: dict[str, Any] | None
    ) -> ErrorResponse:
        code = "DOLPHIN_014"
        msg = get_error_message(code, str(error))
        self._log(code, msg, context)
        return ErrorResponse(
            error_code=code,
            message=msg,
            recoverable=False,
            context=_redact_context(context or {}),
        )
