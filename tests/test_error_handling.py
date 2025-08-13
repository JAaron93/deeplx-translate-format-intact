from __future__ import annotations

from dolphin_ocr.errors import (
    ApiRateLimitError,
    AuthenticationError,
    EncryptedPdfError,
    ErrorHandlingStrategy,
    InvalidDocumentFormatError,
    ProcessingTimeoutError,
    ServiceUnavailableError,
)


def test_handle_rate_limit_maps_to_code_and_recoverable():
    s = ErrorHandlingStrategy()
    resp = s.handle(ApiRateLimitError(), context={"hf_token": "secret"})
    assert resp.error_code == "DOLPHIN_001"
    assert resp.recoverable is True
    assert resp.retry_after is not None
    assert resp.context and resp.context.get("hf_token") == "***REDACTED***"


def test_handle_service_unavailable_maps_to_code_and_estimated_time():
    s = ErrorHandlingStrategy()
    resp = s.handle(ServiceUnavailableError(), context={"op": "ocr"})
    assert resp.error_code == "DOLPHIN_002"
    assert resp.recoverable is True
    assert resp.estimated_recovery_time == 300


def test_handle_auth_failure_not_recoverable():
    s = ErrorHandlingStrategy()
    resp = s.handle(AuthenticationError(), context={"endpoint": "hf"})
    assert resp.error_code == "DOLPHIN_003"
    assert resp.recoverable is False


def test_handle_processing_timeout_recoverable():
    s = ErrorHandlingStrategy()
    resp = s.handle(ProcessingTimeoutError(), context={"doc": "a.pdf"})
    assert resp.error_code == "DOLPHIN_004"
    assert resp.recoverable is True


def test_handle_invalid_format_not_recoverable():
    s = ErrorHandlingStrategy()
    resp = s.handle(InvalidDocumentFormatError(), context={"path": "x.txt"})
    assert resp.error_code == "DOLPHIN_005"
    assert resp.recoverable is False


def test_handle_encrypted_pdf_not_recoverable():
    s = ErrorHandlingStrategy()
    resp = s.handle(EncryptedPdfError(), context={"path": "y.pdf"})
    assert resp.error_code == "DOLPHIN_014"
    assert resp.recoverable is False
