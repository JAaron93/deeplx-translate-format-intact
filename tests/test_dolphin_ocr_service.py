import json
from typing import Any

import pytest

from dolphin_ocr.errors import (
    ApiRateLimitError,
    AuthenticationError,
    ConfigurationError,
    OcrProcessingError,
    ServiceUnavailableError,
)
from services.dolphin_ocr_service import DolphinOCRService


class _MockResponse:
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class _MockClient:
    def __init__(self, response: _MockResponse):
        self._response = response

    # Context manager compatibility
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *_args, **_kwargs):
        return self._response


class _MockClient429TwiceThen200:
    """Simulate two 429 responses followed by 200 success."""

    def __init__(self):
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls <= 2:
            return _MockResponse(429, {"error": "rate-limited"})
        return _MockResponse(200, {"ok": True})


def test_requires_endpoint(monkeypatch):
    svc = DolphinOCRService(hf_token="t", modal_endpoint=None)
    # Ensure env is empty for endpoint
    monkeypatch.delenv("DOLPHIN_MODAL_ENDPOINT", raising=False)
    with pytest.raises(ConfigurationError):
        svc.process_document_images([b"x"])


def test_requires_token(monkeypatch):
    svc = DolphinOCRService(hf_token=None, modal_endpoint="https://example")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Token missing -> AuthenticationError before performing request
    with pytest.raises(AuthenticationError):
        svc.process_document_images([b"x"])


@pytest.mark.parametrize(
    "status,exc",
    [
        (401, AuthenticationError),
        (403, AuthenticationError),
        (429, ApiRateLimitError),
        (500, ServiceUnavailableError),
        (418, OcrProcessingError),
    ],
)
def test_http_status_mapped_to_errors(status, exc):
    svc = DolphinOCRService(hf_token="t", modal_endpoint="https://example")
    svc._client = _MockClient(_MockResponse(status, {"error": "x"}))
    with pytest.raises(exc):
        svc.process_document_images([b"x"])


def test_success_json_parsed():
    svc = DolphinOCRService(hf_token="t", modal_endpoint="https://example")
    body = {"pages": [], "total_pages": 0}
    svc._client = _MockClient(_MockResponse(200, body))
    out = svc.process_document_images([b"x"])
    assert out == body


def test_invalid_json_body_maps_to_processing_error():
    svc = DolphinOCRService(hf_token="t", modal_endpoint="https://example")
    svc._client = _MockClient(_MockResponse(200, "not-json"))
    with pytest.raises(OcrProcessingError):
        svc.process_document_images([b"x"])


def test_empty_images_list_maps_to_processing_error():
    svc = DolphinOCRService(hf_token="t", modal_endpoint="https://example")
    with pytest.raises(OcrProcessingError):
        svc.process_document_images([])


def test_image_size_validation():
    svc = DolphinOCRService(hf_token="t", modal_endpoint="https://example")
    svc.max_image_bytes = 10
    small = b"1234567890"  # exactly 10
    big = b"12345678901"  # 11
    # Exactly at limit passes
    svc._client = _MockClient(_MockResponse(200, {"ok": True}))
    assert svc.process_document_images([small]) == {"ok": True}
    # Over the limit fails
    with pytest.raises(OcrProcessingError):
        svc.process_document_images([big])


def test_max_images_validation():
    svc = DolphinOCRService(hf_token="t", modal_endpoint="https://example")
    svc.max_images = 2
    svc._client = _MockClient(_MockResponse(200, {"ok": True}))
    assert svc.process_document_images([b"a"]) == {"ok": True}
    assert svc.process_document_images([b"a", b"b"]) == {"ok": True}
    with pytest.raises(OcrProcessingError):
        svc.process_document_images([b"a", b"b", b"c"])


def test_retry_mechanism_with_backoff():
    svc = DolphinOCRService(hf_token="t", modal_endpoint="https://example")
    # Inject mock client that returns 429 twice then 200
    svc._client = _MockClient429TwiceThen200()
    # Make backoff predictable and fast
    sleeps: list[float] = []

    def _fake_sleep(s: float):
        sleeps.append(s)

    svc._sleeper = _fake_sleep
    svc.max_attempts = 3
    svc.backoff_base_seconds = 0.01

    out = svc.process_document_images([b"x"])
    assert out == {"ok": True}
    # Ensure exponential backoff applied at least twice
    assert len(sleeps) >= 2
    assert sleeps[0] == pytest.approx(0.01, rel=1e-3)
    assert sleeps[1] == pytest.approx(0.02, rel=1e-3)
