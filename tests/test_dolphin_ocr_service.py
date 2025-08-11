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


@pytest.mark.parametrize("status,exc", [
    (401, AuthenticationError),
    (403, AuthenticationError),
    (429, ApiRateLimitError),
    (500, ServiceUnavailableError),
    (418, OcrProcessingError),
])
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


