"""Dolphin OCR HTTP client with HF token auth and Modal endpoint configuration.

This service is endpoint-agnostic and safe to initialize without a configured
endpoint. Validation is performed at call time so local development can proceed
before a real endpoint is available. Error handling maps to standardized
`dolphin_ocr.errors` codes for consistency across the app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from dolphin_ocr.errors import (
    ApiRateLimitError,
    AuthenticationError,
    ConfigurationError,
    OcrProcessingError,
    ServiceUnavailableError,
)

DEFAULT_TIMEOUT_SECONDS = 30


@dataclass
class DolphinOCRService:
    """Thin HTTP client for the Dolphin OCR Modal service.

    Configuration can be supplied directly or sourced from environment:
    - HF token: env var ``HF_TOKEN``
    - Modal endpoint: env var ``DOLPHIN_MODAL_ENDPOINT`` (e.g. https://...)

    Endpoint validation occurs only when making requests so construction is not
    blocked when values are not yet set up in the environment.
    """

    hf_token: str | None = None
    modal_endpoint: str | None = None
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    _client: httpx.Client | None = None

    # -------------------- Public API --------------------
    def process_document_images(self, images: list[bytes]) -> dict[str, Any]:
        """Send a batch of page images to the OCR service.

        Each item of ``images`` is an image bytes payload (PNG/JPEG). This
        method constructs a multipart request and performs the HTTP call.
        """
        if not images:
            raise OcrProcessingError("No images provided", context={"count": 0})

        endpoint = self._require_endpoint()
        headers = self._build_auth_headers()

        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, data in enumerate(images, start=1):
            # Content type is best-effort; server can sniff type if needed
            files.append((
                "images",
                (f"page_{index}.bin", data, "application/octet-stream"),
            ))

        url = f"{endpoint.rstrip('/')}/process-images"
        try:
            with self._get_client() as client:
                resp = client.post(url, headers=headers, files=files, timeout=self.timeout_seconds)
        except httpx.ReadTimeout as err:
            raise ServiceUnavailableError(
                "OCR service timed out",
                context={"endpoint": endpoint, "timeout": self.timeout_seconds},
            ) from err
        except httpx.RequestError as err:
            raise ServiceUnavailableError(
                "Network error contacting OCR service",
                context={"endpoint": endpoint, "error": str(err)},
            ) from err

        # Map non-2xx statuses to standardized errors
        if resp.status_code == 401 or resp.status_code == 403:
            raise AuthenticationError(
                context={"endpoint": endpoint, "status": resp.status_code}
            )
        if resp.status_code == 429:
            raise ApiRateLimitError(
                context={"endpoint": endpoint, "status": resp.status_code}
            )
        if 500 <= resp.status_code <= 599:
            raise ServiceUnavailableError(
                context={"endpoint": endpoint, "status": resp.status_code}
            )
        if resp.status_code >= 400:
            raise OcrProcessingError(
                context={"endpoint": endpoint, "status": resp.status_code, "body": resp.text[:500]}
            )

        # Parse JSON body
        try:
            return resp.json()
        except ValueError as err:
            raise OcrProcessingError(
                "Invalid JSON response from OCR service",
                context={"endpoint": endpoint, "body": resp.text[:500]},
            ) from err

    # -------------------- Internals --------------------
    def _require_endpoint(self) -> str:
        endpoint = (self.modal_endpoint or os.getenv("DOLPHIN_MODAL_ENDPOINT", "")).strip()
        if not endpoint:
            raise ConfigurationError(
                "DOLPHIN_MODAL_ENDPOINT is required",
                context={"env": bool(os.getenv("DOLPHIN_MODAL_ENDPOINT"))},
            )
        return endpoint

    def _build_auth_headers(self) -> dict[str, str]:
        token = (self.hf_token or os.getenv("HF_TOKEN", "")).strip()
        if not token:
            raise AuthenticationError("HF_TOKEN is required for authentication")
        return {"Authorization": f"Bearer {token}"}

    def _get_client(self) -> httpx.Client:
        # Allow injection for tests, but always return a usable client context
        if self._client is not None:
            return self._client
        return httpx.Client()


