"""Dolphin OCR HTTP client with HF token auth and Modal endpoint configuration.

This service is endpoint-agnostic and safe to initialize without a configured
endpoint. Validation is performed at call time so local development can proceed
before a real endpoint is available. Error handling maps to standardized
`dolphin_ocr.errors` codes for consistency across the app.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

try:
    from dolphin_ocr.logging_config import (
        get_logger as _get_logger,
    )
    from dolphin_ocr.logging_config import (
        setup_logging as _setup_logging,
    )

    _setup_logging()
    logger = _get_logger("dolphin_ocr.service")
except Exception:  # pragma: no cover - fallback logger
    logger = logging.getLogger("dolphin_ocr.service")
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

from dolphin_ocr.errors import (
    ApiRateLimitError,
    AuthenticationError,
    ConfigurationError,
    OcrProcessingError,
    ServiceUnavailableError,
)

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MiB per image
DEFAULT_MAX_IMAGES = 32
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_SECONDS = 0.5


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
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    max_images: int = DEFAULT_MAX_IMAGES
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS
    _client: httpx.Client | None = None
    _sleeper: Callable[[float], None] | None = None

    # Simple performance counters
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_duration_ms: float = 0.0
    _metrics_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    # -------------------- Public API --------------------
    def process_document_images(self, images: list[bytes]) -> dict[str, Any]:
        """Send a batch of page images to the OCR service.

        Each item of ``images`` is an image bytes payload (PNG/JPEG). This
        method constructs a multipart request and performs the HTTP call.
        """
        if not images:
            raise OcrProcessingError(
                "No images provided",
                context={"count": 0},
            )

        self._validate_images(images)

        endpoint = self._require_endpoint()
        headers = self._build_auth_headers()

        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, data in enumerate(images, start=1):
            # Content type is best-effort; server can sniff type if needed
            files.append(
                (
                    "images",
                    (f"page_{index}.bin", data, "application/octet-stream"),
                )
            )

        url = f"{endpoint.rstrip('/')}/process-images"

        attempts = 0
        start = time.perf_counter()
        try:
            while True:
                attempts += 1
                try:
                    with self._get_client() as client:
                        resp = client.post(
                            url,
                            headers=headers,
                            files=files,
                            timeout=self.timeout_seconds,
                        )
                except httpx.ReadTimeout as err:
                    self._record_metrics(start, success=False)
                    raise ServiceUnavailableError(
                        "OCR service timed out",
                        context={
                            "endpoint": endpoint,
                            "timeout": self.timeout_seconds,
                        },
                    ) from err
                except httpx.RequestError as err:
                    self._record_metrics(start, success=False)
                    raise ServiceUnavailableError(
                        "Network error contacting OCR service",
                        context={"endpoint": endpoint, "error": str(err)},
                    ) from err

                # Map non-2xx statuses to standardized errors
                if resp.status_code in (401, 403):
                    self._record_metrics(start, success=False)
                    raise AuthenticationError(
                        context={"endpoint": endpoint, "status": resp.status_code}
                    )
                if resp.status_code == 429 and attempts <= self.max_retries:
                    delay = self._calculate_backoff_delay(attempts)
                    logger.warning(
                        "Rate limited (429). Retrying attempt %s/%s after %.2fs",
                        attempts,
                        self.max_retries,
                        delay,
                    )
                    self._sleep(delay)
                    continue
                if resp.status_code == 429:
                    self._record_metrics(start, success=False)
                    raise ApiRateLimitError(
                        context={"endpoint": endpoint, "status": resp.status_code}
                    )
                if 500 <= resp.status_code <= 599:
                    self._record_metrics(start, success=False)
                    raise ServiceUnavailableError(
                        context={"endpoint": endpoint, "status": resp.status_code}
                    )
                if resp.status_code >= 400:
                    self._record_metrics(start, success=False)
                    raise OcrProcessingError(
                        context={
                            "endpoint": endpoint,
                            "status": resp.status_code,
                            "body": resp.text[:500],
                        }
                    )

                # Success
                data = self._parse_json(resp, endpoint)
                self._record_metrics(start, success=True)
                return data
        finally:
            # Count 1 public request regardless of retries
            with self._metrics_lock:
                self.total_requests += 1

    # -------------------- Internals --------------------
    def _require_endpoint(self) -> str:
        endpoint = (
            (self.modal_endpoint or os.getenv("DOLPHIN_MODAL_ENDPOINT")) or ""
        ).strip()
        if not endpoint:
            raise ConfigurationError(
                "DOLPHIN_MODAL_ENDPOINT is required",
                context={"env": bool(os.getenv("DOLPHIN_MODAL_ENDPOINT"))},
            )
        return endpoint

    def _build_auth_headers(self) -> dict[str, str]:
        token = ((self.hf_token or os.getenv("HF_TOKEN")) or "").strip()
        if not token:
            raise AuthenticationError("HF_TOKEN is required for authentication")
        return {
            "Authorization": f"Bearer {token}",
        }

    def _get_client(self) -> httpx.Client:
        # Allow injection for tests, but always return a usable client context
        if self._client is not None:
            return self._client
        return httpx.Client()

    def _sleep(self, seconds: float) -> None:
        sleeper = self._sleeper or time.sleep
        sleeper(max(0.0, seconds))

    def _calculate_backoff_delay(self, attempt: int) -> float:
        # attempt is 1-based
        return self.backoff_base_seconds * (2 ** max(0, attempt - 1))

    def _parse_json(self, resp: httpx.Response, endpoint: str) -> dict[str, Any]:
        try:
            return resp.json()
        except ValueError as err:
            raise OcrProcessingError(
                "Invalid JSON response from OCR service",
                context={"endpoint": endpoint, "body": resp.text[:500]},
            ) from err

    def _record_metrics(self, start: float, *, success: bool) -> None:
        duration_ms = (time.perf_counter() - start) * 1000.0
        with self._metrics_lock:
            self.last_duration_ms = duration_ms
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            total = self.total_requests
            ok = self.successful_requests
            fail = self.failed_requests
            success_rate = (ok / max(1, total) * 100.0) if total else 0.0
        logger.info(
            "dolphin_ocr_service_request duration_ms=%.2f success=%s total=%d ok=%d fail=%d success_rate=%.1f%%",
            duration_ms,
            success,
            total,
            ok,
            fail,
            success_rate,
        )

    def _validate_images(self, images: list[bytes]) -> None:
        """Validate number and size of images before making HTTP request.

        Raises OcrProcessingError with a helpful message and context when
        limits are exceeded.
        """
        if len(images) > self.max_images:
            raise OcrProcessingError(
                "Too many images for a single request",
                context={
                    "count": len(images),
                    "max_images": self.max_images,
                },
            )
        for idx, data in enumerate(images, start=1):
            size = len(data)
            if size > self.max_image_bytes:
                raise OcrProcessingError(
                    "Image exceeds maximum allowed size",
                    context={
                        "index": idx,
                        "size_bytes": size,
                        "max_bytes": self.max_image_bytes,
                    },
                )
