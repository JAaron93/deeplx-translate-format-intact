"""Async client for the Dolphin PDF layout micro-service.

This client now supports both local microservice and Modal Labs deployment.
The Modal endpoint provides better performance and scalability compared to
the previous HuggingFace Spaces API approach.

The service exposes an endpoint that accepts a multipart upload of a PDF file
and returns JSON describing the page layouts. This helper wraps the HTTP call
so the rest of the codebase doesn't need to know the wire format.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Union

import httpx

logger: logging.Logger = logging.getLogger(__name__)

# Default endpoints - Modal Labs takes priority
DEFAULT_MODAL_ENDPOINT: str = (
    "https://modal-labs--dolphin-ocr-service-dolphin-ocr-endpoint.modal.run"
)
DEFAULT_LOCAL_ENDPOINT: str = "http://localhost:8501/layout"
DEFAULT_TIMEOUT: int = 300  # seconds (increased for Modal processing)


async def get_layout(pdf_path: Union[str, os.PathLike[str]]) -> dict[str, Any]:
    """Send *pdf_path* to the Dolphin service and return the JSON payload.

    Parameters
    ----------
    pdf_path: str | PathLike
        Absolute path to the PDF (single or multi-page) to analyse.

    Returns:
    -------
    dict
        Whatever JSON structure the Dolphin service responds with.  The caller
        is responsible for interpreting the schema.
    """
    endpoint: str = os.getenv("DOLPHIN_ENDPOINT", DEFAULT_MODAL_ENDPOINT)
    pdf_path = pathlib.Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Get timeout from environment variable or use default
    try:
        timeout_seconds: int = int(
            os.getenv("DOLPHIN_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT))
        )
    except ValueError:
        logger.warning(
            f"Invalid DOLPHIN_TIMEOUT_SECONDS value, using default: {DEFAULT_TIMEOUT}"
        )
        timeout_seconds = DEFAULT_TIMEOUT

    # Use streaming upload to avoid loading big PDFs fully into memory.
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        with pdf_path.open("rb") as fp:
            files: dict[str, tuple[str, Any, str]] = {
                "file": (pdf_path.name, fp, "application/pdf")
            }
            response: httpx.Response = await client.post(endpoint, files=files)

    response.raise_for_status()

    try:
        data: dict[str, Any] = response.json()
    except ValueError as e:
        raise ValueError(f"Invalid JSON response from Dolphin service: {e}") from e

    # Basic validation - check if response has the expected structure
    if not isinstance(data, dict) or "pages" not in data:
        raise ValueError(
            "Invalid response format from Dolphin service: missing 'pages' key"
        )

    if not isinstance(data["pages"], list):
        raise ValueError(
            "Invalid response format from Dolphin service: 'pages' is not a list"
        )

    # Validate each page in the response
    for i, page in enumerate(data["pages"]):
        if not isinstance(page, dict):
            raise ValueError(f"Page {i} is not a dictionary")

        # Check for required page-level fields (updated for Modal format)
        required_fields: list[str] = [
            "page_number", "width", "height", "text_blocks"
        ]
        for field in required_fields:
            if field not in page:
                raise ValueError(
                    f"Page {i} is missing required field: {field}"
                )

        # Validate text_blocks array (Modal format uses text_blocks)
        if not isinstance(page["text_blocks"], list):
            raise ValueError(
                f"Page {i} 'text_blocks' is not a list"
            )

        # Validate each text block in the page
        for j, block in enumerate(page["text_blocks"]):
            if not isinstance(block, dict):
                raise ValueError(
                    f"Text block {j} in page {i} is not a dictionary"
                )

            # Check for required text block fields (Modal format)
            block_required: list[str] = [
                "text", "bbox", "confidence", "block_type"
            ]
            for field in block_required:
                if field not in block:
                    raise ValueError(
                        f"Text block {j} in page {i} is missing required field: {field}"
                    )

            # Validate bbox format [x0, y0, x1, y1]
            bbox_coords: list[Union[int, float]] = (
                block.get("bbox", [])
            )
            if not (
                isinstance(bbox_coords, list)
                and len(bbox_coords) == 4
                and all(
                    isinstance(coord, (int, float)) for coord in bbox_coords
                )
            ):
                raise ValueError(
                    f"Element {j} in page {i} has invalid bbox format. "
                    f"Expected [x0, y0, x1, y1], got {bbox_coords}"
                )

    return data
