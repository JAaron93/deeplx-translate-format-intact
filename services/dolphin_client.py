"""Async client for the Dolphin PDF layout micro-service.

The micro-service exposes a single endpoint `/layout` that accepts a multipart
upload of a PDF file and returns JSON describing the page layouts.  This helper
wraps the HTTP call so the rest of the codebase doesn’t need to know the wire
format.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Union

import httpx

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "http://localhost:8501/layout"
DEFAULT_TIMEOUT = 120  # seconds


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
    endpoint = os.getenv("DOLPHIN_ENDPOINT", DEFAULT_ENDPOINT)
    pdf_path = pathlib.Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Get timeout from environment variable or use default
    try:
        timeout_seconds = int(
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
            files = {"file": (pdf_path.name, fp, "application/pdf")}
            response = await client.post(endpoint, files=files)

    response.raise_for_status()

    try:
        data = response.json()
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

        # Check for required page-level fields
        required_fields = ["page_number", "width", "height", "elements"]
        for field in required_fields:
            if field not in page:
                raise ValueError(f"Page {i} is missing required field: {field}")

        # Validate elements array
        if not isinstance(page["elements"], list):
            raise ValueError(f"Page {i} 'elements' is not a list")

        # Validate each element in the page
        for j, element in enumerate(page["elements"]):
            if not isinstance(element, dict):
                raise ValueError(f"Element {j} in page {i} is not a dictionary")

            # Check for required element fields
            element_required = ["type", "bbox", "text"]
            for field in element_required:
                if field not in element:
                    raise ValueError(
                        f"Element {j} in page {i} is missing required field: {field}"
                    )

            # Validate bbox format [x0, y0, x1, y1]
            bbox = element.get("bbox", [])
            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(coord, (int, float)) for coord in bbox)
            ):
                raise ValueError(
                    f"Element {j} in page {i} has invalid bbox format. "
                    f"Expected [x0, y0, x1, y1], got {bbox}"
                )

    return data
