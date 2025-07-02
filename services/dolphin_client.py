"""Async client for the Dolphin PDF layout micro-service.

The micro-service exposes a single endpoint `/layout` that accepts a multipart
upload of a PDF file and returns JSON describing the page layouts.  This helper
wraps the HTTP call so the rest of the codebase doesn’t need to know the wire
format.
"""
from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, Union

import httpx

DEFAULT_ENDPOINT = "http://localhost:8501/layout"
DEFAULT_TIMEOUT = 120  # seconds


async def get_layout(pdf_path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
    """Send *pdf_path* to the Dolphin service and return the JSON payload.

    Parameters
    ----------
    pdf_path: str | PathLike
        Absolute path to the PDF (single or multi-page) to analyse.

    Returns
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
    timeout_seconds = int(os.getenv("DOLPHIN_TIMEOUT_SECONDS", DEFAULT_TIMEOUT))
    
    # Use streaming upload to avoid loading big PDFs fully into memory.
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        with pdf_path.open("rb") as fp:
            files = {"file": (pdf_path.name, fp, "application/pdf")}
            response = await client.post(endpoint, files=files)

    response.raise_for_status()
    
    try:
        data = response.json()
    except ValueError as e:
        raise ValueError(f"Invalid JSON response from Dolphin service: {e}")

    # Basic validation - check if response has the expected structure
    if not isinstance(data, dict) or 'pages' not in data:
        raise ValueError("Invalid response format from Dolphin service: missing 'pages' key")
        
    if not isinstance(data['pages'], list):
        raise ValueError("Invalid response format from Dolphin service: 'pages' is not a list")
        
    return data
