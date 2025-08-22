"""Test utilities for Dolphin OCR components."""

from __future__ import annotations

import base64
import importlib.resources as ir
from pathlib import Path


def load_asset_bytes(name: str) -> bytes:
    """Load an asset file as bytes, robust across environments.

    Prefer importlib.resources to avoid brittle relative paths when tests run
    from different working directories or packaged environments. Falls back to
    the repository "assets/" directory if needed, raising a clear error when
    not found.
    """
    # Try common candidate packages that may bundle test assets
    for pkg in ("tests.assets", "assets"):
        try:
            files = ir.files(pkg)
            with files.joinpath(name).open("rb") as f:
                return f.read()
        except (ModuleNotFoundError, FileNotFoundError):
            # Module may not exist or file not present; try next package
            continue

    # Fallback to the project assets folder (repo root / assets)
    fallback_path = Path(__file__).resolve().parent.parent / "assets" / name
    try:
        if not fallback_path.exists():
            msg = (
                f"Test asset not found: {name!r}! Tried packages 'tests.assets', "
                f"'assets' and path {fallback_path!s}. Ensure the 'assets/' "
                "directory contains the expected file."
            )
            raise FileNotFoundError(msg)
        return fallback_path.read_bytes()
    except FileNotFoundError as e:
        # Provide actionable guidance and preserve underlying error
        raise FileNotFoundError(
            f"Missing test asset {name!r} at {fallback_path}. "
            f"Verify repository assets/ directory and filenames."
        ) from e


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def get_sample_pdfs() -> tuple[str, str, str]:
    return (
        "1-chapter-11-pages-klages.pdf",  # Small multi-page (53KB)
        "complex-layout-1-page-klages.pdf",  # Complex layout (335KB)
        "1-chapter-11-pages-klages.pdf",  # Fallback to multi-page for large tests
    )
