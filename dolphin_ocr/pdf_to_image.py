"""PDF to image conversion utilities for Dolphin OCR.

This module provides a small, dependency-isolated wrapper around
``pdf2image`` (Poppler) and Pillow to turn PDF pages into OCR-ready images.

Design notes (from design.md):
- The converter should be configurable (DPI, image format)
- Conversion should be memory-efficient (page-wise / temp files)
- Images should be optimized for OCR (grayscale, normalization)

The implementation below follows those principles and avoids importing heavy
libraries at module import time to keep startup overhead low.
"""

from __future__ import annotations

import mmap
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from dolphin_ocr.errors import (
    EncryptedPdfError,
    InvalidDocumentFormatError,
    MemoryExhaustionError,
)

_DEFAULT_DPI = 300
_DEFAULT_FORMAT = "PNG"  # Common, lossless for OCR


@dataclass
class PDFToImageConverter:
    """Convert PDF documents to images for OCR processing.

    Parameters
    - dpi: Output resolution (dots per inch). Recommended 300-600 for OCR
    - image_format: Output image format (e.g., "PNG", "JPEG")
    - poppler_path: Optional poppler binaries path (if not on PATH)

    Memory efficiency:
    Uses pdf2image's ``output_folder`` with ``paths_only=True`` so pages are
    rendered to temporary files and streamed back as bytes instead of keeping
    entire documents in memory.
    """

    dpi: int = _DEFAULT_DPI
    image_format: str = _DEFAULT_FORMAT
    poppler_path: str | None = None

    def convert_pdf_to_images(self, pdf_path: str | Path) -> list[bytes]:
        """Convert a PDF to a list of encoded image bytes.

        Each list element contains the bytes of a single page rendered using
        the configured DPI and image format. The function renders page-wise to
        temporary files to reduce peak memory usage.
        """
        # Validate file early before invoking heavy dependencies
        pdf_path = Path(pdf_path)
        self._validate_pdf_or_raise(pdf_path)

        # Lazy import to avoid heavy deps at import time
        try:
            from pdf2image import convert_from_path
        except Exception as err:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "pdf2image is required to convert PDFs. Please install it and "
                "ensure Poppler is available on your system."
            ) from err

        images: list[bytes] = []
        try:
            with TemporaryDirectory(prefix="dolphin_pdf2img_") as tmp_dir:
                tmp = Path(tmp_dir)
                # Render to temp files; get paths back
                paths: Sequence[str] = convert_from_path(
                    str(pdf_path),
                    dpi=self.dpi,
                    fmt=self.image_format.lower(),
                    output_folder=str(tmp),
                    paths_only=True,
                    poppler_path=self.poppler_path,
                )

                for p in paths:
                    data = Path(p).read_bytes()
                    images.append(data)
        except MemoryError as mem_err:
            raise MemoryExhaustionError(
                context={
                    "path": str(pdf_path),
                    "dpi": self.dpi,
                    "image_format": self.image_format,
                }
            ) from mem_err
        except Exception as err:
            # Map common pdf2image/poppler failures to standardized errors
            message = str(err)
            lower = message.lower()
            invalid_markers = (
                "syntax error",
                "not a pdf",
                "unable to get page count",
                "couldn't find trailer dictionary",
                "corrupt",
            )
            password_markers = ("password", "encrypted", "/encrypt")

            if any(m in lower for m in password_markers):
                raise EncryptedPdfError(context={"path": str(pdf_path)}) from err
            if any(m in lower for m in invalid_markers):
                raise InvalidDocumentFormatError(
                    context={"path": str(pdf_path), "reason": message}
                ) from err
            # Unknown error: re-raise as-is to avoid masking issues
            raise

        return images

    # ---------- Internal helpers ----------
    def _validate_pdf_or_raise(self, pdf_path: Path) -> None:
        """Validate that ``pdf_path`` is an existing, non-encrypted PDF.

        Raises InvalidDocumentFormatError for unsupported formats or corrupt
        files and EncryptedPdfError for encrypted PDFs.
        """
        if not pdf_path.exists():
            raise InvalidDocumentFormatError(
                context={"path": str(pdf_path), "reason": "File does not exist"}
            )

        # Extension check (case-insensitive)
        if pdf_path.suffix.lower() != ".pdf":
            raise InvalidDocumentFormatError(
                context={
                    "path": str(pdf_path),
                    "reason": f"Unsupported extension: {pdf_path.suffix}",
                }
            )

        # Magic number check
        try:
            with pdf_path.open("rb") as f:
                header = f.read(5)
        except OSError as os_err:
            raise InvalidDocumentFormatError(
                context={"path": str(pdf_path), "reason": str(os_err)}
            ) from os_err

        if header != b"%PDF-":
            raise InvalidDocumentFormatError(
                context={
                    "path": str(pdf_path),
                    "reason": "Missing %PDF- header",
                }
            )

        # Best-effort encrypted PDF detection using a lightweight scan
        if self._is_encrypted_pdf(pdf_path):
            raise EncryptedPdfError(context={"path": str(pdf_path)})

    @staticmethod
    def _is_encrypted_pdf(pdf_path: Path) -> bool:
        """Return True if the PDF appears to be encrypted.

        Heuristic: search for '/Encrypt' token anywhere in the file using mmap
        when available. Falls back to a bounded chunked scan. False negatives
        are possible; downstream conversion will still be guarded.
        """
        try:
            with pdf_path.open("rb") as f:
                try:
                    with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                        return mm.find(b"/Encrypt") != -1
                except (OSError, ValueError):
                    # mmap failed, fall through to chunked scan
                    pass
        except OSError:
            # File open failed, fall through to chunked scan
            pass
        # Fallback: scan first few megabytes
        try:
            max_scan = 8 * 1024 * 1024  # 8 MiB
            read = 0
            chunk = 256 * 1024
            with pdf_path.open("rb") as f:
                overlap = b""
                while read < max_scan:
                    data = f.read(chunk)
                    if not data:
                        break
                    haystack = overlap + data
                    if b"/Encrypt" in haystack:
                        return True
                    read += len(data)
                    overlap = haystack[-8:]
        except Exception:
            # If detection fails, be conservative and allow downstream handling
            return False
        return False

    def optimize_image_for_ocr(self, image_bytes: bytes) -> bytes:
        """Apply simple Pillow-based optimizations for OCR.

        Steps (conservative defaults):
        - Convert to grayscale
        - Normalize via point mapping
        - Mild sharpening filter
        """
        try:
            from PIL import Image, ImageFilter
        except Exception:  # pragma: no cover - environment dependent
            # If Pillow is missing, return the original bytes
            return image_bytes

        with Image.open(BytesReader(image_bytes)) as im:
            im = im.convert("L")  # grayscale
            # Basic normalization; conservative to avoid overflow and detail loss
            # - Only map near-white pixels to pure white
            # - Clamp scaled values to 255
            im = im.point(lambda x: 255 if x > 250 else min(255, int(x * 1.1)))
            im = im.filter(ImageFilter.SHARPEN)

            out = BytesWriter()
            im.save(out, format=self.image_format)
            return out.getvalue()


class BytesReader:
    """A minimal in-memory bytes reader compatible with PIL.Image.open.

    Provides an explicit, type-checkable interface rather than dynamic
    delegation via ``__getattr__``.
    """

    def __init__(self, data: bytes) -> None:
        """Initialize the in-memory reader with the provided bytes buffer."""
        from io import BytesIO

        self._bio = BytesIO(data)

    def read(self, n: int = -1) -> bytes:
        """Read up to ``n`` bytes; ``-1`` reads to EOF."""
        return self._bio.read(n)

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move the stream position and return the new absolute position."""
        return self._bio.seek(offset, whence)

    def tell(self) -> int:
        """Return the current stream position."""
        return self._bio.tell()

    def readinto(self, b: bytearray | memoryview) -> int:  # pragma: no cover
        """Read bytes into a pre-allocated, writable bytes-like object."""
        return self._bio.readinto(b)  # type: ignore[arg-type]

    def close(self) -> None:  # pragma: no cover
        """Close the underlying buffer."""
        self._bio.close()

    def readable(self) -> bool:  # pragma: no cover
        """Return True indicating stream is readable."""
        return True


class BytesWriter:
    """A minimal in-memory bytes writer that exposes getvalue()."""

    def __init__(self) -> None:
        """Initialize an empty in-memory buffer for writing bytes."""
        from io import BytesIO

        self._bio = BytesIO()

    def write(self, b: bytes) -> int:  # pragma: no cover - passthrough
        """Write bytes to the internal buffer and return the number written."""
        return self._bio.write(b)

    def getvalue(self) -> bytes:
        """Return the full contents of the internal buffer as bytes."""
        return self._bio.getvalue()
