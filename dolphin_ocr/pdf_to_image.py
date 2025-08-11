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

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

_DEFAULT_DPI = 300
_DEFAULT_FORMAT = "PNG"  # Common, lossless for OCR


@dataclass
class PDFToImageConverter:
    """Convert PDF documents to images for OCR processing.

    Parameters
    - dpi: Output resolution (dots per inch). Recommended 300–600 for OCR
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
        # Lazy import to avoid heavy deps at import time
        try:
            from pdf2image import convert_from_path
        except Exception as err:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "pdf2image is required to convert PDFs. Please install it and "
                "ensure Poppler is available on your system."
            ) from err

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        images: list[bytes] = []
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

        return images

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
        from io import BytesIO

        self._bio = BytesIO()

    def write(self, b: bytes) -> int:  # pragma: no cover - passthrough
        return self._bio.write(b)

    def getvalue(self) -> bytes:
        return self._bio.getvalue()
