from __future__ import annotations

import importlib
import time
import warnings
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from dolphin_ocr.layout import BoundingBox, FontInfo


class UnsupportedFormatError(Exception):
    """Raised when an unsupported or invalid document format is encountered.

    Optionally carries an error_code for standardized codes
    (e.g., DOLPHIN_014 for encrypted PDFs).
    """

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        """Initialize the error with an optional standardized error code."""
        super().__init__(message)
        self.error_code = error_code


class DocumentReconstructionError(Exception):
    """Raised when PDF reconstruction fails."""


class PDFEncryptionCheckWarning(RuntimeWarning):
    """Warning raised when PDF encryption could not be checked.

    Emitted when basic validation passes but pypdf-based encryption check
    failed due to environment or parsing issues.
    """


@dataclass(frozen=True)
class TranslatedElement:
    """Translated element with layout and style information."""

    original_text: str
    translated_text: str
    adjusted_text: str | None
    bbox: BoundingBox
    font_info: FontInfo
    layout_strategy: str = "none"
    confidence: float | None = None


@dataclass(frozen=True)
class TranslatedPage:
    """Single page of translated elements."""

    page_number: int
    translated_elements: list[TranslatedElement]
    width: float | None = None
    height: float | None = None
    original_elements: list[TranslatedElement] | None = None


@dataclass(frozen=True)
class TranslatedLayout:
    """All pages of a translated document."""

    pages: list[TranslatedPage]


@dataclass(frozen=True)
class ReconstructionResult:
    """Result of the reconstruction operation."""

    output_path: str
    format: str
    success: bool
    processing_time: float
    warnings: list[str]
    quality_metrics: dict[str, float] | None = None


class PDFDocumentReconstructor:
    """Validate PDF format before any reconstruction work.

    This class focuses on basic format checks:
    - Extension must be .pdf (case-insensitive)
    - File must exist and begin with %PDF- header
    - Reject encrypted PDFs (DOLPHIN_014) when detected via pypdf
    """

    supported_extension = ".pdf"

    @classmethod
    def is_pdf_format(cls, file_path: str | PathLike[str]) -> bool:
        """Return True if the path ends with the configured PDF extension.

        Implemented as a classmethod so subclasses can override
        `supported_extension` and have this method reflect that override.
        Comparison is Unicode-safe and tolerant of missing leading dot in
        subclass overrides.
        """
        ext = Path(file_path).suffix.casefold().lstrip(".")
        configured = str(cls.supported_extension).casefold().lstrip(".")
        return ext == configured

    def validate_pdf_format_or_raise(self, file_path: str | PathLike[str]) -> None:
        """Validate that a file is a readable, non-encrypted PDF.

        Raises UnsupportedFormatError when any requirement is not met.
        - Unsupported extension
        - File not found
        - Missing %PDF- header
        - Encrypted PDF (error_code DOLPHIN_014)
        """
        p = Path(file_path)

        # Extension check first via classmethod to honor subclass overrides
        if not type(self).is_pdf_format(file_path):
            ext = p.suffix or "(none)"
            raise UnsupportedFormatError(
                f"Unsupported format '{ext}'; only PDF is supported."
            )

        # Existence and header check
        if not p.exists():
            raise UnsupportedFormatError(f"File not found: {file_path}")

        try:
            with p.open("rb") as f:
                head = f.read(5)
        except OSError as e:
            raise UnsupportedFormatError(
                f"Unable to read file: {file_path}: {e}"
            ) from e

        if head != b"%PDF-":
            raise UnsupportedFormatError(
                "File content is not a valid PDF (missing %PDF- header)."
            )

        # Encrypted PDF detection via pypdf, if available
        try:
            pypdf = importlib.import_module("pypdf")
            reader = pypdf.PdfReader(str(p))
            if getattr(reader, "is_encrypted", False):
                # Standardized rejection per requirements (DOLPHIN_014)
                raise UnsupportedFormatError(
                    ("Encrypted PDFs not supported - " "please provide unlocked PDF"),
                    error_code="DOLPHIN_014",
                )
        except ModuleNotFoundError:
            # pypdf not installed; skip encryption detection
            pass
        except UnsupportedFormatError:
            # Re-raise our own exceptions
            raise
        except (OSError, ValueError, AttributeError, TypeError) as e:
            # Warn but don't fail on pypdf errors; the file passed basic checks
            msg = f"Could not check PDF encryption: {e}"
            warnings.warn(
                msg,
                category=PDFEncryptionCheckWarning,
                stacklevel=2,
            )

    # ------------------------ Reconstruction API ------------------------
    def reconstruct_pdf_document(
        self,
        *,
        translated_layout: TranslatedLayout,
        original_file_path: str,
        output_path: str,
    ) -> ReconstructionResult:
        """Validate source and reconstruct the translated PDF output."""
        self.validate_pdf_format_or_raise(original_file_path)

        try:
            return self.reconstruct_pdf(
                translated_layout=translated_layout,
                _original_file_path=original_file_path,
                output_path=output_path,
            )
        except (ImportError, OSError, ValueError) as exc:  # pragma: no cover
            raise DocumentReconstructionError(
                f"Failed to reconstruct PDF document: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover
            # Log unexpected exceptions for debugging
            import logging

            logging.exception("Unexpected error during PDF reconstruction")
            raise DocumentReconstructionError(
                f"Unexpected error during PDF reconstruction: {type(exc).__name__}: {exc}"
            ) from exc

    def reconstruct_pdf(
        self,
        *,
        translated_layout: TranslatedLayout,
        _original_file_path: str,
        output_path: str,
    ) -> ReconstructionResult:
        """Render a PDF using ReportLab honoring layout, fonts, and colors."""
        start_time = time.time()
        warnings_out: list[str] = []

        try:
            canvas_mod = importlib.import_module("reportlab.pdfgen.canvas")
            pagesizes_mod = importlib.import_module("reportlab.lib.pagesizes")
            pdfmetrics_mod = importlib.import_module("reportlab.pdfbase.pdfmetrics")
        except ImportError as e:  # pragma: no cover
            raise DocumentReconstructionError(
                f"ReportLab is required for PDF reconstruction: {e}"
            ) from e

        # Create canvas with a default page size; will override per page.
        letter_pagesize = getattr(pagesizes_mod, "letter", (612.0, 792.0))
        c = canvas_mod.Canvas(output_path, pagesize=letter_pagesize)

        total_elements = 0
        overflow_count = 0
        font_fallback_count = 0

        for page in translated_layout.pages:
            # Determine page size
            if page.width and page.height:
                page_width, page_height = float(page.width), float(page.height)
            else:
                candidates: list[TranslatedElement] = []
                if page.translated_elements:
                    candidates.extend(page.translated_elements)
                if page.original_elements:
                    candidates.extend(page.original_elements)
                if candidates:
                    max_x = 0.0
                    max_y = 0.0
                    for elem in candidates:
                        max_x = max(max_x, elem.bbox.x + elem.bbox.width)
                        max_y = max(max_y, elem.bbox.y + elem.bbox.height)
                    page_width = max(1.0, max_x)
                    page_height = max(1.0, max_y)
                else:
                    page_width, page_height = 612.0, 792.0
                    msg = (
                        "No elements found for page "
                        f"{page.page_number}, using default size"
                    )
                    warnings_out.append(msg)

            c.setPageSize((page_width, page_height))

            # Render each element
            for element in page.translated_elements:
                total_elements += 1
                try:
                    x = float(element.bbox.x)
                    y_top = float(element.bbox.y)
                    box_width = float(element.bbox.width)
                    box_height = float(element.bbox.height)

                    font_name = self._select_font_name(element.font_info)
                    font_size = max(1.0, float(element.font_info.size))

                    if pdfmetrics_mod.getFont(font_name) is None:
                        fallback = self._fallback_font_name(element.font_info)
                        if pdfmetrics_mod.getFont(fallback) is None:
                            fallback = "Helvetica"
                        warnings_out.append(
                            f"Font {font_name} not available, using {fallback}"
                        )
                        font_fallback_count += 1
                        font_name = fallback

                    c.setFont(font_name, font_size)

                    # Color preservation (RGB 0-255)
                    if element.font_info.color:
                        r, g, b = element.font_info.color
                        c.setFillColorRGB(r / 255.0, g / 255.0, b / 255.0)

                    # Determine lines
                    if element.adjusted_text and "\n" in element.adjusted_text:
                        text_lines = element.adjusted_text.splitlines()
                    else:
                        text_to_draw = (
                            element.adjusted_text
                            if element.adjusted_text is not None
                            else element.translated_text
                        )
                        text_lines = self._wrap_text_to_width_reportlab(
                            text=text_to_draw,
                            max_width=box_width,
                            font_name=font_name,
                            font_size=font_size,
                            pdfmetrics_module=pdfmetrics_mod,
                        )

                    line_height = font_size * 1.2
                    for idx, line in enumerate(text_lines):
                        line_y = y_top - (idx * line_height)
                        if line_y < (y_top - box_height):
                            msg = (
                                "Text overflow on page "
                                f"{page.page_number}: truncating content"
                            )
                            warnings_out.append(msg)
                            overflow_count += 1
                            break
                        c.drawString(x, line_y, line)

                except (ValueError, RuntimeError, TypeError) as element_error:
                    msg = (
                        "Failed to render element on page "
                        f"{page.page_number}: {element_error}"
                    )
                    warnings_out.append(msg)
                    continue

            c.showPage()

        c.save()

        processing_time = time.time() - start_time
        quality_metrics = {
            "pages": float(len(translated_layout.pages)),
            "elements": float(total_elements),
            "overflow_rate": (
                float(overflow_count) / float(total_elements)
                if total_elements > 0
                else 0.0
            ),
            "font_fallback_rate": (
                float(font_fallback_count) / float(total_elements)
                if total_elements > 0
                else 0.0
            ),
        }

        return ReconstructionResult(
            output_path=output_path,
            format=".pdf",
            success=True,
            processing_time=processing_time,
            warnings=warnings_out,
            quality_metrics=quality_metrics,
        )

    # ----------------------------- Helpers -----------------------------
    def _select_font_name(self, font: FontInfo) -> str:
        family = (font.family or "Helvetica").strip()
        weight = (font.weight or "normal").lower()
        style = (font.style or "normal").lower()

        is_bold = weight == "bold"
        is_italic = style in {"italic", "oblique"}

        if family.lower() in {"times", "times-roman", "times new roman"}:
            if is_bold and is_italic:
                return "Times-BoldItalic"
            if is_bold:
                return "Times-Bold"
            if is_italic:
                return "Times-Italic"
            return "Times-Roman"

        if family.lower() in {"courier", "monospace"}:
            if is_bold and is_italic:
                return "Courier-BoldOblique"
            if is_bold:
                return "Courier-Bold"
            if is_italic:
                return "Courier-Oblique"
            return "Courier"

        if is_bold and is_italic:
            return "Helvetica-BoldOblique"
        if is_bold:
            return "Helvetica-Bold"
        if is_italic:
            return "Helvetica-Oblique"
        return "Helvetica"

    def _fallback_font_name(self, font: FontInfo) -> str:
        weight = (font.weight or "normal").lower()
        style = (font.style or "normal").lower()
        is_bold = weight == "bold"
        is_italic = style in {"italic", "oblique"}

        if is_bold and is_italic:
            return "Helvetica-BoldOblique"
        if is_bold:
            return "Helvetica-Bold"
        if is_italic:
            return "Helvetica-Oblique"
        return "Helvetica"

    def _wrap_text_to_width_reportlab(
        self,
        *,
        text: str,
        max_width: float,
        font_name: str,
        font_size: float,
        pdfmetrics_module,
    ) -> list[str]:
        """Greedy wrap leveraging ReportLab stringWidth metrics."""
        words = text.split()
        if not words:
            return [""]

        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}" if current else word
            if (
                pdfmetrics_module.stringWidth(candidate, font_name, font_size)
                <= max_width
            ):
                current = candidate
            else:
                # If the word itself is too wide, hard-wrap by character
                if (
                    pdfmetrics_module.stringWidth(word, font_name, font_size)
                    > max_width
                ):
                    if current:
                        lines.append(current)
                        current = ""
                    remaining = word
                    while remaining:
                        lo, hi = 1, len(remaining)
                        fit = 1
                        while lo <= hi:
                            mid = (lo + hi) // 2
                            chunk = remaining[:mid]
                            width_ok = (
                                pdfmetrics_module.stringWidth(
                                    chunk, font_name, font_size
                                )
                                <= max_width
                            )
                            if width_ok:
                                fit = mid
                                lo = mid + 1
                            else:
                                hi = mid - 1
                        chunk = remaining[:fit]
                        lines.append(chunk)
                        remaining = remaining[fit:]
                else:
                    lines.append(current)
                    current = word

        if current:
            lines.append(current)

        return lines
