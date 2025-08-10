"""Advanced PDF processor with comprehensive formatting preservation.

Based on amazon-translate-pdf approach using image rendering + text overlay.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import fitz  # type: ignore  # PyMuPDF for PDF manipulation
except Exception:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore
import psutil

logger = logging.getLogger(__name__)


@dataclass
class TextElement:
    """Represents a text element with precise positioning and formatting."""

    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    font_name: str
    font_size: float
    font_flags: int
    color: tuple[float, float, float]
    page_num: int
    confidence: float = 1.0
    line_height: float = 0.0
    char_spacing: float = 0.0
    rotation: float = 0.0


@dataclass
class PageLayout:
    """Represents the complete layout of a PDF page."""

    page_num: int
    width: float
    height: float
    rotation: int
    text_elements: list[TextElement]
    background_image: bytes
    media_box: tuple[float, float, float, float]
    crop_box: tuple[float, float, float, float]

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "page_num": self.page_num,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "text_elements": [asdict(elem) for elem in self.text_elements],
            "media_box": self.media_box,
            "crop_box": self.crop_box,
        }


class AdvancedPDFProcessor:
    """Advanced PDF processor with comprehensive formatting preservation.

    Uses image-based rendering with precise text overlay for maximum accuracy.
    """

    def __init__(self, dpi: int = 300, preserve_images: bool = True) -> None:
        """Initialize the advanced PDF processor.

        Args:
            dpi: Resolution for page rendering (higher = better quality, more memory)
            preserve_images: Whether to preserve embedded images
        """
        self.dpi: int = dpi
        self.preserve_images: bool = preserve_images
        # If current memory usage exceeds this fraction, trigger GC
        self.memory_threshold: float = 0.8  # 80%

    def extract_document_layout(self, pdf_path: str) -> list[PageLayout]:
        """Extract complete document layout with background images and text positioning.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of PageLayout objects containing all page information
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        logger.info(f"Extracting layout from {pdf_path} at {self.dpi} DPI")

        if fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF processing but is not installed"
            )

        doc = fitz.open(pdf_path)
        page_layouts = []

        try:
            total_pages = len(doc)
            logger.info(
                f"Processing {total_pages} pages with advanced layout preservation"
            )

            for page_num in range(total_pages):
                # Check memory usage
                if self._check_memory_usage():
                    logger.warning(
                        "High memory usage detected, running garbage collection"
                    )
                    gc.collect()

                layout = self._extract_page_layout(doc, page_num)
                page_layouts.append(layout)

                logger.info(
                    f"Page {page_num + 1}: {len(layout.text_elements)} text elements extracted"
                )

        finally:
            doc.close()

        logger.info(f"Layout extraction complete: {len(page_layouts)} pages processed")
        return page_layouts

    def _extract_page_layout(self, doc: fitz.Document, page_num: int) -> PageLayout:
        """Extract layout information from a single page."""
        page = doc[page_num]

        # Get page dimensions and properties
        page_rect = page.rect
        rotation = page.rotation
        media_box = tuple(page.mediabox)
        crop_box = tuple(page.cropbox) if page.cropbox != page.mediabox else media_box

        # Render page as high-resolution image
        mat = fitz.Matrix(self.dpi / 72.0, self.dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        background_image = pix.tobytes("png")
        del pix  # Explicitly delete to free memory
        import gc

        gc.collect()  # Force garbage collection

        # Extract text elements with detailed formatting
        text_elements = self._extract_text_elements(page, page_num)

        return PageLayout(
            page_num=page_num,
            width=page_rect.width,
            height=page_rect.height,
            rotation=rotation,
            text_elements=text_elements,
            background_image=background_image,
            media_box=media_box,
            crop_box=crop_box,
        )

    def _extract_text_elements(
        self, page: fitz.Page, page_num: int
    ) -> list[TextElement]:
        """Extract text elements with precise positioning and formatting."""
        text_elements = []

        # Get text with detailed formatting information
        blocks = page.get_text("dict")

        for block in blocks.get("blocks", []):
            if "lines" not in block:
                continue

            for line in block["lines"]:
                line_height = line["bbox"][3] - line["bbox"][1]

                for span in line["spans"]:
                    if not span["text"].strip():
                        continue

                    # Extract color information
                    color = self._normalize_color(span["color"])

                    # Create text element with comprehensive information
                    element = TextElement(
                        text=span["text"],
                        bbox=span["bbox"],
                        font_name=span["font"],
                        font_size=span["size"],
                        font_flags=span["flags"],
                        color=color,
                        page_num=page_num,
                        line_height=line_height,
                        char_spacing=span.get("char_spacing", 0.0),
                        rotation=span.get("angle", 0.0),
                    )

                    text_elements.append(element)

        logger.debug(
            f"Extracted {len(text_elements)} text elements from page {page_num + 1}"
        )
        return text_elements

    def create_translated_pdf(
        self,
        original_layouts: list[PageLayout],
        translated_texts: dict[int, list[str]],
        output_path: str,
    ) -> None:
        """Create a new PDF with translated text while preserving exact formatting.

        Args:
            original_layouts: Original page layouts
            translated_texts: Dictionary mapping page numbers to translated text lists
            output_path: Path for the output PDF
        """
        logger.info(f"Creating translated PDF with {len(original_layouts)} pages")

        # Create new PDF document
        new_doc = fitz.open()

        try:
            for layout in original_layouts:
                self._create_translated_page(
                    new_doc, layout, translated_texts.get(layout.page_num, [])
                )

                logger.info(
                    f"Page {layout.page_num + 1} reconstructed with translation"
                )

            # Save with optimization
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            new_doc.save(output_path, garbage=4, deflate=True, clean=True)

            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Translated PDF saved: {output_path} ({file_size_mb:.2f} MB)")

        finally:
            new_doc.close()

    def _create_translated_page(
        self, doc: fitz.Document, layout: PageLayout, translated_texts: list[str]
    ) -> None:
        """Create a translated page with preserved formatting."""
        # Create new page with exact dimensions
        page = doc.new_page(width=layout.width, height=layout.height)

        # Set page properties
        if layout.rotation != 0:
            page.set_rotation(layout.rotation)

        page.set_mediabox(fitz.Rect(layout.media_box))
        if layout.crop_box != layout.media_box:
            page.set_cropbox(fitz.Rect(layout.crop_box))

        # Insert background image
        if layout.background_image:
            page_rect = fitz.Rect(0, 0, layout.width, layout.height)
            page.insert_image(page_rect, stream=layout.background_image, overlay=False)

        # Add translated text elements
        for i, text_element in enumerate(layout.text_elements):
            # Use translated text if available
            if i < len(translated_texts) and translated_texts[i].strip():
                text_to_insert = translated_texts[i]
            else:
                text_to_insert = text_element.text

            self._insert_text_element(page, text_element, text_to_insert)

    def _insert_text_element(
        self, page: fitz.Page, element: TextElement, text: str
    ) -> None:
        """Insert a text element with precise formatting."""
        try:
            if not text.strip():
                return

            # Calculate insertion point and rectangle
            text_rect = fitz.Rect(element.bbox)
            insert_point = fitz.Point(element.bbox[0], element.bbox[1])

            # Normalize font name
            font_name = self._normalize_font_name(element.font_name)
            font_size = max(element.font_size, 6)  # Minimum readable size

            # Handle text color
            text_color = element.color

            # Try textbox insertion first for better text flow
            rc = page.insert_textbox(
                text_rect,
                text,
                fontname=font_name,
                fontsize=font_size,
                color=text_color,
                align=0,  # Left align
                overlay=True,
            )

            # Fall back to simple text insertion if textbox fails
            if rc < 0:
                page.insert_text(
                    insert_point,
                    text,
                    fontname=font_name,
                    fontsize=font_size,
                    color=text_color,
                    overlay=True,
                )

        except Exception as e:
            logger.warning(f"Failed to insert text element: {e}")

    def create_layout_backup(self, layouts: list[PageLayout], backup_path: str) -> None:
        """Create a backup of the extracted layout information, including background images."""
        try:
            # Determine directory for background images (e.g. my_backup.json -> my_backup_assets/)
            backup_file = Path(backup_path)
            assets_dir = backup_file.with_suffix("")  # strip .json
            assets_dir = assets_dir.parent / f"{assets_dir.name}_assets"
            assets_dir.mkdir(parents=True, exist_ok=True)

            layouts_data: list[dict[str, Any]] = []
            for layout in layouts:
                layout_dict = layout.to_dict()

                # Persist the background image to an external file for storage efficiency
                image_filename = f"page_{layout.page_num}.png"
                image_path = assets_dir / image_filename
                try:
                    if layout.background_image:
                        with open(image_path, "wb") as img_f:
                            img_f.write(layout.background_image)
                    layout_dict["background_image_file"] = image_filename
                except Exception as img_err:
                    logger.warning(
                        f"Failed to save background for page {layout.page_num}: {img_err}"
                    )
                    layout_dict["background_image_file"] = ""

                layouts_data.append(layout_dict)

            backup_data = {
                "version": "1.1",  # bumped version – now stores image assets separately
                "assets_dir": assets_dir.name,
                "total_pages": len(layouts),
                "layouts": layouts_data,
            }

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Layout backup saved to {backup_path} (assets in {assets_dir})"
            )

        except Exception as e:
            logger.error(f"Failed to create layout backup: {e}")

    def load_layout_backup(self, backup_path: str) -> list[PageLayout]:
        """Load layout information from backup and restore background images if available."""
        try:
            backup_file = Path(backup_path)
            with open(backup_file, encoding="utf-8") as f:
                backup_data = json.load(f)

            # Determine assets directory (may be absent in older backups)
            assets_dir_name = backup_data.get("assets_dir")
            assets_dir = (
                (backup_file.parent / assets_dir_name) if assets_dir_name else None
            )

            layouts: list[PageLayout] = []
            for layout_data in backup_data["layouts"]:
                # Reconstruct text elements
                text_elements = [
                    TextElement(**elem) for elem in layout_data["text_elements"]
                ]

                # Restore background image if a path is provided and file exists
                background_image = b""
                image_file = layout_data.get("background_image_file")
                if image_file and assets_dir is not None:
                    image_path = assets_dir / image_file
                    try:
                        if image_path.exists():
                            background_image = image_path.read_bytes()
                    except Exception as img_err:
                        logger.warning(
                            f"Could not load background for page {layout_data['page_num']}: {img_err}"
                        )

                # Create layout object with restored background (if any)
                layout = PageLayout(
                    page_num=layout_data["page_num"],
                    width=layout_data["width"],
                    height=layout_data["height"],
                    rotation=layout_data["rotation"],
                    text_elements=text_elements,
                    background_image=background_image,
                    media_box=tuple(layout_data["media_box"]),
                    crop_box=tuple(layout_data["crop_box"]),
                )
                layouts.append(layout)

            logger.info(f"Loaded {len(layouts)} layouts from backup")
            return layouts

        except Exception as e:
            logger.error(f"Failed to load layout backup: {e}")
            return []

    def generate_preview(self, layouts: list[PageLayout], max_pages: int = 3) -> str:
        """Generate a text preview of the document layout."""
        preview_lines = []

        for _i, layout in enumerate(layouts[:max_pages]):
            preview_lines.append(f"=== Page {layout.page_num + 1} ===")
            preview_lines.append(f"Size: {layout.width:.1f} x {layout.height:.1f}")
            preview_lines.append(f"Text elements: {len(layout.text_elements)}")

            # Show first 10 text elements as preview
            for elem in layout.text_elements[:10]:
                preview_lines.append(
                    f"- {elem.text[:100]}{'...' if len(elem.text) > 100 else ''}"
                )

            if len(layout.text_elements) > 10:
                preview_lines.append(
                    f"... and {len(layout.text_elements) - 10} more elements"
                )

            preview_lines.append("")

        if len(layouts) > max_pages:
            preview_lines.append(f"... and {len(layouts) - max_pages} more pages")

        return "\n".join(preview_lines)

    def _check_memory_usage(self) -> bool:
        """Check if memory usage exceeds threshold."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent > self.memory_threshold
        except (OSError, AttributeError) as e:
            logger.warning(f"Failed to check memory usage: {e}")
            return False

    def _normalize_font_name(self, font_name: str) -> str:
        """Normalize font names for cross-platform compatibility."""
        font_mapping = {
            "Times-Roman": "times-roman",
            "Times-Bold": "times-bold",
            "Times-Italic": "times-italic",
            "Times-BoldItalic": "times-bolditalic",
            "Helvetica": "helv",
            "Helvetica-Bold": "hebo",
            "Helvetica-Oblique": "heob",
            "Helvetica-BoldOblique": "hebobl",  # distinct alias for bold+oblique
            "Courier": "cour",
            "Courier-Bold": "cobo",
            "Courier-Oblique": "coob",
            "Courier-BoldOblique": "cobobl",  # distinct alias for bold+oblique
        }

        return font_mapping.get(font_name, "helv")

    def _normalize_color(self, color_value: int) -> tuple[float, float, float]:
        """Convert color integer to RGB tuple."""
        if isinstance(color_value, int):
            r = (color_value >> 16) & 0xFF
            g = (color_value >> 8) & 0xFF
            b = color_value & 0xFF
            return (r / 255.0, g / 255.0, b / 255.0)
        return (0.0, 0.0, 0.0)

    def _check_memory_usage(self) -> bool:
        """Check if memory usage exceeds threshold."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent > self.memory_threshold
        except (ImportError, AttributeError, OSError):
            return False

    def get_text_for_translation(
        self, layouts: list[PageLayout]
    ) -> dict[int, list[str]]:
        """Extract text from layouts for translation."""
        text_by_page = {}

        for layout in layouts:
            page_texts = []
            for element in layout.text_elements:
                if element.text.strip():
                    page_texts.append(element.text)
            text_by_page[layout.page_num] = page_texts

        return text_by_page

    def cleanup_temp_files(self, temp_dir: str) -> None:
        """Clean up temporary files created during processing."""
        try:
            if os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")
