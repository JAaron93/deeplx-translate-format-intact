"""Asynchronous document processing orchestrator.

Implements an async pipeline that coordinates:
- PDF -> image conversion (CPU-bound) using a small process pool
- OCR requests (IO-bound) with basic rate limiting
- Layout-aware translation (IO-bound) using asyncio tasks and batching
- PDF reconstruction (CPU-bound)

This module complements the synchronous processor by providing a drop-in
async alternative for higher throughput and responsive servers.
"""

import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dolphin_ocr.layout import BoundingBox, FontInfo
from dolphin_ocr.pdf_to_image import PDFToImageConverter
from services.dolphin_ocr_service import DolphinOCRService
from services.layout_aware_translation_service import (
    LayoutAwareTranslationService,
    LayoutContext,
    TextBlock,
)
from services.pdf_document_reconstructor import (
    PDFDocumentReconstructor,
    TranslatedElement,
    TranslatedLayout,
    TranslatedPage,
)


@dataclass(frozen=True)
class AsyncProcessingOptions:
    """Options for async processing pipeline."""

    dpi: int = 300
    output_path: str | None = None


@dataclass(frozen=True)
class AsyncDocumentRequest:
    """Request parameters for async processing.

    Attributes:
        file_path: Input PDF path on disk.
        source_language: Source language code.
        target_language: Target language code.
        options: Pipeline options such as DPI and output path.
    """

    file_path: str
    source_language: str
    target_language: str
    options: AsyncProcessingOptions = field(default_factory=AsyncProcessingOptions)


class _TokenBucket:
    """Integer-based token bucket without busy-waiting.

    Uses integer "micro-tokens" to avoid floating drift and computes the
    exact sleep required when the bucket is empty.
    """

    _SCALE = 1_000_000  # micro-tokens per token

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity_tokens = max(1, int(capacity))
        # convert to micro-token units
        self.capacity_micro = self.capacity_tokens * self._SCALE
        self.tokens_micro = self.capacity_micro
        # refill rate in micro-tokens/second (at least 1 to progress)
        rate = max(0.0, float(refill_rate))
        self.refill_per_sec_micro = max(1, int(rate * self._SCALE))
        self._last = time.perf_counter()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while self.tokens_micro < self._SCALE:
                now = time.perf_counter()
                elapsed = now - self._last
                add = int(elapsed * self.refill_per_sec_micro)
                if add:
                    self.tokens_micro = min(
                        self.capacity_micro, self.tokens_micro + add
                    )
                    self._last = now
                    if self.tokens_micro >= self._SCALE:
                        break
                # compute precise sleep time for at least one token
                deficit = self._SCALE - self.tokens_micro
                wait = deficit / float(self.refill_per_sec_micro)
                await asyncio.sleep(wait)
                self._last = time.perf_counter()
            # consume one token
            self.tokens_micro -= self._SCALE


def _convert_pdf_to_images_proc(
    pdf_path: str, dpi: int, image_format: str, poppler_path: str | None
) -> list[bytes]:
    converter = PDFToImageConverter(
        dpi=dpi, image_format=image_format, poppler_path=poppler_path
    )
    return converter.convert_pdf_to_images(pdf_path)


def _optimize_image_proc(image_bytes: bytes, image_format: str) -> bytes:
    converter = PDFToImageConverter(image_format=image_format)
    return converter.optimize_image_for_ocr(image_bytes)


class AsyncDocumentProcessor:
    """Async orchestrator for document processing with basic concurrency.

    Concurrency model:
    - CPU-bound conversion/optimization: process pool
    - OCR and translation: asyncio tasks (IO-bound) with batching and limits
    - Request concurrency: semaphore to cap concurrent requests
    - Provider rate-limiting: token bucket for OCR calls
    """

    def __init__(
        self,
        *,
        converter: PDFToImageConverter,
        ocr_service: DolphinOCRService,
        translation_service: LayoutAwareTranslationService,
        reconstructor: PDFDocumentReconstructor,
        max_concurrent_requests: int = 4,
        translation_batch_size: int = 100,
        translation_concurrency: int = 4,
        ocr_rate_capacity: int = 2,
        ocr_rate_per_sec: float = 1.0,
        process_pool: ProcessPoolExecutor | None = None,
    ) -> None:
        """Initialize the async processor.

        Parameters mirror the synchronous processor but add concurrency and
        rate-limiting controls suitable for async execution.
        """
        self._converter = converter
        self._ocr = ocr_service
        self._translator = translation_service
        self._reconstructor = reconstructor

        self._req_sema = asyncio.Semaphore(max(1, int(max_concurrent_requests)))
        self._tg_limit = max(1, int(translation_concurrency))
        self._batch_size = max(1, int(translation_batch_size))
        self._ocr_bucket = _TokenBucket(ocr_rate_capacity, ocr_rate_per_sec)
        if process_pool is not None:
            self._pool = process_pool
        else:
            # Use a conservative default based on CPU count
            cpu = os.cpu_count() or 2
            workers = max(1, cpu - 1)
            self._pool = ProcessPoolExecutor(max_workers=workers)

    async def process_document(
        self,
        request: AsyncDocumentRequest,
        *,
        on_progress: Callable[[str, dict], None] | None = None,
    ) -> TranslatedLayout:
        """Run the full async pipeline and return a TranslatedLayout."""
        async with self._req_sema:
            if on_progress:
                on_progress("validated", {"path": request.file_path})

            loop = asyncio.get_running_loop()

            # 1) Convert PDF -> Images (process pool)
            images = await loop.run_in_executor(
                self._pool,
                _convert_pdf_to_images_proc,
                request.file_path,
                request.options.dpi,
                self._converter.image_format,
                self._converter.poppler_path,
            )

            if on_progress:
                on_progress("converted", {"pages": len(images)})

            # Optimize each image (process pool map)
            optimized = []
            for b in images:
                opt = await loop.run_in_executor(
                    self._pool,
                    _optimize_image_proc,
                    b,
                    self._converter.image_format,
                )
                optimized.append(opt)

            # 2) OCR (rate-limited)
            await self._ocr_bucket.acquire()
            # Support both async and sync OCR service implementations
            ocr_async = getattr(self._ocr, "process_document_images_async", None)
            if callable(ocr_async):
                ocr_result = await ocr_async(optimized)  # type: ignore[misc]
            else:
                ocr_result = await asyncio.to_thread(
                    self._ocr.process_document_images, optimized
                )
            if on_progress:
                on_progress("ocr", {"pages": len(optimized)})

            # 3) Build TextBlocks
            blocks_per_page = self._parse_ocr_result(ocr_result)

            # 4) Translation with batching + concurrency
            all_blocks: list[TextBlock] = []
            for blocks in blocks_per_page:
                all_blocks.extend(blocks)

            translations: list = [None] * len(all_blocks)

            async def _translate_batch(
                start_index: int, batch: list[TextBlock]
            ) -> None:
                result = await asyncio.to_thread(
                    self._translator.translate_document_batch,
                    text_blocks=batch,
                    source_lang=request.source_language,
                    target_lang=request.target_language,
                )
                for idx, value in enumerate(result):
                    translations[start_index + idx] = value

            async def _bounded_worker(
                start_index: int,
                batch: list[TextBlock],
                sema: asyncio.Semaphore,
            ) -> None:
                async with sema:
                    await _translate_batch(start_index, batch)

            sema = asyncio.Semaphore(self._tg_limit)
            async with asyncio.TaskGroup() as tg:  # Python 3.11+
                for i in range(0, len(all_blocks), self._batch_size):
                    batch = all_blocks[i : i + self._batch_size]
                    tg.create_task(_bounded_worker(i, batch, sema))

            if on_progress:
                on_progress("translated", {"count": len(all_blocks)})

            # 5) Map back to pages and build TranslatedLayout
            pages: list[TranslatedPage] = []
            ti = 0
            for page_blocks in blocks_per_page:
                elems: list[TranslatedElement] = []
                for _ in page_blocks:
                    t = translations[ti]
                    if t is None:
                        raise RuntimeError(f"Translation at index {ti} is None")
                    elems.append(
                        TranslatedElement(
                            original_text=t.source_text,
                            translated_text=t.raw_translation,
                            adjusted_text=t.adjusted_text,
                            bbox=BoundingBox(
                                t.adjusted_bbox.x,
                                t.adjusted_bbox.y,
                                t.adjusted_bbox.width,
                                t.adjusted_bbox.height,
                            ),
                            font_info=FontInfo(
                                family=t.adjusted_font.family,
                                size=t.adjusted_font.size,
                                weight=t.adjusted_font.weight,
                                style=t.adjusted_font.style,
                                color=t.adjusted_font.color,
                            ),
                            layout_strategy=t.strategy.type.value,
                            confidence=t.translation_confidence,
                        )
                    )
                    ti += 1
                pages.append(
                    TranslatedPage(
                        page_number=len(pages) + 1, translated_elements=elems
                    )
                )

            layout = TranslatedLayout(pages=pages)

            # 6) Reconstruct (run blocking task off the loop)
            output_path = request.options.output_path or _default_output_path(
                request.file_path
            )
            await asyncio.to_thread(
                self._reconstructor.reconstruct_pdf_document,
                translated_layout=layout,
                original_file_path=request.file_path,
                output_path=output_path,
            )

            if on_progress:
                on_progress("reconstructed", {"output_path": output_path})

            return layout

    # -------------------------- Helpers --------------------------
    def _parse_ocr_result(self, result: dict) -> list[list[TextBlock]]:
        """Convert OCR service JSON into per-page lists of TextBlock."""
        pages_out: list[list[TextBlock]] = []
        pages = result.get("pages", []) if isinstance(result, dict) else []
        for page in pages:
            page_blocks: list[TextBlock] = []
            blocks = page.get("text_blocks", []) if isinstance(page, dict) else []
            for blk in blocks:
                text = str(blk.get("text", ""))
                bbox = blk.get("bbox", [0.0, 0.0, 100.0, 20.0])
                font = blk.get("font_info", {})
                color_data = font.get("color", (0, 0, 0))
                if isinstance(color_data, (list, tuple)) and (len(color_data) >= 3):
                    color = tuple(color_data[:3])
                else:
                    color = (0, 0, 0)
                font_info = FontInfo(
                    family=str(font.get("family", "Helvetica")),
                    size=float(font.get("size", 12.0)),
                    weight=str(font.get("weight", "normal")),
                    style=str(font.get("style", "normal")),
                    color=(int(color[0]), int(color[1]), int(color[2])),
                )
                layout_ctx = LayoutContext(
                    bbox=BoundingBox(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    font=font_info,
                    ocr_confidence=(
                        float(blk.get("confidence", 0.0))
                        if "confidence" in blk
                        else None
                    ),
                )
                page_blocks.append(TextBlock(text=text, layout=layout_ctx))
            pages_out.append(page_blocks)
        return pages_out


def _default_output_path(input_path: str) -> str:
    """Return default output path for a translated PDF next to input."""
    p = Path(input_path)
    if not p.suffix:
        raise ValueError(
            "Input path must be a file with an extension: " f"{input_path}"
        )
    return str(p.with_name(f"{p.stem}.translated.pdf"))
