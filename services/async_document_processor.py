from __future__ import annotations

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
    file_path: str
    source_language: str
    target_language: str
    options: AsyncProcessingOptions = field(default_factory=AsyncProcessingOptions)


class _TokenBucket:
    """Simple token-bucket rate limiter.

    Tokens are added over time at ``refill_rate`` per second up to
    ``capacity``. ``acquire`` waits until a token is available.
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = max(1, int(capacity))
        self.refill_rate = float(refill_rate)
        self._tokens = float(self.capacity)
        self._last = time.perf_counter()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while self._tokens < 1.0:
                now = time.perf_counter()
                elapsed = now - self._last
                refill = self._tokens + elapsed * self.refill_rate
                self._tokens = min(float(self.capacity), refill)
                self._last = now
                if self._tokens < 1.0:
                    await asyncio.sleep(0.01)
            self._tokens -= 1.0


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
        self._converter = converter
        self._ocr = ocr_service
        self._translator = translation_service
        self._reconstructor = reconstructor

        self._req_sema = asyncio.Semaphore(max(1, int(max_concurrent_requests)))
        self._tg_limit = max(1, int(translation_concurrency))
        self._batch_size = max(1, int(translation_batch_size))
        self._ocr_bucket = _TokenBucket(ocr_rate_capacity, ocr_rate_per_sec)
        self._pool = process_pool or ProcessPoolExecutor(max_workers=2)

    async def process_document(
        self,
        request: AsyncDocumentRequest,
        *,
        on_progress: Callable[[str, dict], None] | None = None,
    ) -> TranslatedLayout:
        async with self._req_sema:
            if on_progress:
                on_progress("validated", {"path": request.file_path})

            loop = asyncio.get_running_loop()

            # 1) Convert PDF -> Images (process pool)
            images = await loop.run_in_executor(
                self._pool,
                _convert_pdf_to_images_proc,
                request.file_path,
                self._converter.dpi,
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
            start_idx = 0
            async with asyncio.TaskGroup() as tg:  # Python 3.11+
                for i in range(0, len(all_blocks), self._batch_size):
                    batch = all_blocks[i : i + self._batch_size]
                    tg.create_task(_bounded_worker(i, batch, sema))
                    start_idx += len(batch)

            if on_progress:
                on_progress("translated", {"count": len(all_blocks)})

            # 5) Map back to pages and build TranslatedLayout
            pages: list[TranslatedPage] = []
            ti = 0
            for page_blocks in blocks_per_page:
                elems: list[TranslatedElement] = []
                for _ in page_blocks:
                    t = translations[ti]
                    assert t is not None  # for type-checkers
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
            output_path = (
                request.options.output_path
                or _default_output_path(request.file_path)
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
        pages_out: list[list[TextBlock]] = []
        pages = result.get("pages", []) if isinstance(result, dict) else []
        for page in pages:
            page_blocks: list[TextBlock] = []
            blocks = page.get("text_blocks", []) if isinstance(page, dict) else []
            for blk in blocks:
                text = str(blk.get("text", ""))
                bbox = blk.get("bbox", [0.0, 0.0, 100.0, 20.0])
                font = blk.get("font_info", {})
                color = tuple(font.get("color", (0, 0, 0)))
                font_info = FontInfo(
                    family=str(font.get("family", "Helvetica")),
                    size=float(font.get("size", 12.0)),
                    weight=str(font.get("weight", "normal")),
                    style=str(font.get("style", "normal")),
                    color=(int(color[0]), int(color[1]), int(color[2])),
                )
                layout_ctx = LayoutContext(
                    bbox=BoundingBox(
                        float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    ),
                    font=font_info,
                    ocr_confidence=(float(blk["confidence"]) if "confidence" in blk else None),
                )
                page_blocks.append(TextBlock(text=text, layout=layout_ctx))
            pages_out.append(page_blocks)
        return pages_out


def _default_output_path(input_path: str) -> str:
    p = Path(input_path)
    if not p.suffix:
        raise ValueError(f"Input path must be a file with an extension: {input_path}")
    return str(p.with_name(f"{p.stem}.translated.pdf"))


