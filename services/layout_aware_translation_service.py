"""Layout-aware translation service integrating a translation client with
layout preservation decisions.

This service coordinates translation with the `LayoutPreservationEngine` so
that translated text is adjusted (font scale and/or wrapping) to respect the
original document layout. It provides both single-item and batch translation
APIs and preserves layout context per element.

The implementation is synchronous for simplicity and deterministic testing,
but the design keeps the API surface minimal so it can be adapted to async in
the future without breaking callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dolphin_ocr.layout import (
    BoundingBox,
    FitAnalysis,
    FontInfo,
    LayoutPreservationEngine,
    LayoutStrategy,
)


class McpLingoClient(Protocol):
    """Protocol for the Lingo.dev translation client used by the service.

    Tests provide a fake implementation. A real implementation should map to
    the external API.
    """

    def translate(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:  # pragma: no cover
        ...

    def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:  # pragma: no cover
        ...

    # Optional confidence-enabled interfaces
    def translate_with_confidence(
        self, text: str, source_lang: str, target_lang: str
    ) -> tuple[str, float]:  # pragma: no cover
        """Return (translation, confidence in [0,1])."""

    def translate_batch_with_confidence(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[tuple[str, float]]:  # pragma: no cover
        """Return a list of (translation, confidence)."""


@dataclass(frozen=True)
class LayoutContext:
    """Layout context for a single text element.

    Consists of a bounding box and font information. Additional fields can be
    added in the future without breaking the public API.
    """

    bbox: BoundingBox
    font: FontInfo
    ocr_confidence: float | None = None


@dataclass(frozen=True)
class TextBlock:
    """Minimal text block model for batch translation.

    This mirrors the design document's concept and includes only the fields
    required by the layout-aware service.
    """

    text: str
    layout: LayoutContext


@dataclass(frozen=True)
class TranslationResult:
    """Result of a layout-aware translation operation for a single block."""

    source_text: str
    raw_translation: str
    adjusted_text: str
    strategy: LayoutStrategy
    analysis: FitAnalysis
    adjusted_font: FontInfo
    adjusted_bbox: BoundingBox
    quality_score: float
    ocr_confidence: float | None = None
    translation_confidence: float | None = None


class LayoutAwareTranslationService:
    """Translate text while preserving layout constraints.

    - Integrates a translation client (Lingo.dev) via `McpLingoClient`.
    - Uses `LayoutPreservationEngine` to analyze fit and apply adjustments.
    - Provides batch translation that preserves per-element layout context.
    """

    def __init__(
        self,
        lingo_client: McpLingoClient,
        layout_engine: LayoutPreservationEngine,
    ) -> None:
        self._lingo = lingo_client
        self._engine = layout_engine

    # ----------------------------- Public API -----------------------------
    def translate_with_layout_constraints(
        self,
        *,
        text: str,
        source_lang: str,
        target_lang: str,
        layout_context: LayoutContext,
    ) -> TranslationResult:
        """Translate a single text with layout-aware adjustments.

        Steps:
        1) Translate the text via `McpLingoClient`.
        2) Perform length-aware optimization (whitespace compaction) to reduce
           unnecessary growth from translation.
        3) Analyze layout fit and determine a preservation strategy.
        4) Apply adjustments (font scaling and/or wrapping) to produce the
           final adjusted text and layout.
        """
        # Prefer confidence-aware method if available
        translation_conf: float | None = None
        translate_with_conf = getattr(self._lingo, "translate_with_confidence", None)
        if callable(translate_with_conf):
            raw_translation, translation_conf = translate_with_conf(
                text,
                source_lang,
                target_lang,
            )
        else:
            raw_translation = self._lingo.translate(text, source_lang, target_lang)
        optimized_translation = self._optimize_for_length(raw_translation)

        analysis = self._engine.analyze_text_fit(
            original=text,
            translated=optimized_translation,
            bbox=layout_context.bbox,
            font=layout_context.font,
        )
        strategy = self._engine.determine_layout_strategy(analysis)

        (
            adjusted_text,
            adjusted_font,
            adjusted_bbox,
        ) = self._engine.apply_layout_adjustments(
            text=optimized_translation,
            bbox=layout_context.bbox,
            font=layout_context.font,
            strategy=strategy,
        )

        quality_score = self._engine.calculate_quality_score(analysis, strategy)

        return TranslationResult(
            source_text=text,
            raw_translation=raw_translation,
            adjusted_text=adjusted_text,
            strategy=strategy,
            analysis=analysis,
            adjusted_font=adjusted_font,
            adjusted_bbox=adjusted_bbox,
            quality_score=quality_score,
            ocr_confidence=layout_context.ocr_confidence,
            translation_confidence=translation_conf,
        )

    def translate_document_batch(
        self,
        *,
        text_blocks: list[TextBlock],
        source_lang: str,
        target_lang: str,
    ) -> list[TranslationResult]:
        """Batch translate with layout context preservation.

        Attempts a single batched call when the client provides
        `translate_batch`. Falls back to per-item calls otherwise. Each result
        uses the originating block's layout context for analysis and
        adjustments.
        """
        texts = [b.text for b in text_blocks]

        translations, confidences = self._translate_batch_collect_conf(
            texts=texts,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # Ensure size parity to avoid index errors; truncate extras defensively
        translations = translations[: len(text_blocks)]
        if confidences is not None:
            confidences = confidences[: len(text_blocks)]

        results: list[TranslationResult] = []
        for index, (block, raw) in enumerate(
            zip(text_blocks, translations, strict=False)
        ):
            optimized = self._optimize_for_length(raw)
            analysis = self._engine.analyze_text_fit(
                original=block.text,
                translated=optimized,
                bbox=block.layout.bbox,
                font=block.layout.font,
            )
            strategy = self._engine.determine_layout_strategy(analysis)
            (
                adjusted_text,
                adjusted_font,
                adjusted_bbox,
            ) = self._engine.apply_layout_adjustments(
                text=optimized,
                bbox=block.layout.bbox,
                font=block.layout.font,
                strategy=strategy,
            )
            quality = self._engine.calculate_quality_score(analysis, strategy)
            translation_conf: float | None = (
                confidences[index]
                if confidences is not None and index < len(confidences)
                else None
            )
            results.append(
                TranslationResult(
                    source_text=block.text,
                    raw_translation=raw,
                    adjusted_text=adjusted_text,
                    strategy=strategy,
                    analysis=analysis,
                    adjusted_font=adjusted_font,
                    adjusted_bbox=adjusted_bbox,
                    quality_score=quality,
                    ocr_confidence=block.layout.ocr_confidence,
                    translation_confidence=translation_conf,
                )
            )

        return results

    # ---------------------------- Internal utils ---------------------------
    def _translate_batch_collect_conf(
        self,
        *,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> tuple[list[str], list[float] | None]:
        """Translate a batch and collect confidences when available.

        Returns a pair (translations, confidences) where confidences may be
        None if the underlying client does not support confidence outputs.
        """
        translate_batch_with_conf = getattr(
            self._lingo, "translate_batch_with_confidence", None
        )
        if callable(translate_batch_with_conf):
            results_with_conf = translate_batch_with_conf(
                texts, source_lang, target_lang
            )
            translations = [t for (t, _c) in results_with_conf]
            confidences = [c for (_t, c) in results_with_conf]
            return translations, confidences

        translate_batch = getattr(self._lingo, "translate_batch", None)
        if callable(translate_batch):
            return translate_batch(texts, source_lang, target_lang), None

        # Fallback to per-item calls (kept for completeness)
        translations = [
            self._lingo.translate(t, source_lang, target_lang) for t in texts
        ]
        return translations, None

    def _optimize_for_length(self, text: str) -> str:
        """Apply simple, deterministic length optimizations.

        - Collapse repeated whitespace to a single space
        - Trim leading/trailing whitespace

        This is intentionally conservative and language-agnostic. More
        sophisticated strategies (synonym selection, soft constraints to the
        LLM/MT system) can be layered later without changing the external API.
        """
        # Collapse sequences of whitespace to single spaces
        compact = " ".join(text.split())
        return compact.strip()
