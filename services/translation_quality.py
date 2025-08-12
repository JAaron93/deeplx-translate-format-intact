from __future__ import annotations

import os
from dataclasses import dataclass

from dolphin_ocr.layout import FitAnalysis, LayoutStrategy, StrategyType


@dataclass(frozen=True)
class QualityReport:
    overall_score: float
    text_length_ratio: float
    layout_quality: float
    translation_confidence: float | None
    ocr_confidence: float | None
    strategy: StrategyType
    warnings: list[str]


class TranslationQualityValidator:
    """Validate translation quality with layout impact assessment.

    Heuristic scoring:
    - Start from translation confidence (if provided), default 0.8
    - Multiply by layout quality derived from FitAnalysis and LayoutStrategy
    - Clamp to [0, 1]
    """

    # Layout quality penalty/bonus weights (defaults)
    FONT_SCALE_PENALTY_WEIGHT = 0.35
    WRAP_LINES_PENALTY_WEIGHT = 0.25
    NO_ADJUSTMENT_BONUS = 0.05

    def __init__(
        self,
        base_confidence: float = 0.8,
        *,
        large_expansion_threshold: float | None = None,
        warn_on_wrap_overflow: bool | None = None,
        font_scale_penalty_weight: float | None = None,
        wrap_lines_penalty_weight: float | None = None,
        no_adjustment_bonus: float | None = None,
    ) -> None:
        """Create a validator with configurable warning thresholds.

        Parameters
        - base_confidence: default confidence if translation confidence is None
        - large_expansion_threshold: threshold on length_ratio to trigger a
          warning when no layout adjustment is applied. Defaults to env var
          `QUALITY_LARGE_EXPANSION_THRESHOLD` or 1.6.
        - warn_on_wrap_overflow: enable warning when wrapped lines exceed
          capacity. Defaults to env var `QUALITY_WARN_WRAP_OVERFLOW` (truthy)
          or True.
        """
        self._base_conf = max(0.0, min(1.0, base_confidence))

        # Load from environment if not explicitly provided
        if large_expansion_threshold is None:
            env_val = os.getenv("QUALITY_LARGE_EXPANSION_THRESHOLD")
            try:
                large_expansion_threshold = float(env_val) if env_val else 1.6
            except ValueError:
                large_expansion_threshold = 1.6
        # Validate reasonable bounds (must be >= 1.0)
        self._large_expansion_threshold = max(1.0, float(large_expansion_threshold))

        if warn_on_wrap_overflow is None:
            env_flag = os.getenv("QUALITY_WARN_WRAP_OVERFLOW", "true").lower()
            warn_on_wrap_overflow = env_flag in {"1", "true", "yes", "on"}
        self._warn_on_wrap_overflow = bool(warn_on_wrap_overflow)

        # Weights for layout quality scoring (allow overrides)
        def _clamp_01(value: float) -> float:
            return max(0.0, min(1.0, float(value)))

        self._font_scale_penalty_weight = _clamp_01(
            font_scale_penalty_weight
            if font_scale_penalty_weight is not None
            else self.FONT_SCALE_PENALTY_WEIGHT
        )
        self._wrap_lines_penalty_weight = _clamp_01(
            wrap_lines_penalty_weight
            if wrap_lines_penalty_weight is not None
            else self.WRAP_LINES_PENALTY_WEIGHT
        )
        self._no_adjustment_bonus = _clamp_01(
            no_adjustment_bonus
            if no_adjustment_bonus is not None
            else self.NO_ADJUSTMENT_BONUS
        )

    def assess(
        self,
        *,
        analysis: FitAnalysis,
        strategy: LayoutStrategy,
        translation_confidence: float | None,
        ocr_confidence: float | None,
    ) -> QualityReport:
        """Assess quality and produce a `QualityReport`.

        Computes a layout-aware overall score, populates warnings based on
        configured thresholds, and returns a structured report.
        """
        warnings: list[str] = []

        # Layout quality: penalize scaling and wrapping similar to engine logic
        layout_quality = 1.0
        if strategy.type in {StrategyType.FONT_SCALE, StrategyType.HYBRID}:
            delta = abs(1.0 - strategy.font_scale)
            layout_quality -= self._font_scale_penalty_weight * delta
        if strategy.wrap_lines > 1:
            max_lines = max(1, analysis.max_lines)
            normalized = (strategy.wrap_lines - 1) / max(1, max_lines - 1)
            layout_quality -= self._wrap_lines_penalty_weight * normalized
        if strategy.type == StrategyType.NONE:
            layout_quality += self._no_adjustment_bonus
        layout_quality = max(0.0, min(1.0, layout_quality))

        conf = translation_confidence
        if conf is None:
            conf = self._base_conf

        overall = max(0.0, min(1.0, conf * layout_quality))

        # Add gentle warnings
        if analysis.length_ratio > self._large_expansion_threshold and (
            strategy.type == StrategyType.NONE
        ):
            warnings.append("Large expansion without adjustments")
        if self._warn_on_wrap_overflow and (
            strategy.type == StrategyType.TEXT_WRAP
            and analysis.lines_needed > analysis.max_lines
        ):
            warnings.append("Wrapping exceeds capacity; potential truncation")
        return QualityReport(
            overall_score=overall,
            text_length_ratio=analysis.length_ratio,
            layout_quality=layout_quality,
            translation_confidence=translation_confidence,
            ocr_confidence=ocr_confidence,
            strategy=strategy.type,
            warnings=warnings,
        )
