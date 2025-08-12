"""Layout preservation engine and supporting data models.

Implements text fit analysis, strategy determination (font scaling, text
wrapping, hybrid), and a simple quality score for layout preservation
assessment. The implementation uses pragmatic heuristics that are
deterministic for unit testing and do not rely on external font metrics.

This module is intentionally lightweight and self-contained so it can be used
by translation and reconstruction components without adding heavy
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import ceil, floor

# Quality score weights (tunable for layout scoring heuristics)
SCALE_PENALTY_WEIGHT: float = 0.35
WRAP_PENALTY_WEIGHT: float = 0.25
NONE_BONUS: float = 0.05


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box represented as (x, y, width, height).

    Coordinates use a PDF-like coordinate space where (x, y) denote the origin
    of the box and width/height are in the same units as font sizes (points).
    """

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class FontInfo:
    """Font metadata required for layout estimation.

    Only ``size`` is used by the engine, but the rest of the fields are kept to
    align with the design document and future-proof usage.
    """

    family: str = "Helvetica"
    size: float = 12.0
    weight: str = "normal"  # "normal" | "bold"
    style: str = "normal"  # "normal" | "italic"
    color: tuple[int, int, int] = (0, 0, 0)  # RGB 0-255


class StrategyType(str, Enum):
    """High-level strategies for preserving layout when text length varies."""

    NONE = "none"  # Original layout is fine without changes
    FONT_SCALE = "font_scale"
    TEXT_WRAP = "text_wrap"
    HYBRID = "hybrid"  # Combine modest scaling with wrapping


@dataclass(frozen=True)
class FitAnalysis:
    """Result of analyzing whether text fits within a bounding box.

    Attributes:
    - length_ratio: translated_length / original_length (>= 0)
    - one_line_width: Estimated width (points) if rendered on one line at the
      current font size.
    - max_lines: Maximum number of lines that fit the bounding box height.
    - lines_needed: Required number of lines at the current font size to fit
      the text when wrapping at box width.
    - can_fit_without_changes: True if single-line width <= box width.
    - required_scale_for_single_line: Minimal scale factor to fit on one line.
    - can_scale_to_single_line: True if required scale is within limits.
    - can_wrap_within_height: True if lines_needed <= max_lines.
    """

    length_ratio: float
    one_line_width: float
    max_lines: int
    lines_needed: int
    can_fit_without_changes: bool
    required_scale_for_single_line: float
    can_scale_to_single_line: bool
    can_wrap_within_height: bool


@dataclass(frozen=True)
class LayoutStrategy:
    """Selected strategy with concrete parameters to apply."""

    type: StrategyType
    font_scale: float = 1.0
    wrap_lines: int = 1


class LayoutPreservationEngine:
    """Engine that decides how to preserve layout when text length changes.

    The engine uses an average character width heuristic to estimate text width
    and line breaking. The heuristic is tunable to keep the implementation
    deterministic for tests, without depending on real font metrics.
    """

    def __init__(
        self,
        *,
        font_scale_limits: tuple[float, float] = (0.6, 1.2),
        max_bbox_expansion: float = 0.3,  # reserved for future use
        average_char_width_em: float = 0.5,
        line_height_factor: float = 1.2,
    ) -> None:
        """Create a new engine.

        Parameters
        - font_scale_limits: Allowed [min, max] multiplicative scale factors.
        - max_bbox_expansion: Reserved for future use; not applied in this
          implementation.
        - average_char_width_em: Average character width as a multiple of font
          size (em). 0.5 is a reasonable approximation for many fonts.
        - line_height_factor: Multiplier for line height relative to font size.
        """
        self.font_scale_min = float(font_scale_limits[0])
        self.font_scale_max = float(font_scale_limits[1])
        if self.font_scale_min <= 0 or self.font_scale_max <= 0:
            raise ValueError("font_scale_limits must be positive")
        if self.font_scale_min > self.font_scale_max:
            raise ValueError("font_scale_limits must be (min <= max)")
        if average_char_width_em <= 0:
            raise ValueError("average_char_width_em must be positive")
        if line_height_factor <= 0:
            raise ValueError("line_height_factor must be positive")

        self.max_bbox_expansion = float(max_bbox_expansion)
        self.average_char_width_em = float(average_char_width_em)
        self.line_height_factor = float(line_height_factor)

    # -------------------------- Public API --------------------------
    def analyze_text_fit(
        self,
        *,
        original: str,
        translated: str,
        bbox: BoundingBox,
        font: FontInfo,
    ) -> FitAnalysis:
        """Analyze how translated text fits in the bounding box.
        The width heuristic estimates single-line width and derives the
        required number of wrapped lines by dividing by box width. Height fit
        is based on a line-height model.
        """
        # Use 1 as minimum to avoid division by zero and provide meaningful
        # ratio for empty strings
        original_len = max(1, len(original))
        translated_len = len(translated)
        length_ratio = translated_len / float(original_len)

        # Estimate width if rendered on a single line at current font size.
        # Use a simple model: width = font_size * avg_char_em * num_chars
        one_line_width = font.size * self.average_char_width_em * translated_len

        # Height constraints: how many lines can we draw in this bbox?
        line_height = font.size * self.line_height_factor
        max_lines = max(1, floor(bbox.height / max(1.0, line_height)))

        # Lines needed if we wrap to the box width at current font size.
        lines_needed = max(1, ceil(one_line_width / max(1.0, bbox.width)))

        can_fit_without_changes = one_line_width <= bbox.width

        # Minimal scale factor required to fit on a single line.
        required_scale_for_single_line = min(1.0, bbox.width / max(1.0, one_line_width))
        can_scale_to_single_line = (
            self.font_scale_min <= required_scale_for_single_line <= self.font_scale_max
        )

        can_wrap_within_height = lines_needed <= max_lines

        return FitAnalysis(
            length_ratio=length_ratio,
            one_line_width=one_line_width,
            max_lines=max_lines,
            lines_needed=lines_needed,
            can_fit_without_changes=can_fit_without_changes,
            required_scale_for_single_line=required_scale_for_single_line,
            can_scale_to_single_line=can_scale_to_single_line,
            can_wrap_within_height=can_wrap_within_height,
        )

    def determine_layout_strategy(self, analysis: FitAnalysis) -> LayoutStrategy:
        """Choose strategy based on fit analysis.
        Preference aims to preserve appearance when possible.
        1) NONE: No change if it already fits
        2) FONT_SCALE: Modest scaling to keep a single line
        3) TEXT_WRAP: Wrap text if it fits within box height
        4) HYBRID: Combine scaling with wrapping if neither alone is enough.
        """
        # 1) No change needed
        if analysis.can_fit_without_changes:
            return LayoutStrategy(
                type=StrategyType.NONE,
                font_scale=1.0,
                wrap_lines=1,
            )

        # 2) Font scaling to a single line if within allowed bounds
        if analysis.can_scale_to_single_line:
            return LayoutStrategy(
                type=StrategyType.FONT_SCALE,
                font_scale=max(
                    self.font_scale_min,
                    analysis.required_scale_for_single_line,
                ),
                wrap_lines=1,
            )

        # 3) Wrapping without scaling if it fits within height
        if analysis.can_wrap_within_height:
            return LayoutStrategy(
                type=StrategyType.TEXT_WRAP,
                font_scale=1.0,
                wrap_lines=analysis.lines_needed,
            )

        # 4) Hybrid: combine modest scaling with wrapping.
        # Improved accuracy: simulate wrapped line count after applying the
        # candidate scale. Because our width heuristic is linear in font size,
        # we model:
        #   lines_after_scale = ceil(lines_needed * scale)
        # where lines_needed = ceil(one_line_width / box_width) at scale=1.
        # Accept HYBRID only when the simulated line count fits within
        # available height (<= max_lines).
        if analysis.lines_needed > analysis.max_lines:
            scale_needed = analysis.max_lines / float(analysis.lines_needed)
            clamped_scale = max(
                self.font_scale_min, min(self.font_scale_max, scale_needed)
            )
            # Simulate wrapping at the clamped scale.
            lines_after_scale = ceil(analysis.lines_needed * clamped_scale)
            if lines_after_scale <= analysis.max_lines:
                return LayoutStrategy(
                    type=StrategyType.HYBRID,
                    font_scale=clamped_scale,
                    wrap_lines=lines_after_scale,
                )

        # Fallback: wrapping with overflow (best-effort). Keep explicit to
        # avoid surprises.
        return LayoutStrategy(
            type=StrategyType.TEXT_WRAP,
            font_scale=1.0,
            wrap_lines=analysis.max_lines,
        )

    def calculate_quality_score(
        self, analysis: FitAnalysis, decision: LayoutStrategy
    ) -> float:
        """Compute a simple layout preservation quality score in [0, 1].
        Heuristics:
        - Start from 1.0 (perfect preservation).
        - Penalize font scaling proportional to |1 - scale| (weight 0.35).
        - Penalize wrapping proportional to (lines - 1) (weight 0.25 per line),
          normalized by the maximum allowed lines from the analysis.
        - Prefer NONE over other strategies when otherwise equivalent.
        """
        score = 1.0

        # Scaling penalty
        if decision.type in (StrategyType.FONT_SCALE, StrategyType.HYBRID):
            score -= SCALE_PENALTY_WEIGHT * abs(1.0 - decision.font_scale)

        # Wrapping penalty relative to available lines
        if decision.wrap_lines > 1:
            max_lines = max(1, analysis.max_lines)
            normalized_lines = (decision.wrap_lines - 1) / max(1, max_lines - 1)
            score -= WRAP_PENALTY_WEIGHT * normalized_lines

        # Minor preference adjustment: NONE slightly boosted
        if decision.type == StrategyType.NONE:
            score += NONE_BONUS

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
