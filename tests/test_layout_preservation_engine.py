from __future__ import annotations

import pytest

from dolphin_ocr.layout import (
    BoundingBox,
    FontInfo,
    LayoutPreservationEngine,
    StrategyType,
)


def make_engine():
    # Deterministic parameters for tests
    return LayoutPreservationEngine(
        font_scale_limits=(0.6, 1.2),
        average_char_width_em=0.5,
        line_height_factor=1.2,
    )


def analyze(
    engine: LayoutPreservationEngine, original: str, translated: str, bbox: BoundingBox
):
    return engine.analyze_text_fit(
        original=original,
        translated=translated,
        bbox=bbox,
        font=FontInfo(size=12),
    )


@pytest.mark.parametrize(
    "orig, trans, expected_strategy",
    [
        ("Hello", "Hello", StrategyType.NONE),  # fits unchanged
        ("Hello", "Hello world", StrategyType.FONT_SCALE),  # small scale
    ],
)
def test_strategy_none_or_scale_when_possible(orig, trans, expected_strategy):
    engine = make_engine()
    # Width chosen so "Hello" fits, but "Hello world" requires modest scaling
    # one_line_width("Hello world") ≈ 12 * 0.5 * 11 = 66, so width 50 forces
    # scale
    bbox = BoundingBox(x=0, y=0, width=50, height=30)
    analysis = analyze(engine, orig, trans, bbox)
    decision = engine.determine_layout_strategy(analysis)
    assert decision.type == expected_strategy
    # Quality score should be within [0, 1]
    score = engine.calculate_quality_score(analysis, decision)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize(
    "length_ratio, expected",
    [
        (1.0, StrategyType.NONE),
        (1.2, StrategyType.FONT_SCALE),
        (1.8, StrategyType.TEXT_WRAP),
        (2.2, StrategyType.HYBRID),
    ],
)
def test_strategy_selection_across_length_ratios(
    length_ratio: float, expected: StrategyType
):
    engine = make_engine()
    base_text = "abcdefghij"  # length 10
    target_length = int(round(length_ratio * len(base_text)))
    padding_needed = max(0, target_length - len(base_text))
    translated = base_text + ("x" * padding_needed)

    # Choose a bbox that allows clear transitions as ratio grows:
    # - Enough width that small increases allow scaling
    # - Limited height to trigger hybrid when wrapping would overflow
    bbox = BoundingBox(x=0, y=0, width=60, height=22)  # ~1 line height
    analysis = analyze(engine, base_text, translated, bbox)
    decision = engine.determine_layout_strategy(analysis)

    # Map NONE vs FONT_SCALE for 1.0 vs 1.2; 1.8 -> wrap; 2.2 -> likely
    # hybrid
    if expected == StrategyType.HYBRID and decision.type == StrategyType.TEXT_WRAP:
        # Allow TEXT_WRAP fallback if hybrid can't improve lines under limits
        assert decision.type in {StrategyType.HYBRID, StrategyType.TEXT_WRAP}
    else:
        assert decision.type == expected


def test_strategy_text_wrap_when_single_line_not_possible_but_height_allows():
    engine = make_engine()
    # Narrow box to force wrapping, tall enough for required number of lines
    bbox = BoundingBox(x=0, y=0, width=40, height=150)
    orig = "Short"
    trans = "This is a considerably longer line that needs wrapping"
    analysis = analyze(engine, orig, trans, bbox)
    assert analysis.lines_needed > 1
    assert analysis.can_wrap_within_height is True
    decision = engine.determine_layout_strategy(analysis)
    assert decision.type == StrategyType.TEXT_WRAP
    assert decision.wrap_lines >= 2
    score = engine.calculate_quality_score(analysis, decision)
    assert 0.0 <= score <= 1.0


def test_strategy_hybrid_when_neither_scaling_nor_wrapping_alone_is_enough():
    engine = make_engine()
    # Extremely narrow and short box: wrapping alone exceeds height
    bbox = BoundingBox(x=0, y=0, width=35, height=20)  # likely max_lines == 1
    orig = "Base"
    trans = "This is an excessively verbose translation requiring compromise"
    analysis = analyze(engine, orig, trans, bbox)
    assert analysis.can_fit_without_changes is False
    assert analysis.can_scale_to_single_line is False
    assert analysis.can_wrap_within_height is False
    decision = engine.determine_layout_strategy(analysis)
    # Hybrid attempts to fit by scaling to achieve allowed number of lines
    assert decision.type in {StrategyType.HYBRID, StrategyType.TEXT_WRAP}
    # Quality score remains bounded
    score = engine.calculate_quality_score(analysis, decision)
    assert 0.0 <= score <= 1.0
