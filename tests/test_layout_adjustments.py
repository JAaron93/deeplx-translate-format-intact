from __future__ import annotations

from dolphin_ocr.layout import (
    BoundingBox,
    FontInfo,
    LayoutPreservationEngine,
    LayoutStrategy,
    StrategyType,
)


def make_engine() -> LayoutPreservationEngine:
    return LayoutPreservationEngine(font_scale_limits=(0.6, 1.2))


def test_apply_font_scale_only():
    engine = make_engine()
    text = "Hello world"
    bbox = BoundingBox(0, 0, 100, 40)
    font = FontInfo(size=12)
    strat = LayoutStrategy(
        type=StrategyType.FONT_SCALE,
        font_scale=0.8,
        wrap_lines=1,
    )

    adjusted_text, adjusted_font, adjusted_bbox = (
        engine.apply_layout_adjustments(
            text=text,
            bbox=bbox,
            font=font,
            strategy=strat,
        )
    )

    assert adjusted_text == text
    assert 9.0 <= adjusted_font.size <= 10.0
    assert adjusted_bbox == bbox


def test_apply_text_wrap_with_word_boundaries():
    engine = make_engine()
    # Width intentionally small to force wrapping
    bbox = BoundingBox(0, 0, 50, 60)
    font = FontInfo(size=12)
    text = "one two three four five six seven"
    strat = LayoutStrategy(
        type=StrategyType.TEXT_WRAP,
        font_scale=1.0,
        wrap_lines=3,
    )

    adjusted_text, adjusted_font, adjusted_bbox = (
        engine.apply_layout_adjustments(
            text=text,
            bbox=bbox,
            font=font,
            strategy=strat,
        )
    )

    lines = adjusted_text.split("\n")
    assert 1 < len(lines) <= 3
    assert all(line for line in lines)
    assert adjusted_font.size == font.size
    assert adjusted_bbox.height >= bbox.height  # may expand


def test_hybrid_wrap_and_scale_with_bbox_expansion():
    engine = LayoutPreservationEngine(
        font_scale_limits=(0.6, 1.0),
        max_bbox_expansion=0.3,
    )

    text = "This is a long sentence that likely needs wrapping to fit nicely"
    # Short height forces expansion or trimming
    bbox = BoundingBox(0, 0, 60, 20)
    font = FontInfo(size=12)
    strat = LayoutStrategy(
        type=StrategyType.HYBRID,
        font_scale=0.9,
        wrap_lines=2,
    )

    adjusted_text, adjusted_font, adjusted_bbox = (
        engine.apply_layout_adjustments(
            text=text,
            bbox=bbox,
            font=font,
            strategy=strat,
        )
    )

    # Font scaled down a bit
    assert adjusted_font.size < font.size
    # Wrapped to at most requested lines or capacity
    assert 1 <= len(adjusted_text.split("\n")) <= strat.wrap_lines
    # BBox expanded but within limit
    assert (
        bbox.height
        <= adjusted_bbox.height
        <= (bbox.height * (1.0 + engine.max_bbox_expansion))
    )
