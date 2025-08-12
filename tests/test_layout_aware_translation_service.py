from __future__ import annotations

from dataclasses import dataclass
from typing import List

from dolphin_ocr.layout import (
    BoundingBox,
    FontInfo,
    LayoutPreservationEngine,
    StrategyType,
)
from services.layout_aware_translation_service import (
    LayoutAwareTranslationService,
    LayoutContext,
    TextBlock,
)


@dataclass
class _FakeBatchCall:
    texts: List[str]
    source_lang: str
    target_lang: str


class _FakeLingo:
    """Fake translation client with batch support for tests."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping
        self.batch_calls: list[_FakeBatchCall] = []
        self.single_calls: list[tuple[str, str, str]] = []

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        self.single_calls.append((text, source_lang, target_lang))
        return self._mapping.get(text, text)

    def translate_batch(
        self, texts: List[str], source_lang: str, target_lang: str
    ) -> List[str]:
        self.batch_calls.append(
            _FakeBatchCall(
                texts=list(texts),
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )
        return [self._mapping.get(t, t) for t in texts]


def _make_engine() -> LayoutPreservationEngine:
    return LayoutPreservationEngine(
        font_scale_limits=(0.6, 1.2),
        average_char_width_em=0.5,
        line_height_factor=1.2,
    )


def test_single_translation_with_font_scale_and_length_optimization():
    # Arrange: "Hello" -> "Hello   world" (extra spaces to be compacted)
    mapping = {"Hello": "Hello   world"}
    lingo = _FakeLingo(mapping)
    engine = _make_engine()
    service = LayoutAwareTranslationService(
        lingo_client=lingo, layout_engine=engine
    )

    ctx = LayoutContext(bbox=BoundingBox(0, 0, 50, 30), font=FontInfo(size=12))

    # Act
    result = service.translate_with_layout_constraints(
        text="Hello", source_lang="en", target_lang="de", layout_context=ctx
    )

    # Assert: scaling chosen, whitespace compacted
    assert result.strategy.type == StrategyType.FONT_SCALE
    assert result.adjusted_font.size < 12
    assert "  " not in result.adjusted_text
    # Sanity: raw translation contained the extra spaces
    assert result.raw_translation == "Hello   world"
    assert result.quality_score >= 0.0 and result.quality_score <= 1.0


def test_batch_translation_preserves_layout_context_and_uses_batch():
    # Arrange two blocks with different layouts
    # Block A will yield NONE (same text)
    # Block B will induce HYBRID or TEXT_WRAP due to tight box and long
    # translation
    long_translation = (
        "This is an excessively verbose translation requiring compromise"
    )
    mapping = {
        "Short": "Short",
        "Base": long_translation,
    }
    lingo = _FakeLingo(mapping)
    engine = _make_engine()
    service = LayoutAwareTranslationService(
        lingo_client=lingo, layout_engine=engine
    )

    blocks = [
        TextBlock(
            text="Short",
            layout=LayoutContext(
                bbox=BoundingBox(0, 0, 200, 100),
                font=FontInfo(size=12),
            ),
        ),
        TextBlock(
            text="Base",
            layout=LayoutContext(
                bbox=BoundingBox(0, 0, 35, 20),
                font=FontInfo(size=12),
            ),
        ),
    ]

    # Act
    results = service.translate_document_batch(
        text_blocks=blocks, source_lang="en", target_lang="de"
    )

    # Assert: batch was used
    assert len(lingo.batch_calls) == 1
    assert len(lingo.single_calls) == 0
    assert [c.texts for c in lingo.batch_calls][-1] == [
        "Short",
        "Base",
    ]

    # Per-block assertions
    assert len(results) == 2
    r1 = results[0]
    r2 = results[1]
    assert r1.source_text == "Short" and r2.source_text == "Base"

    # r1 should need no changes
    assert r1.strategy.type in {StrategyType.NONE, StrategyType.FONT_SCALE}

    # r2 should be constrained; accept HYBRID or TEXT_WRAP depending on
    # heuristics
    assert r2.strategy.type in {StrategyType.HYBRID, StrategyType.TEXT_WRAP}
    assert r2.adjusted_text  # non-empty
    assert 0.0 <= r2.quality_score <= 1.0
