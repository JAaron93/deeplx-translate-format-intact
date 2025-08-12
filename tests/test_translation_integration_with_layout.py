from __future__ import annotations
from dataclasses import dataclass

from dolphin_ocr.layout import (
    BoundingBox,
    FontInfo,
    LayoutPreservationEngine,
)
from services.layout_aware_translation_service import (
    LayoutAwareTranslationService,
    LayoutContext,
    TextBlock,
)
from services.translation_quality import (
    TranslationQualityValidator as TQValidator,
)


@dataclass
class FakeLingoClient:
    mapping: dict[
        str,
        tuple[str, float],
    ]

    # Minimal methods to satisfy protocol at runtime
    def translate(
        self, text: str, _source_lang: str, _target_lang: str
    ) -> str:
        return self.mapping.get(text, (text, 0.8))[0]

    def translate_batch(
        self, texts: list[str], _source_lang: str, _target_lang: str
    ) -> list[str]:
        return [self.mapping.get(t, (t, 0.8))[0] for t in texts]

    def translate_with_confidence(
        self, text: str, _source_lang: str, _target_lang: str
    ) -> tuple[str, float]:
        # Unused args are part of protocol; intentionally ignored in fake
        return self.mapping.get(text, (text, 0.8))

    def translate_batch_with_confidence(
        self, texts, _source_lang: str, _target_lang: str
    ) -> list[tuple[str, float]]:
        # Unused args are part of protocol; intentionally ignored in fake
        return [self.mapping.get(t, (t, 0.8)) for t in texts]


def test_end_to_end_translation_with_layout_and_quality_assessment():
    # Arrange a short and a long sample
    mapping = {
        "Short": ("Short", 0.95),
        "Base": (
            "This is an excessively verbose translation requiring compromise",
            0.85,
        ),
    }
    lingo = FakeLingoClient(mapping)
    engine = LayoutPreservationEngine(font_scale_limits=(0.6, 1.2))
    service = LayoutAwareTranslationService(
        lingo_client=lingo,
        layout_engine=engine,
    )
    validator = TQValidator(base_confidence=0.8)

    blocks = [
        TextBlock(
            text="Short",
            layout=LayoutContext(
                bbox=BoundingBox(0, 0, 200, 100),
                font=FontInfo(size=12),
                ocr_confidence=0.93,
            ),
        ),
        TextBlock(
            text="Base",
            layout=LayoutContext(
                bbox=BoundingBox(0, 0, 35, 20),
                font=FontInfo(size=12),
                ocr_confidence=0.88,
            ),
        ),
    ]

    # Act: batch translate
    results = service.translate_document_batch(
        text_blocks=blocks,
        source_lang="en",
        target_lang="de",
    )

    # Assert basic shape
    assert len(results) == 2

    # Quality assessment
    q_reports = [
        validator.assess(
            analysis=r.analysis,
            strategy=r.strategy,
            translation_confidence=r.translation_confidence,
            ocr_confidence=r.ocr_confidence,
        )
        for r in results
    ]

    # High-level checks
    for rep in q_reports:
        assert 0.0 <= rep.overall_score <= 1.0
        assert 0.0 <= rep.layout_quality <= 1.0

    # Short sample should likely keep high quality
    assert q_reports[0].overall_score >= 0.7
    # Verify confidence propagation
    assert results[0].translation_confidence == 0.95
    assert results[0].ocr_confidence == 0.93
    # Long sample constrained should still produce bounded score and warnings
    assert q_reports[1].overall_score >= 0.4
    assert isinstance(q_reports[1].warnings, list)
    assert len(q_reports[1].warnings) > 0
    assert results[1].translation_confidence == 0.85
    assert results[1].ocr_confidence == 0.88

    # Custom threshold behavior: set a very high expansion threshold -> no warning
    validator_high_thresh = TQValidator(
        base_confidence=0.8, large_expansion_threshold=10.0
    )
    q_reports_custom = [
        validator_high_thresh.assess(
            analysis=r.analysis,
            strategy=r.strategy,
            translation_confidence=r.translation_confidence,
            ocr_confidence=r.ocr_confidence,
        )
        for r in results
    ]
    # First report should have no expansion warning now
    assert "Large expansion without adjustments" not in q_reports_custom[0].warnings
