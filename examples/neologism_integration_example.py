"""
Example demonstrating the integration of the Neologism Detection Engine
with the existing translation system for philosophy-focused translation.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add the project root to the path (at end to avoid overriding system packages)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from models.neologism_models import (
    DetectedNeologism,
    NeologismAnalysis,
)
from services.neologism_detector import NeologismDetector
from services.translation_service import TranslationService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PhilosophyEnhancedTranslator:
    """
    Enhanced translator that integrates neologism detection with translation services.
    """

    def __init__(
        self,
        terminology_path: str = None,
        min_confidence: float = 0.6,
        preserve_neologisms: bool = True,
    ):
        """
        Initialize the enhanced translator.

        Args:
            terminology_path: Path to terminology JSON file
            min_confidence: Minimum confidence threshold for neologism detection
            preserve_neologisms: Whether to preserve detected neologisms in translation
        """
        self.min_confidence = min_confidence
        self.preserve_neologisms = preserve_neologisms

        # Initialize core components
        self.neologism_detector = NeologismDetector(
            terminology_path=terminology_path, philosophical_threshold=min_confidence
        )

        self.translation_service = TranslationService(
            terminology_map=self.neologism_detector.terminology_map
        )

        logger.info("PhilosophyEnhancedTranslator initialized successfully")

    def translate_with_neologism_detection(
        self,
        text: str,
        source_lang: str = "de",
        target_lang: str = "en",
        provider: str = "auto",
    ) -> Dict[str, Any]:
        """
        Translate text with integrated neologism detection and preservation.

        Args:
            text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            provider: Translation provider to use

        Returns:
            Dictionary containing translation results and neologism analysis
        """
        logger.info(
            f"Starting translation with neologism detection for {len(text)} characters"
        )

        # Step 1: Detect neologisms in the source text
        neologism_analysis = self.neologism_detector.analyze_text(text, "source_text")

        # Step 2: Prepare text for translation
        prepared_text, neologism_map = self._prepare_text_for_translation(
            text, neologism_analysis
        )

        # Step 3: Translate the prepared text
        translated_text = self.translation_service.translate_text(
            prepared_text, source_lang, target_lang, provider
        )

        # Step 4: Post-process translation with neologism handling
        final_translation = self._post_process_translation(
            translated_text, neologism_map
        )

        # Step 5: Generate translation suggestions for neologisms
        neologism_suggestions = self._generate_neologism_suggestions(
            neologism_analysis, target_lang
        )

        return {
            "original_text": text,
            "translated_text": final_translation,
            "neologism_analysis": neologism_analysis.to_dict(),
            "neologism_suggestions": neologism_suggestions,
            "high_confidence_neologisms": [
                n.to_dict() for n in neologism_analysis.get_high_confidence_neologisms()
            ],
            "translation_provider": provider,
            "processing_summary": {
                "total_neologisms": neologism_analysis.total_detections,
                "high_confidence_count": len(
                    neologism_analysis.get_high_confidence_neologisms()
                ),
                "philosophical_density": neologism_analysis.philosophical_density_avg,
                "semantic_fields": neologism_analysis.semantic_fields,
            },
        }

    def _prepare_text_for_translation(
        self, text: str, analysis: NeologismAnalysis
    ) -> tuple[str, Dict[str, DetectedNeologism]]:
        """
        Prepare text for translation by marking neologisms for preservation.

        Args:
            text: Original text
            analysis: Neologism analysis results

        Returns:
            Tuple of (prepared_text, neologism_map)
        """
        prepared_text = text
        neologism_map = {}

        if not self.preserve_neologisms:
            return prepared_text, neologism_map

        # Sort neologisms by position (reverse order to avoid position shifts)
        high_confidence_neologisms = sorted(
            analysis.get_high_confidence_neologisms(),
            key=lambda n: n.start_pos,
            reverse=True,
        )

        # Replace high-confidence neologisms with preservation markers
        for i, neologism in enumerate(high_confidence_neologisms):
            if neologism.confidence >= self.min_confidence:
                marker = f"__NEOLOGISM_{i}__"

                # Replace the neologism with a preservation marker
                prepared_text = (
                    prepared_text[: neologism.start_pos]
                    + marker
                    + prepared_text[neologism.end_pos :]
                )

                neologism_map[marker] = neologism

        return prepared_text, neologism_map

    def _post_process_translation(
        self, translated_text: str, neologism_map: Dict[str, DetectedNeologism]
    ) -> str:
        """
        Post-process translation to handle preserved neologisms.

        Args:
            translated_text: Translated text with markers
            neologism_map: Map of markers to neologisms

        Returns:
            Final processed translation
        """
        final_text = translated_text

        # Replace preservation markers with appropriate handling
        for marker, neologism in neologism_map.items():
            if marker in final_text:
                # For high-confidence neologisms, preserve original term
                # with optional annotation
                replacement = f"{neologism.term}"

                # Add annotation for very high confidence neologisms
                if neologism.confidence >= 0.8:
                    replacement = f"{neologism.term} [philosophical neologism]"

                final_text = final_text.replace(marker, replacement)

        return final_text

    def _generate_neologism_suggestions(
        self, analysis: NeologismAnalysis, target_lang: str
    ) -> List[Dict[str, Any]]:
        """
        Generate translation suggestions for detected neologisms.

        Args:
            analysis: Neologism analysis results
            target_lang: Target language for suggestions

        Returns:
            List of suggestion dictionaries
        """
        suggestions = []

        for neologism in analysis.get_high_confidence_neologisms():
            # Generate morphological breakdown
            morphological_breakdown = self._generate_morphological_breakdown(neologism)

            # Generate contextual suggestions
            contextual_suggestions = self._generate_contextual_suggestions(
                neologism, target_lang
            )

            suggestion = {
                "term": neologism.term,
                "confidence": neologism.confidence,
                "type": neologism.neologism_type.value,
                "morphological_breakdown": morphological_breakdown,
                "contextual_suggestions": contextual_suggestions,
                "philosophical_context": {
                    "semantic_field": neologism.philosophical_context.semantic_field,
                    "related_concepts": neologism.philosophical_context.related_concepts,
                    "philosophical_density": neologism.philosophical_context.philosophical_density,
                },
            }

            suggestions.append(suggestion)

        return suggestions

    def _generate_morphological_breakdown(
        self, neologism: DetectedNeologism
    ) -> Dict[str, Any]:
        """Generate morphological breakdown for a neologism."""
        morphological = neologism.morphological_analysis

        breakdown = {
            "is_compound": morphological.is_compound,
            "compound_parts": morphological.compound_parts,
            "root_words": morphological.root_words,
            "prefixes": morphological.prefixes,
            "suffixes": morphological.suffixes,
            "structural_complexity": morphological.structural_complexity,
        }

        # Add interpretation suggestions
        if morphological.is_compound and morphological.compound_parts:
            breakdown["compound_interpretation"] = " + ".join(
                morphological.compound_parts
            )

        return breakdown

    def _generate_contextual_suggestions(
        self, neologism: DetectedNeologism, target_lang: str
    ) -> List[str]:
        """Generate contextual translation suggestions."""
        suggestions = []

        # Basic morphological suggestions
        if neologism.morphological_analysis.is_compound:
            parts = neologism.morphological_analysis.compound_parts
            if len(parts) >= 2:
                suggestions.append(f"compound of: {' + '.join(parts)}")

        # Semantic field suggestions
        semantic_field = neologism.philosophical_context.semantic_field
        if semantic_field and semantic_field != "general":
            suggestions.append(f"philosophical term in {semantic_field}")

        # Context-based suggestions
        philosophical_keywords = neologism.philosophical_context.philosophical_keywords
        if philosophical_keywords:
            suggestions.append(f"related to: {', '.join(philosophical_keywords[:3])}")

        return suggestions[:5]  # Limit to top 5 suggestions

    def get_detector_statistics(self) -> Dict[str, Any]:
        """Get statistics from the neologism detector."""
        return self.neologism_detector.get_statistics()


def main():
    """Main example function demonstrating the integrated system."""

    # Sample philosophical text in German
    sample_text = """
    Das Wirklichkeitsbewusstsein ist ein zentraler Begriff in der Bewusstseinsphilosophie
    Ludwig Klages'. Die Lebensweltthematik zeigt sich als fundamentale Struktur des
    menschlichen Daseins, die durch die Spontaneität der Erfahrung konstituiert wird.

    In der Zeitlichkeitsanalyse wird deutlich, dass das Bewusstsein nicht nur als
    statische Entität zu verstehen ist, sondern als dynamischer Prozess der
    Wirklichkeitserschließung. Die Bewusstseinsphänomenologie untersucht diese
    Strukturen der Zeitlichkeit und ihre Bedeutung für die Lebenswirklichkeit.

    Die Intentionalitätsstruktur des Bewusstseins zeigt sich in der
    Gegenstandskonstitution, die als Grundlage für jede Erkenntnis fungiert.
    Diese Erkenntnisphilosophie ist von zentraler Bedeutung für das Verständnis
    der menschlichen Existenz.
    """

    print("=== Philosophy-Enhanced Translation System Demo ===\n")

    # Initialize the enhanced translator
    print("1. Initializing Philosophy-Enhanced Translator...")
    translator = PhilosophyEnhancedTranslator(
        terminology_path=str(project_root / "config" / "klages_terminology.json"),
        min_confidence=0.6,
        preserve_neologisms=True,
    )

    # Perform translation with neologism detection
    print("2. Performing translation with neologism detection...")
    result = translator.translate_with_neologism_detection(
        text=sample_text, source_lang="de", target_lang="en", provider="auto"
    )

    # Display results
    print("\n3. Translation Results:")
    print("=" * 50)

    print(f"Original text length: {len(result['original_text'])} characters")
    print(f"Translated text length: {len(result['translated_text'])} characters")

    print(f"\nDetected neologisms: {result['processing_summary']['total_neologisms']}")
    print(
        f"High confidence neologisms: {result['processing_summary']['high_confidence_count']}"
    )
    print(
        f"Philosophical density: {result['processing_summary']['philosophical_density']:.3f}"
    )
    print(
        f"Semantic fields: {', '.join(result['processing_summary']['semantic_fields'])}"
    )

    print("\n4. High-Confidence Neologisms:")
    print("=" * 50)

    for i, neologism in enumerate(result["high_confidence_neologisms"], 1):
        print(f"\n{i}. {neologism['term']}")
        print(f"   Confidence: {neologism['confidence']:.3f}")
        print(f"   Type: {neologism['neologism_type']}")
        print(f"   Context: {neologism['sentence_context'][:100]}...")

    print("\n5. Translation Suggestions:")
    print("=" * 50)

    for i, suggestion in enumerate(result["neologism_suggestions"], 1):
        print(f"\n{i}. {suggestion['term']}")
        print(f"   Confidence: {suggestion['confidence']:.3f}")
        print(f"   Type: {suggestion['type']}")

        if suggestion["morphological_breakdown"]["is_compound"]:
            print(
                f"   Compound parts: {', '.join(suggestion['morphological_breakdown']['compound_parts'])}"
            )

        if suggestion["contextual_suggestions"]:
            print(f"   Suggestions: {'; '.join(suggestion['contextual_suggestions'])}")

    print("\n6. Final Translation:")
    print("=" * 50)
    print(result["translated_text"])

    print("\n7. System Statistics:")
    print("=" * 50)
    stats = translator.get_detector_statistics()
    print(f"Total analyses performed: {stats['total_analyses']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.3f}")
    print(f"Terminology entries: {stats['terminology_entries']}")
    print(f"Philosophical indicators: {stats['philosophical_indicators']}")
    print(f"spaCy available: {stats['spacy_available']}")

    print("\n8. Exporting Results:")
    print("=" * 50)

    # Export results to JSON
    output_file = project_root / "examples" / "translation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Results exported to: {output_file}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
