"""
Comprehensive examples demonstrating the philosophy-enhanced translation system.

This script provides practical examples of how to use the PhilosophyEnhancedTranslationService
and PhilosophyEnhancedDocumentProcessor for various translation scenarios involving
philosophical texts with neologisms.
"""

import asyncio
import logging
from pathlib import Path

from models.user_choice_models import ChoiceType
from services.philosophy_enhanced_document_processor import (
    PhilosophyProcessingProgress,
    create_philosophy_enhanced_document_processor,
    process_document_with_philosophy_awareness,
)
from services.philosophy_enhanced_translation_service import (
    PhilosophyEnhancedTranslationService,
    PhilosophyTranslationProgress,
)
from services.user_choice_manager import UserChoiceManager


def _configure_logging():
    """Configure logging for the examples."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


logger = logging.getLogger(__name__)


class PhilosophyTranslationExamples:
    """Examples demonstrating philosophy-enhanced text translation."""

    def __init__(self):
        self.service = PhilosophyEnhancedTranslationService()
        self.user_choice_manager = UserChoiceManager()

    def example_1_basic_text_translation(self):
        """Example 1: Basic text translation with automatic neologism detection."""
        print("\n" + "=" * 80)
        print("EXAMPLE 1: Basic Text Translation with Neologism Detection")
        print("=" * 80)

        # Sample philosophical text with neologisms
        text = """
        Dasein is a fundamental concept in Heideggerian philosophy. The notion of
        Sein-zum-Tode (being-toward-death) reveals the authenticity of human existence.
        Heidegger's concept of Zuhandenheit (readiness-to-hand) describes our
        relationship with tools and equipment in our everyday world.
        """

        print(f"Original text: {text.strip()}")
        print("Translating from English to German...")

        # Translate with neologism handling
        result = self.service.translate_text_with_neologism_handling(
            text=text.strip(),
            source_lang="en",
            target_lang="de",
            provider="auto",
            session_id="example_session_1",
        )

        print(f"Translated text: {result.translated_text}")
        print(f"Neologisms detected: {result.neologism_analysis.total_neologisms}")
        print(f"Neologisms preserved: {len(result.neologisms_preserved)}")

        if result.neologism_analysis.detected_neologisms:
            print("Detected neologisms:")
            for neologism in result.neologism_analysis.detected_neologisms:
                print(f"  - {neologism.term} (confidence: {neologism.confidence:.2f})")

        return result

    def example_2_batch_translation(self):
        """Example 2: Batch translation of multiple philosophical texts."""
        print("\n" + "=" * 80)
        print("EXAMPLE 2: Batch Translation with Neologism Handling")
        print("=" * 80)

        # Multiple philosophical texts
        texts = [
            "Dasein encompasses the whole of human existence in its temporal structure.",
            "The concept of Angst reveals the fundamental groundlessness of human existence.",
            "Zuhandenheit describes our practical engagement with the world of equipment.",
            "Sein-zum-Tode is the possibility that makes all other possibilities possible.",
        ]

        print(f"Translating {len(texts)} philosophical texts...")

        # Batch translate with neologism handling
        results = self.service.translate_batch_with_neologism_handling(
            texts=texts,
            source_lang="en",
            target_lang="de",
            provider="auto",
            session_id="example_session_2",
        )

        print("Results:")
        for i, result in enumerate(results):
            print(f"\n  Text {i + 1}:")
            print(f"    Original: {texts[i]}")
            print(f"    Translated: {result.translated_text}")
            print(f"    Neologisms: {result.neologism_analysis.total_neologisms}")

        return results

    def example_3_user_choice_management(self):
        """Example 3: Managing user choices for neologism handling."""
        print("\n" + "=" * 80)
        print("EXAMPLE 3: User Choice Management for Neologisms")
        print("=" * 80)

        # Create a session for user choices
        session = self.user_choice_manager.create_session(
            session_name="Philosophy Translation Session",
            document_name="Heidegger Analysis",
            user_id="philosopher_user",
            source_language="en",
            target_language="de",
        )

        print(f"Created session: {session.session_id}")

        # Sample text with neologisms
        text = "The concept of Dasein is central to understanding Heidegger's Angst."

        # First, detect neologisms
        analysis = self.service.neologism_detector.analyze_text(text)
        print(f"Detected {analysis.total_neologisms} neologisms")

        # Set user choices for specific neologisms
        for neologism in analysis.detected_neologisms:
            if neologism.term == "Dasein":
                self.user_choice_manager.set_choice(
                    neologism=neologism,
                    choice_type=ChoiceType.PRESERVE,
                    session_id=session.session_id,
                    custom_translation=None,
                )
                print(f"Set choice for '{neologism.term}': PRESERVE")
            elif neologism.term == "Angst":
                self.user_choice_manager.set_choice(
                    neologism=neologism,
                    choice_type=ChoiceType.CUSTOM,
                    session_id=session.session_id,
                    custom_translation="Anxiety",
                )
                print(f"Set choice for '{neologism.term}': CUSTOM (Anxiety)")

        # Now translate with applied choices
        result = self.service.translate_text_with_neologism_handling(
            text=text,
            source_lang="en",
            target_lang="de",
            provider="auto",
            session_id=session.session_id,
        )

        print(f"Translation with user choices: {result.translated_text}")

        return result, session

    def example_4_progress_tracking(self):
        """Example 4: Progress tracking during translation."""
        print("\n" + "=" * 80)
        print("EXAMPLE 4: Progress Tracking During Translation")
        print("=" * 80)

        # Progress callback function
        def progress_callback(progress: PhilosophyTranslationProgress):
            logger.info(
                "Progress: %.1f%% - Stage: %s",
                progress.overall_progress,
                progress.current_stage,
            )
            logger.info(
                "  Text Processing: %s%%",
                progress.text_processing_progress,
            )
            logger.info(
                "  Neologism Detection: %s%%",
                progress.neologism_detection_progress,
            )
            logger.info(
                "  User Choice Application: %s%%",
                progress.user_choice_application_progress,
            )
            logger.info("  Translation: %s%%", progress.translation_progress)

        # Long philosophical text for demonstration
        long_text = """
        Martin Heidegger's concept of Dasein fundamentally challenges traditional ontology.
        The notion of Sein-zum-Tode (being-toward-death) reveals the temporal structure
        of human existence. Zuhandenheit (readiness-to-hand) describes our practical
        engagement with the world. The concept of Angst (anxiety) discloses the
        fundamental groundlessness of existence. Heidegger's analysis of Zeitlichkeit
        (temporality) shows how past, present, and future are unified in human existence.
        """

        print("Translating long philosophical text with progress tracking...")

        # Translate with progress tracking
        result = self.service.translate_text_with_neologism_handling(
            text=long_text.strip(),
            source_lang="en",
            target_lang="de",
            provider="auto",
            session_id="example_session_4",
            progress_callback=progress_callback,
        )

        print(f"\nFinal result: {result.translated_text}")
        return result


class PhilosophyDocumentExamples:
    """Examples demonstrating philosophy-enhanced document processing."""

    def __init__(self):
        self.processor = create_philosophy_enhanced_document_processor()

    async def example_5_document_processing(self):
        """Example 5: Full document processing with philosophy awareness."""
        print("\n" + "=" * 80)
        print("EXAMPLE 5: Philosophy-Enhanced Document Processing")
        print("=" * 80)

        # Create a sample document for demonstration
        sample_doc_content = """
        Heidegger's Fundamental Ontology

        Martin Heidegger's project of fundamental ontology begins with the question of
        Being (Sein). The central concept in this investigation is Dasein, which refers
        to the kind of being that we ourselves are. Dasein is characterized by its
        relationship to Being itself.

        The concept of Sein-zum-Tode (being-toward-death) reveals the finite nature
        of human existence. This finitude is not a limitation but the very condition
        that makes authentic existence possible. Through Angst (anxiety), Dasein
        encounters the groundlessness of its existence.

        Heidegger's analysis of everyday existence reveals the phenomenon of
        Zuhandenheit (readiness-to-hand), which describes our practical engagement
        with tools and equipment in our surrounding world. This mode of being-with-things
        is more primordial than theoretical knowledge.
        """

        # Create a temporary document
        temp_doc_path = Path("temp_philosophy_doc.txt")
        temp_doc_path.write_text(sample_doc_content)

        try:
            # Progress callback for document processing
            def progress_callback(progress: PhilosophyProcessingProgress):
                logger.info(
                    "Document Processing: %.1f%% - %s",
                    progress.overall_progress,
                    progress.current_stage,
                )
                logger.info(
                    "  Pages: %s/%s",
                    progress.processed_pages,
                    progress.total_pages,
                )
                logger.info(
                    "  Neologisms: %s/%s",
                    progress.processed_neologisms,
                    progress.total_neologisms,
                )

            print("Processing document with philosophy awareness...")

            # Process the document
            result = await self.processor.process_document_with_philosophy_awareness(
                file_path=str(temp_doc_path),
                source_lang="en",
                target_lang="de",
                provider="auto",
                user_id="philosopher_user",
                progress_callback=progress_callback,
            )

            print("\nDocument processing completed!")
            print(f"Session ID: {result.session_id}")
            print(
                f"Total neologisms detected: {result.document_neologism_analysis.total_neologisms}"
            )
            print(f"User choices applied: {result.total_choices_applied}")
            print(f"Processing time: {result.processing_time:.2f} seconds")

            # Create translated document
            output_path = await self.processor.create_translated_document_with_philosophy_awareness(
                result, "translated_philosophy_doc.pdf"
            )

            print(f"Translated document saved to: {output_path}")

            return result

        finally:
            # Clean up temporary file
            if temp_doc_path.exists():
                temp_doc_path.unlink()

    async def example_6_convenience_function(self):
        """Example 6: Using convenience functions for quick processing."""
        print("\n" + "=" * 80)
        print("EXAMPLE 6: Convenience Functions for Quick Processing")
        print("=" * 80)

        # Create a sample document
        sample_content = """
        Phenomenology and Existentialism

        Edmund Husserl's phenomenology introduced the concept of epoché (bracketing),
        which involves suspending judgment about the natural attitude. This method
        reveals the intentional structure of consciousness.

        Sartre's existentialism builds on phenomenology but emphasizes human freedom
        and responsibility. The concept of mauvaise foi (bad faith) describes how
        humans often deny their fundamental freedom.
        """

        temp_doc_path = Path("temp_phenomenology_doc.txt")
        temp_doc_path.write_text(sample_content)

        try:
            print("Using convenience function for quick processing...")

            # Use convenience function
            result, output_path = await process_document_with_philosophy_awareness(
                file_path=str(temp_doc_path),
                source_lang="en",
                target_lang="fr",
                provider="auto",
                user_id="phenomenologist_user",
            )

            print("Processing completed!")
            print(f"Output saved to: {output_path}")
            print(
                f"Neologisms detected: {result.document_neologism_analysis.total_neologisms}"
            )

            return result, output_path

        finally:
            # Clean up temporary file
            if temp_doc_path.exists():
                temp_doc_path.unlink()


class StatisticsAndPerformanceExamples:
    """Examples demonstrating statistics and performance monitoring."""

    def __init__(self):
        self.service = PhilosophyEnhancedTranslationService()
        self.processor = create_philosophy_enhanced_document_processor()

    def example_7_statistics_monitoring(self):
        """Example 7: Statistics and performance monitoring."""
        print("\n" + "=" * 80)
        print("EXAMPLE 7: Statistics and Performance Monitoring")
        print("=" * 80)

        # Perform several translation operations
        sample_texts = [
            "Dasein is the fundamental concept in Heidegger's ontology.",
            "The concept of Angst reveals the groundlessness of existence.",
            "Zuhandenheit describes our practical engagement with tools.",
        ]

        print("Performing multiple translations to generate statistics...")

        for i, text in enumerate(sample_texts):
            result = self.service.translate_text_with_neologism_handling(
                text=text,
                source_lang="en",
                target_lang="de",
                provider="auto",
                session_id=f"stats_session_{i}",
            )
            print(
                f"Translation {i + 1}: {result.neologism_analysis.total_neologisms} neologisms"
            )

        # Get translation service statistics
        translation_stats = self.service.get_statistics()
        print("\nTranslation Service Statistics:")
        print(
            f"  Philosophy-enhanced translations: {translation_stats['philosophy_enhanced_stats']['total_translations']}"
        )
        print(
            f"  Total neologisms detected: {translation_stats['philosophy_enhanced_stats']['total_neologisms_detected']}"
        )
        print(
            f"  Average detection time: {translation_stats['philosophy_enhanced_stats']['average_detection_time']:.3f}s"
        )

        # Get document processor statistics
        processor_stats = self.processor.get_statistics()
        print("\nDocument Processor Statistics:")
        print(
            f"  Documents processed: {processor_stats['philosophy_enhanced_processor_stats']['documents_processed']}"
        )
        print(
            f"  Total processing time: {processor_stats['philosophy_enhanced_processor_stats']['total_processing_time']:.2f}s"
        )

        return translation_stats, processor_stats


async def main():
    """Main function to run all examples."""
    _configure_logging()

    print("Philosophy-Enhanced Translation System Examples")
    print("=" * 80)

    # Text translation examples
    text_examples = PhilosophyTranslationExamples()

    # Example 1: Basic text translation
    text_examples.example_1_basic_text_translation()

    # Example 2: Batch translation
    text_examples.example_2_batch_translation()

    # Example 3: User choice management
    text_examples.example_3_user_choice_management()

    # Example 4: Progress tracking
    text_examples.example_4_progress_tracking()

    # Document processing examples
    doc_examples = PhilosophyDocumentExamples()

    # Example 5: Document processing
    await doc_examples.example_5_document_processing()

    # Example 6: Convenience functions
    await doc_examples.example_6_convenience_function()

    # Statistics examples
    stats_examples = StatisticsAndPerformanceExamples()

    # Example 7: Statistics monitoring
    stats_examples.example_7_statistics_monitoring()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
