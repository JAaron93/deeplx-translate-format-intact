"""Philosophy-Enhanced Translation Service with Neologism Detection and User Choice Management."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from models.neologism_models import DetectedNeologism, NeologismAnalysis
from models.user_choice_models import ChoiceType, UserChoice

from .neologism_detector import NeologismDetector
from .translation_service import TranslationService
from .user_choice_manager import UserChoiceManager

logger = logging.getLogger(__name__)


@dataclass
class PhilosophyTranslationProgress:
    """Progress tracking for philosophy-enhanced translation."""

    total_chunks: int = 0
    processed_chunks: int = 0
    total_neologisms: int = 0
    processed_neologisms: int = 0
    choices_applied: int = 0
    translation_progress: int = 0

    @property
    def overall_progress(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_chunks == 0:
            return 0.0

        # Weight different stages
        detection_weight = 0.3
        choice_weight = 0.2
        translation_weight = 0.5

        detection_progress = self.processed_chunks / self.total_chunks
        choice_progress = (
            (self.processed_neologisms / max(1, self.total_neologisms))
            if self.total_neologisms > 0
            else 1.0
        )
        translation_progress = self.translation_progress / 100.0

        return (
            detection_progress * detection_weight
            + choice_progress * choice_weight
            + translation_progress * translation_weight
        ) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "total_neologisms": self.total_neologisms,
            "processed_neologisms": self.processed_neologisms,
            "choices_applied": self.choices_applied,
            "translation_progress": self.translation_progress,
            "overall_progress": self.overall_progress,
        }


@dataclass
class NeologismPreservationResult:
    """Result of neologism preservation during translation."""

    original_text: str
    modified_text: str
    preserved_neologisms: list[DetectedNeologism]
    applied_choices: list[UserChoice]
    preservation_markers: dict[str, str]  # Maps placeholders to original terms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_text": self.original_text,
            "modified_text": self.modified_text,
            "preserved_neologisms": [n.to_dict() for n in self.preserved_neologisms],
            "applied_choices": [c.to_dict() for c in self.applied_choices],
            "preservation_markers": self.preservation_markers,
        }


class PhilosophyEnhancedTranslationService:
    """Philosophy-Enhanced Translation Service with integrated neologism detection and user choice management.

    This service wraps the existing TranslationService and adds:
    - Automatic neologism detection during translation
    - User choice application for handling detected neologisms
    - Preservation of philosophical terms according to user preferences
    - Enhanced progress tracking for philosophy-aware translation
    """

    def __init__(
        self,
        translation_service: Optional[TranslationService] = None,
        neologism_detector: Optional[NeologismDetector] = None,
        user_choice_manager: Optional[UserChoiceManager] = None,
        terminology_path: Optional[str] = None,
        preserve_neologisms_by_default: bool = True,
        neologism_confidence_threshold: float = 0.5,
        chunk_size: int = 2000,
    ):
        """Initialize the philosophy-enhanced translation service.

        Args:
            translation_service: Base translation service instance
            neologism_detector: Neologism detection service
            user_choice_manager: User choice management service
            terminology_path: Path to terminology file
            preserve_neologisms_by_default: Whether to preserve unhandled neologisms
            neologism_confidence_threshold: Minimum confidence for neologism detection
            chunk_size: Size of text chunks for processing
        """
        # Initialize base translation service
        self.translation_service = translation_service or TranslationService(
            terminology_map=(
                self._load_terminology(terminology_path) if terminology_path else None
            )
        )

        # Initialize neologism detection
        self.neologism_detector = neologism_detector or NeologismDetector(
            terminology_path=terminology_path,
            philosophical_threshold=neologism_confidence_threshold,
        )

        # Initialize user choice management
        self.user_choice_manager = user_choice_manager or UserChoiceManager()

        # Configuration
        self.preserve_neologisms_by_default = preserve_neologisms_by_default
        self.neologism_confidence_threshold = neologism_confidence_threshold
        self.chunk_size = chunk_size

        # Statistics tracking
        self.stats = {
            "total_translations": 0,
            "neologisms_detected": 0,
            "choices_applied": 0,
            "neologisms_preserved": 0,
            "neologisms_translated": 0,
            "processing_time": 0.0,
        }

        logger.info("PhilosophyEnhancedTranslationService initialized")

    def _load_terminology(self, terminology_path: str) -> dict[str, str]:
        """Load terminology mapping from file."""
        try:
            with open(terminology_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load terminology from {terminology_path}: {e}")
            return {}

    async def translate_text_with_neologism_handling(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        provider: str = "auto",
        session_id: Optional[str] = None,
        progress_callback: Optional[
            Callable[[PhilosophyTranslationProgress], None]
        ] = None,
    ) -> dict[str, Any]:
        """Translate text with integrated neologism detection and user choice handling.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            provider: Translation provider to use
            session_id: User session ID for choice context
            progress_callback: Optional progress callback

        Returns:
            Dictionary with translation results and neologism analysis
        """
        start_time = time.time()

        # Initialize progress tracking
        progress = PhilosophyTranslationProgress()

        try:
            # Step 1: Detect neologisms in the text
            logger.info(f"Detecting neologisms in text (length: {len(text)})")
            neologism_analysis = await self._detect_neologisms_async(
                text, progress, progress_callback
            )

            # Step 2: Apply user choices to detected neologisms
            logger.info(
                f"Applying user choices to {len(neologism_analysis.detected_neologisms)} neologisms"
            )
            preservation_result = await self._apply_user_choices_async(
                text, neologism_analysis, session_id, progress, progress_callback
            )

            # Step 3: Translate the modified text
            logger.info(
                f"Translating modified text with {len(preservation_result.preservation_markers)} preserved terms"
            )
            translated_text = await self._translate_with_preservation_async(
                preservation_result.modified_text,
                source_lang,
                target_lang,
                provider,
                progress,
                progress_callback,
            )

            # Step 4: Restore preserved neologisms
            logger.info("Restoring preserved neologisms in translated text")
            final_translation = await self._restore_preserved_neologisms_async(
                translated_text, preservation_result, progress, progress_callback
            )

            # Update statistics
            self.stats["total_translations"] += 1
            self.stats["neologisms_detected"] += len(
                neologism_analysis.detected_neologisms
            )
            self.stats["choices_applied"] += len(preservation_result.applied_choices)
            self.stats["processing_time"] += time.time() - start_time

            # Final progress update
            progress.translation_progress = 100
            if progress_callback:
                progress_callback(progress)

            return {
                "translated_text": final_translation,
                "original_text": text,
                "neologism_analysis": neologism_analysis.to_dict(),
                "preservation_result": preservation_result.to_dict(),
                "translation_metadata": {
                    "provider_used": provider,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "session_id": session_id,
                    "processing_time": time.time() - start_time,
                    "neologisms_detected": len(neologism_analysis.detected_neologisms),
                    "choices_applied": len(preservation_result.applied_choices),
                },
            }

        except Exception as e:
            logger.error(f"Error in philosophy-enhanced translation: {e}")
            # Fallback to regular translation
            fallback_translation = await self.translation_service.translate_text(
                text, source_lang, target_lang, provider
            )

            return {
                "translated_text": fallback_translation,
                "original_text": text,
                "error": str(e),
                "fallback_used": True,
                "translation_metadata": {
                    "provider_used": provider,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "processing_time": time.time() - start_time,
                },
            }

    async def translate_batch_with_neologism_handling(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        provider: str = "auto",
        session_id: Optional[str] = None,
        progress_callback: Optional[
            Callable[[PhilosophyTranslationProgress], None]
        ] = None,
    ) -> list[dict[str, Any]]:
        """Translate a batch of texts with neologism handling.

        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            provider: Translation provider to use
            session_id: User session ID for choice context
            progress_callback: Optional progress callback

        Returns:
            List of translation results
        """
        results = []

        # Initialize progress tracking
        progress = PhilosophyTranslationProgress()
        progress.total_chunks = len(texts)

        for i, text in enumerate(texts):
            try:
                # Update progress
                progress.processed_chunks = i
                if progress_callback:
                    progress_callback(progress)

                # Translate individual text
                result = await self.translate_text_with_neologism_handling(
                    text, source_lang, target_lang, provider, session_id
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error translating text {i}: {e}")
                # Add error result
                results.append(
                    {
                        "translated_text": text,  # Fallback to original
                        "original_text": text,
                        "error": str(e),
                        "text_index": i,
                    }
                )

        # Final progress update
        progress.processed_chunks = len(texts)
        if progress_callback:
            progress_callback(progress)

        return results

    async def _detect_neologisms_async(
        self,
        text: str,
        progress: PhilosophyTranslationProgress,
        progress_callback: Optional[
            Callable[[PhilosophyTranslationProgress], None]
        ] = None,
    ) -> NeologismAnalysis:
        """Detect neologisms in text asynchronously."""
        # Run neologism detection in a thread to avoid blocking
        analysis = await asyncio.to_thread(
            self.neologism_detector.analyze_text, text, chunk_size=self.chunk_size
        )

        # Filter by confidence threshold
        high_confidence_neologisms = [
            n
            for n in analysis.detected_neologisms
            if n.confidence >= self.neologism_confidence_threshold
        ]

        # Update analysis with filtered neologisms
        analysis.detected_neologisms = high_confidence_neologisms
        analysis.total_detections = len(high_confidence_neologisms)

        # Update progress
        progress.total_neologisms = len(high_confidence_neologisms)
        if progress_callback:
            progress_callback(progress)

        return analysis

    async def _apply_user_choices_async(
        self,
        text: str,
        analysis: NeologismAnalysis,
        session_id: Optional[str],
        progress: PhilosophyTranslationProgress,
        progress_callback: Optional[
            Callable[[PhilosophyTranslationProgress], None]
        ] = None,
    ) -> NeologismPreservationResult:
        """Apply user choices to detected neologisms."""
        preservation_markers = {}
        applied_choices = []
        preserved_neologisms = []
        modified_text = text

        # Process each detected neologism
        for i, neologism in enumerate(analysis.detected_neologisms):
            # Update progress
            progress.processed_neologisms = i + 1
            if progress_callback:
                progress_callback(progress)

            # Get user choice for this neologism
            user_choice = await asyncio.to_thread(
                self.user_choice_manager.get_choice_for_neologism, neologism, session_id
            )

            if user_choice:
                # Apply existing choice
                modified_text, marker = self._apply_choice_to_text(
                    modified_text, neologism, user_choice
                )
                if marker:
                    preservation_markers[marker] = neologism.term
                applied_choices.append(user_choice)
                preserved_neologisms.append(neologism)
                progress.choices_applied += 1

            elif self.preserve_neologisms_by_default:
                # Default preservation behavior
                marker = self._create_preservation_marker(neologism.term)
                modified_text = self._replace_term_with_marker(
                    modified_text, neologism, marker
                )
                preservation_markers[marker] = neologism.term
                preserved_neologisms.append(neologism)
                self.stats["neologisms_preserved"] += 1

        return NeologismPreservationResult(
            original_text=text,
            modified_text=modified_text,
            preserved_neologisms=preserved_neologisms,
            applied_choices=applied_choices,
            preservation_markers=preservation_markers,
        )

    def _apply_choice_to_text(
        self, text: str, neologism: DetectedNeologism, choice: UserChoice
    ) -> tuple[str, Optional[str]]:
        """Apply a user choice to text."""
        if choice.choice_type == ChoiceType.PRESERVE:
            # Preserve the term as-is with a marker
            marker = self._create_preservation_marker(neologism.term)
            modified_text = self._replace_term_with_marker(text, neologism, marker)
            return modified_text, marker

        elif choice.choice_type == ChoiceType.TRANSLATE:
            # Let the term be translated normally
            return text, None

        elif choice.choice_type == ChoiceType.CUSTOM_TRANSLATION:
            # Replace with custom translation
            if choice.translation_result:
                marker = self._create_preservation_marker(choice.translation_result)
                modified_text = self._replace_term_with_marker(text, neologism, marker)
                return modified_text, marker
            else:
                return text, None

        elif choice.choice_type == ChoiceType.CONTEXT_DEPENDENT:
            # Apply context-dependent logic
            if choice.translation_result:
                marker = self._create_preservation_marker(choice.translation_result)
                modified_text = self._replace_term_with_marker(text, neologism, marker)
                return modified_text, marker
            else:
                # Default to preserve
                marker = self._create_preservation_marker(neologism.term)
                modified_text = self._replace_term_with_marker(text, neologism, marker)
                return modified_text, marker

        else:
            # Default behavior
            return text, None

    def _create_preservation_marker(self, term: str) -> str:
        """Create a unique preservation marker for a term."""
        import hashlib

        hash_obj = hashlib.sha256(term.encode())
        return f"__PRESERVE_{hash_obj.hexdigest()[:8].upper()}__"

    def _replace_term_with_marker(
        self, text: str, neologism: DetectedNeologism, marker: str
    ) -> str:
        """Replace a neologism term with a preservation marker."""
        # Use word boundaries to avoid partial matches
        pattern = rf"\b{re.escape(neologism.term)}\b"
        return re.sub(pattern, marker, text, flags=re.IGNORECASE)

    def _preserve_neologisms_in_text(
        self, text: str, neologisms: list[DetectedNeologism]
    ) -> tuple[str, dict[str, str]]:
        """Preserve neologisms in text by replacing them with markers.

        Args:
            text: Original text
            neologisms: List of neologisms to preserve

        Returns:
            Tuple of (preserved_text, markers_dict)
        """
        preserved_text = text
        markers = {}

        # Sort neologisms by position (reverse order to avoid position shifts)
        sorted_neologisms = sorted(neologisms, key=lambda n: n.start_pos, reverse=True)

        for i, neologism in enumerate(sorted_neologisms):
            marker = self._create_preservation_marker(neologism.term)
            preserved_text = self._replace_term_with_marker(
                preserved_text, neologism, marker
            )
            markers[marker] = neologism.term

        return preserved_text, markers

    def _restore_neologisms_in_text(self, text: str, markers: dict[str, str]) -> str:
        """Restore neologisms in text by replacing markers with original terms.

        Args:
            text: Text with preservation markers
            markers: Dictionary mapping markers to original terms

        Returns:
            Text with markers replaced by original terms
        """
        restored_text = text

        for marker, original_term in markers.items():
            restored_text = restored_text.replace(marker, original_term)

        return restored_text

    async def _translate_with_preservation_async(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        provider: str,
        progress: PhilosophyTranslationProgress,
        progress_callback: Optional[
            Callable[[PhilosophyTranslationProgress], None]
        ] = None,
    ) -> str:
        """Translate text with preserved markers."""
        # Use the existing translation service
        translated_text = await self.translation_service.translate_text(
            text, source_lang, target_lang, provider
        )

        # Update progress
        progress.translation_progress = 100
        if progress_callback:
            progress_callback(progress)

        return translated_text

    async def _restore_preserved_neologisms_async(
        self,
        translated_text: str,
        preservation_result: NeologismPreservationResult,
        progress: PhilosophyTranslationProgress,
        progress_callback: Optional[
            Callable[[PhilosophyTranslationProgress], None]
        ] = None,
    ) -> str:
        """Restore preserved neologisms in translated text."""
        restored_text = translated_text

        # Replace markers with original terms or custom translations
        for marker, original_term in preservation_result.preservation_markers.items():
            # Find the corresponding choice to see if we should use custom translation
            custom_translation = None
            for choice in preservation_result.applied_choices:
                if choice.neologism_term == original_term and choice.translation_result:
                    custom_translation = choice.translation_result
                    break

            # Use custom translation if available, otherwise use original term
            replacement = custom_translation if custom_translation else original_term
            restored_text = restored_text.replace(marker, replacement)

        return restored_text

    async def translate_document_with_neologism_handling(
        self,
        content: dict[str, Any],
        source_lang: str,
        target_lang: str,
        provider: str = "auto",
        session_id: Optional[str] = None,
        progress_callback: Optional[
            Callable[[PhilosophyTranslationProgress], None]
        ] = None,
    ) -> dict[str, Any]:
        """Translate document content with neologism handling.

        Args:
            content: Document content from document processor
            source_lang: Source language code
            target_lang: Target language code
            provider: Translation provider to use
            session_id: User session ID for choice context
            progress_callback: Optional progress callback

        Returns:
            Translated document content with neologism metadata
        """
        start_time = time.time()

        # Initialize progress tracking
        progress = PhilosophyTranslationProgress()

        try:
            # Extract text blocks for translation
            text_blocks = self._extract_text_blocks(content)
            progress.total_chunks = len(text_blocks)

            # Process each text block
            translated_blocks = []
            neologism_analyses = []

            for i, block in enumerate(text_blocks):
                # Update progress
                progress.processed_chunks = i
                if progress_callback:
                    progress_callback(progress)

                # Translate block with neologism handling
                if block.get("text", "").strip():
                    result = await self.translate_text_with_neologism_handling(
                        block["text"], source_lang, target_lang, provider, session_id
                    )

                    # Update block with translation
                    translated_block = block.copy()
                    translated_block["text"] = result["translated_text"]
                    translated_block["neologism_analysis"] = result.get(
                        "neologism_analysis", {}
                    )
                    translated_blocks.append(translated_block)

                    # Collect neologism analysis
                    if "neologism_analysis" in result:
                        neologism_analyses.append(result["neologism_analysis"])
                else:
                    # Keep empty blocks as-is
                    translated_blocks.append(block)

            # Reconstruct document content
            translated_content = self._reconstruct_content(content, translated_blocks)

            # Final progress update
            progress.processed_chunks = len(text_blocks)
            progress.translation_progress = 100
            if progress_callback:
                progress_callback(progress)

            return {
                "translated_content": translated_content,
                "original_content": content,
                "neologism_analyses": neologism_analyses,
                "translation_metadata": {
                    "provider_used": provider,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "session_id": session_id,
                    "processing_time": time.time() - start_time,
                    "total_blocks": len(text_blocks),
                    "total_neologism_analyses": len(neologism_analyses),
                },
            }

        except Exception as e:
            logger.error(f"Error in document translation with neologism handling: {e}")
            # Fallback to regular document translation
            fallback_result = await self.translation_service.translate_document(
                content, source_lang, target_lang, provider, progress_callback
            )

            return {
                "translated_content": fallback_result,
                "original_content": content,
                "error": str(e),
                "fallback_used": True,
                "translation_metadata": {
                    "provider_used": provider,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "processing_time": time.time() - start_time,
                },
            }

    def _extract_text_blocks(self, content: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract text blocks from document content."""
        text_blocks = []

        if "pages" in content:
            for page_num, page in enumerate(content["pages"]):
                if "text_blocks" in page:
                    for block_num, block in enumerate(page["text_blocks"]):
                        text_blocks.append(
                            {
                                "page": page_num,
                                "block": block_num,
                                "text": block.get("text", ""),
                                "formatting": block.get("formatting", {}),
                                "position": block.get("position", {}),
                            }
                        )

        return text_blocks

    def _reconstruct_content(
        self, original_content: dict[str, Any], translated_blocks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reconstruct document content with translations."""
        content = original_content.copy()

        # Create lookup for translated blocks
        block_lookup = {}
        for block in translated_blocks:
            key = (block["page"], block["block"])
            block_lookup[key] = block

        # Update content with translations
        if "pages" in content:
            for page_num, page in enumerate(content["pages"]):
                if "text_blocks" in page:
                    for block_num, block in enumerate(page["text_blocks"]):
                        key = (page_num, block_num)
                        if key in block_lookup:
                            translated_block = block_lookup[key]
                            block["text"] = translated_block["text"]
                            # Add neologism analysis if available
                            if "neologism_analysis" in translated_block:
                                block["neologism_analysis"] = translated_block[
                                    "neologism_analysis"
                                ]

        return content

    def _select_best_provider(self) -> str:
        """Select the best available provider."""
        return self.translation_service._select_best_provider()

    def get_available_providers(self) -> list[str]:
        """Get list of available translation providers."""
        return self.translation_service.get_available_providers()

    def get_statistics(self) -> dict[str, Any]:
        """Get service statistics."""
        base_stats = self.translation_service.get_available_providers()
        detector_stats = self.neologism_detector.get_statistics()
        choice_stats = self.user_choice_manager.get_statistics()

        return {
            "philosophy_enhanced_stats": self.stats,
            "available_providers": base_stats,
            "neologism_detector_stats": detector_stats,
            "user_choice_manager_stats": choice_stats,
            "configuration": {
                "preserve_neologisms_by_default": self.preserve_neologisms_by_default,
                "neologism_confidence_threshold": self.neologism_confidence_threshold,
                "chunk_size": self.chunk_size,
            },
        }

    def update_configuration(self, **kwargs) -> None:
        """Update service configuration."""
        if "preserve_neologisms_by_default" in kwargs:
            self.preserve_neologisms_by_default = kwargs[
                "preserve_neologisms_by_default"
            ]

        if "neologism_confidence_threshold" in kwargs:
            self.neologism_confidence_threshold = kwargs[
                "neologism_confidence_threshold"
            ]

        if "chunk_size" in kwargs:
            self.chunk_size = kwargs["chunk_size"]

        logger.info("Philosophy-enhanced translation service configuration updated")


# Convenience functions for easy integration


def create_philosophy_enhanced_translation_service(
    terminology_path: Optional[str] = None, db_path: str = "user_choices.db", **kwargs
) -> PhilosophyEnhancedTranslationService:
    """Create a philosophy-enhanced translation service with default components."""
    return PhilosophyEnhancedTranslationService(
        terminology_path=terminology_path,
        user_choice_manager=UserChoiceManager(db_path=db_path),
        **kwargs,
    )


async def translate_with_philosophy_awareness(
    text: str,
    source_lang: str,
    target_lang: str,
    provider: str = "auto",
    session_id: Optional[str] = None,
    terminology_path: Optional[str] = None,
) -> dict[str, Any]:
    """Quick function to translate text with philosophy awareness."""
    service = create_philosophy_enhanced_translation_service(terminology_path)
    return await service.translate_text_with_neologism_handling(
        text, source_lang, target_lang, provider, session_id
    )
