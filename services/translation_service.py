"""Translation service for Dolphin OCR Translate with Lingo.dev provider support"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import requests

# Translation service configuration
TRANSLATION_DELAY = float(
    os.getenv("TRANSLATION_DELAY", "0.1")
)  # Delay between batch requests in seconds

# Configurable User-Agent header
USER_AGENT = os.getenv("DOLPHIN_USER_AGENT", "Dolphin-OCR-Translate/2.0")

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """Abstract base class for translation providers"""

    @abstractmethod
    async def translate_text(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Translate a single text string"""
        pass

    @abstractmethod
    async def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """Translate a batch of text strings"""
        pass


class LingoTranslator(BaseTranslator):
    """Lingo.dev translation implementation"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("LINGO_API_KEY is required")

        self.api_key = api_key
        self.base_url = "https://api.lingo.dev/v1/translate"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            }
        )

    async def translate_text(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        try:
            if not text.strip():
                return text

            payload = {
                "text": text,
                "source": source_lang.lower(),
                "target": target_lang.lower(),
            }

            response = await asyncio.to_thread(
                self.session.post, self.base_url, json=payload, timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if "translation" in result:
                    return result["translation"]
                elif "text" in result:
                    return result["text"]
                else:
                    logger.warning(f"Unexpected Lingo response format: {result}")
                    return text
            else:
                logger.error(
                    f"Lingo HTTP error {response.status_code}: {response.text}"
                )
                return text

        except Exception as e:
            logger.error(f"Lingo translation error: {e}")
            return text

    async def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """Translate batch of texts using Lingo.dev"""
        results = []

        # Process texts individually for better error handling
        for text in texts:
            if not text.strip():
                results.append(text)
                continue

            translated = await self.translate_text(text, source_lang, target_lang)
            results.append(translated)

            # Configurable delay between batch requests to avoid rate limiting
            await asyncio.sleep(TRANSLATION_DELAY)

        return results


class TranslationService:
    """Main translation service with Lingo.dev provider"""

    def __init__(self, terminology_map: Optional[dict[str, str]] = None) -> None:
        # Mapping of provider name to translator instance
        self.providers: dict[str, BaseTranslator] = {}
        # Optional terminology mapping for preprocessing
        self.terminology_map: dict[str, str] = terminology_map or {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize Lingo.dev translation provider"""
        try:
            lingo_key = os.getenv("LINGO_API_KEY")
            if lingo_key:
                self.providers["lingo"] = LingoTranslator(lingo_key)
                logger.info("Lingo translator initialized")
            else:
                raise ValueError("LINGO_API_KEY not found in environment variables")
        except Exception as e:
            logger.error(f"Failed to initialize Lingo translator: {e}")
            raise

    def get_available_providers(self) -> list[str]:
        """Get list of available translation providers"""
        return list(self.providers.keys())

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        provider: str = "auto",
    ) -> list[str]:
        """Translate a list of texts using Lingo.dev provider."""
        if provider == "auto":
            provider = self._select_best_provider()

        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")

        translator = self.providers[provider]
        batch_texts = texts.copy()

        # Apply optional terminology preprocessing
        if self.terminology_map:
            batch_texts = [self._apply_terminology(t) for t in batch_texts]

        try:
            translated = await translator.translate_batch(
                batch_texts, source_lang, target_lang
            )
            # Strip non-translate tags after translation
            translated = [self._strip_non_translate_tags(t) for t in translated]
            return translated
        except Exception as e:
            logger.error(f"Batch translation failed with provider {provider}: {e}")
            return texts  # Fallback to original texts

    async def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        provider: str = "auto",
    ) -> str:
        """Translate a single text string using Lingo.dev provider."""
        if not text.strip():
            return text  # Nothing to translate

        # Apply optional terminology preprocessing
        if self.terminology_map:
            text = self._apply_terminology(text)

        if provider == "auto":
            provider = self._select_best_provider()

        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")

        translator = self.providers[provider]
        try:
            translated = await translator.translate_text(text, source_lang, target_lang)
            return self._strip_non_translate_tags(translated)
        except Exception as e:
            logger.error(f"Translation failed with provider {provider}: {e}")
            return text

    async def translate_document(
        self,
        content: dict[str, Any],
        source_lang: str,
        target_lang: str,
        provider: str = "auto",
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> dict[str, Any]:
        """Translate document content"""
        # Select provider
        if provider == "auto":
            provider = self._select_best_provider()

        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")

        translator = self.providers[provider]

        # Extract text blocks for translation
        text_blocks = self._extract_text_blocks(content)
        total_blocks = len(text_blocks)

        if progress_callback:
            progress_callback(0)

        # Process in batches for efficiency
        batch_size = 20
        translated_blocks = []

        for i in range(0, total_blocks, batch_size):
            batch = text_blocks[i : i + batch_size]
            batch_texts = [block["text"] for block in batch]

            # Apply terminology preprocessing if configured
            if self.terminology_map:
                batch_texts = [self._apply_terminology(t) for t in batch_texts]

            # Perform batch translation with the selected provider
            translated_texts = await translator.translate_batch(
                batch_texts, source_lang, target_lang
            )
            # Strip tags
            translated_texts = [
                self._strip_non_translate_tags(t) for t in translated_texts
            ]

            # Merge translated texts back into their respective blocks
            for j, translated_text in enumerate(translated_texts):
                block = batch[j].copy()
                block["text"] = translated_text
                translated_blocks.append(block)

            # Update progress
            if progress_callback:
                progress = min(100, int((i + len(batch)) / total_blocks * 100))
                progress_callback(progress)

        # Reconstruct document with translated content
        translated_content = self._reconstruct_document(content, translated_blocks)

        if progress_callback:
            progress_callback(100)

        return translated_content

    def _select_best_provider(self) -> str:
        """Select the best available provider (always Lingo.dev)."""
        if "lingo" in self.providers:
            return "lingo"

        raise ValueError("No translation providers available")

    def _extract_text_blocks(self, content: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract text blocks from document content for translation."""
        text_blocks = []

        if "pages" in content:
            for page_num, page in enumerate(content["pages"]):
                if "text_elements" in page:
                    for element in page["text_elements"]:
                        if element.get("text", "").strip():
                            text_blocks.append(
                                {
                                    "text": element["text"],
                                    "page": page_num,
                                    "element_id": element.get("id"),
                                    "metadata": element,
                                }
                            )

        return text_blocks

    def _reconstruct_document(
        self, original_content: dict[str, Any], translated_blocks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reconstruct document with translated text blocks."""
        # Use a deep copy to avoid mutating nested structures in the original
        translated_content = copy.deepcopy(original_content)

        # Create a mapping of translated blocks by page and element
        block_map = {}
        for block in translated_blocks:
            page_num = block["page"]
            element_id = block.get("element_id")
            if page_num not in block_map:
                block_map[page_num] = {}
            # Only index valid element IDs
            if element_id is not None:
                block_map[page_num][element_id] = block["text"]

        # Update the content with translated text
        if "pages" in translated_content:
            for page_num, page in enumerate(translated_content["pages"]):
                # Defensive check on page and expected key
                if page and page_num in block_map and "text_elements" in page:
                    for element in page["text_elements"]:
                        # Skip any null entries
                        if element:
                            element_id = element.get("id")
                            if (
                                element_id is not None
                                and element_id in block_map[page_num]
                            ):
                                element["text"] = block_map[page_num][element_id]

        return translated_content

    def _strip_non_translate_tags(self, text: str) -> str:
        """Remove <span translate="no"> wrappers from translated text."""
        return re.sub(
            r"<span translate=\"no\">(.*?)</span>", r"\1", text, flags=re.IGNORECASE
        )

    def _apply_terminology(self, text: str) -> str:
        """Replace terms in text using self.terminology_map with word-boundary safety."""
        processed = text
        for source, target in self.terminology_map.items():
            # Wrap term in HTML span with translate="no" to preserve it
            pattern = rf"(?<!\w){re.escape(source)}(?!\w)"
            processed = re.sub(
                pattern, f'<span translate="no">{source}</span>', processed
            )
        return processed
