"""Translation service with multiple provider support"""

import asyncio
import logging
from __future__ import annotations

from typing import List, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod
import json
import os
import requests
import deepl
from google.cloud import translate_v2 as translate
import re

# Translation service configuration
TRANSLATION_DELAY = float(os.getenv('TRANSLATION_DELAY', '0.1'))  # Delay between batch requests in seconds

logger = logging.getLogger(__name__)

class BaseTranslator(ABC):
    """Abstract base class for translation providers"""
    
    @abstractmethod
    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        pass
    
    @abstractmethod
    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        pass

class DeepLTranslator(BaseTranslator):
    """DeepL translation implementation"""
    
    def __init__(self, api_key: str):
        self.translator = deepl.Translator(api_key)
    
    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            result = await asyncio.to_thread(
                self.translator.translate_text,
                text,
                source_lang=source_lang,
                target_lang=target_lang,
                preserve_formatting=True,
                tag_handling="html"
            )
            return result.text
        except Exception as e:
            logger.error(f"DeepL translation error: {e}")
            return text
    
    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        try:
            results = await asyncio.to_thread(
                self.translator.translate_text,
                texts,
                source_lang=source_lang,
                target_lang=target_lang,
                preserve_formatting=True,
                tag_handling="html"
            )
            return [result.text for result in results]
        except Exception as e:
            logger.error(f"DeepL batch translation error: {e}")
            return texts

class DeepLXTranslator(BaseTranslator):
    """DeepLX HTTP translation implementation"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'DeepLX-Client/1.0'
        })
    
    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            payload = {
                "text": text,
                "source_lang": source_lang.upper(),
                "target_lang": target_lang.upper()
            }
            
            response = await asyncio.to_thread(
                self.session.post,
                self.endpoint,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'data' in result:
                    return result['data']
                elif 'text' in result:
                    return result['text']
                else:
                    logger.warning(f"Unexpected DeepLX response format: {result}")
                    return text
            else:
                logger.error(f"DeepLX HTTP error {response.status_code}: {response.text}")
                return text
                
        except Exception as e:
            logger.error(f"DeepLX translation error: {e}")
            return text
    
    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate batch of texts using DeepLX"""
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

class GoogleTranslator(BaseTranslator):
    """Google Cloud Translation implementation"""
    
    def __init__(self):
        self.client = translate.Client()
    
    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            result = await asyncio.to_thread(
                self.client.translate,
                text,
                source_language=source_lang.lower(),
                target_language=target_lang.lower()
            )
            return result['translatedText']
        except Exception as e:
            logger.error(f"Google translation error: {e}")
            return text
    
    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        try:
            results = await asyncio.to_thread(
                self.client.translate,
                texts,
                source_language=source_lang.lower(),
                target_language=target_lang.lower()
            )
            return [result['translatedText'] for result in results]
        except Exception as e:
            logger.error(f"Google batch translation error: {e}")
            return texts

class AzureTranslator(BaseTranslator):
    """Azure Cognitive Services Translator implementation"""
    
    def __init__(self, api_key: str, region: str):
        self.api_key = api_key
        self.region = region
        self.endpoint = "https://api.cognitive.microsofttranslator.com"
        self.session = requests.Session()
    
    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            path = '/translate'
            constructed_url = self.endpoint + path
            
            params = {
                'api-version': '3.0',
                'from': source_lang.lower(),
                'to': target_lang.lower(),
                'textType': 'html'
            }
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key,
                'Ocp-Apim-Subscription-Region': self.region,
                'Content-type': 'application/json'
            }
            
            body = [{'text': text}]
            
            response = await asyncio.to_thread(
                self.session.post,
                constructed_url,
                params=params,
                headers=headers,
                json=body,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            return result[0]['translations'][0]['text']
            
        except Exception as e:
            logger.error(f"Azure translation error: {e}")
            return text
    
    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        try:
            path = '/translate'
            constructed_url = self.endpoint + path
            
            params = {
                'api-version': '3.0',
                'from': source_lang.lower(),
                'to': target_lang.lower(),
                'textType': 'html'
            }
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key,
                'Ocp-Apim-Subscription-Region': self.region,
                'Content-type': 'application/json'
            }
            
            body = [{'text': text} for text in texts]
            
            response = await asyncio.to_thread(
                self.session.post,
                constructed_url,
                params=params,
                headers=headers,
                json=body,
                timeout=60
            )
            
            response.raise_for_status()
            results = response.json()
            return [result['translations'][0]['text'] for result in results]
            
        except Exception as e:
            logger.error(f"Azure batch translation error: {e}")
            return texts

class TranslationService:
    """Main translation service with provider management"""
    
    def __init__(self, terminology_map: Optional[Dict[str, str]] = None) -> None:
        # Mapping of provider name to translator instance
        self.providers: Dict[str, BaseTranslator] = {}
        # Optional terminology mapping for preprocessing
        self.terminology_map: Dict[str, str] = terminology_map or {}
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize available translation providers"""
        
        # DeepL
        try:
            deepl_key = os.getenv('DEEPL_API_KEY')
            if deepl_key:
                self.providers['deepl'] = DeepLTranslator(deepl_key)
                logger.info("DeepL translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize DeepL: {e}")
        
        # DeepLX
        try:
            deeplx_endpoint = os.getenv('DEEPLX_ENDPOINT')
            if deeplx_endpoint:
                self.providers['deeplx'] = DeepLXTranslator(deeplx_endpoint)
                logger.info("DeepLX translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize DeepLX: {e}")
        
        # Google
        try:
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                self.providers['google'] = GoogleTranslator()
                logger.info("Google translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Google: {e}")
        
        # Azure
        try:
            azure_key = os.getenv('AZURE_TRANSLATOR_KEY')
            azure_region = os.getenv('AZURE_TRANSLATOR_REGION', 'global')
            if azure_key:
                self.providers['azure'] = AzureTranslator(azure_key, azure_region)
                logger.info("Azure translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure: {e}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available translation providers"""
        return list(self.providers.keys())

    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        provider: str = "auto",
    ) -> List[str]:
        """Translate a list of texts using the chosen provider (or best available)."""
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
            translated = await translator.translate_batch(batch_texts, source_lang, target_lang)
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
        """Translate a single text string using the chosen provider (or best available)."""
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
        content: Dict[str, Any],
        source_lang: str,
        target_lang: str,
        provider: str = "auto",
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Dict[str, Any]:
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
        
        # Translate text blocks using helper method
        translated_blocks = await self._translate_batches(
            translator,
            text_blocks,
            source_lang,
            target_lang,
            batch_size=50,
            progress_callback=progress_callback,
        )
        
        # Reconstruct content with translations
        translated_content = self._reconstruct_content(content, translated_blocks)
        
        if progress_callback:
            progress_callback(100)
        
        return translated_content
    
    def _select_best_provider(self) -> str:
        """Select the best available provider"""
        # Priority order: DeepL > DeepLX > Google > Azure
        priority_order = ['deepl', 'deeplx', 'google', 'azure']
        
        for provider in priority_order:
            if provider in self.providers:
                return provider
        
        raise ValueError("No translation providers available")
    
    def _extract_text_blocks(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract text blocks from document content"""
        text_blocks = []
        
        if 'pages' in content:
            for page_num, page in enumerate(content['pages']):
                if 'text_blocks' in page:
                    for block_num, block in enumerate(page['text_blocks']):
                        text_blocks.append({
                            'page': page_num,
                            'block': block_num,
                            'text': block.get('text', ''),
                            'formatting': block.get('formatting', {}),
                            'position': block.get('position', {})
                        })
        
        return text_blocks
    
    async def _translate_batches(
        self,
        translator: BaseTranslator,
        text_blocks: List[Dict[str, Any]],
        source_lang: str,
        target_lang: str,
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Translate text blocks in batches and optionally report progress"""
        total_blocks = len(text_blocks)
        translated_blocks: List[Dict[str, Any]] = []

        for i in range(0, total_blocks, batch_size):
            batch = text_blocks[i:i + batch_size]
            batch_texts = [block["text"] for block in batch]

            # Apply terminology preprocessing if configured
            if self.terminology_map:
                batch_texts = [self._apply_terminology(t) for t in batch_texts]

            # Perform batch translation with the selected provider
            translated_texts = await translator.translate_batch(
                batch_texts, source_lang, target_lang
            )
            # Strip tags
            translated_texts = [self._strip_non_translate_tags(t) for t in translated_texts]

            # Merge translated texts back into their respective blocks
            for j, translated_text in enumerate(translated_texts):
                block = batch[j].copy()
                block["text"] = translated_text
                translated_blocks.append(block)

            # Progress callback per batch
            if progress_callback:
                progress = int((i + len(batch)) / total_blocks * 100)
                progress_callback(progress)

        return translated_blocks

    def _reconstruct_content(self, original_content: Dict[str, Any], translated_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reconstruct document content with translations"""
        content = original_content.copy()
        
        # Create lookup for translated blocks
        block_lookup = {}
        for block in translated_blocks:
            key = (block['page'], block['block'])
            block_lookup[key] = block
        
        # Update content with translations
        if 'pages' in content:
            for page_num, page in enumerate(content['pages']):
                if 'text_blocks' in page:
                    for block_num, block in enumerate(page['text_blocks']):
                        key = (page_num, block_num)
                        if key in block_lookup:
                            block['text'] = block_lookup[key]['text']
        
        return content

    def _strip_non_translate_tags(self, text: str) -> str:
        """Remove <span translate="no"> wrappers from translated text."""
        return re.sub(r"<span translate=\"no\">(.*?)</span>", r"\1", text, flags=re.IGNORECASE)

    def _apply_terminology(self, text: str) -> str:
        """Replace terms in text using self.terminology_map with word-boundary safety."""
        processed = text
        for source, target in self.terminology_map.items():
            # Wrap term in HTML span with translate="no" to preserve it
            pattern = rf"(?<!\\w){re.escape(source)}(?!\\w)"
            processed = re.sub(pattern, f"<span translate=\"no\">{source}</span>", processed)
        return processed