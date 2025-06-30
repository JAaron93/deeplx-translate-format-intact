"""Translation service with multiple provider support"""

from __future__ import annotations

import asyncio
import logging

from typing import List, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod
import json
import os
import requests
from lara_sdk import Translator as LaraClient, Credentials as LaraCredentials
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
    
    def __init__(self, endpoint: str, api_key: Optional[str] = None):
        # Allow injecting API key via header or query string
        if api_key and 'key=' not in endpoint:
            # Append key as query parameter
            sep = '&' if '?' in endpoint else '?'
            endpoint = f"{endpoint}{sep}key={api_key}"
        self.endpoint = endpoint
        self.api_key = api_key
        self.session = requests.Session()
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'DeepLX-Client/1.0'
        }
        if api_key and 'key=' not in self.endpoint:
            headers['Authorization'] = f'Bearer {api_key}'
        self.session.headers.update(headers)
    
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

class LibreTranslateTranslator(BaseTranslator):
    """LibreTranslate public instance implementation"""

    def __init__(self, endpoint: str = "https://libretranslate.de/translate"):
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "LibreTranslate-Client/1.0",
            "Accept": "application/json"
        })

    async def _post_translate(self, payload: Dict[str, Any]) -> Optional[Any]:
        """Send POST to LibreTranslate and return JSON (or None on error)."""
        try:
            response = await asyncio.to_thread(
                self.session.post,
                self.endpoint,
                json=payload,
                timeout=45,
            )
            if response.status_code == 200 and response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
            logger.error("LibreTranslate HTTP %s: %s", response.status_code, response.text[:200])
            return None
        except Exception as exc:
            logger.error("LibreTranslate request error: %s", exc)
            return None

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            payload = {
                "q": text,
                "source": source_lang.lower(),
                "target": target_lang.lower(),
                "format": "html"
            }
            result = await self._post_translate(payload)
            if result and isinstance(result, dict):
                return result.get("translatedText", text)
            return text
        except Exception as e:
            logger.error(f"LibreTranslate translation error: {e}")
            return text

    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        # LibreTranslate supports multiple 'q' values in one request; use when >1 texts
        texts_clean = [t or " " for t in texts]
        payload = {
            "q": texts_clean,
            "source": source_lang.lower(),
            "target": target_lang.lower(),
            "format": "html",
        }
        result = await self._post_translate(payload)
        if result and isinstance(result, list):
            # Expected list of objects with translatedText
            out = []
            for orig, item in zip(texts, result):
                if isinstance(item, dict):
                    out.append(item.get("translatedText", orig))
                else:
                    out.append(orig)
            return out
        # Fallback to per-text translation on failure
        results = []
        for txt in texts:
            translated = await self.translate_text(txt, source_lang, target_lang)
            results.append(translated)
            await asyncio.sleep(TRANSLATION_DELAY)
        return results


class LaraTranslator(BaseTranslator):
    """Translator using Lara SDK"""

    def __init__(self):
        access_id = os.getenv("LARA_ACCESS_KEY_ID")
        access_secret = os.getenv("LARA_ACCESS_KEY_SECRET")
        if not access_id or not access_secret:
            raise ValueError("LARA_ACCESS_KEY_ID or SECRET not set")
        creds = LaraCredentials(access_key_id=access_id, access_key_secret=access_secret)
        self.client = LaraClient(creds)

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            result = await asyncio.to_thread(self.client.translate, text, source=source_lang, target=target_lang)
            return result.translation if hasattr(result, 'translation') else text
        except Exception as exc:
            logger.error("Lara translate error: %s", exc)
            return text

    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        results: List[str] = []
        for t in texts:
            results.append(await self.translate_text(t, source_lang, target_lang))
            await asyncio.sleep(TRANSLATION_DELAY)
        return results

    

    
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        # openai>=1.0.0 provides OpenAI() client class; older versions use module-level functions
        self._is_v1 = hasattr(openai, "OpenAI") and callable(getattr(openai, "OpenAI"))

        if self._is_v1:
            # v1 client style
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # legacy style
            openai.api_key = api_key

    async def _request_completion(self, prompt_messages: List[dict]) -> str:
        """Internal helper to request a chat completion and return text only."""
        if self._is_v1:
            # new client style
            resp = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=prompt_messages,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        else:
            # legacy module style
            resp = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.model,
                messages=prompt_messages,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            system_prompt = (
                "You are a translation engine. Translate the following text from "
                f"{source_lang} to {target_lang}. Return only the translated text without any extra commentary."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ]
            return await self._request_completion(messages)
        except Exception as exc:
            logger.error(": %s", exc)
            return text

    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        results: List[str] = []
        for txt in texts:
            if not txt.strip():
                results.append(txt)
                continue
            results.append(await self.translate_text(txt, source_lang, target_lang))
            await asyncio.sleep(TRANSLATION_DELAY)
        return results

    

    



    """Free MyMemory translation API (limited)"""

    def _chunk_text(self, text: str, max_len: int = 450) -> List[str]:
        """Splits long text into <= max_len character chunks on whitespace."""
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > max_len and current:
                chunks.append(" ".join(current))
                current = [word]
                current_len = len(word) + 1
            else:
                current.append(word)
                current_len += len(word) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

    """Free MyMemory translation API (limited)"""

    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://api.mymemory.translated.net/get"
        self.session.headers.update({
            "User-Agent": "MyMemory-Client/1.0",
            "Accept": "application/json",
        })

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            # MyMemory free endpoint truncates >500 chars; chunk if needed
            if len(text) > 450:
                chunks = self._chunk_text(text)
                translated_parts: List[str] = []
                for part in chunks:
                    translated_parts.append(await self.translate_text(part, source_lang, target_lang))
                    await asyncio.sleep(TRANSLATION_DELAY)
                return " ".join(translated_parts)

            params = {
                "q": text,
                "langpair": f"{source_lang.lower()}|{target_lang.lower()}",
            }
            response = await asyncio.to_thread(self.session.get, self.base_url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                translated = data.get("responseData", {}).get("translatedText")
                return translated if translated else text
            logger.error("MyMemory HTTP %s: %s", response.status_code, response.text[:100])
            return text
        except Exception as exc:
            logger.error(": %s", exc)
            return text

    async def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        results: List[str] = []
        for txt in texts:
            results.append(await self.translate_text(txt, source_lang, target_lang))
            await asyncio.sleep(TRANSLATION_DELAY)
        return results


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
            deeplx_key = os.getenv('DEEPLX_API_KEY')
            if not deeplx_endpoint and deeplx_key:
                deeplx_endpoint = "https://deeplx.missuo.ru/translate"
            if deeplx_endpoint or deeplx_key:
                self.providers['deeplx'] = DeepLXTranslator(deeplx_endpoint, api_key=deeplx_key)
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
        
        # LibreTranslate
        try:
            libre_endpoint = os.getenv('LIBRE_TRANSLATE_ENDPOINT', 'https://libretranslate.de/translate')
            # Always register libretranslate as fallback provider
            self.providers['libre'] = LibreTranslateTranslator(libre_endpoint)
            logger.info("LibreTranslate translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LibreTranslate: {e}")

        # Lara
        try:
            self.providers['lara'] = LaraTranslator()
            logger.info("Lara translator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Lara: {e}")



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
        """Select the best available provider respecting env override.

        Priority:
        1. PREFERRED_TRANSLATOR env var if available.
        2. DeepLX > DeepL > Google > Azure
        """
        preferred = os.getenv("PREFERRED_TRANSLATOR")
        if preferred and preferred.lower() in self.providers:
            return preferred.lower()

        priority_order = ['lara', 'libre', 'deeplx', 'deepl', 'google', 'azure']
        
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