"""Translation services for converting German text to English."""

import requests
import time
import json
import re
import os
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import deepl
from google.cloud import translate_v2 as translate
from config import Config

class TranslationService(ABC):
    """Abstract base class for translation services."""
    
    @abstractmethod
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        pass
    
    @abstractmethod
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        pass

class BaseKlagesTranslator(TranslationService):
    """Base translator with Klages-specific preprocessing helpers."""
    def __init__(self):
        self.config = Config()

    def _preprocess_klages_terminology(self, text: str) -> str:
        processed_text = text
        for german_term, english_term in self.config.KLAGES_TERMINOLOGY.items():
            pattern = r'\b' + re.escape(german_term) + r'\b'
            processed_text = re.sub(pattern, f"[KLAGES_TERM:{english_term}]", processed_text)
        return processed_text

    def _split_text(self, text: str, max_length: int) -> List[str]:
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

class DeepLTranslator(BaseKlagesTranslator):
    """DeepL translation service implementation."""
    
    def __init__(self, api_key: str):
        super().__init__()
        if not api_key:
            raise ValueError("DeepL API key is required")
        self.translator = deepl.Translator(api_key)
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate single text using DeepL."""
        try:
            # Apply Klages-specific terminology preprocessing
            preprocessed_text = self._preprocess_klages_terminology(text)
            
            result = self.translator.translate_text(
                preprocessed_text,
                source_lang=source_lang,
                target_lang=target_lang,
                preserve_formatting=True
            )
            
            return result.text
        except Exception as e:
            print(f"DeepL translation error: {e}")
            return text  # Return original text on error
    
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate multiple texts using DeepL."""
        translated_texts = []
        
        for text in texts:
            if len(text.strip()) == 0:
                translated_texts.append(text)
                continue
            
            # Split long texts to respect API limits
            if len(text) > self.config.MAX_TEXT_LENGTH:
                chunks = self._split_text(text, self.config.MAX_TEXT_LENGTH)
                translated_chunks = []
                for chunk in chunks:
                    translated_chunk = self.translate_text(chunk, source_lang, target_lang)
                    translated_chunks.append(translated_chunk)
                translated_texts.append(' '.join(translated_chunks))
            else:
                translated_text = self.translate_text(text, source_lang, target_lang)
                translated_texts.append(translated_text)
            
            # Rate limiting
            time.sleep(0.1)
        
        return translated_texts
    
    def _preprocess_klages_terminology(self, text: str) -> str:
        """Apply Klages-specific terminology preprocessing."""
        processed_text = text
        for german_term, english_term in self.config.KLAGES_TERMINOLOGY.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(german_term) + r'\b'
            processed_text = re.sub(pattern, f"[KLAGES_TERM:{english_term}]", processed_text)
        return processed_text
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks while preserving sentence boundaries."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class DeepLXTranslator(BaseKlagesTranslator):
    """DeepLX HTTP translation service implementation."""
    
    def __init__(self, endpoint: str):
        super().__init__()
        if not endpoint:
            raise ValueError("DeepLX endpoint is required")
        self.endpoint = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'DeepLX-Client/1.0'
        })
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate single text using DeepLX."""
        try:
            # Apply Klages-specific terminology preprocessing
            preprocessed_text = self._preprocess_klages_terminology(text)
            
            payload = {
                "text": preprocessed_text,
                "source_lang": source_lang.upper(),
                "target_lang": target_lang.upper()
            }
            
            response = self.session.post(
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
                    print(f"Unexpected DeepLX response format: {result}")
                    return text
            else:
                print(f"DeepLX HTTP error {response.status_code}: {response.text}")
                return text
                
        except Exception as e:
            print(f"DeepLX translation error: {e}")
            return text
    
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate multiple texts using DeepLX."""
        translated_texts = []
        
        for text in texts:
            if len(text.strip()) == 0:
                translated_texts.append(text)
                continue
            
            # Split long texts to respect API limits
            if len(text) > self.config.MAX_TEXT_LENGTH:
                chunks = self._split_text(text, self.config.MAX_TEXT_LENGTH)
                translated_chunks = []
                for chunk in chunks:
                    translated_chunk = self.translate_text(chunk, source_lang, target_lang)
                    translated_chunks.append(translated_chunk)
                translated_texts.append(' '.join(translated_chunks))
            else:
                translated_text = self.translate_text(text, source_lang, target_lang)
                translated_texts.append(translated_text)
            
            # Rate limiting
            time.sleep(0.1)
        
        return translated_texts
    
    def _preprocess_klages_terminology(self, text: str) -> str:
        """Apply Klages-specific terminology preprocessing."""
        processed_text = text
        for german_term, english_term in self.config.KLAGES_TERMINOLOGY.items():
            # Use word boundaries to avoid partial matches
            import re
            pattern = r'\b' + re.escape(german_term) + r'\b'
            processed_text = re.sub(pattern, f"[KLAGES_TERM:{english_term}]", processed_text)
        return processed_text
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks while preserving sentence boundaries."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

class GoogleTranslator(TranslationService):
    """Google Cloud Translation service implementation."""
    
    def __init__(self):
        self.client = translate.Client()
        self.config = Config()
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate single text using Google Translate."""
        try:
            result = self.client.translate(
                text,
                source_language=source_lang.lower(),
                target_language=target_lang.lower(),
                format_='text'
            )
            return result['translatedText']
        except Exception as e:
            print(f"Google Translate error: {e}")
            return text
    
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate multiple texts using Google Translate."""
        try:
            results = self.client.translate(
                texts,
                source_language=source_lang.lower(),
                target_language=target_lang.lower(),
                format_='text'
            )
            return [result['translatedText'] for result in results]
        except Exception as e:
            print(f"Google Translate batch error: {e}")
            return texts

class AzureTranslator(TranslationService):
    """Azure Cognitive Services Translator implementation."""
    
    def __init__(self, api_key: str, region: str):
        if not api_key:
            raise ValueError("Azure Translator API key is required")
        self.api_key = api_key
        self.region = region
        self.endpoint = "https://api.cognitive.microsofttranslator.com"
        self.config = Config()
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate single text using Azure Translator."""
        try:
            path = '/translate'
            constructed_url = self.endpoint + path
            
            params = {
                'api-version': '3.0',
                'from': source_lang.lower(),
                'to': target_lang.lower()
            }
            
            headers = {
                'Ocp-Apim-Subscription-Key': self.api_key,
                'Ocp-Apim-Subscription-Region': self.region,
                'Content-type': 'application/json'
            }
            
            body = [{'text': text}]
            
            response = requests.post(constructed_url, params=params, headers=headers, json=body)
            response.raise_for_status()
            
            result = response.json()
            return result[0]['translations'][0]['text']
        except Exception as e:
            print(f"Azure Translator error: {e}")
            return text
    
    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translate multiple texts using Azure Translator."""
        translated_texts = []
        
        # Azure has a limit on batch size, so process in smaller batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch:
                translated_text = self.translate_text(text, source_lang, target_lang)
                batch_results.append(translated_text)
            
            translated_texts.extend(batch_results)
            time.sleep(0.1)  # Rate limiting
        
        return translated_texts

class TranslatorFactory:
    """Factory for creating translation service instances."""
    
    @staticmethod
    def create_translator(service_type: str = "deepl") -> TranslationService:
        """Create a translator instance based on service type."""
        config = Config()
        
        if service_type.lower() == "deepl":
            if not config.DEEPL_API_KEY:
                raise ValueError("DeepL API key not found in environment variables")
            return DeepLTranslator(config.DEEPL_API_KEY)
        
        elif service_type.lower() == "deeplx":
            deeplx_endpoint = os.getenv('DEEPLX_ENDPOINT')
            if not deeplx_endpoint:
                raise ValueError("DeepLX endpoint not found in environment variables")
            return DeepLXTranslator(deeplx_endpoint)
        
        elif service_type.lower() == "google":
            if not config.GOOGLE_CLOUD_CREDENTIALS:
                raise ValueError("Google Cloud credentials not found in environment variables")
            return GoogleTranslator()
        
        elif service_type.lower() == "azure":
            if not config.AZURE_TRANSLATOR_KEY:
                raise ValueError("Azure Translator API key not found in environment variables")
            return AzureTranslator(config.AZURE_TRANSLATOR_KEY, config.AZURE_TRANSLATOR_REGION)
        
        else:
            raise ValueError(f"Unsupported translation service: {service_type}")