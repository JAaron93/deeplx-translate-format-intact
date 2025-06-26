"""Core services package containing document processing, translation, and language detection functionality."""

from .translation_service import TranslationService
from .language_detector import LanguageDetector
from .enhanced_document_processor import EnhancedDocumentProcessor

__all__ = [
    'TranslationService',
    'LanguageDetector',
    'EnhancedDocumentProcessor'
]