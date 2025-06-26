"""Configuration settings for the PDF translator with enhanced image handling."""

import os
from dotenv import load_dotenv
import json
import logging
from pathlib import Path

load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    # Translation API settings
    DEEPL_API_KEY = os.getenv('DEEPL_API_KEY')
    GOOGLE_CLOUD_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    AZURE_TRANSLATOR_KEY = os.getenv('AZURE_TRANSLATOR_KEY')
    AZURE_TRANSLATOR_REGION = os.getenv('AZURE_TRANSLATOR_REGION', 'global')
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        # Check if at least one translation service is configured
        translation_services = [
            cls.DEEPL_API_KEY,
            cls.GOOGLE_CLOUD_CREDENTIALS,
            cls.AZURE_TRANSLATOR_KEY
        ]
        
        if not any(translation_services):
            errors.append("At least one translation service must be configured")
        
        if cls.GOOGLE_CLOUD_CREDENTIALS and not os.path.exists(cls.GOOGLE_CLOUD_CREDENTIALS):
            errors.append(f"Google credentials file not found: {cls.GOOGLE_CLOUD_CREDENTIALS}")
        
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"- {error}" for error in errors))
    # Processing settings
    CHUNK_SIZE = 25  # Reduced for image-heavy documents
    MAX_TEXT_LENGTH = 4000  # Reduced for better translation quality
    MEMORY_THRESHOLD = 0.75  # Slightly lower threshold for image processing
    
    # Image processing settings
    PRESERVE_IMAGE_QUALITY = True
    MAX_IMAGE_SIZE_MB = 50  # Maximum size per image in MB
    IMAGE_COMPRESSION_QUALITY = 95  # PNG compression quality (0-100)
    MAINTAIN_ASPECT_RATIO = True
    
    # Text flow settings
    TEXT_WRAP_AROUND_IMAGES = True
    MINIMUM_TEXT_SPACING = 2.0  # Minimum spacing between text and images
    PRESERVE_LINE_BREAKS = True
    MAINTAIN_PARAGRAPH_STRUCTURE = True
    
    # File paths
    INPUT_DIR = os.getenv('INPUT_DIR', 'input')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')
    TEMP_DIR = os.getenv('TEMP_DIR', 'temp')
    IMAGE_CACHE_DIR = os.getenv('IMAGE_CACHE_DIR', 'temp/images')
    
    # Translation settings
    SOURCE_LANGUAGE = "DE"
    TARGET_LANGUAGE = "EN"
    
    # Load Klages terminology dictionary from external JSON for easier maintenance
    _TERMINOLOGY_FILE = Path(__file__).parent / 'klages_terminology.json'
    try:
        with _TERMINOLOGY_FILE.open('r', encoding='utf-8') as f:
            KLAGES_TERMINOLOGY = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Klages terminology file not found: {_TERMINOLOGY_FILE}")
        KLAGES_TERMINOLOGY = {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing Klages terminology JSON: {e}")
        KLAGES_TERMINOLOGY = {}
    
    # Quality assurance settings
    VALIDATE_IMAGE_POSITIONING = True
    CHECK_TEXT_OVERLAP = True
    VERIFY_FONT_RENDERING = True
    MAINTAIN_READING_ORDER = True