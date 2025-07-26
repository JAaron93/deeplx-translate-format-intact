"""Configuration settings for Dolphin OCR Translate with enhanced image handling and parallel processing."""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    # Translation API settings - Only Lingo.dev
    LINGO_API_KEY = os.getenv("LINGO_API_KEY")

    # File paths
    INPUT_DIR = os.getenv("INPUT_DIR", "input")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
    TEMP_DIR = os.getenv("TEMP_DIR", "temp")
    IMAGE_CACHE_DIR = os.getenv("IMAGE_CACHE_DIR", "temp/images")

    # Translation settings
    SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "DE")
    TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "EN")

    # Load Klages terminology dictionary from external JSON for easier maintenance
    _TERMINOLOGY_FILE = Path(__file__).parent / "klages_terminology.json"
    try:
        with _TERMINOLOGY_FILE.open("r", encoding="utf-8") as f:
            KLAGES_TERMINOLOGY = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Klages terminology file not found: {_TERMINOLOGY_FILE}")
        KLAGES_TERMINOLOGY = {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing Klages terminology JSON: {e}")
        KLAGES_TERMINOLOGY = {}

    # PDF processing settings
    try:
        PDF_DPI = max(72, int(os.getenv("PDF_DPI", "300")))  # Minimum 72 DPI
        MEMORY_THRESHOLD_MB = max(
            100, int(os.getenv("MEMORY_THRESHOLD_MB", "500"))
        )  # MB, min 100
        TRANSLATION_DELAY = max(0.0, float(os.getenv("TRANSLATION_DELAY", "0.1")))
    except ValueError as e:
        logger.error(f"Invalid configuration value: {e}")
        # Set default values if parsing fails
        PDF_DPI = 300
        MEMORY_THRESHOLD_MB = 500
        TRANSLATION_DELAY = 0.1

    PRESERVE_IMAGES = os.getenv("PRESERVE_IMAGES", "true").lower() == "true"

    # Parallel processing settings
    try:
        MAX_CONCURRENT_REQUESTS = max(
            1, int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        )
        MAX_REQUESTS_PER_SECOND = max(
            0.1, float(os.getenv("MAX_REQUESTS_PER_SECOND", "5.0"))
        )
        TRANSLATION_BATCH_SIZE = max(1, int(os.getenv("TRANSLATION_BATCH_SIZE", "50")))
        TRANSLATION_MAX_RETRIES = max(0, int(os.getenv("TRANSLATION_MAX_RETRIES", "3")))
        TRANSLATION_REQUEST_TIMEOUT = max(
            1.0, float(os.getenv("TRANSLATION_REQUEST_TIMEOUT", "30.0"))
        )
        PARALLEL_PROCESSING_THRESHOLD = max(
            1, int(os.getenv("PARALLEL_PROCESSING_THRESHOLD", "5"))
        )
    except ValueError as e:
        logger.error(f"Invalid parallel processing configuration value: {e}")
        # Set default values if parsing fails
        MAX_CONCURRENT_REQUESTS = 10
        MAX_REQUESTS_PER_SECOND = 5.0
        TRANSLATION_BATCH_SIZE = 50
        TRANSLATION_MAX_RETRIES = 3
        TRANSLATION_REQUEST_TIMEOUT = 30.0
        PARALLEL_PROCESSING_THRESHOLD = 5

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present and valid.

        Returns:
            bool: True if all validations pass, False otherwise
        """
        validation_passed = True

        # Validate API key
        if not cls.LINGO_API_KEY:
            logger.error("LINGO_API_KEY is required but not configured")
            validation_passed = False
        elif len(cls.LINGO_API_KEY.strip()) < 10:  # Basic length check
            logger.error("LINGO_API_KEY appears to be invalid (too short)")
            validation_passed = False

        # Validate directory paths
        required_dirs = {
            "INPUT_DIR": cls.INPUT_DIR,
            "OUTPUT_DIR": cls.OUTPUT_DIR,
            "TEMP_DIR": cls.TEMP_DIR,
            "IMAGE_CACHE_DIR": cls.IMAGE_CACHE_DIR,
        }

        for dir_name, dir_path in required_dirs.items():
            if not dir_path:
                logger.error(f"{dir_name} is required but not configured")
                validation_passed = False
            elif not isinstance(dir_path, str):
                logger.error(f"{dir_name} must be a string, got {type(dir_path)}")
                validation_passed = False
            else:
                # Try to create directory if it doesn't exist
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                except (OSError, PermissionError, ValueError) as e:
                    logger.error(
                        f"Cannot create or access {dir_name} '{dir_path}': {e}"
                    )
                    validation_passed = False

        # Validate numeric settings
        numeric_validations = {
            "PDF_DPI": (cls.PDF_DPI, 72, 600, "DPI must be between 72 and 600"),
            "MEMORY_THRESHOLD_MB": (
                cls.MEMORY_THRESHOLD_MB,
                100,
                10000,
                "Memory threshold must be between 100MB and 10GB",
            ),
            "TRANSLATION_DELAY": (
                cls.TRANSLATION_DELAY,
                0.0,
                10.0,
                "Translation delay must be between 0 and 10 seconds",
            ),
        }

        for setting_name, (
            value,
            min_val,
            max_val,
            error_msg,
        ) in numeric_validations.items():
            if not isinstance(value, (int, float)):
                logger.error(f"{setting_name} must be numeric, got {type(value)}")
                validation_passed = False
            elif not (min_val <= value <= max_val):
                logger.error(
                    f"{setting_name} validation failed: {error_msg}. Got: {value}"
                )
                validation_passed = False

        # Validate language settings
        if not cls.SOURCE_LANGUAGE or not isinstance(cls.SOURCE_LANGUAGE, str):
            logger.error("SOURCE_LANGUAGE is required and must be a string")
            validation_passed = False
        elif len(cls.SOURCE_LANGUAGE) != 2:
            logger.error(
                f"SOURCE_LANGUAGE must be a 2-letter code, got: {cls.SOURCE_LANGUAGE}"
            )
            validation_passed = False

        if not cls.TARGET_LANGUAGE or not isinstance(cls.TARGET_LANGUAGE, str):
            logger.error("TARGET_LANGUAGE is required and must be a string")
            validation_passed = False
        elif len(cls.TARGET_LANGUAGE) != 2:
            logger.error(
                f"TARGET_LANGUAGE must be a 2-letter code, got: {cls.TARGET_LANGUAGE}"
            )
            validation_passed = False

        # Validate boolean settings
        if not isinstance(cls.PRESERVE_IMAGES, bool):
            logger.error(
                f"PRESERVE_IMAGES must be boolean, got {type(cls.PRESERVE_IMAGES)}"
            )
            validation_passed = False

        # Validate terminology dictionary
        if not isinstance(cls.KLAGES_TERMINOLOGY, dict):
            logger.error(
                f"KLAGES_TERMINOLOGY must be a dictionary, got {type(cls.KLAGES_TERMINOLOGY)}"
            )
            validation_passed = False

        # Log validation result
        if validation_passed:
            logger.info("Configuration validation passed successfully")
        else:
            logger.error("Configuration validation failed - check the errors above")

        return validation_passed

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available translation providers"""
        providers = []

        if cls.LINGO_API_KEY:
            providers.append("lingo")

        return providers
