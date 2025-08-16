"""Configuration settings for Dolphin OCR Translate.

Enhanced image handling and parallel processing.
"""

import json
import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional at runtime
    def load_dotenv() -> None:  # type: ignore[misc]
        return None

load_dotenv()

logger = logging.getLogger(__name__)


def _parse_int_env(
    var_name: str, default_value: int, min_value: int | None = None
) -> int:
    """Parse an int env var with optional minimum clamp and default fallback."""
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default_value if min_value is None else max(min_value, default_value)
    try:
        parsed_value = int(raw_value)
    except ValueError as exc:
        logger.error("Invalid %s: %s; default %s", var_name, exc, default_value)
        return default_value if min_value is None else max(min_value, default_value)
    if min_value is not None:
        parsed_value = max(min_value, parsed_value)
    return parsed_value


def _parse_float_env(
    var_name: str, default_value: float, min_value: float | None = None
) -> float:
    """Parse a float env var with optional minimum clamp and default fallback."""
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default_value if min_value is None else max(min_value, default_value)
    try:
        parsed_value = float(raw_value)
    except ValueError as exc:
        logger.error("Invalid %s: %s; default %s", var_name, exc, default_value)
        return default_value if min_value is None else max(min_value, default_value)
    if min_value is not None:
        parsed_value = max(min_value, parsed_value)
    return parsed_value


class Config:
    """Central configuration class for Dolphin OCR Translate application.

    This class loads configuration from environment variables and provides
    validated constants for the entire application. All settings are loaded
    as class attributes and can be accessed directly.

    Environment Variables Read:
        Translation API:
            LINGO_API_KEY: Required API key for Lingo.dev translation service

        File Paths:
            INPUT_DIR: Input directory path (default: "input")
            OUTPUT_DIR: Output directory path (default: "output")
            TEMP_DIR: Temporary files directory (default: "temp")
            IMAGE_CACHE_DIR: Image cache directory (default: "temp/images")

        Translation Settings:
            SOURCE_LANGUAGE: Source language code (default: "DE")
            TARGET_LANGUAGE: Target language code (default: "EN")

        PDF Processing:
            PDF_DPI: PDF rendering DPI, 72-600 (default: 300)
            MEMORY_THRESHOLD_MB: Memory threshold in MB, 100-10000
                (default: 500)
            TRANSLATION_DELAY: Delay between requests in seconds
                (default: 0.1)
            PRESERVE_IMAGES: Whether to preserve images, true/false
                (default: true)

        Parallel Processing:
            MAX_CONCURRENT_REQUESTS: Max concurrent requests, min 1
                (default: 10)
            MAX_REQUESTS_PER_SECOND: Rate limit, min 0.1 (default: 5.0)
            TRANSLATION_BATCH_SIZE: Batch size, min 1 (default: 50)
            TRANSLATION_MAX_RETRIES: Max retries, min 0 (default: 3)
            TRANSLATION_REQUEST_TIMEOUT: Timeout in seconds
                (default: 30.0)
            PARALLEL_PROCESSING_THRESHOLD: Min items for parallel processing
                (default: 5)

    Usage:
        # Access configuration constants directly
        api_key = Config.LINGO_API_KEY
        dpi = Config.PDF_DPI

        # Validate configuration before use
        if not Config.validate_config():
            raise RuntimeError("Invalid configuration")

        # Check available providers
        providers = Config.get_available_providers()

    Methods:
        validate_config(): Validates all configuration settings
            and creates directories
        get_available_providers(): Returns list of available
            translation providers

    Note:
        Configuration is loaded once when the module is imported. Environment
        variables should be set before importing this module. Invalid values
        are replaced with safe defaults and logged as errors.
    """

    # Translation API settings - Only Lingo.dev
    LINGO_API_KEY: str | None = os.getenv("LINGO_API_KEY")

    # File paths
    INPUT_DIR: str = os.getenv("INPUT_DIR", "input")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    IMAGE_CACHE_DIR: str = os.getenv("IMAGE_CACHE_DIR", "temp/images")

    # Translation settings
    SOURCE_LANGUAGE: str = os.getenv("SOURCE_LANGUAGE", "DE")
    TARGET_LANGUAGE: str = os.getenv("TARGET_LANGUAGE", "EN")

    # Load Klages terminology dictionary from external JSON for
    # easier maintenance
    _TERMINOLOGY_FILE = Path(__file__).parent / "klages_terminology.json"
    KLAGES_TERMINOLOGY: dict[str, str] = {}
    try:
        with _TERMINOLOGY_FILE.open("r", encoding="utf-8") as f:
            KLAGES_TERMINOLOGY = json.load(f)
    except FileNotFoundError:
        logger.warning(
            "Klages terminology file not found: %s", _TERMINOLOGY_FILE
        )
    except json.JSONDecodeError as e:
        logger.error(
            "Error parsing Klages terminology JSON at '%s': %s",
            _TERMINOLOGY_FILE,
            e,
        )

    # PDF processing settings
    # Parse independently so one bad value does not reset others
    PDF_DPI: int = _parse_int_env("PDF_DPI", 300, 72)
    MEMORY_THRESHOLD_MB: int = _parse_int_env("MEMORY_THRESHOLD_MB", 500, 100)
    TRANSLATION_DELAY: float = _parse_float_env("TRANSLATION_DELAY", 0.1, 0.0)

    PRESERVE_IMAGES: bool = os.getenv("PRESERVE_IMAGES", "true").lower() == "true"

    # Parallel processing settings
    # Parallel processing settings - parse independently as well
    MAX_CONCURRENT_REQUESTS: int = _parse_int_env(
        "MAX_CONCURRENT_REQUESTS", 10, 1
    )
    MAX_REQUESTS_PER_SECOND: float = _parse_float_env(
        "MAX_REQUESTS_PER_SECOND", 5.0, 0.1
    )
    TRANSLATION_BATCH_SIZE: int = _parse_int_env("TRANSLATION_BATCH_SIZE", 50, 1)
    TRANSLATION_MAX_RETRIES: int = _parse_int_env("TRANSLATION_MAX_RETRIES", 3, 0)
    TRANSLATION_REQUEST_TIMEOUT: float = _parse_float_env(
        "TRANSLATION_REQUEST_TIMEOUT", 30.0, 1.0
    )
    PARALLEL_PROCESSING_THRESHOLD: int = _parse_int_env(
        "PARALLEL_PROCESSING_THRESHOLD", 5, 1
    )

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
                        "Cannot create or access %s '%s': %s",
                        dir_name,
                        dir_path,
                        e,
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
                f"SOURCE_LANGUAGE must be a 2-letter code, "
                f"got: {cls.SOURCE_LANGUAGE}"
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
        """Get list of available translation providers."""
        providers = []

        if cls.LINGO_API_KEY:
            providers.append("lingo")

        return providers
