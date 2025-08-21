"""Configuration settings for Dolphin OCR Translate.

Enhanced image handling and parallel processing.
"""

import json
import logging
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, TypeVar

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at runtime
    # Fallback no-op with signature compatible to python-dotenv
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        """Fallback no-op function when python-dotenv is not available."""
        return False


load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T", int, float)

# Boolean environment variable parsing constants
TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
FALSY: frozenset[str] = frozenset({"0", "false", "no", "off"})


def _parse_env(
    var_name: str,
    default_value: T,
    coerce: Callable[[str], T],
    min_value: T | None = None,
) -> T:
    """Parse an environment variable with a coerce function and optional minimum clamp."""
    fallback_value: T = (
        max(min_value, default_value) if min_value is not None else default_value
    )
    raw_value: str | None = os.getenv(var_name)
    if raw_value is None:
        logger.debug("%s not set; using fallback %s", var_name, fallback_value)
        return fallback_value

    try:
        parsed_value: T = coerce(raw_value)
    except (ValueError, TypeError):
        logger.error(
            "Invalid %s=%r; falling back to %s",
            var_name,
            raw_value,
            fallback_value,
            exc_info=False,
        )
        return fallback_value

    if min_value is not None:
        parsed_value = max(min_value, parsed_value)
    return parsed_value


def _parse_bool_env(var_name: str, default_value: bool) -> bool:
    """Parse a boolean environment variable with explicit truthy/falsy handling.

    Returns True only for explicit truthy values, False only for explicit
    falsy values, and the default_value for any unrecognized values
    (including None).
    """
    raw_value: str | None = os.getenv(var_name)
    if raw_value is None:
        return default_value

    # Normalize the value to lowercase and strip whitespace
    normalized_value: str = raw_value.lower().strip()

    # Return True only for explicit truthy values
    if normalized_value in TRUTHY:
        return True
    # Return False only for explicit falsy values
    elif normalized_value in FALSY:
        return False
    # Return default_value for any unrecognized values
    else:
        return default_value


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
    _TERMINOLOGY_FILE: Path = Path(__file__).parent / "klages_terminology.json"
    KLAGES_TERMINOLOGY: Mapping[str, str] = MappingProxyType({})
    try:
        with _TERMINOLOGY_FILE.open("r", encoding="utf-8") as f:
            _raw: object = json.load(f)
        if isinstance(_raw, dict) and all(
            isinstance(k, str) and isinstance(v, str) for k, v in _raw.items()
        ):
            KLAGES_TERMINOLOGY = MappingProxyType(_raw)
            logger.info(
                "Loaded %d terminology entries from %s",
                len(KLAGES_TERMINOLOGY),
                _TERMINOLOGY_FILE,
            )
        else:
            logger.error(
                "Invalid Klages terminology structure in '%s': "
                "expected mapping[str, str], got: %s",
                _TERMINOLOGY_FILE,
                type(_raw).__name__,
            )
            KLAGES_TERMINOLOGY = MappingProxyType({})
    except FileNotFoundError:
        logger.warning("Klages terminology file not found: %s", _TERMINOLOGY_FILE)
    except json.JSONDecodeError as e:
        logger.error(
            "Error parsing Klages terminology JSON at '%s': %s",
            _TERMINOLOGY_FILE,
            e,
        )

    # PDF processing settings
    # Parse independently so one bad value does not reset others
    PDF_DPI: int = _parse_env("PDF_DPI", 300, int, 72)
    MEMORY_THRESHOLD_MB: int = _parse_env("MEMORY_THRESHOLD_MB", 500, int, 100)
    TRANSLATION_DELAY: float = _parse_env("TRANSLATION_DELAY", 0.1, float, 0.0)

    PRESERVE_IMAGES: bool = _parse_bool_env("PRESERVE_IMAGES", True)

    # Parallel processing settings
    # Parallel processing settings - parse independently as well
    MAX_CONCURRENT_REQUESTS: int = _parse_env("MAX_CONCURRENT_REQUESTS", 10, int, 1)
    MAX_REQUESTS_PER_SECOND: float = _parse_env(
        "MAX_REQUESTS_PER_SECOND", 5.0, float, 0.1
    )
    TRANSLATION_BATCH_SIZE: int = _parse_env("TRANSLATION_BATCH_SIZE", 50, int, 1)
    TRANSLATION_MAX_RETRIES: int = _parse_env("TRANSLATION_MAX_RETRIES", 3, int, 0)
    TRANSLATION_REQUEST_TIMEOUT: float = _parse_env(
        "TRANSLATION_REQUEST_TIMEOUT", 30.0, float, 1.0
    )
    PARALLEL_PROCESSING_THRESHOLD: int = _parse_env(
        "PARALLEL_PROCESSING_THRESHOLD", 5, int, 1
    )

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present and valid.

        Returns:
            bool: True if all validations pass, False otherwise
        """
        validation_passed: bool = True

        # Validate API key
        if not cls.LINGO_API_KEY:
            logger.error("LINGO_API_KEY is required but not configured")
            validation_passed = False
        elif len(cls.LINGO_API_KEY.strip()) < 10:  # Basic length check
            logger.error("LINGO_API_KEY appears to be invalid (too short)")
            validation_passed = False

        # Validate directory paths
        required_dirs: dict[str, str] = {
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
        numeric_validations: dict[
            str, tuple[int | float, int | float, int | float, str]
        ] = {
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
            "MAX_CONCURRENT_REQUESTS": (
                cls.MAX_CONCURRENT_REQUESTS,
                1,
                100,
                "Max concurrent requests must be between 1 and 100",
            ),
            "MAX_REQUESTS_PER_SECOND": (
                cls.MAX_REQUESTS_PER_SECOND,
                0.1,
                100.0,
                "Max requests per second must be between 0.1 and 100.0",
            ),
            "TRANSLATION_BATCH_SIZE": (
                cls.TRANSLATION_BATCH_SIZE,
                1,
                1000,
                "Translation batch size must be between 1 and 1000",
            ),
            "TRANSLATION_MAX_RETRIES": (
                cls.TRANSLATION_MAX_RETRIES,
                0,
                10,
                "Translation max retries must be between 0 and 10",
            ),
            "TRANSLATION_REQUEST_TIMEOUT": (
                cls.TRANSLATION_REQUEST_TIMEOUT,
                1.0,
                300.0,
                "Translation request timeout must be between 1.0 and 300.0 seconds",
            ),
            "PARALLEL_PROCESSING_THRESHOLD": (
                cls.PARALLEL_PROCESSING_THRESHOLD,
                1,
                100,
                "Parallel processing threshold must be between 1 and 100",
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
        # KLAGES_TERMINOLOGY is intentionally immutable via MappingProxyType
        if not isinstance(cls.KLAGES_TERMINOLOGY, Mapping):
            logger.error(
                f"KLAGES_TERMINOLOGY must be a mapping, "
                f"got {type(cls.KLAGES_TERMINOLOGY)}"
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
        providers: list[str] = []

        if cls.LINGO_API_KEY:
            providers.append("lingo")

        return providers
