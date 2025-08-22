"""Configuration settings for the PDF translator."""

import logging
import os
import secrets

from dotenv import load_dotenv

load_dotenv()

logger: logging.Logger = logging.getLogger(__name__)

# Valid ISO 639-1 language codes
VALID_LANGUAGE_CODES: set[str] = {
    "AA",
    "AB",
    "AE",
    "AF",
    "AK",
    "AM",
    "AN",
    "AR",
    "AS",
    "AV",
    "AY",
    "AZ",
    "BA",
    "BE",
    "BG",
    "BH",
    "BI",
    "BM",
    "BN",
    "BO",
    "BR",
    "BS",
    "CA",
    "CE",
    "CH",
    "CO",
    "CR",
    "CS",
    "CU",
    "CV",
    "CY",
    "DA",
    "DE",
    "DV",
    "DZ",
    "EE",
    "EL",
    "EN",
    "EO",
    "ES",
    "ET",
    "EU",
    "FA",
    "FF",
    "FI",
    "FJ",
    "FO",
    "FR",
    "FY",
    "GA",
    "GD",
    "GL",
    "GN",
    "GU",
    "GV",
    "HA",
    "HE",
    "HI",
    "HO",
    "HR",
    "HT",
    "HU",
    "HY",
    "HZ",
    "IA",
    "ID",
    "IE",
    "IG",
    "II",
    "IK",
    "IO",
    "IS",
    "IT",
    "IU",
    "JA",
    "JV",
    "KA",
    "KG",
    "KI",
    "KJ",
    "KK",
    "KL",
    "KM",
    "KN",
    "KO",
    "KR",
    "KS",
    "KU",
    "KV",
    "KW",
    "KY",
    "LA",
    "LB",
    "LG",
    "LI",
    "LN",
    "LO",
    "LT",
    "LU",
    "LV",
    "MG",
    "MH",
    "MI",
    "MK",
    "ML",
    "MN",
    "MR",
    "MS",
    "MT",
    "MY",
    "NA",
    "NB",
    "ND",
    "NE",
    "NG",
    "NL",
    "NN",
    "NO",
    "NR",
    "NV",
    "NY",
    "OC",
    "OJ",
    "OM",
    "OR",
    "OS",
    "PA",
    "PI",
    "PL",
    "PS",
    "PT",
    "QU",
    "RM",
    "RN",
    "RO",
    "RU",
    "RW",
    "SA",
    "SC",
    "SD",
    "SE",
    "SG",
    "SI",
    "SK",
    "SL",
    "SM",
    "SN",
    "SO",
    "SQ",
    "SR",
    "SS",
    "ST",
    "SU",
    "SV",
    "SW",
    "TA",
    "TE",
    "TG",
    "TH",
    "TI",
    "TK",
    "TL",
    "TN",
    "TO",
    "TR",
    "TS",
    "TT",
    "TW",
    "TY",
    "UG",
    "UK",
    "UR",
    "UZ",
    "VE",
    "VI",
    "VO",
    "WA",
    "WO",
    "XH",
    "YI",
    "YO",
    "ZA",
    "ZH",
    "ZU",
}


def _parse_bool_env(env_var: str, default: str = "false") -> bool:
    """Parse boolean environment variable with error handling."""
    try:
        value: str = os.getenv(env_var, default).lower().strip()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        else:
            logger.error(
                "Invalid boolean value '%s' for %s. Valid values: true/false, 1/0, yes/no, on/off",
                value,
                env_var,
            )
            raise ValueError(f"Invalid boolean value for {env_var}: {value}")
    except AttributeError as e:
        logger.exception("Error parsing boolean environment variable %s", env_var)
        raise ValueError(f"Error parsing {env_var}: {e}") from e


def _parse_int_env(
    env_var: str,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Parse integer environment variable with error handling and value clamping.

    Args:
        env_var: Environment variable name to parse
        default: Default value to return on error
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)

    Returns:
        int: Parsed and validated integer value
    """
    try:
        value_str: str = os.getenv(env_var, str(default))
        result: int = int(value_str)

        # Apply clamping if specified
        if min_value is not None and result < min_value:
            logger.warning(
                "Environment variable %s value %d is below minimum %d, clamping to %d",
                env_var,
                result,
                min_value,
                min_value,
            )
            return min_value

        if max_value is not None and result > max_value:
            logger.warning(
                "Environment variable %s value %d is above maximum %d, clamping to %d",
                env_var,
                result,
                max_value,
                max_value,
            )
            return max_value

        return result

    except (ValueError, TypeError) as e:
        logger.error(
            "Invalid integer value '%s' for %s, using default %d: %s",
            os.getenv(env_var, str(default)),
            env_var,
            default,
            str(e),
        )
        return default


class Settings:
    """Application settings with environment variable configuration."""

    # Translation API settings - Only Lingo.dev
    LINGO_API_KEY: str | None = os.getenv("LINGO_API_KEY")

    # Server configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = _parse_int_env("PORT", default=7860, min_value=1, max_value=65535)
    DEBUG: bool = _parse_bool_env("DEBUG", "false")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")

    # File handling
    MAX_FILE_SIZE_MB: int = _parse_int_env("MAX_FILE_SIZE_MB", default=10, min_value=1)
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    DOWNLOAD_DIR: str = os.getenv("DOWNLOAD_DIR", "downloads")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    IMAGE_CACHE_DIR: str = os.getenv("IMAGE_CACHE_DIR", "temp/images")

    # PDF processing
    PDF_DPI: int = _parse_int_env("PDF_DPI", default=300, min_value=72, max_value=600)
    PRESERVE_IMAGES: bool = _parse_bool_env("PRESERVE_IMAGES", "true")
    MEMORY_THRESHOLD: int = _parse_int_env(
        "MEMORY_THRESHOLD", default=500, min_value=100
    )  # MB

    # Translation settings
    SOURCE_LANGUAGE: str = "DE"
    TARGET_LANGUAGE: str = "EN"
    TRANSLATION_DELAY: float = float(os.getenv("TRANSLATION_DELAY", "0.1"))
    # Maximum number of concurrent translation tasks.
    # Environment variable: TRANSLATION_CONCURRENCY_LIMIT
    # Default: 8 (must be >= 1)
    TRANSLATION_CONCURRENCY_LIMIT: int = _parse_int_env(
        "TRANSLATION_CONCURRENCY_LIMIT", default=8, min_value=1
    )

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")

    # Cleanup settings
    CLEANUP_INTERVAL_HOURS: int = _parse_int_env(
        "CLEANUP_INTERVAL_HOURS", default=24, min_value=1
    )
    MAX_FILE_AGE_HOURS: int = _parse_int_env(
        "MAX_FILE_AGE_HOURS", default=48, min_value=1
    )

    def __init__(self) -> None:
        """Initialize settings and create required directories."""
        # Create required directories with proper error handling
        directories_to_create: list[str] = [
            self.UPLOAD_DIR,
            self.DOWNLOAD_DIR,
            self.TEMP_DIR,
            self.IMAGE_CACHE_DIR,
        ]
        for directory in directories_to_create:
            # Normalize path to absolute path
            normalized_path: str = os.path.abspath(directory)

            try:
                os.makedirs(normalized_path, exist_ok=True)
                logger.debug(
                    "Successfully created/verified directory: %s", normalized_path
                )
            except PermissionError as e:
                logger.error(
                    "Permission denied when creating directory '%s': %s. "
                    "Check file system permissions for the application.",
                    normalized_path,
                    e,
                )
                raise
            except OSError as e:
                logger.error(
                    "OS error when creating directory '%s': %s. "
                    "Check path validity and available disk space.",
                    normalized_path,
                    e,
                )
                raise

        # Post-initialization setup
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post-initialization to handle auto-generated settings."""
        # Auto-generate SECRET_KEY for DEBUG mode if not provided
        if self.DEBUG and not self.SECRET_KEY:
            self.SECRET_KEY = secrets.token_urlsafe(32)
            logger.info("Auto-generated SECRET_KEY for DEBUG mode")

    def get_available_translators(self) -> list[str]:
        """Get list of available translation services."""
        available: list[str] = []

        if self.LINGO_API_KEY:
            available.append("lingo")

        return available

    def validate_configuration(self) -> bool:
        """Validate that required configuration is present and all settings are valid.

        Note: Boolean settings (DEBUG, PRESERVE_IMAGES) are validated
        during parsing by _parse_bool_env and will raise ValueError if
        invalid.
        """
        validation_results: list[bool] = [
            self._validate_api_key(),
            self._validate_secret_key(),
            self._validate_language_settings(),
            self._validate_directories(),
            self._validate_numeric_settings(),
            self._validate_terminology(),
        ]

        overall_valid: bool = all(validation_results)

        if overall_valid:
            logger.info("Configuration validation passed")
        else:
            logger.error("Configuration validation failed")

        return overall_valid

    def _validate_api_key(self) -> bool:
        """Validate API key configuration."""
        if not self.LINGO_API_KEY:
            logger.error("LINGO_API_KEY is required but not configured")
            return False
        return True

    def _validate_language_settings(self) -> bool:
        """Validate language code settings."""
        valid: bool = True

        if self.SOURCE_LANGUAGE.upper() not in VALID_LANGUAGE_CODES:
            logger.error(
                "Invalid SOURCE_LANGUAGE code: %s. Must be a valid ISO 639-1 language code.",
                self.SOURCE_LANGUAGE,
            )
            valid = False

        if self.TARGET_LANGUAGE.upper() not in VALID_LANGUAGE_CODES:
            logger.error(
                "Invalid TARGET_LANGUAGE code: %s. Must be a valid ISO 639-1 language code.",
                self.TARGET_LANGUAGE,
            )
            valid = False

        return valid

    def _validate_directories(self) -> bool:
        """Validate directory accessibility and writability."""
        directories_to_check: list[str] = [
            self.UPLOAD_DIR,
            self.DOWNLOAD_DIR,
            self.TEMP_DIR,
            self.IMAGE_CACHE_DIR,
        ]

        valid: bool = True
        for directory in directories_to_check:
            if not self._check_directory_writable(directory):
                # Avoid double-logging here; _check_directory_writable logs at error level
                valid = False

        return valid

    def _validate_numeric_settings(self) -> bool:
        """Validate numeric configuration settings."""
        valid: bool = True

        if self.PORT < 1 or self.PORT > 65535:
            logger.error("Invalid PORT: %s. Must be between 1-65535", self.PORT)
            valid = False

        if self.MAX_FILE_SIZE_MB <= 0:
            logger.error(
                "Invalid MAX_FILE_SIZE_MB: %s. Must be positive", self.MAX_FILE_SIZE_MB
            )
            valid = False

        if self.PDF_DPI < 72 or self.PDF_DPI > 600:
            logger.error("Invalid PDF_DPI: %s. Recommended range: 72-600", self.PDF_DPI)
            valid = False

        if self.MEMORY_THRESHOLD <= 0:
            logger.error(
                "Invalid MEMORY_THRESHOLD: %s. Must be positive", self.MEMORY_THRESHOLD
            )
            valid = False

        if self.TRANSLATION_DELAY < 0:
            logger.error(
                "Invalid TRANSLATION_DELAY: %s. Must be non-negative",
                self.TRANSLATION_DELAY,
            )
            valid = False

        if self.TRANSLATION_CONCURRENCY_LIMIT < 1:
            logger.error(
                "Invalid TRANSLATION_CONCURRENCY_LIMIT: %s. Must be >= 1",
                self.TRANSLATION_CONCURRENCY_LIMIT,
            )
            valid = False

        return valid

    def _validate_terminology(self) -> bool:
        """Validate terminology and translation settings."""
        valid: bool = True

        # Check if source and target languages are different
        if self.SOURCE_LANGUAGE.upper() == self.TARGET_LANGUAGE.upper():
            logger.warning(
                "SOURCE_LANGUAGE and TARGET_LANGUAGE are the same. "
                "Translation may not be necessary."
            )
            # This is a warning, not an error, so don't set valid = False

        return valid

    def _validate_secret_key(self) -> bool:
        """Validate SECRET_KEY configuration for security."""
        # SECRET_KEY is required when DEBUG=False
        if not self.DEBUG and not self.SECRET_KEY:
            logger.error(
                "SECRET_KEY is required but not configured when DEBUG=False. "
                "Set SECRET_KEY environment variable with a secure random value."
            )
            return False

        # Validate SECRET_KEY strength for production
        if not self.DEBUG and len(self.SECRET_KEY) < 32:
            logger.error(
                "SECRET_KEY is too weak: %s characters. "
                "Must be at least 32 characters for production use.",
                len(self.SECRET_KEY),
            )
            return False

        return True

    def _check_directory_writable(self, directory: str) -> bool:
        """Check if a directory is writable by attempting to create a test file."""
        try:
            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)

            # Try to write a test file
            test_file_path: str = os.path.join(directory, ".write_test")
            with open(test_file_path, "w") as test_file:
                test_file.write("test")

            # Clean up test file
            os.remove(test_file_path)
            return True

        except (OSError, PermissionError) as err:
            # Log without traceback at error level for expected validation failures
            logger.error("Directory not writable: '%s'", directory)
            # Emit traceback only in debug mode for diagnostics
            logger.debug(
                "Directory check error for '%s': %s", directory, err, exc_info=True
            )
            return False
