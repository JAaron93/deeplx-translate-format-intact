"""Configuration settings for the PDF translator."""

import logging
import os

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Valid ISO 639-1 language codes
VALID_LANGUAGE_CODES = {
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
        value = os.getenv(env_var, default).lower().strip()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off"):
            return False
        else:
            logger.error(
                f"Invalid boolean value '{value}' for {env_var}. "
                f"Valid values: true/false, 1/0, yes/no, on/off"
            )
            raise ValueError(f"Invalid boolean value for {env_var}: {value}")
    except AttributeError as e:
        logger.error(f"Error parsing boolean environment variable {env_var}: {e}")
        raise ValueError(f"Error parsing {env_var}: {e}") from e


class Settings:
    # Translation API settings - Only Lingo.dev
    LINGO_API_KEY: str = os.getenv("LINGO_API_KEY")

    # Server configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "7860"))
    DEBUG: bool = _parse_bool_env("DEBUG", "false")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY") or (_ for _ in ()).throw(
        ValueError("SECRET_KEY environment variable is required")
    )

    # File handling
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    DOWNLOAD_DIR: str = os.getenv("DOWNLOAD_DIR", "downloads")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "temp")
    IMAGE_CACHE_DIR: str = os.getenv("IMAGE_CACHE_DIR", "temp/images")

    # PDF processing
    PDF_DPI: int = int(os.getenv("PDF_DPI", "300"))
    PRESERVE_IMAGES: bool = _parse_bool_env("PRESERVE_IMAGES", "true")
    MEMORY_THRESHOLD: int = int(os.getenv("MEMORY_THRESHOLD", "500"))  # MB

    # Translation settings
    SOURCE_LANGUAGE: str = "DE"
    TARGET_LANGUAGE: str = "EN"
    TRANSLATION_DELAY: float = float(os.getenv("TRANSLATION_DELAY", "0.1"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "app.log")

    # Cleanup settings
    CLEANUP_INTERVAL_HOURS: int = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))
    MAX_FILE_AGE_HOURS: int = int(os.getenv("MAX_FILE_AGE_HOURS", "48"))

    def __init__(self):
        # Create required directories
        for directory in [
            self.UPLOAD_DIR,
            self.DOWNLOAD_DIR,
            self.TEMP_DIR,
            self.IMAGE_CACHE_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

    def get_available_translators(self) -> list:
        """Get list of available translation services."""
        available = []

        if self.LINGO_API_KEY:
            available.append("lingo")

        return available

    def validate_configuration(self) -> bool:
        """Validate that required configuration is present and all settings are valid."""
        validation_results = [
            self._validate_api_key(),
            self._validate_language_settings(),
            self._validate_directories(),
            self._validate_numeric_settings(),
            self._validate_boolean_settings(),
            self._validate_terminology(),
        ]

        overall_valid = all(validation_results)

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
        valid = True

        if self.SOURCE_LANGUAGE.upper() not in VALID_LANGUAGE_CODES:
            logger.error(
                f"Invalid SOURCE_LANGUAGE code: {self.SOURCE_LANGUAGE}. "
                f"Must be a valid ISO 639-1 language code."
            )
            valid = False

        if self.TARGET_LANGUAGE.upper() not in VALID_LANGUAGE_CODES:
            logger.error(
                f"Invalid TARGET_LANGUAGE code: {self.TARGET_LANGUAGE}. "
                f"Must be a valid ISO 639-1 language code."
            )
            valid = False

        return valid

    def _validate_directories(self) -> bool:
        """Validate directory accessibility and writability."""
        directories_to_check = [
            self.UPLOAD_DIR,
            self.DOWNLOAD_DIR,
            self.TEMP_DIR,
            self.IMAGE_CACHE_DIR,
        ]

        valid = True
        for directory in directories_to_check:
            if not self._check_directory_writable(directory):
                logger.error(f"Directory '{directory}' is not writable")
                valid = False

        return valid

    def _validate_numeric_settings(self) -> bool:
        """Validate numeric configuration settings."""
        valid = True

        if self.PORT < 1 or self.PORT > 65535:
            logger.error(f"Invalid PORT: {self.PORT}. Must be between 1-65535")
            valid = False

        if self.MAX_FILE_SIZE_MB <= 0:
            logger.error(
                f"Invalid MAX_FILE_SIZE_MB: {self.MAX_FILE_SIZE_MB}. Must be positive"
            )
            valid = False

        if self.PDF_DPI < 72 or self.PDF_DPI > 600:
            logger.error(f"Invalid PDF_DPI: {self.PDF_DPI}. Recommended range: 72-600")
            valid = False

        if self.MEMORY_THRESHOLD <= 0:
            logger.error(
                f"Invalid MEMORY_THRESHOLD: {self.MEMORY_THRESHOLD}. Must be positive"
            )
            valid = False

        if self.TRANSLATION_DELAY < 0:
            logger.error(
                f"Invalid TRANSLATION_DELAY: {self.TRANSLATION_DELAY}. Must be non-negative"
            )
            valid = False

        return valid

    def _validate_boolean_settings(self) -> bool:
        """Validate boolean configuration settings."""
        # Boolean settings are validated during parsing by _parse_bool_env
        # This method can be extended for additional boolean-specific validations
        return True

    def _validate_terminology(self) -> bool:
        """Validate terminology and translation settings."""
        valid = True

        # Check if source and target languages are different
        if self.SOURCE_LANGUAGE.upper() == self.TARGET_LANGUAGE.upper():
            logger.warning(
                "SOURCE_LANGUAGE and TARGET_LANGUAGE are the same. "
                "Translation may not be necessary."
            )
            # This is a warning, not an error, so don't set valid = False

        return valid

    def _check_directory_writable(self, directory: str) -> bool:
        """Check if a directory is writable by attempting to create a test file."""
        try:
            # Ensure directory exists
            os.makedirs(directory, exist_ok=True)

            # Try to write a test file
            test_file_path = os.path.join(directory, ".write_test")
            with open(test_file_path, "w") as test_file:
                test_file.write("test")

            # Clean up test file
            os.remove(test_file_path)
            return True

        except (OSError, PermissionError) as e:
            logger.error(f"Directory writability check failed for '{directory}': {e}")
            return False
