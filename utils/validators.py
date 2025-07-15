"""Input validation utilities"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import UploadFile

logger = logging.getLogger(__name__)


class FileValidator:
    """Validates file uploads and inputs"""

    def __init__(self) -> None:
        from config.settings import Settings

        settings = Settings()
        self.max_file_size: int = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        self.allowed_extensions: set[str] = set(settings.ALLOWED_EXTENSIONS)
        self.allowed_mimetypes: set[str] = {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        }

    def validate_file(self, filename: str, file_size: int) -> dict[str, Any]:
        """Validate file upload"""
        try:
            # Check filename
            if not filename:
                return {"valid": False, "error": "No filename provided"}

            # Check file extension
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.allowed_extensions:
                return {
                    "valid": False,
                    "error": f"Unsupported file type. Allowed: {', '.join(self.allowed_extensions)}",
                }

            # Check file size
            if file_size > self.max_file_size:
                max_size_mb = self.max_file_size / (1024 * 1024)
                return {
                    "valid": False,
                    "error": f"File too large. Maximum size: {max_size_mb}MB",
                }

            if file_size == 0:
                return {"valid": False, "error": "File is empty"}

            return {"valid": True}

        except Exception as e:
            logger.error(f"File validation error: {e}")
            return {"valid": False, "error": f"Validation error: {e!s}"}

    def validate_upload_file(self, upload_file: UploadFile) -> dict[str, Any]:
        """Validate FastAPI UploadFile"""
        try:
            # Check content type
            if upload_file.content_type not in self.allowed_mimetypes:
                return {
                    "valid": False,
                    "error": f"Unsupported content type: {upload_file.content_type}",
                }

            # Get file size with safety check
            current_pos = upload_file.file.tell()
            upload_file.file.seek(0, 2)
            file_size = upload_file.file.tell()

            # Safety check before seeking back
            if file_size > self.max_file_size:
                return {
                    "valid": False,
                    "error": f"File too large. Maximum size: {self.max_file_size / (1024 * 1024)}MB",
                }

            upload_file.file.seek(current_pos)  # Reset to original position

            # Validate using standard method
            return self.validate_file(upload_file.filename, file_size)

        except Exception as e:
            logger.error(f"Upload file validation error: {e}")
            return {"valid": False, "error": f"Validation error: {e!s}"}

    def _load_language_config(self) -> dict[str, Any]:
        """Load language configuration from external JSON file"""
        try:
            config_path = Path(__file__).parent.parent / "config" / "languages.json"
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load language config: {e}")
            # Fallback to default values if config loading fails
            return {
                "supported_languages": [
                    "English",
                    "Spanish",
                    "French",
                    "German",
                    "Italian",
                    "Portuguese",
                    "Russian",
                    "Chinese",
                    "Japanese",
                    "Korean",
                    "Arabic",
                    "Hindi",
                    "Dutch",
                    "Swedish",
                    "Norwegian",
                ],
                "supported_formats": ["PDF", "DOCX", "TXT"],
            }

    def validate_language(self, language: str) -> dict[str, Any]:
        """Validate language selection"""
        config = self._load_language_config()
        supported_languages = config.get("supported_languages", [])

        if not language:
            return {"valid": False, "error": "No language selected"}

        if language not in supported_languages:
            return {
                "valid": False,
                "error": f"Unsupported language. Supported: {', '.join(supported_languages)}",
            }

        return {"valid": True}

    def validate_output_format(self, format_type: str) -> dict[str, Any]:
        """Validate output format selection"""
        config = self._load_language_config()
        supported_formats = config.get("supported_formats", ["PDF", "DOCX", "TXT"])

        if not format_type:
            return {"valid": False, "error": "No output format selected"}

        if format_type.upper() not in supported_formats:
            return {
                "valid": False,
                "error": f"Unsupported format. Supported: {', '.join(supported_formats)}",
            }

        return {"valid": True}
