"""Dolphin OCR core package (internal).

Exposes configuration, standardized errors, and logging setup utilities.
"""

from __future__ import annotations

from .config import ConfigurationManager, DolphinConfig
from .errors import ErrorResponse
from .logging_config import get_logger, setup_logging

__all__: list[str] = [
    "ConfigurationManager",
    "DolphinConfig",
    "ErrorResponse",
    "setup_logging",
    "get_logger",
]
