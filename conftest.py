from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest as pytest_api

pytest_plugins: tuple[str, ...] = ("pytest_asyncio",)


def pytest_configure(config: pytest_api.Config) -> None:
    """Set required environment variables for tests early in startup.

    This runs before test collection, ensuring modules that read environment
    variables at import time (e.g., `config.settings.Settings`) see the values.
    """
    # Set DEBUG=True for tests to enable auto-generated SECRET_KEY
    os.environ.setdefault("DEBUG", "true")
    # Provide a fallback SECRET_KEY for tests (will be auto-generated if empty)
    os.environ.setdefault("SECRET_KEY", "")
    # When focusing, quiet output at runtime without relying on pre-parsed
    # addopts
    if os.getenv("FOCUSED"):
        config.option.quiet = max(getattr(config.option, "quiet", 0), 1)
