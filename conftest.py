from __future__ import annotations

import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.config import Config

pytest_plugins: list[str] = ["pytest_asyncio"]


def pytest_configure(config: Config) -> None:
    """Set required environment variables for tests early in startup.

    This runs before test collection, ensuring modules that read environment
    variables at import time (e.g., `config.settings.Settings`) see the values.
    """
    os.environ.setdefault("SECRET_KEY", "test-secret-key")
    # When focusing, remove coverage/time-consuming addopts
    if os.getenv("FOCUSED"):
        config.known_args_namespace.addopts = "-q"
