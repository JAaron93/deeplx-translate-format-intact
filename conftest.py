from __future__ import annotations

import os

pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config):
    """Set required environment variables for tests early in startup.

    This runs before test collection, ensuring modules that read environment
    variables at import time (e.g., `config.settings.Settings`) see the values.
    """
    os.environ.setdefault("SECRET_KEY", "test-secret-key")
    # When focusing, remove coverage/time-consuming addopts
    if os.getenv("FOCUSED"):
        config.known_args_namespace.addopts = "-q"
    # Silence unused-argument linters while keeping the canonical
    # hook signature
    del config
