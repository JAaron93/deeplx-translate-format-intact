import os

import pytest

from services.translation_service import LingoTranslator


@pytest.fixture
def mock_lingo_translator():
    """Provide a LingoTranslator instance with a test API key for any test module/class."""
    env = os.environ.copy()
    env.setdefault("LINGO_API_KEY", "test_api_key")
    # Ensure environment is set for the duration of this fixture
    old = os.environ.get("LINGO_API_KEY")
    os.environ["LINGO_API_KEY"] = env["LINGO_API_KEY"]
    try:
        yield LingoTranslator()
    finally:
        if old is None:
            os.environ.pop("LINGO_API_KEY", None)
        else:
            os.environ["LINGO_API_KEY"] = old
