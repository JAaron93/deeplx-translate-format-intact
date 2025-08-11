import os

import pytest


@pytest.fixture(autouse=True)
def dolphin_env_defaults(monkeypatch):
    """Provide sane default env vars for Dolphin config during tests."""
    monkeypatch.setenv("HF_TOKEN", os.getenv("HF_TOKEN", "mock_hf_token_for_tests"))
    monkeypatch.setenv(
        "DOLPHIN_MODAL_ENDPOINT",
        os.getenv(
            "DOLPHIN_MODAL_ENDPOINT",
            "https://mock-dolphin-endpoint.test.local",
        ),
    )
    yield


@pytest.fixture
def lingo_translator(monkeypatch):
    """Provide a LingoTranslator instance with a test API key for any test module/class."""
    # Set the environment variable using monkeypatch (automatically restored after test)
    monkeypatch.setenv("LINGO_API_KEY", "test_api_key")

    # Import LingoTranslator after setting the environment variable
    from services.translation_service import LingoTranslator

    return LingoTranslator()
