import pytest


@pytest.fixture
def lingo_translator(monkeypatch):
    """Provide a LingoTranslator instance with a test API key for any test module/class."""
    # Set the environment variable using monkeypatch (automatically restored after test)
    monkeypatch.setenv("LINGO_API_KEY", "test_api_key")
    
    # Import LingoTranslator after setting the environment variable
    from services.translation_service import LingoTranslator
    
    return LingoTranslator()
