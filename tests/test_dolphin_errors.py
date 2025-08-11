from dolphin_ocr.errors import (
    DolphinError,
    ApiRateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    ProcessingTimeoutError,
    InvalidDocumentFormatError,
    OcrProcessingError,
    LayoutAnalysisError,
    TranslationServiceError,
    LayoutPreservationError,
    DocumentReconstructionError,
    MemoryExhaustionError,
    StorageError,
    ConfigurationError,
    CODE_TO_MESSAGE,
    get_error_message,
)


def test_error_hierarchy_and_codes():
    e = ApiRateLimitError("Too many requests")
    assert isinstance(e, DolphinError)
    assert e.error_code == "DOLPHIN_001"
    assert "Too many" in str(e)
    assert e.to_dict()["error_code"] == "DOLPHIN_001"

    assert ServiceUnavailableError("svc").error_code == "DOLPHIN_002"
    assert AuthenticationError("auth").error_code == "DOLPHIN_003"
    assert ProcessingTimeoutError("timeout").error_code == "DOLPHIN_004"
    assert InvalidDocumentFormatError("fmt").error_code == "DOLPHIN_005"
    assert OcrProcessingError("ocr").error_code == "DOLPHIN_006"
    assert LayoutAnalysisError("layout").error_code == "DOLPHIN_007"
    assert TranslationServiceError("trans").error_code == "DOLPHIN_008"
    assert LayoutPreservationError("preserve").error_code == "DOLPHIN_009"
    assert DocumentReconstructionError("recon").error_code == "DOLPHIN_010"
    assert MemoryExhaustionError("mem").error_code == "DOLPHIN_011"
    assert StorageError("store").error_code == "DOLPHIN_012"
    assert ConfigurationError("cfg").error_code == "DOLPHIN_013"

    # Mapping coverage and defaulting
    for code, msg in CODE_TO_MESSAGE.items():
        # Build a temporary subclass on the fly for mapping lookup
        class _Tmp(DolphinError):
            error_code = code

        e2 = _Tmp()
        assert msg in str(e2)
        assert get_error_message(code) == msg


