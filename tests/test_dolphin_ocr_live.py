import os

import pytest

from services.dolphin_ocr_service import DolphinOCRService


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_DOLPHIN_TESTS", "false").lower() not in {"1", "true", "yes", "on"},
    reason="Live Dolphin OCR API tests are disabled by default.",
)
def test_live_process_images_smoke():
    # This test requires a real Modal endpoint and HF token to be set in env.
    # It sends a single small image derived from a tiny PDF page.
    endpoint = os.getenv("DOLPHIN_MODAL_ENDPOINT")
    token = os.getenv("HF_TOKEN")
    assert endpoint, "DOLPHIN_MODAL_ENDPOINT must be set for live test"
    assert token, "HF_TOKEN must be set for live test"

    # Create a tiny PNG by writing minimal bytes; if the service requires real
    # image decoding, replace this with a small valid PNG fixture.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0bIDATx\x9cc``\x00\x00\x00\x04\x00\x01\x0b\xe7\x02"
        b"\x9d\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    svc = DolphinOCRService()
    out = svc.process_document_images([png_bytes])
    assert isinstance(out, dict)
    assert "pages" in out or "ok" in out

