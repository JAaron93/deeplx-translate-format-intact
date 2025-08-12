import os

import pytest

from services.dolphin_ocr_service import DolphinOCRService


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_DOLPHIN_TESTS", "false").lower()
    not in {"1", "true", "yes", "on"},
    reason="Live Dolphin OCR API tests are disabled by default.",
)
def test_live_process_images_smoke():
    # This test requires a real Modal endpoint and HF token to be set in env.
    # It sends a single small image derived from a tiny PNG.
    endpoint = os.getenv("DOLPHIN_MODAL_ENDPOINT")
    token = os.getenv("HF_TOKEN")
    assert endpoint, "DOLPHIN_MODAL_ENDPOINT must be set for live test"
    assert token, "HF_TOKEN must be set for live test"

    # 1x1 pixel transparent PNG used as a minimal valid image for tests.
    # If decoding fails in some environments, replace with a small valid sample.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
        b"\x00\x0bIDATx\x9cc``\x00\x00\x00\x04\x00\x01\x0b\xe7\x02"
        b"\x9d\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    svc = DolphinOCRService()
    out = svc.process_document_images([png_bytes])

    # Shape assertions
    assert isinstance(out, dict), f"Expected dict response, got: {type(out)!r}"

    if "pages" in out:
        pages = out["pages"]
        assert isinstance(pages, list), "'pages' must be a list"
        assert len(pages) > 0, "'pages' must contain at least one page"
        first = pages[0]
        assert isinstance(first, dict), "each page must be a dict"
        # Allow either text or block style outputs
        assert (
            "text" in first or "blocks" in first
        ), "page dict must contain 'text' or 'blocks'"
        if "text" in first:
            assert isinstance(first["text"], str)
        if "blocks" in first:
            assert isinstance(first["blocks"], list)
    elif "ok" in out:
        # Some endpoints may return a simple status shape
        assert isinstance(out["ok"], bool)
        assert out["ok"] is True
    else:
        raise AssertionError(f"Unexpected response keys: {list(out.keys())}")
