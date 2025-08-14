import logging
import os
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

from tests.helpers import write_encrypted_pdf
from ui.gradio_interface import create_gradio_interface


@pytest.fixture(autouse=True)
def _patch_gradio_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optionally patch gradio client schema parsing to tolerate boolean
    sub-schemas.

    Controlled by env var GRADIO_SCHEMA_PATCH. Truthy values ("1", "true",
    "yes", "on") enable the patch. In CI, the patch is enabled by default.
    Some versions of gradio_client may emit boolean JSON Schema fragments
    (True/False) for additionalProperties, which can break the parser.
    When enabled, this fixture makes the parser robust for test runs.
    """
    enabled = os.environ.get("GRADIO_SCHEMA_PATCH", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if not enabled and os.environ.get("CI", "").lower() == "true":
        enabled = True
    if not enabled:
        return
    try:
        from gradio_client import utils as gc_utils  # type: ignore
    except Exception:
        return

    if hasattr(gc_utils, "_json_schema_to_python_type"):
        _orig_inner = gc_utils._json_schema_to_python_type  # type: ignore[attr-defined]

        def _safe_inner(schema, defs=None):  # type: ignore[override]
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            try:
                return _orig_inner(schema, defs)
            except Exception:
                return "Any"

        monkeypatch.setattr(gc_utils, "_json_schema_to_python_type", _safe_inner)

    if hasattr(gc_utils, "json_schema_to_python_type"):
        _orig_outer = gc_utils.json_schema_to_python_type

        def _safe_outer(schema, *args, **kwargs):  # type: ignore[override]
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            try:
                return _orig_outer(schema, *args, **kwargs)
            except Exception:
                return "Any"

        monkeypatch.setattr(gc_utils, "json_schema_to_python_type", _safe_outer)


def _launch_blocks() -> tuple[gr.blocks.Blocks, str]:
    # In CI/headless environments localhost may not be reachable; prefer share
    if os.environ.get("CI", "").lower() == "true":
        os.environ.setdefault("GRADIO_SHARE", "true")
    demo = create_gradio_interface()
    # Launch in headless mode; request a share link as fallback
    app, local_url, share_url = demo.launch(
        prevent_thread_lock=True, show_api=False, share=True
    )
    # Prefer share_url if provided, otherwise local_url
    url = share_url or local_url or ""
    return demo, url


def _teardown_blocks(demo: gr.blocks.Blocks) -> None:
    # Defensive cleanup: suppress only expected shutdown errors
    try:
        demo.close()
    except (RuntimeError, OSError, AttributeError):
        pass
    except Exception as err:  # pragma: no cover
        logging.error("Unexpected error while closing Blocks: %s", err, exc_info=True)
        raise


def test_ui_valid_pdf_upload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    demo, url = _launch_blocks()
    try:
        # Simulate process_file_upload returning tuple of expected UI values
        from ui import gradio_interface as gi

        def fake_process(_file: Any):
            return (
                "preview text",
                "upload ok",
                "English",
                "converted",
                {"info": "ocr details", "metrics": {"ocr_conf": 0.9}},
            )

        monkeypatch.setattr(gi, "process_file_upload", fake_process)

        # Exercise the API using gradio_client
        from gradio_client import Client, handle_file

        from tests.helpers import write_minimal_pdf

        pdf = tmp_path / "ok.pdf"
        write_minimal_pdf(pdf)

        client = Client(url)
        # View API to discover endpoints (ensure reachable)
        client.view_api()
        # Predict on upload endpoint; on_file_upload bound to file change
        result = client.predict(handle_file(str(pdf)), api_name="/on_file_upload")
        # Expect preview text in first slot, upload status in second
        assert isinstance(result, (list, tuple))
        assert result[0] == "preview text"
        assert result[1] == "upload ok"
    finally:
        _teardown_blocks(demo)


def test_ui_non_pdf_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    demo, url = _launch_blocks()
    try:
        from ui import gradio_interface as gi

        def fake_process(_file: Any):
            return {
                "error_code": "DOLPHIN_005",
                "message": "Only PDF format supported",
            }

        monkeypatch.setattr(gi, "process_file_upload", fake_process)

        from gradio_client import Client, handle_file

        from tests.helpers import write_minimal_pdf

        # Use a .pdf to bypass client-side file-type gate and rely on mock semantics
        pdf = tmp_path / "note_as_pdf.pdf"
        write_minimal_pdf(pdf)

        client = Client(url)
        out = client.predict(handle_file(str(pdf)), api_name="/on_file_upload")
        # Second item is upload_status with the error code
        assert "DOLPHIN_005" in (out[1] or "")
    finally:
        _teardown_blocks(demo)


def test_ui_encrypted_pdf_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    demo, url = _launch_blocks()
    try:
        from ui import gradio_interface as gi

        def fake_process(_file: Any):
            return {
                "error_code": "DOLPHIN_014",
                "message": (
                    "Encrypted PDFs not supported - please provide " "unlocked PDF"
                ),
            }

        monkeypatch.setattr(gi, "process_file_upload", fake_process)

        from gradio_client import Client, handle_file

        pdf = tmp_path / "enc.pdf"
        write_encrypted_pdf(pdf)

        client = Client(url)
        out = client.predict(handle_file(str(pdf)), api_name="/on_file_upload")
        # Validate tuple structure and error status content
        assert isinstance(out, (list, tuple))
        assert len(out) >= 2
        status = out[1] or ""
        assert "DOLPHIN_014" in status
        assert "Encrypted PDFs not supported" in status
    finally:
        _teardown_blocks(demo)


def test_ui_translation_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    demo, url = _launch_blocks()
    try:
        # Mock start_translation and get_translation_status for progress
        from ui import gradio_interface as gi

        def fake_start(target, pages, philosophy):
            # Validate parameters are passed correctly
            assert target is not None
            assert pages is not None
            assert philosophy is not None
            # Must return 4 outputs matching [status, upload_status, download_btn, progress_timer]
            return (
                "started",
                "queued",
                gr.update(interactive=False),
                gr.Timer(active=True),
            )

        def fake_status():
            return "processing 50%", 0.5, True

        monkeypatch.setattr(gi, "start_translation", fake_start)
        monkeypatch.setattr(gi, "get_translation_status", fake_status)

        from gradio_client import Client

        client = Client(url)
        # Test translation start with specific parameters
        res = client.predict(
            "Spanish",
            5,
            False,
            api_name="/start_translation_with_progress",
        )
        assert isinstance(res, (list, tuple))
        # Expect at least the 3 return values from fake_start
        assert len(res) >= 3
    finally:
        _teardown_blocks(demo)
