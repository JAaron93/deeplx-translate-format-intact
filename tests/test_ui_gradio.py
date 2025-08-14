import contextlib
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

from tests.helpers import write_encrypted_pdf
from ui.gradio_interface import create_gradio_interface


def _launch_blocks() -> tuple[gr.blocks.Blocks, str]:
    demo = create_gradio_interface()
    # Launch in headless mode; returns (app, local_url, share_url)
    _, local_url, _ = demo.launch(
        prevent_thread_lock=True,
        show_api=False,
    )
    return demo, local_url


def _teardown_blocks(demo: gr.blocks.Blocks) -> None:
    # defensive cleanup (no overly broad except during normal runtime)
    with contextlib.suppress(Exception):
        demo.close()


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

        pdf = tmp_path / "ok.pdf"
        pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

        client = Client(url)
        # View API to discover endpoints (ensure reachable)
        client.view_api()
        # Predict on upload endpoint; on_file_upload bound to file change
        result = client.predict(handle_file(str(pdf)))
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

        txt = tmp_path / "note.txt"
        txt.write_text("hello")

        client = Client(url)
        out = client.predict(handle_file(str(txt)))
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
        out = client.predict(handle_file(str(pdf)))
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
            return (
                "started",
                "queued",
                gr.update(interactive=False),
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
        )
        assert isinstance(res, (list, tuple))
        # Expect at least the 3 return values from fake_start
        assert len(res) >= 3
    finally:
        _teardown_blocks(demo)
