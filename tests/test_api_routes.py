from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import api_router, app_router
from tests.test_utils import write_encrypted_pdf, write_minimal_pdf


def _write_min_pdf(p: Path) -> None:
    write_minimal_pdf(p)


def _write_encrypted_pdf(p: Path) -> None:
    write_encrypted_pdf(p)


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(app_router)
    app.include_router(api_router, prefix="/api")
    return TestClient(app)


def test_upload_rejects_non_pdf(tmp_path: Path) -> None:
    client = _make_app()
    nonpdf = tmp_path / "f.txt"
    nonpdf.write_text("hello")
    with nonpdf.open("rb") as fh:
        res = client.post(
            "/api/upload",
            files={"file": ("f.txt", fh, "text/plain")},
        )
    assert res.status_code == 400
    body = res.json()
    assert body["detail"]["error_code"] == "DOLPHIN_005"


def test_upload_rejects_encrypted_pdf(tmp_path: Path) -> None:
    client = _make_app()
    enc = tmp_path / "enc.pdf"
    _write_encrypted_pdf(enc)
    with enc.open("rb") as fh:
        res = client.post(
            "/api/upload",
            files={
                "file": ("enc.pdf", fh, "application/pdf"),
            },
        )
    assert res.status_code == 400
    body = res.json()
    assert body["detail"]["error_code"] == "DOLPHIN_014"


def test_upload_accepts_valid_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _make_app()
    pdf = tmp_path / "ok.pdf"
    _write_min_pdf(pdf)

    # Monkeypatch save to use our tmp file path passthrough
    from core.translation_handler import document_processor, file_handler

    def _save_upload_file(_file):
        # emulate returning a saved path
        return str(pdf)

    def _extract_content(_path: str):
        return {"type": "pdf", "metadata": {}}

    monkeypatch.setattr(file_handler, "save_upload_file", _save_upload_file)
    monkeypatch.setattr(document_processor, "extract_content", _extract_content)

    with pdf.open("rb") as fh:
        res = client.post(
            "/api/upload",
            files={"file": ("ok.pdf", fh, "application/pdf")},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["filename"] == "ok.pdf"
    assert body["content_type"] == "pdf"
