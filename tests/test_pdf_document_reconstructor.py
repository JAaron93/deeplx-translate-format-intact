from __future__ import annotations

from pathlib import Path

import pytest

from services.pdf_document_reconstructor import (
    PDFDocumentReconstructor,
    UnsupportedFormatError,
)


def _write_bytes(tmp_path: Path, name: str, data: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def test_is_pdf_format_extension_only():
    recon = PDFDocumentReconstructor()
    assert recon.is_pdf_format("/x/y/z.pdf")
    # Upper-case still considered PDF by suffix lowercasing
    assert recon.is_pdf_format("/x/y/z.PDF") is True
    # Non-PDF extension
    assert recon.is_pdf_format("/x/y/z.txt") is False


def test_validate_rejects_wrong_extension(tmp_path: Path):
    recon = PDFDocumentReconstructor()
    fpath = _write_bytes(tmp_path, "doc.txt", b"hello")
    with pytest.raises(UnsupportedFormatError) as ei:
        recon.validate_pdf_format_or_raise(fpath)
    assert "Unsupported format" in str(ei.value)


def test_validate_rejects_missing_file(tmp_path: Path):
    recon = PDFDocumentReconstructor()
    with pytest.raises(UnsupportedFormatError) as ei:
        recon.validate_pdf_format_or_raise(str(tmp_path / "nope.pdf"))
    assert "not found" in str(ei.value)


def test_validate_rejects_bad_header(tmp_path: Path):
    recon = PDFDocumentReconstructor()
    fpath = _write_bytes(tmp_path, "bad.pdf", b"%NOT-\nrest")
    with pytest.raises(UnsupportedFormatError) as ei:
        recon.validate_pdf_format_or_raise(fpath)
    assert "missing %PDF- header" in str(ei.value)


def test_validate_accepts_valid_pdf(tmp_path: Path):
    """A valid, non-encrypted PDF should pass validation."""
    recon = PDFDocumentReconstructor()
    fpath = _write_bytes(tmp_path, "valid.pdf", b"%PDF-1.7\nrest")
    # Should not raise
    recon.validate_pdf_format_or_raise(fpath)


def test_validate_without_pypdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Validation works when pypdf is not installed (skip encryption check)."""
    fpath = _write_bytes(tmp_path, "test.pdf", b"%PDF-1.7\nrest")

    # Make import of pypdf appear absent by patching finder
    import importlib.util as _iu

    original_find_spec = _iu.find_spec

    def _find_spec(name, *args, **kwargs):
        if name == "pypdf":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(_iu, "find_spec", _find_spec)

    recon = PDFDocumentReconstructor()
    # Should not raise - encryption check is skipped
    recon.validate_pdf_format_or_raise(fpath)


def test_validate_encrypted_pdf_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Write a valid header to pass the initial check
    fpath = _write_bytes(tmp_path, "enc.pdf", b"%PDF-1.7\nrest")

    class _FakeReader:
        def __init__(self, _path: str):
            self.is_encrypted = True

    # Simplify: create a fake 'pypdf' module with our FakeReader
    import sys
    import types

    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = _FakeReader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    recon = PDFDocumentReconstructor()
    with pytest.raises(UnsupportedFormatError) as ei:
        recon.validate_pdf_format_or_raise(fpath)
    err = ei.value
    assert getattr(err, "error_code", None) == "DOLPHIN_014"
    assert "Encrypted PDFs not supported" in str(err)
