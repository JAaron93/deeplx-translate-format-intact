import pytest
from pathlib import Path

from utils.pdf_validator import (
    validate_pdf,
    validate_pdf_extension_and_header,
    detect_pdf_encryption,
    validate_pdf_structure,
)


def _write_minimal_pdf(path: Path) -> None:
    # Minimal valid-ish PDF with header, xref, trailer, EOF
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj<<>>endobj\n"
        b"xref\n0 2\n0000000000 65535 f \n0000000010 00000 n \n"
        b"trailer<< /Root 1 0 R >>\nstartxref\n9\n%%EOF\n"
    )
    path.write_bytes(content)


def _write_encrypted_pdf(path: Path) -> None:
    import importlib.util
    import importlib

    spec = importlib.util.find_spec("pypdf")
    if spec is None:  # pragma: no cover - dependency missing
        pytest.skip("pypdf not available")
    mod = importlib.import_module("pypdf")
    PdfWriter = getattr(mod, "PdfWriter")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("pwd")
    with path.open("wb") as fh:
        writer.write(fh)


def test_extension_and_header_valid_tmp(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_minimal_pdf(pdf)
    res = validate_pdf_extension_and_header(str(pdf))
    assert res.ok is True
    assert res.is_pdf is True
    assert res.issues == []


def test_extension_and_header_invalid_nonpdf(tmp_path: Path) -> None:
    txt = tmp_path / "doc.txt"
    txt.write_text("hello")
    res = validate_pdf_extension_and_header(str(txt))
    assert res.ok is False
    assert any(issue.code == "DOLPHIN_005" for issue in res.issues)


def test_detect_encryption_for_encrypted_pdf(tmp_path: Path) -> None:
    enc = tmp_path / "enc.pdf"
    _write_encrypted_pdf(enc)
    res = detect_pdf_encryption(str(enc))
    assert res.is_pdf is True
    assert res.is_encrypted is True
    assert any(issue.code == "DOLPHIN_014" for issue in res.issues)


def test_validate_pdf_structure_minimal(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_minimal_pdf(pdf)
    res = validate_pdf_structure(str(pdf))
    assert res.ok is True
    assert res.is_pdf is True


def test_validate_pdf_end_to_end(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_minimal_pdf(pdf)
    res = validate_pdf(str(pdf))
    assert res.ok is True
    assert res.is_pdf is True
    assert res.is_encrypted is False


def test_validate_pdf_nonpdf_file(tmp_path: Path) -> None:
    f = tmp_path / "not.pdf"
    f.write_text("not a pdf")
    res = validate_pdf(str(f))
    assert res.ok is False
    assert any(issue.code == "DOLPHIN_005" for issue in res.issues)
