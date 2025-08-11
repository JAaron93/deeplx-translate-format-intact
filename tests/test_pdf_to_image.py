import sys
import types

import pytest

from dolphin_ocr.errors import EncryptedPdfError, InvalidDocumentFormatError
from dolphin_ocr.pdf_to_image import PDFToImageConverter
from tests.utils_dolphin import get_sample_pdfs, load_asset_bytes


@pytest.mark.parametrize("dpi,fmt", [(300, "PNG"), (200, "JPEG")])
def test_convert_pdf_to_images_roundtrip(tmp_path, dpi, fmt):
    pytest.importorskip("pdf2image")
    converter = PDFToImageConverter(dpi=dpi, image_format=fmt)
    sample, _, _ = get_sample_pdfs()
    pdf_path = tmp_path / sample
    pdf_path.write_bytes(load_asset_bytes(sample))

    pages = converter.convert_pdf_to_images(pdf_path)
    assert isinstance(pages, list)
    assert len(pages) >= 1
    assert isinstance(pages[0], bytes)

    # Validate actual image format using Pillow
    try:
        from io import BytesIO

        from PIL import Image
    except Exception:  # pragma: no cover - if Pillow absent in env
        pytest.skip("Pillow not available to verify image format")

    with Image.open(BytesIO(pages[0])) as img:
        fmt_detected = img.format
    # Account for JPEG naming variations
    expected = "JPEG" if fmt.upper() in {"JPEG", "JPG"} else fmt.upper()
    assert fmt_detected == expected


def test_optimize_image_passthrough_when_pillow_missing(tmp_path, monkeypatch):
    pytest.importorskip("pdf2image")
    converter = PDFToImageConverter()

    # Force import error path by making PIL import fail
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # pragma: no cover - controlled
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("force")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    sample, _, _ = get_sample_pdfs()
    pdf_bytes = load_asset_bytes(sample)

    # Write to temp and convert first
    pdf_path = tmp_path / sample
    pdf_path.write_bytes(pdf_bytes)
    pages = converter.convert_pdf_to_images(pdf_path)
    assert pages, "Expected at least one page"

    out = converter.optimize_image_for_ocr(pages[0])
    # Without Pillow, output equals input
    assert out == pages[0]


def test_convert_pdf_to_images_nonexistent_path(tmp_path):
    converter = PDFToImageConverter()
    missing = tmp_path / "nope.pdf"
    with pytest.raises(InvalidDocumentFormatError):
        converter.convert_pdf_to_images(missing)


def test_optimize_image_changes_bytes_when_pillow_available():
    # Skip if Pillow is unavailable
    try:
        from io import BytesIO

        from PIL import Image
    except Exception:  # pragma: no cover
        pytest.skip("Pillow not available")

    # Create a mid-tone image that normalization will adjust
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="PNG")
    original = buf.getvalue()

    converter = PDFToImageConverter(dpi=300, image_format="PNG")
    optimized = converter.optimize_image_for_ocr(original)

    assert optimized != original
    with Image.open(BytesIO(optimized)) as im2:
        assert im2.mode == "L"
        assert im2.format == "PNG"


def test_convert_multi_page_pdf_and_optimize_each(tmp_path):
    pytest.importorskip("pdf2image")
    converter = PDFToImageConverter(dpi=200, image_format="PNG")
    multi_pdf, _, _ = get_sample_pdfs()
    pdf_path = tmp_path / multi_pdf
    pdf_path.write_bytes(load_asset_bytes(multi_pdf))

    pages = converter.convert_pdf_to_images(pdf_path)
    assert len(pages) >= 2
    for b in pages[:3]:
        out = converter.optimize_image_for_ocr(b)
        assert isinstance(out, bytes)


def test_convert_corrupt_pdf_raises(tmp_path):
    converter = PDFToImageConverter()
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not-a-pdf")
    # Fails at header validation before invoking pdf2image
    with pytest.raises(InvalidDocumentFormatError):
        converter.convert_pdf_to_images(bad)


def test_unsupported_extension_raises(tmp_path):
    converter = PDFToImageConverter()
    not_pdf = tmp_path / "file.txt"
    not_pdf.write_text("hello")
    with pytest.raises(InvalidDocumentFormatError) as exc:
        converter.convert_pdf_to_images(not_pdf)
    assert getattr(exc.value, "error_code", None) == "DOLPHIN_005"


def test_encrypted_pdf_detection_raises(tmp_path):
    # Minimal PDF header with /Encrypt marker
    encrypted = tmp_path / "secret.pdf"
    encrypted.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Encrypt true >>\nendobj\n")
    converter = PDFToImageConverter()
    with pytest.raises(EncryptedPdfError) as exc:
        converter.convert_pdf_to_images(encrypted)
    # Encrypted PDFs are mapped to DOLPHIN_014 per errors; ensure class raised
    assert getattr(exc.value, "error_code", None) == "DOLPHIN_014"


def test_corrupt_pdf_header_error_code(tmp_path):
    converter = PDFToImageConverter()
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not-a-pdf")
    with pytest.raises(InvalidDocumentFormatError) as exc:
        converter.convert_pdf_to_images(bad)
    assert exc.value.error_code == "DOLPHIN_005"


def test_memory_error_maps_to_dolphin_011(tmp_path, monkeypatch):
    # Create a minimal valid-looking PDF file (header only)
    pdf_path = tmp_path / "tiny.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    # Inject a fake pdf2image module that raises MemoryError
    fake_pdf2image = types.ModuleType("pdf2image")

    def _raise_memory_error(*_args, **_kwargs):
        raise MemoryError("simulated")

    fake_pdf2image.convert_from_path = _raise_memory_error
    monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)

    converter = PDFToImageConverter()
    with pytest.raises(Exception) as exc:
        converter.convert_pdf_to_images(pdf_path)
    # Ensure our standardized MemoryExhaustionError (DOLPHIN_011) is raised
    err = exc.value
    from dolphin_ocr.errors import MemoryExhaustionError

    assert isinstance(err, MemoryExhaustionError)
    assert getattr(err, "error_code", None) == "DOLPHIN_011"
