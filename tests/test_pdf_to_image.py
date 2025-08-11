import pytest

from dolphin_ocr.pdf_to_image import PDFToImageConverter
from tests.utils_dolphin import get_sample_pdfs, load_asset_bytes


@pytest.mark.parametrize("dpi,fmt", [(300, "PNG"), (200, "JPEG")])
def test_convert_pdf_to_images_roundtrip(tmp_path, dpi, fmt):
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
    with pytest.raises(FileNotFoundError):
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
    with pytest.raises(Exception):
        converter.convert_pdf_to_images(bad)

