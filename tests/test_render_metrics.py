from ui.gradio_interface import _render_metrics


def test_render_metrics_formats_known_fields() -> None:
    metrics = {
        "ocr_conf": 0.9123,
        "layout_similarity": 0.803,
        "text_accuracy": 0.977,
    }
    s = _render_metrics(metrics)
    assert "OCR confidence: 0.91" in s
    assert "Layout score: 0.80" in s
    assert "Text accuracy: 0.98" in s


def test_render_metrics_fallback_to_json() -> None:
    # No known keys present
    metrics = {"foo": 1, "bar": 2}
    s = _render_metrics(metrics)
    # Should return compact JSON string
    assert s.startswith("{") and s.endswith("}")
