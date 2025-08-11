import pytest
from dolphin_ocr.config import (
    DolphinConfig,
    PerformanceConfig,
    AlertThresholds,
    QualityThresholds,
    ConfigurationManager,
)


def test_dolphin_config_validation(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "testtoken")
    monkeypatch.setenv(
        "DOLPHIN_MODAL_ENDPOINT", "https://example.com/endpoint"
    )
    cfg = DolphinConfig()
    assert cfg.hf_token == "testtoken"
    assert cfg.modal_endpoint.startswith("https://")

@pytest.mark.parametrize("dpi", [150, 300, 600])

def test_performance_config_valid_dpi(monkeypatch, dpi):
    monkeypatch.setenv("PDF_DPI", str(dpi))
    perf = PerformanceConfig()
    assert perf.dpi == dpi


def test_alert_thresholds_ranges(monkeypatch):
    monkeypatch.setenv("ALERT_ERROR_RATE_THRESHOLD", "0.1")
    monkeypatch.setenv("ALERT_ERROR_RATE_WINDOW", "600")
    monkeypatch.setenv("ALERT_CRITICAL_ERROR_THRESHOLD", "5")
    monkeypatch.setenv("ALERT_LATENCY_MULTIPLIER", "2.0")
    monkeypatch.setenv("ALERT_QUOTA_WARNING_THRESHOLD", "0.9")
    alert = AlertThresholds()
    assert 0.01 <= alert.error_rate_threshold <= 0.5
    assert 60 <= alert.error_rate_window <= 3600
    assert 1 <= alert.critical_error_threshold <= 20
    assert 1.1 <= alert.latency_threshold_multiplier <= 3.0
    assert 0.5 <= alert.quota_warning_threshold <= 0.95


def test_quality_thresholds_ranges(monkeypatch):
    monkeypatch.setenv("MIN_OCR_CONFIDENCE", "0.8")
    monkeypatch.setenv("MIN_TRANSLATION_CONFIDENCE", "0.7")
    monkeypatch.setenv("MIN_LAYOUT_PRESERVATION_SCORE", "0.75")
    monkeypatch.setenv("MIN_OVERALL_QUALITY_SCORE", "0.85")
    qual = QualityThresholds()
    assert 0.1 <= qual.min_ocr_confidence <= 1.0
    assert 0.1 <= qual.min_translation_confidence <= 1.0
    assert 0.1 <= qual.min_layout_preservation_score <= 1.0
    assert 0.1 <= qual.min_overall_quality_score <= 1.0


def test_configuration_manager_summary(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "abc")
    monkeypatch.setenv(
        "DOLPHIN_MODAL_ENDPOINT",
        "https://example.com/endpoint",
    )
    cm = ConfigurationManager()
    summary = cm.get_environment_summary()
    assert summary["hf_token_configured"] is True
    assert summary["dolphin_endpoint"].startswith("https://")

