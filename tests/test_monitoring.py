from __future__ import annotations

import logging
import time

import pytest

from dolphin_ocr.monitoring import MonitoringService


def test_record_and_summary_counts():
    m = MonitoringService(window_seconds=60)
    m.record_operation("ocr", 100.0, success=True)
    m.record_operation("ocr", 200.0, success=False, error_code="DOLPHIN_006")
    m.record_operation("convert", 50.0, success=True)

    s = m.get_summary()
    ocr = s["ocr"]
    conv = s["convert"]
    assert isinstance(ocr, dict) and isinstance(conv, dict)
    assert int(ocr["count"]) == 2 and int(ocr["success"]) == 1
    assert int(conv["count"]) == 1 and int(conv["success"]) == 1
    # Validate average latency values
    assert isinstance(ocr["avg_ms"], float)
    assert ocr["avg_ms"] == pytest.approx(150.0, abs=1e-6)
    assert isinstance(conv["avg_ms"], float)
    assert conv["avg_ms"] == pytest.approx(50.0, abs=1e-6)
    err = s["error_rate"]
    assert isinstance(err, float) and 0.0 < err <= 1.0


def test_error_rate_window_pruning(monkeypatch):
    m = MonitoringService(window_seconds=1)
    m.record_operation("x", 10.0, success=False)

    # Advance time beyond window, ensure old event pruned
    orig_time = time.time
    monkeypatch.setattr(time, "time", lambda: orig_time() + 2)
    assert m.get_error_rate() == 0.0


def test_p95_latency_basic():
    m = MonitoringService(window_seconds=10)
    for d in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        m.record_operation("ocr", float(d), success=True)

    p95 = m.get_p95_latency("ocr")
    assert 90.0 <= p95 <= 100.0


def test_log_health_does_not_crash(caplog):
    caplog.set_level(logging.INFO)
    m = MonitoringService(window_seconds=10)
    m.record_operation("y", 5.0, success=True)
    m.log_health()
    assert any("health:" in rec.getMessage() for rec in caplog.records)


def test_p95_latency_edge_cases():
    m = MonitoringService(window_seconds=10)

    # Empty data
    assert m.get_p95_latency("unknown") == 0.0

    # Single sample
    m.record_operation("single", 50.0, success=True)
    assert m.get_p95_latency("single") == 50.0

    # Two samples
    m.record_operation("double", 10.0, success=True)
    m.record_operation("double", 20.0, success=True)
    p95 = m.get_p95_latency("double")
    assert 10.0 <= p95 <= 20.0

    # Odd number of samples (three)
    m.record_operation("triple", 10.0, success=True)
    m.record_operation("triple", 20.0, success=True)
    m.record_operation("triple", 30.0, success=True)
    assert m.get_p95_latency("triple") == 20.0
