#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

# Targeted smoke test for the OCR service to avoid unrelated suite dependencies;
# coverage is enforced by pytest-service.ini
pytest -q -c pytest-service.ini tests/test_dolphin_ocr_service.py
