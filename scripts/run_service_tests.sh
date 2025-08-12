#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

# Targeted smoke test for the OCR service to avoid unrelated suite dependencies;
# coverage is enforced by pytest-service.ini

# Verify pytest is available before attempting to run it
if command -v pytest >/dev/null 2>&1; then
  PYTEST_CMD="pytest"
elif command -v python >/dev/null 2>&1 && python -c 'import pytest' >/dev/null 2>&1; then
  PYTEST_CMD="python -m pytest"
else
  printf "%s\n%s\n%s\n" \
    "Error: pytest is not installed or not found in PATH." \
    "Install with:" \
    "  pip install -U pytest pytest-cov" >&2
  exit 1
fi

# Ensure pytest-cov is available since pytest-service.ini enables coverage
if ! python -c 'import pytest_cov' >/dev/null 2>&1; then
  printf "%s\n%s\n" \
    "Error: pytest-cov is not installed." \
    "Install with:  pip install -U pytest-cov" >&2
  exit 1
fi

"$PYTEST_CMD" -q -c pytest-service.ini tests/test_dolphin_ocr_service.py
