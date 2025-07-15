#!/bin/bash
# Migration Validation Script for Black + Ruff
# =============================================
# This script validates the migration from Flake8 to Black + Ruff

set -e  # Exit on error

echo "==========================================="
echo "Black + Ruff Migration Validation Script"
echo "==========================================="
echo ""

# Check if pre-commit is installed
echo "1. Checking pre-commit installation..."
if ! command -v pre-commit &> /dev/null; then
    echo "   ❌ pre-commit is not installed!"
    echo "   Please install it with: pip install pre-commit"
    exit 1
else
    echo "   ✅ pre-commit is installed ($(pre-commit --version))"
fi
echo ""

# Check if Black is installed
echo "2. Checking Black installation..."
if ! command -v black &> /dev/null; then
    echo "   ❌ Black is not installed!"
    echo "   Please install it with: pip install black==25.1.0"
    exit 1
else
    echo "   ✅ Black is installed ($(black --version 2>&1 | head -n1))"
fi
echo ""

# Check if Ruff is installed
echo "3. Checking Ruff installation..."
if ! command -v ruff &> /dev/null; then
    echo "   ❌ Ruff is not installed!"
    echo "   Please install it with: pip install ruff==0.12.0"
    exit 1
else
    echo "   ✅ Ruff is installed ($(ruff --version))"
fi
echo ""

# Install pre-commit hooks
echo "4. Installing pre-commit hooks..."
pre-commit clean
pre-commit install
echo "   ✅ Pre-commit hooks installed"
echo ""

# Capture before statistics (if baseline exists)
echo "5. Capturing statistics..."
echo ""

# Count Python files
PY_FILE_COUNT=$(find . -name "*.py" -type f ! -path "./.venv/*" ! -path "./__pycache__/*" ! -path "./build/*" ! -path "./dist/*" | wc -l)
echo "   📊 Python files found: $PY_FILE_COUNT"

# Run pre-commit and capture results
echo ""
echo "6. Running pre-commit on all files..."
echo "   This may take a moment and might modify files..."
echo ""

# Create temporary file for output
TEMP_OUTPUT=$(mktemp) || {
    echo "❌ Failed to create temporary file"
    exit 1
}

# Run pre-commit and capture exit code
set +e  # Temporarily disable exit on error
pre-commit run --all-files 2>&1 | tee "$TEMP_OUTPUT"
EXIT_CODE=$?
set -e  # Re-enable exit on error

echo ""
echo "==========================================="
echo "Migration Validation Results"
echo "==========================================="
echo ""

# Analyze results
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All checks passed! No issues found."
else
    echo "⚠️  Some files were modified or issues were found."
    BLACK_MODS=$(grep -c -E "(black|Format Python code with Black).*(Failed|fixed)" "$TEMP_OUTPUT" || true)
    RUFF_MODS=$(grep -c -E "(ruff|Lint Python code with Ruff).*(Failed|fixed)" "$TEMP_OUTPUT" || true)
    # Count modifications by tool
    BLACK_MODS=$(grep -c "Format Python code with Black.*Failed" "$TEMP_OUTPUT" || true)
    RUFF_MODS=$(grep -c "Lint Python code with Ruff.*Failed" "$TEMP_OUTPUT" || true)

    if [ $BLACK_MODS -gt 0 ]; then
        echo "   📝 Black formatted files"
    fi

    if [ $RUFF_MODS -gt 0 ]; then
        echo "   🔧 Ruff fixed linting issues"
# Cleanup function
cleanup() {
    rm -f "$TEMP_OUTPUT"
}

# Set trap for cleanup
trap cleanup EXIT

# Create temporary file for output
TEMP_OUTPUT=$(mktemp)

# ... rest of script ...

# Clean up
rm -f "$TEMP_OUTPUT"

echo ""
echo "==========================================="
echo "Summary"
echo "==========================================="
echo ""
echo "Migration from Flake8 to Black + Ruff has been validated."
echo ""
echo "Tool versions:"
echo "  - Black: $(black --version 2>&1 | head -n1 | sed 's/black, //')"
echo "  - Ruff: $(ruff --version | sed 's/ruff //')"
echo "  - Line length: 88 characters (Black default)"
echo ""
echo "Replaced tools:"
echo "  ❌ flake8 → ✅ ruff"
echo "  ❌ isort → ✅ ruff (built-in)"
echo "  ❌ autoflake → ✅ ruff (F401, F841 rules)"
echo "  ❌ pyupgrade → ✅ ruff (UP rules)"
echo "  ❌ flake8-* plugins → ✅ ruff (D, B, C4, SIM rules)"
echo ""
echo "Next steps:"
echo "  1. Review any modified files"
echo "  2. Commit the changes"
echo "  3. Update CI/CD pipelines"
echo "  4. Notify team members"
echo ""
echo "==========================================="
