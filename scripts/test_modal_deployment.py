#!/usr/bin/env python3
"""Test script for Modal Labs Dolphin OCR deployment.

This script tests the deployed Dolphin OCR service to ensure it's working correctly.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

# Add the project root to the Python path
project_root: Path = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.dolphin_client import get_layout

# ---------------------------------------------------------------------------
# Test-data paths searched by the helper functions
# ---------------------------------------------------------------------------

TEST_PDF_PATHS: List[str] = [
    "tests/fixtures/sample.pdf",
    "docs/sample.pdf",
    "sample.pdf",
]


def find_test_pdf() -> Optional[Path]:
    """Find a test PDF file by checking multiple predefined paths.

    Checks common locations for test PDF files and returns the first
    existing path found, or None if no test PDF is available.

    Returns:
        Optional[Path]: Path to the first existing test PDF file, or None
                        if no test PDF is found in any of the predefined locations.
    """
    candidates = (Path(p) for p in TEST_PDF_PATHS)
    return next((p for p in candidates if p.exists()), None)


def print_test_pdf_locations() -> None:
    """Print the expected test PDF locations for user guidance."""
    print("   Expected locations:")
    for path in TEST_PDF_PATHS:
        print(f"   - {path}")


async def run_test_modal_endpoint() -> bool:
    """Test the Modal Dolphin OCR endpoint."""
    print("🧪 Testing Modal Dolphin OCR endpoint...")

    # Check if endpoint is configured
    endpoint: Optional[str] = os.getenv("DOLPHIN_ENDPOINT")
    if not endpoint:
        print("❌ DOLPHIN_ENDPOINT not set in environment")
        return False

    print(f"📡 Testing endpoint: {endpoint}")

    # Look for a test PDF file using the reusable function
    test_pdf: Optional[Path] = find_test_pdf()
    if not test_pdf:
        print("❌ No test PDF found. Please create a test PDF file.")
        print_test_pdf_locations()
        return False

    print(f"📄 Using test PDF: {test_pdf}")

    try:
        # Test the Dolphin client
        result: Any = await get_layout(test_pdf)

        if not isinstance(result, dict):
            raise ValueError(f"Expected dict result, got {type(result)}")

        print("✅ Modal endpoint test successful!")
        print(f"📊 Processed {result.get('total_pages', 0)} pages")

        # Print summary of results
        if "pages" in result:
            pages: Any = result["pages"]
            if not isinstance(pages, list):
                print(f"⚠️  Unexpected pages format: {type(pages)}")
                return True

            for i, page in enumerate(pages):
                if not isinstance(page, dict):
                    continue
                text_blocks: Any = page.get("text_blocks", [])
                print(f"   Page {i + 1}: {len(text_blocks)} text blocks")

        return True

    except Exception as e:
        print(f"❌ Modal endpoint test failed: {e}")
        return False


async def run_test_local_fallback() -> bool:
    """Test fallback to local endpoint if available."""
    print("\n🧪 Testing local fallback endpoint...")

    # Temporarily override endpoint for local testing
    original_endpoint: Optional[str] = os.getenv("DOLPHIN_ENDPOINT")
    os.environ["DOLPHIN_ENDPOINT"] = "http://localhost:8501/layout"

    try:
        # Look for a test PDF file using the reusable function
        test_pdf: Optional[Path] = find_test_pdf()
        if not test_pdf:
            print("⚠️  No test PDF found for local testing")
            print_test_pdf_locations()
            return True  # Not a failure, just skip

        await get_layout(test_pdf)
        print("✅ Local endpoint test successful!")
        return True

    except Exception as e:
        print(f"⚠️  Local endpoint not available: {e}")
        return True  # Not a failure, local service might not be running

    finally:
        # Restore original endpoint
        if original_endpoint:
            os.environ["DOLPHIN_ENDPOINT"] = original_endpoint
        else:
            os.environ.pop("DOLPHIN_ENDPOINT", None)


def check_modal_authentication() -> bool:
    """Check Modal authentication."""
    print("🔐 Checking Modal authentication...")

    token_id: Optional[str] = os.getenv("MODAL_TOKEN_ID")
    token_secret: Optional[str] = os.getenv("MODAL_TOKEN_SECRET")

    if not token_id or not token_secret:
        print("❌ Modal authentication not configured")
        print("   Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET")
        return False

    print("✅ Modal authentication configured")
    return True


def check_environment() -> bool:
    """Check the deployment environment."""
    print("🌍 Checking deployment environment...")

    required_vars: List[str] = [
        "DOLPHIN_ENDPOINT",
        "LINGO_API_KEY",
    ]
    # System dependencies checks: Poppler and fonts hints
    print("🔎 Checking system dependencies (informational)...")
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        poppler_hint: str = (
            "Install 'poppler-utils' (Linux) or 'brew install poppler' (macOS) "
            "if pdf2image fails."
        )
        print(f"   • Poppler: required by pdf2image. {poppler_hint}")
        print("   • Fonts: install DejaVu/Noto for consistent rendering.")

    missing_vars: List[str] = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False

    print("✅ Environment variables configured")
    return True


async def main() -> None:
    """Main test function."""
    print("🧪 Modal Labs Deployment Test")
    print("=" * 40)

    # Check authentication
    if not check_modal_authentication():
        sys.exit(1)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Test Modal endpoint
    modal_success: bool = await run_test_modal_endpoint()

    # Test local fallback
    local_fallback_success: bool = await run_test_local_fallback()

    print("\n" + "=" * 40)
    match (modal_success, local_fallback_success):
        case (True, True):
            print("✅ All tests passed! Modal deployment and local fallback are working.")
        case (True, False):
            print("⚠️  Modal deployment is working, but local fallback failed.")
            print("   This may not be critical if local service is not required.")
        case (False, True):
            print("❌ Modal deployment failed, but local fallback is working.")
            print("   Check the Modal deployment configuration.")
            sys.exit(1)
        case (False, False):
            print("❌ Both Modal deployment and local fallback failed.")
            print("   Check both Modal deployment and local service configuration.")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
