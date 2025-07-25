#!/usr/bin/env python3
"""Debug script to test philosophical keywords extraction."""

import sys
import traceback
from pathlib import Path

# Add parent directory to path for imports (at end to avoid overriding system packages)
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from services.neologism_detector import NeologismDetector  # noqa: E402


def debug_keywords():
    try:
        detector = NeologismDetector()

        # Verify required attributes exist
        if not hasattr(detector, "philosophical_context_analyzer"):
            raise RuntimeError(
                "NeologismDetector missing philosophical_context_analyzer"
            )

        if not hasattr(detector, "debug_extract_philosophical_keywords"):
            raise RuntimeError("NeologismDetector missing debug method")

        print("✓ NeologismDetector initialized successfully")

        context = (
            "Das Bewusstsein und die Wirklichkeit sind zentrale "
            "Begriffe der Philosophie."
        )

        print(f"Context: {context}")
        # Use consistent tokenization if detector has a tokenization method
        print(f"Context words: {context.lower().split()}")

        # Check philosophical indicators
        indicators = detector.philosophical_context_analyzer.philosophical_indicators
        print(f"Philosophical indicators (first 20): {list(indicators)[:20]}")
        print(f"'philosophie' in indicators: {'philosophie' in indicators}")
        print(f"'bewusstsein' in indicators: {'bewusstsein' in indicators}")
        print(f"'wirklichkeit' in indicators: {'wirklichkeit' in indicators}")

        keywords = detector.debug_extract_philosophical_keywords(context)
        print(f"Extracted keywords: {keywords}")

    except Exception as e:
        print(f"Error during keyword debugging: {e}")
        print(f"Error type: {type(e).__name__}")
        print("Full traceback:")
        traceback.print_exc()
        print("Debug process failed, but continuing execution...")


if __name__ == "__main__":
    debug_keywords()
