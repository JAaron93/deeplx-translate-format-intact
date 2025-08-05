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
    """Debug and verify philosophical keyword detection functionality.

    This function performs a comprehensive test of the NeologismDetector's
    philosophical keyword extraction capabilities by following this workflow:

    1. Instantiates a NeologismDetector instance
    2. Verifies that required attributes exist:
       - philosophical_context_analyzer
       - debug_extract_philosophical_keywords method
    3. Tests keyword extraction on a hard-coded German sentence:
       "Das Bewusstsein und die Wirklichkeit sind zentrale Begriffe der Philosophie."
       (Consciousness and reality are central concepts of philosophy.)
    4. Prints intermediate debugging results including:
       - The test context sentence
       - Context words after tokenization
       - Available philosophical indicators (first 20)
       - Presence checks for specific German philosophical terms
       - Final extracted keywords from the debug method
    5. Handles exceptions with detailed error reporting and traceback printing

    This script is intended for development and debugging purposes to verify
    that the philosophical keyword detection system is working correctly with
    German philosophical texts.

    Returns:
        None: This function performs debugging operations and prints results
              but does not return any value.

    Raises:
        None while the surrounding try/except block is in place.

    Note:
        All exceptions are presently caught and logged internally. If the
        error-handling strategy changes (for example, by re-raising after
        logging), revisit this section to document the new outward behaviour.
    """
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
