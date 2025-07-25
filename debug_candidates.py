#!/usr/bin/env python3
"""Debug script for testing neologism candidate extraction functionality.

This script provides debugging capabilities for the neologism detection system,
specifically focusing on the candidate extraction process. It tests both manual
regex patterns and the actual candidate extraction methods to help developers
understand and troubleshoot the detection logic.

Purpose:
    - Test regex patterns used for identifying potential neologisms
    - Debug the candidate extraction process
    - Validate compound word detection patterns
    - Analyze extraction results for German philosophical texts

Usage:
    Run this script directly to test candidate extraction on sample text:

    $ python debug_candidates.py

    The script will:
    1. Display the test text being analyzed
    2. Test individual regex patterns manually
    3. Run the actual candidate extraction method
    4. Show detailed results including positions and matches

Example Output:
    Text: Die Wirklichkeitsbewusstsein und Bewusstseinsphilosophie sind wichtige Konzepte.

    Testing patterns manually:
    Pattern 1: \b[A-ZÄÖÜ][a-zäöüß]{5,}(?:[A-ZÄÖÜ][a-zäöüß]+)+\b
    Matches: ['Wirklichkeitsbewusstsein', 'Bewusstseinsphilosophie']

    Candidates found: 2
      - Wirklichkeitsbewusstsein at 4-28
      - Bewusstseinsphilosophie at 33-56

Dependencies:
    - services.neologism_detector: For NeologismDetector class
    - re: For regex pattern testing

Note:
    This script is intended for development and debugging purposes.
    It uses private methods (_extract_candidates_regex) for testing.
"""

import re
import sys
from pathlib import Path

# Ensure the project root is in the Python path for proper imports
# This allows the script to be run from any location while maintaining
# proper package structure and avoiding hardcoded path manipulations
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from services.neologism_detector import NeologismDetector
except ImportError as e:
    print(f"Error importing NeologismDetector: {e}")
    print("Make sure you're running this script from the project root directory.")
    print("Current working directory:", Path.cwd())
    print("Script location:", Path(__file__).resolve())
    print("\nTrying alternative import method...")

    # Alternative: try importing without going through services.__init__.py
    try:
        import importlib.util
        neologism_detector_path = project_root / "services" / "neologism_detector.py"
        if not neologism_detector_path.exists():
            print(f"NeologismDetector module not found at: {neologism_detector_path}")
            sys.exit(1)

        spec = importlib.util.spec_from_file_location("neologism_detector", neologism_detector_path)
        if spec is None or spec.loader is None:
            print("Failed to create module spec")
            sys.exit(1)

        neologism_detector_module = importlib.util.module_from_spec(spec)
        sys.modules["neologism_detector"] = neologism_detector_module
        spec.loader.exec_module(neologism_detector_module)
        NeologismDetector = neologism_detector_module.NeologismDetector
        print("Successfully imported using alternative method")
    except Exception as alt_e:
        print(f"Alternative import also failed: {alt_e}")
        sys.exit(1)


def debug_candidates():
    """Debug the candidate extraction functionality of the NeologismDetector.

    This function tests both manual regex patterns and the actual candidate
    extraction method to help identify and troubleshoot issues in the
    neologism detection process.

    The function:
    1. Creates a NeologismDetector instance
    2. Tests predefined regex patterns against sample text
    3. Runs the actual _extract_candidates_regex method with error handling
    4. Displays detailed results for analysis

    Error Handling:
        The function includes comprehensive error handling for the private method call:
        - AttributeError: Method not found or API changed
        - TypeError: Invalid arguments or method signature changed
        - ValueError/IndexError: Data processing errors
        - General exceptions: Unexpected errors with detailed reporting

        All errors are caught and reported without stopping the script execution.

    Note:
        This function uses the private _extract_candidates_regex method
        for debugging purposes only. The error handling ensures robustness
        even if the internal API changes.
    """
    detector = NeologismDetector()

    text = (
        "Die Wirklichkeitsbewusstsein und Bewusstseinsphilosophie "
        "sind wichtige Konzepte."
    )

    print(f"Text: {text}")

    # Test the patterns manually
    compound_patterns = [
        # CapitalizedCompounds
        r"\b[A-ZÄÖÜ][a-zäöüß]{5,}(?:[A-ZÄÖÜ][a-zäöüß]+)+\b",
        # linked compounds
        r"\b[a-zäöüß]+(?:s|n|es|en|er|e|ns|ts)[a-zäöüß]{4,}\b",
        # abstract suffixes
        r"\b[a-zäöüß]+(?:heit|keit|ung|schaft|tum|nis|sal|ismus|ität|logie|sophie)\b",
    ]

    print("\nTesting patterns manually:")
    for i, pattern in enumerate(compound_patterns):
        print(f"Pattern {i+1}: {pattern}")
        matches = re.findall(pattern, text)
        print(f"Matches: {matches}")

    # Test the actual method with error handling
    print("\nTesting actual candidate extraction method:")
    try:
        candidates = detector._extract_candidates_regex(text)
        print(f"Candidates found: {len(candidates)}")

        if candidates:
            for candidate in candidates:
                print(f"  - {candidate.term} at {candidate.start_pos}-{candidate.end_pos}")
        else:
            print("  No candidates detected by the extraction method.")

    except AttributeError as e:
        print(f"Error: Method '_extract_candidates_regex' not found or changed: {e}")
        print("This may indicate that the NeologismDetector API has been modified.")
        print("Please check the NeologismDetector class for available methods.")

    except TypeError as e:
        print(f"Error: Invalid arguments passed to '_extract_candidates_regex': {e}")
        print("This may indicate that the method signature has changed.")
        print("Check if the method expects different parameters.")

    except (ValueError, IndexError) as e:
        print(f"Error: Invalid data or processing error in candidate extraction: {e}")
        print("This may indicate an issue with the input text format or internal logic.")

    except Exception as e:
        print(f"Unexpected error during candidate extraction: {e}")
        print(f"Error type: {type(e).__name__}")
        print("The debug script will continue, but candidate extraction failed.")
        print("This may indicate an issue with dependencies or internal method logic.")

if __name__ == "__main__":
    debug_candidates()
