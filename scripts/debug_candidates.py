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


def find_and_validate_project_root():
    """Find and validate the project root directory.

    Returns:
        Path: Validated project root directory

    Raises:
        RuntimeError: If project root cannot be found or validated
    """
    # Script is in scripts/ subdirectory, so project root is parent.parent
    script_path = Path(__file__).resolve()
    potential_root = script_path.parent.parent

    # Expected directories and files that should exist in project root
    expected_dirs = ["services", "models", "scripts", "database", "config"]
    expected_files = ["app.py", "requirements.txt", "README.md"]

    # Validate that this looks like the correct project root
    missing_dirs = [d for d in expected_dirs if not (potential_root / d).is_dir()]
    missing_files = [f for f in expected_files if not (potential_root / f).is_file()]

    if missing_dirs or missing_files:
        error_msg = f"Project root validation failed for: {potential_root}\n"
        if missing_dirs:
            error_msg += f"Missing directories: {missing_dirs}\n"
        if missing_files:
            error_msg += f"Missing files: {missing_files}\n"
        error_msg += f"Script location: {script_path}\n"
        error_msg += "Please ensure the script is in the correct location relative to the project root."
        raise RuntimeError(error_msg)

    # Additional validation: check for services/neologism_detector.py specifically
    detector_module_path = potential_root / "services" / "neologism_detector.py"
    if not detector_module_path.is_file():
        raise RuntimeError(
            f"Critical file missing: {detector_module_path}\n"
            f"Project root: {potential_root}\n"
            "This script requires the NeologismDetector service to be available."
        )

    return potential_root


# Find and validate project root
try:
    project_root = find_and_validate_project_root()
    print(f"✓ Project root validated: {project_root}")
except RuntimeError as e:
    print("❌ Project root validation failed:")
    print(str(e))
    sys.exit(1)

# Add validated project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import NeologismDetector using the standard package structure
try:
    from services.neologism_detector import NeologismDetector

    print("✓ Successfully imported NeologismDetector")
except ImportError as e:
    print(f"❌ Failed to import NeologismDetector: {e}")
    print(f"Project root: {project_root}")
    print("Current working directory:", Path.cwd())
    print("Script location:", Path(__file__).resolve())
    print("\nThis indicates a problem with the project setup or dependencies.")
    print("Please ensure:")
    print("  1. All required dependencies are installed")
    print("  2. The project structure is intact")
    print("  3. No circular import issues exist")
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

    try:
        detector = NeologismDetector()
    except Exception as e:
        print(f"Error creating NeologismDetector instance: {e}")
        print("Cannot proceed with candidate extraction testing.")
        return

    text = (
        "Die Wirklichkeitsbewusstsein und Bewusstseinsphilosophie "
        "sind wichtige Konzepte."
    )

    print(f"Text: {text}")

    # Test the patterns manually
    # WARNING: These patterns may not match the actual implementation
    # Consider extracting from detector if patterns are accessible
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
    # Test the public candidate extraction method
    print("\nTesting public candidate extraction method:")
    try:
        candidates = detector.debug_extract_candidates(text)
        print(f"Candidates found: {len(candidates)}")

        if candidates:
            # Check if we got an error response
            if len(candidates) == 1 and "error" in candidates[0]:
                error_info = candidates[0]["error"]
                print(
                    f"Error during extraction: {error_info['type']}: {error_info['message']}"
                )
            else:
                for candidate in candidates:
                    term = candidate.get("term", "Unknown")
                    start_pos = candidate.get("start_pos", 0)
                    end_pos = candidate.get("end_pos", 0)
                    context = candidate.get("sentence_context", "No context")
                    print(f"  - '{term}' at {start_pos}-{end_pos}")
                    print(
                        f"    Context: {context[:100]}{'...' if len(context) > 100 else ''}"
                    )
        else:
            print("  No candidates detected by the extraction method.")

    except AttributeError as e:
        print(f"Error: Method 'debug_extract_candidates' not found: {e}")
        print("This may indicate that the NeologismDetector class needs to be updated.")
        print(
            "Please ensure the debug_extract_candidates method has been added to the class."
        )

    except Exception as e:
        print(f"Unexpected error during candidate extraction: {e}")
        print(f"Error type: {type(e).__name__}")
        print("The debug script will continue, but candidate extraction failed.")
        print("This may indicate an issue with dependencies or internal method logic.")


if __name__ == "__main__":
    debug_candidates()
