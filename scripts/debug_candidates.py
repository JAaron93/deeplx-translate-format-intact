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

    Configuration Options:
    The script supports flexible project validation through multiple methods:

    1. Environment Variables:
       $ export PROJECT_EXPECTED_DIRS="services,models,scripts"
       $ export PROJECT_EXPECTED_FILES="app.py,requirements.txt"
       $ export PROJECT_CRITICAL_FILES="services/neologism_detector.py"
       $ python debug_candidates.py

    2. Configuration File:
       Create scripts/project_validation_config.json with custom settings
       (see project_validation_config.json for example)

    3. Programmatic Configuration:
       Modify the script to pass custom parameters to find_and_validate_project_root()

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

import os
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure the project root is in the Python path for proper imports
# This allows the script to be run from any location while maintaining
# proper package structure and avoiding hardcoded path manipulations


def load_project_validation_config(config_path: Optional[Path] = None) -> dict:
    """Load project validation configuration from file or environment.

    Args:
        config_path: Optional path to configuration file

    Returns:
        dict: Configuration with expected_dirs, expected_files, and critical_files
    """
    # Default configuration
    default_config = {
        "expected_dirs": ["services", "models", "scripts", "database", "config"],
        "expected_files": ["app.py", "requirements.txt", "README.md"],
        "critical_files": ["services/neologism_detector.py"],
    }

    # Try to load from environment variables
    env_dirs = os.getenv("PROJECT_EXPECTED_DIRS")
    env_files = os.getenv("PROJECT_EXPECTED_FILES")
    env_critical = os.getenv("PROJECT_CRITICAL_FILES")

    if env_dirs:
        default_config["expected_dirs"] = [d.strip() for d in env_dirs.split(",")]
    if env_files:
        default_config["expected_files"] = [f.strip() for f in env_files.split(",")]
    if env_critical:
        default_config["critical_files"] = [f.strip() for f in env_critical.split(",")]

    # Try to load from configuration file
    if config_path and config_path.exists():
        try:
            import json

            with open(config_path) as f:
                file_config = json.load(f)
                default_config.update(file_config)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    return default_config


def find_and_validate_project_root(
    expected_dirs: Optional[list[str]] = None,
    expected_files: Optional[list[str]] = None,
    critical_files: Optional[list[str]] = None,
    config_path: Optional[Path] = None,
    strict_validation: bool = True,
) -> Path:
    """Find and validate the project root directory.

    Args:
        expected_dirs: List of directories that should exist in project root
        expected_files: List of files that should exist in project root
        critical_files: List of critical files (relative paths) that must exist
        config_path: Optional path to configuration file
        strict_validation: If False, only warn about missing items instead of raising errors

    Returns:
        Path: Validated project root directory

    Raises:
        RuntimeError: If project root cannot be found or validated (when strict_validation=True)
    """
    # Load configuration if parameters not provided
    if not any([expected_dirs, expected_files, critical_files]):
        config = load_project_validation_config(config_path)
        expected_dirs = expected_dirs or config["expected_dirs"]
        expected_files = expected_files or config["expected_files"]
        critical_files = critical_files or config["critical_files"]

    # Use defaults if still None
    expected_dirs = expected_dirs or []
    expected_files = expected_files or []
    critical_files = critical_files or []

    # Script is in scripts/ subdirectory, so project root is parent.parent
    script_path = Path(__file__).resolve()
    potential_root = script_path.parent.parent

    # Validate that this looks like the correct project root
    missing_dirs = [d for d in expected_dirs if not (potential_root / d).is_dir()]
    missing_files = [f for f in expected_files if not (potential_root / f).is_file()]
    missing_critical = [f for f in critical_files if not (potential_root / f).is_file()]

    if missing_dirs or missing_files or missing_critical:
        error_msg = f"Project root validation failed for: {potential_root}\n"
        if missing_dirs:
            error_msg += f"Missing directories: {missing_dirs}\n"
        if missing_files:
            error_msg += f"Missing files: {missing_files}\n"
        if missing_critical:
            error_msg += f"Missing critical files: {missing_critical}\n"
        error_msg += f"Script location: {script_path}\n"
        error_msg += "Please ensure the script is in the correct location relative to the project root."

        if strict_validation:
            raise RuntimeError(error_msg)
        else:
            print(f"Warning: {error_msg}")

    return potential_root


# Find and validate project root with flexible configuration
try:
    # Option 1: Use default configuration (backward compatible)
    project_root = find_and_validate_project_root()

    # Option 2: Use configuration file (uncomment to test)
    # config_file = Path(__file__).parent / "project_validation_config.json"
    # project_root = find_and_validate_project_root(config_path=config_file)

    # Option 3: Use custom parameters (uncomment to test)
    # project_root = find_and_validate_project_root(
    #     expected_dirs=["services", "models"],
    #     expected_files=["app.py"],
    #     critical_files=["services/neologism_detector.py"],
    #     strict_validation=False  # Only warn instead of failing
    # )

    print(f"✓ Project root validated: {project_root}")
except RuntimeError as e:
    print("❌ Project root validation failed:")
    print(str(e))
    sys.exit(1)

# Add validated project root to Python path (at end to avoid overriding system packages)
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

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


def get_compound_patterns_from_detector(detector: NeologismDetector) -> list[str]:
    """Extract compound patterns from detector instance with fallback.

    Args:
        detector: NeologismDetector instance

    Returns:
        list[str]: List of regex patterns for compound detection
    """
    # Try to get patterns from detector instance
    if hasattr(detector, "get_compound_patterns"):
        try:
            patterns = detector.get_compound_patterns()
            print("✓ Using patterns from detector.get_compound_patterns()")
            return patterns
        except Exception as e:
            print(f"⚠️  Failed to get patterns from detector: {e}")

    # Check if detector has the patterns as an attribute or method
    if hasattr(detector, "compound_patterns"):
        try:
            patterns = detector.compound_patterns
            print("✓ Using patterns from detector.compound_patterns attribute")
            return patterns
        except Exception as e:
            print(f"⚠️  Failed to get patterns from detector attribute: {e}")

    # Fallback to hardcoded patterns (may be outdated)
    print("⚠️  Warning: Using fallback hardcoded patterns - these may be outdated!")
    print(
        "   Consider updating the NeologismDetector to expose patterns via get_compound_patterns()"
    )

    fallback_patterns = [
        # CapitalizedCompounds (may be outdated)
        r"\b[A-ZÄÖÜ][a-zäöüß]{5,}(?:[A-ZÄÖÜ][a-zäöüß]+)+\b",
        # Long capitalized words (may be outdated)
        r"\b[A-ZÄÖÜ][a-zäöüß]{10,}\b",
        # linked compounds (may be outdated)
        r"\b[a-zäöüß]+(?:s|n|es|en|er|e|ns|ts)[a-zäöüß]{4,}\b",
        # abstract suffixes (may be outdated)
        r"\b[a-zäöüß]+(?:heit|keit|ung|schaft|tum|nis|sal|ismus|ität|logie|sophie)\b",
    ]

    return fallback_patterns


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

    # Get patterns dynamically from detector with fallback
    print("\n=== Pattern Extraction ===")
    compound_patterns = get_compound_patterns_from_detector(detector)

    print(f"\nTotal patterns extracted: {len(compound_patterns)}")
    print("\n=== Manual Pattern Testing ===")
    total_matches = 0
    for i, pattern in enumerate(compound_patterns):
        print(f"\nPattern {i+1}: {pattern}")
        matches = re.findall(pattern, text)
        print(f"Matches: {matches}")
        total_matches += len(matches)

    print(f"\nTotal matches across all patterns: {total_matches}")
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
