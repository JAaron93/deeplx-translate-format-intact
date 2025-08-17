#!/usr/bin/env python3
"""Debug script for compound word detection using public NeologismDetector API.

This script provides comprehensive debugging information for compound word
detection without directly accessing internal detector structures. It uses only
public APIs and can be configured via command line arguments or configuration
files.
"""

# Add project root to path before any imports to avoid E402 linting errors
import sys
from pathlib import Path

# Always work with an absolute, canonicalised path
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    # Append to give project path lowest precedence, avoiding override of
    # standard libraries
    sys.path.append(project_root_str)

# Standard library imports
import argparse  # noqa: E402
import json  # noqa: E402
import traceback  # noqa: E402
from typing import Any, Dict, List, Optional, Union  # noqa: E402

# Project imports
from services.neologism_detector import NeologismDetector  # noqa: E402


def get_default_categories() -> list[str]:
    """Get default compound categories."""
    return [
        "noun+noun",
        "adjective+noun",
        "verb+noun",
        "noun+verb",
        "adjective+adjective",
        "prefix+word",
        "word+suffix",
        "mixed",
    ]


def get_categories_from_config(config_path: str) -> list[str]:
    """Get compound categories from configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config.get("compound_categories", get_default_categories())
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        logger.warning(f"Could not load categories from {config_path}, using defaults")
        return get_default_categories()


def load_test_words(
    config_path: Optional[Union[str, Path]] = None,
    categories: Optional[List[str]] = None,
    auto_detect_categories: bool = True,
) -> List[Dict[str, Any]]:
    """Load test words from configuration file with flexible category selection.

    Args:
        config_path: Path to configuration file (default: config/debug_test_words.json)
        categories: List of categories to load (default: auto-detect or use defaults)
        auto_detect_categories: If True, automatically detect categories from config file

    Returns:
        List of word dictionaries with category information
    """
    if config_path is None:
        config_path = project_root / "config" / "debug_test_words.json"

    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Determine which categories to use
        if categories is None:
            if auto_detect_categories:
                categories = get_categories_from_config(config_path)
            else:
                categories = get_default_categories()

        # Flatten specified word categories into a single list
        test_words = []
        for category in categories:
            if category in config:
                for word_info in config[category]:
                    # Create a copy to avoid modifying the original
                    word_copy = word_info.copy()
                    word_copy["category"] = category
                    test_words.append(word_copy)
            else:
                print(f"⚠️  Warning: Category '{category}' not found in {config_path}")

        return test_words

    except FileNotFoundError:
        print(f"⚠️  Configuration file not found: {config_path}")
        print("Using fallback test words...")
        return [
            {
                "word": "Bewusstsein",
                "description": "Basic consciousness term",
                "context": "Das Bewusstsein ist ein zentraler Begriff.",
                "category": "fallback",
            },
            {
                "word": "Wirklichkeitsbewusstsein",
                "description": "Complex compound",
                "context": "Das Wirklichkeitsbewusstsein prägt unsere Wahrnehmung.",
                "category": "fallback",
            },
        ]
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing configuration file: {e}")
        return []


def format_debug_output(word_info: Dict[str, Any], debug_result: Dict[str, Any]) -> str:
    """Format debug analysis results for readable output.

    Args:
        word_info: Information about the test word
        debug_result: Debug analysis results from NeologismDetector

    Returns:
        Formatted string for display
    """
    if "error" in debug_result:
        return f"""
❌ ERROR analyzing '{word_info["word"]}':
   Type: {debug_result["error"]["type"]}
   Message: {debug_result["error"]["message"]}
"""

    basic = debug_result.get("basic_info", {})
    compound = debug_result.get("compound_analysis", {})
    morphological = debug_result.get("morphological_analysis", {})
    philosophical = debug_result.get("philosophical_context", {})
    confidence = debug_result.get("confidence_factors", {})
    assessment = debug_result.get("final_assessment", {})

    output = f"""
🔍 ANALYSIS: {basic.get("word", "Unknown")} ({word_info.get("category", "unknown")})
   Description: {word_info.get("description", "No description")}
   Context: {word_info.get("context", "No context")}

📊 BASIC INFO:
   Length: {basic.get("length", 0)} characters
   Capital letters: {basic.get("capital_count", 0)}
   In terminology: {basic.get("in_terminology", False)}

🔗 COMPOUND ANALYSIS:
   Is compound: {compound.get("is_compound", False)}
   Compound parts: {compound.get("compound_parts", [])}
   Compound pattern: {compound.get("compound_pattern", "None")}

🧬 MORPHOLOGICAL:
   Word length: {morphological.get("word_length", 0)}
   Syllables: {morphological.get("syllable_count", 0)}
   Morphemes: {morphological.get("morpheme_count", 0)}
   Root words: {morphological.get("root_words", [])}
   Prefixes: {morphological.get("prefixes", [])}
   Suffixes: {morphological.get("suffixes", [])}
   Structural complexity: {morphological.get("structural_complexity", 0.0):.3f}
   Morphological productivity: {morphological.get("morphological_productivity", 0.0):.3f}

🎓 PHILOSOPHICAL CONTEXT:
   Density: {philosophical.get("philosophical_density", 0.0):.3f}
   Semantic field: {philosophical.get("semantic_field", "unknown")}
   Domain indicators: {philosophical.get("domain_indicators", [])}
   Philosophical keywords: {philosophical.get("philosophical_keywords", [])}
   Surrounding terms: {philosophical.get("surrounding_terms", [])}
   Conceptual clusters: {philosophical.get("conceptual_clusters", [])}

⚖️  CONFIDENCE FACTORS:
   Morphological complexity: {confidence.get("morphological_complexity", 0.0):.3f}
   Compound structure score: {confidence.get("compound_structure_score", 0.0):.3f}
   Context density: {confidence.get("context_density", 0.0):.3f}
   Philosophical indicators: {confidence.get("philosophical_indicators", 0.0):.3f}
   Semantic coherence: {confidence.get("semantic_coherence", 0.0):.3f}
   Rarity score: {confidence.get("rarity_score", 0.0):.3f}
   Pattern productivity: {confidence.get("pattern_productivity", 0.0):.3f}
   Semantic transparency: {confidence.get("semantic_transparency", 0.0):.3f}

🎯 FINAL ASSESSMENT:
   Is neologism: {assessment.get("is_neologism", False)}
   Confidence score: {confidence.get("weighted_score", 0.0):.3f}
   Threshold: {assessment.get("threshold", 0.0):.3f}
   Neologism type: {assessment.get("neologism_type", "None")}
"""
    return output


def debug_compound_detection(
    test_words: Optional[List[str]] = None,
    config_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    filter_category: Optional[str] = None,
    categories: Optional[List[str]] = None,
) -> None:
    """Debug compound word detection using public NeologismDetector API.

    Args:
        test_words: List of words to test. If None, loads from config.
        config_path: Path to configuration file.
        verbose: Whether to show detailed output.
        filter_category: Only test words from this category (legacy, use categories instead).
        categories: List of categories to load and test.
    """
    print("🐬 Dolphin OCR Translate - Compound Word Debug Analysis")
    print("=" * 60)

    try:
        # Initialize detector
        print("🔧 Initializing NeologismDetector...")
        detector = NeologismDetector()
        print("✅ Detector initialized successfully")

        # Load test words
        if test_words:
            word_list = [
                {
                    "word": word,
                    "description": "Command line input",
                    "context": word,
                    "category": "manual",
                }
                for word in test_words
            ]
        else:
            print("📖 Loading test words from configuration...")

            # Determine which categories to load
            load_categories = None
            if categories:
                load_categories = categories
                print(f"🎯 Loading specific categories: {', '.join(categories)}")
            elif filter_category:
                load_categories = [filter_category]
                print(f"🔍 Loading single category: {filter_category}")

            word_list = load_test_words(config_path, categories=load_categories)

        # Legacy filter support (for backward compatibility)
        if filter_category and not categories:
            word_list = [w for w in word_list if w.get("category") == filter_category]
            print(f"🔍 Filtered to category: {filter_category}")

        print(f"📝 Testing {len(word_list)} words...")
        print()

        # Analyze each word
        for i, word_info in enumerate(word_list, 1):
            word = word_info["word"]
            context = word_info.get("context", "")

            print(f"[{i}/{len(word_list)}] Analyzing: {word}")

            try:
                # Use public debug API
                debug_result = detector.debug_analyze_word(word, context)

                if verbose:
                    print(format_debug_output(word_info, debug_result))
                else:
                    # Compact output
                    compound = debug_result.get("compound_analysis", {})
                    assessment = debug_result.get("final_assessment", {})

                    print(f"   Compound: {compound.get('is_compound', False)}")
                    print(f"   Neologism: {assessment.get('is_neologism', False)}")
                    confidence_factors = debug_result.get("confidence_factors", {})
                    score = confidence_factors.get("weighted_score", 0.0)
                    print(f"   Score: {score:.3f}")
                    print()

            except Exception as e:
                print(f"❌ Error analyzing '{word}': {e}")
                if verbose:
                    traceback.print_exc()
                print()

        print("🎉 Analysis complete!")

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


def get_available_categories(config_path: Optional[str] = None) -> list[str]:
    """Get available categories for command line argument choices.

    Args:
        config_path: Path to configuration file

    Returns:
        List of available category names
    """
    if config_path is None:
        config_path = project_root / "config" / "debug_test_words.json"

    try:
        return get_categories_from_config(config_path)
    except Exception:
        return get_default_categories()


def main():
    """Main entry point with command line argument parsing."""
    # First pass: parse config argument to determine available categories
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config", help="Path to configuration file with test words"
    )
    try:
        pre_args, _ = pre_parser.parse_known_args()
        config_path = pre_args.config
    except SystemExit:
        # Handle case where argument parsing fails
        config_path = None

    if config_path is None:
        config_path = project_root / "config" / "debug_test_words.json"

    available_categories = get_available_categories(config_path)

    parser = argparse.ArgumentParser(
        description="Debug compound word detection using NeologismDetector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/debug_compound.py
    # Default: detailed analysis of all categories

  python scripts/debug_compound.py --words Bewusstsein Weltanschauung
    # Analyze specific words with detailed output

  python scripts/debug_compound.py --category {available_categories[0] if available_categories else "compound_words"} --compact
    # Analyze one category with compact output

  python scripts/debug_compound.py --config custom_words.json --verbose
    # Use custom config with explicit detailed output

  python scripts/debug_compound.py --categories {" ".join(available_categories[:2]) if len(available_categories) >= 2 else "compound_words simple_words"} --compact
    # Multiple categories with compact output

Output modes:
  (default)  Detailed analysis with full information
  --verbose  Explicit detailed analysis (same as default)
  --compact  Compact output with only essential information

Available categories: {", ".join(available_categories)}
        """,
    )

    parser.add_argument(
        "--words", nargs="+", help="Specific words to analyze (overrides config file)"
    )
    parser.add_argument("--config", help="Path to configuration file with test words")
    parser.add_argument(
        "--category",
        choices=available_categories,
        help=f"Filter to specific word category. Available: {', '.join(available_categories)}",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=available_categories,
        help=f"Select multiple categories to analyze. Available: {', '.join(available_categories)}",
    )

    # Create mutually exclusive group for output format
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed analysis output with full information",
    )
    output_group.add_argument(
        "--compact",
        action="store_true",
        help="Show compact output with only essential information",
    )

    args = parser.parse_args()

    # Determine output mode (default to verbose if neither specified)
    if args.compact:
        verbose = False
    elif args.verbose:
        verbose = True
    else:
        # Default behavior: verbose mode for better user experience
        verbose = True

    # Determine categories to use
    selected_categories = None
    if args.categories:
        selected_categories = args.categories
    elif args.category:
        # For backward compatibility, convert single category to list
        selected_categories = [args.category]

    debug_compound_detection(
        test_words=args.words,
        config_path=args.config,
        verbose=verbose,
        filter_category=args.category if not args.categories else None,
        categories=selected_categories,
    )


if __name__ == "__main__":
    main()
