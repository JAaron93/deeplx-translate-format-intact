#!/usr/bin/env python3
"""Simple test runner to check if imports work.

This script tests basic imports and functionality of the project modules.

Setup:
    For proper operation, ensure the project is installed in editable mode:
    $ pip install -e .

    Alternatively, set PYTHONPATH in your environment:
    $ export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    $ python run_single_test.py

    Or run with PYTHONPATH inline:
    $ PYTHONPATH=. python run_single_test.py
"""

import sys
import os

def test_basic_imports():
    """Test basic imports."""
    try:
        # Test models import
        from models.user_choice_models import (
            ChoiceType,
            ChoiceScope,
            TranslationContext,
            UserChoice,
            create_choice_id
        )
        print("✓ Successfully imported user choice models")
        
        # Test database import
        from database.choice_database import ChoiceDatabase
        print("✓ Successfully imported choice database")
        
        # Test a simple model creation
        context = TranslationContext(
            sentence_context="Test sentence",
            semantic_field="test",
            philosophical_domain="test",
            author="test",
            source_language="en",
            target_language="de"
        )
        print("✓ Successfully created TranslationContext")
        
        # Test choice creation
        choice = UserChoice(
            choice_id="test_id",
            neologism_term="test_term",
            choice_type=ChoiceType.TRANSLATE,
            choice_scope=ChoiceScope.SESSION,
            context=context,
            translation_result="test_translation"
        )
        print("✓ Successfully created UserChoice")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        raise   # or: sys.exit(1) and drop the bool plumbing

if __name__ == "__main__":
    print("Testing basic functionality...")

    # First, try to run tests with the current Python path
    try:
        success = test_basic_imports()
        if success:
            print("✓ All basic tests passed!")
        else:
            print("✗ Some tests failed!")
            sys.exit(1)
    except ImportError as e:
        print(f"Initial import failed: {e}")
        print("Attempting fallback: adding current directory to Python path...")
        print("Note: For production use, install the package with 'pip install -e .' instead")

        # Fallback: add current directory to Python path only when running as main script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Retry the tests with modified path
        try:
            success = test_basic_imports()
            if success:
                print("✓ All basic tests passed with fallback path!")
            else:
                print("✗ Some tests failed even with fallback!")
                sys.exit(1)
        except Exception as fallback_e:
            print(f"✗ Tests failed even with fallback: {fallback_e}")
            print("\nRecommended solutions:")
            print("1. Install the package in editable mode: pip install -e .")
            print("2. Set PYTHONPATH: export PYTHONPATH=\"${PYTHONPATH}:$(pwd)\"")
            print("3. Run with PYTHONPATH: PYTHONPATH=. python run_single_test.py")
            sys.exit(1)
