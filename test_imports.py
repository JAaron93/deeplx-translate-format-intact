#!/usr/bin/env python3
"""Test script to check if imports work."""

import sys
import traceback

import importlib

def test_import(module_name, from_module=None):
    """Test importing a module."""
    try:
        if from_module:
            module = importlib.import_module(from_module)
            getattr(module, module_name)
            print(f"✓ Successfully imported {module_name} from {from_module}")
        else:
            importlib.import_module(module_name)
            print(f"✓ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to import {module_name}: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing imports...")
    
    # Test basic imports
    standard_modules = ["pytest", "sqlite3", "json", "datetime"]
    project_imports = [
        ("ChoiceDatabase", "database.choice_database"),
        ("UserChoice", "models.user_choice_models")
    ]
    
    success_count = 0
    total_count = 0
    
    for module in standard_modules:
        total_count += 1
        if test_import(module):            success_count += 1
    
    for module_name, from_module in project_imports:
        total_count += 1
        if test_import(module_name, from_module):
            success_count += 1
    
    print(f"Import testing complete: {success_count}/{total_count} successful")
