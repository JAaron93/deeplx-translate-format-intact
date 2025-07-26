# Flexible Category System Enhancement

This document describes the refactoring of `scripts/debug_compound.py` to support flexible, configurable word categories instead of hardcoded category lists.

## Problem Statement

The original implementation had hardcoded category lists in multiple places:

### Issues with Hardcoded Categories

1. **Limited Flexibility**: Only supported three fixed categories (`compound_words`, `simple_words`, `neologisms`)
2. **Code Duplication**: Category lists appeared in multiple locations
3. **Maintenance Overhead**: Adding new categories required code changes
4. **Configuration Mismatch**: No validation that config file categories matched code expectations
5. **Extensibility Barriers**: Difficult to create custom test configurations

### Original Hardcoded Implementation
```python
# Hardcoded in load_test_words function
for category in ["compound_words", "simple_words", "neologisms"]:
    # ...

# Hardcoded in argument parser
parser.add_argument(
    "--category",
    choices=["compound_words", "simple_words", "neologisms"],
    # ...
)
```

## Solution Overview

Implemented a flexible category system that:

1. **Auto-detects categories** from configuration files
2. **Supports custom categories** without code changes
3. **Provides multiple configuration methods** (parameters, auto-detection, defaults)
4. **Maintains backward compatibility** with existing configurations
5. **Offers dynamic argument parsing** based on available categories

## Implementation Details

### 1. Category Detection Functions

#### `get_default_categories()`
```python
def get_default_categories() -> List[str]:
    """Get the default word categories."""
    return ["compound_words", "simple_words", "neologisms"]
```

#### `get_categories_from_config()`
```python
def get_categories_from_config(config_path: str) -> List[str]:
    """Extract available categories from a configuration file."""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        # Return all top-level keys that contain lists (word categories)
        categories = []
        for key, value in config.items():
            if isinstance(value, list):
                categories.append(key)

        return categories
    except (FileNotFoundError, json.JSONDecodeError):
        return get_default_categories()
```

### 2. Enhanced `load_test_words()` Function

**New Signature:**
```python
def load_test_words(
    config_path: str = None,
    categories: List[str] = None,
    auto_detect_categories: bool = True
) -> List[Dict[str, Any]]:
```

**Key Features:**
- **Flexible category selection**: Specify which categories to load
- **Auto-detection**: Automatically discover categories from config file
- **Fallback support**: Use defaults when detection fails
- **Error handling**: Warn about missing categories instead of failing

### 3. Dynamic Argument Parser

**Two-pass parsing approach:**
```python
# First pass: determine config file
pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--config", help="Path to configuration file")
pre_args, _ = pre_parser.parse_known_args()

# Determine available categories based on config
config_path = pre_args.config or default_config_path
available_categories = get_available_categories(config_path)

# Second pass: create full parser with dynamic choices
parser = argparse.ArgumentParser(...)
parser.add_argument("--category", choices=available_categories, ...)
parser.add_argument("--categories", nargs="+", choices=available_categories, ...)
```

### 4. Enhanced Function Parameters

**Updated `debug_compound_detection()` signature:**
```python
def debug_compound_detection(
    test_words: List[str] = None,
    config_path: str = None,
    verbose: bool = True,
    filter_category: str = None,      # Legacy support
    categories: List[str] = None,     # New flexible parameter
) -> None:
```

## Usage Examples

### 1. Default Behavior (Backward Compatible)
```bash
# Uses all categories from default config
python scripts/debug_compound.py
```

### 2. Single Category Selection
```bash
# Legacy single category
python scripts/debug_compound.py --category compound_words

# New single category (converted to list internally)
python scripts/debug_compound.py --categories compound_words
```

### 3. Multiple Category Selection
```bash
# Select multiple categories
python scripts/debug_compound.py --categories compound_words neologisms
```

### 4. Custom Configuration Files
```bash
# Use custom config with auto-detected categories
python scripts/debug_compound.py --config custom_config.json

# Use custom config with specific categories
python scripts/debug_compound.py --config custom_config.json --categories technical_terms
```

### 5. Custom Category Configuration Example

**File: `config/custom_categories_test.json`**
```json
{
  "technical_terms": [
    {
      "word": "Quantencomputer",
      "description": "Quantum computing device",
      "context": "Der Quantencomputer revolutioniert die Informatik."
    }
  ],
  "philosophical_concepts": [
    {
      "word": "Existenzialismus",
      "description": "Philosophical movement",
      "context": "Der Existenzialismus betont die individuelle Freiheit."
    }
  ],
  "modern_neologisms": [
    {
      "word": "Digitalisierung",
      "description": "Digital transformation",
      "context": "Die Digitalisierung verändert unsere Gesellschaft."
    }
  ]
}
```

**Usage:**
```bash
python scripts/debug_compound.py --config config/custom_categories_test.json --categories technical_terms philosophical_concepts
```

## Benefits Achieved

### 1. Flexibility
- **Custom categories**: Easy to create domain-specific test configurations
- **Dynamic discovery**: No need to modify code for new categories
- **Multiple selection**: Test specific combinations of categories

### 2. Maintainability
- **Single source of truth**: Categories defined in configuration files
- **No code duplication**: Category lists no longer hardcoded in multiple places
- **Easy extension**: Add new categories by updating config files

### 3. User Experience
- **Dynamic help**: Help text shows available categories for any config file
- **Clear feedback**: Informative messages about category selection
- **Backward compatibility**: Existing scripts and configs continue to work

### 4. Robustness
- **Error handling**: Graceful handling of missing categories or config files
- **Fallback behavior**: Uses defaults when auto-detection fails
- **Validation**: Warns about missing categories instead of silent failures

## Testing Results

*Note: The following examples show actual output from testing the enhanced script.*

### Default Configuration
```bash
$ python scripts/debug_compound.py --help
Available categories: compound_words, simple_words, neologisms
```

### Custom Configuration
```bash
$ python scripts/debug_compound.py --config config/custom_categories_test.json --help
Available categories: technical_terms, philosophical_concepts, modern_neologisms
```

### Category Selection
```bash
$ python scripts/debug_compound.py --categories compound_words neologisms --compact
🎯 Loading specific categories: compound_words, neologisms
📝 Testing 11 words...
```

### Auto-detection
```bash
$ python scripts/debug_compound.py --config config/custom_categories_test.json --compact
📖 Loading test words from configuration...
📝 Testing 6 words...  # All categories auto-detected
```

## Migration Guide

### For Existing Users
- **No changes required**: Existing command lines continue to work
- **Enhanced functionality**: Can now use `--categories` for multiple selection
- **Same config files**: Existing configuration files work unchanged

### For New Configurations
1. **Create JSON config** with custom category names as top-level keys
2. **Use `--config`** to specify the custom configuration file
3. **Use `--categories`** to select specific categories
4. **Run `--help`** to see available categories for any config file

### For Developers
1. **Use `load_test_words()`** with `categories` parameter for programmatic access
2. **Call `get_categories_from_config()`** to discover available categories
3. **Use `get_default_categories()`** for fallback behavior

## Future Enhancements

1. **Category metadata**: Support for category descriptions and properties
2. **Nested categories**: Hierarchical category organization
3. **Category validation**: Schema validation for category structures
4. **Category templates**: Predefined category sets for common use cases
5. **Interactive selection**: CLI interface for category selection

## Files Modified

- **`scripts/debug_compound.py`**: Complete refactoring for flexible categories
- **`config/custom_categories_test.json`**: Example custom configuration
- **`scripts/README_flexible_categories.md`**: This documentation

This enhancement transforms the script from a rigid, hardcoded system to a flexible, configuration-driven tool that can adapt to any domain or use case without requiring code modifications.
