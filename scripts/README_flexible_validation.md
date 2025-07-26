# Flexible Project Root Validation

This document describes the flexible project root validation system implemented in `debug_candidates.py` and available for use in other scripts.

## Overview

The validation system has been refactored to eliminate hardcoded project structure dependencies and provide multiple configuration methods. This makes the validation more adaptable to changes in project layout without requiring code modifications.

## Configuration Methods

### 1. Default Configuration (Backward Compatible)

```python
project_root = find_and_validate_project_root()
```

Uses built-in defaults:
- **Expected directories**: `services`, `models`, `scripts`, `database`, `config`
- **Expected files**: `app.py`, `requirements.txt`, `README.md`
- **Critical files**: `services/neologism_detector.py`

### 2. Environment Variables

Set environment variables to customize validation:

```bash
export PROJECT_EXPECTED_DIRS="services,models,scripts"
export PROJECT_EXPECTED_FILES="app.py,requirements.txt"
export PROJECT_CRITICAL_FILES="services/neologism_detector.py"
python your_script.py
```

### 3. Configuration File

Create a JSON configuration file:

```json
{
  "expected_dirs": ["services", "models", "scripts"],
  "expected_files": ["app.py", "requirements.txt"],
  "critical_files": ["services/neologism_detector.py"]
}
```

Use it in your script:

```python
config_path = Path("path/to/config.json")
project_root = find_and_validate_project_root(config_path=config_path)
```

### 4. Programmatic Configuration

Pass parameters directly:

```python
project_root = find_and_validate_project_root(
    expected_dirs=["services", "models"],
    expected_files=["app.py"],
    critical_files=["services/neologism_detector.py"],
    strict_validation=False  # Only warn instead of failing
)
```

## Configuration Priority

The system uses the following priority order:

1. **Direct parameters** (highest priority)
2. **Environment variables**
3. **Configuration file**
4. **Default configuration** (lowest priority)

## Validation Modes

### Strict Validation (Default)

- Raises `RuntimeError` if any expected items are missing
- Suitable for production scripts that require complete project structure

### Non-Strict Validation

- Only prints warnings for missing items
- Continues execution even if some items are missing
- Useful for development or when working with partial project structures

```python
project_root = find_and_validate_project_root(strict_validation=False)
```

## API Reference

### `find_and_validate_project_root()`

```python
def find_and_validate_project_root(
    expected_dirs: Optional[list[str]] = None,
    expected_files: Optional[list[str]] = None,
    critical_files: Optional[list[str]] = None,
    config_path: Optional[Path] = None,
    strict_validation: bool = True
) -> Path:
```

**Parameters:**
- `expected_dirs`: List of directories that should exist in project root
- `expected_files`: List of files that should exist in project root
- `critical_files`: List of critical files (relative paths) that must exist
- `config_path`: Optional path to JSON configuration file
- `strict_validation`: If False, only warn about missing items

**Returns:** `Path` object pointing to validated project root

**Raises:** `RuntimeError` if validation fails (when `strict_validation=True`)

### `load_project_validation_config()`

```python
def load_project_validation_config(config_path: Optional[Path] = None) -> dict:
```

**Parameters:**
- `config_path`: Optional path to JSON configuration file

**Returns:** Dictionary with configuration loaded from environment variables, config file, or defaults

## Examples

See `scripts/example_flexible_validation.py` for comprehensive examples of all configuration methods.

## Migration Guide

### From Hardcoded Validation

**Before:**
```python
# Hardcoded validation
expected_dirs = ["services", "models", "scripts"]
expected_files = ["app.py", "requirements.txt"]
# ... validation logic
```

**After:**
```python
# Flexible validation
project_root = find_and_validate_project_root(
    expected_dirs=["services", "models", "scripts"],
    expected_files=["app.py", "requirements.txt"]
)
```

### Adding New Project Structure Requirements

Instead of modifying code, update configuration:

1. **Environment variables**: Update your deployment scripts
2. **Configuration file**: Update the JSON file
3. **Default configuration**: Modify the defaults in `load_project_validation_config()`

## Benefits

1. **Flexibility**: Easy to adapt to project structure changes
2. **Maintainability**: No need to modify code for structure changes
3. **Environment-specific**: Different validation rules for different environments
4. **Backward compatibility**: Existing code continues to work unchanged
5. **Testability**: Easy to test with different project structures

## Files

- `scripts/debug_candidates.py`: Main implementation
- `scripts/project_validation_config.json`: Example configuration file
- `scripts/example_flexible_validation.py`: Comprehensive examples
- `scripts/README_flexible_validation.md`: This documentation
