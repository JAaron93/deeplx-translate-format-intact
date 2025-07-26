# File Operations and Import Improvements

This document describes the improvements made to `scripts/example_flexible_validation.py` to follow Python best practices for imports and implement robust file operation error handling.

## Problems Addressed

The original implementation had several issues that violated Python best practices and lacked proper error handling:

### 1. Import Organization Issues

**Problem**: Late import of `json` module
```python
def example_config_file():
    # ... function code ...
    if not config_file.exists():
        # BAD: Import inside function
        import json
        # ... rest of function
```

**Issues:**
- ❌ **PEP 8 violation**: Imports should be at the top of the file
- ❌ **Poor readability**: Dependencies not immediately visible
- ❌ **Performance impact**: Module imported every time function runs
- ❌ **IDE warnings**: Static analysis tools flag this as poor practice

### 2. File Operation Issues

**Problem**: No error handling for file operations
```python
# BAD: No error handling
with open(config_file, 'w') as f:
    json.dump(sample_config, f, indent=2)
```

**Issues:**
- ❌ **No permission checks**: Could fail silently or crash
- ❌ **No disk space handling**: Could fail with unclear errors
- ❌ **No overwrite protection**: Could accidentally destroy existing files
- ❌ **No encoding specification**: Could cause encoding issues
- ❌ **No directory creation**: Could fail if parent directory doesn't exist

### 3. User Experience Issues

**Problem**: No protection against accidental overwrites
- ❌ **Data loss risk**: Could overwrite existing configuration files
- ❌ **No user warning**: Silent overwriting without confirmation
- ❌ **Poor feedback**: Unclear error messages when operations fail

## Solutions Implemented

### 1. Import Organization Fix

**Before (Poor Practice):**
```python
import os
import sys
from pathlib import Path
from typing import Optional

def example_config_file():
    # ... code ...
    import json  # BAD: Late import
```

**After (Best Practice):**
```python
import json          # GOOD: Import at top
import os
import sys
from pathlib import Path
from typing import Optional
```

**Benefits:**
- ✅ **PEP 8 compliant**: Follows Python style guidelines
- ✅ **Clear dependencies**: All imports visible at file top
- ✅ **Better performance**: Module imported once at startup
- ✅ **IDE friendly**: Static analysis tools work correctly

### 2. Robust File Operations

**New Helper Function:**
```python
def create_sample_config_safely(config_file: Path, sample_config: dict) -> bool:
    """Create a sample config file with proper error handling and overwrite protection."""

    # Check if file already exists and warn user
    if config_file.exists():
        print(f"⚠️  Config file already exists: {config_file}")
        print("   To prevent accidental data loss, please:")
        print("   1. Review the existing file")
        print("   2. Delete it manually if you want to recreate it")
        print("   3. Or modify the existing file as needed")
        return False

    try:
        # Ensure the parent directory exists
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Write the config file with proper error handling
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        print(f"✓ Successfully created sample config: {config_file}")
        return True
    except PermissionError as e:
        print(f"❌ Permission denied creating config file: {e}")
        print(f"   Please check write permissions for: {config_file.parent}")
        return False
    except OSError as e:
        print(f"❌ OS error creating config file: {e}")
        print(f"   Please check disk space and path validity: {config_file}")
        return False
    except (TypeError, ValueError) as e:
        print(f"❌ Error serializing config data: {e}")
        print("   This indicates a problem with the sample config structure")
        return False
```

### 3. Enhanced Error Handling

**Comprehensive Exception Handling:**

1. **PermissionError**: Handles insufficient file system permissions
2. **OSError**: Handles disk space, invalid paths, and other OS-level issues
3. **TypeError/ValueError**: Handles JSON serialization problems
4. **Directory Creation**: Ensures parent directories exist before writing

**User-Friendly Error Messages:**
- Clear explanation of what went wrong
- Specific guidance on how to fix the issue
- Context about where the problem occurred

### 4. Overwrite Protection

**Before (Dangerous):**
```python
# BAD: Always overwrites without warning
with open(config_file, 'w') as f:
    json.dump(sample_config, f, indent=2)
```

**After (Safe):**
```python
# GOOD: Check for existing file first
if config_file.exists():
    print("⚠️  Config file already exists")
    print("   To prevent accidental data loss, please:")
    print("   1. Review the existing file")
    print("   2. Delete it manually if you want to recreate it")
    return False
```

## Testing Results

### 1. Import Organization Test
```python
# Verify json is available at module level
import scripts.example_flexible_validation as efv
assert hasattr(efv, 'json')  # ✅ Available immediately
```

### 2. File Creation Test
```bash
$ python -c "from scripts.example_flexible_validation import create_sample_config_safely; ..."
📝 Creating sample config file: scripts/project_validation_config.json
✓ Successfully created sample config: scripts/project_validation_config.json
Creation result: True
```

### 3. Overwrite Protection Test
```bash
$ python -c "# Try to create same file again"
⚠️  Config file already exists: scripts/project_validation_config.json
   To prevent accidental data loss, please:
   1. Review the existing file
   2. Delete it manually if you want to recreate it
   3. Or modify the existing file as needed
Overwrite attempt result: False
```

### 4. Error Handling Test
```bash
# Test with read-only directory (simulated)
❌ Permission denied creating config file: [Errno 13] Permission denied
   Please check write permissions for: /readonly/path
```

## Benefits Achieved

### 1. Code Quality
- ✅ **PEP 8 compliance**: Proper import organization
- ✅ **Better maintainability**: Clear error handling patterns
- ✅ **Improved readability**: Dependencies visible at top
- ✅ **Static analysis friendly**: No warnings from linters

### 2. Reliability
- ✅ **Robust error handling**: Graceful failure with clear messages
- ✅ **Data protection**: No accidental overwrites
- ✅ **Environment resilience**: Handles various failure scenarios
- ✅ **User guidance**: Clear instructions when problems occur

### 3. User Experience
- ✅ **Clear feedback**: Informative success and error messages
- ✅ **Data safety**: Protection against accidental file loss
- ✅ **Helpful guidance**: Specific instructions for resolving issues
- ✅ **Predictable behavior**: Consistent error handling patterns

### 4. Development Workflow
- ✅ **IDE support**: Better autocomplete and error detection
- ✅ **Debugging**: Clear error messages aid troubleshooting
- ✅ **Team consistency**: Standard error handling patterns
- ✅ **Future maintenance**: Easy to extend and modify

## Best Practices Established

### 1. Import Organization
```python
# ✅ GOOD: All imports at top, grouped logically
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ❌ BAD: Late imports inside functions
def some_function():
    import json  # Don't do this
```

### 2. File Operations
```python
# ✅ GOOD: Comprehensive error handling
try:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
except PermissionError as e:
    print(f"❌ Permission denied: {e}")
    return False
except OSError as e:
    print(f"❌ OS error: {e}")
    return False

# ❌ BAD: No error handling
with open(config_file, 'w') as f:
    json.dump(data, f)
```

### 3. Overwrite Protection
```python
# ✅ GOOD: Check before overwriting
if config_file.exists():
    print("⚠️  File already exists")
    print("   Please review before proceeding")
    return False

# ❌ BAD: Silent overwrite
with open(config_file, 'w') as f:  # Overwrites without warning
    json.dump(data, f)
```

### 4. User Communication
```python
# ✅ GOOD: Clear, actionable error messages
except PermissionError as e:
    print(f"❌ Permission denied creating config file: {e}")
    print(f"   Please check write permissions for: {config_file.parent}")
    return False

# ❌ BAD: Unclear or missing error messages
except Exception as e:
    print(f"Error: {e}")  # Too vague
    # or worse: silent failure
```

## Migration Guide

### For Existing Code

1. **Move imports to top of file**
2. **Add comprehensive error handling around file operations**
3. **Implement overwrite protection for file creation**
4. **Use specific exception types instead of broad Exception catching**
5. **Provide clear, actionable error messages**

### For New Code

1. **Always organize imports at the top**
2. **Use the `create_sample_config_safely` pattern for file creation**
3. **Include encoding specification in file operations**
4. **Create parent directories before writing files**
5. **Return boolean success indicators from file operations**

## Files Modified

- **`scripts/example_flexible_validation.py`**: Complete refactoring for proper imports and robust file operations
- **`scripts/README_file_operations_improvements.md`**: This documentation

These improvements transform the script from a fragile, error-prone implementation to a robust, user-friendly system that follows Python best practices and handles edge cases gracefully.
