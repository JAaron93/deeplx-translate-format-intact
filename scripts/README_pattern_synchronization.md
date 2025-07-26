# Pattern Synchronization Enhancement

This document describes the pattern synchronization enhancement implemented in `debug_candidates.py` to ensure consistency between debug scripts and the actual detector implementation.

## Problem Statement

Previously, the debug script used hardcoded regex patterns for compound word detection that could become outdated when the actual detector implementation changed. This led to:

- **Inconsistent results** between debug output and actual detection
- **Maintenance overhead** requiring manual updates to debug scripts
- **Potential bugs** going unnoticed due to pattern mismatches
- **Confusion** when debugging detection issues

## Solution Overview

The enhancement introduces a dynamic pattern extraction system with intelligent fallback:

1. **Primary**: Extract patterns from detector instance via `get_compound_patterns()`
2. **Secondary**: Check for patterns as detector attributes
3. **Fallback**: Use hardcoded patterns with clear warnings

## Implementation Details

### 1. Detector Enhancement

Added `get_compound_patterns()` method to `NeologismDetector`:

```python
def get_compound_patterns(self) -> list[str]:
    """Get the regex patterns used for compound word detection."""
    return [
        # CapitalizedCompounds (internal capitals)
        r"\b[A-ZÄÖÜ][a-zäöüß]{5,}(?:[A-ZÄÖÜ][a-zäöüß]+)+\b",
        # Long capitalized words (potential compounds)
        r"\b[A-ZÄÖÜ][a-zäöüß]{10,}\b",
        # linked compounds
        r"\b[a-zäöüß]+(?:s|n|es|en|er|e|ns|ts)[a-zäöüß]{4,}\b",
        # abstract suffixes including philosophical terms
        r"\b[a-zäöüß]+(?:heit|keit|ung|schaft|tum|nis|sal|ismus|ität|logie|sophie|bewusstsein|philosophie)\b",
    ]
```

### 2. Debug Script Enhancement

Added `get_compound_patterns_from_detector()` function with intelligent fallback:

```python
def get_compound_patterns_from_detector(detector: NeologismDetector) -> list[str]:
    """Extract compound patterns from detector instance with fallback."""

    # Try detector.get_compound_patterns()
    if hasattr(detector, 'get_compound_patterns'):
        try:
            patterns = detector.get_compound_patterns()
            print("✓ Using patterns from detector.get_compound_patterns()")
            return patterns
        except Exception as e:
            print(f"⚠️  Failed to get patterns from detector: {e}")

    # Try detector.compound_patterns attribute
    if hasattr(detector, 'compound_patterns'):
        try:
            patterns = detector.compound_patterns
            print("✓ Using patterns from detector.compound_patterns attribute")
            return patterns
        except Exception as e:
            print(f"⚠️  Failed to get patterns from detector attribute: {e}")

    # Fallback to hardcoded patterns with warning
    print("⚠️  Warning: Using fallback hardcoded patterns - these may be outdated!")
    return fallback_patterns
```

## Benefits

### 1. Pattern Consistency
- Debug scripts always use the same patterns as the actual implementation
- Eliminates discrepancies between debug output and real detection results

### 2. Automatic Synchronization
- No manual updates needed when detector patterns change
- Patterns stay synchronized automatically

### 3. Maintenance Reduction
- Single source of truth for patterns
- Reduces code duplication and maintenance overhead

### 4. Clear Warnings
- Explicit warnings when fallback patterns are used
- Helps identify when detector needs pattern exposure methods

### 5. Backward Compatibility
- Graceful degradation when new methods aren't available
- Existing code continues to work

## Usage Examples

### Normal Operation
```bash
$ python scripts/debug_candidates.py
✓ Using patterns from detector.get_compound_patterns()
Total patterns extracted: 4
```

### Fallback Operation
```bash
$ python scripts/debug_candidates.py
⚠️  Warning: Using fallback hardcoded patterns - these may be outdated!
Total patterns extracted: 3
```

## Pattern Evolution Demonstration

The enhancement immediately revealed pattern evolution:

**Before (Hardcoded)**:
- 3 patterns
- Missing long capitalized words pattern
- Limited philosophical terms

**After (From Detector)**:
- 4 patterns
- Includes `\b[A-ZÄÖÜ][a-zäöüß]{10,}\b` for long compounds
- Enhanced philosophical terms: `bewusstsein|philosophie`

## Testing

Run the test suite to verify functionality:

```bash
# Test normal operation
python scripts/debug_candidates.py

# Test fallback functionality
python scripts/test_fallback_patterns.py
```

## Migration Guide

### For Other Debug Scripts

1. **Import the helper function**:
   ```python
   from scripts.debug_candidates import get_compound_patterns_from_detector
   ```

2. **Replace hardcoded patterns**:
   ```python
   # Before
   patterns = [r"pattern1", r"pattern2"]

   # After
   patterns = get_compound_patterns_from_detector(detector)
   ```

### For Detector Enhancements

1. **Add pattern exposure method**:
   ```python
   def get_compound_patterns(self) -> list[str]:
       return self.compound_patterns
   ```

2. **Update internal usage**:
   ```python
   # Use centralized patterns
   patterns = self.get_compound_patterns()
   ```

## Future Enhancements

1. **Configuration-based patterns**: Load patterns from config files
2. **Pattern versioning**: Track pattern changes over time
3. **Pattern validation**: Validate pattern syntax and effectiveness
4. **Pattern metrics**: Collect statistics on pattern usage and effectiveness

## Files Modified

- `services/neologism_detector.py`: Added `get_compound_patterns()` method
- `scripts/debug_candidates.py`: Added dynamic pattern extraction
- `scripts/test_fallback_patterns.py`: Test suite for fallback functionality
- `scripts/README_pattern_synchronization.md`: This documentation

## Impact

This enhancement ensures that debug tools remain synchronized with the actual implementation, reducing maintenance overhead and improving debugging accuracy. The pattern evolution from 3 to 4 patterns demonstrates the immediate value of this synchronization approach.
