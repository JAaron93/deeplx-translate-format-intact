#!/usr/bin/env python3
"""Test file to validate Black and Ruff linting setup.

This file intentionally contains various Python constructs and formatting issues
that should trigger both Black formatting and Ruff linting rules.
"""

# Test imports (some unused, some unsorted)
from typing import Dict, List, Optional

# Test long line that exceeds 88 characters (Black's default line length)
really_long_variable_name_that_should_trigger_line_length_formatting = "This is a very long string that will definitely exceed the 88 character limit set by Black"


# Test function with poor formatting
def poorly_formatted_function(arg1, arg2, arg3=None):
    """Docstring with trailing whitespace"""
    # Inconsistent indentation and spacing
    if arg1 == arg2:
        print("Equal values")
    else:
        print("Different values")  # inconsistent indentation (6 spaces instead of 8)

    # Dead code (Ruff F841)
    unused_variable = 42

    # Using deprecated syntax (for Ruff UP rules)
    old_style_format = "arg1: %s, arg2: %s" % (arg1, arg2)

    # Unnecessary parentheses
    return arg1 + arg2


# Test class with various issues
class TestClass:
    """Class docstring"""

    def __init__(self, value):
        self.value = value
        self.unused_attr = "unused"  # Ruff might flag this

    def method_with_complexity(self, x, y, z):
        # Complex conditional that could be simplified (Ruff SIM rules)
        if x == True:  # Should use 'if x:'
            if y == False:  # Should use 'if not y:'
                if z != None:  # Should use 'if z is not None:'
                    return True

        # Mutable default argument (Ruff B006)
        def inner_function(items=[]):
            items.append(1)
            return items

        # Using bare except (Ruff E722)
        try:
            result = 10 / 0
        except:
            pass

        return False


# Test list/dict comprehensions with poor formatting
messy_list = [i for i in range(10) if i % 2 == 0]
messy_dict = {k: v for k, v in zip(["a", "b", "c"], [1, 2, 3])}


# Test type annotations with poor spacing
def typed_function(arg1: str, arg2: int) -> Optional[Dict[str, List[int]]]:
    """Function with type hints"""
    return None


# Test string concatenation that could use f-strings (Ruff UP rules)
name = "Test"
old_concat = "Hello " + name + "!"
old_format = f"Hello {name}!"

# Test comparison issues
value = 5
if value == None:  # Should use 'is None'
    pass
if type(value) == int:  # Should use isinstance()
    pass

# Test trailing whitespace and blank lines issues


# Too many blank lines above

# Missing final newline below
print("End of test file")
