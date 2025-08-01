#!/usr/bin/env python3
# Test file with intentional style violations to test non-blocking pre-commit hooks

import os,sys,json # Multiple imports on one line (violates E401)
import   requests   # Extra whitespace (violates E201, E202)

def bad_function(x,y,z):  # Missing spaces after commas (violates E999)
    # Missing docstring (violates D100)
    if x==y:  # Missing spaces around operator (violates E225)
        print("This line is way too long and exceeds the 88 character limit set by Black and will trigger line length violations")  # Line too long (violates E501)
    
    unused_variable = "This variable is never used"  # Unused variable (violates F841)
    
    # Trailing whitespace on next line (violates W291)
    result = x+y+z    
    
    return result

class badClassName:  # Class name should be PascalCase (violates N801)
    def __init__(self):
        pass
    
    def another_bad_method(self):
        # More intentional violations
        l = [1,2,3,4,5]  # Variable name 'l' is ambiguous (violates E741)
        d = {'key':'value'}  # Missing spaces around colon (violates E231)
        
        # Security issue for bandit
        import subprocess
        subprocess.call("ls -la", shell=True)  # Security issue (violates B602)
        
        return l,d  # Missing space after comma (violates E999)

# Missing final newline (violates W292)

# Adding more violations to trigger hooks
def another_function():
    x=1+2+3  # No spaces around operators
    return x
