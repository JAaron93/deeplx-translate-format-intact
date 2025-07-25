#!/usr/bin/env python3
"""Basic test to check if Python is working and display environment info."""

import os
import sys
+if __name__ == "__main__":
+    print("Python is working!")
+    print("Script file path:", __file__)
+    print("Current working directory:", os.getcwd())
+    print("Python version:", sys.version)
+    # Show first 3 entries of sys.path
+    print("Python path:", sys.path[:3])
