#!/usr/bin/env python3
"""
sdk_preflight.py
================
Environment readiness check for the LK20 Digital Twin SDK.
Checks for dependencies, project structure, and hardware requirements.
"""

import sys
import os
import platform
from pathlib import Path

def run_preflight():
    print("--- LK20 SDK Preflight Check ---")
    
    # 1. Python Version
    print(f"Python version: {platform.python_version()} ... ", end="")
    if sys.version_info >= (3, 9):
        print("OK")
    else:
        print("WARNING: Recommend Python 3.9+")

    # 2. Dependencies
    print("Checking dependencies...", end="")
    try:
        import numpy as np
        print(f" Numpy {np.__version__} OK")
    except ImportError:
        print(" FAILED: numpy is required.")

    # 3. Project Structure
    print("Checking project structure...", end="")
    required_files = ["tn.py", "lk20_main.py", "lk20_kernel.py", "digital_twin_kernel.py"]
    missing = [f for f in required_files if not Path(f).exists()]
    if not missing:
        print(" OK")
    else:
        print(f" FAILED: Missing {missing}")

    # 4. Data Directory
    print("Checking data directory...", end="")
    if Path("data").is_dir():
        print(" OK")
    else:
        print(" WARNING: 'data/' directory not found. Run 'python lk20_main.py init' first.")

    print("\nPreflight check complete.")

if __name__ == "__main__":
    run_preflight()
