#!/usr/bin/env python3
"""
run_regression.py
=================
Single-command regression suite for the LK20 Digital Twin.
Runs preflight, contract, privacy, and integrity checks.
Does NOT start the web server.
"""

import subprocess
import sys

def run_script(name, cmd):
    print(f"Running {name}...", end=" ", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("PASS")
        return True
    except subprocess.CalledProcessError as e:
        print("FAIL")
        print(f"\n--- ERROR in {name} ---")
        print(e.stdout)
        print(e.stderr)
        return False
    except Exception as e:
        print("ERROR")
        print(f"Failed to execute {name}: {e}")
        return False

def main():
    print("--- LK20 Regression Suite ---")
    
    checks = [
        ("SDK Preflight", [sys.executable, "sdk_preflight.py"]),
        ("SDK Contract", [sys.executable, "sdk_contract_test.py"]),
        ("Privacy Boundary", [sys.executable, "privacy_boundary_test.py"]),
        ("Network Integrity", [sys.executable, "verify_network_integrity.py"])
    ]
    
    all_passed = True
    for name, cmd in checks:
        if not run_script(name, cmd):
            all_passed = False
            
    print("\n-----------------------------")
    if all_passed:
        print("SUMMARY: ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("SUMMARY: REGRESSION FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
