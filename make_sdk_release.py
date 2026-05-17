#!/usr/bin/env python3
"""
make_sdk_release.py
===================
Bundles the LK20 Digital Twin SDK for distribution.
Collects core scripts, documentation, and tests into a 'release' directory.
"""

import shutil
import os
import time
from pathlib import Path

def make_release():
    print("--- Building LK20 SDK Release ---")
    
    release_dir = Path("sdk_release")
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir()

    # Files to include
    core_files = [
        "tn.py",
        "lk20_main.py",
        "lk20_kernel.py",
        "digital_twin_kernel.py",
        "twin_anything.py",
        "local_ai_adapter.py",
        "approval_workflow.py",
        "idporten_auth_adapter_placeholder.py"
    ]
    
    docs_and_tests = [
        "README_SDK.md",
        "API_ROUTES.md",
        "PRIVACY_MODEL.md",
        "sdk_preflight.py",
        "sdk_contract_test.py",
        "privacy_boundary_test.py"
    ]

    # Copy files
    for f in core_files + docs_and_tests:
        if Path(f).exists():
            print(f"Adding {f}...")
            shutil.copy2(f, release_dir / f)
        else:
            print(f"WARNING: Skipping {f} (not found)")

    # Create a simple version stamp
    with open(release_dir / "VERSION.txt", "w") as f:
        f.write(f"LK20-SDK-v1.0-dev-{int(time.time())}")

    print(f"\nRelease bundled in: {release_dir.absolute()}")

if __name__ == "__main__":
    make_release()
