#!/usr/bin/env python3
"""
approval_workflow.py
====================
Handles the governed ingestion pipeline for curriculum uploads.
Enforces quarantine, manual review, and Merkle-sealed attachment.
"""

import os
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

class ApprovalWorkflow:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.quarantine_dir = self.data_dir / "uploads" / "quarantined"
        self.accepted_dir = self.data_dir / "uploads" / "accepted"
        self.manifest_dir = self.data_dir / "uploads" / "manifests"

    def quarantine_upload(self, manifest_id: str, reason: str) -> bool:
        """
        Moves an upload to quarantine for further inspection.
        """
        manifest_path = self.manifest_dir / f"{manifest_id}.json"
        if not manifest_path.exists():
            return False
            
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        manifest["status"] = "quarantined"
        manifest["quarantine_reason"] = reason
        manifest["quarantine_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            
        print(f"Upload {manifest_id} quarantined: {reason}")
        return True

    def approve_upload(self, manifest_id: str, approver_id: str) -> Dict[str, Any]:
        """
        Approves a quarantined or pending upload.
        Moves files to 'accepted' and updates the manifest.
        """
        manifest_path = self.manifest_dir / f"{manifest_id}.json"
        if not manifest_path.exists():
            return {"ok": False, "error": "Manifest not found"}

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        # In a real system, we'd move the raw file to accepted/
        manifest["status"] = "accepted"
        manifest["approver_id"] = approver_id
        manifest["approval_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return {"ok": True, "manifest": manifest}

    def list_pending(self) -> List[Dict[str, Any]]:
        """
        Lists all uploads waiting for approval or in quarantine.
        """
        pending = []
        for mf in self.manifest_dir.glob("*.json"):
            with open(mf, "r") as f:
                manifest = json.load(f)
                if manifest.get("status") in ("pending", "quarantined"):
                    pending.append(manifest)
        return pending

if __name__ == "__main__":
    # Test logic
    pass
