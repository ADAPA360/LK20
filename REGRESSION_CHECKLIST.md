# LK20 Digital Twin: Regression Checklist

Use this checklist before every release or after significant local modifications.

## 1. Baseline Integrity
- [ ] **Import Stability**: All modules in root and `akkurat_atomtn_stack` import without error.
- [ ] **Scaffold Integrity**: The 19 populated scaffolds are non-empty and retain their logic.
- [ ] **Dual TN Preservation**: `tn.py` in root and `akkurat_atomtn_stack/tn.py` remain distinct and unmerged.

## 2. Automated Validation
- [ ] **SDK Preflight**: `python sdk_preflight.py` returns OK.
- [ ] **SDK Contract**: `python sdk_contract_test.py` passes all tests.
- [ ] **Privacy Boundaries**: `python privacy_boundary_test.py` passes (Boundary A/B/C confirmed).
- [ ] **Network Integrity**: `python verify_network_integrity.py` confirms Merkle validity.

## 3. Web & API Services
- [ ] **Server Startup**: `python lk20_server.py` starts without traceback.
- [ ] **API Availability**: `/api/status` and `/api/routes` return valid JSON.
- [ ] **Frontend Load**: Dashboard loads at `http://127.0.0.1:8000/`.
- [ ] **Smoke Test**: `python smoke_test_web.py` completes 6/6 tests successfully.

## 4. Feature Sanity
- [ ] **Local AI Fallback**: `local_ai_adapter.py` imports safely without model assets.
- [ ] **Approval Workflow**: Quarantine-to-Accepted flow is functional in `approval_workflow.py`.
- [ ] **Federation**: `federation_export.py` and `federation_import.py` logic remains intact.
- [ ] **GREP Normalization**: `grep_normalizer.py` correctly cleans sample curriculum text.

## 5. Deployment & Release
- [ ] **SDK Release**: `python make_sdk_release.py` successfully bundles the SDK.
- [ ] **Documentation**: `README_WEB.md`, `README_SDK.md`, and `LOCALHOST_RUNBOOK.md` are up to date.
