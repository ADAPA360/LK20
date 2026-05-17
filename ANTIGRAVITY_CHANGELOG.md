# ANTIGRAVITY CHANGELOG - LK20 Digital Twin

## Summary of Changes
Minimal repairs and population of missed scaffolds to bring the project to a verified state.

## Modified Files

### 1. `embedding_index.py`
- **Type**: Functional / Scaffold Population
- **Reason**: The file was 0 bytes.
- **Before**: Empty file.
- **After**: Contains `EmbeddingIndex` class with NumPy-based cosine similarity search for curriculum embeddings.

### 2. `smoke_test_web.py`
- **Type**: Test-only
- **Reason**: Fix for flaky "Guest" role check.
- **Before**: Assumed the session started as guest, but local persistent state could cause it to start as admin.
- **After**: Added an explicit `api_call("/logout", "POST")` at the beginning of `run_tests()` to ensure a clean state.

### 3. `ANTIGRAVITY_PROJECT_MAP.md`
- **Type**: Documentation-only [NEW]
- **Reason**: PHASE 1 requirement to map the project tree.

### 4. `ANTIGRAVITY_VERIFICATION_REPORT.md`
- **Type**: Documentation-only [NEW]
- **Reason**: PHASE 8 requirement to summarize verification results.

### 5. `ANTIGRAVITY_CHANGELOG.md`
- **Type**: Documentation-only [NEW]
- **Reason**: PHASE 8 requirement to track all modifications.

## [2026-05-12] Baseline Freeze & Regression Hardening

### 6. `LOCALHOST_RUNBOOK.md`
- **Type**: Documentation-only [NEW]
- **Reason**: Provides reproducible local operation instructions.

### 7. `REGRESSION_CHECKLIST.md`
- **Type**: Documentation-only [NEW]
- **Reason**: Manual verification guide for project stability.

### 8. `LK20_BASELINE_MANIFEST.json`
- **Type**: Metadata [NEW]
- **Reason**: Captured SHA256 hashes and verification state for baseline freeze.

### 9. `run_regression.py`
- **Type**: Test-only [NEW]
- **Reason**: Single-command runner for non-interactive validation checks.

