# LK20 Digital Twin: Localhost Runbook

This document provides exact instructions for starting, verifying, and troubleshooting the LK20 digital-twin platform in a local development environment.

## 1. Quick Start (Full Verification)

Run the following command sequence to verify the environment and baseline state:

```powershell
python sdk_preflight.py
python sdk_contract_test.py
python privacy_boundary_test.py
python verify_network_integrity.py
```

Expected Output: All tests should report **OK** or **SUCCESS**. `verify_network_integrity.py` should output a JSON object with `"is_valid": true`.

## 2. Starting the Backend Server

To start the local HTTP server:

```powershell
python lk20_server.py
```

- **Backend API URL**: `http://127.0.0.1:8000/api`
- **Frontend Dashboard URL**: `http://127.0.0.1:8000/`

## 3. Verifying the Web Platform

While the server is running, execute the smoke test:

```powershell
python smoke_test_web.py
```

Expected Output: `--- Smoke Test Completed Successfully ---` with 6/6 tests passing.

## 4. Full Regression Suite

To run all non-server validation checks in one pass:

```powershell
python run_regression.py
```

## 5. Shutdown Instructions

- **Server**: Press `Ctrl+C` in the terminal running `lk20_server.py`.
- **Background Processes**: Ensure no stray `python` processes are holding port 8000.

## 6. Troubleshooting

### Port 8000 already in use
If the server fails to start with an `Address already in use` error:
1. Identify the process: `netstat -ano | findstr :8000`
2. Kill the process (if safe): `taskkill /F /PID <PID>`

### Missing NumPy
The system requires NumPy for tensor operations and embedding similarity.
- Install: `pip install numpy`

### Local AI Model Assets Absent
The system is designed to fail safely if local LLM weights or embedding models are missing from `local_ai/models/`.
- Behavior: `local_ai_adapter.py` will switch to **Deterministic Fallback Mode**. This is normal for basic curriculum twin inspection.

### Stale Session State
If `smoke_test_web.py` fails on the "Guest" check:
- Cause: A persistent session is saved in `data/config/session.json`.
- Fix: The smoke test now includes an auto-logout at the start, but you can also manually delete `data/config/session.json` to reset.

### Windows Path Issues (Spaces in 'ANU AI')
If scripts fail due to path resolution in `C:\Users\ali_z\ANU AI\LK20`:
- Fix: Always wrap paths in double quotes in CLI commands. The Python scripts use `pathlib` and absolute paths to handle this internally.
