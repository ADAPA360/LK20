# ANTIGRAVITY PROJECT MAP - LK20 Digital Twin

## Project Overview
Local-first LK20 governed curriculum digital-twin platform.

## Folder Structure
- `C:\Users\ali_z\ANU AI\LK20` (Root)
  - `akkurat_atomtn_stack/`: AtomTN unified tensor network library.
  - `local_ai/`: Local AI processing, semantic attractors, and language models.
  - `web/`: Frontend dashboard (Vanilla HTML/JS/CSS).
  - `data/`: Governed state, networks, audits, and configuration.

## Python Entrypoints
- `lk20_main.py`: Command gateway for the system.
- `lk20_server.py`: Production-quality local HTTP server (default: http://127.0.0.1:8000).
- `smoke_test_web.py`: Automated smoke test for the web platform.
- `sdk_preflight.py`: Environment and dependency pre-check.
- `sdk_contract_test.py`: API stability and contract verification.
- `privacy_boundary_test.py`: Governed privacy boundary automated tests.
- `verify_network_integrity.py`: Merkle-root and network integrity validation.
- `make_sdk_release.py`: SDK distribution builder.

## Documentation (Markdown)
- `README_WEB.md`: Instructions for running the local web platform.
- `README_SDK.md`: Instructions for using the Digital Twin SDK.
- `API_ROUTES.md`: Detailed JSON API endpoint mapping.
- `PRIVACY_MODEL.md`: Privacy boundary and data classification definitions.
- `NATIONAL_IMPLEMENTATION_NOTES.md`: LK20 domain and national standard notes.
- `SECURITY_LOCAL_DEV.md`: Local development security and auth protocols.

## Tensor Network Modules (Distinct)
1. **Root `tn.py`**: `C:\Users\ali_z\ANU AI\LK20\tn.py` (Root-level customized LK20 tensor-network script).
2. **Stack `tn.py`**: `C:\Users\ali_z\ANU AI\LK20\akkurat_atomtn_stack\tn.py` (AtomTN unified tensor network library).

## Local AI / Bridge Components
- `local_ai_adapter.py`: Bridge between LK20 logic and AtomTN stack.
- `embedding_index.py`: (Scaffold) Intended for embedding indexing logic.
- `local_ai/`: Contains core AI modules (`local_ai.py`, `semantic_attractors.py`, `sentence_builder.py`, etc.).

## Web Frontend Structure
- `web/index.html`: Dashboard UI.
- `web/app.js`: Vanilla JS API client.
- `web/style.css`: Clean enterprise dashboard styling.

## Data Store
- `data/audit/`: Governed action logs.
- `data/canonical/`: National curriculum snapshots.
- `data/config/`: Session and system configuration.
- `data/exports/`: Federated sub-tree exports.
- `data/networks/`: Tree Tensor Network (TTN) state files.
- `data/reports/`: Generated system reports.
- `data/uploads/`: Curriculum ingestion pipeline (raw, accepted, quarantined).
