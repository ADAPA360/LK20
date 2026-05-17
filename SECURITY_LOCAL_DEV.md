# LK20 Local Development Security

This document outlines the security protocols for running the LK20 Digital Twin in a local development environment.

## 1. Local Authentication
- **Mechanism**: The current system uses a "trust-on-login" model for local development.
- **Session Storage**: Sessions are stored in `data/config/session.json`. 
- **Role Enforcement**: Roles are enforced within `lk20_main.py` via the `PermissionEngine`.
- **Warning**: Do not expose `lk20_server.py` to the public internet. It is designed to bind to `127.0.0.1` only.

## 2. File System Security
- **Uploads**: All uploads are sanitized for path traversal in `lk20_server.py`.
- **Permissions**: Ensure the `data/` directory has restricted read/write permissions for the local user running the service.
- **Audit Logs**: Audit logs in `data/audit/` are append-only by convention. Verification of these logs is handled via `verify_network_integrity.py`.

## 3. Local AI & Privacy
- **Processing**: Large Language Model (LLM) processing should happen via the `local_ai_adapter.py` using locally hosted weights or the `akkurat_atomtn_stack`.
- **Data Leaks**: Ensure that no student-identifiable information (PII) is sent to external API providers (e.g., OpenAI/Vertex AI) if the system is configured in "Federated Mode".

## 4. Development Best Practices
- **No Hardcoded Keys**: Use environment variables or a local `.env` file (not checked into git) for any external service keys.
- **Mocking**: For IDPorten or other national auth systems, use the provided `idporten_auth_adapter_placeholder.py`.
