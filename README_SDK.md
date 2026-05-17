# LK20 Digital Twin SDK

Welcome to the SDK for the governed curriculum digital twin. This SDK allows you to build local-first applications that interact with the LK20 curriculum state, perform AI-assisted mapping, and ensure governed compliance.

## Installation
The SDK is designed to be lightweight. Ensure you have `numpy` installed for the tensor-network operations.

```bash
pip install -r requirements_sdk.txt
```

## Core Modules
- **`LK20MainApp`**: The primary gateway for all curriculum operations.
- **`DigitalTwinKernel`**: Lower-level access to the Tree Tensor Network (TTN).
- **`LocalAIAdapter`**: Interface for curriculum embeddings and LLM suggestions.

## Basic Usage

```python
from lk20_main import LK20MainApp, LK20MainConfig

# Initialize
config = LK20MainConfig.from_project_root(".")
app = LK20MainApp(config)

# Get status
status = app.status()
print(f"Network Active: {status['network_exists']}")

# Search curriculum
results = app.search(query="programmering", limit=5)
for r in results['results']:
    print(f"Found: {r['name']} (Score: {r['score']})")
```

## Testing
Run the following to verify your local environment:
```bash
python sdk_preflight.py
python sdk_contract_test.py
```

## Security
Always use the `PermissionEngine` to verify user roles before performing mutating actions. Refer to `SECURITY_LOCAL_DEV.md` for more information.
