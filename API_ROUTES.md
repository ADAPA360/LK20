# LK20 Digital Twin API Routes

This document outlines the JSON API endpoints exposed by `lk20_server.py`. 
All endpoints are relative to `http://127.0.0.1:8000/api`.

## System Status
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/status` | GET | Returns general system status, version, and network state. |
| `/health` | GET | Returns detailed health checks for all core dependencies. |
| `/init` | POST | Initializes the project data directory structure. |

## Session & Identity
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/whoami` | GET | Returns the current user session details. |
| `/login` | POST | Authenticates a user with a specific role and ID. |
| `/logout` | POST | Clears the current session. |

## Network Operations
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/create-network` | POST | Generates the initial Tree Tensor Network (TTN). |
| `/verify` | GET | Performs a Merkle-root integrity check of the network. |
| `/snapshot` | POST | Creates a persistent snapshot of the current state. |

## Curriculum Exploration
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/inspect` | GET | Inspects a target (grade, subject, node, or upload). |
| `/search` | GET | Vector/text search across the curriculum network. |
| `/student/view` | GET | Returns a student-safe projection of a grade/subject. |

## Upload Workflow
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/upload` | POST | (Multipart) Uploads curriculum documents with metadata. |
| `/uploads` | GET | Lists recent upload manifests. |
| `/upload?id=X` | GET | Inspects a specific upload manifest. |
| `/validate-upload` | POST | Validates an upload manifest against local rules. |
| `/attach-upload` | POST | Attaches a validated upload to the digital twin. |

## Coverage & Gaps
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/coverage` | GET | Returns coverage percentage for competence aims. |
| `/gaps` | GET | Highlights missing coverage in the curriculum. |

## Governance & Canonical
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/gov/benefits` | GET | Reports on the societal and educational benefits of the twin. |
| `/gov/inspect-system` | GET | Provides a government-level inspection view. |
| `/canonical-status` | GET | Returns the status of the governing canonical curriculum. |
| `/sample-canonical` | POST | Generates sample canonical data for testing. |
| `/ingest-canonical`| POST | Ingests a governing curriculum snapshot. |

## Audit
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/audit` | GET | Returns a list of recent governed actions from the audit log. |
