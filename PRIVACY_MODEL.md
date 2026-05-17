# LK20 Digital Twin Privacy Model

This document defines the privacy boundaries and governing rules for the LK20 digital twin. The system is "Privacy-by-Design," ensuring that student data is decoupled from public curriculum governance.

## 1. Data Classification

| Layer | Type | Visibility | Governance |
| :--- | :--- | :--- | :--- |
| **L0: Canonical** | National Curriculum (LK20) | Public | National (Udir) |
| **L1: Local** | School/Teacher Plans | School-Internal | School Leader |
| **L2: Evidence** | Student Work / Proof | Private | Teacher/Parent |
| **L3: Metadata** | Audit Logs / Coverage | Protected | Admin |

## 2. Privacy Boundaries

### Boundary A: Student De-identification
- **Rule**: Student evidence (L2) must never be stored in the primary TTN (Tree Tensor Network) nodes.
- **Implementation**: The `digital_twin_kernel` stores a Merkle hash (pointer) to the evidence in the `local_curriculum_exception` or `student_evidence` nodes, while the raw data remains in a separate encrypted blob-store.

### Boundary B: Role-Based Projection
- **Guest**: Sees only L0 (Canonical) and aggregated L3 (Public Coverage).
- **Teacher**: Sees L0, L1 (Their school), and L2 (Their students).
- **Government**: Sees L0 and aggregated L3. No access to L1 or L2 details.

### Boundary C: DPIA Enforcement
- **Requirement**: Any upload marked `requires_dpia=True` must undergo a verification step before the TTN root is updated.
- **Workflow**: Upload -> Quarantine -> Manual Approval -> Integration.

## 3. Governance Constraints
- **Local-First**: Data is stored locally on the school/municipality infrastructure.
- **Governed Audit**: Every access to Boundary B/C is logged with a role-signature.
- **Merkle Verification**: The `verify_network_integrity.py` script checks that no unauthorized data-injection has occurred in the protected nodes.
