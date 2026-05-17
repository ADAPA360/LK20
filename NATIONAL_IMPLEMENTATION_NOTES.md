# National Implementation Notes: LK20 Digital Twin

This document tracks implementation details related to the Norwegian LK20 (Kunnskapsløftet 2020) curriculum standards and their digital representation.

## 1. Curriculum Structure (Udir)
The twin mirrors the structure provided by the Udir (Utdanningsdirektoratet) API:
- **Kunnskapsområder**: Broad knowledge areas.
- **Kompetansemål**: Specific competence aims for each grade/stage.
- **Tverrgående temaer**: Interdisciplinary themes (Folkehelse, Demokrati, Bærekraft).
- **Kjerneelementer**: Core elements of each subject.

## 2. Integration with National Infrastructure
- **Felles Kommunal Journal (FKJ)**: The twin should ideally synchronize with school administrative systems to pull student/teacher rosters.
- **ID-porten**: Mandatory for teacher/leader authentication in production.
- **Feide**: Alternative authentication path for educational institutions.

## 3. Metadata Standards
- **LOM (Learning Object Metadata)**: Used for indexing teacher-uploaded unit plans.
- **GREP**: The national database for curriculum data. `grep_normalizer.py` and `grep_import_manifest.py` are specifically for handling these data formats.

## 4. Federated Governance
In the national rollout, each municipality or county acts as a "Governance Hub". They can export their "Local Curriculum Provision" (Lokal Læreplanarbeid) via `federation_export.py` to be ingested by others or reported to Udir.
