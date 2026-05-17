#!/usr/bin/env python3
# lk20_kernel.py
r"""
Project Chimera / Akkurat - LK20 Curriculum Digital Twin Kernel
===============================================================

Production LK20-domain adapter for the Akkurat governed digital-twin stack.

This module depends on:

    digital_twin_kernel.py

and optionally detects / cooperates with:

    twin_anything.py
    tn.py

It does not reimplement digital_twin_kernel.py or tn.py.

Purpose
-------
Build and operate an LK20 curriculum digital twin for Norway's school system,
including:

- Canonical national LK20 / Udir / Grep curriculum substrate.
- Grade/stage structure from 1st grade through Vg4/påbygging.
- Grunnskole subject shells.
- Videregående programme shells.
- Local school curriculum overlays.
- School upload manifests.
- Upload validation and quarantine.
- Coverage/alignment placeholders ready for Grep population.
- Versioning, Merkle audit, and federation-ready snapshots.

Design rule
-----------
Udir/Grep canonical curriculum data is authoritative.

School uploads are local overlays. They may bind to, annotate, operationalize,
or provide evidence against canonical nodes, but they must not mutate canonical
national curriculum nodes.

Primary LK20 branch
-------------------
    7.0 LK20 Curriculum Twin
    ├── 7.1  Canonical Governance / Source of Truth
    ├── 7.2  Grade / Stage Structure
    ├── 7.3  Subject Curriculum Registry
    ├── 7.4  Upper Secondary Programme Structure
    ├── 7.5  Local School Curriculum Overlay
    ├── 7.6  Uploads / Artefacts / Evidence
    ├── 7.7  Coverage / Alignment / Gap Model
    ├── 7.8  Assessment / Vurdering Model
    ├── 7.9  Privacy / Access / Data Protection
    ├── 7.10 Federation / Versioning / Merkle Audit
    └── 7.11 Memory / Logs / Snapshots

Public API
----------
- LK20KernelConfig
- LK20SourceSnapshot
- LK20UploadManifest
- LK20ValidationResult
- LK20BuildResult
- LK20KernelFactory
- build_lk20_twin(...)
- create_lk20_twin(...)
- status_report(...)

CLI examples
------------
Status:

    python lk20_kernel.py --mode status

Create empty LK20 twin:

    python lk20_kernel.py --mode create --output lk20_empty_twin.json

Create sample upload manifest:

    python lk20_kernel.py --mode sample-upload --output sample_upload_manifest.json

Validate upload manifest:

    python lk20_kernel.py --mode validate-upload --manifest sample_upload_manifest.json

Attach upload manifest into a twin:

    python lk20_kernel.py --mode attach-upload ^
      --input-network lk20_empty_twin.json ^
      --manifest sample_upload_manifest.json ^
      --output lk20_with_upload.json

Create sample canonical snapshot:

    python lk20_kernel.py --mode sample-canonical --output sample_canonical_snapshot.json

Ingest canonical snapshot:

    python lk20_kernel.py --mode ingest-canonical ^
      --input-network lk20_empty_twin.json ^
      --canonical sample_canonical_snapshot.json ^
      --output lk20_with_canonical.json
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib
import json
import math
import mimetypes
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Version
# =============================================================================

LK20_KERNEL_VERSION = "lk20-kernel-v1.0-production-dtk4-tn31"


# =============================================================================
# Path bootstrap
# =============================================================================

def _module_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()


def _add_path(path: Union[str, os.PathLike, None], *, prepend: bool = True) -> Optional[Path]:
    if path is None:
        return None

    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        return None

    if not p.exists():
        return None

    s = str(p)
    if s not in sys.path:
        if prepend:
            sys.path.insert(0, s)
        else:
            sys.path.append(s)

    return p


def configure_paths(
    *,
    project_root: Optional[Union[str, os.PathLike]] = None,
    akkurat_root: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Optional[str]]:
    here = _module_dir()

    candidates = [
        project_root,
        akkurat_root,
        os.environ.get("LK20_ROOT"),
        os.environ.get("AKKURAT_ROOT"),
        os.environ.get("AKKURAT_COGNITIVE_ROOT"),
        here,
        here.parent,
        here.parent / "app",
        here.parent / "Akkurat",
        here.parent / "Akkurat" / "qwen3.5_4b",
        Path.cwd(),
        Path("/mnt/data"),
    ]

    found_dtk = None
    found_tn = None
    found_twin_anything = None
    found_self = None

    for candidate in candidates:
        p = _add_path(candidate, prepend=True)
        if p is None:
            continue

        if found_dtk is None and (p / "digital_twin_kernel.py").exists():
            found_dtk = p

        if found_tn is None and (p / "tn.py").exists():
            found_tn = p

        if found_twin_anything is None and (p / "twin_anything.py").exists():
            found_twin_anything = p

        if found_self is None and (p / "lk20_kernel.py").exists():
            found_self = p

    _add_path(here, prepend=True)

    return {
        "module_dir": str(here),
        "project_root": None if project_root is None else str(Path(project_root).expanduser()),
        "digital_twin_kernel_root": None if found_dtk is None else str(found_dtk),
        "tn_root": None if found_tn is None else str(found_tn),
        "twin_anything_root": None if found_twin_anything is None else str(found_twin_anything),
        "lk20_kernel_root": None if found_self is None else str(found_self),
    }


_PATHS = configure_paths()


# =============================================================================
# Generic helpers
# =============================================================================

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stable_hash_u32(s: str) -> int:
    h = 2166136261
    for b in (s or "").encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _slug(value: Any, default: str = "item", max_len: int = 96) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = default
    if s[0].isdigit():
        s = f"{default}_{s}"
    return s[: int(max_len)]


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_mapping(x: Any) -> Dict[str, Any]:
    if isinstance(x, Mapping):
        return dict(x)
    return {}


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, set):
        return list(x)
    return [x]


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)

    if isinstance(obj, np.generic):
        return _json_safe(obj.item())

    if isinstance(obj, np.ndarray):
        return np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0).astype(float).tolist()

    if is_dataclass(obj):
        return _json_safe(asdict(obj))

    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass

    if hasattr(obj, "describe") and callable(obj.describe):
        try:
            return _json_safe(obj.describe())
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return _json_safe(vars(obj))
        except Exception:
            pass

    return str(obj)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    b = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False).encode(
        "utf-8",
        errors="ignore",
    )
    return hashlib.sha256(b).hexdigest()


def _atomic_write_text(path: Union[str, os.PathLike], text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, p)


def _write_json(path: Union[str, os.PathLike], payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
    )


def _read_json(path: Union[str, os.PathLike], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(default)

    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return copy.deepcopy(default)


def _sha256_file(path: Union[str, os.PathLike]) -> Optional[str]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None

    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def _guess_mime(filename: str) -> str:
    if not filename:
        return "application/octet-stream"
    guess, _ = mimetypes.guess_type(filename)
    return guess or "application/octet-stream"


def _dataclass_kwargs(cls: Any, values: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        names = {f.name for f in dataclasses.fields(cls)}
        return {k: v for k, v in dict(values).items() if k in names}
    except Exception:
        return dict(values)


# =============================================================================
# Dependency status
# =============================================================================

def digital_twin_status() -> Dict[str, Any]:
    configure_paths()

    try:
        dtk = importlib.import_module("digital_twin_kernel")
        return {
            "available": True,
            "module": getattr(dtk, "__file__", ""),
            "attrs": {
                "DIGITAL_TWIN_KERNEL_VERSION": hasattr(dtk, "DIGITAL_TWIN_KERNEL_VERSION"),
                "DigitalTwinsBuilder": hasattr(dtk, "DigitalTwinsBuilder"),
                "TreeTensorNetwork": hasattr(dtk, "TreeTensorNetwork"),
                "AkkuratInterface": hasattr(dtk, "AkkuratInterface"),
                "NodeKind": hasattr(dtk, "NodeKind"),
                "FusionConfig": hasattr(dtk, "FusionConfig"),
                "FusionMode": hasattr(dtk, "FusionMode"),
                "LatentGeometry": hasattr(dtk, "LatentGeometry"),
                "MerkleHasher": hasattr(dtk, "MerkleHasher"),
                "Action": hasattr(dtk, "Action"),
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "module": None,
            "error": repr(exc),
            "attrs": {},
        }


def tn_status() -> Dict[str, Any]:
    configure_paths()

    try:
        tn = importlib.import_module("tn")
        return {
            "available": True,
            "module": getattr(tn, "__file__", ""),
            "attrs": {
                "TensorTrain": hasattr(tn, "TensorTrain"),
                "TTConfig": hasattr(tn, "TTConfig"),
                "TuckerFusionLayer": hasattr(tn, "TuckerFusionLayer"),
                "tensor_train_svd": hasattr(tn, "tensor_train_svd"),
                "get_factors": hasattr(tn, "get_factors"),
                "factorize_into_modes": hasattr(tn, "factorize_into_modes"),
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "module": None,
            "error": repr(exc),
            "attrs": {},
        }


def twin_anything_status() -> Dict[str, Any]:
    configure_paths()

    try:
        ta = importlib.import_module("twin_anything")
        return {
            "available": True,
            "module": getattr(ta, "__file__", ""),
            "attrs": {
                "TwinAnythingFactory": hasattr(ta, "TwinAnythingFactory"),
                "TwinAnythingConfig": hasattr(ta, "TwinAnythingConfig"),
                "build_anything_twin": hasattr(ta, "build_anything_twin"),
                "create_twin": hasattr(ta, "create_twin"),
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "module": None,
            "error": repr(exc),
            "attrs": {},
        }


def status_report() -> Dict[str, Any]:
    return {
        "ok": True,
        "ts": _now_iso(),
        "version": LK20_KERNEL_VERSION,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "paths": configure_paths(),
        "digital_twin_kernel": digital_twin_status(),
        "tn": tn_status(),
        "twin_anything": twin_anything_status(),
    }


# =============================================================================
# LK20 constants
# =============================================================================

LK20_ROOT_ID = "7.0"

LK20_BRANCHES: Tuple[Tuple[str, str, str], ...] = (
    ("7.1", "Canonical Governance / Source of Truth", "governance"),
    ("7.2", "Grade / Stage Structure", "entity"),
    ("7.3", "Subject Curriculum Registry", "model"),
    ("7.4", "Upper Secondary Programme Structure", "model"),
    ("7.5", "Local School Curriculum Overlay", "storage"),
    ("7.6", "Uploads / Artefacts / Evidence", "storage"),
    ("7.7", "Coverage / Alignment / Gap Model", "observable"),
    ("7.8", "Assessment / Vurdering Model", "governance"),
    ("7.9", "Privacy / Access / Data Protection", "governance"),
    ("7.10", "Federation / Versioning / Merkle Audit", "governance"),
    ("7.11", "Memory / Logs / Snapshots", "storage"),
)

LK20_GOVERNANCE_NODES: Tuple[Tuple[str, str, str, Dict[str, Any]], ...] = (
    (
        "7.1.1",
        "Overordnet del",
        "governance",
        {
            "canonical": True,
            "content_status": "empty",
            "source_expected": "Utdanningsdirektoratet",
            "mutation_policy": "canonical_population_only",
        },
    ),
    (
        "7.1.2",
        "Fag- og timefordeling",
        "governance",
        {
            "canonical": True,
            "content_status": "empty",
            "source_expected": "Udir circular / timetable regulations",
            "mutation_policy": "canonical_population_only",
        },
    ),
    (
        "7.1.3",
        "Grep Canonical Registry",
        "data",
        {
            "canonical": True,
            "content_status": "empty",
            "source_expected": "Grep API / Udir machine-readable curriculum data",
            "mutation_policy": "canonical_population_only",
        },
    ),
    (
        "7.1.4",
        "Tilbudsstruktur Videregående Opplæring",
        "governance",
        {
            "canonical": True,
            "content_status": "empty",
            "source_expected": "Udir programme structure",
            "mutation_policy": "canonical_population_only",
        },
    ),
    (
        "7.1.5",
        "LK20 Version Ledger",
        "storage",
        {
            "canonical": True,
            "content_status": "empty",
            "purpose": "Track source snapshots, effective dates, hashes, and deprecations.",
        },
    ),
)

LK20_GRADE_MAP: Dict[str, Dict[str, Any]] = {
    "G1": {
        "node_id": "7.2.1",
        "label": "1. trinn",
        "stage": "primary_1_4",
        "curriculum_block": "1-4",
        "competence_checkpoint": "after_grade_2",
        "grade_equivalent": 1,
        "school_upload_allowed": True,
    },
    "G2": {
        "node_id": "7.2.2",
        "label": "2. trinn",
        "stage": "primary_1_4",
        "curriculum_block": "1-4",
        "competence_checkpoint": "after_grade_2",
        "grade_equivalent": 2,
        "school_upload_allowed": True,
    },
    "G3": {
        "node_id": "7.2.3",
        "label": "3. trinn",
        "stage": "primary_1_4",
        "curriculum_block": "1-4",
        "competence_checkpoint": "after_grade_4",
        "grade_equivalent": 3,
        "school_upload_allowed": True,
    },
    "G4": {
        "node_id": "7.2.4",
        "label": "4. trinn",
        "stage": "primary_1_4",
        "curriculum_block": "1-4",
        "competence_checkpoint": "after_grade_4",
        "grade_equivalent": 4,
        "school_upload_allowed": True,
    },
    "G5": {
        "node_id": "7.2.5",
        "label": "5. trinn",
        "stage": "primary_5_7",
        "curriculum_block": "5-7",
        "competence_checkpoint": "after_grade_7",
        "grade_equivalent": 5,
        "school_upload_allowed": True,
    },
    "G6": {
        "node_id": "7.2.6",
        "label": "6. trinn",
        "stage": "primary_5_7",
        "curriculum_block": "5-7",
        "competence_checkpoint": "after_grade_7",
        "grade_equivalent": 6,
        "school_upload_allowed": True,
    },
    "G7": {
        "node_id": "7.2.7",
        "label": "7. trinn",
        "stage": "primary_5_7",
        "curriculum_block": "5-7",
        "competence_checkpoint": "after_grade_7",
        "grade_equivalent": 7,
        "school_upload_allowed": True,
    },
    "G8": {
        "node_id": "7.2.8",
        "label": "8. trinn",
        "stage": "lower_secondary_8_10",
        "curriculum_block": "8-10",
        "competence_checkpoint": "after_grade_10",
        "grade_equivalent": 8,
        "school_upload_allowed": True,
        "choice_group_required": True,
    },
    "G9": {
        "node_id": "7.2.9",
        "label": "9. trinn",
        "stage": "lower_secondary_8_10",
        "curriculum_block": "8-10",
        "competence_checkpoint": "after_grade_10",
        "grade_equivalent": 9,
        "school_upload_allowed": True,
        "choice_group_required": True,
    },
    "G10": {
        "node_id": "7.2.10",
        "label": "10. trinn",
        "stage": "lower_secondary_8_10",
        "curriculum_block": "8-10",
        "competence_checkpoint": "after_grade_10",
        "grade_equivalent": 10,
        "transition_target": "VG1",
        "school_upload_allowed": True,
        "choice_group_required": True,
    },
    "VG1": {
        "node_id": "7.2.11",
        "label": "Vg1",
        "stage": "upper_secondary",
        "grade_equivalent": 11,
        "paths": ["study_preparatory", "vocational"],
        "school_upload_allowed": True,
    },
    "VG2": {
        "node_id": "7.2.12",
        "label": "Vg2",
        "stage": "upper_secondary",
        "grade_equivalent": 12,
        "paths": ["study_preparatory_programme_area", "vocational_programme_area"],
        "school_upload_allowed": True,
    },
    "VG3": {
        "node_id": "7.2.13",
        "label": "Vg3",
        "stage": "upper_secondary",
        "grade_equivalent": 13,
        "paths": ["study_preparatory_completion", "vocational_vg3", "apprenticeship_training"],
        "school_upload_allowed": True,
    },
    "VG4": {
        "node_id": "7.2.14",
        "label": "Vg4 / Påbygging",
        "stage": "upper_secondary_supplementary",
        "grade_equivalent": 14,
        "optional": True,
        "paths": ["general_study_competence_supplement"],
        "school_upload_allowed": True,
    },
}

GRUNNSKOLE_SUBJECT_SHELLS: Dict[str, Dict[str, Any]] = {
    "KRLE": {"label": "KRLE", "stages": ["1-7", "8-10"]},
    "NOR": {"label": "Norsk", "stages": ["1-4", "5-7", "8-10"]},
    "MAT": {"label": "Matematikk", "stages": ["1-4", "5-7", "8-10"]},
    "NAT": {"label": "Naturfag", "stages": ["1-4", "5-7", "8-10"]},
    "ENG": {"label": "Engelsk", "stages": ["1-4", "5-7", "8-10"]},
    "SAF": {"label": "Samfunnsfag", "stages": ["1-7", "8-10"]},
    "KHV": {"label": "Kunst og håndverk", "stages": ["1-7", "8-10"]},
    "MUS": {"label": "Musikk", "stages": ["1-7", "8-10"]},
    "MHE": {"label": "Mat og helse", "stages": ["1-7", "8-10"]},
    "KRO": {"label": "Kroppsøving", "stages": ["1-7", "8-10"]},
    "FSP": {
        "label": "Fremmedspråk / fordypning / arbeidslivsfag",
        "stages": ["8-10"],
        "choice_group": True,
    },
    "VAL": {"label": "Valgfag", "stages": ["8-10"], "choice_group": True},
    "UTV": {"label": "Utdanningsvalg", "stages": ["8-10"]},
    "FAK": {"label": "Fysisk aktivitet", "stages": ["5-7"]},
}

VGO_PROGRAMME_SHELLS: Dict[str, Dict[str, Any]] = {
    "ST": {
        "label": "Studiespesialisering",
        "family": "study_preparatory",
        "programme_areas": ["realfag", "språk_samfunnsfag_økonomi"],
    },
    "ID": {"label": "Idrettsfag", "family": "study_preparatory"},
    "MDD": {"label": "Musikk, dans og drama", "family": "study_preparatory"},
    "KDA": {"label": "Kunst, design og arkitektur", "family": "study_preparatory"},
    "MK": {"label": "Medier og kommunikasjon", "family": "study_preparatory"},
    "BA": {"label": "Bygg- og anleggsteknikk", "family": "vocational"},
    "EL": {"label": "Elektro og datateknologi", "family": "vocational"},
    "FBIE": {"label": "Frisør, blomster, interiør og eksponeringsdesign", "family": "vocational"},
    "HO": {"label": "Helse- og oppvekstfag", "family": "vocational"},
    "HDP": {"label": "Håndverk, design og produktutvikling", "family": "vocational"},
    "IM": {"label": "Informasjonsteknologi og medieproduksjon", "family": "vocational"},
    "NA": {"label": "Naturbruk", "family": "vocational"},
    "RM": {"label": "Restaurant- og matfag", "family": "vocational"},
    "SR": {"label": "Salg, service og reiseliv", "family": "vocational"},
    "TP": {"label": "Teknologi- og industrifag", "family": "vocational"},
}

UPLOAD_TYPE_SPECS: Dict[str, Dict[str, Any]] = {
    "annual_plan": {
        "label": "Årsplan",
        "allowed_formats": ["pdf", "docx", "xlsx", "csv", "json", "md", "markdown"],
        "requires_grade": True,
        "requires_subject": True,
        "contains_student_data_allowed": False,
    },
    "term_plan": {
        "label": "Terminplan",
        "allowed_formats": ["pdf", "docx", "xlsx", "json", "md", "markdown"],
        "requires_grade": True,
        "requires_subject": True,
        "contains_student_data_allowed": False,
    },
    "unit_plan": {
        "label": "Undervisningsopplegg / periodeplan",
        "allowed_formats": ["pdf", "docx", "json", "md", "markdown"],
        "requires_grade": True,
        "requires_subject": True,
        "contains_student_data_allowed": False,
    },
    "lesson_plan": {
        "label": "Leksjonsplan",
        "allowed_formats": ["pdf", "docx", "json", "md", "markdown"],
        "requires_grade": True,
        "requires_subject": True,
        "contains_student_data_allowed": False,
    },
    "assessment_rubric": {
        "label": "Vurderingsrubrikk",
        "allowed_formats": ["pdf", "docx", "xlsx", "json"],
        "requires_grade": True,
        "requires_subject": True,
        "contains_student_data_allowed": False,
    },
    "student_evidence": {
        "label": "Elevdokumentasjon / kompetansebevis",
        "allowed_formats": ["pdf", "docx", "png", "jpg", "jpeg", "webp", "mp3", "wav", "mp4", "json"],
        "requires_grade": True,
        "requires_subject": True,
        "contains_student_data_allowed": True,
    },
    "local_curriculum_exception": {
        "label": "Lokal tilpasning / unntak",
        "allowed_formats": ["pdf", "docx", "json"],
        "requires_grade": False,
        "requires_subject": False,
        "contains_student_data_allowed": False,
    },
    "canonical_snapshot": {
        "label": "Canonical Udir/Grep Snapshot",
        "allowed_formats": ["json"],
        "requires_grade": False,
        "requires_subject": False,
        "contains_student_data_allowed": False,
    },
}

VISIBILITY_LEVELS = {
    "private_school",
    "municipality",
    "county",
    "national_sandbox",
}

APPROVAL_STATUSES = {
    "draft",
    "approved",
    "rejected",
    "archived",
    "quarantined",
}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class LK20KernelConfig:
    vector_dim: int = 256
    sketch_dim: int = 96
    history_capacity: int = 64
    seed: int = 2027
    use_tn_projection: bool = True
    latent_geometry: str = "euclidean"
    tt_rank: int = 8

    install_platform_taxonomy: bool = True
    install_twin_anything_context: bool = True

    lk20_root_id: str = LK20_ROOT_ID
    project_root: str = ""

    def normalized(self) -> "LK20KernelConfig":
        cfg = copy.deepcopy(self)

        cfg.vector_dim = int(max(8, cfg.vector_dim))
        cfg.sketch_dim = int(max(8, min(int(cfg.sketch_dim), cfg.vector_dim)))
        cfg.history_capacity = int(max(1, cfg.history_capacity))
        cfg.seed = int(cfg.seed)
        cfg.tt_rank = int(max(1, cfg.tt_rank))

        cfg.latent_geometry = str(cfg.latent_geometry or "euclidean").lower().strip()
        if cfg.latent_geometry not in {"euclidean", "hyperbolic", "poincare", "poincaré"}:
            cfg.latent_geometry = "euclidean"

        cfg.use_tn_projection = bool(cfg.use_tn_projection)
        cfg.install_platform_taxonomy = bool(cfg.install_platform_taxonomy)
        cfg.install_twin_anything_context = bool(cfg.install_twin_anything_context)

        cfg.lk20_root_id = str(cfg.lk20_root_id or LK20_ROOT_ID)
        cfg.project_root = str(cfg.project_root or "")

        return cfg


@dataclass
class LK20SourceSnapshot:
    source_name: str = "udir_grep"
    source_url: str = ""
    source_version: str = ""
    retrieved_at: str = ""
    effective_from: str = ""
    payload_hash: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "LK20SourceSnapshot":
        s = copy.deepcopy(self)

        if not s.retrieved_at:
            s.retrieved_at = _now_iso()

        s.payload = _as_mapping(s.payload)
        s.metadata = _as_mapping(s.metadata)

        if not s.payload_hash:
            s.payload_hash = _payload_hash(
                {
                    "source_name": s.source_name,
                    "source_version": s.source_version,
                    "effective_from": s.effective_from,
                    "payload": s.payload,
                }
            )

        return s

    @classmethod
    def empty(cls) -> "LK20SourceSnapshot":
        return cls(
            source_name="not_loaded",
            source_url="",
            source_version="",
            retrieved_at="",
            effective_from="",
            payload_hash="",
            payload={},
            metadata={"content_status": "empty"},
        ).normalized()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LK20SourceSnapshot":
        p = dict(payload or {})
        return cls(
            source_name=str(p.get("source_name", p.get("name", "udir_grep"))),
            source_url=str(p.get("source_url", p.get("url", ""))),
            source_version=str(p.get("source_version", p.get("version", ""))),
            retrieved_at=str(p.get("retrieved_at", "")),
            effective_from=str(p.get("effective_from", "")),
            payload_hash=str(p.get("payload_hash", "")),
            payload=_as_mapping(p.get("payload", {})),
            metadata=_as_mapping(p.get("metadata", {})),
        ).normalized()

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))


@dataclass
class LK20UploadManifest:
    upload_id: str = ""
    school_org_id: str = ""
    school_name: str = ""

    uploaded_by_role: str = "teacher"
    uploaded_by_user_id: str = ""

    upload_type: str = "annual_plan"

    file_path: str = ""
    filename: str = ""
    mime_type: str = ""
    file_sha256: str = ""

    framework: str = "LK20"
    grade: str = ""
    stage: str = ""
    subject_code: str = ""
    subject_name: str = ""
    programme_code: str = ""
    programme_name: str = ""

    competence_aim_ids: List[str] = field(default_factory=list)
    core_element_ids: List[str] = field(default_factory=list)
    interdisciplinary_theme_ids: List[str] = field(default_factory=list)
    basic_skill_ids: List[str] = field(default_factory=list)

    school_year: str = ""
    term: str = "full_year"
    valid_from: str = ""
    valid_to: str = ""

    visibility: str = "private_school"
    contains_student_data: bool = False
    requires_dpia: bool = False
    approval_status: str = "draft"
    approved_by: Optional[str] = None

    extracted_text: str = ""
    extracted_json: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "LK20UploadManifest":
        m = copy.deepcopy(self)

        if not m.upload_id:
            seed = f"{m.school_org_id}:{m.filename}:{m.file_sha256}:{_now_iso()}"
            m.upload_id = f"upl_{_stable_hash_u32(seed):08x}"

        if m.file_path and not m.filename:
            m.filename = Path(m.file_path).name

        if m.file_path and not m.file_sha256:
            m.file_sha256 = _sha256_file(m.file_path) or ""

        if not m.mime_type:
            m.mime_type = _guess_mime(m.filename)

        m.upload_type = _slug(m.upload_type, default="annual_plan")
        m.framework = str(m.framework or "LK20").upper().strip()

        m.grade = str(m.grade or "").upper().strip()
        m.stage = str(m.stage or "").strip()

        if m.grade in LK20_GRADE_MAP and not m.stage:
            m.stage = str(LK20_GRADE_MAP[m.grade].get("stage", ""))

        m.subject_code = str(m.subject_code or "").strip()
        m.subject_name = str(m.subject_name or "").strip()
        m.programme_code = str(m.programme_code or "").upper().strip()
        m.programme_name = str(m.programme_name or "").strip()

        m.visibility = str(m.visibility or "private_school").strip()
        if m.visibility not in VISIBILITY_LEVELS:
            m.visibility = "private_school"

        m.approval_status = str(m.approval_status or "draft").strip()
        if m.approval_status not in APPROVAL_STATUSES:
            m.approval_status = "draft"

        m.competence_aim_ids = [str(x).strip() for x in _as_list(m.competence_aim_ids) if str(x).strip()]
        m.core_element_ids = [str(x).strip() for x in _as_list(m.core_element_ids) if str(x).strip()]
        m.interdisciplinary_theme_ids = [
            str(x).strip() for x in _as_list(m.interdisciplinary_theme_ids) if str(x).strip()
        ]
        m.basic_skill_ids = [str(x).strip() for x in _as_list(m.basic_skill_ids) if str(x).strip()]

        m.extracted_json = _as_mapping(m.extracted_json)
        m.metadata = _as_mapping(m.metadata)

        return m

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LK20UploadManifest":
        p = dict(payload or {})

        uploaded_by = _as_mapping(p.get("uploaded_by", {}))
        source_file = _as_mapping(p.get("source_file", {}))
        binding = _as_mapping(p.get("curriculum_binding", {}))
        validity = _as_mapping(p.get("local_validity", {}))
        governance = _as_mapping(p.get("governance", {}))

        flat = {
            "upload_id": p.get("upload_id", ""),
            "school_org_id": p.get("school_org_id", ""),
            "school_name": p.get("school_name", ""),
            "uploaded_by_role": p.get("uploaded_by_role", uploaded_by.get("role", "teacher")),
            "uploaded_by_user_id": p.get("uploaded_by_user_id", uploaded_by.get("user_id", "")),
            "upload_type": p.get("upload_type", p.get("type", "annual_plan")),
            "file_path": p.get("file_path", source_file.get("file_path", "")),
            "filename": p.get("filename", source_file.get("filename", "")),
            "mime_type": p.get("mime_type", source_file.get("mime_type", "")),
            "file_sha256": p.get("file_sha256", source_file.get("sha256", "")),
            "framework": p.get("framework", binding.get("framework", "LK20")),
            "grade": p.get("grade", binding.get("grade", "")),
            "stage": p.get("stage", binding.get("stage", "")),
            "subject_code": p.get("subject_code", binding.get("subject_code", "")),
            "subject_name": p.get("subject_name", binding.get("subject_name", "")),
            "programme_code": p.get("programme_code", binding.get("programme_code", "")),
            "programme_name": p.get("programme_name", binding.get("programme_name", "")),
            "competence_aim_ids": p.get("competence_aim_ids", binding.get("competence_aim_ids", [])),
            "core_element_ids": p.get("core_element_ids", binding.get("core_element_ids", [])),
            "interdisciplinary_theme_ids": p.get(
                "interdisciplinary_theme_ids",
                binding.get("interdisciplinary_theme_ids", []),
            ),
            "basic_skill_ids": p.get("basic_skill_ids", binding.get("basic_skill_ids", [])),
            "school_year": p.get("school_year", validity.get("school_year", "")),
            "term": p.get("term", validity.get("term", "full_year")),
            "valid_from": p.get("valid_from", validity.get("valid_from", "")),
            "valid_to": p.get("valid_to", validity.get("valid_to", "")),
            "visibility": p.get("visibility", governance.get("visibility", "private_school")),
            "contains_student_data": p.get(
                "contains_student_data",
                governance.get("contains_student_data", False),
            ),
            "requires_dpia": p.get("requires_dpia", governance.get("requires_dpia", False)),
            "approval_status": p.get("approval_status", governance.get("approval_status", "draft")),
            "approved_by": p.get("approved_by", governance.get("approved_by", None)),
            "extracted_text": p.get("extracted_text", ""),
            "extracted_json": p.get("extracted_json", {}),
            "metadata": p.get("metadata", {}),
        }

        return cls(**_dataclass_kwargs(cls, flat)).normalized()

    def manifest_hash(self) -> str:
        m = copy.deepcopy(self.normalized())
        payload = asdict(m)
        payload.pop("extracted_text", None)
        return _payload_hash(payload)

    def to_envelope(self) -> Dict[str, Any]:
        m = self.normalized()

        return {
            "upload_id": m.upload_id,
            "school_org_id": m.school_org_id,
            "school_name": m.school_name,
            "uploaded_by": {
                "role": m.uploaded_by_role,
                "user_id": m.uploaded_by_user_id,
            },
            "upload_type": m.upload_type,
            "source_file": {
                "file_path": m.file_path,
                "filename": m.filename,
                "mime_type": m.mime_type,
                "sha256": m.file_sha256,
            },
            "curriculum_binding": {
                "framework": m.framework,
                "grade": m.grade,
                "stage": m.stage,
                "subject_code": m.subject_code,
                "subject_name": m.subject_name,
                "programme_code": m.programme_code,
                "programme_name": m.programme_name,
                "competence_aim_ids": list(m.competence_aim_ids),
                "core_element_ids": list(m.core_element_ids),
                "interdisciplinary_theme_ids": list(m.interdisciplinary_theme_ids),
                "basic_skill_ids": list(m.basic_skill_ids),
            },
            "local_validity": {
                "school_year": m.school_year,
                "term": m.term,
                "valid_from": m.valid_from,
                "valid_to": m.valid_to,
            },
            "governance": {
                "visibility": m.visibility,
                "contains_student_data": bool(m.contains_student_data),
                "requires_dpia": bool(m.requires_dpia),
                "approval_status": m.approval_status,
                "approved_by": m.approved_by,
            },
            "extracted_text": m.extracted_text,
            "extracted_json": _json_safe(m.extracted_json),
            "metadata": _json_safe(m.metadata),
            "manifest_hash": m.manifest_hash(),
        }


@dataclass
class LK20ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalized_manifest: Optional[LK20UploadManifest] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "normalized_manifest": None
            if self.normalized_manifest is None
            else self.normalized_manifest.to_envelope(),
        }


@dataclass
class LK20BuildResult:
    ok: bool
    network: Any
    control_plane: Any
    metadata: Dict[str, Any]
    node_map: Dict[str, str]
    snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "metadata": _json_safe(self.metadata),
            "node_map": _json_safe(self.node_map),
            "snapshot": _json_safe(self.snapshot),
        }


# =============================================================================
# Factory
# =============================================================================

class LK20KernelFactory:
    VERSION = LK20_KERNEL_VERSION

    def __init__(
        self,
        *,
        vector_dim: int = 256,
        sketch_dim: int = 96,
        history_capacity: int = 64,
        seed: int = 2027,
        use_tn_projection: bool = True,
        latent_geometry: str = "euclidean",
        tt_rank: int = 8,
        install_platform_taxonomy: bool = True,
        install_twin_anything_context: bool = True,
        project_root: Optional[Union[str, os.PathLike]] = None,
        config: Optional[LK20KernelConfig] = None,
    ):
        self.cfg = (
            config
            or LK20KernelConfig(
                vector_dim=vector_dim,
                sketch_dim=sketch_dim,
                history_capacity=history_capacity,
                seed=seed,
                use_tn_projection=use_tn_projection,
                latent_geometry=latent_geometry,
                tt_rank=tt_rank,
                install_platform_taxonomy=install_platform_taxonomy,
                install_twin_anything_context=install_twin_anything_context,
                project_root="" if project_root is None else str(project_root),
            )
        ).normalized()

        configure_paths(project_root=self.cfg.project_root or None)

    # -------------------------------------------------------------------------
    # Main creation API
    # -------------------------------------------------------------------------

    def create_empty_lk20_twin(self) -> LK20BuildResult:
        dtk = importlib.import_module("digital_twin_kernel")

        ttn = self._build_base_network(dtk)
        cp = dtk.AkkuratInterface(ttn, strict=False)

        node_map: Dict[str, str] = {}

        if self.cfg.install_twin_anything_context:
            self._try_install_twin_anything_context(dtk, ttn, node_map)

        node_map.update(self._install_lk20_root(dtk, ttn))
        node_map.update(self._install_governance_layer(dtk, ttn))
        node_map.update(self._install_grade_layer(dtk, ttn))
        node_map.update(self._install_subject_registry_shell(dtk, ttn))
        node_map.update(self._install_vgo_programme_shell(dtk, ttn))
        node_map.update(self._install_local_overlay_layer(dtk, ttn))
        node_map.update(self._install_upload_layer(dtk, ttn))
        node_map.update(self._install_coverage_layer(dtk, ttn))
        node_map.update(self._install_assessment_layer(dtk, ttn))
        node_map.update(self._install_privacy_layer(dtk, ttn))
        node_map.update(self._install_federation_layer(dtk, ttn))
        node_map.update(self._install_memory_layer(dtk, ttn))

        self._install_fusion_profiles(dtk, ttn)
        self._seed_empty_state(ttn)
        self._finalize_histories(ttn)

        snapshot = self._snapshot(dtk, ttn)

        metadata = {
            "factory": "LK20KernelFactory",
            "factory_version": self.VERSION,
            "created_at": _now_iso(),
            "root_node": self.cfg.lk20_root_id,
            "node_count": len(getattr(ttn, "nodes", {})),
            "digital_twin_kernel": getattr(dtk, "__file__", ""),
            "tn": tn_status(),
            "twin_anything": twin_anything_status(),
            "config": _json_safe(self.cfg),
            "content_status": "empty_lk20_substrate",
            "canonical_rule": (
                "Udir/Grep canonical data may populate national nodes. "
                "School uploads must remain local overlays."
            ),
        }

        return LK20BuildResult(
            ok=True,
            network=ttn,
            control_plane=cp,
            metadata=metadata,
            node_map=node_map,
            snapshot=snapshot,
        )

    # -------------------------------------------------------------------------
    # Base network
    # -------------------------------------------------------------------------

    def _build_base_network(self, dtk: Any) -> Any:
        geometry = (
            dtk.LatentGeometry.HYPERBOLIC.value
            if self.cfg.latent_geometry in {"hyperbolic", "poincare", "poincaré"}
            else dtk.LatentGeometry.EUCLIDEAN.value
        )

        if bool(self.cfg.install_platform_taxonomy):
            kwargs = {
                "vector_dim": int(self.cfg.vector_dim),
                "seed": int(self.cfg.seed),
                "sketch_dim": int(self.cfg.sketch_dim),
                "history_capacity": int(self.cfg.history_capacity),
                "use_tn_projection": bool(self.cfg.use_tn_projection),
                "latent_geometry": geometry,
                "tt_rank": int(self.cfg.tt_rank),
            }

            try:
                return dtk.DigitalTwinsBuilder.build_platform(**kwargs)
            except TypeError:
                kwargs.pop("tt_rank", None)
                return dtk.DigitalTwinsBuilder.build_platform(**kwargs)

        return dtk.TreeTensorNetwork(
            vector_dim=int(self.cfg.vector_dim),
            seed=int(self.cfg.seed),
            root_name="LK20 Curriculum Digital Twin",
            sketch_dim=int(self.cfg.sketch_dim),
            history_capacity=int(self.cfg.history_capacity),
            use_tn_projection=bool(self.cfg.use_tn_projection),
            latent_geometry=geometry,
            tt_rank=int(self.cfg.tt_rank),
        )

    def _try_install_twin_anything_context(self, dtk: Any, ttn: Any, node_map: MutableMapping[str, str]) -> None:
        status = twin_anything_status()
        node_map["twin_anything_context"] = "7.ta"

        self._add_node(
            dtk,
            ttn,
            "7.ta",
            "Twin Anything Integration Context",
            parent_id="0",
            kind="model",
            metadata={
                "available": bool(status.get("available", False)),
                "module": status.get("module", ""),
                "status": status,
                "role": "Optional generic factory context. LK20 branch remains owned by lk20_kernel.py.",
            },
        )

    # -------------------------------------------------------------------------
    # Node utilities
    # -------------------------------------------------------------------------

    def _node_kind(self, dtk: Any, kind_name: str) -> Any:
        name = str(kind_name or "generic").lower().strip()

        try:
            mapping = {
                "root": dtk.NodeKind.ROOT,
                "physical": dtk.NodeKind.PHYSICAL,
                "virtual": dtk.NodeKind.VIRTUAL,
                "data": dtk.NodeKind.DATA,
                "entity": dtk.NodeKind.ENTITY,
                "resource": dtk.NodeKind.RESOURCE,
                "sensor": dtk.NodeKind.SENSOR,
                "actuator": dtk.NodeKind.ACTUATOR,
                "property": dtk.NodeKind.PROPERTY,
                "model": dtk.NodeKind.MODEL,
                "pipeline": dtk.NodeKind.PIPELINE,
                "storage": dtk.NodeKind.STORAGE,
                "visual": dtk.NodeKind.VISUAL,
                "cognitive": dtk.NodeKind.COGNITIVE,
                "atomtn": dtk.NodeKind.ATOMTN,
                "quantum": dtk.NodeKind.QUANTUM,
                "flow": dtk.NodeKind.FLOW,
                "governance": dtk.NodeKind.GOVERNANCE,
                "observable": dtk.NodeKind.OBSERVABLE,
                "generic": dtk.NodeKind.GENERIC,
            }
            return mapping.get(name, dtk.NodeKind.GENERIC)
        except Exception:
            return dtk.NodeKind.GENERIC

    def _add_node(
        self,
        dtk: Any,
        ttn: Any,
        node_id: str,
        name: str,
        *,
        parent_id: str,
        kind: str = "generic",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if node_id in getattr(ttn, "nodes", {}):
            try:
                ttn.nodes[node_id].metadata.update(dict(metadata or {}))
                ttn.bump_tick()
            except Exception:
                pass
            return

        ttn.add_node(
            node_id,
            name,
            parent_id=parent_id,
            kind=self._node_kind(dtk, kind),
            metadata=dict(metadata or {}),
        )

    def _safe_update(self, ttn: Any, node_id: str, payload: Any, *, note: str) -> None:
        try:
            ttn.update_node_data(node_id, payload, note=note)
        except TypeError:
            ttn.update_node_data(node_id, payload)

    # -------------------------------------------------------------------------
    # LK20 branch installation
    # -------------------------------------------------------------------------

    def _install_lk20_root(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        root = str(self.cfg.lk20_root_id)

        self._add_node(
            dtk,
            ttn,
            root,
            "LK20 Curriculum Twin",
            parent_id="0",
            kind="governance",
            metadata={
                "domain": "education",
                "country": "Norway",
                "framework": "LK20",
                "content_status": "empty_substrate",
                "installed_by": self.VERSION,
                "canonical_rule": "Udir/Grep canonical; school uploads are overlays.",
                "adapter": "lk20_kernel.py",
            },
        )

        node_map = {"lk20_root": root}

        for node_id, name, kind_name in LK20_BRANCHES:
            self._add_node(
                dtk,
                ttn,
                node_id,
                name,
                parent_id=root,
                kind=kind_name,
                metadata={
                    "content_status": "empty",
                    "branch": _slug(name),
                    "lk20_branch": True,
                },
            )
            node_map[_slug(name)] = node_id

        return node_map

    def _install_governance_layer(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        node_map: Dict[str, str] = {}

        for node_id, name, kind_name, metadata in LK20_GOVERNANCE_NODES:
            self._add_node(
                dtk,
                ttn,
                node_id,
                name,
                parent_id="7.1",
                kind=kind_name,
                metadata=metadata,
            )
            node_map[_slug(name)] = node_id

        return node_map

    def _install_grade_layer(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        node_map: Dict[str, str] = {"grades": "7.2"}

        stage_nodes = {
            "primary_1_4": ("7.2.stage.primary_1_4", "Primary Stage 1–4"),
            "primary_5_7": ("7.2.stage.primary_5_7", "Primary Stage 5–7"),
            "lower_secondary_8_10": ("7.2.stage.lower_secondary_8_10", "Lower Secondary Stage 8–10"),
            "upper_secondary": ("7.2.stage.upper_secondary", "Upper Secondary Vg1–Vg3"),
            "upper_secondary_supplementary": (
                "7.2.stage.upper_secondary_supplementary",
                "Upper Secondary Supplementary / Vg4",
            ),
        }

        for stage, (node_id, label) in stage_nodes.items():
            self._add_node(
                dtk,
                ttn,
                node_id,
                label,
                parent_id="7.2",
                kind="entity",
                metadata={
                    "stage": stage,
                    "content_status": "empty",
                    "stage_shell": True,
                },
            )
            node_map[f"stage_{stage}"] = node_id

        for grade_key, cfg in LK20_GRADE_MAP.items():
            stage = str(cfg.get("stage", ""))
            parent = stage_nodes.get(stage, ("7.2", ""))[0]

            self._add_node(
                dtk,
                ttn,
                cfg["node_id"],
                cfg["label"],
                parent_id=parent,
                kind="entity",
                metadata={
                    **cfg,
                    "grade_key": grade_key,
                    "content_status": "empty",
                    "canonical_content_loaded": False,
                    "local_overlay_allowed": bool(cfg.get("school_upload_allowed", True)),
                },
            )
            node_map[f"grade_{grade_key.lower()}"] = cfg["node_id"]

        return node_map

    def _install_subject_registry_shell(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        node_map: Dict[str, str] = {"subjects": "7.3"}

        self._add_node(
            dtk,
            ttn,
            "7.3.GS",
            "Grunnskole Subject Shells",
            parent_id="7.3",
            kind="model",
            metadata={"content_status": "empty_shells"},
        )

        for subject_key, meta in GRUNNSKOLE_SUBJECT_SHELLS.items():
            subject_node = f"7.3.GS.{subject_key}"

            self._add_node(
                dtk,
                ttn,
                subject_node,
                str(meta.get("label", subject_key)),
                parent_id="7.3.GS",
                kind="model",
                metadata={
                    **meta,
                    "subject_key": subject_key,
                    "content_status": "empty",
                    "source_expected": "Grep subject curriculum",
                },
            )
            node_map[f"subject_{subject_key.lower()}"] = subject_node

            parts = {
                "about": "Om faget",
                "core_elements": "Kjerneelementer",
                "interdisciplinary_themes": "Tverrfaglige temaer",
                "basic_skills": "Grunnleggende ferdigheter",
                "competence_aims": "Kompetansemål",
                "assessment": "Vurderingsordning",
                "local_bindings": "Local School Bindings",
            }

            for part_key, part_label in parts.items():
                part_node = f"{subject_node}.{part_key}"

                self._add_node(
                    dtk,
                    ttn,
                    part_node,
                    part_label,
                    parent_id=subject_node,
                    kind="governance" if part_key == "assessment" else "data",
                    metadata={
                        "subject_key": subject_key,
                        "section": part_key,
                        "content_status": "empty",
                    },
                )
                node_map[f"subject_{subject_key.lower()}_{part_key}"] = part_node

        return node_map

    def _install_vgo_programme_shell(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        node_map: Dict[str, str] = {"vgo_programmes": "7.4"}

        self._add_node(
            dtk,
            ttn,
            "7.4.STUDY",
            "Study-preparatory Programmes",
            parent_id="7.4",
            kind="model",
            metadata={"family": "study_preparatory", "content_status": "empty"},
        )

        self._add_node(
            dtk,
            ttn,
            "7.4.YF",
            "Vocational Programmes",
            parent_id="7.4",
            kind="model",
            metadata={"family": "vocational", "content_status": "empty"},
        )

        for code, meta in VGO_PROGRAMME_SHELLS.items():
            family = str(meta.get("family", ""))
            parent = "7.4.STUDY" if family == "study_preparatory" else "7.4.YF"
            node_id = f"7.4.{code}"

            self._add_node(
                dtk,
                ttn,
                node_id,
                str(meta.get("label", code)),
                parent_id=parent,
                kind="model",
                metadata={
                    **meta,
                    "programme_code_shell": code,
                    "content_status": "empty",
                    "source_expected": "Grep/Udir programme structure",
                },
            )
            node_map[f"programme_{code.lower()}"] = node_id

            for level in ("VG1", "VG2", "VG3"):
                level_node = f"{node_id}.{level}"

                self._add_node(
                    dtk,
                    ttn,
                    level_node,
                    f"{meta.get('label', code)} {level}",
                    parent_id=node_id,
                    kind="model",
                    metadata={
                        "programme_code_shell": code,
                        "level": level,
                        "content_status": "empty",
                    },
                )
                node_map[f"programme_{code.lower()}_{level.lower()}"] = level_node

        self._add_node(
            dtk,
            ttn,
            "7.4.VG4",
            "Vg4 / Påbygging Shell",
            parent_id="7.4",
            kind="model",
            metadata={
                "level": "VG4",
                "optional": True,
                "content_status": "empty",
            },
        )

        node_map["programme_vg4"] = "7.4.VG4"
        return node_map

    def _install_local_overlay_layer(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        nodes = [
            ("7.5.1", "School Registry", "storage"),
            ("7.5.2", "Local Annual Plans", "storage"),
            ("7.5.3", "Local Term Plans", "storage"),
            ("7.5.4", "Local Unit Plans", "storage"),
            ("7.5.5", "Local Lesson Sequences", "storage"),
            ("7.5.6", "Local Assessment Rubrics", "storage"),
            ("7.5.7", "Local Deviations / Adaptations", "governance"),
            ("7.5.8", "Local Approval Workflow", "governance"),
        ]

        node_map = {}

        for node_id, name, kind in nodes:
            self._add_node(
                dtk,
                ttn,
                node_id,
                name,
                parent_id="7.5",
                kind=kind,
                metadata={
                    "content_status": "empty",
                    "local_overlay": True,
                },
            )
            node_map[_slug(name)] = node_id

        return node_map

    def _install_upload_layer(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        nodes = [
            ("7.6.1", "Upload Manifest Index", "storage"),
            ("7.6.2", "Raw Uploaded Artefacts", "storage"),
            ("7.6.3", "Extracted Text / Tables", "data"),
            ("7.6.4", "Curriculum Binding Candidates", "model"),
            ("7.6.5", "Accepted Curriculum Bindings", "data"),
            ("7.6.6", "Quarantined Uploads", "governance"),
            ("7.6.7", "Upload Validation Rules", "governance"),
        ]

        node_map = {}

        for node_id, name, kind in nodes:
            self._add_node(
                dtk,
                ttn,
                node_id,
                name,
                parent_id="7.6",
                kind=kind,
                metadata={
                    "content_status": "empty",
                    "upload_layer": True,
                },
            )
            node_map[_slug(name)] = node_id

        try:
            ttn.nodes["7.6.7"].metadata["upload_type_specs"] = _json_safe(UPLOAD_TYPE_SPECS)
            ttn.nodes["7.6.7"].metadata["visibility_levels"] = sorted(VISIBILITY_LEVELS)
            ttn.nodes["7.6.7"].metadata["approval_statuses"] = sorted(APPROVAL_STATUSES)
            ttn.bump_tick()
        except Exception:
            pass

        return node_map

    def _install_coverage_layer(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        nodes = [
            ("7.7.1", "Competence Aim Coverage", "observable"),
            ("7.7.2", "Core Element Coverage", "observable"),
            ("7.7.3", "Basic Skills Coverage", "observable"),
            ("7.7.4", "Interdisciplinary Theme Coverage", "observable"),
            ("7.7.5", "Grade / Subject Gap Detection", "model"),
            ("7.7.6", "Alignment Scoring", "model"),
            ("7.7.7", "Drift / Version Mismatch Detection", "observable"),
        ]

        node_map = {}

        for node_id, name, kind in nodes:
            self._add_node(
                dtk,
                ttn,
                node_id,
                name,
                parent_id="7.7",
                kind=kind,
                metadata={
                    "content_status": "empty",
                    "analytics_layer": True,
                },
            )
            node_map[_slug(name)] = node_id

        return node_map

    def _install_assessment_layer(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        nodes = [
            ("7.8.1", "Underveisvurdering Model", "governance"),
            ("7.8.2", "Standpunkt Assessment Model", "governance"),
            ("7.8.3", "Exam / Sluttvurdering Rules", "governance"),
            ("7.8.4", "Rubric-to-Competence Binding", "model"),
            ("7.8.5", "Evidence-of-Competence Model", "data"),
        ]

        node_map = {}

        for node_id, name, kind in nodes:
            self._add_node(
                dtk,
                ttn,
                node_id,
                name,
                parent_id="7.8",
                kind=kind,
                metadata={
                    "content_status": "empty",
                    "assessment_layer": True,
                },
            )
            node_map[_slug(name)] = node_id

        return node_map

    def _install_privacy_layer(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        nodes = [
            ("7.9.1", "Access Control Policy", "governance"),
            ("7.9.2", "Student Data Boundary", "governance"),
            ("7.9.3", "DPIA / Risk Assessment Flags", "governance"),
            ("7.9.4", "Pseudonymisation / Minimisation", "governance"),
            ("7.9.5", "Retention / Deletion Policy", "governance"),
        ]

        node_map = {}

        for node_id, name, kind in nodes:
            self._add_node(
                dtk,
                ttn,
                node_id,
                name,
                parent_id="7.9",
                kind=kind,
                metadata={
                    "content_status": "empty",
                    "privacy_layer": True,
                },
            )
            node_map[_slug(name)] = node_id

        return node_map

    def _install_federation_layer(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        nodes = [
            ("7.10.1", "Udir / Grep Snapshot Ledger", "storage"),
            ("7.10.2", "School Snapshot Ledger", "storage"),
            ("7.10.3", "Municipality / County Federation", "governance"),
            ("7.10.4", "Merkle Verification", "governance"),
            ("7.10.5", "Source Version Checks", "governance"),
            ("7.10.6", "Trust Contracts", "governance"),
        ]

        node_map = {}

        for node_id, name, kind in nodes:
            self._add_node(
                dtk,
                ttn,
                node_id,
                name,
                parent_id="7.10",
                kind=kind,
                metadata={
                    "content_status": "empty",
                    "federation_layer": True,
                },
            )
            node_map[_slug(name)] = node_id

        return node_map

    def _install_memory_layer(self, dtk: Any, ttn: Any) -> Dict[str, str]:
        nodes = [
            ("7.11.1", "LK20 Kernel Event Log", "storage"),
            ("7.11.2", "Curriculum Population Log", "storage"),
            ("7.11.3", "Upload Processing Log", "storage"),
            ("7.11.4", "Approval / Rejection Memory", "storage"),
            ("7.11.5", "Saved Snapshots", "storage"),
        ]

        node_map = {}

        for node_id, name, kind in nodes:
            self._add_node(
                dtk,
                ttn,
                node_id,
                name,
                parent_id="7.11",
                kind=kind,
                metadata={
                    "content_status": "empty",
                    "memory_layer": True,
                },
            )
            node_map[_slug(name)] = node_id

        return node_map

    # -------------------------------------------------------------------------
    # Fusion and seeding
    # -------------------------------------------------------------------------

    def _install_fusion_profiles(self, dtk: Any, ttn: Any) -> None:
        fusion_specs = {
            "7.0": dict(attention_beta=2.6, residual_mix=0.16),
            "7.1": dict(attention_beta=2.8, residual_mix=0.12),
            "7.2": dict(attention_beta=2.4, residual_mix=0.18),
            "7.3": dict(attention_beta=2.5, residual_mix=0.16),
            "7.4": dict(attention_beta=2.5, residual_mix=0.16),
            "7.5": dict(attention_beta=2.2, residual_mix=0.20),
            "7.6": dict(attention_beta=2.4, residual_mix=0.18),
            "7.7": dict(attention_beta=2.7, residual_mix=0.14),
            "7.8": dict(attention_beta=2.7, residual_mix=0.14),
            "7.9": dict(attention_beta=3.0, residual_mix=0.10),
            "7.10": dict(attention_beta=2.9, residual_mix=0.12),
            "7.11": dict(attention_beta=2.0, residual_mix=0.24),
        }

        for node_id, spec in fusion_specs.items():
            if node_id not in getattr(ttn, "nodes", {}):
                continue

            try:
                ttn.set_fusion(
                    node_id,
                    dtk.FusionConfig(
                        mode=dtk.FusionMode.ATTENTION,
                        attention_beta=float(spec["attention_beta"]),
                        residual_mix=float(spec["residual_mix"]),
                        auto_tune_beta=True,
                    ),
                )
            except Exception:
                pass

    def _seed_empty_state(self, ttn: Any) -> None:
        seed_payloads = {
            "7.0": {
                "framework": "LK20",
                "domain": "Norwegian curriculum digital twin",
                "content_status": "empty_substrate",
                "canonical_source": "Udir/Grep",
                "local_overlay_policy": "school_uploads_do_not_mutate_canonical_nodes",
                "created_at": _now_iso(),
                "kernel_version": self.VERSION,
            },
            "7.1": {
                "layer": "canonical_governance",
                "content_status": "empty",
                "expected_population": ["overordnet_del", "fag_og_timefordeling", "grep"],
            },
            "7.2": {
                "layer": "grade_stage_structure",
                "grade_keys": sorted(LK20_GRADE_MAP.keys()),
                "content_status": "empty_shells",
            },
            "7.3": {
                "layer": "subject_curriculum_registry",
                "subject_shells": sorted(GRUNNSKOLE_SUBJECT_SHELLS.keys()),
                "content_status": "empty_shells",
            },
            "7.4": {
                "layer": "vgo_programme_structure",
                "programme_shells": sorted(VGO_PROGRAMME_SHELLS.keys()),
                "content_status": "empty_shells",
            },
            "7.5": {
                "layer": "local_school_overlay",
                "content_status": "empty",
                "policy": "local artefacts bind to canonical targets",
            },
            "7.6": {
                "layer": "uploads",
                "upload_types": sorted(UPLOAD_TYPE_SPECS.keys()),
                "content_status": "empty",
            },
            "7.7": {
                "layer": "coverage_alignment_gap_model",
                "metrics": [
                    "competence_aim_coverage",
                    "core_element_coverage",
                    "basic_skills_coverage",
                    "theme_coverage",
                    "alignment_score",
                    "drift_score",
                ],
                "content_status": "empty",
            },
            "7.8": {
                "layer": "assessment_model",
                "content_status": "empty",
            },
            "7.9": {
                "layer": "privacy_access_data_protection",
                "student_data_default_visibility": "private_school",
                "content_status": "empty",
            },
            "7.10": {
                "layer": "federation_versioning_merkle_audit",
                "content_status": "empty",
            },
            "7.11": {
                "layer": "memory_logs_snapshots",
                "created_at": _now_iso(),
                "event_log": [],
                "content_status": "empty",
            },
        }

        for node_id, payload in seed_payloads.items():
            if node_id in getattr(ttn, "nodes", {}):
                self._safe_update(ttn, node_id, payload, note="lk20_empty_state_seed")

    # -------------------------------------------------------------------------
    # Canonical ingestion
    # -------------------------------------------------------------------------

    def ingest_canonical_snapshot(
        self,
        network: Any,
        snapshot: Union[LK20SourceSnapshot, Mapping[str, Any]],
        *,
        strict: bool = False,
    ) -> Dict[str, Any]:
        dtk = importlib.import_module("digital_twin_kernel")

        if isinstance(snapshot, LK20SourceSnapshot):
            snap = snapshot.normalized()
        else:
            snap = LK20SourceSnapshot.from_mapping(snapshot)

        payload = snap.to_dict()
        snap_id = f"canonical_{_slug(snap.source_name, 'source')}_{_slug(snap.source_version or snap.payload_hash[:12], 'version', 32)}"
        node_id = f"7.10.1.{snap_id}"

        self._add_node(
            dtk,
            network,
            node_id,
            f"Canonical Snapshot: {snap.source_name} {snap.source_version}".strip(),
            parent_id="7.10.1",
            kind="data",
            metadata={
                "source_name": snap.source_name,
                "source_version": snap.source_version,
                "payload_hash": snap.payload_hash,
                "retrieved_at": snap.retrieved_at,
                "effective_from": snap.effective_from,
            },
        )

        self._safe_update(network, node_id, payload, note="lk20_canonical_snapshot_ingested")

        # Update ledgers and canonical registry nodes.
        try:
            network.nodes["7.1.5"].metadata.setdefault("snapshots", {})
            network.nodes["7.1.5"].metadata["snapshots"][snap.payload_hash] = node_id
            network.nodes["7.10.1"].metadata.setdefault("snapshot_index", {})
            network.nodes["7.10.1"].metadata["snapshot_index"][snap.payload_hash] = node_id
            network.nodes["7.1.3"].metadata["last_snapshot_hash"] = snap.payload_hash
            network.nodes["7.1.3"].metadata["last_snapshot_node"] = node_id
            network.nodes["7.1.3"].metadata["content_status"] = "snapshot_loaded"
            network.bump_tick()
        except Exception:
            pass

        self._populate_from_canonical_payload(network, snap.payload)
        self._append_log(network, "7.11.2", {"event": "canonical_snapshot_ingested", "snapshot": payload, "node_id": node_id})
        self._finalize_histories(network)

        return {
            "ok": True,
            "snapshot_node_id": node_id,
            "snapshot": payload,
            "merkle_root": self._safe_merkle_hex(network),
        }

    def _populate_from_canonical_payload(self, network: Any, payload: Mapping[str, Any]) -> None:
        """
        Conservative population helper.

        Supported payload sections:
            subjects: list[dict]
            competence_aims: list[dict]
            programmes: list[dict]
            grade_metadata: dict[str, dict]
            timetable: dict
            overordnet_del: dict|str
        """
        dtk = importlib.import_module("digital_twin_kernel")
        p = _as_mapping(payload)

        if "overordnet_del" in p and "7.1.1" in network.nodes:
            self._safe_update(network, "7.1.1", p["overordnet_del"], note="canonical_overordnet_del")

        if "timetable" in p and "7.1.2" in network.nodes:
            self._safe_update(network, "7.1.2", p["timetable"], note="canonical_timetable")

        grade_metadata = _as_mapping(p.get("grade_metadata", {}))
        for grade_key, meta in grade_metadata.items():
            gk = str(grade_key).upper()
            cfg = LK20_GRADE_MAP.get(gk)
            if cfg and cfg["node_id"] in network.nodes:
                network.nodes[cfg["node_id"]].metadata.update(_json_safe(meta))
                network.nodes[cfg["node_id"]].metadata["canonical_content_loaded"] = True

        for subject in _as_list(p.get("subjects", [])):
            if isinstance(subject, Mapping):
                self.upsert_subject_curriculum(network, subject)

        for aim in _as_list(p.get("competence_aims", [])):
            if isinstance(aim, Mapping):
                self.upsert_competence_aim(network, aim)

        for programme in _as_list(p.get("programmes", [])):
            if isinstance(programme, Mapping):
                self.upsert_programme(network, programme)

        try:
            network.bump_tick()
        except Exception:
            pass

    def upsert_subject_curriculum(self, network: Any, subject: Mapping[str, Any]) -> Dict[str, Any]:
        dtk = importlib.import_module("digital_twin_kernel")
        s = dict(subject or {})

        code = str(s.get("code", s.get("subject_code", s.get("id", "")))).strip()
        label = str(s.get("label", s.get("name", s.get("subject_name", code or "Subject")))).strip()
        shell = str(s.get("shell", s.get("subject_shell", ""))).upper().strip()

        if shell and f"7.3.GS.{shell}" in network.nodes:
            node_id = f"7.3.GS.{shell}"
        else:
            node_id = f"7.3.subject.{_slug(code or label, 'subject')}"

        parent = "7.3.GS" if node_id.startswith("7.3.GS.") else "7.3"

        self._add_node(
            dtk,
            network,
            node_id,
            label,
            parent_id=parent,
            kind="model",
            metadata={
                "subject_code": code,
                "subject_name": label,
                "content_status": "canonical_loaded",
                "source": "canonical_snapshot",
                **_json_safe(s),
            },
        )

        sections = {
            "about": s.get("about", s.get("om_faget", {})),
            "core_elements": s.get("core_elements", s.get("kjerneelementer", [])),
            "interdisciplinary_themes": s.get("interdisciplinary_themes", s.get("tverrfaglige_temaer", [])),
            "basic_skills": s.get("basic_skills", s.get("grunnleggende_ferdigheter", [])),
            "assessment": s.get("assessment", s.get("vurderingsordning", {})),
        }

        for section_key, section_payload in sections.items():
            section_node = f"{node_id}.{section_key}"
            self._add_node(
                dtk,
                network,
                section_node,
                section_key.replace("_", " ").title(),
                parent_id=node_id,
                kind="governance" if section_key == "assessment" else "data",
                metadata={
                    "subject_code": code,
                    "section": section_key,
                    "content_status": "canonical_loaded" if section_payload else "empty",
                },
            )
            if section_payload:
                self._safe_update(network, section_node, section_payload, note=f"canonical_subject_{section_key}")

        self._safe_update(network, node_id, s, note="canonical_subject_curriculum")
        return {"ok": True, "node_id": node_id, "subject_code": code}

    def upsert_competence_aim(self, network: Any, aim: Mapping[str, Any]) -> Dict[str, Any]:
        dtk = importlib.import_module("digital_twin_kernel")
        a = dict(aim or {})

        aim_id = str(a.get("id", a.get("aim_id", a.get("competence_aim_id", "")))).strip()
        subject_code = str(a.get("subject_code", a.get("subject", ""))).strip()
        grade = str(a.get("grade", a.get("grade_key", ""))).upper().strip()
        text = str(a.get("text", a.get("description", ""))).strip()

        if not aim_id:
            aim_id = f"aim_{_stable_hash_u32(subject_code + grade + text):08x}"

        subject_node = self._find_subject_node(network, subject_code)
        parent = f"{subject_node}.competence_aims" if subject_node and f"{subject_node}.competence_aims" in network.nodes else "7.3"

        node_id = f"{parent}.{_slug(aim_id, 'aim')}"

        self._add_node(
            dtk,
            network,
            node_id,
            f"Kompetansemål: {aim_id}",
            parent_id=parent,
            kind="data",
            metadata={
                "aim_id": aim_id,
                "subject_code": subject_code,
                "grade": grade,
                "content_status": "canonical_loaded",
            },
        )

        self._safe_update(network, node_id, a, note="canonical_competence_aim")

        if grade in LK20_GRADE_MAP:
            grade_node = LK20_GRADE_MAP[grade]["node_id"]
            try:
                network.nodes[grade_node].metadata.setdefault("competence_aim_ids", [])
                if aim_id not in network.nodes[grade_node].metadata["competence_aim_ids"]:
                    network.nodes[grade_node].metadata["competence_aim_ids"].append(aim_id)
                network.bump_tick()
            except Exception:
                pass

        return {"ok": True, "node_id": node_id, "aim_id": aim_id}

    def upsert_programme(self, network: Any, programme: Mapping[str, Any]) -> Dict[str, Any]:
        dtk = importlib.import_module("digital_twin_kernel")
        p = dict(programme or {})

        code = str(p.get("code", p.get("programme_code", p.get("id", "")))).upper().strip()
        label = str(p.get("label", p.get("name", p.get("programme_name", code or "Programme")))).strip()
        family = str(p.get("family", "")).strip()

        if code in VGO_PROGRAMME_SHELLS:
            node_id = f"7.4.{code}"
        else:
            parent = "7.4.STUDY" if family == "study_preparatory" else "7.4.YF" if family == "vocational" else "7.4"
            node_id = f"{parent}.{_slug(code or label, 'programme')}"

        parent = "7.4"
        if code in VGO_PROGRAMME_SHELLS:
            family = VGO_PROGRAMME_SHELLS[code].get("family", family)
            parent = "7.4.STUDY" if family == "study_preparatory" else "7.4.YF"
        elif "." in node_id:
            parent = ".".join(node_id.split(".")[:-1])

        self._add_node(
            dtk,
            network,
            node_id,
            label,
            parent_id=parent,
            kind="model",
            metadata={
                "programme_code": code,
                "programme_name": label,
                "family": family,
                "content_status": "canonical_loaded",
                **_json_safe(p),
            },
        )

        self._safe_update(network, node_id, p, note="canonical_programme")
        return {"ok": True, "node_id": node_id, "programme_code": code}

    def _find_subject_node(self, network: Any, subject_code: str) -> Optional[str]:
        sc = str(subject_code or "").strip()

        if not sc:
            return None

        for nid, node in getattr(network, "nodes", {}).items():
            meta = getattr(node, "metadata", {})
            if str(meta.get("subject_code", "")).strip() == sc:
                return nid

        shell = sc.split("-", 1)[0].upper()
        if shell in GRUNNSKOLE_SUBJECT_SHELLS and f"7.3.GS.{shell}" in network.nodes:
            return f"7.3.GS.{shell}"

        return None

    # -------------------------------------------------------------------------
    # Schools and uploads
    # -------------------------------------------------------------------------

    def register_school(
        self,
        network: Any,
        *,
        school_org_id: str,
        school_name: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        dtk = importlib.import_module("digital_twin_kernel")

        org = str(school_org_id or "").strip()
        name = str(school_name or org or "Unnamed School").strip()

        if not org:
            org = f"school_{_stable_hash_u32(name):08x}"

        node_id = f"7.5.1.{_slug(org, 'school')}"

        self._add_node(
            dtk,
            network,
            node_id,
            f"School: {name}",
            parent_id="7.5.1",
            kind="entity",
            metadata={
                "school_org_id": org,
                "school_name": name,
                "registered_at": _now_iso(),
                **_json_safe(dict(metadata or {})),
            },
        )

        self._safe_update(
            network,
            node_id,
            {
                "school_org_id": org,
                "school_name": name,
                "metadata": dict(metadata or {}),
            },
            note="school_registered",
        )

        try:
            network.nodes["7.5.1"].metadata.setdefault("school_index", {})
            network.nodes["7.5.1"].metadata["school_index"][org] = node_id
            network.bump_tick()
        except Exception:
            pass

        self._append_log(network, "7.11.1", {"event": "school_registered", "school_org_id": org, "node_id": node_id})
        self._finalize_histories(network)

        return {"ok": True, "school_node_id": node_id, "school_org_id": org}

    def validate_upload_manifest(self, manifest: Union[LK20UploadManifest, Mapping[str, Any]]) -> LK20ValidationResult:
        if isinstance(manifest, LK20UploadManifest):
            m = manifest.normalized()
        else:
            m = LK20UploadManifest.from_mapping(manifest)

        errors: List[str] = []
        warnings: List[str] = []

        spec = UPLOAD_TYPE_SPECS.get(m.upload_type)
        if spec is None:
            errors.append(f"unknown_upload_type:{m.upload_type}")
        else:
            if spec.get("requires_grade") and not m.grade:
                errors.append("missing_grade")

            if spec.get("requires_subject") and not (m.subject_code or m.subject_name):
                errors.append("missing_subject_binding")

            if bool(m.contains_student_data) and not bool(spec.get("contains_student_data_allowed", False)):
                errors.append(f"student_data_not_allowed_for_upload_type:{m.upload_type}")

        if m.framework != "LK20":
            errors.append(f"unsupported_framework:{m.framework}")

        if m.grade and m.grade not in LK20_GRADE_MAP:
            errors.append(f"unknown_grade:{m.grade}")

        if m.grade in LK20_GRADE_MAP:
            expected_stage = str(LK20_GRADE_MAP[m.grade].get("stage", ""))
            if m.stage and expected_stage and m.stage != expected_stage:
                warnings.append(f"stage_mismatch:manifest={m.stage}:expected={expected_stage}")

        if m.programme_code and m.programme_code not in VGO_PROGRAMME_SHELLS and m.grade not in {"VG1", "VG2", "VG3", "VG4"}:
            warnings.append(f"programme_code_present_outside_vgo:{m.programme_code}")

        if m.visibility not in VISIBILITY_LEVELS:
            errors.append(f"invalid_visibility:{m.visibility}")

        if m.approval_status not in APPROVAL_STATUSES:
            errors.append(f"invalid_approval_status:{m.approval_status}")

        if bool(m.contains_student_data):
            if m.visibility != "private_school":
                errors.append("student_data_requires_private_school_visibility")
            if not m.requires_dpia:
                warnings.append("student_data_without_dpia_flag")

        if m.file_path:
            p = Path(m.file_path)
            if not p.exists():
                warnings.append(f"file_path_not_found:{m.file_path}")

        if m.filename and spec is not None:
            suffix = Path(m.filename).suffix.lower().lstrip(".")
            allowed = set(str(x).lower() for x in spec.get("allowed_formats", []))
            if suffix and suffix not in allowed:
                errors.append(f"file_format_not_allowed:{suffix}:upload_type={m.upload_type}")

        if not m.school_org_id:
            warnings.append("missing_school_org_id")

        if m.approval_status == "approved" and not m.approved_by:
            warnings.append("approved_status_without_approved_by")

        if m.upload_type in {"annual_plan", "term_plan", "unit_plan", "assessment_rubric"} and not m.competence_aim_ids:
            warnings.append("no_competence_aim_bindings_yet")

        return LK20ValidationResult(
            ok=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            normalized_manifest=m,
        )

    def attach_upload_manifest(
        self,
        network: Any,
        manifest: Union[LK20UploadManifest, Mapping[str, Any]],
        *,
        strict: bool = False,
    ) -> Dict[str, Any]:
        dtk = importlib.import_module("digital_twin_kernel")

        validation = self.validate_upload_manifest(manifest)
        m = validation.normalized_manifest

        if m is None:
            return {"ok": False, "error": "manifest_normalization_failed"}

        if strict and not validation.ok:
            return validation.to_dict()

        # Register school if present.
        if m.school_org_id or m.school_name:
            self.register_school(
                network,
                school_org_id=m.school_org_id,
                school_name=m.school_name or m.school_org_id,
                metadata={"source": "upload_manifest", "upload_id": m.upload_id},
            )

        parent = "7.6.5" if validation.ok and m.approval_status == "approved" else "7.6.6" if not validation.ok else "7.6.1"
        upload_node_id = f"{parent}.{_slug(m.upload_id, 'upload')}"

        self._add_node(
            dtk,
            network,
            upload_node_id,
            f"Upload: {m.upload_id}",
            parent_id=parent,
            kind="data" if validation.ok else "governance",
            metadata={
                "upload_id": m.upload_id,
                "school_org_id": m.school_org_id,
                "upload_type": m.upload_type,
                "approval_status": m.approval_status,
                "validation_ok": bool(validation.ok),
                "manifest_hash": m.manifest_hash(),
            },
        )

        payload = m.to_envelope()
        payload["validation"] = validation.to_dict()
        payload["coverage_placeholder"] = self.compute_upload_coverage_placeholder(m)

        self._safe_update(
            network,
            upload_node_id,
            payload,
            note="lk20_upload_manifest_attached",
        )

        self._bind_upload_to_overlay_indexes(network, m, upload_node_id, validation)

        self._append_log(
            network,
            "7.11.3",
            {
                "event": "upload_manifest_attached",
                "upload_id": m.upload_id,
                "node_id": upload_node_id,
                "validation": validation.to_dict(),
            },
        )

        self._finalize_histories(network)

        return {
            "ok": bool(validation.ok),
            "upload_node_id": upload_node_id,
            "validation": validation.to_dict(),
            "coverage_placeholder": self.compute_upload_coverage_placeholder(m),
            "manifest": m.to_envelope(),
            "merkle_root": self._safe_merkle_hex(network),
        }

    def _bind_upload_to_overlay_indexes(
        self,
        network: Any,
        manifest: LK20UploadManifest,
        upload_node_id: str,
        validation: LK20ValidationResult,
    ) -> None:
        m = manifest.normalized()

        try:
            network.nodes["7.6.1"].metadata.setdefault("upload_index", {})
            network.nodes["7.6.1"].metadata["upload_index"][m.upload_id] = upload_node_id

            if m.school_org_id:
                network.nodes["7.5.1"].metadata.setdefault("school_uploads", {})
                network.nodes["7.5.1"].metadata["school_uploads"].setdefault(m.school_org_id, [])
                if upload_node_id not in network.nodes["7.5.1"].metadata["school_uploads"][m.school_org_id]:
                    network.nodes["7.5.1"].metadata["school_uploads"][m.school_org_id].append(upload_node_id)

            if m.grade in LK20_GRADE_MAP:
                grade_node = LK20_GRADE_MAP[m.grade]["node_id"]
                network.nodes[grade_node].metadata.setdefault("local_uploads", [])
                if upload_node_id not in network.nodes[grade_node].metadata["local_uploads"]:
                    network.nodes[grade_node].metadata["local_uploads"].append(upload_node_id)

            subject_node = self._find_subject_node(network, m.subject_code)
            if subject_node and subject_node in network.nodes:
                network.nodes[subject_node].metadata.setdefault("local_uploads", [])
                if upload_node_id not in network.nodes[subject_node].metadata["local_uploads"]:
                    network.nodes[subject_node].metadata["local_uploads"].append(upload_node_id)

            if m.upload_type == "annual_plan":
                network.nodes["7.5.2"].metadata.setdefault("uploads", [])
                if upload_node_id not in network.nodes["7.5.2"].metadata["uploads"]:
                    network.nodes["7.5.2"].metadata["uploads"].append(upload_node_id)
            elif m.upload_type == "term_plan":
                network.nodes["7.5.3"].metadata.setdefault("uploads", [])
                if upload_node_id not in network.nodes["7.5.3"].metadata["uploads"]:
                    network.nodes["7.5.3"].metadata["uploads"].append(upload_node_id)
            elif m.upload_type == "unit_plan":
                network.nodes["7.5.4"].metadata.setdefault("uploads", [])
                if upload_node_id not in network.nodes["7.5.4"].metadata["uploads"]:
                    network.nodes["7.5.4"].metadata["uploads"].append(upload_node_id)
            elif m.upload_type == "lesson_plan":
                network.nodes["7.5.5"].metadata.setdefault("uploads", [])
                if upload_node_id not in network.nodes["7.5.5"].metadata["uploads"]:
                    network.nodes["7.5.5"].metadata["uploads"].append(upload_node_id)
            elif m.upload_type == "assessment_rubric":
                network.nodes["7.5.6"].metadata.setdefault("uploads", [])
                if upload_node_id not in network.nodes["7.5.6"].metadata["uploads"]:
                    network.nodes["7.5.6"].metadata["uploads"].append(upload_node_id)
            elif m.upload_type == "local_curriculum_exception":
                network.nodes["7.5.7"].metadata.setdefault("uploads", [])
                if upload_node_id not in network.nodes["7.5.7"].metadata["uploads"]:
                    network.nodes["7.5.7"].metadata["uploads"].append(upload_node_id)

            network.bump_tick()

        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------------------

    def compute_upload_coverage_placeholder(
        self,
        manifest: Union[LK20UploadManifest, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        validation = self.validate_upload_manifest(manifest)
        m = validation.normalized_manifest

        if m is None:
            return {"ok": False, "error": "manifest_normalization_failed"}

        mapped_aims = len(m.competence_aim_ids)
        mapped_core = len(m.core_element_ids)
        mapped_themes = len(m.interdisciplinary_theme_ids)
        mapped_skills = len(m.basic_skill_ids)

        has_subject = bool(m.subject_code or m.subject_name)
        has_grade = bool(m.grade)
        has_validity = bool(m.school_year or m.valid_from or m.valid_to)
        has_evidence = bool(m.extracted_text or m.extracted_json or m.file_sha256 or m.filename)

        readiness_score = 0.0
        readiness_score += 0.20 if has_grade else 0.0
        readiness_score += 0.20 if has_subject else 0.0
        readiness_score += 0.25 if mapped_aims > 0 else 0.0
        readiness_score += 0.10 if mapped_core > 0 else 0.0
        readiness_score += 0.10 if mapped_themes > 0 else 0.0
        readiness_score += 0.05 if mapped_skills > 0 else 0.0
        readiness_score += 0.05 if has_validity else 0.0
        readiness_score += 0.05 if has_evidence else 0.0

        return {
            "ok": bool(validation.ok),
            "coverage_status": "placeholder_until_full_grep_loaded",
            "grade": m.grade,
            "stage": m.stage,
            "subject_code": m.subject_code,
            "subject_name": m.subject_name,
            "programme_code": m.programme_code,
            "mapped_competence_aim_count": mapped_aims,
            "mapped_core_element_count": mapped_core,
            "mapped_interdisciplinary_theme_count": mapped_themes,
            "mapped_basic_skill_count": mapped_skills,
            "readiness_score": round(readiness_score, 4),
            "validation": validation.to_dict(),
        }

    def compute_network_coverage_summary(self, network: Any) -> Dict[str, Any]:
        uploads = []
        upload_index = {}

        try:
            upload_index = dict(network.nodes["7.6.1"].metadata.get("upload_index", {}))
        except Exception:
            upload_index = {}

        for upload_id, node_id in upload_index.items():
            if node_id not in network.nodes:
                continue
            node = network.nodes[node_id]
            uploads.append(
                {
                    "upload_id": upload_id,
                    "node_id": node_id,
                    "metadata": _json_safe(node.metadata),
                }
            )

        by_grade: Dict[str, int] = {}
        by_subject: Dict[str, int] = {}
        by_school: Dict[str, int] = {}

        for item in uploads:
            meta = item.get("metadata", {})
            grade = str(meta.get("grade", meta.get("curriculum_binding", {}).get("grade", "")))
            subject = str(meta.get("subject_code", meta.get("curriculum_binding", {}).get("subject_code", "")))
            school = str(meta.get("school_org_id", ""))

            if grade:
                by_grade[grade] = by_grade.get(grade, 0) + 1
            if subject:
                by_subject[subject] = by_subject.get(subject, 0) + 1
            if school:
                by_school[school] = by_school.get(school, 0) + 1

        summary = {
            "ok": True,
            "ts": _now_iso(),
            "upload_count": len(uploads),
            "by_grade": by_grade,
            "by_subject": by_subject,
            "by_school": by_school,
            "status": "placeholder_summary_until_canonical_competence_targets_loaded",
        }

        if "7.7.1" in network.nodes:
            self._safe_update(network, "7.7.1", summary, note="network_coverage_summary")

        self._finalize_histories(network)
        return summary

    # -------------------------------------------------------------------------
    # Logs, finalization, snapshots
    # -------------------------------------------------------------------------

    def _append_log(self, network: Any, node_id: str, event: Mapping[str, Any]) -> None:
        if node_id not in getattr(network, "nodes", {}):
            return

        try:
            log = network.nodes[node_id].metadata.setdefault("event_log", [])
            rec = {"ts": _now_iso(), **_json_safe(dict(event))}
            log.append(rec)
            log[:] = log[-500:]
            network.bump_tick()
            self._safe_update(network, node_id, {"event_log_tail": log[-25:]}, note="lk20_log_append")
        except Exception:
            pass

    def _safe_merkle_hex(self, network: Any) -> Optional[str]:
        try:
            dtk = importlib.import_module("digital_twin_kernel")
            return dtk.MerkleHasher.hex64(network.merkle_root())
        except Exception:
            return None

    def _finalize_histories(self, ttn: Any) -> None:
        try:
            ttn.push_global_histories()
            return
        except Exception:
            pass

        try:
            for nid in sorted(getattr(ttn, "nodes", {}).keys()):
                if hasattr(ttn, "_push_path_histories"):
                    ttn._push_path_histories(nid)
        except Exception:
            pass

    def _snapshot(self, dtk: Any, ttn: Any) -> Dict[str, Any]:
        try:
            root_hash = dtk.MerkleHasher.hex64(ttn.merkle_root())
        except Exception:
            root_hash = None

        try:
            lk20_measure = ttn.measure(str(self.cfg.lk20_root_id))
        except Exception as exc:
            lk20_measure = {"error": repr(exc)}

        return {
            "ts": _now_iso(),
            "node_count": len(getattr(ttn, "nodes", {})),
            "lk20_root": str(self.cfg.lk20_root_id),
            "merkle_root": root_hash,
            "lk20_measure": _json_safe(lk20_measure),
        }


# =============================================================================
# Convenience functions
# =============================================================================

def build_lk20_twin(
    *,
    config: Optional[LK20KernelConfig] = None,
    **factory_kwargs: Any,
) -> Dict[str, Any]:
    factory = LK20KernelFactory(config=config, **factory_kwargs)
    result = factory.create_empty_lk20_twin()

    return {
        "network": result.network,
        "control_plane": result.control_plane,
        "metadata": result.metadata,
        "node_map": result.node_map,
        "snapshot": result.snapshot,
    }


def create_lk20_twin(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    return build_lk20_twin(*args, **kwargs)


def sample_upload_manifest() -> Dict[str, Any]:
    manifest = LK20UploadManifest(
        school_org_id="NO-SCHOOL-ORG-ID",
        school_name="Example School",
        uploaded_by_role="teacher",
        uploaded_by_user_id="teacher-pseudonymous-id",
        upload_type="annual_plan",
        filename="norsk_5_trinn_arsplan.pdf",
        mime_type="application/pdf",
        grade="G5",
        subject_code="NOR",
        subject_name="Norsk",
        competence_aim_ids=[],
        core_element_ids=[],
        interdisciplinary_theme_ids=[],
        basic_skill_ids=[],
        school_year="2025-2026",
        term="full_year",
        valid_from="2025-08-01",
        valid_to="2026-06-30",
        visibility="private_school",
        contains_student_data=False,
        requires_dpia=False,
        approval_status="draft",
        metadata={
            "note": "Sample manifest. Competence aims are intentionally empty until Grep population is implemented.",
        },
    ).normalized()

    return manifest.to_envelope()


def sample_canonical_snapshot() -> Dict[str, Any]:
    snap = LK20SourceSnapshot(
        source_name="sample_udir_grep",
        source_url="",
        source_version="sample-local-v0",
        effective_from="",
        payload={
            "overordnet_del": {
                "title": "Overordnet del",
                "content_status": "sample_placeholder",
            },
            "timetable": {
                "content_status": "sample_placeholder",
            },
            "subjects": [
                {
                    "code": "NOR",
                    "label": "Norsk",
                    "shell": "NOR",
                    "about": {"content_status": "sample_placeholder"},
                    "core_elements": [],
                    "interdisciplinary_themes": [],
                    "basic_skills": [],
                    "assessment": {},
                }
            ],
            "competence_aims": [
                {
                    "id": "SAMPLE-NOR-G5-001",
                    "subject_code": "NOR",
                    "grade": "G5",
                    "text": "Sample competence aim placeholder.",
                }
            ],
            "programmes": [
                {
                    "code": "ST",
                    "label": "Studiespesialisering",
                    "family": "study_preparatory",
                }
            ],
            "grade_metadata": {
                "G5": {
                    "sample_loaded": True,
                }
            },
        },
        metadata={
            "note": "Local sample only. Replace with real Udir/Grep data ingestion.",
        },
    ).normalized()

    return snap.to_dict()


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Akkurat LK20 Curriculum Digital Twin Kernel")

    p.add_argument(
        "--mode",
        choices=[
            "status",
            "create",
            "sample-upload",
            "validate-upload",
            "attach-upload",
            "sample-canonical",
            "ingest-canonical",
            "coverage-summary",
        ],
        default="status",
    )

    p.add_argument("--output", default="")
    p.add_argument("--manifest", default="")
    p.add_argument("--canonical", default="")
    p.add_argument("--input-network", default="")
    p.add_argument("--project-root", default="")

    p.add_argument("--vector-dim", type=int, default=256)
    p.add_argument("--sketch-dim", type=int, default=96)
    p.add_argument("--history-capacity", type=int, default=64)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--tt-rank", type=int, default=8)

    p.add_argument("--disable-tn-projection", action="store_true")
    p.add_argument("--disable-platform-taxonomy", action="store_true")
    p.add_argument("--disable-twin-anything-context", action="store_true")
    p.add_argument("--latent-geometry", choices=["euclidean", "hyperbolic"], default="euclidean")
    p.add_argument("--strict", action="store_true")

    return p


def _config_from_args(args: argparse.Namespace) -> LK20KernelConfig:
    return LK20KernelConfig(
        vector_dim=int(args.vector_dim),
        sketch_dim=int(args.sketch_dim),
        history_capacity=int(args.history_capacity),
        seed=int(args.seed),
        use_tn_projection=not bool(args.disable_tn_projection),
        latent_geometry=str(args.latent_geometry),
        tt_rank=int(args.tt_rank),
        install_platform_taxonomy=not bool(args.disable_platform_taxonomy),
        install_twin_anything_context=not bool(args.disable_twin_anything_context),
        project_root=str(args.project_root or ""),
    ).normalized()


def _load_or_create_network(args: argparse.Namespace, factory: LK20KernelFactory) -> Any:
    dtk = importlib.import_module("digital_twin_kernel")

    if args.input_network:
        path = Path(args.input_network)
        if not path.exists():
            raise FileNotFoundError(f"input network not found: {args.input_network}")
        return dtk.TreeTensorNetwork.load_json(path)

    return factory.create_empty_lk20_twin().network


def _create_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _config_from_args(args)
    factory = LK20KernelFactory(config=cfg)
    result = factory.create_empty_lk20_twin()
    out = result.to_dict()

    if args.output:
        try:
            result.network.save_json(args.output, include_histories=True)
            out["saved_network_json"] = str(Path(args.output).resolve())
        except Exception:
            _write_json(args.output, out)
            out["saved_summary_json"] = str(Path(args.output).resolve())

    return out


def _validate_upload_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.manifest:
        raise ValueError("--manifest is required for validate-upload mode")

    payload = _read_json(args.manifest, {})
    if not isinstance(payload, Mapping):
        raise ValueError("--manifest must point to a JSON object")

    factory = LK20KernelFactory(config=_config_from_args(args))
    result = factory.validate_upload_manifest(payload).to_dict()

    if args.output:
        _write_json(args.output, result)
        result["saved_validation_json"] = str(Path(args.output).resolve())

    return result


def _attach_upload_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.manifest:
        raise ValueError("--manifest is required for attach-upload mode")

    payload = _read_json(args.manifest, {})
    if not isinstance(payload, Mapping):
        raise ValueError("--manifest must point to a JSON object")

    factory = LK20KernelFactory(config=_config_from_args(args))
    network = _load_or_create_network(args, factory)

    result = factory.attach_upload_manifest(network, payload, strict=bool(args.strict))

    if args.output:
        network.save_json(args.output, include_histories=True)
        result["saved_network_json"] = str(Path(args.output).resolve())

    return result


def _sample_upload_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    payload = sample_upload_manifest()

    if args.output:
        _write_json(args.output, payload)
        payload["saved_sample_manifest_json"] = str(Path(args.output).resolve())

    return payload


def _sample_canonical_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    payload = sample_canonical_snapshot()

    if args.output:
        _write_json(args.output, payload)
        payload["saved_sample_canonical_json"] = str(Path(args.output).resolve())

    return payload


def _ingest_canonical_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.canonical:
        raise ValueError("--canonical is required for ingest-canonical mode")

    payload = _read_json(args.canonical, {})
    if not isinstance(payload, Mapping):
        raise ValueError("--canonical must point to a JSON object")

    factory = LK20KernelFactory(config=_config_from_args(args))
    network = _load_or_create_network(args, factory)

    result = factory.ingest_canonical_snapshot(network, payload, strict=bool(args.strict))

    if args.output:
        network.save_json(args.output, include_histories=True)
        result["saved_network_json"] = str(Path(args.output).resolve())

    return result


def _coverage_summary_from_cli(args: argparse.Namespace) -> Dict[str, Any]:
    factory = LK20KernelFactory(config=_config_from_args(args))
    network = _load_or_create_network(args, factory)

    result = factory.compute_network_coverage_summary(network)

    if args.output:
        network.save_json(args.output, include_histories=True)
        result["saved_network_json"] = str(Path(args.output).resolve())

    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    try:
        configure_paths(project_root=args.project_root or None)

        if args.mode == "status":
            out = status_report()

        elif args.mode == "create":
            out = _create_from_cli(args)

        elif args.mode == "sample-upload":
            out = _sample_upload_from_cli(args)

        elif args.mode == "validate-upload":
            out = _validate_upload_from_cli(args)

        elif args.mode == "attach-upload":
            out = _attach_upload_from_cli(args)

        elif args.mode == "sample-canonical":
            out = _sample_canonical_from_cli(args)

        elif args.mode == "ingest-canonical":
            out = _ingest_canonical_from_cli(args)

        elif args.mode == "coverage-summary":
            out = _coverage_summary_from_cli(args)

        else:
            raise ValueError(f"unknown mode: {args.mode}")

        print(json.dumps(_json_safe(out), indent=2, ensure_ascii=False))
        return 0 if bool(out.get("ok", True)) else 2

    except Exception as exc:
        err = {
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(_json_safe(err), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())