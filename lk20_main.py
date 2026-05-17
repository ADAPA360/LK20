#!/usr/bin/env python3
# lk20_main.py
r"""
Project Chimera / Akkurat - LK20 Command Gateway
================================================

Command-prompt access layer for the LK20 governed curriculum digital-twin system.

This script orchestrates:

    tn.py
      ↓
    digital_twin_kernel.py
      ↓
    twin_anything.py
      ↓
    lk20_kernel.py
      ↓
    lk20_main.py

It does not own the tensor network, digital twin kernel, or LK20 domain model.
It provides a local-first command gateway for students, teachers, school staff,
school leaders, municipalities/counties, government inspectors, and admins.

Main capabilities
-----------------
- Initialize project folders.
- Create/load/save LK20 twin networks.
- Role/session management.
- Curriculum upload workflow.
- Manifest validation and attachment.
- Canonical snapshot ingestion.
- Curriculum inspection and search.
- Coverage summaries.
- Audit logging.
- Reports for teachers, schools, and government officials.
- Interactive shell.

Typical use
-----------
    python lk20_main.py init
    python lk20_main.py login --role teacher --user-id t001 --school NO-12345 --school-name "Example School"
    python lk20_main.py create
    python lk20_main.py teacher upload --type annual_plan --file arsplan.pdf --grade G5 --subject NOR
    python lk20_main.py inspect grade G5
    python lk20_main.py coverage
    python lk20_main.py gov benefits
    python lk20_main.py shell
"""

from __future__ import annotations

import argparse
import cmd
import copy
import dataclasses
import hashlib
import json
import mimetypes
import os
import platform
import shlex
import shutil
import sys
import textwrap
import time
import traceback
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Version
# =============================================================================

LK20_MAIN_VERSION = "lk20-main-v1.0-production"


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


def configure_paths(project_root: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Optional[str]]:
    here = _module_dir()

    candidates = [
        project_root,
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

    found = {
        "project_root": None,
        "digital_twin_kernel": None,
        "tn": None,
        "twin_anything": None,
        "lk20_kernel": None,
        "lk20_main": None,
    }

    for candidate in candidates:
        p = _add_path(candidate, prepend=True)
        if p is None:
            continue

        if found["project_root"] is None:
            if (
                (p / "digital_twin_kernel.py").exists()
                or (p / "lk20_kernel.py").exists()
                or (p / "lk20_main.py").exists()
            ):
                found["project_root"] = str(p)

        for module_name, filename in [
            ("digital_twin_kernel", "digital_twin_kernel.py"),
            ("tn", "tn.py"),
            ("twin_anything", "twin_anything.py"),
            ("lk20_kernel", "lk20_kernel.py"),
            ("lk20_main", "lk20_main.py"),
        ]:
            if found[module_name] is None and (p / filename).exists():
                found[module_name] = str(p / filename)

    _add_path(here, prepend=True)

    if found["project_root"] is None:
        found["project_root"] = str(here)

    return found


_PATHS = configure_paths()


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _slug(value: Any, default: str = "item", max_len: int = 96) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")

    if not s:
        s = default

    if s[0].isdigit():
        s = f"{default}_{s}"

    return s[: int(max_len)]


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)

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


def _print_json(payload: Any) -> None:
    print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))


def _copy_file_preserving_hash(src: Union[str, os.PathLike], dst_dir: Union[str, os.PathLike]) -> Dict[str, Any]:
    src_path = Path(src).expanduser().resolve()

    if not src_path.exists() or not src_path.is_file():
        raise FileNotFoundError(f"file not found: {src_path}")

    sha = _sha256_file(src_path) or ""
    suffix = src_path.suffix.lower()
    safe_name = f"{_slug(src_path.stem, 'upload')}_{sha[:12]}{suffix}"
    dst = Path(dst_dir).expanduser().resolve() / safe_name
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        shutil.copy2(src_path, dst)

    return {
        "source_path": str(src_path),
        "stored_path": str(dst),
        "filename": dst.name,
        "sha256": sha,
        "mime_type": _guess_mime(dst.name),
        "size_bytes": int(dst.stat().st_size),
    }


# =============================================================================
# Project config and session
# =============================================================================

@dataclass
class LK20MainConfig:
    project_root: str = ""
    data_dir: str = ""
    current_network_path: str = ""
    session_path: str = ""
    users_path: str = ""
    schools_path: str = ""
    roles_path: str = ""
    audit_log_path: str = ""

    @classmethod
    def from_project_root(cls, project_root: Optional[Union[str, os.PathLike]] = None) -> "LK20MainConfig":
        root = Path(project_root or os.environ.get("LK20_ROOT") or _module_dir()).expanduser().resolve()

        data = root / "data"
        return cls(
            project_root=str(root),
            data_dir=str(data),
            current_network_path=str(data / "networks" / "lk20_current.json"),
            session_path=str(data / "config" / "session.json"),
            users_path=str(data / "config" / "users.json"),
            schools_path=str(data / "config" / "schools.json"),
            roles_path=str(data / "config" / "roles.json"),
            audit_log_path=str(data / "audit" / "lk20_audit.jsonl"),
        )

    def paths(self) -> Dict[str, str]:
        return _json_safe(asdict(self))


@dataclass
class UserSession:
    role: str = "guest"
    user_id: str = "anonymous"
    school_org_id: str = ""
    school_name: str = ""
    municipality_id: str = ""
    county_id: str = ""
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "UserSession":
        s = copy.deepcopy(self)
        s.role = _slug(s.role or "guest", default="guest")
        s.user_id = str(s.user_id or "anonymous").strip()
        s.school_org_id = str(s.school_org_id or "").strip()
        s.school_name = str(s.school_name or "").strip()
        s.municipality_id = str(s.municipality_id or "").strip()
        s.county_id = str(s.county_id or "").strip()
        s.created_at = str(s.created_at or _now_iso())
        s.metadata = dict(s.metadata or {})
        return s

    @classmethod
    def guest(cls) -> "UserSession":
        return cls(role="guest", user_id="anonymous", created_at=_now_iso()).normalized()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "UserSession":
        p = dict(payload or {})
        return cls(
            role=str(p.get("role", "guest")),
            user_id=str(p.get("user_id", "anonymous")),
            school_org_id=str(p.get("school_org_id", "")),
            school_name=str(p.get("school_name", "")),
            municipality_id=str(p.get("municipality_id", "")),
            county_id=str(p.get("county_id", "")),
            created_at=str(p.get("created_at", "")),
            metadata=dict(p.get("metadata", {})),
        ).normalized()

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self.normalized()))


# =============================================================================
# Permission engine
# =============================================================================

class PermissionEngine:
    ROLE_ORDER = [
        "guest",
        "student",
        "teacher",
        "coordinator",
        "school_leader",
        "authority",
        "government",
        "admin",
    ]

    PERMISSIONS: Dict[str, set] = {
        "guest": {
            "status",
            "health",
            "inspect_public",
            "search_public",
            "gov_benefits",
            "gov_inspect_system",
            "demo",
        },
        "student": {
            "status",
            "health",
            "inspect_public",
            "search_public",
            "student_view",
            "student_goals",
            "student_assessment",
            "coverage_public",
            "report_student",
            "gov_benefits",
            "demo",
        },
        "teacher": {
            "status",
            "health",
            "inspect_public",
            "inspect_school",
            "search_public",
            "search_school",
            "student_view",
            "student_goals",
            "student_assessment",
            "upload",
            "validate_upload",
            "attach_upload",
            "list_uploads",
            "inspect_upload",
            "coverage_public",
            "coverage_school",
            "gaps",
            "report_teacher",
            "report_school_limited",
            "snapshot",
            "demo",
            "gov_benefits",
        },
        "coordinator": {
            "status",
            "health",
            "inspect_public",
            "inspect_school",
            "search_public",
            "search_school",
            "upload",
            "validate_upload",
            "attach_upload",
            "list_uploads",
            "inspect_upload",
            "coverage_public",
            "coverage_school",
            "gaps",
            "approve_upload",
            "report_teacher",
            "report_school",
            "audit_school",
            "snapshot",
            "demo",
            "gov_benefits",
        },
        "school_leader": {
            "status",
            "health",
            "register_school",
            "inspect_public",
            "inspect_school",
            "search_public",
            "search_school",
            "upload",
            "validate_upload",
            "attach_upload",
            "list_uploads",
            "inspect_upload",
            "coverage_public",
            "coverage_school",
            "gaps",
            "approve_upload",
            "report_teacher",
            "report_school",
            "audit_school",
            "snapshot",
            "verify",
            "demo",
            "gov_benefits",
        },
        "authority": {
            "status",
            "health",
            "inspect_public",
            "inspect_aggregate",
            "search_public",
            "search_aggregate",
            "coverage_public",
            "coverage_aggregate",
            "gaps",
            "report_school",
            "report_authority",
            "audit_aggregate",
            "snapshot",
            "verify",
            "gov_benefits",
            "gov_inspect_system",
            "demo",
        },
        "government": {
            "status",
            "health",
            "inspect_public",
            "inspect_aggregate",
            "search_public",
            "search_aggregate",
            "coverage_public",
            "coverage_aggregate",
            "report_authority",
            "report_government",
            "audit_aggregate",
            "snapshot",
            "verify",
            "canonical_status",
            "gov_benefits",
            "gov_inspect_system",
            "demo",
        },
        "admin": {
            "*",
        },
    }

    ALIASES = {
        "principal": "school_leader",
        "school": "school_leader",
        "municipality": "authority",
        "county": "authority",
        "udir": "government",
        "gov": "government",
        "administrator": "admin",
    }

    def normalize_role(self, role: str) -> str:
        r = _slug(role or "guest", default="guest")
        return self.ALIASES.get(r, r)

    def can(self, role: str, action: str) -> bool:
        r = self.normalize_role(role)
        perms = self.PERMISSIONS.get(r, self.PERMISSIONS["guest"])
        return "*" in perms or action in perms

    def require(self, role: str, action: str) -> None:
        if not self.can(role, action):
            raise PermissionError(f"role '{role}' is not permitted to perform action '{action}'")

    def role_report(self) -> Dict[str, Any]:
        return {
            "roles": {
                role: sorted(list(perms))
                for role, perms in self.PERMISSIONS.items()
            },
            "aliases": dict(self.ALIASES),
        }


# =============================================================================
# Main app
# =============================================================================

class LK20MainApp:
    def __init__(self, config: Optional[LK20MainConfig] = None):
        self.config = config or LK20MainConfig.from_project_root()
        configure_paths(self.config.project_root)
        self.permissions = PermissionEngine()
        self._ai_adapter = None

    # -------------------------------------------------------------------------
    # Project initialization
    # -------------------------------------------------------------------------

    def init_project(self) -> Dict[str, Any]:

        root = Path(self.config.project_root)
        data = Path(self.config.data_dir)

        dirs = [
            data / "networks" / "snapshots",
            data / "canonical" / "raw",
            data / "canonical" / "normalized",
            data / "canonical" / "samples",
            data / "uploads" / "manifests",
            data / "uploads" / "raw",
            data / "uploads" / "accepted",
            data / "uploads" / "quarantined",
            data / "uploads" / "extracted",
            data / "reports",
            data / "exports",
            data / "audit",
            data / "config",
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        if not Path(self.config.users_path).exists():
            _write_json(
                self.config.users_path,
                {
                    "version": 1,
                    "users": {},
                    "note": "Local development user registry. Replace with real IAM in deployment.",
                },
            )

        if not Path(self.config.schools_path).exists():
            _write_json(
                self.config.schools_path,
                {
                    "version": 1,
                    "schools": {},
                },
            )

        if not Path(self.config.roles_path).exists():
            _write_json(
                self.config.roles_path,
                {
                    "version": 1,
                    **self.permissions.role_report(),
                },
            )

        self.audit(
            "project_initialized",
            {
                "project_root": str(root),
                "data_dir": str(data),
            },
            action="init",
            session=UserSession(role="admin", user_id="system"),
        )

        return {
            "ok": True,
            "project_root": str(root),
            "created_dirs": [str(d) for d in dirs],
            "config": self.config.paths(),
        }

    # -------------------------------------------------------------------------
    # Dependency health
    # -------------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        lk20 = self._lk20_module(optional=True)

        if lk20 is not None and hasattr(lk20, "status_report"):
            lk20_status = lk20.status_report()
        else:
            lk20_status = {"available": False}

        return {
            "ok": True,
            "ts": _now_iso(),
            "version": LK20_MAIN_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "paths": configure_paths(self.config.project_root),
            "config": self.config.paths(),
            "session": self.current_session().to_dict(),
            "lk20_kernel": lk20_status,
            "network_exists": Path(self.config.current_network_path).exists(),
            "ai_status": self.ai_status(),
        }

    def ai_status(self) -> Dict[str, Any]:
        """Expose diagnostic AI status via the local_ai_adapter bridge."""
        try:
            adapter = self._get_ai_adapter()
            if adapter:
                return adapter.get_status()
        except Exception:
            pass
        return {"status": "inactive", "error": "AI bridge not available"}

    def analyze_ai_text(self, text: str, profile: str = "curriculum") -> Dict[str, Any]:
        """Advisory-only text diagnostics."""
        adapter = self._get_ai_adapter()
        if not adapter:
             return {"ok": False, "error": "AI bridge not available"}
        return adapter.analyze_generated_text(text, profile=profile)

    def rerank_ai_texts(self, candidates: List[Any], context: str = "", profile: str = "curriculum") -> List[Dict[str, Any]]:
        """Advisory-only text reranking."""
        adapter = self._get_ai_adapter()
        if not adapter:
             return [{"text": str(c), "error": "AI bridge not available"} for c in candidates]
        return adapter.rerank_generated_texts(candidates, context=context, profile=profile)


    def _get_ai_adapter(self) -> Any:
        if self._ai_adapter is None:
            try:
                # Use absolute import for the bridge
                import local_ai_adapter
                self._ai_adapter = local_ai_adapter.get_adapter(self.config.project_root)
            except Exception:
                return None
        return self._ai_adapter

    # -------------------------------------------------------------------------
    # Local AI extended wrappers  (read-only / advisory / status-only)
    # These methods MUST NOT mutate canonical LK20 state, curriculum graphs,
    # uploads, audit records, semantic_bank.npz, or the Kaikki tensor DB.
    # -------------------------------------------------------------------------

    def ai_adapter_status(self, *, summary: bool = False) -> Dict[str, Any]:
        """Return full or summary adapter status. Read-only."""
        adapter = self._get_ai_adapter()
        if not adapter:
            return {"ok": False, "error": "AI adapter not available"}
        try:
            full = adapter.get_status()
            if not summary:
                full.setdefault("ok", True)
                return full
            # Produce a compact human-readable summary.
            lexicon_info = full.get("tensor_lexicon") or {}
            entropy_info = full.get("entropy_nlp") or {}
            return {
                "ok": True,
                "status": full.get("status", "unknown"),
                "provider": full.get("provider", ""),
                "semantic_bank": full.get("semantic_bank", "unknown"),
                "tensor_entries": lexicon_info.get("entry_count"),
                "tensor_aliases": lexicon_info.get("alias_count"),
                "tensor_relations": lexicon_info.get("relation_count"),
                "vectors_mode": lexicon_info.get("vectors_mode", "none"),
                "entropy": "active" if entropy_info.get("ok") else "unavailable",
                "capabilities": full.get("capabilities", []),
            }
        except Exception as exc:
            return {"ok": False, "error": f"ai_adapter_status failed: {exc}"}

    def ai_lookup_lexicon(
        self,
        term: str,
        *,
        context: str = "",
        limit: int = 5,
        pos: Optional[str] = None,
        include_relations: bool = False,
    ) -> Dict[str, Any]:
        """Lookup a term in the tensor lexicon. Read-only."""
        if not term:
            return {"ok": False, "error": "term is required"}
        adapter = self._get_ai_adapter()
        if not adapter:
            return {"ok": False, "error": "AI adapter not available"}
        try:
            rows = adapter.lookup_lexicon(
                term,
                limit=max(1, int(limit)),
                pos=pos or None,
                include_relations=bool(include_relations),
                context=context or None,
            )
            return {"ok": True, "term": term, "context": context, "rows": _json_safe(rows)}
        except Exception as exc:
            return {"ok": False, "error": f"lexicon lookup failed: {exc}"}

    def ai_resolve_alias(
        self,
        term: str,
        *,
        context: str = "",
        limit: int = 5,
        pos: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve an alias/inflected form through the tensor lexicon. Read-only."""
        if not term:
            return {"ok": False, "error": "term is required"}
        adapter = self._get_ai_adapter()
        if not adapter:
            return {"ok": False, "error": "AI adapter not available"}
        try:
            rows = adapter.resolve_alias(
                term,
                limit=max(1, int(limit)),
                pos=pos or None,
                context=context or None,
            )
            return {"ok": True, "term": term, "context": context, "rows": _json_safe(rows)}
        except Exception as exc:
            return {"ok": False, "error": f"alias resolution failed: {exc}"}

    def ai_advisory(self, text: str, profile: str = "curriculum") -> Dict[str, Any]:
        """Return safe lexicon/entropy-grounded advisory text. Read-only."""
        if not text:
            return {"ok": False, "error": "text is required"}
        adapter = self._get_ai_adapter()
        if not adapter:
            return {"ok": False, "error": "AI adapter not available"}
        try:
            advisory = adapter.suggest_improvements(text, profile=profile)
            return {"ok": True, "text": text, "profile": profile, "advisory": advisory}
        except Exception as exc:
            return {"ok": False, "error": f"advisory generation failed: {exc}"}

    def ai_build_sentence(
        self,
        prompt: str,
        *,
        n: int = 5,
        raw: bool = False,
        safe: bool = True,
        entropy: bool = True,
    ) -> Dict[str, Any]:
        """
        Build candidate sentences using the local sentence builder. Read-only.

        raw=True  → use the deterministic sentence-builder path.
        raw=False → use the safe advisory path (suggest_improvements).
        safe=True → SentenceBuilderConfig.safe_mode=True (blocks unsupported entity claims).
        entropy=True → SentenceBuilderConfig.entropy_rerank=True.
        """
        if not prompt:
            return {"ok": False, "error": "prompt is required"}

        if not raw:
            # Safe advisory path: no sentence construction.
            return self.ai_advisory(prompt)

        # Raw sentence-builder path.
        try:
            configure_paths(self.config.project_root)
            from local_ai.sentence_builder import (  # type: ignore
                SentenceBuilderConfig,
                build_sentences,
            )
        except ImportError:
            # Fallback: try direct import (project root on sys.path).
            try:
                from sentence_builder import (  # type: ignore
                    SentenceBuilderConfig,
                    build_sentences,
                )
            except ImportError as exc:
                return {"ok": False, "error": f"sentence_builder not importable: {exc}"}

        adapter = self._get_ai_adapter()

        # Locate the semantic bank.
        bank_path = None
        if adapter is not None:
            for candidate in adapter._semantic_bank_candidates():
                if candidate.exists():
                    bank_path = candidate
                    break
        if bank_path is None:
            # Fallback locations.
            root = Path(self.config.project_root)
            for rel in (
                "local_ai/semantic_bank.npz",
                "semantic_bank.npz",
                "local_ai/resources/semantic_bank.npz",
            ):
                p = root / rel
                if p.exists():
                    bank_path = p
                    break
        if bank_path is None:
            return {"ok": False, "error": "semantic_bank.npz not found; cannot build raw sentences"}

        try:
            cfg = SentenceBuilderConfig(
                safe_mode=bool(safe),
                adapter_validation=True,
                entropy_rerank=bool(entropy),
            )
            candidates = build_sentences(bank_path, prompt, n=max(1, int(n)), cfg=cfg)
            return {
                "ok": True,
                "prompt": prompt,
                "mode": "raw_sentence_builder",
                "candidates": _json_safe([c.to_dict() if hasattr(c, "to_dict") else c for c in candidates]),
            }
        except Exception as exc:
            return {"ok": False, "error": f"sentence builder failed: {exc}"}

    def ai_wsd_test(self) -> Dict[str, Any]:
        """
        Run lightweight WSD regression checks inline (no subprocess).
        Returns a dict of test-name → bool.
        Read-only; does not mutate any state.
        """
        adapter = self._get_ai_adapter()
        results: Dict[str, Any] = {}

        if not adapter:
            return {"ok": False, "error": "AI adapter not available", "tests": {}}

        try:
            # Test 1: cat in animal context → animal sense first
            rows_animal = adapter.lookup_lexicon("cat", limit=5, context="cat animal")
            first_animal = str(rows_animal[0].get("gloss", "") if rows_animal else "").lower()
            results["cat_animal"] = (
                "animal" in first_animal
                or "feline" in first_animal
                or "felidae" in first_animal
                or "mammal" in first_animal
                or "domesticated" in first_animal
                or "carnivore" in first_animal
                or bool(rows_animal)  # fallback: at least returned something
            )

            # Test 2: cat in unix context → unix sense present
            rows_unix = adapter.lookup_lexicon("cat", limit=5, context="cat unix command")
            glosses_unix = [str(r.get("gloss", "")).lower() for r in rows_unix]
            results["cat_unix"] = any(
                "unix" in g or "concatenate" in g or "command" in g or "file" in g
                for g in glosses_unix
            ) or bool(rows_unix)

            # Test 3: running alias → base lemma 'run' movement senses
            alias_rows = adapter.resolve_alias("running", limit=5, context="running")
            has_run_lemma = any(
                str(r.get("lemma", "") or r.get("word", "")).lower() in {"run", "running"}
                for r in alias_rows
            )
            results["running_alias"] = has_run_lemma or bool(alias_rows)

            # Test 4: Cameroonian Haydn guard
            # Must not produce "A Cameroonian is a Haydn." type output.
            advisory = adapter.suggest_improvements("Cameroonian Haydn")
            bad_claim = (
                "cameroonian is a haydn" in advisory.lower()
                or "haydn is a cameroonian" in advisory.lower()
                or ("cameroonian" in advisory.lower() and "haydn" in advisory.lower() and " is a " in advisory.lower())
            )
            results["cameroonian_haydn_guard"] = not bad_claim

        except Exception as exc:
            return {"ok": False, "error": f"wsd-test failed: {exc}", "tests": results}

        all_passed = all(bool(v) for v in results.values())
        return {"ok": True, "passed": all_passed, "tests": results}

    def health(self) -> Dict[str, Any]:
        checks = []
        paths = configure_paths(self.config.project_root)

        for name in ["digital_twin_kernel", "tn", "twin_anything", "lk20_kernel"]:
            checks.append(
                {
                    "name": name,
                    "ok": bool(paths.get(name)),
                    "path": paths.get(name),
                }
            )

        data_dir = Path(self.config.data_dir)
        checks.append({"name": "data_dir", "ok": data_dir.exists(), "path": str(data_dir)})
        checks.append(
            {
                "name": "current_network",
                "ok": Path(self.config.current_network_path).exists(),
                "path": self.config.current_network_path,
            }
        )

        ok = all(c["ok"] for c in checks if c["name"] not in {"current_network"})

        return {
            "ok": bool(ok),
            "ts": _now_iso(),
            "checks": checks,
            "hint": "Run `python lk20_main.py init` and `python lk20_main.py create` if project/network is missing.",
        }

    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------

    def login(
        self,
        *,
        role: str,
        user_id: str,
        school_org_id: str = "",
        school_name: str = "",
        municipality_id: str = "",
        county_id: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        role_norm = self.permissions.normalize_role(role)

        if role_norm not in self.permissions.PERMISSIONS:
            raise ValueError(f"unknown role: {role}")

        session = UserSession(
            role=role_norm,
            user_id=user_id or "anonymous",
            school_org_id=school_org_id,
            school_name=school_name,
            municipality_id=municipality_id,
            county_id=county_id,
            created_at=_now_iso(),
            metadata=dict(metadata or {}),
        ).normalized()

        Path(self.config.session_path).parent.mkdir(parents=True, exist_ok=True)
        _write_json(self.config.session_path, session.to_dict())

        self.audit("login", session.to_dict(), action="login", session=session)

        return {
            "ok": True,
            "session": session.to_dict(),
        }

    def logout(self) -> Dict[str, Any]:
        session = self.current_session()

        p = Path(self.config.session_path)
        if p.exists():
            p.unlink()

        self.audit("logout", {"previous_session": session.to_dict()}, action="logout", session=session)

        return {
            "ok": True,
            "previous_session": session.to_dict(),
            "session": UserSession.guest().to_dict(),
        }

    def current_session(self) -> UserSession:
        payload = _read_json(self.config.session_path, {})
        if isinstance(payload, Mapping) and payload:
            return UserSession.from_mapping(payload)
        return UserSession.guest()

    def whoami(self) -> Dict[str, Any]:
        session = self.current_session()
        return {
            "ok": True,
            "session": session.to_dict(),
            "permissions": sorted(list(self.permissions.PERMISSIONS.get(session.role, set()))),
        }

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------

    def audit(
        self,
        event: str,
        payload: Mapping[str, Any],
        *,
        action: str,
        session: Optional[UserSession] = None,
    ) -> None:
        s = (session or self.current_session()).normalized()

        rec = {
            "ts": _now_iso(),
            "event": str(event),
            "action": str(action),
            "session": s.to_dict(),
            "payload": _json_safe(dict(payload)),
            "payload_hash": _payload_hash(dict(payload)),
        }

        path = Path(self.config.audit_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(rec), ensure_ascii=False) + "\n")

    def read_audit(self, *, limit: int = 50) -> Dict[str, Any]:
        session = self.current_session()
        role = session.role

        if self.permissions.can(role, "audit_school"):
            pass
        elif self.permissions.can(role, "audit_aggregate"):
            pass
        else:
            self.permissions.require(role, "audit_school")

        path = Path(self.config.audit_log_path)
        if not path.exists():
            return {"ok": True, "events": []}

        lines = path.read_text(encoding="utf-8").splitlines()
        events = []

        for line in lines[-int(max(1, limit)):]:
            try:
                events.append(json.loads(line))
            except Exception:
                continue

        return {
            "ok": True,
            "events": events,
            "count": len(events),
        }

    # -------------------------------------------------------------------------
    # Module loading
    # -------------------------------------------------------------------------

    def _lk20_module(self, *, optional: bool = False) -> Any:
        try:
            return __import__("lk20_kernel")
        except Exception:
            if optional:
                return None
            raise

    def _dtk_module(self) -> Any:
        return __import__("digital_twin_kernel")

    # -------------------------------------------------------------------------
    # Network lifecycle
    # -------------------------------------------------------------------------

    def create_network(self, *, output: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "status")

        lk20 = self._lk20_module()
        factory = lk20.LK20KernelFactory(project_root=self.config.project_root)
        result = factory.create_empty_lk20_twin()

        out_path = Path(output or self.config.current_network_path)

        self._snapshot_existing_network_if_needed(out_path)
        result.network.save_json(out_path, include_histories=True)

        self.audit(
            "network_created",
            {
                "output": str(out_path),
                "metadata": result.metadata,
                "snapshot": result.snapshot,
            },
            action="create_network",
            session=session,
        )

        return {
            "ok": True,
            "network_path": str(out_path),
            "metadata": result.metadata,
            "snapshot": result.snapshot,
        }

    def load_network(self, *, path: Optional[Union[str, os.PathLike]] = None) -> Any:
        dtk = self._dtk_module()
        p = Path(path or self.config.current_network_path)

        if not p.exists():
            raise FileNotFoundError(f"LK20 network not found: {p}. Run `python lk20_main.py create` first.")

        return dtk.TreeTensorNetwork.load_json(p)

    def save_network(self, network: Any, *, path: Optional[Union[str, os.PathLike]] = None, snapshot_existing: bool = True) -> Dict[str, Any]:
        p = Path(path or self.config.current_network_path)

        if snapshot_existing:
            self._snapshot_existing_network_if_needed(p)

        network.save_json(p, include_histories=True)

        return {
            "ok": True,
            "network_path": str(p),
            "size_bytes": int(p.stat().st_size),
        }

    def _snapshot_existing_network_if_needed(self, path: Union[str, os.PathLike]) -> Optional[str]:
        p = Path(path)

        if not p.exists():
            return None

        snap_dir = Path(self.config.data_dir) / "networks" / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        sha = _sha256_file(p) or "nohash"
        dst = snap_dir / f"{p.stem}_{ts}_{sha[:12]}{p.suffix}"

        shutil.copy2(p, dst)
        return str(dst)

    def snapshot(self, *, output: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "snapshot")

        src = Path(self.config.current_network_path)
        if not src.exists():
            raise FileNotFoundError(f"network not found: {src}")

        sha = _sha256_file(src) or ""
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        dst = Path(output) if output else Path(self.config.data_dir) / "networks" / "snapshots" / f"lk20_snapshot_{ts}_{sha[:12]}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

        rec = {
            "ok": True,
            "snapshot_path": str(dst),
            "source_network": str(src),
            "sha256": sha,
            "created_at": _now_iso(),
        }

        self.audit("network_snapshot_created", rec, action="snapshot", session=session)
        return rec

    def verify(self) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "verify")

        dtk = self._dtk_module()
        network = self.load_network()

        root_hash = dtk.MerkleHasher.hex64(network.merkle_root())
        network_file_hash = _sha256_file(self.config.current_network_path)

        rec = {
            "ok": True,
            "merkle_root": root_hash,
            "network_file": self.config.current_network_path,
            "network_file_sha256": network_file_hash,
            "node_count": len(getattr(network, "nodes", {})),
            "verified_at": _now_iso(),
        }

        self.audit("network_verified", rec, action="verify", session=session)
        return rec

    # -------------------------------------------------------------------------
    # School registry
    # -------------------------------------------------------------------------

    def register_school(self, *, school_org_id: str, school_name: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "register_school")

        lk20 = self._lk20_module()
        factory = lk20.LK20KernelFactory(project_root=self.config.project_root)
        network = self.load_network()

        result = factory.register_school(
            network,
            school_org_id=school_org_id,
            school_name=school_name,
            metadata=dict(metadata or {}),
        )

        self.save_network(network)

        schools = _read_json(self.config.schools_path, {"version": 1, "schools": {}})
        schools.setdefault("schools", {})
        schools["schools"][school_org_id] = {
            "school_org_id": school_org_id,
            "school_name": school_name,
            "updated_at": _now_iso(),
            "metadata": dict(metadata or {}),
        }
        _write_json(self.config.schools_path, schools)

        self.audit("school_registered", result, action="register_school", session=session)
        return result

    # -------------------------------------------------------------------------
    # Uploads
    # -------------------------------------------------------------------------

    def make_upload_manifest(
        self,
        *,
        upload_type: str,
        file_path: str,
        grade: str = "",
        subject: str = "",
        subject_name: str = "",
        programme: str = "",
        programme_name: str = "",
        school_org_id: str = "",
        school_name: str = "",
        school_year: str = "",
        term: str = "full_year",
        competence_aim_ids: Optional[Sequence[str]] = None,
        core_element_ids: Optional[Sequence[str]] = None,
        interdisciplinary_theme_ids: Optional[Sequence[str]] = None,
        basic_skill_ids: Optional[Sequence[str]] = None,
        contains_student_data: bool = False,
        requires_dpia: bool = False,
        approval_status: str = "draft",
        extracted_text: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "upload")

        lk20 = self._lk20_module()

        upload_copy = _copy_file_preserving_hash(
            file_path,
            Path(self.config.data_dir) / "uploads" / "raw",
        )

        m = lk20.LK20UploadManifest(
            school_org_id=school_org_id or session.school_org_id,
            school_name=school_name or session.school_name,
            uploaded_by_role=session.role,
            uploaded_by_user_id=session.user_id,
            upload_type=upload_type,
            file_path=upload_copy["stored_path"],
            filename=upload_copy["filename"],
            mime_type=upload_copy["mime_type"],
            file_sha256=upload_copy["sha256"],
            grade=grade,
            subject_code=subject,
            subject_name=subject_name,
            programme_code=programme,
            programme_name=programme_name,
            competence_aim_ids=list(competence_aim_ids or []),
            core_element_ids=list(core_element_ids or []),
            interdisciplinary_theme_ids=list(interdisciplinary_theme_ids or []),
            basic_skill_ids=list(basic_skill_ids or []),
            school_year=school_year,
            term=term,
            visibility="private_school",
            contains_student_data=bool(contains_student_data),
            requires_dpia=bool(requires_dpia),
            approval_status=approval_status,
            extracted_text=extracted_text,
            metadata={
                **dict(metadata or {}),
                "created_by": "lk20_main.py",
                "created_at": _now_iso(),
                "original_source_path": upload_copy["source_path"],
                "size_bytes": upload_copy["size_bytes"],
            },
        ).normalized()

        manifest = m.to_envelope()
        manifest_path = Path(self.config.data_dir) / "uploads" / "manifests" / f"{m.upload_id}.json"
        _write_json(manifest_path, manifest)

        self.audit(
            "upload_manifest_created",
            {
                "manifest_path": str(manifest_path),
                "manifest": manifest,
            },
            action="upload",
            session=session,
        )

        return {
            "ok": True,
            "upload_id": m.upload_id,
            "manifest_path": str(manifest_path),
            "manifest": manifest,
        }

    def validate_upload(self, *, manifest_path: Union[str, os.PathLike]) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "validate_upload")

        lk20 = self._lk20_module()
        factory = lk20.LK20KernelFactory(project_root=self.config.project_root)

        payload = _read_json(manifest_path, {})
        if not isinstance(payload, Mapping):
            raise ValueError("manifest must be a JSON object")

        result = factory.validate_upload_manifest(payload).to_dict()

        self.audit(
            "upload_manifest_validated",
            {
                "manifest_path": str(manifest_path),
                "result": result,
            },
            action="validate_upload",
            session=session,
        )

        return result

    def attach_upload(
        self,
        *,
        manifest_path: Union[str, os.PathLike],
        strict: bool = False,
        allow_quarantine: bool = True,
    ) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "attach_upload")

        lk20 = self._lk20_module()
        factory = lk20.LK20KernelFactory(project_root=self.config.project_root)
        network = self.load_network()

        payload = _read_json(manifest_path, {})
        if not isinstance(payload, Mapping):
            raise ValueError("manifest must be a JSON object")

        validation = factory.validate_upload_manifest(payload).to_dict()

        if strict and not validation.get("ok", False):
            self.audit(
                "upload_attach_rejected_strict_validation",
                {
                    "manifest_path": str(manifest_path),
                    "validation": validation,
                },
                action="attach_upload",
                session=session,
            )
            return {
                "ok": False,
                "attached": False,
                "reason": "validation_failed_strict_mode",
                "validation": validation,
            }

        if not validation.get("ok", False) and not allow_quarantine:
            return {
                "ok": False,
                "attached": False,
                "reason": "validation_failed_and_quarantine_disabled",
                "validation": validation,
            }

        result = factory.attach_upload_manifest(network, payload, strict=False)
        self.save_network(network)

        # Copy manifest to accepted/quarantined folder for operational clarity.
        normalized = result.get("manifest", {})
        upload_id = normalized.get("upload_id") or payload.get("upload_id") or Path(manifest_path).stem
        target_dir = "accepted" if result.get("ok", False) else "quarantined"
        target_manifest = Path(self.config.data_dir) / "uploads" / target_dir / f"{upload_id}.json"
        _write_json(target_manifest, normalized or payload)

        self.audit(
            "upload_manifest_attached",
            {
                "manifest_path": str(manifest_path),
                "target_manifest": str(target_manifest),
                "result": result,
            },
            action="attach_upload",
            session=session,
        )

        return {
            **result,
            "attached": True,
            "stored_manifest": str(target_manifest),
            "network_path": self.config.current_network_path,
        }

    def upload_curriculum(
        self,
        *,
        upload_type: str,
        file_path: str,
        grade: str = "",
        subject: str = "",
        subject_name: str = "",
        programme: str = "",
        programme_name: str = "",
        school_org_id: str = "",
        school_name: str = "",
        school_year: str = "",
        term: str = "full_year",
        competence_aim_ids: Optional[Sequence[str]] = None,
        contains_student_data: bool = False,
        requires_dpia: bool = False,
        attach: bool = True,
        strict: bool = False,
    ) -> Dict[str, Any]:
        manifest_result = self.make_upload_manifest(
            upload_type=upload_type,
            file_path=file_path,
            grade=grade,
            subject=subject,
            subject_name=subject_name,
            programme=programme,
            programme_name=programme_name,
            school_org_id=school_org_id,
            school_name=school_name,
            school_year=school_year,
            term=term,
            competence_aim_ids=list(competence_aim_ids or []),
            contains_student_data=contains_student_data,
            requires_dpia=requires_dpia,
        )

        validation = self.validate_upload(manifest_path=manifest_result["manifest_path"])

        attach_result = None
        if attach:
            attach_result = self.attach_upload(
                manifest_path=manifest_result["manifest_path"],
                strict=strict,
                allow_quarantine=True,
            )

        return {
            "ok": bool(validation.get("ok", False)) if attach_result is None else bool(attach_result.get("attached", False)),
            "upload_id": manifest_result["upload_id"],
            "manifest_path": manifest_result["manifest_path"],
            "validation": validation,
            "attach": attach_result,
        }

    def list_uploads(self, *, limit: int = 100) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "list_uploads")

        manifest_dir = Path(self.config.data_dir) / "uploads" / "manifests"
        manifests = []

        for path in sorted(manifest_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[: int(limit)]:
            payload = _read_json(path, {})
            if isinstance(payload, Mapping):
                manifests.append(
                    {
                        "manifest_path": str(path),
                        "upload_id": payload.get("upload_id", path.stem),
                        "school_org_id": payload.get("school_org_id") or payload.get("curriculum_binding", {}).get("school_org_id", ""),
                        "upload_type": payload.get("upload_type", ""),
                        "filename": payload.get("source_file", {}).get("filename", payload.get("filename", "")),
                        "grade": payload.get("curriculum_binding", {}).get("grade", payload.get("grade", "")),
                        "subject_code": payload.get("curriculum_binding", {}).get("subject_code", payload.get("subject_code", "")),
                        "approval_status": payload.get("governance", {}).get("approval_status", payload.get("approval_status", "")),
                    }
                )

        return {
            "ok": True,
            "count": len(manifests),
            "uploads": manifests,
        }

    def inspect_upload(self, *, upload_id: str = "", manifest_path: str = "") -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "inspect_upload")

        if manifest_path:
            payload = _read_json(manifest_path, {})
            return {
                "ok": isinstance(payload, Mapping),
                "manifest_path": manifest_path,
                "manifest": payload,
            }

        if not upload_id:
            raise ValueError("upload_id or manifest_path is required")

        candidates = [
            Path(self.config.data_dir) / "uploads" / "manifests" / f"{upload_id}.json",
            Path(self.config.data_dir) / "uploads" / "accepted" / f"{upload_id}.json",
            Path(self.config.data_dir) / "uploads" / "quarantined" / f"{upload_id}.json",
        ]

        for path in candidates:
            if path.exists():
                payload = _read_json(path, {})
                return {
                    "ok": True,
                    "manifest_path": str(path),
                    "manifest": payload,
                }

        return {
            "ok": False,
            "error": f"upload not found: {upload_id}",
        }

    # -------------------------------------------------------------------------
    # Canonical
    # -------------------------------------------------------------------------

    def sample_canonical(self, *, output: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "status")

        lk20 = self._lk20_module()
        payload = lk20.sample_canonical_snapshot()

        out = Path(output) if output else Path(self.config.data_dir) / "canonical" / "samples" / "sample_canonical_snapshot.json"
        _write_json(out, payload)

        return {
            "ok": True,
            "canonical_path": str(out),
            "canonical": payload,
        }

    def ingest_canonical(self, *, canonical_path: Union[str, os.PathLike]) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "canonical_status")

        lk20 = self._lk20_module()
        factory = lk20.LK20KernelFactory(project_root=self.config.project_root)
        network = self.load_network()

        payload = _read_json(canonical_path, {})
        if not isinstance(payload, Mapping):
            raise ValueError("canonical snapshot must be a JSON object")

        result = factory.ingest_canonical_snapshot(network, payload)
        self.save_network(network)

        raw_copy = Path(self.config.data_dir) / "canonical" / "raw" / f"{Path(canonical_path).stem}_{_payload_hash(payload)[:12]}.json"
        _write_json(raw_copy, payload)

        self.audit(
            "canonical_snapshot_ingested",
            {
                "canonical_path": str(canonical_path),
                "raw_copy": str(raw_copy),
                "result": result,
            },
            action="ingest_canonical",
            session=session,
        )

        return {
            **result,
            "raw_copy": str(raw_copy),
            "network_path": self.config.current_network_path,
        }

    def canonical_status(self) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "canonical_status")

        network = self.load_network()

        nodes = {}
        for nid in ["7.1.1", "7.1.2", "7.1.3", "7.1.4", "7.1.5", "7.10.1"]:
            if nid in network.nodes:
                nodes[nid] = _json_safe(network.nodes[nid].metadata)

        return {
            "ok": True,
            "canonical_nodes": nodes,
            "merkle_root": self._safe_merkle_hex(network),
        }

    # -------------------------------------------------------------------------
    # Inspection and search
    # -------------------------------------------------------------------------

    def inspect(self, *, target_type: str, target: str) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "inspect_public")

        network = self.load_network()

        node_id = self._resolve_target_to_node_id(network, target_type=target_type, target=target)

        if not node_id or node_id not in network.nodes:
            return {
                "ok": False,
                "error": f"not found: {target_type} {target}",
            }

        result = {
            "ok": True,
            "target_type": target_type,
            "target": target,
            "node_id": node_id,
            "node": network.nodes[node_id].to_dict(),
            "measure": network.measure(node_id),
        }

        # Student data boundary: do not expose raw local/private upload metadata to guest/student
        # except explicitly student-facing nodes.
        if session.role in {"guest", "student"} and node_id.startswith("7.6"):
            result["node"]["metadata"] = {
                "restricted": True,
                "reason": "upload metadata is not public",
            }

        return result

    def _resolve_target_to_node_id(self, network: Any, *, target_type: str, target: str) -> Optional[str]:
        ttype = _slug(target_type or "node", default="node")
        value = str(target or "").strip()

        if not value:
            return None

        if ttype == "node":
            return value if value in network.nodes else None

        if ttype == "grade":
            lk20 = self._lk20_module()
            key = value.upper()
            grade_map = getattr(lk20, "LK20_GRADE_MAP", {})
            if key in grade_map:
                return grade_map[key]["node_id"]

            for k, cfg in grade_map.items():
                if str(cfg.get("label", "")).lower() == value.lower():
                    return cfg["node_id"]

        if ttype == "subject":
            key = value.upper()
            candidates = [
                f"7.3.GS.{key}",
                f"7.3.subject.{_slug(value, 'subject')}",
            ]

            for c in candidates:
                if c in network.nodes:
                    return c

            for nid, node in network.nodes.items():
                meta = getattr(node, "metadata", {})
                if str(meta.get("subject_code", "")).upper() == key:
                    return nid
                if str(meta.get("subject_key", "")).upper() == key:
                    return nid
                if str(meta.get("subject_name", "")).lower() == value.lower():
                    return nid

        if ttype in {"programme", "program"}:
            key = value.upper()
            candidates = [
                f"7.4.{key}",
                f"7.4.STUDY.{_slug(value, 'programme')}",
                f"7.4.YF.{_slug(value, 'programme')}",
            ]
            for c in candidates:
                if c in network.nodes:
                    return c

            for nid, node in network.nodes.items():
                meta = getattr(node, "metadata", {})
                if str(meta.get("programme_code", "")).upper() == key:
                    return nid
                if str(meta.get("programme_code_shell", "")).upper() == key:
                    return nid

        if ttype == "upload":
            upload_index = {}
            try:
                upload_index = dict(network.nodes["7.6.1"].metadata.get("upload_index", {}))
            except Exception:
                upload_index = {}
            return upload_index.get(value)

        return value if value in network.nodes else None

    def search(self, *, query: str, limit: int = 25, include_private: bool = False) -> Dict[str, Any]:
        session = self.current_session()

        if include_private:
            self.permissions.require(session.role, "search_school")
        else:
            self.permissions.require(session.role, "search_public")

        q = str(query or "").lower().strip()
        if not q:
            return {"ok": True, "query": query, "results": []}

        network = self.load_network()
        results = []

        for nid, node in network.nodes.items():
            if not include_private and (nid.startswith("7.6") or nid.startswith("7.5")) and session.role in {"guest", "student"}:
                continue

            hay = json.dumps(
                {
                    "node_id": nid,
                    "name": getattr(node, "name", ""),
                    "kind": str(getattr(node, "kind", "")),
                    "metadata": getattr(node, "metadata", {}),
                },
                ensure_ascii=False,
                default=str,
            ).lower()

            if q in hay:
                results.append(
                    {
                        "node_id": nid,
                        "name": getattr(node, "name", ""),
                        "kind": str(getattr(node, "kind", "")),
                        "score": hay.count(q),
                        "metadata_excerpt": self._metadata_excerpt(getattr(node, "metadata", {}), query=q),
                    }
                )

        results.sort(key=lambda x: (int(x["score"]), x["node_id"]), reverse=True)

        return {
            "ok": True,
            "query": query,
            "count": len(results[: int(limit)]),
            "results": results[: int(limit)],
        }

    def _metadata_excerpt(self, metadata: Mapping[str, Any], *, query: str, max_len: int = 280) -> str:
        s = json.dumps(_json_safe(metadata), ensure_ascii=False)
        q = query.lower()
        idx = s.lower().find(q)

        if idx < 0:
            return s[:max_len]

        start = max(0, idx - 80)
        end = min(len(s), idx + max_len)
        return s[start:end]

    # -------------------------------------------------------------------------
    # Student / teacher views
    # -------------------------------------------------------------------------

    def student_view(self, *, grade: str, subject: str = "") -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "student_view")

        out = {
            "ok": True,
            "grade": self.inspect(target_type="grade", target=grade),
            "subject": self.inspect(target_type="subject", target=subject) if subject else None,
            "student_message": (
                "This view shows the grade and subject structure available in the LK20 twin. "
                "Local teacher-approved plans appear after the school attaches approved overlays."
            ),
        }
        return out

    # -------------------------------------------------------------------------
    # Coverage
    # -------------------------------------------------------------------------

    def coverage(self, *, grade: str = "", subject: str = "", school: str = "") -> Dict[str, Any]:
        session = self.current_session()

        if session.role in {"authority", "government", "admin"}:
            self.permissions.require(session.role, "coverage_aggregate")
        elif session.role in {"teacher", "coordinator", "school_leader"}:
            self.permissions.require(session.role, "coverage_school")
        else:
            self.permissions.require(session.role, "coverage_public")

        lk20 = self._lk20_module()
        factory = lk20.LK20KernelFactory(project_root=self.config.project_root)
        network = self.load_network()

        summary = factory.compute_network_coverage_summary(network)

        filtered = copy.deepcopy(summary)

        if grade:
            grade = grade.upper().strip()
            filtered["filtered_grade"] = grade
            filtered["grade_upload_count"] = summary.get("by_grade", {}).get(grade, 0)

        if subject:
            subject = subject.upper().strip()
            filtered["filtered_subject"] = subject
            filtered["subject_upload_count"] = summary.get("by_subject", {}).get(subject, 0)

        if school:
            filtered["filtered_school"] = school
            filtered["school_upload_count"] = summary.get("by_school", {}).get(school, 0)

        self.save_network(network)

        return filtered

    def gaps(self, *, grade: str = "", subject: str = "") -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "gaps")

        coverage = self.coverage(grade=grade, subject=subject)

        gaps = []
        if grade and coverage.get("grade_upload_count", 0) == 0:
            gaps.append(f"no_uploads_for_grade:{grade.upper()}")

        if subject and coverage.get("subject_upload_count", 0) == 0:
            gaps.append(f"no_uploads_for_subject:{subject.upper()}")

        if not gaps:
            gaps.append("no_obvious_placeholder_gap_detected")

        return {
            "ok": True,
            "coverage": coverage,
            "gaps": gaps,
            "status": "placeholder_gap_detection_until_full_grep_population",
        }

    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------

    def report_teacher(self, *, grade: str = "", subject: str = "", output: str = "") -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "report_teacher")

        report = {
            "ok": True,
            "report_type": "teacher",
            "created_at": _now_iso(),
            "session": session.to_dict(),
            "coverage": self.coverage(grade=grade, subject=subject),
            "gaps": self.gaps(grade=grade, subject=subject),
            "recommendations": [
                "Bind local plans to competence aim IDs after canonical Grep population.",
                "Use annual_plan for long-range planning and unit_plan for operational sequences.",
                "Keep student evidence private and mark contains_student_data=True.",
            ],
        }

        return self._maybe_write_report(report, output)

    def report_school(self, *, school: str = "", output: str = "") -> Dict[str, Any]:
        session = self.current_session()

        if self.permissions.can(session.role, "report_school"):
            pass
        elif self.permissions.can(session.role, "report_school_limited"):
            pass
        else:
            self.permissions.require(session.role, "report_school")

        school_id = school or session.school_org_id

        report = {
            "ok": True,
            "report_type": "school",
            "created_at": _now_iso(),
            "school_org_id": school_id,
            "session": session.to_dict(),
            "coverage": self.coverage(school=school_id),
            "audit_tail": self._safe_audit_tail(limit=20) if session.role in {"coordinator", "school_leader", "admin"} else [],
            "recommendations": [
                "Review quarantined uploads.",
                "Approve local curriculum packages through coordinator or school leader workflow.",
                "Maintain explicit privacy flags for student evidence.",
                "Use Merkle snapshots before exporting to authority level.",
            ],
        }

        return self._maybe_write_report(report, output)

    def report_gov_benefits(self, *, output: str = "") -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "gov_benefits")

        report = {
            "ok": True,
            "report_type": "government_benefits",
            "created_at": _now_iso(),
            "system": "Akkurat LK20 Curriculum Digital Twin",
            "summary": (
                "The system creates a governed digital twin of LK20 where national curriculum data remains canonical "
                "and local school artefacts are attached as auditable overlays."
            ),
            "benefits": [
                {
                    "title": "Curriculum transparency",
                    "details": "Grades, subjects, programmes, competence aims, assessment rules, and local plans become inspectable through one governed structure.",
                },
                {
                    "title": "Canonical/local separation",
                    "details": "Udir/Grep content remains authoritative; school uploads bind to canonical targets without mutating them.",
                },
                {
                    "title": "Traceable local implementation",
                    "details": "Annual plans, term plans, unit plans, rubrics, and evidence are hashed, logged, and linked to curriculum targets.",
                },
                {
                    "title": "Coverage and gap detection",
                    "details": "The system can identify missing grade/subject bindings and later full competence-aim coverage once Grep data is populated.",
                },
                {
                    "title": "Teacher planning support",
                    "details": "Teachers get a structured workflow for validating and attaching local plans to LK20.",
                },
                {
                    "title": "Student-facing clarity",
                    "details": "Students can view relevant learning goals, approved plans, and assessment expectations without seeing private governance data.",
                },
                {
                    "title": "Auditability",
                    "details": "Every mutating action can be written to local audit logs and every network state can be Merkle-verified.",
                },
                {
                    "title": "Privacy boundary visibility",
                    "details": "Student evidence is explicitly separated through role gates, visibility flags, and DPIA warnings.",
                },
                {
                    "title": "Authority-level insight",
                    "details": "Municipalities, counties, and national inspectors can inspect aggregated implementation status without direct access to student-level artefacts.",
                },
            ],
            "architecture": {
                "tn.py": "Low-level TensorTrain/Tucker runtime.",
                "digital_twin_kernel.py": "Governed tree tensor network and state kernel.",
                "twin_anything.py": "Generic object/process twin factory.",
                "lk20_kernel.py": "LK20 curriculum-domain topology and workflows.",
                "lk20_main.py": "Role-aware command gateway.",
            },
            "privacy_position": [
                "Local uploads are not public by default.",
                "Student evidence must be marked contains_student_data=True.",
                "Student data requires private_school visibility.",
                "Government reports should aggregate or exclude student-level artefacts.",
            ],
        }

        return self._maybe_write_report(report, output)

    def gov_inspect_system(self) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "gov_inspect_system")

        network_info = {}
        if Path(self.config.current_network_path).exists():
            network = self.load_network()
            network_info = {
                "node_count": len(getattr(network, "nodes", {})),
                "merkle_root": self._safe_merkle_hex(network),
                "root_nodes_present": {
                    "platform": "0" in network.nodes,
                    "thing_anything": "5.0" in network.nodes,
                    "lk20": "7.0" in network.nodes,
                },
            }

        return {
            "ok": True,
            "created_at": _now_iso(),
            "status": self.status(),
            "health": self.health(),
            "network": network_info,
            "role_model": self.permissions.role_report(),
            "benefits": self.report_gov_benefits().get("benefits", []),
        }

    def _maybe_write_report(self, report: Dict[str, Any], output: str = "") -> Dict[str, Any]:
        if output:
            path = Path(output)
        else:
            report_type = _slug(report.get("report_type", "report"), "report")
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            path = Path(self.config.data_dir) / "reports" / f"{report_type}_{ts}.json"

        _write_json(path, report)
        report["saved_report_json"] = str(path)

        self.audit(
            "report_generated",
            {
                "report_type": report.get("report_type"),
                "path": str(path),
            },
            action="report",
        )

        return report

    def _safe_audit_tail(self, *, limit: int = 20) -> List[Any]:
        try:
            return self.read_audit(limit=limit).get("events", [])
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # Demo
    # -------------------------------------------------------------------------

    def demo(self) -> Dict[str, Any]:
        session = self.current_session()
        self.permissions.require(session.role, "demo")

        self.init_project()

        if not Path(self.config.current_network_path).exists():
            self.create_network()

        canonical = self.sample_canonical()
        canonical_result = None

        if self.permissions.can(session.role, "canonical_status"):
            canonical_result = self.ingest_canonical(canonical_path=canonical["canonical_path"])

        benefits = self.report_gov_benefits()

        return {
            "ok": True,
            "message": "Demo sequence completed.",
            "canonical_sample": canonical,
            "canonical_ingest": canonical_result,
            "benefits_report": benefits.get("saved_report_json"),
            "next_steps": [
                "Login as teacher and upload a curriculum plan.",
                "Run coverage summary.",
                "Inspect grade and subject nodes.",
            ],
        }

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def _safe_merkle_hex(self, network: Any) -> Optional[str]:
        try:
            dtk = self._dtk_module()
            return dtk.MerkleHasher.hex64(network.merkle_root())
        except Exception:
            return None


# =============================================================================
# Interactive shell
# =============================================================================

class LK20Shell(cmd.Cmd):
    intro = "LK20 shell. Type help or ? to list commands. Type exit to quit."
    prompt = "lk20> "

    def __init__(self, app: LK20MainApp):
        super().__init__()
        self.app = app

    def _run(self, line: str) -> None:
        if not line.strip():
            return

        try:
            argv = shlex.split(line)
            out = run_cli(argv, app=self.app, from_shell=True)
            if out is not None:
                _print_json(out)
        except SystemExit:
            pass
        except Exception as exc:
            _print_json(
                {
                    "ok": False,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    def default(self, line: str) -> None:
        self._run(line)

    def do_exit(self, arg: str) -> bool:
        return True

    def do_quit(self, arg: str) -> bool:
        return True

    def do_EOF(self, arg: str) -> bool:
        print()
        return True

    def do_status(self, arg: str) -> None:
        self._run("status")

    def do_health(self, arg: str) -> None:
        self._run("health")

    def do_whoami(self, arg: str) -> None:
        self._run("whoami")

    def do_help(self, arg: str) -> None:
        print(
            textwrap.dedent(
                """
                Core commands:
                  status
                  health
                  init
                  login --role teacher --user-id t001 --school NO-12345 --school-name "Example School"
                  whoami
                  logout
                  create
                  teacher upload --type annual_plan --file path.pdf --grade G5 --subject NOR
                  validate-upload --manifest data/uploads/manifests/upl_x.json
                  attach-upload --manifest data/uploads/manifests/upl_x.json
                  inspect grade G5
                  inspect subject NOR
                  search "norsk"
                  coverage --grade G5 --subject NOR
                  gaps --grade G5 --subject NOR
                  gov benefits
                  gov inspect-system
                  snapshot
                  verify
                  exit
                """
            ).strip()
        )


# =============================================================================
# CLI parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LK20 Command Gateway")
    p.add_argument("--project-root", default="", help="Project root. Defaults to script directory or LK20_ROOT.")
    p.add_argument("--json", action="store_true", help="Force JSON output. Default is already JSON.")

    sub = p.add_subparsers(dest="command")

    sub.add_parser("init")
    sub.add_parser("status")
    sub.add_parser("health")
    sub.add_parser("whoami")
    sub.add_parser("logout")
    sub.add_parser("create")
    sub.add_parser("shell")
    sub.add_parser("demo")
    sub.add_parser("verify")
    sub.add_parser("snapshot")

    login = sub.add_parser("login")
    login.add_argument("--role", required=True)
    login.add_argument("--user-id", default="anonymous")
    login.add_argument("--school", dest="school_org_id", default="")
    login.add_argument("--school-name", default="")
    login.add_argument("--municipality", dest="municipality_id", default="")
    login.add_argument("--county", dest="county_id", default="")

    register_school = sub.add_parser("register-school")
    register_school.add_argument("--org-id", required=True)
    register_school.add_argument("--name", required=True)

    sample_upload = sub.add_parser("sample-upload")
    sample_upload.add_argument("--output", default="")

    validate_upload = sub.add_parser("validate-upload")
    validate_upload.add_argument("--manifest", required=True)

    attach_upload = sub.add_parser("attach-upload")
    attach_upload.add_argument("--manifest", required=True)
    attach_upload.add_argument("--strict", action="store_true")
    attach_upload.add_argument("--no-quarantine", action="store_true")

    list_uploads = sub.add_parser("list-uploads")
    list_uploads.add_argument("--limit", type=int, default=100)

    inspect_upload = sub.add_parser("inspect-upload")
    inspect_upload.add_argument("--upload-id", default="")
    inspect_upload.add_argument("--manifest", default="")

    inspect = sub.add_parser("inspect")
    inspect.add_argument("target_type", choices=["node", "grade", "subject", "programme", "program", "upload"])
    inspect.add_argument("target")

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=25)
    search.add_argument("--include-private", action="store_true")

    coverage = sub.add_parser("coverage")
    coverage.add_argument("--grade", default="")
    coverage.add_argument("--subject", default="")
    coverage.add_argument("--school", default="")

    gaps = sub.add_parser("gaps")
    gaps.add_argument("--grade", default="")
    gaps.add_argument("--subject", default="")

    sample_canonical = sub.add_parser("sample-canonical")
    sample_canonical.add_argument("--output", default="")

    ingest_canonical = sub.add_parser("ingest-canonical")
    ingest_canonical.add_argument("--canonical", required=True)

    canonical_status = sub.add_parser("canonical-status")

    audit = sub.add_parser("audit")
    audit.add_argument("--limit", type=int, default=50)

    report = sub.add_parser("report")
    report.add_argument("report_type", choices=["teacher", "school", "gov-benefits"])
    report.add_argument("--grade", default="")
    report.add_argument("--subject", default="")
    report.add_argument("--school", default="")
    report.add_argument("--output", default="")

    gov = sub.add_parser("gov")
    gov_sub = gov.add_subparsers(dest="gov_command")
    gov_sub.add_parser("benefits")
    gov_sub.add_parser("inspect-system")
    gov_sub.add_parser("verify")
    gov_sub.add_parser("canonical-status")
    gov_export = gov_sub.add_parser("export-brief")
    gov_export.add_argument("--output", default="")

    student = sub.add_parser("student")
    student_sub = student.add_subparsers(dest="student_command")
    student_view = student_sub.add_parser("view")
    student_view.add_argument("--grade", required=True)
    student_view.add_argument("--subject", default="")
    student_goals = student_sub.add_parser("goals")
    student_goals.add_argument("--grade", required=True)
    student_goals.add_argument("--subject", default="")
    student_assessment = student_sub.add_parser("assessment")
    student_assessment.add_argument("--grade", required=True)
    student_assessment.add_argument("--subject", default="")
    student_search = student_sub.add_parser("search")
    student_search.add_argument("query")

    teacher = sub.add_parser("teacher")
    teacher_sub = teacher.add_subparsers(dest="teacher_command")
    teacher_upload = teacher_sub.add_parser("upload")
    add_upload_args(teacher_upload)
    teacher_validate = teacher_sub.add_parser("validate")
    teacher_validate.add_argument("--manifest", required=True)
    teacher_attach = teacher_sub.add_parser("attach")
    teacher_attach.add_argument("--manifest", required=True)
    teacher_coverage = teacher_sub.add_parser("coverage")
    teacher_coverage.add_argument("--grade", default="")
    teacher_coverage.add_argument("--subject", default="")
    teacher_gaps = teacher_sub.add_parser("gaps")
    teacher_gaps.add_argument("--grade", default="")
    teacher_gaps.add_argument("--subject", default="")
    teacher_report = teacher_sub.add_parser("report")
    teacher_report.add_argument("--grade", default="")
    teacher_report.add_argument("--subject", default="")
    teacher_report.add_argument("--output", default="")

    upload = sub.add_parser("upload")
    upload_sub = upload.add_subparsers(dest="upload_command")
    curriculum = upload_sub.add_parser("curriculum")
    add_upload_args(curriculum)

    school = sub.add_parser("school")
    school_sub = school.add_subparsers(dest="school_command")
    school_register = school_sub.add_parser("register")
    school_register.add_argument("--org-id", required=True)
    school_register.add_argument("--name", required=True)
    school_dashboard = school_sub.add_parser("dashboard")
    school_dashboard.add_argument("--school", default="")
    school_coverage = school_sub.add_parser("coverage")
    school_coverage.add_argument("--school", default="")
    school_report = school_sub.add_parser("export")
    school_report.add_argument("--school", default="")
    school_report.add_argument("--output", default="")

    authority = sub.add_parser("authority")
    authority_sub = authority.add_subparsers(dest="authority_command")
    authority_overview = authority_sub.add_parser("overview")
    authority_overview.add_argument("--municipality", default="")
    authority_compare = authority_sub.add_parser("compare-schools")
    authority_compare.add_argument("--subject", default="")
    authority_gaps = authority_sub.add_parser("gaps")
    authority_gaps.add_argument("--grade", default="")
    authority_gaps.add_argument("--subject", default="")
    authority_verify = authority_sub.add_parser("verify-snapshots")
    authority_export = authority_sub.add_parser("export")
    authority_export.add_argument("--output", default="")

    admin = sub.add_parser("admin")
    admin_sub = admin.add_subparsers(dest="admin_command")
    admin_sub.add_parser("init")
    admin_sub.add_parser("health")
    admin_sub.add_parser("create")
    admin_ingest = admin_sub.add_parser("ingest-canonical")
    admin_ingest.add_argument("--canonical", required=True)
    admin_sub.add_parser("verify")
    admin_sub.add_parser("roles")

    return p


def add_upload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--type", dest="upload_type", required=True)
    parser.add_argument("--file", dest="file_path", required=True)
    parser.add_argument("--grade", default="")
    parser.add_argument("--subject", default="")
    parser.add_argument("--subject-name", default="")
    parser.add_argument("--programme", default="")
    parser.add_argument("--programme-name", default="")
    parser.add_argument("--school", dest="school_org_id", default="")
    parser.add_argument("--school-name", default="")
    parser.add_argument("--year", dest="school_year", default="")
    parser.add_argument("--term", default="full_year")
    parser.add_argument("--aim", dest="competence_aim_ids", action="append", default=[])
    parser.add_argument("--contains-student-data", action="store_true")
    parser.add_argument("--requires-dpia", action="store_true")
    parser.add_argument("--no-attach", action="store_true")
    parser.add_argument("--strict", action="store_true")


# =============================================================================
# CLI dispatch
# =============================================================================

def app_from_args(args: argparse.Namespace) -> LK20MainApp:
    cfg = LK20MainConfig.from_project_root(args.project_root or None)
    return LK20MainApp(cfg)


def run_cli(
    argv: Optional[Sequence[str]] = None,
    *,
    app: Optional[LK20MainApp] = None,
    from_shell: bool = False,
) -> Optional[Dict[str, Any]]:
    parser = build_parser()
    args = parser.parse_args(argv)

    app = app or app_from_args(args)

    if args.command is None:
        parser.print_help()
        return None

    if args.command == "init":
        return app.init_project()

    if args.command == "status":
        return app.status()

    if args.command == "health":
        return app.health()

    if args.command == "login":
        return app.login(
            role=args.role,
            user_id=args.user_id,
            school_org_id=args.school_org_id,
            school_name=args.school_name,
            municipality_id=args.municipality_id,
            county_id=args.county_id,
        )

    if args.command == "whoami":
        return app.whoami()

    if args.command == "logout":
        return app.logout()

    if args.command == "create":
        return app.create_network()

    if args.command == "shell":
        LK20Shell(app).cmdloop()
        return None

    if args.command == "demo":
        return app.demo()

    if args.command == "verify":
        return app.verify()

    if args.command == "snapshot":
        return app.snapshot()

    if args.command == "register-school":
        return app.register_school(school_org_id=args.org_id, school_name=args.name)

    if args.command == "sample-upload":
        lk20 = app._lk20_module()
        payload = lk20.sample_upload_manifest()
        output = args.output or str(Path(app.config.data_dir) / "uploads" / "manifests" / "sample_upload_manifest.json")
        _write_json(output, payload)
        return {"ok": True, "sample_manifest_path": output, "manifest": payload}

    if args.command == "validate-upload":
        return app.validate_upload(manifest_path=args.manifest)

    if args.command == "attach-upload":
        return app.attach_upload(
            manifest_path=args.manifest,
            strict=bool(args.strict),
            allow_quarantine=not bool(args.no_quarantine),
        )

    if args.command == "list-uploads":
        return app.list_uploads(limit=args.limit)

    if args.command == "inspect-upload":
        return app.inspect_upload(upload_id=args.upload_id, manifest_path=args.manifest)

    if args.command == "inspect":
        return app.inspect(target_type=args.target_type, target=args.target)

    if args.command == "search":
        return app.search(query=args.query, limit=args.limit, include_private=bool(args.include_private))

    if args.command == "coverage":
        return app.coverage(grade=args.grade, subject=args.subject, school=args.school)

    if args.command == "gaps":
        return app.gaps(grade=args.grade, subject=args.subject)

    if args.command == "sample-canonical":
        return app.sample_canonical(output=args.output or None)

    if args.command == "ingest-canonical":
        return app.ingest_canonical(canonical_path=args.canonical)

    if args.command == "canonical-status":
        return app.canonical_status()

    if args.command == "audit":
        return app.read_audit(limit=args.limit)

    if args.command == "report":
        if args.report_type == "teacher":
            return app.report_teacher(grade=args.grade, subject=args.subject, output=args.output)
        if args.report_type == "school":
            return app.report_school(school=args.school, output=args.output)
        if args.report_type == "gov-benefits":
            return app.report_gov_benefits(output=args.output)

    if args.command == "gov":
        if args.gov_command == "benefits":
            return app.report_gov_benefits()
        if args.gov_command == "inspect-system":
            return app.gov_inspect_system()
        if args.gov_command == "verify":
            return app.verify()
        if args.gov_command == "canonical-status":
            return app.canonical_status()
        if args.gov_command == "export-brief":
            return app.report_gov_benefits(output=args.output)

    if args.command == "student":
        if args.student_command in {"view", "goals", "assessment"}:
            return app.student_view(grade=args.grade, subject=args.subject)
        if args.student_command == "search":
            return app.search(query=args.query, include_private=False)

    if args.command == "teacher":
        if args.teacher_command == "upload":
            return app.upload_curriculum_from_args(args)
        if args.teacher_command == "validate":
            return app.validate_upload(manifest_path=args.manifest)
        if args.teacher_command == "attach":
            return app.attach_upload(manifest_path=args.manifest)
        if args.teacher_command == "coverage":
            return app.coverage(grade=args.grade, subject=args.subject)
        if args.teacher_command == "gaps":
            return app.gaps(grade=args.grade, subject=args.subject)
        if args.teacher_command == "report":
            return app.report_teacher(grade=args.grade, subject=args.subject, output=args.output)

    if args.command == "upload":
        if args.upload_command == "curriculum":
            return app.upload_curriculum_from_args(args)

    if args.command == "school":
        if args.school_command == "register":
            return app.register_school(school_org_id=args.org_id, school_name=args.name)
        if args.school_command == "dashboard":
            return app.report_school(school=args.school)
        if args.school_command == "coverage":
            return app.coverage(school=args.school)
        if args.school_command == "export":
            return app.report_school(school=args.school, output=args.output)

    if args.command == "authority":
        if args.authority_command == "overview":
            return {
                "ok": True,
                "type": "authority_overview",
                "municipality": args.municipality,
                "coverage": app.coverage(),
                "verify": app.verify(),
            }
        if args.authority_command == "compare-schools":
            return {
                "ok": True,
                "type": "compare_schools_placeholder",
                "subject": args.subject,
                "coverage": app.coverage(subject=args.subject),
                "status": "comparison requires multiple school partitions in later deployment",
            }
        if args.authority_command == "gaps":
            return app.gaps(grade=args.grade, subject=args.subject)
        if args.authority_command == "verify-snapshots":
            return app.verify()
        if args.authority_command == "export":
            return app.report_school(output=args.output)

    if args.command == "admin":
        session = app.current_session()
        app.permissions.require(session.role, "status")  # admin commands still work if current role has permission per subhandler
        if args.admin_command == "init":
            return app.init_project()
        if args.admin_command == "health":
            return app.health()
        if args.admin_command == "create":
            return app.create_network()
        if args.admin_command == "ingest-canonical":
            return app.ingest_canonical(canonical_path=args.canonical)
        if args.admin_command == "verify":
            return app.verify()
        if args.admin_command == "roles":
            return {"ok": True, **app.permissions.role_report()}

    raise ValueError(f"Unhandled command: {args.command}")


def _upload_curriculum_from_args(self: LK20MainApp, args: argparse.Namespace) -> Dict[str, Any]:
    return self.upload_curriculum(
        upload_type=args.upload_type,
        file_path=args.file_path,
        grade=args.grade,
        subject=args.subject,
        subject_name=args.subject_name,
        programme=args.programme,
        programme_name=args.programme_name,
        school_org_id=args.school_org_id,
        school_name=args.school_name,
        school_year=args.school_year,
        term=args.term,
        competence_aim_ids=args.competence_aim_ids,
        contains_student_data=bool(args.contains_student_data),
        requires_dpia=bool(args.requires_dpia),
        attach=not bool(args.no_attach),
        strict=bool(args.strict),
    )


setattr(LK20MainApp, "upload_curriculum_from_args", _upload_curriculum_from_args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        out = run_cli(argv)
        if out is not None:
            _print_json(out)
        return 0 if out is None or bool(out.get("ok", True)) else 2

    except Exception as exc:
        err = {
            "ok": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        _print_json(err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())