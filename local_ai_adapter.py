#!/usr/bin/env python3
"""
local_ai_adapter.py
===================
Import-safe orchestration layer for LK20 local AI.

This module is intentionally advisory/status-only. It bridges the LK20 server or
kernel code to local_ai modules and local lexical resources without mutating
canonical LK20 state.

Primary responsibilities
------------------------
- Load the SemanticAttractorBank when available.
- Expose entropy_nlp diagnostics and reranking when available.
- Expose the Kaikki tensor lexicon hybrid SQLite bank for direct, alias, and
  relation lookup.
- Report tensor-core and manifest status.
- Preserve legacy compatibility methods used elsewhere in LK20.

Important design note
---------------------
The adapter must not generate arbitrary prose from the legacy semantic-bank
sentence builder by default. Advisory output is now lexicon/entropy based.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some deployments
    np = None  # type: ignore


# -----------------------------------------------------------------------------
# Paths and import isolation
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_AI_DIR = PROJECT_ROOT / "local_ai"

DEFAULT_LEXICON_DB = LOCAL_AI_DIR / "resources" / "kaikki_tensor" / "lexicon_kaikki_hybrid_v3.db"
DEFAULT_LEXICON_CORE = LOCAL_AI_DIR / "resources" / "kaikki_tensor" / "lexicon_kaikki_hybrid_v3.core.npz"
DEFAULT_LEXICON_MANIFEST = LOCAL_AI_DIR / "resources" / "kaikki_tensor" / "lexicon_kaikki_hybrid_v3.manifest.json"

DEFAULT_SEMANTIC_BANK_CANDIDATES = (
    LOCAL_AI_DIR / "semantic_bank.npz",
    PROJECT_ROOT / "semantic_bank.npz",
    LOCAL_AI_DIR / "resources" / "semantic_bank.npz",
)

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Ranking configuration
# -----------------------------------------------------------------------------

# These are intentionally conservative. They are not a full morphological model;
# they are a deterministic IR scoring layer that makes raw Wiktextract/Kaikki
# entries behave more naturally for common LK20 lookup use.
DEFAULT_POS_RANK: Dict[str, int] = {
    "noun": 0,
    "verb": 0,
    "adjective": 1,
    "adverb": 2,
    "conjunction": 3,
    "determiner": 4,
    "preposition": 4,
    "pronoun": 4,
    "interjection": 5,
}

# Explicit overrides for high-frequency terms where a generic noun/verb-first
# rule is insufficient. The list is intentionally small and explainable.
LEXICAL_POS_PREFERENCES: Dict[str, str] = {
    "cat": "noun",
    "dog": "noun",
    "animal": "noun",
    "person": "noun",
    "student": "noun",
    "teacher": "noun",
    "school": "noun",
    "run": "verb",
    "runs": "verb",
    "running": "verb",
    "ran": "verb",
    "think": "verb",
    "thinks": "verb",
    "thinking": "verb",
    "thought": "verb",
    "learn": "verb",
    "learning": "verb",
    "teach": "verb",
    "read": "verb",
    "write": "verb",
    "because": "conjunction",
    "although": "conjunction",
    "though": "conjunction",
    "unless": "conjunction",
    "until": "conjunction",
    "while": "conjunction",
    "whereas": "conjunction",
    "and": "conjunction",
    "or": "conjunction",
    "but": "conjunction",
    "if": "conjunction",
    "the": "determiner",
    "a": "determiner",
    "an": "determiner",
    "this": "determiner",
    "that": "determiner",
    "these": "determiner",
    "those": "determiner",
    "he": "pronoun",
    "she": "pronoun",
    "it": "pronoun",
    "we": "pronoun",
    "they": "pronoun",
    "beautiful": "adjective",
}

ADJECTIVE_SUFFIXES = (
    "able",
    "ible",
    "al",
    "ial",
    "ant",
    "ary",
    "ful",
    "ic",
    "ical",
    "ive",
    "less",
    "ous",
)

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "as",
    "by",
    "at",
    "from",
    "into",
    "about",
    "than",
    "then",
    "so",
    "very",
    "can",
    "could",
    "should",
    "would",
    "will",
    "just",
    "please",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def ensure_paths(project_root: Optional[str | Path] = None) -> None:
    """Ensure LK20 root and local_ai are importable without package installation."""
    root = Path(project_root or PROJECT_ROOT).resolve()
    local_ai = root / "local_ai"

    local_ai_str = str(local_ai)
    root_str = str(root)

    if local_ai_str not in sys.path:
        sys.path.insert(0, local_ai_str)
    if root_str not in sys.path:
        sys.path.append(root_str)


def _path_info(path: Path) -> Dict[str, Any]:
    """Return stable file status metadata for a path."""
    try:
        exists = path.exists()
        return {
            "path": str(path),
            "exists": bool(exists),
            "size": int(path.stat().st_size) if exists else None,
        }
    except Exception as exc:
        return {"path": str(path), "exists": False, "size": None, "error": str(exc)}


def _safe_json_loads(value: Any, default: Any = None) -> Any:
    """Parse JSON strings defensively."""
    if value is None:
        return default
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if not isinstance(value, str):
        return default
    text = value.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _read_json_file(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk. Returns {} on any failure."""
    try:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception as exc:
        LOGGER.warning("Could not read JSON file %s: %s", path, exc)
        return {}


def _limit_value(limit: Any, default: int = 10, maximum: int = 200) -> int:
    """Normalize user/API supplied limits."""
    try:
        n = int(limit)
    except Exception:
        n = default
    if n < 1:
        return 1
    if n > maximum:
        return maximum
    return n


def _candidate_limit(limit: int) -> int:
    """Fetch more than the requested result count so ranking can work correctly."""
    return max(50, min(1000, int(limit) * 50))


def _normalize_token(text: Any) -> str:
    """Normalize user lookup tokens for lexicon/alias queries."""
    if text is None:
        return ""
    return str(text).strip().lower()


def _raw_token(text: Any) -> str:
    """Preserve user casing while trimming whitespace."""
    if text is None:
        return ""
    return str(text).strip()


def _coerce_candidate_text(candidate: Any) -> str:
    """Extract text from strings, dictionaries, or candidate-like objects."""
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, dict):
        for key in ("text", "sentence", "content", "output"):
            if key in candidate and candidate[key] is not None:
                return str(candidate[key])
        return json.dumps(candidate, ensure_ascii=False)
    for attr in ("text", "sentence", "content", "output"):
        if hasattr(candidate, attr):
            value = getattr(candidate, attr)
            if value is not None:
                return str(value)
    return str(candidate)


def _is_all_caps_acronym(text: Any) -> bool:
    """Return True for uppercase acronym-like strings such as CAT or USA."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    return stripped.upper() == stripped and stripped.lower() != stripped




def _morphology_penalty(gloss: Any) -> int:
    """Penalize pure inflectional/form entries before lexical matching."""
    g = str(gloss or "").strip().lower()
    morphology_starts = (
        "plural of ",
        "inflection of ",
        "form of ",
        "present participle",
        "past participle",
        "simple past",
        "third-person singular",
        "comparative form of ",
        "superlative form of ",
    )
    return 1 if g.startswith(morphology_starts) else 0

def _gloss_penalty(gloss: Any) -> int:
    """Penalize derivative, restricted-register, and non-curriculum senses.

    The score is intentionally coarse and deterministic. Lower is better.
    Raw Wiktextract/Kaikki entries frequently reset sense_index across
    etymology sections, so this penalty is needed before final tie-breaking.
    """
    g = str(gloss or "").strip().lower()
    if not g:
        return 20

    penalty = 0

    # Pure morphology / redirects should usually lose to base semantic senses,
    # especially for aliases such as running -> run, cats -> cat, thought -> think.
    morphology_starts = (
        "plural of ",
        "inflection of ",
        "form of ",
        "alternative form of ",
        "alt form of ",
        "misspelling of ",
        "nonstandard spelling of ",
        "obsolete spelling of ",
        "archaic spelling of ",
        "present participle",
        "past participle",
        "simple past",
        "third-person singular",
        "comparative form of ",
        "superlative form of ",
    )
    if g.startswith(morphology_starts):
        penalty += 60

    # Abbreviations/acronyms are valid entries but should not beat normal
    # lexical senses for lowercase educational queries.
    abbreviation_starts = (
        "acronym of ",
        "initialism of ",
        "abbreviation of ",
        "short for ",
        "ellipsis of ",
    )
    if g.startswith(abbreviation_starts):
        penalty += 35
    if any(x in g for x in (" acronym of ", " initialism of ", " abbreviation of ")):
        penalty += 20

    # Register/domain penalties. Context overlap can still rescue these when
    # the user explicitly asks about that domain, e.g. "cat unix command".
    severe = (
        "street name",
        "drug",
        "vagina",
        "vulva",
        "prostitute",
        "slur",
        "vulgar",
        "offensive",
        "derogatory",
    )
    if any(x in g for x in severe):
        penalty += 25

    restricted = (
        "obsolete",
        "archaic",
        "dated",
        "rare",
        "dialectal",
        "nonstandard",
        "slang",
        "chiefly",
        "now only",
        "historical",
    )
    if any(x in g for x in restricted):
        penalty += 8

    # Raw dictionary redirects are usually less useful as first results.
    if g.startswith("synonym of "):
        penalty += 12

    # Computer command senses are legitimate but should not beat ordinary
    # curriculum senses unless the context explicitly mentions computing.
    if any(x in g for x in ("unix", "command", "standard output", "program and command")):
        penalty += 10

    return penalty


def _rank_tokens(text: Any) -> set[str]:
    """Tokenize text for context/gloss overlap scoring."""
    toks = re.findall(r"[A-Za-z][A-Za-z'\-]*", str(text or "").lower())
    out: set[str] = set()
    for tok in toks:
        clean = tok.strip("-'")
        if not clean or clean in STOPWORDS or len(clean) < 2:
            continue
        out.add(clean)
        # Cheap singular normalization helps animal/animals and cat/cats.
        if clean.endswith("ies") and len(clean) > 4:
            out.add(clean[:-3] + "y")
        elif clean.endswith("s") and len(clean) > 3:
            out.add(clean[:-1])
    return out


def _context_overlap_bonus(item: Dict[str, Any], context: Optional[str], *, query_norm: str = "") -> float:
    """Return a positive context compatibility score. Higher is better."""
    if not context:
        return 0.0
    ctx = _rank_tokens(context)
    if query_norm:
        ctx.add(query_norm)
    if not ctx:
        return 0.0

    candidate_text = " ".join(
        str(item.get(k) or "")
        for k in ("word", "lemma", "alias_key", "pos", "gloss")
    )
    cand = _rank_tokens(candidate_text)
    if not cand:
        return 0.0

    overlap = ctx.intersection(cand)
    score = float(len(overlap))

    gloss_l = str(item.get("gloss") or "").lower()
    pos_l = str(item.get("pos") or "").lower()

    # Topic marker emitted into glosses by the ingestor; very useful for LK20.
    if "animal" in ctx and ("terms relating to animals" in gloss_l or "animal" in gloss_l or "mammal" in gloss_l or "felidae" in gloss_l or "feline" in gloss_l):
        score += 6.0
    if {"pet", "domestic", "domesticated", "feline", "mammal"}.intersection(ctx) and any(x in gloss_l for x in ("domestic", "domesticated", "feline", "felidae", "mammal", "pet")):
        score += 5.0
    if {"run", "running", "move", "swiftly", "quickly", "feet", "race"}.intersection(ctx) and any(x in gloss_l for x in ("move swiftly", "quickly", "feet", "running", "race")):
        score += 5.0
    if {"think", "thought", "mind", "ponder", "consider", "believe", "mental", "mentally"}.intersection(ctx) and any(x in gloss_l for x in ("mind", "ponder", "mentally", "consider", "believe", "opinion")):
        score += 5.0
    if {"because", "reason", "cause"}.intersection(ctx) and pos_l == "conjunction" and any(x in gloss_l for x in ("reason", "cause", "account")):
        score += 4.0

    # If context clearly names a restricted domain, allow that domain to compete.
    if {"unix", "linux", "command", "terminal", "shell", "file", "files"}.intersection(ctx) and any(x in gloss_l for x in ("unix", "command", "standard output", "file", "files")):
        score += 8.0
    if {"drug", "methcathinone", "slang"}.intersection(ctx) and any(x in gloss_l for x in ("drug", "methcathinone", "street name", "slang")):
        score += 8.0

    return score

def _preferred_pos_for_query(query_norm: str, explicit_pos: Optional[str] = None) -> Optional[str]:
    """
    Infer a natural preferred POS for common lookup terms.

    This is deliberately a heuristic, not a linguistic authority. Explicit
    caller-supplied POS always wins.
    """
    if explicit_pos:
        pos = _normalize_token(explicit_pos)
        return pos or None

    if not query_norm:
        return None

    if query_norm in LEXICAL_POS_PREFERENCES:
        return LEXICAL_POS_PREFERENCES[query_norm]

    # Light suffix hints for cases where the user queries an inflected or
    # derived form and no explicit override exists.
    if query_norm.endswith("ly") and len(query_norm) > 4:
        return "adverb"
    if query_norm.endswith(("ing", "ed")) and len(query_norm) > 4:
        return "verb"
    if query_norm.endswith(ADJECTIVE_SUFFIXES) and len(query_norm) > 5:
        return "adjective"

    return None


def _pos_rank(pos: Any, preferred_pos: Optional[str]) -> int:
    """Calculate deterministic POS rank."""
    pos_norm = _normalize_token(pos)
    if preferred_pos:
        if pos_norm == preferred_pos:
            return 0
        return 1 + DEFAULT_POS_RANK.get(pos_norm, 9)
    return DEFAULT_POS_RANK.get(pos_norm, 9)


def _safe_int(value: Any, default: int = 999_999) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _tokenize_for_advisory(text: str, *, max_tokens: int = 8) -> List[str]:
    """Extract stable lexical tokens for advisory lookup."""
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", str(text or ""))
    normalized: List[str] = []
    seen: set[str] = set()

    # First pass: prefer content words.
    for tok in raw_tokens:
        clean = tok.strip("-'").lower()
        if not clean or clean in STOPWORDS or len(clean) < 2:
            continue
        if clean not in seen:
            seen.add(clean)
            normalized.append(clean)
        if len(normalized) >= max_tokens:
            return normalized

    # Second pass: if the text only contains function words, keep them. This
    # allows examples like "because" to still resolve to conjunction.
    for tok in raw_tokens:
        clean = tok.strip("-'").lower()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
        if len(normalized) >= max_tokens:
            break

    return normalized


def _compact_entropy_summary(diag: Dict[str, Any]) -> str:
    """Convert entropy_nlp diagnostics into a short safe string."""
    if not diag:
        return ""

    if diag.get("ok") is False:
        err = diag.get("error") or diag.get("message")
        return f"Entropy diagnostic unavailable: {err}" if err else "Entropy diagnostic unavailable."

    parts: List[str] = []

    for key in ("profile", "target_profile", "status"):
        val = diag.get(key)
        if val:
            parts.append(f"{key}={val}")

    # Different versions of entropy_nlp may use different field names.
    for key in (
        "entropy_nats",
        "estimated_entropy_nats",
        "tree_entropy_nats",
        "entropy",
        "mean_branching_factor",
        "branching_factor",
    ):
        val = diag.get(key)
        if isinstance(val, (int, float)):
            parts.append(f"{key}={val:.3g}")

    warnings = diag.get("warnings") or diag.get("warning") or diag.get("messages")
    if isinstance(warnings, str) and warnings.strip():
        parts.append(warnings.strip())
    elif isinstance(warnings, list) and warnings:
        safe_warnings = [str(w).strip() for w in warnings if str(w).strip()]
        if safe_warnings:
            parts.append("; ".join(safe_warnings[:3]))

    return "; ".join(parts)


# -----------------------------------------------------------------------------
# Tensor lexicon bridge
# -----------------------------------------------------------------------------

class TensorLexiconHybridBank:
    """
    Read-only bridge to the LK20 TensorLexiconHybridBank SQLite database.

    Expected schema version 3 tables:
    - entries(key, shard_name, source_line, sense_index, word, lemma, alias_key,
      pos, gloss, weight, source, lang_code, sense_id, metadata_json, vector,
      vector_dim, vector_dtype, created_at)
    - aliases(alias, key, alias_type)
    - relations(source_key, relation_type, target_text)
    - processed_shards(...)
    - shard_pos_sums(...)
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        core_path: Optional[str | Path] = None,
        manifest_path: Optional[str | Path] = None,
        *,
        connect: bool = False,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self.db_path = Path(db_path or DEFAULT_LEXICON_DB)
        self.core_path = Path(core_path or self.db_path.with_suffix(".core.npz"))
        self.manifest_path = Path(manifest_path or self.db_path.with_suffix(".manifest.json"))
        self.busy_timeout_ms = int(busy_timeout_ms)

        self._con: Optional[sqlite3.Connection] = None
        self._manifest: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()

        if connect:
            self.connect()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self.db_path.exists()

    @property
    def manifest(self) -> Dict[str, Any]:
        if self._manifest is None:
            self._manifest = _read_json_file(self.manifest_path)
        return self._manifest

    def connect(self) -> sqlite3.Connection:
        """Open a read-only SQLite connection when possible."""
        with self._lock:
            if self._con is not None:
                return self._con

            if not self.db_path.exists():
                raise FileNotFoundError(f"Tensor lexicon DB not found: {self.db_path}")

            # Use URI read-only mode to protect canonical state. Fall back to a
            # normal connection if the local sqlite build rejects URI mode.
            try:
                uri = self.db_path.resolve().as_uri() + "?mode=ro"
                con = sqlite3.connect(uri, uri=True, check_same_thread=False)
            except Exception:
                con = sqlite3.connect(str(self.db_path), check_same_thread=False)

            con.row_factory = sqlite3.Row
            try:
                con.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
                con.execute("PRAGMA query_only=ON")
                con.execute("PRAGMA temp_store=MEMORY")
            except Exception:
                # Some PRAGMA statements are advisory only. Continue if rejected.
                pass

            self._con = con
            return con

    def close(self) -> None:
        with self._lock:
            if self._con is not None:
                try:
                    self._con.close()
                finally:
                    self._con = None

    def __enter__(self) -> "TensorLexiconHybridBank":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Status and schema
    # ------------------------------------------------------------------

    def status(self, *, include_integrity_check: bool = False) -> Dict[str, Any]:
        """Return file, manifest, and optional DB status."""
        manifest = self.manifest

        out: Dict[str, Any] = {
            "available": self.db_path.exists(),
            "db_path": str(self.db_path),
            "db_size": self.db_path.stat().st_size if self.db_path.exists() else None,
            "core_path": str(self.core_path),
            "core_size": self.core_path.stat().st_size if self.core_path.exists() else None,
            "manifest_path": str(self.manifest_path),
            "manifest_size": self.manifest_path.stat().st_size if self.manifest_path.exists() else None,
            "format": manifest.get("format"),
            "schema_version": manifest.get("schema_version"),
            "entry_count": manifest.get("entry_count"),
            "alias_count": manifest.get("alias_count"),
            "relation_count": manifest.get("relation_count"),
            "complete_shards": manifest.get("complete_shards"),
            "vectors_mode": manifest.get("vectors_mode"),
            "pos_counts": manifest.get("pos_counts"),
        }

        tt = manifest.get("tensor_train") or {}
        tt_meta = tt.get("metadata") or {}
        out["tensor_train"] = {
            "param_count": tt.get("param_count"),
            "core_shapes": tt.get("core_shapes"),
            "input_dim": tt.get("input_dim"),
            "output_dim": tt.get("output_dim"),
            "pos_list": tt.get("pos_list") or tt_meta.get("pos_list") or manifest.get("pos_list"),
        }

        if include_integrity_check and self.db_path.exists():
            out["integrity_check"] = self.integrity_check()

        return out

    def schema(self) -> Dict[str, List[Tuple[Any, ...]]]:
        """Return PRAGMA table_info rows for key lexicon tables."""
        con = self.connect()
        tables = ["entries", "aliases", "relations", "processed_shards", "shard_pos_sums"]
        result: Dict[str, List[Tuple[Any, ...]]] = {}
        with self._lock:
            for table in tables:
                try:
                    result[table] = [tuple(row) for row in con.execute(f"PRAGMA table_info({table})")]
                except Exception as exc:
                    result[table] = [("error", str(exc))]
        return result

    def integrity_check(self) -> str:
        """Run SQLite PRAGMA integrity_check. This can be slow on a large DB."""
        con = self.connect()
        with self._lock:
            try:
                row = con.execute("PRAGMA integrity_check").fetchone()
                return str(row[0]) if row else "unknown"
            except Exception as exc:
                return f"error: {exc}"

    def counts(self) -> Dict[str, Optional[int]]:
        """Return table counts using the live DB."""
        con = self.connect()
        result: Dict[str, Optional[int]] = {}
        with self._lock:
            for table in ["entries", "aliases", "relations", "shard_pos_sums", "processed_shards"]:
                try:
                    result[table] = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                except Exception:
                    result[table] = None
        return result

    def tensor_core_status(self) -> Dict[str, Any]:
        """Return tensor-core file and npz metadata without performing inference."""
        info: Dict[str, Any] = {
            "available": self.core_path.exists(),
            "core_path": str(self.core_path),
            "core_size": self.core_path.stat().st_size if self.core_path.exists() else None,
            "numpy_available": np is not None,
        }

        manifest_tt = (self.manifest.get("tensor_train") or {}) if self.manifest else {}
        if manifest_tt:
            info["manifest_tensor_train"] = manifest_tt

        if not self.core_path.exists() or np is None:
            return info

        try:
            npz = np.load(self.core_path, allow_pickle=False)  # type: ignore[union-attr]
            keys = sorted(str(k) for k in npz.files)
            arrays: Dict[str, Dict[str, Any]] = {}
            for key in keys:
                arr = npz[key]
                arrays[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            info["npz_keys"] = keys
            info["arrays"] = arrays
        except Exception as exc:
            info["error"] = str(exc)

        return info

    # ------------------------------------------------------------------
    # Row conversion and ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_from_row(row: sqlite3.Row, *, include_raw_metadata_json: bool = False) -> Dict[str, Any]:
        keys = set(row.keys())

        item: Dict[str, Any] = {
            "key": row["key"] if "key" in keys else None,
            "word": row["word"] if "word" in keys else None,
            "lemma": row["lemma"] if "lemma" in keys else None,
            "alias_key": row["alias_key"] if "alias_key" in keys else None,
            "pos": row["pos"] if "pos" in keys else None,
            "gloss": row["gloss"] if "gloss" in keys else None,
            "weight": row["weight"] if "weight" in keys else None,
            "source": row["source"] if "source" in keys else None,
            "lang_code": row["lang_code"] if "lang_code" in keys else None,
            "sense_id": row["sense_id"] if "sense_id" in keys else None,
            "source_line": row["source_line"] if "source_line" in keys else None,
            "sense_index": row["sense_index"] if "sense_index" in keys else None,
            "metadata": _safe_json_loads(row["metadata_json"], default={}) if "metadata_json" in keys else {},
            "has_vector": bool(row["vector"] is not None) if "vector" in keys else False,
            "vector_dim": row["vector_dim"] if "vector_dim" in keys else None,
            "vector_dtype": row["vector_dtype"] if "vector_dtype" in keys else None,
            "match_type": row["match_type"] if "match_type" in keys else None,
            "matched_alias": row["matched_alias"] if "matched_alias" in keys else None,
            "alias_type": row["alias_type"] if "alias_type" in keys else None,
        }

        if include_raw_metadata_json and "metadata_json" in keys:
            item["metadata_json"] = row["metadata_json"]

        return item

    @staticmethod
    def _ranking_tuple(
        item: Dict[str, Any],
        *,
        query_raw: str,
        query_norm: str,
        explicit_pos: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Tuple[Any, ...]:
        """Python-side final ranking. Mirrors SQL CASE logic and adds safeguards."""
        word = str(item.get("word") or "")
        lemma = str(item.get("lemma") or "")
        alias_key = str(item.get("alias_key") or "")
        matched_alias = str(item.get("matched_alias") or "")
        alias_type = str(item.get("alias_type") or "")
        match_type = str(item.get("match_type") or "")
        pos = str(item.get("pos") or "")
        gloss = str(item.get("gloss") or "")

        word_l = word.lower()
        lemma_l = lemma.lower()
        alias_key_l = alias_key.lower()
        matched_alias_l = matched_alias.lower()
        preferred = _preferred_pos_for_query(query_norm, explicit_pos)

        # POS comes first because common exact spellings such as "cat" and "run"
        # have valid but fringe senses in other POS categories.
        pos_rank = _pos_rank(pos, preferred)

        # Lowercase user queries should not surface uppercase acronyms before
        # regular lexical entries unless the user explicitly typed uppercase.
        query_is_lower = query_raw == query_raw.lower()
        acronym_penalty = 1 if query_is_lower and _is_all_caps_acronym(word) else 0

        # Direct exact form/lemma beats case-insensitive, which beats alias_key,
        # which beats alias-table form resolution.
        if word == query_raw or lemma == query_raw:
            lexical_rank = 0
        elif word_l == query_norm or lemma_l == query_norm:
            lexical_rank = 1
        elif alias_key_l == query_norm:
            lexical_rank = 2
        elif matched_alias_l == query_norm:
            lexical_rank = 3
        else:
            lexical_rank = 4

        if match_type == "direct":
            match_rank = 0
        elif alias_type in ("lemma", "canonical", "headword"):
            match_rank = 1
        elif alias_type in ("form", "inflection", "variant"):
            match_rank = 2
        else:
            match_rank = 3

        q_len = len(query_norm)
        length_candidates = [
            abs(len(word_l) - q_len) if word_l else 99_999,
            abs(len(lemma_l) - q_len) if lemma_l else 99_999,
        ]
        length_delta = min(length_candidates)

        morphology_rank = _morphology_penalty(gloss)
        source_line = _safe_int(item.get("source_line"))
        sense_index = _safe_int(item.get("sense_index"))
        weight_tiebreak = -_safe_float(item.get("weight"), default=0.0)
        gloss_rank = _gloss_penalty(gloss)
        context_rank = -_context_overlap_bonus(item, context, query_norm=query_norm)

        return (
            pos_rank,
            acronym_penalty,
            morphology_rank,
            lexical_rank,
            context_rank,
            gloss_rank,
            match_rank,
            source_line,
            sense_index,
            length_delta,
            weight_tiebreak,
            word_l,
            lemma_l,
            str(item.get("key") or ""),
        )

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def lookup(
        self,
        term: str,
        *,
        limit: int = 10,
        pos: Optional[str] = None,
        include_aliases: bool = True,
        include_relations: bool = False,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Look up a term in entries, then aliases.

        Direct matching uses entries.word, entries.lemma, and entries.alias_key.
        Alias matching uses aliases.alias joined back to entries.key.

        Ranking is intentionally not plain weight DESC. It uses deterministic
        linguistic heuristics:
        - case-exact lowercase forms outrank uppercase acronym forms;
        - direct word/lemma matches outrank alias matches;
        - common POS preferences are applied when no POS is specified;
        - acronym/initialism glosses are de-prioritized;
        - Kaikki/Wiktextract weight is only a late tie-breaker.
        """
        query_raw = _raw_token(term)
        query_norm = _normalize_token(term)
        if not query_norm:
            return []

        lim = _limit_value(limit)
        candidate_lim = _candidate_limit(lim)
        pos_norm = _normalize_token(pos) if pos else ""
        preferred_pos = _preferred_pos_for_query(query_norm, pos_norm or None) or ""
        query_is_lower = 1 if query_raw == query_raw.lower() else 0

        con = self.connect()
        candidates: List[Dict[str, Any]] = []

        direct_sql = """
            SELECT
                key, shard_name, source_line, sense_index, word, lemma, alias_key,
                pos, gloss, weight, source, lang_code, sense_id, metadata_json,
                vector, vector_dim, vector_dtype, created_at,
                'direct' AS match_type,
                NULL AS matched_alias,
                NULL AS alias_type
            FROM entries
            WHERE (lower(word) = ? OR lower(lemma) = ? OR alias_key = ?)
        """
        params: List[Any] = [query_norm, query_norm, query_norm]

        if pos_norm:
            direct_sql += " AND lower(pos) = ?"
            params.append(pos_norm)

        direct_sql += """
            ORDER BY
                CASE
                    WHEN ? != '' AND lower(pos) = ? THEN 0
                    WHEN ? != '' THEN 1
                    ELSE 0
                END,
                CASE
                    WHEN word = ? THEN 0
                    WHEN lemma = ? THEN 0
                    WHEN lower(word) = ? THEN 1
                    WHEN lower(lemma) = ? THEN 1
                    WHEN alias_key = ? THEN 2
                    ELSE 3
                END,
                CASE
                    WHEN ? = 1 AND word = upper(word) AND word != lower(word) THEN 1
                    ELSE 0
                END,
                CASE lower(pos)
                    WHEN ? THEN 0
                    WHEN 'noun' THEN 1
                    WHEN 'verb' THEN 1
                    WHEN 'adjective' THEN 2
                    WHEN 'adverb' THEN 3
                    WHEN 'conjunction' THEN 4
                    WHEN 'determiner' THEN 5
                    WHEN 'preposition' THEN 5
                    WHEN 'pronoun' THEN 5
                    WHEN 'interjection' THEN 6
                    ELSE 9
                END,
                CASE
                    WHEN lower(gloss) LIKE 'acronym of%' THEN 3
                    WHEN lower(gloss) LIKE 'initialism of%' THEN 3
                    WHEN lower(gloss) LIKE 'abbreviation of%' THEN 3
                    WHEN lower(gloss) LIKE 'alternative form of%' THEN 2
                    WHEN lower(gloss) LIKE 'alt form of%' THEN 2
                    ELSE 0
                END,
                COALESCE(ABS(length(word) - ?), 99999),
                COALESCE(source_line, 999999),
                COALESCE(sense_index, 999999),
                COALESCE(weight, 0.0) DESC,
                lower(word),
                key
            LIMIT ?
        """
        params.extend(
            [
                preferred_pos,
                preferred_pos,
                preferred_pos,
                query_raw,
                query_raw,
                query_norm,
                query_norm,
                query_norm,
                query_is_lower,
                preferred_pos,
                len(query_norm),
                candidate_lim,
            ]
        )

        with self._lock:
            for row in con.execute(direct_sql, params):
                candidates.append(self._entry_from_row(row))

            if include_aliases:
                alias_sql = """
                    SELECT
                        e.key, e.shard_name, e.source_line, e.sense_index,
                        e.word, e.lemma, e.alias_key, e.pos, e.gloss, e.weight,
                        e.source, e.lang_code, e.sense_id, e.metadata_json,
                        e.vector, e.vector_dim, e.vector_dtype, e.created_at,
                        'alias' AS match_type,
                        a.alias AS matched_alias,
                        a.alias_type AS alias_type
                    FROM aliases a
                    JOIN entries e ON e.key = a.key
                    WHERE a.alias = ?
                """
                alias_params: List[Any] = [query_norm]

                if pos_norm:
                    alias_sql += " AND lower(e.pos) = ?"
                    alias_params.append(pos_norm)

                alias_sql += """
                    ORDER BY
                        CASE
                            WHEN ? != '' AND lower(e.pos) = ? THEN 0
                            WHEN ? != '' THEN 1
                            ELSE 0
                        END,
                        CASE a.alias_type
                            WHEN 'lemma' THEN 0
                            WHEN 'canonical' THEN 0
                            WHEN 'headword' THEN 0
                            WHEN 'form' THEN 1
                            WHEN 'inflection' THEN 1
                            ELSE 2
                        END,
                        CASE
                            WHEN e.word = ? THEN 0
                            WHEN e.lemma = ? THEN 0
                            WHEN lower(e.word) = ? THEN 1
                            WHEN lower(e.lemma) = ? THEN 1
                            WHEN e.alias_key = ? THEN 2
                            ELSE 3
                        END,
                        CASE
                            WHEN ? = 1 AND e.word = upper(e.word) AND e.word != lower(e.word) THEN 1
                            ELSE 0
                        END,
                        CASE lower(e.pos)
                            WHEN ? THEN 0
                            WHEN 'noun' THEN 1
                            WHEN 'verb' THEN 1
                            WHEN 'adjective' THEN 2
                            WHEN 'adverb' THEN 3
                            WHEN 'conjunction' THEN 4
                            WHEN 'determiner' THEN 5
                            WHEN 'preposition' THEN 5
                            WHEN 'pronoun' THEN 5
                            WHEN 'interjection' THEN 6
                            ELSE 9
                        END,
                        CASE
                            WHEN lower(e.gloss) LIKE 'acronym of%' THEN 3
                            WHEN lower(e.gloss) LIKE 'initialism of%' THEN 3
                            WHEN lower(e.gloss) LIKE 'abbreviation of%' THEN 3
                            WHEN lower(e.gloss) LIKE 'alternative form of%' THEN 2
                            WHEN lower(e.gloss) LIKE 'alt form of%' THEN 2
                            ELSE 0
                        END,
                        COALESCE(ABS(length(e.word) - ?), 99999),
                        COALESCE(e.source_line, 999999),
                        COALESCE(e.sense_index, 999999),
                        COALESCE(e.weight, 0.0) DESC,
                        lower(e.word),
                        e.key
                    LIMIT ?
                """
                alias_params.extend(
                    [
                        preferred_pos,
                        preferred_pos,
                        preferred_pos,
                        query_raw,
                        query_raw,
                        query_norm,
                        query_norm,
                        query_norm,
                        query_is_lower,
                        preferred_pos,
                        len(query_norm),
                        candidate_lim,
                    ]
                )

                for row in con.execute(alias_sql, alias_params):
                    candidates.append(self._entry_from_row(row))

        ranked = sorted(
            candidates,
            key=lambda item: self._ranking_tuple(
                item,
                query_raw=query_raw,
                query_norm=query_norm,
                explicit_pos=pos_norm or None,
                context=context,
            ),
        )

        rows: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for item in ranked:
            key = str(item.get("key") or "")
            if key and key in seen_keys:
                continue
            if key:
                seen_keys.add(key)
            rows.append(item)
            if len(rows) >= lim:
                break

        if include_relations:
            for item in rows:
                key = item.get("key")
                if key:
                    item["relations"] = self.relations_for_key(str(key), limit=25)

        return rows[:lim]

    def get_entry(self, key: str, *, include_relations: bool = False) -> Optional[Dict[str, Any]]:
        """Fetch a single entry by primary key."""
        if not key:
            return None
        con = self.connect()
        sql = """
            SELECT
                key, shard_name, source_line, sense_index, word, lemma, alias_key,
                pos, gloss, weight, source, lang_code, sense_id, metadata_json,
                vector, vector_dim, vector_dtype, created_at,
                'key' AS match_type,
                NULL AS matched_alias,
                NULL AS alias_type
            FROM entries
            WHERE key = ?
            LIMIT 1
        """
        with self._lock:
            row = con.execute(sql, (key,)).fetchone()
        if row is None:
            return None
        item = self._entry_from_row(row)
        if include_relations:
            item["relations"] = self.relations_for_key(str(key), limit=100)
        return item

    def aliases_for_key(self, key: str, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return aliases attached to an entry key."""
        if not key:
            return []
        lim = _limit_value(limit)
        con = self.connect()
        sql = """
            SELECT alias, alias_type, key
            FROM aliases
            WHERE key = ?
            ORDER BY alias_type, alias
            LIMIT ?
        """
        with self._lock:
            return [dict(row) for row in con.execute(sql, (key, lim))]

    def resolve_alias(
        self,
        alias: str,
        *,
        limit: int = 10,
        pos: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Resolve an alias/form to entries through the aliases table only."""
        query_raw = _raw_token(alias)
        query_norm = _normalize_token(alias)
        if not query_norm:
            return []

        lim = _limit_value(limit)
        candidate_lim = _candidate_limit(lim)
        pos_norm = _normalize_token(pos) if pos else ""
        preferred_pos = _preferred_pos_for_query(query_norm, pos_norm or None) or ""
        query_is_lower = 1 if query_raw == query_raw.lower() else 0
        con = self.connect()

        sql = """
            SELECT
                e.key, e.shard_name, e.source_line, e.sense_index,
                e.word, e.lemma, e.alias_key, e.pos, e.gloss, e.weight,
                e.source, e.lang_code, e.sense_id, e.metadata_json,
                e.vector, e.vector_dim, e.vector_dtype, e.created_at,
                'alias' AS match_type,
                a.alias AS matched_alias,
                a.alias_type AS alias_type
            FROM aliases a
            JOIN entries e ON e.key = a.key
            WHERE a.alias = ?
        """
        params: List[Any] = [query_norm]
        if pos_norm:
            sql += " AND lower(e.pos) = ?"
            params.append(pos_norm)
        sql += """
            ORDER BY
                CASE
                    WHEN ? != '' AND lower(e.pos) = ? THEN 0
                    WHEN ? != '' THEN 1
                    ELSE 0
                END,
                CASE a.alias_type
                    WHEN 'lemma' THEN 0
                    WHEN 'canonical' THEN 0
                    WHEN 'headword' THEN 0
                    WHEN 'form' THEN 1
                    WHEN 'inflection' THEN 1
                    ELSE 2
                END,
                CASE
                    WHEN e.word = ? THEN 0
                    WHEN e.lemma = ? THEN 0
                    WHEN lower(e.word) = ? THEN 1
                    WHEN lower(e.lemma) = ? THEN 1
                    WHEN e.alias_key = ? THEN 2
                    ELSE 3
                END,
                CASE
                    WHEN ? = 1 AND e.word = upper(e.word) AND e.word != lower(e.word) THEN 1
                    ELSE 0
                END,
                CASE lower(e.pos)
                    WHEN ? THEN 0
                    WHEN 'noun' THEN 1
                    WHEN 'verb' THEN 1
                    WHEN 'adjective' THEN 2
                    WHEN 'adverb' THEN 3
                    WHEN 'conjunction' THEN 4
                    WHEN 'determiner' THEN 5
                    WHEN 'preposition' THEN 5
                    WHEN 'pronoun' THEN 5
                    WHEN 'interjection' THEN 6
                    ELSE 9
                END,
                CASE
                    WHEN lower(e.gloss) LIKE 'acronym of%' THEN 3
                    WHEN lower(e.gloss) LIKE 'initialism of%' THEN 3
                    WHEN lower(e.gloss) LIKE 'abbreviation of%' THEN 3
                    WHEN lower(e.gloss) LIKE 'alternative form of%' THEN 2
                    WHEN lower(e.gloss) LIKE 'alt form of%' THEN 2
                    ELSE 0
                END,
                COALESCE(ABS(length(e.word) - ?), 99999),
                COALESCE(e.sense_index, 999999),
                COALESCE(e.weight, 0.0) DESC,
                lower(e.word),
                e.key
            LIMIT ?
        """
        params.extend(
            [
                preferred_pos,
                preferred_pos,
                preferred_pos,
                query_raw,
                query_raw,
                query_norm,
                query_norm,
                query_norm,
                query_is_lower,
                preferred_pos,
                len(query_norm),
                candidate_lim,
            ]
        )

        with self._lock:
            candidates = [self._entry_from_row(row) for row in con.execute(sql, params)]

        ranked = sorted(
            candidates,
            key=lambda item: self._ranking_tuple(
                item,
                query_raw=query_raw,
                query_norm=query_norm,
                explicit_pos=pos_norm or None,
                context=context,
            ),
        )

        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in ranked:
            key = str(item.get("key") or "")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            out.append(item)
            if len(out) >= lim:
                break

        return out

    # Backward-compatible alias name.
    lookup_alias = resolve_alias

    def relations_for_key(
        self,
        key: str,
        *,
        relation_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return relation rows for one source entry key."""
        if not key:
            return []
        lim = _limit_value(limit, maximum=500)
        con = self.connect()

        sql = """
            SELECT source_key, relation_type, target_text
            FROM relations
            WHERE source_key = ?
        """
        params: List[Any] = [key]
        if relation_type:
            sql += " AND relation_type = ?"
            params.append(str(relation_type))
        sql += " ORDER BY relation_type, target_text LIMIT ?"
        params.append(lim)

        with self._lock:
            return [dict(row) for row in con.execute(sql, params)]

    def relation_counts(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return relation counts by relation_type."""
        lim = _limit_value(limit)
        con = self.connect()
        sql = """
            SELECT relation_type, COUNT(*) AS n
            FROM relations
            GROUP BY relation_type
            ORDER BY n DESC, relation_type
            LIMIT ?
        """
        with self._lock:
            return [dict(row) for row in con.execute(sql, (lim,))]

    def lookup_relations(
        self,
        term: str,
        *,
        entry_limit: int = 5,
        relation_limit: int = 25,
        relation_type: Optional[str] = None,
        pos: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Look up a term and attach relation rows to each matching entry."""
        entries = self.lookup(term, limit=entry_limit, pos=pos, include_aliases=True)
        out: List[Dict[str, Any]] = []
        for entry in entries:
            key = entry.get("key")
            rels = self.relations_for_key(str(key), relation_type=relation_type, limit=relation_limit) if key else []
            out.append({"entry": entry, "relations": rels})
        return out

    def make_pos_lookup(self, *, max_entries: int = 20) -> Callable[[str], List[str]]:
        """Return a lightweight token -> POS-list function backed by the lexicon."""
        def _lookup(token: str) -> List[str]:
            seen: set[str] = set()
            result: List[str] = []
            for row in self.lookup(token, limit=max_entries):
                pos = row.get("pos")
                if pos and str(pos) not in seen:
                    seen.add(str(pos))
                    result.append(str(pos))
            return result

        return _lookup


# -----------------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------------

class LocalAIAdapter:
    """Governing bridge between LK20 and local AI advisory modules."""

    def __init__(
        self,
        project_root: Optional[str | Path] = None,
        *,
        semantic_bank_path: Optional[str | Path] = None,
        lexicon_db_path: Optional[str | Path] = None,
        lexicon_core_path: Optional[str | Path] = None,
        lexicon_manifest_path: Optional[str | Path] = None,
    ) -> None:
        self.root = Path(project_root or PROJECT_ROOT).resolve()
        self.local_ai_dir = self.root / "local_ai"
        ensure_paths(self.root)

        self.semantic_bank_path = Path(semantic_bank_path) if semantic_bank_path else None
        self.lexicon_db_path = Path(lexicon_db_path) if lexicon_db_path else self.local_ai_dir / "resources" / "kaikki_tensor" / "lexicon_kaikki_hybrid_v3.db"
        self.lexicon_core_path = Path(lexicon_core_path) if lexicon_core_path else self.lexicon_db_path.with_suffix(".core.npz")
        self.lexicon_manifest_path = Path(lexicon_manifest_path) if lexicon_manifest_path else self.lexicon_db_path.with_suffix(".manifest.json")

        self._bank: Any = None
        self._bank_load_attempted = False
        self._lexicon: Optional[TensorLexiconHybridBank] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Resource properties
    # ------------------------------------------------------------------

    def _semantic_bank_candidates(self) -> List[Path]:
        candidates: List[Path] = []
        if self.semantic_bank_path is not None:
            candidates.append(self.semantic_bank_path)
        candidates.extend(
            [
                self.local_ai_dir / "semantic_bank.npz",
                self.root / "semantic_bank.npz",
                self.local_ai_dir / "resources" / "semantic_bank.npz",
            ]
        )

        # Preserve order while removing duplicates.
        seen: set[str] = set()
        unique: List[Path] = []
        for path in candidates:
            key = str(path.resolve()) if path.exists() else str(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    @property
    def bank(self) -> Any:
        """Lazy-load the SemanticAttractorBank if available."""
        with self._lock:
            if self._bank_load_attempted:
                return self._bank

            self._bank_load_attempted = True
            ensure_paths(self.root)

            try:
                from semantic_attractors import SemanticAttractorBank  # type: ignore
            except Exception as exc:
                LOGGER.info("SemanticAttractorBank import unavailable: %s", exc)
                self._bank = None
                return None

            for bank_path in self._semantic_bank_candidates():
                try:
                    if bank_path.exists():
                        self._bank = SemanticAttractorBank.load_npz(bank_path)
                        return self._bank
                except Exception as exc:
                    LOGGER.error("Failed to load SemanticAttractorBank from %s: %s", bank_path, exc)

            self._bank = None
            return None

    @property
    def lexicon(self) -> Optional[TensorLexiconHybridBank]:
        """Lazy-create the tensor lexicon bridge if the DB exists."""
        with self._lock:
            if self._lexicon is not None:
                return self._lexicon

            if not self.lexicon_db_path.exists():
                return None

            self._lexicon = TensorLexiconHybridBank(
                db_path=self.lexicon_db_path,
                core_path=self.lexicon_core_path,
                manifest_path=self.lexicon_manifest_path,
            )
            return self._lexicon

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Status-only advisory report. No mutation of canonical state."""
        has_semantic_bank = self.bank is not None
        lexicon = self.lexicon
        has_lexicon = lexicon is not None and lexicon.available

        capabilities = [
            "advisory",
            "explanation",
            "status_only",
            "entropy_diagnostics",
            "safe_lexicon_advisory",
        ]
        if has_semantic_bank:
            capabilities.extend(["semantic_attractor_bank"])
        if has_lexicon:
            capabilities.extend(
                [
                    "tensor_lexicon_lookup",
                    "alias_resolution",
                    "relation_lookup",
                    "tensor_core_status",
                    "ranked_lexicon_lookup",
                    "context_aware_sense_ranking",
                ]
            )

        return {
            "status": "active" if (has_semantic_bank or has_lexicon) else "limited",
            "provider": "Akkurat.LocalAI",
            "project_root": str(self.root),
            "semantic_bank": "loaded" if has_semantic_bank else "missing",
            "tensor_lexicon": lexicon.status() if has_lexicon and lexicon else {"available": False, "db_path": str(self.lexicon_db_path)},
            "capabilities": capabilities,
            "entropy_nlp": self.ai_entropy_status(),
        }

    def ai_entropy_status(self) -> Dict[str, Any]:
        """Advisory block for entropy NLP status."""
        try:
            ensure_paths(self.root)
            import entropy_nlp  # type: ignore
            return entropy_nlp.status()
        except Exception as exc:
            return {"ok": False, "status": "unavailable", "error": str(exc)}

    def tensor_core_status(self) -> Dict[str, Any]:
        """Return tensor-core status through the lexicon bridge."""
        lexicon = self.lexicon
        if lexicon is None:
            return {"available": False, "core_path": str(self.lexicon_core_path)}
        return lexicon.tensor_core_status()

    # ------------------------------------------------------------------
    # Entropy NLP bridge
    # ------------------------------------------------------------------

    def _make_entropy_pos_lookup(self) -> Optional[Callable[[str], Any]]:
        """Prefer entropy_nlp's semantic-bank POS lookup; fallback to lexicon."""
        try:
            ensure_paths(self.root)
            import entropy_nlp  # type: ignore
            if self.bank is not None and hasattr(entropy_nlp, "make_pos_lookup_from_semantic_bank"):
                return entropy_nlp.make_pos_lookup_from_semantic_bank(self.bank)
        except Exception as exc:
            LOGGER.debug("Semantic-bank POS lookup unavailable: %s", exc)

        if self.lexicon is not None:
            try:
                return self.lexicon.make_pos_lookup()
            except Exception as exc:
                LOGGER.debug("Lexicon POS lookup unavailable: %s", exc)

        return None

    def analyze_generated_text(self, text: str, profile: str = "curriculum") -> Dict[str, Any]:
        """Diagnose text using entropy NLP diagnostics. Status-only."""
        try:
            ensure_paths(self.root)
            import entropy_nlp  # type: ignore

            pos_lookup = self._make_entropy_pos_lookup()
            diag = entropy_nlp.diagnose_text(text, profile=profile, pos_lookup=pos_lookup)
            if hasattr(diag, "to_dict"):
                return diag.to_dict()
            if isinstance(diag, dict):
                return diag
            return {"ok": True, "diagnostic": str(diag)}
        except Exception as exc:
            return {"ok": False, "error": f"Entropy NLP diagnostic failed: {exc}"}

    def rerank_generated_texts(
        self,
        candidates: Sequence[Any],
        context: str = "",
        profile: str = "curriculum",
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using entropy NLP. Advisory only."""
        try:
            ensure_paths(self.root)
            import entropy_nlp  # type: ignore

            pos_lookup = self._make_entropy_pos_lookup()
            result = entropy_nlp.rerank_texts(
                candidates,
                context=context,
                profile=profile,
                pos_lookup=pos_lookup,
            )
            if isinstance(result, list):
                return result
            return [{"text": _coerce_candidate_text(c), "result": result} for c in candidates]
        except Exception as exc:
            LOGGER.error("Reranking failed: %s", exc)
            return [{"text": _coerce_candidate_text(c), "error": str(exc)} for c in candidates]

    # ------------------------------------------------------------------
    # Safe advisory bridge
    # ------------------------------------------------------------------

    def suggest_improvements(self, text: str, **kwargs: Any) -> str:
        """
        Advisory-only suggestions. Does not mutate LK20 state.

        This method intentionally does not call sentence_builder.build_sentences.
        The legacy sentence builder can create semantically arbitrary sentences
        from attractor-space traversal. For local-AI governance, the safer
        default is factual lexicon anchoring plus optional entropy diagnostics.
        """
        profile = str(kwargs.get("profile", "curriculum"))
        max_terms = _limit_value(kwargs.get("max_terms", 3), default=3, maximum=8)

        tokens = _tokenize_for_advisory(str(text or ""), max_tokens=max_terms)
        lexical_lines: List[str] = []

        if self.lexicon is not None and tokens:
            for tok in tokens:
                rows = self.lookup_lexicon(tok, limit=1, context=str(text or ""))
                if not rows:
                    continue
                row = rows[0]
                word = row.get("word") or tok
                lemma = row.get("lemma") or word
                pos = row.get("pos") or "unknown-pos"
                gloss = str(row.get("gloss") or "").strip()
                if not gloss:
                    gloss = "no gloss available"
                if str(lemma).lower() != str(word).lower():
                    lexical_lines.append(f"'{tok}' → {word}/{lemma} ({pos}) — {gloss}")
                else:
                    lexical_lines.append(f"'{tok}' ({pos}) — {gloss}")

        entropy_summary = ""
        try:
            diag = self.analyze_generated_text(str(text or ""), profile=profile)
            entropy_summary = _compact_entropy_summary(diag)
        except Exception as exc:
            entropy_summary = f"Entropy diagnostic unavailable: {exc}"

        if lexical_lines and entropy_summary:
            return "Advisory: " + " | ".join(lexical_lines) + f" | Entropy: {entropy_summary}"
        if lexical_lines:
            return "Advisory: " + " | ".join(lexical_lines)
        if entropy_summary:
            return f"Advisory: No high-confidence lexicon anchor found. Entropy: {entropy_summary}"
        if self.lexicon is None:
            return "AI Status: Tensor lexicon is unavailable; no safe lexical advisory can be generated."
        return "AI Status: Unable to generate safe lexicon-based advice for this context."

    # ------------------------------------------------------------------
    # Tensor lexicon API
    # ------------------------------------------------------------------

    def lookup_lexicon(
        self,
        term: str,
        *,
        limit: int = 10,
        pos: Optional[str] = None,
        include_aliases: bool = True,
        include_relations: bool = False,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lookup a term in the tensor lexicon. Returns [] when unavailable."""
        lexicon = self.lexicon
        if lexicon is None:
            return []
        try:
            return lexicon.lookup(
                term,
                limit=limit,
                pos=pos,
                include_aliases=include_aliases,
                include_relations=include_relations,
                context=context,
            )
        except Exception as exc:
            LOGGER.error("Lexicon lookup failed for %r: %s", term, exc)
            return []

    def resolve_alias(
        self,
        alias: str,
        *,
        limit: int = 10,
        pos: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Resolve an inflected/alias form through the tensor lexicon."""
        lexicon = self.lexicon
        if lexicon is None:
            return []
        try:
            return lexicon.resolve_alias(alias, limit=limit, pos=pos, context=context)
        except Exception as exc:
            LOGGER.error("Alias resolution failed for %r: %s", alias, exc)
            return []

    # Backward-compatible alias name for call sites that use lookup_alias.
    lookup_alias = resolve_alias

    def get_lexicon_entry(self, key: str, *, include_relations: bool = False) -> Optional[Dict[str, Any]]:
        """Fetch a lexicon entry by its primary key."""
        lexicon = self.lexicon
        if lexicon is None:
            return None
        try:
            return lexicon.get_entry(key, include_relations=include_relations)
        except Exception as exc:
            LOGGER.error("Lexicon entry fetch failed for %r: %s", key, exc)
            return None

    def lookup_relations(
        self,
        term: str,
        *,
        entry_limit: int = 5,
        relation_limit: int = 25,
        relation_type: Optional[str] = None,
        pos: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lookup term entries and attach table-mode relation rows."""
        lexicon = self.lexicon
        if lexicon is None:
            return []
        try:
            return lexicon.lookup_relations(
                term,
                entry_limit=entry_limit,
                relation_limit=relation_limit,
                relation_type=relation_type,
                pos=pos,
            )
        except Exception as exc:
            LOGGER.error("Relation lookup failed for %r: %s", term, exc)
            return []

    def relation_counts(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Return relation counts from the tensor lexicon."""
        lexicon = self.lexicon
        if lexicon is None:
            return []
        try:
            return lexicon.relation_counts(limit=limit)
        except Exception as exc:
            LOGGER.error("Relation count query failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def embed_curriculum_text(self, text: str) -> List[float]:
        """
        Legacy deterministic curriculum embedding.

        Returns 128 floats by repeating a SHA-256 byte digest four times. This is
        intentionally not a neural embedding; it preserves old call-site shape
        expectations without importing model runtimes.
        """
        h = hashlib.sha256(str(text).encode("utf-8", errors="replace")).digest()
        return [(b / 255.0) for b in h] * 4

    def close(self) -> None:
        """Close local read-only resources."""
        with self._lock:
            if self._lexicon is not None:
                self._lexicon.close()


# -----------------------------------------------------------------------------
# Public factory and convenience functions
# -----------------------------------------------------------------------------

def get_adapter(root: Optional[str | Path] = None) -> LocalAIAdapter:
    """Return a new LocalAIAdapter instance."""
    return LocalAIAdapter(root)


def get_status(root: Optional[str | Path] = None) -> Dict[str, Any]:
    """Convenience status function for lightweight integrations."""
    adapter = get_adapter(root)
    try:
        return adapter.get_status()
    finally:
        adapter.close()


def lookup_lexicon(
    term: str,
    *,
    limit: int = 10,
    pos: Optional[str] = None,
    root: Optional[str | Path] = None,
    include_aliases: bool = True,
    context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience lexicon lookup function."""
    adapter = get_adapter(root)
    try:
        return adapter.lookup_lexicon(term, limit=limit, pos=pos, include_aliases=include_aliases, context=context)
    finally:
        adapter.close()


def lookup_alias(
    alias: str,
    *,
    limit: int = 10,
    pos: Optional[str] = None,
    root: Optional[str | Path] = None,
    context: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convenience alias lookup function."""
    adapter = get_adapter(root)
    try:
        return adapter.resolve_alias(alias, limit=limit, pos=pos, context=context)
    finally:
        adapter.close()


# -----------------------------------------------------------------------------
# CLI self-test
# -----------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    adapter = get_adapter()

    try:
        print(json.dumps(adapter.get_status(), indent=2, ensure_ascii=False), flush=True)

        print("\nLEXICON SAMPLE:", flush=True)
        for term in ["cat", "run", "beautiful", "because", "think"]:
            rows = adapter.lookup_lexicon(term, limit=3)
            print(f"\n{term}: {len(rows)}", flush=True)

            for row in rows:
                print(
                    json.dumps(
                        {
                            "key": row.get("key"),
                            "word": row.get("word"),
                            "lemma": row.get("lemma"),
                            "pos": row.get("pos"),
                            "gloss": row.get("gloss"),
                            "match_type": row.get("match_type"),
                            "matched_alias": row.get("matched_alias"),
                            "alias_type": row.get("alias_type"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        print("\nALIAS SAMPLE:", flush=True)
        for term in ["running", "runs", "cats", "thought"]:
            rows = adapter.resolve_alias(term, limit=2)
            print(f"\n{term}: {len(rows)}", flush=True)
            for row in rows:
                print(
                    json.dumps(
                        {
                            "word": row.get("word"),
                            "lemma": row.get("lemma"),
                            "pos": row.get("pos"),
                            "gloss": row.get("gloss"),
                            "matched_alias": row.get("matched_alias"),
                            "alias_type": row.get("alias_type"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        print("\nADVISORY SAMPLE:", flush=True)
        print(adapter.suggest_improvements("cat animal"), flush=True)

        return 0

    except Exception as exc:
        logging.exception("local_ai_adapter self-test failed")
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 1

    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
