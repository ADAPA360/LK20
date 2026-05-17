#!/usr/bin/env python3
# tensor_lexicon_ingestor.py
"""
LK20 Local AI — Production Streaming Tensor Lexicon Ingestor
============================================================

Streams Kaikki/Wiktextract raw JSONL(.GZ) shards into a restartable hybrid bank:

  1. SQLite metadata store
  2. shard/POS semantic aggregate table
  3. optional TensorTrain aggregate core using root tn.py

Production defaults in this version target the observed Windows bottleneck:
  - per-entry vectors are OFF by default
  - shard sorting is numeric, not lexicographic
  - bad Wiktextract form aliases are filtered and purged from older DBs
  - incomplete shards are cleaned on resume
  - secondary indexes are deferred until finalization
  - SQLite DELETE journal mode is the default for long single-writer Windows runs

Typical safe full run, from local_ai:

    python .\\tensor_lexicon_ingestor.py run ^
      --shards-dir .\\resources\\kaikki\\raw_shards ^
      --out .\\resources\\kaikki_tensor\\lexicon_kaikki_hybrid_v3.db ^
      --dim 128 ^
      --seed 1729 ^
      --vectors-mode none
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Project-root import setup
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tn import TensorTrain, factorize_into_modes
except Exception as exc:  # pragma: no cover
    print(
        "ERROR: Could not import TensorTrain/factorize_into_modes from root tn.py.\n"
        f"Project root expected at: {PROJECT_ROOT}\n"
        f"Import error: {exc}",
        file=sys.stderr,
    )
    raise


# =============================================================================
# Constants
# =============================================================================

SCHEMA_VERSION = 3

DEFAULT_ALLOWED_POS = (
    "noun",
    "verb",
    "adjective",
    "adverb",
    "preposition",
    "pronoun",
    "determiner",
    "conjunction",
    "interjection",
)

DEFAULT_RELATION_FIELDS = (
    "synonyms",
    "antonyms",
    "hypernyms",
    "hyponyms",
    "holonyms",
    "meronyms",
    "related",
    "derived",
    "coordinate_terms",
)

POS_ALIASES = {
    "": "",
    "n": "noun",
    "noun": "noun",
    "proper_noun": "noun",
    "proper noun": "noun",
    "proper-noun": "noun",
    "name": "noun",
    "num": "noun",
    "number": "noun",
    "numeral": "noun",
    "v": "verb",
    "verb": "verb",
    "vb": "verb",
    "adj": "adjective",
    "adjective": "adjective",
    "adv": "adverb",
    "adverb": "adverb",
    "prep": "preposition",
    "preposition": "preposition",
    "pron": "pronoun",
    "pronoun": "pronoun",
    "det": "determiner",
    "determiner": "determiner",
    "conj": "conjunction",
    "conjunction": "conjunction",
    "coord_conj": "conjunction",
    "subordinating_conjunction": "conjunction",
    "intj": "interjection",
    "interjection": "interjection",
}

# Wiktextract form pollution observed in Kaikki rows. These are metadata/control
# values, not lexical forms. Only form aliases are purged; real lemmas remain.
FORM_VALUE_BLACKLIST = {
    "",
    "-",
    "—",
    "–",
    "?",
    "unknown",
    "no-table-tags",
    "table-tags",
    "glossary",
    "inflection-table",
    "inflection-template",
    "conjugation-table",
    "declension-table",
}

FORM_TAG_BLACKLIST = {
    "no-table-tags",
    "table-tags",
    "glossary",
    "maintenance",
    "request",
    "deprecated",
    "obsolete-template",
    "inflection-table",
    "inflection-template",
    "conjugation-table",
    "declension-table",
}

_SPACE_RE = re.compile(r"\s+", re.UNICODE)
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", re.UNICODE)
_SHARD_RANGE_RE = re.compile(r"raw_(\d+)_(\d+)\.jsonl(?:\.gz)?$", re.IGNORECASE)


# =============================================================================
# SQLite schema
# =============================================================================

SQL_SCHEMA_CORE = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    key TEXT PRIMARY KEY,
    shard_name TEXT NOT NULL,
    source_line INTEGER NOT NULL,
    sense_index INTEGER NOT NULL,
    word TEXT NOT NULL,
    lemma TEXT NOT NULL,
    alias_key TEXT NOT NULL,
    pos TEXT NOT NULL,
    gloss TEXT NOT NULL,
    weight REAL NOT NULL,
    source TEXT NOT NULL,
    lang_code TEXT NOT NULL,
    sense_id TEXT,
    metadata_json TEXT NOT NULL,
    vector BLOB,
    vector_dim INTEGER,
    vector_dtype TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT NOT NULL,
    key TEXT NOT NULL,
    alias_type TEXT NOT NULL,
    PRIMARY KEY (alias, key, alias_type),
    FOREIGN KEY (key) REFERENCES entries(key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS relations (
    source_key TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    target_text TEXT NOT NULL,
    PRIMARY KEY (source_key, relation_type, target_text),
    FOREIGN KEY (source_key) REFERENCES entries(key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS shard_pos_sums (
    shard_name TEXT NOT NULL,
    pos TEXT NOT NULL,
    count INTEGER NOT NULL,
    sum_vector BLOB NOT NULL,
    vector_dim INTEGER NOT NULL,
    vector_dtype TEXT NOT NULL,
    PRIMARY KEY (shard_name, pos)
);

CREATE TABLE IF NOT EXISTS processed_shards (
    shard_name TEXT PRIMARY KEY,
    shard_path TEXT NOT NULL,
    shard_size INTEGER NOT NULL,
    shard_mtime_ns INTEGER NOT NULL,
    status TEXT NOT NULL,
    records_read INTEGER NOT NULL DEFAULT 0,
    records_selected INTEGER NOT NULL DEFAULT 0,
    skipped_wrong_language INTEGER NOT NULL DEFAULT 0,
    skipped_bad_pos INTEGER NOT NULL DEFAULT 0,
    entries_ingested INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0,
    started_at REAL NOT NULL,
    finished_at REAL
);
"""

SQL_SECONDARY_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_entries_lemma ON entries(lemma);
CREATE INDEX IF NOT EXISTS idx_entries_alias_key ON entries(alias_key);
CREATE INDEX IF NOT EXISTS idx_entries_pos ON entries(pos);
CREATE INDEX IF NOT EXISTS idx_entries_shard ON entries(shard_name);
CREATE INDEX IF NOT EXISTS idx_entries_word_pos ON entries(word, pos);
CREATE INDEX IF NOT EXISTS idx_aliases_alias ON aliases(alias);
CREATE INDEX IF NOT EXISTS idx_aliases_key ON aliases(key);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_key);
CREATE INDEX IF NOT EXISTS idx_relations_type_target ON relations(relation_type, target_text);
CREATE INDEX IF NOT EXISTS idx_processed_status ON processed_shards(status);
"""

REQUIRED_ENTRY_COLUMNS = {
    "key",
    "shard_name",
    "source_line",
    "sense_index",
    "word",
    "lemma",
    "alias_key",
    "pos",
    "gloss",
    "weight",
    "source",
    "lang_code",
    "sense_id",
    "metadata_json",
    "vector",
    "vector_dim",
    "vector_dtype",
    "created_at",
}


# =============================================================================
# Generic helpers
# =============================================================================

def now_ts() -> float:
    return float(time.time())


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return _SPACE_RE.sub(" ", text).strip()


def alias_text(value: Any) -> str:
    return canonical_text(value).casefold()


def normalize_pos(pos: Any) -> str:
    p = canonical_text(pos).casefold().replace("-", "_").replace(" ", "_")
    return POS_ALIASES.get(p, p)


def parse_csv_set(value: str) -> Tuple[str, ...]:
    out: List[str] = []
    for item in str(value or "").split(","):
        item = canonical_text(item)
        if item:
            out.append(item)
    return tuple(dict.fromkeys(out))


def tokenize(text: str, max_tokens: int = 128) -> List[str]:
    if not text:
        return []
    return [m.group(0).casefold() for m in _TOKEN_RE.finditer(text)][: int(max_tokens)]


def sha256_hex(text: str, length: Optional[int] = None) -> str:
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    return h if length is None else h[: int(length)]


def hash_feature(feature: str, dim: int, seed: int) -> Tuple[int, float]:
    payload = f"{seed}|{feature}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    sign_value = int.from_bytes(digest[8:16], "little", signed=False)
    return value % int(dim), (1.0 if (sign_value & 1) == 0 else -1.0)


def add_hashed_feature(vec: np.ndarray, feature: str, scale: float, seed: int) -> None:
    idx, sign = hash_feature(feature, int(vec.size), int(seed))
    vec[idx] += np.float32(float(scale) * sign)


def vectorize_lexicon_entry(
    *,
    lemma: str,
    pos: str,
    gloss: str,
    relations: Mapping[str, Sequence[str]],
    weight: float,
    dim: int,
    seed: int,
) -> np.ndarray:
    """Deterministic signed feature hashing; no Python hash() dependency."""
    d = int(dim)
    if d <= 0:
        raise ValueError("dim must be positive.")

    vec = np.zeros(d, dtype=np.float32)
    lemma_a = alias_text(lemma)
    pos_a = alias_text(pos)

    if lemma_a:
        add_hashed_feature(vec, f"lemma:{lemma_a}", 2.25, seed)
        if len(lemma_a) >= 3:
            add_hashed_feature(vec, f"prefix3:{lemma_a[:3]}", 0.25, seed)
            add_hashed_feature(vec, f"suffix3:{lemma_a[-3:]}", 0.25, seed)
        if len(lemma_a) >= 4:
            add_hashed_feature(vec, f"suffix4:{lemma_a[-4:]}", 0.20, seed)

    if pos_a:
        add_hashed_feature(vec, f"pos:{pos_a}", 1.25, seed)
        if lemma_a:
            add_hashed_feature(vec, f"lemma_pos:{lemma_a}:{pos_a}", 1.00, seed)

    toks = tokenize(gloss, max_tokens=128)
    if toks:
        token_scale = 1.0 / max(1.0, float(len(toks)) ** 0.5)
        for tok in toks:
            add_hashed_feature(vec, f"gloss_tok:{tok}", token_scale, seed)
            if pos_a:
                add_hashed_feature(vec, f"pos_gloss_tok:{pos_a}:{tok}", 0.35 * token_scale, seed)

    for rel_type, targets in sorted(relations.items()):
        rt = alias_text(rel_type).replace(" ", "_")
        if not rt:
            continue
        for target in list(targets)[:32]:
            t = alias_text(target)
            if not t:
                continue
            add_hashed_feature(vec, f"rel:{rt}:{t}", 0.80, seed)
            if lemma_a:
                add_hashed_feature(vec, f"lemma_rel:{lemma_a}:{rt}:{t}", 0.30, seed)

    norm = float(np.linalg.norm(vec))
    if np.isfinite(norm) and norm > 1e-12:
        vec = vec / np.float32(norm)

    w = float(weight)
    if np.isfinite(w):
        vec = vec * np.float32(w)

    return np.ascontiguousarray(vec, dtype=np.float32)


def array_to_blob(arr: np.ndarray) -> sqlite3.Binary:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return sqlite3.Binary(a.tobytes(order="C"))


def blob_to_array(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != int(dim):
        raise ValueError(f"Vector blob size {arr.size} does not match expected dim {dim}.")
    return arr.astype(np.float32, copy=True)


def open_text_auto(path: Path):
    p = Path(path)
    if p.suffix.casefold() == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace", newline="")
    return p.open("rt", encoding="utf-8", errors="replace", newline="")


def atomic_write_json(path: Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)


def safe_stat(path: Path) -> Tuple[int, int]:
    st = Path(path).stat()
    return int(st.st_size), int(st.st_mtime_ns)


def shard_range_tuple(name_or_path: Any) -> Tuple[int, int, str]:
    name = Path(name_or_path).name
    m = _SHARD_RANGE_RE.match(name)
    if m:
        return (int(m.group(1)), int(m.group(2)), name)
    return (10**30, 10**30, name)


def shard_sort_key(path: Path) -> Tuple[int, int, str]:
    return shard_range_tuple(path)


def clean_relation_type(name: str) -> str:
    return alias_text(name).replace(" ", "_").replace("-", "_")


def extract_relation_targets(values: Any, limit: int) -> List[str]:
    out: List[str] = []
    seen = set()

    if not isinstance(values, list):
        return out

    for item in values:
        target = ""
        if isinstance(item, str):
            target = item
        elif isinstance(item, Mapping):
            target = (
                item.get("word")
                or item.get("term")
                or item.get("english")
                or item.get("roman")
                or item.get("alt")
                or ""
            )

        target = canonical_text(target)
        if not target:
            continue

        key = alias_text(target)
        if not key or key in seen:
            continue

        seen.add(key)
        out.append(target)

        if len(out) >= int(limit):
            break

    return out


def normalize_form_tag(value: Any) -> str:
    return alias_text(value).replace("_", "-").replace(" ", "-")


def is_usable_form_alias(value: Any, tags: Any = None) -> bool:
    form = canonical_text(value)
    if not form:
        return False

    a = alias_text(form)
    if not a:
        return False

    if a in FORM_VALUE_BLACKLIST:
        return False

    # Avoid obvious template/control artifacts.
    if any(ch in form for ch in "{}[]|#<>="):
        return False

    if len(form) > 120:
        return False

    if len(form.split()) > 8:
        return False

    if isinstance(tags, list):
        norm_tags = {normalize_form_tag(t) for t in tags if canonical_text(t)}
        if norm_tags & FORM_TAG_BLACKLIST:
            return False

    return True


def extract_forms(record: Mapping[str, Any], limit: int = 128) -> List[str]:
    forms = record.get("forms", [])
    if not isinstance(forms, list):
        return []

    out: List[str] = []
    seen = set()

    for item in forms:
        value = ""
        tags: Any = []

        if isinstance(item, str):
            value = item
            tags = []
        elif isinstance(item, Mapping):
            value = item.get("form") or item.get("word") or ""
            tags = item.get("tags", [])
        else:
            continue

        if not is_usable_form_alias(value, tags):
            continue

        value = canonical_text(value)
        key = alias_text(value)

        if not key or key in seen:
            continue

        seen.add(key)
        out.append(value)

        if len(out) >= int(limit):
            break

    return out


def get_record_lang_code(record: Mapping[str, Any]) -> str:
    return alias_text(record.get("lang_code") or record.get("langCode") or "")


def get_record_lang_name(record: Mapping[str, Any]) -> str:
    return alias_text(record.get("lang") or record.get("language") or "")


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class LexiconEntry:
    key: str
    shard_name: str
    source_line: int
    sense_index: int
    word: str
    lemma: str
    alias_key: str
    pos: str
    gloss: str
    weight: float
    source: str
    lang_code: str
    sense_id: str
    metadata: Dict[str, Any]
    relations: Dict[str, List[str]]
    aliases: List[Tuple[str, str]]
    vector: np.ndarray


@dataclass
class ShardStats:
    shard_name: str
    records_read: int = 0
    records_selected: int = 0
    skipped_wrong_language: int = 0
    skipped_bad_pos: int = 0
    entries_ingested: int = 0
    errors: int = 0
    started_at: float = field(default_factory=now_ts)

    def elapsed(self) -> float:
        return max(1e-6, now_ts() - self.started_at)

    def rate(self) -> float:
        return float(self.entries_ingested) / self.elapsed()


@dataclass
class IngestStats:
    shards_total: int = 0
    shards_processed: int = 0
    shards_skipped: int = 0
    records_read: int = 0
    records_selected: int = 0
    skipped_wrong_language: int = 0
    skipped_bad_pos: int = 0
    entries_ingested: int = 0
    errors: int = 0
    started_at: float = field(default_factory=now_ts)

    def elapsed(self) -> float:
        return max(1e-6, now_ts() - self.started_at)

    def rate(self) -> float:
        return float(self.entries_ingested) / self.elapsed()


# =============================================================================
# Main ingestor
# =============================================================================

class TensorLexiconIngestor:
    def __init__(
        self,
        *,
        db_path: Path,
        dim: int = 128,
        seed: int = 2027,
        lang_code: str = "en",
        allowed_pos: Sequence[str] = DEFAULT_ALLOWED_POS,
        relation_fields: Sequence[str] = DEFAULT_RELATION_FIELDS,
        relation_limit: int = 32,
        batch_size: int = 1000,
        vectors_mode: str = "none",
        relations_mode: str = "table",
        aliases_mode: str = "clean-forms",
        store_categories: bool = False,
        max_categories: int = 32,
        max_gloss_chars: int = 2000,
        busy_timeout_ms: int = 30000,
        journal_mode: str = "DELETE",
        synchronous: str = "NORMAL",
        create_indexes_on_init: bool = False,
        final_indexes: bool = True,
        purge_bad_aliases: bool = True,
        wal_checkpoint_every: int = 25,
    ):
        self.db_path = Path(db_path)
        self.dim = int(dim)
        self.seed = int(seed)
        self.lang_code = alias_text(lang_code or "en")
        self.allowed_pos = tuple(normalize_pos(p) for p in allowed_pos if normalize_pos(p))
        self.allowed_pos_set = set(self.allowed_pos)
        self.relation_fields = tuple(relation_fields)
        self.relation_limit = int(relation_limit)
        self.batch_size = int(max(1, batch_size))
        self.vectors_mode = str(vectors_mode or "none").lower().strip()
        self.relations_mode = str(relations_mode or "table").lower().strip()
        self.aliases_mode = str(aliases_mode or "clean-forms").lower().strip()
        self.store_categories = bool(store_categories)
        self.max_categories = int(max(0, max_categories))
        self.max_gloss_chars = int(max(1, max_gloss_chars))
        self.busy_timeout_ms = int(busy_timeout_ms)
        self.journal_mode = str(journal_mode or "DELETE").upper().strip()
        self.synchronous = str(synchronous or "NORMAL").upper().strip()
        self.create_indexes_on_init = bool(create_indexes_on_init)
        self.final_indexes = bool(final_indexes)
        self.purge_bad_aliases = bool(purge_bad_aliases)
        self.wal_checkpoint_every = int(max(0, wal_checkpoint_every))
        self.stats = IngestStats()

        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        if not self.allowed_pos:
            raise ValueError("allowed_pos cannot be empty.")
        if self.vectors_mode not in {"none", "entry"}:
            raise ValueError("vectors_mode must be 'none' or 'entry'.")
        if self.relations_mode not in {"table", "json", "off"}:
            raise ValueError("relations_mode must be 'table', 'json', or 'off'.")
        if self.aliases_mode not in {"lemma-only", "clean-forms", "off"}:
            raise ValueError("aliases_mode must be 'lemma-only', 'clean-forms', or 'off'.")

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms}")
        conn.execute(f"PRAGMA journal_mode = {self.journal_mode}")
        conn.execute(f"PRAGMA synchronous = {self.synchronous}")
        conn.execute("PRAGMA temp_store = MEMORY")
        return conn

    def init_db(self, conn: sqlite3.Connection) -> None:
        self._guard_existing_schema(conn)
        conn.executescript(SQL_SCHEMA_CORE)
        if self.create_indexes_on_init:
            conn.executescript(SQL_SECONDARY_INDEXES)

        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (
                "ingestor_config",
                stable_json(
                    {
                        "dim": self.dim,
                        "seed": self.seed,
                        "lang_code": self.lang_code,
                        "allowed_pos": list(self.allowed_pos),
                        "relation_fields": list(self.relation_fields),
                        "relation_limit": self.relation_limit,
                        "vectors_mode": self.vectors_mode,
                        "relations_mode": self.relations_mode,
                        "aliases_mode": self.aliases_mode,
                        "store_categories": self.store_categories,
                        "max_categories": self.max_categories,
                        "max_gloss_chars": self.max_gloss_chars,
                        "journal_mode": self.journal_mode,
                        "synchronous": self.synchronous,
                    }
                ),
            ),
        )
        conn.commit()

        if self.purge_bad_aliases:
            self.purge_polluted_form_aliases(conn)

    def create_secondary_indexes(self, conn: sqlite3.Connection) -> None:
        print("[INDEX] creating secondary indexes...", flush=True)
        conn.executescript(SQL_SECONDARY_INDEXES)
        conn.commit()
        print("[INDEX] done.", flush=True)

    def _guard_existing_schema(self, conn: sqlite3.Connection) -> None:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='entries'"
        ).fetchone()

        if not table:
            return

        cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(entries)").fetchall()}
        missing = sorted(REQUIRED_ENTRY_COLUMNS - cols)
        if missing:
            raise RuntimeError(
                "Existing SQLite DB has an incompatible old entries schema. "
                "Use --reset-output, choose a new --out path, or migrate manually. "
                f"Missing columns: {missing}"
            )

    def purge_polluted_form_aliases(self, conn: sqlite3.Connection) -> None:
        aliases = sorted(a for a in FORM_VALUE_BLACKLIST if a)
        if not aliases:
            return
        q = ",".join("?" for _ in aliases)
        n = int(
            conn.execute(
                f"SELECT COUNT(*) FROM aliases WHERE alias_type='form' AND alias IN ({q})",
                aliases,
            ).fetchone()[0]
        )
        if n:
            conn.execute(
                f"DELETE FROM aliases WHERE alias_type='form' AND alias IN ({q})",
                aliases,
            )
            conn.commit()
            print(f"[CLEAN] removed polluted form aliases: {n}", flush=True)

    def cleanup_incomplete_shards(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT shard_name FROM processed_shards WHERE status != 'complete'"
        ).fetchall()

        if not rows:
            return

        print(f"[RECOVERY] Cleaning {len(rows)} incomplete shard ledger entries.")
        conn.execute("BEGIN IMMEDIATE")
        try:
            for (shard_name,) in rows:
                self._delete_shard_locked(conn, str(shard_name))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _delete_shard_locked(self, conn: sqlite3.Connection, shard_name: str) -> None:
        conn.execute("DELETE FROM entries WHERE shard_name = ?", (shard_name,))
        conn.execute("DELETE FROM shard_pos_sums WHERE shard_name = ?", (shard_name,))
        conn.execute("DELETE FROM processed_shards WHERE shard_name = ?", (shard_name,))

    def is_shard_complete(self, conn: sqlite3.Connection, shard_path: Path) -> bool:
        size, mtime_ns = safe_stat(shard_path)
        row = conn.execute(
            """
            SELECT shard_size, shard_mtime_ns, status
            FROM processed_shards
            WHERE shard_name = ?
            """,
            (shard_path.name,),
        ).fetchone()

        if not row:
            return False

        old_size, old_mtime_ns, status = row
        return (
            int(old_size) == int(size)
            and int(old_mtime_ns) == int(mtime_ns)
            and str(status) == "complete"
        )

    def mark_shard_started(self, conn: sqlite3.Connection, shard_path: Path) -> None:
        size, mtime_ns = safe_stat(shard_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO processed_shards (
                shard_name,
                shard_path,
                shard_size,
                shard_mtime_ns,
                status,
                records_read,
                records_selected,
                skipped_wrong_language,
                skipped_bad_pos,
                entries_ingested,
                errors,
                started_at,
                finished_at
            )
            VALUES (?, ?, ?, ?, 'started', 0, 0, 0, 0, 0, 0, ?, NULL)
            """,
            (shard_path.name, str(shard_path), int(size), int(mtime_ns), now_ts()),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def iter_entries_from_record(
        self,
        *,
        record: Mapping[str, Any],
        shard_name: str,
        line_no: int,
    ) -> Iterator[LexiconEntry]:
        word = canonical_text(record.get("word") or record.get("title") or "")
        if not word:
            return

        raw_pos = record.get("pos", "")
        pos = normalize_pos(raw_pos)
        if not pos or pos not in self.allowed_pos_set:
            return

        senses = record.get("senses", [])
        if not isinstance(senses, list):
            return

        forms = extract_forms(record, limit=128) if self.aliases_mode == "clean-forms" else []

        for sense_index, sense in enumerate(senses):
            if not isinstance(sense, Mapping):
                continue

            sense_pos = normalize_pos(sense.get("pos") or pos)
            if not sense_pos or sense_pos not in self.allowed_pos_set:
                continue

            glosses = sense.get("glosses", [])
            if not isinstance(glosses, list) or not glosses:
                continue

            gloss_parts = [canonical_text(g) for g in glosses if canonical_text(g)]
            gloss = canonical_text(" ".join(gloss_parts))
            if not gloss:
                continue
            if len(gloss) > self.max_gloss_chars:
                gloss = gloss[: self.max_gloss_chars].rstrip()

            relations: Dict[str, List[str]] = {}
            if self.relations_mode != "off":
                for rel_field in self.relation_fields:
                    targets = extract_relation_targets(sense.get(rel_field, []), limit=self.relation_limit)
                    if targets:
                        relations[clean_relation_type(rel_field)] = targets

            sense_id_raw = (
                sense.get("id")
                or sense.get("senseid")
                or sense.get("sense_id")
                or sense.get("wikidata")
                or ""
            )
            sense_id = canonical_text(sense_id_raw)
            if not sense_id:
                sense_id = f"{shard_name}:{line_no}:{sense_index}"

            key_payload = stable_json(
                {
                    "source": "kaikki:wiktextract",
                    "lang_code": self.lang_code,
                    "shard_name": shard_name,
                    "source_line": int(line_no),
                    "sense_index": int(sense_index),
                    "word": word,
                    "pos": sense_pos,
                    "gloss_sha256": sha256_hex(gloss),
                }
            )
            key = f"kaikki:{sha256_hex(key_payload, 32)}"

            metadata: Dict[str, Any] = {
                "source_shard": shard_name,
                "source_line": int(line_no),
                "record_pos": canonical_text(raw_pos),
                "sense_tags": sense.get("tags", []) if isinstance(sense.get("tags", []), list) else [],
                "record_tags": record.get("tags", []) if isinstance(record.get("tags", []), list) else [],
            }

            if self.relations_mode == "json" and relations:
                metadata["relations"] = relations

            if self.store_categories:
                cats = sense.get("categories") or record.get("categories") or []
                if isinstance(cats, list):
                    metadata["categories"] = [canonical_text(c) for c in cats[: self.max_categories] if canonical_text(c)]

            vector = vectorize_lexicon_entry(
                lemma=word,
                pos=sense_pos,
                gloss=gloss,
                relations=relations,
                weight=1.0,
                dim=self.dim,
                seed=self.seed,
            )

            aliases: List[Tuple[str, str]] = []
            if self.aliases_mode != "off":
                main_alias = alias_text(word)
                if main_alias:
                    aliases.append((main_alias, "lemma"))

                if self.aliases_mode == "clean-forms":
                    for form in forms:
                        a = alias_text(form)
                        if a and a != main_alias and is_usable_form_alias(a):
                            aliases.append((a, "form"))

            aliases = list(dict.fromkeys(aliases))

            yield LexiconEntry(
                key=key,
                shard_name=shard_name,
                source_line=int(line_no),
                sense_index=int(sense_index),
                word=word,
                lemma=word,
                alias_key=alias_text(word),
                pos=sense_pos,
                gloss=gloss,
                weight=1.0,
                source="kaikki:wiktextract",
                lang_code=self.lang_code,
                sense_id=sense_id,
                metadata=metadata,
                relations=relations,
                aliases=aliases,
                vector=vector,
            )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_shard(
        self,
        *,
        conn: sqlite3.Connection,
        shard_path: Path,
        force_reprocess: bool = False,
        progress_every: int = 0,
    ) -> ShardStats:
        shard_path = Path(shard_path)
        shard_name = shard_path.name

        if not force_reprocess and self.is_shard_complete(conn, shard_path):
            self.stats.shards_skipped += 1
            print(f"[SKIP] {shard_name} already complete.")
            return ShardStats(shard_name=shard_name)

        self.mark_shard_started(conn, shard_path)
        shard_stats = ShardStats(shard_name=shard_name)

        entries_batch: List[Tuple[Any, ...]] = []
        aliases_batch: List[Tuple[str, str, str]] = []
        relations_batch: List[Tuple[str, str, str]] = []
        pos_sums: Dict[str, np.ndarray] = {}
        pos_counts: Dict[str, int] = {}

        def flush_batches() -> None:
            nonlocal entries_batch, aliases_batch, relations_batch
            if entries_batch:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO entries (
                        key,
                        shard_name,
                        source_line,
                        sense_index,
                        word,
                        lemma,
                        alias_key,
                        pos,
                        gloss,
                        weight,
                        source,
                        lang_code,
                        sense_id,
                        metadata_json,
                        vector,
                        vector_dim,
                        vector_dtype,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    entries_batch,
                )
                entries_batch = []

            if aliases_batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO aliases(alias, key, alias_type) VALUES (?, ?, ?)",
                    aliases_batch,
                )
                aliases_batch = []

            if relations_batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO relations(source_key, relation_type, target_text) VALUES (?, ?, ?)",
                    relations_batch,
                )
                relations_batch = []

        print(f"[RUN] {shard_name}", flush=True)

        conn.execute("BEGIN IMMEDIATE")
        try:
            self._delete_shard_locked(conn, shard_name)

            with open_text_auto(shard_path) as f:
                for line_no, line in enumerate(f, start=1):
                    shard_stats.records_read += 1
                    self.stats.records_read += 1

                    if not line.strip():
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        shard_stats.errors += 1
                        self.stats.errors += 1
                        continue

                    rec_lang = get_record_lang_code(record)
                    rec_lang_name = get_record_lang_name(record)

                    if rec_lang:
                        if rec_lang != self.lang_code:
                            shard_stats.skipped_wrong_language += 1
                            self.stats.skipped_wrong_language += 1
                            continue
                    elif rec_lang_name and rec_lang_name not in {"english", self.lang_code}:
                        shard_stats.skipped_wrong_language += 1
                        self.stats.skipped_wrong_language += 1
                        continue
                    elif not rec_lang and not rec_lang_name:
                        shard_stats.skipped_wrong_language += 1
                        self.stats.skipped_wrong_language += 1
                        continue

                    raw_pos = normalize_pos(record.get("pos", ""))
                    if not raw_pos or raw_pos not in self.allowed_pos_set:
                        shard_stats.skipped_bad_pos += 1
                        self.stats.skipped_bad_pos += 1
                        continue

                    shard_stats.records_selected += 1
                    self.stats.records_selected += 1

                    for entry in self.iter_entries_from_record(record=record, shard_name=shard_name, line_no=line_no):
                        if self.vectors_mode == "entry":
                            vec_blob = array_to_blob(entry.vector)
                            vec_dim: Optional[int] = int(self.dim)
                            vec_dtype: Optional[str] = "float32"
                        else:
                            vec_blob = None
                            vec_dim = None
                            vec_dtype = None

                        entries_batch.append(
                            (
                                entry.key,
                                entry.shard_name,
                                entry.source_line,
                                entry.sense_index,
                                entry.word,
                                entry.lemma,
                                entry.alias_key,
                                entry.pos,
                                entry.gloss,
                                float(entry.weight),
                                entry.source,
                                entry.lang_code,
                                entry.sense_id,
                                stable_json(entry.metadata),
                                vec_blob,
                                vec_dim,
                                vec_dtype,
                                now_ts(),
                            )
                        )

                        for alias, alias_type in entry.aliases:
                            # Defense-in-depth: never insert polluted form aliases.
                            if alias_type == "form" and alias in FORM_VALUE_BLACKLIST:
                                continue
                            aliases_batch.append((alias, entry.key, alias_type))

                        if self.relations_mode == "table":
                            for rel_type, targets in entry.relations.items():
                                rt = clean_relation_type(rel_type)
                                for target in targets[: self.relation_limit]:
                                    target_clean = canonical_text(target)
                                    if target_clean:
                                        relations_batch.append((entry.key, rt, target_clean))

                        if entry.pos not in pos_sums:
                            pos_sums[entry.pos] = np.zeros(self.dim, dtype=np.float32)
                            pos_counts[entry.pos] = 0
                        pos_sums[entry.pos] += entry.vector
                        pos_counts[entry.pos] += 1

                        shard_stats.entries_ingested += 1
                        self.stats.entries_ingested += 1

                        if len(entries_batch) >= self.batch_size:
                            flush_batches()

                    if progress_every and progress_every > 0 and shard_stats.records_read % int(progress_every) == 0:
                        print(
                            "[PROGRESS] "
                            f"shard={shard_name} "
                            f"records={shard_stats.records_read:,} "
                            f"selected={shard_stats.records_selected:,} "
                            f"entries={shard_stats.entries_ingested:,} "
                            f"skipped_lang={shard_stats.skipped_wrong_language:,} "
                            f"errors={shard_stats.errors:,}",
                            flush=True,
                        )

            flush_batches()

            for pos, sum_vec in sorted(pos_sums.items()):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO shard_pos_sums (
                        shard_name, pos, count, sum_vector, vector_dim, vector_dtype
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (shard_name, pos, int(pos_counts[pos]), array_to_blob(sum_vec), int(self.dim), "float32"),
                )

            size, mtime_ns = safe_stat(shard_path)
            conn.execute(
                """
                INSERT OR REPLACE INTO processed_shards (
                    shard_name,
                    shard_path,
                    shard_size,
                    shard_mtime_ns,
                    status,
                    records_read,
                    records_selected,
                    skipped_wrong_language,
                    skipped_bad_pos,
                    entries_ingested,
                    errors,
                    started_at,
                    finished_at
                )
                VALUES (?, ?, ?, ?, 'complete', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    shard_name,
                    str(shard_path),
                    int(size),
                    int(mtime_ns),
                    int(shard_stats.records_read),
                    int(shard_stats.records_selected),
                    int(shard_stats.skipped_wrong_language),
                    int(shard_stats.skipped_bad_pos),
                    int(shard_stats.entries_ingested),
                    int(shard_stats.errors),
                    float(shard_stats.started_at),
                    now_ts(),
                ),
            )

            conn.commit()

        except Exception:
            conn.rollback()
            raise

        self.stats.shards_processed += 1

        print(
            "[DONE] "
            f"{shard_name} "
            f"records={shard_stats.records_read:,} "
            f"selected={shard_stats.records_selected:,} "
            f"entries={shard_stats.entries_ingested:,} "
            f"skipped_lang={shard_stats.skipped_wrong_language:,} "
            f"skipped_pos={shard_stats.skipped_bad_pos:,} "
            f"errors={shard_stats.errors:,} "
            f"rate={shard_stats.rate():.1f} entry/s",
            flush=True,
        )

        return shard_stats

    # ------------------------------------------------------------------
    # TensorTrain aggregate core
    # ------------------------------------------------------------------

    def build_tensor_core(
        self,
        *,
        conn: sqlite3.Connection,
        core_out: Path,
        manifest_out: Path,
        tt_rank: int = 8,
        energy_tol: float = 0.999,
    ) -> Dict[str, Any]:
        rows = conn.execute(
            """
            SELECT pos, SUM(count) AS total_count
            FROM shard_pos_sums
            GROUP BY pos
            ORDER BY pos
            """
        ).fetchall()

        if not rows:
            raise RuntimeError("Cannot build TensorTrain core: no shard_pos_sums rows found.")

        pos_list = [str(r[0]) for r in rows]
        counts = {str(pos): int(count) for pos, count in rows}
        matrix = np.zeros((len(pos_list), self.dim), dtype=np.float32)
        pos_to_row = {pos: i for i, pos in enumerate(pos_list)}

        for pos, _count, blob, vector_dim in conn.execute(
            "SELECT pos, count, sum_vector, vector_dim FROM shard_pos_sums ORDER BY pos"
        ).fetchall():
            pos = str(pos)
            if int(vector_dim) != self.dim:
                raise RuntimeError(f"shard_pos_sums vector_dim mismatch for pos={pos}: {vector_dim} != {self.dim}")
            matrix[pos_to_row[pos]] += blob_to_array(blob, self.dim)

        for pos, row_idx in pos_to_row.items():
            matrix[row_idx] /= np.float32(max(1, counts[pos]))

        output_dims = factorize_into_modes(matrix.shape[0], num_modes=2)
        input_dims = factorize_into_modes(matrix.shape[1], num_modes=2)

        tt = TensorTrain.from_dense(
            matrix,
            output_dims=output_dims,
            input_dims=input_dims,
            max_bond_dim=int(tt_rank),
            dtype=np.float32,
            energy_tol=float(energy_tol),
            device="cpu",
        )

        tt.metadata.update(
            {
                "format": "LK20.TensorLexiconAggregateCore",
                "schema_version": SCHEMA_VERSION,
                "dim": self.dim,
                "seed": self.seed,
                "lang_code": self.lang_code,
                "pos_list": pos_list,
                "pos_counts": counts,
                "source_db": str(self.db_path),
                "matrix_shape": list(matrix.shape),
                "compression": "TensorTrain.from_dense(pos_average_matrix)",
            }
        )

        core_out = Path(core_out)
        core_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_core = core_out.with_name(core_out.stem + ".tmp.npz")
        tt.save_npz(tmp_core, compressed=True)
        os.replace(tmp_core, core_out)

        entry_count = int(conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
        alias_count = int(conn.execute("SELECT COUNT(*) FROM aliases").fetchone()[0])
        relation_count = int(conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0])
        complete_shards = int(conn.execute("SELECT COUNT(*) FROM processed_shards WHERE status='complete'").fetchone()[0])

        manifest = {
            "format": "LK20.TensorLexiconHybridBank",
            "schema_version": SCHEMA_VERSION,
            "created_at": now_ts(),
            "db_path": str(self.db_path),
            "core_path": str(core_out),
            "entry_count": entry_count,
            "alias_count": alias_count,
            "relation_count": relation_count,
            "complete_shards": complete_shards,
            "dim": self.dim,
            "seed": self.seed,
            "lang_code": self.lang_code,
            "allowed_pos": list(self.allowed_pos),
            "vectors_mode": self.vectors_mode,
            "relations_mode": self.relations_mode,
            "aliases_mode": self.aliases_mode,
            "pos_list": pos_list,
            "pos_counts": counts,
            "tensor_train": tt.describe(),
        }

        atomic_write_json(manifest_out, manifest)

        print(
            "[CORE] "
            f"saved={core_out} "
            f"manifest={manifest_out} "
            f"shape={tuple(matrix.shape)} "
            f"tt_params={tt.parameter_count():,}",
            flush=True,
        )
        return manifest

    # ------------------------------------------------------------------
    # Top-level run helpers
    # ------------------------------------------------------------------

    def list_shards(
        self,
        *,
        shards_dir: Path,
        start_shard: Optional[str] = None,
        start_after: Optional[str] = None,
        end_shard: Optional[str] = None,
        max_shards: Optional[int] = None,
    ) -> List[Path]:
        paths = sorted(
            [
                p
                for p in Path(shards_dir).glob("*.jsonl*")
                if p.is_file() and (p.name.endswith(".jsonl") or p.name.endswith(".jsonl.gz"))
            ],
            key=shard_sort_key,
        )

        if start_after:
            start_key = shard_range_tuple(start_after)
            paths = [p for p in paths if shard_range_tuple(p) > start_key]

        if start_shard:
            start_key = shard_range_tuple(start_shard)
            paths = [p for p in paths if shard_range_tuple(p) >= start_key]

        if end_shard:
            end_key = shard_range_tuple(end_shard)
            paths = [p for p in paths if shard_range_tuple(p) <= end_key]

        if max_shards is not None:
            paths = paths[: int(max_shards)]

        return paths

    def run(
        self,
        *,
        shards_dir: Path,
        core_out: Optional[Path] = None,
        manifest_out: Optional[Path] = None,
        max_shards: Optional[int] = None,
        start_shard: Optional[str] = None,
        start_after: Optional[str] = None,
        end_shard: Optional[str] = None,
        progress_every: int = 0,
        force_reprocess: bool = False,
        reset_output: bool = False,
        rebuild_core_only: bool = False,
        no_core: bool = False,
        tt_rank: int = 8,
        energy_tol: float = 0.999,
        dry_run: bool = False,
        maintenance_every: int = 25,
    ) -> None:
        core_out = Path(core_out) if core_out else self.db_path.with_suffix(".core.npz")
        manifest_out = Path(manifest_out) if manifest_out else self.db_path.with_suffix(".manifest.json")

        if reset_output:
            self.remove_output_files(core_out=core_out, manifest_out=manifest_out)

        shards = self.list_shards(
            shards_dir=Path(shards_dir),
            start_shard=start_shard,
            start_after=start_after,
            end_shard=end_shard,
            max_shards=max_shards,
        )

        self.stats.shards_total = len(shards)

        if dry_run:
            print(f"[DRY-RUN] shards_dir={shards_dir}")
            print(f"[DRY-RUN] shards_selected={len(shards)}")
            if shards[:8]:
                print("[DRY-RUN] first shards:")
                for p in shards[:8]:
                    print(f"  {p.name}")
            if len(shards) > 8:
                print("[DRY-RUN] last shards:")
                for p in shards[-8:]:
                    print(f"  {p.name}")
            print(f"[DRY-RUN] out={self.db_path}")
            print(f"[DRY-RUN] core_out={core_out}")
            print(f"[DRY-RUN] manifest_out={manifest_out}")
            return

        conn = self.connect()
        try:
            self.init_db(conn)
            self.cleanup_incomplete_shards(conn)

            if not rebuild_core_only:
                if not shards:
                    print(f"No shards found in {shards_dir}")
                    return

                print(
                    "[START] "
                    f"shards={len(shards)} "
                    f"out={self.db_path} "
                    f"dim={self.dim} "
                    f"seed={self.seed} "
                    f"lang_code={self.lang_code} "
                    f"allowed_pos={','.join(self.allowed_pos)} "
                    f"vectors_mode={self.vectors_mode} "
                    f"relations_mode={self.relations_mode} "
                    f"aliases_mode={self.aliases_mode} "
                    f"journal={self.journal_mode}",
                    flush=True,
                )

                for idx, shard_path in enumerate(shards, start=1):
                    self.ingest_shard(
                        conn=conn,
                        shard_path=shard_path,
                        force_reprocess=force_reprocess,
                        progress_every=progress_every,
                    )

                    if maintenance_every and idx % int(maintenance_every) == 0:
                        self.light_maintenance(conn, idx)

                print(
                    "[INGEST COMPLETE] "
                    f"processed={self.stats.shards_processed:,} "
                    f"skipped={self.stats.shards_skipped:,} "
                    f"records={self.stats.records_read:,} "
                    f"selected={self.stats.records_selected:,} "
                    f"entries={self.stats.entries_ingested:,} "
                    f"skipped_lang={self.stats.skipped_wrong_language:,} "
                    f"skipped_pos={self.stats.skipped_bad_pos:,} "
                    f"errors={self.stats.errors:,} "
                    f"rate={self.stats.rate():.1f} entry/s",
                    flush=True,
                )

            if self.final_indexes:
                self.create_secondary_indexes(conn)

            if not no_core:
                self.build_tensor_core(
                    conn=conn,
                    core_out=core_out,
                    manifest_out=manifest_out,
                    tt_rank=tt_rank,
                    energy_tol=energy_tol,
                )

        finally:
            conn.close()

    def light_maintenance(self, conn: sqlite3.Connection, idx: int) -> None:
        try:
            if self.journal_mode == "WAL" and self.wal_checkpoint_every and idx % self.wal_checkpoint_every == 0:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except Exception:
            pass

    def remove_output_files(self, *, core_out: Optional[Path] = None, manifest_out: Optional[Path] = None) -> None:
        paths = [
            self.db_path,
            Path(str(self.db_path) + "-wal"),
            Path(str(self.db_path) + "-shm"),
            core_out or self.db_path.with_suffix(".core.npz"),
            manifest_out or self.db_path.with_suffix(".manifest.json"),
        ]

        for p in paths:
            try:
                p = Path(p)
                if p.exists():
                    p.unlink()
                    print(f"[RESET] removed {p}")
            except FileNotFoundError:
                pass


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LK20 TensorTrain-backed streaming lexicon ingestor")
    sub = parser.add_subparsers(dest="command")

    run_cmd = sub.add_parser("run", help="Ingest Kaikki raw shards into SQLite and build TensorTrain aggregate core.")
    run_cmd.add_argument("--shards-dir", required=True, help="Directory containing raw .jsonl or .jsonl.gz shards.")
    run_cmd.add_argument("--out", default="lexicon.db", help="Output SQLite database path.")
    run_cmd.add_argument("--core-out", default=None, help="Output TensorTrain core .npz path. Default: OUT with .core.npz suffix.")
    run_cmd.add_argument("--manifest-out", default=None, help="Output manifest .json path. Default: OUT with .manifest.json suffix.")
    run_cmd.add_argument("--dim", type=int, default=128, help="Semantic vector dimension.")
    run_cmd.add_argument("--seed", type=int, default=2027, help="Deterministic hashing seed.")
    run_cmd.add_argument("--lang-code", default="en", help="Wiktextract language code to ingest. Default: en.")
    run_cmd.add_argument("--allowed-pos", default=",".join(DEFAULT_ALLOWED_POS), help="Comma-separated allowed POS set.")
    run_cmd.add_argument("--relation-fields", default=",".join(DEFAULT_RELATION_FIELDS), help="Comma-separated Wiktextract relation fields to keep.")
    run_cmd.add_argument("--relation-limit", type=int, default=32, help="Maximum targets per relation type per sense.")
    run_cmd.add_argument("--batch-size", type=int, default=1000, help="SQLite executemany batch size within each shard transaction.")
    run_cmd.add_argument("--max-shards", type=int, default=None, help="Maximum number of shards to process.")
    run_cmd.add_argument("--start-shard", default=None, help="Start at this shard filename, inclusive.")
    run_cmd.add_argument("--start-after", default=None, help="Start after this shard filename, exclusive.")
    run_cmd.add_argument("--end-shard", default=None, help="End at this shard filename, inclusive.")
    run_cmd.add_argument("--progress-every", type=int, default=0, help="Print within-shard progress every N records. 0 disables.")
    run_cmd.add_argument("--tt-rank", type=int, default=8, help="Maximum TensorTrain bond rank for aggregate core.")
    run_cmd.add_argument("--energy-tol", type=float, default=0.999, help="Energy tolerance for TensorTrain SVD compression.")
    run_cmd.add_argument("--busy-timeout-ms", type=int, default=30000, help="SQLite busy timeout in milliseconds.")
    run_cmd.add_argument("--max-gloss-chars", type=int, default=2000, help="Maximum stored gloss length per sense.")

    run_cmd.add_argument("--vectors-mode", choices=["none", "entry"], default="none", help="Per-entry vector storage mode. Default: none.")
    run_cmd.add_argument("--no-vectors", action="store_true", help="Compatibility alias for --vectors-mode none.")
    run_cmd.add_argument("--store-vectors", action="store_true", help="Compatibility alias for --vectors-mode entry. Not recommended for full Kaikki.")
    run_cmd.add_argument("--relations-mode", choices=["table", "json", "off"], default="table", help="Relation storage mode.")
    run_cmd.add_argument("--aliases-mode", choices=["lemma-only", "clean-forms", "off"], default="clean-forms", help="Alias storage mode.")

    run_cmd.add_argument("--journal-mode", choices=["DELETE", "WAL", "TRUNCATE", "PERSIST", "MEMORY", "OFF"], default="DELETE", help="SQLite journal mode. DELETE is safer for long Windows runs.")
    run_cmd.add_argument("--synchronous", choices=["OFF", "NORMAL", "FULL", "EXTRA"], default="NORMAL", help="SQLite synchronous setting.")
    run_cmd.add_argument("--create-indexes-on-init", action="store_true", help="Create secondary indexes before ingestion. Slower; mainly for compatibility.")
    run_cmd.add_argument("--no-final-indexes", action="store_true", help="Do not create secondary indexes after ingestion.")
    run_cmd.add_argument("--maintenance-every", type=int, default=25, help="Run light DB maintenance every N processed shard attempts.")

    run_cmd.add_argument("--store-categories", action="store_true", help="Store capped categories in metadata. Never used as relations.")
    run_cmd.add_argument("--max-categories", type=int, default=32, help="Maximum categories to store when enabled.")
    run_cmd.add_argument("--force-reprocess", action="store_true", help="Reprocess shards even if marked complete.")
    run_cmd.add_argument("--reset-output", action="store_true", help="Delete output DB/core/manifest before running.")
    run_cmd.add_argument("--rebuild-core-only", action="store_true", help="Do not ingest shards; rebuild TensorTrain core from existing shard_pos_sums.")
    run_cmd.add_argument("--no-core", action="store_true", help="Do not build TensorTrain core after ingestion.")
    run_cmd.add_argument("--dry-run", action="store_true", help="List selected shards and output paths without ingesting.")
    run_cmd.add_argument("--no-purge-bad-aliases", action="store_true", help="Disable startup purge of polluted form aliases from old DBs.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 2

    vectors_mode = str(args.vectors_mode)
    if bool(args.store_vectors):
        vectors_mode = "entry"
    if bool(args.no_vectors):
        vectors_mode = "none"

    ingestor = TensorLexiconIngestor(
        db_path=Path(args.out),
        dim=int(args.dim),
        seed=int(args.seed),
        lang_code=args.lang_code,
        allowed_pos=parse_csv_set(args.allowed_pos),
        relation_fields=parse_csv_set(args.relation_fields),
        relation_limit=int(args.relation_limit),
        batch_size=int(args.batch_size),
        vectors_mode=vectors_mode,
        relations_mode=args.relations_mode,
        aliases_mode=args.aliases_mode,
        store_categories=bool(args.store_categories),
        max_categories=int(args.max_categories),
        max_gloss_chars=int(args.max_gloss_chars),
        busy_timeout_ms=int(args.busy_timeout_ms),
        journal_mode=args.journal_mode,
        synchronous=args.synchronous,
        create_indexes_on_init=bool(args.create_indexes_on_init),
        final_indexes=not bool(args.no_final_indexes),
        purge_bad_aliases=not bool(args.no_purge_bad_aliases),
    )

    ingestor.run(
        shards_dir=Path(args.shards_dir),
        core_out=Path(args.core_out) if args.core_out else None,
        manifest_out=Path(args.manifest_out) if args.manifest_out else None,
        max_shards=args.max_shards,
        start_shard=args.start_shard,
        start_after=args.start_after,
        end_shard=args.end_shard,
        progress_every=int(args.progress_every),
        force_reprocess=bool(args.force_reprocess),
        reset_output=bool(args.reset_output),
        rebuild_core_only=bool(args.rebuild_core_only),
        no_core=bool(args.no_core),
        tt_rank=int(args.tt_rank),
        energy_tol=float(args.energy_tol),
        dry_run=bool(args.dry_run),
        maintenance_every=int(args.maintenance_every),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
