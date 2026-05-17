#!/usr/bin/env python3
# dictionary_lexicon_ingestor.py
"""
LK20 / Akkurat Local AI — Production Dictionary Lexicon Ingestor
=================================================================

Builds an NPZ SemanticAttractorBank-compatible lexical substrate from
moderate-size dictionary resources such as:

- Princeton WordNet data.* files
- Moby word lists
- JSON / JSONL / NDJSON
- CSV / TSV
- TXT word lists
- small LK20 overlay lexicons

This module is intentionally distinct from tensor_lexicon_ingestor.py.
Use this file for NPZ banks and curated/moderate lexical resources.
Use tensor_lexicon_ingestor.py for Kaikki-scale raw Wiktextract ingestion.

Primary outputs
---------------
The generated .npz contains the compatibility keys expected by the local AI
stack:

    labels, lemmas, words, pos, glosses, definitions, sources, sense_ids,
    weights, vectors, relation_matrix, relations_json, entries_json,
    metadata_json, __metadata_json__

Commands
--------
From local_ai:

    python dictionary_lexicon_ingestor.py install-resources --resources-dir resources

    python dictionary_lexicon_ingestor.py ingest ^
      --use-installed ^
      --resources-dir resources ^
      --out semantic_bank_bootstrap_wordnet_moby.npz ^
      --dim 128 ^
      --no-relation-matrix

    python dictionary_lexicon_ingestor.py inspect-bank --bank semantic_bank_bootstrap_wordnet_moby.npz

Runtime requirements
--------------------
- Python 3.10+
- NumPy
- Standard library only otherwise
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Paths / constants
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_RESOURCES_DIR = SCRIPT_DIR / "resources"
DEFAULT_BANK_PATH = SCRIPT_DIR / "semantic_bank.npz"

DEFAULT_DIM = 64
DEFAULT_MAX_RELATION_MATRIX = 4000
DEFAULT_RELATION_LIMIT = 64
DEFAULT_MAX_GLOSS_CHARS = 4000
EPS = 1e-12

POS_ALIASES = {
    "": "",
    "n": "noun",
    "noun": "noun",
    "nn": "noun",
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
    "a": "adjective",
    "s": "adjective",
    "adj": "adjective",
    "adjective": "adjective",
    "r": "adverb",
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
    "interj": "interjection",
    "intj": "interjection",
    "interjection": "interjection",
}

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

WORDNET_DATA_FILES = {
    "data.noun": "noun",
    "data.verb": "verb",
    "data.adj": "adjective",
    "data.adv": "adverb",
}

# Relation fields deliberately exclude categories. Wiktionary categories and many
# dictionary category fields contain maintenance labels, rhyme pages, request
# labels, translation labels, and other non-semantic artifacts.
DEFAULT_RELATION_FIELDS = (
    "synonyms",
    "antonyms",
    "hypernyms",
    "hyponyms",
    "meronyms",
    "holonyms",
    "related",
    "related_terms",
    "see_also",
    "also_see",
    "derived",
    "derivations",
    "coordinate_terms",
)

RESOURCE_MANIFEST = {
    "wordnet_db_3_0": {
        "kind": "tar_gz",
        "url": "https://wordnetcode.princeton.edu/3.0/WNdb-3.0.tar.gz",
        "target": "wordnet",
        "license_note": "Princeton WordNet database. See WordNet license in archive.",
    },
    "moby_word_lists": {
        "kind": "plain_text",
        "url": "https://www.gutenberg.org/files/3201/3201-0.txt",
        "target": "moby/moby_word_lists.txt",
        "license_note": "Project Gutenberg Moby Word Lists. Public domain in the USA per Gutenberg metadata.",
    },
}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class LexiconEntry:
    lemma: str
    pos: str = ""
    gloss: str = ""
    source: str = "input"
    sense_id: str = ""
    weight: float = 1.0
    relations: Dict[str, List[str]] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def key(self) -> Tuple[str, str, str, str]:
        return (
            alias_text(self.lemma),
            normalize_pos(self.pos),
            canonical_text(self.gloss).casefold(),
            canonical_text(self.sense_id).casefold(),
        )


@dataclass
class IngestStats:
    record_count: int = 0
    entry_count: int = 0
    unique_lemma_count: int = 0
    pos_counts: Dict[str, int] = field(default_factory=dict)
    relation_counts: Dict[str, int] = field(default_factory=dict)
    gloss_token_mean: float = 0.0
    gloss_token_max: int = 0
    raw_record_count: int = 0
    empty_lemma_dropped: int = 0
    bad_pos_dropped: int = 0
    duplicate_dropped: int = 0
    warnings_count: int = 0
    sources: Dict[str, int] = field(default_factory=dict)


@dataclass
class BankBuildConfig:
    dim: int = DEFAULT_DIM
    deduplicate: bool = True
    include_relation_matrix: bool = True
    max_relation_matrix: int = DEFAULT_MAX_RELATION_MATRIX
    relation_limit: int = DEFAULT_RELATION_LIMIT
    min_token_len: int = 1
    seed: int = 0
    vector_norm: str = "l2"
    allowed_pos: Tuple[str, ...] = DEFAULT_ALLOWED_POS
    max_entries: int = 0
    max_gloss_chars: int = DEFAULT_MAX_GLOSS_CHARS


# =============================================================================
# Generic helpers
# =============================================================================

_SPACE_RE = re.compile(r"\s+", re.UNICODE)
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]*|\d+(?:\.\d+)?", re.UNICODE)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "real": np.nan_to_num(obj.real).astype(float).tolist(),
                "imag": np.nan_to_num(obj.imag).astype(float).tolist(),
            }
        return np.nan_to_num(obj).astype(float).tolist()
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))
    return str(obj)


def stable_json(obj: Any) -> str:
    return json.dumps(json_safe(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\x00", " ")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def alias_text(x: Any) -> str:
    return canonical_text(x).casefold()


def normalize_pos(pos: Any) -> str:
    p = canonical_text(pos).casefold().replace(".", "").replace("-", "_")
    p = p.replace(" ", "_")
    return POS_ALIASES.get(p, p)


def tokenize(text: str, *, min_len: int = 1, max_tokens: int = 512) -> List[str]:
    text = canonical_text(text).casefold()
    toks = [m.group(0).strip("-'") for m in _TOKEN_RE.finditer(text)]
    toks = [t for t in toks if len(t) >= int(min_len)]
    return toks[: int(max_tokens)]


def parse_csv_set(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = str(value or "").split(",")
    out: List[str] = []
    for item in raw:
        s = canonical_text(item)
        if s:
            out.append(s)
    return tuple(dict.fromkeys(out))


def resolve_path(path: str | Path, *, base: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return ((base or Path.cwd()) / p).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stable_hash_bytes(text: str, seed: int = 0) -> bytes:
    h = hashlib.blake2b(digest_size=16, person=b"LK20Lexicon")
    h.update(str(int(seed)).encode("utf-8", errors="ignore"))
    h.update(b"\0")
    h.update(str(text).encode("utf-8", errors="ignore"))
    return h.digest()


def stable_hash_int(text: str, seed: int = 0) -> int:
    return int.from_bytes(stable_hash_bytes(text, seed=seed)[:8], "little", signed=False)


def open_text_auto(path: Path) -> str:
    """Read a moderate text/gzip file into memory with robust encoding fallback."""
    p = Path(path)
    raw = p.read_bytes()
    if raw.startswith(b"\x1f\x8b"):
        raw = gzip.decompress(raw)

    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def iter_text_lines_auto(path: Path) -> Iterator[str]:
    """Stream text lines from plain or gzip files."""
    p = Path(path)
    if p.suffix.casefold() == ".gz":
        with gzip.open(p, "rt", encoding="utf-8", errors="replace", newline="") as f:
            for line in f:
                yield line
        return

    # For non-gzip files, decode line by line as UTF-8 with replacement. This is
    # safer for larger JSONL/word-list inputs than reading all bytes first.
    with p.open("rt", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            yield line


def first_present(record: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    p = Path(path).resolve()
    ensure_parent(p)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)


def atomic_write_json(path: Path, obj: Any) -> None:
    p = Path(path).resolve()
    ensure_parent(p)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# =============================================================================
# Record normalization
# =============================================================================


def coerce_relation_values(value: Any, *, limit: int = DEFAULT_RELATION_LIMIT) -> List[str]:
    if value is None:
        return []

    out: List[str] = []
    seen: set[str] = set()

    def add(token: Any) -> None:
        nonlocal out
        s = canonical_text(token)
        if not s:
            return
        if len(s) > 200:
            return
        k = alias_text(s)
        if not k or k in seen:
            return
        seen.add(k)
        out.append(s)

    if isinstance(value, str):
        for part in re.split(r"[,;|]", value):
            add(part)
            if len(out) >= int(limit):
                break
        return out

    if isinstance(value, Mapping):
        token = first_present(value, ("lemma", "word", "target", "name", "id", "term", "form"), "")
        if token:
            add(token)
        else:
            for v in value.values():
                for item in coerce_relation_values(v, limit=max(1, int(limit) - len(out))):
                    add(item)
                    if len(out) >= int(limit):
                        break
                if len(out) >= int(limit):
                    break
        return out[: int(limit)]

    if isinstance(value, Iterable):
        for x in value:
            if isinstance(x, Mapping):
                add(first_present(x, ("lemma", "word", "target", "name", "id", "term", "form"), ""))
            else:
                add(x)
            if len(out) >= int(limit):
                break
        return out

    add(value)
    return out


def normalize_relation_name(name: Any) -> str:
    return alias_text(name).replace(" ", "_").replace("-", "_")


def extract_relations_from_record(
    record: Mapping[str, Any],
    *,
    relation_fields: Sequence[str] = DEFAULT_RELATION_FIELDS,
    relation_limit: int = DEFAULT_RELATION_LIMIT,
) -> Dict[str, List[str]]:
    relations: Dict[str, List[str]] = {}

    rel_obj = record.get("relations", {})
    if isinstance(rel_obj, Mapping):
        for k, v in rel_obj.items():
            rel_name = normalize_relation_name(k)
            if rel_name in {"category", "categories"}:
                continue
            vals = coerce_relation_values(v, limit=relation_limit)
            if vals:
                relations[rel_name] = vals[: int(relation_limit)]

    for rel_name_raw in relation_fields:
        rel_name = normalize_relation_name(rel_name_raw)
        if rel_name in {"category", "categories"}:
            continue
        if rel_name_raw in record:
            vals = coerce_relation_values(record.get(rel_name_raw), limit=relation_limit)
            if vals:
                relations[rel_name] = vals[: int(relation_limit)]

    return relations


def normalize_raw_record(
    record: Mapping[str, Any],
    *,
    source: str = "input",
    relation_fields: Sequence[str] = DEFAULT_RELATION_FIELDS,
    relation_limit: int = DEFAULT_RELATION_LIMIT,
    max_gloss_chars: int = DEFAULT_MAX_GLOSS_CHARS,
) -> Dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}

    lemma = first_present(record, ("lemma", "word", "headword", "term", "entry", "name", "label"), "")
    pos = first_present(record, ("pos", "part_of_speech", "lexical_category", "type"), "")
    gloss = first_present(record, ("gloss", "definition", "meaning", "description", "def"), "")

    if isinstance(gloss, list):
        gloss = "; ".join(canonical_text(x) for x in gloss if canonical_text(x))

    gloss = canonical_text(gloss)
    if max_gloss_chars and len(gloss) > int(max_gloss_chars):
        gloss = gloss[: int(max_gloss_chars)].rstrip()

    sense_id = first_present(record, ("sense_id", "sense", "id", "synset", "offset"), "")
    weight = first_present(record, ("weight", "frequency", "score", "prior"), 1.0)

    out = dict(record)
    out["lemma"] = canonical_text(lemma)
    out["pos"] = normalize_pos(pos)
    out["gloss"] = gloss
    out["sense_id"] = canonical_text(sense_id)
    out["source"] = canonical_text(record.get("source", source)) or source
    out["relations"] = extract_relations_from_record(
        record,
        relation_fields=relation_fields,
        relation_limit=relation_limit,
    )

    try:
        out["weight"] = float(weight)
    except Exception:
        out["weight"] = 1.0

    if not math.isfinite(float(out["weight"])):
        out["weight"] = 1.0

    out.setdefault("word", out["lemma"])
    out.setdefault("definition", out["gloss"])
    out.setdefault("part_of_speech", out["pos"])
    return out


def entry_from_record(
    record: Mapping[str, Any],
    *,
    source: str = "input",
    relation_fields: Sequence[str] = DEFAULT_RELATION_FIELDS,
    relation_limit: int = DEFAULT_RELATION_LIMIT,
    max_gloss_chars: int = DEFAULT_MAX_GLOSS_CHARS,
) -> Optional[LexiconEntry]:
    r = normalize_raw_record(
        record,
        source=source,
        relation_fields=relation_fields,
        relation_limit=relation_limit,
        max_gloss_chars=max_gloss_chars,
    )
    lemma = canonical_text(r.get("lemma", ""))
    if not lemma:
        return None

    return LexiconEntry(
        lemma=lemma,
        pos=normalize_pos(r.get("pos", "")),
        gloss=canonical_text(r.get("gloss", "")),
        source=canonical_text(r.get("source", source)) or source,
        sense_id=canonical_text(r.get("sense_id", "")),
        weight=float(r.get("weight", 1.0) or 1.0),
        relations=dict(r.get("relations", {}) or {}),
        raw=dict(r),
    )


# =============================================================================
# Readers
# =============================================================================


def read_json_records(path: Path) -> List[Dict[str, Any]]:
    text = open_text_auto(path)
    data = json.loads(text)

    if isinstance(data, list):
        return [x if isinstance(x, dict) else {"word": str(x)} for x in data]

    if isinstance(data, dict):
        for key in ("entries", "records", "dictionary", "lexicon", "words", "data", "items"):
            val = data.get(key)
            if isinstance(val, list):
                return [x if isinstance(x, dict) else {"word": str(x)} for x in val]

        records: List[Dict[str, Any]] = []
        for k, v in data.items():
            if isinstance(v, Mapping):
                rec = dict(v)
                rec.setdefault("word", k)
                records.append(rec)
            elif isinstance(v, str):
                records.append({"word": k, "definition": v})
            elif isinstance(v, list):
                if all(isinstance(x, str) for x in v):
                    records.append({"word": k, "definition": "; ".join(v)})
                else:
                    for item in v:
                        if isinstance(item, Mapping):
                            rec = dict(item)
                            rec.setdefault("word", k)
                            records.append(rec)
            else:
                records.append({"word": k, "definition": str(v)})
        return records

    raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")


def read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line_no, line in enumerate(iter_text_lines_auto(path), start=1):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        out.append(obj if isinstance(obj, dict) else {"word": str(obj)})
    return out


def read_delimited_records(path: Path, *, delimiter: str) -> List[Dict[str, Any]]:
    text = open_text_auto(path)
    f = io.StringIO(text)

    try:
        has_header = csv.Sniffer().has_header(text[:4096])
    except Exception:
        has_header = True

    if has_header:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [dict(row) for row in reader]

    reader2 = csv.reader(f, delimiter=delimiter)
    records: List[Dict[str, Any]] = []
    for row in reader2:
        if not row:
            continue
        if len(row) == 1:
            records.append({"word": row[0]})
        elif len(row) == 2:
            records.append({"word": row[0], "definition": row[1]})
        else:
            records.append({"word": row[0], "part_of_speech": row[1], "definition": " ".join(row[2:])})
    return records


def read_txt_word_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in iter_text_lines_auto(path):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith(";"):
            continue

        if "\t" in s:
            parts = [p.strip() for p in s.split("\t") if p.strip()]
        else:
            parts = [p.strip() for p in re.split(r"\s{2,}", s) if p.strip()]

        if len(parts) == 1:
            records.append({"word": parts[0], "source": path.name})
        elif len(parts) == 2:
            records.append({"word": parts[0], "definition": parts[1], "source": path.name})
        else:
            records.append(
                {
                    "word": parts[0],
                    "part_of_speech": parts[1],
                    "definition": " ".join(parts[2:]),
                    "source": path.name,
                }
            )
    return records


def looks_like_wordnet_data(path: Path) -> bool:
    try:
        sample: List[str] = []
        for i, line in enumerate(iter_text_lines_auto(path)):
            sample.append(line)
            if i >= 200:
                break
    except Exception:
        return False

    for line in sample:
        s = line.strip()
        if not s or s.startswith("  "):
            continue
        if re.match(r"^\d{8}\s+\d+\s+[nvar]\s+", s):
            return True
    return False


def infer_wordnet_pos_from_name(name: str) -> str:
    lower = name.casefold()
    if "noun" in lower:
        return "noun"
    if "verb" in lower:
        return "verb"
    if "adj" in lower:
        return "adjective"
    if "adv" in lower:
        return "adverb"
    return ""


def wordnet_pointer_name(symbol: str) -> str:
    return {
        "!": "antonyms",
        "@": "hypernyms",
        "@i": "instance_hypernyms",
        "~": "hyponyms",
        "~i": "instance_hyponyms",
        "#m": "member_holonyms",
        "#s": "substance_holonyms",
        "#p": "part_holonyms",
        "%m": "member_meronyms",
        "%s": "substance_meronyms",
        "%p": "part_meronyms",
        "=": "attributes",
        "+": "derivationally_related",
        ";c": "domain_topic",
        "-c": "member_topic",
        ";r": "domain_region",
        "-r": "member_region",
        ";u": "domain_usage",
        "-u": "member_usage",
        "*": "entailments",
        ">": "causes",
        "^": "also_see",
        "$": "verb_group",
        "&": "similar_to",
        "<": "participle_of",
        "\\": "pertainyms",
    }.get(symbol, f"wn_pointer_{symbol}")


def read_wordnet_data_file(path: Path, *, pos: str = "") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pos = normalize_pos(pos)

    for line in iter_text_lines_auto(path):
        s = line.strip()
        if not s or s.startswith("  "):
            continue
        if not re.match(r"^\d{8}\s+\d+\s+[nvar]\s+", s):
            continue

        if "|" in s:
            head, gloss = s.split("|", 1)
            gloss = gloss.strip()
        else:
            head, gloss = s, ""

        parts = head.split()
        if len(parts) < 5:
            continue

        synset_offset = parts[0]
        wn_pos_raw = parts[2]
        wn_pos = normalize_pos(wn_pos_raw) or pos

        try:
            w_cnt = int(parts[3], 16)
        except Exception:
            continue

        idx = 4
        lemmas: List[str] = []
        for _ in range(w_cnt):
            if idx >= len(parts):
                break
            lemma = parts[idx].replace("_", " ")
            lemmas.append(lemma)
            idx += 2

        relations: Dict[str, List[str]] = {}
        if idx < len(parts):
            try:
                p_cnt = int(parts[idx])
                idx += 1
                for _ in range(p_cnt):
                    if idx + 3 >= len(parts):
                        break
                    symbol = parts[idx]
                    target_offset = parts[idx + 1]
                    target_pos = normalize_pos(parts[idx + 2])
                    rel_name = wordnet_pointer_name(symbol)
                    relations.setdefault(rel_name, []).append(f"wn:{target_pos}:{target_offset}")
                    idx += 4
            except Exception:
                pass

        for lemma in lemmas:
            out.append(
                {
                    "word": lemma,
                    "part_of_speech": wn_pos,
                    "definition": gloss,
                    "sense_id": f"wn:{wn_pos}:{synset_offset}:{lemma}",
                    "source": f"wordnet:{path.name}",
                    "relations": relations,
                }
            )

    return out


def read_any_records(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Dictionary input does not exist: {path}")

    name = path.name.casefold()
    suffix = path.suffix.casefold()

    if name in WORDNET_DATA_FILES:
        return read_wordnet_data_file(path, pos=WORDNET_DATA_FILES[name]), warnings

    if suffix == ".json":
        return read_json_records(path), warnings

    if suffix in {".jsonl", ".ndjson", ".gz"}:
        # .gz is treated as JSONL only when its inner content is line-oriented.
        # For full arbitrary compressed JSON files, decompress externally or use
        # a .json extension before gzip support is needed.
        return read_jsonl_records(path), warnings

    if suffix == ".csv":
        return read_delimited_records(path, delimiter=","), warnings

    if suffix == ".tsv":
        return read_delimited_records(path, delimiter="\t"), warnings

    if suffix in {".txt", ".dic", ".words", ".lst"}:
        if looks_like_wordnet_data(path):
            return read_wordnet_data_file(path, pos=infer_wordnet_pos_from_name(path.name)), warnings
        return read_txt_word_records(path), warnings

    text = open_text_auto(path)
    stripped = text.lstrip()

    if stripped.startswith("[") or stripped.startswith("{"):
        return read_json_records(path), warnings

    if "\t" in text[:4096]:
        return read_delimited_records(path, delimiter="\t"), warnings

    if "," in text[:4096]:
        try:
            return read_delimited_records(path, delimiter=","), warnings
        except Exception as exc:
            warnings.append(f"CSV fallback failed: {exc!r}")

    return read_txt_word_records(path), warnings


# =============================================================================
# Resource installer
# =============================================================================


def download_bytes(url: str, *, timeout: int = 60) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LK20-LocalAI-LexiconIngestor/2.0"},
    )
    with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
        return resp.read()


def safe_extract_tar(tf: tarfile.TarFile, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    for member in tf.getmembers():
        member_path = (target_dir / member.name).resolve()
        if not str(member_path).startswith(str(target_dir)):
            raise RuntimeError(f"Unsafe tar path in archive: {member.name}")
    tf.extractall(target_dir)


def install_plain_text_resource(url: str, target: Path, *, timeout: int = 60) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    data = download_bytes(url, timeout=timeout)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def install_tar_gz_resource(url: str, target_dir: Path, *, timeout: int = 60) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    data = download_bytes(url, timeout=timeout)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        safe_extract_tar(tf, target_dir)


def resource_installed(resources_dir: Path, resource_name: str) -> bool:
    spec = RESOURCE_MANIFEST[resource_name]
    target = resources_dir / spec["target"]

    if resource_name == "wordnet_db_3_0":
        if not target.exists():
            return False
        return any(p.name.casefold() in WORDNET_DATA_FILES for p in target.rglob("*") if p.is_file())

    return target.exists()


def install_resources(
    resources_dir: Path = DEFAULT_RESOURCES_DIR,
    *,
    force: bool = False,
    timeout: int = 60,
    quiet: bool = False,
) -> Dict[str, Any]:
    resources_dir = Path(resources_dir).resolve()
    resources_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "resources_dir": str(resources_dir),
        "installed": {},
        "errors": {},
        "timestamp": now_iso(),
        "resources": {},
    }

    for name, spec in RESOURCE_MANIFEST.items():
        result["resources"][name] = {
            "url": spec["url"],
            "target": str(resources_dir / spec["target"]),
            "license_note": spec.get("license_note", ""),
        }

        try:
            if resource_installed(resources_dir, name) and not force:
                result["installed"][name] = "already_present"
                if not quiet:
                    print(f"[OK] {name}: already present")
                continue

            if not quiet:
                print(f"[GET] {name}: {spec['url']}")

            target = resources_dir / spec["target"]
            if spec["kind"] == "tar_gz":
                install_tar_gz_resource(spec["url"], target, timeout=timeout)
            elif spec["kind"] == "plain_text":
                install_plain_text_resource(spec["url"], target, timeout=timeout)
            else:
                raise ValueError(f"Unknown resource kind: {spec['kind']}")

            result["installed"][name] = "installed"
            if not quiet:
                print(f"[OK] {name}: installed")

        except Exception as exc:
            result["errors"][name] = repr(exc)
            if not quiet:
                print(f"[WARN] {name}: {exc}")

    manifest_path = resources_dir / "resource_manifest.json"
    atomic_write_json(manifest_path, result)
    return result


def discover_installed_resource_inputs(resources_dir: Path = DEFAULT_RESOURCES_DIR) -> List[Path]:
    resources_dir = Path(resources_dir).resolve()
    if not resources_dir.exists():
        return []

    paths: List[Path] = []

    for p in resources_dir.rglob("*"):
        if p.is_file() and p.name.casefold() in WORDNET_DATA_FILES:
            paths.append(p)

    for p in resources_dir.rglob("*.txt"):
        if "moby" in str(p).casefold():
            paths.append(p)

    return sorted(set(paths), key=lambda x: str(x).casefold())


# =============================================================================
# Vectorization
# =============================================================================


def char_ngrams(text: str, n: int) -> Iterator[str]:
    s = f"^{text}$"
    if len(s) < int(n):
        return
    for i in range(0, len(s) - int(n) + 1):
        yield s[i : i + int(n)]


def add_feature(v: np.ndarray, token: str, *, weight: float, seed: int = 0) -> None:
    dim = int(v.shape[0])
    h = stable_hash_int(token, seed=seed)

    idx1 = h % dim
    idx2 = (h >> 16) % dim

    sign1 = 1.0 if ((h >> 32) & 1) == 0 else -1.0
    sign2 = 1.0 if ((h >> 33) & 1) == 0 else -1.0

    v[idx1] += float(weight) * sign1
    v[idx2] += float(weight) * 0.5 * sign2


def vectorize_entry(
    entry: LexiconEntry,
    *,
    dim: int = DEFAULT_DIM,
    seed: int = 0,
    min_token_len: int = 1,
    vector_norm: str = "l2",
) -> np.ndarray:
    dim = int(max(4, dim))
    v = np.zeros((dim,), dtype=np.float64)

    lemma = alias_text(entry.lemma)
    pos = normalize_pos(entry.pos)
    gloss = canonical_text(entry.gloss).casefold()

    add_feature(v, f"lemma:{lemma}", weight=3.0, seed=seed)
    add_feature(v, f"pos:{pos}", weight=0.75, seed=seed)
    add_feature(v, f"lemma_pos:{lemma}:{pos}", weight=0.75, seed=seed)
    add_feature(v, f"source:{entry.source}", weight=0.15, seed=seed)

    compact = re.sub(r"[^a-z0-9]+", "_", lemma).strip("_")
    for n in (2, 3, 4):
        for gram in char_ngrams(compact, n):
            add_feature(v, f"char{n}:{gram}", weight=0.8 / n, seed=seed)

    tokens = tokenize(gloss, min_len=min_token_len, max_tokens=192)
    if tokens:
        tok_weight = 0.45 / max(1.0, float(len(tokens)) ** 0.35)
        for tok in tokens:
            add_feature(v, f"gloss_tok:{tok}", weight=tok_weight, seed=seed)
            if pos:
                add_feature(v, f"pos_gloss_tok:{pos}:{tok}", weight=0.15 * tok_weight, seed=seed)

    for a, b in zip(tokens, tokens[1:]):
        add_feature(v, f"gloss_bigram:{a}_{b}", weight=0.18, seed=seed)

    for rel_name, vals in sorted(entry.relations.items()):
        rn = normalize_relation_name(rel_name)
        add_feature(v, f"relname:{rn}", weight=0.35, seed=seed)
        for target in vals[:16]:
            add_feature(v, f"reltarget:{rn}:{alias_text(target)}", weight=0.20, seed=seed)

    if entry.weight and math.isfinite(float(entry.weight)):
        add_feature(v, "entry_weight", weight=0.05 * float(entry.weight), seed=seed)

    nrm = float(np.linalg.norm(v))
    if nrm <= EPS or not np.isfinite(nrm):
        for i in range(dim):
            raw = stable_hash_int(f"{lemma}:{i}", seed=seed)
            v[i] = ((raw % 20001) / 10000.0) - 1.0
        nrm = float(np.linalg.norm(v))

    norm_name = canonical_text(vector_norm).casefold()
    if norm_name in {"l2", "unit", "normalize"} and nrm > EPS:
        v = v / nrm
    elif norm_name in {"none", "raw"}:
        pass
    else:
        if nrm > EPS:
            v = v / nrm

    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def relation_weight(rel_name: str) -> float:
    r = rel_name.casefold()
    if "synonym" in r or "similar" in r:
        return 0.95
    if "hypernym" in r or "hyponym" in r:
        return 0.75
    if "antonym" in r:
        return -0.65
    if "related" in r or "also" in r:
        return 0.60
    return 0.45


def build_relation_matrix(entries: Sequence[LexiconEntry], vectors: np.ndarray, *, max_n: int) -> np.ndarray:
    n = len(entries)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n > int(max_n):
        return np.zeros((0, 0), dtype=np.float32)

    V = np.asarray(vectors, dtype=np.float32)
    sim = V @ V.T
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    lemma_to_indices: Dict[str, List[int]] = {}
    for i, e in enumerate(entries):
        lemma_to_indices.setdefault(alias_text(e.lemma), []).append(i)

    for i, e in enumerate(entries):
        for rel_name, targets in e.relations.items():
            w = relation_weight(rel_name)
            for target in targets:
                key = alias_text(target)
                if key.startswith("wn:"):
                    continue
                for j in lemma_to_indices.get(key, []):
                    if abs(w) > abs(float(sim[i, j])):
                        sim[i, j] = w
                    if abs(w) > abs(float(sim[j, i])):
                        sim[j, i] = w

    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)


# =============================================================================
# Ingestion pipeline
# =============================================================================


def is_probably_boilerplate(text: str) -> bool:
    s = canonical_text(text).casefold()
    if not s:
        return True

    bad_prefixes = (
        "project gutenberg",
        "the project gutenberg",
        "ebook",
        "release date",
        "language:",
        "copyright",
        "produced by",
        "end of the project gutenberg",
        "*** start",
        "*** end",
    )
    if any(s.startswith(x) for x in bad_prefixes):
        return True

    if len(s) > 80 and not re.match(r"^[a-zA-Z][a-zA-Z'\- ]+$", s):
        return True

    return False


def load_entries_from_paths(
    paths: Sequence[Path],
    *,
    source_label: Optional[str] = None,
    allowed_pos: Sequence[str] = DEFAULT_ALLOWED_POS,
    relation_fields: Sequence[str] = DEFAULT_RELATION_FIELDS,
    relation_limit: int = DEFAULT_RELATION_LIMIT,
    max_entries: int = 0,
    max_gloss_chars: int = DEFAULT_MAX_GLOSS_CHARS,
) -> Tuple[List[LexiconEntry], List[str], int, int, int]:
    entries: List[LexiconEntry] = []
    warnings: List[str] = []
    raw_count = 0
    empty_dropped = 0
    bad_pos_dropped = 0
    allowed = {normalize_pos(p) for p in allowed_pos if normalize_pos(p)}

    for path in paths:
        p = Path(path).resolve()

        try:
            records, read_warnings = read_any_records(p)
            warnings.extend([f"{p.name}: {w}" for w in read_warnings])
        except Exception as exc:
            warnings.append(f"{p}: read failed: {exc!r}")
            continue

        raw_count += len(records)
        src = source_label or p.name

        for rec in records:
            ent = entry_from_record(
                rec,
                source=src,
                relation_fields=relation_fields,
                relation_limit=relation_limit,
                max_gloss_chars=max_gloss_chars,
            )
            if ent is None:
                empty_dropped += 1
                continue
            if is_probably_boilerplate(ent.lemma):
                empty_dropped += 1
                continue
            if allowed and normalize_pos(ent.pos) not in allowed:
                bad_pos_dropped += 1
                continue

            entries.append(ent)
            if max_entries and len(entries) >= int(max_entries):
                return entries, warnings, raw_count, empty_dropped, bad_pos_dropped

    return entries, warnings, raw_count, empty_dropped, bad_pos_dropped


def deduplicate_entries(entries: Sequence[LexiconEntry]) -> Tuple[List[LexiconEntry], int]:
    seen: set[Tuple[str, str, str, str]] = set()
    out: List[LexiconEntry] = []
    dropped = 0

    for e in entries:
        key = e.key()
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(e)

    return out, dropped


def compute_stats(
    entries: Sequence[LexiconEntry],
    *,
    raw_record_count: int = 0,
    empty_lemma_dropped: int = 0,
    bad_pos_dropped: int = 0,
    duplicate_dropped: int = 0,
    warnings_count: int = 0,
) -> IngestStats:
    pos_counts: Dict[str, int] = {}
    relation_counts: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    gloss_token_counts: List[int] = []

    for e in entries:
        pos = normalize_pos(e.pos)
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
        sources[e.source] = sources.get(e.source, 0) + 1
        gloss_token_counts.append(len(tokenize(e.gloss)))

        for rel, vals in e.relations.items():
            relation_counts[rel] = relation_counts.get(rel, 0) + len(vals)

    unique_lemmas = len({alias_text(e.lemma) for e in entries})

    return IngestStats(
        record_count=len(entries),
        entry_count=len(entries),
        unique_lemma_count=unique_lemmas,
        pos_counts=dict(sorted(pos_counts.items())),
        relation_counts=dict(sorted(relation_counts.items())),
        gloss_token_mean=float(np.mean(gloss_token_counts)) if gloss_token_counts else 0.0,
        gloss_token_max=int(max(gloss_token_counts)) if gloss_token_counts else 0,
        raw_record_count=int(raw_record_count),
        empty_lemma_dropped=int(empty_lemma_dropped),
        bad_pos_dropped=int(bad_pos_dropped),
        duplicate_dropped=int(duplicate_dropped),
        warnings_count=int(warnings_count),
        sources=dict(sorted(sources.items())),
    )


def build_semantic_bank(
    entries: Sequence[LexiconEntry],
    *,
    out: Path = DEFAULT_BANK_PATH,
    config: BankBuildConfig = BankBuildConfig(),
    raw_record_count: int = 0,
    empty_lemma_dropped: int = 0,
    bad_pos_dropped: int = 0,
    duplicate_dropped: int = 0,
    warnings: Optional[List[str]] = None,
    manifest_out: Optional[Path] = None,
) -> Dict[str, Any]:
    out = Path(out).resolve()
    ensure_parent(out)

    warnings = warnings or []
    dim = int(max(4, config.dim))
    entries_list = list(entries)

    if config.deduplicate:
        entries_list, dropped = deduplicate_entries(entries_list)
        duplicate_dropped += dropped

    if config.max_entries and len(entries_list) > int(config.max_entries):
        entries_list = entries_list[: int(config.max_entries)]

    vectors = np.zeros((len(entries_list), dim), dtype=np.float32)
    for i, e in enumerate(entries_list):
        vectors[i] = vectorize_entry(
            e,
            dim=dim,
            seed=int(config.seed),
            min_token_len=int(config.min_token_len),
            vector_norm=str(config.vector_norm),
        )

    relation_matrix = np.zeros((0, 0), dtype=np.float32)
    if config.include_relation_matrix:
        relation_matrix = build_relation_matrix(entries_list, vectors, max_n=int(config.max_relation_matrix))

    labels = np.asarray([str(e.lemma) for e in entries_list], dtype=np.str_)
    pos = np.asarray([str(normalize_pos(e.pos)) for e in entries_list], dtype=np.str_)
    glosses = np.asarray([str(e.gloss) for e in entries_list], dtype=np.str_)
    sources = np.asarray([str(e.source) for e in entries_list], dtype=np.str_)
    sense_ids = np.asarray([str(e.sense_id) for e in entries_list], dtype=np.str_)
    weights = np.asarray([float(e.weight) for e in entries_list], dtype=np.float32)

    relations_json = np.asarray(
        [stable_json(e.relations) for e in entries_list],
        dtype=np.str_,
    )

    entries_json = np.asarray(
        [
            stable_json(
                {
                    "lemma": e.lemma,
                    "word": e.lemma,
                    "label": e.lemma,
                    "pos": normalize_pos(e.pos),
                    "part_of_speech": normalize_pos(e.pos),
                    "gloss": e.gloss,
                    "definition": e.gloss,
                    "source": e.source,
                    "sense_id": e.sense_id,
                    "weight": float(e.weight),
                    "relations": e.relations,
                }
            )
            for e in entries_list
        ],
        dtype=np.str_,
    )

    stats = compute_stats(
        entries_list,
        raw_record_count=raw_record_count,
        empty_lemma_dropped=empty_lemma_dropped,
        bad_pos_dropped=bad_pos_dropped,
        duplicate_dropped=duplicate_dropped,
        warnings_count=len(warnings),
    )

    metadata = {
        "format": "LK20.SemanticBank",
        "version": 3,
        "created_at": now_iso(),
        "dim": dim,
        "entry_count": len(entries_list),
        "vectorizer": "deterministic_hash_lemma_gloss_relation_v3",
        # Keep deliberately empty for compatibility with loaders that pass this
        # object into SemanticAttractorConfig(**metadata['config']).
        "config": {},
        "ingestor_config": json_safe(asdict(config)),
        "stats": json_safe(asdict(stats)),
        "warnings": warnings[:200],
    }

    metadata_json = np.asarray(stable_json(metadata), dtype=np.str_)

    atomic_save_npz(
        out,
        labels=labels,
        lemmas=labels,
        words=labels,
        pos=pos,
        glosses=glosses,
        definitions=glosses,
        sources=sources,
        sense_ids=sense_ids,
        weights=weights,
        vectors=np.asarray(vectors, dtype=np.float32),
        relation_matrix=np.asarray(relation_matrix, dtype=np.float32),
        relations_json=relations_json,
        entries_json=entries_json,
        metadata_json=metadata_json,
        __metadata_json__=metadata_json,
    )

    health = inspect_bank(out)
    result = {
        "stats": asdict(stats),
        "bank_health": health,
        "out": str(out),
        "warnings": warnings[:20],
    }

    if manifest_out is not None:
        atomic_write_json(Path(manifest_out), result)

    return result


def ingest(
    input_path: Optional[Path] = None,
    out: Path = DEFAULT_BANK_PATH,
    *,
    use_installed: bool = False,
    install_missing: bool = False,
    resources_dir: Path = DEFAULT_RESOURCES_DIR,
    dim: int = DEFAULT_DIM,
    max_relation_matrix: int = DEFAULT_MAX_RELATION_MATRIX,
    relation_limit: int = DEFAULT_RELATION_LIMIT,
    deduplicate: bool = True,
    include_relation_matrix: bool = True,
    seed: int = 0,
    allowed_pos: Sequence[str] = DEFAULT_ALLOWED_POS,
    max_entries: int = 0,
    max_gloss_chars: int = DEFAULT_MAX_GLOSS_CHARS,
    source_label: Optional[str] = None,
    manifest_out: Optional[Path] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    warnings: List[str] = []
    paths: List[Path] = []

    if input_path is not None:
        paths.append(Path(input_path).resolve())

    resources_dir = Path(resources_dir).resolve()

    if install_missing:
        install_result = install_resources(resources_dir, force=False, quiet=quiet)
        if install_result.get("errors"):
            warnings.append(f"Resource installer warnings: {install_result['errors']}")

    if use_installed:
        resource_paths = discover_installed_resource_inputs(resources_dir)
        paths.extend(resource_paths)
        if not resource_paths:
            warnings.append(f"No installed resource inputs found under {resources_dir}")

    if not paths:
        raise ValueError("No input paths supplied. Use --input, --use-installed, or --install-missing --use-installed.")

    entries, read_warnings, raw_count, empty_dropped, bad_pos_dropped = load_entries_from_paths(
        paths,
        source_label=source_label,
        allowed_pos=allowed_pos,
        relation_fields=DEFAULT_RELATION_FIELDS,
        relation_limit=relation_limit,
        max_entries=max_entries,
        max_gloss_chars=max_gloss_chars,
    )
    warnings.extend(read_warnings)

    cfg = BankBuildConfig(
        dim=int(dim),
        deduplicate=bool(deduplicate),
        include_relation_matrix=bool(include_relation_matrix),
        max_relation_matrix=int(max_relation_matrix),
        relation_limit=int(relation_limit),
        seed=int(seed),
        allowed_pos=tuple(normalize_pos(p) for p in allowed_pos if normalize_pos(p)),
        max_entries=int(max_entries),
        max_gloss_chars=int(max_gloss_chars),
    )

    if not quiet:
        print("Ingesting lexicon")
        print(f"  inputs: {len(paths)}")
        for p in paths[:16]:
            print(f"    - {p}")
        if len(paths) > 16:
            print(f"    ... {len(paths) - 16} more")
        print(f"  entries loaded: {len(entries):,}")
        print(f"  out:            {Path(out).resolve()}")

    result = build_semantic_bank(
        entries,
        out=Path(out).resolve(),
        config=cfg,
        raw_record_count=raw_count,
        empty_lemma_dropped=empty_dropped,
        bad_pos_dropped=bad_pos_dropped,
        warnings=warnings,
        manifest_out=manifest_out,
    )

    if not quiet:
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
        print(f"\nSemantic bank written: {Path(out).resolve()}")

    return result


# =============================================================================
# Inspection
# =============================================================================


def inspect_bank(path: Path) -> Dict[str, Any]:
    p = Path(path).resolve()
    if not p.exists():
        return {
            "kind": "SemanticAttractorBank",
            "exists": False,
            "path": str(p),
            "is_stable": False,
        }

    with np.load(p, allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32) if "vectors" in data else np.zeros((0, DEFAULT_DIM), dtype=np.float32)

        if "labels" in data:
            labels = data["labels"]
        elif "lemmas" in data:
            labels = data["lemmas"]
        elif "words" in data:
            labels = data["words"]
        else:
            labels = np.asarray([], dtype=np.str_)

        relation = np.asarray(data["relation_matrix"], dtype=np.float32) if "relation_matrix" in data else np.zeros((0, 0), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1) if vectors.ndim == 2 and vectors.shape[0] else np.zeros((0,), dtype=np.float32)

        metadata = {}
        meta_key = "__metadata_json__" if "__metadata_json__" in data else "metadata_json" if "metadata_json" in data else None
        if meta_key:
            try:
                metadata = json.loads(str(data[meta_key].item()))
            except Exception:
                metadata = {}

        finite = bool(np.all(np.isfinite(vectors))) and bool(np.all(np.isfinite(relation)))
        dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
        entry_count = int(vectors.shape[0]) if vectors.ndim == 2 else int(len(labels))
        all_dims_match = bool(vectors.ndim == 2 and len(labels) == entry_count)

        return {
            "kind": "SemanticAttractorBank",
            "path": str(p),
            "exists": True,
            "entry_count": entry_count,
            "dim": dim,
            "all_dims_match": all_dims_match,
            "finite": finite,
            "vector_norm_min": float(np.min(norms)) if norms.size else 0.0,
            "vector_norm_mean": float(np.mean(norms)) if norms.size else 0.0,
            "vector_norm_max": float(np.max(norms)) if norms.size else 0.0,
            "has_relation_matrix": bool(relation.size > 0),
            "relation_shape": list(map(int, relation.shape)),
            "metadata_version": metadata.get("version"),
            "metadata_stats": metadata.get("stats", {}),
            "is_stable": bool(finite and all_dims_match and vectors.ndim == 2),
        }


# =============================================================================
# Backward-compatible public aliases
# =============================================================================


def ingest_lexicon(
    input_path: str | Path,
    out_path: str | Path = DEFAULT_BANK_PATH,
    **kwargs: Any,
) -> Dict[str, Any]:
    return ingest(input_path=Path(input_path), out=Path(out_path), **kwargs)


def ingest_dictionary(
    input_path: str | Path,
    out_path: str | Path = DEFAULT_BANK_PATH,
    **kwargs: Any,
) -> Dict[str, Any]:
    return ingest_lexicon(input_path, out_path, **kwargs)


def build_bank(
    input_path: str | Path,
    out_path: str | Path = DEFAULT_BANK_PATH,
    **kwargs: Any,
) -> Dict[str, Any]:
    return ingest_lexicon(input_path, out_path, **kwargs)


def build_from_records(
    records: Sequence[Mapping[str, Any]],
    out_path: str | Path = DEFAULT_BANK_PATH,
    *,
    dim: int = DEFAULT_DIM,
    seed: int = 0,
) -> Dict[str, Any]:
    entries: List[LexiconEntry] = []
    empty = 0

    for rec in records:
        ent = entry_from_record(rec, source="records")
        if ent is None:
            empty += 1
        else:
            entries.append(ent)

    return build_semantic_bank(
        entries,
        out=Path(out_path),
        config=BankBuildConfig(dim=dim, seed=seed),
        raw_record_count=len(records),
        empty_lemma_dropped=empty,
    )


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dictionary_lexicon_ingestor.py",
        description="Ingest dictionary files into LK20 semantic_bank.npz.",
    )
    sub = parser.add_subparsers(dest="command")

    p_ingest = sub.add_parser("ingest", help="Ingest dictionary input into a semantic bank.")
    p_ingest.add_argument("--input", "-i", default=None, help="Input path: JSON, JSONL, CSV, TSV, TXT, or WordNet data.*.")
    p_ingest.add_argument("--out", "-o", default=str(DEFAULT_BANK_PATH), help="Output .npz semantic bank path.")
    p_ingest.add_argument("--manifest-out", default=None, help="Optional JSON manifest/report path.")
    p_ingest.add_argument("--use-installed", action="store_true", help="Also ingest installed resources from resources directory.")
    p_ingest.add_argument("--install-missing", action="store_true", help="Install open/public resources if not already present before ingest.")
    p_ingest.add_argument("--resources-dir", default=str(DEFAULT_RESOURCES_DIR), help="Resource directory.")
    p_ingest.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Semantic vector dimension.")
    p_ingest.add_argument("--max-relation-matrix", type=int, default=DEFAULT_MAX_RELATION_MATRIX, help="Maximum entries for dense relation matrix.")
    p_ingest.add_argument("--relation-limit", type=int, default=DEFAULT_RELATION_LIMIT, help="Maximum relation targets retained per relation type per entry.")
    p_ingest.add_argument("--allowed-pos", default=",".join(DEFAULT_ALLOWED_POS), help="Comma-separated allowed POS list.")
    p_ingest.add_argument("--max-entries", type=int, default=0, help="Maximum entries to retain. 0 means unlimited.")
    p_ingest.add_argument("--max-gloss-chars", type=int, default=DEFAULT_MAX_GLOSS_CHARS, help="Maximum stored gloss length per entry.")
    p_ingest.add_argument("--source-label", default=None, help="Override source label for explicit --input records.")
    p_ingest.add_argument("--no-deduplicate", action="store_true", help="Do not deduplicate entries.")
    p_ingest.add_argument("--no-relation-matrix", action="store_true", help="Do not build dense relation matrix.")
    p_ingest.add_argument("--seed", type=int, default=0, help="Deterministic vector seed.")
    p_ingest.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    p_install = sub.add_parser("install-resources", help="Install optional open/public lexical resources locally.")
    p_install.add_argument("--resources-dir", default=str(DEFAULT_RESOURCES_DIR), help="Resource directory.")
    p_install.add_argument("--force", action="store_true", help="Redownload/reinstall resources.")
    p_install.add_argument("--timeout", type=int, default=60, help="Download timeout seconds.")
    p_install.add_argument("--quiet", action="store_true", help="Suppress progress output.")

    p_inspect = sub.add_parser("inspect-bank", help="Inspect a semantic bank .npz.")
    p_inspect.add_argument("--bank", default=str(DEFAULT_BANK_PATH), help="Semantic bank path.")

    return parser


def cmd_ingest(args: argparse.Namespace) -> int:
    input_path = None
    if getattr(args, "input", None):
        input_path = resolve_path(args.input, base=Path.cwd())

    out_path = resolve_path(getattr(args, "out", str(DEFAULT_BANK_PATH)), base=Path.cwd())
    resources_dir = resolve_path(getattr(args, "resources_dir", str(DEFAULT_RESOURCES_DIR)), base=Path.cwd())
    manifest_out = None
    if getattr(args, "manifest_out", None):
        manifest_out = resolve_path(args.manifest_out, base=Path.cwd())

    try:
        ingest(
            input_path=input_path,
            out=out_path,
            use_installed=bool(getattr(args, "use_installed", False)),
            install_missing=bool(getattr(args, "install_missing", False)),
            resources_dir=resources_dir,
            dim=int(getattr(args, "dim", DEFAULT_DIM)),
            max_relation_matrix=int(getattr(args, "max_relation_matrix", DEFAULT_MAX_RELATION_MATRIX)),
            relation_limit=int(getattr(args, "relation_limit", DEFAULT_RELATION_LIMIT)),
            deduplicate=not bool(getattr(args, "no_deduplicate", False)),
            include_relation_matrix=not bool(getattr(args, "no_relation_matrix", False)),
            seed=int(getattr(args, "seed", 0)),
            allowed_pos=parse_csv_set(getattr(args, "allowed_pos", ",".join(DEFAULT_ALLOWED_POS))),
            max_entries=int(getattr(args, "max_entries", 0)),
            max_gloss_chars=int(getattr(args, "max_gloss_chars", DEFAULT_MAX_GLOSS_CHARS)),
            source_label=getattr(args, "source_label", None),
            manifest_out=manifest_out,
            quiet=bool(getattr(args, "quiet", False)),
        )
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def cmd_install_resources(args: argparse.Namespace) -> int:
    resources_dir = resolve_path(getattr(args, "resources_dir", str(DEFAULT_RESOURCES_DIR)), base=Path.cwd())

    try:
        result = install_resources(
            resources_dir=resources_dir,
            force=bool(getattr(args, "force", False)),
            timeout=int(getattr(args, "timeout", 60)),
            quiet=bool(getattr(args, "quiet", False)),
        )
        if not bool(getattr(args, "quiet", False)):
            print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
        return 0 if not result.get("errors") else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def cmd_inspect_bank(args: argparse.Namespace) -> int:
    bank_path = resolve_path(getattr(args, "bank", str(DEFAULT_BANK_PATH)), base=Path.cwd())

    try:
        result = inspect_bank(bank_path)
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
        return 0 if result.get("is_stable") else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_arg_parser()

    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(list(argv))

    if args.command == "ingest":
        return cmd_ingest(args)

    if args.command == "install-resources":
        return cmd_install_resources(args)

    if args.command == "inspect-bank":
        return cmd_inspect_bank(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LexiconEntry",
    "IngestStats",
    "BankBuildConfig",
    "normalize_raw_record",
    "entry_from_record",
    "read_any_records",
    "install_resources",
    "discover_installed_resource_inputs",
    "build_semantic_bank",
    "ingest",
    "ingest_lexicon",
    "ingest_dictionary",
    "build_bank",
    "build_from_records",
    "inspect_bank",
    "main",
]
