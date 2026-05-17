#!/usr/bin/env python3
# convert_kaikki_wiktionary_to_lk20_jsonl.py
"""
LK20 / Akkurat Local AI — Kaikki/Wiktextract Converter
======================================================

Converts Kaikki/Wiktextract JSONL into the flat LK20 lexicon JSONL format
accepted by dictionary_lexicon_ingestor.py.

Input
-----
Kaikki raw Wiktextract JSONL or JSONL.GZ, e.g.:

    resources/kaikki/raw-wiktextract-data.jsonl.gz

Output
------
One flat JSON object per sense:

    {
      "word": "cat",
      "lemma": "cat",
      "part_of_speech": "noun",
      "definition": "...",
      "gloss": "...",
      "sense_id": "kaikki:...",
      "source": "kaikki:wiktextract",
      "relations": {
        "synonyms": [...],
        "antonyms": [...],
        "hypernyms": [...]
      },
      "metadata": {...},
      "raw": {...}
    }

Important design choices
------------------------
- Wiktionary categories are NOT inserted into semantic relations.
  They are often maintenance labels, translation labels, rhyme labels,
  page labels, or request labels. They are preserved only in raw metadata.

- Relation lists are capped per relation type to prevent a single entry
  from dominating the vector space.

- POS labels are normalized before output so the ingestor receives stable
  categories.

Typical command
---------------
From C:\\Users\\ali_z\\ANU AI\\LK20\\local_ai:

    python convert_kaikki_wiktionary_to_lk20_jsonl.py ^
      --input resources\\kaikki\\raw-wiktextract-data.jsonl.gz ^
      --out resources\\normalized\\kaikki_english_lk20_smoke_v2.jsonl ^
      --lang-code en ^
      --max-entries 1000 ^
      --allowed-pos noun,verb,adjective,adverb,preposition,pronoun,determiner,conjunction,interjection

Then:

    python dictionary_lexicon_ingestor.py ingest ^
      --input resources\\normalized\\kaikki_english_lk20_smoke_v2.jsonl ^
      --out semantic_bank_kaikki_smoke_v2.npz ^
      --dim 128 ^
      --no-relation-matrix
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


# =============================================================================
# Paths / constants
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "resources" / "kaikki" / "raw-wiktextract-data.jsonl.gz"
DEFAULT_OUT = SCRIPT_DIR / "resources" / "normalized" / "kaikki_english_lk20.jsonl"

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")


# POS values normalized to the vocabulary expected by dictionary_lexicon_ingestor.py.
POS_ALIASES = {
    "": "",

    # Nouns
    "n": "noun",
    "noun": "noun",
    "nouns": "noun",
    "proper noun": "noun",
    "proper name": "noun",
    "name": "noun",
    "num": "noun",
    "number": "noun",
    "numeral": "noun",

    # Verbs
    "v": "verb",
    "verb": "verb",
    "verbs": "verb",

    # Adjectives
    "a": "adjective",
    "adj": "adjective",
    "adjective": "adjective",
    "adjectives": "adjective",

    # Adverbs
    "adv": "adverb",
    "adverb": "adverb",
    "adverbs": "adverb",

    # Function words
    "prep": "preposition",
    "preposition": "preposition",
    "postposition": "preposition",
    "prepositional phrase": "preposition",
    "pron": "pronoun",
    "pronoun": "pronoun",
    "det": "determiner",
    "determiner": "determiner",
    "article": "determiner",
    "conj": "conjunction",
    "conjunction": "conjunction",
    "coordinator": "conjunction",

    # Interjections
    "intj": "interjection",
    "interj": "interjection",
    "interjection": "interjection",

    # Deliberately weak / skipped categories
    "interfix": "",
    "prefix": "",
    "suffix": "",
    "particle": "",
    "phrase": "",
    "proverb": "",
    "punctuation": "",
    "symbol": "",
    "character": "",
    "letter": "",
}


# Semantic relation fields we want inside dictionary_lexicon_ingestor.py.
# Important: categories are deliberately excluded.
RELATION_FIELDS = {
    "synonyms": "synonyms",
    "antonyms": "antonyms",
    "hypernyms": "hypernyms",
    "hyponyms": "hyponyms",
    "holonyms": "holonyms",
    "meronyms": "meronyms",
    "derived": "derived",
    "related": "related",
    "coordinate_terms": "coordinate_terms",
    "troponyms": "troponyms",
    "alt_of": "alt_of",
    "form_of": "form_of",
    "instances": "instances",
    "topics": "topics",

    # Deliberately excluded from semantic relations:
    # "categories": "categories",
}


RELATION_LIMITS = {
    "synonyms": 64,
    "antonyms": 64,
    "hypernyms": 64,
    "hyponyms": 64,
    "related": 64,
    "derived": 48,
    "coordinate_terms": 48,
    "holonyms": 32,
    "meronyms": 32,
    "troponyms": 32,
    "alt_of": 32,
    "form_of": 32,
    "instances": 32,
    "topics": 24,
    "tags": 24,
}

DEFAULT_RELATION_LIMIT = 32


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class ConvertStats:
    input_path: str = ""
    output_path: str = ""
    started_at: str = ""
    finished_at: str = ""

    records_read: int = 0
    records_selected: int = 0
    senses_read: int = 0
    entries_written: int = 0

    skipped_wrong_language: int = 0
    skipped_no_word: int = 0
    skipped_no_sense: int = 0
    skipped_no_gloss: int = 0

    skipped_pos: Dict[str, int] = field(default_factory=dict)
    pos_counts: Dict[str, int] = field(default_factory=dict)
    relation_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Generic helpers
# =============================================================================

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_pos(pos: Any) -> str:
    p = canonical_text(pos).lower().replace("_", " ")
    p = p.replace(".", "")
    p = re.sub(r"\s+", " ", p).strip()
    return POS_ALIASES.get(p, p)


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(canonical_text(text))]


def safe_float(x: Any, default: float = 1.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))
    return str(obj)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def open_jsonl_text(path: Path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open_jsonl_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield line_no, obj


def list_text(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        s = canonical_text(value)
        return [s] if s else []

    if isinstance(value, Mapping):
        candidates: List[str] = []
        for key in ("word", "term", "text", "english", "name", "sense", "gloss", "id"):
            if key in value:
                candidates.extend(list_text(value.get(key)))
        return candidates

    if isinstance(value, Iterable):
        out: List[str] = []
        for item in value:
            out.extend(list_text(item))
        return out

    s = canonical_text(value)
    return [s] if s else []


def first_text(value: Any) -> str:
    vals = list_text(value)
    return vals[0] if vals else ""


def unique_ordered(items: Iterable[str], limit: int = 64) -> List[str]:
    out: List[str] = []
    seen = set()

    for item in items:
        s = canonical_text(item)
        if not s:
            continue

        key = s.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append(s)

        if len(out) >= int(limit):
            break

    return out


def relation_limit(rel: str) -> int:
    return int(RELATION_LIMITS.get(rel, DEFAULT_RELATION_LIMIT))


def compact_categories(value: Any, limit: int = 32) -> List[str]:
    """Keep categories as metadata only, not as semantic relations."""
    return unique_ordered(list_text(value), limit=limit)


# =============================================================================
# Kaikki extraction helpers
# =============================================================================

def extract_relation_values(payload: Any) -> List[str]:
    if payload is None:
        return []

    if isinstance(payload, str):
        return [payload]

    if isinstance(payload, Mapping):
        vals: List[str] = []

        for key in ("word", "term", "text", "english", "name"):
            if key in payload:
                vals.extend(list_text(payload.get(key)))

        if not vals:
            for value in payload.values():
                vals.extend(extract_relation_values(value))

        return vals

    if isinstance(payload, Iterable):
        vals: List[str] = []
        for item in payload:
            vals.extend(extract_relation_values(item))
        return vals

    return list_text(payload)


def extract_relations(record: Mapping[str, Any], sense: Mapping[str, Any]) -> Dict[str, List[str]]:
    relations: Dict[str, List[str]] = {}

    # Sense-level and record-level lexical relations.
    # Categories are deliberately excluded from RELATION_FIELDS.
    for source in (record, sense):
        for field, rel_name in RELATION_FIELDS.items():
            if field in source:
                vals = unique_ordered(
                    extract_relation_values(source.get(field)),
                    limit=relation_limit(rel_name),
                )
                if vals:
                    relations.setdefault(rel_name, [])
                    relations[rel_name].extend(vals)

    # Keep usage/register tags as weak relation hints.
    # Do not include Wiktionary categories as semantic relations.
    tags: List[str] = []
    tags.extend(list_text(sense.get("tags")))
    tags.extend(list_text(record.get("tags")))

    if tags:
        relations.setdefault("tags", [])
        relations["tags"].extend(unique_ordered(tags, limit=relation_limit("tags")))

    # Normalize and deduplicate per relation.
    clean: Dict[str, List[str]] = {}
    for rel, vals in relations.items():
        cleaned = unique_ordered(vals, limit=relation_limit(rel))
        if cleaned:
            clean[rel] = cleaned

    return clean


def extract_glosses(sense: Mapping[str, Any]) -> List[str]:
    glosses: List[str] = []

    for key in ("glosses", "raw_glosses"):
        vals = sense.get(key)

        if isinstance(vals, list):
            glosses.extend(canonical_text(v) for v in vals if canonical_text(v))
        elif isinstance(vals, str):
            glosses.append(canonical_text(vals))

    out: List[str] = []
    for gloss in glosses:
        g = canonical_text(gloss)
        if not g:
            continue

        low = g.lower()
        if low in {"no gloss", "unknown", "definition needed"}:
            continue

        out.append(g)

    return unique_ordered(out, limit=16)


def extract_examples(sense: Mapping[str, Any], max_examples: int = 4) -> List[str]:
    examples = sense.get("examples", [])
    out: List[str] = []

    if isinstance(examples, list):
        for item in examples:
            if isinstance(item, Mapping):
                for key in ("text", "english", "translation"):
                    txt = canonical_text(item.get(key))
                    if txt:
                        out.append(txt)
                        break
            else:
                txt = canonical_text(item)
                if txt:
                    out.append(txt)

    return unique_ordered(out, limit=max_examples)


def make_sense_id(record: Mapping[str, Any], sense: Mapping[str, Any], line_no: int, sense_index: int) -> str:
    word = canonical_text(record.get("word", ""))
    pos = normalize_pos(record.get("pos", ""))

    raw_id = (
        sense.get("id")
        or sense.get("senseid")
        or sense.get("sense_id")
        or sense.get("wikidata")
        or ""
    )

    sense_id = first_text(raw_id)
    if sense_id:
        return f"kaikki:{sense_id}"

    safe_word = re.sub(r"[^A-Za-z0-9_\-]+", "_", word).strip("_")[:80]
    safe_pos = re.sub(r"[^A-Za-z0-9_\-]+", "_", pos).strip("_")[:32]
    return f"kaikki:line:{line_no}:sense:{sense_index}:{safe_word}:{safe_pos}"


def make_entry(
    record: Mapping[str, Any],
    sense: Mapping[str, Any],
    *,
    line_no: int,
    sense_index: int,
    include_examples: bool = True,
    include_raw: bool = True,
    source: str = "kaikki:wiktextract",
) -> Optional[Dict[str, Any]]:
    word = canonical_text(record.get("word") or record.get("title") or "")
    if not word:
        return None

    pos = normalize_pos(record.get("pos", ""))
    glosses = extract_glosses(sense)
    if not glosses:
        return None

    definition = "; ".join(glosses)
    examples = extract_examples(sense) if include_examples else []
    relations = extract_relations(record, sense)

    weight = 1.0

    if examples:
        weight += 0.05

    if relations:
        relation_value_count = sum(len(v) for v in relations.values())
        weight += min(0.25, 0.01 * relation_value_count)

    lang = canonical_text(record.get("lang", ""))
    lang_code = canonical_text(record.get("lang_code", ""))

    raw_summary = {
        "kaikki_line_no": line_no,
        "sense_index": sense_index,
        "lang": lang,
        "lang_code": lang_code,
        "tags": unique_ordered(list_text(sense.get("tags")), limit=24),
        "record_tags": unique_ordered(list_text(record.get("tags")), limit=24),
        "categories": compact_categories(sense.get("categories"), limit=32),
        "record_categories": compact_categories(record.get("categories"), limit=32),
        "topics": unique_ordered(list_text(sense.get("topics")), limit=24),
        "examples": examples,
        "forms_sample": record.get("forms", [])[:20] if isinstance(record.get("forms"), list) else [],
        "etymology_text": canonical_text(record.get("etymology_text", ""))[:1200],
    }

    entry: Dict[str, Any] = {
        "word": word,
        "lemma": word,
        "part_of_speech": pos,
        "pos": pos,
        "definition": definition,
        "gloss": definition,
        "sense_id": make_sense_id(record, sense, line_no, sense_index),
        "source": source,
        "weight": weight,
        "relations": relations,
        "metadata": {
            "lang": lang,
            "lang_code": lang_code,
            "source_record_line": line_no,
            "sense_index": sense_index,
            "gloss_count": len(glosses),
            "example_count": len(examples),
            "category_count": len(raw_summary["categories"]) + len(raw_summary["record_categories"]),
        },
    }

    if examples:
        entry["examples"] = examples

    if include_raw:
        entry["raw"] = raw_summary

    return entry


def selected_language(record: Mapping[str, Any], lang_code: str, include_non_english: bool = False) -> bool:
    if include_non_english:
        return True

    expected = canonical_text(lang_code).lower()
    if not expected:
        return True

    actual = canonical_text(record.get("lang_code", "")).lower()
    return actual == expected


# =============================================================================
# Conversion pipeline
# =============================================================================

def convert_kaikki(
    input_path: Path,
    out_path: Path,
    *,
    lang_code: str = "en",
    include_non_english: bool = False,
    include_examples: bool = True,
    include_raw: bool = True,
    allowed_pos: Optional[Sequence[str]] = None,
    max_records: int = 0,
    max_entries: int = 0,
    progress_every: int = 100000,
    quiet: bool = False,
) -> ConvertStats:
    input_path = Path(input_path).resolve()
    out_path = Path(out_path).resolve()
    ensure_parent(out_path)

    allowed = {normalize_pos(p) for p in allowed_pos or [] if normalize_pos(p)}

    stats = ConvertStats(
        input_path=str(input_path),
        output_path=str(out_path),
        started_at=now_iso(),
    )

    if not input_path.exists():
        raise FileNotFoundError(f"input does not exist: {input_path}")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8", newline="\n") as out:
        for line_no, record in iter_jsonl(input_path):
            stats.records_read += 1

            if max_records and stats.records_read > max_records:
                break

            if not selected_language(record, lang_code, include_non_english=include_non_english):
                stats.skipped_wrong_language += 1
                continue

            word = canonical_text(record.get("word") or record.get("title") or "")
            if not word:
                stats.skipped_no_word += 1
                continue

            pos = normalize_pos(record.get("pos", ""))

            if allowed and pos not in allowed:
                stats.skipped_pos[pos or ""] = stats.skipped_pos.get(pos or "", 0) + 1
                continue

            senses = record.get("senses", [])
            if not isinstance(senses, list) or not senses:
                stats.skipped_no_sense += 1
                continue

            stats.records_selected += 1

            for sense_index, sense in enumerate(senses):
                if not isinstance(sense, Mapping):
                    continue

                stats.senses_read += 1

                entry = make_entry(
                    record,
                    sense,
                    line_no=line_no,
                    sense_index=sense_index,
                    include_examples=include_examples,
                    include_raw=include_raw,
                )

                if entry is None:
                    stats.skipped_no_gloss += 1
                    continue

                out.write(json.dumps(json_safe(entry), ensure_ascii=False, sort_keys=True) + "\n")
                stats.entries_written += 1

                entry_pos = str(entry.get("part_of_speech", ""))
                stats.pos_counts[entry_pos] = stats.pos_counts.get(entry_pos, 0) + 1

                rels = entry.get("relations", {})
                if isinstance(rels, Mapping):
                    for rel, vals in rels.items():
                        n = len(vals) if isinstance(vals, list) else 1
                        stats.relation_counts[rel] = stats.relation_counts.get(rel, 0) + n

                if max_entries and stats.entries_written >= max_entries:
                    break

            if max_entries and stats.entries_written >= max_entries:
                break

            if not quiet and progress_every and stats.records_read % int(progress_every) == 0:
                print(
                    f"[PROGRESS] records={stats.records_read:,} "
                    f"selected={stats.records_selected:,} "
                    f"entries={stats.entries_written:,}",
                    file=sys.stderr,
                )

    tmp_path.replace(out_path)

    stats.finished_at = now_iso()
    stats.pos_counts = dict(sorted(stats.pos_counts.items()))
    stats.relation_counts = dict(sorted(stats.relation_counts.items()))
    stats.skipped_pos = dict(sorted(stats.skipped_pos.items()))

    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    stats_path.write_text(
        json.dumps(json_safe(asdict(stats)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return stats


# =============================================================================
# CLI
# =============================================================================

def parse_pos_list(value: str) -> List[str]:
    if not value:
        return []
    return [normalize_pos(x) for x in re.split(r"[,;| ]+", value) if normalize_pos(x)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Kaikki/Wiktextract JSONL to LK20 lexicon JSONL.")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT), help="Input .jsonl or .jsonl.gz from Kaikki/Wiktextract.")
    parser.add_argument("--out", "-o", default=str(DEFAULT_OUT), help="Output LK20-normalized JSONL.")
    parser.add_argument("--lang-code", default="en", help="Language code to keep. Default: en.")
    parser.add_argument("--include-non-english", action="store_true", help="Keep all languages from the extraction.")
    parser.add_argument("--no-examples", action="store_true", help="Do not include examples in output.")
    parser.add_argument("--no-raw", action="store_true", help="Do not include raw summary metadata.")
    parser.add_argument(
        "--allowed-pos",
        default="",
        help="Optional comma-separated POS allowlist, e.g. noun,verb,adjective,adverb.",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Stop after this many input records. Useful for smoke tests.")
    parser.add_argument("--max-entries", type=int, default=0, help="Stop after this many output entries. Useful for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        stats = convert_kaikki(
            Path(args.input),
            Path(args.out),
            lang_code=str(args.lang_code),
            include_non_english=bool(args.include_non_english),
            include_examples=not bool(args.no_examples),
            include_raw=not bool(args.no_raw),
            allowed_pos=parse_pos_list(args.allowed_pos),
            max_records=int(args.max_records),
            max_entries=int(args.max_entries),
            progress_every=int(args.progress_every),
            quiet=bool(args.quiet),
        )

        print(json.dumps(json_safe(asdict(stats)), indent=2, ensure_ascii=False))
        return 0

    except Exception as exc:
        print(f"ERROR: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())