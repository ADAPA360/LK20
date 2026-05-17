#!/usr/bin/env python3
# convert_kaikki_slice_to_lk20_jsonl.py
"""
Convert one bounded Kaikki/Wiktextract record range into LK20 JSONL.

This script imports the already-populated converter module and reuses its
normalization/extraction logic, but adds start/end record bounds so each shard
can run as a short independent Python process.

Typical use:

    python convert_kaikki_slice_to_lk20_jsonl.py ^
      --input resources\\kaikki\\raw-wiktextract-data.jsonl.gz ^
      --out resources\\normalized\\shards\\kaikki_000001_250000.jsonl ^
      --start-record 1 ^
      --end-record 250000 ^
      --lang-code en ^
      --allowed-pos noun,verb,adjective,adverb,preposition,pronoun,determiner,conjunction,interjection ^
      --no-raw
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import convert_kaikki_wiktionary_to_lk20_jsonl as base


@dataclass
class SliceStats:
    input_path: str = ""
    output_path: str = ""
    started_at: str = ""
    finished_at: str = ""

    start_record: int = 1
    end_record: int = 0

    records_seen: int = 0
    records_in_range: int = 0
    records_selected: int = 0
    senses_read: int = 0
    entries_written: int = 0

    skipped_wrong_language: int = 0
    skipped_no_word: int = 0
    skipped_no_sense: int = 0
    skipped_no_gloss: int = 0
    skipped_too_large_line: int = 0

    skipped_pos: Dict[str, int] = field(default_factory=dict)
    pos_counts: Dict[str, int] = field(default_factory=dict)
    relation_counts: Dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def atomic_replace(src: Path, dst: Path) -> None:
    for _ in range(10):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(1.0)
    os.replace(src, dst)


def iter_jsonl_range(path: Path, start_record: int, end_record: int, max_line_chars: int):
    """
    Stream records from start_record to end_record, inclusive.

    This function avoids retaining prior records and optionally skips oversized
    lines before json.loads.
    """
    import gzip

    start_record = int(max(1, start_record))
    end_record = int(max(start_record, end_record))

    opener = gzip.open if str(path).lower().endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no < start_record:
                continue
            if line_no > end_record:
                break

            if max_line_chars and len(line) > max_line_chars:
                yield line_no, None, "too_large_line"
                continue

            s = line.strip()
            if not s:
                continue

            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                yield line_no, None, "json_decode_error"
                continue

            if isinstance(obj, dict):
                yield line_no, obj, ""


def convert_slice(
    input_path: Path,
    out_path: Path,
    *,
    start_record: int,
    end_record: int,
    lang_code: str = "en",
    include_non_english: bool = False,
    include_examples: bool = True,
    include_raw: bool = False,
    allowed_pos: Optional[Sequence[str]] = None,
    progress_every: int = 25000,
    max_line_chars: int = 2_000_000,
    quiet: bool = False,
) -> SliceStats:
    input_path = Path(input_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    allowed = {base.normalize_pos(p) for p in allowed_pos or [] if base.normalize_pos(p)}

    stats = SliceStats(
        input_path=str(input_path),
        output_path=str(out_path),
        started_at=now_iso(),
        start_record=int(start_record),
        end_record=int(end_record),
    )

    if not input_path.exists():
        raise FileNotFoundError(f"input does not exist: {input_path}")

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8", newline="\n") as out:
        for line_no, record, error in iter_jsonl_range(
            input_path,
            int(start_record),
            int(end_record),
            int(max_line_chars),
        ):
            stats.records_seen = line_no
            stats.records_in_range += 1

            if error == "too_large_line":
                stats.skipped_too_large_line += 1
                continue
            if error:
                stats.warnings.append(f"line {line_no}: {error}")
                continue

            if record is None:
                continue

            if not base.selected_language(record, lang_code, include_non_english=include_non_english):
                stats.skipped_wrong_language += 1
                continue

            word = base.canonical_text(record.get("word") or record.get("title") or "")
            if not word:
                stats.skipped_no_word += 1
                continue

            pos = base.normalize_pos(record.get("pos", ""))

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

                entry = base.make_entry(
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

                out.write(json.dumps(base.json_safe(entry), ensure_ascii=False, sort_keys=True) + "\n")
                stats.entries_written += 1

                entry_pos = str(entry.get("part_of_speech", ""))
                stats.pos_counts[entry_pos] = stats.pos_counts.get(entry_pos, 0) + 1

                rels = entry.get("relations", {})
                if isinstance(rels, Mapping):
                    for rel, vals in rels.items():
                        n = len(vals) if isinstance(vals, list) else 1
                        stats.relation_counts[rel] = stats.relation_counts.get(rel, 0) + n

            if not quiet and progress_every and stats.records_in_range % int(progress_every) == 0:
                print(
                    f"[SLICE PROGRESS] range={start_record}-{end_record} "
                    f"in_range={stats.records_in_range:,} "
                    f"selected={stats.records_selected:,} "
                    f"entries={stats.entries_written:,}",
                    file=sys.stderr,
                    flush=True,
                )

    atomic_replace(tmp_path, out_path)

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert one bounded Kaikki/Wiktextract slice to LK20 JSONL.")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--out", "-o", required=True)
    parser.add_argument("--start-record", type=int, required=True)
    parser.add_argument("--end-record", type=int, required=True)
    parser.add_argument("--lang-code", default="en")
    parser.add_argument("--include-non-english", action="store_true")
    parser.add_argument("--no-examples", action="store_true")
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--allowed-pos", default="")
    parser.add_argument("--progress-every", type=int, default=25000)
    parser.add_argument("--max-line-chars", type=int, default=2_000_000)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        stats = convert_slice(
            Path(args.input),
            Path(args.out),
            start_record=int(args.start_record),
            end_record=int(args.end_record),
            lang_code=str(args.lang_code),
            include_non_english=bool(args.include_non_english),
            include_examples=not bool(args.no_examples),
            include_raw=not bool(args.no_raw),
            allowed_pos=base.parse_pos_list(args.allowed_pos),
            progress_every=int(args.progress_every),
            max_line_chars=int(args.max_line_chars),
            quiet=bool(args.quiet),
        )

        print(json.dumps(json_safe(asdict(stats)), indent=2, ensure_ascii=False))
        return 0

    except Exception as exc:
        print(f"ERROR: {exc!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())