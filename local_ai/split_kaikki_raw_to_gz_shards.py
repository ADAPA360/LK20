#!/usr/bin/env python3
# split_kaikki_raw_to_gz_shards.py
"""
Split a large Kaikki/Wiktextract JSONL.GZ file into smaller JSONL.GZ shards.

This does not parse JSON. It only streams lines and writes bounded compressed
raw shards. This makes later conversion safer because each converter process
handles a small independent input file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass
class SplitStats:
    input_path: str
    out_dir: str
    lines_per_shard: int
    started_at: str
    finished_at: str = ""
    total_lines: int = 0
    shard_count: int = 0
    last_shard: str = ""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool, float)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))
    return str(obj)


def atomic_replace(src: Path, dst: Path) -> None:
    for _ in range(12):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(1.0)
    os.replace(src, dst)


def split_raw(
    input_path: Path,
    out_dir: Path,
    *,
    lines_per_shard: int = 10000,
    force: bool = False,
    progress_every: int = 250000,
) -> SplitStats:
    input_path = Path(input_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input does not exist: {input_path}")

    if lines_per_shard <= 0:
        raise ValueError("lines_per_shard must be positive")

    stats = SplitStats(
        input_path=str(input_path),
        out_dir=str(out_dir),
        lines_per_shard=int(lines_per_shard),
        started_at=now_iso(),
    )

    current_out = None
    current_tmp = None
    current_final = None
    shard_start = 1
    shard_end = 0
    shard_index = 0

    def close_current() -> None:
        nonlocal current_out, current_tmp, current_final, shard_index

        if current_out is None:
            return

        current_out.close()
        atomic_replace(current_tmp, current_final)

        shard_index += 1
        stats.shard_count = shard_index
        stats.last_shard = str(current_final)

        current_out = None
        current_tmp = None
        current_final = None

    def open_shard(start_line: int) -> None:
        nonlocal current_out, current_tmp, current_final, shard_start, shard_end

        shard_start = int(start_line)
        shard_end = shard_start + int(lines_per_shard) - 1

        name = f"raw_{shard_start:07d}_{shard_end:07d}.jsonl.gz"
        current_final = out_dir / name
        current_tmp = current_final.with_suffix(current_final.suffix + ".tmp")

        if current_final.exists() and not force:
            raise FileExistsError(
                f"shard already exists: {current_final}. "
                "Delete existing shards or pass --force."
            )

        current_tmp.unlink(missing_ok=True)
        current_out = gzip.open(current_tmp, "wt", encoding="utf-8", newline="\n")

    with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as src:
        for line_no, line in enumerate(src, start=1):
            if current_out is None:
                open_shard(line_no)

            current_out.write(line)
            stats.total_lines = line_no

            if line_no >= shard_end:
                close_current()

            if progress_every and line_no % int(progress_every) == 0:
                print(
                    f"[SPLIT] lines={line_no:,} shards={stats.shard_count:,}",
                    flush=True,
                )

    close_current()

    stats.finished_at = now_iso()

    stats_path = out_dir / "raw_split_stats.json"
    stats_path.write_text(
        json.dumps(json_safe(asdict(stats)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split Kaikki raw JSONL.GZ into smaller JSONL.GZ shards.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", default="resources/kaikki/raw_shards")
    parser.add_argument("--lines-per-shard", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        stats = split_raw(
            Path(args.input),
            Path(args.out_dir),
            lines_per_shard=int(args.lines_per_shard),
            force=bool(args.force),
            progress_every=int(args.progress_every),
        )
        print(json.dumps(json_safe(asdict(stats)), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())