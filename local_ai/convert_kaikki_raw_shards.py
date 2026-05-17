#!/usr/bin/env python3
# convert_kaikki_raw_shards.py
"""
Convert raw Kaikki JSONL.GZ shards into LK20-normalized JSONL shards.

Each raw shard is converted by launching convert_kaikki_wiktionary_to_lk20_jsonl.py
as a separate Python process. This isolates crashes and allows restart.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
CONVERTER = SCRIPT_DIR / "convert_kaikki_wiktionary_to_lk20_jsonl.py"

RAW_RX = re.compile(r"raw_(\d{7})_(\d{7})\.jsonl\.gz$")


@dataclass
class BatchStats:
    started_at: str
    finished_at: str = ""
    raw_dir: str = ""
    out_dir: str = ""
    total_raw_shards: int = 0
    converted: int = 0
    skipped: int = 0
    failed: List[Dict[str, Any]] = field(default_factory=list)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))
    return str(obj)


def raw_shards(raw_dir: Path) -> List[tuple[int, int, Path]]:
    out = []
    for p in raw_dir.glob("raw_*.jsonl.gz"):
        m = RAW_RX.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), int(m.group(2)), p))
    out.sort()
    return out


def run_one(
    raw_path: Path,
    out_path: Path,
    *,
    allowed_pos: str,
    retries: int = 2,
    retry_delay: float = 5.0,
    quiet: bool = False,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(CONVERTER),
        "--input",
        str(raw_path),
        "--out",
        str(out_path),
        "--lang-code",
        "en",
        "--allowed-pos",
        allowed_pos,
        "--no-raw",
        "--progress-every",
        "0",
    ]

    for attempt in range(1, int(retries) + 2):
        if out_path.exists() and Path(str(out_path) + ".stats.json").exists():
            return 0

        Path(str(out_path) + ".tmp").unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        Path(str(out_path) + ".stats.json").unlink(missing_ok=True)

        if not quiet:
            print(f"[CONVERT] {raw_path.name} -> {out_path.name} attempt={attempt}", flush=True)

        proc = subprocess.run(cmd)

        if proc.returncode == 0:
            return 0

        if not quiet:
            print(f"[WARN] failed {raw_path.name} returncode={proc.returncode}", flush=True)

        time.sleep(float(retry_delay))

    return int(proc.returncode)


def convert_all(
    raw_dir: Path,
    out_dir: Path,
    *,
    allowed_pos: str,
    retries: int = 2,
    stop_on_error: bool = True,
    quiet: bool = False,
) -> BatchStats:
    raw_dir = Path(raw_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = BatchStats(
        started_at=now_iso(),
        raw_dir=str(raw_dir),
        out_dir=str(out_dir),
    )

    shards = raw_shards(raw_dir)
    stats.total_raw_shards = len(shards)

    for start, end, raw_path in shards:
        out_path = out_dir / f"kaikki_{start:07d}_{end:07d}.jsonl"

        if out_path.exists() and Path(str(out_path) + ".stats.json").exists():
            stats.skipped += 1
            if not quiet:
                print(f"[SKIP] {out_path.name}", flush=True)
            continue

        rc = run_one(
            raw_path,
            out_path,
            allowed_pos=allowed_pos,
            retries=retries,
            quiet=quiet,
        )

        if rc == 0:
            stats.converted += 1
        else:
            fail = {
                "raw": str(raw_path),
                "out": str(out_path),
                "returncode": rc,
            }
            stats.failed.append(fail)

            if stop_on_error:
                break

    stats.finished_at = now_iso()

    stats_path = out_dir / "conversion_batch_stats.json"
    stats_path.write_text(
        json.dumps(json_safe(asdict(stats)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Kaikki raw shards to LK20 normalized shards.")
    parser.add_argument("--raw-dir", default="resources/kaikki/raw_shards")
    parser.add_argument("--out-dir", default="resources/normalized/shards")
    parser.add_argument(
        "--allowed-pos",
        default="noun,verb,adjective,adverb,preposition,pronoun,determiner,conjunction,interjection",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stats = convert_all(
        Path(args.raw_dir),
        Path(args.out_dir),
        allowed_pos=str(args.allowed_pos),
        retries=int(args.retries),
        stop_on_error=not bool(args.continue_on_error),
        quiet=bool(args.quiet),
    )

    print(json.dumps(json_safe(asdict(stats)), indent=2, ensure_ascii=False))
    return 0 if not stats.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())