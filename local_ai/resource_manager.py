#!/usr/bin/env python3
# resource_manager.py
"""
LK20 / Akkurat Local AI — Resource Manager
==========================================

Downloads and verifies external open lexical resources for the local AI stack.

This script is intentionally dependency-free:
- Python standard library only.
- No ML dependencies.
- No mutation of semantic_bank.npz.
- Downloads are written to local_ai/resources by default.
- Every downloaded file gets SHA-256 metadata in resource_lock.json.

Primary resource for next lift:
- Kaikki / Wiktextract raw English Wiktionary JSONL gzip.

Typical commands
----------------
From C:\\Users\\ali_z\\ANU AI\\LK20\\local_ai:

    python resource_manager.py list

    python resource_manager.py install --include kaikki_raw_enwiktionary_jsonl_gz

    python resource_manager.py verify

Then convert:

    python convert_kaikki_wiktionary_to_lk20_jsonl.py ^
      --input resources\\kaikki\\raw-wiktextract-data.jsonl.gz ^
      --out resources\\normalized\\kaikki_english_lk20.jsonl ^
      --lang-code en

Then ingest:

    python dictionary_lexicon_ingestor.py ingest ^
      --input resources\\normalized\\kaikki_english_lk20.jsonl ^
      --use-installed ^
      --out semantic_bank_english_wiktionary.npz ^
      --dim 128 ^
      --no-relation-matrix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESOURCES_DIR = SCRIPT_DIR / "resources"
DEFAULT_LOCK_PATH = DEFAULT_RESOURCES_DIR / "resource_lock.json"

USER_AGENT = "LK20-LocalAI-ResourceManager/1.0"

CHUNK_SIZE = 1024 * 1024 * 4


RESOURCE_MANIFEST: Dict[str, Dict[str, Any]] = {
    "kaikki_raw_enwiktionary_jsonl_gz": {
        "url": "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz",
        "target": "kaikki/raw-wiktextract-data.jsonl.gz",
        "kind": "jsonl.gz",
        "license_note": (
            "Kaikki/Wiktextract data follows the licenses of Wiktionary content. "
            "Wiktionary content is generally CC BY-SA and GFDL; preserve attribution."
        ),
        "description": "Raw Wiktextract JSONL from English Wiktionary, compressed.",
        "large": True,
        "preferred": True,
    },
    "kaikki_english_postprocessed_jsonl": {
        "url": "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl",
        "target": "kaikki/kaikki.org-dictionary-English.jsonl",
        "kind": "jsonl",
        "license_note": (
            "Postprocessed Kaikki English dictionary JSONL. Kaikki marks this source as deprecated; "
            "prefer raw-wiktextract-data.jsonl.gz."
        ),
        "description": "Postprocessed English dictionary JSONL from Kaikki. Deprecated upstream.",
        "large": True,
        "preferred": False,
    },
    "open_english_wordnet_2025_json_zip": {
        "url": "https://en-word.net/static/english-wordnet-2025-json.zip",
        "target": "oewn/english-wordnet-2025-json.zip",
        "kind": "zip",
        "license_note": "Open English Wordnet 2025, CC-BY 4.0.",
        "description": "Open English Wordnet 2025 JSON release.",
        "large": False,
        "preferred": True,
    },
    "open_english_wordnet_2025_wndb_zip": {
        "url": "https://en-word.net/static/english-wordnet-2025.zip",
        "target": "oewn/english-wordnet-2025-wndb.zip",
        "kind": "zip",
        "license_note": "Open English Wordnet 2025, CC-BY 4.0.",
        "description": "Open English Wordnet 2025 WNDB legacy release.",
        "large": False,
        "preferred": False,
    },
    "gcide_latest_tar_xz": {
        "url": "https://ftp.gnu.org/gnu/gcide/gcide-latest.tar.xz",
        "target": "gcide/gcide-latest.tar.xz",
        "kind": "tar.xz",
        "license_note": "GNU Collaborative International Dictionary of English. Check included license files.",
        "description": "GCIDE latest source archive.",
        "large": False,
        "preferred": False,
    },
    "wordnet_db_3_0_tar_gz": {
        "url": "https://wordnetcode.princeton.edu/3.0/WNdb-3.0.tar.gz",
        "target": "wordnet/WNdb-3.0.tar.gz",
        "kind": "tar.gz",
        "license_note": "Princeton WordNet database. See archive license.",
        "description": "Princeton WordNet 3.0 database archive.",
        "large": False,
        "preferred": False,
    },
    "moby_word_lists_txt": {
        "url": "https://www.gutenberg.org/files/3201/3201-0.txt",
        "target": "moby/moby_word_lists.txt",
        "kind": "txt",
        "license_note": "Project Gutenberg Moby Word Lists. Public domain in the USA per Gutenberg metadata.",
        "description": "Moby word lists text from Project Gutenberg.",
        "large": False,
        "preferred": False,
    },
}


GROUPS: Dict[str, List[str]] = {
    "kaikki": ["kaikki_raw_enwiktionary_jsonl_gz"],
    "wiktionary": ["kaikki_raw_enwiktionary_jsonl_gz"],
    "oewn": ["open_english_wordnet_2025_json_zip"],
    "wordnet": ["wordnet_db_3_0_tar_gz", "open_english_wordnet_2025_json_zip"],
    "core": [
        "kaikki_raw_enwiktionary_jsonl_gz",
        "open_english_wordnet_2025_json_zip",
    ],
    "all": list(RESOURCE_MANIFEST.keys()),
}


@dataclass
class DownloadRecord:
    name: str
    url: str
    path: str
    size_bytes: int
    sha256: str
    downloaded_at: str
    license_note: str = ""
    description: str = ""
    kind: str = ""


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))
    return str(obj)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def format_bytes(n: Optional[int]) -> str:
    if n is None or n < 0:
        return "unknown"
    value = float(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{n}B"


def expand_includes(items: Sequence[str]) -> List[str]:
    if not items:
        return ["kaikki_raw_enwiktionary_jsonl_gz"]

    out: List[str] = []
    seen = set()

    for item in items:
        key = str(item).strip()
        if not key:
            continue

        expanded = GROUPS.get(key, [key])
        for name in expanded:
            if name not in RESOURCE_MANIFEST:
                raise KeyError(f"unknown resource or group: {name}")
            if name not in seen:
                seen.add(name)
                out.append(name)

    return out


def target_path(resources_dir: Path, name: str) -> Path:
    spec = RESOURCE_MANIFEST[name]
    return (resources_dir / spec["target"]).resolve()


def head_content_length(url: str, timeout: int = 30) -> Optional[int]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            val = resp.headers.get("Content-Length")
            if val:
                return int(val)
    except Exception:
        return None
    return None


def download_url(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    resume: bool = True,
    timeout: int = 60,
    quiet: bool = False,
) -> Dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not force:
        return {
            "status": "already_present",
            "path": str(dest),
            "size_bytes": int(dest.stat().st_size),
            "sha256": sha256_file(dest),
        }

    if force and dest.exists():
        dest.unlink()

    existing = tmp.stat().st_size if tmp.exists() and resume else 0

    headers = {"User-Agent": USER_AGENT}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)

    mode = "ab" if existing > 0 else "wb"
    started_at = time.time()
    last_report = 0.0
    bytes_written = existing
    expected_total = head_content_length(url, timeout=min(30, timeout)) or None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)

            if existing > 0 and status != 206:
                if not quiet:
                    print("[INFO] server did not honor resume; restarting download")
                tmp.unlink(missing_ok=True)
                bytes_written = 0
                mode = "wb"

            if not quiet:
                total_msg = format_bytes(expected_total)
                print(f"[GET] {url}")
                print(f"[OUT] {dest}")
                print(f"[SIZE] {total_msg}")

            with tmp.open(mode + "") as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break

                    f.write(chunk)
                    bytes_written += len(chunk)

                    now = time.time()
                    if not quiet and (now - last_report > 5.0):
                        elapsed = max(0.001, now - started_at)
                        rate = bytes_written / elapsed
                        if expected_total:
                            pct = 100.0 * bytes_written / expected_total
                            print(f"[PROGRESS] {format_bytes(bytes_written)} / {format_bytes(expected_total)} ({pct:.2f}%) @ {format_bytes(int(rate))}/s")
                        else:
                            print(f"[PROGRESS] {format_bytes(bytes_written)} @ {format_bytes(int(rate))}/s")
                        last_report = now

    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error downloading {url}: {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"URL error downloading {url}: {exc.reason}") from exc

    os.replace(tmp, dest)

    digest = sha256_file(dest)
    sidecar = dest.with_suffix(dest.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {dest.name}\n", encoding="utf-8")

    return {
        "status": "downloaded",
        "path": str(dest),
        "size_bytes": int(dest.stat().st_size),
        "sha256": digest,
    }


def load_lock(resources_dir: Path) -> Dict[str, Any]:
    path = resources_dir / "resource_lock.json"
    payload = read_json(path, {"version": 1, "resources": {}})
    if not isinstance(payload, dict):
        payload = {"version": 1, "resources": {}}
    payload.setdefault("version", 1)
    payload.setdefault("resources", {})
    return payload


def save_lock(resources_dir: Path, lock: Mapping[str, Any]) -> None:
    atomic_write_json(resources_dir / "resource_lock.json", dict(lock))


def install_resources(
    resources_dir: Path,
    names: Sequence[str],
    *,
    force: bool = False,
    resume: bool = True,
    timeout: int = 60,
    quiet: bool = False,
) -> Dict[str, Any]:
    resources_dir = resources_dir.resolve()
    resources_dir.mkdir(parents=True, exist_ok=True)

    names = expand_includes(names)
    lock = load_lock(resources_dir)

    result: Dict[str, Any] = {
        "ok": True,
        "resources_dir": str(resources_dir),
        "installed": {},
        "errors": {},
        "timestamp": now_iso(),
    }

    for name in names:
        spec = RESOURCE_MANIFEST[name]
        dest = target_path(resources_dir, name)

        try:
            if not quiet:
                print()
                print(f"[RESOURCE] {name}")
                print(f"[DESC] {spec.get('description', '')}")

            dl = download_url(
                str(spec["url"]),
                dest,
                force=force,
                resume=resume,
                timeout=timeout,
                quiet=quiet,
            )

            record = DownloadRecord(
                name=name,
                url=str(spec["url"]),
                path=str(dest),
                size_bytes=int(dl["size_bytes"]),
                sha256=str(dl["sha256"]),
                downloaded_at=now_iso(),
                license_note=str(spec.get("license_note", "")),
                description=str(spec.get("description", "")),
                kind=str(spec.get("kind", "")),
            )

            lock["resources"][name] = asdict(record)
            result["installed"][name] = dl

            if not quiet:
                print(f"[OK] {name}: {dl['status']}")
                print(f"[SHA256] {dl['sha256']}")

        except Exception as exc:
            result["ok"] = False
            result["errors"][name] = repr(exc)
            if not quiet:
                print(f"[ERROR] {name}: {exc}")

    save_lock(resources_dir, lock)
    return result


def verify_resources(resources_dir: Path, names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    resources_dir = resources_dir.resolve()
    lock = load_lock(resources_dir)

    selected = expand_includes(names or list(lock.get("resources", {}).keys()))
    result: Dict[str, Any] = {
        "ok": True,
        "resources_dir": str(resources_dir),
        "verified": {},
        "missing": {},
        "mismatched": {},
    }

    for name in selected:
        spec = RESOURCE_MANIFEST[name]
        path = target_path(resources_dir, name)
        rec = lock.get("resources", {}).get(name, {})
        expected = rec.get("sha256", "")

        if not path.exists():
            result["ok"] = False
            result["missing"][name] = str(path)
            continue

        actual = sha256_file(path)
        item = {
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256": actual,
            "expected_sha256": expected,
        }

        if expected and actual != expected:
            result["ok"] = False
            result["mismatched"][name] = item
        else:
            result["verified"][name] = item

    return result


def list_resources() -> Dict[str, Any]:
    return {
        "resources": RESOURCE_MANIFEST,
        "groups": GROUPS,
        "default_install": ["kaikki_raw_enwiktionary_jsonl_gz"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LK20 Local AI resource manager")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List known resources and groups.")
    p_list.add_argument("--json", action="store_true")

    p_install = sub.add_parser("install", help="Download one or more resources.")
    p_install.add_argument("--resources-dir", default=str(DEFAULT_RESOURCES_DIR))
    p_install.add_argument("--include", nargs="+", default=["kaikki_raw_enwiktionary_jsonl_gz"], help="Resource names or groups. Example: --include kaikki")
    p_install.add_argument("--force", action="store_true")
    p_install.add_argument("--no-resume", action="store_true")
    p_install.add_argument("--timeout", type=int, default=60)
    p_install.add_argument("--quiet", action="store_true")

    p_verify = sub.add_parser("verify", help="Verify downloaded resources against resource_lock.json.")
    p_verify.add_argument("--resources-dir", default=str(DEFAULT_RESOURCES_DIR))
    p_verify.add_argument("--include", nargs="*", default=None)

    p_lock = sub.add_parser("lock", help="Print resource_lock.json.")
    p_lock.add_argument("--resources-dir", default=str(DEFAULT_RESOURCES_DIR))

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        payload = list_resources()
        print(json.dumps(json_safe(payload), indent=2, ensure_ascii=False))
        return 0

    if args.command == "install":
        result = install_resources(
            Path(args.resources_dir),
            args.include,
            force=bool(args.force),
            resume=not bool(args.no_resume),
            timeout=int(args.timeout),
            quiet=bool(args.quiet),
        )
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
        return 0 if result.get("ok") else 1

    if args.command == "verify":
        result = verify_resources(Path(args.resources_dir), args.include)
        print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
        return 0 if result.get("ok") else 1

    if args.command == "lock":
        lock = load_lock(Path(args.resources_dir))
        print(json.dumps(json_safe(lock), indent=2, ensure_ascii=False))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())