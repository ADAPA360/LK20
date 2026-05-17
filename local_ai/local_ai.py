#!/usr/bin/env python3
# local_ai.py
"""
LK20 / Akkurat Local AI — CLI contact point
===========================================

Production role
---------------
This script is the main command-line contact point for the LK20 local AI stack.

It coordinates the local advisory, lexical, semantic-attractor, ingestion, and
sentence-building layers:

    local_ai_adapter.py                         root-level safe adapter
    local_ai/semantic_attractors.py             semantic attractor bank
    local_ai/dictionary_lexicon_ingestor.py     semantic-bank builder
    local_ai/sentence_builder.py                explicit sentence candidate builder
    akkurat_atomtn_stack/*                      tensor-network substrate

Safety posture
--------------
The root-level local_ai_adapter.py is the default safety boundary for CLI-facing
lexical/advisory behavior. It provides:

    - read-only tensor lexicon lookup;
    - context-aware word-sense ranking;
    - alias / inflection resolution;
    - entropy diagnostics;
    - safe advisory output.

Direct sentence generation through sentence_builder.py is still available, but
is treated as raw/legacy generation until the builder itself is hardened.

Supported commands
------------------
    doctor
    adapter-status
    lexicon
    alias
    advisory
    wsd-test
    ingest
    install-resources
    inspect-bank
    build
    chat
    attractors
    run-script

Windows usage from project root:
    cd "C:\\Users\\ali_z\\ANU AI\\LK20"
    python local_ai\\local_ai.py doctor --compile
    python local_ai\\local_ai.py adapter-status
    python local_ai\\local_ai.py lexicon --term cat --context "cat animal"
    python local_ai\\local_ai.py alias --term running
    python local_ai\\local_ai.py advisory --text "cat animal"
    python local_ai\\local_ai.py build --prompt "cat animal"
    python local_ai\\local_ai.py build --prompt "cat animal" --raw

Windows usage from local_ai folder:
    cd "C:\\Users\\ali_z\\ANU AI\\LK20\\local_ai"
    python local_ai.py doctor --compile
    python local_ai.py adapter-status
    python local_ai.py lexicon --term cat --context "cat animal"
"""

from __future__ import annotations

import argparse
import compileall
import importlib
import inspect
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Path model
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
LOCAL_AI_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = LOCAL_AI_DIR.parent
STACK_DIR = PROJECT_ROOT / "akkurat_atomtn_stack"

ROOT_ADAPTER = PROJECT_ROOT / "local_ai_adapter.py"

DEFAULT_BANK = LOCAL_AI_DIR / "semantic_bank.npz"
DEFAULT_DICTIONARY = LOCAL_AI_DIR / "dictionary.json"
DEFAULT_RESOURCES_DIR = LOCAL_AI_DIR / "resources"
DEFAULT_TENSOR_LEXICON_DB = LOCAL_AI_DIR / "resources" / "kaikki_tensor" / "lexicon_kaikki_hybrid_v3.db"
DEFAULT_TENSOR_LEXICON_CORE = LOCAL_AI_DIR / "resources" / "kaikki_tensor" / "lexicon_kaikki_hybrid_v3.core.npz"
DEFAULT_TENSOR_LEXICON_MANIFEST = LOCAL_AI_DIR / "resources" / "kaikki_tensor" / "lexicon_kaikki_hybrid_v3.manifest.json"

VERSION = 4

LOCAL_EXPECTED = [
    "semantic_attractors.py",
    "dictionary_lexicon_ingestor.py",
    "sentence_builder.py",
    "entropy_nlp.py",
    "local_ai.py",
]

ROOT_EXPECTED = [
    "local_ai_adapter.py",
]

STACK_REQUIRED = [
    "math_utils.py",
    "geometry.py",
    "ttn_state.py",
]

STACK_OPTIONAL_FILES = [
    "flow.py",
    "fiber.py",
    "projection.py",
    "fuzzy_backend.py",
    "hamiltonian.py",
    "apply.py",
    "evolve.py",
]

LOCAL_IMPORTS_REQUIRED = [
    "semantic_attractors",
    "dictionary_lexicon_ingestor",
    "sentence_builder",
]

LOCAL_IMPORTS_OPTIONAL = [
    "entropy_nlp",
]

ROOT_IMPORTS_REQUIRED = [
    "local_ai_adapter",
]

STACK_IMPORTS_REQUIRED = [
    "math_utils",
    "geometry",
    "ttn_state",
]

STACK_IMPORTS_OPTIONAL = [
    "flow",
    "fiber",
    "projection",
    "fuzzy_backend",
    "hamiltonian",
    "apply",
    "evolve",
]


# =============================================================================
# Generic helpers
# =============================================================================

def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {
                "real": np.nan_to_num(obj.real).astype(float).tolist(),
                "imag": np.nan_to_num(obj.imag).astype(float).tolist(),
            }
        return np.nan_to_num(obj).astype(float).tolist()
    if is_dataclass(obj):
        return json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))
    return str(obj)


def print_json(obj: Any) -> None:
    print(json.dumps(json_safe(obj), indent=2, ensure_ascii=False))


def status_line(label: str, ok: bool, detail: str = "", *, warn: bool = False) -> None:
    if warn:
        tag = "WARN"
    else:
        tag = "OK" if ok else "FAIL"
    if detail:
        print(f"  [{tag:<7}] {label} -> {detail}")
    else:
        print(f"  [{tag:<7}] {label}")


def ensure_import_paths() -> None:
    """
    Ensure local_ai, stack, and project-root modules are importable regardless
    of current working directory.
    """
    for p in (str(PROJECT_ROOT), str(LOCAL_AI_DIR), str(STACK_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def resolve_path(path: Optional[str | Path], *, default: Optional[Path] = None) -> Path:
    """
    Resolve paths robustly from either:
      - project root, or
      - local_ai folder.

    Rules:
      absolute path -> itself
      relative existing from cwd -> cwd/path
      relative existing from local_ai -> local_ai/path
      relative existing from project root -> project_root/path
      otherwise -> project_root/path, except bare filenames default to local_ai/path
    """
    if path is None or str(path).strip() == "":
        if default is None:
            raise ValueError("Path is required.")
        return Path(default).resolve()

    p = Path(str(path).strip().strip('"'))
    if p.is_absolute():
        return p.resolve()

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    local_candidate = (LOCAL_AI_DIR / p).resolve()
    if local_candidate.exists():
        return local_candidate

    root_candidate = (PROJECT_ROOT / p).resolve()
    if root_candidate.exists():
        return root_candidate

    if len(p.parts) == 1:
        return local_candidate

    return root_candidate


def resolve_bank_path(path: Optional[str | Path]) -> Path:
    return resolve_path(path, default=DEFAULT_BANK)


def resolve_input_path(path: Optional[str | Path]) -> Optional[Path]:
    if path is None or str(path).strip() == "":
        return None
    return resolve_path(path)


def module_file(mod: Any) -> str:
    return str(Path(getattr(mod, "__file__", "")).resolve()) if getattr(mod, "__file__", None) else "<unknown>"


def import_module_safe(name: str) -> Tuple[bool, Optional[Any], Optional[str]]:
    try:
        mod = importlib.import_module(name)
        return True, mod, None
    except Exception as exc:
        return False, None, "".join(traceback.format_exception_only(type(exc), exc)).strip()


def callable_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", repr(fn))


def get_first_callable(module: Any, names: Sequence[str]) -> Optional[Callable[..., Any]]:
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj
    return None


def _call_with_supported_kwargs(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """
    Call a function with only the keyword arguments it supports.

    If a callable accepts **kwargs, all provided kwargs are passed through.
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_kwargs:
        return fn(**kwargs)

    usable = {
        k: v
        for k, v in kwargs.items()
        if k in params
        and params[k].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return fn(**usable)


def _compact_gloss(text: Any, *, max_len: int = 240) -> str:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 1)].rstrip() + "…"


def _format_lexicon_rows(rows: Sequence[Mapping[str, Any]], *, json_output: bool = False) -> None:
    if json_output:
        print_json(list(rows))
        return

    if not rows:
        print("[]")
        return

    for i, row in enumerate(rows, start=1):
        word = row.get("word")
        lemma = row.get("lemma")
        pos = row.get("pos")
        match_type = row.get("match_type")
        alias_type = row.get("alias_type")
        matched_alias = row.get("matched_alias")
        sense_index = row.get("sense_index")
        source_line = row.get("source_line")
        gloss = _compact_gloss(row.get("gloss"))

        label_parts = [str(word or lemma or "<unknown>")]
        if lemma and lemma != word:
            label_parts.append(f"lemma={lemma}")
        if pos:
            label_parts.append(str(pos))
        if match_type:
            label_parts.append(f"match={match_type}")
        if alias_type:
            label_parts.append(f"alias_type={alias_type}")
        if matched_alias:
            label_parts.append(f"matched={matched_alias}")
        if sense_index is not None:
            label_parts.append(f"sense={sense_index}")
        if source_line is not None:
            label_parts.append(f"line={source_line}")

        print(f"{i}. " + " | ".join(label_parts))
        if gloss:
            print(f"   {gloss}")


def format_result(result: Any, *, json_output: bool = False) -> None:
    if json_output:
        print_json(result)
        return

    if result is None:
        print("(no result)")
        return

    if isinstance(result, str):
        print(result)
        return

    if isinstance(result, Mapping):
        for key in ("sentence", "text", "output", "result", "answer"):
            if key in result and isinstance(result[key], str):
                print(result[key])
                return
        print_json(result)
        return

    if isinstance(result, (list, tuple)):
        if not result:
            print("[]")
            return
        for i, item in enumerate(result, start=1):
            if isinstance(item, str):
                print(f"{i}. {item}")
            elif isinstance(item, Mapping):
                if "sentence" in item:
                    print(f"{i}. {item['sentence']}")
                elif "text" in item:
                    print(f"{i}. {item['text']}")
                elif "lemma" in item or "word" in item:
                    label = item.get("lemma", item.get("word"))
                    score = item.get("score", item.get("similarity", ""))
                    print(f"{i}. {label} {score}".rstrip())
                else:
                    print(f"{i}. {json.dumps(json_safe(item), ensure_ascii=False)}")
            else:
                sentence = getattr(item, "sentence", None)
                if isinstance(sentence, str):
                    print(f"{i}. {sentence}")
                else:
                    print(f"{i}. {item}")
        return

    print(result)


# =============================================================================
# Adapter bridge
# =============================================================================

def get_local_adapter() -> Any:
    ensure_import_paths()
    try:
        from local_ai_adapter import get_adapter  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Could not import root local_ai_adapter.py: {exc}") from exc
    return get_adapter(PROJECT_ROOT)


def adapter_status() -> Dict[str, Any]:
    adapter = get_local_adapter()
    try:
        return adapter.get_status()
    finally:
        adapter.close()


def run_adapter_status(args: argparse.Namespace) -> int:
    """Print root local_ai_adapter.py status. JSON is the default output."""
    try:
        result = adapter_status()
        if not isinstance(result, Mapping) or not result:
            print_json({"ok": False, "status": "unavailable", "error": "empty adapter status"})
            sys.stdout.flush()
            return 1

        if bool(getattr(args, "summary", False)):
            tensor = result.get("tensor_lexicon") or {}
            entropy = result.get("entropy_nlp") or {}
            print(f"status:         {result.get('status')}")
            print(f"provider:       {result.get('provider')}")
            print(f"semantic_bank:  {result.get('semantic_bank')}")
            print(f"tensor_entries: {tensor.get('entry_count')}")
            print(f"vectors_mode:   {tensor.get('vectors_mode')}")
            print(f"entropy:        {entropy.get('status')}")
            caps = result.get("capabilities") or []
            if caps:
                print("capabilities:   " + ", ".join(str(x) for x in caps))
        else:
            print_json(result)

        sys.stdout.flush()
        return 0 if result.get("status") in {"active", "limited"} else 1
    except Exception:
        eprint("Adapter status failed.")
        traceback.print_exc()
        return 1


def run_lexicon(args: argparse.Namespace) -> int:
    term = str(args.term or "").strip()
    if not term:
        eprint("Term is required. Use --term cat")
        return 1

    adapter = get_local_adapter()
    try:
        rows = adapter.lookup_lexicon(
            term,
            limit=int(args.limit),
            pos=args.pos,
            include_aliases=not bool(args.no_aliases),
            include_relations=bool(args.relations),
            context=str(args.context or term),
        )
        _format_lexicon_rows(rows, json_output=bool(args.json))
        return 0 if rows else 1
    except Exception:
        eprint("Lexicon lookup failed.")
        traceback.print_exc()
        return 1
    finally:
        adapter.close()


def run_alias(args: argparse.Namespace) -> int:
    term = str(args.term or "").strip()
    if not term:
        eprint("Alias term is required. Use --term running")
        return 1

    adapter = get_local_adapter()
    try:
        rows = adapter.resolve_alias(
            term,
            limit=int(args.limit),
            pos=args.pos,
            context=str(args.context or term),
        )
        _format_lexicon_rows(rows, json_output=bool(args.json))
        return 0 if rows else 1
    except Exception:
        eprint("Alias resolution failed.")
        traceback.print_exc()
        return 1
    finally:
        adapter.close()


def run_advisory(args: argparse.Namespace) -> int:
    text = str(args.text or args.prompt or "").strip()
    if not text:
        eprint("Text is required. Use --text \"cat animal\"")
        return 1

    adapter = get_local_adapter()
    try:
        if args.json:
            result = {
                "text": text,
                "advisory": adapter.suggest_improvements(text),
                "diagnostic": adapter.analyze_generated_text(text, profile=str(args.profile)),
            }
            print_json(result)
        else:
            print(adapter.suggest_improvements(text))
        return 0
    except Exception:
        eprint("Advisory failed.")
        traceback.print_exc()
        return 1
    finally:
        adapter.close()


def run_wsd_test(args: argparse.Namespace) -> int:
    """
    Regression smoke test for adapter WSD behavior.
    """
    adapter = get_local_adapter()
    failed: List[str] = []

    checks = [
        ("cat", "cat animal", "noun", "felidae"),
        ("cat", "cat unix command", "noun", "unix"),
        ("run", "run quickly on feet", "verb", "move swiftly"),
        ("think", "think in the mind", "verb", "ponder"),
        ("because", "because reason cause", "conjunction", "reason"),
    ]

    alias_checks = [
        ("running", "run"),
        ("runs", "run"),
        ("cats", "cat"),
        ("thought", "think"),
    ]

    details: Dict[str, Any] = {"lexicon": [], "aliases": [], "advisory": None}

    try:
        for term, context, expected_pos, expected_fragment in checks:
            rows = adapter.lookup_lexicon(term, limit=3, context=context)
            top = rows[0] if rows else {}
            gloss = str(top.get("gloss", ""))
            item = {
                "term": term,
                "context": context,
                "expected_pos": expected_pos,
                "expected_fragment": expected_fragment,
                "top": top,
                "ok": bool(rows)
                and top.get("pos") == expected_pos
                and expected_fragment.lower() in gloss.lower(),
            }
            details["lexicon"].append(item)
            if not item["ok"]:
                failed.append(f"{term}: expected POS={expected_pos}, fragment={expected_fragment!r}, got={top}")

        for alias, expected_lemma in alias_checks:
            rows = adapter.resolve_alias(alias, limit=3, context=alias)
            top = rows[0] if rows else {}
            item = {
                "alias": alias,
                "expected_lemma": expected_lemma,
                "top": top,
                "ok": bool(rows) and top.get("lemma") == expected_lemma,
            }
            details["aliases"].append(item)
            if not item["ok"]:
                failed.append(f"{alias}: expected lemma={expected_lemma}, got={top}")

        advisory = adapter.suggest_improvements("cat animal")
        advisory_ok = (
            ("felidae" in advisory.lower() or "animal" in advisory.lower())
            and "methcathinone" not in advisory.lower()
            and "cameroonian" not in advisory.lower()
            and "haydn" not in advisory.lower()
        )
        details["advisory"] = {"text": advisory, "ok": advisory_ok}
        if not advisory_ok:
            failed.append(f"advisory unsafe/unexpected: {advisory}")

        result = {"ok": not failed, "failures": failed, "details": details}
        if args.json:
            print_json(result)
        else:
            if failed:
                print("local_ai_adapter WSD regression tests: FAIL")
                for f in failed:
                    print(f"  - {f}")
            else:
                print("local_ai_adapter WSD regression tests: OK")
        return 0 if not failed else 1

    except Exception:
        eprint("WSD regression test failed with exception.")
        traceback.print_exc()
        return 1
    finally:
        adapter.close()


# =============================================================================
# Bank utilities
# =============================================================================

def inspect_bank_fallback(path: Path) -> Dict[str, Any]:
    p = Path(path).resolve()
    if not p.exists():
        return {
            "kind": "SemanticAttractorBank",
            "path": str(p),
            "exists": False,
            "is_stable": False,
        }

    pickle_warning = False
    try:
        data_ctx = np.load(p, allow_pickle=False)
    except Exception:
        data_ctx = np.load(p, allow_pickle=True)
        pickle_warning = True

    with data_ctx as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32) if "vectors" in data else np.zeros((0, 0), dtype=np.float32)

        def _safe_arr(name: str, default: Any) -> Any:
            try:
                return data[name] if name in data else default
            except Exception:
                return default

        lemmas = _safe_arr("lemmas", _safe_arr("words", np.asarray([], dtype=str)))
        relation = np.asarray(data["relation_matrix"], dtype=np.float32) if "relation_matrix" in data else np.zeros((0, 0), dtype=np.float32)

        if vectors.ndim == 2 and vectors.shape[0] > 0:
            norms = np.linalg.norm(vectors, axis=1)
        else:
            norms = np.zeros((0,), dtype=np.float32)

        metadata = {}
        for meta_key in ("metadata_json", "__metadata_json__"):
            if meta_key in data:
                try:
                    metadata = json.loads(str(data[meta_key].item()))
                    break
                except Exception:
                    metadata = {}

        finite = bool(np.all(np.isfinite(vectors))) and bool(np.all(np.isfinite(relation)))
        entry_count = int(vectors.shape[0]) if vectors.ndim == 2 else int(len(lemmas))
        dim = int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 0

        return {
            "kind": "SemanticAttractorBank",
            "path": str(p),
            "exists": True,
            "entry_count": entry_count,
            "dim": dim,
            "all_dims_match": bool(vectors.ndim == 2 and len(lemmas) == entry_count),
            "finite": finite,
            "vector_norm_min": float(np.min(norms)) if norms.size else 0.0,
            "vector_norm_mean": float(np.mean(norms)) if norms.size else 0.0,
            "vector_norm_max": float(np.max(norms)) if norms.size else 0.0,
            "has_relation_matrix": bool(relation.size > 0),
            "relation_shape": list(map(int, relation.shape)),
            "metadata_format": metadata.get("format"),
            "metadata_version": metadata.get("version"),
            "metadata_stats": metadata.get("stats"),
            "pickle_fallback_used": bool(pickle_warning),
            "is_stable": bool(finite and vectors.ndim == 2 and len(lemmas) == entry_count),
        }


def inspect_bank(path: Path) -> Dict[str, Any]:
    ensure_import_paths()
    ok, mod, _err = import_module_safe("dictionary_lexicon_ingestor")
    if ok and mod is not None and callable(getattr(mod, "inspect_bank", None)):
        try:
            return getattr(mod, "inspect_bank")(Path(path))
        except Exception:
            return inspect_bank_fallback(path)
    return inspect_bank_fallback(path)


# =============================================================================
# Sentence builder and attractor call adapters
# =============================================================================

def call_sentence_builder(
    fn: Callable[..., Any],
    *,
    bank_path: Path,
    prompt: str,
    n: int,
    temperature: float = 0.0,
    seed: int = 0,
) -> Any:
    """
    Robust adapter for current and older sentence_builder.py signatures.
    """
    kwargs = {
        "bank_or_path": str(bank_path),
        "bank": str(bank_path),
        "bank_path": str(bank_path),
        "semantic_bank": str(bank_path),
        "semantic_bank_path": str(bank_path),
        "prompt": prompt,
        "text": prompt,
        "query": prompt,
        "n": int(n),
        "count": int(n),
        "num_sentences": int(n),
        "max_sentences": int(n),
        "temperature": float(temperature),
        "seed": int(seed),
    }

    try:
        return _call_with_supported_kwargs(fn, **kwargs)
    except TypeError as first_error:
        fallbacks = [
            lambda: fn(str(bank_path), prompt),
            lambda: fn(str(bank_path), prompt, n=int(n)),
            lambda: fn(str(bank_path), prompt, int(n)),
            lambda: fn(prompt, str(bank_path)),
            lambda: fn(prompt, str(bank_path), int(n)),
            lambda: fn(prompt),
        ]

        last_error: BaseException = first_error
        for attempt in fallbacks:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
                continue

        raise TypeError(
            f"Could not call sentence builder function {callable_name(fn)}. "
            f"First error: {first_error!r}; last fallback error: {last_error!r}"
        ) from last_error


def call_attractor_function(
    fn: Callable[..., Any],
    *,
    bank_path: Path,
    query: str,
    top_k: int,
    seed: int = 0,
) -> Any:
    kwargs = {
        "bank_or_path": str(bank_path),
        "bank": str(bank_path),
        "bank_path": str(bank_path),
        "semantic_bank": str(bank_path),
        "semantic_bank_path": str(bank_path),
        "query": query,
        "prompt": query,
        "text": query,
        "top_k": int(top_k),
        "k": int(top_k),
        "n": int(top_k),
        "seed": int(seed),
    }

    try:
        return _call_with_supported_kwargs(fn, **kwargs)
    except TypeError as first_error:
        fallbacks = [
            lambda: fn(str(bank_path), query, int(top_k)),
            lambda: fn(str(bank_path), query),
            lambda: fn(query, str(bank_path), int(top_k)),
            lambda: fn(query, str(bank_path)),
            lambda: fn(query),
        ]
        last_error: BaseException = first_error
        for attempt in fallbacks:
            try:
                return attempt()
            except TypeError as exc:
                last_error = exc
                continue
        raise TypeError(
            f"Could not call attractor function {callable_name(fn)}. "
            f"First error: {first_error!r}; last fallback error: {last_error!r}"
        ) from last_error


# =============================================================================
# Command implementations
# =============================================================================

def run_doctor(args: argparse.Namespace) -> int:
    ensure_import_paths()

    print("LK20 Local AI doctor")
    print("=" * 72)
    print(f"Python:       {platform.python_version()} ({platform.platform()})")
    print(f"Executable:   {sys.executable}")
    print(f"Project root: {PROJECT_ROOT} [{'OK' if PROJECT_ROOT.exists() else 'MISSING'}]")
    print(f"local_ai:     {LOCAL_AI_DIR} [{'OK' if LOCAL_AI_DIR.exists() else 'MISSING'}]")
    print(f"stack:        {STACK_DIR} [{'OK' if STACK_DIR.exists() else 'MISSING'}]")
    print()

    failed = False

    print("Expected files:")
    for name in ROOT_EXPECTED:
        p = PROJECT_ROOT / name
        ok = p.exists()
        failed = failed or not ok
        print(f"  [{'OK' if ok else 'MISSING':<7}] {p}")
    for name in LOCAL_EXPECTED:
        p = LOCAL_AI_DIR / name
        ok = p.exists()
        failed = failed or not ok
        print(f"  [{'OK' if ok else 'MISSING':<7}] {p}")
    for name in STACK_REQUIRED:
        p = STACK_DIR / name
        ok = p.exists()
        failed = failed or not ok
        print(f"  [{'OK' if ok else 'MISSING':<7}] {p}")
    for name in STACK_OPTIONAL_FILES:
        p = STACK_DIR / name
        ok = p.exists()
        print(f"  [{'OK' if ok else 'OPTIONAL':<7}] {p}")
    print()

    print("Tensor lexicon artifacts:")
    for p in (DEFAULT_TENSOR_LEXICON_DB, DEFAULT_TENSOR_LEXICON_CORE, DEFAULT_TENSOR_LEXICON_MANIFEST):
        ok = p.exists()
        failed = failed or not ok
        detail = f"{p.stat().st_size} bytes" if ok else "missing"
        status_line(str(p), ok, detail)
    print()

    print("Required import checks:")
    for name in ROOT_IMPORTS_REQUIRED + LOCAL_IMPORTS_REQUIRED + STACK_IMPORTS_REQUIRED:
        ok, mod, err = import_module_safe(name)
        if ok and mod is not None:
            status_line(name, True, module_file(mod))
        else:
            failed = True
            status_line(name, False, err or "")
    print()

    print("Optional import checks:")
    for name in LOCAL_IMPORTS_OPTIONAL + STACK_IMPORTS_OPTIONAL:
        ok, mod, err = import_module_safe(name)
        if ok and mod is not None:
            status_line(name, True, module_file(mod))
        else:
            status_line(name, False, err or "", warn=True)
    print()

    print("Adapter check:")
    try:
        st = adapter_status()
        active = st.get("status") in {"active", "limited"}
        failed = failed or not active
        tensor = st.get("tensor_lexicon") or {}
        entropy = st.get("entropy_nlp") or {}
        status_line("local_ai_adapter.get_status()", active, f"status={st.get('status')}; provider={st.get('provider')}")
        status_line("tensor_lexicon", bool(tensor.get("available")), f"entries={tensor.get('entry_count')}; vectors_mode={tensor.get('vectors_mode')}")
        status_line("entropy_nlp", bool(entropy.get("ok")), f"status={entropy.get('status')}")
    except Exception as exc:
        failed = True
        status_line("local_ai_adapter.get_status()", False, repr(exc))
    print()

    if args.compile:
        print("Compile checks:")
        compile_targets = [
            *(PROJECT_ROOT / x for x in ROOT_EXPECTED),
            *(LOCAL_AI_DIR / x for x in LOCAL_EXPECTED),
            *(STACK_DIR / x for x in STACK_REQUIRED),
        ]
        for p in compile_targets:
            if not p.exists():
                failed = True
                print(f"  [{'MISSING':<7}] {p.name}")
                continue
            try:
                ok = compileall.compile_file(str(p), quiet=1, force=True)
                failed = failed or not ok
                print(f"  [{'OK' if ok else 'FAIL':<7}] {p.name}")
            except Exception as exc:
                failed = True
                print(f"  [{'FAIL':<7}] {p.name} -> {exc!r}")
        print()

    bank = resolve_bank_path(args.bank)
    print(f"Default semantic bank: {bank} [{'OK' if bank.exists() else 'MISSING'}]")

    candidates = []
    for raw in (
        DEFAULT_DICTIONARY,
        LOCAL_AI_DIR / "dictionary.json",
        PROJECT_ROOT / "dictionary.json",
        Path.cwd() / "dictionary.json",
    ):
        p = raw.resolve()
        if p not in candidates and p.exists():
            candidates.append(p)

    if candidates:
        print("Dictionary candidates:")
        for p in candidates:
            print(f"  [{'OK':<7}] {p}")
    else:
        print("Dictionary candidates: none found in default locations")

    print()
    print(f"Doctor result: {'FAIL' if failed else 'OK'}")
    return 1 if failed else 0


def run_ingest(args: argparse.Namespace) -> int:
    ensure_import_paths()

    ok, mod, err = import_module_safe("dictionary_lexicon_ingestor")
    if not ok or mod is None:
        eprint(f"Could not import dictionary_lexicon_ingestor: {err}")
        return 1

    fn = getattr(mod, "ingest", None)
    if not callable(fn):
        eprint("dictionary_lexicon_ingestor.py does not expose ingest(...).")
        return 1

    input_path = resolve_input_path(args.input)
    out_path = resolve_path(args.out, default=DEFAULT_BANK)
    resources_dir = resolve_path(args.resources_dir, default=DEFAULT_RESOURCES_DIR)

    if input_path is not None and not input_path.exists():
        eprint(f"Dictionary input does not exist: {input_path}")
        return 1

    try:
        result = _call_with_supported_kwargs(
            fn,
            input_path=input_path,
            input=input_path,
            out=out_path,
            out_path=out_path,
            use_installed=bool(args.use_installed),
            install_missing=bool(args.install_missing),
            resources_dir=resources_dir,
            dim=int(args.dim),
            max_relation_matrix=int(args.max_relation_matrix),
            deduplicate=not bool(args.no_deduplicate),
            include_relation_matrix=not bool(args.no_relation_matrix),
            seed=int(args.seed),
            quiet=bool(args.quiet),
        )
        if args.json:
            print_json(result)
        return 0
    except Exception:
        eprint("Lexicon ingestion failed.")
        traceback.print_exc()
        return 1


def run_install_resources(args: argparse.Namespace) -> int:
    ensure_import_paths()

    ok, mod, err = import_module_safe("dictionary_lexicon_ingestor")
    if not ok or mod is None:
        eprint(f"Could not import dictionary_lexicon_ingestor: {err}")
        return 1

    fn = getattr(mod, "install_resources", None)
    if not callable(fn):
        eprint("dictionary_lexicon_ingestor.py does not expose install_resources(...).")
        return 1

    resources_dir = resolve_path(args.resources_dir, default=DEFAULT_RESOURCES_DIR)

    try:
        result = _call_with_supported_kwargs(
            fn,
            resources_dir=resources_dir,
            force=bool(args.force),
            timeout=int(args.timeout),
            quiet=bool(args.quiet),
        )
        if args.json:
            print_json(result)
        return 0
    except Exception:
        eprint("Resource installation failed.")
        traceback.print_exc()
        return 1


def run_inspect_bank(args: argparse.Namespace) -> int:
    bank = resolve_bank_path(args.bank)
    result = inspect_bank(bank)
    print_json(result)
    return 0 if result.get("exists") and result.get("is_stable") else 1


def _raw_sentence_build(prompt: str, args: argparse.Namespace) -> int:
    ensure_import_paths()

    bank_path = resolve_bank_path(args.bank)
    if not bank_path.exists() and not args.allow_missing_bank:
        eprint(f"Semantic bank does not exist: {bank_path}")
        eprint("Run ingest first, or pass --allow-missing-bank if your builder does not require a bank.")
        return 1

    ok, mod, err = import_module_safe("sentence_builder")
    if not ok or mod is None:
        eprint(f"Could not import sentence_builder: {err}")
        return 1

    fn = get_first_callable(
        mod,
        [
            "build_sentences",
            "build_sentence",
            "generate_sentences",
            "generate_sentence",
            "sentence_builder",
            "build",
            "run",
        ],
    )

    if fn is None:
        eprint("sentence_builder.py exposes no supported build function.")
        eprint("Expected one of: build_sentences, build_sentence, generate_sentences, generate_sentence, build, run.")
        return 1

    if not args.quiet:
        print("Mode:   raw sentence_builder")
        print(f"Prompt: {prompt}")
        print(f"Bank:   {bank_path}")

    try:
        result = call_sentence_builder(
            fn,
            bank_path=bank_path,
            prompt=prompt,
            n=int(args.n),
            temperature=float(args.temperature),
            seed=int(args.seed),
        )
        format_result(result, json_output=bool(args.json))
        return 0
    except Exception:
        eprint("Sentence builder function failed.")
        traceback.print_exc()
        return 1


def run_build(args: argparse.Namespace) -> int:
    """
    Safe-by-default build command.

    Without --raw, this command routes through local_ai_adapter.suggest_improvements
    and returns factual advisory output. Use --raw to call sentence_builder.py
    directly.
    """
    prompt = str(args.prompt or "").strip()
    if not prompt:
        eprint("Prompt is required. Use --prompt \"...\".")
        return 1

    if bool(args.raw):
        return _raw_sentence_build(prompt, args)

    adapter = get_local_adapter()
    try:
        if not args.quiet:
            print("Mode:   safe adapter advisory")
            print(f"Prompt: {prompt}")
        if args.json:
            result = {
                "mode": "safe_adapter_advisory",
                "prompt": prompt,
                "advisory": adapter.suggest_improvements(prompt),
                "diagnostic": adapter.analyze_generated_text(prompt, profile=str(args.profile)),
            }
            print_json(result)
        else:
            print(adapter.suggest_improvements(prompt))
        return 0
    except Exception:
        eprint("Safe build/advisory failed.")
        traceback.print_exc()
        return 1
    finally:
        adapter.close()


def fallback_query_vector(query: str, dim: int, seed: int = 0) -> np.ndarray:
    """
    Local fallback vectorizer for attractor search. Prefer
    SemanticAttractorBank.nearest(...) when available.
    """
    v = np.zeros((dim,), dtype=np.float64)

    def stable_hash(text: str) -> int:
        h = (1469598103934665603 ^ int(seed)) & 0xFFFFFFFFFFFFFFFF
        for b in text.encode("utf-8", errors="ignore"):
            h ^= int(b)
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return int(h)

    toks = re.findall(r"[a-zA-Z][a-zA-Z'\-]*|\d+(?:\.\d+)?", query.lower().strip())
    for tok in toks:
        h = stable_hash(tok)
        idx = h % dim
        sign = -1.0 if ((h >> 63) & 1) else 1.0
        v[idx] += sign
        h2 = stable_hash(f"lemma:{tok}")
        v[h2 % dim] += 0.5 * (-1.0 if ((h2 >> 63) & 1) else 1.0)

    n = float(np.linalg.norm(v))
    if n > 1e-12:
        v /= n
    return v.astype(np.float32)


def fallback_attractor_search(bank_path: Path, query: str, *, top_k: int = 8, seed: int = 0) -> List[Dict[str, Any]]:
    ensure_import_paths()

    try:
        from semantic_attractors import SemanticAttractorBank  # type: ignore

        bank = SemanticAttractorBank.load_npz(bank_path)
        rows = bank.nearest(query, top_k=int(top_k))
        out: List[Dict[str, Any]] = []
        for rank, (key, score) in enumerate(rows, start=1):
            attr = bank.attractors.get(key)
            meta = getattr(attr, "metadata", {}) or {}
            out.append(
                {
                    "rank": rank,
                    "key": key,
                    "lemma": getattr(attr, "label", key) if attr is not None else key,
                    "pos": meta.get("pos"),
                    "gloss": meta.get("gloss"),
                    "score": float(score),
                    "source": "SemanticAttractorBank.nearest",
                }
            )
        return out
    except Exception:
        pass

    with np.load(bank_path, allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
        lemmas = data["lemmas"] if "lemmas" in data else data["words"]
        pos = data["pos"] if "pos" in data else np.asarray([""] * len(lemmas), dtype=str)
        glosses = data["glosses"] if "glosses" in data else data["definitions"] if "definitions" in data else np.asarray([""] * len(lemmas), dtype=str)

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return []

    q = fallback_query_vector(query, vectors.shape[1], seed=seed)
    scores = vectors @ q
    order = np.argsort(-scores)[: max(1, int(top_k))]

    out = []
    for idx in order:
        i = int(idx)
        out.append(
            {
                "rank": len(out) + 1,
                "lemma": str(lemmas[i]),
                "pos": str(pos[i]) if i < len(pos) else "",
                "gloss": str(glosses[i]) if i < len(glosses) else "",
                "score": float(scores[i]),
                "source": "fallback_query_vector",
            }
        )
    return out


def run_attractors(args: argparse.Namespace) -> int:
    ensure_import_paths()

    query = str(args.query or args.prompt or "").strip()
    if not query:
        eprint("Query is required. Use --query \"...\".")
        return 1

    bank_path = resolve_bank_path(args.bank)
    if not bank_path.exists():
        eprint(f"Semantic bank does not exist: {bank_path}")
        return 1

    ok, mod, err = import_module_safe("semantic_attractors")
    if not ok or mod is None:
        eprint(f"Could not import semantic_attractors: {err}")
        return 1

    fn = get_first_callable(
        mod,
        [
            "find_attractors",
            "query_attractors",
            "nearest",
            "search",
            "attractors",
            "run",
        ],
    )

    if fn is None:
        result = fallback_attractor_search(bank_path, query, top_k=int(args.top_k), seed=int(args.seed))
        format_result(result, json_output=bool(args.json))
        return 0

    try:
        result = call_attractor_function(
            fn,
            bank_path=bank_path,
            query=query,
            top_k=int(args.top_k),
            seed=int(args.seed),
        )
        format_result(result, json_output=bool(args.json))
        return 0
    except Exception:
        # Prefer a working fallback over failing the CLI.
        try:
            result = fallback_attractor_search(bank_path, query, top_k=int(args.top_k), seed=int(args.seed))
            format_result(result, json_output=bool(args.json))
            return 0
        except Exception:
            eprint("Semantic attractor function failed.")
            traceback.print_exc()
            return 1


def run_chat(args: argparse.Namespace) -> int:
    ensure_import_paths()

    print("LK20 local_ai interactive CLI")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Mode: {'raw sentence_builder' if args.raw else 'safe adapter advisory'}")
    print("Type 'help' for commands. Type 'exit' to quit.")
    print()

    bank_path = resolve_bank_path(args.bank)
    build_fn: Optional[Callable[..., Any]] = None

    if args.raw:
        if not bank_path.exists() and not args.allow_missing_bank:
            eprint(f"Semantic bank does not exist: {bank_path}")
            eprint("Run ingest first, or pass --allow-missing-bank.")
            return 1

        ok, sb_mod, sb_err = import_module_safe("sentence_builder")
        if not ok or sb_mod is None:
            eprint(f"Could not import sentence_builder: {sb_err}")
            return 1

        build_fn = get_first_callable(
            sb_mod,
            [
                "build_sentences",
                "build_sentence",
                "generate_sentences",
                "generate_sentence",
                "sentence_builder",
                "build",
                "run",
            ],
        )
        if build_fn is None:
            eprint("sentence_builder.py exposes no supported build function.")
            return 1

    adapter = None if args.raw else get_local_adapter()

    try:
        while True:
            try:
                line = input("local_ai> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            low = line.lower().strip()
            if low in {"exit", "quit", ":q"}:
                break

            if low in {"help", ":help", "?"}:
                print("Commands:")
                print("  help                    Show this help.")
                print("  exit                    Quit.")
                print("  /bank                   Inspect current semantic bank.")
                print("  /status                 Show adapter status.")
                print("  /lexicon <term> [ctx]   Adapter lexicon lookup. Use ctx after term optionally.")
                print("  /alias <term>           Resolve alias/inflected form.")
                print("  /attractors <query>     Search semantic attractors.")
                print("  <text>                  Safe advisory by default; raw generation if --raw was used.")
                continue

            if low == "/bank":
                print_json(inspect_bank(bank_path))
                continue

            if low == "/status":
                try:
                    print_json(adapter_status())
                except Exception:
                    traceback.print_exc()
                continue

            if low.startswith("/lexicon"):
                payload = line[len("/lexicon"):].strip()
                if not payload:
                    print("Usage: /lexicon <term> [context words...]")
                    continue
                pieces = payload.split(maxsplit=1)
                term = pieces[0]
                context = payload if len(pieces) > 1 else term
                try:
                    a = adapter or get_local_adapter()
                    rows = a.lookup_lexicon(term, limit=int(args.top_k), context=context)
                    _format_lexicon_rows(rows, json_output=False)
                    if adapter is None:
                        a.close()
                except Exception:
                    traceback.print_exc()
                continue

            if low.startswith("/alias"):
                term = line[len("/alias"):].strip()
                if not term:
                    print("Usage: /alias <term>")
                    continue
                try:
                    a = adapter or get_local_adapter()
                    rows = a.resolve_alias(term, limit=int(args.top_k), context=term)
                    _format_lexicon_rows(rows, json_output=False)
                    if adapter is None:
                        a.close()
                except Exception:
                    traceback.print_exc()
                continue

            if low.startswith("/attractors"):
                query = line[len("/attractors"):].strip()
                if not query:
                    print("Usage: /attractors <query>")
                    continue
                try:
                    result = fallback_attractor_search(bank_path, query, top_k=int(args.top_k), seed=int(args.seed))
                    format_result(result, json_output=False)
                except Exception:
                    traceback.print_exc()
                continue

            try:
                if args.raw and build_fn is not None:
                    result = call_sentence_builder(
                        build_fn,
                        bank_path=bank_path,
                        prompt=line,
                        n=int(args.n),
                        temperature=float(args.temperature),
                        seed=int(args.seed),
                    )
                    format_result(result, json_output=False)
                else:
                    if adapter is None:
                        adapter = get_local_adapter()
                    print(adapter.suggest_improvements(line))
            except Exception:
                print("CLI request failed:")
                traceback.print_exc()

    finally:
        if adapter is not None:
            adapter.close()

    return 0


def run_script(args: argparse.Namespace) -> int:
    """
    Run one of the local scripts directly, while preserving local_ai and stack
    import paths.
    """
    ensure_import_paths()

    script_name = str(args.script).strip()
    allowed = {
        "semantic_attractors": LOCAL_AI_DIR / "semantic_attractors.py",
        "semantic_attractors.py": LOCAL_AI_DIR / "semantic_attractors.py",
        "dictionary_lexicon_ingestor": LOCAL_AI_DIR / "dictionary_lexicon_ingestor.py",
        "dictionary_lexicon_ingestor.py": LOCAL_AI_DIR / "dictionary_lexicon_ingestor.py",
        "sentence_builder": LOCAL_AI_DIR / "sentence_builder.py",
        "sentence_builder.py": LOCAL_AI_DIR / "sentence_builder.py",
    }

    script_path = allowed.get(script_name)
    if script_path is None:
        candidate = resolve_path(script_name)
        if candidate.exists() and candidate.suffix == ".py":
            script_path = candidate
        else:
            eprint(f"Unknown script: {script_name}")
            eprint(f"Allowed: {', '.join(sorted(allowed))}")
            return 1

    if not script_path.exists():
        eprint(f"Script does not exist: {script_path}")
        return 1

    cmd = [sys.executable, str(script_path), *list(args.args or [])]
    if args.print_command:
        print(" ".join(shlex.quote(x) for x in cmd))

    env = os.environ.copy()
    py_path_parts = [str(PROJECT_ROOT), str(LOCAL_AI_DIR), str(STACK_DIR)]
    if env.get("PYTHONPATH"):
        py_path_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_path_parts)

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(proc.returncode)


# =============================================================================
# CLI parser
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local_ai.py",
        description="LK20 / Akkurat Local AI command-line contact point.",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print local_ai.py version metadata and exit.",
    )

    sub = parser.add_subparsers(dest="command")

    p_doc = sub.add_parser("doctor", help="Check local AI, adapter, tensor lexicon, and AtomTN stack health.")
    p_doc.add_argument("--compile", action="store_true", help="Compile-check key Python files.")
    p_doc.add_argument("--bank", default=str(DEFAULT_BANK), help="Semantic bank path.")

    p_adapt = sub.add_parser("adapter-status", aliases=["status"], help="Print root local_ai_adapter.py status.")
    p_adapt.add_argument("--summary", action="store_true", help="Print compact status summary instead of full JSON.")
    p_adapt.add_argument("--json", action="store_true", help="Accepted for symmetry; full JSON is already the default.")
    p_adapt.set_defaults(command="adapter-status", handler=run_adapter_status)

    p_lex = sub.add_parser("lexicon", help="Context-aware tensor lexicon lookup through local_ai_adapter.py.")
    p_lex.add_argument("--term", "-t", required=True, help="Lookup term, e.g. cat.")
    p_lex.add_argument("--context", "-c", default=None, help="Optional context for WSD, e.g. 'cat animal'.")
    p_lex.add_argument("--pos", default=None, help="Optional POS filter.")
    p_lex.add_argument("--limit", "-n", type=int, default=5, help="Maximum rows.")
    p_lex.add_argument("--relations", action="store_true", help="Attach relations where available.")
    p_lex.add_argument("--no-aliases", action="store_true", help="Disable alias fallback.")
    p_lex.add_argument("--json", action="store_true", help="Print JSON result.")

    p_alias = sub.add_parser("alias", help="Resolve alias/inflected form through local_ai_adapter.py.")
    p_alias.add_argument("--term", "-t", required=True, help="Alias or inflected form, e.g. running.")
    p_alias.add_argument("--context", "-c", default=None, help="Optional context.")
    p_alias.add_argument("--pos", default=None, help="Optional POS filter.")
    p_alias.add_argument("--limit", "-n", type=int, default=5, help="Maximum rows.")
    p_alias.add_argument("--json", action="store_true", help="Print JSON result.")

    p_adv = sub.add_parser("advisory", help="Safe lexicon/entropy advisory through local_ai_adapter.py.")
    p_adv.add_argument("--text", "-t", default=None, help="Text to diagnose.")
    p_adv.add_argument("--prompt", "-p", default=None, help="Alias for --text.")
    p_adv.add_argument("--profile", default="curriculum", help="Entropy profile.")
    p_adv.add_argument("--json", action="store_true", help="Print JSON result.")

    p_wsd = sub.add_parser("wsd-test", help="Run context-aware WSD regression smoke tests.")
    p_wsd.add_argument("--json", action="store_true", help="Print JSON result.")

    p_ing = sub.add_parser("ingest", help="Ingest dictionary resources into a semantic bank.")
    p_ing.add_argument("--input", "-i", default=None, help="Input dictionary path.")
    p_ing.add_argument("--out", "-o", default=str(DEFAULT_BANK), help="Output semantic bank .npz path.")
    p_ing.add_argument("--use-installed", action="store_true", help="Also ingest installed resources.")
    p_ing.add_argument("--install-missing", action="store_true", help="Install resources before ingesting.")
    p_ing.add_argument("--resources-dir", default=str(DEFAULT_RESOURCES_DIR), help="Resource directory.")
    p_ing.add_argument("--dim", type=int, default=64, help="Semantic vector dimension.")
    p_ing.add_argument("--max-relation-matrix", type=int, default=4000, help="Maximum dense relation matrix entries.")
    p_ing.add_argument("--no-deduplicate", action="store_true", help="Disable deduplication.")
    p_ing.add_argument("--no-relation-matrix", action="store_true", help="Do not build dense relation matrix.")
    p_ing.add_argument("--seed", type=int, default=0, help="Deterministic vector seed.")
    p_ing.add_argument("--quiet", action="store_true", help="Suppress progress output from ingestor.")
    p_ing.add_argument("--json", action="store_true", help="Print JSON result.")

    p_res = sub.add_parser("install-resources", help="Install optional open/public lexical resources.")
    p_res.add_argument("--resources-dir", default=str(DEFAULT_RESOURCES_DIR), help="Resource directory.")
    p_res.add_argument("--force", action="store_true", help="Redownload resources.")
    p_res.add_argument("--timeout", type=int, default=60, help="Download timeout seconds.")
    p_res.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    p_res.add_argument("--json", action="store_true", help="Print JSON result.")

    p_ins = sub.add_parser("inspect-bank", help="Inspect a semantic bank .npz.")
    p_ins.add_argument("--bank", default=str(DEFAULT_BANK), help="Semantic bank path.")

    p_build = sub.add_parser("build", help="Safe advisory by default; use --raw for direct sentence_builder generation.")
    p_build.add_argument("--bank", default=str(DEFAULT_BANK), help="Semantic bank path.")
    p_build.add_argument("--prompt", "-p", required=True, help="Prompt text.")
    p_build.add_argument("-n", type=int, default=1, help="Number of outputs requested where supported.")
    p_build.add_argument("--temperature", type=float, default=0.0, help="Generation temperature where supported.")
    p_build.add_argument("--seed", type=int, default=0, help="Deterministic seed where supported.")
    p_build.add_argument("--profile", default="curriculum", help="Entropy profile for safe mode.")
    p_build.add_argument("--raw", action="store_true", help="Call sentence_builder.py directly. Use only for explicit generation tests.")
    p_build.add_argument("--allow-missing-bank", action="store_true", help="Allow raw builder call even if bank is missing.")
    p_build.add_argument("--quiet", action="store_true", help="Suppress prompt/bank header.")
    p_build.add_argument("--json", action="store_true", help="Print JSON result.")

    p_chat = sub.add_parser("chat", help="Interactive local AI CLI. Safe advisory by default; use --raw for direct builder.")
    p_chat.add_argument("--bank", default=str(DEFAULT_BANK), help="Semantic bank path.")
    p_chat.add_argument("-n", type=int, default=1, help="Number of outputs requested where supported.")
    p_chat.add_argument("--top-k", type=int, default=8, help="Attractor/lexicon count for slash commands.")
    p_chat.add_argument("--temperature", type=float, default=0.0, help="Generation temperature where supported.")
    p_chat.add_argument("--seed", type=int, default=0, help="Deterministic seed where supported.")
    p_chat.add_argument("--raw", action="store_true", help="Use direct sentence_builder generation instead of safe advisory.")
    p_chat.add_argument("--allow-missing-bank", action="store_true", help="Allow raw chat even if bank is missing.")

    p_att = sub.add_parser("attractors", help="Search semantic attractors.")
    p_att.add_argument("--bank", default=str(DEFAULT_BANK), help="Semantic bank path.")
    p_att.add_argument("--query", "-q", default=None, help="Query text.")
    p_att.add_argument("--prompt", default=None, help="Alias for --query.")
    p_att.add_argument("--top-k", "-k", type=int, default=8, help="Number of attractors.")
    p_att.add_argument("--seed", type=int, default=0, help="Deterministic seed.")
    p_att.add_argument("--json", action="store_true", help="Print JSON result.")

    p_run = sub.add_parser("run-script", help="Run an underlying local_ai script.")
    p_run.add_argument("--print-command", action="store_true", help="Print subprocess command before running.")
    p_run.add_argument("script", help="Script name or path.")
    p_run.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the script.")

    return parser


def print_version() -> None:
    info = {
        "kind": "LK20.local_ai",
        "version": VERSION,
        "script": str(SCRIPT_PATH),
        "project_root": str(PROJECT_ROOT),
        "local_ai_dir": str(LOCAL_AI_DIR),
        "stack_dir": str(STACK_DIR),
        "root_adapter": str(ROOT_ADAPTER),
        "python": sys.version,
    }
    print_json(info)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_import_paths()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print_version()
        return 0

    if args.command is None:
        parser.print_help()
        return 0

    handler = getattr(args, "handler", None)
    if callable(handler):
        return int(handler(args))

    if args.command == "doctor":
        return run_doctor(args)

    if args.command == "adapter-status":
        return run_adapter_status(args)

    if args.command == "lexicon":
        return run_lexicon(args)

    if args.command == "alias":
        return run_alias(args)

    if args.command == "advisory":
        return run_advisory(args)

    if args.command == "wsd-test":
        return run_wsd_test(args)

    if args.command == "ingest":
        return run_ingest(args)

    if args.command == "install-resources":
        return run_install_resources(args)

    if args.command == "inspect-bank":
        return run_inspect_bank(args)

    if args.command == "build":
        return run_build(args)

    if args.command == "chat":
        return run_chat(args)

    if args.command == "attractors":
        return run_attractors(args)

    if args.command == "run-script":
        return run_script(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PROJECT_ROOT",
    "LOCAL_AI_DIR",
    "STACK_DIR",
    "DEFAULT_BANK",
    "DEFAULT_TENSOR_LEXICON_DB",
    "resolve_path",
    "resolve_bank_path",
    "inspect_bank",
    "get_local_adapter",
    "adapter_status",
    "call_sentence_builder",
    "call_attractor_function",
    "run_doctor",
    "run_adapter_status",
    "run_lexicon",
    "run_alias",
    "run_advisory",
    "run_wsd_test",
    "run_ingest",
    "run_build",
    "run_chat",
    "run_attractors",
    "main",
]
