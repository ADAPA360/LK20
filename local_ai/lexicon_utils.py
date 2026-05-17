#!/usr/bin/env python3
# lexicon_utils.py
"""
Shared lexical utilities for LK20 vectorization and normalization.
"""

from __future__ import annotations
import hashlib
import json
import math
import re
import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Sequence

# Constants matching legacy ingestor for determinism
DEFAULT_DIM = 128
EPS = 1e-12

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")

POS_ALIASES = {
    "n": "noun", "noun": "noun", "proper noun": "noun", "proper name": "noun",
    "v": "verb", "verb": "verb",
    "adj": "adjective", "adjective": "adjective", "a": "adjective",
    "adv": "adverb", "adverb": "adverb",
    "prep": "preposition", "preposition": "preposition",
    "pron": "pronoun", "pronoun": "pronoun",
    "det": "determiner", "determiner": "determiner",
    "conj": "conjunction", "conjunction": "conjunction",
    "intj": "interjection", "interjection": "interjection",
}

def canonical_text(x: Any) -> str:
    if x is None: return ""
    s = str(x)
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip()

def normalize_pos(pos: Any) -> str:
    p = canonical_text(pos).lower().replace("_", " ").replace(".", "")
    p = re.sub(r"\s+", " ", p).strip()
    return POS_ALIASES.get(p, p)

def stable_hash_int(s: str, seed: int = 0) -> int:
    h = (2166136261 ^ int(seed)) & 0xFFFFFFFF
    for b in str(s).encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)

def tokenize(text: str, min_len: int = 1) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(canonical_text(text)) if len(m.group(0)) >= min_len]

def char_ngrams(text: str, n: int) -> List[str]:
    s = str(text)
    if len(s) < n: return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]

def add_feature(v: np.ndarray, token: str, *, weight: float, seed: int = 0) -> None:
    dim = int(v.shape[0])
    h = stable_hash_int(token, seed=seed)
    idx1 = h % dim
    idx2 = (h >> 16) % dim
    sign1 = 1.0 if ((h >> 32) & 1) == 0 else -1.0
    sign2 = 1.0 if ((h >> 33) & 1) == 0 else -1.0
    v[idx1] += float(weight) * sign1
    v[idx2] += float(weight) * 0.5 * sign2

def vectorize_lexicon_entry(
    lemma: str, 
    pos: str, 
    gloss: str, 
    source: str = "kaikki",
    relations: Optional[Mapping[str, List[str]]] = None,
    weight: float = 1.0,
    dim: int = DEFAULT_DIM, 
    seed: int = 0, 
    min_token_len: int = 1
) -> np.ndarray:
    dim = int(max(4, dim))
    v = np.zeros((dim,), dtype=np.float64)
    l_can = canonical_text(lemma).lower()
    p_can = normalize_pos(pos)
    g_can = canonical_text(gloss).lower()
    
    add_feature(v, f"lemma:{l_can}", weight=3.0, seed=seed)
    add_feature(v, f"pos:{p_can}", weight=0.75, seed=seed)
    add_feature(v, f"source:{source}", weight=0.15, seed=seed)

    compact = re.sub(r"[^a-z0-9]+", "_", l_can).strip("_")
    for n in (2, 3, 4):
        for gram in char_ngrams(compact, n):
            add_feature(v, f"char{n}:{gram}", weight=0.8 / n, seed=seed)

    tokens = tokenize(g_can, min_len=min_token_len)
    for tok in tokens:
        add_feature(v, f"gloss_tok:{tok}", weight=0.45, seed=seed)
    for a, b in zip(tokens, tokens[1:]):
        add_feature(v, f"gloss_bigram:{a}_{b}", weight=0.25, seed=seed)

    if relations:
        for rel_name, vals in sorted(relations.items()):
            add_feature(v, f"relname:{rel_name}", weight=0.35, seed=seed)
            for target in vals[:16]:
                add_feature(v, f"reltarget:{rel_name}:{canonical_text(target).lower()}", weight=0.2, seed=seed)

    add_feature(v, "entry_weight", weight=0.05 * float(weight), seed=seed)

    nrm = float(np.linalg.norm(v))
    if nrm <= EPS or not np.isfinite(nrm):
        for i in range(dim):
            raw = stable_hash_int(f"{l_can}:{i}", seed=seed)
            v[i] = ((raw % 20001) / 10000.0) - 1.0
        nrm = float(np.linalg.norm(v))

    if nrm > EPS:
        v = v / nrm
    return v.astype(np.float32)

def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)): return obj
    if isinstance(obj, float): return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, Mapping): return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [json_safe(v) for v in obj]
    return str(obj)
