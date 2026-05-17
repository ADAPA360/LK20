#!/usr/bin/env python3
# entropy_nlp.py
r"""
LK20 / Akkurat Local AI — Entropy NLP Diagnostics
=================================================

Production role
---------------
This module adds a local, deterministic, governance-safe NLP layer inspired by
recent work on semantic chunking and natural-language entropy. It is designed to
live at:

    C:\Users\ali_z\ANU AI\LK20\local_ai\entropy_nlp.py

The module does **not** mutate the LK20 digital twin. It provides read-only
analysis, candidate scoring, and generation-control hints that can be consumed by
`sentence_builder.py`, `local_ai_adapter.py`, or a future `/api/ai/...` endpoint.

Core idea
---------
Natural text can be treated as a hierarchy of semantic chunks. A text whose
chunks are too flat is usually under-structured; a text whose chunk tree is too
branchy or too irregular can become hard to follow. This module converts that
idea into practical production signals:

- recursive semantic-ish chunk trees using deterministic local heuristics;
- random K-ary-tree negative log-likelihood and entropy-rate estimates;
- style/complexity profiles for curriculum, explanation, narrative, academic,
  and poetic text;
- candidate re-ranking for generated text;
- warnings for common local-generation failure modes such as repetition,
  missing terminal punctuation, and unknown verbification.

Design constraints
------------------
- NumPy is optional. The module runs with the Python standard library alone.
- No network calls.
- No model calls.
- No filesystem writes except explicit CLI output if a caller redirects stdout.
- No mutation of canonical LK20 state.
- Public API is JSON-safe.

Suggested integration points
----------------------------
1. In `sentence_builder.py`, generate multiple candidates, then call:

       from entropy_nlp import rerank_texts
       ranked = rerank_texts(candidates, context=prompt, profile="curriculum")

2. In `local_ai_adapter.py`, expose:

       adapter.ai_entropy_status()
       adapter.analyze_generated_text(text)

3. In `lk20_server.py`, optionally expose status-only endpoints:

       GET  /api/ai/entropy/status
       POST /api/ai/entropy/analyze
       POST /api/ai/entropy/rerank

The returned diagnostics are advisory and should be labelled as such in the UI.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # Optional: used only for small vector summaries if available.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - standard-library fallback is supported.
    _np = None


# =============================================================================
# Constants
# =============================================================================

MODULE_VERSION = "entropy-nlp-v1.0"
EPS = 1e-12
DEFAULT_K = 4
DEFAULT_MAX_DEPTH = 32
DEFAULT_MIN_SPLIT_TOKENS = 2
DEFAULT_TARGET_N_FOR_HK = 384

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*|[^\w\s]", re.UNICODE)
SENTENCE_END = {".", "!", "?"}
PUNCTUATION = {".", "!", "?", ";", ":", ",", "—", "–", "-", "(" , ")", "[", "]", "{" , "}"}

COORDINATORS = {"and", "or", "but", "nor", "yet", "so"}
SUBORDINATORS = {
    "because", "although", "while", "when", "if", "unless", "since", "before",
    "after", "where", "whereas", "that", "which", "who", "whom", "whose",
}
DISCOURSE_MARKERS = {
    "therefore", "however", "moreover", "finally", "first", "second", "third",
    "also", "instead", "meanwhile", "consequently", "nevertheless", "notably",
}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"}
AUXILIARIES = {
    "am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must",
}
COMMON_VERBS = {
    "be", "am", "is", "are", "was", "were", "have", "has", "had", "do", "does", "did",
    "make", "makes", "made", "say", "says", "said", "go", "goes", "went", "see", "sees",
    "saw", "seen", "use", "uses", "used", "learn", "learns", "learned", "teach", "teaches",
    "taught", "write", "writes", "wrote", "read", "reads", "show", "shows", "shown", "explain",
    "explains", "explained", "describe", "describes", "described", "compare", "compares",
    "compared", "analyze", "analyzes", "analysed", "analyse", "understand", "understands",
    "understood", "work", "works", "worked", "build", "builds", "built", "create", "creates",
    "created", "develop", "develops", "developed", "support", "supports", "supported", "connect",
    "connects", "connected", "improve", "improves", "improved", "need", "needs", "needed",
    "should", "must", "can", "could", "would", "will", "help", "helps", "helped", "give", "gives",
    "gave", "take", "takes", "took", "find", "finds", "found", "mean", "means", "meant",
}
COMMON_ADJECTIVES = {
    "good", "better", "best", "clear", "simple", "complex", "gentle", "strong", "weak", "local",
    "semantic", "canonical", "diagnostic", "private", "public", "safe", "stable", "coherent",
}

# Entropy regimes are intentionally broad. They are not treated as hard truth;
# they are used as production-control priors.
PROFILE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "children": {
        "target_k": 2,
        "target_entropy_nats": 1.25,
        "entropy_band": [0.8, 1.8],
        "description": "simple, low branching, high redundancy",
    },
    "simple": {
        "target_k": 3,
        "target_entropy_nats": 1.9,
        "entropy_band": [1.2, 2.4],
        "description": "plain-language explanation",
    },
    "curriculum": {
        "target_k": 4,
        "target_entropy_nats": 2.5,
        "entropy_band": [1.8, 3.0],
        "description": "regular structured prose suitable for curriculum explanations",
    },
    "regular": {
        "target_k": 4,
        "target_entropy_nats": 2.5,
        "entropy_band": [1.8, 3.0],
        "description": "regular narrative/expository text",
    },
    "academic": {
        "target_k": 5,
        "target_entropy_nats": 2.9,
        "entropy_band": [2.2, 3.4],
        "description": "dense expository or technical writing",
    },
    "poetic": {
        "target_k": 6,
        "target_entropy_nats": 3.2,
        "entropy_band": [2.5, 4.0],
        "description": "high-complexity, atypical, figurative text",
    },
}


# =============================================================================
# JSON safety and token utilities
# =============================================================================


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if is_dataclass(obj):
        return json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if _np is not None:
        try:
            if isinstance(obj, _np.generic):
                return json_safe(obj.item())
            if isinstance(obj, _np.ndarray):
                return _np.nan_to_num(obj).astype(float).tolist()
        except Exception:
            pass
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return json_safe(vars(obj))
    return str(obj)


def canonical_text(text: Any) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\t\n]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(text: str) -> List[str]:
    return [m.group(0) for m in TOKEN_RE.finditer(canonical_text(text))]


def token_is_word(tok: str) -> bool:
    return bool(re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ0-9]", tok or ""))


def detokenize(tokens: Sequence[str]) -> str:
    out: List[str] = []
    no_space_before = {".", ",", "!", "?", ";", ":", "%", ")", "]", "}"}
    no_space_after = {"(", "[", "{", "#", "£", "$", "€"}
    for raw in tokens:
        tok = str(raw)
        if not tok:
            continue
        if not out:
            out.append(tok)
        elif tok in no_space_before:
            out[-1] += tok
        elif out[-1] in no_space_after:
            out[-1] += tok
        elif tok.startswith("'"):
            out[-1] += tok
        else:
            out.append(tok)
    return " ".join(out).strip()


def split_sentences_tokens(tokens: Sequence[str]) -> List[List[str]]:
    sentences: List[List[str]] = []
    cur: List[str] = []
    for tok in tokens:
        cur.append(tok)
        if tok in SENTENCE_END:
            if cur:
                sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    return sentences


# =============================================================================
# Random K-ary tree entropy model
# =============================================================================


def log_comb(n: int, k: int) -> float:
    n = int(n)
    k = int(k)
    if k < 0 or n < 0 or k > n:
        return float("-inf")
    return float(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def log_z_k(n: int, k: int) -> float:
    """
    Log number of weak ordered K-way partitions of a length-n span.

    A parent span of n tokens can be split into K non-negative child sizes in
    C(n + K - 1, K - 1) ways. This is the weak-composition count used here as
    the local branching multiplicity.
    """
    n = int(max(0, n))
    k = int(max(1, k))
    return log_comb(n + k - 1, k - 1)


@lru_cache(maxsize=256)
def theoretical_hk(k: int, n_max: int = DEFAULT_TARGET_N_FOR_HK) -> float:
    """
    Numeric entropy-rate estimate h_K from the recursive tree entropy model.

    The recurrence is a practical finite-N approximation used as a local prior.
    It is not used as a hard validator.
    """
    k = int(max(1, k))
    n_max = int(max(8, n_max))
    if k == 1:
        return 0.0

    H = [0.0] * (n_max + 1)
    for n in range(2, n_max + 1):
        lz = log_z_k(n, k)
        accum = 0.0
        for child_n in range(2, n):
            # Probability mass for a selected child size under the splitting kernel.
            lp = log_z_k(n - child_n, k - 1) - lz if k > 1 else float("-inf")
            if lp > -745:  # avoid underflow noise
                accum += math.exp(lp) * H[child_n]
        H[n] = lz + float(k) * accum
    return float(H[n_max] / max(1, n_max))


def k_for_profile(profile: str = "curriculum") -> int:
    p = PROFILE_REGISTRY.get(str(profile or "curriculum").lower(), PROFILE_REGISTRY["curriculum"])
    return int(p.get("target_k", DEFAULT_K))


def entropy_band_for_profile(profile: str = "curriculum") -> Tuple[float, float]:
    p = PROFILE_REGISTRY.get(str(profile or "curriculum").lower(), PROFILE_REGISTRY["curriculum"])
    band = p.get("entropy_band", [1.8, 3.0])
    return float(band[0]), float(band[1])


# =============================================================================
# Chunk tree
# =============================================================================


@dataclass
class ChunkNode:
    tokens: List[str]
    children: List["ChunkNode"] = field(default_factory=list)
    depth: int = 0
    boundary_reason: str = ""

    @property
    def size(self) -> int:
        return int(len(self.tokens))

    @property
    def text(self) -> str:
        return detokenize(self.tokens)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def walk(self) -> Iterable["ChunkNode"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def internal_nodes(self) -> Iterable["ChunkNode"]:
        for node in self.walk():
            if node.children:
                yield node

    def leaves(self) -> Iterable["ChunkNode"]:
        for node in self.walk():
            if not node.children:
                yield node

    def max_depth(self) -> int:
        if not self.children:
            return int(self.depth)
        return max(c.max_depth() for c in self.children)

    def to_dict(self, *, include_text: bool = True, max_children_text: int = 32) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "size": self.size,
            "depth": int(self.depth),
            "leaf": self.is_leaf(),
            "child_count": len(self.children),
            "boundary_reason": self.boundary_reason,
            "children": [c.to_dict(include_text=include_text, max_children_text=max_children_text) for c in self.children],
        }
        if include_text:
            payload["text"] = self.text[: int(max_children_text) * 20]
        return payload


@dataclass
class EntropyNLPConfig:
    k: int = DEFAULT_K
    profile: str = "curriculum"
    max_depth: int = DEFAULT_MAX_DEPTH
    min_split_tokens: int = DEFAULT_MIN_SPLIT_TOKENS
    force_token_leaves: bool = False
    sentence_bias: float = 2.5
    phrase_bias: float = 1.0
    balance_bias: float = 0.65

    def normalized(self) -> "EntropyNLPConfig":
        profile = str(self.profile or "curriculum").lower()
        k = int(self.k or 0)
        if k <= 0:
            k = k_for_profile(profile)
        return EntropyNLPConfig(
            k=int(max(1, min(16, k))),
            profile=profile,
            max_depth=int(max(1, self.max_depth)),
            min_split_tokens=int(max(1, self.min_split_tokens)),
            force_token_leaves=bool(self.force_token_leaves),
            sentence_bias=float(self.sentence_bias),
            phrase_bias=float(self.phrase_bias),
            balance_bias=float(self.balance_bias),
        )


def boundary_score(tokens: Sequence[str], i: int, *, k: int, depth: int, cfg: EntropyNLPConfig) -> Tuple[float, str]:
    """Score a potential boundary before tokens[i]."""
    n = len(tokens)
    if i <= 0 or i >= n:
        return -1e9, "invalid"

    prev = str(tokens[i - 1])
    nxt = str(tokens[i])
    prev_l = prev.lower()
    nxt_l = nxt.lower()

    score = 0.0
    reasons: List[str] = []

    if prev in SENTENCE_END:
        score += 5.0 * cfg.sentence_bias
        reasons.append("sentence_end")
    elif prev == ";":
        score += 4.0
        reasons.append("semicolon")
    elif prev == ":":
        score += 3.2
        reasons.append("colon")
    elif prev == ",":
        score += 1.3
        reasons.append("comma")

    if nxt_l in DISCOURSE_MARKERS:
        score += 3.0
        reasons.append("discourse_marker")
    if nxt_l in COORDINATORS:
        score += 1.7
        reasons.append("coordinator")
    if nxt_l in SUBORDINATORS:
        score += 1.4
        reasons.append("subordinator")
    if nxt_l in DETERMINERS and i > 1:
        score += 0.8
        reasons.append("determiner_np_start")

    if token_is_word(prev) and token_is_word(nxt):
        # Mild phrase-boundary prior: split before likely new content words.
        if nxt_l not in {"of", "to", "in", "on", "with", "for"}:
            score += 0.2 * cfg.phrase_bias

    # Avoid pathological edge splits.
    edge = min(i, n - i) / max(1, n)
    score += cfg.balance_bias * math.log1p(edge * k)

    # Prefer deeper phrase-level splits to be local and balanced.
    if depth > 1 and prev not in SENTENCE_END:
        score += 0.4 * cfg.phrase_bias

    return float(score), "+".join(reasons) if reasons else "balance"


def choose_cut_points(tokens: Sequence[str], *, cfg: EntropyNLPConfig, depth: int = 0) -> Tuple[List[int], str]:
    n = len(tokens)
    k = int(cfg.k)
    if n <= max(1, cfg.min_split_tokens) or k <= 1:
        return [], "too_small"

    # Target number of segments is smaller for short spans and at phrase level.
    if n <= 5:
        target_segments = min(k, n)
    elif n <= 14:
        target_segments = min(k, max(2, int(round(math.sqrt(n)))))
    else:
        target_segments = min(k, max(2, int(round(math.log2(n + 1)))))

    need = max(1, target_segments - 1)
    candidates: List[Tuple[float, int, str]] = []
    min_size = 1 if n <= 6 else max(1, min(4, n // (k * 2 + 1)))

    for i in range(1, n):
        if i < min_size or n - i < min_size:
            continue
        score, reason = boundary_score(tokens, i, k=k, depth=depth, cfg=cfg)
        candidates.append((score, i, reason))

    if not candidates:
        return [], "no_candidates"

    candidates.sort(key=lambda x: (x[0], -abs((n / 2) - x[1])), reverse=True)
    selected: List[Tuple[int, str]] = []

    for score, idx, reason in candidates:
        if len(selected) >= need:
            break
        # Keep cuts separated enough to avoid tiny repeated fragments.
        existing = [x[0] for x in selected]
        if any(abs(idx - j) < min_size for j in existing):
            continue
        proposed = sorted(existing + [idx])
        spans = [proposed[0], *[b - a for a, b in zip(proposed, proposed[1:])], n - proposed[-1]]
        if min(spans) < min_size:
            continue
        selected.append((idx, reason))

    if not selected and n > 2:
        # Deterministic fallback: balanced split. This keeps the tree usable.
        midpoint = n // 2
        return [midpoint], "balanced_fallback"

    selected.sort(key=lambda x: x[0])
    return [x[0] for x in selected], ";".join(x[1] for x in selected)


def build_chunk_tree(
    text_or_tokens: Union[str, Sequence[str]],
    *,
    config: Optional[EntropyNLPConfig] = None,
    depth: int = 0,
) -> ChunkNode:
    cfg = (config or EntropyNLPConfig()).normalized()
    tokens = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else [str(t) for t in text_or_tokens]
    node = ChunkNode(tokens=list(tokens), depth=int(depth))

    if depth >= cfg.max_depth or len(tokens) <= 1:
        return node

    if cfg.force_token_leaves and len(tokens) <= max(2, cfg.k):
        if len(tokens) > 1:
            node.children = [ChunkNode(tokens=[t], depth=depth + 1, boundary_reason="token_leaf") for t in tokens]
        return node

    cuts, reason = choose_cut_points(tokens, cfg=cfg, depth=depth)
    if not cuts:
        if cfg.force_token_leaves and len(tokens) > 1:
            node.children = [ChunkNode(tokens=[t], depth=depth + 1, boundary_reason="token_leaf") for t in tokens]
        return node

    bounds = [0] + cuts + [len(tokens)]
    children: List[ChunkNode] = []
    for start, end in zip(bounds, bounds[1:]):
        if end <= start:
            continue
        child = build_chunk_tree(tokens[start:end], config=cfg, depth=depth + 1)
        child.boundary_reason = reason
        children.append(child)

    # Avoid false recursion if split did not change structure.
    if len(children) == 1 and children[0].size == node.size:
        return node
    node.children = children
    node.boundary_reason = reason
    return node


# =============================================================================
# Diagnostics
# =============================================================================


@dataclass
class TreeStats:
    token_count: int
    internal_count: int
    leaf_count: int
    max_depth: int
    mean_branching: float
    max_branching: int
    negative_log_probability: float
    entropy_rate_nats: float
    theoretical_hk_nats: float
    k: int
    profile: str
    entropy_band: Tuple[float, float]
    band_status: str

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class TextDiagnostics:
    ok: bool
    text: str
    profile: str
    k: int
    stats: TreeStats
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    chunk_tree: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


def tree_negative_log_probability(root: ChunkNode, *, k: int) -> float:
    total = 0.0
    for node in root.internal_nodes():
        if node.size > 1:
            total += log_z_k(node.size, k)
    return float(total)


def compute_tree_stats(root: ChunkNode, *, cfg: EntropyNLPConfig) -> TreeStats:
    token_count = max(1, root.size)
    internal_nodes = list(root.internal_nodes())
    leaves = list(root.leaves())
    branch_counts = [len(n.children) for n in internal_nodes if n.children]
    nll = tree_negative_log_probability(root, k=cfg.k)
    rate = nll / token_count
    hk = theoretical_hk(cfg.k)
    band = entropy_band_for_profile(cfg.profile)
    if rate < band[0]:
        status = "below_target"
    elif rate > band[1]:
        status = "above_target"
    else:
        status = "inside_target"
    return TreeStats(
        token_count=int(root.size),
        internal_count=int(len(internal_nodes)),
        leaf_count=int(len(leaves)),
        max_depth=int(root.max_depth()),
        mean_branching=float(sum(branch_counts) / max(1, len(branch_counts))),
        max_branching=int(max(branch_counts) if branch_counts else 0),
        negative_log_probability=float(nll),
        entropy_rate_nats=float(rate),
        theoretical_hk_nats=float(hk),
        k=int(cfg.k),
        profile=str(cfg.profile),
        entropy_band=(float(band[0]), float(band[1])),
        band_status=status,
    )


def repetition_score(tokens: Sequence[str], ngram: int = 3) -> float:
    words = [t.lower() for t in tokens if token_is_word(t)]
    if len(words) < ngram * 2:
        return 0.0
    grams = [tuple(words[i : i + ngram]) for i in range(0, len(words) - ngram + 1)]
    if not grams:
        return 0.0
    unique = len(set(grams))
    return float(1.0 - unique / max(1, len(grams)))


def detect_unknown_verbification(tokens: Sequence[str], pos_lookup: Optional[Callable[[str], Sequence[str]]] = None) -> List[str]:
    """
    Detect likely noun/adjective used as an invented verb in a simple SVO frame.

    This is deliberately conservative and intended to catch failures like:
        "A cat gentles an animal."
    It does not reject creative writing; it returns warnings for a reranker.
    """
    warnings: List[str] = []
    toks = [t for t in tokens if token_is_word(t)]
    low = [t.lower() for t in toks]
    if len(low) < 3:
        return warnings

    for i in range(0, len(low) - 2):
        if low[i] in DETERMINERS or low[i] in {"i", "you", "he", "she", "it", "we", "they"}:
            # Find likely verb after a one-word subject or det+noun subject.
            verb_idx = i + 2 if low[i] in DETERMINERS and i + 2 < len(low) else i + 1
            verb = low[verb_idx]
            if verb in COMMON_VERBS or verb in AUXILIARIES:
                continue
            if not (verb.endswith("s") or verb.endswith("ed") or verb.endswith("ing")):
                continue
            lemma = re.sub(r"(ing|ed|s)$", "", verb)
            known_pos: List[str] = []
            if pos_lookup is not None:
                try:
                    known_pos = [str(x).lower() for x in pos_lookup(lemma)]
                except Exception:
                    known_pos = []
            adjective_like = lemma in COMMON_ADJECTIVES or verb.rstrip("s") in COMMON_ADJECTIVES
            if adjective_like or (known_pos and "verb" not in known_pos and "v" not in known_pos):
                warnings.append(f"possible_unknown_verbification:{verb}")
    return sorted(set(warnings))


def diagnose_text(
    text: str,
    *,
    profile: str = "curriculum",
    k: Optional[int] = None,
    include_tree: bool = False,
    pos_lookup: Optional[Callable[[str], Sequence[str]]] = None,
) -> TextDiagnostics:
    profile = str(profile or "curriculum").lower()
    cfg = EntropyNLPConfig(k=int(k or 0), profile=profile).normalized()
    clean = canonical_text(text)
    root = build_chunk_tree(clean, config=cfg)
    stats = compute_tree_stats(root, cfg=cfg)
    tokens = root.tokens
    warnings: List[str] = []
    suggestions: List[str] = []

    if not clean:
        warnings.append("empty_text")
        suggestions.append("Provide non-empty text.")

    if clean and tokens and tokens[-1] not in SENTENCE_END:
        warnings.append("missing_terminal_punctuation")
        suggestions.append("End generated prose with a clear terminal punctuation mark.")

    rep = repetition_score(tokens)
    if rep > 0.10:
        warnings.append(f"high_repetition:{rep:.3f}")
        suggestions.append("Reduce repeated phrases or regenerate with stronger anti-repetition constraints.")

    warnings.extend(detect_unknown_verbification(tokens, pos_lookup=pos_lookup))
    if any(w.startswith("possible_unknown_verbification") for w in warnings):
        suggestions.append("Constrain verb slots to entries known as verbs or use a safer copular/quality template.")

    if stats.band_status == "below_target":
        suggestions.append("The text is likely too flat or generic for the selected profile; add one specific relation, example, or curriculum anchor.")
    elif stats.band_status == "above_target":
        suggestions.append("The text is likely too branchy or dense for the selected profile; split it into shorter sentences or reduce clause nesting.")

    if stats.max_branching > cfg.k:
        warnings.append("branching_exceeds_k")

    ok = not any(w.startswith("possible_unknown_verbification") for w in warnings) and "empty_text" not in warnings
    return TextDiagnostics(
        ok=bool(ok),
        text=clean,
        profile=profile,
        k=int(cfg.k),
        stats=stats,
        warnings=sorted(set(warnings)),
        suggestions=suggestions,
        chunk_tree=root.to_dict(include_text=True) if include_tree else None,
        metadata={
            "module": MODULE_VERSION,
            "created_at": now_iso(),
            "profile_description": PROFILE_REGISTRY.get(profile, PROFILE_REGISTRY["curriculum"]).get("description", ""),
        },
    )


# Backward-compatible alias style.
analyze_text = diagnose_text


# =============================================================================
# Candidate scoring and reranking
# =============================================================================


@dataclass
class CandidateScore:
    text: str
    score: float
    entropy_score: float
    quality_score: float
    penalty: float
    diagnostics: TextDiagnostics
    rank_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))


def _candidate_text(item: Union[str, Mapping[str, Any]]) -> str:
    if isinstance(item, str):
        return item
    for key in ("text", "sentence", "output", "result", "answer"):
        if key in item and isinstance(item[key], str):
            return item[key]
    return str(item)


def _candidate_base_score(item: Union[str, Mapping[str, Any]]) -> float:
    if isinstance(item, Mapping):
        for key in ("score", "base_score", "similarity", "confidence"):
            if key in item:
                try:
                    val = float(item[key])
                    return val if math.isfinite(val) else 0.0
                except Exception:
                    continue
    return 0.0


def entropy_fit_score(rate: float, *, profile: str) -> float:
    low, high = entropy_band_for_profile(profile)
    if low <= rate <= high:
        # Peak around middle of target band.
        mid = (low + high) * 0.5
        half = max(EPS, (high - low) * 0.5)
        return float(max(0.0, 1.0 - 0.25 * abs(rate - mid) / half))
    if rate < low:
        return float(max(0.0, 1.0 - (low - rate) / max(EPS, low)))
    return float(max(0.0, 1.0 - (rate - high) / max(EPS, high)))


def score_text_candidate(
    candidate: Union[str, Mapping[str, Any]],
    *,
    context: str = "",
    profile: str = "curriculum",
    k: Optional[int] = None,
    pos_lookup: Optional[Callable[[str], Sequence[str]]] = None,
) -> CandidateScore:
    text = _candidate_text(candidate)
    base = _candidate_base_score(candidate)
    diag = diagnose_text(text, profile=profile, k=k, include_tree=False, pos_lookup=pos_lookup)
    rate = diag.stats.entropy_rate_nats
    e_score = entropy_fit_score(rate, profile=profile)

    penalty = 0.0
    for w in diag.warnings:
        if w == "missing_terminal_punctuation":
            penalty += 0.05
        elif w.startswith("high_repetition"):
            penalty += 0.20
        elif w.startswith("possible_unknown_verbification"):
            penalty += 0.35
        elif w == "empty_text":
            penalty += 1.0
        else:
            penalty += 0.08

    toks = tokenize(text)
    # Curriculum/diagnostic text should usually be concise.
    if profile in {"curriculum", "simple"} and len(toks) > 48:
        penalty += min(0.25, (len(toks) - 48) / 200.0)

    # Mild context anchoring: reward overlap with prompt terms without forcing copying.
    ctx_terms = {t.lower() for t in tokenize(context) if token_is_word(t) and len(t) > 3}
    cand_terms = {t.lower() for t in toks if token_is_word(t) and len(t) > 3}
    overlap = len(ctx_terms & cand_terms) / max(1, min(len(ctx_terms), 12)) if ctx_terms else 0.0
    context_score = min(1.0, overlap)

    quality = 0.58 * e_score + 0.22 * context_score + 0.20 * (1.0 if diag.ok else 0.45)
    # Combine with incoming model score only mildly so diagnostics can suppress bad candidates.
    final = (0.80 * quality + 0.20 * max(0.0, min(1.0, base))) - penalty
    final = max(0.0, min(1.0, final))

    return CandidateScore(
        text=text,
        score=float(final),
        entropy_score=float(e_score),
        quality_score=float(quality),
        penalty=float(penalty),
        diagnostics=diag,
        rank_metadata={
            "base_score": float(base),
            "context_overlap": float(context_score),
            "profile": profile,
            "module": MODULE_VERSION,
        },
    )


def rerank_texts(
    candidates: Sequence[Union[str, Mapping[str, Any]]],
    *,
    context: str = "",
    profile: str = "curriculum",
    k: Optional[int] = None,
    top_k: Optional[int] = None,
    pos_lookup: Optional[Callable[[str], Sequence[str]]] = None,
) -> List[Dict[str, Any]]:
    scored = [score_text_candidate(c, context=context, profile=profile, k=k, pos_lookup=pos_lookup) for c in candidates]
    scored.sort(key=lambda x: (x.score, x.entropy_score, -x.penalty), reverse=True)
    if top_k is not None:
        scored = scored[: int(max(1, top_k))]
    out: List[Dict[str, Any]] = []
    for rank, item in enumerate(scored, start=1):
        d = item.to_dict()
        d["rank"] = rank
        out.append(d)
    return out


# =============================================================================
# Generation control hints
# =============================================================================


def generation_control(
    *,
    profile: str = "curriculum",
    target_audience: str = "teacher",
    max_sentences: int = 3,
) -> Dict[str, Any]:
    profile = str(profile or "curriculum").lower()
    p = PROFILE_REGISTRY.get(profile, PROFILE_REGISTRY["curriculum"])
    k = int(p.get("target_k", DEFAULT_K))
    low, high = entropy_band_for_profile(profile)
    return {
        "ok": True,
        "module": MODULE_VERSION,
        "profile": profile,
        "target_audience": target_audience,
        "target_k": k,
        "target_entropy_band_nats_per_token": [low, high],
        "theoretical_hk_nats_per_token": theoretical_hk(k),
        "max_sentences": int(max_sentences),
        "generation_rules": [
            "Plan the answer as a small hierarchy: gist -> keypoints -> surface sentence.",
            f"Use at most {k} major semantic chunks at each level.",
            "Prefer coherent chunks over fixed token windows.",
            "Generate multiple candidates and rerank with entropy_nlp.rerank_texts.",
            "Reject candidates with unknown verbification warnings unless creative style is explicitly requested.",
            "Keep AI output advisory and non-mutating inside the LK20 governance boundary.",
        ],
    }


def status() -> Dict[str, Any]:
    return {
        "ok": True,
        "status": "active",
        "provider": "Akkurat.LocalAI.EntropyNLP",
        "version": MODULE_VERSION,
        "capabilities": [
            "semantic_chunk_tree_diagnostics",
            "tree_entropy_rate_estimation",
            "generation_candidate_reranking",
            "unknown_verbification_warning",
            "profile_based_generation_control",
            "status_only_governance_safe",
        ],
        "profiles": PROFILE_REGISTRY,
        "numpy_available": bool(_np is not None),
    }


# =============================================================================
# Optional semantic-bank POS adapter
# =============================================================================


def make_pos_lookup_from_semantic_bank(bank: Any) -> Callable[[str], Sequence[str]]:
    """
    Build a POS lookup callable from SemanticAttractorBank-like objects.

    This keeps entropy_nlp.py decoupled from semantic_attractors.py while letting
    sentence_builder/local_ai_adapter pass the bank in when available.
    """
    def lookup(lemma: str) -> Sequence[str]:
        q = str(lemma or "").lower().strip()
        if not q:
            return []
        keys: List[str] = []
        try:
            if hasattr(bank, "resolve") and callable(bank.resolve):
                keys = list(bank.resolve(q, max_results=8))
            elif hasattr(bank, "aliases"):
                keys = list(getattr(bank, "aliases", {}).get(q, []))
        except Exception:
            keys = []
        pos: List[str] = []
        for key in keys:
            try:
                a = getattr(bank, "attractors", {}).get(key)
                meta = getattr(a, "metadata", {}) if a is not None else {}
                p = str(meta.get("pos", "")).lower().strip()
                if p:
                    pos.append(p)
            except Exception:
                continue
        return sorted(set(pos))
    return lookup


# =============================================================================
# CLI
# =============================================================================


def _read_text_arg(text: str = "", file: str = "") -> str:
    if text:
        return text
    if file:
        return Path(file).read_text(encoding="utf-8")
    return sys.stdin.read()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local entropy/NLP diagnostics for LK20 Local AI.")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("status", help="Print module status.")

    a = sub.add_parser("analyze", help="Analyze text entropy/chunk structure.")
    a.add_argument("--text", default="")
    a.add_argument("--file", default="")
    a.add_argument("--profile", default="curriculum")
    a.add_argument("--k", type=int, default=0)
    a.add_argument("--include-tree", action="store_true")

    r = sub.add_parser("rerank", help="Rerank candidate texts from a JSON array.")
    r.add_argument("--candidates-json", default="", help="JSON array of strings or candidate objects. Reads stdin if omitted.")
    r.add_argument("--context", default="")
    r.add_argument("--profile", default="curriculum")
    r.add_argument("--k", type=int, default=0)
    r.add_argument("--top-k", type=int, default=0)

    g = sub.add_parser("control", help="Print generation-control hints.")
    g.add_argument("--profile", default="curriculum")
    g.add_argument("--target-audience", default="teacher")
    g.add_argument("--max-sentences", type=int, default=3)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command is None or args.command == "status":
            print(json.dumps(json_safe(status()), indent=2, ensure_ascii=False))
            return 0

        if args.command == "analyze":
            text = _read_text_arg(args.text, args.file)
            result = diagnose_text(
                text,
                profile=args.profile,
                k=args.k or None,
                include_tree=bool(args.include_tree),
            ).to_dict()
            print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
            return 0 if result.get("ok", False) else 2

        if args.command == "rerank":
            raw = args.candidates_json or sys.stdin.read()
            candidates = json.loads(raw)
            if not isinstance(candidates, list):
                raise ValueError("candidates JSON must be an array")
            result = rerank_texts(
                candidates,
                context=args.context,
                profile=args.profile,
                k=args.k or None,
                top_k=args.top_k or None,
            )
            print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))
            return 0

        if args.command == "control":
            print(json.dumps(json_safe(generation_control(
                profile=args.profile,
                target_audience=args.target_audience,
                max_sentences=args.max_sentences,
            )), indent=2, ensure_ascii=False))
            return 0

        raise ValueError(f"unknown command: {args.command}")
    except Exception as exc:
        err = {"ok": False, "error": repr(exc)}
        print(json.dumps(json_safe(err), indent=2, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODULE_VERSION",
    "PROFILE_REGISTRY",
    "EntropyNLPConfig",
    "ChunkNode",
    "TreeStats",
    "TextDiagnostics",
    "CandidateScore",
    "tokenize",
    "detokenize",
    "log_z_k",
    "theoretical_hk",
    "build_chunk_tree",
    "compute_tree_stats",
    "diagnose_text",
    "analyze_text",
    "score_text_candidate",
    "rerank_texts",
    "generation_control",
    "make_pos_lookup_from_semantic_bank",
    "status",
    "main",
]
