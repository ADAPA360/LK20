#!/usr/bin/env python3
# sentence_builder.py
"""
Semantic sentence builder for AtomTN / Akkurat.

Production role
---------------
Conservative deterministic sentence composition over:

    semantic_attractors.py
    local_ai_adapter.py  (optional read-only lexical validation)
    entropy_nlp.py      (optional advisory reranking)

This builder is deliberately not an open-ended generator. It creates only simple
sentences that are supported by prompt anchors, semantic-bank metadata, or the
verified root local_ai_adapter. In safe mode, which is the default, it will not
emit unsupported definitional claims such as "A Cameroonian is a Haydn."

Public API retained
-------------------
    SentenceBuilderConfig
    SentenceCandidate
    SentencePlan
    SlotCandidate
    SemanticSentenceBuilder
    SemanticSentenceBuilder.from_bank_path(...)
    SemanticSentenceBuilder.build(...)
    build_sentences(...)
    candidates_to_json(...)

Version
-------
6: strict entity-pair safety notices for unverified proper/entity prompts.

Operational invariant
---------------------
For any non-empty prompt, build(...) returns at least one visible
SentenceCandidate. The CLI never exits silently unless Python itself is not
executing this file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Ensure direct execution, runpy execution, and project-root invocation can all
# resolve sibling local_ai modules without requiring PYTHONPATH mutation.
SCRIPT_PATH = Path(__file__).resolve()
LOCAL_AI_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = LOCAL_AI_DIR.parent
for _p in (str(LOCAL_AI_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from semantic_attractors import SemanticAttractorBank, SemanticEntry
except Exception as exc:  # pragma: no cover
    raise ImportError("sentence_builder.py requires semantic_attractors.py in the same import path") from exc


_EPS = 1e-12
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_\-']+", re.UNICODE)
_VOWEL_SOUND_RE = re.compile(r"^[aeiou]", re.IGNORECASE)

_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "of", "to", "in", "on",
    "for", "with", "and", "or", "but", "is", "are", "was", "were", "be", "being",
    "been", "as", "by", "from", "at", "it", "its", "their", "his", "her", "our",
}

_POS_ALIASES = {
    "": "",
    "noun": "n", "nouns": "n", "n": "n", "proper_noun": "n", "proper noun": "n", "name": "n",
    "verb": "v", "verbs": "v", "v": "v",
    "adj": "adj", "adjective": "adj", "adjectives": "adj", "a": "adj", "s": "adj",
    "adv": "adv", "adverb": "adv", "adverbs": "adv", "r": "adv",
    "conj": "conj", "conjunction": "conj",
    "prep": "prep", "preposition": "prep",
    "pron": "pron", "pronoun": "pron",
    "det": "det", "determiner": "det",
    "interj": "interj", "interjection": "interj", "intj": "interj",
}

_DEFINITION_RELATIONS = {
    "hypernym", "hypernyms", "instance_hypernym", "instance_hypernyms",
    "is_a", "type", "kind", "kind_of", "subclass", "subclass_of",
    "synonym", "synonyms", "similar_to", "also_see",
}

_UNSAFE_GLOSS_PATTERNS = (
    "street name", "drug", "methcathinone", "vulgar", "offensive", "derogatory",
    "prostitute", "vagina", "vulva", "slur", "sexual", "pornographic",
    "archaic", "obsolete", "dated", "dialectal", "rare", "slang", "colloquial",
    "abbreviation of", "acronym of", "initialism of", "misspelling of",
    "alternative form of", "ellipsis of", "synonym of", "plural of",
    "past participle", "present participle", "gerund of", "third-person singular",
    "simple past", "comparative form of", "superlative form of",
)

_ENTITY_GLOSS_PATTERNS = (
    "surname", "given name", "male given name", "female given name", "proper noun",
    "native or inhabitant", "of or relating to", "capital and largest city",
    "composer", "writer", "poet", "politician", "person who", "people of",
    "inhabitant of", "the music of", "located on", "born", "died",
)

_BROAD_ANIMAL_TERMS = {
    "animal", "animals", "mammal", "mammals", "feline", "felidae", "cat", "cats",
    "species", "domesticated", "carnivorous", "vertebrate", "organism", "pet",
    "dog", "canine", "bird", "fish", "reptile", "insect", "primate", "rodent",
}


# =============================================================================
# Helpers
# =============================================================================


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj)
        if np.iscomplexobj(arr):
            return {"real": arr.real.astype(float).tolist(), "imag": arr.imag.astype(float).tolist()}
        return np.nan_to_num(arr.astype(float), nan=0.0, posinf=0.0, neginf=0.0).tolist()
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return _json_safe(vars(obj))
    return str(obj)


def _stable_hash_u64(text: str, seed: int = 0) -> int:
    h = (1469598103934665603 ^ int(seed)) & 0xFFFFFFFFFFFFFFFF
    for b in str(text).encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(h)


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _norm_token(text: Any) -> str:
    return _normalize_space(text).lower()


def _tokenize(text: Any) -> List[str]:
    return [m.group(0).lower().strip("_-") for m in _WORD_RE.finditer(str(text or "")) if m.group(0).strip("_-")]


def _content_tokens(text: Any) -> List[str]:
    return [t for t in _tokenize(text) if t and t not in _STOPWORDS]


def _singularish(token: str) -> str:
    t = _norm_token(token)
    if len(t) > 3 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("es"):
        return t[:-2]
    if len(t) > 2 and t.endswith("s"):
        return t[:-1]
    return t


def _canonical_pos(pos: Any) -> str:
    p = _norm_token(pos).replace("-", "_").replace(" ", "_")
    return _POS_ALIASES.get(p, p)


def _unit_vector(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n <= _EPS:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr / n).astype(np.float64, copy=False)


def _cosine(a: Any, b: Any) -> float:
    aa = _unit_vector(a)
    bb = _unit_vector(b)
    if aa.size == 0 or bb.size == 0 or aa.shape != bb.shape:
        return 0.0
    val = float(np.dot(aa, bb))
    return val if np.isfinite(val) else 0.0


def _article_for(word: str, definite: bool = False) -> str:
    if definite:
        return "the"
    w = _norm_token(word)
    if not w:
        return "a"
    if w.startswith(("honest", "hour", "heir")):
        return "an"
    if w.startswith(("user", "university", "unicorn", "european", "one")):
        return "a"
    return "an" if _VOWEL_SOUND_RE.match(w) else "a"


def _pluralize(noun: str) -> str:
    n = str(noun)
    low = n.lower()
    irregular = {"child": "children", "person": "people", "man": "men", "woman": "women", "mouse": "mice", "goose": "geese"}
    if low in irregular:
        return irregular[low]
    if low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        return n[:-1] + "ies"
    if low.endswith(("s", "x", "z", "ch", "sh")):
        return n + "es"
    return n + "s"


def _third_person_singular(verb: str) -> str:
    v = str(verb)
    low = v.lower()
    irregular = {"be": "is", "have": "has", "do": "does", "go": "goes"}
    if low in irregular:
        return irregular[low]
    if low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        return v[:-1] + "ies"
    if low.endswith(("s", "x", "z", "ch", "sh", "o")):
        return v + "es"
    return v + "s"


def _past_tense(verb: str) -> str:
    v = str(verb)
    low = v.lower()
    irregular = {"be": "was", "have": "had", "do": "did", "go": "went", "run": "ran", "see": "saw", "make": "made"}
    if low in irregular:
        return irregular[low]
    if low.endswith("e"):
        return v + "d"
    if low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        return v[:-1] + "ied"
    return v + "ed"


def _capitalize_sentence(s: str) -> str:
    s = _normalize_space(s)
    if not s:
        return s
    s = s[0].upper() + s[1:]
    if s[-1] not in ".!?":
        s += "."
    return s


def _shorten(text: str, limit: int = 220) -> str:
    s = _normalize_space(text)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 1)].rstrip() + "…"


def _metadata(entry: Any) -> Dict[str, Any]:
    meta = getattr(entry, "metadata", {}) or {}
    return dict(meta) if isinstance(meta, Mapping) else {}


def _lemma(entry: Any, default: str = "") -> str:
    val = getattr(entry, "lemma", None)
    if val:
        return str(val)
    meta = _metadata(entry)
    return str(meta.get("lemma") or meta.get("word") or meta.get("label") or default)


def _pos(entry: Any) -> str:
    val = getattr(entry, "pos", None)
    if val:
        return _canonical_pos(val)
    meta = _metadata(entry)
    return _canonical_pos(meta.get("pos") or meta.get("part_of_speech") or "")


def _gloss(entry: Any) -> str:
    val = getattr(entry, "gloss", None)
    if val:
        return str(val)
    meta = _metadata(entry)
    return str(meta.get("gloss") or meta.get("definition") or "")


def _relations(entry: Any) -> Dict[str, Any]:
    val = getattr(entry, "relations", None)
    if isinstance(val, Mapping):
        return dict(val)
    meta = _metadata(entry)
    raw = meta.get("relations")
    if isinstance(raw, Mapping):
        return dict(raw)
    embedded = meta.get("entry")
    if isinstance(embedded, Mapping) and isinstance(embedded.get("relations"), Mapping):
        return dict(embedded.get("relations"))
    return {}


def _entry_text(entry: Any) -> str:
    parts = [_lemma(entry), _pos(entry), _gloss(entry)]
    meta = _metadata(entry)
    embedded = meta.get("entry")
    if isinstance(embedded, Mapping):
        for key in ("definition", "gloss", "label", "word", "lemma"):
            if embedded.get(key):
                parts.append(str(embedded.get(key)))
    return " ".join(str(x) for x in parts if x)


def _entry_tokens(entry: Any) -> set[str]:
    toks = set(_content_tokens(_entry_text(entry)))
    lem = _norm_token(_lemma(entry))
    if lem:
        toks.add(lem)
        toks.add(lem.replace("_", " "))
    return {t for t in toks if t}


def _unsafe_gloss_penalty(text: Any) -> float:
    low = _norm_token(text)
    penalty = 0.0
    for pat in _UNSAFE_GLOSS_PATTERNS:
        if pat in low:
            penalty += 0.75
    for pat in _ENTITY_GLOSS_PATTERNS:
        if pat in low:
            penalty += 0.35
    return float(penalty)


def _is_capitalized_entity(lemma: Any, gloss: Any = "") -> bool:
    s = str(lemma or "").strip()
    if re.match(r"^[A-Z][a-z]+(?:[\s\-_][A-Z][a-z]+)*$", s or ""):
        return True
    low = _norm_token(gloss)
    return any(p in low for p in _ENTITY_GLOSS_PATTERNS)


def _safe_len_penalty(lemma: str) -> float:
    words = [w for w in str(lemma or "").replace("_", " ").split() if w]
    return 0.02 * max(0, len(words) - 1) + 0.002 * max(0, len(str(lemma)) - 16)


def _pos_priority(pos: str, desired: str = "") -> int:
    p = _canonical_pos(pos)
    d = _canonical_pos(desired)
    if d and p == d:
        return 0
    order = {"n": 1, "v": 2, "adj": 3, "adv": 4, "conj": 5, "prep": 6, "det": 7, "pron": 8, "interj": 9}
    return order.get(p, 20)


# =============================================================================
# Data containers
# =============================================================================


@dataclass(frozen=True)
class SentenceBuilderConfig:
    seed: int = 0
    default_template: str = "auto"
    max_candidates_per_slot: int = 16
    max_combinations: int = 512
    relation_boost: float = 0.12
    lexical_overlap_boost: float = 0.08
    weight_boost: float = 0.05
    anchor_boost: float = 0.65
    exact_lemma_boost: float = 0.35
    diversity_penalty: float = 0.08
    allow_reuse: bool = False
    require_known_pos: bool = False
    fallback_to_any_pos: bool = True
    prefer_simple_words: bool = True
    definite_article: bool = False
    tense: str = "present"
    number: str = "singular"

    safe_mode: bool = True
    adapter_validation: bool = True
    entropy_rerank: bool = True
    allow_unsafe_definition: bool = False
    reject_proper_noun_definitions: bool = True
    definition_evidence_threshold: float = 0.45
    fallback_safe_sentence: bool = True
    require_prompt_anchored_core_slots: bool = True
    debug: bool = False

    def normalized(self) -> "SentenceBuilderConfig":
        tense = str(self.tense or "present").lower().strip()
        if tense not in {"present", "past", "bare"}:
            tense = "present"
        number = str(self.number or "singular").lower().strip()
        if number not in {"singular", "plural"}:
            number = "singular"
        return SentenceBuilderConfig(
            seed=int(self.seed),
            default_template=str(self.default_template or "auto").lower().strip(),
            max_candidates_per_slot=int(max(1, self.max_candidates_per_slot)),
            max_combinations=int(max(1, self.max_combinations)),
            relation_boost=float(self.relation_boost),
            lexical_overlap_boost=float(self.lexical_overlap_boost),
            weight_boost=float(self.weight_boost),
            anchor_boost=float(self.anchor_boost),
            exact_lemma_boost=float(self.exact_lemma_boost),
            diversity_penalty=float(max(0.0, self.diversity_penalty)),
            allow_reuse=bool(self.allow_reuse),
            require_known_pos=bool(self.require_known_pos),
            fallback_to_any_pos=bool(self.fallback_to_any_pos),
            prefer_simple_words=bool(self.prefer_simple_words),
            definite_article=bool(self.definite_article),
            tense=tense,
            number=number,
            safe_mode=bool(self.safe_mode),
            adapter_validation=bool(self.adapter_validation),
            entropy_rerank=bool(self.entropy_rerank),
            allow_unsafe_definition=bool(self.allow_unsafe_definition),
            reject_proper_noun_definitions=bool(self.reject_proper_noun_definitions),
            definition_evidence_threshold=float(max(0.0, self.definition_evidence_threshold)),
            fallback_safe_sentence=bool(self.fallback_safe_sentence),
            require_prompt_anchored_core_slots=bool(self.require_prompt_anchored_core_slots),
            debug=bool(self.debug),
        )


@dataclass(frozen=True)
class SentencePlan:
    name: str
    template: str
    slots: Tuple[Tuple[str, str], ...]
    description: str = ""
    factual_strength: str = "low"

    def slot_names(self) -> List[str]:
        return [s for s, _ in self.slots]


@dataclass
class SlotCandidate:
    slot: str
    key: str
    lemma: str
    pos: str
    score: float
    reasons: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class SentenceCandidate:
    sentence: str
    score: float
    plan: str
    slots: Dict[str, SlotCandidate] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.sentence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "score": float(self.score),
            "plan": self.plan,
            "slots": {k: v.to_dict() for k, v in self.slots.items()},
            "diagnostics": _json_safe(self.diagnostics),
        }


@dataclass
class PromptAnalysis:
    text: str
    tokens: List[str]
    exact_keys_by_token: Dict[str, List[str]] = field(default_factory=dict)
    token_to_keys: Dict[str, List[str]] = field(default_factory=dict)
    anchors_by_pos: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    def keys_for_pos(self, pos: str) -> List[str]:
        wanted = _canonical_pos(pos)
        out: List[str] = []
        seen: set[str] = set()
        for _tok, key in self.anchors_by_pos.get(wanted, []):
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out


BUILTIN_PLANS: Dict[str, SentencePlan] = {
    "definition": SentencePlan("definition", "{subj_np} is {obj_np}", (("subject", "n"), ("object", "n")), "X is Y.", "high"),
    "quality": SentencePlan("quality", "{subj_np} is {adj}", (("subject", "n"), ("adjective", "adj")), "X is ADJ.", "medium"),
    "action": SentencePlan("action", "{subj_np} {verb} {obj_np}", (("subject", "n"), ("verb", "v"), ("object", "n")), "X verbs Y.", "medium"),
    "relation": SentencePlan("relation", "{subj_np} relates to {obj_np}", (("subject", "n"), ("object", "n")), "Cautious relation.", "low"),
    "comparison": SentencePlan("comparison", "{subj_np} differs from {obj_np}", (("subject", "n"), ("object", "n")), "Contrast.", "medium"),
    "observation": SentencePlan("observation", "{subj_np} can be {adj} in context", (("subject", "n"), ("adjective", "adj")), "Cautious attribute.", "low"),
    "context_note": SentencePlan("context_note", "{subj} and {obj} are related terms in this context", (("subject", "n"), ("object", "n")), "Non-factual fallback.", "low"),
}


# =============================================================================
# Builder
# =============================================================================


class SemanticSentenceBuilder:
    def __init__(self, bank: SemanticAttractorBank, cfg: Optional[SentenceBuilderConfig] = None, *, adapter: Any = None) -> None:
        self.bank = bank
        self.cfg = (cfg or SentenceBuilderConfig()).normalized()
        self.entries: Dict[str, Any] = self._extract_entries(bank)
        self.vectors: Dict[str, np.ndarray] = self._extract_vectors(bank)
        self.index_by_pos: Dict[str, List[str]] = self._build_pos_index()
        self.index_by_lemma_exact: Dict[str, List[str]] = self._build_exact_lemma_index()
        self.index_by_token: Dict[str, List[str]] = self._build_token_index()
        self._adapter = adapter
        self._adapter_owned = False

    @classmethod
    def from_bank_path(cls, path: Union[str, Path], cfg: Optional[SentenceBuilderConfig] = None, *, adapter: Any = None) -> "SemanticSentenceBuilder":
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".npz":
            bank = SemanticAttractorBank.load_npz(p)
        elif suffix == ".json":
            bank = SemanticAttractorBank.load_json(p)
        elif suffix == ".db":
            from semantic_attractors import HybridSemanticBank  # type: ignore
            bank = HybridSemanticBank(p)
        else:
            try:
                bank = SemanticAttractorBank.load_npz(p)
            except Exception:
                bank = SemanticAttractorBank.load_json(p)
        return cls(bank, cfg=cfg, adapter=adapter)

    @staticmethod
    def _extract_entries(bank: SemanticAttractorBank) -> Dict[str, Any]:
        entries = getattr(bank, "entries", None)
        if isinstance(entries, Mapping):
            return {str(k): v for k, v in entries.items()}

        attr = getattr(bank, "attractors", None)
        if isinstance(attr, Mapping):
            out: Dict[str, Any] = {}
            for k, a in attr.items():
                meta = getattr(a, "metadata", {}) or {}
                lemma = str(meta.get("lemma", getattr(a, "label", k)))
                pos = str(meta.get("pos", ""))
                gloss = str(meta.get("gloss", ""))
                sense_id = str(meta.get("sense_id", ""))
                rels: Dict[str, Any] = {}
                embedded = meta.get("entry") if isinstance(meta, Mapping) else None
                if isinstance(embedded, Mapping) and isinstance(embedded.get("relations"), Mapping):
                    rels = dict(embedded.get("relations"))
                elif isinstance(meta, Mapping) and isinstance(meta.get("relations"), Mapping):
                    rels = dict(meta.get("relations"))
                try:
                    out[str(k)] = SemanticEntry(
                        key=str(k),
                        lemma=lemma,
                        sense_id=sense_id,
                        pos=pos,
                        gloss=gloss,
                        tokens=tuple(_content_tokens(f"{lemma} {gloss} {pos}")),
                        relations=rels,
                        weight=float(getattr(a, "weight", 1.0)),
                        metadata=dict(meta),
                    )
                except Exception:
                    out[str(k)] = a
            return out

        keys_attr = getattr(bank, "keys", [])
        keys = list(keys_attr()) if callable(keys_attr) else list(keys_attr or [])
        return {str(k): SemanticEntry(key=str(k), lemma=str(k), tokens=tuple(_content_tokens(k))) for k in keys}

    @staticmethod
    def _extract_vectors(bank: SemanticAttractorBank) -> Dict[str, np.ndarray]:
        attr = getattr(bank, "attractors", None)
        if isinstance(attr, Mapping):
            out: Dict[str, np.ndarray] = {}
            for k, a in attr.items():
                vec = getattr(a, "vector", None)
                if vec is not None:
                    out[str(k)] = _unit_vector(vec)
            return out
        return {}

    def close(self) -> None:
        if self._adapter_owned and self._adapter is not None and hasattr(self._adapter, "close"):
            try:
                self._adapter.close()
            except Exception:
                pass
        self._adapter = None
        self._adapter_owned = False

    def _get_adapter(self) -> Any:
        if not self.cfg.adapter_validation:
            return None
        if self._adapter is not None:
            return self._adapter
        try:
            project_root = Path(__file__).resolve().parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from local_ai_adapter import get_adapter  # type: ignore
            self._adapter = get_adapter(project_root)
            self._adapter_owned = True
            return self._adapter
        except Exception:
            return None

    def _entry_sort_key(self, key: str, desired_pos: str = "") -> Tuple[Any, ...]:
        e = self.entries[key]
        return (
            _pos_priority(_pos(e), desired_pos),
            _unsafe_gloss_penalty(_gloss(e)),
            _safe_len_penalty(_lemma(e, key)),
            _norm_token(_lemma(e, key)),
            key,
        )

    def _build_pos_index(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for key, entry in self.entries.items():
            out.setdefault(_pos(entry) or "", []).append(key)
        for p in list(out.keys()):
            out[p] = sorted(set(out[p]), key=lambda k: self._entry_sort_key(k, p))
        return out

    def _build_exact_lemma_index(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for key, entry in self.entries.items():
            lemma = _norm_token(_lemma(entry, key)).replace("_", " ")
            if lemma:
                out.setdefault(lemma, []).append(key)
                raw = _norm_token(_lemma(entry, key))
                out.setdefault(raw, []).append(key)
        for lemma in list(out.keys()):
            out[lemma] = sorted(set(out[lemma]), key=lambda k: self._entry_sort_key(k))
        return out

    def _build_token_index(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for key, entry in self.entries.items():
            for tok in _entry_tokens(entry):
                out.setdefault(tok, []).append(key)
        for tok in list(out.keys()):
            out[tok] = sorted(set(out[tok]), key=lambda k: self._entry_sort_key(k))
        return out

    # ------------------------------------------------------------------
    # Prompt and retrieval
    # ------------------------------------------------------------------
    def exact_keys(self, token: str, *, pos: Optional[str] = None, limit: int = 16) -> List[str]:
        t = _norm_token(token).replace("_", " ")
        keys = list(self.index_by_lemma_exact.get(t, []))
        if not keys and _singularish(t) != t:
            keys = list(self.index_by_lemma_exact.get(_singularish(t), []))
        if pos:
            wanted = _canonical_pos(pos)
            keys = [k for k in keys if _pos(self.entries.get(k)) == wanted]
        return keys[: int(max(1, limit))]

    def resolve_keys(self, terms: Union[str, Sequence[str]], *, max_per_token: int = 12, exact_first: bool = True) -> List[str]:
        toks = _content_tokens(terms if isinstance(terms, str) else " ".join(str(t) for t in terms))
        out: List[str] = []
        seen: set[str] = set()
        for tok in toks:
            exact = self.exact_keys(tok, limit=max_per_token) if exact_first else []
            rest = [k for k in self.index_by_token.get(tok, []) if k not in exact]
            for key in (exact + rest)[:max_per_token]:
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def analyze_prompt(self, prompt: Union[str, Sequence[str]]) -> PromptAnalysis:
        text = prompt if isinstance(prompt, str) else " ".join(str(x) for x in prompt)
        toks = _content_tokens(text)
        analysis = PromptAnalysis(text=str(text), tokens=toks)
        for tok in toks:
            exact = self.exact_keys(tok, limit=32)
            keys = self.resolve_keys([tok], max_per_token=32)
            analysis.exact_keys_by_token[tok] = exact
            analysis.token_to_keys[tok] = keys
            for key in exact:
                e = self.entries.get(key)
                if e is None:
                    continue
                p = _pos(e)
                if p:
                    analysis.anchors_by_pos.setdefault(p, []).append((tok, key))
        return analysis

    def intent_vector(self, prompt: Union[str, Sequence[str]], *, extra_keys: Optional[Sequence[str]] = None) -> np.ndarray:
        keys = self.resolve_keys(prompt)
        if extra_keys:
            keys.extend(str(k) for k in extra_keys if str(k) in self.vectors)
        vecs = [self.vectors[k] for k in keys if k in self.vectors]
        if vecs:
            return _unit_vector(np.mean(np.stack(vecs, axis=0), axis=0))
        dim = next(iter(self.vectors.values())).size if self.vectors else 64
        tokens = _content_tokens(prompt if isinstance(prompt, str) else " ".join(str(x) for x in prompt))
        acc = np.zeros((dim,), dtype=np.float64)
        for tok in tokens:
            rng = np.random.default_rng((_stable_hash_u64(tok, seed=self.cfg.seed) & 0xFFFFFFFF))
            acc += rng.normal(size=dim)
        return _unit_vector(acc)

    def _relation_neighbors(self, seed_keys: Sequence[str]) -> set[str]:
        out: set[str] = set()
        for key in seed_keys:
            e = self.entries.get(str(key))
            if e is None:
                continue
            for payload in _relations(e).values():
                if isinstance(payload, Mapping):
                    out.update(str(k) for k in payload.keys() if str(k) in self.entries)
                else:
                    try:
                        out.update(str(x) for x in payload if str(x) in self.entries)
                    except Exception:
                        pass
        return out

    def _slot_pool(self, pos: str, analysis: Optional[PromptAnalysis] = None) -> List[str]:
        wanted = _canonical_pos(pos)
        anchors = analysis.keys_for_pos(wanted) if analysis is not None else []
        base = list(self.index_by_pos.get(wanted, []))
        if not base and self.cfg.fallback_to_any_pos:
            for vals in self.index_by_pos.values():
                base.extend(vals)
        out: List[str] = []
        seen: set[str] = set()
        for key in anchors + base:
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def rank_slot_candidates(
        self,
        slot: str,
        pos: str,
        intent: np.ndarray,
        *,
        prompt: Union[str, Sequence[str]],
        analysis: Optional[PromptAnalysis] = None,
        seed_keys: Optional[Sequence[str]] = None,
        exclude_keys: Optional[set[str]] = None,
        limit: Optional[int] = None,
    ) -> List[SlotCandidate]:
        cfg = self.cfg
        prompt_text = prompt if isinstance(prompt, str) else " ".join(str(x) for x in prompt)
        analysis = analysis or self.analyze_prompt(prompt_text)
        seed_keys = list(seed_keys or self.resolve_keys(prompt_text))
        neighbor_keys = self._relation_neighbors(seed_keys)
        prompt_tokens = set(_content_tokens(prompt_text))
        exclude_keys = set(exclude_keys or set())
        limit = int(limit or cfg.max_candidates_per_slot)
        required = _canonical_pos(pos)
        anchor_set = set(analysis.keys_for_pos(required))

        cands: List[SlotCandidate] = []
        for key in self._slot_pool(required, analysis=analysis):
            if key in exclude_keys and not cfg.allow_reuse:
                continue
            entry = self.entries.get(key)
            if entry is None:
                continue
            epos = _pos(entry)
            if cfg.require_known_pos and required and epos != required:
                continue

            vec = self.vectors.get(key)
            semantic = _cosine(intent, vec) if vec is not None else 0.0
            relation = cfg.relation_boost if key in neighbor_keys else 0.0
            overlap_n = len(prompt_tokens.intersection(_entry_tokens(entry)))
            overlap = cfg.lexical_overlap_boost * float(overlap_n)
            try:
                weight = cfg.weight_boost * math.log1p(max(0.0, float(getattr(entry, "weight", 1.0))))
            except Exception:
                weight = 0.0
            anchor = cfg.anchor_boost if key in anchor_set else 0.0
            exact = cfg.exact_lemma_boost if _norm_token(_lemma(entry, key)).replace("_", " ") in prompt_tokens else 0.0
            simplicity = -_safe_len_penalty(_lemma(entry, key)) if cfg.prefer_simple_words else 0.0
            unsafe = -_unsafe_gloss_penalty(_gloss(entry))
            pos_bonus = 0.18 if required and epos == required else -0.08
            tie = 1e-9 * ((_stable_hash_u64(f"{slot}|{key}", seed=cfg.seed) % 1000003) / 1000003.0)
            score = semantic + relation + overlap + weight + anchor + exact + simplicity + unsafe + pos_bonus + tie
            cands.append(SlotCandidate(
                slot=slot, key=key, lemma=_lemma(entry, key), pos=epos, score=float(score),
                reasons={
                    "semantic": float(semantic), "relation": float(relation),
                    "lexical_overlap": float(overlap), "weight": float(weight),
                    "anchor": float(anchor), "exact_lemma": float(exact),
                    "simplicity": float(simplicity), "unsafe": float(unsafe),
                    "pos_bonus": float(pos_bonus), "tie": float(tie),
                },
            ))
        cands.sort(key=lambda c: (-c.score, _safe_len_penalty(c.lemma), c.lemma.lower(), c.key))
        return cands[:limit]

    # ------------------------------------------------------------------
    # Planning and realization
    # ------------------------------------------------------------------
    def choose_plans(self, prompt: Union[str, Sequence[str]], template: Optional[str] = None) -> List[SentencePlan]:
        requested = str(template or self.cfg.default_template or "auto").lower().strip()
        if requested and requested != "auto":
            if requested not in BUILTIN_PLANS:
                raise ValueError(f"unknown sentence template {requested!r}; available: {sorted(BUILTIN_PLANS)}")
            return [BUILTIN_PLANS[requested]]
        toks = set(_content_tokens(prompt if isinstance(prompt, str) else " ".join(str(x) for x in prompt)))
        if toks.intersection({"different", "contrast", "opposite", "antonym", "differs"}):
            return [BUILTIN_PLANS["comparison"], BUILTIN_PLANS["relation"], BUILTIN_PLANS["context_note"]]
        if toks.intersection({"do", "does", "action", "act", "move", "cause", "make", "run"}):
            return [BUILTIN_PLANS["action"], BUILTIN_PLANS["definition"], BUILTIN_PLANS["relation"], BUILTIN_PLANS["context_note"]]
        if toks.intersection({"quality", "attribute", "property", "feel", "looks"}):
            return [BUILTIN_PLANS["quality"], BUILTIN_PLANS["observation"], BUILTIN_PLANS["definition"], BUILTIN_PLANS["context_note"]]
        return [BUILTIN_PLANS["definition"], BUILTIN_PLANS["quality"], BUILTIN_PLANS["relation"], BUILTIN_PLANS["action"], BUILTIN_PLANS["context_note"]]

    def _noun_phrase(self, cand: SlotCandidate) -> str:
        lemma = cand.lemma.replace("_", " ")
        if self.cfg.number == "plural":
            return _pluralize(lemma)
        if cand.pos in {"pron", "det"}:
            return lemma
        return f"{_article_for(lemma, definite=self.cfg.definite_article)} {lemma}"

    def _realize_verb(self, cand: SlotCandidate) -> str:
        v = cand.lemma.replace("_", " ")
        if self.cfg.tense == "bare":
            return v
        if self.cfg.tense == "past":
            return _past_tense(v)
        if self.cfg.number == "singular":
            return _third_person_singular(v)
        return v

    def realize(self, plan: SentencePlan, slots: Mapping[str, SlotCandidate]) -> str:
        vals: Dict[str, str] = {}
        subj = slots.get("subject")
        obj = slots.get("object")
        adj = slots.get("adjective")
        verb = slots.get("verb")
        if subj is not None:
            vals["subj"] = subj.lemma.replace("_", " ")
            vals["subj_np"] = self._noun_phrase(subj)
        if obj is not None:
            vals["obj"] = obj.lemma.replace("_", " ")
            vals["obj_np"] = self._noun_phrase(obj)
        if adj is not None:
            vals["adj"] = adj.lemma.replace("_", " ")
        if verb is not None:
            vals["verb"] = self._realize_verb(verb)
        try:
            sent = plan.template.format(**vals)
        except KeyError:
            pieces = [vals.get("subj_np", vals.get("subj", "something"))]
            if verb is not None:
                pieces.append(vals.get("verb", verb.lemma))
            elif adj is not None:
                pieces.extend(["is", vals.get("adj", adj.lemma)])
            else:
                pieces.extend(["relates to", vals.get("obj_np", vals.get("obj", "something"))])
            sent = " ".join(pieces)
        return _capitalize_sentence(sent)

    def _slot_matches_any_prompt_token(self, cand: Optional[SlotCandidate], analysis: PromptAnalysis) -> bool:
        if cand is None:
            return False
        lemma = _norm_token(cand.lemma).replace("_", " ")
        lemma_s = _singularish(lemma)
        for tok in analysis.tokens:
            tt = _norm_token(tok).replace("_", " ")
            if lemma == tt or lemma_s == _singularish(tt):
                return True
        return False

    def _slot_matches_prompt_token(self, cand: Optional[SlotCandidate], token: str) -> bool:
        if cand is None:
            return False
        lemma = _norm_token(cand.lemma).replace("_", " ")
        tok = _norm_token(token).replace("_", " ")
        return lemma == tok or _singularish(lemma) == _singularish(tok)

    def _candidate_prompt_anchor_check(self, candidate: SentenceCandidate, analysis: PromptAnalysis) -> Tuple[bool, Dict[str, Any]]:
        """
        Reject safe-mode candidates that use unprompted core slots.

        This is the guard that prevents outputs such as:
          - "A cat brooms an animal."
          - "A love-potion is a red."
          - "A denial is Cameroonian."

        In safe mode, factual and semi-factual templates must be built from
        explicit prompt anchors. Generic fallback text is generated separately.
        """
        if not self.cfg.require_prompt_anchored_core_slots:
            return True, {"required": False}

        toks = list(analysis.tokens or [])
        if not toks:
            return True, {"required": True, "reason": "no_prompt_tokens"}

        plan = candidate.plan
        required_slots: List[str] = []
        require_order = False

        if plan == "definition":
            required_slots = ["subject", "object"]
            require_order = len(toks) >= 2
        elif plan in {"relation", "context_note", "comparison"}:
            required_slots = ["subject", "object"]
            require_order = len(toks) >= 2
        elif plan == "action":
            required_slots = ["subject", "verb", "object"]
            require_order = len(toks) >= 3
        elif plan in {"quality", "observation"}:
            required_slots = ["subject", "adjective"]
            require_order = len(toks) >= 2
        else:
            return True, {"required": True, "plan": plan, "checked": False}

        missing: List[str] = []
        for slot in required_slots:
            if not self._slot_matches_any_prompt_token(candidate.slots.get(slot), analysis):
                missing.append(slot)

        info: Dict[str, Any] = {
            "required": True,
            "plan": plan,
            "required_slots": required_slots,
            "missing_prompt_anchors": missing,
        }
        if missing:
            return False, info

        if require_order and "subject" in required_slots and "object" in required_slots:
            subj_ok = self._slot_matches_prompt_token(candidate.slots.get("subject"), toks[0])
            obj_ok = self._slot_matches_prompt_token(candidate.slots.get("object"), toks[1])
            info["prompt_order"] = {"subject_matches_first": subj_ok, "object_matches_second": obj_ok}
            if not (subj_ok and obj_ok):
                return False, info

        if require_order and plan in {"quality", "observation"}:
            subj_ok = self._slot_matches_prompt_token(candidate.slots.get("subject"), toks[0])
            adj_ok = self._slot_matches_prompt_token(candidate.slots.get("adjective"), toks[1])
            info["prompt_order"] = {"subject_matches_first": subj_ok, "adjective_matches_second": adj_ok}
            if not (subj_ok and adj_ok):
                return False, info

        return True, info

    def _candidate_has_entity_slot(self, candidate: SentenceCandidate) -> Tuple[bool, List[Dict[str, Any]]]:
        entities: List[Dict[str, Any]] = []
        for slot_name, slot in candidate.slots.items():
            e = self.entries.get(slot.key)
            gloss = _gloss(e) if e is not None else ""
            if _is_capitalized_entity(slot.lemma, gloss):
                entities.append({"slot": slot_name, "lemma": slot.lemma, "gloss": gloss[:160]})
        return bool(entities), entities

    # ------------------------------------------------------------------
    # Adapter and safety
    # ------------------------------------------------------------------
    def _adapter_lookup(self, term: str, context: str, *, limit: int = 5) -> List[Dict[str, Any]]:
        adapter = self._get_adapter()
        if adapter is None or not hasattr(adapter, "lookup_lexicon"):
            return []
        try:
            return list(adapter.lookup_lexicon(term, limit=limit, context=context))
        except TypeError:
            try:
                return list(adapter.lookup_lexicon(term, limit=limit))
            except Exception:
                return []
        except Exception:
            return []

    def _adapter_supports_definition(self, subj: str, obj: str, context: str) -> Tuple[float, Dict[str, Any]]:
        rows = self._adapter_lookup(subj, f"{context} {subj} {obj}", limit=5)
        obj_l = _norm_token(obj)
        evidence: Dict[str, Any] = {"adapter_rows": len(rows), "matched": False, "top_gloss": ""}
        score = 0.0
        for i, row in enumerate(rows):
            gloss = str(row.get("gloss") or "")
            pos = _canonical_pos(row.get("pos") or "")
            low = _norm_token(gloss)
            if i == 0:
                evidence["top_gloss"] = gloss[:260]
            if pos == "n":
                score = max(score, 0.05)
            if obj_l and obj_l in low:
                score = max(score, 0.72 - 0.05 * i)
                evidence["matched"] = True
                evidence["match_type"] = "object_in_adapter_gloss"
            if obj_l == "animal" and (_BROAD_ANIMAL_TERMS & set(_content_tokens(low))):
                score = max(score, 0.68 - 0.04 * i)
                evidence["matched"] = True
                evidence["match_type"] = "animal_taxonomy_gloss"
        return float(score), evidence

    def _definition_evidence(self, subject: SlotCandidate, obj: SlotCandidate, prompt_text: str) -> Tuple[float, Dict[str, Any]]:
        se = self.entries.get(subject.key)
        oe = self.entries.get(obj.key)
        subj_lemma = _norm_token(subject.lemma)
        obj_lemma = _norm_token(obj.lemma)
        subj_gloss = _gloss(se) if se is not None else ""
        obj_gloss = _gloss(oe) if oe is not None else ""
        subj_tokens = set(_content_tokens(subj_gloss))
        evidence: Dict[str, Any] = {
            "subject": subject.lemma, "object": obj.lemma, "signals": [],
            "subject_gloss": subj_gloss[:260], "object_gloss": obj_gloss[:260],
        }
        score = 0.0

        if not subject.lemma or not obj.lemma or subj_lemma == obj_lemma:
            evidence["signals"].append("empty_or_duplicate")
            return 0.0, evidence

        low_subj_gloss = _norm_token(subj_gloss)
        if obj_lemma and (obj_lemma in low_subj_gloss or _singularish(obj_lemma) in subj_tokens):
            score = max(score, 0.78)
            evidence["signals"].append("object_in_subject_gloss")

        if obj_lemma == "animal" and (_BROAD_ANIMAL_TERMS & subj_tokens):
            score = max(score, 0.70)
            evidence["signals"].append("broad_animal_taxonomy")

        if se is not None:
            rels = _relations(se)
            rel_text = json.dumps(_json_safe(rels), ensure_ascii=False).lower()
            rel_names = {_norm_token(k) for k in rels.keys()}
            if rel_names & _DEFINITION_RELATIONS and obj_lemma and obj_lemma in rel_text:
                score = max(score, 0.83)
                evidence["signals"].append("relation_mentions_object")
            elif rel_names & _DEFINITION_RELATIONS:
                score = max(score, 0.30)
                evidence["signals"].append("definition_relation_present")

        adapter_score, adapter_evidence = self._adapter_supports_definition(subject.lemma, obj.lemma, prompt_text)
        if adapter_score > 0:
            score = max(score, adapter_score)
            evidence["adapter"] = adapter_evidence
            evidence["signals"].append("adapter_support")

        if self.cfg.reject_proper_noun_definitions:
            subj_entity = _is_capitalized_entity(subject.lemma, subj_gloss)
            obj_entity = _is_capitalized_entity(obj.lemma, obj_gloss)
            evidence["entity_guard"] = {"subject_entity": subj_entity, "object_entity": obj_entity}
            if (subj_entity or obj_entity) and "object_in_subject_gloss" not in evidence["signals"] and "relation_mentions_object" not in evidence["signals"]:
                score = min(score, 0.20)
                evidence["signals"].append("entity_guard_penalty")

        unsafe = _unsafe_gloss_penalty(subj_gloss) + _unsafe_gloss_penalty(obj_gloss)
        if unsafe > 0:
            evidence["unsafe_penalty"] = unsafe
            score = max(0.0, score - min(0.50, unsafe * 0.15))

        evidence["score"] = float(score)
        return float(score), evidence

    def _candidate_is_safe(self, candidate: SentenceCandidate, prompt_text: str, analysis: PromptAnalysis) -> Tuple[bool, Dict[str, Any]]:
        if not self.cfg.safe_mode:
            return True, {"safe_mode": False, "accepted": True}
        diagnostics: Dict[str, Any] = {"safe_mode": True, "plan": candidate.plan, "accepted": True}

        anchor_ok, anchor_info = self._candidate_prompt_anchor_check(candidate, analysis)
        diagnostics["prompt_anchor"] = anchor_info
        if not anchor_ok:
            diagnostics.update({"accepted": False, "reason": "unanchored_core_slot"})
            return False, diagnostics

        has_entity, entity_slots = self._candidate_has_entity_slot(candidate)
        if has_entity:
            diagnostics["entity_slots"] = entity_slots

            # Entity-heavy prompts must not be converted into factual or
            # relation-looking statements unless the definition gate below can
            # verify the relation. For example, "Cameroonian Haydn" should not
            # become "A Cameroonian relates to a Haydn"; it should surface a
            # safety notice that the relation is unverified.
            if candidate.plan in {"action", "quality", "comparison", "observation", "relation", "context_note"}:
                diagnostics.update({
                    "accepted": False,
                    "reason": "entity_pair_requires_verified_relation",
                    "entity_slots": entity_slots,
                })
                return False, diagnostics

        if candidate.plan == "definition":
            subj = candidate.slots.get("subject")
            obj = candidate.slots.get("object")
            if subj is None or obj is None:
                diagnostics.update({"accepted": False, "reason": "missing_definition_slots"})
                return False, diagnostics
            score, evidence = self._definition_evidence(subj, obj, prompt_text)
            diagnostics["definition_evidence"] = evidence
            if self.cfg.allow_unsafe_definition or score >= self.cfg.definition_evidence_threshold:
                diagnostics["accepted"] = True
                return True, diagnostics
            diagnostics.update({"accepted": False, "reason": "insufficient_definition_evidence"})
            return False, diagnostics

        if candidate.plan in {"action", "quality", "comparison"}:
            for slot in candidate.slots.values():
                e = self.entries.get(slot.key)
                if e is not None and _unsafe_gloss_penalty(_gloss(e)) >= 0.75:
                    diagnostics.update({"accepted": False, "reason": "unsafe_slot_gloss", "slot": slot.to_dict()})
                    return False, diagnostics
        return True, diagnostics

    # ------------------------------------------------------------------
    # Candidate construction
    # ------------------------------------------------------------------
    def _combine_plan_candidates(self, plan: SentencePlan, slot_cands: Mapping[str, List[SlotCandidate]], *, prompt_text: str, analysis: PromptAnalysis) -> List[SentenceCandidate]:
        slots = plan.slot_names()
        if any(not slot_cands.get(s) for s in slots):
            return []
        out: List[SentenceCandidate] = []

        def rec(i: int, chosen: Dict[str, SlotCandidate], used: set[str]) -> None:
            if len(out) >= self.cfg.max_combinations:
                return
            if i >= len(slots):
                score = float(np.mean([c.score for c in chosen.values()])) if chosen else 0.0
                lemmas = [_norm_token(c.lemma) for c in chosen.values()]
                duplicate_count = len(lemmas) - len(set(lemmas))
                score -= self.cfg.diversity_penalty * duplicate_count
                cand = SentenceCandidate(
                    sentence=self.realize(plan, chosen),
                    score=float(score),
                    plan=plan.name,
                    slots=dict(chosen),
                    diagnostics={"duplicate_lemma_count": int(duplicate_count)},
                )
                ok, safety = self._candidate_is_safe(cand, prompt_text, analysis)
                cand.diagnostics["safety"] = safety
                if ok:
                    out.append(cand)
                return

            slot = slots[i]
            for cand in slot_cands[slot]:
                if not self.cfg.allow_reuse and cand.key in used:
                    continue
                if not self.cfg.allow_reuse and any(_norm_token(cand.lemma) == _norm_token(prev.lemma) for prev in chosen.values()):
                    continue
                chosen[slot] = cand
                used.add(cand.key)
                rec(i + 1, chosen, used)
                used.discard(cand.key)
                chosen.pop(slot, None)

        rec(0, {}, set())
        out.sort(key=lambda c: (-c.score, c.sentence))
        return out

    def _adapter_definition_fallback(self, prompt: str) -> List[SentenceCandidate]:
        toks = _content_tokens(prompt)
        if len(toks) < 2:
            return []
        subj, obj = toks[0], toks[1]
        rows = self._adapter_lookup(subj, f"{prompt} {subj} {obj}", limit=3)
        if not rows:
            return []
        top_gloss = str(rows[0].get("gloss") or "")
        if obj in _norm_token(top_gloss) or (obj == "animal" and (_BROAD_ANIMAL_TERMS & set(_content_tokens(top_gloss)))):
            s = SlotCandidate("subject", f"adapter:{subj}", subj, "n", 0.5, {"adapter_fallback": 1.0})
            o = SlotCandidate("object", f"adapter:{obj}", obj, "n", 0.5, {"adapter_fallback": 1.0})
            sent = _capitalize_sentence(f"{_article_for(subj)} {subj} is {_article_for(obj)} {obj}")
            return [SentenceCandidate(sent, 0.5, "definition", {"subject": s, "object": o}, {"fallback": True, "source": "adapter_definition", "adapter_top_gloss": top_gloss[:260]})]
        return []

    def _fallback_candidates(self, prompt: str, analysis: PromptAnalysis, *, n: int = 1) -> List[SentenceCandidate]:
        adapter_defs = self._adapter_definition_fallback(prompt)
        if adapter_defs:
            return adapter_defs[: int(max(1, n))]

        noun_keys = analysis.keys_for_pos("n")
        if len(noun_keys) >= 2:
            s_key, o_key = noun_keys[0], noun_keys[1]
            se, oe = self.entries.get(s_key), self.entries.get(o_key)
            if se is not None and oe is not None:
                subj = SlotCandidate("subject", s_key, _lemma(se, s_key), _pos(se), 0.0, {"fallback": 1.0})
                obj = SlotCandidate("object", o_key, _lemma(oe, o_key), _pos(oe), 0.0, {"fallback": 1.0})
                if _is_capitalized_entity(subj.lemma, _gloss(se)) or _is_capitalized_entity(obj.lemma, _gloss(oe)):
                    sent = _capitalize_sentence(f"{subj.lemma.replace('_', ' ')} and {obj.lemma.replace('_', ' ')} need a verified semantic relation before a sentence can be generated")
                    return [SentenceCandidate(sent, 0.0, "safety_notice", {"subject": subj, "object": obj}, {"fallback": True, "reason": "entity_pair_needs_verified_relation"})]
                plan = BUILTIN_PLANS["context_note"]
                sent = self.realize(plan, {"subject": subj, "object": obj})
                return [SentenceCandidate(sent, 0.0, "context_note", {"subject": subj, "object": obj}, {"fallback": True, "reason": "no_verified_factual_sentence"})]

        clean = _shorten(prompt, 120) or "the prompt"
        return [SentenceCandidate(
            sentence=_capitalize_sentence(f"The prompt '{clean}' needs a verified semantic relation before sentence generation"),
            score=0.0,
            plan="safety_notice",
            slots={},
            diagnostics={"fallback": True, "reason": "insufficient_anchors"},
        )]

    def _entropy_rerank(self, ranked_list: List[SentenceCandidate], prompt: str, n: int) -> List[SentenceCandidate]:
        if not self.cfg.entropy_rerank or not ranked_list:
            return ranked_list[: int(max(1, n))]
        try:
            try:
                import entropy_nlp  # type: ignore
            except ImportError:
                import local_ai.entropy_nlp as entropy_nlp  # type: ignore
            pos_lookup = None
            if hasattr(entropy_nlp, "make_pos_lookup_from_semantic_bank"):
                try:
                    pos_lookup = entropy_nlp.make_pos_lookup_from_semantic_bank(self.bank)
                except Exception:
                    pos_lookup = None
            reranked = entropy_nlp.rerank_texts(
                [c.to_dict() for c in ranked_list],
                context=str(prompt),
                profile="curriculum",
                pos_lookup=pos_lookup,
                top_k=int(max(1, n)),
            )
            by_sentence = {c.sentence.lower(): c for c in ranked_list}
            final: List[SentenceCandidate] = []
            for item in reranked or []:
                if not isinstance(item, Mapping):
                    continue
                text = str(item.get("text") or item.get("sentence") or "")
                orig = by_sentence.get(text.lower())
                if orig is None:
                    continue
                final.append(SentenceCandidate(
                    sentence=orig.sentence,
                    score=float(item.get("score", orig.score)),
                    plan=orig.plan,
                    slots=orig.slots,
                    diagnostics={**orig.diagnostics, "entropy_nlp": item.get("diagnostics", {}), "entropy_rank": item.get("rank")},
                ))
            if final:
                return final[: int(max(1, n))]
        except Exception as exc:
            if self.cfg.debug:
                print(f"DEBUG: entropy_nlp reranking failed: {exc}", file=sys.stderr, flush=True)
        return ranked_list[: int(max(1, n))]

    def _slot_lemma_matches_token(self, cand: Optional[SlotCandidate], token: str) -> bool:
        if cand is None:
            return False
        lemma = _norm_token(cand.lemma).replace("_", " ")
        tok = _norm_token(token).replace("_", " ")
        return lemma == tok or _singularish(lemma) == _singularish(tok)

    def _prompt_order_score(self, cand: SentenceCandidate, analysis: PromptAnalysis) -> float:
        """Reward subject/object order matching prompt token order."""
        toks = list(analysis.tokens or [])
        if len(toks) < 2:
            return 0.0
        subj = cand.slots.get("subject")
        obj = cand.slots.get("object")
        forward = self._slot_lemma_matches_token(subj, toks[0]) and self._slot_lemma_matches_token(obj, toks[1])
        reverse = self._slot_lemma_matches_token(subj, toks[1]) and self._slot_lemma_matches_token(obj, toks[0])
        if forward:
            return 1.0
        if reverse:
            return -0.35
        return 0.0

    def _definition_evidence_score(self, cand: SentenceCandidate) -> float:
        safety = cand.diagnostics.get("safety", {}) if isinstance(cand.diagnostics, Mapping) else {}
        evidence = safety.get("definition_evidence", {}) if isinstance(safety, Mapping) else {}
        try:
            return float(evidence.get("score", 0.0))
        except Exception:
            return 0.0

    def _candidate_sort_key(self, cand: SentenceCandidate, analysis: PromptAnalysis) -> Tuple[Any, ...]:
        """
        Production-facing ordering:
        1. verified definitions in prompt order,
        2. other verified definitions,
        3. cautious relation/context outputs,
        4. safety notices.
        """
        safety = cand.diagnostics.get("safety", {}) if isinstance(cand.diagnostics, Mapping) else {}
        accepted = bool(safety.get("accepted", True)) if isinstance(safety, Mapping) else True
        ev_score = self._definition_evidence_score(cand)
        order_score = self._prompt_order_score(cand, analysis)

        if cand.plan == "definition" and accepted and ev_score >= self.cfg.definition_evidence_threshold:
            plan_rank = 0
        elif cand.plan == "definition" and accepted:
            plan_rank = 1
        elif cand.plan == "relation":
            plan_rank = 3
        elif cand.plan == "context_note":
            plan_rank = 4
        elif cand.plan == "safety_notice":
            plan_rank = 6
        else:
            plan_rank = 2

        return (
            plan_rank,
            -order_score,
            -ev_score,
            -float(cand.score),
            cand.sentence.lower(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        prompt: Union[str, Sequence[str]],
        *,
        n: int = 5,
        template: Optional[str] = None,
        forced_slots: Optional[Mapping[str, str]] = None,
        return_all_plans: bool = False,
    ) -> List[SentenceCandidate]:
        prompt_text = prompt if isinstance(prompt, str) else " ".join(str(x) for x in prompt)
        prompt_text = _normalize_space(prompt_text)
        if not prompt_text:
            return [SentenceCandidate("The prompt needs text before sentence generation.", 0.0, "safety_notice", {}, {"fallback": True, "reason": "empty_prompt"})]

        cfg = self.cfg
        analysis = self.analyze_prompt(prompt_text)
        intent = self.intent_vector(prompt_text)
        seed_keys = self.resolve_keys(prompt_text)
        forced_slots = dict(forced_slots or {})
        plans = self.choose_plans(prompt_text, template=template)

        all_candidates: List[SentenceCandidate] = []
        for plan in plans:
            slot_cands: Dict[str, List[SlotCandidate]] = {}
            for slot, pos in plan.slots:
                forced = forced_slots.get(slot)
                if forced:
                    forced_keys = self.resolve_keys([forced], max_per_token=cfg.max_candidates_per_slot)
                    if forced in self.entries:
                        forced_keys = [forced]
                    forced_list: List[SlotCandidate] = []
                    for fk in forced_keys[: cfg.max_candidates_per_slot]:
                        e = self.entries.get(fk)
                        if e is not None:
                            forced_list.append(SlotCandidate(slot, fk, _lemma(e, fk), _pos(e), 1.25, {"forced": 1.0}))
                    slot_cands[slot] = forced_list
                else:
                    slot_cands[slot] = self.rank_slot_candidates(
                        slot,
                        pos,
                        intent,
                        prompt=prompt_text,
                        analysis=analysis,
                        seed_keys=seed_keys,
                        exclude_keys=set(),
                        limit=cfg.max_candidates_per_slot,
                    )

            plan_candidates = self._combine_plan_candidates(plan, slot_cands, prompt_text=prompt_text, analysis=analysis)
            if cfg.debug:
                for cand in plan_candidates:
                    cand.diagnostics["slot_candidate_counts"] = {s: len(v) for s, v in slot_cands.items()}
                    cand.diagnostics["seed_keys"] = list(seed_keys[:24])
                    cand.diagnostics["prompt_tokens"] = list(analysis.tokens)
                    cand.diagnostics["exact_keys_by_token"] = {k: v[:8] for k, v in analysis.exact_keys_by_token.items()}
            all_candidates.extend(plan_candidates if return_all_plans else plan_candidates[: max(1, int(n))])

        best_by_sentence: Dict[str, SentenceCandidate] = {}
        for cand in all_candidates:
            key = cand.sentence.lower()
            prev = best_by_sentence.get(key)
            if prev is None or cand.score > prev.score:
                best_by_sentence[key] = cand

        ranked_list = sorted(best_by_sentence.values(), key=lambda c: self._candidate_sort_key(c, analysis))
        if not ranked_list and cfg.fallback_safe_sentence:
            ranked_list = self._fallback_candidates(prompt_text, analysis, n=n)

        if not ranked_list:
            ranked_list = [SentenceCandidate(
                sentence=_capitalize_sentence(f"The prompt '{_shorten(prompt_text, 120)}' needs a verified semantic relation before sentence generation"),
                score=0.0,
                plan="safety_notice",
                slots={},
                diagnostics={"fallback": True, "reason": "non_empty_invariant"},
            )]

        return self._entropy_rerank(ranked_list, prompt_text, int(max(1, n)))

    def explain(self, candidate: SentenceCandidate) -> Dict[str, Any]:
        return candidate.to_dict()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "SemanticSentenceBuilder",
            "entry_count": int(len(self.entries)),
            "vector_count": int(len(self.vectors)),
            "pos_counts": {k: len(v) for k, v in self.index_by_pos.items()},
            "exact_lemma_count": int(len(self.index_by_lemma_exact)),
            "token_index_count": int(len(self.index_by_token)),
            "config": _json_safe(self.cfg),
            "safe_mode": bool(self.cfg.safe_mode),
        }

    def health_metrics(self) -> Dict[str, Any]:
        vector_dims = sorted(set(int(v.size) for v in self.vectors.values()))
        return {
            "kind": "SemanticSentenceBuilder",
            "is_stable": bool(self.entries and (not self.vectors or len(vector_dims) <= 1)),
            "entry_count": int(len(self.entries)),
            "vector_count": int(len(self.vectors)),
            "vector_dims": vector_dims,
            "known_pos": sorted(k for k in self.index_by_pos.keys() if k),
            "exact_lemma_count": int(len(self.index_by_lemma_exact)),
        }


# =============================================================================
# Convenience API
# =============================================================================


def build_sentences(
    bank_or_path: Union[SemanticAttractorBank, str, Path],
    prompt: Union[str, Sequence[str]],
    *,
    n: int = 5,
    template: Optional[str] = None,
    cfg: Optional[SentenceBuilderConfig] = None,
) -> List[SentenceCandidate]:
    builder = SemanticSentenceBuilder.from_bank_path(bank_or_path, cfg=cfg) if isinstance(bank_or_path, (str, Path)) else SemanticSentenceBuilder(bank_or_path, cfg=cfg)
    try:
        return builder.build(prompt, n=n, template=template)
    finally:
        builder.close()


def candidates_to_json(candidates: Sequence[SentenceCandidate], *, indent: Optional[int] = 2) -> str:
    return json.dumps([c.to_dict() if hasattr(c, "to_dict") else _json_safe(c) for c in candidates], ensure_ascii=False, indent=indent)


# =============================================================================
# CLI
# =============================================================================


def _parse_forced_slots(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and v:
            out[k] = v
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build safe semantic sentence candidates from a SemanticAttractorBank.")
    ap.add_argument("bank", help="Path to semantic bank .npz, .json, or compatible DB")
    ap.add_argument("prompt", nargs="+", help="Prompt / seed terms")
    ap.add_argument("--n", type=int, default=5, help="Number of candidates to print")
    ap.add_argument("--template", default="auto", choices=["auto"] + sorted(BUILTIN_PLANS.keys()), help="Sentence template")
    ap.add_argument("--seed", type=int, default=0, help="Deterministic tie-break seed")
    ap.add_argument("--tense", default="present", choices=["present", "past", "bare"], help="Verb tense")
    ap.add_argument("--number", default="singular", choices=["singular", "plural"], help="Subject/object number")
    ap.add_argument("--definite", action="store_true", help="Use definite articles where applicable")
    ap.add_argument("--allow-reuse", action="store_true", help="Allow the same entry to fill multiple slots")
    ap.add_argument("--force", action="append", default=[], help="Force slot value, e.g. --force subject=cat --force object=animal")
    ap.add_argument("--unsafe", action="store_true", help="Disable safety gates for debugging only")
    ap.add_argument("--no-adapter", action="store_true", help="Disable local_ai_adapter validation")
    ap.add_argument("--no-entropy", action="store_true", help="Disable entropy_nlp reranking")
    ap.add_argument("--json", action="store_true", help="Print full JSON diagnostics")
    ap.add_argument("--debug", action="store_true", help="Include debug diagnostics")
    ap.add_argument("--self-test", action="store_true", help="Run built-in self-test and exit")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    cfg = SentenceBuilderConfig(
        seed=int(args.seed),
        default_template=args.template,
        tense=args.tense,
        number=args.number,
        definite_article=bool(args.definite),
        allow_reuse=bool(args.allow_reuse),
        safe_mode=not bool(args.unsafe),
        adapter_validation=not bool(args.no_adapter),
        entropy_rerank=not bool(args.no_entropy),
        debug=bool(args.debug),
    )
    builder = SemanticSentenceBuilder.from_bank_path(args.bank, cfg=cfg)
    try:
        prompt = " ".join(args.prompt)
        candidates = builder.build(prompt, n=int(args.n), template=args.template, forced_slots=_parse_forced_slots(args.force))
        if args.json:
            try:
                print(candidates_to_json(candidates, indent=2), flush=True)
            except Exception as exc:
                print(json.dumps({"ok": False, "error": f"JSON formatting failed: {exc}", "candidate_count": len(candidates)}, ensure_ascii=False, indent=2), flush=True)
                return 1
        else:
            if not candidates:
                print("No safe sentence candidates were generated.", flush=True)
            for i, cand in enumerate(candidates, start=1):
                print(f"{i}. {cand.sentence}  [score={cand.score:.4f}; plan={cand.plan}]", flush=True)
                if args.debug:
                    print(json.dumps(_json_safe(cand.diagnostics), ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        builder.close()


# =============================================================================
# Self-test
# =============================================================================


def _self_test() -> None:
    from semantic_attractors import SemanticAttractorConfig, SemanticAttractorBank, SemanticEntry

    entries = [
        SemanticEntry(key="en:cat:n:1", lemma="cat", pos="n", gloss="small domesticated feline animal", tokens=("cat", "feline", "animal"), relations={"hypernyms": ["en:animal:n:1"]}),
        SemanticEntry(key="en:animal:n:1", lemma="animal", pos="n", gloss="living organism", tokens=("animal", "organism")),
        SemanticEntry(key="en:gentle:adj:1", lemma="gentle", pos="adj", gloss="mild kind soft", tokens=("gentle", "kind", "soft")),
        SemanticEntry(key="en:chase:v:1", lemma="chase", pos="v", gloss="pursue quickly", tokens=("chase", "pursue", "move")),
        SemanticEntry(key="en:cameroonian:n:1", lemma="Cameroonian", pos="n", gloss="a native or inhabitant of Cameroon", tokens=("cameroonian",)),
        SemanticEntry(key="en:haydn:n:1", lemma="Haydn", pos="n", gloss="surname of a composer", tokens=("haydn",)),
    ]
    bank = SemanticAttractorBank.from_entries(entries, dim=32, seed=3, config=SemanticAttractorConfig(dim=32, seed=3))
    builder = SemanticSentenceBuilder(bank, cfg=SentenceBuilderConfig(seed=3, debug=True, adapter_validation=False, entropy_rerank=False))
    try:
        cands = builder.build("cat animal", n=3)
        assert cands, "expected candidates"
        assert any(c.sentence.lower() == "a cat is an animal." for c in cands), [str(c) for c in cands]
        bad = builder.build("Cameroonian Haydn", n=1)
        assert bad, "expected safe fallback"
        assert "is a haydn" not in bad[0].sentence.lower(), bad[0].sentence
        print("sentence_builder.py self-test passed", flush=True)
    finally:
        builder.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SentenceBuilderConfig",
    "SentencePlan",
    "SlotCandidate",
    "SentenceCandidate",
    "SemanticSentenceBuilder",
    "BUILTIN_PLANS",
    "build_sentences",
    "candidates_to_json",
]
