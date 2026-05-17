#!/usr/bin/env python3
# twin_any_language.py
r"""
Project Chimera / Akkurat - Twin Any Language
=============================================

Production CPU-first linguistic twin layer for the Akkurat cognitive stack.

This module converts raw linguistic objects — words, phrases, clauses,
sentences, structured intents, semantic frames, and grammar plans — into stable,
serializable, verifiable language twins. It is designed to sit between:

    - tn.py
        TensorTrain / Tucker / tree tensor primitives used for compressed
        projection, grammar-law operators, and semantic-to-grammar planning.

    - digital_twin_kernel.py
        Governed digital-twin substrate used for deterministic projection,
        latent geometry, node-state tracking, sandboxed actions, histories,
        Merkle-style verification, and governed execution.

Design goals
------------
- Deterministic and portable: NumPy-only required, optional tn.py and
  digital_twin_kernel.py integration.
- Production-safe parsing defaults: no network calls, no model dependency,
  no hidden mutable global runtime beyond bounded caches.
- Structured linguistic twins: lexical, phrase, sentence, semantic-frame,
  grammar-plan, and language-intent twins.
- Explainable grammar laws: subject-verb agreement, determiner-noun agreement,
  clause completeness, verb valency, basic English word order, and semantic-role
  compatibility.
- Projection-ready payloads: JSON-safe canonical representation and optional
  digital_twin_kernel.ProjectionEngine latent encoding.
- Tensor-network ready features: deterministic categorical encoders and compact
  numeric feature vectors suitable for tn.TensorTrain operators.

Important scope note
--------------------
This module is not a full statistical NLP pipeline. It provides a governed,
mathematically structured language-twin substrate. It can later be connected to
LLMs, morphological analyzers, dependency parsers, or curated lexicons, but it
intentionally ships with deterministic rule-based fallback behavior.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np


# =============================================================================
# Optional local imports: tn.py and digital_twin_kernel.py
# =============================================================================


def _add_project_paths() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent]
    for p in candidates:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return here.parent.parent if len(here.parents) >= 2 else here.parent


_AKKURAT_ROOT = _add_project_paths()

_TN_OK = False
_TN_ERR = ""
tn = None
try:
    import tn as tn  # type: ignore
    _TN_OK = True
except Exception as e1:  # pragma: no cover - optional dependency
    try:
        from . import tn as tn  # type: ignore
        _TN_OK = True
    except Exception as e2:  # pragma: no cover - optional dependency
        _TN_OK = False
        _TN_ERR = f"tn.py import failed: {repr(e1)} | {repr(e2)}"

_DTK_OK = False
_DTK_ERR = ""
dtk = None
try:
    import digital_twin_kernel as dtk  # type: ignore
    _DTK_OK = True
except Exception as e1:  # pragma: no cover - optional dependency
    try:
        from . import digital_twin_kernel as dtk  # type: ignore
        _DTK_OK = True
    except Exception as e2:  # pragma: no cover - optional dependency
        _DTK_OK = False
        _DTK_ERR = f"digital_twin_kernel.py import failed: {repr(e1)} | {repr(e2)}"


# =============================================================================
# Constants and low-level utilities
# =============================================================================

_EPS = 1e-9
_DEFAULT_VECTOR_DIM = 256
_WORD_RE = re.compile(r"[\w]+(?:[-'][\w]+)*|[^\w\s]", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?]+$")
_VOWEL_SOUND_RE = re.compile(r"^[aeiouAEIOU]")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _stable_hash_u32(s: str) -> int:
    h = 2166136261
    for b in (s or "").encode("utf-8", errors="ignore"):
        h ^= int(b)
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _bytes_hash_u64(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h ^= int(b)
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(h)


def _canonical_text(s: Any) -> str:
    if s is None:
        return ""
    text = str(s)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonical_key(s: Any) -> str:
    text = _canonical_text(s).casefold()
    text = re.sub(r"\s+", "_", text)
    return text


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else str(obj)
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, np.ndarray):
        arr = np.nan_to_num(np.asarray(obj, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(float).tolist()
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
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


def stable_payload_hash(payload: Mapping[str, Any]) -> str:
    b = json.dumps(_json_safe(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8", errors="ignore")
    return f"{_bytes_hash_u64(b):016x}"


def _sanitize_array(x: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    try:
        arr = np.asarray(x, dtype=dtype).reshape(-1)
    except Exception:
        arr = np.zeros(0, dtype=dtype)
    if arr.size == 0:
        return arr.astype(dtype, copy=False)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype, copy=False)


def _l2_normalize(v: Any, eps: float = _EPS) -> np.ndarray:
    x = _sanitize_array(v, np.float32)
    if x.size == 0:
        return x
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / n).astype(np.float32, copy=False)


def _hash_vector(payload: Any, *, dim: int = _DEFAULT_VECTOR_DIM, seed: int = 2027, key: str = "language") -> np.ndarray:
    """Deterministic signed hashing projection for any JSON-safe payload."""
    dim = int(max(1, dim))
    out = np.zeros(dim, dtype=np.float32)
    safe = _json_safe(payload)
    text = json.dumps(safe, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    if not text:
        return out
    tokens = re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)
    base = _stable_hash_u32(f"{seed}:{key}:{len(tokens)}")
    for i, tok in enumerate(tokens[:8192]):
        h = _stable_hash_u32(f"{base}:{i}:{tok}")
        sign = -1.0 if (h & 1) else 1.0
        out[int(h % dim)] += np.float32(sign)
    return _l2_normalize(out)


def tokenize(text: str) -> List[str]:
    text = _canonical_text(text)
    if not text:
        return []
    return [m.group(0) for m in _WORD_RE.finditer(text)]


def detokenize(tokens: Sequence[str]) -> str:
    out: List[str] = []
    no_space_before = set(".,!?;:%)]}")
    no_space_after = set("([{£$€#")
    for tok in tokens:
        if not tok:
            continue
        if not out:
            out.append(tok)
        elif tok in no_space_before:
            out[-1] = out[-1] + tok
        elif out[-1] and out[-1][-1] in no_space_after:
            out[-1] = out[-1] + tok
        elif tok == "'" or tok.startswith("'"):
            out[-1] = out[-1] + tok
        else:
            out.append(tok)
    return " ".join(out).strip()


def ensure_sentence_terminal(text: str, terminal: str = ".") -> str:
    text = _canonical_text(text)
    if not text:
        return text
    if _SENTENCE_END_RE.search(text):
        return text
    return text + terminal


# =============================================================================
# Public enums and configs
# =============================================================================


class TwinKind(str, Enum):
    LEXICAL = "lexical"
    PHRASE = "phrase"
    SENTENCE = "sentence"
    SEMANTIC_FRAME = "semantic_frame"
    GRAMMAR_PLAN = "grammar_plan"
    INTENT = "intent"
    GENERIC_LANGUAGE = "generic_language"


class PartOfSpeech(str, Enum):
    UNKNOWN = "unknown"
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    DETERMINER = "determiner"
    PRONOUN = "pronoun"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    AUXILIARY = "auxiliary"
    PARTICLE = "particle"
    PUNCTUATION = "punctuation"
    NUMERAL = "numeral"


class Number(str, Enum):
    UNKNOWN = "unknown"
    SINGULAR = "singular"
    PLURAL = "plural"
    MASS = "mass"


class Person(str, Enum):
    UNKNOWN = "unknown"
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"


class Tense(str, Enum):
    UNKNOWN = "unknown"
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    INFINITIVE = "infinitive"
    GERUND = "gerund"
    PARTICIPLE = "participle"


class Voice(str, Enum):
    UNKNOWN = "unknown"
    ACTIVE = "active"
    PASSIVE = "passive"


class ClauseType(str, Enum):
    UNKNOWN = "unknown"
    DECLARATIVE = "declarative"
    INTERROGATIVE = "interrogative"
    IMPERATIVE = "imperative"
    EXCLAMATIVE = "exclamative"
    FRAGMENT = "fragment"


class SemanticRole(str, Enum):
    UNKNOWN = "unknown"
    AGENT = "agent"
    PATIENT = "patient"
    THEME = "theme"
    RECIPIENT = "recipient"
    EXPERIENCER = "experiencer"
    INSTRUMENT = "instrument"
    LOCATION = "location"
    TIME = "time"
    MANNER = "manner"
    CAUSE = "cause"
    ATTRIBUTE = "attribute"


class GrammarSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class LanguageTwinConfig:
    language: str = "en"
    vector_dim: int = _DEFAULT_VECTOR_DIM
    seed: int = 2027
    strict: bool = False
    use_digital_projection: bool = True
    use_tn_projection: bool = True
    latent_geometry: str = "euclidean"
    default_style: str = "neutral"
    allow_fragments: bool = False
    allow_poetic_semantics: bool = False
    max_tokens: int = 512
    eps: float = _EPS

    def normalized(self) -> "LanguageTwinConfig":
        lang = _canonical_key(self.language or "en") or "en"
        geom = str(self.latent_geometry or "euclidean").lower().strip()
        if geom not in {"euclidean", "hyperbolic", "poincare", "poincaré"}:
            geom = "euclidean"
        return LanguageTwinConfig(
            language=lang,
            vector_dim=int(max(8, self.vector_dim)),
            seed=int(self.seed),
            strict=bool(self.strict),
            use_digital_projection=bool(self.use_digital_projection),
            use_tn_projection=bool(self.use_tn_projection),
            latent_geometry=geom,
            default_style=str(self.default_style or "neutral"),
            allow_fragments=bool(self.allow_fragments),
            allow_poetic_semantics=bool(self.allow_poetic_semantics),
            max_tokens=int(max(1, self.max_tokens)),
            eps=float(self.eps if self.eps > 0 else _EPS),
        )


# =============================================================================
# Linguistic feature dataclasses
# =============================================================================


@dataclass
class MorphFeatures:
    pos: str = PartOfSpeech.UNKNOWN.value
    lemma: str = ""
    number: str = Number.UNKNOWN.value
    person: str = Person.UNKNOWN.value
    tense: str = Tense.UNKNOWN.value
    voice: str = Voice.UNKNOWN.value
    gender: str = "unknown"
    case: str = "unknown"
    degree: str = "unknown"
    aspect: str = "unknown"
    polarity: str = "positive"
    definiteness: str = "unknown"
    countability: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class SemanticFeatures:
    semantic_type: str = "unknown"
    animacy: str = "unknown"
    concreteness: str = "unknown"
    can_be_agent: bool = False
    can_be_patient: bool = False
    can_be_theme: bool = False
    can_be_recipient: bool = False
    roles_required: List[str] = field(default_factory=list)
    role_constraints: Dict[str, List[str]] = field(default_factory=dict)
    alternations: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class GrammarIssue:
    code: str
    message: str
    severity: str = GrammarSeverity.ERROR.value
    span: Optional[Tuple[int, int]] = None
    repair: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class GrammarReport:
    ok: bool
    score: float
    issues: List[GrammarIssue] = field(default_factory=list)
    laws_satisfied: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class LanguageTwin:
    kind: str
    text: str = ""
    language: str = "en"
    canonical: str = ""
    morph: MorphFeatures = field(default_factory=MorphFeatures)
    semantics: SemanticFeatures = field(default_factory=SemanticFeatures)
    tokens: List[str] = field(default_factory=list)
    children: List["LanguageTwin"] = field(default_factory=list)
    roles: Dict[str, str] = field(default_factory=dict)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    latent: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    twin_id: str = ""

    def __post_init__(self) -> None:
        self.text = _canonical_text(self.text)
        self.language = _canonical_key(self.language or "en") or "en"
        self.canonical = self.canonical or _canonical_key(self.text)
        if not self.tokens and self.text:
            self.tokens = tokenize(self.text)
        if not self.twin_id:
            self.twin_id = self.compute_hash()

    def payload(self, *, include_latent: bool = False) -> Dict[str, Any]:
        data = {
            "kind": self.kind,
            "text": self.text,
            "language": self.language,
            "canonical": self.canonical,
            "morph": self.morph.to_dict(),
            "semantics": self.semantics.to_dict(),
            "tokens": list(self.tokens),
            "children": [c.payload(include_latent=include_latent) for c in self.children],
            "roles": dict(self.roles),
            "dependencies": _json_safe(self.dependencies),
            "metadata": _json_safe(self.metadata),
            "created_at": self.created_at,
        }
        if include_latent and self.latent is not None:
            data["latent"] = _json_safe(self.latent)
        return data

    def compute_hash(self) -> str:
        return stable_payload_hash(self.payload(include_latent=False))

    def refresh_id(self) -> str:
        self.twin_id = self.compute_hash()
        return self.twin_id

    def to_dict(self, *, include_latent: bool = False) -> Dict[str, Any]:
        data = self.payload(include_latent=include_latent)
        data["twin_id"] = self.twin_id
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LanguageTwin":
        morph = payload.get("morph", {})
        sem = payload.get("semantics", {})
        children = [cls.from_dict(c) for c in payload.get("children", []) if isinstance(c, Mapping)]
        latent = payload.get("latent", None)
        return cls(
            kind=str(payload.get("kind", TwinKind.GENERIC_LANGUAGE.value)),
            text=str(payload.get("text", "")),
            language=str(payload.get("language", "en")),
            canonical=str(payload.get("canonical", "")),
            morph=MorphFeatures(**{k: morph.get(k, getattr(MorphFeatures(), k)) for k in asdict(MorphFeatures()).keys()}) if isinstance(morph, Mapping) else MorphFeatures(),
            semantics=SemanticFeatures(**{k: sem.get(k, getattr(SemanticFeatures(), k)) for k in asdict(SemanticFeatures()).keys()}) if isinstance(sem, Mapping) else SemanticFeatures(),
            tokens=list(payload.get("tokens", [])),
            children=children,
            roles=dict(payload.get("roles", {})),
            dependencies=list(payload.get("dependencies", [])),
            latent=_sanitize_array(latent, np.float32) if latent is not None else None,
            metadata=dict(payload.get("metadata", {})),
            created_at=str(payload.get("created_at", _now_iso())),
            twin_id=str(payload.get("twin_id", "")),
        )


@dataclass
class SentenceTwinState:
    sentence: str = ""
    tokens: List[str] = field(default_factory=list)
    lexical_twins: List[LanguageTwin] = field(default_factory=list)
    semantic_frame: Dict[str, Any] = field(default_factory=dict)
    grammar_plan: Dict[str, Any] = field(default_factory=dict)
    open_slots: List[str] = field(default_factory=list)
    closed: bool = False
    latent: Optional[np.ndarray] = None
    grammar_report: Optional[GrammarReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_latent: bool = False) -> Dict[str, Any]:
        return {
            "sentence": self.sentence,
            "tokens": list(self.tokens),
            "lexical_twins": [t.to_dict(include_latent=include_latent) for t in self.lexical_twins],
            "semantic_frame": _json_safe(self.semantic_frame),
            "grammar_plan": _json_safe(self.grammar_plan),
            "open_slots": list(self.open_slots),
            "closed": bool(self.closed),
            "latent": _json_safe(self.latent) if include_latent and self.latent is not None else None,
            "grammar_report": self.grammar_report.to_dict() if self.grammar_report else None,
            "metadata": _json_safe(self.metadata),
        }


# =============================================================================
# Lexicon and deterministic morphology fallback
# =============================================================================


_IRREGULAR_VERBS: Dict[str, Dict[str, str]] = {
    "be": {"past_singular": "was", "past_plural": "were", "present_1s": "am", "present_2": "are", "present_3s": "is", "present_plural": "are", "past_participle": "been", "gerund": "being"},
    "have": {"past": "had", "present_3s": "has", "past_participle": "had", "gerund": "having"},
    "do": {"past": "did", "present_3s": "does", "past_participle": "done", "gerund": "doing"},
    "go": {"past": "went", "present_3s": "goes", "past_participle": "gone", "gerund": "going"},
    "give": {"past": "gave", "present_3s": "gives", "past_participle": "given", "gerund": "giving"},
    "eat": {"past": "ate", "present_3s": "eats", "past_participle": "eaten", "gerund": "eating"},
    "run": {"past": "ran", "present_3s": "runs", "past_participle": "run", "gerund": "running"},
    "chase": {"past": "chased", "present_3s": "chases", "past_participle": "chased", "gerund": "chasing"},
    "discover": {"past": "discovered", "present_3s": "discovers", "past_participle": "discovered", "gerund": "discovering"},
    "make": {"past": "made", "present_3s": "makes", "past_participle": "made", "gerund": "making"},
    "see": {"past": "saw", "present_3s": "sees", "past_participle": "seen", "gerund": "seeing"},
    "write": {"past": "wrote", "present_3s": "writes", "past_participle": "written", "gerund": "writing"},
    "say": {"past": "said", "present_3s": "says", "past_participle": "said", "gerund": "saying"},
}

_IRREGULAR_NOUNS: Dict[str, str] = {
    "child": "children",
    "person": "people",
    "man": "men",
    "woman": "women",
    "mouse": "mice",
    "goose": "geese",
    "tooth": "teeth",
    "foot": "feet",
    "ox": "oxen",
}

_REVERSE_IRREGULAR_NOUNS = {v: k for k, v in _IRREGULAR_NOUNS.items()}

_DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their", "some", "any", "each", "every", "no"}
_PREPOSITIONS = {"to", "from", "with", "by", "for", "in", "on", "at", "into", "onto", "over", "under", "through", "during", "before", "after", "of", "about", "as"}
_CONJUNCTIONS = {"and", "or", "but", "because", "although", "while", "if", "when", "since", "unless"}
_PRONOUNS = {
    "i": (Person.FIRST.value, Number.SINGULAR.value),
    "me": (Person.FIRST.value, Number.SINGULAR.value),
    "we": (Person.FIRST.value, Number.PLURAL.value),
    "us": (Person.FIRST.value, Number.PLURAL.value),
    "you": (Person.SECOND.value, Number.UNKNOWN.value),
    "he": (Person.THIRD.value, Number.SINGULAR.value),
    "him": (Person.THIRD.value, Number.SINGULAR.value),
    "she": (Person.THIRD.value, Number.SINGULAR.value),
    "her": (Person.THIRD.value, Number.SINGULAR.value),
    "it": (Person.THIRD.value, Number.SINGULAR.value),
    "they": (Person.THIRD.value, Number.PLURAL.value),
    "them": (Person.THIRD.value, Number.PLURAL.value),
}
_AUXILIARIES = {"am", "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "will", "would", "shall", "should", "can", "could", "may", "might", "must"}

_ENTITY_SEMANTICS: Dict[str, Dict[str, Any]] = {
    "dog": {"semantic_type": "animal", "animacy": "animate", "concreteness": "concrete", "can_be_agent": True, "can_be_patient": True, "can_be_theme": True},
    "cat": {"semantic_type": "animal", "animacy": "animate", "concreteness": "concrete", "can_be_agent": True, "can_be_patient": True, "can_be_theme": True},
    "boy": {"semantic_type": "human", "animacy": "animate", "concreteness": "concrete", "can_be_agent": True, "can_be_patient": True, "can_be_theme": True, "can_be_recipient": True},
    "girl": {"semantic_type": "human", "animacy": "animate", "concreteness": "concrete", "can_be_agent": True, "can_be_patient": True, "can_be_theme": True, "can_be_recipient": True},
    "scientist": {"semantic_type": "human", "animacy": "animate", "concreteness": "concrete", "can_be_agent": True, "can_be_patient": True},
    "book": {"semantic_type": "object", "animacy": "inanimate", "concreteness": "concrete", "can_be_patient": True, "can_be_theme": True},
    "apple": {"semantic_type": "food", "animacy": "inanimate", "concreteness": "concrete", "can_be_patient": True, "can_be_theme": True},
    "particle": {"semantic_type": "physical_object", "animacy": "inanimate", "concreteness": "concrete", "can_be_patient": True, "can_be_theme": True},
    "idea": {"semantic_type": "abstract", "animacy": "inanimate", "concreteness": "abstract", "can_be_theme": True},
}

_VERB_FRAMES: Dict[str, Dict[str, Any]] = {
    "run": {"roles_required": ["agent"], "role_constraints": {"agent": ["human", "animal", "machine"]}, "valency": "intransitive"},
    "sleep": {"roles_required": ["agent"], "role_constraints": {"agent": ["human", "animal"]}, "valency": "intransitive"},
    "bark": {"roles_required": ["agent"], "role_constraints": {"agent": ["animal"]}, "valency": "intransitive"},
    "chase": {"roles_required": ["agent", "patient"], "role_constraints": {"agent": ["human", "animal", "machine"], "patient": ["human", "animal", "object"]}, "valency": "transitive"},
    "eat": {"roles_required": ["agent", "patient"], "role_constraints": {"agent": ["human", "animal"], "patient": ["food", "object"]}, "valency": "transitive"},
    "discover": {"roles_required": ["agent", "patient"], "role_constraints": {"agent": ["human", "organization", "machine"], "patient": ["object", "physical_object", "abstract"]}, "valency": "transitive"},
    "give": {"roles_required": ["agent", "theme", "recipient"], "role_constraints": {"agent": ["human", "organization"], "theme": ["object", "food", "physical_object", "abstract"], "recipient": ["human", "organization", "animal"]}, "valency": "ditransitive", "alternations": ["theme_to_recipient", "recipient_theme"]},
    "write": {"roles_required": ["agent", "theme"], "role_constraints": {"agent": ["human", "machine"], "theme": ["object", "abstract"]}, "valency": "transitive"},
    "make": {"roles_required": ["agent", "theme"], "role_constraints": {"agent": ["human", "machine", "organization"], "theme": ["object", "abstract", "food"]}, "valency": "transitive"},
    "see": {"roles_required": ["experiencer", "theme"], "role_constraints": {"experiencer": ["human", "animal", "machine"], "theme": ["object", "human", "animal", "abstract", "physical_object"]}, "valency": "transitive"},
    "say": {"roles_required": ["agent", "theme"], "role_constraints": {"agent": ["human", "organization"], "theme": ["abstract", "utterance"]}, "valency": "transitive"},
}


class LexiconProvider(Protocol):
    def lookup(self, text: str, *, language: str = "en") -> Optional[Mapping[str, Any]]: ...


class DictLexicon:
    """Small deterministic lexicon with optional user overrides."""

    def __init__(self, entries: Optional[Mapping[str, Mapping[str, Any]]] = None):
        self.entries: Dict[str, Dict[str, Any]] = {}
        if entries:
            for k, v in entries.items():
                self.entries[_canonical_key(k)] = dict(v)

    def lookup(self, text: str, *, language: str = "en") -> Optional[Mapping[str, Any]]:
        key = _canonical_key(text)
        if key in self.entries:
            return copy.deepcopy(self.entries[key])
        return None

    def add(self, text: str, payload: Mapping[str, Any]) -> None:
        self.entries[_canonical_key(text)] = copy.deepcopy(dict(payload))


def singularize_noun(word: str) -> str:
    w = _canonical_key(word)
    if w in _REVERSE_IRREGULAR_NOUNS:
        return _REVERSE_IRREGULAR_NOUNS[w]
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and (w.endswith("ses") or w.endswith("xes") or w.endswith("zes") or w.endswith("ches") or w.endswith("shes")):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def pluralize_noun(lemma: str) -> str:
    w = _canonical_key(lemma)
    if w in _IRREGULAR_NOUNS:
        return _IRREGULAR_NOUNS[w]
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    if re.search(r"(s|x|z|ch|sh)$", w):
        return w + "es"
    return w + "s"


def infer_noun_number(word: str) -> str:
    w = _canonical_key(word)
    if w in _REVERSE_IRREGULAR_NOUNS:
        return Number.PLURAL.value
    if w in _IRREGULAR_NOUNS:
        return Number.SINGULAR.value
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return Number.PLURAL.value
    return Number.SINGULAR.value


def conjugate_verb(lemma: str, *, tense: str = Tense.PRESENT.value, person: str = Person.THIRD.value, number: str = Number.SINGULAR.value, voice: str = Voice.ACTIVE.value) -> str:
    lemma = _canonical_key(lemma)
    tense = str(tense or Tense.PRESENT.value)
    person = str(person or Person.THIRD.value)
    number = str(number or Number.SINGULAR.value)
    irregular = _IRREGULAR_VERBS.get(lemma, {})

    if tense == Tense.PAST.value:
        if lemma == "be":
            return irregular.get("past_plural" if number == Number.PLURAL.value else "past_singular", "was")
        return irregular.get("past", regular_past(lemma))
    if tense == Tense.FUTURE.value:
        return f"will {lemma}"
    if tense == Tense.GERUND.value:
        return irregular.get("gerund", regular_gerund(lemma))
    if tense == Tense.PARTICIPLE.value:
        return irregular.get("past_participle", regular_past(lemma))
    if tense == Tense.INFINITIVE.value:
        return lemma
    if tense == Tense.PRESENT.value:
        if lemma == "be":
            if person == Person.FIRST.value and number == Number.SINGULAR.value:
                return irregular.get("present_1s", "am")
            if person == Person.SECOND.value:
                return irregular.get("present_2", "are")
            if number == Number.PLURAL.value:
                return irregular.get("present_plural", "are")
            return irregular.get("present_3s", "is")
        if person == Person.THIRD.value and number == Number.SINGULAR.value:
            return irregular.get("present_3s", regular_present_3s(lemma))
        return lemma
    return lemma


def regular_past(lemma: str) -> str:
    if lemma.endswith("e"):
        return lemma + "d"
    if re.search(r"[^aeiou]y$", lemma):
        return lemma[:-1] + "ied"
    return lemma + "ed"


def regular_present_3s(lemma: str) -> str:
    if re.search(r"(s|x|z|ch|sh|o)$", lemma):
        return lemma + "es"
    if re.search(r"[^aeiou]y$", lemma):
        return lemma[:-1] + "ies"
    return lemma + "s"


def regular_gerund(lemma: str) -> str:
    if lemma.endswith("ie"):
        return lemma[:-2] + "ying"
    if lemma.endswith("e") and lemma not in {"be", "see"}:
        return lemma[:-1] + "ing"
    return lemma + "ing"


def infer_verb_lemma(word: str) -> str:
    w = _canonical_key(word)
    for lemma, forms in _IRREGULAR_VERBS.items():
        if w == lemma or w in set(forms.values()):
            return lemma
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        stem = w[:-2]
        if re.search(r"(s|x|z|ch|sh|o)$", stem):
            return stem
    if w.endswith("s") and len(w) > 2:
        return w[:-1]
    if w.endswith("ied") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ed") and len(w) > 3:
        stem = w[:-2]
        if stem.endswith("e"):
            return stem
        return stem
    if w.endswith("ing") and len(w) > 4:
        stem = w[:-3]
        return stem
    return w


def infer_verb_tense_number_person(word: str) -> Tuple[str, str, str]:
    w = _canonical_key(word)
    if w in {"was", "were", "did", "had", "went", "gave", "ate", "ran", "saw", "wrote", "said"} or w.endswith("ed"):
        return (Tense.PAST.value, Person.UNKNOWN.value, Number.UNKNOWN.value)
    if w in {"am"}:
        return (Tense.PRESENT.value, Person.FIRST.value, Number.SINGULAR.value)
    if w in {"is", "does", "has"}:
        return (Tense.PRESENT.value, Person.THIRD.value, Number.SINGULAR.value)
    if w in {"are"}:
        return (Tense.PRESENT.value, Person.UNKNOWN.value, Number.PLURAL.value)
    if w.endswith("ing"):
        return (Tense.GERUND.value, Person.UNKNOWN.value, Number.UNKNOWN.value)
    if len(w) > 2 and w.endswith("s") and w not in {"is", "was"}:
        return (Tense.PRESENT.value, Person.THIRD.value, Number.SINGULAR.value)
    return (Tense.PRESENT.value, Person.UNKNOWN.value, Number.UNKNOWN.value)


def choose_indefinite_article(word: str) -> str:
    return "an" if _VOWEL_SOUND_RE.search(_canonical_text(word)) else "a"


# =============================================================================
# Encoders for tensor-network-ready categorical features
# =============================================================================


class CategoricalEncoder:
    def __init__(self, values: Sequence[str], *, unknown: str = "unknown"):
        vals = [str(v) for v in values]
        if unknown not in vals:
            vals = [unknown] + vals
        self.values = vals
        self.unknown = unknown
        self.index = {v: i for i, v in enumerate(vals)}

    def encode(self, value: Any) -> int:
        return int(self.index.get(str(value), self.index[self.unknown]))

    def one_hot(self, value: Any) -> np.ndarray:
        out = np.zeros(len(self.values), dtype=np.float32)
        out[self.encode(value)] = 1.0
        return out

    def __len__(self) -> int:
        return len(self.values)


POS_ENCODER = CategoricalEncoder([p.value for p in PartOfSpeech])
NUMBER_ENCODER = CategoricalEncoder([n.value for n in Number])
PERSON_ENCODER = CategoricalEncoder([p.value for p in Person])
TENSE_ENCODER = CategoricalEncoder([t.value for t in Tense])
VOICE_ENCODER = CategoricalEncoder([v.value for v in Voice])
ROLE_ENCODER = CategoricalEncoder([r.value for r in SemanticRole])
CLAUSE_ENCODER = CategoricalEncoder([c.value for c in ClauseType])
SEM_TYPE_ENCODER = CategoricalEncoder(["unknown", "human", "animal", "object", "food", "physical_object", "abstract", "organization", "machine", "utterance"])
ANIMACY_ENCODER = CategoricalEncoder(["unknown", "animate", "inanimate"])


def lexical_feature_vector(twin: LanguageTwin) -> np.ndarray:
    parts = [
        POS_ENCODER.one_hot(twin.morph.pos),
        NUMBER_ENCODER.one_hot(twin.morph.number),
        PERSON_ENCODER.one_hot(twin.morph.person),
        TENSE_ENCODER.one_hot(twin.morph.tense),
        VOICE_ENCODER.one_hot(twin.morph.voice),
        SEM_TYPE_ENCODER.one_hot(twin.semantics.semantic_type),
        ANIMACY_ENCODER.one_hot(twin.semantics.animacy),
        np.array([
            1.0 if twin.semantics.can_be_agent else 0.0,
            1.0 if twin.semantics.can_be_patient else 0.0,
            1.0 if twin.semantics.can_be_theme else 0.0,
            1.0 if twin.semantics.can_be_recipient else 0.0,
            float(min(32, len(twin.text))) / 32.0,
            float(min(32, len(twin.tokens))) / 32.0,
        ], dtype=np.float32),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


def sentence_feature_vector(state: SentenceTwinState) -> np.ndarray:
    token_count = len(state.tokens)
    pos_counts = np.zeros(len(POS_ENCODER), dtype=np.float32)
    for t in state.lexical_twins:
        pos_counts[POS_ENCODER.encode(t.morph.pos)] += 1.0
    if token_count:
        pos_counts = pos_counts / float(token_count)
    report_score = float(state.grammar_report.score) if state.grammar_report else 0.0
    issue_count = float(len(state.grammar_report.issues)) if state.grammar_report else 0.0
    base = np.array([
        min(token_count, 512) / 512.0,
        1.0 if state.closed else 0.0,
        min(len(state.open_slots), 32) / 32.0,
        report_score,
        min(issue_count, 32) / 32.0,
    ], dtype=np.float32)
    return np.concatenate([base, pos_counts], axis=0).astype(np.float32)


# =============================================================================
# Language twin factory
# =============================================================================


class LanguageTwinFactory:
    """
    Main public factory for creating language twins.

    Parameters
    ----------
    config:
        LanguageTwinConfig for deterministic behavior and projection settings.
    lexicon:
        Optional LexiconProvider for project/domain-specific lexical overrides.
    projection_engine:
        Optional digital_twin_kernel.ProjectionEngine-compatible object. If not
        supplied and digital_twin_kernel.py is importable, a local ProjectionEngine
        is created.
    """

    def __init__(
        self,
        config: Optional[LanguageTwinConfig] = None,
        *,
        lexicon: Optional[LexiconProvider] = None,
        projection_engine: Optional[Any] = None,
    ):
        self.config = (config or LanguageTwinConfig()).normalized()
        self.lexicon = lexicon or DictLexicon()
        self.projection_engine = projection_engine or self._make_projection_engine()

    def _make_projection_engine(self) -> Optional[Any]:
        if not (self.config.use_digital_projection and _DTK_OK and dtk is not None):
            return None
        try:
            cfg = dtk.ProjectionConfig(  # type: ignore[attr-defined]
                vector_dim=int(self.config.vector_dim),
                seed=int(self.config.seed),
                use_tn_projection=bool(self.config.use_tn_projection),
                latent_geometry=str(self.config.latent_geometry),
            )
            return dtk.ProjectionEngine(cfg)  # type: ignore[attr-defined]
        except Exception:
            return None

    def project_payload(self, payload: Any, *, key: str) -> np.ndarray:
        if self.projection_engine is not None:
            try:
                if isinstance(payload, str):
                    return _sanitize_array(self.projection_engine.encode_text(payload, key=key), np.float32)
                if isinstance(payload, np.ndarray) or isinstance(payload, (list, tuple)) and all(isinstance(x, (int, float, np.number)) for x in payload[: min(16, len(payload))]):
                    return _sanitize_array(self.projection_engine.encode_numbers(payload, key=key), np.float32)
                return _sanitize_array(self.projection_engine.encode_json(payload, key=key), np.float32)
            except Exception:
                pass
        return _hash_vector(payload, dim=self.config.vector_dim, seed=self.config.seed, key=key)

    def lexical_twin(self, text: str, *, metadata: Optional[Mapping[str, Any]] = None) -> LanguageTwin:
        text = _canonical_text(text)
        lex = self.lexicon.lookup(text, language=self.config.language)
        if lex is not None:
            twin = self._from_lexicon(text, lex, metadata=metadata)
        else:
            twin = self._infer_lexical_twin(text, metadata=metadata)
        twin.latent = self.project_payload(twin.payload(include_latent=False), key=f"lexical::{twin.canonical}")
        twin.refresh_id()
        return twin

    def _from_lexicon(self, text: str, lex: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]]) -> LanguageTwin:
        morph_payload = dict(lex.get("morph", {})) if isinstance(lex.get("morph", {}), Mapping) else {}
        sem_payload = dict(lex.get("semantics", {})) if isinstance(lex.get("semantics", {}), Mapping) else {}
        morph = MorphFeatures(**{k: morph_payload.get(k, getattr(MorphFeatures(), k)) for k in asdict(MorphFeatures()).keys()})
        sem = SemanticFeatures(**{k: sem_payload.get(k, getattr(SemanticFeatures(), k)) for k in asdict(SemanticFeatures()).keys()})
        md = dict(metadata or {})
        md.update(dict(lex.get("metadata", {})) if isinstance(lex.get("metadata", {}), Mapping) else {})
        return LanguageTwin(
            kind=TwinKind.LEXICAL.value,
            text=text,
            language=self.config.language,
            canonical=str(lex.get("canonical", _canonical_key(text))),
            morph=morph,
            semantics=sem,
            metadata=md,
        )

    def _infer_lexical_twin(self, text: str, *, metadata: Optional[Mapping[str, Any]]) -> LanguageTwin:
        key = _canonical_key(text)
        morph = MorphFeatures(lemma=key)
        sem = SemanticFeatures()
        md = dict(metadata or {})

        if not key:
            pass
        elif re.fullmatch(r"[^\w\s]+", text, re.UNICODE):
            morph.pos = PartOfSpeech.PUNCTUATION.value
            morph.lemma = text
        elif key in _DETERMINERS:
            morph.pos = PartOfSpeech.DETERMINER.value
            morph.lemma = key
            if key in {"a", "an", "this", "that", "each", "every"}:
                morph.number = Number.SINGULAR.value
            elif key in {"these", "those"}:
                morph.number = Number.PLURAL.value
            morph.definiteness = "definite" if key in {"the", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"} else "indefinite"
        elif key in _PRONOUNS:
            morph.pos = PartOfSpeech.PRONOUN.value
            morph.lemma = key
            morph.person, morph.number = _PRONOUNS[key]
            sem.semantic_type = "human" if key in {"i", "me", "we", "us", "you", "he", "him", "she", "her", "they", "them"} else "unknown"
            sem.animacy = "animate" if sem.semantic_type == "human" else "unknown"
            sem.can_be_agent = True
            sem.can_be_patient = True
            sem.can_be_theme = True
            sem.can_be_recipient = True
        elif key in _PREPOSITIONS:
            morph.pos = PartOfSpeech.PREPOSITION.value
            morph.lemma = key
        elif key in _CONJUNCTIONS:
            morph.pos = PartOfSpeech.CONJUNCTION.value
            morph.lemma = key
        elif key in _AUXILIARIES:
            morph.pos = PartOfSpeech.AUXILIARY.value
            morph.lemma = infer_verb_lemma(key)
            morph.tense, morph.person, morph.number = infer_verb_tense_number_person(key)
        elif key in _VERB_FRAMES or key in _IRREGULAR_VERBS or any(key in set(forms.values()) for forms in _IRREGULAR_VERBS.values()) or key.endswith(("ed", "ing")):
            lemma = infer_verb_lemma(key)
            morph.pos = PartOfSpeech.VERB.value
            morph.lemma = lemma
            morph.tense, morph.person, morph.number = infer_verb_tense_number_person(key)
            frame = _VERB_FRAMES.get(lemma, {})
            sem.roles_required = list(frame.get("roles_required", []))
            sem.role_constraints = copy.deepcopy(frame.get("role_constraints", {}))
            sem.alternations = list(frame.get("alternations", []))
            sem.attributes["valency"] = frame.get("valency", "unknown")
        elif re.fullmatch(r"\d+(?:\.\d+)?", key):
            morph.pos = PartOfSpeech.NUMERAL.value
            morph.lemma = key
            sem.semantic_type = "number"
        elif key.endswith("ly"):
            morph.pos = PartOfSpeech.ADVERB.value
            morph.lemma = key
        elif key.endswith(("ous", "ful", "ive", "al", "ic", "able", "ible", "less")):
            morph.pos = PartOfSpeech.ADJECTIVE.value
            morph.lemma = key
        else:
            lemma = singularize_noun(key)
            morph.pos = PartOfSpeech.NOUN.value
            morph.lemma = lemma
            morph.number = infer_noun_number(key)
            morph.person = Person.THIRD.value
            morph.countability = "count"
            ent = _ENTITY_SEMANTICS.get(lemma, {})
            if ent:
                for k, v in ent.items():
                    if hasattr(sem, k):
                        setattr(sem, k, copy.deepcopy(v))
                    else:
                        sem.attributes[k] = copy.deepcopy(v)
            else:
                sem.semantic_type = "unknown"
                sem.animacy = "unknown"
                sem.concreteness = "unknown"
                sem.can_be_patient = True
                sem.can_be_theme = True

        return LanguageTwin(
            kind=TwinKind.LEXICAL.value,
            text=text,
            language=self.config.language,
            canonical=key,
            morph=morph,
            semantics=sem,
            metadata=md,
        )

    def phrase_twin(self, text_or_tokens: Union[str, Sequence[str]], *, metadata: Optional[Mapping[str, Any]] = None) -> LanguageTwin:
        tokens = tokenize(text_or_tokens) if isinstance(text_or_tokens, str) else [str(t) for t in text_or_tokens]
        tokens = tokens[: self.config.max_tokens]
        children = [self.lexical_twin(tok) for tok in tokens]
        text = detokenize(tokens)
        morph = self._infer_phrase_morph(children)
        sem = self._infer_phrase_semantics(children)
        twin = LanguageTwin(
            kind=TwinKind.PHRASE.value,
            text=text,
            language=self.config.language,
            canonical=_canonical_key(text),
            morph=morph,
            semantics=sem,
            tokens=tokens,
            children=children,
            metadata=dict(metadata or {}),
        )
        twin.latent = self.project_payload(twin.payload(include_latent=False), key=f"phrase::{twin.canonical}")
        twin.refresh_id()
        return twin

    def _infer_phrase_morph(self, children: Sequence[LanguageTwin]) -> MorphFeatures:
        if not children:
            return MorphFeatures()
        head = self._find_phrase_head(children)
        return copy.deepcopy(head.morph)

    def _infer_phrase_semantics(self, children: Sequence[LanguageTwin]) -> SemanticFeatures:
        if not children:
            return SemanticFeatures()
        head = self._find_phrase_head(children)
        sem = copy.deepcopy(head.semantics)
        sem.attributes.setdefault("phrase_length", len(children))
        return sem

    def _find_phrase_head(self, children: Sequence[LanguageTwin]) -> LanguageTwin:
        # English fallback: prefer rightmost noun/pronoun, then verb, then last lexical token.
        for pos in (PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value, PartOfSpeech.VERB.value):
            for child in reversed(children):
                if child.morph.pos == pos:
                    return child
        return children[-1]

    def sentence_twin(self, text: str, *, metadata: Optional[Mapping[str, Any]] = None, validate: bool = True) -> LanguageTwin:
        tokens = tokenize(text)[: self.config.max_tokens]
        children = [self.lexical_twin(tok) for tok in tokens]
        sent = detokenize(tokens)
        state = SentenceTwinState(sentence=sent, tokens=tokens, lexical_twins=children)
        state.semantic_frame = infer_semantic_frame(children)
        state.grammar_plan = infer_basic_grammar_plan(children)
        if validate:
            state.grammar_report = GrammarValidator(self.config).validate_state(state)
        twin = LanguageTwin(
            kind=TwinKind.SENTENCE.value,
            text=sent,
            language=self.config.language,
            canonical=_canonical_key(sent),
            morph=MorphFeatures(pos="sentence"),
            semantics=SemanticFeatures(attributes={"semantic_frame": state.semantic_frame}),
            tokens=tokens,
            children=children,
            roles={str(k): str(v) for k, v in state.semantic_frame.get("roles", {}).items()},
            dependencies=infer_basic_dependencies(children),
            metadata={**dict(metadata or {}), "state": state.to_dict(include_latent=False)},
        )
        twin.latent = self.project_payload(twin.payload(include_latent=False), key=f"sentence::{twin.canonical}")
        twin.refresh_id()
        return twin

    def semantic_frame_twin(self, frame: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None) -> LanguageTwin:
        frame = copy.deepcopy(dict(frame))
        text = frame_to_text_label(frame)
        children: List[LanguageTwin] = []
        for key in ("event", "action", "verb", "agent", "patient", "theme", "recipient", "experiencer", "location", "time"):
            if key in frame and isinstance(frame[key], str):
                children.append(self.lexical_twin(frame[key], metadata={"semantic_role": key}))
        twin = LanguageTwin(
            kind=TwinKind.SEMANTIC_FRAME.value,
            text=text,
            language=self.config.language,
            canonical=_canonical_key(text),
            morph=MorphFeatures(pos="semantic_frame"),
            semantics=SemanticFeatures(attributes={"frame": frame}),
            tokens=tokenize(text),
            children=children,
            roles={k: str(v) for k, v in frame.items() if isinstance(v, str)},
            metadata=dict(metadata or {}),
        )
        twin.latent = self.project_payload(twin.payload(include_latent=False), key=f"semantic_frame::{twin.canonical}")
        twin.refresh_id()
        return twin

    def grammar_plan_twin(self, plan: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None) -> LanguageTwin:
        plan = copy.deepcopy(dict(plan))
        text = "grammar_plan:" + stable_payload_hash(plan)
        twin = LanguageTwin(
            kind=TwinKind.GRAMMAR_PLAN.value,
            text=text,
            language=self.config.language,
            canonical=_canonical_key(text),
            morph=MorphFeatures(pos="grammar_plan"),
            semantics=SemanticFeatures(attributes={"grammar_plan": plan}),
            metadata=dict(metadata or {}),
        )
        twin.latent = self.project_payload(twin.payload(include_latent=False), key=f"grammar_plan::{twin.canonical}")
        twin.refresh_id()
        return twin

    def intent_twin(self, intent: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None) -> LanguageTwin:
        intent = copy.deepcopy(dict(intent))
        text = frame_to_text_label(intent)
        twin = LanguageTwin(
            kind=TwinKind.INTENT.value,
            text=text,
            language=self.config.language,
            canonical=_canonical_key(text),
            morph=MorphFeatures(pos="intent"),
            semantics=SemanticFeatures(attributes={"intent": intent}),
            children=[self.lexical_twin(str(v), metadata={"intent_key": k}) for k, v in intent.items() if isinstance(v, str)],
            metadata=dict(metadata or {}),
        )
        twin.latent = self.project_payload(twin.payload(include_latent=False), key=f"intent::{twin.canonical}")
        twin.refresh_id()
        return twin

    def any(self, obj: Any, *, metadata: Optional[Mapping[str, Any]] = None) -> LanguageTwin:
        """Universal language-twin constructor."""
        if isinstance(obj, LanguageTwin):
            return obj
        if isinstance(obj, str):
            toks = tokenize(obj)
            if len(toks) <= 1:
                return self.lexical_twin(obj, metadata=metadata)
            return self.sentence_twin(obj, metadata=metadata) if _looks_like_sentence(obj, toks) else self.phrase_twin(toks, metadata=metadata)
        if isinstance(obj, Mapping):
            keys = set(str(k) for k in obj.keys())
            if keys & {"event", "action", "verb", "agent", "patient", "theme", "recipient"}:
                return self.semantic_frame_twin(obj, metadata=metadata)
            if keys & {"clause_type", "voice", "subject", "object", "verb_form", "order"}:
                return self.grammar_plan_twin(obj, metadata=metadata)
            return self.intent_twin(obj, metadata=metadata)
        if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray)):
            return self.phrase_twin([str(x) for x in obj], metadata=metadata)
        return self.lexical_twin(str(obj), metadata=metadata)


# =============================================================================
# Basic inference helpers
# =============================================================================


def _looks_like_sentence(text: str, tokens: Sequence[str]) -> bool:
    if _SENTENCE_END_RE.search(_canonical_text(text)):
        return True
    posish = [_canonical_key(t) for t in tokens]
    return any(t in _AUXILIARIES or t in _VERB_FRAMES or t.endswith(("ed", "ing")) for t in posish) and len(tokens) >= 3


def frame_to_text_label(frame: Mapping[str, Any]) -> str:
    parts = []
    for key in ("event", "action", "verb", "agent", "experiencer", "patient", "theme", "recipient", "tense", "style"):
        if key in frame:
            parts.append(f"{key}={frame[key]}")
    if not parts:
        parts = [f"{k}={v}" for k, v in sorted(frame.items(), key=lambda kv: str(kv[0]))[:8]]
    return "; ".join(parts)


def infer_basic_dependencies(twins: Sequence[LanguageTwin]) -> List[Dict[str, Any]]:
    deps: List[Dict[str, Any]] = []
    verb_idx = next((i for i, t in enumerate(twins) if t.morph.pos in {PartOfSpeech.VERB.value, PartOfSpeech.AUXILIARY.value}), -1)
    if verb_idx >= 0:
        subj_idx = next((i for i in range(verb_idx - 1, -1, -1) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), -1)
        obj_idx = next((i for i in range(verb_idx + 1, len(twins)) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), -1)
        if subj_idx >= 0:
            deps.append({"head": verb_idx, "dependent": subj_idx, "label": "nsubj"})
        if obj_idx >= 0:
            deps.append({"head": verb_idx, "dependent": obj_idx, "label": "obj"})
    for i, t in enumerate(twins[:-1]):
        if t.morph.pos == PartOfSpeech.DETERMINER.value and twins[i + 1].morph.pos == PartOfSpeech.NOUN.value:
            deps.append({"head": i + 1, "dependent": i, "label": "det"})
    return deps


def infer_semantic_frame(twins: Sequence[LanguageTwin]) -> Dict[str, Any]:
    frame: Dict[str, Any] = {"roles": {}}
    verb_idx = next((i for i, t in enumerate(twins) if t.morph.pos == PartOfSpeech.VERB.value), -1)
    if verb_idx < 0:
        return frame
    verb = twins[verb_idx]
    frame["event"] = verb.morph.lemma
    frame["tense"] = verb.morph.tense
    frame["valency"] = verb.semantics.attributes.get("valency", "unknown")
    subject = next((twins[i] for i in range(verb_idx - 1, -1, -1) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), None)
    obj = next((twins[i] for i in range(verb_idx + 1, len(twins)) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), None)
    roles = frame["roles"]
    req = set(verb.semantics.roles_required)
    if subject is not None:
        roles["agent" if "agent" in req or not req else "experiencer"] = subject.morph.lemma
    if obj is not None:
        if "patient" in req:
            roles["patient"] = obj.morph.lemma
        elif "theme" in req:
            roles["theme"] = obj.morph.lemma
        else:
            roles["object"] = obj.morph.lemma
    return frame


def infer_basic_grammar_plan(twins: Sequence[LanguageTwin]) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "clause_type": ClauseType.DECLARATIVE.value,
        "voice": Voice.ACTIVE.value,
        "order": [],
    }
    verb_idx = next((i for i, t in enumerate(twins) if t.morph.pos == PartOfSpeech.VERB.value), -1)
    if verb_idx >= 0:
        plan["verb"] = twins[verb_idx].morph.lemma
        plan["verb_form"] = twins[verb_idx].text
        plan["tense"] = twins[verb_idx].morph.tense
        subj = next((twins[i] for i in range(verb_idx - 1, -1, -1) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), None)
        obj = next((twins[i] for i in range(verb_idx + 1, len(twins)) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), None)
        if subj:
            plan["subject"] = subj.morph.lemma
            plan["subject_number"] = subj.morph.number
        if obj:
            plan["object"] = obj.morph.lemma
        if subj and obj:
            plan["order"] = ["subject", "verb", "object"]
        elif subj:
            plan["order"] = ["subject", "verb"]
        else:
            plan["order"] = ["verb"]
    return plan


# =============================================================================
# Grammar validator
# =============================================================================


class GrammarValidator:
    def __init__(self, config: Optional[LanguageTwinConfig] = None):
        self.config = (config or LanguageTwinConfig()).normalized()

    def validate_sentence(self, twin: LanguageTwin) -> GrammarReport:
        state_payload = twin.metadata.get("state") if isinstance(twin.metadata, Mapping) else None
        if isinstance(state_payload, Mapping):
            state = SentenceTwinState(
                sentence=str(state_payload.get("sentence", twin.text)),
                tokens=list(state_payload.get("tokens", twin.tokens)),
                lexical_twins=twin.children,
                semantic_frame=dict(state_payload.get("semantic_frame", {})),
                grammar_plan=dict(state_payload.get("grammar_plan", {})),
            )
        else:
            state = SentenceTwinState(
                sentence=twin.text,
                tokens=twin.tokens,
                lexical_twins=twin.children,
                semantic_frame=infer_semantic_frame(twin.children),
                grammar_plan=infer_basic_grammar_plan(twin.children),
            )
        return self.validate_state(state)

    def validate_state(self, state: SentenceTwinState) -> GrammarReport:
        issues: List[GrammarIssue] = []
        laws: List[str] = []
        twins = state.lexical_twins

        self._law_clause_completeness(twins, issues, laws)
        self._law_subject_verb_agreement(twins, issues, laws)
        self._law_determiner_noun_agreement(twins, issues, laws)
        self._law_verb_valency(twins, state.semantic_frame, issues, laws)
        self._law_basic_word_order(twins, issues, laws)
        self._law_semantic_role_compatibility(twins, state.semantic_frame, issues, laws)

        error_count = sum(1 for i in issues if i.severity == GrammarSeverity.ERROR.value)
        warning_count = sum(1 for i in issues if i.severity == GrammarSeverity.WARNING.value)
        penalty = error_count * 0.22 + warning_count * 0.08
        score = _clamp01(1.0 - penalty)
        ok = error_count == 0 and (self.config.strict is False or warning_count == 0)
        return GrammarReport(ok=ok, score=score, issues=issues, laws_satisfied=laws, metadata={"language": self.config.language})

    def _law_clause_completeness(self, twins: Sequence[LanguageTwin], issues: List[GrammarIssue], laws: List[str]) -> None:
        has_verb = any(t.morph.pos == PartOfSpeech.VERB.value for t in twins)
        has_subject = any(t.morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value} for t in twins)
        if self.config.allow_fragments:
            laws.append("fragment_allowed")
            return
        if not twins:
            issues.append(GrammarIssue("empty_sentence", "The sentence has no tokens.", GrammarSeverity.ERROR.value))
            return
        if not has_verb:
            issues.append(GrammarIssue("missing_finite_verb", "A complete declarative clause normally requires a finite verb.", GrammarSeverity.ERROR.value, repair="Add a finite verb."))
        if not has_subject:
            issues.append(GrammarIssue("missing_subject", "A complete declarative clause normally requires a subject.", GrammarSeverity.ERROR.value, repair="Add a subject noun phrase."))
        if has_verb and has_subject:
            laws.append("clause_completeness")

    def _law_subject_verb_agreement(self, twins: Sequence[LanguageTwin], issues: List[GrammarIssue], laws: List[str]) -> None:
        verb_idx = next((i for i, t in enumerate(twins) if t.morph.pos == PartOfSpeech.VERB.value), -1)
        if verb_idx < 0:
            return
        subj_idx = next((i for i in range(verb_idx - 1, -1, -1) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), -1)
        if subj_idx < 0:
            return
        subj = twins[subj_idx]
        verb = twins[verb_idx]
        if verb.morph.tense != Tense.PRESENT.value:
            laws.append("subject_verb_agreement")
            return
        subj_num = subj.morph.number
        subj_person = subj.morph.person if subj.morph.person != Person.UNKNOWN.value else Person.THIRD.value
        expected = conjugate_verb(verb.morph.lemma, tense=Tense.PRESENT.value, person=subj_person, number=subj_num)
        if verb.text.casefold() != expected.casefold() and verb.morph.lemma not in {"be"}:
            issues.append(GrammarIssue(
                "subject_verb_disagreement",
                f"Subject '{subj.text}' expects verb form '{expected}', not '{verb.text}'.",
                GrammarSeverity.ERROR.value,
                span=(verb_idx, verb_idx + 1),
                repair=expected,
                metadata={"subject": subj.text, "verb": verb.text, "expected": expected},
            ))
        else:
            laws.append("subject_verb_agreement")

    def _law_determiner_noun_agreement(self, twins: Sequence[LanguageTwin], issues: List[GrammarIssue], laws: List[str]) -> None:
        checked = False
        for i in range(len(twins) - 1):
            det = twins[i]
            noun = twins[i + 1]
            if det.morph.pos != PartOfSpeech.DETERMINER.value or noun.morph.pos != PartOfSpeech.NOUN.value:
                continue
            checked = True
            d = det.text.casefold()
            if d in {"a", "an"} and noun.morph.number != Number.SINGULAR.value:
                issues.append(GrammarIssue(
                    "determiner_number_mismatch",
                    f"Determiner '{det.text}' requires a singular count noun, but '{noun.text}' is {noun.morph.number}.",
                    GrammarSeverity.ERROR.value,
                    span=(i, i + 2),
                    repair=f"the {noun.text}",
                ))
            elif d == "a" and choose_indefinite_article(noun.text) == "an":
                issues.append(GrammarIssue(
                    "indefinite_article_sound_mismatch",
                    f"Use 'an' before '{noun.text}' under the fallback vowel-sound heuristic.",
                    GrammarSeverity.WARNING.value,
                    span=(i, i + 1),
                    repair="an",
                ))
            elif d == "an" and choose_indefinite_article(noun.text) == "a":
                issues.append(GrammarIssue(
                    "indefinite_article_sound_mismatch",
                    f"Use 'a' before '{noun.text}' under the fallback vowel-sound heuristic.",
                    GrammarSeverity.WARNING.value,
                    span=(i, i + 1),
                    repair="a",
                ))
        if checked and not any(i.code.startswith("determiner") for i in issues):
            laws.append("determiner_noun_agreement")

    def _law_verb_valency(self, twins: Sequence[LanguageTwin], frame: Mapping[str, Any], issues: List[GrammarIssue], laws: List[str]) -> None:
        verb = next((t for t in twins if t.morph.pos == PartOfSpeech.VERB.value), None)
        if verb is None:
            return
        required = list(verb.semantics.roles_required)
        if not required:
            laws.append("verb_valency")
            return
        roles = frame.get("roles", {}) if isinstance(frame, Mapping) else {}
        role_keys = set(roles.keys()) if isinstance(roles, Mapping) else set()
        # Surface fallback: subject/object existence can satisfy common roles.
        has_subject = any(t.morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value} for t in twins[: max(0, twins.index(verb))]) if verb in twins else False
        has_object = any(t.morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value} for t in twins[(twins.index(verb) + 1):]) if verb in twins else False
        satisfied = set(role_keys)
        if has_subject:
            satisfied.add("agent")
            satisfied.add("experiencer")
        if has_object:
            satisfied.add("patient")
            satisfied.add("theme")
        missing = [r for r in required if r not in satisfied]
        if missing:
            issues.append(GrammarIssue(
                "verb_valency_unsatisfied",
                f"Verb '{verb.text}' requires role(s): {', '.join(missing)}.",
                GrammarSeverity.ERROR.value,
                repair=f"Add {', '.join(missing)} argument(s).",
                metadata={"verb": verb.text, "required": required, "missing": missing},
            ))
        else:
            laws.append("verb_valency")

    def _law_basic_word_order(self, twins: Sequence[LanguageTwin], issues: List[GrammarIssue], laws: List[str]) -> None:
        verb_idx = next((i for i, t in enumerate(twins) if t.morph.pos == PartOfSpeech.VERB.value), -1)
        if verb_idx < 0:
            return
        subj_idx = next((i for i, t in enumerate(twins) if i < verb_idx and t.morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), -1)
        # Imperatives may start with a verb; this fallback only warns.
        if subj_idx < 0 and not self.config.allow_fragments:
            issues.append(GrammarIssue(
                "marked_or_missing_subject_order",
                "The fallback English declarative order expects a subject before the main verb.",
                GrammarSeverity.WARNING.value,
            ))
            return
        if subj_idx >= 0:
            laws.append("basic_english_word_order")

    def _law_semantic_role_compatibility(self, twins: Sequence[LanguageTwin], frame: Mapping[str, Any], issues: List[GrammarIssue], laws: List[str]) -> None:
        if self.config.allow_poetic_semantics:
            laws.append("semantic_role_compatibility_poetic_mode")
            return
        verb_idx = next((i for i, t in enumerate(twins) if t.morph.pos == PartOfSpeech.VERB.value), -1)
        if verb_idx < 0:
            return
        verb = twins[verb_idx]
        constraints = verb.semantics.role_constraints or {}
        if not constraints:
            laws.append("semantic_role_compatibility")
            return
        candidates: Dict[str, Optional[LanguageTwin]] = {
            "agent": next((twins[i] for i in range(verb_idx - 1, -1, -1) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), None),
            "patient": next((twins[i] for i in range(verb_idx + 1, len(twins)) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), None),
            "theme": next((twins[i] for i in range(verb_idx + 1, len(twins)) if twins[i].morph.pos in {PartOfSpeech.NOUN.value, PartOfSpeech.PRONOUN.value}), None),
            "recipient": None,
        }
        incompatible: List[Tuple[str, str, str, List[str]]] = []
        for role, allowed in constraints.items():
            ent = candidates.get(role)
            if ent is None:
                continue
            sem_type = ent.semantics.semantic_type
            if sem_type != "unknown" and allowed and sem_type not in allowed:
                incompatible.append((role, ent.text, sem_type, list(allowed)))
        for role, ent_text, sem_type, allowed in incompatible:
            issues.append(GrammarIssue(
                "semantic_role_mismatch",
                f"Entity '{ent_text}' has semantic type '{sem_type}', which is unusual for role '{role}' of verb '{verb.text}'.",
                GrammarSeverity.WARNING.value,
                metadata={"role": role, "entity": ent_text, "semantic_type": sem_type, "allowed": allowed},
            ))
        if not incompatible:
            laws.append("semantic_role_compatibility")


# =============================================================================
# Grammar planner and surface realizer
# =============================================================================


class SemanticGrammarPlanner:
    """
    Deterministic semantic-frame to grammar-plan compiler.

    Optional tn.TensorTrain planners can be injected later. The fallback planner
    is symbolic and intentionally transparent.
    """

    def __init__(self, factory: Optional[LanguageTwinFactory] = None, *, planner_tt: Optional[Any] = None):
        self.factory = factory or LanguageTwinFactory()
        self.planner_tt = planner_tt

    def plan(self, frame: Mapping[str, Any], *, style: Optional[str] = None, focus: Optional[str] = None) -> Dict[str, Any]:
        frame = copy.deepcopy(dict(frame))
        event = str(frame.get("event") or frame.get("action") or frame.get("verb") or "be")
        lemma = infer_verb_lemma(event)
        tense = str(frame.get("tense", Tense.PRESENT.value))
        voice = str(frame.get("voice", Voice.ACTIVE.value))
        focus = str(focus or frame.get("focus", "agent"))
        style = str(style or frame.get("style", self.factory.config.default_style))

        if self.planner_tt is not None:
            # Hook point for future tensorized semantic-to-grammar planning.
            # The fallback remains authoritative if planner output is invalid.
            try:
                vec = self.factory.project_payload(frame, key="semantic_planner_input")
                _ = self.planner_tt.apply(vec)  # type: ignore[attr-defined]
            except Exception:
                pass

        plan: Dict[str, Any] = {
            "clause_type": ClauseType.DECLARATIVE.value,
            "voice": voice,
            "style": style,
            "focus": focus,
            "tense": tense,
            "verb_lemma": lemma,
        }

        agent = frame.get("agent") or frame.get("experiencer")
        patient = frame.get("patient")
        theme = frame.get("theme") or patient
        recipient = frame.get("recipient")

        frame_info = _VERB_FRAMES.get(lemma, {})
        valency = frame_info.get("valency", "unknown")
        plan["valency"] = valency

        if voice == Voice.PASSIVE.value and (patient or theme):
            plan["subject_role"] = "patient" if patient else "theme"
            plan["subject"] = patient or theme
            plan["agent_by_phrase"] = bool(agent)
            if agent:
                plan["agent"] = agent
            plan["verb_form"] = conjugate_verb("be", tense=tense, person=Person.THIRD.value, number=Number.SINGULAR.value) + " " + conjugate_verb(lemma, tense=Tense.PARTICIPLE.value)
            plan["order"] = ["subject", "auxiliary", "participle", "by_agent"] if agent else ["subject", "auxiliary", "participle"]
            return plan

        plan["subject_role"] = "agent" if agent is not None else "theme"
        plan["subject"] = agent if agent is not None else (theme or patient)
        plan["object"] = patient if patient is not None else None
        if theme is not None and patient is None:
            plan["object"] = theme
        if recipient is not None:
            if lemma == "give" and style in {"compact", "informal"}:
                plan["indirect_object"] = recipient
                plan["direct_object"] = theme
                plan["order"] = ["subject", "verb", "indirect_object", "direct_object"]
            else:
                plan["direct_object"] = theme
                plan["prepositional_object"] = {"preposition": "to", "object": recipient, "role": "recipient"}
                plan["order"] = ["subject", "verb", "direct_object", "prepositional_object"]
        elif plan.get("object"):
            plan["order"] = ["subject", "verb", "object"]
        else:
            plan["order"] = ["subject", "verb"]

        subject_twin = self.factory.lexical_twin(str(plan.get("subject", "it")))
        plan["subject_number"] = subject_twin.morph.number if subject_twin.morph.number != Number.UNKNOWN.value else Number.SINGULAR.value
        plan["subject_person"] = subject_twin.morph.person if subject_twin.morph.person != Person.UNKNOWN.value else Person.THIRD.value
        plan["verb_form"] = conjugate_verb(lemma, tense=tense, person=plan["subject_person"], number=plan["subject_number"])
        return plan


class SurfaceRealizer:
    def __init__(self, factory: Optional[LanguageTwinFactory] = None):
        self.factory = factory or LanguageTwinFactory()

    def realize(self, plan: Mapping[str, Any], *, terminal: str = ".") -> str:
        plan = dict(plan)
        if plan.get("voice") == Voice.PASSIVE.value:
            return ensure_sentence_terminal(self._realize_passive(plan), terminal)
        return ensure_sentence_terminal(self._realize_active(plan), terminal)

    def _realize_active(self, plan: Mapping[str, Any]) -> str:
        subject = self._np(plan.get("subject", "it"), role="subject")
        verb = str(plan.get("verb_form") or conjugate_verb(str(plan.get("verb_lemma", "be"))))
        parts = [subject, verb]
        if plan.get("indirect_object"):
            parts.append(self._np(plan["indirect_object"], role="object"))
        if plan.get("direct_object"):
            parts.append(self._np(plan["direct_object"], role="object"))
        elif plan.get("object"):
            parts.append(self._np(plan["object"], role="object"))
        if isinstance(plan.get("prepositional_object"), Mapping):
            pp = plan["prepositional_object"]
            parts.append(str(pp.get("preposition", "to")))
            parts.append(self._np(pp.get("object", "it"), role="object"))
        return capitalize_sentence(" ".join(p for p in parts if p))

    def _realize_passive(self, plan: Mapping[str, Any]) -> str:
        subject = self._np(plan.get("subject", "it"), role="subject")
        verb_form = str(plan.get("verb_form") or "was done")
        parts = [subject, verb_form]
        if plan.get("agent_by_phrase") and plan.get("agent"):
            parts.extend(["by", self._np(plan["agent"], role="object")])
        return capitalize_sentence(" ".join(p for p in parts if p))

    def _np(self, value: Any, *, role: str) -> str:
        text = _canonical_text(value)
        if not text:
            return "it"
        toks = tokenize(text)
        if len(toks) > 1 or _canonical_key(text) in _PRONOUNS:
            return text
        twin = self.factory.lexical_twin(text)
        if twin.morph.pos == PartOfSpeech.NOUN.value:
            if twin.morph.number == Number.PLURAL.value:
                return f"the {text}"
            # Deterministic default: definite article for known concrete entities.
            return f"the {text}"
        return text


def capitalize_sentence(text: str) -> str:
    text = _canonical_text(text)
    if not text:
        return text
    return text[0].upper() + text[1:]


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


# =============================================================================
# Tensor-network grammar helper builders
# =============================================================================


class TensorGrammarOps:
    """Optional TensorTrain grammar operators with safe symbolic fallback."""

    def __init__(self, *, dtype: np.dtype = np.float32):
        self.dtype = np.dtype(dtype)
        self.agreement_tt: Optional[Any] = None
        self.metadata: Dict[str, Any] = {"tn_available": bool(_TN_OK), "tn_error": _TN_ERR}

    def build_subject_verb_agreement_tt(self, *, max_bond_dim: int = 8, energy_tol: float = 0.999) -> Optional[Any]:
        if not (_TN_OK and tn is not None and hasattr(tn, "TensorTrain")):
            return None
        try:
            table = np.zeros((len(PERSON_ENCODER), len(NUMBER_ENCODER), len(PERSON_ENCODER), len(NUMBER_ENCODER), len(TENSE_ENCODER)), dtype=np.float32)
            for sp in PERSON_ENCODER.values:
                for sn in NUMBER_ENCODER.values:
                    for vp in PERSON_ENCODER.values:
                        for vn in NUMBER_ENCODER.values:
                            for te in TENSE_ENCODER.values:
                                valid = 1.0
                                if te == Tense.PRESENT.value:
                                    if sn in {Number.SINGULAR.value, Number.PLURAL.value} and vn in {Number.SINGULAR.value, Number.PLURAL.value}:
                                        valid = 1.0 if sn == vn else 0.0
                                    if sp != Person.UNKNOWN.value and vp != Person.UNKNOWN.value:
                                        valid *= 1.0 if sp == vp else 0.0
                                table[PERSON_ENCODER.encode(sp), NUMBER_ENCODER.encode(sn), PERSON_ENCODER.encode(vp), NUMBER_ENCODER.encode(vn), TENSE_ENCODER.encode(te)] = valid
            matrix = table.reshape(1, -1)
            self.agreement_tt = tn.TensorTrain.from_dense(  # type: ignore[attr-defined]
                matrix,
                output_dims=[1],
                input_dims=[len(PERSON_ENCODER), len(NUMBER_ENCODER), len(PERSON_ENCODER), len(NUMBER_ENCODER), len(TENSE_ENCODER)],
                max_bond_dim=int(max_bond_dim),
                energy_tol=float(energy_tol),
                dtype=self.dtype,
                device="cpu",
            )
            return self.agreement_tt
        except Exception as e:
            self.metadata["agreement_tt_error"] = repr(e)
            self.agreement_tt = None
            return None

    def score_agreement(self, subject: LanguageTwin, verb: LanguageTwin) -> float:
        if self.agreement_tt is None:
            return symbolic_agreement_score(subject, verb)
        try:
            idx = np.zeros((len(PERSON_ENCODER), len(NUMBER_ENCODER), len(PERSON_ENCODER), len(NUMBER_ENCODER), len(TENSE_ENCODER)), dtype=np.float32)
            idx[
                PERSON_ENCODER.encode(subject.morph.person if subject.morph.person != Person.UNKNOWN.value else Person.THIRD.value),
                NUMBER_ENCODER.encode(subject.morph.number),
                PERSON_ENCODER.encode(verb.morph.person if verb.morph.person != Person.UNKNOWN.value else subject.morph.person),
                NUMBER_ENCODER.encode(verb.morph.number if verb.morph.number != Number.UNKNOWN.value else subject.morph.number),
                TENSE_ENCODER.encode(verb.morph.tense),
            ] = 1.0
            y = self.agreement_tt.apply(idx.reshape(-1))  # type: ignore[attr-defined]
            arr = _sanitize_array(y, np.float32)
            return _clamp01(float(arr[0]) if arr.size else 0.0)
        except Exception:
            return symbolic_agreement_score(subject, verb)


def symbolic_agreement_score(subject: LanguageTwin, verb: LanguageTwin) -> float:
    if verb.morph.tense != Tense.PRESENT.value:
        return 1.0
    subj_person = subject.morph.person if subject.morph.person != Person.UNKNOWN.value else Person.THIRD.value
    subj_num = subject.morph.number if subject.morph.number != Number.UNKNOWN.value else Number.SINGULAR.value
    expected = conjugate_verb(verb.morph.lemma, tense=Tense.PRESENT.value, person=subj_person, number=subj_num)
    return 1.0 if verb.text.casefold() == expected.casefold() else 0.0


# =============================================================================
# High-level sentence builder
# =============================================================================


class GovernedLanguageBuilder:
    """
    High-level semantic-to-sentence builder using language twins.

    The class is intentionally runtime-agnostic. If a digital twin kernel object
    is supplied, it can be used by the caller to govern actions externally. This
    builder returns proposed traces and reports rather than mutating external
    kernel topology by default.
    """

    def __init__(
        self,
        config: Optional[LanguageTwinConfig] = None,
        *,
        factory: Optional[LanguageTwinFactory] = None,
        planner: Optional[SemanticGrammarPlanner] = None,
        realizer: Optional[SurfaceRealizer] = None,
        validator: Optional[GrammarValidator] = None,
        tensor_ops: Optional[TensorGrammarOps] = None,
        kernel: Optional[Any] = None,
    ):
        self.config = (config or LanguageTwinConfig()).normalized()
        self.factory = factory or LanguageTwinFactory(self.config)
        self.planner = planner or SemanticGrammarPlanner(self.factory)
        self.realizer = realizer or SurfaceRealizer(self.factory)
        self.validator = validator or GrammarValidator(self.config)
        self.tensor_ops = tensor_ops or TensorGrammarOps()
        self.kernel = kernel

    def build_from_intent(self, intent: Mapping[str, Any], *, terminal: str = ".") -> Dict[str, Any]:
        intent_twin = self.factory.intent_twin(intent)
        frame_twin = self.factory.semantic_frame_twin(intent)
        plan = self.planner.plan(intent, style=str(intent.get("style", self.config.default_style)))
        plan_twin = self.factory.grammar_plan_twin(plan)
        sentence = self.realizer.realize(plan, terminal=terminal)
        sentence_twin = self.factory.sentence_twin(sentence)
        report = self.validator.validate_sentence(sentence_twin)
        sentence_latent = self.factory.project_payload(sentence_twin.payload(include_latent=False), key="built_sentence")

        trace = {
            "intent_hash": intent_twin.twin_id,
            "semantic_frame_hash": frame_twin.twin_id,
            "grammar_plan_hash": plan_twin.twin_id,
            "sentence_hash": sentence_twin.twin_id,
            "tn_available": _TN_OK,
            "digital_twin_kernel_available": _DTK_OK,
        }

        return {
            "sentence": sentence,
            "grammar_score": report.score,
            "ok": report.ok,
            "issues": [i.to_dict() for i in report.issues],
            "laws_satisfied": list(report.laws_satisfied),
            "semantic_frame": frame_twin.to_dict(include_latent=False),
            "grammar_plan": plan,
            "sentence_twin": sentence_twin.to_dict(include_latent=False),
            "sentence_latent": sentence_latent.astype(float).tolist(),
            "trace": trace,
        }

    def validate(self, text: str) -> Dict[str, Any]:
        twin = self.factory.sentence_twin(text)
        report = self.validator.validate_sentence(twin)
        return {
            "sentence": twin.text,
            "ok": report.ok,
            "score": report.score,
            "issues": [i.to_dict() for i in report.issues],
            "laws_satisfied": report.laws_satisfied,
            "sentence_twin": twin.to_dict(include_latent=False),
        }

    def propose_next_tokens(self, partial: str, candidates: Sequence[str]) -> List[Dict[str, Any]]:
        partial_tokens = tokenize(partial)
        proposals: List[Dict[str, Any]] = []
        for cand in candidates:
            candidate_text = detokenize(partial_tokens + [str(cand)])
            twin = self.factory.sentence_twin(candidate_text, validate=True)
            report = self.validator.validate_sentence(twin)
            proposals.append({
                "candidate": cand,
                "sentence": candidate_text,
                "score": report.score,
                "ok": report.ok,
                "issues": [i.to_dict() for i in report.issues],
                "latent": _json_safe(twin.latent),
            })
        proposals.sort(key=lambda x: (float(x["score"]), bool(x["ok"])), reverse=True)
        return proposals


# =============================================================================
# Public convenience API
# =============================================================================


_DEFAULT_FACTORY: Optional[LanguageTwinFactory] = None


def get_default_factory() -> LanguageTwinFactory:
    global _DEFAULT_FACTORY
    if _DEFAULT_FACTORY is None:
        _DEFAULT_FACTORY = LanguageTwinFactory()
    return _DEFAULT_FACTORY


def twin_any_language(obj: Any, *, config: Optional[LanguageTwinConfig] = None, metadata: Optional[Mapping[str, Any]] = None) -> LanguageTwin:
    factory = LanguageTwinFactory(config) if config is not None else get_default_factory()
    return factory.any(obj, metadata=metadata)


def build_sentence(intent: Mapping[str, Any], *, config: Optional[LanguageTwinConfig] = None) -> Dict[str, Any]:
    builder = GovernedLanguageBuilder(config)
    return builder.build_from_intent(intent)


def validate_sentence(text: str, *, config: Optional[LanguageTwinConfig] = None) -> Dict[str, Any]:
    builder = GovernedLanguageBuilder(config)
    return builder.validate(text)


def backend_status() -> Dict[str, Any]:
    return {
        "tn_available": bool(_TN_OK),
        "tn_error": _TN_ERR,
        "digital_twin_kernel_available": bool(_DTK_OK),
        "digital_twin_kernel_error": _DTK_ERR,
    }


# Backward-compatible alias for conceptual discussions that used twin_anything.
def twin_anything(obj: Any, *, config: Optional[LanguageTwinConfig] = None, metadata: Optional[Mapping[str, Any]] = None) -> LanguageTwin:
    return twin_any_language(obj, config=config, metadata=metadata)


# =============================================================================
# CLI smoke test
# =============================================================================


def _demo() -> None:
    print(json.dumps(backend_status(), indent=2))
    builder = GovernedLanguageBuilder()
    examples = [
        {"agent": "dog", "event": "chase", "patient": "cat", "tense": "past", "style": "simple"},
        {"agent": "boy", "event": "give", "theme": "book", "recipient": "girl", "tense": "past", "style": "simple"},
        {"agent": "scientist", "event": "discover", "patient": "particle", "tense": "past", "style": "formal"},
    ]
    for ex in examples:
        out = builder.build_from_intent(ex)
        print(json.dumps({"intent": ex, "sentence": out["sentence"], "score": out["grammar_score"], "issues": out["issues"]}, indent=2, ensure_ascii=False))
    print(json.dumps(builder.validate("The dog run."), indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    _demo()
