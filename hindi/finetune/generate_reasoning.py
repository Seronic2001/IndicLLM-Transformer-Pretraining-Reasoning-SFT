"""Hindi reasoning data generator — pure Python, no model dependency.

Produces ``{train,val,test}.jsonl`` of synthetic reasoning examples whose answers
are computed by a *symbolic solver* over a relation graph (>, <, =) — never by a
template author and never by an LLM.

Anti-leakage (mandatory, test-enforced):
  * Entity pools are split three ways — train vs val vs test — *before* any
    example is generated; the splits are explicit, checked-in lists below.
  * The multi_hop paradigm is reserved for val/test only; negation makes up 5%
    of train and an equal share of val/test.
  * Premises are stated in a random direction and shuffled, and the query pair
    and its order are random, so the answer cannot be read off any one premise.
  * The question asks about a random relation (more / less / equal), drawn
    independently of the answer, so echoing the question's relation word scores
    below the majority baseline.
  * Half of the training queries are at least two hops apart, and the train pool
    adds 3,000 syllable-built names (disjoint from val/test) so models learn to
    copy names they have never seen.

Reliability fallbacks:
  * Contradictory/cyclic premise sets (shouldn't happen by construction) make
    solve_relation raise; the generator catches and skips that example.
  * Per-template compatibility rules: every template pairs entities with a single
    attribute (उम्र/लंबाई/बचत) so agreement/gender/honorific issues can't arise.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- entity pools
# Three disjoint pools: a name seen in training never appears in val or test, and
# val and test share no names either, so picking a checkpoint on val cannot leak test.
TRAIN_ENTITY_POOL: list[str] = [
    "राम", "सीता", "गीता", "मोहन", "सुनीता", "राजेश", "कविता", "अमित", "प्रिया", "विकास",
    "नीता", "सुरेश", "मीना", "रवि", "अनिता", "दीपक", "रेखा", "संजय", "पूजा", "मनोज", "विजय",
    "रोहन", "सोहन", "अजय", "नेहा", "किरण", "सीमा", "राहुल", "पायल", "गौरव", "वरुण", "दिव्या",
    "श्वेता", "साक्षी", "आँचल", "अनुप", "मयंक", "आलोक", "विवेक", "मनीषा", "भारती", "वर्षा",
    "संगीता", "भावना", "संदीप", "पंकज", "हेमंत", "लोकेश", "तरुण", "कार्तिक", "तन्मय", "नीतेश",
    "मोनिका", "प्रीति", "पल्लवी", "स्वाती", "गरिमा", "चेतना", "नूतन", "वंदना", "कल्पना", "रीना",
    "शालिनी", "आरती", "कुणाल", "अतुल", "चेतन", "प्रणव", "भास्कर", "मानस", "दीपांकर", "हिरण्य",
    "प्रांजल", "ध्रुव", "अनुराग", "सौरभ", "नवजीत", "ऋतुपर्ण", "मृदुल", "विपुल", "उत्पल",
    "निरंजन", "जगदीश", "देवजीत", "रूपक", "अंकुर", "प्रशांत", "कौशिक", "रूबुल", "मुकुल", "अरूप",
    "भास्वती", "रूमी", "मृदुला", "निवेदिता", "मोनालिसा", "रीमा", "दीपाली", "मलया", "काकली",
    "जूमणी", "रुणझुम", "मुनमी", "पापड़ी", "सोनाली", "बर्णाली", "रश्मि", "स्मिता", "मधुस्मिता",
    "प्रणामी", "गीतांजलि", "पूरबी", "मीनाक्षी", "नवनीता", "डिंपल", "पल्लव", "पार्थ", "अभिजीत",
    "देवव्रत", "चिन्मय", "सिद्धार्थ", "सत्येन", "रमेन", "हरेन", "भूपेन", "खगेन", "प्रमथ",
    "रत्न", "लक्ष्मी", "ज्ञानेन", "धीरेन", "गुणेन", "बीरेन", "महेन", "गोविंद", "माधव", "केशव",
    "दामोदर", "हरिहर", "नारायण", "अनंत", "मोहिधर", "घनश्याम", "कमलेश", "जगन्नाथ", "गोपाल",
    "त्रैलोक्य", "भवेन", "लीलाधर", "हेमचंद्र", "शरद", "महिम", "पद्म", "आनंद", "हेमकांत",
    "धर्मेंद्र", "उपेन", "गिरीन", "कमलजीत", "सुरेन", "जोगेन", "भोगेन", "दिनेश", "महेश", "रमेश",
    "उमेश", "हिमांशु", "प्रभात", "भास्वर", "अचिंत्य", "प्रदीप", "संजीव", "राजीव", "कुलदीप",
    "हरिप्रसाद", "देवाशीष", "परिमल", "सुबोध", "अरविंद", "अपूर्व", "शांतनु", "प्रफुल्ल", "योगेश",
    "त्रिदीप", "रूपम", "प्रीतम",
]
VAL_ENTITY_POOL: list[str] = [
    "अर्जुन", "लता", "किशोर", "माला", "हरीश", "ज्योति", "नरेश", "अभिलाष", "निलय", "सुब्रत",
    "समीर", "सुशील", "हीरक", "द्विपेन", "खगेंद्र", "युगल", "उज्ज्वल", "ब्रजेन", "रत्नेश्वर",
    "गोलेश्वर", "भोगेश्वर", "योगेश्वर", "गुणेश्वर", "लीलेश्वर", "महेश्वर", "रमाकांत", "सूर्य",
]
TEST_ENTITY_POOL: list[str] = [
    "सरिता", "प्रमोद", "विनीता", "कमल", "सुधा", "यश", "इंदु", "तारा", "चंद्र", "इंद्र",
    "उपेंद्र", "जितेन", "नरेन", "हितेश", "पराग", "रूपाली", "प्रतिभा", "उर्मिला", "अंजना",
    "दीपांविता", "पूर्णिमा", "कल्पिता", "प्रियंका", "त्रिवेणी", "अनुराधा", "मिताली",
    "रूपज्योति", "रूपश्री", "सुचित्रा", "सुजाता", "मालविका", "अनसूया", "अपराजिता", "मैत्रेयी",
    "शर्मिष्ठा", "अरुंधती", "गार्गी", "देवयानी",
]
# The curated train pool has ~190 names, and the val/test names are never seen in
# training, so models misspelled unseen names when copying them into the answer.
# Training therefore also uses N_SYNTHETIC_TRAIN_NAMES names built from syllables;
# they are filtered against the val/test pools, so all three pools stay disjoint.
N_SYNTHETIC_TRAIN_NAMES = 3000
_SYN_ONSETS = ["अ", "आ", "इ", "उ", "क", "कि", "कु", "का", "ग", "गो", "ज", "जि", "त", "ता",
               "दे", "दी", "न", "नि", "प", "प्र", "बि", "भा", "म", "मी", "मो", "र", "रा", "रू",
               "ल", "ली", "व", "वि", "श", "शी", "स", "सु", "सो", "ह", "हे", "चा", "धी", "यो"]
_SYN_MIDDLES = ["", "न", "र", "म", "ल", "व", "श", "त", "ध", "ज", "क", "द", "प", "ब", "ग", "ह",
                "नी", "रा", "मि", "ले"]
_SYN_ENDINGS = ["ना", "ता", "ती", "नी", "का", "री", "या", "रा", "ल", "न", "श", "त", "र", "म", "व",
                "क", "द", "प", "ज", "ली", "मा", "शा", "वी", "धा", "नू"]
# Template words a syllable combination could spell; never used as names.
_TEMPLATE_WORDS = {"कम", "अधिक", "बराबर", "नहीं", "और", "कहा", "पूछा", "गाँव", "एक", "इसलिए", "कारण"}


def synthetic_names(n: int, exclude: set[str], seed: int = 0) -> list[str]:
    """n distinct onset + middle + ending syllable names, none of them in exclude."""
    rng = random.Random(seed)
    combos = sorted({o + m + e for o in _SYN_ONSETS for m in _SYN_MIDDLES for e in _SYN_ENDINGS}
                    - exclude - _TEMPLATE_WORDS)
    if len(combos) < n:
        raise ValueError(f"only {len(combos)} synthetic names for n={n}")
    return rng.sample(combos, n)


# Fresh names: built on demand from a syllable grammar, so a training example's names
# are (almost surely) never seen again and the only way to answer is to copy them from
# the prompt. Unlike the fixed synthetic pool above, the grammar covers conjuncts,
# anusvara and long compounds, which the held-out names need. It does not use any
# val/test name; those strings are rejected through ``exclude``.
_FRESH_INITIAL_VOWELS = ["अ", "आ", "इ", "ई", "उ", "ए", "ओ"]
_FRESH_CONSONANTS = list("कखगघचजटडतथदधनपफबभमयरलवशसह")
_FRESH_CLUSTERS = ["क्र", "ग्र", "त्र", "प्र", "श्र", "स्त", "न्द", "ज्य", "स्व", "क्ष", "द्व",
                   "र्म", "ल्प", "न्त", "म्ब", "ष्ण", "त्य", "ध्र", "स्न", "ज्ञ"]
_FRESH_MATRAS = ["", "", "ा", "ि", "ी", "ु", "ू", "े", "ो", "ां", "ं"]
_FRESH_ENDINGS = ["", "", "", "न", "र", "ल", "त", "श", "म", "व"]


def fresh_name(rng: random.Random, exclude: Optional[set] = None) -> str:
    """One name from the syllable grammar: 2-4 syllables, conjuncts and anusvara allowed."""
    while True:
        parts: list = []
        for i in range(rng.choice([2, 2, 3, 3, 4])):
            if i == 0 and rng.random() < 0.2:
                parts.append(rng.choice(_FRESH_INITIAL_VOWELS))
                continue
            onset = rng.choice(_FRESH_CLUSTERS) if rng.random() < 0.3 else rng.choice(_FRESH_CONSONANTS)
            parts.append(onset + rng.choice(_FRESH_MATRAS))
        name = "".join(parts) + rng.choice(_FRESH_ENDINGS)
        if 3 <= len(name) <= 14 and name not in _TEMPLATE_WORDS and not (exclude and name in exclude):
            return name


def fresh_names(k: int, rng: random.Random, exclude: Optional[set] = None) -> list:
    """k distinct fresh names."""
    out: list = []
    while len(out) < k:
        n = fresh_name(rng, exclude)
        if n not in out:
            out.append(n)
    return out


# The hand-picked train names alone: fresh-name training can mix them back in, so the
# model also sees real-looking names like the held-out val/test ones.
CURATED_TRAIN_POOL: list[str] = list(TRAIN_ENTITY_POOL)

TRAIN_ENTITY_POOL = TRAIN_ENTITY_POOL + synthetic_names(
    N_SYNTHETIC_TRAIN_NAMES, set(TRAIN_ENTITY_POOL) | set(VAL_ENTITY_POOL) | set(TEST_ENTITY_POOL))

ATTRIBUTES: list[str] = ["उम्र", "लंबाई", "बचत"]
RESERVED_PARADIGMS = {"multi_hop"}  # eval splits only
TRAIN_PARADIGMS = ("transitive", "word_problem", "conversational")
EVAL_PARADIGMS = TRAIN_PARADIGMS + ("multi_hop", "negation")
TRAIN_NEGATION_FRACTION = 0.05  # negation curriculum share of the train split
TRAIN_LONG_QUERY_FRACTION = 0.5  # train queries forced >= 2 hops apart (otherwise >= 1)
QUESTION_RELATIONS = (">", "<", "=")  # relation the question asks about...
QUESTION_WEIGHTS = (0.4, 0.4, 0.2)    # ...drawn independently of the true answer
# Harder eval splits (--hard-eval), never used in training: chains longer than any
# training or test_depth chain, and depth-3/4 chains with premises the answer does
# not need, so the model has to pick out the relevant path.
HARD_DEPTHS = (6, 8, 10)
HARD_N_EXTRA_PREMISES = 3


@dataclass
class ReasoningExample:
    id: str
    split: str
    paradigm: str
    attribute: str
    entities: list[str]
    premises: list[tuple[str, str, str]]  # (subj, rel, obj), rel in {">", "<", "="}
    query: tuple[str, str]
    answer_rel: str          # canonical solver output, e.g. "राम > सीता"
    answer_text: str         # rendered direct answer sentence
    cot_text: str            # step-by-step chain-of-thought explanation + answer
    text: str                # rendered question + premises
    distractors: list = field(default_factory=list)  # (subj, rel, obj, attr), another attribute

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "split": self.split,
            "paradigm": self.paradigm,
            "attribute": self.attribute,
            "entities": self.entities,
            "premises": [list(p) for p in self.premises],
            "query": list(self.query),
            "answer_rel": self.answer_rel,
            "answer_text": self.answer_text,
            "cot_text": self.cot_text,
            "text": self.text,
            **({"distractors": [list(d) for d in self.distractors]} if self.distractors else {}),
        }


# ---------------------------------------------------------------- solver

class ContradictionError(ValueError):
    """Raised when the premise graph is cyclic/contradictory or underdetermined."""


def solve_relation(premises: list[tuple[str, str, str]], query: tuple[str, str]) -> str:
    """Pure symbolic reasoning over a relation graph (>, <, =).

    ``=`` edges form equivalence classes (union-find); ``>`` edges form a directed
    graph between classes (``<`` is normalized to the reverse ``>``). The answer is
    the transitive closure: query (x, y) -> "x > y", "x < y", or "x = y".

    Raises ContradictionError on cycles/contradictions and on underdetermined
    queries — a generated example that hits either is skipped, never mislabeled.
    """
    x, y = query
    if x == y:
        return f"{x} = {y}"

    nodes: list[str] = []
    for s, _r, o in premises:
        for n in (s, o):
            if n not in nodes:
                nodes.append(n)

    parent = {n: n for n in nodes}
    if x not in parent or y not in parent:
        raise ContradictionError(f"query entity not in premise graph: {query}")

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for s, rel, o in premises:
        if rel == "=":
            union(s, o)

    gt: dict[str, set[str]] = {n: set() for n in nodes}
    for s, rel, o in premises:
        if rel == ">":
            gt[find(s)].add(find(o))
        elif rel == "<":
            gt[find(o)].add(find(s))
        elif rel != "=":
            raise ContradictionError(f"unknown relation {rel!r}")

    def reachable(src: str, dst: str) -> bool:
        seen: set[str] = set()
        stack = [src]
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(gt.get(cur, ()))
        return False

    cx, cy = find(x), find(y)
    if cx == cy:
        return f"{x} = {y}"
    fwd = reachable(cx, cy)
    bwd = reachable(cy, cx)
    if fwd and bwd:
        raise ContradictionError(f"cycle between {x} and {y}")
    if fwd:
        return f"{x} > {y}"
    if bwd:
        return f"{x} < {y}"
    raise ContradictionError(f"underdetermined query: {query} vs premises {premises}")


# ---------------------------------------------------------------- templates
# Placeholders: {A}, {B} entity names; {attr} attribute noun.
# rel=='>'  -> "A की attr B की attr से अधिक है"
# rel=='<'  -> "A की attr B की attr से कम है"
# rel=='='  -> "A की attr B की attr के बराबर है"
POS = {
    ">": "{A} की {attr} {B} की {attr} से अधिक है",
    "<": "{A} की {attr} {B} की {attr} से कम है",
    "=": "{A} की {attr} {B} की {attr} के बराबर है",
}
# Negated phrasing — semantically the strict opposite (generator ensures no ties).
NEG = {
    ">": "{A} की {attr} {B} की {attr} से अधिक नहीं है",
    "<": "{A} की {attr} {B} की {attr} से कम नहीं है",
    "=": "{A} की {attr} {B} की {attr} के बराबर नहीं है",
}
# The question asks about a random relation (not always "more"), so copying the
# question's relation word into the answer is right only by chance.
QUESTIONS = {
    ">": "क्या {A} की {attr} {B} की {attr} से अधिक है?",
    "<": "क्या {A} की {attr} {B} की {attr} से कम है?",
    "=": "क्या {A} की {attr} {B} की {attr} के बराबर है?",
}
QUESTION = QUESTIONS[">"]
OPPOSITE = {">": "<", "<": ">", "=": "="}


def render_premise(a: str, b: str, rel: str, attr: str, negate: bool = False) -> str:
    """Render a premise. With negate=True the TRUE relation rel is phrased as its
    strict opposite ("A is not more than B" means A < B, no ties by construction),
    so parse_premises inverts rendering exactly."""
    tpl = NEG if negate else POS
    return tpl[OPPOSITE[rel] if negate else rel].format(A=a, B=b, attr=attr) + "।"


def render_answer(a: str, b: str, rel: str, attr: str, negate: bool = False) -> str:
    return render_premise(a, b, rel, attr, negate=negate)


def render_cot(premises: list[tuple[str, str, str]], query: tuple[str, str], attr: str, negate: bool = False) -> str:
    """Render the Chain-of-Thought rationale: the premises linking the two query
    entities, each oriented along the path from the first to the second, followed
    by the relation they imply."""
    qa, qb = query
    neighbours: dict[str, list[str]] = {}
    for s, _rel, o in premises:
        neighbours.setdefault(s, []).append(o)
        neighbours.setdefault(o, []).append(s)
    # The premises form a chain, so the path between two entities is unique.
    prev: dict[str, Optional[str]] = {qa: None}
    stack = [qa]
    while stack:
        cur = stack.pop()
        for nxt in neighbours.get(cur, ()):
            if nxt not in prev:
                prev[nxt] = cur
                stack.append(nxt)
    path = [qb]
    while path[-1] != qa:
        path.append(prev[path[-1]])
    path.reverse()
    steps = [solve_relation(premises, (u, v)) for u, v in zip(path, path[1:])]
    return f"[कारण: {' और '.join(steps)} इसलिए {solve_relation(premises, query)}]"


def render_text(paradigm: str, premises: list[tuple[str, str, str]], query: tuple[str, str],
                 attr: str, entity_pool_names: list[str], question_rel: str = ">",
                stated: Optional[list[tuple[str, str, str, str]]] = None) -> str:
    """Wrap the premises + question in the paradigm's framing."""
    a, b = query
    negate = paradigm == "negation"
    # stated: (subj, rel, obj, attr) in display order, when it includes distractors
    stated = stated or [(s, rel, o, attr) for s, rel, o in premises]
    body = " ".join(render_premise(s, o, rel, at, negate=negate) for s, rel, o, at in stated)
    q = QUESTIONS[question_rel].format(A=a, B=b, attr=attr)  # already ends with "?"
    if paradigm == "transitive":
        return body + " " + q
    if paradigm == "multi_hop":
        return body + " " + q
    if paradigm == "negation":
        return body + " " + q
    if paradigm == "conversational":
        # Devanagari-only framing (no ASCII punctuation) keeps script purity >99%.
        lines = []
        for i, (s, rel, o, at) in enumerate(stated):
            speaker = s if i % 2 == 0 else o
            lines.append(f"{speaker} ने कहा कि {render_premise(s, o, rel, at, negate=negate)}")
        lines.append(f"{a} ने पूछा कि {q}")
        return " ".join(lines)
    if paradigm == "word_problem":
        names = " और ".join(entity_pool_names)
        intro = f"एक गाँव में {names} रहते हैं। " if len(names) <= 40 else ""
        return intro + body + " " + q
    raise ValueError(f"unknown paradigm {paradigm!r}")


# ---------------------------------------------------------------- parsing (for test_label_correctness)

_PREMISE_RE = re.compile(
    r"(?P<A>[^\s।:?]+) की (?P<attr>[^\s]+) (?P<B>[^\s।:?]+) की (?P<attr2>[^\s]+) से "
    r"(?P<rel>अधिक|कम) (?P<neg>नहीं )?है"
)
_EQ_RE = re.compile(
    r"(?P<A>[^\s।:?]+) की (?P<attr>[^\s]+) (?P<B>[^\s।:?]+) की (?P<attr2>[^\s]+) के बराबर "
    r"(?P<neg>नहीं )?है"
)


def parse_premises(text: str) -> list[tuple[str, str, str]]:
    """Re-parse rendered Hindi text into (subj, rel, obj) premises.

    Used by test_label_correctness to prove template <-> solver agreement:
    premises parsed from the text are fed back to solve_relation and must match
    the stored answer. Negated phrasing maps to the strict opposite relation.
    Only the statement part before the first "क्या" (question) is parsed, so the
    question never masquerades as a premise.
    """
    statement = text.split("क्या", 1)[0]
    found: list[tuple[str, str, str]] = []
    for m in _PREMISE_RE.finditer(statement):
        a, b, rel = m.group("A"), m.group("B"), m.group("rel")
        neg = bool(m.group("neg"))
        if rel == "अधिक":
            found.append((a, "<" if neg else ">", b))
        else:  # कम
            found.append((a, ">" if neg else "<", b))
    for m in _EQ_RE.finditer(statement):
        a, b = m.group("A"), m.group("B")
        neg = bool(m.group("neg"))
        if neg:
            # "बराबर नहीं" — the generator never emits this for resolvable labels;
            # it only appears in negation-framed val/test content where the true
            # relation is strict (no ties). Map to "<" is invalid in general, so we
            # refuse rather than guess: this branch should be unreachable by
            # construction (generator avoids = in negation examples).
            raise ContradictionError("negated '=' premise is not supported by construction")
        found.append((a, "=", b))
    return found


# ---------------------------------------------------------------- generation

def _pick_entities(pool: list[str], k: int, rng: random.Random) -> list[str]:
    if len(pool) < k:
        raise ValueError(f"pool too small ({len(pool)}) for k={k}")
    return rng.sample(pool, k)


def _make_chain(entities: list[str], attr: str, rng: random.Random, allow_equals: bool,
                min_hops: int = 1, exact_hops: Optional[int] = None,
                force_equal_run: bool = False) -> tuple[list[tuple[str, str, str]], tuple[str, str]]:
    """Assign values, emit one truthful premise per consecutive pair, return (premises, query).

    Each premise is stated in a random direction ("A < B" or "B > A"), the premises
    are shuffled, and the query is a random pair at least min_hops apart in a
    random order — so neither the wording of a premise nor the position of an
    entity in the text gives the answer away.

    exact_hops queries a pair exactly that many hops apart (depth-controlled eval).
    force_equal_run gives 3+ consecutive entities one value and queries two of them, so
    the answer is "=" and is only reachable by chaining equalities.
    """
    if allow_equals:
        values = {e: rng.randint(1, 50) for e in entities}
        if rng.random() < 0.3:
            # Force one equality pair by duplicating a value.
            a, b = rng.sample(entities, 2)
            values[b] = values[a]
    else:
        distinct = rng.sample(range(1, 100), len(entities))
        values = dict(zip(entities, distinct))
    run: list[str] = []
    if force_equal_run and allow_equals and len(entities) >= 3:
        run = rng.sample(entities, 3 if len(entities) == 3 or rng.random() < 0.5 else 4)
        for e in run[1:]:
            values[e] = values[run[0]]
    order = sorted(entities, key=lambda e: (values[e], e))
    premises: list[tuple[str, str, str]] = []
    for lo, hi in zip(order, order[1:]):
        if values[lo] == values[hi]:
            premises.append(rng.choice([(lo, "=", hi), (hi, "=", lo)]))
        else:
            premises.append(rng.choice([(lo, "<", hi), (hi, ">", lo)]))
    rng.shuffle(premises)
    if run:
        idx = sorted(order.index(e) for e in run)
        i, j = rng.choice([(a, b) for a in idx for b in idx if abs(a - b) >= 2])
        return premises, (order[i], order[j])
    if exact_hops is not None:
        i, j = rng.choice([
            (i, j) for i in range(len(order)) for j in range(len(order)) if abs(i - j) == exact_hops
        ])
        return premises, (order[i], order[j])
    min_hops = max(1, min(min_hops, len(order) - 1))
    i, j = rng.choice([
        (i, j) for i in range(len(order)) for j in range(len(order)) if abs(i - j) >= min_hops
    ])
    return premises, (order[i], order[j])


def _branch_premises(chain: list[str], extra: list[str], rng: random.Random,
                     allow_equals: bool) -> list[tuple[str, str, str]]:
    """Attach each extra entity to a random entity already in the graph by one premise.

    The graph stays a tree, so the path between the query entities is unchanged and
    the extra premises are true but irrelevant to the answer."""
    attached = list(chain)
    rels = [">", "<"] + (["="] if allow_equals else [])
    out = []
    for x in extra:
        y, rel = rng.choice(attached), rng.choice(rels)
        out.append(rng.choice([(x, rel, y), (y, OPPOSITE[rel], x)]))
        attached.append(x)
    return out


def _shortcut_premises(premises: list[tuple[str, str, str]], chain: list[str], query: tuple[str, str],
                       n: int, rng: random.Random) -> list[tuple[str, str, str]]:
    """Up to n true premises between chain entities that are not already linked,
    never the query pair itself, so the answer still has to be composed."""
    linked = {frozenset((s, o)) for s, _r, o in premises} | {frozenset(query)}
    pairs = [(u, v) for i, u in enumerate(chain) for v in chain[i + 1:] if frozenset((u, v)) not in linked]
    out = []
    for u, v in rng.sample(pairs, min(n, len(pairs))):
        rel = solve_relation(premises, (u, v)).split()[1]
        out.append(rng.choice([(u, rel, v), (v, OPPOSITE[rel], u)]))
    return out


def _attribute_distractors(chain: list[str], attr: str, n: int,
                           rng: random.Random) -> list[tuple[str, str, str, str]]:
    """Up to n true premises about the same entities under another attribute (no ties,
    so they are valid under negation phrasing too); they say nothing about attr."""
    other = rng.choice([a for a in ATTRIBUTES if a != attr])
    order = rng.sample(chain, len(chain))  # ascending in the other attribute
    pairs = rng.sample(list(zip(order, order[1:])), min(n, len(order) - 1))
    return [(*rng.choice([(lo, "<", hi), (hi, ">", lo)]), other) for lo, hi in pairs]


def _mix_real_names(entities: list[str], real_names: list[str], fraction: float,
                    rng: random.Random) -> list[str]:
    """Replace each name with a real one with probability fraction; names stay distinct."""
    out: list[str] = []
    for name in entities:
        if rng.random() < fraction:
            real = rng.choice(real_names)
            if real not in out and real not in entities:
                name = real
        out.append(name)
    return out


def generate_examples(
    n: int,
    entity_pool: list[str],
    paradigms: tuple[str, ...],
    split: str,
    seed: int,
    allow_equals: bool = True,
    min_hops: int = 1,
    paradigm_weights: Optional[tuple[float, ...]] = None,
    long_query_fraction: float = 0.0,
    fresh: bool = False,
    name_exclude: Optional[set] = None,
    real_names: Optional[list[str]] = None,
    real_name_fraction: float = 0.0,
    balance_answers: bool = False,
    n_entities: Optional[int] = None,
    exact_hops: Optional[int] = None,
    branch_entities: int = 0,
    shortcut_premises: int = 0,
    attribute_distractors: int = 0,
) -> list[ReasoningExample]:
    """Generate n distinct reasoning examples (paradigms drawn uniformly unless weighted).

    long_query_fraction of the non-multi_hop examples query a pair at least two hops
    apart, so the answer has to be composed from premises instead of read off one.

    The extra options are off by default, so the default output is unchanged:
      fresh            names are drawn from the syllable grammar per example (entity_pool
                       is ignored), never from name_exclude
      real_name_fraction  with fresh, each name is swapped for one from real_names
                       with this probability
      balance_answers  the answers are held to equal shares of >, < and = (negation
                       examples cannot be "=", so they take the other two)
      n_entities       fixed number of entities per example
      exact_hops       the query pair is exactly this many hops apart
      branch_entities  extra entities hung off the chain (irrelevant premises)
      shortcut_premises  extra true premises between non-adjacent chain entities
      attribute_distractors  extra premises about another attribute
    """
    rng = random.Random(seed)
    name_rng = random.Random(seed + 7919)
    quota = -(-n // 3)
    answer_counts = {">": 0, "<": 0, "=": 0}
    out: list[ReasoningExample] = []
    seen_texts: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        paradigm = rng.choices(list(paradigms), weights=paradigm_weights)[0]
        attr = rng.choice(ATTRIBUTES)
        k = rng.choice([3, 4] if paradigm != "multi_hop" else [4, 5])
        if n_entities is not None:
            k = n_entities
        k_all = k + branch_entities
        if fresh:
            entities = fresh_names(k_all, name_rng, name_exclude)
            if real_names and real_name_fraction > 0:
                entities = _mix_real_names(entities, real_names, real_name_fraction, name_rng)
        else:
            entities = _pick_entities(entity_pool, k_all, rng)
        long_hops = 2 if rng.random() < long_query_fraction else 1
        question_rel = rng.choices(QUESTION_RELATIONS, weights=QUESTION_WEIGHTS)[0]
        equal_run = balance_answers and paradigm != "negation" and rng.random() < 0.5
        try:
            premises, query = _make_chain(
                # "=" is ambiguous under negation phrasing, so negation examples never get ties.
                entities[:k], attr, rng,
                allow_equals=allow_equals and paradigm != "negation",
                min_hops=min_hops if paradigm == "multi_hop" else long_hops,
                exact_hops=exact_hops,
                force_equal_run=equal_run,
            )
            stated = None
            if branch_entities or shortcut_premises or attribute_distractors:
                chain = entities[:k]
                premises = premises + _branch_premises(chain, entities[k:], rng,
                                                       allow_equals and paradigm != "negation")
                premises = premises + _shortcut_premises(premises, chain, query, shortcut_premises, rng)
                distractors = _attribute_distractors(chain, attr, attribute_distractors, rng)
                stated = [(s, rel, o, attr) for s, rel, o in premises] + distractors
                rng.shuffle(stated)
            answer_rel = solve_relation(premises, query)
        except ContradictionError:
            continue  # skip, never mislabel
        if balance_answers:
            if answer_counts[answer_rel.split()[1]] >= quota:
                continue

        qa, qb = query
        rel = answer_rel.split()[1]
        text = render_text(paradigm, premises, query, attr, entities, question_rel=question_rel,
                           stated=stated)
        if text in seen_texts:
            continue
        seen_texts.add(text)
        ans_clean = render_answer(qa, qb, rel, attr, negate=paradigm == "negation")
        cot_str = f"{render_cot(premises, query, attr, negate=paradigm == 'negation')} {ans_clean}"
        ex = ReasoningExample(
            id=f"hin_{split}_{len(out):06d}",
            split=split,
            paradigm=paradigm,
            attribute=attr,
            entities=entities,
            premises=premises,
            query=query,
            answer_rel=answer_rel,
            answer_text=ans_clean,
            cot_text=cot_str,
            text=text,
            distractors=[d for d in (stated or []) if d[3] != attr],
        )
        out.append(ex)
        answer_counts[answer_rel.split()[1]] += 1
    if len(out) < n:
        raise RuntimeError(f"only generated {len(out)}/{n} examples (pool exhausted?)")
    return out


def generate_dataset(
    n_train: int, n_val: int, n_test: int,
    out_dir: str,
    seed: int = 1337,
    train_pool: Optional[list[str]] = None,
    val_pool: Optional[list[str]] = None,
    test_pool: Optional[list[str]] = None,
    fresh_train_names: bool = False,
    real_name_fraction: float = 0.0,
    extra_eval: bool = False,
    n_extra: int = 500,
    hard_eval: bool = False,
) -> dict:
    """Generate train/val/test with the mandated anti-leakage split and write jsonl.

    fresh_train_names draws every training example's names from the syllable grammar
    instead of the fixed pool; real_name_fraction then swaps that share of them for
    curated train names (fresh names look unlike the real held-out names, and v5 runs
    scored far higher on val_train_names than on val). extra_eval also writes test_balanced.jsonl (equal shares
    of >, < and =), test_depth{2..5}.jsonl (query exactly that many hops apart) and
    val_train_names.jsonl (built like val, but with training-style names, so comparing
    it with val separates failing the task from failing to carry it to unseen names).
    test.jsonl is the legacy benchmark and is not affected by either option."""
    train_pool = train_pool or TRAIN_ENTITY_POOL
    val_pool = val_pool or VAL_ENTITY_POOL
    test_pool = test_pool or TEST_ENTITY_POOL
    assert not (set(train_pool) & set(val_pool)), "train/val entity pools must be disjoint"
    assert not (set(train_pool) & set(test_pool)), "train/test entity pools must be disjoint"
    assert not (set(val_pool) & set(test_pool)), "val/test entity pools must be disjoint"

    # Train split gets 5% negation curriculum using ONLY train_pool entities
    train_paradigms = TRAIN_PARADIGMS + ("negation",)
    other_share = (1.0 - TRAIN_NEGATION_FRACTION) / len(TRAIN_PARADIGMS)
    train = generate_examples(
        n_train, train_pool, train_paradigms, "train", seed=seed,
        allow_equals=True, min_hops=1,
        paradigm_weights=(other_share,) * len(TRAIN_PARADIGMS) + (TRAIN_NEGATION_FRACTION,),
        long_query_fraction=TRAIN_LONG_QUERY_FRACTION,
        fresh=fresh_train_names,
        name_exclude=set(val_pool) | set(test_pool),
        real_names=CURATED_TRAIN_POOL, real_name_fraction=real_name_fraction,
    )
    # Eval splits get all paradigms, in equal shares, incl. the reserved multi_hop.
    all_paradigms = EVAL_PARADIGMS
    val = generate_examples(
        n_val, val_pool, all_paradigms, "val", seed=seed + 1,
        allow_equals=True, min_hops=2,
    )
    test = generate_examples(
        n_test, test_pool, all_paradigms, "test", seed=seed + 2,
        allow_equals=True, min_hops=2,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
    if extra_eval:
        splits["test_balanced"] = generate_examples(
            n_extra, test_pool, all_paradigms, "test_balanced", seed=seed + 3,
            allow_equals=True, min_hops=2, balance_answers=True,
        )
        for hops in (2, 3, 4, 5):
            splits[f"test_depth{hops}"] = generate_examples(
                n_extra, test_pool, all_paradigms, f"test_depth{hops}", seed=seed + 10 + hops,
                allow_equals=True, n_entities=hops + 1, exact_hops=hops,
            )
        splits["val_train_names"] = generate_examples(
            n_extra, train_pool, all_paradigms, "val_train_names", seed=seed + 20,
            allow_equals=True, min_hops=2,
            fresh=fresh_train_names, name_exclude=set(val_pool) | set(test_pool),
            real_names=CURATED_TRAIN_POOL, real_name_fraction=real_name_fraction,
        )
    if hard_eval:
        # Test names only; no training example has more than 4 entities or 3 hops.
        for hops in HARD_DEPTHS:
            splits[f"test_depth{hops}"] = generate_examples(
                n_extra, test_pool, all_paradigms, f"test_depth{hops}", seed=seed + 30 + hops,
                allow_equals=True, n_entities=hops + 1, exact_hops=hops,
            )
        # Same chains as test_depth3 / test_depth4 plus HARD_N_EXTRA_PREMISES premises the
        # answer does not need: compare each with that depth split.
        splits["test_branch"] = generate_examples(
            n_extra, test_pool, all_paradigms, "test_branch", seed=seed + 50,
            allow_equals=True, n_entities=4, exact_hops=3, branch_entities=HARD_N_EXTRA_PREMISES,
        )
        splits["test_distractor_attr"] = generate_examples(
            n_extra, test_pool, all_paradigms, "test_distractor_attr", seed=seed + 51,
            allow_equals=True, n_entities=4, exact_hops=3, attribute_distractors=HARD_N_EXTRA_PREMISES,
        )
        splits["test_redundant"] = generate_examples(
            n_extra, test_pool, all_paradigms, "test_redundant", seed=seed + 52,
            allow_equals=True, n_entities=5, exact_hops=4, shortcut_premises=HARD_N_EXTRA_PREMISES,
        )
    counts = {}
    for name, examples in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        counts[name] = len(examples)

    manifest = {
        "n_train": counts["train"],
        "n_val": counts["val"],
        "n_test": counts["test"],
        "train_entities": train_pool,
        "val_entities": val_pool,
        "test_entities": test_pool,
        "reserved_paradigms": sorted(RESERVED_PARADIGMS),
        "train_negation_fraction": TRAIN_NEGATION_FRACTION,
        "fresh_train_names": fresh_train_names,
        "real_name_fraction": real_name_fraction,
        "extra_eval": sorted(k for k in counts if k not in ("train", "val", "test")),
        "note": "multi_hop reserved for eval; train/val/test entity pools disjoint",
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate reasoning data")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "finetune" / "reasoning"))
    parser.add_argument("--n-train", type=int, default=300000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fresh-train-names", action=argparse.BooleanOptionalAction, default=True,
                        help="draw each training example's names from the syllable grammar")
    parser.add_argument("--real-name-fraction", type=float, default=0.5,
                        help="with --fresh-train-names, share of names taken from the curated train pool")
    parser.add_argument("--extra-eval", action="store_true",
                        help="also write test_balanced.jsonl and test_depth{2..5}.jsonl")
    parser.add_argument("--n-extra", type=int, default=500, help="examples per extra eval file")
    parser.add_argument("--hard-eval", action="store_true",
                        help="also write test_depth{6,8,10}, test_branch, test_distractor_attr, test_redundant")
    args = parser.parse_args(argv)
    manifest = generate_dataset(
        args.n_train, args.n_val, args.n_test, args.out_dir, seed=args.seed,
        fresh_train_names=args.fresh_train_names, real_name_fraction=args.real_name_fraction,
        extra_eval=args.extra_eval, n_extra=args.n_extra,
        hard_eval=args.hard_eval,
    )
    _print_utf8(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _print_utf8(text: str) -> None:
    """Print UTF-8 text safely even when the console codepage is e.g. cp1252."""
    import sys as _sys

    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
