"""Hindi reasoning data generator — pure Python, no model dependency.

Produces ``{train,val,test}.jsonl`` of synthetic reasoning examples whose answers
are computed by a *symbolic solver* over a relation graph (>, <, =, ?) — never by a
template author and never by an LLM.

Anti-leakage (mandatory, test-enforced):
  * Entity pools are split three ways — train vs val vs test — *before* any
    example is generated; the splits are explicit, checked-in lists below.
  * Held-out evaluation evaluates transitive, multi-hop, negation, conversational,
    word problem, and indeterminate paradigms on strictly unseen entities, and
    val entities never appear in test (so early stopping on val can't leak test).
  * Template diversity: premises/questions are rendered from paraphrase variants
    (synonyms, adverb, swapped तुलना order, question framings) sampled with the
    seeded RNG. Every variant round-trips through ``parse_premises`` and the
    symbolic solver, so surface diversity never changes the gold answer.

Reliability fallbacks:
  * Contradictory/cyclic premise sets make solve_relation raise; the generator catches
    and skips that example.
  * Per-template compatibility rules: every template pairs entities with a single
    attribute (उम्र/लंबाई/बचत) so agreement/gender/honorific issues can't arise.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- entity pools
# Disjoint by design: train / val / test entities never overlap (three-way split,
# checked by generate_dataset asserts and by unit tests). Val and test get
# separate pools so early stopping on val cannot leak test entities.
TRAIN_ENTITY_POOL: list[str] = [
    "राम", "सीता", "गीता", "मोहन", "सुनीता", "राजेश", "कविता", "अमित", "प्रिया",
    "विकास", "नीता", "सुरेश", "मीना", "रवि", "अनिता", "दीपक", "रेखा", "संजय",
    "पूजा", "मनोज", "विजय", "रोहन", "सोहन", "अजय", "नेहा", "किरण", "सीमा",
    "राहुल", "पायल", "गौरव", "वरुण", "दिव्या", "श्वेता", "साक्षी", "आँचल",
    "अनुप", "मयंक", "आलोक", "विवेक", "मनीषा", "भारती", "वर्षा", "संगीता",
    "भावना", "संदीप", "पंकज", "हेमंत", "लोकेश", "तरुण", "कार्तिक", "तन्मय",
    "नीतेश", "मोनिका", "प्रीति", "पल्लवी", "स्वाती", "गरिमा", "चेतना", "नूतन",
    "वंदना", "कल्पना", "रीना", "शालिनी", "आरती", "कुणाल", "अतुल", "चेतन",
    "प्रणव", "भास्कर", "मानस", "दीपांकर", "हिरण्य", "प्रांजल", "ध्रुव", "अनुराग",
    "सौरभ", "नवजीत", "ऋतुपर्ण", "मृदुल", "विपुल", "उत्पल", "निरंजन", "जगदीश",
    "देवजीत", "रूपक", "अंकुर", "प्रशांत", "कौशिक", "रूबुल", "मुकुल", "अरूप",
    "भास्वती", "रूमी", "मृदुला", "निवेदिता", "मोनालिसा", "रीमा", "दीपाली", "मलया",
    "काकली", "जूमणी", "रुणझुम", "मुनमी", "पापड़ी", "सोनाली", "बर्णाली", "रश्मि",
    "स्मिता", "मधुस्मिता", "प्रणामी", "गीतांजलि", "पूरबी", "मीनाक्षी", "नवनीता",
    "डिंपल", "पल्लव", "पार्थ", "अभिजीत", "देवव्रत", "चिन्मय", "सिद्धार्थ", "सत्येन",
    "रमेन", "हरेन", "भूपेन", "खगेन", "प्रमथ", "रत्न", "लक्ष्मी", "ज्ञानेन", "धीरेन",
    "गुणेन", "बीरेन", "महेन", "गोविंद", "माधव", "केशव", "दामोदर", "हरिहर",
    "नारायण", "अनंत", "मोहिधर", "घनश्याम", "कमलेश", "जगन्नाथ", "गोपाल", "त्रैलोक्य",
    "भवेन", "लीलाधर", "हेमचंद्र", "शरद", "महिम", "पद्म", "आनंद", "हेमकांत",
    "धर्मेंद्र", "उपेन", "गिरीन", "कमलजीत", "सुरेन", "जोगेन", "भोगेन", "दिनेश",
    "महेश", "रमेश", "उमेश", "हिमांशु", "प्रभात", "भास्वर", "अचिंत्य", "प्रदीप",
    "संजीव", "राजीव", "कुलदीप", "हरिप्रसाद", "देवाशीष", "परिमल", "सुबोध", "अरविंद",
    "अपूर्व", "शांतनु", "प्रफुल्ल", "योगेश", "त्रिदीप", "रूपम", "प्रीतम", "अभिलाष",
    "निलय", "सुब्रत", "समीर", "सुशील", "हीरक", "द्विपेन", "खगेंद्र", "युगल",
    "उज्ज्वल", "ब्रजेन", "रत्नेश्वर", "गोलेश्वर", "भोगेश्वर", "योगेश्वर", "गुणेश्वर",
    "लीलेश्वर", "महेश्वर", "रमाकांत", "सूर्य", "चंद्र", "इंद्र", "उपेंद्र", "जितेन",
    "नरेन", "हितेश", "पराग", "रूपाली", "प्रतिभा", "उर्मिला", "अंजना", "दीपांविता",
    "पूर्णिमा", "कल्पिता", "प्रियंका", "त्रिवेणी", "अनुराधा", "मिताली", "रूपज्योति",
    "रूपश्री", "सुचित्रा", "सुजाता", "मालविका", "अनसूया", "अपराजिता", "मैत्रेयी",
    "शर्मिष्ठा", "अरुंधती", "गार्गी", "देवयानी",
]
SYMBOLIC_ENTITIES: list[str] = [
    "क", "ख", "ग", "घ", "च", "छ", "ज", "झ", "ट", "ठ", "ड", "ढ", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म",
    "A", "B", "C", "D", "E", "X", "Y", "Z", "W",
]
VAL_ENTITY_POOL: list[str] = [
    "अर्जुन", "लता", "किशोर", "माला", "हरीश", "ज्योति", "नरेश",
]
TEST_ENTITY_POOL: list[str] = [
    "सरिता", "प्रमोद", "विनीता", "कमल", "सुधा", "यश", "इंदु", "तारा",
]
# Backward-compatibility alias (prefer VAL_/TEST_ENTITY_POOL in new code).
EVAL_ENTITY_POOL: list[str] = VAL_ENTITY_POOL + TEST_ENTITY_POOL
ATTRIBUTES: list[str] = ["उम्र", "लंबाई", "बचत"]
# Paradigms emphasized in evaluation (harder reasoning categories). They are NOT
# held out of training — TRAIN_PARADIGMS below includes them; the name records
# which categories the eval analysis focuses on.
EVAL_FOCUS_PARADIGMS = {"multi_hop", "negation", "indeterminate"}
TRAIN_PARADIGMS = ("transitive", "word_problem", "conversational", "multi_hop", "negation", "indeterminate")
ALL_PARADIGMS = ("transitive", "word_problem", "conversational", "negation", "multi_hop", "indeterminate")


@dataclass
class ReasoningExample:
    id: str
    split: str
    paradigm: str
    attribute: str
    entities: list[str]
    premises: list[tuple[str, str, str]]  # (subj, rel, obj), rel in {">", "<", "="}
    query: tuple[str, str]
    answer_rel: str          # canonical solver output, e.g. "राम > सीता" or "राम ? सीता"
    answer_text: str         # rendered direct answer sentence
    cot_text: str            # step-by-step chain-of-thought explanation + answer
    text: str                # rendered question + premises

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
        }


# ---------------------------------------------------------------- solver

class ContradictionError(ValueError):
    """Raised when the premise graph is cyclic/contradictory or underdetermined (when not allowed)."""


def solve_relation(
    premises: list[tuple[str, str, str]],
    query: tuple[str, str],
    allow_underdetermined: bool = False,
) -> str:
    """Pure symbolic reasoning over a relation graph (>, <, =, ?).

    ``=`` edges form equivalence classes (union-find); ``>`` edges form a directed
    graph between classes (``<`` is normalized to the reverse ``>``). The answer is
    the transitive closure: query (x, y) -> "x > y", "x < y", "x = y", or "x ? y".
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
        if allow_underdetermined:
            return f"{x} ? {y}"
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
    if allow_underdetermined:
        return f"{x} ? {y}"
    raise ContradictionError(f"underdetermined query: {query} vs premises {premises}")


# ---------------------------------------------------------------- templates
# Paraphrase variants per relation (template v2). Index 0 is the canonical form;
# the rest add lexical/structural diversity (synonym ज्यादा, adverb अपेक्षाकृत,
# swapped तुलना order, equality synonym समान). render_premise samples an index
# with the seeded RNG; every variant is parsed back by parse_premises below.
POS_VARIANTS: dict[str, list[str]] = {
    ">": [
        "{A} की {attr} {B} की {attr} से अधिक है",
        "{A} की {attr} {B} की {attr} से ज्यादा है",
        "{A} की {attr} {B} की {attr} से अपेक्षाकृत अधिक है",
        "{B} की {attr} की तुलना में {A} की {attr} अधिक है",
    ],
    "<": [
        "{A} की {attr} {B} की {attr} से कम है",
        "{A} की {attr} {B} की {attr} से अपेक्षाकृत कम है",
        "{B} की {attr} की तुलना में {A} की {attr} कम है",
    ],
    "=": [
        "{A} की {attr} {B} की {attr} के बराबर है",
        "{A} की {attr} {B} की {attr} के समान है",
    ],
}
# Negated forms are derived mechanically ("है" -> "नहीं है") so each POS variant
# has a parallel NEG variant; the caller pre-maps rel through OPPOSITE.
NEG_VARIANTS: dict[str, list[str]] = {
    k: [t.replace(" है", " नहीं है") for t in v] for k, v in POS_VARIANTS.items()
}
# Canonical-form aliases (backward compatible; render defaults to variant 0).
POS = {k: v[0] for k, v in POS_VARIANTS.items()}
NEG = {k: v[0] for k, v in NEG_VARIANTS.items()}
# Question framings: polarity "gt" (अधिक) vs "lt" (कम). Questions are never
# parsed (parse_premises splits them off), so these are free-form — but each
# keeps the क्या marker the splitter relies on.
QUESTION_VARIANTS: list[str] = [
    "क्या {A} की {attr} {B} की {attr} से अधिक है?",
    "क्या {A} की {attr} {B} की {attr} से ज्यादा है?",
    "क्या यह सच है कि {A} की {attr} {B} की {attr} से अधिक है?",
]
QUESTION_LT_VARIANTS: list[str] = [
    "क्या {A} की {attr} {B} की {attr} से कम है?",
    "क्या {B} की {attr} की तुलना में {A} की {attr} कम है?",
    "क्या यह सच है कि {A} की {attr} {B} की {attr} से कम है?",
]
QUESTION = QUESTION_VARIANTS[0]
QUESTION_LT = QUESTION_LT_VARIANTS[0]
QUESTION_MARKER = " क्या "
OPPOSITE = {">": "<", "<": ">", "=": "="}
# Bump when surface templates change (datasets generated under different
# versions are not directly comparable); recorded in manifest.json.
TEMPLATE_VERSION = "v2-paraphrase"

# Balanced paradigm sampling matching evaluation distribution (prevents indeterminate over-conservatism)
PARADIGM_TRAIN_WEIGHTS: dict[str, float] = {
    "transitive": 0.20,
    "multi_hop": 0.20,
    "negation": 0.18,
    "word_problem": 0.16,
    "conversational": 0.16,
    "indeterminate": 0.10,
}


def render_premise(
    a: str, b: str, rel: str, attr: str,
    negate: bool = False,
    rng: Optional[random.Random] = None,
    variant: Optional[int] = None,
) -> str:
    """Render one premise, sampling a paraphrase variant with the seeded RNG.

    ``rng=None`` (or explicit ``variant``) renders the canonical form, so answer
    rendering — which never passes an RNG — stays in a single surface template.
    Variant tables differ in length by design (e.g. ">" has a ज्यादा synonym
    form that "<" lacks), so an explicit index is taken modulo the table size.
    """
    table = NEG_VARIANTS if negate else POS_VARIANTS
    variants = table[OPPOSITE[rel] if negate else rel]
    if variant is None:
        variant = rng.randrange(len(variants)) if rng is not None else 0
    return variants[variant % len(variants)].format(A=a, B=b, attr=attr) + "।"


def render_answer(a: str, b: str, rel: str, attr: str, negate: bool = False) -> str:
    if rel == "?":
        return "दी गई जानकारी से यह तय नहीं किया जा सकता है।"
    return render_premise(a, b, rel, attr, negate=negate)


HI_POLARITY_WORDS: tuple[str, ...] = ("अधिक", "कम", "ज्यादा", "बराबर", "समान")


def extract_prompt_lemma_token_ids(prompt_text: str, tokenizer) -> list[int]:
    """Extract token IDs for words in the prompt and relational tokens in Hindi.

    Whitelists symmetric comparative and equality tokens so neither question polarity
    nor prompt entities are suppressed by prompt logit biasing.
    """
    token_ids = set()
    for pw in HI_POLARITY_WORDS:
        token_ids.update(tokenizer.encode(pw))
        token_ids.update(tokenizer.encode(" " + pw))

    words = prompt_text.replace("?", " ").replace("।", " ").replace(",", " ").replace(":", " ").split()
    for w in words:
        if len(w) > 1:
            token_ids.update(tokenizer.encode(w))
            token_ids.update(tokenizer.encode(" " + w))

    return list(token_ids)


def render_cot(
    deductive_premises: list[tuple[str, str, str]],
    query: tuple[str, str],
    attr: str,
    negate: bool = False,
    use_symbolic: bool = False,
) -> str:
    """Render explicit Chain-of-Thought (CoT) step-by-step intermediate deduction hops.
    
    If use_symbolic=True, uses dedicated logic tokens (<COT_START>, <REL_GT>, etc.)
    instead of brackets and ASCII operators for subword-safe rendering.
    """
    if not deductive_premises:
        a, b = query
        if use_symbolic:
            return f"<COT_START> {a} <REL_DISJOINT> {b} <COT_END>"
        return f"[कारण: {a} और {b} दो अलग-अलग समूहों में हैं, कोई पथ नहीं है]"
    
    if use_symbolic:
        _SYM = {">": "<REL_GT>", "<": "<REL_LT>", "=": "<REL_EQ>"}
        steps = [f"{s} {_SYM.get(rel, rel)} {o}" for s, rel, o in deductive_premises]
        chain_str = " और ".join(steps)
        return f"<COT_START> {chain_str} <COT_END>"
    
    steps = [f"{s} {rel} {o}" for s, rel, o in deductive_premises]
    chain_str = " और ".join(steps)
    return f"[कारण: {chain_str}]"


def render_text(
    paradigm: str,
    premises: list[tuple[str, str, str]],
    query: tuple[str, str],
    attr: str,
    entity_pool_names: list[str],
    rng: Optional[random.Random] = None,
    **kwargs,
) -> str:
    """Wrap the premises + question in the paradigm's framing, with shuffled premise presentation."""
    a, b = query
    negate = paradigm == "negation"

    # Present premises in non-linear order so the model must assemble the graph
    display_premises = list(premises)
    if rng is not None and len(display_premises) > 1 and paradigm != "conversational":
        rng.shuffle(display_premises)

    body = " ".join(
        render_premise(s, o, rel, attr, negate=negate, rng=rng)
        for s, rel, o in display_premises
    )
    # Contrastive polarity: sample a question framing for this polarity.
    q_template = kwargs.get("question_template")
    if q_template is None:
        qpool = QUESTION_VARIANTS if kwargs.get("question_polarity", "gt") == "gt" else QUESTION_LT_VARIANTS
        q_template = qpool[rng.randrange(len(qpool))] if rng is not None else qpool[0]
    q = q_template.format(A=a, B=b, attr=attr)

    if paradigm in ("transitive", "multi_hop", "negation", "indeterminate"):
        return body + " " + q
    if paradigm == "conversational":
        lines = []
        for i, (s, rel, o) in enumerate(display_premises):
            speaker = s if i % 2 == 0 else o
            lines.append(f"{speaker} ने कहा कि {render_premise(s, o, rel, attr, negate=negate, rng=rng)}")
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
    r"(?:अपेक्षाकृत )?(?P<rel>अधिक|ज्यादा|कम) (?P<neg>नहीं )?है"
)
# Swapped-order variant: "{B} की {attr} की तुलना में {A} की {attr} अधिक है".
_TULNA_RE = re.compile(
    r"(?P<X>[^\s।:?]+) की (?P<attr>[^\s]+) की तुलना में (?P<Y>[^\s।:?]+) की (?P<attr2>[^\s]+) "
    r"(?P<rel>अधिक|ज्यादा|कम) (?P<neg>नहीं )?है"
)
_EQ_RE = re.compile(
    r"(?P<A>[^\s।:?]+) की (?P<attr>[^\s]+) (?P<B>[^\s।:?]+) की (?P<attr2>[^\s]+) के "
    r"(?P<rel>बराबर|समान) (?P<neg>नहीं )?है"
)


def parse_premises(text: str) -> list[tuple[str, str, str]]:
    """Re-parse rendered Hindi text into (subj, rel, obj) premises."""
    statement = text.split(QUESTION_MARKER, 1)[0]
    found: list[tuple[str, str, str]] = []
    for m in _PREMISE_RE.finditer(statement):
        a, b, rel = m.group("A"), m.group("B"), m.group("rel")
        neg = bool(m.group("neg"))
        base = ">" if rel in ("अधिक", "ज्यादा") else "<"
        found.append((a, OPPOSITE[base] if neg else base, b))
    for m in _TULNA_RE.finditer(statement):
        # The sentence claims "Y is base-er than X" (denied when negated).
        x, y, rel = m.group("X"), m.group("Y"), m.group("rel")
        neg = bool(m.group("neg"))
        base = ">" if rel in ("अधिक", "ज्यादा") else "<"
        found.append((y, OPPOSITE[base] if neg else base, x))
    for m in _EQ_RE.finditer(statement):
        a, b = m.group("A"), m.group("B")
        if m.group("neg"):
            raise ContradictionError("negated '=' premise is not supported by construction")
        found.append((a, "=", b))
    return found


# ---------------------------------------------------------------- generation

def _pick_entities(pool: list[str], k: int, rng: random.Random) -> list[str]:
    if len(pool) < k:
        raise ValueError(f"pool too small ({len(pool)}) for k={k}")
    return rng.sample(pool, k)


def _make_chain(
    entities: list[str],
    attr: str,
    rng: random.Random,
    allow_equals: bool,
    min_hops: int = 1,
    is_indeterminate: bool = False,
    pool_for_distractors: Optional[list[str]] = None,
) -> tuple[list[tuple[str, str, str]], tuple[str, str], list[tuple[str, str, str]]]:
    """Emit comparison premises, return (all_premises, query, deductive_premises)."""
    if is_indeterminate:
        # Form two disconnected subgraphs where no path connects query endpoints
        mid = max(2, len(entities) // 2)
        c1, c2 = entities[:mid], entities[mid:]
        if len(c2) < 2 and pool_for_distractors:
            extra = [e for e in pool_for_distractors if e not in entities]
            if extra:
                c2 = c2 + [rng.choice(extra)]
        p1 = []
        for i in range(len(c1) - 1):
            rel = "<" if rng.random() < 0.5 else ">"
            p1.append((c1[i], rel, c1[i + 1]))
        p2 = []
        for i in range(len(c2) - 1):
            rel = "<" if rng.random() < 0.5 else ">"
            p2.append((c2[i], rel, c2[i + 1]))
        all_premises = p1 + p2
        if not all_premises:
            all_premises = [(c1[0], ">", c1[-1])]
        q = (c1[rng.randint(0, len(c1) - 1)], c2[rng.randint(0, len(c2) - 1)]) if c2 else (c1[0], c1[-1])
        return all_premises, q, []

    while True:
        if allow_equals:
            values = {e: rng.randint(1, 50) for e in entities}
            if rng.random() < 0.2:
                a, b = rng.sample(entities, 2)
                values[b] = values[a]
        else:
            distinct = rng.sample(range(1, 100), len(entities))
            values = {e: v for e, v in zip(entities, distinct)}

        order = sorted(entities, key=lambda e: (values[e], e))
        use_descending = (rng.random() < 0.5)
        if use_descending:
            order = list(reversed(order))

        deductive_premises = []
        for i in range(len(order) - 1):
            e1, e2 = order[i], order[i + 1]
            if values[e1] == values[e2]:
                deductive_premises.append((e1, "=", e2))
            else:
                rel = ">" if use_descending else "<"
                deductive_premises.append((e1, rel, e2))

        if len(order) - 1 >= min_hops:
            q = (order[0], order[-1]) if rng.random() < 0.5 else (order[-1], order[0])
        else:
            q = tuple(rng.sample(entities, 2))

        try:
            ans = solve_relation(deductive_premises, q, allow_underdetermined=False)
            if ans.split()[1] == "=" and not allow_equals:
                continue
        except ContradictionError:
            continue

        all_premises = list(deductive_premises)
        # 35% chance to inject an irrelevant distractor premise from outside the chain
        if pool_for_distractors and rng.random() < 0.35:
            available = [e for e in pool_for_distractors if e not in entities]
            if len(available) >= 2:
                d1, d2 = rng.sample(available, 2)
                d_rel = ">" if rng.random() < 0.5 else "<"
                all_premises.append((d1, d_rel, d2))

        return all_premises, q, deductive_premises


def _weighted_paradigm_choice(paradigms: tuple[str, ...], rng: random.Random) -> str:
    """Choose a paradigm using PARADIGM_TRAIN_WEIGHTS for balanced sampling."""
    available = [p for p in paradigms if p in PARADIGM_TRAIN_WEIGHTS]
    if not available:
        return rng.choice(list(paradigms))
    weights = [PARADIGM_TRAIN_WEIGHTS[p] for p in available]
    total = sum(weights)
    weights = [w / total for w in weights]
    r = rng.random()
    cumulative = 0.0
    for p, w in zip(available, weights):
        cumulative += w
        if r <= cumulative:
            return p
    return available[-1]


def generate_examples(
    n: int,
    entity_pool: list[str],
    paradigms: tuple[str, ...],
    split: str,
    seed: int,
    allow_equals: bool = True,
    min_hops: int = 1,
    use_contrastive: bool = True,
    use_symbolic_tokens: bool = True,
) -> list[ReasoningExample]:
    """Generate n reasoning examples (deduped ids; skips solver failures).
    
    Contrastive polarity: During training, each relational chain is queried from
    both polarity perspectives ("अधिक" and "कम"), generating contrastive pairs that
    decouple answer decoding from question stance priming.
    
    Symbolic tokens: 50% of CoT examples use <COT_START>/<COT_END>/<REL_GT>/etc.
    instead of brackets, preventing subword fragmentation in low-resource tokenizers.
    """
    rng = random.Random(seed)
    out: list[ReasoningExample] = []
    attempts = 0
    is_train = (split == "train")
    while len(out) < n and attempts < n * 60:
        attempts += 1
        # Use weighted sampling during training for 30% indeterminate
        if is_train:
            paradigm = _weighted_paradigm_choice(paradigms, rng)
        else:
            paradigm = rng.choice(list(paradigms))
        attr = rng.choice(ATTRIBUTES)
        is_indet = (paradigm == "indeterminate")
        if is_indet:
            k = 4
        elif paradigm == "multi_hop":
            k = 3  # strictly 2-hop relational chaining across 3 entities (A > B, B > C => A vs C)
        else:
            k = rng.choice([2, 3])  # 1 to 2 hops, 2 to 3 entities per assignment specification
        
        # 30% Abstract / Symbolic variable curriculum during training
        if is_train and rng.random() < 0.30:
            current_pool = SYMBOLIC_ENTITIES
        else:
            current_pool = entity_pool
            
        entities = _pick_entities(current_pool, k, rng)

        try:
            all_premises, query, deductive_premises = _make_chain(
                entities, attr, rng,
                allow_equals=allow_equals and paradigm != "negation" and not is_indet,
                min_hops=min_hops if paradigm == "multi_hop" else 1,
                is_indeterminate=is_indet,
                pool_for_distractors=current_pool,
            )
            answer_rel = solve_relation(all_premises, query, allow_underdetermined=True)
        except ContradictionError:
            continue

        qa, qb = query
        rel = answer_rel.split()[1]
        use_sym = use_symbolic_tokens

        # === Primary example (balanced "अधिक" / "कम" question framing) ===
        pol = "gt" if rng.random() < 0.5 else "lt"
        text = render_text(paradigm, all_premises, query, attr, entities, rng=rng,
                           question_polarity=pol)
        ans_clean = render_answer(qa, qb, rel, attr, negate=paradigm == "negation")
        cot_str = f"{render_cot(deductive_premises, query, attr, negate=paradigm == 'negation', use_symbolic=use_sym)} {ans_clean}"
        ex = ReasoningExample(
            id=f"hin_{split}_{len(out):06d}",
            split=split,
            paradigm=paradigm,
            attribute=attr,
            entities=entities,
            premises=all_premises,
            query=query,
            answer_rel=answer_rel,
            answer_text=ans_clean,
            cot_text=cot_str,
            text=text,
        )
        out.append(ex)
        if len(out) >= n:
            break

        # === Contrastive / Bidirectional counterpart — training only ===
        if use_contrastive and is_train and rng.random() < 0.85:
            # 50% inverted directional query (qb, qa), 50% same query opposite question framing
            if rng.random() < 0.5:
                inv_query = (qb, qa)
                if is_indet:
                    inv_ans_rel = f"{qb} ? {qa}"
                    inv_rel = "?"
                    inv_deductive = []
                else:
                    inv_ans_rel = solve_relation(all_premises, inv_query, allow_underdetermined=True)
                    inv_rel = inv_ans_rel.split()[1]
                    inv_deductive = [(o, OPPOSITE[r], s) for s, r, o in reversed(deductive_premises)]

                inv_pol = "lt" if pol == "gt" else "gt"
                text_inv = render_text(paradigm, all_premises, inv_query, attr, entities, rng=rng,
                                       question_polarity=inv_pol)
                ans_clean_inv = render_answer(qb, qa, inv_rel, attr, negate=paradigm == "negation")
                cot_str_inv = f"{render_cot(inv_deductive, inv_query, attr, negate=paradigm == 'negation', use_symbolic=use_sym)} {ans_clean_inv}"
                ex_inv = ReasoningExample(
                    id=f"hin_{split}_{len(out):06d}",
                    split=split,
                    paradigm=paradigm,
                    attribute=attr,
                    entities=entities,
                    premises=all_premises,
                    query=inv_query,
                    answer_rel=inv_ans_rel,
                    answer_text=ans_clean_inv,
                    cot_text=cot_str_inv,
                    text=text_inv,
                )
                out.append(ex_inv)
            elif not is_indet:
                opp_pol = "lt" if pol == "gt" else "gt"
                text_opp = render_text(paradigm, all_premises, query, attr, entities, rng=rng,
                                       question_polarity=opp_pol)
                ex_opp = ReasoningExample(
                    id=f"hin_{split}_{len(out):06d}",
                    split=split,
                    paradigm=paradigm,
                    attribute=attr,
                    entities=entities,
                    premises=all_premises,
                    query=query,
                    answer_rel=answer_rel,
                    answer_text=ans_clean,
                    cot_text=cot_str,
                    text=text_opp,
                )
                out.append(ex_opp)

    if len(out) < n:
        raise RuntimeError(f"only generated {len(out)}/{n} examples (pool exhausted?)")
    return out[:n]


def generate_dataset(
    n_train: int, n_val: int, n_test: int,
    out_dir: str,
    seed: int = 1337,
    train_pool: Optional[list[str]] = None,
    val_pool: Optional[list[str]] = None,
    test_pool: Optional[list[str]] = None,
    eval_pool: Optional[list[str]] = None,
) -> dict:
    """Generate train/val/test with the mandated three-way anti-leakage split.

    ``eval_pool`` is a deprecated fallback: when given without explicit
    val/test pools it is split deterministically in half (val first).
    """
    train_pool = train_pool or TRAIN_ENTITY_POOL
    if eval_pool is not None and val_pool is None and test_pool is None:
        half = len(eval_pool) // 2
        val_pool, test_pool = eval_pool[:half], eval_pool[half:]
    val_pool = val_pool or VAL_ENTITY_POOL
    test_pool = test_pool or TEST_ENTITY_POOL
    assert not (set(train_pool) & set(val_pool)), "train/val entity pools must be disjoint"
    assert not (set(train_pool) & set(test_pool)), "train/test entity pools must be disjoint"
    assert not (set(val_pool) & set(test_pool)), "val/test entity pools must be disjoint"
    assert not (set(SYMBOLIC_ENTITIES) & (set(val_pool) | set(test_pool))), \
        "symbolic train curriculum must not leak into eval pools"

    # Train split gets full curriculum across paradigms using ONLY train_pool entities
    train = generate_examples(
        n_train, train_pool, TRAIN_PARADIGMS, "train", seed=seed,
        allow_equals=True, min_hops=1,
    )
    # Val and test get all paradigms evaluated on their own held-out entities
    val = generate_examples(
        n_val, val_pool, ALL_PARADIGMS, "val", seed=seed + 1,
        allow_equals=True, min_hops=2,
    )
    test = generate_examples(
        n_test, test_pool, ALL_PARADIGMS, "test", seed=seed + 2,
        allow_equals=True, min_hops=2,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "val": val, "test": test}
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
        "eval_entities": sorted(set(val_pool) | set(test_pool)),
        "template_version": TEMPLATE_VERSION,
        "eval_focus_paradigms": sorted(EVAL_FOCUS_PARADIGMS),
        "note": "multi_hop/negation/indeterminate included in eval; train/val/test entity pools 100% disjoint",
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate reasoning data")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "finetune" / "reasoning"))
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    manifest = generate_dataset(
        args.n_train, args.n_val, args.n_test, args.out_dir, seed=args.seed
    )
    _print_utf8(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _print_utf8(text: str) -> None:
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
