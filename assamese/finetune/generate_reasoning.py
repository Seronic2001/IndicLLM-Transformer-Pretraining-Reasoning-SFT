"""Assamese reasoning data generator (Reasoning Pipeline mirror) — pure Python.

Identical design to hindi/finetune/generate_reasoning.py but with Assamese
(Bengali-Assamese script) templates and entity pools. No code is imported from
``hindi/`` (spec Section 0.2: the two languages share nothing).

Anti-leakage: disjoint train/val/test entity pools; multi_hop reserved for val/test
(negation is 5% of train). Premises are stated in a random direction and shuffled,
and the query pair and its order are random, so the answer cannot be read off any
one premise. The question asks about a random relation, drawn independently of the
answer; half of the training queries are at least two hops apart; and the train pool
adds 3,000 syllable-built names (disjoint from val/test). Answers come only from the
symbolic ``solve_relation`` solver.
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
# Three disjoint pools: a name seen in training never appears in val or test, and
# val and test share no names either, so picking a checkpoint on val cannot leak test.
TRAIN_ENTITY_POOL: list[str] = [
    "ৰাম", "সীতা", "গীতা", "মোহন", "সুনীতা", "ৰাজেশ", "কবিতা", "অমিত", "প্ৰিয়া", "বিকাশ",
    "নীতা", "সুৰেশ", "মীনা", "ৰবি", "অনিতা", "দীপক", "ৰেখা", "সঞ্জয়", "পূজা", "মনোজ", "বিজয়",
    "ৰোহন", "সোহন", "অজয়", "নেহা", "কিৰণ", "সীমা", "ৰাহুল", "পায়েল", "গৌৰৱ", "বৰুণ", "দিব্যা",
    "শ্বেতা", "সাক্ষী", "আঁচল", "অনুপ", "ময়ংক", "অলোক", "বিবেক", "মনীষা", "ভাৰতী", "বৰ্ষা",
    "সংগীতা", "ভাৱনা", "সন্দীপ", "পংকজ", "হেমন্ত", "লোকেশ", "তৰুণ", "কাৰ্তিক", "তন্ময়",
    "নীতেশ", "মণিকা", "প্ৰীতি", "পল্লৱী", "স্বাতী", "গৰিমা", "চেতনা", "নূতন", "বন্দনা",
    "কল্পনা", "ৰীনা", "শালিনী", "আৰতি", "কুণাল", "অতুল", "চেতন", "প্ৰণৱ", "ভাস্কৰ", "মানস",
    "দীপাংকৰ", "হিৰণ্য", "প্ৰাঞ্জল", "ধ্ৰুৱ", "অনুৰাগ", "সৌৰভ", "নৱজিত", "ঋতুপৰ্ণ", "মৃদুল",
    "বিপুল", "উৎপল", "নিৰঞ্জন", "জগদীশ", "দেৱজিত", "ৰূপক", "অংকুৰ", "প্ৰশান্ত", "কৌশিক",
    "ৰুবুল", "মুকুল", "অৰূপ", "ভাস্বতী", "ৰুমী", "মৃদুলা", "নিবেদিতা", "মনালিচা", "ৰীমা",
    "দিপালী", "মলয়া", "কাকলি", "জুমণি", "ৰুণজুমি", "মুনমী", "পাপৰি", "জোনালী", "বৰ্ণালী",
    "ৰশ্মি", "স্মিতা", "মধুস্মিতা", "প্ৰণামী", "গীতাঞ্জলি", "পূৰবী", "মিনাক্ষী", "নৱনীতা",
    "ডিম্পল", "পল্লৱ", "পাৰ্থ", "অভিজিৎ", "দেৱব্ৰত", "চিন্ময়", "সিদ্ধাৰ্থ", "সত্যেন", "ৰমেন",
    "হৰেন", "ভূপেন", "খগেন", "প্ৰমথ", "ৰত্ন", "লক্ষ্মী", "জ্ঞানেন", "ধীৰেন", "গুণেন", "বীৰেন",
    "মহেন", "গোবিন্দ", "মাধৱ", "কেশৱ", "দামোদৰ", "হৰিহৰ", "নাৰায়ণ", "অনন্ত", "মুহিধৰ",
    "ঘনশ্যাম", "কমলেশ", "জগন্নাথ", "গোপাল", "ত্ৰৈলোক্য", "ভবেন", "লীলাধৰ", "হেমচন্দ্ৰ", "শৰৎ",
    "মহিম", "পদ্ম", "আনন্দ", "হেমকান্ত", "ধৰ্মেন্দ্ৰ", "উপেন", "গিৰীন", "কমলজিত", "সুৰেন",
    "যোগেন", "ভোগেন", "দীনেশ", "মহেশ", "ৰমেশ", "উমেশ", "হিমাংশু", "প্ৰভাত", "ভাস্বৰ",
    "অচিন্ত্য", "প্ৰদীপ", "সঞ্জীৱ", "ৰাজীৱ", "কুলদীপ", "হৰিপ্ৰসাদ", "দেৱাশীষ", "পৰিমল", "সুবোধ",
    "অৰবিন্দ", "অপূৰ্ব", "শান্তনু", "প্ৰফুল্ল", "যোগেশ", "ত্ৰিদীপ", "ৰূপম", "প্ৰীতম",
]
VAL_ENTITY_POOL: list[str] = [
    "অৰ্জুন", "লতা", "কিশোৰ", "মালা", "হৰিশ", "জ্যোতি", "নাৰেশ", "অভিলাষ", "নিলয়", "সুব্ৰত",
    "সমীৰ", "সুশীল", "হীৰক", "দ্বিপেন", "খগেন্দ্ৰ", "যুগল", "উজ্জ্বল", "ব্ৰজেন", "ৰত্নেশ্বৰ",
    "গোলেশ্বৰ", "ভোগেশ্বৰ", "যোগেশ্বৰ", "গুণেশ্বৰ", "লীলেশ্বৰ", "মহেশ্বৰ", "ৰমাকান্ত", "সূৰ্য",
]
TEST_ENTITY_POOL: list[str] = [
    "সৰিতা", "প্ৰমোদ", "বিনীতা", "কমল", "সুধা", "যশ", "ইন্দু", "তাৰা", "চন্দ্ৰ", "ইন্দ্ৰ",
    "উপেন্দ্ৰ", "জিতেন", "নৰেন", "হিতেশ", "পৰাগ", "ৰূপালী", "প্ৰতিভা", "উৰ্মিলা", "অঞ্জনা",
    "দীপান্বিতা", "পূৰ্ণিমা", "কল্পিতা", "প্ৰিয়ংকা", "ত্ৰিবেণী", "অনুৰাধা", "মিতালী",
    "ৰূপজ্যোতি", "ৰূপশ্রী", "সুচিত্ৰা", "সুজাতা", "মালবিকা", "অনসূয়া", "অপৰাজিতা", "মৈত্ৰেয়ী",
    "শৰ্মিষ্ঠা", "অৰুন্ধতী", "গাৰ্গী", "দেৱযানী",
]
# The curated train pool has ~190 names, and the val/test names are never seen in
# training, so models misspelled unseen names when copying them into the answer.
# Training therefore also uses N_SYNTHETIC_TRAIN_NAMES names built from syllables;
# they are filtered against the val/test pools, so all three pools stay disjoint.
N_SYNTHETIC_TRAIN_NAMES = 3000
_SYN_ONSETS = ["অ", "আ", "ই", "উ", "ক", "কি", "কু", "কা", "গ", "গো", "জ", "জি", "ত", "তা",
               "দে", "দী", "ন", "নি", "প", "প্ৰ", "বি", "ভা", "ম", "মী", "মো", "ৰ", "ৰা", "ৰূ",
               "ল", "লী", "শ", "শী", "স", "সু", "সো", "হ", "হে", "চা", "ধী", "যো"]
_SYN_MIDDLES = ["", "ন", "ৰ", "ম", "ল", "ব", "শ", "ত", "ধ", "জ", "ক", "দ", "প", "গ", "হ",
                "নী", "ৰা", "মি", "লে"]
_SYN_ENDINGS = ["না", "তা", "তী", "নী", "কা", "ৰী", "য়া", "ৰা", "ল", "ন", "শ", "ত", "ৰ", "ম", "ব",
                "ক", "দ", "প", "জ", "লী", "মা", "শা", "ধা"]
# Template words a syllable combination could spell; never used as names.
_TEMPLATE_WORDS = {"কম", "বেছি", "সমান", "নহয়", "আৰু", "কলে", "সুধিলে", "যে", "কি", "এখন",
                   "গাঁৱত", "থাকে", "সেয়ে", "কাৰণ"}


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
_FRESH_INITIAL_VOWELS = ["অ", "আ", "ই", "ঈ", "উ", "এ", "ও"]
_FRESH_CONSONANTS = list("কখগঘচজটডতথদধনপফবভমযৰলৱশসহ")
_FRESH_CLUSTERS = ["ক্ৰ", "গ্ৰ", "ত্ৰ", "প্ৰ", "শ্ৰ", "স্ত", "ন্দ", "জ্য", "স্ব", "ক্ষ", "দ্ব",
                   "ৰ্ম", "ল্প", "ন্ত", "ম্ব", "ষ্ণ", "ত্য", "ধ্ৰ", "স্ন", "জ্ঞ"]
_FRESH_MATRAS = ["", "", "া", "ি", "ী", "ু", "ূ", "ে", "ো", "ং", "ংগ"]
_FRESH_ENDINGS = ["", "", "", "ন", "ৰ", "ল", "ত", "শ", "ম", "ৱ"]


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

ATTRIBUTES: list[str] = ["বয়স", "উচ্চতা", "সঞ্চয়"]
RESERVED_PARADIGMS = {"multi_hop"}  # eval splits only
TRAIN_PARADIGMS = ("transitive", "word_problem", "conversational")
EVAL_PARADIGMS = TRAIN_PARADIGMS + ("multi_hop", "negation")
TRAIN_NEGATION_FRACTION = 0.05  # negation curriculum share of the train split
TRAIN_LONG_QUERY_FRACTION = 0.5  # train queries forced >= 2 hops apart (otherwise >= 1)
QUESTION_RELATIONS = (">", "<", "=")  # relation the question asks about...
QUESTION_WEIGHTS = (0.4, 0.4, 0.2)    # ...drawn independently of the true answer


@dataclass
class ReasoningExample:
    id: str
    split: str
    paradigm: str
    attribute: str
    entities: list[str]
    premises: list[tuple[str, str, str]]
    query: tuple[str, str]
    answer_rel: str
    answer_text: str
    cot_text: str
    text: str

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
    pass


def solve_relation(premises: list[tuple[str, str, str]], query: tuple[str, str]) -> str:
    """Same union-find + DAG solver as the Hindi mirror (see that module for the
    full explanation). Relations: ">" "<" "=" between entity names."""
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
    raise ContradictionError(f"underdetermined query: {query}")


# ---------------------------------------------------------------- templates (Assamese)
# rel=='>'  -> "Aৰ attr Bৰ attrতকৈ বেছি" (more than)
# rel=='<'  -> "Aৰ attr Bৰ attrতকৈ কম" (less than)
# rel=='='  -> "Aৰ attr Bৰ attrৰ সমান" (equal to)
POS = {
    ">": "{A}ৰ {attr} {B}ৰ {attr}তকৈ বেছি",
    "<": "{A}ৰ {attr} {B}ৰ {attr}তকৈ কম",
    "=": "{A}ৰ {attr} {B}ৰ {attr}ৰ সমান",
}
NEG = {
    ">": "{A}ৰ {attr} {B}ৰ {attr}তকৈ বেছি নহয়",
    "<": "{A}ৰ {attr} {B}ৰ {attr}তকৈ কম নহয়",
    "=": "{A}ৰ {attr} {B}ৰ {attr}ৰ সমান নহয়",
}
# The question asks about a random relation (not always "more"), so copying the
# question's relation word into the answer is right only by chance.
QUESTIONS = {
    ">": "কি {A}ৰ {attr} {B}ৰ {attr}তকৈ বেছি?",
    "<": "কি {A}ৰ {attr} {B}ৰ {attr}তকৈ কম?",
    "=": "কি {A}ৰ {attr} {B}ৰ {attr}ৰ সমান?",
}
QUESTION = QUESTIONS[">"]
OPPOSITE = {">": "<", "<": ">", "=": "="}
QUESTION_MARKER = " কি "  # space-delimited question marker for parse splitting


def render_premise(a: str, b: str, rel: str, attr: str, negate: bool = False) -> str:
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
    return f"[কাৰণ: {' আৰু '.join(steps)} সেয়ে {solve_relation(premises, query)}]"


def render_text(paradigm: str, premises: list[tuple[str, str, str]], query: tuple[str, str],
                attr: str, entity_pool_names: list[str], question_rel: str = ">") -> str:
    a, b = query
    negate = paradigm == "negation"
    body = " ".join(render_premise(s, o, rel, attr, negate=negate) for s, rel, o in premises)
    q = QUESTIONS[question_rel].format(A=a, B=b, attr=attr)
    if paradigm in ("transitive", "multi_hop", "negation"):
        return body + " " + q
    if paradigm == "conversational":
        lines = []
        for i, (s, rel, o) in enumerate(premises):
            speaker = s if i % 2 == 0 else o
            lines.append(f"{speaker} কলে যে {render_premise(s, o, rel, attr, negate=negate)}")
        lines.append(f"{a} সুধিলে যে {q}")
        return " ".join(lines)
    if paradigm == "word_problem":
        names = " আৰু ".join(entity_pool_names)
        intro = f"এখন গাঁৱত {names} থাকে। " if len(names) <= 40 else ""
        return intro + body + " " + q
    raise ValueError(f"unknown paradigm {paradigm!r}")


# ---------------------------------------------------------------- parsing (for test_label_correctness)
_PREMISE_RE = re.compile(
    r"(?P<A>[^\s।?]+)ৰ (?P<attr>[^\s]+) (?P<B>[^\s।?]+)ৰ (?P<attr2>[^\s]+)তকৈ "
    r"(?P<rel>বেছি|কম)(?P<neg> নহয়)?"
)
_EQ_RE = re.compile(
    r"(?P<A>[^\s।?]+)ৰ (?P<attr>[^\s]+) (?P<B>[^\s।?]+)ৰ (?P<attr2>[^\s]+)ৰ সমান(?P<neg> নহয়)?"
)


def parse_premises(text: str) -> list[tuple[str, str, str]]:
    """Re-parse rendered Assamese text into (subj, rel, obj) premises.

    Only the statement part before the space-delimited question marker is parsed,
    so the question never masquerades as a premise. Negation ("বেছি নহয়") maps to
    the strict opposite relation.
    """
    statement = text.split(QUESTION_MARKER, 1)[0]
    found: list[tuple[str, str, str]] = []
    for m in _PREMISE_RE.finditer(statement):
        a, b, rel = m.group("A"), m.group("B"), m.group("rel")
        neg = bool(m.group("neg"))
        if rel == "বেছি":
            found.append((a, "<" if neg else ">", b))
        else:  # কম
            found.append((a, ">" if neg else "<", b))
    for m in _EQ_RE.finditer(statement):
        a, b = m.group("A"), m.group("B")
        if m.group("neg"):
            raise ContradictionError("negated '=' premise unsupported by construction")
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
        if fresh:
            entities = fresh_names(k, name_rng, name_exclude)
            if real_names and real_name_fraction > 0:
                entities = _mix_real_names(entities, real_names, real_name_fraction, name_rng)
        else:
            entities = _pick_entities(entity_pool, k, rng)
        long_hops = 2 if rng.random() < long_query_fraction else 1
        question_rel = rng.choices(QUESTION_RELATIONS, weights=QUESTION_WEIGHTS)[0]
        equal_run = balance_answers and paradigm != "negation" and rng.random() < 0.5
        try:
            premises, query = _make_chain(
                # "=" is ambiguous under negation phrasing, so negation examples never get ties.
                entities, attr, rng,
                allow_equals=allow_equals and paradigm != "negation",
                min_hops=min_hops if paradigm == "multi_hop" else long_hops,
                exact_hops=exact_hops,
                force_equal_run=equal_run,
            )
            answer_rel = solve_relation(premises, query)
        except ContradictionError:
            continue  # skip, never mislabel
        if balance_answers:
            if answer_counts[answer_rel.split()[1]] >= quota:
                continue

        qa, qb = query
        rel = answer_rel.split()[1]
        text = render_text(paradigm, premises, query, attr, entities, question_rel=question_rel)
        if text in seen_texts:
            continue
        seen_texts.add(text)
        ans_clean = render_answer(qa, qb, rel, attr, negate=paradigm == "negation")
        cot_str = f"{render_cot(premises, query, attr, negate=paradigm == 'negation')} {ans_clean}"
        ex = ReasoningExample(
            id=f"asm_{split}_{len(out):06d}",
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
    parser = argparse.ArgumentParser(description="Generate Assamese reasoning data")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "finetune" / "reasoning"))
    parser.add_argument("--n-train", type=int, default=480000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fresh-train-names", action=argparse.BooleanOptionalAction, default=True,
                        help="draw each training example's names from the syllable grammar")
    parser.add_argument("--real-name-fraction", type=float, default=0.0,
                        help="with --fresh-train-names, share of names taken from the curated train pool")
    parser.add_argument("--extra-eval", action="store_true",
                        help="also write test_balanced.jsonl and test_depth{2..5}.jsonl")
    parser.add_argument("--n-extra", type=int, default=500, help="examples per extra eval file")
    args = parser.parse_args(argv)
    manifest = generate_dataset(
        args.n_train, args.n_val, args.n_test, args.out_dir, seed=args.seed,
        fresh_train_names=args.fresh_train_names, real_name_fraction=args.real_name_fraction,
        extra_eval=args.extra_eval, n_extra=args.n_extra,
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
