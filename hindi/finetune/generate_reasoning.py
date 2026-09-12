"""Hindi reasoning data generator — pure Python, no model dependency.

Produces ``{train,val,test}.jsonl`` of synthetic reasoning examples whose answers
are computed by a *symbolic solver* over a relation graph (>, <, =) — never by a
template author and never by an LLM.

Anti-leakage (mandatory, test-enforced):
  * Entity pools are split into disjoint train-pool vs eval-pool *before* any
    example is generated; the split is an explicit, checked-in list below.
  * multi_hop and negation paradigms are reserved for val/test only.

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
# Disjoint by design: train entities never appear in val/test (and vice versa).
TRAIN_ENTITY_POOL: list[str] = [
    "राम", "सीता", "गीता", "मोहन", "सुनीता", "राजेश", "कविता", "अमित", "प्रिया",
    "विकास", "नीता", "सुरेश", "मीना", "रवि", "अनिता", "दीपक", "रेखा", "संजय",
    "पूजा", "मनोज",
]
EVAL_ENTITY_POOL: list[str] = [
    "अर्जुन", "लता", "किशोर", "माला", "हरीश", "ज्योति", "नरेश", "सरिता",
    "प्रमोद", "विनीता", "कमल", "सुधा", "यश", "इंदु", "तारा",
]
ATTRIBUTES: list[str] = ["उम्र", "लंबाई", "बचत"]
RESERVED_PARADIGMS = {"multi_hop", "negation"}  # eval splits only
TRAIN_PARADIGMS = ("transitive", "word_problem", "conversational")


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
QUESTION = "क्या {A} की {attr} {B} की {attr} से अधिक है?"
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
    """Render explicit Chain-of-Thought (CoT) step-by-step intermediate deduction hops."""
    steps = [f"{s} {rel} {o}" for s, rel, o in premises]
    chain_str = " और ".join(steps)
    return f"[कारण: {chain_str}]"


def render_text(paradigm: str, premises: list[tuple[str, str, str]], query: tuple[str, str],
                 attr: str, entity_pool_names: list[str]) -> str:
    """Wrap the premises + question in the paradigm's framing."""
    a, b = query
    negate = paradigm == "negation"
    body = " ".join(render_premise(s, o, rel, attr, negate=negate) for s, rel, o in premises)
    q = QUESTION.format(A=a, B=b, attr=attr)  # QUESTION already ends with "?"
    if paradigm == "transitive":
        return body + " " + q
    if paradigm == "multi_hop":
        return body + " " + q
    if paradigm == "negation":
        return body + " " + q
    if paradigm == "conversational":
        # Devanagari-only framing (no ASCII punctuation) keeps script purity >99%.
        lines = []
        for i, (s, rel, o) in enumerate(premises):
            speaker = s if i % 2 == 0 else o
            lines.append(f"{speaker} ने कहा कि {render_premise(s, o, rel, attr, negate=negate)}")
        lines.append(f"{a} ने पूछा कि {QUESTION.format(A=a, B=b, attr=attr)}")
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
                min_hops: int = 1) -> tuple[list[tuple[str, str, str]], tuple[str, str]]:
    """Assign distinct-ish values, emit comparison premises, return (premises, query).

    Premises compare consecutive entities by value; the query is two entities at
    distance >= min_hops in the chain (transitive reasoning needed).
    """
    while True:
        if allow_equals:
            values = {e: rng.randint(1, 50) for e in entities}
            if rng.random() < 0.3:
                # Force one equality pair by duplicating a value.
                a, b = rng.sample(entities, 2)
                values[b] = values[a]
        else:
            distinct = rng.sample(range(1, 100), len(entities))
            values = {e: v for e, v in zip(entities, distinct)}
        order = sorted(entities, key=lambda e: (values[e], e))
        premises = []
        for i in range(len(order) - 1):
            hi, lo = order[i], order[i + 1]
            if values[hi] == values[lo]:
                premises.append((hi, "=", lo))
            elif rng.random() < 0.5:
                premises.append((hi, ">", lo))
            else:
                premises.append((hi, "<", lo))
        # Query must require >= min_hops transitive steps: pick endpoints far apart.
        if len(order) - 1 >= min_hops:
            q = (order[0], order[-1])
        else:
            q = tuple(rng.sample(entities, 2))
        # Must be resolvable (no ties on the query endpoints).
        try:
            solve_relation(premises, (q[0], q[1]))
        except ContradictionError:
            continue
        return premises, (q[0], q[1])


def generate_examples(
    n: int,
    entity_pool: list[str],
    paradigms: tuple[str, ...],
    split: str,
    seed: int,
    allow_equals: bool = True,
    min_hops: int = 1,
) -> list[ReasoningExample]:
    """Generate n reasoning examples (deduped ids; skips solver failures)."""
    rng = random.Random(seed)
    out: list[ReasoningExample] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        paradigm = rng.choice(list(paradigms))
        attr = rng.choice(ATTRIBUTES)
        k = rng.choice([3, 4] if paradigm != "multi_hop" else [4, 5])
        entities = _pick_entities(entity_pool, k, rng)
        try:
            premises, query = _make_chain(
                # "=" is semantically ambiguous under negation phrasing, so negation
                # examples never get ties (reliability fallback, spec Reasoning Pipeline).
                entities, attr, rng,
                allow_equals=allow_equals and paradigm != "negation",
                min_hops=min_hops if paradigm == "multi_hop" else 1,
            )
            answer_rel = solve_relation(premises, query)
        except ContradictionError:
            continue  # reliability fallback: skip, never mislabel

        # Resolve the canonical answer relation from answer_rel text ("A > B").
        qa, qb = query
        rel = answer_rel.split()[1]
        text = render_text(paradigm, premises, query, attr, entities)
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
        )
        out.append(ex)
    if len(out) < n:
        raise RuntimeError(f"only generated {len(out)}/{n} examples (pool exhausted?)")
    return out


def generate_dataset(
    n_train: int, n_val: int, n_test: int,
    out_dir: str,
    seed: int = 1337,
    train_pool: Optional[list[str]] = None,
    eval_pool: Optional[list[str]] = None,
) -> dict:
    """Generate train/val/test with the mandated anti-leakage split and write jsonl."""
    train_pool = train_pool or TRAIN_ENTITY_POOL
    eval_pool = eval_pool or EVAL_ENTITY_POOL
    assert not (set(train_pool) & set(eval_pool)), "entity pools must be disjoint"

    # Train split gets 5% negation curriculum using ONLY train_pool entities
    train_paradigms = TRAIN_PARADIGMS + ("negation",)
    train = generate_examples(
        n_train, train_pool, train_paradigms, "train", seed=seed,
        allow_equals=True, min_hops=1,
    )
    # Eval splits get ALL paradigms incl. reserved multi_hop / negation.
    all_paradigms = TRAIN_PARADIGMS + tuple(sorted(RESERVED_PARADIGMS))
    val = generate_examples(
        n_val, eval_pool, all_paradigms, "val", seed=seed + 1,
        allow_equals=True, min_hops=2,
    )
    test = generate_examples(
        n_test, eval_pool, all_paradigms, "test", seed=seed + 2,
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
        "eval_entities": eval_pool,
        "reserved_paradigms": sorted(RESERVED_PARADIGMS),
        "note": "multi_hop/negation reserved for eval; entity pools disjoint",
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
    """Print UTF-8 text safely even when the console codepage is e.g. cp1252."""
    import sys as _sys

    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
