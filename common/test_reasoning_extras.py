"""Fresh names, balanced answers, depth-controlled queries and relation-level scoring.

Run for both languages: the Hindi and Assamese generators and finetune modules mirror
each other, so one parametrised test covers both.
"""
from __future__ import annotations

import collections
import importlib
import random
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(lang: str):
    """(generate_reasoning, finetune) of one language, imported as that language's package."""
    for p in (str(_ROOT), str(_ROOT / lang)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for name in ("finetune", "finetune.generate_reasoning", "finetune.finetune"):
        sys.modules.pop(name, None)  # `finetune` is a top-level package name in both languages
    gen = importlib.import_module(f"{lang}.finetune.generate_reasoning")
    sys.modules["finetune.generate_reasoning"] = gen  # what finetune.asserted_relation imports
    ft = importlib.import_module(f"{lang}.finetune.finetune")
    return gen, ft


@pytest.fixture(params=["hindi", "assamese"])
def lang(request):
    return _load(request.param)


def test_fresh_names_avoid_eval_pools(lang):
    gen, _ = lang
    exclude = set(gen.VAL_ENTITY_POOL) | set(gen.TEST_ENTITY_POOL)
    rng = random.Random(0)
    names = {n for _ in range(300) for n in gen.fresh_names(4, rng, exclude)}
    assert len(names) > 600 and not names & exclude
    train = gen.generate_examples(
        200, [], gen.TRAIN_PARADIGMS, "train", seed=1, fresh=True, name_exclude=exclude
    )
    assert not {e for ex in train for e in ex.entities} & exclude


def test_defaults_are_unchanged(lang):
    gen, _ = lang
    a = gen.generate_examples(50, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=3, min_hops=2)
    b = gen.generate_examples(50, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=3, min_hops=2,
                              fresh=False, balance_answers=False)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]


def test_balanced_answers_and_labels(lang):
    gen, _ = lang
    ex = gen.generate_examples(300, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=4,
                               min_hops=2, balance_answers=True)
    counts = collections.Counter(e.answer_rel.split()[1] for e in ex)
    assert set(counts) == {">", "<", "="} and max(counts.values()) <= 100
    for e in ex:
        assert gen.solve_relation(e.premises, e.query) == e.answer_rel


@pytest.mark.parametrize("hops", [2, 3, 4, 5, 6, 8, 10])
def test_depth_split_queries_exact_hops(lang, hops):
    gen, _ = lang
    ex = gen.generate_examples(60, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=5,
                               n_entities=hops + 1, exact_hops=hops)
    for e in ex:
        assert len(e.entities) == hops + 1 and len(e.premises) == hops
        assert gen.solve_relation(e.premises, e.query) == e.answer_rel


def test_val_train_names_split(lang, tmp_path):
    gen, ft = lang
    gen.generate_dataset(60, 30, 30, str(tmp_path), fresh_train_names=True, extra_eval=True, n_extra=40)
    ex = ft.load_reasoning_jsonl(tmp_path / "val_train_names.jsonl")
    names = {e for d in ex for e in d["entities"]}
    assert len(ex) == 40 and "val_train_names" in ft.EXTRA_EVAL_FILES
    assert not names & (set(gen.VAL_ENTITY_POOL) | set(gen.TEST_ENTITY_POOL))
    assert "multi_hop" in {d["paradigm"] for d in ex}  # built like val, names aside


def _parsed(gen, e):
    return sorted(map(tuple, gen.parse_premises(e.text)))


def test_branch_premises_keep_the_query_path(lang):
    gen, ft = lang
    ex = gen.generate_examples(80, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=6,
                               n_entities=4, exact_hops=3, branch_entities=3)
    for e in ex:
        assert len(e.entities) == 7 and len(e.premises) == 6  # a tree: one path per pair
        assert ft.query_hops(e.to_dict()) == 3
        assert gen.solve_relation(e.premises, e.query) == e.answer_rel
        assert _parsed(gen, e) == sorted(e.premises)


def test_shortcut_premises_never_link_the_query_pair(lang):
    gen, _ = lang
    ex = gen.generate_examples(80, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=7,
                               n_entities=5, exact_hops=4, shortcut_premises=3)
    for e in ex:
        assert len(e.premises) == 7
        assert all({s, o} != set(e.query) for s, _r, o in e.premises)
        assert gen.solve_relation(e.premises, e.query) == e.answer_rel
        assert _parsed(gen, e) == sorted(e.premises)


def test_attribute_distractors_are_stated_but_not_premises(lang):
    gen, _ = lang
    ex = gen.generate_examples(80, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=8,
                               n_entities=4, exact_hops=3, attribute_distractors=3)
    for e in ex:
        assert len(e.premises) == 3 and len(e.distractors) == 3
        assert all(at != e.attribute and at in gen.ATTRIBUTES for *_x, at in e.distractors)
        assert gen.solve_relation(e.premises, e.query) == e.answer_rel
        assert _parsed(gen, e) == sorted(e.premises + [tuple(d[:3]) for d in e.distractors])
        assert "distractors" in e.to_dict()
    plain = gen.generate_examples(5, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=8)
    assert all("distractors" not in e.to_dict() for e in plain)


def test_hard_eval_splits(lang, tmp_path):
    gen, ft = lang
    gen.generate_dataset(60, 30, 30, str(tmp_path), extra_eval=True, hard_eval=True, n_extra=40)
    for name in ("test_depth6", "test_depth8", "test_depth10", "test_branch",
                 "test_distractor_attr", "test_redundant"):
        ex = ft.load_reasoning_jsonl(tmp_path / f"{name}.jsonl")
        assert len(ex) == 40
        assert {n for d in ex for n in d["entities"]} <= set(gen.TEST_ENTITY_POOL)


def test_split_names_encoding(lang):
    gen, ft = lang
    from tokenizer.tokenizer import Tokenizer

    lang_name = gen.__name__.split(".")[0]
    base = Tokenizer(str(_ROOT / lang_name / "tokenizer" / f"{lang_name}.model"))
    ex = gen.generate_examples(100, [], gen.TRAIN_PARADIGMS, "train", seed=7, fresh=True,
                               name_exclude=set(gen.VAL_ENTITY_POOL) | set(gen.TEST_ENTITY_POOL))
    ex += gen.generate_examples(100, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=8, min_hops=2)
    plain = ft.NameSplitTokenizer(base, [])
    split = ft.NameSplitTokenizer(base, {n for e in ex for n in e.entities})
    for e in ex:
        for text in (e.text, " " + e.answer_text, " " + e.cot_text):
            assert plain.encode(text) == base.encode(text)
            ids = split.encode(text)
            assert split.decode(ids) == " ".join(text.split())
        ids = split.encode(e.text)
        for name in e.entities:  # every occurrence starts with the bare name's tokens
            name_ids = base.encode(name)
            hits = sum(ids[i:i + len(name_ids)] == name_ids for i in range(len(ids)))
            assert hits == e.text.count(name)


def test_asserted_relation_tolerates_misspelt_names(lang):
    gen, ft = lang
    ex = gen.generate_examples(40, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=6, min_hops=2)
    for e in ex:
        d = e.to_dict()
        gold = e.answer_rel.split()[1]
        assert ft.asserted_relation(e.answer_text, d) == gold
        qa = e.query[0]
        broken = e.answer_text.replace(qa, qa[:-1] + qa[-1] * 2, 1)  # one wrong character
        assert ft.asserted_relation(broken, d) == gold
        assert ft.asserted_relation("", d) is None


def test_relation_metrics(lang):
    _, ft = lang
    m = ft.relation_metrics(["<", "<", ">", "="], ["<", ">", ">", None])
    assert m["accuracy_relation"] == 0.5
    assert m["per_relation_recall"] == {"<": 0.5, "=": 0.0, ">": 1.0}
    assert m["n_per_relation"] == {"<": 2, "=": 1, ">": 1}


def test_real_name_mix(lang):
    gen, _ = lang
    exclude = set(gen.VAL_ENTITY_POOL) | set(gen.TEST_ENTITY_POOL)
    kwargs = dict(fresh=True, name_exclude=exclude)
    plain = gen.generate_examples(100, [], gen.TRAIN_PARADIGMS, "train", seed=9, **kwargs)
    off = gen.generate_examples(100, [], gen.TRAIN_PARADIGMS, "train", seed=9, **kwargs,
                                real_names=gen.CURATED_TRAIN_POOL, real_name_fraction=0.0)
    assert [e.to_dict() for e in plain] == [e.to_dict() for e in off]
    mixed = gen.generate_examples(300, [], gen.TRAIN_PARADIGMS, "train", seed=9, **kwargs,
                                  real_names=gen.CURATED_TRAIN_POOL, real_name_fraction=0.5)
    names = [n for e in mixed for n in e.entities]
    real = sum(n in set(gen.CURATED_TRAIN_POOL) for n in names) / len(names)
    assert 0.35 < real < 0.6 and not set(names) & exclude
    for e in mixed:
        assert len(set(e.entities)) == len(e.entities)
        assert gen.solve_relation(e.premises, e.query) == e.answer_rel


@pytest.mark.parametrize("hops", [2, 3, 4])
def test_query_hops(lang, hops):
    gen, ft = lang
    ex = gen.generate_examples(30, gen.TEST_ENTITY_POOL, gen.EVAL_PARADIGMS, "test", seed=10,
                               n_entities=hops + 1, exact_hops=hops)
    assert all(ft.query_hops(e.to_dict()) == hops for e in ex)


class _CharTokenizer:
    eos_id = 1

    def encode(self, text):
        return [2 + ord(c) % 200 for c in text]


def test_curriculum_limits_sampled_hops(lang):
    gen, ft = lang
    ex = [e.to_dict() for e in gen.generate_examples(
        200, gen.TRAIN_ENTITY_POOL, gen.TRAIN_PARADIGMS, "train", seed=11, long_query_fraction=0.5)]
    data = ft.PromptMaskedSFTDataset(ex, _CharTokenizer(), block_size=256, use_cot=False)
    assert {1, 2} <= set(data.hops)
    assert [ft.curriculum_max_hops(s, 4000) for s in (0, 1999, 2000, 3999, 4000)] == [1, 1, 2, 2, None]
    data.set_max_hops(1)
    g = torch.Generator().manual_seed(0)
    x, _ = data.get_batch(64, "cpu", generator=g)
    w = x.shape[1]
    allowed = {tuple(data.samples[i][0][:w].tolist()) for i, h in enumerate(data.hops) if h == 1}
    assert all(tuple(row.tolist()) in allowed for row in x)
    data.set_max_hops(None)
    assert data.active is None
