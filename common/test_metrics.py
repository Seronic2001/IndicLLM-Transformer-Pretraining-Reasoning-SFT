"""Tests for common/metrics.py — Agent-E acceptance tests (metrics part)."""

import math

import pytest

from common.metrics import (
    bits_per_byte,
    bleu,
    chrf,
    distinct_n,
    perplexity,
    repetition_rate,
    rouge_l,
)


# ---------------------------------------------------------------- known values

def test_ppl_bpb_consistency():
    assert perplexity(0.0) == 1.0
    assert perplexity(2.0) == pytest.approx(math.exp(2.0))
    # Doubling utf8_bytes halves bits/byte at fixed loss/tokens.
    b1 = bits_per_byte(1.0, tokens=100, utf8_bytes=100)
    b2 = bits_per_byte(1.0, tokens=100, utf8_bytes=200)
    assert b2 == pytest.approx(b1 / 2)
    assert b1 == pytest.approx(1.0 / math.log(2.0))
    with pytest.raises(ValueError):
        bits_per_byte(1.0, tokens=0, utf8_bytes=100)


def test_bleu_known_value():
    # hyp="hello world" vs ref="hello there": 1 unigram match / 2 unigrams, BP=1.
    assert bleu(["hello world"], ["hello there"], n=1) == pytest.approx(50.0, abs=1e-6)
    # Perfect match with >=4 words -> BLEU-4 = 100.
    line = "the quick brown fox jumps over the lazy dog"
    assert bleu([line], [line]) == pytest.approx(100.0, abs=1e-6)


def test_chrf_known_value():
    assert chrf(["नमस्ते दुनिया"], ["नमस्ते दुनिया"]) == pytest.approx(100.0, abs=1e-6)
    # Different script text scores much lower than identical text.
    assert chrf(["नमस्ते दुनिया"], ["hello world"]) < 30.0


def test_rouge_l_known_value():
    assert rouge_l(["the cat sat on the mat"], ["the cat sat on the mat"]) == pytest.approx(1.0)
    # Partial overlap has F1 strictly between 0 and 1.
    r = rouge_l(["the cat sat on the mat"], ["a cat sat on a mat"])
    assert 0.0 < r < 1.0
    assert rouge_l(["a b c"], ["x y z"]) == pytest.approx(0.0)


def test_bleu_devnagari_script_no_false_penalty():
    # Identical Devanagari text must score 100 under tokenize='none' — proves we
    # aren't applying English whitespace tokenization assumptions to Indic script.
    line = "भारत की राजधानी नई दिल्ली है"
    assert bleu([line], [line]) == pytest.approx(100.0, abs=1e-6)


def test_repetition_rate_known_value():
    assert repetition_rate(["a a a"], n=1) == pytest.approx(1 - 1 / 3)
    assert repetition_rate(["a a a"], n=2) == pytest.approx(1 - 1 / 2)
    assert repetition_rate(["a b c"], n=1) == pytest.approx(0.0)
    assert repetition_rate([], n=1) == 0.0
    with pytest.raises(ValueError):
        repetition_rate(["a"], n=0)


def test_distinct_n_known_value():
    texts = ["a b c", "a b d"]
    assert distinct_n(texts, 1) == pytest.approx(4 / 6)
    assert distinct_n(texts, 2) == pytest.approx(3 / 4)
    assert distinct_n(["zzz"], 2) == 0.0


# ---------------------------------------------------------------- robustness

def test_no_crash_on_empty_generation():
    """Edge-case prefixes (e.g. BOS-only) producing empty/whitespace output must not
    crash the metric pipeline — they should yield finite scores."""
    hyps = ["", " ", "\u2581", "नमस्ते"]
    refs = ["", "कुछ", "\u2581भारत", "नमस्ते"]
    for fn in (bleu, chrf, rouge_l):
        score = fn(hyps, refs)  # must not raise
        assert math.isfinite(score)
    assert repetition_rate(hyps, n=3) == 0.0
    assert distinct_n(hyps, n=2) >= 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        bleu(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        chrf(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        rouge_l(["a"], ["a", "b"])
