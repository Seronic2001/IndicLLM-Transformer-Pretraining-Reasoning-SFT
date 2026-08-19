"""Evaluation metrics — pure functions, no model code .

BLEU / chrF / ROUGE-L are delegated to the standard libraries (sacrebleu, rouge-score)
for correctness and reproducibility, pinned in ``requirements.txt``. For Devanagari /
Bengali-Assamese script we pass ``tokenize="none"`` to sacrebleu: whitespace-aware
tokenization is unreliable for Indic text, and chrF is character-based anyway.

All functions here are deterministic and free of model/checkpoint logic so they can be
unit-tested against hand-computed values.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import sacrebleu
from rouge_score import rouge_scorer


def perplexity(loss: float) -> float:
    """Perplexity = exp(cross-entropy loss)."""
    return math.exp(float(loss))


def bits_per_byte(loss: float, tokens: int, utf8_bytes: int) -> float:
    """Bits per byte = loss / log(2) * tokens / utf8_bytes.

    Converts a per-token cross-entropy (nats) into bits per UTF-8 byte of the
    underlying text, which is comparable across tokenizers with different
    token/byte ratios.
    """
    if utf8_bytes <= 0:
        raise ValueError("utf8_bytes must be positive")
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    nats_per_byte = float(loss) * float(tokens) / float(utf8_bytes)
    return nats_per_byte / math.log(2.0)


def _as_lines(texts: Sequence[str]) -> list[str]:
    return [str(t) for t in texts]


def bleu(hyps: Sequence[str], refs: Sequence[str], n: int = 4) -> float:
    """Corpus-level BLEU-n via sacrebleu, tokenize='none' (Indic-script safe).

    ``refs`` must be the same length as ``hyps`` (one reference per hypothesis,
    passed as a list of lists for sacrebleu's multi-ref interface).
    """
    if len(hyps) != len(refs):
        raise ValueError("hyps and refs must be same length")
    hyps = _as_lines(hyps)
    refs_multi = [[r] for r in _as_lines(refs)]
    metric = sacrebleu.metrics.bleu.BLEU(
        tokenize="none", max_ngram_order=n
    )
    return float(metric.corpus_score(hyps, refs_multi).score)


def chrf(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Corpus-level chrF++ via sacrebleu (character n-gram F-score, tokenize='none')."""
    if len(hyps) != len(refs):
        raise ValueError("hyps and refs must be same length")
    hyps = _as_lines(hyps)
    refs_multi = [[r] for r in _as_lines(refs)]
    metric = sacrebleu.metrics.chrf.CHRF(char_order=6, word_order=2)  # chrF++
    return float(metric.corpus_score(hyps, refs_multi).score)


def rouge_l(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Corpus-level ROUGE-L F1 via rouge-score (longest common subsequence based).

    Computed per pair and macro-averaged. rouge-score's default tokenizer splits on
    whitespace/punctuation and lowercases; both Hindi and Assamese use spaces between
    words, so this is a reasonable baseline and is documented as such in the report.
    """
    if len(hyps) != len(refs):
        raise ValueError("hyps and refs must be same length")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(str(h), str(r))["rougeL"].fmeasure
        for h, r in zip(hyps, refs)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenization shared by the n-gram metrics (Indic scripts space-separate)."""
    return [t for t in text.split() if t]


def _ngrams(tokens: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def repetition_rate(texts: Sequence[str], n: int = 3) -> float:
    """Fraction of n-grams that are repeats of an earlier occurrence, averaged per text.

    For each text: 1 - (unique n-grams / total n-grams), then macro-averaged over
    texts. 0.0 = fully novel text; 1.0 = everything is a repeat.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not texts:
        return 0.0
    rates = []
    for text in texts:
        grams = list(_ngrams(_tokenize(text), n))
        if not grams:
            continue
        unique = len(set(grams))
        rates.append(1.0 - unique / len(grams))
    return sum(rates) / len(rates) if rates else 0.0


def distinct_n(texts: Sequence[str], n: int) -> float:
    """Corpus-level distinct-n: unique n-grams / total n-grams, pooled across texts."""
    if n < 1:
        raise ValueError("n must be >= 1")
    counter: Counter = Counter()
    for text in texts:
        counter.update(_ngrams(_tokenize(text), n))
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return len(counter) / total
