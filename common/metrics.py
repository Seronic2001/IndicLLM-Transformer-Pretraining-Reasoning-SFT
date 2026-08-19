"""Evaluation metrics — pure functions, no model code.

BLEU / chrF / ROUGE-L are delegated to the standard libraries (sacrebleu, rouge-score)
for correctness and reproducibility, pinned in ``requirements.txt``. For Devanagari /
Bengali-Assamese script we pass ``tokenize="none"`` to sacrebleu: whitespace-aware
tokenization is unreliable for Indic text, and chrF is character-based anyway.

All functions here are deterministic and free of model/checkpoint logic.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


def perplexity(loss: float) -> float:
    """Perplexity = exp(cross-entropy loss)."""
    return math.exp(float(loss))


def bits_per_byte(loss: float, tokens: int, utf8_bytes: int) -> float:
    """Bits per byte = loss / log(2) * tokens / utf8_bytes.

    Converts a per-token cross-entropy (nats) into bits per UTF-8 byte of the
    underlying source text, allowing cross-tokenizer / cross-language comparison.
    """
    if utf8_bytes <= 0:
        raise ValueError("utf8_bytes must be positive")
    nats_to_bits = 1.0 / math.log(2.0)
    return float(loss) * nats_to_bits * (float(tokens) / float(utf8_bytes))


def bleu_4(
    hypotheses: Sequence[str],
    references: Sequence[str | Sequence[str]],
    smooth_method: str = "exp",
) -> float:
    """Corpus-level BLEU-4 via sacrebleu with no whitespace-based tokenization.

    Returns the float BLEU score (0–100 scale, standard sacrebleu convention).
    """
    if sacrebleu is None:
        raise ImportError("sacrebleu is required for BLEU evaluation: pip install sacrebleu")
    if len(hypotheses) == 0:
        return 0.0
    if isinstance(references[0], str):
        refs_for_sacre = [[r for r in references]]  # type: ignore[list-item]
    else:
        max_refs = max(len(r) for r in references)  # type: ignore[arg-type]
        refs_for_sacre = [
            [r[i] if i < len(r) else "" for r in references]  # type: ignore[index]
            for i in range(max_refs)
        ]
    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        refs_for_sacre,
        smooth_method=smooth_method,
        tokenize="none",
    )
    return float(bleu.score)


def chrf(
    hypotheses: Sequence[str],
    references: Sequence[str | Sequence[str]],
    word_order: int = 0,
) -> float:
    """Corpus-level chrF (or chrF++ if word_order > 0) via sacrebleu.

    Returns the float chrF score (0–100 scale). Character n-gram metrics are
    well-matched to morphologically rich Indian languages where surface forms
    vary while roots match.
    """
    if sacrebleu is None:
        raise ImportError("sacrebleu is required for chrF evaluation: pip install sacrebleu")
    if len(hypotheses) == 0:
        return 0.0
    if isinstance(references[0], str):
        refs_for_sacre = [[r for r in references]]  # type: ignore[list-item]
    else:
        max_refs = max(len(r) for r in references)  # type: ignore[arg-type]
        refs_for_sacre = [
            [r[i] if i < len(r) else "" for r in references]  # type: ignore[index]
            for i in range(max_refs)
        ]
    score = sacrebleu.corpus_chrf(
        hypotheses,
        refs_for_sacre,
        word_order=word_order,
    )
    return float(score.score)


def rouge_l(
    hypotheses: Sequence[str],
    references: Sequence[str],
) -> float:
    """Mean ROUGE-L F1 score over (hyp, ref) pairs using google/rouge_score.

    Returns the mean F1 score on a 0–100 scale.
    """
    if rouge_scorer is None:
        raise ImportError("rouge-score is required for ROUGE-L: pip install rouge-score")
    if len(hypotheses) == 0:
        return 0.0
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    f1_sum = sum(
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    )
    return float((f1_sum / len(hypotheses)) * 100.0)


def repetition_rate(token_ids: Sequence[int], n: int = 4) -> float:
    """Fraction of n-grams in the sequence that are duplicates.

    Returns 0.0 for sequences shorter than n.
    """
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    counts = Counter(ngrams)
    duplicates = sum(c - 1 for c in counts.values() if c > 1)
    return float(duplicates / len(ngrams))


def distinct_n(token_ids: Sequence[int], n: int = 1) -> float:
    """Distinct-n = unique n-grams / total n-grams.

    Standard diversity diagnostic for generated text (Li et al., 2016).
    """
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return float(len(set(ngrams)) / len(ngrams))
