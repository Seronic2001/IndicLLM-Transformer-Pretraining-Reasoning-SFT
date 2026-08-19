"""MinHash near-duplicate detection — reusable, invoked separately per language.

Used by the data pipeline to flag near-duplicate documents that exact-hash dedup
would miss (e.g. identical articles with slightly different whitespace/punctuation).
"""

from __future__ import annotations

import random
import zlib
from typing import Iterable, Sequence

import numpy as np

# Deterministic 64-bit universal hashing parameters
_RNG = random.Random(1337)
_A = np.array([_RNG.getrandbits(63) | 1 for _ in range(64)], dtype=np.uint64)
_B = np.array([_RNG.getrandbits(64) for _ in range(64)], dtype=np.uint64)


def shingles(text: str, k: int = 8) -> list[str]:
    """Character k-shingles (no whitespace removal — cheap and effective)."""
    text = " ".join(text.split())
    if len(text) < k:
        return [text]
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def minhash_signature(text: str, n_hashes: int = 16, k: int = 8) -> list[int]:
    """MinHash signature: min of each 64-bit universal hash function over all shingles."""
    sh = shingles(text, k)
    if not sh:
        return [0] * n_hashes
    crcs = np.array([zlib.crc32(s.encode("utf-8")) for s in sh], dtype=np.uint64)
    # (n_hashes, 1) * (1, N) + (n_hashes, 1) using 64-bit uint64 wrapping arithmetic
    hashes = _A[:n_hashes, None] * crcs[None, :] + _B[:n_hashes, None]
    return np.min(hashes, axis=1).tolist()


def jaccard_estimate(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not sig_a or len(sig_a) != len(sig_b):
        return 0.0
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


def find_near_duplicates(
    docs: Iterable[tuple[str, str]],
    threshold: float = 0.6,
    n_hashes: int = 16,
    k: int = 8,
    n_bands: int = 4,
    max_bucket_size: int = 200,
) -> list[tuple[str, str, float]]:
    """Return [(doc_id_a, doc_id_b, similarity)] pairs above threshold.

    Uses LSH (Locality Sensitive Hashing) band indexing with bucket size capping
    for fast O(N) candidate filtering at scale, preventing memory blowup on large corpora.
    """
    entries = [(doc_id, minhash_signature(text, n_hashes, k)) for doc_id, text in docs]
    if not entries:
        return []

    # Fast path for small document sets (< 100 docs)
    if len(entries) < 100:
        pairs: list[tuple[str, str, float]] = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                sim = jaccard_estimate(entries[i][1], entries[j][1])
                if sim >= threshold:
                    pairs.append((entries[i][0], entries[j][0], sim))
        return pairs

    # LSH Banding for large document collections
    rows_per_band = max(1, n_hashes // n_bands)
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    for idx, (_doc_id, sig) in enumerate(entries):
        for band_idx in range(n_bands):
            start = band_idx * rows_per_band
            band_key = (band_idx, tuple(sig[start : start + rows_per_band]))
            buckets.setdefault(band_key, []).append(idx)

    pairs: list[tuple[str, str, float]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for doc_indices in buckets.values():
        if 1 < len(doc_indices) <= max_bucket_size:
            for i in range(len(doc_indices)):
                for j in range(i + 1, len(doc_indices)):
                    u, v = doc_indices[i], doc_indices[j]
                    if u > v:
                        u, v = v, u
                    if (u, v) not in seen_pairs:
                        seen_pairs.add((u, v))
                        sim = jaccard_estimate(entries[u][1], entries[v][1])
                        if sim >= threshold:
                            pairs.append((entries[u][0], entries[v][0], sim))

    return pairs
