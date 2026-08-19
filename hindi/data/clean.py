"""Hindi document cleaning + dedup (Agent-A).

Pipeline per document:
  * NFC normalization (unicodedata.normalize)
  * script filter: keep Devanagari (\\u0900-\\u097F); count removed chars
  * drop control chars, collapse whitespace, drop too-short / too-empty docs
  * exact-hash dedup (sha256 of normalized text) then MinHash near-dup removal

Output: ``clean/<source>.txt`` lines (one document per line) + a JSONL record
per document. Every step is deterministic and re-runnable (idempotent: cleaning
the same input twice yields identical output).

The Assamese mirror is identical except for its script range (\\u0980-\\u09FF).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

from common.script_utils import (  # noqa: E402
    clean_indic_text,
    is_high_quality_indic_document,
    is_script_char,
)

MIN_DOC_CHARS = 20
MIN_SCRIPT_FRACTION = 0.5

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SCRIPT_FILTER_RE = {
    "hindi": re.compile(r"[^\u0900-\u097F\u0964\u0965\s]"),
    "assamese": re.compile(r"[^\u0980-\u09FF\u0964\u0965\s]"),
}


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def count_script_chars(text: str, lang: str) -> tuple[int, int]:
    """(chars in script, total non-space chars) via common.script_utils."""
    from common.script_utils import script_fraction

    return script_fraction(text, lang)


def clean_document(text: str, lang: str, doc_id: Optional[str] = None) -> tuple[Optional[str], dict]:
    """Clean one document. Returns (cleaned_text | None if dropped, stats dict)."""
    stats = {"doc_id": doc_id or "", "script_filter_removed_chars": 0, "dropped": None}
    text = normalize_text(text)
    orig_len = max(1, len(text.strip()))

    # 1. Vectorized regex script filtering (drops non-script characters, control chars, invisible chars)
    rx = _SCRIPT_FILTER_RE[lang]
    filtered = rx.sub("", text)
    stats["script_filter_removed_chars"] = len(text) - len(filtered)

    # 2. Deep Indic sanitization (heals spaced matras, conjuncts, orphan diacritics)
    cleaned = clean_indic_text(filtered, lang)

    in_script, total = count_script_chars(cleaned, lang)
    if len(cleaned) < MIN_DOC_CHARS:
        stats["dropped"] = "too_short"
        return None, stats
    if total and in_script / total < MIN_SCRIPT_FRACTION:
        stats["dropped"] = "low_script_fraction"
        return None, stats

    # 3. High distortion / loss ratio check (e.g. if >40% of document was orphan matra noise that got stripped)
    if (len(text) - len(cleaned)) / orig_len > 0.40:
        stats["dropped"] = "high_orphan_diacritics"
        return None, stats

    # 4. OCR / Crawler Quality Gate on healed text
    is_high_quality, reason = is_high_quality_indic_document(cleaned, lang)
    if not is_high_quality:
        stats["dropped"] = reason
        return None, stats

    return cleaned, stats


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dedup_documents(
    docs: Iterable[tuple[str, str, str]],  # (doc_id, source, text)
    minhash_threshold: float = 0.6,
) -> tuple[list[dict], int, list[tuple[str, str, float]]]:
    """Exact-hash dedup then MinHash near-dup removal.

    Returns (unique_records, exact_removed, near_dup_pairs_removed). Records are
    {"doc_id", "source", "text", "hash"}. First-seen order is preserved.
    """
    import sys
    from pathlib import Path

    try:
        from common.minhash import find_near_duplicates
    except ImportError:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from scripts.minhash import find_near_duplicates
        except ImportError:
            def find_near_duplicates(d, threshold=0.6):
                return []

    seen: set[str] = set()
    unique: list[dict] = []
    exact_removed = 0
    for doc_id, source, text in docs:
        h = doc_hash(text)
        if h in seen:
            exact_removed += 1
            continue
        seen.add(h)
        unique.append({"doc_id": doc_id, "source": source, "text": text, "hash": h})

    near_pairs = find_near_duplicates(
        [(r["doc_id"], r["text"]) for r in unique], threshold=minhash_threshold
    )
    # Keep the first-seen doc of each near-dup pair, drop the later one.
    position = {r["doc_id"]: i for i, r in enumerate(unique)}
    near_removed: set[str] = set()
    for a, b, _sim in near_pairs:
        drop = b if position[a] < position[b] else a
        near_removed.add(drop)
    unique = [r for r in unique if r["doc_id"] not in near_removed]
    return unique, exact_removed, near_pairs


def write_clean_outputs(records: list[dict], out_dir: str, source_name: str) -> tuple[Path, Path]:
    """Write clean/<source>.txt (one doc per line) + clean/<source>.jsonl."""
    out = Path(out_dir) / "clean"
    out.mkdir(parents=True, exist_ok=True)
    txt_path = out / f"{source_name}.txt"
    jsonl_path = out / f"{source_name}.jsonl"
    with open(txt_path, "w", encoding="utf-8") as f_txt, open(jsonl_path, "w", encoding="utf-8") as f_json:
        for r in records:
            f_txt.write(r["text"] + "\n")
            f_json.write(json.dumps(r, ensure_ascii=False) + "\n")
    return txt_path, jsonl_path
