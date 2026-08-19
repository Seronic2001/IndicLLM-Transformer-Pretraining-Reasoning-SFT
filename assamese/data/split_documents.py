"""Document-level 98/1/1 split (Agent-A).

Splits are by DOCUMENT, never by line — a document never appears in more than one
split (test_no_document_leakage). Deterministic seeded shuffle.

Outputs (in --out-dir/splits):
  * {train,val,test}.jsonl   one record per line: {doc_id, source, text}
  * {train,val,test}.txt     raw text, one document per line (tokenizer input)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional


def load_doc_records(clean_dir: str) -> list[dict]:
    """Load every clean/<source>.jsonl into a flat record list."""
    records: list[dict] = []
    clean = Path(clean_dir)
    for path in sorted(clean.glob("*.jsonl")):
        if path.name.startswith("manifest") or path.name.startswith("dataset_stats"):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and "text" in rec and rec["text"].strip():
                            records.append(rec)
                    except Exception:
                        continue
    return records


def split_records(
    records: list[dict],
    ratios: tuple[float, float, float] = (0.98, 0.01, 0.01),
    seed: int = 1337,
) -> dict[str, list[dict]]:
    """Deterministic document-level split."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_splits(
    splits: dict[str, list[dict]], out_dir: str, txt_only: bool = False
 ) -> dict[str, int]:
    p = Path(out_dir)
    out = p if p.name == "splits" else p / "splits"
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, records in splits.items():
        if txt_only:
            with open(out / f"{name}.txt", "w", encoding="utf-8") as f_txt:
                for r in records:
                    text = r.get("text", "").strip()
                    if text:
                        f_txt.write(text.replace("\n", " ") + "\n")
        else:
            with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f_json, open(
                out / f"{name}.txt", "w", encoding="utf-8"
            ) as f_txt:
                for r in records:
                    text = r.get("text", "").strip()
                    if text:
                        f_json.write(json.dumps(r, ensure_ascii=False) + "\n")
                        f_txt.write(text.replace("\n", " ") + "\n")
        counts[name] = len(records)
    return counts


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Document-level train/val/test split (Agent-A)")
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--txt-only",
        action="store_true",
        help="write only {train,val,test}.txt to avoid duplicate JSONL disk usage",
    )
    args = parser.parse_args(argv)
    records = load_doc_records(args.clean_dir)
    if not records:
        print("no documents found in clean dir", file=sys.stderr)
        return 1
    splits = split_records(records, seed=args.seed)
    counts = write_splits(splits, args.out_dir, txt_only=args.txt_only)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
