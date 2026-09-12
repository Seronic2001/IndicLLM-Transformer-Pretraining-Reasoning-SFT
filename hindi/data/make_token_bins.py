"""Convert splits into flat token-id .bin files (Agent-A output contract).

Reads splits/{train,val,test}.jsonl (or .txt) and the trained tokenizer, writes
``{train,val,test}.bin`` as raw uint16 arrays (vocab < 65536) — the exact input
the Agent-D ``TokenDataset`` consumes via numpy.memmap.

Resumable: skips a split whose .bin already exists unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union, Sequence

import numpy as np


def tokenize_to_bin(
    src: Union[Path, str, Sequence[str]],
    tokenizer,
    out_path: str,
    batch_size: int = 10000,
) -> int:
    """Stream documents from JSONL/TXT file or list of texts directly into binary token file."""
    import time
    t0 = time.time()
    total_docs = 0
    total_tokens = 0
    log_interval = 50000

    def _iter_texts():
        nonlocal total_docs
        if isinstance(src, (list, tuple)):
            for doc in src:
                doc_str = doc.strip() if isinstance(doc, str) else str(doc).strip()
                if doc_str:
                    total_docs += 1
                    yield doc_str
        else:
            p = Path(src)
            with open(p, "r", encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    if p.suffix == ".jsonl":
                        try:
                            text = json.loads(line).get("text", "").strip()
                        except Exception:
                            continue
                    else:
                        text = line
                    if text:
                        total_docs += 1
                        yield text

    with open(out_path, "wb") as f_out:
        batch: list[str] = []
        for text in _iter_texts():
            batch.append(text)
            if len(batch) >= batch_size:
                if hasattr(tokenizer, "sp"):
                    encoded_batch = tokenizer.sp.encode(batch, out_type=int)
                    batch_ids = [tid for doc_ids in encoded_batch for tid in doc_ids]
                else:
                    batch_ids = [tid for doc in batch for tid in tokenizer.encode(doc)]
                if batch_ids:
                    arr = np.array(batch_ids, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                    total_tokens += int(arr.size)
                batch = []
                if total_docs % log_interval == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[{Path(out_path).name}] Processed {total_docs:,} docs "
                        f"| {total_tokens:,} tokens | {total_docs / max(elapsed, 0.001):.0f} docs/sec",
                        flush=True,
                    )

        if batch:
            if hasattr(tokenizer, "sp"):
                encoded_batch = tokenizer.sp.encode(batch, out_type=int)
                batch_ids = [tid for doc_ids in encoded_batch for tid in doc_ids]
            else:
                batch_ids = [tid for doc in batch for tid in tokenizer.encode(doc)]
            if batch_ids:
                arr = np.array(batch_ids, dtype=np.uint16)
                f_out.write(arr.tobytes())
                total_tokens += int(arr.size)

    elapsed = time.time() - t0
    print(
        f"[{Path(out_path).name}] FINISHED: {total_docs:,} docs "
        f"| {total_tokens:,} tokens in {elapsed:.1f}s ({total_docs / max(elapsed, 0.001):.0f} docs/sec)",
        flush=True,
    )
    if total_tokens == 0:
        raise ValueError(f"tokenized output for {out_path} is empty")
    return total_tokens


tokenize_file_to_bin = tokenize_to_bin


def tokenize_clean_dir_to_bins(
    clean_dir: Path,
    tokenizer,
    out_dir: Path,
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    batch_size: int = 10000,
    seed: int = 1337,
) -> dict[str, int]:
    """Single-pass streaming directly from clean/*.jsonl into train.bin, val.bin, test.bin.

    Zero intermediate disk files created.
    """
    import random
    import time
    rng = random.Random(seed)
    t0 = time.time()

    train_bin_path = out_dir / "train.bin"
    val_bin_path = out_dir / "val.bin"
    test_bin_path = out_dir / "test.bin"

    train_f = open(train_bin_path, "wb")
    val_f = open(val_bin_path, "wb")
    test_f = open(test_bin_path, "wb")

    batches: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    token_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    handles = {"train": train_f, "val": val_f, "test": test_f}

    def _flush_batch(split: str) -> None:
        b = batches[split]
        if not b:
            return
        if hasattr(tokenizer, "sp"):
            encoded_batch = tokenizer.sp.encode(b, out_type=int)
            batch_ids = [tid for doc_ids in encoded_batch for tid in doc_ids]
        else:
            batch_ids = [tid for doc in b for tid in tokenizer.encode(doc)]
        if batch_ids:
            arr = np.array(batch_ids, dtype=np.uint16)
            handles[split].write(arr.tobytes())
            token_counts[split] += int(arr.size)
        batches[split] = []

    total_docs = 0
    log_interval = 50000

    EXCLUDED_CORPUS_FILES = {
        "train.jsonl", "val.jsonl", "test.jsonl",
        "clean__train.jsonl", "clean__val.jsonl", "clean__test.jsonl",
        "reasoning.jsonl", "train_reasoning.jsonl", "val_reasoning.jsonl", "test_reasoning.jsonl",
    }
    try:
        for cfile in sorted(clean_dir.glob("*.jsonl")):
            if cfile.name in EXCLUDED_CORPUS_FILES or "reasoning" in cfile.name.lower():
                print(f"[*] Skipping non-pretraining / reasoning benchmark file: '{cfile.name}'", flush=True)
                continue
            print(f"[*] Streaming & encoding '{cfile.name}'...", flush=True)
            with open(cfile, "r", encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "").strip()
                    except Exception:
                        continue
                    if not text:
                        continue

                    r = rng.random()
                    if r < train_ratio:
                        split = "train"
                    elif r < (train_ratio + val_ratio):
                        split = "val"
                    else:
                        split = "test"

                    batches[split].append(text)
                    counts[split] += 1
                    total_docs += 1

                    if len(batches[split]) >= batch_size:
                        _flush_batch(split)

                    if total_docs % log_interval == 0:
                        elapsed = time.time() - t0
                        print(
                            f"[StreamTokenBins] Processed {total_docs:,} docs "
                            f"(train: {token_counts['train']:,} tokens, val: {token_counts['val']:,}, test: {token_counts['test']:,}) "
                            f"| {total_docs / max(elapsed, 0.001):.0f} docs/sec",
                            flush=True,
                        )

        for split in ("train", "val", "test"):
            _flush_batch(split)

    finally:
        train_f.close()
        val_f.close()
        test_f.close()

    elapsed = time.time() - t0
    print(
        f"[StreamTokenBins] COMPLETE: {total_docs:,} docs encoded in {elapsed:.1f}s "
        f"(train.bin: {token_counts['train']:,} tokens, val.bin: {token_counts['val']:,} tokens, test.bin: {token_counts['test']:,} tokens)",
        flush=True,
    )
    return token_counts


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Make train/val/test .bin files")
    parser.add_argument("--tokenizer", required=True, help="path to <lang>.model")
    parser.add_argument("--splits-dir", help="data/splits/")
    parser.add_argument("--clean-dir", help="data/clean/ (direct single-pass streaming without intermediate splits)")
    parser.add_argument("--out-dir", required=True, help="data/")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from tokenizer.tokenizer import Tokenizer  # noqa: E402
    except ImportError:
        from hindi.tokenizer.tokenizer import Tokenizer  # noqa: E402

    tok = Tokenizer(args.tokenizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clean_dir:
        clean_dir = Path(args.clean_dir)
        tokenize_clean_dir_to_bins(clean_dir, tok, out_dir)
        return 0

    if not args.splits_dir:
        print("Error: Either --clean-dir or --splits-dir must be provided.", file=sys.stderr)
        return 1

    splits_dir = Path(args.splits_dir)
    totals = {}
    for split in ("train", "val", "test"):
        jsonl = splits_dir / f"{split}.jsonl"
        txt = splits_dir / f"{split}.txt"
        src = jsonl if jsonl.exists() else txt
        if not src.exists():
            print(f"[skip] {split}: no input ({src})", file=sys.stderr)
            continue
        dest = out_dir / f"{split}.bin"
        if dest.exists() and not args.force:
            print(f"[skip] {split}: {dest} already exists (use --force to rebuild)")
            continue
        n = tokenize_file_to_bin(src, tok, str(dest))
        totals[split] = n
        print(f"[ok] {split}: {n:,} tokens -> {dest}")

    if not totals:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
