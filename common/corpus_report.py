"""Phase 1 corpus and tokenizer statistics, recomputed from the published artifacts.

usage: python -m common.corpus_report --lang hindi --artifacts <dir> --tokenizer hindi/tokenizer/hindi.model \
           --out corpus_report_hindi.json

<dir> is an artifact dataset (shubhadeepmandal/lma-<lang>-artifact[s]): clean/<source>.jsonl,
{train,val,test}.bin and tokenizer_candidates/*.model. Nothing is estimated:

  * per source: documents, words, characters, UTF-8 bytes and exact token counts with the
    chosen tokenizer, split into train/val/test by replaying make_token_bins' assignment
    (one Random(1337) draw per non-empty document, files in sorted order, 98/1/1);
  * bins: tokens and documents (EOS count) of each .bin, and whether they equal the replayed
    totals, which shows the clean files are the ones the bins were built from;
  * quality: share of letters in the language's script block and, for the Bengali block,
    the share of Bengali র (U+09B0) among র + Assamese ৰ (U+09F0), which flags Bengali text;
  * tokenizers: each candidate scored on held-out documents (the val split).

Manual vs downloaded follows SOURCE_KINDS below: "manual" is text this project collected
itself (its own crawlers of primary websites, its own PDF extraction); a published corpus
or dump is "downloaded", even when the project's code fetched and cleaned it.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import unicodedata
from pathlib import Path

import numpy as np
import sentencepiece as spm

SEED, TRAIN_RATIO, VAL_RATIO = 1337, 0.98, 0.01  # make_token_bins defaults
BATCH = 10_000
N_TOKENIZER_DOCS = 2000
SCRIPT_BLOCKS = {"hindi": (0x0900, 0x097F), "assamese": (0x0980, 0x09FF)}
BENGALI_RA, ASSAMESE_RA = "র", "ৰ"

_CRAWL, _PDF = "project crawler of the site", "project PDF text extraction"
SOURCE_KINDS = {
    "hindi": {
        "crawler_amarujala": ("manual", _CRAWL),
        "crawler_bbc_hi": ("manual", _CRAWL),
        "crawler_gadyakosh": ("manual", "project crawler (MediaWiki API of gadyakosh.org)"),
        "crawler_kavitakosh": ("manual", "project crawler (MediaWiki API of kavitakosh.org)"),
        "crawler_vikaspedia_hi": ("manual", _CRAWL),
        "ncert_pdfs": ("manual", _PDF),
        "crawler_cc100_hi": ("downloaded", "CC-100 dump (data.statmt.org)"),
        "oscar_hindi": ("downloaded", "OSCAR (Kaggle dataset)"),
        "sangraha_verified": ("downloaded", "AI4Bharat Sangraha"),
        "sangraha_unverified": ("downloaded", "AI4Bharat Sangraha"),
        "wikipedia_hindi": ("downloaded", "Wikipedia articles (Kaggle dataset)"),
        "generated_samples": ("excluded", "model generations, no text field"),
    },
    "assamese": {
        "crawler_agradoot": ("manual", _CRAWL),
        "crawler_asomiya_pratidin": ("manual", _CRAWL),
        "crawler_nenow_asm": ("manual", _CRAWL),
        "crawler_niyomiya_barta": ("manual", _CRAWL),
        "crawler_vikaspedia_as": ("manual", _CRAWL),
        "crawler_xahitya_literary": ("manual", _CRAWL),
        "ncert_assamese_pdfs": ("manual", _PDF),
        "scert_pdfs": ("manual", _PDF),
        "seba_pdfs": ("manual", _PDF),
        "asm_corpus": ("downloaded", "published Assamese corpus"),
        "crawler_cc100_as": ("downloaded", "CC-100 dump (data.statmt.org)"),
        "indiccorp_v2": ("downloaded", "AI4Bharat IndicCorp v2"),
        "news_portals": ("downloaded", "Assamese news articles (Kaggle dataset)"),
        "sangraha_verified": ("downloaded", "AI4Bharat Sangraha"),
        "sangraha_unverified": ("downloaded", "AI4Bharat Sangraha"),
        "wikipedia_assamese": ("downloaded", "Wikipedia articles"),
        "wikipedia_dump_manual": ("downloaded", "Wikimedia dump, parsed by project code"),
        "generated_samples": ("excluded", "model generations, no text field"),
    },
}


def iter_texts(path: Path):
    """Document texts of one clean/<source>.jsonl, filtered exactly as make_token_bins does."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "").strip()
            except Exception:
                continue
            if text:
                yield text


def script_purity(texts: list[str], block: tuple[int, int]) -> float | None:
    """Share of letters (Unicode L*/M* categories) inside the script block."""
    in_block = letters = 0
    for text in texts:
        for ch in text:
            if unicodedata.category(ch)[0] in "LM":
                letters += 1
                in_block += block[0] <= ord(ch) <= block[1]
    return round(in_block / letters, 4) if letters else None


def scan_sources(clean_dir: Path, sp, lang: str) -> tuple[list[dict], list[str]]:
    """Per-source counts, plus the val-split texts (held-out documents for the tokenizers)."""
    rng = random.Random(SEED)
    kinds = SOURCE_KINDS[lang]
    rows, val_texts = [], []
    for path in sorted(clean_dir.glob("*.jsonl")):
        name = path.stem
        if name in ("train", "val", "test") or "reasoning" in name.lower():
            continue  # make_token_bins skips these too
        start = time.time()
        kind, how = kinds.get(name, ("unclassified", "not in SOURCE_KINDS"))
        row = {"name": name, "kind": kind, "how": how, "docs": 0, "words": 0, "chars": 0,
               "utf8_bytes": 0, "tokens": 0, "bengali_ra": 0, "assamese_ra": 0,
               "split_docs": {"train": 0, "val": 0, "test": 0},
               "split_tokens": {"train": 0, "val": 0, "test": 0}}
        source_val, batch = [], []

        def flush():
            for text, split, ids in zip([b[0] for b in batch], [b[1] for b in batch],
                                        sp.encode([b[0] for b in batch], out_type=int)):
                row["tokens"] += len(ids)
                row["split_tokens"][split] += len(ids)
            batch.clear()

        for text in iter_texts(path):
            r = rng.random()
            split = "train" if r < TRAIN_RATIO else "val" if r < TRAIN_RATIO + VAL_RATIO else "test"
            row["docs"] += 1
            row["split_docs"][split] += 1
            row["words"] += len(text.split())
            row["chars"] += len(text)
            row["utf8_bytes"] += len(text.encode("utf-8"))
            row["bengali_ra"] += text.count(BENGALI_RA)
            row["assamese_ra"] += text.count(ASSAMESE_RA)
            if split == "val":
                source_val.append(text)
            batch.append((text, split))
            if len(batch) >= BATCH:
                flush()
        flush()
        ra = row["bengali_ra"] + row["assamese_ra"]
        row["bengali_ra_share"] = round(row["bengali_ra"] / ra, 4) if ra else None
        row["script_purity_val"] = script_purity(source_val, SCRIPT_BLOCKS[lang])
        row["tokens_per_word"] = round(row["tokens"] / row["words"], 4) if row["words"] else None
        val_texts += source_val
        rows.append(row)
        print(f"  {name:28s} {kind:12s} docs={row['docs']:>9,} tokens={row['tokens']:>12,} "
              f"({time.time() - start:.0f}s)", flush=True)
    return rows, val_texts


def bin_counts(path: Path, eos_id: int) -> dict:
    ids = np.memmap(path, dtype=np.uint16, mode="r")
    return {"tokens": int(ids.size), "docs": int(np.count_nonzero(ids == eos_id))}


def evaluate_tokenizer(sp, texts: list[str]) -> dict:
    """The metrics of <lang>.tokenizer.train_tokenizer.evaluate_tokenizer, on documents."""
    tokens = words = chars = nbytes = unk = exact = 0
    for text, ids in zip(texts, sp.encode(texts, out_type=int)):
        tokens += len(ids)
        words += max(1, len(text.split()))
        chars += len(text)
        nbytes += len(text.encode("utf-8"))
        unk += ids.count(sp.unk_id())
        exact += sp.decode(ids) == text
    return {
        "vocab_size": sp.get_piece_size(),
        "fertility_tokens_per_word": round(tokens / words, 4),
        "compression_ratio_chars_per_token": round(chars / tokens, 4),
        "bytes_per_token": round(nbytes / tokens, 4),
        "unk_rate": round(unk / tokens, 6),
        "roundtrip_accuracy": round(exact / len(texts), 4),
        "n_docs": len(texts),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lang", choices=sorted(SOURCE_KINDS), required=True)
    parser.add_argument("--artifacts", required=True, help="artifact dataset root")
    parser.add_argument("--tokenizer", required=True, help="the chosen <lang>.model")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    root = Path(args.artifacts)
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    print(f"[{args.lang}] scanning {root / 'clean'}", flush=True)
    sources, val_texts = scan_sources(root / "clean", sp, args.lang)

    totals = {k: sum(s[k] for s in sources) for k in ("docs", "words", "chars", "utf8_bytes", "tokens")}
    by_kind = {}
    for s in sources:
        k = by_kind.setdefault(s["kind"], {"docs": 0, "tokens": 0})
        k["docs"] += s["docs"]
        k["tokens"] += s["tokens"]
    for k in by_kind.values():
        k["token_share"] = round(k["tokens"] / totals["tokens"], 4)

    bins = {}
    for split in ("train", "val", "test"):
        b = bin_counts(root / f"{split}.bin", sp.eos_id())
        replay_docs = sum(s["split_docs"][split] for s in sources)
        replay_tokens = sum(s["split_tokens"][split] for s in sources) + replay_docs  # + one EOS per doc
        b.update(replay_tokens=replay_tokens, replay_docs=replay_docs,
                 matches=b["tokens"] == replay_tokens and b["docs"] == replay_docs)
        bins[split] = b
        print(f"  {split}.bin tokens={b['tokens']:,} replay={replay_tokens:,} match={b['matches']}", flush=True)

    sample = random.Random(SEED).sample(val_texts, min(N_TOKENIZER_DOCS, len(val_texts)))
    tokenizers = {"chosen": evaluate_tokenizer(sp, sample)}
    for model in sorted((root / "tokenizer_candidates").glob("*.model")):
        tokenizers[model.stem.replace("candidate_", "")] = evaluate_tokenizer(
            spm.SentencePieceProcessor(model_file=str(model)), sample)

    report = {
        "lang": args.lang,
        "classification_rule": __doc__.split("Manual vs downloaded follows SOURCE_KINDS below: ")[1].strip(),
        "totals": totals, "by_kind": by_kind, "sources": sources, "bins": bins,
        "tokenizers": tokenizers,
        "split_rule": f"Random({SEED}) per document, {TRAIN_RATIO}/{VAL_RATIO}/rest",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"by_kind": by_kind, "tokenizers": tokenizers}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
