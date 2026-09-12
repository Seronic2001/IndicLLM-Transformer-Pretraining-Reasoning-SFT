"""Train and evaluate SentencePiece BPE & Unigram tokenizers with Indic enhancements.

Usage:
    python -m hindi.tokenizer.train_tokenizer --config configs/tokenizer_H.yaml \
        --corpus data/splits/train.txt --val data/splits/val.txt \
        --out-dir tokenizer --lang hindi

Outputs (in --out-dir):
    <lang>.model              SentencePiece model (resume/encode artifact)
    <lang>.vocab              piece -> id table, one per line
    tokenizer_stats.json      chosen model stats, fertility, unk-rate, rationale
    tokenizer_comparison.json full comparative metrics (BPE vs Unigram, 16K vs 32K)

Design decisions (locked in project specifications Section 3 Tokenizer):
  * SentencePiece with byte_fallback=True -> zero <unk> on any input.
  * 100% character coverage (character_coverage=1.0) -> zero dropped Indic glyphs/nuktas.
  * split_digits=True -> separates digits into single tokens, saving vocab slots.
  * split_by_unicode_script=True -> prevents cross-script token merges.
  * Automated comparison across candidate model types {bpe, unigram} and vocab sizes {16384, 32768}.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

try:
    import sentencepiece as spm
except ImportError:
    spm = None

# Make `model.tokenizer` style imports work when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .tokenizer import Tokenizer  # noqa: E402
except ImportError:
    try:
        from tokenizer.tokenizer import Tokenizer  # noqa: E402
    except ImportError:
        from hindi.tokenizer.tokenizer import Tokenizer  # noqa: E402

DEFAULT_VOCAB_CANDIDATES = (16384, 32768)
DEFAULT_MODEL_TYPES = ("bpe", "unigram")


def _load_config(path: Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _sample_lines(path: Path, n: int, seed: int) -> list[str]:
    """Deterministic sample of up to n non-empty lines from a corpus file."""
    random.seed(seed)
    lines = [ln.rstrip("\n") for ln in open(path, encoding="utf-8")]
    lines = [ln for ln in lines if ln.strip()]
    if len(lines) <= n:
        return lines
    return random.sample(lines, n)


def _fertility(tok: Tokenizer, lines: list[str]) -> tuple[float, float]:
    """(avg tokens per whitespace word, unk token rate) over sampled lines."""
    total_tokens = 0
    total_words = 0
    unk = 0
    for ln in lines:
        ids = tok.encode(ln)
        total_tokens += len(ids)
        total_words += max(1, len(ln.split()))
        unk += ids.count(tok.unk_id)
    if total_words == 0:
        return 0.0, 0.0
    return total_tokens / total_words, unk / max(1, total_tokens)


def evaluate_tokenizer(tok: Tokenizer, lines: list[str]) -> dict:
    """Comprehensive evaluation metrics on held-out text."""
    if not lines:
        return {
            "fertility_tokens_per_word": 0.0,
            "unk_rate": 0.0,
            "compression_ratio_chars_per_token": 0.0,
            "bytes_per_token": 0.0,
            "roundtrip_accuracy": 1.0,
        }

    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    unk_count = 0
    exact_roundtrip = 0

    for ln in lines:
        total_chars += len(ln)
        total_bytes += len(ln.encode("utf-8"))
        total_words += max(1, len(ln.split()))

        ids = tok.encode(ln)
        total_tokens += len(ids)
        unk_count += ids.count(tok.unk_id)

        try:
            decoded = tok.decode(ids)
            if decoded == ln:
                exact_roundtrip += 1
        except Exception:
            pass

    return {
        "fertility_tokens_per_word": round(total_tokens / max(1, total_words), 4),
        "unk_rate": round(unk_count / max(1, total_tokens), 6),
        "compression_ratio_chars_per_token": round(total_chars / max(1, total_tokens), 4),
        "bytes_per_token": round(total_bytes / max(1, total_tokens), 4),
        "roundtrip_accuracy": round(exact_roundtrip / max(1, len(lines)), 4),
        "total_eval_lines": len(lines),
    }


def train_tokenizer(
    corpus_path: Path,
    model_prefix: str,
    vocab_size: int,
    config: dict,
    model_type: Optional[str] = None,
    seed: int = 1337,
) -> dict:
    """Train a SentencePiece BPE or Unigram model and write .model + .vocab."""
    m_type = model_type or config.get("model_type", "bpe")
    params = {
        "input": str(corpus_path),
        "model_prefix": model_prefix,
        "model_type": m_type,
        "vocab_size": int(vocab_size),
        "byte_fallback": bool(config.get("byte_fallback", True)),
        "character_coverage": float(config.get("character_coverage", 1.0)),
        "split_digits": bool(config.get("split_digits", True)),
        "split_by_unicode_script": bool(config.get("split_by_unicode_script", True)),
        "split_by_whitespace": bool(config.get("split_by_whitespace", True)),
        "normalization_rule_name": str(config.get("normalization_rule_name", "nfkc")),
        "pad_id": int(config.get("pad_id", 0)),
        "unk_id": int(config.get("unk_id", 1)),
        "bos_id": int(config.get("bos_id", 2)),
        "eos_id": int(config.get("eos_id", 3)),
        "seed_sentencepiece_size": int(config.get("seed_sentencepiece_size", 1000000)),
        "num_threads": int(config.get("num_threads", os_cpu_count())),
        "hard_vocab_limit": False,  # allow slightly-smaller vocab if data is small
    }
    if config.get("input_sentence_size"):
        params["input_sentence_size"] = int(config["input_sentence_size"])
    if spm is not None:
        spm.SentencePieceTrainer.train(**params)
        sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
        vocab_size_actual = sp.get_piece_size()
        vocab_path = Path(model_prefix).with_suffix(".vocab")
        with open(vocab_path, "w", encoding="utf-8") as f:
            for i in range(vocab_size_actual):
                f.write(f"{sp.id_to_piece(i)}\t{i}\n")
    else:
        # Write dummy model and vocab files for offline testing
        model_path = Path(model_prefix).with_suffix(".model")
        model_path.write_bytes(b"dummy_model")
        vocab_path = Path(model_prefix).with_suffix(".vocab")
        vocab_size_actual = int(vocab_size)
        with open(vocab_path, "w", encoding="utf-8") as f:
            for i in range(vocab_size_actual):
                f.write(f"tok_{i}\t{i}\n")

    return {
        "model_path": str(Path(model_prefix).with_suffix(".model")),
        "vocab_path": str(vocab_path),
        "vocab_size": vocab_size_actual,
        "config_vocab_size": int(vocab_size),
        "model_type": m_type,
        "byte_fallback": params["byte_fallback"],
        "character_coverage": params["character_coverage"],
        "split_digits": params["split_digits"],
    }


def compare_and_decide_tokenizer(
    corpus_path: Path,
    val_path: Path,
    out_dir: Path,
    config: dict,
    candidate_vocab_sizes: tuple[int, ...] = DEFAULT_VOCAB_CANDIDATES,
    candidate_model_types: tuple[str, ...] = DEFAULT_MODEL_TYPES,
    seed: int = 1337,
) -> tuple[str, int, dict]:
    """Train BPE and Unigram candidates across candidate vocab sizes and compare on val.txt.

    Returns:
        (chosen_model_type, chosen_vocab_size, decision_dict)
    """
    lines = _sample_lines(val_path, n=2000, seed=seed)
    results: dict[str, dict] = {}

    for m_type in candidate_model_types:
        for vs in candidate_vocab_sizes:
            key = f"{m_type}_{vs}"
            prefix = str(out_dir / f"candidate_{key}")
            stats = train_tokenizer(corpus_path, prefix, vs, config, model_type=m_type, seed=seed)
            tok = Tokenizer(prefix + ".model")
            eval_stats = evaluate_tokenizer(tok, lines)
            stats.update(eval_stats)
            results[key] = stats

    # Decision Rule:
    # 1. Compare 16K vs 32K within the default/configured model_type
    target_type = config.get("model_type", "bpe")
    if target_type not in candidate_model_types:
        target_type = "bpe"

    key_32k = f"{target_type}_32768" if f"{target_type}_32768" in results else list(results.keys())[-1]
    key_16k = f"{target_type}_16384" if f"{target_type}_16384" in results else list(results.keys())[0]

    res_32k = results[key_32k]
    res_16k = results[key_16k]

    fertility_penalty = res_16k["fertility_tokens_per_word"] / max(1e-9, res_32k["fertility_tokens_per_word"])
    choose_small = (
        fertility_penalty <= 1.15
        and res_16k["unk_rate"] <= 1e-4
        and res_32k["unk_rate"] <= 1e-4
    )
    chosen_vs = 16384 if choose_small else 32768
    chosen_mtype = target_type
    chosen_key = f"{chosen_mtype}_{chosen_vs}"
    if chosen_key not in results:
        chosen_key = list(results.keys())[0]
        chosen_vs = results[chosen_key]["config_vocab_size"]
        chosen_mtype = results[chosen_key]["model_type"]

    rationale = (
        f"Selected {chosen_mtype.upper()} with {chosen_vs} vocab size: "
        f"fertility = {results[chosen_key]['fertility_tokens_per_word']} tokens/word, "
        f"compression = {results[chosen_key]['compression_ratio_chars_per_token']} chars/token, "
        f"roundtrip accuracy = {results[chosen_key]['roundtrip_accuracy']*100:.2f}%."
    )

    decision_record = {
        "candidates": results,
        "chosen_model_type": chosen_mtype,
        "chosen_vocab_size": chosen_vs,
        "chosen_key": chosen_key,
        "rationale": rationale,
        "note": (
            "16K choice requires model_L_16k.yaml (n_layer=8) to keep ~25M params; "
            "32K pairs with model_H.yaml / model_L.yaml (n_layer=6)."
        ),
    }

    # Copy chosen candidate to official location
    winning_model = out_dir / f"candidate_{chosen_key}.model"
    winning_vocab = out_dir / f"candidate_{chosen_key}.vocab"
    target_model = out_dir / f"{config.get('lang', 'hindi')}.model"
    target_vocab = out_dir / f"{config.get('lang', 'hindi')}.vocab"

    if winning_model.exists():
        shutil.copy(str(winning_model), str(target_model))
        shutil.copy(str(winning_vocab), str(target_vocab))

    return chosen_mtype, chosen_vs, decision_record


def decide_vocab_size(
    corpus_path: Path,
    val_path: Path,
    out_dir: Path,
    config: dict,
    candidates: tuple[int, ...] = DEFAULT_VOCAB_CANDIDATES,
    seed: int = 1337,
) -> tuple[int, dict]:
    """Backward-compatible wrapper for candidate evaluation."""
    target_type = config.get("model_type", "bpe")
    chosen_mtype, chosen_vs, decision = compare_and_decide_tokenizer(
        corpus_path, val_path, out_dir, config,
        candidate_vocab_sizes=candidates,
        candidate_model_types=(target_type,),
        seed=seed,
    )
    legacy_candidates = {}
    for k, v in decision["candidates"].items():
        vs_str = str(v.get("config_vocab_size", k.split("_")[-1]))
        legacy_candidates[vs_str] = v
    decision["candidates"] = legacy_candidates
    return chosen_vs, decision


def os_cpu_count() -> int:
    import os

    return os.cpu_count() or 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the language tokenizer with BPE/Unigram comparison")
    parser.add_argument("--config", default=None, help="tokenizer_<lang>.yaml path")
    parser.add_argument("--corpus", "--input-txt", dest="corpus", required=True, help="data/splits/train.txt")
    parser.add_argument("--val", default=None, help="data/splits/val.txt (for stats)")
    parser.add_argument("--out-dir", required=True, help="tokenizer/ output dir")
    parser.add_argument("--lang", default="hindi", help="hindi or assamese (names the .model)")
    parser.add_argument("--model-type", default=None, choices=["bpe", "unigram", "compare"], help="bpe, unigram, or compare both")
    parser.add_argument("--vocab-size", type=int, default=None, help="skip selection; force size")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    config_path = args.config
    if config_path is None:
        candidate_cfgs = [
            Path(f"hindi/configs/tokenizer_H.yaml"),
            Path(f"assamese/configs/tokenizer_L.yaml"),
            Path(f"configs/tokenizer_H.yaml"),
            Path(__file__).resolve().parents[1] / "configs" / "tokenizer_H.yaml",
        ]
        for c in candidate_cfgs:
            if c.exists():
                config_path = str(c)
                break
        if config_path is None:
            raise FileNotFoundError("Could not find tokenizer config. Please provide --config.")

    config = _load_config(Path(config_path))
    config["lang"] = args.lang
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)

    val_path = Path(args.val) if args.val else (corpus.parent / "val.txt")
    if not val_path.exists():
        val_path = corpus

    candidate_vs = (args.vocab_size,) if args.vocab_size else tuple(config.get("candidate_vocab_sizes", DEFAULT_VOCAB_CANDIDATES))
    
    if args.model_type and args.model_type != "compare":
        candidate_mtypes = (args.model_type,)
    else:
        candidate_mtypes = tuple(config.get("candidate_model_types", DEFAULT_MODEL_TYPES))

    print(f"[*] Training & Evaluating Tokenizer candidates across: Types={candidate_mtypes}, VocabSizes={candidate_vs}...")
    chosen_mtype, chosen_vs, decision = compare_and_decide_tokenizer(
        corpus, val_path, out_dir, config,
        candidate_vocab_sizes=candidate_vs,
        candidate_model_types=candidate_mtypes,
        seed=args.seed,
    )

    prefix = str(out_dir / args.lang)
    stats = train_tokenizer(corpus, prefix, chosen_vs, config, model_type=chosen_mtype, seed=args.seed)

    # Recompute final fertility/unk on the chosen model for the stats file.
    tok = Tokenizer(prefix + ".model")
    eval_metrics = evaluate_tokenizer(tok, _sample_lines(val_path, n=2000, seed=args.seed))
    stats.update(eval_metrics)
    stats.update(decision)

    # Save tokenizer_stats.json and tokenizer_comparison.json
    stats_path = out_dir / "tokenizer_stats.json"
    comp_path = out_dir / "tokenizer_comparison.json"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(decision["candidates"], f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"🎉 TOKENIZER TRAINING & COMPARISON COMPLETE ({args.lang.upper()})")
    print("=" * 60)
    print(f"  • Chosen Algorithm : {chosen_mtype.upper()}")
    print(f"  • Chosen Vocab Size: {chosen_vs}")
    print(f"  • Fertility Ratio  : {stats['fertility_tokens_per_word']} tokens/word")
    print(f"  • Compression      : {stats['compression_ratio_chars_per_token']} chars/token")
    print(f"  • UNK Rate         : {stats['unk_rate']}")
    print(f"  • Roundtrip Acc    : {stats['roundtrip_accuracy']*100:.2f}%")
    print(f"  • Artifacts        : {prefix}.model, {prefix}.vocab, {stats_path.name}, {comp_path.name}")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
