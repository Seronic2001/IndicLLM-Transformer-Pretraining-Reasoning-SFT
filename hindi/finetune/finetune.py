"""Hindi finetuning pipeline.

Supports both Direct SFT and Chain-of-Thought (CoT) Fine-Tuning across
both Version 1.0 (Baseline) and Version 2.0 (Modern Transformer) architectures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

import yaml
import torch
import numpy as np

_LANG_ROOT = Path(__file__).resolve().parents[1]
if str(_LANG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LANG_ROOT))

from common.checkpoint import save_checkpoint, load_checkpoint
from tokenizer.tokenizer import Tokenizer

FINETUNE_LR = 3e-5
FINETUNE_MAX_STEPS = 300
FINETUNE_WARMUP = 10
FINETUNE_BLOCK_SIZE = 128


# ---------------------------------------------------------------- data conversion

def load_reasoning_jsonl(path: Union[str, Path]) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


class PromptMaskedSFTDataset:
    """Dataset of (prompt, completion) sequences where prompt tokens have label=-100."""

    def __init__(self, examples: list[dict], tokenizer, block_size: int = 128, use_cot: bool = True):
        self.block_size = block_size
        self.samples = []
        for ex in examples:
            prompt_text = ex["text"]
            ans_text = ex.get("cot_text", ex.get("answer_text", "")) if use_cot else ex.get("answer_text", "")

            p_ids = tokenizer.encode(prompt_text)
            a_ids = tokenizer.encode(" " + ans_text)
            if not a_ids:
                continue

            max_p_len = max(1, block_size - len(a_ids))
            p_ids = p_ids[:max_p_len]
            input_ids = p_ids + a_ids
            if len(input_ids) < 2:
                continue
            input_ids = input_ids[: block_size + 1]

            prompt_len = len(p_ids)
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            pad_len = (block_size + 1) - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len
                labels = labels + [-100] * pad_len

            target_slice = labels[1:]
            if not any(lbl != -100 for lbl in target_slice):
                continue

            x = torch.tensor(input_ids[:-1], dtype=torch.long)
            y = torch.tensor(target_slice, dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(
        self, batch_size: int, device: str, generator: Optional[torch.Generator] = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if generator is not None:
            indices = torch.randint(len(self.samples), (batch_size,), generator=generator)
        else:
            indices = torch.randint(len(self.samples), (batch_size,))
        bx = torch.stack([self.samples[i][0] for i in indices]).to(device)
        by = torch.stack([self.samples[i][1] for i in indices]).to(device)
        return bx, by


def build_finetune_bin(
    examples: list[dict], tokenizer, out_path: str, max_seq_len: int = 256, use_cot: bool = True
) -> int:
    all_ids: list[int] = []
    for ex in examples:
        ans = ex.get("cot_text", ex.get("answer_text", "")) if use_cot else ex.get("answer_text", "")
        full_text = ex["text"] + " " + ans
        ids = tokenizer.encode(full_text.strip())[:max_seq_len]
        all_ids.extend(ids)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(out_path)
    return int(arr.size)


# ---------------------------------------------------------------- exact match & multi-tier metrics

_COT_PREFIX_RE = __import__("re").compile(r"\[कारण:[^\]]*\]\s*")
_COT_EXTRACT_RE = __import__("re").compile(r"\[कारण:\s*([^\]]*)\]")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def extract_answer_from_prediction(pred: str) -> str:
    return _COT_PREFIX_RE.sub("", pred).strip()


def extract_rationale_and_answer(text: str) -> tuple[str, str]:
    match = _COT_EXTRACT_RE.search(text)
    if match:
        rationale = match.group(1).strip()
        answer = _COT_EXTRACT_RE.sub("", text).strip()
        return rationale, answer
    return "", text.strip()


def exact_match(pred: str, gold: str) -> bool:
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return p == g
    return g in p or p in g


def exact_match_accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    hits = sum(1 for p, g in zip(preds, golds) if exact_match(p, g))
    return hits / len(preds)


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def char_similarity(s1: str, s2: str) -> float:
    p, g = _normalize(s1), _normalize(s2)
    if not p and not g:
        return 1.0
    max_len = max(len(p), len(g))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(p, g)
    return max(0.0, 1.0 - dist / max_len)


def token_f1_score(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    from collections import Counter
    p_counts = Counter(p_toks)
    g_counts = Counter(g_toks)
    common = sum((p_counts & g_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall = common / len(g_toks)
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- model helper

def load_model_and_config(model_config_path: Union[str, Path]):
    with open(model_config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    if raw_cfg.get("arch_version") == "v2" or "rope_theta" in raw_cfg or raw_cfg.get("d_ff") == 1376:
        from model.gpt_v2 import GPTConfigV2, GPTLanguageModelV2
        model_cfg = GPTConfigV2.from_yaml(model_config_path)
        model = GPTLanguageModelV2(model_cfg)
    else:
        from model.gpt import GPTConfig, GPTLanguageModel
        model_cfg = GPTConfig.from_yaml(model_config_path)
        model = GPTLanguageModel(model_cfg)
    return model, model_cfg


# ---------------------------------------------------------------- reasoning eval

@torch.no_grad()
def evaluate_reasoning(
    model: torch.nn.Module,
    tokenizer,
    test_examples: list[dict],
    device: str = "cpu",
    max_new_tokens: int = 60,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    model.eval()
    preds: list[str] = []
    golds_ans: list[str] = []
    golds_cot: list[str] = []
    per_paradigm_ans: dict[str, list[float]] = {}
    per_paradigm_cot: dict[str, list[float]] = {}
    per_paradigm_f1_ans: dict[str, list[float]] = {}
    per_paradigm_f1_cot: dict[str, list[float]] = {}
    per_paradigm_sim_ans: dict[str, list[float]] = {}
    per_paradigm_decomp: dict[str, list[float]] = {}

    for ex in test_examples:
        prompt_ids = tokenizer.encode(ex["text"])
        if not prompt_ids:
            continue
        idx = torch.tensor(
            [prompt_ids[: model.config.block_size - max_new_tokens]],
            dtype=torch.long,
            device=device,
        )
        gen = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.1,
        )
        pred = tokenizer.decode(gen[0, idx.shape[1] :].tolist())
        gold_ans = ex["answer_text"]
        gold_cot = ex.get("cot_text", gold_ans)

        pred_ans = extract_answer_from_prediction(pred)
        hit_ans = 1.0 if exact_match(pred_ans, gold_ans) else 0.0
        hit_cot = 1.0 if exact_match(pred, gold_cot) else 0.0

        f1_ans = token_f1_score(pred_ans, gold_ans)
        f1_cot = token_f1_score(pred, gold_cot)
        sim_ans = char_similarity(pred_ans, gold_ans)

        pred_rat, _ = extract_rationale_and_answer(pred)
        gold_rat, _ = extract_rationale_and_answer(gold_cot)
        rat_f1 = token_f1_score(pred_rat, gold_rat)
        decomp_score = 0.4 * rat_f1 + 0.6 * f1_ans

        preds.append(pred)
        golds_ans.append(gold_ans)
        golds_cot.append(gold_cot)
        p_name = ex.get("paradigm", "general")
        per_paradigm_ans.setdefault(p_name, []).append(hit_ans)
        per_paradigm_cot.setdefault(p_name, []).append(hit_cot)
        per_paradigm_f1_ans.setdefault(p_name, []).append(f1_ans)
        per_paradigm_f1_cot.setdefault(p_name, []).append(f1_cot)
        per_paradigm_sim_ans.setdefault(p_name, []).append(sim_ans)
        per_paradigm_decomp.setdefault(p_name, []).append(decomp_score)

    n = len(preds)
    acc_ans = sum(sum(v) for v in per_paradigm_ans.values()) / max(n, 1)
    acc_cot = sum(sum(v) for v in per_paradigm_cot.values()) / max(n, 1)
    avg_f1_ans = sum(sum(v) for v in per_paradigm_f1_ans.values()) / max(n, 1)
    avg_f1_cot = sum(sum(v) for v in per_paradigm_f1_cot.values()) / max(n, 1)
    avg_sim_ans = sum(sum(v) for v in per_paradigm_sim_ans.values()) / max(n, 1)
    avg_decomp = sum(sum(v) for v in per_paradigm_decomp.values()) / max(n, 1)

    return {
        "accuracy_answer_only": round(acc_ans, 6),
        "accuracy_exact_match": round(acc_cot, 6),
        "f1_answer_only": round(avg_f1_ans, 6),
        "f1_exact_match": round(avg_f1_cot, 6),
        "char_similarity_answer_only": round(avg_sim_ans, 6),
        "cot_decomposed_score": round(avg_decomp, 6),
        "n_examples": n,
        "per_paradigm_accuracy_answer_only": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(per_paradigm_ans.items())
        },
        "per_paradigm_f1_answer_only": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(per_paradigm_f1_ans.items())
        },
        "per_paradigm_accuracy": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(per_paradigm_cot.items())
        },
    }


# ---------------------------------------------------------------- main pipeline

def finetune(
    pretrained_ckpt: str,
    model_config_path: str,
    tokenizer_path: str,
    data_dir: str,
    out_dir: str,
    device: Optional[str] = None,
    max_steps: int = FINETUNE_MAX_STEPS,
    lr: float = FINETUNE_LR,
    n_val: int = 50,
    n_test: int = 100,
    use_cot: bool = True,
    seed: int = 42,
    custom_ckpt_name: Optional[str] = None,
) -> dict:
    from train.train import TrainConfig, Trainer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_str = "cot" if use_cot else "direct"

    # --- Data
    train_ex = load_reasoning_jsonl(data_dir / "train.jsonl")
    val_ex = load_reasoning_jsonl(data_dir / "val.jsonl")
    test_ex = load_reasoning_jsonl(data_dir / "test.jsonl")
    if len(test_ex) > n_test:
        rng = np.random.default_rng(seed)
        test_ex = [test_ex[i] for i in rng.choice(len(test_ex), n_test, replace=False)]
    if len(val_ex) > n_val:
        rng = np.random.default_rng(seed + 1)
        val_ex = [val_ex[i] for i in rng.choice(len(val_ex), n_val, replace=False)]

    # --- Model + Tokenizer
    model, model_cfg = load_model_and_config(model_config_path)
    tokenizer = Tokenizer(tokenizer_path)
    load_checkpoint(pretrained_ckpt, model, restore_rng=False)

    block_size = min(FINETUNE_BLOCK_SIZE, model_cfg.block_size)
    train_data = PromptMaskedSFTDataset(train_ex, tokenizer, block_size=block_size, use_cot=use_cot)
    val_data = PromptMaskedSFTDataset(val_ex, tokenizer, block_size=block_size, use_cot=use_cot)

    eff_batch = 32
    micro = 8
    accum = eff_batch // micro
    is_v2 = getattr(model_cfg, "arch_version", None) == "v2" or hasattr(model_cfg, "rope_theta")
    eff_warmup = min(30, max_steps // 10) if is_v2 else min(10, max_steps // 10)
    eff_wd = 0.01 if is_v2 else 0.1
    eff_lr = (lr * 1.66) if (is_v2 and lr <= 3.5e-5) else lr

    train_cfg = TrainConfig(
        micro_batch_size=micro,
        gradient_accumulation_steps=accum,
        effective_batch_size=eff_batch,
        learning_rate=eff_lr,
        min_lr=eff_lr / 10,
        warmup_steps=eff_warmup,
        max_steps=max_steps,
        max_grad_norm=1.0,
        weight_decay=eff_wd,
        eval_interval=max(1, max_steps // 10),
        save_interval=max_steps,
        eval_batches=8,
        mixed_precision=device.startswith("cuda"),
        seed=seed,
        notes=f"finetune ({mode_str})",
    )

    ckpt_subdir = out_dir / f"checkpoints_{mode_str}"
    trainer = Trainer(
        model,
        train_data,
        val_data,
        train_cfg,
        checkpoint_dir=str(ckpt_subdir),
        device=device,
    )

    # --- Pretrained Baseline Evaluation (Zero-shot)
    baseline = evaluate_reasoning(model, tokenizer, test_ex, device=device, seed=seed)
    print(f"[{mode_str.upper()}] PRETRAINED BASELINE: {json.dumps(baseline, ensure_ascii=False)}")

    # --- Training Loop
    trainer.train(max_steps=max_steps, resume=True)

    # --- Save Designated Checkpoint
    ckpt_name = custom_ckpt_name or f"finetuned_{mode_str}.pt"
    finetuned_ckpt = str(out_dir / ckpt_name)
    save_checkpoint(
        finetuned_ckpt,
        model,
        trainer.optimizer,
        trainer.scheduler,
        max_steps,
        config={"model": vars(model_cfg), "train": train_cfg.to_dict()},
    )
    print(f"[{mode_str.upper()}] Saved fine-tuned checkpoint: {finetuned_ckpt}")

    # --- Fine-Tuned Evaluation
    finetuned = evaluate_reasoning(model, tokenizer, test_ex, device=device, seed=seed)
    print(f"[{mode_str.upper()}] FINETUNED: {json.dumps(finetuned, ensure_ascii=False)}")

    results = {
        "mode": mode_str,
        "pretrained_baseline": baseline,
        "finetuned": finetuned,
        "delta_accuracy": round(
            finetuned["accuracy_answer_only"] - baseline["accuracy_answer_only"], 6
        ),
        "delta_accuracy_answer_only": round(
            finetuned["accuracy_answer_only"] - baseline["accuracy_answer_only"], 6
        ),
        "delta_accuracy_cot_exact": round(
            finetuned["accuracy_exact_match"] - baseline["accuracy_exact_match"], 6
        ),
        "config": {
            "lr": lr,
            "max_steps": max_steps,
            "block_size": block_size,
            "n_train_examples": len(train_ex),
            "n_val_examples": len(val_ex),
            "n_test_examples": len(test_ex),
            "checkpoint": finetuned_ckpt,
        },
    }

    eval_json_path = out_dir / f"eval_results_{mode_str}.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Also write to top-level eval_results.json for standard compatibility
    with open(out_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Finetune on reasoning data")
    parser.add_argument("--pretrained-ckpt", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", default=str(_LANG_ROOT / "finetune" / "reasoning"))
    parser.add_argument("--out-dir", default=str(_LANG_ROOT / "finetune" / "out"))
    parser.add_argument("--max-steps", type=int, default=FINETUNE_MAX_STEPS)
    parser.add_argument("--lr", type=float, default=FINETUNE_LR)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--mode", choices=["direct", "cot", "both"], default="both")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.mode in ("direct", "both"):
        print("\n--- Running Direct SFT ---")
        finetune(
            args.pretrained_ckpt, args.model_config, args.tokenizer,
            args.data_dir, args.out_dir,
            device=args.device, max_steps=args.max_steps, lr=args.lr,
            n_val=args.n_val, n_test=args.n_test, use_cot=False,
            custom_ckpt_name="finetuned_direct.pt",
        )

    if args.mode in ("cot", "both"):
        print("\n--- Running CoT SFT ---")
        finetune(
            args.pretrained_ckpt, args.model_config, args.tokenizer,
            args.data_dir, args.out_dir,
            device=args.device, max_steps=args.max_steps, lr=args.lr,
            n_val=args.n_val, n_test=args.n_test, use_cot=True,
            custom_ckpt_name="finetuned_cot.pt",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
