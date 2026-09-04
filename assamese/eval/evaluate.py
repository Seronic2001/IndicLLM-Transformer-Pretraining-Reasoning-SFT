"""Hindi evaluation suite (Agent-E).

Standardized Generation Protocol (AGENT_BUILD_SPEC §3 Agent-E):
  * Extract N=500 held-out prompt prefixes of T_prefix=32 tokens from test.bin.
  * Generate a T_gen=64-token continuation per prefix at temperatures
    {0.0 (greedy), 0.5, 1.0, 1.5}.
  * Compare against the actual 64-token ground-truth continuation.

Outputs (in --out-dir):
  * ppl_bpb_table.json          test-set loss / perplexity / bits-per-byte
  * generation_metrics.json     per-temperature BLEU / chrF / ROUGE-L /
                                repetition-rate / distinct-1..4
  * generated_samples.jsonl     per-sample prefix, reference, and all 4 outputs

Degenerate/high-temperature repetition is an expected, reportable finding — it is
recorded, never silently filtered (spec Agent-E reliability fallbacks).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# Make `model.` / `tokenizer.` style imports work when run as a script.
_LANG_ROOT = Path(__file__).resolve().parents[1]
if str(_LANG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LANG_ROOT))

from common.metrics import (  # noqa: E402
    bits_per_byte,
    bleu,
    chrf,
    distinct_n,
    perplexity,
    repetition_rate,
    rouge_l,
)
from model.gpt import GPTConfig, GPTLanguageModel  # noqa: E402

N_PROMPTS = 500
T_PREFIX = 32
T_GEN = 64
TEMPERATURES = (0.0, 0.5, 1.0, 1.5)


@dataclass
class EvalSpec:
    test_path: str
    tokenizer: object  # Tokenizer-like (encode/decode)
    n_prompts: int = N_PROMPTS
    prefix_len: int = T_PREFIX
    gen_len: int = T_GEN
    temperatures: tuple[float, ...] = TEMPERATURES
    device: str = "cpu"
    seed: int = 1337


def extract_prefixes(spec: EvalSpec) -> list[tuple[list[int], list[int]]]:
    """Draw N (prefix_ids, target_ids) pairs from test.bin deterministically.

    Each window starts at a seeded-random position; prefix is the first
    prefix_len tokens, target the next gen_len tokens (the ground truth).
    """
    import numpy as np

    data = np.memmap(spec.test_path, dtype=np.uint16, mode="r")
    span = spec.prefix_len + spec.gen_len
    max_start = len(data) - span
    if max_start < 1:
        raise ValueError(
            f"test corpus too small ({len(data)} tokens) for prefix+gen span {span}"
        )
    rng = np.random.default_rng(spec.seed)
    starts = rng.choice(max_start, size=spec.n_prompts, replace=False)
    return [
        (data[s : s + spec.prefix_len].tolist(), data[s + spec.prefix_len : s + span].tolist())
        for s in sorted(starts)
    ]


@torch.no_grad()
def generate_continuations(
    model: GPTLanguageModel,
    spec: EvalSpec,
    pairs: list[tuple[list[int], list[int]]],
) -> dict[str, list[str]]:
    """Run the 4-temperature protocol. Returns {temp_label: [decoded_text, ...]}."""
    model.eval()
    out: dict[str, list[str]] = {str(t): [] for t in spec.temperatures}
    for prefix_ids, _target in pairs:
        idx = torch.tensor([prefix_ids], dtype=torch.long, device=spec.device)
        for temp in spec.temperatures:
            gen = model.generate(
                idx, max_new_tokens=spec.gen_len, temperature=temp, top_k=None
            )
            new_ids = gen[0, -spec.gen_len:].tolist()
            out[str(temp)].append(spec.tokenizer.decode(new_ids))
    return out


def compute_generation_metrics(
    references: list[str], generations: dict[str, list[str]]
) -> dict:
    """Corpus-level metrics per temperature (never filters degenerate outputs)."""
    result: dict[str, dict] = {}
    for label, hyps in generations.items():
        result[label] = {
            "bleu": bleu(hyps, references),
            "chrf": chrf(hyps, references),
            "rouge_l": rouge_l(hyps, references),
            "repetition_rate_3": repetition_rate(hyps, n=3),
            "distinct_1": distinct_n(hyps, n=1),
            "distinct_2": distinct_n(hyps, n=2),
            "distinct_4": distinct_n(hyps, n=4),
        }
    return result


@torch.no_grad()
def compute_ppl_bpb(model: GPTLanguageModel, spec: EvalSpec) -> dict:
    """Mean loss / perplexity / bits-per-byte over the test windows (Agent-E)."""
    import numpy as np

    data = np.memmap(spec.test_path, dtype=np.uint16, mode="r")
    span = spec.prefix_len + spec.gen_len
    max_start = len(data) - span
    rng = np.random.default_rng(spec.seed + 1)
    starts = rng.choice(max_start, size=spec.n_prompts, replace=False)

    total_loss = 0.0
    total_tokens = 0
    total_bytes = 0
    for s in starts:
        window = data[s : s + span].astype(np.int64)
        x = torch.tensor(window[:-1], dtype=torch.long, device=spec.device).unsqueeze(0)
        y = torch.tensor(window[1:], dtype=torch.long, device=spec.device).unsqueeze(0)
        loss = model(x, targets=y)["loss"].item()
        n_tokens = window.size - 1
        text = spec.tokenizer.decode(window.tolist())
        total_loss += loss * n_tokens
        total_tokens += n_tokens
        total_bytes += len(text.encode("utf-8"))

    mean_loss = total_loss / max(1, total_tokens)
    return {
        "loss": mean_loss,
        "perplexity": perplexity(mean_loss),
        "bits_per_byte": bits_per_byte(mean_loss, total_tokens, total_bytes),
        "tokens_evaluated": int(total_tokens),
        "utf8_bytes": int(total_bytes),
        "windows": len(starts),
        "prefix_len": spec.prefix_len,
        "gen_len": spec.gen_len,
    }


def run_evaluation(
    model: GPTLanguageModel,
    spec: EvalSpec,
    out_dir: str,
) -> dict:
    """Run the full protocol and write the three deliverable JSONs.

    Returns the combined results dict (also written to disk).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = extract_prefixes(spec)
    references = [spec.tokenizer.decode(t) for _p, t in pairs]
    generations = generate_continuations(model, spec, pairs)
    gen_metrics = compute_generation_metrics(references, generations)
    ppl = compute_ppl_bpb(model, spec)

    with open(out / "generation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(gen_metrics, f, ensure_ascii=False, indent=2)
    with open(out / "ppl_bpb_table.json", "w", encoding="utf-8") as f:
        json.dump(ppl, f, ensure_ascii=False, indent=2)

    with open(out / "generated_samples.jsonl", "w", encoding="utf-8") as f:
        for i, (prefix_ids, target_ids) in enumerate(pairs):
            rec = {
                "index": i,
                "prefix": spec.tokenizer.decode(prefix_ids),
                "reference": references[i],
            }
            for label, hyps in generations.items():
                rec[f"generated_temp_{label}"] = hyps[i]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {"generation_metrics": gen_metrics, "ppl_bpb": ppl}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation suite (Agent-E)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--tokenizer", required=True, help="path to <lang>.model")
    parser.add_argument("--test-bin", required=True)
    parser.add_argument("--out-dir", default=str(_LANG_ROOT / "eval"))
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--arch", choices=("v1", "v2"), default="v1",
                        help="model architecture: v1 (learned-abs/GELU/LayerNorm) or v2 (RoPE/SwiGLU/RMSNorm)")
    args = parser.parse_args(argv)

    from tokenizer.tokenizer import Tokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.arch == "v2":
        from model.gpt_v2 import GPTConfigV2, GPTLanguageModelV2

        model_cfg = GPTConfigV2.from_yaml(args.model_config)
        model = GPTLanguageModelV2(model_cfg).to(device)
    else:
        model_cfg = GPTConfig.from_yaml(args.model_config)
        model = GPTLanguageModel(model_cfg).to(device)

    from common.checkpoint import load_checkpoint

    load_checkpoint(args.checkpoint, model, restore_rng=False)

    spec = EvalSpec(
        test_path=args.test_bin,
        tokenizer=Tokenizer(args.tokenizer),
        n_prompts=args.n_prompts,
        device=device,
    )
    results = run_evaluation(model, spec, args.out_dir)
    print(json.dumps({"ppl_bpb": results["ppl_bpb"]}, indent=2))
    for label, m in results["generation_metrics"].items():
        print(f"temp={label}: bleu={m['bleu']:.2f} chrf={m['chrf']:.2f} "
              f"rouge_l={m['rouge_l']:.3f} rep3={m['repetition_rate_3']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
