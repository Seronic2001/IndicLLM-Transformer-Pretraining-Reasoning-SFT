"""Hindi evaluation suite.

Standardized Generation Protocol (project specifications Section 3 Evaluator):
  * Extract N=500 held-out prompt prefixes of T_prefix=32 tokens from test.bin.
  * Generate a T_gen=64-token continuation per prefix at temperatures
    {0.0 (greedy), 0.5, 1.0, 1.5}.
  * Compare against the actual 64-token ground-truth continuation.

Outputs (in --out-dir):
  * ppl_bpb_table.json          test-set loss / perplexity / bits-per-byte
  * generation_metrics.json     per-temperature BLEU / chrF / ROUGE-L /
                                repetition-rate / distinct-1..4
  * generated_samples.jsonl     per-sample prefix, reference, and all 4 outputs
  * full_context_ppl.json       (--full-context-windows > 0) loss / perplexity /
                                bits-per-byte on evenly spaced, non-overlapping
                                block_size-token windows, the same windows cut to
                                T_prefix + T_gen tokens, and loss by context length

ppl_bpb_table.json scores short (T_prefix + T_gen = 96-token) windows, so its loss
is higher than the trainer's validation loss, which uses block_size-token windows.
full_context_ppl.json uses fixed windows that do not depend on the seed or the
model, so numbers from different checkpoints are directly comparable. Pass
--ppl-only to skip the generation protocol.

Degenerate/high-temperature repetition is an expected, reportable finding — it is
recorded, never silently filtered (spec Evaluator reliability fallbacks).
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
FULL_CONTEXT_WINDOWS = 2000
# Target-position ranges (0-based, end-exclusive) labelled by preceding-token count.
CONTEXT_BUCKETS = {"1-31": (0, 31), "32-95": (31, 95), "96-255": (95, 255), "256-511": (255, 511)}


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
    """Mean loss / perplexity / bits-per-byte over the test windows."""
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


@torch.no_grad()
def compute_full_context_ppl(
    model: GPTLanguageModel,
    spec: EvalSpec,
    n_windows: int = FULL_CONTEXT_WINDOWS,
    batch_size: int = 32,
) -> dict:
    """Loss / perplexity / bits-per-byte on fixed full-length test windows.

    Windows are evenly spaced, non-overlapping block_size-token slices of
    test.bin, identical for every model. Each is scored at full length and cut
    to prefix_len + gen_len tokens (the ppl_bpb_table.json span), and the
    full-length loss is also averaged per context-length bucket.
    """
    import numpy as np

    model.eval()
    block = model.config.block_size
    short = min(spec.prefix_len + spec.gen_len, block)
    data = np.memmap(spec.test_path, dtype=np.uint16, mode="r")
    n_possible = len(data) // block
    if n_possible < 1:
        raise ValueError(
            f"test corpus too small ({len(data)} tokens) for a {block}-token window"
        )
    n_windows = min(n_windows, n_possible)
    starts = np.linspace(0, n_possible - 1, n_windows).astype(np.int64) * block
    windows = np.stack([data[s : s + block] for s in starts]).astype(np.int64)

    def position_losses(length: int) -> "np.ndarray":
        total = torch.zeros(length - 1, dtype=torch.float64)
        for i in range(0, n_windows, batch_size):
            w = torch.from_numpy(windows[i : i + batch_size, :length]).to(spec.device)
            x, y = w[:, :-1], w[:, 1:]
            logits = model(x)["logits"].float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none"
            )
            total += loss.view(y.shape).sum(dim=0).double().cpu()
        return (total / n_windows).numpy()

    def summary(losses: "np.ndarray", length: int) -> dict:
        mean_loss = float(losses.mean())
        n_tokens = n_windows * (length - 1)
        n_bytes = sum(
            len(spec.tokenizer.decode(w[:length].tolist()).encode("utf-8")) for w in windows
        )
        return {
            "loss": mean_loss,
            "perplexity": perplexity(mean_loss),
            "bits_per_byte": bits_per_byte(mean_loss, n_tokens, n_bytes),
            "tokens_evaluated": int(n_tokens),
            "utf8_bytes": int(n_bytes),
            "window_tokens": int(length),
        }

    full_losses = position_losses(block)
    return {
        "windows": int(n_windows),
        "full_context": summary(full_losses, block),
        "short_context": summary(position_losses(short), short),
        "loss_by_context_length": {
            label: float(full_losses[a:b].mean())
            for label, (a, b) in CONTEXT_BUCKETS.items()
            if b <= len(full_losses)
        },
    }


def run_evaluation(
    model: GPTLanguageModel,
    spec: EvalSpec,
    out_dir: str,
    full_context_windows: int = 0,
    ppl_only: bool = False,
) -> dict:
    """Run the full protocol and write the deliverable JSONs.

    With full_context_windows > 0 also writes full_context_ppl.json; with
    ppl_only the generation protocol is skipped. Returns the combined results
    dict (also written to disk).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    if full_context_windows > 0:
        results["full_context_ppl"] = compute_full_context_ppl(
            model, spec, n_windows=full_context_windows
        )
        with open(out / "full_context_ppl.json", "w", encoding="utf-8") as f:
            json.dump(results["full_context_ppl"], f, ensure_ascii=False, indent=2)
    if ppl_only:
        results["ppl_bpb"] = compute_ppl_bpb(model, spec)
        with open(out / "ppl_bpb_table.json", "w", encoding="utf-8") as f:
            json.dump(results["ppl_bpb"], f, ensure_ascii=False, indent=2)
        return results

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

    results.update({"generation_metrics": gen_metrics, "ppl_bpb": ppl})
    return results


# Checkpoints written by the Kaggle training scripts name parameters differently
# from the model classes here; (old, new) substrings applied in order to each key.
_KEY_RENAMES = (
    ("_orig_mod.", ""),
    ("transformer.wte.", "token_embedding."),
    ("transformer.wpe.", "position_embedding."),
    ("transformer.h.", "blocks."),
    ("transformer.ln_f.", "ln_f."),
    (".ln1.", ".ln_1."),
    (".ln2.", ".ln_2."),
    (".attn.qkv_proj.", ".attn.c_attn."),
    (".attn.out_proj.", ".attn.c_proj."),
    (".mlp.fc1.", ".mlp.c_fc."),
    (".mlp.fc2.", ".mlp.c_proj."),
)


def load_model_for_eval(checkpoint: str, model_config: str, device: str):
    """Build the model named by the config (V1, V1 no_pos, or V2) and load weights.

    Parameter names are remapped and the non-learned causal-mask buffers are
    left to the model; any other missing or unexpected key raises.
    """
    import yaml

    with open(model_config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    if raw_cfg.get("arch_version") == "v2" or "rope_theta" in raw_cfg:
        from model.gpt_v2 import GPTConfigV2, GPTLanguageModelV2

        model = GPTLanguageModelV2(GPTConfigV2.from_yaml(model_config))
    else:
        model = GPTLanguageModel(GPTConfig.from_yaml(model_config))

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {}
    for key, value in ckpt["model_state_dict"].items():
        if key.endswith("attn.mask"):
            continue
        for old, new in _KEY_RENAMES:
            key = key.replace(old, new)
        state[key] = value
    load_result = model.load_state_dict(state, strict=False)
    missing = [k for k in load_result.missing_keys if not k.endswith("attn.mask")]
    if missing or load_result.unexpected_keys:
        raise ValueError(
            f"Checkpoint {checkpoint} is incompatible with {model_config}: "
            f"missing={missing}, unexpected={load_result.unexpected_keys}."
        )
    return model.to(device)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation suite")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--tokenizer", required=True, help="path to <lang>.model")
    parser.add_argument("--test-bin", required=True)
    parser.add_argument("--out-dir", default=str(_LANG_ROOT / "eval"))
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument(
        "--full-context-windows", type=int, default=FULL_CONTEXT_WINDOWS,
        help="fixed block_size-token test windows for full_context_ppl.json (0 disables)",
    )
    parser.add_argument("--ppl-only", action="store_true", help="skip the generation protocol")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    from tokenizer.tokenizer import Tokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_eval(args.checkpoint, args.model_config, device)

    spec = EvalSpec(
        test_path=args.test_bin,
        tokenizer=Tokenizer(args.tokenizer),
        n_prompts=args.n_prompts,
        device=device,
    )
    results = run_evaluation(
        model, spec, args.out_dir,
        full_context_windows=args.full_context_windows, ppl_only=args.ppl_only,
    )
    print(json.dumps({"ppl_bpb": results["ppl_bpb"]}, indent=2))
    if "full_context_ppl" in results:
        print(json.dumps({"full_context_ppl": results["full_context_ppl"]}, indent=2))
    for label, m in results.get("generation_metrics", {}).items():
        print(f"temp={label}: bleu={m['bleu']:.2f} chrf={m['chrf']:.2f} "
              f"rouge_l={m['rouge_l']:.3f} rep3={m['repetition_rate_3']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
