"""Score a finetuned reasoning checkpoint on reasoning jsonl files, without training.

usage: python -m common.evaluate_finetuned --lang hindi --ckpt finetuned_direct.pt \
           --tokenizer hindi/tokenizer/hindi.model --data-dir <dir> --out results.json \
           [--files test test_depth6 ...] [--n 2000] [--max-new-tokens 80]

The model is rebuilt from the config stored in the checkpoint, so the architecture is
exactly the one that was finetuned. Scoring is <lang>.finetune.finetune.evaluate_reasoning,
the function the finetuning run used, so the numbers are comparable with its logs.
One language per process: both languages import their model code as `model`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import tempfile
import time
from pathlib import Path

import torch
import yaml

DEFAULT_FILES = ("test", "test_depth3", "test_depth4", "test_depth5", "test_depth6", "test_depth8",
                 "test_depth10", "test_branch", "test_distractor_attr", "test_redundant")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lang", choices=["hindi", "assamese"], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--n", type=int, default=2000, help="examples per file (first n)")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="default: finetune MAX_NEW_TOKENS (Direct); CoT on long chains needs ~160")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    ft = importlib.import_module(f"{args.lang}.finetune.finetune")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "model.yaml"
        cfg_path.write_text(yaml.safe_dump(saved["config"]["model"]), encoding="utf-8")
        model, _ = ft.load_model_and_config(cfg_path)
    model.load_state_dict(saved["model_state_dict"])
    model.to(device).eval()
    del saved
    tokenizer = ft.Tokenizer(args.tokenizer)
    max_new_tokens = args.max_new_tokens or ft.MAX_NEW_TOKENS

    results = {"checkpoint": str(args.ckpt), "max_new_tokens": max_new_tokens, "splits": {}}
    for name in args.files:
        path = Path(args.data_dir) / f"{name}.jsonl"
        if not path.exists():
            print(f"[skip] {name}: {path} not found", flush=True)
            continue
        start = time.time()
        examples = ft.load_reasoning_jsonl(path)[: args.n]
        r = ft.evaluate_reasoning(model, tokenizer, examples, device=device, max_new_tokens=max_new_tokens)
        r["chance_baseline"] = ft.chance_baselines(examples)
        results["splits"][name] = r
        print(f"{name:22s} n={r['n_examples']:5d} strict={r['accuracy_answer_strict']:.4f} "
              f"relation={r['accuracy_relation']:.4f} macroF1={r['macro_f1_relation']:.4f} "
              f"majority={r['chance_baseline']['majority_accuracy']:.4f} ({time.time() - start:.0f}s)",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
