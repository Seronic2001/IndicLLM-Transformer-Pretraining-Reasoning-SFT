"""Assemble the Hugging Face Space for the demo, and slim checkpoints for it.

usage:
  python -m demo.build_space --out <space dir> [--checkpoint NAME=PATH ...]

The Space holds demo/app.py (as app.py), its README and requirements, the code the
app imports from both languages (model, tokenizer, reasoning generator) and the
checkpoints under checkpoints/<NAME>.pt. NAME is <lang>_<role>, role "lm"
(pretrained, free generation) or "reasoning" (Direct-finetuned).

slim_checkpoint keeps only the weights (fp16) and the model config: a training
checkpoint also holds optimizer and scheduler state and is ~300 MB; the slim one ~50 MB.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path(__file__).resolve().parent
CODE_FILES = [
    "{lang}/model/__init__.py", "{lang}/model/gpt.py", "{lang}/model/gpt_v2.py",
    "{lang}/tokenizer/__init__.py", "{lang}/tokenizer/tokenizer.py", "{lang}/tokenizer/{lang}.model",
    "{lang}/finetune/__init__.py", "{lang}/finetune/generate_reasoning.py",
]
LANGS = ("hindi", "assamese")


def build_model(cfg: dict) -> torch.nn.Module:
    """The model described by a checkpoint's config (V1 or V2), with fresh weights.
    Uses the Hindi model code: the two languages' model files are identical."""
    import tempfile

    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        if cfg.get("arch_version") == "v2" or "rope_theta" in cfg:
            from hindi.model.gpt_v2 import GPTConfigV2, GPTLanguageModelV2
            return GPTLanguageModelV2(GPTConfigV2.from_yaml(path))
        from hindi.model.gpt import GPTConfig, GPTLanguageModel
        return GPTLanguageModel(GPTConfig.from_yaml(path))


def slim_checkpoint(src: Path, dst: Path, note: str = "") -> int:
    """Write {"config": {"model": ...}, "model_state_dict": fp16 weights, "note"} to dst;
    returns the parameter count. Tied embeddings are saved once (shared storage)."""
    saved = torch.load(src, map_location="cpu", weights_only=False)
    model = build_model(saved["config"]["model"])
    model.load_state_dict(saved["model_state_dict"])
    model.half()
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"model": saved["config"]["model"]}, "model_state_dict": model.state_dict(),
                "note": note or str(src)}, dst)
    return sum(p.numel() for p in model.parameters())


def build_space(out: Path, checkpoints: dict[str, Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DEMO / "app.py", out / "app.py")
    shutil.copy2(DEMO / "requirements.txt", out / "requirements.txt")
    shutil.copy2(DEMO / "SPACE_README.md", out / "README.md")
    for lang in LANGS:
        for pattern in CODE_FILES:
            rel = pattern.format(lang=lang)
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / rel, out / rel)
    for name, path in checkpoints.items():
        n = slim_checkpoint(path, out / "checkpoints" / f"{name}.pt", note=str(path))
        print(f"  {name}: {n:,} params <- {path}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", nargs="*", default=[], help="NAME=PATH, e.g. hindi_lm=best.pt")
    args = parser.parse_args(argv)
    build_space(Path(args.out), dict(c.split("=", 1) for c in args.checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
