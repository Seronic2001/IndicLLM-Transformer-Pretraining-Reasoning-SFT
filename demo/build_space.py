"""Assemble the demo's host folder (Hugging Face Space or Railway), and slim checkpoints.

usage:
  python -m demo.build_space --out <dir> --site <built static site>
  python -m demo.build_space --out <dir> --checkpoint NAME=PATH ...

--site: <dir> gets demo/app.py (as app.py, the server), its requirements, the Space README,
the Dockerfile (Railway) and the site built by demo/static/build_static under site/. The
page is the only demo UI; every host serves the same files. Hosts with an upload limit
(Railway) skip site/models/*.onnx (.railwayignore) and the Dockerfile downloads them.

--checkpoint: slims each checkpoint into <dir>/checkpoints/<NAME>.pt, the input of
build_static. NAME is <lang>_<role>, role "lm" (pretrained, free generation) or
<arch>_<mode> (reasoning fine-tune: v1/v2, direct/cot; "reasoning" is still read as v1_direct).

slim_checkpoint keeps only the weights (fp16) and the model config: a training
checkpoint also holds optimizer and scheduler state and is ~300 MB; the slim one ~50 MB.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

DEMO = Path(__file__).resolve().parent
HOST_FILES = {"app.py": "app.py", "requirements.txt": "requirements.txt",
              "SPACE_README.md": "README.md", "Dockerfile": "Dockerfile", "railwayignore": ".railwayignore"}


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


def build_space(out: Path, site: Path | None, checkpoints: dict[str, Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if site is not None:
        for src, dst in HOST_FILES.items():
            shutil.copy2(DEMO / src, out / dst)
        shutil.copytree(site, out / "site", dirs_exist_ok=True, ignore=shutil.ignore_patterns("README.md"))
    for name, path in checkpoints.items():
        n = slim_checkpoint(path, out / "checkpoints" / f"{name}.pt", note=str(path))
        print(f"  {name}: {n:,} params <- {path}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--site", help="output dir of demo.static.build_static")
    parser.add_argument("--checkpoint", nargs="*", default=[], help="NAME=PATH, e.g. hindi_lm=best.pt")
    args = parser.parse_args(argv)
    build_space(Path(args.out), Path(args.site) if args.site else None,
                dict(c.split("=", 1) for c in args.checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
