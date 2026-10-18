"""How reasoning fine-tuning changed a model's attention, on the reasoning prompts.

usage: python -m common.attention_shift --lang hindi --pretrained best.pt \
           --finetuned direct=finetuned_direct.pt [cot=finetuned_cot.pt] \
           --tokenizer hindi/tokenizer/hindi.model --data test.jsonl --out-dir <dir> [--n 200]

The pretrained checkpoint and each finetuned one read the same n reasoning prompts (the
question text, without the answer). Per layer, averaged over heads, query positions and
prompts:
  entropy          attention entropy (nats); lower = sharper
  mean_distance    attention-weighted |query - key| in tokens
  jsd_vs_pretrained  Jensen-Shannon divergence (nats) of each attention row from the
                     pretrained model's row for the same prompt, layer, head and query
  entity_mass_last   attention from the last prompt token to tokens of entity names,
                     the information the answer depends on
Writes attention_shift.json and one heatmap figure per finetuned model (layer-mean of the
last layer, pretrained vs finetuned, first prompt).
"""
from __future__ import annotations

import argparse
import importlib
import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from common.checkpoint import load_checkpoint  # noqa: E402
from common.plot_fonts import use_indic_fonts  # noqa: E402

use_indic_fonts()


def load_model(ft, path: str, device: str):
    """The model of a pretrained or finetuned checkpoint, built from the config stored in it."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "model.yaml"
        cfg_path.write_text(yaml.safe_dump(saved["config"]["model"]), encoding="utf-8")
        model, _ = ft.load_model_and_config(cfg_path)
    del saved
    load_checkpoint(path, model, restore_rng=False)
    return model.to(device).eval()


def entity_positions(tokenizer, ids: list[int], text: str, names: list[str]) -> list[int]:
    """Indices of tokens overlapping any occurrence of an entity name in the decoded text."""
    ends = [len(tokenizer.decode(ids[: i + 1])) for i in range(len(ids))]
    starts = [0] + ends[:-1]
    decoded = tokenizer.decode(ids)
    spans = []
    for name in names:
        at = decoded.find(name)
        while at != -1:
            spans.append((at, at + len(name)))
            at = decoded.find(name, at + 1)
    return [i for i, (s, e) in enumerate(zip(starts, ends)) if any(s < b and e > a for a, b in spans)]


@torch.no_grad()
def attention_maps(model, ids: list[int], device: str) -> torch.Tensor:
    """(n_layer, n_head, T, T) post-softmax attention for one prompt."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    return torch.stack([a[0].float().cpu() for a in model(x, return_attn=True)["attn_weights"]])


def row_stats(att: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per layer entropy and mean distance, averaged over heads and queries."""
    p = att.clamp_min(1e-12)
    entropy = -(att * p.log()).sum(-1).mean(dim=(1, 2))
    T = att.shape[-1]
    pos = torch.arange(T, dtype=torch.float)
    dist = (pos[None, :] - pos[:, None]).abs()
    return entropy, (att * dist).sum(-1).mean(dim=(1, 2))


def jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Per layer JSD between matching attention rows, averaged over heads and queries."""
    m = 0.5 * (p + q)

    def kl(a, b):
        return (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(-1)

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean(dim=(1, 2))


def plot_pair(pre: torch.Tensor, post: torch.Tensor, pieces: list[str], title: str, path: Path) -> None:
    # byte-fallback pieces such as <0x3F> ("?", not in the vocabulary) shown as their character
    labels = [chr(int(p[3:-1], 16)) if p.startswith("<0x") and 0x20 <= int(p[3:-1], 16) < 0x7F
              else p.replace("▁", "") for p in pieces]
    T = len(labels)
    size = max(5, T / 7)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 1.5, size + 1))
    for ax, att, name in zip(axes, (pre, post), ("pretrained", "finetuned")):
        im = ax.imshow(att.numpy(), cmap="viridis", vmin=0.0, vmax=float(max(pre.max(), post.max())))
        ax.set_title(name)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        step = 1 if T <= 40 else max(1, T // 20)
        ticks = list(range(0, T, step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([labels[i] for i in ticks], rotation=90, fontsize=6)
        ax.set_yticklabels([labels[i] for i in ticks], fontsize=6)
    fig.colorbar(im, ax=axes, label="attention weight (mean over heads)")
    fig.suptitle(title)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lang", choices=["hindi", "assamese"], required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--finetuned", nargs="+", required=True, help="name=path, e.g. direct=finetuned_direct.pt")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data", required=True, help="reasoning jsonl (prompts = its text field)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    ft = importlib.import_module(f"{args.lang}.finetune.finetune")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ft.Tokenizer(args.tokenizer)
    examples = ft.load_reasoning_jsonl(Path(args.data))[: args.n]
    prompts = [(tokenizer.encode(ex["text"]), ex) for ex in examples]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = {"pretrained": args.pretrained, **dict(f.split("=", 1) for f in args.finetuned)}
    sums, first = {}, {}
    pre_maps = []
    for name, path in models.items():
        model = load_model(ft, path, device)
        acc = {"entropy": 0.0, "mean_distance": 0.0, "jsd_vs_pretrained": 0.0, "entity_mass_last": 0.0}
        for k, (ids, ex) in enumerate(prompts):
            att = attention_maps(model, ids, device)
            ent, dist = row_stats(att)
            ents = entity_positions(tokenizer, ids, ex["text"], ex["entities"])
            acc["entropy"] = acc["entropy"] + ent
            acc["mean_distance"] = acc["mean_distance"] + dist
            acc["entity_mass_last"] = acc["entity_mass_last"] + att[:, :, -1, ents].sum(-1).mean(1)
            if name == "pretrained":
                pre_maps.append(att.half())
            else:
                acc["jsd_vs_pretrained"] = acc["jsd_vs_pretrained"] + jsd(att, pre_maps[k].float())
            if k == 0:
                first[name] = att[-1].mean(0)
        sums[name] = {key: [round(float(v) / len(prompts), 4) for v in val] if torch.is_tensor(val) else None
                      for key, val in acc.items()}
        del model
        print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in sums[name].items()), flush=True)

    pieces = [tokenizer.sp.id_to_piece(i) for i in prompts[0][0]]
    for name in models:
        if name != "pretrained":
            plot_pair(first["pretrained"], first[name], pieces,
                      f"{args.lang} last layer, mean over heads: pretrained vs {name}",
                      out / f"attention_pair_{name}.png")
    (out / "attention_shift.json").write_text(json.dumps({
        "lang": args.lang, "n_prompts": len(prompts), "data": args.data,
        "checkpoints": models, "per_layer": sums, "first_prompt": examples[0]["text"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
