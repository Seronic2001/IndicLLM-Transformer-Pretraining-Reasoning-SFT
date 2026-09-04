"""Assamese attention analysis toolkit.

Consumes the model's ``return_attn=True`` output — post-softmax attention weights
(B, n_head, T, T) — and produces:

  * ``attention_entropy``         per-head entropy averaged over query positions
  * ``mean_attention_distance``   per-head mean |query - key| weighted by attention
  * ``plot_attention_heatmap``    labeled query-vs-key heatmap for one head/layer
  * ``analyze_attention``         driver: runs early + late layers, several heads,
                                  writes heatmap PNGs and a summary JSON

Every plot sets title, x-label, y-label (spec §0.4) — enforced by test.
SentencePiece's leading ``▁`` markers are stripped for display only; token ids
are never altered. Full (T, T) matrices are computed and discarded per example,
never held for all examples at once (spec reliability fallbacks).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

_LANG_ROOT = Path(__file__).resolve().parents[1]
if str(_LANG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LANG_ROOT))


def attention_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """attn_weights: (n_head, T, T) post-softmax, rows sum to 1.

    Returns (n_head,) entropy per head, averaged over query positions.
    """
    p = attn_weights.float()
    log_p = torch.where(p > 0, p.log(), torch.zeros_like(p))
    entropies = -(p * log_p).sum(dim=-1)  # (n_head, T)
    return entropies.mean(dim=-1)


def mean_attention_distance(attn_weights: torch.Tensor) -> torch.Tensor:
    """attn_weights: (n_head, T, T) post-softmax.

    Returns (n_head,) mean |query_pos - key_pos| weighted by attention mass —
    higher = more long-range attention for that head.
    """
    p = attn_weights.float()
    n_head, T, _ = p.shape
    positions = torch.arange(T, dtype=torch.float, device=p.device)
    dist = (positions[None, None, :] - positions[None, :, None]).abs()  # (1, T, T)
    mean_d = (p * dist).sum(dim=-1)  # (n_head, T)
    return mean_d.mean(dim=-1)


def plot_attention_heatmap(
    attn_weights: torch.Tensor,
    tokens: list[str],
    layer: int,
    head: int,
    title: str,
    save_path: str,
) -> plt.Axes:
    """Query vs key positions, color scale for weights.

    ``tokens`` are display-only (SentencePiece ``▁`` markers stripped); must set
    title, x-label, y-label. Returns the Axes for programmatic assertions.
    """
    w = attn_weights.detach().cpu().float()
    assert w.dim() == 2 and w.shape[0] == w.shape[1], "expected (T, T) matrix"
    T = w.shape[0]

    fig, ax = plt.subplots(figsize=(max(4, T / 8 + 2), max(4, T / 8 + 2)))
    im = ax.imshow(w.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax, label="attention weight")

    # Font discovery for Indic scripts (Nirmala UI on Windows)
    nirmala_path = r"C:\Windows\Fonts\Nirmala.ttc"
    indic_font = (
        matplotlib.font_manager.FontProperties(fname=nirmala_path, size=6.5)
        if Path(nirmala_path).exists()
        else None
    )

    # Tick labels: full token strings only for short sentences, else sparse.
    display = [t.replace("\u2581", "") for t in tokens][:T]
    if T <= 32:
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        if indic_font:
            ax.set_xticklabels(display, rotation=45, ha="right", rotation_mode="anchor", fontproperties=indic_font)
            ax.set_yticklabels(display, fontproperties=indic_font)
        else:
            ax.set_xticklabels(display, rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
            ax.set_yticklabels(display, fontsize=6)
    else:
        step = max(1, T // 8)
        ticks = list(range(0, T, step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        sub_display = [display[i] for i in ticks]
        if indic_font:
            ax.set_xticklabels(sub_display, rotation=45, ha="right", rotation_mode="anchor", fontproperties=indic_font)
            ax.set_yticklabels(sub_display, fontproperties=indic_font)
        else:
            ax.set_xticklabels(sub_display, rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
            ax.set_yticklabels(sub_display, fontsize=6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return ax


@torch.no_grad()
def analyze_attention(
    model,
    tokenizer,
    texts: list[str],
    out_dir: str,
    layers: Optional[list[int]] = None,
    heads: Optional[list[int]] = None,
    device: str = "cpu",
) -> dict:
    """Run the per-example attention analysis, write heatmaps + summary JSON.

    Returns {"per_example": [...], "summary": {...}} and writes
    ``attention_summary.json`` plus one PNG per (example, layer, head).
    """
    model.eval()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_layers = len(model.blocks)
    if layers is None:
        layers = [0, n_layers // 2, n_layers - 1]
    n_heads = model.config.n_head
    if heads is None:
        heads = list(range(min(4, n_heads)))

    per_example: list[dict] = []
    summary_rows: list[dict] = []

    for ei, text in enumerate(texts):
        ids = tokenizer.encode(text)
        if len(ids) == 0:
            continue
        x = torch.tensor([ids[: model.config.block_size]], dtype=torch.long, device=device)
        attn_layers = model(x, return_attn=True)["attn_weights"]  # one per layer

        example = {"example": ei, "text": text, "layer_head": []}
        for li in layers:
            att = attn_layers[li][0]  # (n_head, T, T)
            ent = attention_entropy(att)
            dist = mean_attention_distance(att)
            for hi in heads:
                if hi >= att.shape[0]:
                    continue
                title = (
                    f"{getattr(model.config, 'model_name', '')} layer {li} head {hi} "
                    f"| example {ei}"
                ).strip()
                path = out / f"ex{ei}_layer{li}_head{hi}.png"
                plot_attention_heatmap(
                    att[hi],
                    tokenizer.encode_as_pieces(text),
                    layer=li,
                    head=hi,
                    title=title,
                    save_path=str(path),
                )
                row = {
                    "example": ei,
                    "layer": li,
                    "head": hi,
                    "entropy": float(ent[hi]),
                    "mean_distance": float(dist[hi]),
                    "heatmap": str(path),
                }
                summary_rows.append(row)
                example["layer_head"].append({"layer": li, "head": hi})
        per_example.append(example)

    summary = {
        "layers_analyzed": layers,
        "heads_analyzed": heads,
        "n_examples": len(per_example),
        "rows": summary_rows,
    }
    with open(out / "attention_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {"per_example": per_example, "summary": summary}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Attention analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", default=str(_LANG_ROOT / "eval" / "attention"))
    parser.add_argument("--texts", nargs="+", default=[
        "অসমৰ ৰাজধানী গুৱাহাটী আৰু ই এখন ডাঙৰ চহৰ",
        "লৰাবোৰ বিদ্যালয়ত কিতাপ পঢ়ে আৰু খেলে",
        "সূৰ্য পূবত উদয় হয় আৰু পশ্চিমত অস্ত যায়",
    ])
    parser.add_argument("--device", default=None)
    parser.add_argument("--arch", choices=("v1", "v2"), default="v1",
                        help="model architecture: v1 (learned-abs/GELU/LayerNorm) or v2 (RoPE/SwiGLU/RMSNorm)")
    args = parser.parse_args(argv)

    from tokenizer.tokenizer import Tokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from common.checkpoint import load_checkpoint

    if args.arch == "v2":
        from model.gpt_v2 import GPTConfigV2, GPTLanguageModelV2

        model = GPTLanguageModelV2(GPTConfigV2.from_yaml(args.model_config)).to(device)
    else:
        from model.gpt import GPTConfig, GPTLanguageModel

        model = GPTLanguageModel(GPTConfig.from_yaml(args.model_config)).to(device)
    load_checkpoint(args.checkpoint, model, restore_rng=False)
    tok = Tokenizer(args.tokenizer)
    results = analyze_attention(model, tok, args.texts, args.out_dir, device=device)
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
