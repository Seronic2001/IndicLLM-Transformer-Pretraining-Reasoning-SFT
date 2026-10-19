"""Draw every report figure from the result files, so each figure traces to an output.

usage: python -m report.make_figures   (from the repo root; writes report/figures/)

Inputs: report/results/*.json (copied unchanged from the Kaggle runs listed in
report/results/README.md), <lang>/data/dataset_stats.json and the Phase 2 heatmaps in
<lang>/eval/attention/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.plot_fonts import use_indic_fonts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "report" / "results"
FIGURES = ROOT / "report" / "figures"
LANGS = ("hindi", "assamese")
MODELS = ("hindi_v1", "hindi_v2", "assamese_v1", "assamese_v2")
LABEL = {"hindi": "Hindi", "assamese": "Assamese", "hindi_v1": "Hindi V1", "hindi_v2": "Hindi V2",
         "assamese_v1": "Assamese V1", "assamese_v2": "Assamese V2"}
ARCH_COLOR = {"v1": "#4C72B0", "v2": "#DD8452"}
MODE_STYLE = {"direct": "-", "cot": "--"}
KIND_COLOR = {"manual": "#55A868", "downloaded": "#8172B2"}


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- Phase 1

def phase1() -> None:
    stats = {lang: load(f"phase1_tokenizer_stats_{lang}.json")["candidates"] for lang in LANGS}
    keys = ("bpe_16384", "bpe_32768", "unigram_16384", "unigram_32768")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(keys))
    for i, lang in enumerate(LANGS):
        vals = [stats[lang][k]["fertility_tokens_per_word"] for k in keys]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, label=LABEL[lang])
        ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, ["BPE 16K\n(chosen)", "BPE 32K", "Unigram 16K", "Unigram 32K"])
    ax.set_ylabel("tokens per word (lower is better)")
    ax.set_title("Tokenizer fertility on held-out text (2,000 documents)")
    ax.set_ylim(0, 1.7)
    ax.legend()
    save(fig, "tokenizer_fertility_comparison.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for lang, marker in zip(LANGS, ("o", "s")):
        for kind, style in (("bpe", "-"), ("unigram", "--")):
            sizes = [16384, 32768]
            vals = [stats[lang][f"{kind}_{v}"]["compression_ratio_chars_per_token"] for v in sizes]
            ax.plot(sizes, vals, style, marker=marker, label=f"{LABEL[lang]} {kind.upper() if kind == 'bpe' else 'Unigram'}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([16384, 32768], ["16,384", "32,768"])
    ax.set_xlabel("vocabulary size")
    ax.set_ylabel("characters per token (higher is better)")
    ax.set_title("Compression vs vocabulary size")
    ax.legend(fontsize=8)
    save(fig, "compression_vs_vocab_size.png")

    corpus = {lang: json.loads((ROOT / lang / "data" / "dataset_stats.json").read_text(encoding="utf-8"))
              for lang in LANGS}
    bins = load("phase1_bins.json")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, lang in enumerate(LANGS):
        d = corpus[lang]
        m, dl = d["manual_tokens"] / 1e6, d["downloaded_tokens"] / 1e6
        ax.bar(i, dl, color=KIND_COLOR["downloaded"], label="downloaded" if i == 0 else None)
        ax.bar(i, m, bottom=dl, color=KIND_COLOR["manual"], label="manual" if i == 0 else None)
        ax.text(i, dl + m + 8, f"{dl + m:.1f}M\n({d['manual_fraction']:.1%} manual)", ha="center", fontsize=8)
        train = bins[lang]["train_bytes"] / 2 / 1e6
        ax.plot([i - 0.4, i + 0.4], [train, train], "k:", lw=1.5,
                label="train.bin tokens (trained on)" if i == 0 else None)
    ax.set_xticks(range(len(LANGS)), [LABEL[lang] for lang in LANGS])
    ax.set_ylabel("tokens (millions)")
    ax.set_title("Corpus composition (dataset_stats.json) and training tokens")
    ax.set_ylim(0, 900)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "manual_vs_downloaded_tokens.png")

    for lang in LANGS:
        sources = sorted((s for s in corpus[lang]["sources"] if s["token_estimate"] > 0),
                         key=lambda s: s["token_estimate"])
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(sources) + 1.2))
        ax.barh([s["name"] for s in sources], [s["token_estimate"] / 1e6 for s in sources],
                color=[KIND_COLOR[s["type"]] for s in sources])
        for k, c in KIND_COLOR.items():
            ax.barh([], [], color=c, label=k)
        ax.set_xscale("log")
        ax.set_xlabel("tokens (millions, log scale)")
        ax.set_title(f"{LABEL[lang]} corpus by source")
        ax.legend(fontsize=8)
        save(fig, f"corpus_distribution_{lang}.png")


# ---------------------------------------------------------------- Phase 2

def phase2() -> None:
    logs = {m: load(f"phase2_train_log_{m}.json") for m in MODELS}
    tokens_per_step = 32768
    for lang in LANGS:
        log = logs[f"{lang}_v2"]
        steps = np.array([r["step"] for r in log])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps * tokens_per_step / 1e6, [r["loss"] for r in log], "--", label="train loss")
        ax.plot(steps * tokens_per_step / 1e6, [r["val_loss"] for r in log], "-", label="validation loss")
        best = min(log, key=lambda r: r["val_loss"])
        ax.annotate(f"best val {best['val_loss']:.3f} @ step {best['step']:,}",
                    (best["step"] * tokens_per_step / 1e6, best["val_loss"]), xytext=(-170, 40),
                    textcoords="offset points", arrowprops={"arrowstyle": "->"}, fontsize=8)
        ax.set_xlabel("training tokens (millions)")
        ax.set_ylabel("cross-entropy (nats)")
        ax.set_title(f"{LABEL[lang]} V2 pretraining (15,259 steps x 32,768 tokens)")
        ax.legend()
        save(fig, f"loss_curve_{lang}.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        for arch in ("v1", "v2"):
            log = logs[f"{lang}_{arch}"]
            ax.plot([r["step"] * tokens_per_step / 1e6 for r in log], [r["val_loss"] for r in log],
                    color=ARCH_COLOR[arch], label=f"{arch.upper()} (final {log[-1]['val_loss']:.3f})")
        ax.set_xlabel("training tokens (millions)")
        ax.set_ylabel("validation loss (nats)")
        ax.set_ylim(3.3, 6.0)
        ax.set_title(f"{LABEL[lang]}: Baseline V1 vs Modern V2")
        ax.legend()
        save(fig, f"loss_curve_comparison_{lang}.png")

    # attention entropy and distance of the pretrained models (layers 0, 4, 7; heads 0-3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in MODELS:
        rows = load(f"phase2_attention_summary_{m}.json")["rows"]
        layers = sorted({r["layer"] for r in rows})
        style = "-" if m.startswith("hindi") else "--"
        for ax, key in zip(axes, ("entropy", "mean_distance")):
            ax.plot(layers, [np.mean([r[key] for r in rows if r["layer"] == li]) for li in layers], style,
                    marker="o", color=ARCH_COLOR[m[-2:]], label=LABEL[m])
    for ax, title in zip(axes, ("attention entropy (nats)", "mean attention distance (tokens)")):
        ax.set_xlabel("layer")
        ax.set_xticks([0, 4, 7])
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Pretrained models, mean over heads 0-3 and the three example sentences")
    save(fig, "phase2_attention_entropy_distance.png")

    use_indic_fonts()
    for lang in LANGS:
        picks = [(0, 0), (0, 3), (7, 1), (7, 2)]
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        for ax, (layer, head) in zip(axes.flat, picks):
            ax.imshow(plt.imread(ROOT / lang / "eval" / "attention" / f"ex0_layer{layer}_head{head}.png"))
            ax.axis("off")
        fig.suptitle(f"{LABEL[lang]} V2, example 0: layer 0 heads 0 and 3 (top), layer 7 heads 1 and 2 (bottom)")
        fig.tight_layout()
        save(fig, f"attn_{lang}_panel.png")


# ---------------------------------------------------------------- Phase 3

def phase3() -> None:
    sweep = {**load("phase3_finetune_hindi.json"), **load("phase3_finetune_assamese.json")}
    hard = load("phase3_eval_hard.json")
    chance = None

    # test relation accuracy of every (model, mode, variant), mean and range over the two seeds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, mode in zip(axes, ("direct", "cot")):
        labels, means, lo, hi, colors = [], [], [], [], []
        for m in MODELS:
            for variant, seeds in sweep[m][mode]["by_variant"].items():
                accs = [s["finetuned"]["accuracy_relation"] for s in seeds.values()]
                chance = chance or next(iter(seeds.values()))["chance_baseline"]["majority_accuracy"]
                labels.append(f"{LABEL[m]}\n{variant}")
                means.append(np.mean(accs))
                lo.append(np.mean(accs) - min(accs))
                hi.append(max(accs) - np.mean(accs))
                colors.append(ARCH_COLOR[m[-2:]])
        ax.bar(range(len(labels)), means, yerr=[lo, hi], color=colors, capsize=3)
        ax.axhline(chance, color="k", ls=":", lw=1, label=f"majority baseline {chance:.3f}")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_title(f"{'CoT' if mode == 'cot' else 'Direct'} SFT")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("test relation accuracy (mean, range over 2 seeds)")
    axes[0].legend(fontsize=8)
    save(fig, "phase3_reasoning_accuracy_comparison.png")

    # depth generalisation of the kept checkpoints
    depths = [2, 3, 4, 5, 6, 8, 10]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in MODELS:
        for mode in ("direct", "cot"):
            job = hard[m][mode]["job"]
            variant, _mode, seed = job.rsplit("_", 2)
            d2 = sweep[m][mode]["by_variant"][variant][seed.replace("seed", "")]["extra_eval"]["test_depth2"]
            accs = [d2["accuracy_relation"]] + [hard[m][mode]["splits"][f"test_depth{d}"]["accuracy_relation"]
                                                for d in depths[1:]]
            ax.plot(depths, accs, MODE_STYLE[mode], marker="o", ms=3,
                    color=ARCH_COLOR[m[-2:]], alpha=1.0 if m.startswith("hindi") else 0.55,
                    label=f"{LABEL[m]} {mode}")
    ax.axhline(chance, color="k", ls=":", lw=1)
    ax.axvspan(1.5, 3.5, color="grey", alpha=0.12, label="hop counts seen in training")
    ax.set_xticks(depths)
    ax.set_xlabel("hops between the queried entities")
    ax.set_ylabel("relation accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Depth generalisation of the kept checkpoints")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "phase3_depth_generalisation.png")

    # harder splits
    splits = [("test", "main test"), ("test_branch", "branches"), ("test_distractor_attr", "other-attribute\npremises"),
              ("test_redundant", "redundant\nshortcuts")]
    fig, ax = plt.subplots(figsize=(10, 4))
    jobs = [(m, mode) for m in MODELS for mode in ("direct", "cot")]
    width = 0.8 / len(splits)
    for i, (split, name) in enumerate(splits):
        ax.bar(np.arange(len(jobs)) + (i - 1.5) * width,
               [hard[m][mode]["splits"][split]["accuracy_relation"] for m, mode in jobs], width, label=name)
    ax.axhline(chance, color="k", ls=":", lw=1)
    ax.set_xticks(range(len(jobs)), [f"{LABEL[m]}\n{mode}" for m, mode in jobs], fontsize=8)
    ax.set_ylabel("relation accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Kept checkpoints on the harder splits")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "phase3_hard_splits.png")

    # per-paradigm accuracy of the kept checkpoints on the main test
    paradigms = ("transitive", "multi_hop", "negation", "word_problem", "conversational")
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(paradigms)
    for i, p in enumerate(paradigms):
        vals = []
        for m, mode in jobs:
            variant, _mode, seed = hard[m][mode]["job"].rsplit("_", 2)
            ft = sweep[m][mode]["by_variant"][variant][seed.replace("seed", "")]["finetuned"]
            vals.append(ft["per_paradigm_accuracy_answer_only"][p])
        ax.bar(np.arange(len(jobs)) + (i - 2) * width, vals, width, label=p.replace("_", " "))
    ax.set_xticks(range(len(jobs)), [f"{LABEL[m]}\n{mode}" for m, mode in jobs], fontsize=8)
    ax.set_ylabel("answer accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-paradigm accuracy on the main test (kept checkpoints)")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "phase3_per_paradigm_breakdown.png")

    # attention before vs after finetuning
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for col, m in enumerate(MODELS):
        shift = load(f"phase3_attention_shift_{m}.json")["per_layer"]
        layers = range(len(shift["pretrained"]["entropy"]))
        for name, color in (("pretrained", "grey"), ("direct", "#C44E52"), ("cot", "#4C72B0")):
            if shift[name]["jsd_vs_pretrained"]:
                axes[0, col].plot(layers, shift[name]["jsd_vs_pretrained"], marker="o", color=color, label=name)
            axes[1, col].plot(layers, shift[name]["entity_mass_last"], marker="o", color=color, label=name)
        axes[0, col].set_title(LABEL[m])
        axes[1, col].set_xlabel("layer")
        axes[1, col].set_ylim(0, 1)
    axes[0, 0].set_ylabel("JSD from pretrained (nats)")
    axes[1, 0].set_ylabel("last-token attention\non entity names")
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Attention on 200 depth-4 reasoning prompts, mean over heads")
    fig.tight_layout()
    save(fig, "phase3_attention_shift.png")
    attention_change_map()
    entity_attention_dumbbell()


# Attention diagrams: colours from the dataviz reference palette (blue sequential ramp,
# blue <-> red diverging around a grey zero, categorical blue / orange for Direct / CoT).
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
MODE_COLOR = {"direct": "#2a78d6", "cot": "#eb6834"}
MODE_LABEL = {"direct": "Direct", "cot": "CoT"}


def attention_change_map() -> None:
    """Per layer and fine-tuned model: how far attention moved from the pretrained model
    (JSD), and how its entropy and mean distance changed (fine-tuned minus pretrained)."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    rows, labels = {"jsd": [], "entropy": [], "distance": []}, []
    for m in MODELS:
        shift = load(f"phase3_attention_shift_{m}.json")["per_layer"]
        base = shift["pretrained"]
        for mode in ("direct", "cot"):
            cur = shift[mode]
            rows["jsd"].append(cur["jsd_vs_pretrained"])
            rows["entropy"].append(np.subtract(cur["entropy"], base["entropy"]))
            rows["distance"].append(np.subtract(cur["mean_distance"], base["mean_distance"]))
            labels.append(f"{LABEL[m]} · {MODE_LABEL[mode]}")
    sequential = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
    diverging = LinearSegmentedColormap.from_list("blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948"])
    panels = (
        ("jsd", "Change from pretrained (JSD, nats)", sequential, None, "{:.2f}"),
        ("entropy", "Δ entropy (nats)\nblue = sharper", diverging, "sym", "{:+.2f}"),
        ("distance", "Δ mean distance (tokens)\nblue = more local", diverging, "sym", "{:+.1f}"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, (key, title, cmap, mode, fmt) in zip(axes, panels):
        data = np.array(rows[key], dtype=float)
        if mode == "sym":
            lim = float(np.abs(data).max())
            norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        else:
            norm = plt.Normalize(0.0, float(data.max()))
        mesh = ax.pcolormesh(data, cmap=cmap, norm=norm, edgecolors="white", linewidth=1.5)
        for (r, c), v in np.ndenumerate(data):
            r_, g_, b_, _ = cmap(norm(v))
            light = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ > 0.55
            ax.text(c + 0.5, r + 0.5, fmt.format(v), ha="center", va="center", fontsize=7,
                    color=INK if light else "white")
        ax.set_xticks(np.arange(data.shape[1]) + 0.5, range(data.shape[1]))
        ax.set_xlabel("layer", color=INK_2)
        ax.set_title(title, fontsize=10, color=INK)
        ax.tick_params(length=0, colors=INK_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y in (2, 4, 6):  # one band per pretrained model
            ax.axhline(y, color="white", linewidth=4)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.02)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    axes[0].set_yticks(np.arange(len(labels)) + 0.5, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Where fine-tuning changes attention: 200 depth-4 prompts, mean over heads", color=INK)
    fig.tight_layout()
    save(fig, "phase3_attention_change_map.png")


def entity_attention_dumbbell() -> None:
    """Share of the last prompt token's final-layer attention on entity names,
    pretrained vs each fine-tuned model."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    offsets = {"direct": -0.14, "cot": 0.14}
    for i, m in enumerate(MODELS):
        shift = load(f"phase3_attention_shift_{m}.json")["per_layer"]
        base = shift["pretrained"]["entity_mass_last"][-1]
        for mode, dy in offsets.items():
            v, y = shift[mode]["entity_mass_last"][-1], i + dy
            ax.plot([base, v], [y, y], color=MODE_COLOR[mode], linewidth=2, solid_capstyle="round", zorder=1)
            ax.scatter([v], [y], s=64, color=MODE_COLOR[mode], edgecolors="white", linewidths=2, zorder=3,
                       label=MODE_LABEL[mode] if i == 0 else None)
            ax.text(v + (0.025 if v >= base else -0.025), y, f"{v:.2f}", va="center",
                    ha="left" if v >= base else "right", fontsize=8, color=INK)
        ax.scatter([base] * 2, [i + dy for dy in offsets.values()], s=64, facecolors="white", edgecolors=MUTED,
                   linewidths=2, zorder=2, label="pretrained" if i == 0 else None)
        ax.text(base, i - 0.3, f"{base:.2f}", va="center", ha="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(MODELS)), [LABEL[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("final-layer attention of the last prompt token on entity names", color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, colors=INK_2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    ax.set_title("Direct fine-tuning points the answer position at the names; CoT less so", fontsize=10, color=INK,
                 pad=24)
    fig.tight_layout()
    save(fig, "phase3_entity_attention_dumbbell.png")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, fn in (("Phase 1", phase1), ("Phase 2", phase2), ("Phase 3", phase3)):
        print(name)
        fn()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
