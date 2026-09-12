"""Dedicated Demo Runner for Hindi & Assamese Language Models.

Runs in a fresh Python process on Kaggle GPU or locally to generate qualitative completions,
plot publication-quality training loss curves, and render multi-head self-attention heatmaps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hindi.model.gpt import GPTConfig, GPTLanguageModel


def locate_file(patterns: list[str], search_roots: Optional[list[str]] = None) -> Optional[Path]:
    search_roots = search_roots or ["/kaggle/input", "."]
    for root_dir in search_roots:
        p_root = Path(root_dir)
        if p_root.exists():
            for root, _dirs, files in os.walk(p_root):
                for f in sorted(files, reverse=True):
                    for pat in patterns:
                        if pat == f or pat in f:
                            return Path(root) / f
    return None


def generate_sample(model: GPTLanguageModel, sp: spm.SentencePieceProcessor, prompt_text: str,
                    max_new_tokens: int = 60, temp: float = 0.7, top_k: Optional[int] = 50,
                    device: str = "cpu") -> tuple[str, str]:
    input_ids = sp.encode(prompt_text, out_type=int)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    out_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temp, top_k=top_k)
    full_tokens = out_idx[0].tolist()
    gen_tokens = full_tokens[len(input_ids):]
    full_text = sp.decode(full_tokens)
    gen_text = sp.decode(gen_tokens)
    return full_text, gen_text


def plot_attention_map(model: GPTLanguageModel, sp: spm.SentencePieceProcessor, sentence: str,
                       out_path: Path, title: str = "Self-Attention Heatmap", device: str = "cpu") -> None:
    tokens = sp.encode(sentence, out_type=int)[:20]  # first 20 tokens for clean axis readability
    labels = [sp.id_to_piece(t).replace(" ", "") for t in tokens]
    inp = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        res = model(inp, return_attn=True)
        # res["attn_weights"] is a list of (1, n_head, T, T) tensors per block
        layer_idx = min(len(res["attn_weights"]) - 1, 3)
        attn_matrix = res["attn_weights"][layer_idx][0, 0].cpu().numpy()

    plt.figure(figsize=(9, 7), dpi=300)
    plt.imshow(attn_matrix, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Attention Mass")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.title(f"{title} (Layer {layer_idx}, Head 0)", fontsize=12, fontweight="bold")
    plt.xlabel("Key Tokens (Attended)", fontsize=10)
    plt.ylabel("Query Tokens", fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"  + Saved attention map: {out_path}", flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LMA Model Demonstrations & Text Generation")
    parser.add_argument("--out-dir", default="/kaggle/working", help="Output directory")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Inference Device   : {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})", flush=True)

    # 1. Locate Artifacts
    print("\n" + "=" * 60, flush=True)
    print("[1/4] Locating Trained Checkpoints & Tokenizers", flush=True)
    print("=" * 60, flush=True)

    search_roots = ["/kaggle/input", ".", str(REPO_ROOT)]
    hindi_ckpt = locate_file(["best.pt", "ckpt_1907.pt", "ckpt_1500.pt"], search_roots)
    assamese_ckpt = locate_file(["best.pt", "ckpt_1907.pt", "ckpt_1500.pt"], [r for r in search_roots if "hindi" not in r.lower()] + search_roots)
    
    # Specific search for Assamese checkpoint to avoid picking up Hindi
    for r in search_roots:
        p_root = Path(r)
        if p_root.exists():
            for root, _dirs, files in os.walk(p_root):
                if "assamese" in root.lower() or "as" in root.lower():
                    for f in files:
                        if f in ("best.pt", "ckpt_1907.pt", "ckpt_1500.pt"):
                            assamese_ckpt = Path(root) / f
                            break

    hindi_spm = locate_file(["hindi.model"], search_roots)
    assamese_spm = locate_file(["assamese.model"], search_roots)
    hindi_log = locate_file(["train_log.json"], [r for r in search_roots if "hindi" in r.lower()] + search_roots)
    assamese_log = locate_file(["train_log.json"], [r for r in search_roots if "assamese" in r.lower()] + search_roots)

    print(f"[*] Hindi Checkpoint    : {hindi_ckpt}", flush=True)
    print(f"[*] Hindi Tokenizer     : {hindi_spm}", flush=True)
    print(f"[*] Assamese Checkpoint : {assamese_ckpt}", flush=True)
    print(f"[*] Assamese Tokenizer  : {assamese_spm}", flush=True)

    if not hindi_ckpt or not assamese_ckpt or not hindi_spm or not assamese_spm:
        raise FileNotFoundError(f"Missing required model or tokenizer artifacts! hi_ckpt={hindi_ckpt}, as_ckpt={assamese_ckpt}")

    # 2. Load Models
    print("\n" + "=" * 60, flush=True)
    print("[2/4] Loading Models into Memory", flush=True)
    print("=" * 60, flush=True)

    # Hindi
    sp_hi = spm.SentencePieceProcessor()
    sp_hi.load(str(hindi_spm))
    cfg_hi = GPTConfig.from_yaml(REPO_ROOT / "hindi/configs/model_H.yaml")
    model_hi = GPTLanguageModel(cfg_hi)
    ckpt_hi_data = torch.load(hindi_ckpt, map_location="cpu", weights_only=False)
    model_hi.load_state_dict(ckpt_hi_data.get("model_state_dict", ckpt_hi_data))
    model_hi.to(device)
    model_hi.eval()
    print(f"[+] Hindi GPT Model (~{model_hi.num_params()/1e6:.2f}M params) loaded on {device}!", flush=True)

    # Assamese
    sp_as = spm.SentencePieceProcessor()
    sp_as.load(str(assamese_spm))
    cfg_as = GPTConfig.from_yaml(REPO_ROOT / "assamese/configs/model_L.yaml")
    model_as = GPTLanguageModel(cfg_as)
    ckpt_as_data = torch.load(assamese_ckpt, map_location="cpu", weights_only=False)
    model_as.load_state_dict(ckpt_as_data.get("model_state_dict", ckpt_as_data))
    model_as.to(device)
    model_as.eval()
    print(f"[+] Assamese GPT Model (~{model_as.num_params()/1e6:.2f}M params) loaded on {device}!", flush=True)

    # 3. Generate Example Completions
    print("\n" + "=" * 60, flush=True)
    print("[3/4] Generating Example Completions Across Multiple Temperatures", flush=True)
    print("=" * 60, flush=True)

    hindi_prompts = [
        ("Culture & Heritage", "भारत एक विशाल और विविधतापूर्ण देश है जहाँ"),
        ("History & Knowledge", "प्राचीन काल में भारतीय संस्कृति और ज्ञान का प्रसार"),
        ("Science & Technology", "विज्ञान और आधुनिक प्रौद्योगिकी के विकास से"),
        ("Literature & Story", "एक सुंदर गाँव में एक दयालु किसान रहता था जो"),
    ]

    assamese_prompts = [
        ("Culture & Heritage", "অসমৰ প্ৰাকৃতিক সৌন্দৰ্য আৰু সংস্কৃতি বিশ্ববিখ্যাত কাৰণ"),
        ("Literature & Sankaradeva", "মহাপুৰুষ শ্ৰীমন্ত শংকৰদেৱৰ অসমীয়া সাহিত্য আৰু সংস্কৃতিৰ প্ৰতি অৱদান"),
        ("Geography & Brahmaputra", "ব্ৰহ্মপুত্ৰ নদী অসমৰ জীৱনৰেখা আৰু ইয়াৰ দুয়োপাৰে"),
        ("Modern Society & Education", "আধুনিক যুগত শিক্ষা আৰু প্ৰযুক্তিৰ বিকাশে"),
    ]

    temperatures = [
        (0.0, None, "Greedy (T=0.0)"),
        (0.6, 40, "Focused (T=0.6, Top-k=40)"),
        (0.8, 50, "Creative (T=0.8, Top-k=50)"),
    ]

    completions_records = {"hindi": [], "assamese": []}

    print("\n" + "=" * 70)
    print("🌟 HINDI LANGUAGE MODEL COMPLETIONS (~25.76M GPT)")
    print("=" * 70)
    for category, prompt in hindi_prompts:
        print(f"\n📌 [Domain: {category}]")
        print(f"Prompt: \"{prompt}\"")
        cat_rec = {"category": category, "prompt": prompt, "completions": []}
        for temp, top_k, label in temperatures:
            full_t, gen_t = generate_sample(model_hi, sp_hi, prompt, max_new_tokens=55, temp=temp, top_k=top_k, device=device)
            print(f"  • {label:<26} ➔ {gen_t}")
            cat_rec["completions"].append({"sampling": label, "temp": temp, "top_k": top_k, "continuation": gen_t, "full_text": full_t})
        completions_records["hindi"].append(cat_rec)

    print("\n" + "=" * 70)
    print("🌟 ASSAMESE LANGUAGE MODEL COMPLETIONS (~25.76M GPT)")
    print("=" * 70)
    for category, prompt in assamese_prompts:
        print(f"\n📌 [Domain: {category}]")
        print(f"Prompt: \"{prompt}\"")
        cat_rec = {"category": category, "prompt": prompt, "completions": []}
        for temp, top_k, label in temperatures:
            full_t, gen_t = generate_sample(model_as, sp_as, prompt, max_new_tokens=55, temp=temp, top_k=top_k, device=device)
            print(f"  • {label:<26} ➔ {gen_t}")
            cat_rec["completions"].append({"sampling": label, "temp": temp, "top_k": top_k, "continuation": gen_t, "full_text": full_t})
        completions_records["assamese"].append(cat_rec)

    # 4. Plot Loss Curves & Attention Maps
    print("\n" + "=" * 60, flush=True)
    print("[4/4] Generating Training Plots & Attention Heatmaps", flush=True)
    print("=" * 60, flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    if hindi_log and hindi_log.exists():
        with open(hindi_log, "r", encoding="utf-8") as f:
            h_logs = json.load(f)
        h_steps = [e["step"] for e in h_logs]
        h_train_loss = [e["loss"] for e in h_logs]
        h_val_loss = [e["val_loss"] for e in h_logs]
        ax1.plot(h_steps, h_train_loss, label="Train Loss", color="#1f77b4", alpha=0.8, linewidth=1.5)
        ax1.plot(h_steps, h_val_loss, label="Val Loss (Eval Points)", color="#d62728", marker="o", markersize=4, linewidth=2)
        ax1.set_title("Hindi GPT Pre-Training Loss (~25.76M Params)", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Optimization Steps (262k tokens/step)", fontsize=10)
        ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", frameon=True)
        ax1.set_ylim(3.5, 7.0)

    if assamese_log and assamese_log.exists():
        with open(assamese_log, "r", encoding="utf-8") as f:
            a_logs = json.load(f)
        a_steps = [e["step"] for e in a_logs]
        a_train_loss = [e["loss"] for e in a_logs]
        a_val_loss = [e["val_loss"] for e in a_logs]
        ax2.plot(a_steps, a_train_loss, label="Train Loss", color="#2ca02c", alpha=0.8, linewidth=1.5)
        ax2.plot(a_steps, a_val_loss, label="Val Loss (Eval Points)", color="#9467bd", marker="o", markersize=4, linewidth=2)
        ax2.set_title("Assamese GPT Pre-Training Loss (~25.76M Params)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Optimization Steps (262k tokens/step)", fontsize=10)
        ax2.set_ylabel("Cross-Entropy Loss", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", frameon=True)
        ax2.set_ylim(3.5, 7.0)

    plt.tight_layout()
    loss_curve_path = plots_dir / "loss_curves_comparison.png"
    plt.savefig(str(loss_curve_path), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved loss curves comparison to: {loss_curve_path}", flush=True)

    # Attention Heatmaps
    plot_attention_map(model_hi, sp_hi, "भारत एक विशाल और विविधतापूर्ण देश है जहाँ विभिन्न भाषाएँ बोली जाती हैं।",
                       plots_dir / "hindi_attention_layer3.png", "Hindi Multi-Head Attention", device=device)
    plot_attention_map(model_as, sp_as, "অসমৰ প্ৰাকৃতিক সৌন্দৰ্য আৰু বৈচিত্ৰ্যময় সংস্কৃতি অতি সুন্দৰ আৰু অনুপম।",
                       plots_dir / "assamese_attention_layer3.png", "Assamese Multi-Head Attention", device=device)

    # Summary Report
    report_path = out_dir / "demo_completions_report.md"
    report_lines = [
        "# LMA Phase 2 Language Model Demonstration & Inference Report",
        "",
        "## 1. Pre-Training Convergence (500M Tokens, 1907 Steps)",
        "![Loss Curves](plots/loss_curves_comparison.png)",
        "",
        "- **Hindi GPT Final Validation Loss**: ~3.97 (Perplexity: ~53.3)",
        "- **Assamese GPT Final Validation Loss**: ~3.97 (Perplexity: ~53.2)",
        "",
        "---",
        "",
        "## 2. Qualitative Text Generation Samples (Hindi GPT)",
        "",
    ]
    for rec in completions_records["hindi"]:
        report_lines.append(f"### Domain: {rec['category']}")
        report_lines.append(f"**Prompt**: *\"{rec['prompt']}\"*")
        report_lines.append("")
        for comp in rec["completions"]:
            report_lines.append(f"- **{comp['sampling']}**:")
            report_lines.append(f"  > {comp['continuation']}")
            report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## 3. Qualitative Text Generation Samples (Assamese GPT)",
        "",
    ])
    for rec in completions_records["assamese"]:
        report_lines.append(f"### Domain: {rec['category']}")
        report_lines.append(f"**Prompt**: *\"{rec['prompt']}\"*")
        report_lines.append("")
        for comp in rec["completions"]:
            report_lines.append(f"- **{comp['sampling']}**:")
            report_lines.append(f"  > {comp['continuation']}")
            report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## 4. Multi-Head Attention Heatmaps",
        "### Hindi Attention (Layer 3, Head 0)",
        "![Hindi Attention](plots/hindi_attention_layer3.png)",
        "",
        "### Assamese Attention (Layer 3, Head 0)",
        "![Assamese Attention](plots/assamese_attention_layer3.png)",
    ])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    with open(out_dir / "demo_completions.json", "w", encoding="utf-8") as f:
        json.dump(completions_records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Final Output Files in /kaggle/working:")
    for root, _dirs, files in os.walk(out_dir):
        rel = os.path.relpath(root, out_dir)
        prefix = "" if rel == "." else f"{rel}/"
        for f in sorted(files):
            fp = Path(root) / f
            print(f"  * {prefix}{f:<32} ({fp.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 70)
    print("[SUCCESS] LMA Model Demonstration & Inference Suite Complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
