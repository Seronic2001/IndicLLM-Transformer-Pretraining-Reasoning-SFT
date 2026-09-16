# Technical Reports & Visualizations (Phases 1, 2, and 3)

This directory contains the comprehensive technical reports, statistical summaries, publication-quality figures, and evaluation matrices for the **Language Models and Agents (Monsoon 2026)** Individual Project.

* **[Phase 3 Technical Report (Markdown)](phase3_report.md)**: Full 35-mark report detailing anti-leakage relational reasoning, target-only prompt masking, 8-model benchmark matrix (V1 vs. V2 $\times$ Direct SFT vs. CoT), 6-paradigm breakdown, 4-layer attention redistribution (Layers 0, 2, 5, 7), and qualitative diagnostic taxonomy.
* **[Phase 2 Technical Report (Markdown)](phase2_report.md)**: Full 40-mark report covering 16K transformer architectures (Baseline V1-16K vs Modern V2-16K, ~25M params), 500M-token pretraining convergence, held-out test PPL & BPB, generative quality diagnostics, and cross-layer attention pattern analysis.
* **[Phase 1 Technical Report (Markdown)](phase1_report.md)**: Full 25-mark report covering language selection, corpus collection, Indic normalizer, deduplication, tokenizer bake-off, and case studies.
* **[Figures Directory (figures/)](figures/)**:
  * *Phase 3 Figures*: Reasoning answer accuracy (`phase3_reasoning_accuracy_comparison.png`), multi-tier token $F_1$ (`phase3_multi_tier_f1_comparison.png`), character similarity & decomposed scores (`phase3_char_similarity_and_decomp.png`), standalone per-paradigm breakdowns (`phase3_per_paradigm_hindi.png`, `phase3_per_paradigm_assamese.png`), and empirical 4-layer attention heatmaps (`phase3_pretrain_vs_finetune_attention_hindi.png`, `phase3_pretrain_vs_finetune_attention_assamese.png`).
  * *Phase 2 Figures*: Pretraining loss curves (`loss_curve_hindi.png`, `loss_curve_assamese.png`), architecture comparisons (`loss_curve_comparison_hindi.png`, `loss_curve_comparison_assamese.png`), and cross-layer attention specialization heatmaps (`attn_hindi_panel.png`, `attn_assamese_panel.png`).
  * *Phase 1 Figures*: Corpus distributions, manual collection fraction, tokenizer fertility, and compression curves.

