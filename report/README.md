# Reports, Results & Figures

Reports for the **Language Models and Agents (Monsoon 2026)** individual project:

* **[Phase 1 report](phase1_report.md)**: language choice, corpus collection, cleaning pipeline, tokenizer bake-off, case studies, split sizes.
* **[Phase 2 report](phase2_report.md)**: Baseline V1 and Modern V2 architectures, 500M-token pretraining, perplexity and bits per byte, generation quality, attention heatmaps.
* **[Phase 3 report](phase3_report.md)**: Direct vs CoT reasoning fine-tuning, depth and harder splits, leakage analysis, error analysis, attention before vs after fine-tuning.
* **[Final report](final_report.md)** ([PDF](final_report.pdf), source [`final_report.tex`](final_report.tex)): synthesis across the three phases.
* **[Bonus report](bonus_report.md)**: positional-embedding ablation (on the earlier 1,907-step pretraining runs).

Every number in the Phase 1–3 and final reports comes from the files in **[`results/`](results/)** ([provenance](results/README.md)), and every figure in **[`figures/`](figures/)** is drawn from them by `python -m report.make_figures` (except the bonus ablation figure). The PDF is built with `xelatex final_report.tex` from this directory.
