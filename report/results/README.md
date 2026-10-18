# Result files behind the reports

Every number in the Phase 1–3 and final reports is read from these files, and every figure in
`report/figures/` (except `ablation_no_pos_attention_heatmaps.png`, from the bonus run) is drawn
from them by `python -m report.make_figures`. The JSON files are copied unchanged from the
outputs of the Kaggle runs below, unless noted.

| File | Source |
|---|---|
| `phase1_tokenizer_stats_<lang>.json` | `shubhadeepmandal/lma-unify-<lang>` (2026-10-11): tokenizer candidates scored on 2,000 held-out documents; also in the artifact datasets |
| `phase1_bins.json` | sizes of `train.bin` / `val.bin` / `test.bin` in `shubhadeepmandal/lma-hindi-artifacts` and `lma-assamese-artifact` (written by hand from the dataset file listing) |
| `phase2_pretrained_eval.json` | `shubhadeepmandal/lma-eval-pretrained` (2026-10-12): test PPL / BPB, full-context PPL, generation metrics of the four pretrained models |
| `phase2_train_log_<model>.json` | `train_log.json` of `seronic2001/lma-pretrain-<lang>-v1` and `shubhadeepmandal/lma-pretrain-<lang>-v2` (2026-10-11) |
| `phase2_attention_summary_<model>.json` | `seronic2001/lma-eval-attention` (2026-10-19): `<lang>.eval.attention_analysis` on each pretrained `best.pt` |
| `phase3_finetune_<lang>.json` | `summary.json` of `seronic2001/lma-finetune` and `lma-finetune-assamese` (the 2-seed sweep, 2026-10-17) |
| `phase3_eval_hard.json` | `seronic2001/lma-eval-hard` (2026-10-18): kept checkpoints on the standard and harder splits |
| `phase3_attention_shift_<model>.json` | `seronic2001/lma-eval-attention` (2026-10-19): `common.attention_shift` |
| `phase3_leakage_<lang>.json` | `python -m common.leakage_check` run locally on data regenerated with `python -m <lang>.finetune.generate_reasoning --extra-eval --hard-eval` (the sweep's defaults) |

Corpus statistics come from `<lang>/data/dataset_stats.json`; the V2 evaluation outputs and Phase 2
heatmaps are also kept in `<lang>/eval/`.
