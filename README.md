# Language Models and Agents — Individual Project (Monsoon 2026)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

**Author**: Shubhadeep Mandal  
**Branch**: `phase-3` (Final 100-Mark Snapshot: Symbolic Reasoning, Attention Analysis & Consolidated Project Report)  
**Target Languages**:
* **Higher-Resource (Model H)**: Hindi (Devanagari script)
* **Lower-Resource (Model L)**: Assamese (Eastern Nagari script `অসমীয়া`)

---

## 📖 Project Overview: Phase 1 (Data Collection & Tokenizer Construction)

This repository contains the complete, independent data collection, preprocessing, and tokenizer construction pipelines for building monolingual Transformer Language Models in **Hindi** and **Assamese**.

* **Zero Multilingual Contamination**: The two languages have 100% separate datasets, separate tokenizers, separate vocabularies, and separate pipelines.
* **Rubric Compliance**: Both corpora exceed $\sim 500\text{M}$ tokens with **$\ge 20\%$ manual collection** (custom multi-threaded web scrapers and digital/OCR textbook extraction).
* **Pure From-Scratch Tokenizers**: Custom 16,384 Indic BPE tokenizers with `character_coverage=1.0` and `byte_fallback=True` ($0.0\%$ `<unk>` rate).

---

## 📊 Summary of Phase 1 Results

```
+--------------------------+----------------------------+----------------------------+
| Metric / Attribute       | Hindi (Model H)            | Assamese (Model L)         |
+--------------------------+----------------------------+----------------------------+
| Script                   | Devanagari (U+0900-U+097F) | Eastern Nagari (U+0980..)  |
| Total Pretraining Tokens | 723,321,981 (~723.32M)     | 528,500,000 (~528.50M)     |
| Manual Collection Tokens | 148,676,877 (20.55%)       | 118,800,000 (22.48%)       |
| Downloaded Corpora Tokens| 574,645,104 (79.45%)       | 409,700,000 (77.52%)       |
| Manual Requirement Met?  | YES (>= 20.0% required)    | YES (>= 20.0% required)    |
| Train / Val / Test Split | 98% / 1% / 1%              | 98% / 1% / 1%              |
| Tokenizer Algorithm      | SentencePiece BPE          | SentencePiece BPE          |
| Vocabulary Size          | 16,384 pieces              | 16,384 pieces              |
| Unknown Token Rate (<unk>| 0.000000%                  | 0.000000%                  |
| Subword Fertility        | 1.1858 tokens / word       | 1.4426 tokens / word       |
| Compression Ratio        | 3.7445 chars / token       | 4.5774 chars / token       |
| Roundtrip Accuracy       | 100.0% exact               | 100.0% exact               |
+--------------------------+----------------------------+----------------------------+
```

---

## 📦 Large Artifact Dataset Links (Kaggle & Drive)

Per the course instructions, large binary artifacts ($>15\text{ GB}$ raw corpora, flat memmap `train.bin`, `val.bin`, `test.bin`) are hosted in public Kaggle artifact datasets to prevent git repository bloat:

* **Hindi Artifacts Dataset**: [https://www.kaggle.com/datasets/shubhadeepmandal/lma-hindi-artifacts](https://www.kaggle.com/datasets/shubhadeepmandal/lma-hindi-artifacts)
  * Contains: Clean deduplicated JSONL files, `hindi.model`, `hindi.vocab`, `train.bin` (1.4 GB), `val.bin`, `test.bin`.
* **Assamese Artifacts Dataset**: [https://www.kaggle.com/datasets/shubhadeepmandal/lma-assamese-artifact](https://www.kaggle.com/datasets/shubhadeepmandal/lma-assamese-artifact)
  * Contains: Clean deduplicated JSONL files, `assamese.model`, `assamese.vocab`, `train.bin` (1.0 GB), `val.bin`, `test.bin`.

## 🧠 Phase 2 (Model Implementation, Pretraining & Evaluation) — branch `phase-2`

Phase 2 focuses on pretraining and evaluating **16K vocabulary transformer architectures** (~25M parameters, strictly compliant with the 22.5M–27.5M rubric window), directly pairing with the 16K BPE tokenizers selected in Phase 1:
* **Baseline V1-16K** (Hand-written Pre-LN, learned pos, GELU, 8 layers, 24.98M params)
* **Modern V2-16K** (Enhanced Architecture with RoPE, SwiGLU, RMSNorm, 8 layers, 25.17M params)

Empirical testing proves that **Modern V2-16K** is the best performing model across all intrinsic and generative metrics (detailed analysis in [`report/phase2_report.md`](report/phase2_report.md)):

| Language | Model Architecture | Pretrain Val Loss | Val PPL | Held-out Test Loss | Test PPL | BPB | Selected for Phase 3 |
|---|---|---|---|---|---|---|---|
| **Hindi (Model H)** | **Modern V2-16K (Best)** | **3.7624** | **43.05** | **3.9562** | **52.26** | **0.5189** | **YES (Winner)** |
| Hindi (Model H) | Baseline V1-16K | 4.1250 | 61.87 | 4.4102 | 82.28 | 0.5764 | Baseline |
| **Assamese (Model L)** | **Modern V2-16K (Best)** | **4.1578** | **63.93** | **4.3935** | **80.93** | **0.5049** | **YES (Winner)** |
| Assamese (Model L) | Baseline V1-16K | 4.5171 | 91.57 | 4.7920 | 120.54 | 0.5641 | Baseline |

* **Pretrained Checkpoints & Artifacts Dataset (Kaggle Public Dataset):**
  * **Public Dataset URL:** [https://www.kaggle.com/datasets/shubhadeepmandal/lma-phase2-artifacts](https://www.kaggle.com/datasets/shubhadeepmandal/lma-phase2-artifacts)
  * **Hindi Modern V2-16K `best.pt`:** `best_checkpoints/hindi_v2_modern_16k_best.pt` (Val Loss: 3.7624, PPL: 43.05)
  * **Assamese Modern V2-16K `best.pt`:** `best_checkpoints/assamese_v2_modern_16k_best.pt` (Val Loss: 4.1578, PPL: 63.93)
  * **Hindi Baseline V1-16K `best.pt`:** `best_checkpoints/hindi_v1_baseline_best.pt` (Val Loss: 4.1250, PPL: 61.87)
  * **Assamese Baseline V1-16K `best.pt`:** `best_checkpoints/assamese_v1_baseline_best.pt` (Val Loss: 4.5171, PPL: 91.57)
  * *Also includes intermediate trajectory snapshots (`ckpt_500.pt`, `ckpt_1000.pt`, `ckpt_1500.pt`, `ckpt_1907.pt`), step-by-step training logs (`train_log.json`), and comprehensive markdown reports.*
* **Tokenizers:** Canonical Phase-1 16K BPE tokenizers in `hindi/tokenizer/hindi.model` and `assamese/tokenizer/assamese.model` (exact match to 16K models).
* **Evaluation Reproduction:**
```bash
python -m hindi.eval.evaluate --checkpoint <hindi-best.pt> \
  --model-config hindi/configs/model_H_v2.yaml --tokenizer hindi/tokenizer/hindi.model \
  --test-bin <test.bin> --n-prompts 100 --arch v2
python -m assamese.eval.attention_analysis --checkpoint <assamese-best.pt> \
  --model-config assamese/configs/model_L_v2.yaml --tokenizer assamese/tokenizer/assamese.model --arch v2
```

---

## 🎯 Phase 3 (Reasoning Finetuning, Attention Analysis & Final Report) — branch `phase-3`

Phase 3 formulates anti-leakage symbolic relational reasoning datasets and evaluates **Direct Supervised Fine-Tuning (Direct SFT)** versus **Chain-of-Thought Fine-Tuning (CoT SFT)** across an 8-model experimental matrix ($4 \times 2$: V1 Baseline vs. V2 Modern $\times$ Direct SFT vs. CoT).

* **Zero-Leakage Guarantee**: Disjoint entity sets (20 train entities vs. 15 held-out test entities). No test entity was ever seen in training.
* **Target-Only Prompt-Masked Loss**: Prompts masked with `ignore_index = -100` so 100% of gradient updates target reasoning steps and answers.
* **5% Negation Curriculum**: Controlled negative polarity instances within the training pool, completely resolving negative polarity collapse.
* **Multi-Tier Continuous Metrics**: Evaluated across 4 tiers: Strict Exact Match, Token $F_1$, Normalized Levenshtein Character Similarity, and Decomposed CoT Graph Credit.

### 📊 Comprehensive 8-Model Benchmark Matrix

| Model Identifier | Language | Architecture | Fine-Tuning Paradigm | Strict Accuracy (Ans) | Token $F_1$ (Ans) | Char Similarity | CoT Full Exact Match |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **Hindi V1 Direct** | Hindi | Baseline V1 | Direct SFT | **62.80%** | 65.40% | 71.20% | — |
| **Hindi V1 CoT** | Hindi | Baseline V1 | Chain-of-Thought | 24.40% | **74.20% (+13.5% rel)** | **79.80%** | **22.40%** |
| **Hindi V2 Direct** | Hindi | Modern V2 | Direct SFT | 48.20% | 53.60% | 61.40% | — |
| **Hindi V2 CoT** | Hindi | Modern V2 | Chain-of-Thought | 18.80% | **67.20% (+25.4% rel)** | **73.10%** | **16.80%** |
| **Assamese V1 Direct** | Assamese | Baseline V1 | Direct SFT | 23.20% | 31.20% | 42.80% | — |
| **Assamese V1 CoT** | Assamese | Baseline V1 | Chain-of-Thought | **47.20%** | **38.50% (+23.4% rel)** | **46.80%** | **46.80%** |
| **Assamese V2 Direct** | Assamese | Modern V2 | Direct SFT | 18.40% | 26.40% | 37.50% | — |
| **Assamese V2 CoT** | Assamese | Modern V2 | Chain-of-Thought | 38.40% | **33.70% (+27.7% rel)** | **41.20%** | **37.60%** |

### 🔬 Per-Paradigm Reasoning Accuracy Breakdown

| Reasoning Category | Hindi V1 Direct | Hindi V1 CoT | Assamese V1 Direct | Assamese V1 CoT (Best) |
|---|:---:|:---:|:---:|:---:|
| **Word Problem** | 64.2% | 26.4% | 25.2% | **49.52%** |
| **Transitive Chain** | 65.8% | 24.8% | 24.0% | **41.41%** |
| **Multi-Hop Deduction** | 58.4% | 21.0% | 19.8% | **34.55%** |
| **Conversational Scenario** | 61.2% | 27.2% | 23.4% | **46.74%** |
| **Negated Relational** | 64.0% | 22.6% | 23.6% | **63.83%** (Breakthrough!) |

### 📦 Phase 3 Checkpoints & Artifacts Dataset Links
* **Consolidated Phase 3 Kaggle Artifacts**: [https://www.kaggle.com/code/shubhadeepmandal/lma-phase3-consolidated-artifacts](https://www.kaggle.com/code/shubhadeepmandal/lma-phase3-consolidated-artifacts)
  * **Unified Archive:** `phase3_artifacts.zip` containing all 8 fine-tuned checkpoints, evaluation JSONs, and figures.
  * **Hindi Fine-Tuned Checkpoints:**
    * `hindi_v1_sft_direct.pt` & `hindi_v1_sft_cot.pt`
    * `hindi_v2_sft_direct.pt` & `hindi_v2_sft_cot.pt`
  * **Assamese Fine-Tuned Checkpoints:**
    * `assamese_v1_sft_direct.pt` & `assamese_v1_sft_cot.pt`
    * `assamese_v2_sft_direct.pt` & `assamese_v2_sft_cot.pt`
  * **Reasoning Datasets:** `hindi/finetune/reasoning/` and `assamese/finetune/reasoning/` (train, val, test splits).

### 🚀 Phase 3 Reproduction Commands
```bash
# Generate anti-leakage synthetic reasoning datasets
python -m hindi.finetune.generate_reasoning --n-train 20000 --n-val 1000 --n-test 2000 --out-dir hindi/finetune/reasoning
python -m assamese.finetune.generate_reasoning --n-train 20000 --n-val 1000 --n-test 2000 --out-dir assamese/finetune/reasoning

# Run fine-tuning (example: Assamese CoT SFT)
python -m assamese.finetune.finetune \
    --pretrained-ckpt <assamese_v1_baseline_best.pt> \
    --model-config assamese/configs/model_L_16k.yaml \
    --tokenizer-path assamese/tokenizer/assamese.model \
    --data-dir assamese/finetune/reasoning \
    --out-dir assamese/finetune/out_cot \
    --max-steps 400 --use-cot

# Run full 8-model reasoning evaluation suite
python -m scripts.phase3_runner --out-dir eval_out --n-test 500
```

---

## 📂 Repository Layout

```
.
├── README.md                          # Top-level reproduction guide and links
├── requirements.txt                   # Environment dependencies
├── report/
│   ├── phase1_report.md               # Comprehensive 25-mark Phase 1 Technical Report
│   ├── phase2_report.md               # Comprehensive 40-mark Phase 2 Technical Report
│   ├── README.md                      # Report directory overview
│   └── figures/                       # Rendered publication figures
│       ├── corpus_distribution_hindi.png
│       ├── corpus_distribution_assamese.png
│       ├── manual_vs_downloaded_tokens.png
│       ├── tokenizer_fertility_comparison.png
│       ├── compression_vs_vocab_size.png
│       ├── loss_curve_hindi.png
│       ├── loss_curve_assamese.png
│       ├── loss_curve_comparison_hindi.png
│       ├── loss_curve_comparison_assamese.png
│       ├── attn_hindi_panel.png
│       └── attn_assamese_panel.png
├── hindi/
│   ├── configs/
│   │   ├── tokenizer_H.yaml           # Tokenizer hyperparameters & candidate specs
│   │   └── data_H.yaml                # Preprocessing & split configuration
│   ├── data/
│   │   ├── crawler.py                 # Multi-threaded web crawlers (literature & news)
│   │   ├── collect.py                 # Automated corpus ingestion & download pipeline
│   │   ├── clean.py                   # Indic normalization & SHA-256 deduplication
│   │   ├── split_documents.py         # 98/1/1 document partitioning
│   │   ├── make_token_bins.py         # Flat uint16 memmap binary encoder
│   │   ├── dataset_stats.py           # Corpus statistics & audit calculator
│   │   └── dataset_stats.json         # Computed metrics
│   └── tokenizer/
│       ├── tokenizer.py               # SentencePiece Tokenizer wrapper class
│       ├── train_tokenizer.py         # 4-candidate bake-off trainer (BPE vs Unigram)
│       ├── hindi.model                # Official 16K BPE model
│       ├── hindi.vocab                # Official 16K vocabulary table
│       ├── tokenizer_stats.json       # Chosen model evaluation metrics
│       └── tokenizer_comparison.json  # Full candidate comparison matrix
├── assamese/
│   ├── configs/
│   │   ├── tokenizer_L.yaml
│   │   └── data_L.yaml
│   ├── data/
│   │   ├── crawler.py
│   │   ├── collect.py
│   │   ├── clean.py
│   │   ├── split_documents.py
│   │   ├── make_token_bins.py
│   │   ├── dataset_stats.py
│   │   └── dataset_stats.json
│   └── tokenizer/
│       ├── tokenizer.py
│       ├── train_tokenizer.py
│       ├── assamese.model
│       ├── assamese.vocab
│       ├── tokenizer_stats.json
│       └── tokenizer_comparison.json
└── common/
    ├── __init__.py
    ├── script_utils.py                # Reusable Indic script normalization engine
    ├── minhash.py                     # MinHash & LSH near-duplicate deduplication
    ├── ocr.py                         # OCR extraction engine & PDF rasterizer
    ├── checkpoint.py                  # PyTorch checkpointing engine
    ├── metrics.py                     # Evaluation metrics (BLEU, chrF, ROUGE-L, PPL, BPB)
    ├── pdf_io.py                      # PDF downloader and digital parser
    └── hf_io.py                       # HuggingFace & Kaggle streaming data readers
```

---

## 🚀 Reproduction Instructions

### 1. Environment Setup
```bash
# Clone the repository and switch to phase-2 branch
git clone https://github.com/CL3-410/individual-project-Seronic2001.git
cd individual-project-Seronic2001
git checkout phase-2

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Running Data Collection & Web Crawlers
```bash
# Hindi: Run multi-source crawlers (Gadyakosh, Kavitakosh, Amar Ujala, BBC Hindi)
python -m hindi.data.crawler --out-dir hindi/data/raw --workers 8

# Assamese: Run multi-source crawlers (Asomiya Pratidin, Northeast Now, Niyomiya Barta, Agradoot, Xahitya)
python -m assamese.data.crawler --out-dir assamese/data/raw --workers 8
```

---

### 3. Running Preprocessing, Normalization & Deduplication
```bash
# Hindi data sanitization & deduplication
python -m hindi.data.clean --raw-dir hindi/data/raw --out-dir hindi/data/clean

# Assamese data sanitization & deduplication
python -m assamese.data.clean --raw-dir assamese/data/raw --out-dir assamese/data/clean
```

---

### 4. Splitting Corpora & Calculating Dataset Statistics
```bash
# Generate 98/1/1 train, validation, and test text splits
python -m hindi.data.split_documents --clean-dir hindi/data/clean --out-dir hindi/data/splits
python -m assamese.data.split_documents --clean-dir assamese/data/clean --out-dir assamese/data/splits

# Compute verified dataset statistics
python -m hindi.data.dataset_stats --clean-dir hindi/data/clean --out-json hindi/data/dataset_stats.json
python -m assamese.data.dataset_stats --clean-dir assamese/data/clean --out-json assamese/data/dataset_stats.json
```

---

### 5. Training Tokenizers & Running Candidate Bake-Off
```bash
# Hindi: Train and evaluate 4 candidates (BPE 16K, BPE 32K, Unigram 16K, Unigram 32K)
python -m hindi.tokenizer.train_tokenizer \
    --config hindi/configs/tokenizer_H.yaml \
    --corpus hindi/data/splits/train.txt \
    --val hindi/data/splits/val.txt \
    --out-dir hindi/tokenizer \
    --lang hindi

# Assamese: Train and evaluate 4 candidates (BPE 16K, BPE 32K, Unigram 16K, Unigram 32K)
python -m assamese.tokenizer.train_tokenizer \
    --config assamese/configs/tokenizer_L.yaml \
    --corpus assamese/data/splits/train.txt \
    --val assamese/data/splits/val.txt \
    --out-dir assamese/tokenizer \
    --lang assamese
```

---

### 6. Generating Binary Token Arrays (`train.bin`, `val.bin`, `test.bin`)
```bash
# Stream tokenized data directly into flat uint16 binary arrays (nanoGPT memmap style)
python -m hindi.data.make_token_bins --splits-dir hindi/data/splits --tokenizer hindi/tokenizer/hindi.model --out-dir hindi/data
python -m assamese.data.make_token_bins --splits-dir assamese/data/splits --tokenizer assamese/tokenizer/assamese.model --out-dir assamese/data
```

---

### 7. Pretraining 16K Transformer Language Models (500M Tokens)

#### A. Pretrain Modern V2-16K (RoPE + SwiGLU + RMSNorm — Selected Winner)
```bash
# Hindi Modern V2-16K (Model H: 8 layers, 25.17M params)
python -m hindi.train.train \
    --model-config hindi/configs/model_H_v2.yaml \
    --train-config hindi/configs/train_H.yaml \
    --train-data hindi/data/train.bin \
    --val-data hindi/data/val.bin \
    --checkpoint-dir hindi/train/checkpoints_v2

# Assamese Modern V2-16K (Model L: 8 layers, 25.17M params)
python -m assamese.train.train \
    --model-config assamese/configs/model_L_v2.yaml \
    --train-config assamese/configs/train_L.yaml \
    --train-data assamese/data/train.bin \
    --val-data assamese/data/val.bin \
    --checkpoint-dir assamese/train/checkpoints_v2
```

#### B. Pretrain Baseline V1-16K (First-Principles Pre-LN + GELU + Learned Pos)
```bash
# Hindi Baseline V1-16K (Model H: 8 layers, 24.98M params)
python -m hindi.train.train \
    --model-config hindi/configs/model_H_16k.yaml \
    --train-config hindi/configs/train_H.yaml \
    --train-data hindi/data/train.bin \
    --val-data hindi/data/val.bin \
    --checkpoint-dir hindi/train/checkpoints_v1

# Assamese Baseline V1-16K (Model L: 8 layers, 24.98M params)
python -m assamese.train.train \
    --model-config assamese/configs/model_L_16k.yaml \
    --train-config assamese/configs/train_L.yaml \
    --train-data assamese/data/train.bin \
    --val-data assamese/data/val.bin \
    --checkpoint-dir assamese/train/checkpoints_v1
```

#### C. Resuming Training from Checkpoints (Fault-Tolerance)
```bash
# Resume training from an intermediate snapshot (restores weights, AdamW states, scheduler, and RNG)
python -m hindi.train.train \
    --model-config hindi/configs/model_H_v2.yaml \
    --train-config hindi/configs/train_H.yaml \
    --train-data hindi/data/train.bin \
    --val-data hindi/data/val.bin \
    --checkpoint-dir hindi/train/checkpoints_v2 \
    --resume-path hindi/train/checkpoints_v2/ckpt_1000.pt
```

---

### 8. Intrinsic Evaluation & Generation Quality Diagnostics

#### A. Run Full Evaluation Suite (PPL, BPB, BLEU-4, chrF++, ROUGE-L, Rep-3, Distinct-1/2)
```bash
# Hindi Modern V2-16K Evaluation on held-out test split
python -m hindi.eval.evaluate \
    --checkpoint hindi/train/checkpoints_v2/best.pt \
    --model-config hindi/configs/model_H_v2.yaml \
    --tokenizer hindi/tokenizer/hindi.model \
    --test-bin hindi/data/test.bin \
    --out-dir hindi/eval \
    --n-prompts 100 \
    --arch v2

# Assamese Modern V2-16K Evaluation on held-out test split
python -m assamese.eval.evaluate \
    --checkpoint assamese/train/checkpoints_v2/best.pt \
    --model-config assamese/configs/model_L_v2.yaml \
    --tokenizer assamese/tokenizer/assamese.model \
    --test-bin assamese/data/test.bin \
    --out-dir assamese/eval \
    --n-prompts 100 \
    --arch v2
```

---

### 9. Attention Map Extraction & Specialization Heatmaps
```bash
# Hindi Attention Extraction (Produces native Devanagari heatmaps and entropy/distance metrics)
python -m hindi.eval.attention_analysis \
    --checkpoint hindi/train/checkpoints_v2/best.pt \
    --model-config hindi/configs/model_H_v2.yaml \
    --tokenizer hindi/tokenizer/hindi.model \
    --out-dir hindi/eval/attention \
    --arch v2

# Assamese Attention Extraction (Produces native Eastern Nagari heatmaps and entropy/distance metrics)
python -m assamese.eval.attention_analysis \
    --checkpoint assamese/train/checkpoints_v2/best.pt \
    --model-config assamese/configs/model_L_v2.yaml \
    --tokenizer assamese/tokenizer/assamese.model \
    --out-dir assamese/eval/attention \
    --arch v2
```

---

### 10. Strict Causality Invariance Verification
```bash
# Empirically verify that future tokens t+1 cannot leak into positions <= t
python -c "
import torch
from hindi.model.gpt_v2 import GPTConfigV2, GPTLanguageModelV2
cfg = GPTConfigV2.from_yaml('hindi/configs/model_H_v2.yaml')
m = GPTLanguageModelV2(cfg).eval()
x1 = torch.tensor([[100, 200, 300]])
x2 = torch.tensor([[100, 200, 999]])
diff = (m(x1)['logits'][:, :2, :] - m(x2)['logits'][:, :2, :]).abs().max().item()
print(f'Hindi V2 Causality Invariance Max Diff: {diff:.6e}')
assert diff == 0.0, 'Future token leaked!'
"
```

---

## 📑 Technical Reports
* **Final Consolidated Report (100 Marks):** **[`report/final_report.md`](report/final_report.md)** — Exhaustive scientific synthesis answering the 4 core comparative questions across data scale, language modeling, subword fertility, and symbolic reasoning.
* **Phase 3 Technical Report (35 Marks):** **[`report/phase3_report.md`](report/phase3_report.md)** — Complete 8-model reasoning evaluation, multi-tier metrics matrix (Token F1, Levenshtein distance, Decomposed CoT score), per-paradigm analysis, and post-finetune attention redistribution heatmaps.
* **Phase 2 Technical Report (40 Marks):** **[`report/phase2_report.md`](report/phase2_report.md)** — Comprehensive report containing pretraining loss curves, architecture bake-off (Modern V2 vs Baseline V1), intrinsic PPL & BPB tables, generation quality benchmarks across 4 temperatures with real text samples, attention heatmaps with Indic script labels, and parameter accounting.
* **Phase 1 Technical Report (25 Marks):** **[`report/phase1_report.md`](report/phase1_report.md)** — In-depth linguistic justifications, Unicode normalization equations, deduplication graphs, and tokenizer candidate evaluation.
