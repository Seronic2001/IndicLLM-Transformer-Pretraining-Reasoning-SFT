# Language Models and Agents — Individual Project (Monsoon 2026)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

**Author**: Shubhadeep Mandal (Roll No: 2025201056)  
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

* **Zero-Leakage Guarantee**: Disjoint entity pools (236 training names plus ~30 abstract symbols vs. 15 held-out names: 7 validation, 8 test). No held-out entity was ever seen in training.
* **Target-Only Prompt-Masked Loss**: Prompts masked with `ignore_index = -100` so 100% of gradient updates target reasoning steps and answers.
* **Negation Curriculum**: Negated premises make up about 18% of training examples, all within the training pool; fine-tuned models reach 50.6–55.4% (Hindi) and 8.4–36.1% (Assamese) strict accuracy on held-out negation queries.
* **Multi-Tier Continuous Metrics**: Evaluated across 4 tiers: Strict Exact Match, Token $F_1$, Normalized Levenshtein Character Similarity, and Decomposed CoT Graph Credit.

### 📊 Comprehensive 8-Model Benchmark Matrix (Hardened Multi-Hop & Anti-Leakage Suite)

| Model Identifier | Language | Architecture | Fine-Tuning Paradigm | Strict Accuracy (Ans) | Decision Stance Acc | Token $F_1$ (Ans) | Char Similarity | CoT Decomp Score |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Hindi V1 Direct** | Hindi | Baseline V1 | Direct SFT | **44.00%** | 38.00% | **83.30%** | **85.10%** | — |
| **Hindi V1 CoT** | Hindi | Baseline V1 | Chain-of-Thought | 40.60% | 40.40% | 81.89% | 83.72% | 50.80% |
| **Hindi V2 Direct** | Hindi | Modern V2 | Direct SFT | 42.40% | **40.80%** | 82.27% | 84.19% | — |
| **Hindi V2 CoT** | Hindi | Modern V2 | Chain-of-Thought | 38.20% | 38.00% | 82.04% | 84.51% | **51.64%** |
| **Assamese V1 Direct** | Assamese | Baseline V1 | Direct SFT | **29.40%** | 34.60% | **66.78%** | **75.36%** | — |
| **Assamese V1 CoT** | Assamese | Baseline V1 | Chain-of-Thought | 24.60% | **34.80%** | 65.16% | 72.90% | **39.52%** |
| **Assamese V2 Direct** | Assamese | Modern V2 | Direct SFT | 21.00% | 30.40% | 56.33% | 66.36% | — |
| **Assamese V2 CoT** | Assamese | Modern V2 | Chain-of-Thought | 11.80% | 29.20% | 50.65% | 59.29% | 30.65% |

### 🔬 Per-Paradigm Reasoning Accuracy Breakdown (Strict Ans / Decision Stance)

#### 🇮🇳 Hindi (Model H, Devanagari)
| Reasoning Category | Hindi V1 Direct | Hindi V1 CoT | Hindi V2 Direct | Hindi V2 CoT |
|---|:---:|:---:|:---:|:---:|
| **Conversational Scenario** | 48.9% / 44.3% | 45.5% / 44.3% | **51.1% / 47.7%** | 44.3% / 47.7% |
| **Indeterminate Component** | 0.0% / 0.0% | 0.0% / **12.2%** | 0.0% / **12.2%** | 0.0% / 0.0% |
| **Multi-Hop Deduction** | **47.9%** / 38.3% | 43.6% / 41.5% | 43.6% / 37.2% | 44.7% / **41.5%** |
| **Negated Relational** | 50.6% / 50.6% | 50.6% / 50.6% | **55.4% / 56.6%** | 53.0% / 55.4% |
| **Transitive Chain** | **56.4% / 47.4%** | 47.4% / 43.6% | 47.4% / 41.0% | 39.7% / 38.5% |
| **Word Problem** | **55.4%** / 43.4% | 51.8% / 47.0% | 51.8% / 47.0% | 42.2% / 39.8% |

#### 🌿 Assamese (Model L, Eastern Nagari)
| Reasoning Category | Assamese V1 Direct | Assamese V1 CoT | Assamese V2 Direct | Assamese V2 CoT |
|---|:---:|:---:|:---:|:---:|
| **Conversational Scenario** | **34.1% / 42.0%** | 30.7% / 36.4% | 28.4% / 37.5% | 14.8% / 30.7% |
| **Indeterminate Component** | 0.0% / 8.1% | 0.0% / 8.1% | 0.0% / **36.5%** | 4.1% / 17.6% |
| **Multi-Hop Deduction** | **36.2%** / 37.2% | **36.2% / 44.7%** | 21.3% / 21.3% | 8.5% / 30.9% |
| **Negated Relational** | **36.1%** / 42.2% | 28.9% / **48.2%** | 24.1% / 32.5% | 8.4% / 30.1% |
| **Transitive Chain** | **33.3% / 38.5%** | 17.9% / 30.8% | 21.8% / 25.6% | 14.1% / 35.9% |
| **Word Problem** | **32.5% / 36.1%** | 28.9% / **36.1%** | 27.7% / 30.1% | 20.5% / 28.9% |

### 📦 Phase 3 Checkpoints & Artifacts Dataset Links
* **Consolidated Phase 3 Kaggle Artifacts**: [https://www.kaggle.com/datasets/shubhadeepmandal/lma-phase3-artifact](https://www.kaggle.com/datasets/shubhadeepmandal/lma-phase3-artifact)
  * **Artifact Directory:** `phase3_artifacts/` containing all 8 fine-tuned checkpoints (`checkpoints/`), evaluation JSONs, and figures (`figures/`).
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

### 11. Phase 3: Symbolic Reasoning (Direct SFT vs. Chain-of-Thought)

#### 8-Model Benchmark Matrix (Tiers 1–4 Metrics: Hardened Multi-Hop Suite)
```
+------------------------------------------------------------------------------------------------------------------+
| Model Variant        | Language  | SFT Mode | Ans Acc  | Stance   | Graph Valid | Token F1 | Char Sim | CoT Decomp |
+------------------------------------------------------------------------------------------------------------------+
| Hindi V1 Direct      | Hindi     | Direct   | 44.00%   | 38.00%   | —           | 83.30%   | 85.10%   | —          |
| Hindi V1 CoT         | Hindi     | CoT      | 40.60%   | 40.40%   | 9.40%       | 81.89%   | 83.72%   | 50.80%     |
| Hindi V2 Direct      | Hindi     | Direct   | 42.40%   | 40.80%   | —           | 82.27%   | 84.19%   | —          |
| Hindi V2 CoT         | Hindi     | CoT      | 38.20%   | 38.00%   | 6.40%       | 82.04%   | 84.51%   | 51.64%     |
| Assamese V1 Direct   | Assamese  | Direct   | 29.40%   | 34.60%   | —           | 66.78%   | 75.36%   | —          |
| Assamese V1 CoT      | Assamese  | CoT      | 24.60%   | 34.80%   | 1.80%       | 65.16%   | 72.90%   | 39.52%     |
| Assamese V2 Direct   | Assamese  | Direct   | 21.00%   | 30.40%   | —           | 56.33%   | 66.36%   | —          |
| Assamese V2 CoT      | Assamese  | CoT      | 11.80%   | 29.20%   | 3.00%       | 50.65%   | 59.29%   | 30.65%     |
+------------------------------------------------------------------------------------------------------------------+
```

#### Reproduce Reasoning Data Generation & Fine-Tuning
```bash
# 1. Generate 20,000 synthetic reasoning examples with disjoint splits
python -m hindi.finetune.generate_reasoning --out-dir hindi/finetune/reasoning
python -m assamese.finetune.generate_reasoning --out-dir assamese/finetune/reasoning

# 2. Run Direct SFT & CoT SFT on Hindi
python -m hindi.finetune.finetune --mode direct --arch v1 --epochs 3
python -m hindi.finetune.finetune --mode cot --arch v1 --epochs 3
python -m hindi.finetune.finetune --mode direct --arch v2 --epochs 3
python -m hindi.finetune.finetune --mode cot --arch v2 --epochs 3

# 3. Run Direct SFT & CoT SFT on Assamese
python -m assamese.finetune.finetune --mode direct --arch v1 --epochs 3
python -m assamese.finetune.finetune --mode cot --arch v1 --epochs 3
python -m assamese.finetune.finetune --mode direct --arch v2 --epochs 3
python -m assamese.finetune.finetune --mode cot --arch v2 --epochs 3
```

---

## 📑 Technical Reports
* **Final Consolidated Report (100 Marks):** **[`report/final_report.md`](report/final_report.md)** — Exhaustive scientific synthesis answering the 4 core comparative questions across data scale, language modeling, subword fertility, and symbolic reasoning.
* **Phase 3 Technical Report (35 Marks):** **[`report/phase3_report.md`](report/phase3_report.md)** — Complete 8-model reasoning evaluation, multi-tier metrics matrix (Token F1, Levenshtein distance, Decomposed CoT score), per-paradigm analysis, and post-finetune attention redistribution heatmaps.
* **Phase 2 Technical Report (40 Marks):** **[`report/phase2_report.md`](report/phase2_report.md)** — Comprehensive report containing pretraining loss curves, architecture bake-off (Modern V2 vs Baseline V1), intrinsic PPL & BPB tables, generation quality benchmarks across 4 temperatures with real text samples, attention heatmaps with Indic script labels, and parameter accounting.
* **Phase 1 Technical Report (25 Marks):** **[`report/phase1_report.md`](report/phase1_report.md)** — In-depth linguistic justifications, Unicode normalization equations, deduplication graphs, and tokenizer candidate evaluation.
