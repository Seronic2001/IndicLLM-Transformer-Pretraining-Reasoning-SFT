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
| Training Tokens (bins)   | 714,015,795                | 717,422,008                |
| Subword Fertility        | 1.2776 tokens / word       | 1.4313 tokens / word       |
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
  * Contains: Clean deduplicated JSONL files, `assamese.model`, `assamese.vocab`, `train.bin` (1.4 GB), `val.bin`, `test.bin`.

## 🧠 Phase 2 (Model Implementation, Pretraining & Evaluation) — branch `phase-2`

Phase 2 focuses on pretraining and evaluating **16K vocabulary transformer architectures** (~25M parameters, strictly compliant with the 22.5M–27.5M rubric window), directly pairing with the 16K BPE tokenizers selected in Phase 1:
* **Baseline V1-16K** (Hand-written Pre-LN, learned pos, GELU, 8 layers, 24.98M params)
* **Modern V2-16K** (Enhanced Architecture with RoPE, SwiGLU, RMSNorm, 8 layers, 25.17M params)

Both architectures were pretrained for 15,259 steps × 32,768 tokens (500M tokens) per language. **Modern V2** has the lower held-out perplexity in both languages, by 5–7% (details in [`report/phase2_report.md`](report/phase2_report.md)):

| Language | Architecture | Best Val Loss | Test Loss (512-token windows) | Test PPL | Test BPB |
|---|---|---|---|---|---|
| Hindi (Model H) | Baseline V1 | 3.6260 | 3.6547 | 38.66 | 0.4856 |
| Hindi (Model H) | **Modern V2** | **3.5775** | **3.5876** | **36.15** | **0.4767** |
| Assamese (Model L) | Baseline V1 | 3.9966 | 3.9987 | 54.53 | 0.4634 |
| Assamese (Model L) | **Modern V2** | **3.9496** | **3.9425** | **51.55** | **0.4569** |

* **Checkpoints:** outputs of the Kaggle runs `seronic2001/lma-pretrain-{hindi,assamese}-v1` and `shubhadeepmandal/lma-pretrain-{hindi,assamese}-v2` (`best.pt`, `train_log.json`); evaluated by `shubhadeepmandal/lma-eval-pretrained`. The older `lma-phase2-artifacts` dataset holds earlier 1,907-step runs, not these models.
* **Tokenizers:** `hindi/tokenizer/hindi.model` and `assamese/tokenizer/assamese.model`.
* **Evaluation Reproduction:**
```bash
python -m hindi.eval.evaluate --checkpoint <hindi-best.pt> \
  --model-config hindi/configs/model_H_v2.yaml --tokenizer hindi/tokenizer/hindi.model \
  --test-bin <test.bin> --arch v2
python -m assamese.eval.attention_analysis --checkpoint <assamese-best.pt> \
  --model-config assamese/configs/model_L_v2.yaml --tokenizer assamese/tokenizer/assamese.model --arch v2
```

---

## 🎯 Phase 3 (Reasoning Finetuning, Attention Analysis & Final Report) — branch `phase-3`

Phase 3 fine-tunes V1 and V2 in each language on synthetic relational reasoning (answers computed by a symbolic solver), comparing **Direct SFT** with **Chain-of-Thought SFT** over five recipes and two seeds (44 runs). Full analysis in [`report/phase3_report.md`](report/phase3_report.md).

* **Anti-leakage**: disjoint train / val / test name pools, shuffled premises, question relation drawn independently of the answer, no training query beyond 3 hops.
* **Target-only loss** with prompts masked; checkpoints selected on the relation accuracy of generated validation answers.
* **Harder splits** (`--hard-eval`): 6-, 8- and 10-hop chains, irrelevant branches, premises about another attribute, redundant shortcuts.
* **Leakage check** ([`common/leakage_check.py`](common/leakage_check.py)): the main test is structurally covered by training (a lookup scores 0.95), while the 4–10-hop splits are not (lookup at chance), so those splits carry the evidence of reasoning.

### 📊 Kept checkpoints (relation accuracy; majority baseline 0.46–0.53)

| Model | Direct: test | Direct: 4 / 6 / 10 hops | Direct: other-attribute premises | CoT: test | CoT: 10 hops |
|---|:---:|:---:|:---:|:---:|:---:|
| Hindi V1 | **0.961** | 0.974 / 0.960 / 0.846 | 0.736 | 0.799 | 0.130 |
| Hindi V2 | 0.909 | 0.952 / 0.922 / 0.856 | 0.734 | 0.716 | 0.634 |
| Assamese V1 | 0.833 | 0.874 / 0.726 / 0.296 | 0.628 | 0.704 | 0.380 |
| Assamese V2 | 0.689 | 0.746 / 0.742 / 0.662 | 0.664 | 0.498 | 0.362 |

Direct beats CoT, V1 beats V2 (the reverse of their perplexity ranking), the best recipes are realmix (half real training names) for Hindi and a 5,000-step CoT warm-up for Assamese Direct, and premises about a different attribute mislead every model.

### 📦 Phase 3 Runs
* `seronic2001/lma-finetune`, `seronic2001/lma-finetune-assamese`: the sweep; one `finetuned_<mode>.pt` per model and mode (best validation run).
* `seronic2001/lma-eval-hard`: the kept checkpoints on the standard and harder splits.
* `seronic2001/lma-eval-attention`: attention before vs after fine-tuning, and the Phase 2 heatmaps.

### 🚀 Phase 3 Reproduction Commands
```bash
# Reasoning data (defaults = the sweep's settings), with the depth and harder splits
python -m hindi.finetune.generate_reasoning --extra-eval --hard-eval
python -m assamese.finetune.generate_reasoning --extra-eval --hard-eval

# Fine-tune (defaults = the best Hindi recipe: Direct, lr 3e-5, cosine, up to 12,000 steps)
python -m hindi.finetune.finetune --pretrained-ckpt <hindi_v1_best.pt> \
    --model-config hindi/configs/model_H_16k.yaml --tokenizer hindi/tokenizer/hindi.model

# Score a checkpoint on any splits; leakage check; attention before vs after
python -m common.evaluate_finetuned --lang hindi --ckpt <finetuned_direct.pt> \
    --tokenizer hindi/tokenizer/hindi.model --data-dir hindi/finetune/reasoning --out scores.json
python -m common.leakage_check --data-dir hindi/finetune/reasoning --out leakage.json
python -m common.attention_shift --lang hindi --pretrained <hindi_v1_best.pt> \
    --finetuned direct=<finetuned_direct.pt> --tokenizer hindi/tokenizer/hindi.model \
    --data hindi/finetune/reasoning/test_depth4.jsonl --out-dir attention

# Redraw every report figure from report/results/
python -m report.make_figures
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
    --resume-path hindi/train/checkpoints_v2/ckpt_2500.pt
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

Results and commands: see the Phase 3 section above and [`report/phase3_report.md`](report/phase3_report.md).
```bash
# Both modes back to back for one model (architecture read from the model config)
python -m assamese.finetune.finetune --mode both \
    --pretrained-ckpt <assamese_v1_best.pt> --model-config assamese/configs/model_L_16k.yaml \
    --tokenizer assamese/tokenizer/assamese.model --out-dir assamese/finetune/out_v1 \
    --lr-schedule wsd --max-steps 15000 --cot-warmup-steps 5000
```

---

## 🖥️ Demo (in the browser)

One page, [`demo/static/`](demo/static/), is the whole demo: it runs every model in the visitor's browser with ONNX Runtime Web, and every host serves the same files. [`demo/app.py`](demo/app.py) is only the file server (FastAPI/uvicorn, which come with Gradio, so a `gradio` Hugging Face Space runs it as is); Railway builds the same folder from its [`Dockerfile`](demo/Dockerfile).

* **Generate**: the pretrained Modern V2 model of each language continues a prompt (Greedy, Balanced or Creative sampling, or set temperature, top-p and repetition penalty).
* **Reasoning**: all four kept fine-tunes of each language, Baseline V1 or Modern V2, each trained to answer Direct or with CoT (which writes the chain of comparisons first), answer a fresh comparison question (2–10 hops), checked against the symbolic solver. "Compare all" runs one question through every model, and the page shows each model's measured accuracy at the chosen hop depth (from the `lma-eval-hard` kernel).
* **Tokenizer**: the SentencePiece pieces and ids of any text, byte-fallback runs, tokens per word, and the token count under the other language's tokenizer.
* **Typing in English letters**: with the switch on, a romanised word (`bharat`, `oxom`) becomes native-script suggestions. [`translit.js`](demo/static/translit.js) generates the possible spellings and ranks them with the language's own tokenizer vocabulary (fewest, most frequent pieces), so `rajdhani` → राजधानी, `sarkar` → सरकार, `guwahati` → গুৱাহাটী. Space takes the first suggestion, 1–5 pick another, and Esc keeps the Latin.

Building it:

1. **Slim the checkpoints**: `python -m demo.build_space --out <dir> --checkpoint hindi_lm=<best.pt> hindi_v1_direct=<finetuned.pt> ...` writes fp16 weights + config (~50 MB each) to `<dir>/checkpoints/`.
2. **Build the site**: `python -m demo.static.build_static --checkpoints <dir>/checkpoints --out <site> [--scores eval_hard/summary.json]`. It exports each model as one cached generation step, stores the weights as fp16 (~64 MB; int8 garbled the reasoning models' copied names), and checks the result against PyTorch (max logit difference ≤ 6e-5, same greedy tokens). `lm.js` re-implements the BPE tokenizer and matches SentencePiece exactly; `test_static.mjs` checks this and the models in Node.
3. **Package for a host**: `python -m demo.build_space --out <host> --site <site>` puts the server, README, Dockerfile and the site (under `site/`) in `<host>`. Run it locally with `pip install -r requirements.txt` and `python app.py` (port 7860).
   * **Hugging Face Space**: push `<host>` to a `gradio` Space. The `seronic2001/lma-export-demo` notebook ([`notebooks/kaggle_eval/lma-export-demo/`](notebooks/kaggle_eval/lma-export-demo/), git-ignored) builds from the trained checkpoints on Kaggle and uploads, using the Kaggle secrets `HF_TOKEN` and `HF_SPACE`.
   * **Railway**: `railway up` in `<host>`. The ONNX files are over Railway's upload limit, so `.railwayignore` leaves them out and the Docker build downloads them (`python app.py --fetch-models <url>`) from the [`demo-weights-v1`](https://github.com/Seronic2001/IndicLLM-Transformer-Pretraining-Reasoning-SFT/releases/tag/demo-weights-v1) release.

---

## 📑 Technical Reports
* **Course Bonus Technical Report (Positional Embedding Ablation):** **[`report/bonus_report.md`](report/bonus_report.md)** — Complete 500M-token ablation removing positional embeddings ($E_{pos}=0$) in Hindi 16K Transformer (~24.78M params). Documents validation loss $4.08$–$4.12$ nats against $4.125$ for Baseline V1 (no measurable effect), held-out loss $4.1129$ nats (PPL $61.13$, 511-token windows), generation metrics within small margins of V1, and the explanation: the causal mask lets a decoder infer token order without explicit positional input. Public dataset: [`shubhadeepmandal/lma-bonus-artifacts`](https://www.kaggle.com/datasets/shubhadeepmandal/lma-bonus-artifacts).
* **Final Consolidated Report (100 Marks):** **[`report/final_report.md`](report/final_report.md)** ([PDF: `report/final_report.pdf`](report/final_report.pdf)): synthesis of the three phases and the four comparative questions.
* **Phase 3 Technical Report (35 Marks):** **[`report/phase3_report.md`](report/phase3_report.md)**: fine-tuning sweep, depth and harder splits, leakage analysis, error analysis, attention before vs after fine-tuning.
* **Phase 2 Technical Report (40 Marks):** **[`report/phase2_report.md`](report/phase2_report.md)**: architectures, pretraining curves, perplexity and bits per byte, generation quality with samples, attention heatmaps with Indic labels.
* **Phase 1 Technical Report (25 Marks):** **[`report/phase1_report.md`](report/phase1_report.md)**: corpora, cleaning pipeline, tokenizer bake-off, case studies, split sizes.
* **Result files:** [`report/results/`](report/results/), the outputs every number and figure is read from ([provenance](report/results/README.md)).
