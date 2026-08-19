# Language Models and Agents — Individual Project (Monsoon 2026)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q6gOCxoh)

**Author**: Shubhadeep Mandal  
**Branch**: `phase-1` (Phase 1 Deliverables Submission)  
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
| Total Pretraining Tokens | 624,145,435 (~624.15M)     | 528,500,000 (~528.50M)     |
| Manual Collection Tokens | 135,012,475 (21.63%)       | 118,214,000 (22.37%)       |
| Downloaded Corpora Tokens| 489,132,960 (78.37%)       | 410,286,000 (77.63%)       |
| Manual Requirement Met?  | YES (>= 20.0% required)    | YES (>= 20.0% required)    |
| Train / Val / Test Split | 98% / 1% / 1%              | 98% / 1% / 1%              |
| Tokenizer Algorithm      | SentencePiece BPE          | SentencePiece BPE          |
| Vocabulary Size          | 16,384 pieces              | 16,384 pieces              |
| Unknown Token Rate (<unk>| 0.000000%                  | 0.000000%                  |
| Subword Fertility        | 1.2776 tokens / word       | 1.4313 tokens / word       |
| Compression Ratio        | 3.7445 chars / token       | 4.5774 chars / token       |
| Roundtrip Accuracy       | 100.0% exact               | 100.0% exact               |
+--------------------------+----------------------------+----------------------------+
```

---

## 📦 Large Artifact Dataset Links (Kaggle & Drive)

Per the course instructions, large binary artifacts ($>15\text{ GB}$ raw corpora, flat memmap `train.bin`, `val.bin`, `test.bin`) are hosted in public Kaggle artifact datasets to prevent git repository bloat:

* **Hindi Artifacts Dataset**: [https://www.kaggle.com/datasets/shubhadeepmandal/lma-hindi-artifacts](https://www.kaggle.com/datasets/shubhadeepmandal/lma-hindi-artifacts)
  * Contains: Clean deduplicated JSONL files, `hindi.model`, `hindi.vocab`, `train.bin` (1.2 GB), `val.bin`, `test.bin`.
* **Assamese Artifacts Dataset**: [https://www.kaggle.com/datasets/shubhadeepmandal/lma-assamese-artifact](https://www.kaggle.com/datasets/shubhadeepmandal/lma-assamese-artifact)
  * Contains: Clean deduplicated JSONL files, `assamese.model`, `assamese.vocab`, `train.bin` (1.0 GB), `val.bin`, `test.bin`.
* **Sanitized Source Corpora**:
  * [Hindi Clean Corpora](https://www.kaggle.com/datasets/shubhadeepmandal/lma-clean-hindi-artifacts)
  * [Assamese Clean Corpora](https://www.kaggle.com/datasets/shubhadeepmandal/lma-clean-assamese-artifacts)

---

## 📂 Repository Layout

```
.
├── README.md                          # Top-level reproduction guide and links
├── requirements.txt                   # Environment dependencies
├── report/
│   ├── phase1_report.md               # Comprehensive 25-mark Phase 1 Technical Report
│   ├── README.md                      # Report directory overview
│   └── figures/                       # Rendered publication figures
│       ├── corpus_distribution_hindi.png
│       ├── corpus_distribution_assamese.png
│       ├── manual_vs_downloaded_tokens.png
│       ├── tokenizer_fertility_comparison.png
│       └── compression_vs_vocab_size.png
├── hindi/
│   ├── configs/
│   │   ├── tokenizer_H.yaml           # Tokenizer hyperparameters & candidate specs
│   │   └── data_H.yaml                # Preprocessing & split configuration
│   ├── data/
│   │   ├── crawler.py                 # Multi-threaded web crawlers (literature & news)
│   │   ├── download_corpora.py        # Curated public dataset fetchers
│   │   ├── ocr_extract.py             # NCERT PDF digital extraction & Tesseract OCR
│   │   ├── clean_pipeline.py          # Indic normalization & SHA-256 deduplication
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
│   │   ├── download_corpora.py
│   │   ├── ocr_extract.py
│   │   ├── clean_pipeline.py
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
    ├── indic_normalizer.py            # Reusable Indic script normalization engine
    └── text_utils.py                  # Character coverage & script regex utilities
```

---

## 🚀 Reproduction Instructions

### 1. Environment Setup
```bash
# Clone the repository and switch to phase-1 branch
git clone https://github.com/Language-Models-and-Agents-2026/individual-project-Seronic2001.git
cd individual-project-Seronic2001
git checkout phase-1

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
python -m hindi.data.clean_pipeline --raw-dir hindi/data/raw --out-dir hindi/data/clean

# Assamese data sanitization & deduplication
python -m assamese.data.clean_pipeline --raw-dir assamese/data/raw --out-dir assamese/data/clean
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

## 📑 Full Phase 1 Report
For the complete technical report with in-depth linguistic justifications, Unicode normalization equations, deduplication graphs, and tokenizer parameter trade-offs, see **[`report/phase1_report.md`](report/phase1_report.md)**.
