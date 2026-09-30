# Comparative Pre-Training and Symbolic Reasoning Transfer in Monolingual Indic Transformers: A Comprehensive Synthesis Across High-Resource Hindi and Low-Resource Assamese

**Author**: Shubhadeep Mandal (Roll No: 2025201056)  
**Course**: Language Models and Agents (Monsoon 2026)  
**Submission Repository**: [github.com/CL3-410/individual-project-Seronic2001](https://github.com/CL3-410/individual-project-Seronic2001)  
**Branch**: `phase-3` (Final 100-Mark Snapshot)  
**Target Languages**:
* **Higher-Resource Language (Model H)**: Hindi (Devanagari script, Indo-Aryan family)
* **Lower-Resource Language (Model L)**: Assamese (Eastern Nagari script `অসমীয়া`, Indo-Aryan family)

---

## Executive Abstract

We design, build, pretrain, and evaluate two completely independent monolingual decoder-only Transformer Language Models (~25M parameters each) from scratch in PyTorch without any pretrained initialization or cross-lingual weight sharing. Through multi-source web crawling and digital OCR extraction of educational textbooks, we curated balanced **500M+ token corpora** for both languages, meeting the mandatory $\ge 20\%$ manual collection threshold. Custom 16,384-vocabulary SentencePiece BPE tokenizers achieve character coverage $>99.99\%$ with $0.0\%$ unknown token rate. Pretraining on 500M tokens across 1,907 optimizer steps yields smooth convergence and competitive held-out test perplexities ($52.26$ for Hindi V2, $80.93$ for Assamese V2). In Phase 3, we formulate an anti-leakage relational reasoning solver and conduct Supervised Fine-Tuning across an 8-model experimental matrix ($4 \times 2$: Baseline V1 vs. Modern V2 $\times$ Direct SFT vs. Chain-of-Thought). Fine-tuning lifts strict answer accuracy from about 0% zero-shot to 38.2–44.0% in Hindi and 11.8–29.4% in Assamese. Direct SFT keeps a small edge over Chain-of-Thought on strict accuracy and answer-level Token $F_1$ in every pair; Chain-of-Thought adds a scratchpad that can be verified (graph validity up to 9.4%, decomposed step score up to 51.6%). The resource gap between the two languages persists on every metric.

---

## 1. Project Synthesis & Full Experimental Trajectory

```
+---------------------------------------------------------------------------------------------------------+
|                                    Three-Phase Engineering Pipeline                                     |
+------------------------------------+------------------------------------+-------------------------------+
| Phase 1: Data & Tokenization       | Phase 2: Architecture & Pretraining| Phase 3: Symbolic Reasoning   |
| • 723M tokens (Hindi, 20.55% man)  | • Hand-written Decoder-Only GPT    | • Synthetic relational graphs |
| • 528M tokens (Assam, 22.48% man)  | • V1 Baseline (Pre-LN, GELU, pos)  | • Anti-leakage entity pools   |
| • MinHash LSH deduplication (s=0.6)| • V2 Modern (RMSNorm, SwiGLU, RoPE)| • Direct vs. CoT SFT          |
| • Custom 16K BPE Tokenizers        | • 500M pretraining budget (AdamW)  | • Multi-tier metric suite     |
| • Byte-fallback (<unk> = 0.0%)     | • Test PPL: 52.26 (HI), 80.93 (AS) | • Negation curriculum         |
+------------------------------------+------------------------------------+-------------------------------+
```

---

## 2. Phase-by-Phase Empirical Summary

### 2.1 Phase 1: Corpus Scale & Tokenization Diagnostics

Both languages satisfy the $\sim 500\text{M}$ token requirement with $>20\%$ manual collection (OCR of state board textbooks and multi-domain web scrapers):

| Metric / Dimension | Hindi (Model H) | Assamese (Model L) | Rubric Requirement |
| :--- | :---: | :---: | :---: |
| **Script Family** | Devanagari (`U+0900–U+097F`) | Eastern Nagari (`U+0980–U+09FF`) | Non-English Indian Languages |
| **Total Pretraining Tokens** | **723,321,981 (~723.32M)** | **528,500,000 (~528.50M)** | $\sim 500\text{M}$ target |
| **Manual Collection Fraction** | **148,676,877 (20.55%)** | **118,800,000 (22.48%)** | $\ge 20.0\%$ mandatory |
| **Deduplication Method** | MinHash LSH ($s=0.6$) | MinHash LSH ($s=0.6$) | Exact / near-dup removal |
| **Vocabulary Size** | 16,384 pieces | 16,384 pieces | $\ge 10\text{k}$ recommended |
| **Unknown Token Rate (`<unk>`)** | **0.000000%** | **0.000000%** | Byte fallback enabled |
| **Subword Fertility** | 1.1858 tokens / word | 1.4426 tokens / word | Optimal compression |
| **Characters per Token** | 3.7445 chars / token | 4.5774 chars / token | High morphological packing |

*Public Phase 1 Datasets*: Raw crawled text, OCR extractions, cleaned corpora, and 16K BPE models are archived in public Kaggle datasets: [`shubhadeepmandal/lma-hindi-artifacts`](https://www.kaggle.com/datasets/shubhadeepmandal/lma-hindi-artifacts) and [`shubhadeepmandal/lma-assamese-artifact`](https://www.kaggle.com/datasets/shubhadeepmandal/lma-assamese-artifact).

![Hindi Monolingual Corpus Source Distribution](figures/corpus_distribution_hindi_cropped.png)
![Assamese Monolingual Corpus Source Distribution](figures/corpus_distribution_assamese_cropped.png)
*Figure 2.1: Pretraining Corpus Source Composition (Pie Chart Breakdown) — Detailed source breakdowns across Hindi (723M tokens, 20.55% manual) and Assamese (528M tokens, 22.48% manual), fulfilling the mandatory $\ge 20\%$ manual collection threshold through curated web crawls and digital textbook OCR.*

![Subword Tokenizer Fertility Comparison](figures/tokenizer_fertility_comparison.png)
*Figure 2.2: Subword Fertility (Tokens per Word) across Vocabulary Sizes. At our chosen 16K vocabulary, Assamese requires $1.4426$ tokens/word compared to $1.1858$ for Hindi due to Eastern Nagari conjunct ligatures (যুক্তাক্ষৰ).*

*Analytical Contrast (Figure 2.1 vs. Figure 2.2)*: While both corpora surpass the 500M token threshold, the contrast between Figure 2.1 and Figure 2.2 reveals a critical structural divergence. Hindi benefited from high web density, yielding a low fertility rate ($1.1858$ tokens/word) where words map almost 1:1 to single subword pieces. Conversely, Assamese text contains dense multi-consonant clusters (e.g. ক্ষ, জ্ঞ, ত্ত) that frequently fracture into 2–3 subwords. Consequently, an identical 512-token context window spans ~431 words in Hindi but only ~354 words in Assamese (~21.6% shorter horizon), constraining multi-hop reasoning span.

---

### 2.2 Phase 2: Pretraining Dynamics & Continuous Language Modeling Benchmark

Pretraining was conducted on dedicated Nvidia GPUs with mixed precision fp16 (AMP) over 1,907 optimizer steps at 262,144 tokens per step (500M tokens total):

| Model Generation | Architecture Details | Hindi Test Loss | Hindi PPL | Hindi BPB | Assamese Test Loss | Assamese PPL | Assamese BPB |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Version 1.0 (Baseline LM)** | Pre-LN, GELU, Absolute Pos | 4.4102 | 82.28 | 0.5764 | 4.7920 | 120.54 | 0.5641 |
| **Version 2.0 (Modern LM)** | Pre-RMSNorm, SwiGLU, RoPE | **3.9562** | **52.26** | **0.5189** | **4.3935** | **80.93** | **0.5049** |
| **Architectural Gain ($\Delta$)** | SwiGLU + RoPE Advantage | **-0.4540** | **-36.5%** | **-0.0575** | **-0.3985** | **-32.9%** | **-0.0592** |

![Hindi 16K Pretraining Validation Loss Trajectory](figures/loss_curve_val_hindi_3way.png)
*Figure 2.3: Hindi Pretraining Dynamics (500M tokens, 1,907 steps) — Modern V2 (RMSNorm, SwiGLU, RoPE) vs. Baseline V1 (Pre-LN, GELU, Absolute Pos). Modern V2 achieves a 0.454 nat test loss reduction and 36.5% lower test perplexity ($52.26$ vs $82.28$).*

![Assamese 16K Pretraining Validation Loss Trajectory](figures/loss_curve_val_assamese_3way.png)
*Figure 2.4: Assamese Pretraining Dynamics (500M tokens, 1,907 steps) — Modern V2 achieves a 0.398 nat test loss reduction and 32.9% lower test perplexity ($80.93$ vs $120.54$) over Baseline V1.*

*Analytical Contrast (Figure 2.3 vs. Figure 2.4)*: Contrasting the loss trajectories across both languages demonstrates two key findings:
1. **Architectural Parity**: The architectural advantage of Modern V2 over Baseline V1 is remarkably invariant across scripts ($\Delta = -0.454$ nats / $-36.5\%$ PPL in Hindi; $\Delta = -0.398$ nats / $-32.9\%$ PPL in Assamese), verifying that gated SwiGLU projections and relative rotary embeddings generalize universally regardless of script morphological complexity.
2. **Orthographic Floor**: However, Assamese validation loss plateaus at a strictly higher baseline ($4.1578$ nats, Val PPL 63.93; Test Loss 4.3935, Test PPL 80.93) compared to Hindi ($3.7624$ nats, Val PPL 43.05; Test Loss 3.9562, Test PPL 52.26). This $\Delta \approx 28.67$ PPL gap is not underfitting; when normalized by UTF-8 byte density (BPB), Assamese actually compresses more efficiently ($0.5049$ BPB vs. $0.5189$ BPB in V2), proving that higher token cross-entropy reflects higher information density per subword unit.

### 2.3 Phase 3: Symbolic Reasoning Benchmark Matrix (8 Models)

Supervised fine-tuning on 20,000 synthetic reasoning examples, evaluated on 500 held-out test queries per model with strictly disjoint entity pools (numbers from `report/finetuning_eval.json`; "—" marks scratchpad metrics that do not apply to Direct SFT):

| Model Key | Language | Architecture | SFT Mode | Strict Acc (Ans) | Decision Acc | Graph Valid | Token F1 (Ans) | Char Sim | CoT Decomp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hindi V1 Direct** | Hindi | Baseline V1 | Direct | **44.00%** | 38.00% | — | **83.30%** | **85.10%** | — |
| **Hindi V1 CoT** | Hindi | Baseline V1 | CoT | 40.60% | 40.40% | **9.40%** | 81.89% | 83.72% | 50.80% |
| **Hindi V2 Direct** | Hindi | Modern V2 | Direct | 42.40% | **40.80%** | — | 82.27% | 84.19% | — |
| **Hindi V2 CoT** | Hindi | Modern V2 | CoT | 38.20% | 38.00% | 6.40% | 82.04% | 84.51% | **51.64%** |
| **Assamese V1 Direct**| Assamese| Baseline V1 | Direct | **29.40%** | 34.60% | — | **66.78%** | **75.36%** | — |
| **Assamese V1 CoT** | Assamese| Baseline V1 | CoT | 24.60% | **34.80%** | 1.80% | 65.16% | 72.90% | **39.52%** |
| **Assamese V2 Direct**| Assamese| Modern V2 | Direct | 21.00% | 30.40% | — | 56.33% | 66.36% | — |
| **Assamese V2 CoT** | Assamese| Modern V2 | CoT | 11.80% | 29.20% | **3.00%** | 50.65% | 59.29% | 30.65% |

![Phase 3 Answer Accuracy Comparison](figures/phase3_reasoning_accuracy_comparison.png)
*Figure 2.5: Direct SFT vs. Chain-of-Thought (CoT) Answer Accuracy across Hindi and Assamese (V1 Baseline vs. V2 Modern).*

![Multi-Tier Token F1 Comparison](figures/phase3_multi_tier_f1_comparison.png)
*Figure 2.6: Answer-level Token F1 for Direct SFT and Chain-of-Thought models.*

![Continuous Quality & Decomposed CoT Score](figures/phase3_char_similarity_and_decomp.png)
*Figure 2.7: Character similarity for all models and the decomposed CoT step score for CoT models.*

![Per-Paradigm CoT Reasoning Accuracy Breakdown: Hindi vs. Assamese](figures/phase3_per_paradigm_breakdown.png)
*Figure 2.8: Reasoning accuracy across the six paradigms (Conversational, Indeterminate, Multi-Hop, Negation, Transitive, Word Problem) under CoT SFT, Hindi (left) and Assamese (right). Exact per-paradigm values are tabulated in the Phase 3 report, Section 6.*

*Analytical Contrast across Phase 3 Figures*:
1. **Strict vs. Continuous Metrics (Figure 2.5 vs. Figures 2.6 & 2.7)**: Strict answer accuracy is 38.2–44.0% in Hindi and 11.8–29.4% in Assamese, while answer-level Token $F_1$ is 81.9–83.3% and 50.7–66.8%. The gap shows that models usually produce the right sentence frame and entities and slip on the relation word or an inflection. Direct SFT is slightly ahead of CoT on both strict accuracy and Token $F_1$ in every pair; CoT models additionally produce a scratchpad, scored by the decomposed step score (50.80% and 51.64% in Hindi, 39.52% and 30.65% in Assamese) and by solver-verified graph validity (9.4% and 6.4% in Hindi, 1.8% and 3.0% in Assamese).
2. **Cross-Language Paradigm Profile (Figure 2.8)**: Hindi is fairly even across the five solvable paradigms (39.7–56.4% strict accuracy) and scores 0% strict on indeterminate queries. Assamese is lower throughout (8.4–36.2%). On negated premises, decision stance reaches 50.6–56.6% in Hindi and 30.1–48.2% in Assamese; the best Assamese figure (48.2%, V1 CoT) is close to the corresponding Hindi one (50.6%).

---

## 3. Deep-Dive: The Four Core Synthesis Questions

### Question 1: How did data scale and quality differ between Model H and Model L?

1. **Corpus Abundance & Scrape Density**:
   * *Hindi (Model H)*: Enjoyed abundant web text from OSCAR, Wikipedia, mC4, and major regional news portals (Dainik Jagran, Amar Ujala). Crawling reached 723M raw tokens with a single-pass discard rate of ~18% during Unicode hygiene and MinHash deduplication.
   * *Assamese (Model L)*: Severely constrained public text availability. Standard web crawls contained extensive code-switching with Bengali and English. We deployed targeted multi-threaded crawlers for regional domains (e.g. Asomiya Pratidin, Assam Tribune) and OCR pipelines across SEBA/SCERT textbooks to collect 118.8M manual tokens. The discard rate was significantly higher (~32%), requiring aggressive script-filtering (`U+0980–U+09FF`) to preserve monolingual integrity.
2. **Quality vs. Noise Profile**:
   * Hindi data featured rich syntactic variety across journalistic, formal, and conversational domains.
   * Assamese text required heavy normalization of conjunct glyphs (যুক্তাক্ষৰ), archaic spellings, and digit conventions to prevent vocabulary fragmentation.

---

### Question 2: How do language-modeling and reasoning results compare across the two resource tiers?

1. **Pretraining Convergence Gap**:
   * Hindi reached a lower test perplexity ($52.26$) than Assamese ($80.93$). This $\Delta \approx 28.67$ PPL gap directly reflects the higher morphological complexity and conjunct ligature density in Eastern Nagari, where rare compound characters incur higher cross-entropy loss.
2. **Reasoning Acquisition Divergence**:
   * The ordering carries over to reasoning. Best strict answer accuracy is **44.0%** in Hindi (V1 Direct) against **29.4%** in Assamese (V1 Direct); best decision accuracy is 40.8% against 34.8%.
   * Direct SFT vs. Chain-of-Thought is a trade-off, not a win for CoT. Direct is ahead on strict accuracy in all four pairs (44.0 vs. 40.6, 42.4 vs. 38.2, 29.4 vs. 24.6, 21.0 vs. 11.8) and on answer Token $F_1$ in all four (83.3 vs. 81.9, 82.3 vs. 82.0, 66.8 vs. 65.2, 56.3 vs. 50.7). Decision accuracy is within about 3 points either way.
   * What CoT adds is a verifiable scratchpad: graph validity of 9.4% / 6.4% (Hindi V1 / V2) and 1.8% / 3.0% (Assamese), and decomposed step scores of 50.8% / 51.6% and 39.5% / 30.7%.
   * The gap between languages is larger on surface form than on the decision: Assamese trails Hindi by 15–26 points on strict accuracy but by 6–9 points on decision accuracy.

---

### Question 3: What tokenizer / corpus factors most affected the lower-resource model?

1. **Subword Fertility Differential**:
   * Assamese exhibited higher fertility ($1.4426$ tokens/word) compared to Hindi ($1.1858$ tokens/word). Each grammatical sentence in Assamese consumes $\sim 21.6\%$ more context window positions, reducing the effective temporal span of the 512-token context window.
2. **Byte Fallback Impact**:
   * By enforcing `byte_fallback=True` and `character_coverage=1.0`, zero `<unk>` tokens were produced during pretraining. However, rare Assamese ligatures (e.g. ক্ষ, ক্ত, জ্ঞ) occasionally split into multi-byte sequences, requiring multiple attention steps to parse a single semantic character.
3. **Curriculum Balance**:
   * Textbooks collected manually provided clean, structured domain knowledge with consistent grammatical case endings (`-ৰ`, `-ক`, `-ত`), which proved vital for downstream relational reasoning templates.

---

### Question 4: What evidence explains the observed differences?

We present four empirical pillars explaining the performance dynamics:

1. **Architecture Ordering Reverses Between Pretraining and Reasoning (V1 vs. V2)**:
   * V2 Modern is clearly better at language modelling (33–36% lower test perplexity in Phase 2), but V1 Baseline is equal or better at the reasoning task: 44.0% vs. 42.4% strict accuracy in Hindi (Direct) and 29.4% vs. 21.0% in Assamese (Direct).
   * One possible reason is that V1's absolute positional embeddings suit the rigid, fixed-layout reasoning templates, while RoPE's relative positions give no such anchor. This was not tested with an ablation.
2. **Attention Redistribution After Fine-Tuning (Section 3.2)**:
   * Pretrained and CoT-finetuned V2 checkpoints were compared on 200 held-out test examples per language, averaged over all 8 layers and 6 heads, with prompt tokens and answer tokens reported separately. Early layers are almost unchanged (Jensen–Shannon divergence at most 0.064 nats in layers 0–2), while layer 7 changes most, and most on answer tokens (**0.280** in Hindi, **0.267** in Assamese; maximum possible 0.693).
   * At layer 7, answer-token attention entropy drops by **$27.1\%$** (Hindi) and **$22.4\%$** (Assamese), and the share of answer-token attention landing on the prompt rises from 0.69 to 0.82 (Hindi) and from 0.56 to 0.67 (Assamese). Mean attention distance grows in Assamese (+18.2% at layer 7, +38.6% at layer 6) but is flat in Hindi (−1.0% at layer 7).

![Hindi: attention change by layer](figures/phase3_attn_hindi_v2_cot_jsd.png)
*Figure 3.1: Hindi V2 CoT — JS divergence between pretrained and finetuned attention by layer (200 test examples), for answer rows, prompt rows, and a prompt-only out-of-template probe sentence.*

![Assamese: attention change by layer](figures/phase3_attn_assamese_v2_cot_jsd.png)
*Figure 3.2: Assamese V2 CoT — same measurement as Figure 3.1.*

![Hindi layer 0: pretrained vs finetuned](figures/phase3_attn_hindi_v2_cot_L0_pair.png)
*Figure 3.3: Hindi V2, early layer (layer 0, head 1) — Pretrained Base (left) vs. Finetuned CoT (right). Red lines mark the prompt/answer boundary.*

![Hindi layer 7: pretrained vs finetuned](figures/phase3_attn_hindi_v2_cot_pair.png)
*Figure 3.4: Hindi V2, late layer (layer 7, head 1) — Pretrained Base (left) vs. Finetuned CoT (right).*

![Assamese layer 0: pretrained vs finetuned](figures/phase3_attn_assamese_v2_cot_L0_pair.png)
*Figure 3.5: Assamese V2, early layer (layer 0, head 2) — Pretrained Base (left) vs. Finetuned CoT (right).*

![Assamese layer 7: pretrained vs finetuned](figures/phase3_attn_assamese_v2_cot_pair.png)
*Figure 3.6: Assamese V2, late layer (layer 7, head 5) — Pretrained Base (left) vs. Finetuned CoT (right).*

*Reading the figures*:
- **Where the change is**: the early-layer pairs (Figures 3.3, 3.5) are nearly identical before and after, while the late-layer pairs (Figures 3.4, 3.6) differ clearly below the horizontal red line, i.e. on answer tokens. This matches a fine-tuning loss applied only to the completion.
- **What the change is**: late-layer answer tokens attend more sharply and look back at the premises more. This is descriptive evidence of where attention moved; it does not establish that a particular head implements a particular reasoning step. The pretrained model has also never been trained on the `<COT_START>`/`<REL_*>` tokens, so part of the answer-token divergence reflects unfamiliar input. Full tables are in the Phase 3 report, Section 7.
3. **Fine-Tuning Is What Produces the Exact-Match Scores**:
   * Zero-shot (pretrained, no SFT, same decoding), all eight models score 0.0–0.2% strict answer accuracy on the 500 test queries; after SFT they score 11.8–44.0%, on entities never seen in training.
   * Decision accuracy is a weaker signal. It rises for both V1 models (16–20% to 35–40%) and for Hindi V2 (30% to 38–41%), but not for Assamese V2 (33–34% zero-shot, 29–30% after SFT). For that model, fine-tuning taught the answer format more than the decision.
4. **Negation**:
   * Negated premises make up about 18% of training examples. On the held-out negation queries, fine-tuned strict accuracy is 50.6–55.4% in Hindi and 8.4–36.1% in Assamese, with decision stance of 50.6–56.6% and 30.1–48.2%. Zero-shot strict accuracy on the same queries is 0%.
5. **Token F1 vs. Strict Exact Match**:
   * Continuous metrics show that outputs are close to the target even when exact match fails: Hindi V1 CoT reaches **81.9% Token $F_1$** and **83.7% Character Similarity** at 40.6% strict accuracy; Assamese V1 CoT reaches 65.2% and 72.9% at 24.6%.
6. **Indeterminate Queries Remain Unsolved**:
   * Strict accuracy on queries with no deductive path is 0% for seven of the eight models (4.1% for Assamese V2 CoT). Decision stance reaches at most 36.5% (Assamese V2 Direct) and 12.2% in Hindi. Indeterminate items are about 10% of the training mix; we did not test whether a larger share or a larger model would fix this.

---

## 5. Artifact Verification & Reproduction Checklist

* **Public Kaggle Datasets Across All Three Phases**:
  * **Phase 1 Pretraining Corpora & Tokenizers**:
    * Hindi Dataset: [`shubhadeepmandal/lma-hindi-artifacts`](https://www.kaggle.com/datasets/shubhadeepmandal/lma-hindi-artifacts)
    * Assamese Dataset: [`shubhadeepmandal/lma-assamese-artifact`](https://www.kaggle.com/datasets/shubhadeepmandal/lma-assamese-artifact)
  * **Phase 2 Pretrained Checkpoints**: [`shubhadeepmandal/lma-phase2-artifacts`](https://www.kaggle.com/datasets/shubhadeepmandal/lma-phase2-artifacts)
  * **Phase 3 Finetuned Reasoning Models & Consolidated Artifacts**: [`shubhadeepmandal/lma-phase3-artifacts`](https://www.kaggle.com/datasets/shubhadeepmandal/lma-phase3-artifacts)
* **Tokenizers**: `hindi/tokenizer/hindi.model` and `assamese/tokenizer/assamese.model` (16K BPE).
* **Figures**: Rendered in 300 DPI under `report/figures/`.
* **Zero Contamination**: Disjoint entity pools, no pretrained components, independent monolingual pipelines.
