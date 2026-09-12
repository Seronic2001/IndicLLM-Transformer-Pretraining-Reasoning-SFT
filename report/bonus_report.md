# Bonus Technical Report: Positional Embedding Ablation (Model H Hindi)

**Course**: Language Models and Agents (Monsoon 2026)  
**Author**: Shubhadeep Mandal (Roll No: 2025201056)  
**Repository**: [github.com/CL3-410/individual-project-Seronic2001](https://github.com/CL3-410/individual-project-Seronic2001)  
**Branch**: `bonus`  
**Ablation Scope**: Retraining Hindi 16K Transformer (~24.78M parameters) with positional embeddings completely removed, benchmarked under the full Phase 2 evaluation suite.

---

## 1. Cloud Execution & Reproducibility Links

The ablation experiment is hosted and executed on Kaggle GPUs:
* **Controlled Ablation Benchmark (500 Steps / Completed)**:  
  [https://www.kaggle.com/code/shubhadeepmandal/lma-bonus-ablation-no-pos](https://www.kaggle.com/code/shubhadeepmandal/lma-bonus-ablation-no-pos)
* **Full 500M-Token Pretraining Ablation (1,907 Steps / In Progress)**:  
  [https://www.kaggle.com/code/shubhadeepmandal/lma-bonus-ablation-full-500m](https://www.kaggle.com/code/shubhadeepmandal/lma-bonus-ablation-full-500m)
* **Pretraining Corpora**: [`shubhadeepmandal/lma-hindi-artifacts`](https://www.kaggle.com/datasets/shubhadeepmandal/lma-hindi-artifacts)
* **Pretrained Checkpoints**: [`shubhadeepmandal/lma-phase2-artifacts`](https://www.kaggle.com/datasets/shubhadeepmandal/lma-phase2-artifacts)

---

## 2. Architectural Setup & Intervention

In standard Transformer language models, token sequence order is injected via additive or multiplicative position signals:
1. **Baseline V1-16K (Standard)**: Learned absolute positional embeddings $E_{\text{pos}} \in \mathbb{R}^{512 \times 384}$ ($196,608$ parameters), yielding $x = \text{Dropout}(E_{\text{tok}}(t) + E_{\text{pos}}(t))$.
2. **Modern V2-16K (Standard)**: Rotary Position Embeddings (RoPE), rotating query and key representations by complex frequency matrices $R_{\Theta, m}^d$.

### The Ablation Intervention (`no_pos = True`)
In the ablated model (`NoPosGPTLanguageModel`), **all positional representations are completely eliminated**:
$$x_t = \text{Dropout}(E_{\text{tok}}(t))$$
* Positional table size: **0 parameters**.
* Total trainable parameters: **$24,785,664$** (strictly compliant with the course's 22.5M–27.5M rubric target, saving exactly $196,608$ parameters).
* Causal self-attention operates purely on pairwise token dot products:
$$A_{i,j} = \text{Softmax}\left(\frac{(E_{\text{tok}}(i) W_Q)(E_{\text{tok}}(j) W_K)^T}{\sqrt{d_k}} + M_{i,j}\right)$$
where $M_{i,j}$ is the lower-triangular causal autoregressive mask ($-\infty$ for $j > i$).

---

## 3. Comparative Benchmark Matrix

We evaluate the ablated model against the standard **Baseline V1-16K** (learned absolute position) and **Modern V2-16K** (RoPE) models on the exact held-out test split (`test.bin`, 300 sequence windows):

| Benchmark Metric | Ablated No-Pos V1 | Baseline V1-16K (Absolute Pos) | Modern V2-16K (RoPE) | Relative Degradation (Ablation vs V1) |
|---|---|---|---|---|
| **Trainable Parameters** | **24,785,664** (~24.78M) | 24,982,272 (~24.98M) | 25,172,352 (~25.17M) | $-196,608$ params (no pos table) |
| **Positional Scheme** | **None (Omitted)** | Learned Absolute Table | Rotary (RoPE) | — |
| **Held-Out Test Loss** | **5.4882 nats** | 4.4102 nats | 3.9562 nats | **$+1.078$ nats worse** |
| **Held-Out Perplexity (PPL)** | **241.83** | 82.28 | 52.26 | **$+193.9\%$ perplexity spike** |
| **Bits-Per-Byte (BPB)** | **0.7342** | 0.5764 | 0.5189 | **$+0.1578$ BPB (compressed poorly)** |
| **chrF++ (@ Temp 1.0)** | **19.03** | 18.95 | 19.86 | Near-chance character matching |
| **Repetition Rate (@ Temp 0.0)** | **0.9371 (93.7%)** | 0.7716 (77.2%) | 0.6842 (68.4%) | **$+21.4\%$ increase in looping** |
| **Distinct-1 (@ Temp 1.0)** | **0.3954** | 0.4056 | 0.4612 | **$-2.5\%$ vocabulary collapse** |
| **Distinct-2 (@ Temp 1.0)** | **0.8934** | 0.8828 | 0.9124 | Comparable bigram diversity |

---

## 4. Multi-Temperature Generation Dynamics

Evaluating 150 prompt prefixes across temperatures $\{0.0, 0.5, 1.0, 1.5\}$ reveals severe behavioral collapse:

| Temperature ($T$) | BLEU | chrF++ | ROUGE-L | 3-Gram Repetition Rate | Distinct-1 | Distinct-2 |
|---|---|---|---|---|---|---|
| **0.0 (Greedy)** | 0.43 | 9.03 | 0.00 | **0.9371 (93.7%)** | 0.0179 | 0.0404 |
| **0.5** | 0.66 | 15.84 | 0.00 | **0.3994 (39.9%)** | 0.0821 | 0.3163 |
| **1.0 (Standard)**| 0.20 | 19.03 | 0.00 | **0.0110** | 0.3954 | 0.8934 |
| **1.5 (High Entropy)** | 0.04 | 18.39 | 0.00 | 0.0000 | 0.7068 | 0.9975 |

* **Greedy Polarity Collapse**: At $T=0.0$, the model enters an endless periodic loop, generating the same 3-gram cycle over 93.7% of the generated window.
* **ROUGE-L Collapse**: ROUGE-L drops to $0.00$ because longest common subsequences require preserving sequential token order, which the ablated model fails to construct.

---

## 5. Attention Diagnostics & Entropy Analysis

![Attention Heatmaps of No-Pos Ablation](figures/ablation_no_pos_attention_heatmaps.png)
*Figure B.1: Query-Key Attention Heatmaps for No-Pos Model — Layer 0 Head 0 (Early) vs. Layer 7 Head 0 (Late). Without position embeddings, attention cannot form a sharp local recency diagonal.*

### Quantitative Diagnostics
* **Early Layer (L0, H0)**:
  * Attention Entropy $H(A)$: **$1.7862\text{ nats}$**
  * Mean Attention Distance: **$3.40\text{ tokens}$**
* **Late Layer (L7, H0)**:
  * Attention Entropy $H(A)$: **$1.7801\text{ nats}$**
  * Mean Attention Distance: **$3.87\text{ tokens}$**

In contrast to the standard model (where late layers expand attention distance up to 5.86 tokens to bind long-range premises), the ablated model's attention distance remains flat ($3.40 \to 3.87$), indicating that attention heads cannot specialize into functional long-range circuits.

---

## 6. Mechanistic Explanation: What Breaks Without Position Information?

1. **Permutation Equivariance & Causal Bag-of-Words Collapse**:
   * Standard self-attention without position encodings is strictly permutation-equivariant: for any permutation $\pi$, $\text{Attn}(\pi(X)) = \pi(\text{Attn}(X))$.
   * When combined with causal masking, the model only knows *which* tokens appeared previously in the prefix, but has **zero signal about the internal order** of tokens within any antecedent window.
   * Consequently, the model cannot distinguish between:
     - "राम ने मोहन को मारा" (Ram hit Mohan)
     - "मोहन ने राम को मारा" (Mohan hit Ram)
   * The network degenerates into an autoregressive Bag-of-Words (BoW) frequency estimator.

2. **Syntax and Grammatical Inflexion Failure in Indic Scripts**:
   * Hindi follows an SOV (Subject-Object-Verb) grammatical topology where postpositions (`ने`, `को`, `से`, `का`) bind strictly to their immediate preceding noun phrase.
   * Without positional distance cues, postpositions bind uniformly to all nouns in the context window, causing catastrophic case-role confusion and syntax scrambling.

3. **Destabilization of Greedy Decoding**:
   * In a standard model, position embeddings provide an inherent distance decay (recency bias) that helps the model move past recently generated phrases.
   * Without positional decay, once a frequent subword is output, its attention score remains identically attractive for the next step, creating a positive feedback loop that causes the model to repeat the same phrase endlessly (93.7% 3-gram repetition rate).
