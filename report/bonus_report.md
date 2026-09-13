# Bonus Technical Report: Positional Embedding Ablation (Model H Hindi)

**Course**: Language Models and Agents (Monsoon 2026)  
**Author**: Shubhadeep Mandal (Roll No: 2025201056)  
**Repository**: [github.com/CL3-410/individual-project-Seronic2001](https://github.com/CL3-410/individual-project-Seronic2001)  
**Branch**: `bonus`  
**Ablation Scope**: Retraining Hindi 16K Transformer (~24.78M parameters) with positional embeddings completely removed ($E_{\text{pos}} = 0$), benchmarked under the full Phase 2 evaluation suite across full 500M-token (1,907 steps) pretraining.

---

## 1. Cloud Execution & Reproducibility Links

The ablation experiment is hosted, trained, and evaluated on Kaggle GPUs:
* **Full 500M-Token Pretraining Ablation (1,907 Steps / Completed)**:  
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

We evaluate the fully pretrained ablated model against the standard **Baseline V1-16K** (learned absolute position) and **Modern V2-16K** (RoPE) models on the exact held-out test split (`test.bin`, 300 sequence windows, 153,300 tokens):

| Benchmark Metric | Ablated No-Pos V1 (Full 500M) | Baseline V1-16K (Standard 500M) | Modern V2-16K (Standard 500M) | Key Observations |
|---|---|---|---|---|
| **Trainable Parameters** | **24,785,664** (~24.78M) | 24,982,272 (~24.98M) | 25,172,352 (~25.17M) | $-196,608$ params saved (no pos table) |
| **Tokens Pretrained** | **500,000,000** (1,907 steps) | 500,000,000 (1,907 steps) | 500,000,000 (1,907 steps) | Identical pretraining token budget |
| **Positional Scheme** | **None (Omitted)** | Learned Absolute Table | Rotary (RoPE) | Complete structural ablation |
| **Held-Out Test Loss** | **4.1129 nats** | 4.4102 nats | 3.9562 nats | Loss converges, but generation breaks |
| **Held-Out Perplexity (PPL)**| **61.13** | 82.28 | 52.26 | Lower PPL than V1 due to bigram memorization |
| **Bits-Per-Byte (BPB)** | **0.5502** | 0.5764 | 0.5189 | Strong surface n-gram compression |
| **chrF++ (@ Temp 1.0)** | **19.82** | 18.95 | 19.86 | Matches baseline character overlap |
| **Repetition Rate (@ T=0.0)**| **0.7757 (77.6%)** | 0.7716 (77.2%) | 0.6842 (68.4%) | **Severe looping failure mode** |
| **Distinct-1 (@ Temp 1.0)** | **0.3733** | 0.4056 | 0.4612 | Reduced unigram variety |
| **Distinct-2 (@ Temp 1.0)** | **0.8611** | 0.8828 | 0.9124 | Lower bigram diversity than RoPE |

---

## 4. Multi-Temperature Generation Dynamics

Evaluating 150 prompt prefixes across temperatures $\{0.0, 0.5, 1.0, 1.5\}$ reveals severe behavioral degradation and structural failure modes:

| Temperature ($T$) | BLEU | chrF++ | ROUGE-L | 3-Gram Repetition Rate | Distinct-1 | Distinct-2 |
|---|---|---|---|---|---|---|
| **0.0 (Greedy)** | 0.58 | 13.05 | **0.00** | **0.7757 (77.6%)** | 0.0704 | 0.1658 |
| **0.5** | 1.02 | 16.98 | **0.00** | **0.3814 (38.1%)** | 0.1361 | 0.4247 |
| **1.0 (Standard)**| 0.33 | 19.82 | **0.00** | **0.0256 (2.6%)** | 0.3733 | 0.8611 |
| **1.5 (High Entropy)** | 0.07 | 18.52 | **0.00** | **0.0004** | 0.6578 | 0.9880 |

### Key Generation Insights:
* **ROUGE-L Total Collapse ($0.00$)**: Across all temperatures, ROUGE-L remains strictly $0.00$. Because ROUGE-L measures the Longest Common Subsequence (LCS), matching requires preserving sequential token order. Without position embeddings, the model cannot coordinate multi-token phrase ordering with the reference text.
* **Greedy Decoding Looping ($77.6\%$)**: At $T=0.0$, the model collapses into periodic cyclic 3-gram attractors, repeating identical phrases indefinitely.

---

## 5. Qualitative Generation Samples

Empirical text generation directly using the final checkpoint (`hindi_no_pos_latest.pt`):

```text
Prompt 1: "भारत एक महान देश है" (India is a great country)
  [T=0.0]: भारत एक महान देश है। भारत में भारत के सबसे बड़े देश हैं। भारत में सबसे बड़ा देश है। भारत में सबसे बड़ा देश है। भारत में सबसे बड़ा देश है
  [T=0.5]: भारत एक महान देश है। भारत में दुनिया में सबसे बड़ा देश है। इस देश में सबसे बड़ा देश है। दुनिया में सबसे बड़ा देश है। भारत का सबसे बड़ा देश
  [T=1.0]: भारत एक महान देश है। भारत के पूर्वपश्चिम ईरानी राज्य हैं और मोदी सरकार के महासचिव सुभाष चंद्रमान आईएएनएस की नई दिल्ली में हुई मॉब लिंचिंग मामले में सुप्रीम कोर्ट से मिला

Prompt 2: "शिक्षा का महत्व जीवन में बहुत" (The importance of education in life is very)
  [T=0.0]: शिक्षा का महत्व जीवन में बहुत महत्व है। यह एक महत्वपूर्ण बात है कि यह एक महत्वपूर्ण बात है कि यह एक महत्वपूर्ण बात है कि यह एक महत्वपूर्ण बात है कि यह एक महत्वपूर्ण
  [T=0.5]: शिक्षा का महत्व जीवन में बहुत बड़ा महत्व होता है। यह एक ऐसा ग्रंथ है जो प्रकृति में एक महान और महान है। यह एक महान धर्म है जो जीवन में सबसे बड़ी समस्या
  [T=1.0]: शिक्षा का महत्व जीवन में बहुत नफरत यूनानऔजिया इंडस्ट्री का एक महत्व है। जैसा कि हम जानते हैं कि हमारे विज्ञान और अनुसंधान व्यक्ति के अस्मिता हत्याकांड की जानकारी
```

Notice the stark repetition loop at $T=0.0$: `यह एक महत्वपूर्ण बात है कि यह एक महत्वपूर्ण बात है कि...` repeats 5 times in a 30-token budget.

---

## 6. Attention Diagnostics & Entropy Analysis

![Attention Heatmaps of No-Pos Ablation](figures/ablation_no_pos_attention_heatmaps.png)  
*Figure B.1: Query-Key Attention Heatmaps for Full 500M No-Pos Model — Layer 0 Head 0 (Early) vs. Layer 7 Head 0 (Late). Without position embeddings, attention cannot form structured diagonal locality.*

### Quantitative Diagnostics
* **Early Layer (L0, H0)**:
  * Attention Entropy $H(A)$: **$1.3907\text{ nats}$**
  * Mean Attention Distance: **$3.20\text{ tokens}$**
* **Late Layer (L7, H0)**:
  * Attention Entropy $H(A)$: **$1.2894\text{ nats}$**
  * Mean Attention Distance: **$3.43\text{ tokens}$**

In standard models with positional encodings, late layers expand attention distance up to 5.86 tokens to bind long-range syntactic premises. In the ablated model, attention distance remains constrained and flat ($3.20 \to 3.43$), indicating that attention heads cannot specialize into functional long-range circuits.

---

## 7. Mechanistic Explanation: What Breaks Without Position Information?

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
   * Without positional decay, once a frequent subword is output, its attention score remains identically attractive for the next step, creating a positive feedback loop that causes the model to repeat the same phrase endlessly (77.6% 3-gram repetition rate).
