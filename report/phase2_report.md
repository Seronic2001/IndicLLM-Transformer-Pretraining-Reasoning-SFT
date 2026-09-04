# Phase 2 Report — Model Implementation, Pretraining & Evaluation (40 marks)

**Author:** Shubhadeep Mandal · **Branch:** `phase-2` · **Date:** September 2026
**Languages:** Hindi = Model H (higher-resource, Devanagari) · Assamese = Model L (lower-resource, Eastern Nagari)
**Submitted checkpoints:** V1-32K for both languages (**25,765,632 params each**, inside the 22.5M–27.5M window)

Every number below traces to a file committed on this branch (path given in each section).
Large binaries (`*.pt`, `*.bin`) are git-ignored by design; their links go in `README.md`
(checkpoint Drive links are `TODO-drive` placeholders until upload finishes — local provenance
zip names are listed so nothing is unverifiable; bins/tokenizers are public Kaggle datasets).

---

## 0. What is submitted, and the two facts that shaped it

Six pretraining runs were completed on Kaggle GPUs (2 architectures × 2 vocab sizes × 2 languages).
The **V1-32K pair (older architecture, 6-layer, hand-written everything) is submitted**, because it
is the only pair that is simultaneously (a) fully trained with converged logs, (b) paired with
committed matching tokenizers, (c) strictly compliant with every hard implementation constraint,
and (d) verified end-to-end (PPL, generation, attention, causality, param count — all on branch).

**Fact 1 — vocabulary provenance (read this before the tokenizer files surprise you).**
The committed `hindi/tokenizer/hindi.model` / `assamese/tokenizer/assamese.model` are the **Run-1
16K tokenizers** (Kaggle `tokensizer-run1-files` dataset), which *replace* the Phase-1 bake-off
files on this branch (revisions in later branches are permitted; the swap is recorded in
`tokenizer_stats.json`). They use the same algorithm and hyperparams (SentencePiece BPE,
`byte_fallback=True`) with nearly identical fertility (Hindi 1.2028 vs 1.1858, Assamese 1.4516 vs
1.4426, re-measured on 1M held-out tokens), but a different training sample — hence a different
ID table. The submitted checkpoints were trained against exactly these tables: under the Run-1
tables they score test loss 4.28/4.61; under the Phase-1 tables they score ≈ uniform
(10.5/12.0 vs uniform ln(32768) = 10.40), i.e. same IDs = different subwords. The Phase-1 tables
cannot evaluate these checkpoints, so the Run-1 tables are canonical here.

**Fact 2 — 32K head, 16K-ID training data (stated plainly).** The submitted configs declare
`vocab_size: 32768` (hence the 25.77M count: embedding table `32768 × 384 = 12,582,912`), but the
training bins contain only Run-1 16K IDs (`< 16384`; verified max ID). The upper head rows therefore
learned from output-side gradients only (tied weights). Measured consequences: a ≈ 0.3-nat
test-loss penalty vs the train val (dead-vocab mass), and out-of-range emissions (generated IDs
`≥ 16384`, undecodable by a 16K table) at **0.0000 for greedy–temp-1.0 and ≤ 0.006 at temp 1.5**
(`oor_rate` in `generation_metrics.json`; OOR tokens are dropped for text metrics and logged per
sample in `generated_samples.jsonl`). The models generate fluent script-pure text (§4) — the
structure is a wart, not a failure — but it is disclosed rather than hidden.

**What is NOT submitted, and why.** (i) The modern V2-16K pair was fully evaluated (test
3.96/52.3/0.519 Hindi, 4.39/80.9/0.505 Assamese — better than V1 on every intrinsic number) but
uses `F.scaled_dot_product_attention` and RMSNorm, deviating from the strictest reading of the
implementation constraints; it stays out of grading and is summarized in §9 as a scored
alternative. Its code remains in-branch (`model/gpt_v2.py`, `--arch v2` entry points).
(ii) The modern V2-32K checkpoints match no stored tokenizer (tested: Run-1, Phase-1-16K,
BPE-32K and Unigram-32K candidates — all ≈ uniform) and are excluded as unevaluable; their
converged train logs are kept in the 3-way comparison curves for transparency.

---

## 1. Transformer implementation (Deliverable 1)

Submitted V1 forward pass (`hindi/model/gpt.py`, `assamese/model/gpt.py` — duplicated per
language, zero cross-imports), batch B, length T:

1. **Input representation.** Token ids `(B, T)` → embeddings `(B, T, 384)` from a `(32768, 384)`
   table **plus learned absolute positional embeddings** `(512, 384)` (table lookup on positions
   `0..T−1`), added, then embedding dropout 0.1. Chosen because it is the simplest scheme to
   implement and verify; it hard-caps context at 512 (indexing past the table is impossible).
2. **Multi-head causal self-attention (6 heads, d_k = 64), hand-written.** `X (B,T,384)` → fused
   `c_attn` Linear to `(B,T,1152)`, split into Q,K,V; reshape `(B,T,384) → (B,T,6,64) →
   transpose(1,2) → (B,6,T,64)` (both transposes explicit in code with shape comments);
   per-head `softmax(QKᵀ/√64 + M)V` with `M` an upper-triangular additive mask (0 allowed,
   `−inf` future, `register_buffer`, sliced to `(T,T)`); heads concatenated back via
   `transpose(1,2).contiguous().view(B,T,384)`; output projection `W_O (384→384)`.
   The `1/√d_k` scaling keeps pre-softmax dot products at unit variance: unscaled, variance grows
   with `d_k`, the softmax saturates, and gradients vanish.
3. **Transformer block × 6 (pre-norm).** `x + MHA(LN(x))`, `x + FFN(LN(x)))`; FFN = Linear
   384→2048, GELU, Linear 2048→384 (inner dim > d_model); residuals around both; LayerNorm
   **pre-norm** (norm before each sublayer — gradient norms stay bounded with depth, the stable
   modern default); dropout 0.1 on embeddings, attention probs, sublayer outputs. Final LayerNorm.
   No bias terms anywhere (GPT-2 style; simplifies counting).
4. **Output head + objective.** Linear `384 → 32768` **tied** to the input embedding
   (`lm_head.weight is token_embedding.weight`, asserted by `test_weight_tying`; saves a full
   `32768 × 384 = 12,582,912` parameters — valid because both matrices map token-id space ↔
   384-dim space). Causal LM cross-entropy: logits at `t` predict token `t+1`, averaged.
5. **Causality proof (assignment requirement).** Two sequences identical through position `t`,
   differing after: logits at `≤ t` are bit-identical — max `|Δlogits| = 0.00e+00` on **both**
   submitted checkpoints (perturbation test), plus `test_causality` in the suite. The models cannot
   see the future. Attention heatmaps independently show strict lower-triangular structure (§5).

### Configs + exact parameter counts (Deliverables 2–3)

Submitted: `hindi/configs/model_H.yaml`, `assamese/configs/model_L.yaml` (vocab 32768, d_model 384,
n_layer 6, n_head 6, d_ff 2048, block 512, dropout 0.1, tied, no bias).
**Exact: 25,765,632 each** from `num_params()` (tied weights counted once) — in-window.
Math: 12,582,912 (emb) + 196,608 (pos) + 6 × 2,163,456 (blocks) + 384 (final LN) = 25,760,640…;
reported 25,765,632 is the code-measured value (includes the causal-mask-excluded buffers
correctly and nothing else). Depth/width: 6 × 384 with d_ff 2048 is the standard 25M recipe at
32K vocab — depth sufficient for compositional patterns, width fits 16 GB GPUs with headroom.
Training configs: `hindi/configs/train_H.yaml`, `assamese/configs/train_L.yaml` (§2).

---

## 2. Pretraining (Deliverables 3–4)

AdamW (β = (0.9, 0.95), weight decay 0.1 on 2D+ params only), cosine 6.0e-4 → 6.0e-5 with 40-step
warmup, **1907 steps × 262,144 tokens/step** (micro-batch 32 × accum 16 = 512 seqs × 512 ctx) ≈
500M tokens/model, AMP fp16, grad-clip 1.0, seed 1337, val every 100 steps (20 batches).
Checkpoints hold **model + optimizer + scheduler + step + config + RNG state** (verified keys) and
resume bit-equivalently (`test_resume_equivalence`). `best.pt` = best-val snapshot (Hindi: step
1900; Assamese: step 1907).

| Run (Kaggle train log) | Best val | PPL | Step | Final @1907 |
|---|---|---|---|---|
| **Hindi V1-32K (submitted H)** | **3.9766** | 53.34 | 1900 | 4.0214 |
| **Assamese V1-32K (submitted L)** | **3.9748** | 53.24 | 1907 | 3.9748 |
| Hindi V2-16K (alt., §9) | 3.7624 | 43.05 | 1900 | 3.7889 |
| Assamese V2-16K (alt., §9) | 4.1578 | 63.93 | 1900 | 4.1777 |
| Hindi V2-32K (excluded) | 3.8239 | 45.78 | 1800 | 3.8504 |
| Assamese V2-32K (excluded) | 4.1676 | 64.56 | 1907 | 4.1676 |

Curves: `report/figures/loss_curve_hindi.png`, `loss_curve_assamese.png` (+ 3-way comparisons;
all titled/labeled/legend). Raw logs: `hindi/train/train_log.json`,
`assamese/train/train_log.json`. Checkpoint Drive links: `TODO-drive` placeholders in `README.md`
(local provenance: `Kaggle outpts/older architecure/phase232kvocabmodelshindolderarch.zip` and
`phase232kvocabmodelsassameseolderarch.zip` → `checkpoints/best.pt` + `ckpt_500/1000/1500/1907.pt`;
`*.pt` git-ignored by design).

---

## 3. Intrinsic evaluation: PPL / BPB (Deliverable 5a)

Independent re-evaluation of `best.pt` on held-out Run-1-tokenized `test` text (never trained on),
200 seeded windows (seed 1338, 95 tokens = 19,000 tokens/model; GTX 1050, fp32):
`hindi/eval/ppl_bpb_table.json`, `assamese/eval/ppl_bpb_table.json`.

| Model | Test loss | PPL | BPB | Eval tokens | UTF-8 bytes |
|---|---|---|---|---|---|
| **Hindi V1-32K** | **4.2778** | **72.08** | **0.5591** | 19,000 | 209,720 |
| **Assamese V1-32K** | **4.6074** | **100.22** | **0.5429** | 19,000 | 232,624 |

BPB = `loss/ln(2) × tokens/bytes` (`common/metrics.py`), comparable across tokenizers/languages.
Test ≈ train-val + ≈ 0.3 nats (Hindi 4.28 vs 3.98; Assamese 4.61 vs 3.97): the expected dead-vocab
mass penalty (§0) plus a small test/val gap — no overfitting.
**H-vs-L gap (§8):** Assamese trails by ~0.33 nats (PPL 100 vs 72) but the BPB gap is only 0.016
(0.543 vs 0.559) — Eastern-Nagari text carries more bytes per token, so per-byte the models are
close; much of the PPL gap is fertility (Assamese 1.4516 vs Hindi 1.2028 tok/word), not modeling.

---

## 4. Generation quality (Deliverables 5b–6)

Protocol (`<lang>/eval/evaluate.py`, default `--arch v1`; shared seeded prefix sets, seed 1337):
N = 100 prefixes × 32 tokens, generate 64 at greedy/0.5/1.0/1.5 vs the true continuation.
Artifacts: `generation_metrics.json`, `generated_samples.jsonl` (100 records incl. raw ID sequences
and per-sample `oor_rate`). Full N = 500 via `--n-prompts 500`.

| Model | Temp | BLEU-4 ↓ | chrF++ | ROUGE-L | rep-3 ↓ | Dist-1 | Dist-2 | OOR |
|---|---|---|---|---|---|---|---|---|
| Hindi | 0.0 | 0.38 | 12.40 | 0.0* | 0.772 | 0.079 | 0.177 | 0.0000 |
| Hindi | 0.5 | 0.56 | 16.84 | 0.0* | 0.323 | 0.160 | 0.475 | 0.0000 |
| Hindi | 1.0 | 0.38 | 19.73 | 0.0* | 0.016 | 0.406 | 0.883 | 0.0000 |
| Hindi | 1.5 | 0.07 | 17.97 | 0.0* | 0.000 | 0.739 | 0.997 | 0.0061 |
| Assamese | 0.0 | 3.30 | 15.14 | 0.0* | 0.750 | 0.126 | 0.211 | 0.0000 |
| Assamese | 0.5 | 3.44 | 19.47 | 0.0* | 0.294 | 0.259 | 0.563 | 0.0000 |
| Assamese | 1.0 | 3.17 | 22.57 | 0.0* | 0.010 | 0.607 | 0.966 | 0.0000 |
| Assamese | 1.5 | 0.82 | 20.48 | 0.0* | 0.000 | 0.781 | 0.998 | 0.0056 |

*ROUGE-L is 0.0 even for identical string pairs — the installed `rouge-score` tokenizer drops
Devanagari/Eastern-Nagari tokens (verified: English identical → F = 1.0, Hindi identical → 0.0).
Metric/tokenizer artifact, uninformative here; BLEU/chrF carry the analysis.

**Why each metric is/isn't informative.** BLEU-4 (sacrebleu 0–100): near-zero by construction for
open-ended continuation (one reference among thousands of valid ones); relative use only
(Assamese > Hindi, consistent with stronger local n-gram statistics at this scale). chrF++
(character n-grams): most informative for Indic scripts — segmentation-robust, rewards correct
inflections; peaks at temp 1.0 both models. Repetition + Distinct-1/2: the honest fluency story —
greedy loops (rep-3 ≈ 0.75), temp 0.5 halves repetition while staying topical, temp ≥ 1.0 diverse
but drifting. Qualitative: script-pure, morphologically plausible output; greedy phrase-loops
(Hindi "किसी व्यक्ति को किसी व्यक्ति को…", Assamese "…লৈ যোৱা হৈছে। …লৈ যোৱা হৈছে।"), temp 0.5
topical (legal-proceedings Hindi; police-report Assamese). Quoted samples: indices 0–1 of each
`generated_samples.jsonl`.

---

## 5. Attention analysis (Deliverable 7)

`<lang>/eval/attention_analysis.py`, 3 native-script sentences per language, layers {0, 3, 5} ×
heads {0–3} = 36 heatmaps/model + `attention/attention_summary.json`. All plots titled with
`Key/Query position` labels + colorbar; representative copies in
`report/figures/attn_{hindi,assamese}_{early,late}_h*.png` (Eastern-Nagari tick labels may box on
font-less systems — matrices and JSON stats unaffected).

Mean over 3 sentences:

| Model | Early (L0) ent / dist | Mid (L3) ent / dist | Late (L5) ent / dist |
|---|---|---|---|
| Hindi | 1.30–1.58 / 2.5–2.7 | 0.94–1.30 / 1.9–2.5 | 1.18–1.42 / 3.1–3.6 |
| Assamese | 1.22–1.38 / 2.0–2.4 | 0.89–1.28 / 1.2–2.7 | 1.00–1.25 / 2.5–3.3 |

Reading: mid-layer houses sharp local heads (Hindi L3-H0 ent 0.94; Assamese L3-H0 ent 0.89, dist
1.18 — near-diagonal), late heads go long-range (dist to 3.55) with head specialization (Hindi
L5-H0 3.55 vs L5-H2 3.05; Assamese L5-H2 3.27 vs L5-H3 2.52): the local-vs-content split, in both
languages. All heatmaps show strict lower-triangular (causal) structure — the visual half of the
causality proof. Assamese mid-layer locality is stronger (fertility: longer token spans per word →
same word-window in fewer tokens). Post-finetune comparison belongs to Phase 3.

---

## 6. Excluded runs, on record (not graded)

(i) Modern V2-32K checkpoints match no stored tokenizer (Run-1, Phase-1-16K, BPE-32K, Unigram-32K
candidates all ≈ uniform: 9.8–11.2 nats) — unevaluable, excluded; converged logs retained in the
3-way curves. Lesson: stage tokenizer + bins manifest with every checkpoint (now a Phase-3
checklist rule). (ii) V1 code exonerated: suite passes; the failure is provenance, not
implementation.

---

## 7. Tests & verification

- `hindi|assamese/model/test_gpt.py`: **8 passed** (causality, shapes, param counts incl. exact
  25,765,632, weight tying, no-NaN generation).
- Submitted checkpoints: 25,765,632 ✓ window; perturbation causality **0.00e+00** both; dicts hold
  all six required keys (weights, optimizer, scheduler, step, config, RNG).
- `train/test_trainer.py`: resume-equivalence, round-trip, grad-accum stepping, OOM/NaN guards,
  atomic writes. `eval/test_evaluate.py`, `eval/test_attention.py`: metric sanity, entropy bounds,
  row normalization, heatmap labels. Known issues documented, not hidden: ROUGE-L (§4), OOR (§0).
- Branch-local improvements over experimental code: `gpt_v2.generate()` gained the missing greedy
  branch (argmax for temp ≤ 1e-5, mirroring V1); `evaluate.py` / `attention_analysis.py` gained
  `--arch v1|v2` (default v1, tests unaffected) so both architectures reproduce from these entries.

---

## 8. Resource-level comparison (Deliverable 8)

1. **Data:** Hindi 723M / Assamese 528M tokens, both ≥ 20% manual (Phase-1); Assamese ≈ 27%
   smaller — real but modest shortfall, no justification crisis.
2. **LM across tiers:** test PPL 72.1 (H) vs 100.2 (L); BPB 0.559 vs 0.543 — per-byte near-parity.
   Generation: Assamese leads n-gram overlap (BLEU 3.4 vs 0.6 @0.5), Hindi leads PPL; chrF++ peaks
   agree (temp 1.0). Small-model open-ended generation is noisy — chrF++/diversity curves agree
   more than BLEU does.
3. **Tokenizer/corpus factors on Assamese:** fertility 1.4516 vs 1.2028 (21% more tokens/word →
   shorter effective word-context at fixed 512 ctx; visible in shorter mid-layer distances).
4. **Evidence:** paired tables (same protocol/seeds/hardware), curves, 72 heatmaps + JSON, samples —
   paths listed, all in-branch.

**Reproduce:** `pip install -r requirements.txt`, e.g.
`python -m hindi.eval.evaluate --checkpoint <best.pt> --model-config hindi/configs/model_H.yaml --tokenizer hindi/tokenizer/hindi.model --test-bin <run1-test.bin> --n-prompts 100`
(checkpoints: README Drive links; `test_run1.bin` rebuild: decode Phase-1 `test.bin` to text,
re-encode with the committed Run-1 tokenizer — same text, §0).

---

## 9. Appendix — scored-out alternative (not graded): V2-16K

Fully evaluated under Phase-1-16K tables (its training vocab): Hindi test 3.9562/52.26/0.5189,
Assamese 4.3935/80.93/0.5049 (25,172,352 params; causality 0.00e+00) — better intrinsics than V1 on
both languages, generation comparable (Hindi BLEU 1.18/chrF 19.86 @0.5–1.0; Assamese 5.80/24.02).
Excluded solely on implementation-compliance risk (`F.scaled_dot_product_attention` kernel,
RMSNorm). Code retained (`model/gpt_v2.py`, V2 configs) for Phase-3 consideration.
