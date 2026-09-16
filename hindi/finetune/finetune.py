"""Hindi finetuning pipeline.

Supports both Direct SFT and Chain-of-Thought (CoT) Fine-Tuning across
both Version 1.0 (Baseline) and Version 2.0 (Modern Transformer) architectures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

import yaml
import torch
import numpy as np

_LANG_ROOT = Path(__file__).resolve().parents[1]
if str(_LANG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LANG_ROOT))

from common.checkpoint import save_checkpoint, load_checkpoint
from tokenizer.tokenizer import Tokenizer

FINETUNE_LR = 3e-5
FINETUNE_MAX_STEPS = 300
FINETUNE_WARMUP = 10
FINETUNE_BLOCK_SIZE = 256

# Assamese stance tokens for contrastive margin loss
_STANCE_GT_TOKEN = "अधिक"  # "greater"
_STANCE_LT_TOKEN = "कम"     # "less"


# ---------------------------------------------------------------- data conversion

def load_reasoning_jsonl(path: Union[str, Path]) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _find_subseq(haystack: list[int], needle: list[int]) -> list[int]:
    """All start offsets where token subsequence `needle` occurs in `haystack`."""
    if not needle or len(needle) > len(haystack):
        return []
    n = len(needle)
    return [s for s in range(len(haystack) - n + 1) if haystack[s : s + n] == needle]


class PromptMaskedSFTDataset:
    """Dataset of (prompt, completion) sequences where prompt tokens have label=-100.
    
    Also tracks the position of the stance token (বেছি/কম) in the answer for
    contrastive margin loss computation.
    """

    def __init__(self, examples: list[dict], tokenizer, block_size: int = 128, use_cot: bool = True):
        self.block_size = block_size
        self.samples = []  # (x, y)
        self.stance_metadata = []  # (target_stance_pos, stance_gt_id, stance_opp_id)
        self.entity_metadata = []  # (entity_target_positions, entity_token_ids) per sample
        
        # Pre-encode stance tokens
        gt_ids = tokenizer.encode(_STANCE_GT_TOKEN)
        lt_ids = tokenizer.encode(_STANCE_LT_TOKEN)
        self._stance_gt_id = gt_ids[-1] if gt_ids else -1
        self._stance_lt_id = lt_ids[-1] if lt_ids else -1
        
        for ex in examples:
            prompt_text = ex["text"]
            ans_text = ex.get("cot_text", ex.get("answer_text", "")) if use_cot else ex.get("answer_text", "")

            p_ids = tokenizer.encode(prompt_text)
            a_ids = tokenizer.encode(" " + ans_text)
            eos_id = getattr(tokenizer, "eos_id", None)
            if eos_id is not None and (not a_ids or a_ids[-1] != eos_id):
                a_ids = a_ids + [eos_id]
            if not a_ids:
                continue

            max_p_len = max(1, block_size - len(a_ids))
            p_ids = p_ids[:max_p_len]
            input_ids = p_ids + a_ids
            if len(input_ids) < 2:
                continue
            input_ids = input_ids[: block_size + 1]

            prompt_len = len(p_ids)
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            # Find stance token position in the answer portion
            stance_pos = -1
            stance_gt_id = self._stance_gt_id
            stance_opp_id = self._stance_lt_id
            for pos in range(len(a_ids) - 1, -1, -1):
                tid = a_ids[pos]
                if tid == self._stance_gt_id:
                    stance_pos = prompt_len + pos
                    stance_gt_id = self._stance_gt_id
                    stance_opp_id = self._stance_lt_id
                    break
                elif tid == self._stance_lt_id:
                    stance_pos = prompt_len + pos
                    stance_gt_id = self._stance_lt_id
                    stance_opp_id = self._stance_gt_id
                    break

            # Query-entity mention spans in the completion (target/logit index
            # space) for the entity-coverage auxiliary loss. Rewards emitting
            # prompt entities in rationales/answers instead of hallucinating.
            ent_tidx: list[int] = []
            ent_tids: list[int] = []
            for ent in ex.get("query", []) or []:
                if not ent:
                    continue
                for variant in (" " + ent, ent):
                    try:
                        e_ids = tokenizer.encode(variant)
                    except Exception:
                        e_ids = []
                    starts = _find_subseq(a_ids, e_ids)
                    if starts:
                        for s in starts:
                            for j, tid in enumerate(e_ids):
                                t_idx = prompt_len + s + j - 1
                                if 0 <= t_idx < block_size:
                                    ent_tidx.append(t_idx)
                                    ent_tids.append(tid)
                        break

            pad_len = (block_size + 1) - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len
                labels = labels + [-100] * pad_len

            target_slice = labels[1:]
            if not any(lbl != -100 for lbl in target_slice):
                continue

            x = torch.tensor(input_ids[:-1], dtype=torch.long)
            y = torch.tensor(target_slice, dtype=torch.long)
            target_stance_pos = stance_pos - 1 if stance_pos > 0 else -1
            self.samples.append((x, y))
            self.stance_metadata.append((target_stance_pos, stance_gt_id, stance_opp_id))
            self.entity_metadata.append((ent_tidx, ent_tids))

        if self.samples:
            self._all_x = torch.stack([s[0] for s in self.samples])
            self._all_y = torch.stack([s[1] for s in self.samples])
        else:
            self._all_x = torch.empty((0, block_size), dtype=torch.long)
            self._all_y = torch.empty((0, block_size), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]

    def get_batch(
        self, batch_size: int, device: str, generator: Optional[torch.Generator] = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[int], list[list[int]], list[list[int]]]:
        """Returns (x, y, stance_positions, stance_gt_ids, stance_opp_ids, entity_positions, entity_token_ids)."""
        if generator is not None:
            indices = torch.randint(len(self.samples), (batch_size,), generator=generator)
        else:
            indices = torch.randint(len(self.samples), (batch_size,))
        bx = self._all_x[indices].to(device, non_blocking=True)
        by = self._all_y[indices].to(device, non_blocking=True)
        idx_list = indices.tolist()
        stance_pos = [self.stance_metadata[i][0] for i in idx_list]
        stance_gt = [self.stance_metadata[i][1] for i in idx_list]
        stance_opp = [self.stance_metadata[i][2] for i in idx_list]
        ent_pos = [self.entity_metadata[i][0] for i in idx_list]
        ent_ids = [self.entity_metadata[i][1] for i in idx_list]
        return bx, by, stance_pos, stance_gt, stance_opp, ent_pos, ent_ids


# ---------------------------------------------------------------- exact match & multi-tier metrics

_COT_PREFIX_RE = __import__("re").compile(r"(?:\[(?:कारण|কাৰণ):.*?\]|<COT_START>.*?<COT_END>)\s*")
_COT_EXTRACT_RE = __import__("re").compile(r"(?:\[(?:कारण|কাৰণ):\s*(.*?)\]|<COT_START>\s*(.*?)<COT_END>)")


def _extract_matched_rationale(match) -> str:
    if not match:
        return ""
    for g in reversed(match.groups()):
        if g is not None and g not in ("कारण", "কাৰণ"):
            return g.strip()
    return ""

try:
    from finetune.generate_reasoning import solve_relation, ContradictionError, extract_prompt_lemma_token_ids
except ImportError:
    try:
        from hindi.finetune.generate_reasoning import solve_relation, ContradictionError, extract_prompt_lemma_token_ids
    except ImportError:
        solve_relation = None
        ContradictionError = None
        extract_prompt_lemma_token_ids = None


def verify_cot_derivation(pred: str, query: tuple[str, str], gold_rel_str: str) -> bool:
    """Graph-equivalent / Permutation-invariant CoT verification."""
    if solve_relation is None:
        return True
    if not query or len(query) != 2 or not gold_rel_str:
        return False

    gold_rel = gold_rel_str.split()[1] if " " in gold_rel_str else gold_rel_str
    
    # Check if rationale is present
    match = _COT_EXTRACT_RE.search(pred)
    if match:
        rat = _extract_matched_rationale(match)
    elif "<COT_START>" in pred:
        rat = pred.split("<COT_START>", 1)[1].split("<COT_END>", 1)[0].split("]")[0].strip()
    elif any(tag in pred for tag in ["[कारण:"]):
        for tag in ["[कारण:"]:
            if tag in pred:
                rat = pred.split(tag, 1)[1].split("]")[0].strip()
                break
    else:
        return False

    # Pre-normalize symbolic relational tokens so graph solver always works
    norm_rat = rat.replace("<REL_GT>", " > ").replace("<REL_LT>", " < ").replace("<REL_EQ>", " = ")

    # Handle indeterminate
    if gold_rel == "?":
        undet_markers = [
            "संबंध नहीं", "तय नहीं किया जा सकता", "जानकारी अपर्याप्त",
            "कहा नहीं जा सकता", "ज्ञात नहीं", "स्पष्ट नहीं",
            "अलग-अलग समूहों", "कोई पथ नहीं", "DISJOINT", "<REL_DISJOINT>", "?"
        ]
        return any(m in norm_rat for m in undet_markers) or any(m in pred for m in undet_markers)

    # Extract relational premise tuples (e.g. "A < B" or "A = B" or "A > B")
    tuples = __import__("re").findall(r"([^\s><=,।]+)\s*([><=])\s*([^\s><=,।]+)", norm_rat)
    if not tuples:
        return False

    try:
        ans = solve_relation(tuples, query, allow_underdetermined=True)
        pred_rel = ans.split()[1]
        return pred_rel == gold_rel
    except Exception:
        return False


def _normalize(text: str) -> str:
    return " ".join(text.split())


def extract_answer_from_prediction(pred: str) -> str:
    """Extract answer text robustly without swallowing sentences on unclosed tags."""
    if "<COT_END>" in pred:
        ans = pred.split("<COT_END>", 1)[1].strip()
    elif "]" in pred and any(tag in pred for tag in ["[कारण:"]):
        ans = pred.split("]", 1)[1].strip()
    else:
        clean = pred.replace("<COT_START>", "").replace("[कारण:", "").strip()
        m = __import__("re").search(r"([^\s><=]+(?:\s+की|\s+का|\s+दी|\s+से|की उम्र|का वजन|की लंबाई|की बचत|दी गई जानकारी).*)$", clean)
        if m:
            ans = m.group(1).strip()
        else:
            ans = _COT_PREFIX_RE.sub("", pred).strip() or clean

    for delim in ["।", "?", "."]:
        if delim in ans:
            parts = ans.split(delim)
            first_sent = parts[0].strip() + delim
            if len(first_sent) >= 5:
                return first_sent
    return ans


def extract_rationale_and_answer(text: str) -> tuple[str, str]:
    match = _COT_EXTRACT_RE.search(text)
    if match:
        rationale = _extract_matched_rationale(match)
        raw_ans = _COT_EXTRACT_RE.sub("", text).strip()
        answer = extract_answer_from_prediction(raw_ans)
        return rationale, answer
    if "<COT_START>" in text:
        content = text.split("<COT_START>", 1)[1]
        if "<COT_END>" in content:
            r_part, a_part = content.split("<COT_END>", 1)
            return r_part.strip(), extract_answer_from_prediction(a_part)
        return content.strip(), extract_answer_from_prediction(text)
    for tag in ["[कारण:"]:
        if tag in text:
            content = text.split(tag, 1)[1]
            if "]" in content:
                r_part, a_part = content.split("]", 1)
                return r_part.strip(), extract_answer_from_prediction(a_part)
            return content.strip(), extract_answer_from_prediction(text)
    return "", extract_answer_from_prediction(text)


def strict_exact_match(pred: str, gold: str) -> bool:
    """Strict exact match with canonical CoT tag normalization."""
    p = _normalize(pred).rstrip("।.")
    g = _normalize(gold).rstrip("।.")
    if p == g:
        return True
    p_norm = p.replace("<COT_START>", "[कारण:").replace("<COT_END>", "]")
    g_norm = g.replace("<COT_START>", "[कारण:").replace("<COT_END>", "]")
    p_norm = p_norm.replace("<REL_GT>", ">").replace("<REL_LT>", "<").replace("<REL_EQ>", "=").replace("<REL_DISJOINT>", "?")
    g_norm = g_norm.replace("<REL_GT>", ">").replace("<REL_LT>", "<").replace("<REL_EQ>", "=").replace("<REL_DISJOINT>", "?")
    p_norm = __import__("re").sub(r"\s*\]", "]", p_norm)
    g_norm = __import__("re").sub(r"\s*\]", "]", g_norm)
    p_norm = __import__("re").sub(r"\[कारण:\s*", "[कारण: ", p_norm)
    g_norm = __import__("re").sub(r"\[कारण:\s*", "[कारण: ", g_norm)
    return _normalize(p_norm) == _normalize(g_norm)


def extract_semantic_decision(text: str) -> str:
    """Extract relational decision stance from generated text without substring loopholes.
    Returns: 'GREATER' | 'LESS' | 'EQUAL' | 'UNDETERMINED' | 'UNKNOWN'
    """
    cleaned = _normalize(text)

    # 1. Check for indeterminacy
    undet_patterns = [
        "तय नहीं", "निर्धारित नहीं", "जानकारी अपर्याप्त", "कहा नहीं जा सकता",
        "ज्ञात नहीं", "निश्चित नहीं", "स्पष्ट नहीं", "संभव नहीं", "अलग-अलग समूहों", "कोई पथ नहीं",
        "<REL_DISJOINT>", "DISJOINT"
    ]
    if any(pat in cleaned for pat in undet_patterns):
        return "UNDETERMINED"

    # 2. Extract final answer portion if CoT prefix exists
    ans_only = _COT_PREFIX_RE.sub("", cleaned).strip()
    target = ans_only if ans_only else cleaned

    # 3. Equality
    if "बराबर" in target or "समान" in target:
        if "बराबर नहीं" not in target and "समान नहीं" not in target:
            return "EQUAL"

    # 4. Negated comparison
    if "अधिक नहीं" in target or "ज्यादा नहीं" in target:
        return "LESS"
    if "कम नहीं" in target:
        return "GREATER"

    # 5. Direct polarity
    has_gt = any(k in target for k in ["अधिक", "ज्यादा", "बड़ा", "बड़ी"])
    has_lt = any(k in target for k in ["कम", "छोटा", "छोटी"])
    if has_gt and not has_lt:
        return "GREATER"
    if has_lt and not has_gt:
        return "LESS"

    return "UNKNOWN"


def exact_match(pred: str, gold: str) -> bool:
    """Strict Exact Match."""
    return strict_exact_match(pred, gold)


def exact_match_accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    hits = sum(1 for p, g in zip(preds, golds) if strict_exact_match(p, g))
    return hits / len(preds)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Character-level edit distance with unit insert/delete/substitute cost."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if not s2:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def char_similarity(s1: str, s2: str) -> float:
    """Length-normalized character similarity in [0, 1] from Levenshtein distance."""
    p, g = _normalize(s1), _normalize(s2)
    if not p and not g:
        return 1.0
    max_len = max(len(p), len(g))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(p, g)
    return max(0.0, 1.0 - dist / max_len)


def token_f1_score(pred: str, gold: str) -> float:
    """Bag-of-words token F1 between prediction and gold (partial-credit metric)."""
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    from collections import Counter
    p_counts = Counter(p_toks)
    g_counts = Counter(g_toks)
    common = sum((p_counts & g_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall = common / len(g_toks)
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------- model helper

def load_model_and_config(model_config_path: Union[str, Path], vocab_size_override: Optional[int] = None):
    """Build the model from a YAML config, optionally overriding the vocabulary size.

    Pass the tokenizer's full vocabulary size (base BPE + special logic tokens,
    i.e. ``Tokenizer(...).vocab_size``) as ``vocab_size_override``; ``None``
    keeps the config file's value. The size is always caller-derived, never a
    hardcoded constant, so a retrained tokenizer can't silently mismatch the
    embedding table.
    """
    with open(model_config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    # Override vocab_size to include special logic tokens (when provided)
    if vocab_size_override is not None:
        raw_cfg["vocab_size"] = vocab_size_override

    if raw_cfg.get("arch_version") == "v2" or "rope_theta" in raw_cfg or raw_cfg.get("d_ff") == 1376:
        from model.gpt_v2 import GPTConfigV2, GPTLanguageModelV2
        model_cfg = GPTConfigV2.from_yaml(model_config_path)
        if vocab_size_override is not None:
            model_cfg.vocab_size = vocab_size_override
        model = GPTLanguageModelV2(model_cfg)
    else:
        from model.gpt import GPTConfig, GPTLanguageModel
        model_cfg = GPTConfig.from_yaml(model_config_path)
        if vocab_size_override is not None:
            model_cfg.vocab_size = vocab_size_override
        model = GPTLanguageModel(model_cfg)
    return model, model_cfg


# ---------------------------------------------------------------- reasoning eval

@torch.no_grad()
def evaluate_reasoning(
    model: torch.nn.Module,
    tokenizer,
    test_examples: list[dict],
    device: str = "cpu",
    max_new_tokens: int = 120,
    seed: int = 0,
    prompt_bias: float = 3.5,
) -> dict:
    """Generate answers for test examples and score every metric tier.

    Returns strict answer match, full-derivation match, solver-verified CoT
    graph validity, semantic decision stance, token F1, character similarity,
    and decomposed CoT credit — each overall and broken down per paradigm.
    """
    torch.manual_seed(seed)
    model.eval()
    preds: list[str] = []
    golds_ans: list[str] = []
    golds_cot: list[str] = []
    per_paradigm_ans: dict[str, list[float]] = {}
    per_paradigm_cot: dict[str, list[float]] = {}
    per_paradigm_cot_graph: dict[str, list[float]] = {}
    per_paradigm_dec: dict[str, list[float]] = {}
    per_paradigm_f1_ans: dict[str, list[float]] = {}
    per_paradigm_f1_cot: dict[str, list[float]] = {}
    per_paradigm_sim_ans: dict[str, list[float]] = {}
    per_paradigm_decomp: dict[str, list[float]] = {}

    for ex in test_examples:
        prompt_ids = tokenizer.encode(ex["text"])
        if not prompt_ids:
            continue
        idx = torch.tensor(
            [prompt_ids[: model.config.block_size - max_new_tokens]],
            dtype=torch.long,
            device=device,
        )
        extra_allowed = None
        if prompt_bias > 0.0 and extract_prompt_lemma_token_ids is not None:
            extra_allowed = extract_prompt_lemma_token_ids(ex["text"], tokenizer)

        gen = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.15,
            eos_id=getattr(tokenizer, "eos_id", None),
            prompt_bias=prompt_bias,
            extra_allowed_tokens=extra_allowed,
        )
        raw_pred = tokenizer.decode(gen[0, idx.shape[1] :].tolist()).strip()
        pred_rat, pred_ans = extract_rationale_and_answer(raw_pred)
        gold_ans = ex["answer_text"]
        gold_cot = ex.get("cot_text", gold_ans)
        if pred_rat:
            if "<COT_START>" in gold_cot or "<COT_START>" in raw_pred:
                pred = f"<COT_START> {pred_rat} <COT_END> {pred_ans}"
            else:
                pred = f"[कारण: {pred_rat}] {pred_ans}"
        else:
            pred = pred_ans if pred_ans else raw_pred
        query = tuple(ex.get("query", []))
        ans_rel = ex.get("answer_rel", "")

        pred_ans = extract_answer_from_prediction(pred)
        hit_ans = 1.0 if strict_exact_match(pred_ans, gold_ans) else 0.0
        hit_cot = 1.0 if strict_exact_match(pred, gold_cot) else 0.0
        hit_cot_graph = 1.0 if verify_cot_derivation(pred, query, ans_rel) else 0.0

        pred_dec = extract_semantic_decision(pred)
        gold_dec = extract_semantic_decision(gold_ans)
        hit_dec = 1.0 if (pred_dec == gold_dec and pred_dec != "UNKNOWN") else 0.0

        f1_ans = token_f1_score(pred_ans, gold_ans)
        f1_cot = token_f1_score(pred, gold_cot)
        sim_ans = char_similarity(pred_ans, gold_ans)

        pred_rat, _ = extract_rationale_and_answer(pred)
        gold_rat, _ = extract_rationale_and_answer(gold_cot)
        rat_f1 = token_f1_score(pred_rat, gold_rat)
        decomp_score = 0.4 * rat_f1 + 0.6 * f1_ans

        preds.append(pred)
        golds_ans.append(gold_ans)
        golds_cot.append(gold_cot)
        p_name = ex.get("paradigm", "general")
        per_paradigm_ans.setdefault(p_name, []).append(hit_ans)
        per_paradigm_cot.setdefault(p_name, []).append(hit_cot)
        per_paradigm_cot_graph.setdefault(p_name, []).append(hit_cot_graph)
        per_paradigm_dec.setdefault(p_name, []).append(hit_dec)
        per_paradigm_f1_ans.setdefault(p_name, []).append(f1_ans)
        per_paradigm_f1_cot.setdefault(p_name, []).append(f1_cot)
        per_paradigm_sim_ans.setdefault(p_name, []).append(sim_ans)
        per_paradigm_decomp.setdefault(p_name, []).append(decomp_score)

    n = len(preds)
    acc_ans = sum(sum(v) for v in per_paradigm_ans.values()) / max(n, 1)
    acc_cot = sum(sum(v) for v in per_paradigm_cot.values()) / max(n, 1)
    acc_cot_graph = sum(sum(v) for v in per_paradigm_cot_graph.values()) / max(n, 1)
    acc_dec = sum(sum(v) for v in per_paradigm_dec.values()) / max(n, 1)
    avg_f1_ans = sum(sum(v) for v in per_paradigm_f1_ans.values()) / max(n, 1)
    avg_f1_cot = sum(sum(v) for v in per_paradigm_f1_cot.values()) / max(n, 1)
    avg_sim_ans = sum(sum(v) for v in per_paradigm_sim_ans.values()) / max(n, 1)
    avg_decomp = sum(sum(v) for v in per_paradigm_decomp.values()) / max(n, 1)

    return {
        "accuracy_answer_only": round(acc_ans, 6),
        "accuracy_exact_match": round(acc_cot, 6),
        "accuracy_cot_graph_valid": round(acc_cot_graph, 6),
        "accuracy_semantic_decision": round(acc_dec, 6),
        "f1_answer_only": round(avg_f1_ans, 6),
        "f1_exact_match": round(avg_f1_cot, 6),
        "char_similarity_answer_only": round(avg_sim_ans, 6),
        "cot_decomposed_score": round(avg_decomp, 6),
        "n_examples": n,
        "per_paradigm_accuracy_answer_only": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(per_paradigm_ans.items())
        },
        "per_paradigm_accuracy_cot_graph_valid": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(per_paradigm_cot_graph.items())
        },
        "per_paradigm_decision_accuracy": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(per_paradigm_dec.items())
        },
        "per_paradigm_f1_answer_only": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(per_paradigm_f1_ans.items())
        },
        "per_paradigm_accuracy": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(per_paradigm_cot.items())
        },
    }


# ---------------------------------------------------------------- main pipeline

def finetune(
    pretrained_ckpt: str,
    model_config_path: str,
    tokenizer_path: str,
    data_dir: str,
    out_dir: str,
    device: Optional[str] = None,
    max_steps: int = FINETUNE_MAX_STEPS,
    lr: float = FINETUNE_LR,
    n_val: int = 50,
    n_test: int = 100,
    use_cot: bool = True,
    seed: int = 42,
    custom_ckpt_name: Optional[str] = None,
    max_epochs: Optional[float] = None,
    early_stop_patience: int = 3,
    eval_interval: Optional[int] = None,
) -> dict:
    """Run one full SFT experiment (Direct when use_cot=False, CoT otherwise).

    Loads the reasoning splits, builds the model from its pretrained checkpoint,
    trains with prompt-masked loss plus stance-margin and entity-coverage
    auxiliaries (best-validation restore with early stopping), evaluates the
    pretrained baseline against the finetuned model, and persists the
    checkpoint, eval JSONs, and a results dict with deltas.

    When ``max_epochs`` is set, steps are capped at
    ``max_epochs * (n_train_samples // effective_batch_size)`` so a run never
    trains past the requested epoch budget even if ``max_steps`` is larger.
    """
    from train.train import TrainConfig, Trainer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_str = "cot" if use_cot else "direct"

    # --- Data
    train_ex = load_reasoning_jsonl(data_dir / "train.jsonl")
    val_ex = load_reasoning_jsonl(data_dir / "val.jsonl")
    test_ex = load_reasoning_jsonl(data_dir / "test.jsonl")
    if len(test_ex) > n_test:
        rng = np.random.default_rng(seed)
        test_ex = [test_ex[i] for i in rng.choice(len(test_ex), n_test, replace=False)]
    if len(val_ex) > n_val:
        rng = np.random.default_rng(seed + 1)
        # Balanced stratified sampling across all paradigms
        by_paradigm: dict[str, list[dict]] = {}
        for ex in val_ex:
            by_paradigm.setdefault(ex.get("paradigm", "general"), []).append(ex)
        
        per_p = n_val // max(len(by_paradigm), 1)
        stratified_val: list[dict] = []
        for p_name, p_list in sorted(by_paradigm.items()):
            take = min(len(p_list), per_p)
            selected_indices = rng.choice(len(p_list), take, replace=False)
            stratified_val.extend([p_list[i] for i in selected_indices])
            
        if len(stratified_val) < n_val:
            rem = [ex for ex in val_ex if ex not in stratified_val]
            if rem:
                take_more = min(len(rem), n_val - len(stratified_val))
                stratified_val.extend([rem[i] for i in rng.choice(len(rem), take_more, replace=False)])
        val_ex = stratified_val

    # --- Model + Tokenizer (tokenizer first: its vocab size sizes the embedding table)
    tokenizer = Tokenizer(tokenizer_path)
    model, model_cfg = load_model_and_config(model_config_path, vocab_size_override=tokenizer.vocab_size)
    load_checkpoint(pretrained_ckpt, model, restore_rng=False)

    # Inject SFT dropout regularization (especially for V2 which defaults to 0.0)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.1

    # Tight sequence length tailored to reasoning task (160 for CoT, 128 for Direct),
    # reducing attention compute from O(256^2) to O(128^2)—up to a 4x attention speedup.
    finetune_block = 160 if use_cot else 128
    block_size = min(finetune_block, model_cfg.block_size)
    train_data = PromptMaskedSFTDataset(train_ex, tokenizer, block_size=block_size, use_cot=use_cot)
    val_data = PromptMaskedSFTDataset(val_ex, tokenizer, block_size=block_size, use_cot=use_cot)

    eff_batch = 32
    micro = 32  # at T<=160, micro=32 fits easily on any GPU, yielding 1-step updates (accum=1)
    accum = eff_batch // micro
    is_v2 = getattr(model_cfg, "arch_version", None) == "v2" or hasattr(model_cfg, "rope_theta")
    eff_warmup = min(20, max_steps // 10) if is_v2 else min(10, max_steps // 10)
    eff_wd = 0.05 if is_v2 else 0.1
    eff_lr = (lr * 0.85) if (is_v2 and lr <= 3.5e-5) else lr  # 2.5e-5 for stable SwiGLU fine-tuning

    if max_epochs is not None:
        epoch_steps = max(1, len(train_data.samples) // eff_batch)
        capped = max(1, int(max_epochs * epoch_steps))
        if capped < max_steps:
            print(f"[{mode_str.upper()}] Capping max_steps {max_steps} -> {capped} "
                  f"({max_epochs} epoch(s) x {epoch_steps} steps/epoch)", flush=True)
            max_steps = capped

    actual_eval_interval = eval_interval or min(50, max(1, max_steps // 10))
    train_cfg = TrainConfig(
        micro_batch_size=micro,
        gradient_accumulation_steps=accum,
        effective_batch_size=eff_batch,
        learning_rate=eff_lr,
        min_lr=eff_lr / 10,
        warmup_steps=eff_warmup,
        max_steps=max_steps,
        max_grad_norm=1.0,
        weight_decay=eff_wd,
        eval_interval=actual_eval_interval,
        save_interval=max_steps,
        eval_batches=8,
        mixed_precision=device.startswith("cuda"),
        seed=seed,
        early_stop_patience=early_stop_patience,
        stance_margin_weight=1.5,
        stance_margin_gamma=2.0,
        notes=f"finetune ({mode_str})",
    )

    ckpt_subdir = out_dir / f"checkpoints_{mode_str}"
    trainer = Trainer(
        model,
        train_data,
        val_data,
        train_cfg,
        checkpoint_dir=str(ckpt_subdir),
        device=device,
    )

    eval_max_tokens = 96 if use_cot else 36
    # --- Pretrained Baseline Evaluation (Zero-shot)
    baseline = evaluate_reasoning(
        model, tokenizer, test_ex, device=device, max_new_tokens=eval_max_tokens, seed=seed
    )
    print(f"[{mode_str.upper()}] PRETRAINED BASELINE: {json.dumps(baseline, ensure_ascii=False)}")

    # --- Training Loop
    trainer.train(max_steps=max_steps, resume=True)

    # --- Restore Best Validation Checkpoint (Avoids Overfitting & Val Loss Creep)
    best_ckpt = ckpt_subdir / "best.pt"
    if best_ckpt.exists():
        print(f"[{mode_str.upper()}] Restoring best validation checkpoint from {best_ckpt}...", flush=True)
        load_checkpoint(str(best_ckpt), model, restore_rng=False)

    # --- Save Designated Checkpoint
    ckpt_name = custom_ckpt_name or f"finetuned_{mode_str}.pt"
    finetuned_ckpt = str(out_dir / ckpt_name)
    save_checkpoint(
        finetuned_ckpt,
        model,
        trainer.optimizer,
        trainer.scheduler,
        max_steps,
        config={"model": vars(model_cfg), "train": train_cfg.to_dict()},
    )
    print(f"[{mode_str.upper()}] Saved fine-tuned checkpoint: {finetuned_ckpt}")

    # --- Fine-Tuned Evaluation
    finetuned = evaluate_reasoning(
        model, tokenizer, test_ex, device=device, max_new_tokens=eval_max_tokens, seed=seed
    )
    print(f"[{mode_str.upper()}] FINETUNED: {json.dumps(finetuned, ensure_ascii=False)}")

    results = {
        "mode": mode_str,
        "pretrained_baseline": baseline,
        "finetuned": finetuned,
        "delta_accuracy": round(
            finetuned["accuracy_answer_only"] - baseline["accuracy_answer_only"], 6
        ),
        "delta_accuracy_answer_only": round(
            finetuned["accuracy_answer_only"] - baseline["accuracy_answer_only"], 6
        ),
        "delta_accuracy_cot_exact": round(
            finetuned["accuracy_exact_match"] - baseline["accuracy_exact_match"], 6
        ),
        "config": {
            "lr": eff_lr,  # effective LR actually used (differs from `lr` for V2)
            "max_steps": max_steps,
            "block_size": block_size,
            "n_train_examples": len(train_ex),
            "n_val_examples": len(val_ex),
            "n_test_examples": len(test_ex),
            "checkpoint": finetuned_ckpt,
        },
    }

    eval_json_path = out_dir / f"eval_results_{mode_str}.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Also write to top-level eval_results.json for standard compatibility
    with open(out_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Finetune on reasoning data")
    parser.add_argument("--pretrained-ckpt", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--data-dir", default=str(_LANG_ROOT / "finetune" / "reasoning"))
    parser.add_argument("--out-dir", default=str(_LANG_ROOT / "finetune" / "out"))
    parser.add_argument("--max-steps", type=int, default=FINETUNE_MAX_STEPS)
    parser.add_argument("--lr", type=float, default=FINETUNE_LR)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--mode", choices=["direct", "cot", "both"], default="both")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.mode in ("direct", "both"):
        print("\n--- Running Direct SFT ---")
        finetune(
            args.pretrained_ckpt, args.model_config, args.tokenizer,
            args.data_dir, args.out_dir,
            device=args.device, max_steps=args.max_steps, lr=args.lr,
            n_val=args.n_val, n_test=args.n_test, use_cot=False,
            custom_ckpt_name="finetuned_direct.pt",
        )

    if args.mode in ("cot", "both"):
        print("\n--- Running CoT SFT ---")
        finetune(
            args.pretrained_ckpt, args.model_config, args.tokenizer,
            args.data_dir, args.out_dir,
            device=args.device, max_steps=args.max_steps, lr=args.lr,
            n_val=args.n_val, n_test=args.n_test, use_cot=True,
            custom_ckpt_name="finetuned_cot.pt",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
