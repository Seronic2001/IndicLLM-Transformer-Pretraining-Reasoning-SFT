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

from common.checkpoint import checkpoint_paths_in_dir, save_checkpoint, load_checkpoint
from tokenizer.tokenizer import Tokenizer

# Defaults are the best recipe of the 2-seed sweep (finetuned-logs-summary-v6.txt):
# Direct + realmix (half the training names real): test relation 0.945 (V1), 0.880 (V2).
# Direct beat CoT and dropout 0.1 stopped most cells learning. The checkpoint is selected
# on generated answers, not on the validation loss, which rises while answers improve.
FINETUNE_LR = 3e-5
FINETUNE_BATCH_SIZE = 32
FINETUNE_MAX_STEPS = 12_000  # up to 12,000 steps, early-stopped
FINETUNE_LR_SCHEDULE = "cosine"
FINETUNE_WARMUP_STEPS = 300
FINETUNE_COT_WARMUP_STEPS = 0  # Direct only: CoT targets for these steps first
FINETUNE_MODE = "direct"
FINETUNE_N_VAL = 1000
FINETUNE_N_TEST = 2000
FINETUNE_PROBE_INTERVAL = 500  # 10 probes per run could not locate a peak finer than ~900 steps
FINETUNE_PATIENCE = 3          # probes without improvement, counted after onset
FINETUNE_DROPOUT = 0.0     # as in pretraining and the lr 1e-4 run that learned the relation
FINETUNE_BLOCK_SIZE = 192  # longest prompt + CoT target is ~145 tokens
MAX_NEW_TOKENS = 80        # longest CoT target is ~65 tokens
N_PROBES = 10              # generative validation probes spread over the run
N_PROBE_EXAMPLES = 500     # validation examples generated per probe (the whole val subset)
# Greedy decoding with no repetition penalty: answers and CoT steps must copy the
# entity names and relation words from the prompt, which a penalty pushes away from.
REPETITION_PENALTY = 1.0
GEN_BATCH_SIZE = 32        # equal-length prompts generated together in evaluation
# Checkpoint selection. "accuracy_answer_strict" requires the entity names to be spelt
# exactly, which on unseen names measures copying as much as reasoning;
# "accuracy_relation" scores the relation the answer asserts and tolerates misspelt
# names, so it is the default.
SELECTION_METRICS = ("accuracy_answer_strict", "accuracy_relation")
DEFAULT_SELECTION_METRIC = "accuracy_relation"
RELATION_MATCH_THRESHOLD = 0.8  # char similarity a rendered answer needs to count as asserted
# Early stopping counts probes without improvement only after onset, i.e. once the
# best probe beats the majority-relation rate by ONSET_MARGIN: some runs sit at chance
# for thousands of steps before learning the task, and must not be stopped there.
ONSET_MARGIN = 0.10
EXTRA_EVAL_FILES = ("test_balanced", "test_depth2", "test_depth3", "test_depth4", "test_depth5",
                    "val_train_names")


# ---------------------------------------------------------------- data conversion

def load_reasoning_jsonl(path: Union[str, Path]) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


class PromptMaskedSFTDataset:
    """Dataset of (prompt, completion) sequences where prompt tokens have label=-100.

    The completion ends with EOS so the model learns where the answer stops.
    """

    def __init__(self, examples: list[dict], tokenizer, block_size: int = 128, use_cot: bool = True):
        self.block_size = block_size
        self.samples = []
        self.hops: list[int] = []  # query distance of each sample, for the curriculum
        self.active: Optional[torch.Tensor] = None  # sample indices get_batch draws from
        for ex in examples:
            prompt_text = ex["text"]
            ans_text = ex.get("cot_text", ex.get("answer_text", "")) if use_cot else ex.get("answer_text", "")

            p_ids = tokenizer.encode(prompt_text)
            a_ids = tokenizer.encode(" " + ans_text)
            if not a_ids:
                continue
            a_ids = a_ids + [tokenizer.eos_id]

            max_p_len = max(1, block_size - len(a_ids))
            p_ids = p_ids[:max_p_len]
            input_ids = p_ids + a_ids
            if len(input_ids) < 2:
                continue
            input_ids = input_ids[: block_size + 1]

            prompt_len = len(p_ids)
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            n_real = len(input_ids) - 1  # positions before the padding

            pad_len = (block_size + 1) - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len
                labels = labels + [-100] * pad_len

            target_slice = labels[1:]
            if not any(lbl != -100 for lbl in target_slice):
                continue

            x = torch.tensor(input_ids[:-1], dtype=torch.long)
            y = torch.tensor(target_slice, dtype=torch.long)
            self.samples.append((x, y, n_real))
            self.hops.append(query_hops(ex) if "premises" in ex else 1)

    def __len__(self) -> int:
        return len(self.samples)

    def set_max_hops(self, max_hops: Optional[int]) -> None:
        """Sample only examples whose query is at most max_hops apart (None: all)."""
        if max_hops is None:
            self.active = None
            return
        idx = [i for i, h in enumerate(self.hops) if h <= max_hops]
        if not idx:
            raise ValueError(f"no training examples with at most {max_hops} hops")
        self.active = torch.tensor(idx, dtype=torch.long)

    def get_batch(
        self, batch_size: int, device: str, generator: Optional[torch.Generator] = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.samples) if self.active is None else len(self.active)
        if generator is not None:
            indices = torch.randint(n, (batch_size,), generator=generator)
        else:
            indices = torch.randint(n, (batch_size,))
        if self.active is not None:
            indices = self.active[indices]
        # Trailing padding is cut down to the longest sequence in the batch. The model
        # is causal and padded positions carry label -100, so the loss is unchanged,
        # but most sequences are far shorter than block_size.
        width = max(self.samples[i][2] for i in indices)
        bx = torch.stack([self.samples[i][0][:width] for i in indices]).to(device)
        by = torch.stack([self.samples[i][1][:width] for i in indices]).to(device)
        return bx, by


def query_hops(ex: dict) -> int:
    """Number of premises on the path between the two query entities."""
    neighbours: dict[str, list[str]] = {}
    for s, _rel, o in ex["premises"]:
        neighbours.setdefault(s, []).append(o)
        neighbours.setdefault(o, []).append(s)
    start, goal = ex["query"]
    dist = {start: 0}
    frontier = [start]
    while frontier:
        nxt = []
        for cur in frontier:
            if cur == goal:
                return dist[cur]
            for n in neighbours.get(cur, ()):
                if n not in dist:
                    dist[n] = dist[cur] + 1
                    nxt.append(n)
        frontier = nxt
    return len(ex["premises"])  # not reached: a solvable query is connected


def curriculum_max_hops(step: int, curriculum_steps: int) -> Optional[int]:
    """Hop cap at a training step: 1-hop queries for the first half of the curriculum,
    up to 2 hops for the second half, then everything."""
    if step < curriculum_steps // 2:
        return 1
    if step < curriculum_steps:
        return 2
    return None


class NameSplitTokenizer:
    """Tokenizer wrapper that encodes each entity name the same way wherever it occurs.

    Assamese fuses the case ending onto the name, and the BPE often merges it into the
    name's last token (টন্তল -> ▁ট ন্ত ল, but টন্তলৰ -> ▁ট ন্ত লৰ), so one entity
    can be two token sequences in one example. Here a word made of a known name plus a
    short ending is encoded as the bare name's tokens followed by the ending's own
    tokens. SentencePiece encodes words independently, so the text is encoded word by
    word, which matches the plain tokenizer exactly for every other word. Decoding is
    unchanged: the ending's pieces carry no word-boundary marker, so they join the name.
    """

    MAX_SUFFIX = 3     # characters of case ending split off a name
    MAX_CACHE = 200_000

    def __init__(self, tokenizer, names):
        self.base = tokenizer
        self.names = set(names)
        self._cache: dict[str, list[int]] = {}
        self.n_split = 0  # name+ending words split (a cached word is counted once)

    def __getattr__(self, attr):  # decode, eos_id, sp, ... come from the wrapped tokenizer
        if attr == "base":
            raise AttributeError(attr)
        return getattr(self.base, attr)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for word in text.split():
            ids = self._cache.get(word)
            if ids is None:
                ids = self._encode_word(word)
                if len(self._cache) < self.MAX_CACHE:
                    self._cache[word] = ids
            out.extend(ids)
        return out

    def _encode_word(self, word: str) -> list[int]:
        if self.base.sp is not None and word not in self.names:  # longest name wins
            for k in range(1, min(self.MAX_SUFFIX, len(word) - 1) + 1):
                if word[:-k] in self.names:
                    self.n_split += 1
                    return self.base.encode(word[:-k]) + self._bare_ids(word[-k:])
        return self.base.encode(word)

    def _bare_ids(self, text: str) -> list[int]:
        """Ids of text with no word-boundary marker in front (byte pieces as fallback)."""
        sp = self.base.sp
        pieces = sp.encode(text, out_type=str)
        pieces = pieces[1:] if pieces[0] == "\u2581" else [pieces[0].lstrip("\u2581")] + pieces[1:]
        ids = []
        for piece in pieces:
            pid = sp.piece_to_id(piece)
            if pid == sp.unk_id():
                ids.extend(sp.piece_to_id(f"<0x{b:02X}>") for b in piece.encode("utf-8"))
            else:
                ids.append(pid)
        return ids


def build_finetune_bin(
    examples: list[dict], tokenizer, out_path: str, max_seq_len: int = 256, use_cot: bool = True
) -> int:
    all_ids: list[int] = []
    for ex in examples:
        ans = ex.get("cot_text", ex.get("answer_text", "")) if use_cot else ex.get("answer_text", "")
        full_text = ex["text"] + " " + ans
        ids = tokenizer.encode(full_text.strip())[:max_seq_len]
        all_ids.extend(ids)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(out_path)
    return int(arr.size)


# ---------------------------------------------------------------- exact match & multi-tier metrics

_COT_PREFIX_RE = __import__("re").compile(r"\[कारण:[^\]]*\]\s*")
_COT_EXTRACT_RE = __import__("re").compile(r"\[कारण:\s*([^\]]*)\]")


def _normalize(text: str) -> str:
    return " ".join(text.split())


def extract_answer_from_prediction(pred: str) -> str:
    return _COT_PREFIX_RE.sub("", pred).strip()


def extract_rationale_and_answer(text: str) -> tuple[str, str]:
    match = _COT_EXTRACT_RE.search(text)
    if match:
        rationale = match.group(1).strip()
        answer = _COT_EXTRACT_RE.sub("", text).strip()
        return rationale, answer
    return "", text.strip()


def exact_match(pred: str, gold: str) -> bool:
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return p == g
    return g in p or p in g


SENTENCE_END = "\u0964"  # danda: every gold answer is one sentence ending with it


def strict_match(pred: str, gold: str) -> bool:
    """Equality of the prediction's first sentence with the gold answer.

    exact_match above accepts any substring in either direction (a lone "है"
    matches every gold, as does the right answer followed by its opposite).
    Here the prediction is cut after its first sentence end, so trailing text
    from a model that does not stop is ignored, and the rest must be identical.
    """
    end = pred.find(SENTENCE_END)
    if end != -1:
        pred = pred[: end + 1]
    return _normalize(pred) == _normalize(gold)


def exact_match_accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    hits = sum(1 for p, g in zip(preds, golds) if exact_match(p, g))
    return hits / len(preds)


def levenshtein_distance(s1: str, s2: str) -> int:
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
    p, g = _normalize(s1), _normalize(s2)
    if not p and not g:
        return 1.0
    max_len = max(len(p), len(g))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(p, g)
    return max(0.0, 1.0 - dist / max_len)


def token_f1_score(pred: str, gold: str) -> float:
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

def load_model_and_config(model_config_path: Union[str, Path]):
    with open(model_config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    if raw_cfg.get("arch_version") == "v2" or "rope_theta" in raw_cfg:
        from model.gpt_v2 import GPTConfigV2, GPTLanguageModelV2
        model_cfg = GPTConfigV2.from_yaml(model_config_path)
        model = GPTLanguageModelV2(model_cfg)
    else:
        from model.gpt import GPTConfig, GPTLanguageModel
        model_cfg = GPTConfig.from_yaml(model_config_path)
        model = GPTLanguageModel(model_cfg)
    return model, model_cfg


# ---------------------------------------------------------------- relation-level metrics

_FLIP_REL = {"<": ">", ">": "<", "=": "="}


def asserted_relation(pred: str, ex: dict) -> Optional[str]:
    """The relation, in query order, that the first sentence of pred asserts.

    The sentence is matched against every rendering of the answer template (3 relations
    x 2 phrasings x 2 entity orders) and the closest one wins, so a misspelt name or a
    missing space does not hide the relation. Negated phrasing is just another wording
    of the same relation (it renders the opposite word), so it does not change the
    result. None when nothing is close enough, i.e. the answer is not well-formed.
    """
    from finetune.generate_reasoning import render_answer

    end = pred.find(SENTENCE_END)
    sentence = _normalize(pred[: end + 1] if end != -1 else pred)
    if not sentence:
        return None
    qa, qb = ex["query"]
    attr = ex["attribute"]
    best_sim, best_rel = 0.0, None
    for rel in "<=>":
        for negate in (False, True):
            for a, b, swapped in ((qa, qb, False), (qb, qa, True)):
                sim = char_similarity(sentence, render_answer(a, b, rel, attr, negate=negate))
                if sim > best_sim:
                    best_sim, best_rel = sim, (_FLIP_REL[rel] if swapped else rel)
    return best_rel if best_sim >= RELATION_MATCH_THRESHOLD else None


def relation_metrics(gold: list[str], pred: list[Optional[str]]) -> dict:
    """Accuracy, per-relation recall and macro-F1 over the relations > < =."""
    n = max(len(gold), 1)
    out: dict = {"accuracy_relation": round(sum(g == p for g, p in zip(gold, pred)) / n, 6)}
    recall, f1s = {}, []
    for rel in "<=>":
        tp = sum(g == rel and p == rel for g, p in zip(gold, pred))
        n_gold = sum(g == rel for g in gold)
        n_pred = sum(p == rel for p in pred)
        if not n_gold:
            continue
        precision = tp / n_pred if n_pred else 0.0
        rec = tp / n_gold
        recall[rel] = round(rec, 6)
        f1s.append(2 * precision * rec / (precision + rec) if precision + rec else 0.0)
    out["per_relation_recall"] = recall
    out["macro_f1_relation"] = round(sum(f1s) / max(len(f1s), 1), 6)
    out["n_per_relation"] = {rel: sum(g == rel for g in gold) for rel in "<=>"}
    return out


# ---------------------------------------------------------------- reasoning eval

def set_dropout(model: torch.nn.Module, p: float) -> None:
    """Set every dropout rate in a built model (V1 uses nn.Dropout everywhere; V2's
    attention reads a dropout_p attribute)."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p
        if hasattr(module, "dropout_p"):
            module.dropout_p = p
    model.config.dropout = p


def chance_baselines(examples: list[dict]) -> dict:
    """Accuracy of answering every example with the same relation.

    Answers differ only in the relation word, so the most frequent relation is the
    accuracy a model gets by learning the answer format without the reasoning.
    """
    counts: dict[str, int] = {}
    for ex in examples:
        rel = ex["answer_rel"].split()[1]
        counts[rel] = counts.get(rel, 0) + 1
    n = max(sum(counts.values()), 1)
    distribution = {rel: round(c / n, 6) for rel, c in sorted(counts.items())}
    majority = max(distribution, key=distribution.get)
    return {
        "relation_distribution": distribution,
        "majority_relation": majority,
        "majority_accuracy": distribution[majority],
    }


@torch.no_grad()
def generate_predictions(
    model: torch.nn.Module,
    tokenizer,
    examples: list[dict],
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[Optional[str]]:
    """Greedy prediction for each example, in order (None where the prompt is empty).

    Prompts with the same token length are generated as one batch, so no padding or
    attention mask is needed and each row is the same computation as generating it
    alone; generating one prompt at a time left the GPU idle and dominated run time.
    """
    eos_id = getattr(tokenizer, "eos_id", None)
    max_prompt_len = model.config.block_size - max_new_tokens
    prompts = [tokenizer.encode(ex["text"])[:max_prompt_len] for ex in examples]
    by_length: dict[int, list[int]] = {}
    for i, prompt_ids in enumerate(prompts):
        if prompt_ids:
            by_length.setdefault(len(prompt_ids), []).append(i)

    preds: list[Optional[str]] = [None] * len(examples)
    for length, members in sorted(by_length.items()):
        for start in range(0, len(members), GEN_BATCH_SIZE):
            batch = members[start : start + GEN_BATCH_SIZE]
            idx = torch.tensor([prompts[i] for i in batch], dtype=torch.long, device=device)
            gen = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                repetition_penalty=REPETITION_PENALTY,
                eos_id=eos_id,
            )
            for i, new_ids in zip(batch, gen[:, length:].tolist()):
                if eos_id in new_ids:
                    new_ids = new_ids[: new_ids.index(eos_id)]
                preds[i] = tokenizer.decode(new_ids)
    return preds


@torch.no_grad()
def evaluate_reasoning(
    model: torch.nn.Module,
    tokenizer,
    test_examples: list[dict],
    device: str = "cpu",
    max_new_tokens: int = MAX_NEW_TOKENS,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    model.eval()
    preds: list[str] = []
    golds_ans: list[str] = []
    golds_cot: list[str] = []
    per_paradigm_ans: dict[str, list[float]] = {}
    per_paradigm_cot: dict[str, list[float]] = {}
    per_paradigm_f1_ans: dict[str, list[float]] = {}
    per_paradigm_f1_cot: dict[str, list[float]] = {}
    per_paradigm_sim_ans: dict[str, list[float]] = {}
    per_paradigm_decomp: dict[str, list[float]] = {}
    strict_ans_hits = 0
    strict_cot_hits = 0
    gold_rels: list[str] = []
    pred_rels: list[Optional[str]] = []
    name_hits = 0

    all_preds = generate_predictions(model, tokenizer, test_examples, device, max_new_tokens)
    for ex, pred in zip(test_examples, all_preds):
        if pred is None:
            continue
        gold_ans = ex["answer_text"]
        gold_cot = ex.get("cot_text", gold_ans)

        pred_ans = extract_answer_from_prediction(pred)
        gold_rels.append(ex["answer_rel"].split()[1])
        pred_rels.append(asserted_relation(pred_ans, ex))
        first_sentence = pred_ans.split(SENTENCE_END, 1)[0]
        name_hits += all(name in first_sentence for name in ex["query"])
        hit_ans = 1.0 if exact_match(pred_ans, gold_ans) else 0.0
        hit_cot = 1.0 if exact_match(pred, gold_cot) else 0.0
        strict_ans_hits += strict_match(pred_ans, gold_ans)
        strict_cot_hits += strict_match(pred, gold_cot)

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
        per_paradigm_f1_ans.setdefault(p_name, []).append(f1_ans)
        per_paradigm_f1_cot.setdefault(p_name, []).append(f1_cot)
        per_paradigm_sim_ans.setdefault(p_name, []).append(sim_ans)
        per_paradigm_decomp.setdefault(p_name, []).append(decomp_score)

    n = len(preds)
    acc_ans = sum(sum(v) for v in per_paradigm_ans.values()) / max(n, 1)
    acc_cot = sum(sum(v) for v in per_paradigm_cot.values()) / max(n, 1)
    avg_f1_ans = sum(sum(v) for v in per_paradigm_f1_ans.values()) / max(n, 1)
    avg_f1_cot = sum(sum(v) for v in per_paradigm_f1_cot.values()) / max(n, 1)
    avg_sim_ans = sum(sum(v) for v in per_paradigm_sim_ans.values()) / max(n, 1)
    avg_decomp = sum(sum(v) for v in per_paradigm_decomp.values()) / max(n, 1)

    return {
        "accuracy_answer_only": round(acc_ans, 6),
        "accuracy_exact_match": round(acc_cot, 6),
        "accuracy_answer_strict": round(strict_ans_hits / max(n, 1), 6),
        "accuracy_cot_strict": round(strict_cot_hits / max(n, 1), 6),
        "f1_answer_only": round(avg_f1_ans, 6),
        "f1_exact_match": round(avg_f1_cot, 6),
        "char_similarity_answer_only": round(avg_sim_ans, 6),
        "cot_decomposed_score": round(avg_decomp, 6),
        "n_examples": n,
        **relation_metrics(gold_rels, pred_rels),
        "accuracy_name_copy": round(name_hits / max(n, 1), 6),
        "per_paradigm_accuracy_answer_only": {
            k: round(sum(v) / len(v), 6) for k, v in sorted(per_paradigm_ans.items())
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
    n_val: int = FINETUNE_N_VAL,
    n_test: int = FINETUNE_N_TEST,
    use_cot: bool = True,
    seed: int = 42,
    custom_ckpt_name: Optional[str] = None,
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    min_lr: Optional[float] = None,
    warmup_steps: Optional[int] = FINETUNE_WARMUP_STEPS,
    lr_schedule: str = FINETUNE_LR_SCHEDULE,
    probe_interval: Optional[int] = FINETUNE_PROBE_INTERVAL,
    patience: Optional[int] = FINETUNE_PATIENCE,
    onset_margin: float = ONSET_MARGIN,
    overfit: Optional[int] = None,
    split_names: bool = False,
    dropout: float = FINETUNE_DROPOUT,
    weight_decay: float = 0.1,
    curriculum_steps: int = 0,
    cot_warmup_steps: int = 0,
) -> dict:
    """Finetune one model in one mode and evaluate it.

    curriculum_steps > 0 trains on 1-hop queries first, then up to 2 hops, then all
    (see curriculum_max_hops); cot_warmup_steps > 0 (Direct only) trains on CoT targets
    first and switches to Direct targets afterwards, and only probes after the switch
    can be selected. Both take effect at probe boundaries, so they are rounded up to a
    multiple of the probe interval.
    """
    from train.train import TrainConfig, Trainer

    probe_interval = probe_interval or max(1, max_steps // N_PROBES)
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}, got {selection_metric!r}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_str = "cot" if use_cot else "direct"
    if cot_warmup_steps and use_cot:
        raise ValueError("cot_warmup_steps only applies to Direct finetuning")
    if cot_warmup_steps > max_steps - probe_interval:
        raise ValueError("cot_warmup_steps must leave at least one probe interval of Direct training")

    # --- Data
    train_ex = load_reasoning_jsonl(data_dir / "train.jsonl")
    val_ex = load_reasoning_jsonl(data_dir / "val.jsonl")
    test_ex = load_reasoning_jsonl(data_dir / "test.jsonl")
    names = {n for split in (train_ex, val_ex, test_ex) for ex in split for n in ex["entities"]}
    for name in EXTRA_EVAL_FILES:
        if (data_dir / f"{name}.jsonl").exists():
            names.update(n for ex in load_reasoning_jsonl(data_dir / f"{name}.jsonl") for n in ex["entities"])
    if len(test_ex) > n_test:
        rng = np.random.default_rng(seed)
        test_ex = [test_ex[i] for i in rng.choice(len(test_ex), n_test, replace=False)]
    if len(val_ex) > n_val:
        rng = np.random.default_rng(seed + 1)
        val_ex = [val_ex[i] for i in rng.choice(len(val_ex), n_val, replace=False)]
    if overfit:
        # Capacity check: train on a small subset and validate on those same examples,
        # so a model that cannot reach high validation accuracy cannot fit the task at all.
        train_ex = train_ex[:overfit]
        val_ex = train_ex[:n_val]

    # --- Model + Tokenizer
    model, model_cfg = load_model_and_config(model_config_path)
    tokenizer = Tokenizer(tokenizer_path)
    if split_names:
        tokenizer = NameSplitTokenizer(tokenizer, names)
    load_checkpoint(pretrained_ckpt, model, restore_rng=False)
    set_dropout(model, dropout)

    block_size = min(FINETUNE_BLOCK_SIZE, model_cfg.block_size)
    train_data = PromptMaskedSFTDataset(train_ex, tokenizer, block_size=block_size, use_cot=use_cot)
    val_data = PromptMaskedSFTDataset(val_ex, tokenizer, block_size=block_size, use_cot=use_cot)
    cot_train_data = (PromptMaskedSFTDataset(train_ex, tokenizer, block_size=block_size, use_cot=True)
                      if cot_warmup_steps else None)
    if split_names:
        print(f"[{mode_str.upper()}] split-names encoding: {tokenizer.n_split:,} name+ending words split "
              f"in the training and validation sequences")

    # One recipe for both architectures, so V1 vs V2 differ only in the model.
    train_cfg = TrainConfig(
        micro_batch_size=FINETUNE_BATCH_SIZE,
        gradient_accumulation_steps=1,
        effective_batch_size=FINETUNE_BATCH_SIZE,
        learning_rate=lr,
        min_lr=lr / 10 if min_lr is None else min_lr,
        warmup_steps=max(1, max_steps // 20) if warmup_steps is None else warmup_steps,
        max_steps=max_steps,
        lr_schedule=lr_schedule,
        max_grad_norm=1.0,
        weight_decay=weight_decay,
        eval_interval=probe_interval,
        save_interval=max_steps,
        eval_batches=16,
        mixed_precision=device.startswith("cuda"),
        seed=seed,
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

    # --- Pretrained Baseline Evaluation (Zero-shot)
    # It depends only on the pretrained weights and the test set, not on the mode, so
    # the Direct and CoT runs of one model share it through out_dir.
    baseline_key = {
        "pretrained_ckpt": str(pretrained_ckpt),
        "test_texts": [ex["text"] for ex in test_ex],
        "max_new_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "split_names": split_names,
    }
    baseline_path = out_dir / "pretrained_baseline.json"
    baseline = None
    if baseline_path.exists():
        saved = json.loads(baseline_path.read_text(encoding="utf-8"))
        if saved.get("key") == baseline_key:
            baseline = saved["baseline"]
    if baseline is None:
        baseline = evaluate_reasoning(model, tokenizer, test_ex, device=device, seed=seed)
        baseline_path.write_text(
            json.dumps({"key": baseline_key, "baseline": baseline}, ensure_ascii=False), encoding="utf-8"
        )
    print(f"[{mode_str.upper()}] PRETRAINED BASELINE: {json.dumps(baseline, ensure_ascii=False)}")
    chance = chance_baselines(test_ex)
    print(f"[{mode_str.upper()}] CHANCE BASELINE: {json.dumps(chance, ensure_ascii=False)}")

    # --- Training Loop, with checkpoint selection by generated-answer accuracy.
    # Validation loss is a poor guide to whether the generated answer is right, so
    # the run is probed N_PROBES times: answers are generated for a fixed validation
    # subset and the weights with the highest strict answer accuracy are kept.
    probe_ex = val_ex[:max(N_PROBE_EXAMPLES, n_val)]
    best_path = out_dir / f"best_val_accuracy_{mode_str}.pt"
    best = {"accuracy": -1.0, "step": 0}
    if best_path.exists():  # resumed run: keep the best found before the interruption
        saved = torch.load(best_path, map_location="cpu", weights_only=False)
        best = {"accuracy": saved["accuracy"], "step": saved["step"]}
    history = []
    # With patience set, stop once `patience` probes in a row fail to beat the best,
    # counted only after onset (best >= majority relation rate + onset_margin).
    onset_threshold = chance_baselines(probe_ex)["majority_accuracy"] + onset_margin
    stopped_early_at = None
    targets = list(range(probe_interval, max_steps, probe_interval)) + [max_steps]
    direct_train_data = train_data
    # Curriculum stage and CoT warmup are set from the step each segment starts at.
    for start, target in zip([0] + targets[:-1], targets):
        in_warmup = start < cot_warmup_steps
        trainer.train_data = cot_train_data if in_warmup else direct_train_data
        if curriculum_steps:
            trainer.train_data.set_max_hops(curriculum_max_hops(start, curriculum_steps))
        trainer.train(max_steps=target, resume=True)
        if trainer.step > target:
            continue  # resumed past this probe point
        for old_ckpt in checkpoint_paths_in_dir(str(ckpt_subdir))[:-1]:
            os.remove(old_ckpt)  # only the latest full checkpoint is needed to resume
        rng_state = torch.get_rng_state()  # the probe must not disturb the training data order
        probe = evaluate_reasoning(model, tokenizer, probe_ex, device=device, seed=seed)
        torch.set_rng_state(rng_state)
        accuracy = probe[selection_metric]
        history.append({
            "step": trainer.step,
            "val_accuracy_answer_strict": probe["accuracy_answer_strict"],
            "val_accuracy_relation": probe["accuracy_relation"],
            "stage": "cot_warmup" if in_warmup else "main",
            "max_hops": curriculum_max_hops(start, curriculum_steps) if curriculum_steps else None,
        })
        if in_warmup:  # the model still answers with CoT here; not a Direct candidate
            print(f"[{mode_str.upper()}] step {trainer.step}: val {selection_metric}={accuracy:.4f} "
                  f"(CoT warmup, not selectable)")
            continue
        if accuracy > best["accuracy"]:
            best = {"accuracy": accuracy, "step": trainer.step}
            torch.save({"model_state_dict": model.state_dict(), **best}, best_path)
        print(f"[{mode_str.upper()}] step {trainer.step}: val {selection_metric}={accuracy:.4f} "
              f"(best {best['accuracy']:.4f} @ step {best['step']})")
        since_best = sum(1 for h in history if h["step"] > best["step"])
        if patience and best["accuracy"] >= onset_threshold and since_best >= patience:
            stopped_early_at = trainer.step
            print(f"[{mode_str.upper()}] early stop at step {trainer.step}: no gain in {patience} "
                  f"probes since step {best['step']}")
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"])
        model.to(device)

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
    finetuned = evaluate_reasoning(model, tokenizer, test_ex, device=device, seed=seed)
    print(f"[{mode_str.upper()}] FINETUNED: {json.dumps(finetuned, ensure_ascii=False)}")

    # Extra benchmarks (relation-balanced and depth-controlled), when the data has them.
    extra = {}
    for name in EXTRA_EVAL_FILES:
        path = data_dir / f"{name}.jsonl"
        if path.exists():
            extra[name] = evaluate_reasoning(
                model, tokenizer, load_reasoning_jsonl(path)[:n_test], device=device, seed=seed
            )
            print(f"[{mode_str.upper()}] {name}: strict={extra[name]['accuracy_answer_strict']:.4f} "
                  f"relation={extra[name]['accuracy_relation']:.4f} "
                  f"macroF1={extra[name]['macro_f1_relation']:.4f}")

    results = {
        "mode": mode_str,
        "pretrained_baseline": baseline,
        "chance_baseline": chance,
        "finetuned": finetuned,
        "extra_eval": extra,
        "delta_accuracy": round(
            finetuned["accuracy_answer_only"] - baseline["accuracy_answer_only"], 6
        ),
        "delta_accuracy_answer_only": round(
            finetuned["accuracy_answer_only"] - baseline["accuracy_answer_only"], 6
        ),
        "delta_accuracy_cot_exact": round(
            finetuned["accuracy_exact_match"] - baseline["accuracy_exact_match"], 6
        ),
        "selection": {
            "metric": f"{selection_metric} on validation (generated answers)",
            "best_step": best["step"],
            "best_val_accuracy": best["accuracy"],
            "history": history,
            "probe_interval": probe_interval,
            "patience": patience,
            "onset_threshold": round(onset_threshold, 6),
            "stopped_early_at": stopped_early_at,
        },
        "config": {
            "lr": lr,
            "min_lr": train_cfg.min_lr,
            "warmup_steps": train_cfg.warmup_steps,
            "lr_schedule": lr_schedule,
            "max_steps": max_steps,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "curriculum_steps": curriculum_steps,
            "cot_warmup_steps": cot_warmup_steps,
            "seed": seed,
            "repetition_penalty": REPETITION_PENALTY,
            "n_probe_examples": len(probe_ex),
            "block_size": block_size,
            "n_train_examples": len(train_ex),
            "n_val_examples": len(val_ex),
            "n_test_examples": len(test_ex),
            "overfit": overfit,
            "split_names": split_names,
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
    parser.add_argument("--min-lr", type=float, default=None, help="LR floor (default lr/10)")
    parser.add_argument("--warmup-steps", type=int, default=FINETUNE_WARMUP_STEPS)
    parser.add_argument("--lr-schedule", choices=["cosine", "wsd"], default=FINETUNE_LR_SCHEDULE,
                        help="cosine decay, or warmup-stable-decay (peak LR held for 80%% of steps)")
    parser.add_argument("--probe-interval", type=int, default=FINETUNE_PROBE_INTERVAL,
                        help="steps between validation probes")
    parser.add_argument("--patience", type=int, default=FINETUNE_PATIENCE,
                        help="stop after this many probes without improvement, counted after onset")
    parser.add_argument("--onset-margin", type=float, default=ONSET_MARGIN,
                        help="onset = best probe >= majority relation rate + this margin")
    parser.add_argument("--overfit", type=int, default=None,
                        help="train on this many training examples and validate on them (capacity check)")
    parser.add_argument("--split-names", action="store_true",
                        help="encode each name the same way wherever it occurs (case ending split off)")
    parser.add_argument("--dropout", type=float, default=FINETUNE_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--curriculum-steps", type=int, default=0,
                        help="1-hop queries for the first half of these steps, <=2 hops for the second half")
    parser.add_argument("--cot-warmup-steps", type=int, default=FINETUNE_COT_WARMUP_STEPS,
                        help="Direct only: train on CoT targets for these steps first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-val", type=int, default=FINETUNE_N_VAL)
    parser.add_argument("--n-test", type=int, default=FINETUNE_N_TEST)
    parser.add_argument("--mode", choices=["direct", "cot", "both"], default=FINETUNE_MODE)
    parser.add_argument("--device", default=None)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default=DEFAULT_SELECTION_METRIC,
                        help="validation metric that picks the saved checkpoint")
    args = parser.parse_args(argv)

    if args.mode in ("direct", "both"):
        print("\n--- Running Direct SFT ---")
        finetune(
            args.pretrained_ckpt, args.model_config, args.tokenizer,
            args.data_dir, args.out_dir,
            device=args.device, max_steps=args.max_steps, lr=args.lr,
            n_val=args.n_val, n_test=args.n_test, use_cot=False,
            custom_ckpt_name="finetuned_direct.pt", selection_metric=args.selection_metric,
            min_lr=args.min_lr, warmup_steps=args.warmup_steps, lr_schedule=args.lr_schedule,
            probe_interval=args.probe_interval, patience=args.patience, onset_margin=args.onset_margin,
            overfit=args.overfit,
            split_names=args.split_names,
            dropout=args.dropout, weight_decay=args.weight_decay,
            curriculum_steps=args.curriculum_steps, seed=args.seed,
            cot_warmup_steps=args.cot_warmup_steps,
        )

    if args.mode in ("cot", "both"):
        print("\n--- Running CoT SFT ---")
        finetune(
            args.pretrained_ckpt, args.model_config, args.tokenizer,
            args.data_dir, args.out_dir,
            device=args.device, max_steps=args.max_steps, lr=args.lr,
            n_val=args.n_val, n_test=args.n_test, use_cot=True,
            custom_ckpt_name="finetuned_cot.pt", selection_metric=args.selection_metric,
            min_lr=args.min_lr, warmup_steps=args.warmup_steps, lr_schedule=args.lr_schedule,
            probe_interval=args.probe_interval, patience=args.patience, onset_margin=args.onset_margin,
            overfit=args.overfit,
            split_names=args.split_names,
            dropout=args.dropout, weight_decay=args.weight_decay,
            curriculum_steps=args.curriculum_steps, seed=args.seed,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
