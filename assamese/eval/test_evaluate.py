"""Agent-E tests for the Hindi evaluation suite (mirrored in assamese/)."""

import json
import math
import random

import numpy as np
import pytest
import torch

from assamese.eval.evaluate import (
    EvalSpec,
    compute_generation_metrics,
    compute_ppl_bpb,
    extract_prefixes,
    generate_continuations,
    run_evaluation,
)
from assamese.model.gpt import GPTConfig, GPTLanguageModel
from assamese.tokenizer.train_tokenizer import train_tokenizer
from assamese.tokenizer.tokenizer import Tokenizer

WORDS = [
    "অসম", "আছে", "আৰু", "নহয়", "মানুহ", "কথা", "কয়", "ৰাজধানী",
    "গুৱাহাটী", "চহৰ", "গাঁও", "পানী", "আকাশ", "সূৰ্য", "জোন", "তৰা",
    "লৰা", "ছোৱালী", "বিদ্যালয়", "কিতাপ", "পঢ়ে", "লিখে", "ঘৰ", "পৰিয়াল",
    "ভাই", "ভনী", "বন্ধু", "ভাত", "দালি", "মাছ", "গাখীৰ", "চাহ", "বজাৰ",
    "পথ", "গাড়ী", "বাছ", "ৰেল", "বিমান", "সাগৰ", "নদী", "পৰ্বত", "অৰণ্য",
    "গছ", "ফুল", "পাত", "নীলা", "ৰঙা", "হালধীয়া", "বগা", "পুৱা", "সন্ধ্যা",
]

SMALL_MODEL = GPTConfig(
    vocab_size=512, d_model=32, n_layer=2, n_head=2, d_ff=64,
    block_size=32, dropout=0.0, tie_weights=True, bias=False,
)


@pytest.fixture(scope="module")
def eval_fixture(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    rng = random.Random(42)
    lines = []
    for _ in range(400):
        k = rng.randint(4, 12)
        lines.append(" ".join(rng.choice(WORDS) for _ in range(k)))

    corpus = tmp / "corpus.txt"
    corpus.write_text("\n".join(lines), encoding="utf-8")

    stats = train_tokenizer(
        corpus, str(tmp / "tiny"), vocab_size=512,
        config={"model_type": "bpe", "byte_fallback": True,
                "character_coverage": 0.9995,
                "pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3},
        seed=1,
    )
    tok = Tokenizer(stats["model_path"])

    # Encode everything into test.bin (uint16).
    ids = []
    for ln in lines:
        ids.extend(tok.encode(ln))
    np.array(ids, dtype=np.uint16).tofile(tmp / "test.bin")

    torch.manual_seed(0)
    model = GPTLanguageModel(SMALL_MODEL)

    spec = EvalSpec(
        test_path=str(tmp / "test.bin"),
        tokenizer=tok,
        n_prompts=8,
        prefix_len=8,
        gen_len=16,
        device="cpu",
        seed=7,
    )
    return {"tmp": tmp, "tok": tok, "model": model, "spec": spec}


def test_extract_prefixes_deterministic_and_shaped(eval_fixture):
    spec = eval_fixture["spec"]
    p1 = extract_prefixes(spec)
    p2 = extract_prefixes(spec)
    assert p1 == p2
    assert len(p1) == 8
    for prefix, target in p1:
        assert len(prefix) == 8
        assert len(target) == 16


def test_greedy_generation_deterministic(eval_fixture):
    spec = eval_fixture["spec"]
    model = eval_fixture["model"]
    pairs = extract_prefixes(spec)
    g1 = generate_continuations(model, spec, pairs)
    g2 = generate_continuations(model, spec, pairs)
    assert g1["0.0"] == g2["0.0"]
    assert len(g1["0.0"]) == 8
    # All decoded texts decode cleanly (byte fallback -> no <unk>).
    assert all("\u2581" not in t for t in g1["0.0"])


def test_generation_metrics_schema(eval_fixture):
    spec = eval_fixture["spec"]
    pairs = extract_prefixes(spec)
    generations = generate_continuations(eval_fixture["model"], spec, pairs)
    references = [spec.tokenizer.decode(t) for _, t in pairs]
    metrics = compute_generation_metrics(references, generations)
    for label in ("0.0", "0.5", "1.0", "1.5"):
        m = metrics[label]
        for key in ("bleu", "chrf", "rouge_l", "repetition_rate_3",
                    "distinct_1", "distinct_2", "distinct_4"):
            assert math.isfinite(m[key]), f"{label}/{key} not finite"
        assert 0.0 <= m["repetition_rate_3"] <= 1.0


def test_ppl_bpb_consistency(eval_fixture):
    spec = eval_fixture["spec"]
    table = compute_ppl_bpb(eval_fixture["model"], spec)
    assert table["perplexity"] == pytest.approx(math.exp(table["loss"]), rel=1e-9)
    assert table["bits_per_byte"] > 0
    assert table["tokens_evaluated"] == 8 * (8 + 16 - 1)
    assert table["windows"] == 8


def test_run_evaluation_writes_deliverables(eval_fixture):
    spec = eval_fixture["spec"]
    results = run_evaluation(eval_fixture["model"], spec, str(eval_fixture["tmp"] / "out"))

    out = eval_fixture["tmp"] / "out"
    assert (out / "ppl_bpb_table.json").exists()
    assert (out / "generation_metrics.json").exists()
    assert (out / "generated_samples.jsonl").exists()

    gen_metrics = json.loads((out / "generation_metrics.json").read_text(encoding="utf-8"))
    assert set(gen_metrics) == {"0.0", "0.5", "1.0", "1.5"}

    samples = [
        json.loads(ln) for ln in
        (out / "generated_samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(samples) == 8
    for s in samples:
        assert "prefix" in s and "reference" in s
        for label in ("0.0", "0.5", "1.0", "1.5"):
            assert f"generated_temp_{label}" in s
    assert "generation_metrics" in results and "ppl_bpb" in results
