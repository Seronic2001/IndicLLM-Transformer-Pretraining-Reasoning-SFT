"""Agent-F acceptance tests for the attention analysis toolkit (mirrored in assamese/)."""

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from assamese.eval.attention_analysis import (
    analyze_attention,
    attention_entropy,
    mean_attention_distance,
    plot_attention_heatmap,
)
from assamese.model.gpt import GPTConfig, GPTLanguageModel
from assamese.tokenizer.train_tokenizer import train_tokenizer
from assamese.tokenizer.tokenizer import Tokenizer

WORDS = [
    "অসম", "আছে", "আৰু", "নহয়", "মানুহ", "কথা", "কয়", "ৰাজধানী",
    "গুৱাহাটী", "চহৰ", "গাঁও", "পানী", "আকাশ", "সূৰ্য", "জোন", "তৰা",
    "লৰা", "ছোৱালী", "বিদ্যালয়", "কিতাপ", "পঢ়ে", "লিখে", "ঘৰ", "পৰিয়াল",
]

SMALL_MODEL = GPTConfig(
    vocab_size=512, d_model=32, n_layer=2, n_head=2, d_ff=64,
    block_size=32, dropout=0.0, tie_weights=True, bias=False,
)


def row_stochastic(n_head, T, seed):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(n_head, T, T, generator=g)
    return m / m.sum(dim=-1, keepdim=True)


def test_entropy_range():
    T = 8
    att = row_stochastic(4, T, seed=1)
    ent = attention_entropy(att)
    assert ent.shape == (4,)
    assert torch.all(ent >= 0)
    assert torch.all(ent <= math.log(T) + 1e-6)


def test_uniform_attention_max_entropy():
    T = 16
    att = torch.full((3, T, T), 1.0 / T)
    ent = attention_entropy(att)
    assert torch.allclose(ent, torch.full((3,), math.log(T)), atol=1e-5)


def test_attention_entropy_zero_on_deterministic():
    """A one-hot attention matrix has entropy 0 (the formula sanity check)."""
    att = torch.zeros(2, 4, 4)
    att[:, :, 0] = 1.0  # all mass on the first key
    assert torch.allclose(attention_entropy(att), torch.zeros(2), atol=1e-6)


def test_mean_attention_distance_range():
    T = 8
    att = row_stochastic(3, T, seed=2)
    d = mean_attention_distance(att)
    assert d.shape == (3,)
    assert torch.all(d >= 0) and torch.all(d <= T - 1 + 1e-6)
    # Uniform attention over a causal context: distance is the mean |i-j|.
    causal = torch.tril(torch.ones(T, T))
    causal = causal / causal.sum(dim=-1, keepdim=True)
    expected = (
        torch.arange(T, dtype=torch.float)[None, :] - torch.arange(T, dtype=torch.float)[:, None]
    ).abs()
    uniform_dist = (causal * expected).sum(dim=-1).mean()
    d2 = mean_attention_distance(causal.unsqueeze(0))
    assert d2[0] == pytest.approx(uniform_dist.item())


def test_causal_attn_weights_sum_to_one():
    torch.manual_seed(0)
    model = GPTLanguageModel(SMALL_MODEL).eval()
    x = torch.randint(0, SMALL_MODEL.vocab_size, (1, 8))
    attn = model(x, return_attn=True)["attn_weights"]
    assert len(attn) == SMALL_MODEL.n_layer
    for att in attn:
        assert att.shape == (1, SMALL_MODEL.n_head, 8, 8)
        rows = att[0].sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
        # Future keys are exactly zero (causal).
        assert torch.allclose(att[0].triu(1), torch.zeros_like(att[0].triu(1)), atol=1e-8)


def test_heatmap_has_labels(tmp_path):
    att = row_stochastic(1, 6, seed=3)[0]
    tokens = ["\u2581অসম", "\u2581আছে", "\u2581আৰু", "\u2581নহয়", "\u2581কথা", "\u2581কয়"]
    save = str(tmp_path / "heat.png")
    ax = plot_attention_heatmap(
        att, tokens, layer=0, head=0, title="Layer 0 Head 0", save_path=save
    )
    assert ax.get_title() != ""
    assert ax.get_xlabel() == "Key position"
    assert ax.get_ylabel() == "Query position"
    assert (tmp_path / "heat.png").exists()
    # Display tick labels have the SentencePiece marker stripped.
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "অসম" in labels and "\u2581" not in labels


@pytest.fixture(scope="module")
def attention_fixture(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("attn")
    rng = random.Random(11)
    lines = []
    for _ in range(200):
        k = rng.randint(4, 10)
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
    torch.manual_seed(0)
    model = GPTLanguageModel(SMALL_MODEL)
    return {
        "tmp": tmp,
        "tok": Tokenizer(stats["model_path"]),
        "model": model,
    }


def test_analyze_attention_runs(attention_fixture):
    texts = [
        "অসমৰ ৰাজধানী গুৱাহাটী হয়",
        "লৰা বিদ্যালয়ত কিতাপ পঢ়ে",
    ]
    results = analyze_attention(
        attention_fixture["model"],
        attention_fixture["tok"],
        texts,
        str(attention_fixture["tmp"] / "attn_out"),
        layers=[0],
        heads=[0, 1],
        device="cpu",
    )
    out = attention_fixture["tmp"] / "attn_out"
    summary = json.loads((out / "attention_summary.json").read_text(encoding="utf-8"))
    assert summary["n_examples"] == 2
    assert len(summary["rows"]) == 2 * 2  # 2 examples x 2 heads
    for row in summary["rows"]:
        assert 0.0 <= row["entropy"] <= math.log(row["heatmap"].count("_") and 32)
        assert 0.0 <= row["mean_distance"] <= 31
        assert (out / Path(row["heatmap"]).name).exists()
    assert (out / "ex0_layer0_head0.png").exists()
    assert (out / "ex1_layer0_head1.png").exists()
