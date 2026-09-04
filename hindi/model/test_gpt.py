"""Agent-C acceptance tests for the Hindi model (mirrored in assamese/)."""

import math

import pytest
import torch

from hindi.model.gpt import GPTConfig, GPTLanguageModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Small config keeps loop tests fast on CPU/GTX-1050; the full locked configs are
# exercised by test_param_count and test_output_shape_full_config.
SMALL = GPTConfig(
    vocab_size=128,
    d_model=64,
    n_layer=2,
    n_head=2,
    d_ff=256,
    block_size=64,
    dropout=0.0,
    tie_weights=True,
    bias=False,
)


def make_model(config: GPTConfig = SMALL, device: str = DEVICE) -> GPTLanguageModel:
    torch.manual_seed(0)
    return GPTLanguageModel(config).to(device)


# ---------------------------------------------------------------- shapes / wiring

def test_output_shape():
    model = make_model()
    model.eval()
    x = torch.randint(0, SMALL.vocab_size, (4, 16), device=DEVICE)
    out = model(x)
    assert out["logits"].shape == (4, 16, SMALL.vocab_size)
    assert out["loss"] is None
    assert out["attn_weights"] is None

    targets = torch.randint(0, SMALL.vocab_size, (4, 16), device=DEVICE)
    out = model(x, targets=targets)
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"])

    out = model(x, return_attn=True)
    assert out["attn_weights"] is not None
    assert len(out["attn_weights"]) == SMALL.n_layer
    for att in out["attn_weights"]:
        assert att.shape == (4, SMALL.n_head, 16, 16)


def test_output_shape_full_config():
    """Full locked configs (vocab 32768 / 16384) produce the right logits shape."""
    from common.config_loader import model_config_path

    for name in ("model_H.yaml", "model_L.yaml", "model_L_16k.yaml"):
        lang = "hindi" if name == "model_H.yaml" else "assamese"
        cfg = GPTConfig.from_yaml(model_config_path(lang, name))
        model = make_model(cfg, "cpu")
        x = torch.randint(0, cfg.vocab_size, (1, 8))
        logits = model(x)["logits"]
        assert logits.shape == (1, 8, cfg.vocab_size)


def test_weight_tying():
    model = make_model()
    assert (
        model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
    ), "input embedding and output projection must share the same tensor"
    # Tied pair must be counted once.
    n = sum(p.numel() for p in model.parameters())
    assert n == model.num_params()


# ---------------------------------------------------------------- param counts

@pytest.mark.parametrize("lang,name,lo,hi", [
    ("hindi", "model_H.yaml", 22_500_000, 27_500_000),
    ("assamese", "model_L.yaml", 22_500_000, 27_500_000),
    ("assamese", "model_L_16k.yaml", 22_500_000, 27_500_000),
])
def test_param_count(lang, name, lo, hi):
    from common.config_loader import model_config_path

    cfg = GPTConfig.from_yaml(model_config_path(lang, name))
    model = GPTLanguageModel(cfg)
    n = model.num_params()
    print(f"{lang}/{name}: {n:,} params ({n/1e6:.2f}M)")  # exact count, printed (spec §1)
    assert lo <= n <= hi, f"{lang}/{name} param count {n} outside [{lo}, {hi}]"


# ---------------------------------------------------------------- causality (required)

def test_causality():
    """Logits at positions <= t must be identical when token t+1 differs.

    This is the assignment's required proof the model cannot see the future.
    """
    model = make_model()
    model.eval()
    B, T, t = 2, 16, 8
    x = torch.randint(0, SMALL.vocab_size, (B, T), device=DEVICE)
    x2 = x.clone()
    x2[:, t + 1] = (x2[:, t + 1] + 1) % SMALL.vocab_size

    with torch.no_grad():
        logits1 = model(x)["logits"]
        logits2 = model(x2)["logits"]

    assert torch.allclose(
        logits1[:, : t + 1], logits2[:, : t + 1], atol=1e-5
    ), "model attended to future tokens (t+1 affected logits at <= t)"
    # And the future position itself must differ.
    assert not torch.allclose(logits1[:, t + 1], logits2[:, t + 1])


def test_attention_rows_sum_to_one_and_mask_future():
    """Post-softmax causal attention: each query row sums to 1 and future keys are 0."""
    model = make_model()
    model.eval()
    x = torch.randint(0, SMALL.vocab_size, (1, 8), device=DEVICE)
    att = model(x, return_attn=True)["attn_weights"][0][0]  # layer 0: (n_head, 8, 8)
    assert att.shape == (SMALL.n_head, 8, 8)
    rows = att.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
    # Upper triangle (future) must be exactly zero after masking.
    assert torch.all(att.tril() >= 0)
    assert torch.allclose(att.triu(1), torch.zeros_like(att.triu(1)), atol=1e-8)


# ---------------------------------------------------------------- training wiring

def test_overfit_tiny_batch():
    """A single repeated batch of ~1000 tokens must overfit to near-zero loss,
    proving forward/backward are wired correctly."""
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    B, T = 16, 64  # 1024 tokens
    x = torch.randint(0, SMALL.vocab_size, (B, T), device=DEVICE)
    targets = torch.roll(x, shifts=-1, dims=1)

    for _ in range(200):
        model.train()
        optimizer.zero_grad()
        loss = model(x, targets=targets)["loss"]
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.1, f"model failed to overfit one batch: loss={loss.item():.4f}"


# ---------------------------------------------------------------- generation

def test_generate_no_nan():
    model = make_model()
    prefix = torch.randint(0, SMALL.vocab_size, (1, 8), device=DEVICE)

    checked = {"n": 0}

    def checking_forward(self, idx, targets=None, return_attn=False):
        out = model.__class__.forward(self, idx, targets=targets, return_attn=return_attn)
        logits = out["logits"]
        assert torch.isfinite(logits).all(), f"NaN/Inf logits at generation step {checked['n']}"
        checked["n"] += 1
        return out

    import types

    model.forward = types.MethodType(checking_forward, model)
    for temp in (0.0, 0.5, 1.0, 1.5):
        out = model.generate(prefix.clone(), max_new_tokens=50, temperature=temp, top_k=None)
        assert out.shape == (1, 58)
        assert out.min() >= 0 and out.max() < SMALL.vocab_size
    assert checked["n"] >= 4 * 50

    model.forward = model.__class__.forward  # restore


def test_generate_top_k_no_nan():
    model = make_model()
    prefix = torch.randint(0, SMALL.vocab_size, (1, 8), device=DEVICE)
    out = model.generate(prefix.clone(), max_new_tokens=20, temperature=0.8, top_k=10)
    assert out.shape == (1, 28)
    assert torch.isfinite(out.float()).all()


def test_forward_backward_param_grads():
    """Optimizer touches every trainable tensor (ties still counted once)."""
    model = make_model()
    x = torch.randint(0, SMALL.vocab_size, (2, 8), device=DEVICE)
    targets = torch.randint(0, SMALL.vocab_size, (2, 8), device=DEVICE)
    loss = model(x, targets=targets)["loss"]
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    assert len(grads) == sum(1 for _ in model.named_parameters())
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())


def test_block_size_boundary():
    model = make_model()
    x = torch.randint(0, SMALL.vocab_size, (1, SMALL.block_size), device=DEVICE)
    model(x)  # exactly block_size is fine
    with pytest.raises(AssertionError):
        model(torch.randint(0, SMALL.vocab_size, (1, SMALL.block_size + 1), device=DEVICE))
