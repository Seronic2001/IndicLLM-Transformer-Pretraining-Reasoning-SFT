"""Agent-D acceptance tests for the Hindi Trainer (mirrored in assamese/)."""

import numpy as np
import pytest
import torch

from hindi.model.gpt import GPTConfig, GPTLanguageModel
from hindi.train.train import TokenDataset, TrainConfig, Trainer, build_optimizer

DEVICE = "cpu"  # deterministic, fast for small-model loop tests

MODEL_CFG = GPTConfig(
    vocab_size=64,
    d_model=32,
    n_layer=2,
    n_head=2,
    d_ff=64,
    block_size=16,
    dropout=0.0,
    tie_weights=True,
    bias=False,
)


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    train = rng.integers(0, MODEL_CFG.vocab_size, size=50_000, dtype=np.uint16)
    val = rng.integers(0, MODEL_CFG.vocab_size, size=20_000, dtype=np.uint16)
    train.tofile(tmp_path / "train.bin")
    val.tofile(tmp_path / "val.bin")
    return tmp_path


def make_trainer(tmp_path, resume_path=None, ckpt_subdir="ckpts", **cfg_overrides):
    torch.manual_seed(0)
    model = GPTLanguageModel(MODEL_CFG)
    defaults = dict(
        micro_batch_size=8,
        gradient_accumulation_steps=2,
        eval_interval=10_000,  # avoid eval noise in loop tests
        save_interval=10_000,
        mixed_precision=False,
        seed=1234,
    )
    defaults.update(cfg_overrides)
    cfg = TrainConfig(**defaults)
    train_data = TokenDataset(tmp_path / "train.bin", MODEL_CFG.block_size)
    val_data = TokenDataset(tmp_path / "val.bin", MODEL_CFG.block_size)
    trainer = Trainer(
        model, train_data, val_data, cfg,
        checkpoint_dir=str(tmp_path / ckpt_subdir),
        device=DEVICE,
        resume_path=resume_path,
    )
    return trainer


# ---------------------------------------------------------------- data loading

def test_token_dataset_shapes_and_shift(data_dir):
    ds = TokenDataset(data_dir / "train.bin", block_size=16)
    assert len(ds) == 50_000
    x, y = ds.get_batch(4, "cpu")
    assert x.shape == (4, 16) and y.shape == (4, 16)
    # y is x shifted one position right (next-token targets).
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert x.dtype == torch.int64


def test_token_dataset_dtype_sniff(data_dir):
    assert TokenDataset._sniff_dtype(str(data_dir / "train.bin")) == "uint16"
    big = (np.random.default_rng(1).integers(0, 100_000, size=1000)).astype(np.uint32)
    big.tofile(data_dir / "big.bin")
    assert TokenDataset._sniff_dtype(str(data_dir / "big.bin")) == "uint32"


def test_optimizer_weight_decay_groups():
    model = GPTLanguageModel(MODEL_CFG)
    opt = build_optimizer(model, learning_rate=1e-3, weight_decay=0.1)
    assert len(opt.param_groups) == 2
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    # Every 1D param (LayerNorm weight/bias) is in the no-decay group (by identity).
    no_decay_ids = {id(p) for p in no_decay["params"]}
    decay_ids = {id(p) for p in decay["params"]}
    for p in model.parameters():
        if p.dim() < 2:
            assert id(p) in no_decay_ids, f"{p.shape} should not decay"
        else:
            assert id(p) in decay_ids, f"{p.shape} should decay"


# ---------------------------------------------------------------- loop semantics

def test_gradient_accumulation_step(data_dir):
    """Optimizer is stepped once per gradient_accumulation_steps micro-batches."""
    trainer = make_trainer(data_dir)
    calls = {"opt": 0, "batch": 0}
    real_step = trainer.optimizer.step

    def counting_step():
        calls["opt"] += 1
        real_step()

    real_get_batch = trainer.train_data.get_batch

    def counting_get_batch(*a, **kw):
        calls["batch"] += 1
        return real_get_batch(*a, **kw)

    trainer.optimizer.step = counting_step
    trainer.train_data.get_batch = counting_get_batch
    trainer.train(max_steps=3)
    assert calls["opt"] == 3  # one optimizer step per global step
    assert calls["batch"] == 3 * trainer.config.gradient_accumulation_steps


def test_oom_fallback_reduces_batch(data_dir):
    trainer = make_trainer(data_dir)
    orig_micro, orig_accum = (
        trainer.config.micro_batch_size,
        trainer.config.gradient_accumulation_steps,
    )
    state = {"raised": False}

    def oom_once(micro, accum):
        if not state["raised"]:
            state["raised"] = True
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return 0.5

    trainer._forward_backward = oom_once
    trainer.train(max_steps=1)
    assert state["raised"]
    assert trainer.config.micro_batch_size == orig_micro // 2
    assert trainer.config.gradient_accumulation_steps == orig_accum * 2
    assert trainer.config.effective_batch_size == orig_micro * orig_accum  # invariant
    assert trainer.step == 1


def test_oom_at_batch_one_halts(data_dir):
    trainer = make_trainer(data_dir, micro_batch_size=1, gradient_accumulation_steps=1)

    def always_oom(micro, accum):
        raise torch.cuda.OutOfMemoryError("persistent")

    trainer._forward_backward = always_oom
    with pytest.raises(RuntimeError, match="persists"):
        trainer.train(max_steps=1)


def test_nan_loss_skips_step(data_dir):
    trainer = make_trainer(data_dir)
    calls = {"opt": 0}
    real_step = trainer.optimizer.step

    def counting_step():
        calls["opt"] += 1
        real_step()

    state = {"nan": True}

    def forward_that_nans_first_step(idx, targets=None, return_attn=False):
        out = type(trainer.model).forward(trainer.model, idx, targets=targets, return_attn=return_attn)
        if state["nan"] and out["loss"] is not None:
            state["nan"] = False
            out["loss"] = out["loss"] * float("nan")
        return out

    trainer.optimizer.step = counting_step
    # Plain function attribute -> no self auto-binding (called as model(x, targets=y)).
    trainer.model.forward = forward_that_nans_first_step
    trainer.train(max_steps=3)
    # The NaN step was skipped (no optimizer call), the other 3 steps ran normally.
    assert calls["opt"] == 3
    assert trainer.step == 3


def test_resume_equivalence(data_dir):
    """20 uninterrupted steps == 10 + save + resume for 10 more (bit-close)."""
    a = make_trainer(data_dir, ckpt_subdir="ck_a")
    a.train(max_steps=20)
    loss_a = a.last_train_loss

    b1 = make_trainer(data_dir, ckpt_subdir="ck_b")
    b1.train(max_steps=10)  # saves ckpt_10.pt at end of run

    b2 = make_trainer(
        data_dir, ckpt_subdir="ck_b",
        resume_path=str(data_dir / "ck_b" / "ckpt_10.pt"),
    )
    b2.train(max_steps=20)
    loss_b = b2.last_train_loss

    assert b2.step == 20
    assert abs(loss_a - loss_b) < 1e-6, f"{loss_a} vs {loss_b}"


def test_checkpoint_round_trip_through_trainer(data_dir):
    trainer = make_trainer(data_dir)
    trainer.train(max_steps=2)
    path = str(data_dir / "ckpts" / "ckpt_2.pt")
    assert trainer._latest_checkpoint() == path

    fresh = make_trainer(data_dir, resume_path=path)
    for (k1, t1), (k2, t2) in zip(
        trainer.model.state_dict().items(), fresh.model.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(t1, t2)


def test_trainer_creates_best_checkpoint(data_dir):
    trainer = make_trainer(data_dir, eval_interval=1, eval_batches=2)
    trainer.train(max_steps=3)
    import os

    assert os.path.exists(os.path.join(str(data_dir / "ckpts"), "best.pt"))
    assert len(trainer.logs) > 0
    entry = trainer.logs[-1]
    assert "val_loss" in entry and "lr" in entry and "step" in entry


def test_scheduler_warmup_then_cosine():
    from hindi.train.train import WarmupCosineScheduler

    s = WarmupCosineScheduler(base_lr=6e-4, min_lr=6e-5, warmup_steps=40, max_steps=1907)
    s.t = 0
    assert s.get_lr() == pytest.approx(6e-4 / 40)
    s.t = 39
    assert s.get_lr() == pytest.approx(6e-4)
    s.t = 1906
    assert s.get_lr() == pytest.approx(6e-5, rel=1e-3)
    # Resume round-trip.
    sd = s.state_dict()
    s2 = WarmupCosineScheduler(6e-4, 6e-5, 40, 1907)
    s2.load_state_dict(sd)
    assert s2.t == s.t
