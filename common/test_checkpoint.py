"""Tests for common/checkpoint.py — Agent-D acceptance tests 1, 2, 6 + validation."""

import os
import threading

import pytest
import torch


def _make_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4)
    )


@pytest.fixture
def model_opt_sched():
    model = _make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    return model, optimizer, scheduler


def test_checkpoint_round_trip(tmp_path, model_opt_sched):
    """Save then load: every tensor in state_dict bit-identical, step returned."""
    from common.checkpoint import load_checkpoint, save_checkpoint

    model, optimizer, scheduler = model_opt_sched
    path = str(tmp_path / "ckpt_42.pt")
    save_checkpoint(path, model, optimizer, scheduler, step=42, config={"d_model": 8})

    fresh_model = _make_model()
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_opt, T_max=10)

    step = load_checkpoint(path, fresh_model, fresh_opt, fresh_sched)
    assert step == 42

    for (k1, t1), (k2, t2) in zip(
        model.state_dict().items(), fresh_model.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(t1, t2)


def test_atomic_checkpoint_write(tmp_path, model_opt_sched):
    """Simulate a crash mid-save: previous checkpoint stays valid (.tmp + os.replace)."""
    from common.checkpoint import load_checkpoint, save_checkpoint, validate_checkpoint

    model, optimizer, scheduler = model_opt_sched
    path = str(tmp_path / "ckpt_10.pt")
    save_checkpoint(path, model, optimizer, scheduler, step=10, config={})
    before = open(path, "rb").read()

    # Crash mid-write: monkeypatch torch.save to raise right after the .tmp file
    # has been created/partially written.
    import common.checkpoint as ckpt_mod

    real_save = ckpt_mod.torch.save
    calls = {"n": 0}

    def crashing_save(obj, f, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash mid-write")
        return real_save(obj, f, *a, **kw)

    ckpt_mod.torch.save = crashing_save
    try:
        with pytest.raises(RuntimeError):
            save_checkpoint(path, model, optimizer, scheduler, step=11, config={})
    finally:
        ckpt_mod.torch.save = real_save

    # Original checkpoint must be untouched and valid.
    assert open(path, "rb").read() == before
    assert validate_checkpoint(path)
    # No stray .tmp files left behind.
    assert [f for f in os.listdir(tmp_path) if ".tmp-" in f] == []


def test_validate_checkpoint_rejects_missing_keys(tmp_path, model_opt_sched):
    from common.checkpoint import save_checkpoint, validate_checkpoint

    model, optimizer, scheduler = model_opt_sched
    path = str(tmp_path / "ckpt_1.pt")
    save_checkpoint(path, model, optimizer, scheduler, step=1, config={"a": 1})

    payload = torch.load(path, weights_only=False)
    del payload["scheduler_state_dict"]
    bad_path = str(tmp_path / "bad.pt")
    torch.save(payload, bad_path)
    assert not validate_checkpoint(bad_path)
    assert validate_checkpoint(path)


def test_load_checkpoint_raises_on_partial(tmp_path, model_opt_sched):
    from common.checkpoint import load_checkpoint

    model, _, _ = model_opt_sched
    path = str(tmp_path / "partial.pt")
    torch.save({"model_state_dict": model.state_dict(), "step": 3}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, model)


def test_resume_equivalence(tmp_path):
    """Agent-D acceptance test 1: 20 uninterrupted steps == 10 + save + 10 resumed.

    Uses a small model + fixed seed so CPU runs are fast; equivalence is enforced by
    restoring saved RNG state.
    """
    from common.checkpoint import load_checkpoint, save_checkpoint

    def run(steps, start_ckpt=None, out_ckpt=None, seed=1234):
        torch.manual_seed(seed)
        model = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.9)
        step0 = 0
        if start_ckpt is not None:
            # Restores saved RNG state, so the batch drawn below (from the global
            # RNG) matches the interrupted run exactly — same as a TokenDataset
            # whose batch positions are drawn from the global RNG each step.
            step0 = load_checkpoint(start_ckpt, model, opt, sched)
        loss = None
        for i in range(step0, steps):
            # Batch must be drawn *inside* the loop from the global RNG so a
            # resume (with restored RNG) reproduces the original data order.
            data = torch.randn(8, 4)
            targets = torch.randn(8, 4)
            opt.zero_grad()
            loss = ((model(data) - targets) ** 2).mean()
            loss.backward()
            opt.step()
            sched.step()
        if out_ckpt is not None:
            save_checkpoint(out_ckpt, model, opt, sched, step=steps, config={})
        assert loss is not None
        return loss.item()

    ckpt = str(tmp_path / "ckpt_10.pt")
    loss_a = run(steps=20)
    run(steps=10, out_ckpt=ckpt)
    loss_b = run(steps=20, start_ckpt=ckpt)
    assert abs(loss_a - loss_b) < 1e-6
