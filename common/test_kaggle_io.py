"""Tests for common/kaggle_io.py — checkpoint discovery, snapshot staging, Drive mirror."""

import os

import pytest


@pytest.fixture
def kaggle_mod(monkeypatch, tmp_path):
    import common.kaggle_io as k

    monkeypatch.setattr(k, "KAGGLE_WORKING", str(tmp_path / "working"))
    monkeypatch.setattr(k, "KAGGLE_INPUT", str(tmp_path / "input"))
    return k, tmp_path


def _write_valid_checkpoint(path, step):
    import torch
    from common.checkpoint import save_checkpoint

    os.makedirs(os.path.dirname(path), exist_ok=True)
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    save_checkpoint(path, model, opt, sched, step=step, config={"d_model": 2})


def test_find_latest_checkpoint_picks_highest_valid(tmp_path, kaggle_mod):
    k, _ = kaggle_mod
    input_dir = str(tmp_path / "dataset_input" / "checkpoints")
    _write_valid_checkpoint(os.path.join(input_dir, "ckpt_100.pt"), step=100)
    _write_valid_checkpoint(os.path.join(input_dir, "ckpt_50.pt"), step=50)
    # Invalid file (not a real checkpoint) with a high step must be skipped.
    os.makedirs(os.path.join(input_dir, "nested"), exist_ok=True)
    with open(os.path.join(input_dir, "nested", "ckpt_999.pt"), "w") as f:
        f.write("not a checkpoint")

    found = k.find_latest_checkpoint(input_dir)
    assert found is not None
    assert os.path.basename(found) == "ckpt_100.pt"


def test_find_latest_checkpoint_none_when_empty(tmp_path, kaggle_mod):
    k, _ = kaggle_mod
    assert k.find_latest_checkpoint(str(tmp_path / "empty")) is None


def test_commit_checkpoint_snapshot_writes_manifest(tmp_path, kaggle_mod):
    k, _ = kaggle_mod
    os.makedirs(k.KAGGLE_WORKING, exist_ok=True)  # Kaggle always has /kaggle/working
    src = str(tmp_path / "ckpt_7.pt")
    _write_valid_checkpoint(src, step=7)
    dest = k.commit_checkpoint_snapshot(src, notes="test snapshot")

    assert dest.startswith(k.KAGGLE_WORKING)
    assert os.path.exists(dest)
    manifest = os.path.join(k.KAGGLE_WORKING, k.MANIFEST_NAME)
    assert os.path.exists(manifest)
    with open(manifest, encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 1
    import json

    entry = json.loads(lines[0])
    assert entry["notes"] == "test snapshot"
    assert entry["file"] == dest


def test_commit_checkpoint_snapshot_missing_file_raises(tmp_path, kaggle_mod):
    k, _ = kaggle_mod
    with pytest.raises(FileNotFoundError):
        k.commit_checkpoint_snapshot(str(tmp_path / "nope.pt"), notes="x")


# ---------------------------------------------------------------- cli_commit

def test_cli_commit_versions_and_records_manifest(monkeypatch, tmp_path):
    import json

    import common.kaggle_io as k

    payload = tmp_path / "payload"
    payload.mkdir()
    calls = {"version": 0}

    def fake_datasets_version(folder, message, **kw):
        calls["version"] += 1
        return None

    # cli_commit lazily imports from common.kaggle_cli, so patch there.
    monkeypatch.setattr("common.kaggle_cli.cli_available", lambda: True)
    monkeypatch.setattr("common.kaggle_cli.has_credentials", lambda: True)
    monkeypatch.setattr("common.kaggle_cli.datasets_version", fake_datasets_version)

    entry = k.cli_commit(
        str(payload), "me", "ckpts", "Hindi checkpoints",
        message="v1 from local", create_if_missing=False,
    )
    assert calls["version"] == 1
    assert entry["kind"] == "kaggle_commit"
    assert entry["dataset"] == "me/ckpts"
    assert entry["action"] == "versioned"

    # Metadata written into the payload (CLI needs datasets-metadata.json).
    meta = json.loads((payload / "datasets-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "me/ckpts"
    # Commit recorded in the manifest for the paper trail.
    lines = (payload / k.MANIFEST_NAME).read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["kind"] == "kaggle_commit"


def test_cli_commit_create_if_missing(monkeypatch, tmp_path):
    import common.kaggle_io as k

    payload = tmp_path / "payload"
    payload.mkdir()
    monkeypatch.setattr("common.kaggle_cli.cli_available", lambda: True)
    monkeypatch.setattr("common.kaggle_cli.has_credentials", lambda: True)
    monkeypatch.setattr(
        "common.kaggle_cli.ensure_dataset",
        lambda *a, **kw: "created",
    )
    entry = k.cli_commit(
        str(payload), "me", "ckpts", "Checkpoints", create_if_missing=True
    )
    assert entry["action"] == "created"


def test_cli_commit_requires_credentials(monkeypatch, tmp_path):
    import common.kaggle_io as k
    from common.kaggle_cli import KaggleCLIError

    payload = tmp_path / "payload"
    payload.mkdir()
    monkeypatch.setattr("common.kaggle_cli.cli_available", lambda: False)
    with pytest.raises(KaggleCLIError, match="CLI not available"):
        k.cli_commit(str(payload), "me", "ckpts", "Checkpoints")

    monkeypatch.setattr("common.kaggle_cli.cli_available", lambda: True)
    monkeypatch.setattr("common.kaggle_cli.has_credentials", lambda: False)
    with pytest.raises(KaggleCLIError, match="no Kaggle credentials"):
        k.cli_commit(str(payload), "me", "ckpts", "Checkpoints")


def test_mirror_to_drive(monkeypatch, tmp_path):
    import common.kaggle_io as k

    mount = str(tmp_path / "drive")
    os.makedirs(mount, exist_ok=True)  # a mounted Drive root exists in production
    monkeypatch.setenv("DRIVE_MOUNT", mount)
    src = str(tmp_path / "ckpt_3.pt")
    _write_valid_checkpoint(src, step=3)

    dest = k.mirror_to_drive(src, "hindi/checkpoints")
    assert dest is not None
    assert dest == os.path.join(mount, "hindi/checkpoints", "ckpt_3.pt")
    assert os.path.exists(dest)


def test_mirror_to_drive_noop_without_mount(monkeypatch, tmp_path):
    import common.kaggle_io as k

    monkeypatch.delenv("DRIVE_MOUNT", raising=False)
    src = str(tmp_path / "ckpt_3.pt")
    _write_valid_checkpoint(src, step=3)
    assert k.mirror_to_drive(src, "hindi/checkpoints") is None
