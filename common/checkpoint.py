"""Checkpoint save / load / validate shared by both languages.

A valid checkpoint dict is (see AGENT_BUILD_SPEC.md §0.3 / §3 Agent-D):

    {
        "model_state_dict":      OrderedDict,
        "optimizer_state_dict":  dict,
        "scheduler_state_dict":  dict,
        "step":                  int,       # training step this checkpoint was written at
        "config":                dict,      # full config (model + training) used to produce it
        "rng_state":             dict,      # torch/random/numpy RNG state for bit-exact resume
    }

Writing is atomic (write to ``<path>.tmp`` then ``os.replace``) so a crash mid-write
never corrupts the last-good checkpoint. Loading validates every required key and
raises a clear error instead of silently doing a partial load.

This module contains no language-specific logic; it is the single implementation of
checkpoint I/O used by both ``hindi/`` and ``assamese/``.
"""

from __future__ import annotations

import os
import random
import tempfile
from typing import Any, Optional

import numpy as np
import torch

REQUIRED_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "step",
    "config",
    "rng_state",
)


def _capture_rng_state() -> dict:
    """Snapshot the global RNG state of torch (cpu + cuda), numpy, and random."""
    state = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    """Restore a snapshot produced by ``_capture_rng_state``."""
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "random" in state:
        random.setstate(state["random"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: dict,
    rng_state: Optional[dict] = None,
) -> None:
    """Atomically write a full, resume-capable checkpoint to ``path``.

    ``scheduler`` may be any object exposing ``state_dict()``/``load_state_dict()``
    (a torch LR scheduler or a lightweight wrapper).
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": int(step),
        "config": config,
        "rng_state": rng_state if rng_state is not None else _capture_rng_state(),
    }

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    # Atomic write: same directory + fsync + os.replace means readers never see a
    # torn file, and the previous checkpoint survives any crash mid-write.
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Never leave a stale .tmp file behind on failure.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    restore_rng: bool = True,
) -> int:
    """Load a checkpoint in place and return the step to resume from.

    Raises ``ValueError`` if any required key is missing (never a silent partial
    load). Optimizer/scheduler state is applied only when the corresponding object
    is provided — a caller that wants optimizer state must pass it.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ValueError(
            f"Checkpoint {path} is missing required keys {missing}; refusing partial load."
        )
    if not isinstance(ckpt["step"], int) or ckpt["step"] < 0:
        raise ValueError(f"Checkpoint {path} has invalid step {ckpt['step']!r}.")

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if restore_rng:
        _restore_rng_state(ckpt["rng_state"])
    return ckpt["step"]


def validate_checkpoint(path: str) -> bool:
    """Validate a checkpoint on disk without loading tensors onto the GPU.

    Checks that every required key exists and that ``step`` is a non-negative int.
    Used by the resume-drill test and before committing a Kaggle Dataset version.
    """
    if not os.path.exists(path):
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if not all(k in ckpt for k in REQUIRED_KEYS):
        return False
    if not isinstance(ckpt.get("step"), int) or ckpt["step"] < 0:
        return False
    if not isinstance(ckpt.get("config"), dict):
        return False
    return True


def checkpoint_paths_in_dir(directory: str) -> list[str]:
    """Return existing ``ckpt_<step>.pt`` paths in ``directory`` sorted by step."""
    if not os.path.isdir(directory):
        return []
    paths = []
    for name in os.listdir(directory):
        if not name.startswith("ckpt_") or not name.endswith(".pt"):
            continue
        step_str = name[len("ckpt_") : -len(".pt")]
        if not step_str.isdigit():
            continue
        paths.append((int(step_str), os.path.join(directory, name)))
    paths.sort()
    return [p for _, p in paths]
