"""Hindi training infrastructure.

Implements the locked Trainer contract of project specifications Section 3 Trainer:

  * TokenDataset over flat uint16/uint32 token arrays via numpy.memmap
    (nanoGPT-style random block_size windows — 500M tokens never enter RAM).
  * AdamW with weight_decay=0.1 on 2D+ params only (no decay on biases/LayerNorm).
  * Cosine LR schedule with linear warmup (~2% of total steps), min_lr floor.
  * Mixed precision (torch.cuda.amp autocast + GradScaler) when enabled and on CUDA.
  * Gradient clipping at 1.0 (post-unscale).
  * Resume-capable by construction: on start, load the latest valid checkpoint and
    continue from its step; RNG state is saved/restored so data order reproduces.
  * Optional wall-clock budget (--time-limit-hours): once it is spent, save
    ckpt_<step>.pt and exit cleanly so the next session resumes from there.
  * OOM fallback: halve micro_batch_size / double grad-accum (effective batch
    invariant), halt if micro-batch 1 still OOMs.
  * NaN/Inf loss: skip the optimizer step; reload from last checkpoint if it
    happens >2 steps in a row.
  * Loud warning when val loss worsens 3 consecutive eval points.
  * Checkpoint auto-staging to Kaggle working + Drive mirror (opt-in via config).

The identical file is duplicated into assamese/train/train.py — the two languages
share no code (spec Section 0.2).
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

# Make `model.` / `tokenizer.` style imports work when run as a script from <lang>/.
_LANG_ROOT = Path(__file__).resolve().parents[1]
if str(_LANG_ROOT) not in sys.path:
    sys.path.insert(0, str(_LANG_ROOT))

from common.checkpoint import (  # noqa: E402
    checkpoint_paths_in_dir,
    load_checkpoint,
    save_checkpoint,
)
from model.gpt import GPTConfig, GPTLanguageModel  # noqa: E402

log = logging.getLogger("train")


# ---------------------------------------------------------------- data loading

class TokenDataset:
    """Random-window access to a flat token-id array (numpy.memmap, nanoGPT-style).

    The file is a raw uint16 (vocab < 65536) or uint32 array. Windows are drawn
    from the *global* torch RNG, so saving/restoring RNG state in the checkpoint
    makes resumes reproduce the exact data order.
    """

    def __init__(self, path: Union[str, Path], block_size: int, dtype: Optional[str] = None):
        self.path = str(path)
        self.block_size = block_size
        if dtype is None:
            dtype = self._sniff_dtype(self.path)
        self.dtype = dtype
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        if self.data.size < block_size + 1:
            raise ValueError(
                f"corpus {self.path} too small ({self.data.size} tokens) for "
                f"block_size={block_size}"
            )

    @staticmethod
    def _sniff_dtype(path: str) -> str:
        """uint16 unless the file clearly holds larger ids."""
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        sample = mm[: min(mm.size, 1_000_000)]
        return "uint16" if sample.max() < 65000 else "uint32"

    def __len__(self) -> int:
        return int(self.data.size)

    def get_batch(
        self, batch_size: int, device: str = "cpu", generator: Optional[torch.Generator] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random (x, y) windows of shape (batch_size, block_size).

        Drawn from the global torch RNG by default (deliberately — see class
        docstring). Pass a local ``generator`` to draw from a private stream
        (used by eval so it never perturbs training RNG).
        """
        gen = generator
        if gen is not None and getattr(gen, "device", None) is not None and gen.device.type != "cpu":
            gen = None
        ix = torch.randint(
            len(self.data) - self.block_size, (batch_size,), generator=gen, device="cpu"
        )
        x = np.stack([self.data[i : i + self.block_size] for i in ix.tolist()])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix.tolist()])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        return x.to(device), y.to(device)


# ---------------------------------------------------------------- LR schedule

class WarmupCosineScheduler:
    """Linear warmup then cosine decay to min_lr (spec Section 1)."""

    def __init__(
        self,
        base_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
    ):
        assert warmup_steps >= 0 and max_steps > 0
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.t = 0  # optimizer steps completed

    def get_lr(self) -> float:
        if self.warmup_steps > 0 and self.t < self.warmup_steps:
            return self.base_lr * (self.t + 1) / self.warmup_steps
        progress = (self.t - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> None:
        self.t += 1

    def state_dict(self) -> dict:
        return {"t": self.t}

    def load_state_dict(self, state: dict) -> None:
        self.t = int(state["t"])


class WarmupStableDecayScheduler(WarmupCosineScheduler):
    """Linear warmup, constant base_lr until stable_fraction of max_steps, then linear
    decay to min_lr. Keeps the rate high while a run is still searching (finetuning
    runs that sit at chance for thousands of steps), and anneals only at the end."""

    def __init__(self, base_lr: float, min_lr: float, warmup_steps: int, max_steps: int,
                 stable_fraction: float = 0.8):
        super().__init__(base_lr, min_lr, warmup_steps, max_steps)
        assert 0.0 <= stable_fraction <= 1.0
        self.decay_start = max(warmup_steps, int(stable_fraction * max_steps))

    def get_lr(self) -> float:
        if self.warmup_steps > 0 and self.t < self.warmup_steps:
            return self.base_lr * (self.t + 1) / self.warmup_steps
        if self.t < self.decay_start:
            return self.base_lr
        progress = (self.t - self.decay_start) / max(1, self.max_steps - self.decay_start)
        progress = min(1.0, progress)
        return self.base_lr + (self.min_lr - self.base_lr) * progress


LR_SCHEDULES = {"cosine": WarmupCosineScheduler, "wsd": WarmupStableDecayScheduler}


# ---------------------------------------------------------------- config

@dataclass
class TrainConfig:
    micro_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = 64  # micro_batch_size * gradient_accumulation_steps
    learning_rate: float = 6.0e-4
    min_lr: float = 6.0e-5
    warmup_steps: int = 300
    max_steps: int = 15259
    lr_schedule: str = "cosine"     # "cosine" or "wsd" (warmup-stable-decay)
    stable_fraction: float = 0.8    # wsd only: share of max_steps held at the peak rate
    max_grad_norm: float = 1.0
    weight_decay: float = 0.1
    eval_interval: int = 250
    save_interval: int = 2500
    eval_batches: int = 20
    mixed_precision: bool = True
    seed: int = 1337
    kaggle_auto_stage: bool = False  # stage snapshots to /kaggle/working + Drive
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TrainConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------- optimizer

def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """AdamW with weight decay on 2D+ params only (no decay on biases/LayerNorm)."""
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)


# ---------------------------------------------------------------- trainer

class Trainer:
    """Resume-capable pretraining loop. train() is correct for fresh start OR resume."""

    def __init__(
        self,
        model: GPTLanguageModel,
        train_data: TokenDataset,
        val_data: Optional[TokenDataset],
        config: TrainConfig,
        checkpoint_dir: Union[str, Path],
        device: Optional[str] = None,
        resume_path: Optional[str] = None,
    ):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.checkpoint_dir = str(checkpoint_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        self.model.to(self.device)

        self.optimizer = build_optimizer(
            model, config.learning_rate, weight_decay=config.weight_decay
        )
        if config.lr_schedule not in LR_SCHEDULES:
            raise ValueError(f"lr_schedule must be one of {sorted(LR_SCHEDULES)}, got {config.lr_schedule!r}")
        schedule_kwargs = {"stable_fraction": config.stable_fraction} if config.lr_schedule == "wsd" else {}
        self.scheduler = LR_SCHEDULES[config.lr_schedule](
            base_lr=config.learning_rate,
            min_lr=config.min_lr,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            **schedule_kwargs,
        )
        self.use_amp = config.mixed_precision and self.device.startswith("cuda")
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            except (TypeError, AttributeError):
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            self.scaler = None
        # Eval draws from a private RNG stream so it never perturbs the training
        # data order (keeps resume bit-equivalence intact). It is re-seeded on every
        # evaluate() call, so all eval points score the same windows and best.pt
        # selection is not driven by sampling noise.
        self._eval_seed = (config.seed or 0) + 999_983
        self.step = 0
        self.best_val_loss = float("inf")
        self._consecutive_nan = 0
        self._val_history: list[float] = []
        self.logs: list[dict] = []

        if resume_path:
            self._resume_from(resume_path)

    # ------------------------------------------------------------------ resume

    def _resume_from(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        step = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler, restore_rng=True
        )
        self.step = step
        # The checkpoint holds the full config used to produce it — surface arch
        # mismatches loudly rather than silently training a different architecture.
        saved_cfg = torch.load(path, map_location="cpu", weights_only=False).get("config", {})
        self.best_val_loss = float(saved_cfg.get("best_val_loss", self.best_val_loss))
        saved_model = saved_cfg.get("model", {})
        cur_model = {
            "vocab_size": self.model.config.vocab_size,
            "d_model": self.model.config.d_model,
            "n_layer": self.model.config.n_layer,
            "n_head": self.model.config.n_head,
            "block_size": self.model.config.block_size,
        }
        mismatches = {k: (saved_model.get(k), v) for k, v in cur_model.items()
                      if saved_model.get(k) is not None and saved_model.get(k) != v}
        if mismatches:
            log.warning(
                "CHECKPOINT MODEL CONFIG MISMATCH vs current model: %s — continuing with "
                "the current model architecture as loaded above.",
                mismatches,
            )
        log.info("Resumed from %s at step %d", path, self.step)

    def _latest_checkpoint(self) -> Optional[str]:
        paths = checkpoint_paths_in_dir(self.checkpoint_dir)
        return paths[-1] if paths else None

    # ------------------------------------------------------------------ eval

    def _autocast(self):
        return torch.autocast(device_type="cuda", enabled=self.use_amp)

    @torch.no_grad()
    def evaluate(self, data: Optional[TokenDataset] = None) -> float:
        """Mean loss over eval_batches seeded-random windows (the same windows on
        every call); train/eval mode is saved and restored here."""
        data = data or self.val_data
        if data is None:
            return float("nan")
        was_training = self.model.training
        self.model.eval()
        generator = torch.Generator(device="cpu").manual_seed(self._eval_seed)
        total, count = 0.0, 0
        for _ in range(self.config.eval_batches):
            x, y = data.get_batch(self.config.micro_batch_size, self.device, generator=generator)
            with self._autocast():
                loss = self.model(x, targets=y)["loss"]
            total += loss.item()
            count += 1
        if was_training:
            self.model.train()
        return total / max(1, count)

    # ------------------------------------------------------------------ step

    def _forward_backward(self, micro_batch_size: int, accum: int) -> float:
        """Run `accum` micro-batches, accumulating gradients; loss is scaled by 1/accum
        so the accumulated gradient matches one big batch. Returns the mean loss
        over the micro-batches.

        Any non-finite micro-loss aborts the whole step immediately (returns NaN)
        so a NaN in an early micro-batch can't poison later gradients."""
        total = 0.0
        for _ in range(accum):
            x, y = self.train_data.get_batch(micro_batch_size, self.device)
            with self._autocast():
                loss = self.model(x, targets=y)["loss"] / accum
            if not torch.isfinite(loss):
                return float("nan")
            total += loss.item()
            if self.scaler is not None and self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        return total

    def _clip_and_step(self) -> None:
        # Apply this step's LR before stepping; otherwise step 1 runs at the
        # optimizer's initial base LR instead of the warmup LR.
        lr = self.scheduler.get_lr()
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()

    def _run_one_optimizer_step(self, step_idx: int) -> float:
        """Run one full optimizer step (grad-accum micro-batches) with OOM retry.

        Returns the mean micro-batch loss. Raises if micro-batch 1 still OOMs.
        """
        micro = self.config.micro_batch_size
        accum = self.config.gradient_accumulation_steps
        while True:
            try:
                self.model.zero_grad(set_to_none=True)
                loss_val = self._forward_backward(micro, accum)
                if loss_val is None or not math.isfinite(loss_val):
                    return loss_val  # NaN/Inf: skip the optimizer step; train() handles it
                self._clip_and_step()
                return loss_val
            except torch.cuda.OutOfMemoryError:
                if micro <= 1:
                    raise
                torch.cuda.empty_cache()
                micro //= 2
                accum *= 2
                log.warning(
                    "CUDA OOM at micro_batch_size=%d — retrying with %d (accum %d); "
                    "effective batch unchanged.",
                    micro * 2, micro, accum,
                )
                self.config.micro_batch_size = micro
                self.config.gradient_accumulation_steps = accum
                self.config.effective_batch_size = micro * accum

    # ------------------------------------------------------------------ main loop

    def train(
        self,
        max_steps: Optional[int] = None,
        resume: bool = True,
        time_limit_hours: Optional[float] = None,
    ) -> bool:
        """Run the training loop to max_steps (defaults to config.max_steps).

        Correct whether fresh or resuming: if `resume` and a valid checkpoint
        exists (or resume_path was passed to __init__), continue from its step.
        With time_limit_hours set, the loop saves ckpt_<step>.pt and stops once
        that much wall-clock time has passed. Returns True if max_steps was reached.
        """
        deadline = time.monotonic() + time_limit_hours * 3600 if time_limit_hours else None
        max_steps = max_steps or self.config.max_steps
        if self.step == 0 and resume:
            latest = self._latest_checkpoint()
            if latest is not None:
                self._resume_from(latest)
        if self.step >= max_steps:
            log.info("Already at step %d >= max_steps %d; nothing to do.", self.step, max_steps)
            return True

        self.model.train()
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        log.info(
            "Training: device=%s amp=%s micro=%d accum=%d lr=%g steps %d..%d",
            self.device, self.use_amp, self.config.micro_batch_size,
            self.config.gradient_accumulation_steps, self.config.learning_rate,
            self.step + 1, max_steps,
        )

        while self.step < max_steps:
            self.last_train_loss = None
            # NaN guard: NaN/Inf loss skips the step; >2 in a row -> reload.
            nan_skip = False
            try:
                loss_val = self._run_one_optimizer_step(self.step)
                if loss_val is None or not math.isfinite(loss_val):
                    nan_skip = True
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    "CUDA OOM persists at micro_batch_size=1 — halting. Reduce model "
                    "size or free memory; do not silently continue."
                ) from None

            if nan_skip:
                self._consecutive_nan += 1
                self.model.zero_grad(set_to_none=True)  # drop any partial gradients
                log.warning(
                    "NaN/Inf loss at step %d (streak %d) — optimizer step skipped.",
                    self.step + 1, self._consecutive_nan,
                )
                if self._consecutive_nan > 2:
                    latest = self._latest_checkpoint()
                    if latest is not None:
                        log.error("NaN streak >2 — reloading from %s", latest)
                        self._resume_from(latest)
                        self._consecutive_nan = 0
                    else:
                        raise RuntimeError(
                            "NaN/Inf loss on >2 consecutive steps and no checkpoint to "
                            "reload from — halting."
                        )
                continue

            self._consecutive_nan = 0
            self.last_train_loss = loss_val
            self.step += 1

            # Evaluate first so the periodic checkpoint records the current best_val_loss.
            if self.step % self.config.eval_interval == 0 or self.step == max_steps:
                self._evaluate_and_log(self.step)
            if self.step % self.config.save_interval == 0 or self.step == max_steps:
                self._save(self.step)
            if deadline is not None and self.step < max_steps and time.monotonic() >= deadline:
                if self.step % self.config.save_interval != 0:
                    self._save(self.step)
                log.info(
                    "Time limit of %.2f h reached at step %d/%d — rerun to resume.",
                    time_limit_hours, self.step, max_steps,
                )
                return False

        log.info("Training complete at step %d.", self.step)
        return True

    # ------------------------------------------------------------------ helpers

    def _evaluate_and_log(self, step: int) -> None:
        val_loss = self.evaluate()
        lr = self.scheduler.get_lr()
        entry = {"step": step, "val_loss": val_loss, "lr": lr, "loss": self.last_train_loss}
        self.logs.append(entry)
        print(
            f"[step {step:5d}] train_loss={self.last_train_loss:.4f} "
            f"val_loss={val_loss:.4f} lr={lr:.2e}"
        )
        # Val-loss-worsening detector: loud warning, no silent bad training.
        if val_loss == val_loss:  # not NaN
            self._val_history.append(val_loss)
            if len(self._val_history) >= 4 and all(
                self._val_history[i] > self._val_history[i - 1] for i in (-1, -2, -3)
            ):
                log.warning(
                    "WARNING: val loss increased for 3 consecutive eval points "
                    "(%s) — check LR / data; not auto-fixing.",
                    [round(v, 4) for v in self._val_history[-3:]],
                )
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.checkpoint_dir, "best.pt")
                self._save(step, path=best_path)

    def _save(self, step: int, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.checkpoint_dir, f"ckpt_{step}.pt")
        config = {
            # Every model field (incl. no_pos / arch_version / rope_theta), so the
            # checkpoint alone identifies the architecture it was trained with.
            "model": asdict(self.model.config),
            "train": self.config.to_dict(),
            "best_val_loss": self.best_val_loss,
        }
        save_checkpoint(path, self.model, self.optimizer, self.scheduler, step, config)
        if self.config.kaggle_auto_stage:
            from common.kaggle_io import commit_checkpoint_snapshot, mirror_to_drive

            try:
                commit_checkpoint_snapshot(
                    path, notes=f"step {step}" + (f" | {self.config.notes}" if self.config.notes else "")
                )
                mirror_to_drive(path, f"{os.path.basename(_LANG_ROOT)}/checkpoints")
            except Exception as exc:  # staging must never kill training
                log.warning("checkpoint auto-stage failed: %s", exc)
        return path


def train_entrypoint(argv: Optional[list[str]] = None) -> int:
    """CLI: python -m hindi.train.train --model-config configs/model_H.yaml
    --train-config configs/train_H.yaml --train-data data/train.bin
    --val-data data/val.bin --checkpoint-dir train/checkpoints [--resume-path ...]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain the language model")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume-path", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--time-limit-hours", type=float, default=None,
        help="save a checkpoint and exit cleanly after this much wall-clock time",
    )
    args = parser.parse_args(argv)

    import yaml

    with open(args.model_config, "r", encoding="utf-8") as f:
        raw_mcfg = yaml.safe_load(f) or {}

    if raw_mcfg.get("arch_version") == "v2" or "rope_theta" in raw_mcfg:
        from model.gpt_v2 import GPTConfigV2, GPTLanguageModelV2
        model_cfg = GPTConfigV2.from_yaml(args.model_config)
        model = GPTLanguageModelV2(model_cfg)
        print(f"[*] Initialized V2 Modern Transformer LM (RoPE, SwiGLU, RMSNorm, FlashAttention-2/SDPA)")
    else:
        model_cfg = GPTConfig.from_yaml(args.model_config)
        model = GPTLanguageModel(model_cfg)
        print(f"[*] Initialized Baseline V1 GPT Transformer LM")

    train_cfg = TrainConfig.from_yaml(args.train_config)
    print(f"params: {model.num_params():,} ({model.num_params()/1e6:.2f}M)")

    train_data = TokenDataset(args.train_data, model_cfg.block_size)
    val_data = TokenDataset(args.val_data, model_cfg.block_size) if args.val_data else None
    trainer = Trainer(
        model, train_data, val_data, train_cfg,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        resume_path=args.resume_path,
    )
    finished = trainer.train(
        max_steps=args.max_steps,
        resume=not args.no_resume,
        time_limit_hours=args.time_limit_hours,
    )

    # A resumed run continues the log of the sessions before it: keep their
    # entries up to the step this session started from.
    stats_path = os.path.join(args.checkpoint_dir, "train_log.json")
    logs = trainer.logs
    if os.path.exists(stats_path) and not args.no_resume:
        with open(stats_path, "r", encoding="utf-8") as f:
            previous = json.load(f)
        first_new = logs[0]["step"] if logs else trainer.step + 1
        logs = [e for e in previous if e["step"] < first_new] + logs
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    print(f"[*] {'Training complete' if finished else 'Stopped on time limit'} "
          f"at step {trainer.step}/{args.max_steps or train_cfg.max_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(train_entrypoint())
