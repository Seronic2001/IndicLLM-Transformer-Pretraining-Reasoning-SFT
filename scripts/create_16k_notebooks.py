"""Generate standalone Jupyter Notebooks for Assamese and Hindi 16K Pretraining."""
import json
from pathlib import Path

def make_cell(cell_type: str, source: str) -> dict:
    lines = [line + "\n" for line in source.split("\n")]
    if lines and lines[-1] == "\n":
        lines[-1] = ""
    # Strip the trailing newline from the very last line if empty
    if lines and lines[-1] == "":
        lines.pop()
    elif lines:
        lines[-1] = lines[-1].rstrip("\n")

    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
    }

def generate_notebook(lang: str, output_path: Path):
    is_hi = lang == "hindi"
    lang_name = "Hindi" if is_hi else "Assamese"
    lang_script = "Devanagari script" if is_hi else "Eastern Nagari script"
    artifact_name = "lma-hindi-artifacts" if is_hi else "lma-assamese-artifact"
    model_name = "hindi" if is_hi else "assamese"
    other_name = "assamese" if is_hi else "hindi"

    cells = []

    # Markdown Intro
    intro_md = f"""# LMA Phase 2: {lang_name} 16K Transformer Pretraining (Baseline V1)

This notebook executes the complete **16K Vocabulary Baseline Transformer Language Model** pretraining for **{lang_name}** on Kaggle GPU.

### Architecture Highlights (16K Optimal Configuration):
- **Model Type**: Decoder-Only GPT (Pre-LN, Learned Absolute Positional Embeddings, GELU)
- **Vocabulary Size**: 16,384 tokens (SentencePiece BPE)
- **Depth / Width**: 8 Layers, 6 Heads ($d_{{\\text{{model}}}}=384, d_{{\\text{{ff}}}}=2240, d_k=64$)
- **Total Parameters**: **24,982,272 (~24.98M)** (within 0.07% of 25.0M target, strictly inside ACL Rubric 22.5M – 27.5M)
- **Training Strategy**: Mixed Precision (fp16 AMP), AdamW, Cosine LR (6.0e-4 -> 6.0e-5), effective batch 512 seqs (262,144 tokens/step)
- **Token Budget**: 500,000,000 tokens (1,907 optimizer steps)
- **Input Dataset**: Public `{artifact_name}` (`train.bin`, `val.bin`, `test.bin`)
"""
    cells.append(make_cell("markdown", intro_md))

    # Cell 1: Environment & GPU Verification
    c1 = """import os
import sys
import subprocess
import shutil
import json
import time
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Device Name : {device_name}")
    print(f"Compute Cap     : sm_{cap[0]}{cap[1]}")
    print(f"Total GPU VRAM  : {vram_gb:.2f} GB")
    torch.backends.cudnn.benchmark = True

    # P100 legacy architecture compatibility check (only inside Kaggle Cloud)
    if cap < (7, 0) and Path("/kaggle").exists():
        print("[!] Legacy GPU architecture detected (sm_60 < sm_70).")
        print("[*] Installing sm_60 compatible PyTorch build (torch==2.4.1+cu121)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q",
            "torch==2.4.1+cu121", "torchvision==0.19.1+cu121", "torchaudio==2.4.1+cu121",
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        print("[+] Installed PyTorch 2.4.1+cu121 with native sm_60 Pascal acceleration!")
    elif cap < (7, 0):
        print(f"[*] Local GPU detected: {device_name} (sm_{cap[0]}{cap[1]}). Using current PyTorch environment.")
else:
    print("[!] WARNING: Running on CPU. GPU is strongly recommended.")
print("=" * 70)
"""
    cells.append(make_cell("code", c1))

    # Cell 2: Dataset Discovery & Mounting
    c2 = f"""# Locate and Mount Pretraining Token Binaries from /kaggle/input
data_dir = Path("/kaggle/working/data")
data_dir.mkdir(parents=True, exist_ok=True)

print("[*] Scanning /kaggle/input for {lang_name} artifacts ({artifact_name})...")
found_artifacts = {{}}
for root, dirs, files in os.walk("/kaggle/input"):
    r_str = str(root).replace("\\\\", "/")
    if "{other_name}" in r_str.lower():
        continue
    for f in files:
        f_clean = f
        if f_clean.startswith("data__"): f_clean = f_clean[6:]
        if f_clean.startswith("clean__"): f_clean = f_clean[7:]
        if f_clean in ("train.bin", "val.bin", "test.bin", "{model_name}.model", "{model_name}.vocab", "dataset_stats.json"):
            full_p = Path(root) / f
            if f_clean not in found_artifacts or full_p.stat().st_size > found_artifacts[f_clean].stat().st_size:
                found_artifacts[f_clean] = full_p
        if f in ("candidate_bpe_16384.model", "candidate_16384.model") and "{model_name}.model" not in found_artifacts:
            found_artifacts["{model_name}.model"] = Path(root) / f
        if f in ("candidate_bpe_16384.vocab", "candidate_16384.vocab") and "{model_name}.vocab" not in found_artifacts:
            found_artifacts["{model_name}.vocab"] = Path(root) / f

for fname, fpath in sorted(found_artifacts.items()):
    dest = data_dir / fname
    shutil.copy2(fpath, dest)
    size_mb = fpath.stat().st_size / (1024 * 1024)
    print(f"  + Mounted {{fname:<24}} ({{size_mb:.2f}} MB) from {{fpath}}")

train_bin = data_dir / "train.bin"
val_bin = data_dir / "val.bin"
test_bin = data_dir / "test.bin"
assert train_bin.exists(), f"Critical error: train.bin not found! Please attach '{artifact_name}' to the notebook."
print(f"\\n[+] Successfully mounted train.bin ({{train_bin.stat().st_size / (1024*1024):.2f}} MB)")
"""
    cells.append(make_cell("code", c2))

    # Cell 3: Self-Contained Model Definition
    c3 = f"""# Hand-written Baseline Transformer Language Model (Baseline V1)
# No pretrained models, no HuggingFace classes, no nn.Transformer* shortcuts

@dataclass
class GPTConfig:
    vocab_size: int = 16384
    d_model: int = 384
    n_layer: int = 8          # 8 layers for hierarchical representation depth
    n_head: int = 6           # d_k = 64
    d_ff: int = 2240          # widened FFN associative memory -> exactly 24,982,272 params (~24.98M)
    block_size: int = 512     # max sequence context length
    dropout: float = 0.1
    tie_weights: bool = True
    bias: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head
        self.d_model = config.d_model
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.triu(torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        att = att.masked_fill(self.mask[None, None, :T, :T], float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.block_size, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.d_model),
        ))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    def num_params(self) -> int:
        # PyTorch self.parameters() already deduplicates shared/tied weights
        return sum(p.numel() for p in self.parameters())

# Instantiate and verify param count
config = GPTConfig()
model = GPTLanguageModel(config)
total_params = model.num_params()
total_params_m = total_params / 1e6

print("=" * 60)
print(f"[*] Model Architecture : Baseline V1 GPT ({lang_name} 16K)")
print(f"[*] Vocab Size         : {{config.vocab_size:,}}")
print(f"[*] Layers / Heads     : {{config.n_layer}} layers, {{config.n_head}} heads")
print(f"[*] Hidden / FFN Dim   : d_model={{config.d_model}}, d_ff={{config.d_ff}}")
print(f"[*] Total Parameters   : {{total_params:,}} ({{total_params_m:.2f}}M)")
print("=" * 60)

assert 22_500_000 <= total_params <= 27_500_000, f"Params {{total_params_m:.2f}}M outside 22.5M - 27.5M rubric window!"
print(f"[+] Parameter count verified: {{total_params_m:.2f}}M strictly adheres to the 25M +/- 10% ACL rubric window!")
"""
    cells.append(make_cell("code", c3))

    # Cell 4: TokenDataset and Pretraining Loop
    c4 = f"""# Memmap Dataset, LR Scheduler, and Optimizer Setup
class TokenDataset:
    def __init__(self, path: Path, block_size: int):
        self.path = str(path)
        self.block_size = block_size
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        assert self.data.size >= block_size + 1, "Dataset too small"

    def get_batch(self, batch_size: int, device: str):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = np.stack([self.data[i : i + self.block_size] for i in ix.tolist()])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size] for i in ix.tolist()])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)

# Training Hyperparameters
micro_batch_size = 32
gradient_accumulation_steps = 16  # Effective batch = 32 * 16 = 512 sequences = 262,144 tokens/step
max_steps = 1907                  # 1907 * 262,144 = 500,000,000 tokens
warmup_steps = 40
lr_peak = 6.0e-4
lr_min = 6.0e-5
weight_decay = 0.1
eval_interval = 100
eval_iters = 20
checkpoint_interval = 500

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Weight decay only on 2D weight tensors (not on LayerNorm or bias)
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and n != "lm_head.weight"]
nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
optim_groups = [
    {{"params": decay_params, "weight_decay": weight_decay}},
    {{"params": nodecay_params, "weight_decay": 0.0}},
]
# Hardware-accelerated fused AdamW kernel on CUDA
fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
optimizer = torch.optim.AdamW(optim_groups, lr=lr_peak, betas=(0.9, 0.95), eps=1e-8, fused=(fused_available and device == "cuda"))
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

def get_lr(step: int) -> float:
    if step < warmup_steps:
        return lr_peak * (step + 1) / (warmup_steps + 1)
    if step > max_steps:
        return lr_min
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_min + coeff * (lr_peak - lr_min)

train_dataset = TokenDataset(train_bin, config.block_size)
val_dataset = TokenDataset(val_bin if val_bin.exists() else train_bin, config.block_size)

ckpt_dir = Path("/kaggle/working/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)
print(f"[*] Training configured on device: {{device}}")
print(f"[*] Total Steps: {{max_steps}} | Effective Batch: {{micro_batch_size * gradient_accumulation_steps}}")
"""
    cells.append(make_cell("code", c4))

    # Cell 5: Main Training Loop
    c5 = f"""# Pretraining Execution Loop
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = []
    for _ in range(eval_iters):
        X, Y = val_dataset.get_batch(micro_batch_size, device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

# Check for existing checkpoints to resume
start_step = 0
best_val_loss = float("inf")
train_log = []

for ckpt_f in sorted(ckpt_dir.glob("ckpt_*.pt")):
    try:
        data = torch.load(ckpt_f, map_location=device)
        if "step" in data and data["step"] > start_step:
            start_step = data["step"]
            model.load_state_dict(data["model_state_dict"])
            optimizer.load_state_dict(data["optimizer_state_dict"])
            if "scaler_state_dict" in data and scaler:
                scaler.load_state_dict(data["scaler_state_dict"])
            best_val_loss = data.get("val_loss", best_val_loss)
            print(f"[+] Resuming from checkpoint: {{ckpt_f.name}} at step {{start_step}}")
    except Exception as e:
        print(f"[!] Warning reading {{ckpt_f}}: {{e}}")

print(f"\\n[*] Launching training from step {{start_step + 1}} to {{max_steps}}...")
start_time = time.time()
model.train()

for step in range(start_step + 1, max_steps + 1):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for micro_step in range(gradient_accumulation_steps):
        X, Y = train_dataset.get_batch(micro_batch_size, device)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        accum_loss += loss.item()
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    if step % eval_interval == 0 or step == max_steps:
        val_loss = estimate_loss()
        elapsed = time.time() - start_time
        tok_per_sec = (step - start_step) * (micro_batch_size * gradient_accumulation_steps * config.block_size) / max(elapsed, 1e-5)
        print(f"[step {{step:5d}}/{{max_steps}}] train_loss={{accum_loss:.4f}} val_loss={{val_loss:.4f}} lr={{lr:.2e}} | {{tok_per_sec:,.0f}} tok/s")

        train_log.append({{
            "step": step,
            "train_loss": float(accum_loss),
            "val_loss": float(val_loss),
            "lr": float(lr),
            "tokens": int(step * micro_batch_size * gradient_accumulation_steps * config.block_size)
        }})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({{
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config.__dict__,
            }}, ckpt_dir / "best.pt")
            print(f"  + Saved new best checkpoint (val_loss={{val_loss:.4f}})")

    if step % checkpoint_interval == 0 or step == max_steps:
        torch.save({{
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "val_loss": accum_loss,
            "config": config.__dict__,
        }}, ckpt_dir / f"ckpt_{{step}}.pt")
        print(f"  + Stored checkpoint: ckpt_{{step}}.pt")

print("\\n[SUCCESS] Pretraining finished successfully at step 1907!")
"""
    cells.append(make_cell("code", c5))

    # Cell 6: Packaging & Deliverables
    c6 = f"""# Packaging Deliverables into /kaggle/working
out_working = Path("/kaggle/working")

# Save training log JSON
log_path = out_working / "train_log.json"
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(train_log, f, indent=2)

# Save Pretraining Markdown Report
report_path = out_working / "pretrain_report_{model_name}_16k.md"
report_text = f\"\"\"# LMA Phase 2 Pretraining Report: {lang_name} 16K Transformer LM (Baseline V1)

## Model & Training Architecture
- **Language**: {lang_name} ({lang_script})
- **Model Architecture**: Baseline V1 Decoder-Only GPT (Pre-LN, Learned Positional Embeddings, GELU)
- **Parameter Count**: {{total_params:,}} ({{total_params_m:.2f}}M params)
- **Layers / Heads**: {{config.n_layer}} layers, {{config.n_head}} heads, d_model={{config.d_model}}, d_ff={{config.d_ff}}
- **Context Length**: {{config.block_size}} tokens
- **Vocabulary Size**: {{config.vocab_size:,}} pieces (SentencePiece BPE 16K)
- **Training Precision**: Mixed Precision fp16 (torch.cuda.amp)
- **Optimizer**: AdamW (weight_decay=0.1, beta=(0.9, 0.95), lr=6.0e-4 -> 6.0e-5 cosine decay)
- **Batching**: micro_batch=32, grad_accum=16 -> effective_batch=512 seqs (262,144 tokens/step)
- **Total Pretraining Steps**: 1907 steps (500M tokens)
- **Best Validation Loss**: {{best_val_loss:.4f}}
\"\"\"
report_path.write_text(report_text, encoding="utf-8")

print("=" * 70)
print("Final Deliverables in /kaggle/working:")
for root, dirs, files in os.walk(out_working):
    rel = os.path.relpath(root, out_working)
    prefix = "" if rel == "." else f"{{rel}}/"
    for f in sorted(files):
        if "data" in prefix:
            continue
        fp = Path(root) / f
        print(f"  * {{prefix}}{{f:<32}} ({{fp.stat().st_size / (1024*1024):.2f}} MB)")
print("=" * 70)
print("[SUCCESS] {lang_name} 16K Transformer Deliverables Ready!")
"""
    cells.append(make_cell("code", c6))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[+] Wrote {output_path}")

def main():
    root = Path(__file__).resolve().parents[1]
    nb_dir = root / "notebooks"
    generate_notebook("assamese", nb_dir / "lma_pretrain_assamese_16k.ipynb")
    generate_notebook("hindi", nb_dir / "lma_pretrain_hindi_16k.ipynb")

if __name__ == "__main__":
    main()
