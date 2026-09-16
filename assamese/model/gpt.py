"""Hindi GPT language model — hand-written Transformer, no pretrained pieces.

Implements the locked decisions of project specifications Section 1:
  * learned absolute positional embeddings, table (block_size, d_model)
  * hand-written multi-head scaled dot-product attention with causal mask
    (explicit matmul/softmax — no F.scaled_dot_product_attention shortcut)
  * pre-norm (LayerNorm before each sublayer)
  * GELU in the FFN, inner dim 4 x d_model
  * tied input embedding / output projection
  * no bias in Linear layers (GPT-2 style); LayerNorm retains its bias

Only primitives are used: nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout,
nn.GELU, nn.Parameter. No nn.Transformer*, no pretrained weights/tokenizers.

The identical file is duplicated into assamese/model/gpt.py by design — the two
languages share no code, no weights, no vocab (spec Section 0.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int
    block_size: int
    dropout: float = 0.1
    tie_weights: bool = True
    bias: bool = False

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "GPTConfig":
        """Load a GPTConfig from a config yaml (unknown keys are ignored)."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product attention, causal mask, hand-written.

    forward: x (B, T, d_model) -> (B, T, d_model). With return_attn=True also
    returns post-softmax attention weights (B, n_head, T, T) for the attention
    analysis toolkit.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head
        self.d_model = config.d_model
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: upper-triangular boolean buffer, not a learned Parameter.
        # Shape (block_size, block_size); sliced to (T, T) per forward call.
        mask = torch.triu(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)  # (B, T, d_model) each

        # --- multi-head reshape (this exact pattern is what Attention Analyzer's hooks assume) ---
        # (B, T, d_model) -> (B, T, n_head, d_k) -> transpose(1, 2) -> (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # Scale by 1/sqrt(d_k) BEFORE masking.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))  # (B, n_head, T, T)
        # Additive -inf on the upper triangle: future positions excluded by softmax.
        att = att.masked_fill(self.mask[None, None, :T, :T], float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, d_k)
        # --- merge heads: transpose back and concat ---
        # (B, n_head, T, d_k) -> transpose(1,2) -> (B, T, n_head, d_k) -> view (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        if return_attn:
            return y, att
        return y


class MLP(nn.Module):
    """Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Pre-norm transformer block: ln -> attn -> residual -> ln -> mlp -> residual."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            y, att = self.attn(self.ln_1(x), return_attn=True)
            x = x + y
            x = x + self.mlp(self.ln_2(x))
            return x, att
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTLanguageModel(nn.Module):
    """Full GPT-style autoregressive LM per the locked spec."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: output projection shares the input embedding matrix.
        # Both map between token-id space and d_model space, so sharing is valid.
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layer) (GPT-2 style stabilisation).
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> dict:
        """idx: (B, T) token ids. targets: (B, T) next-token ids or None.

        Returns {"logits": (B, T, vocab), "loss": scalar | None,
                 "attn_weights": list | None}  # one (B,n_head,T,T) tensor per layer
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        )

        tok = self.token_embedding(idx)  # (B, T, d_model)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T, d_model)
        x = self.drop(tok + pos)

        attn_weights: list[torch.Tensor] = []
        for block in self.blocks:
            if return_attn:
                x, att = block(x, return_attn=True)
                attn_weights.append(att)
            else:
                x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )

        return {
            "logits": logits,
            "loss": loss,
            "attn_weights": attn_weights if return_attn else None,
        }

    def num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count. Tied weights are counted exactly once.

        With non_embedding=True the token and position embeddings are excluded
        (useful for reasoning about the transformer core size).
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
            n -= self.position_embedding.weight.numel()
        return n

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        repetition_window: int = 64,
        eos_id: Optional[int] = None,
        prompt_bias: float = 0.0,
        extra_allowed_tokens: Optional[Union[torch.Tensor, list, set, tuple]] = None,
    ) -> torch.Tensor:
        """Autoregressively extend idx (B, T) by max_new_tokens tokens with top-k, top-p, repetition penalty, and optional prompt_bias."""
        self.eval()
        allowed_prompt_tokens = None
        if prompt_bias > 0.0 and idx.size(1) > 0:
            prompt_tokens = torch.unique(idx)
            special_ids = torch.tensor(
                [tid for tid in range(16384, 16390) if tid < self.config.vocab_size],
                device=idx.device,
                dtype=idx.dtype,
            )
            cat_list = [prompt_tokens, special_ids]
            if extra_allowed_tokens is not None:
                if isinstance(extra_allowed_tokens, (list, tuple, set)):
                    extra_tensor = torch.tensor(list(extra_allowed_tokens), device=idx.device, dtype=idx.dtype)
                else:
                    extra_tensor = extra_allowed_tokens.to(device=idx.device, dtype=idx.dtype).view(-1)
                if extra_tensor.numel() > 0:
                    cat_list.append(extra_tensor)
            allowed_prompt_tokens = torch.unique(torch.cat(cat_list))

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits = self(idx_cond)["logits"][:, -1, :].clone()

            # Boost tokens appearing in the input prompt (and special logic tokens)
            if prompt_bias > 0.0 and allowed_prompt_tokens is not None:
                logits[:, allowed_prompt_tokens] += prompt_bias

                # Constrained Rationale Entity Decoding:
                # If currently inside <COT_START> (16384) without <COT_END> (16385),
                # boost prompt tokens and logic tokens by an extra +2.0 to eliminate
                # out-of-prompt training entity intrusion / unigram prior bleed.
                for b in range(idx.size(0)):
                    seq = idx[b]
                    cot_start_seen = (seq == 16384).any().item()
                    cot_end_seen = (seq == 16385).any().item()
                    if cot_start_seen and not cot_end_seen:
                        logits[b, allowed_prompt_tokens] += 2.0

            # Apply repetition penalty to recently generated tokens
            if repetition_penalty > 1.0 and idx.size(1) > 0:
                for b in range(idx.size(0)):
                    recent_tokens = torch.unique(idx[b, -repetition_window:])
                    logits[b, recent_tokens] = torch.where(
                        logits[b, recent_tokens] > 0,
                        logits[b, recent_tokens] / repetition_penalty,
                        logits[b, recent_tokens] * repetition_penalty,
                    )

            if temperature is None or temperature <= 1e-5:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    for b in range(logits.size(0)):
                        indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                        logits[b, indices_to_remove] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_id is not None and (idx_next == eos_id).all():
                break
        return idx


def build_model(config: GPTConfig, device: Optional[str] = None) -> GPTLanguageModel:
    model = GPTLanguageModel(config)
    if device is not None:
        model = model.to(device)
    return model


def print_param_counts() -> dict[str, int]:
    """Build both locked configs and print exact parameter counts (spec Section 1).

    Halts (raises) if either model is outside the 22.5M–27.5M rubric window rather
    than silently proceeding with a wrong-sized model.
    """
    repo = Path(__file__).resolve().parents[1]  # <lang>/  (this module is <lang>/model/gpt.py)
    counts: dict[str, int] = {}
    for name in ("model_H.yaml", "model_L.yaml", "model_L_16k.yaml"):
        path = repo / "configs" / name
        if not path.exists():
            continue
        cfg = GPTConfig.from_yaml(path)
        n = GPTLanguageModel(cfg).num_params()
        counts[name] = n
        flag = "OK" if 22_500_000 <= n <= 27_500_000 else "OUT OF RANGE"
        print(f"{name}: {n:,} params ({n/1e6:.2f}M) [{flag}]")
        if flag != "OK":
            raise RuntimeError(f"{name} param count {n} outside 22.5M–27.5M rubric window")
    return counts


if __name__ == "__main__":
    print_param_counts()
