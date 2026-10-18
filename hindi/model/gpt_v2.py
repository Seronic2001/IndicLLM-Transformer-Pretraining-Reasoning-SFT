"""Hindi GPT Version 2.0 Language Model — Modern Transformer Architecture.

Upgrades:
1. Positional Encoding : RoPE (Rotary Position Embeddings) — eliminates static embedding tables, enables extrapolation beyond 512 tokens.
2. Activation Function : SwiGLU (LLaMA/Mistral style, d_ff=1376) — superior gradient dynamics with ~25.64M parameter target.
3. Normalization       : RMSNorm (Root Mean Square LayerNorm) — eliminates mean-centering, speeding up forward/backward passes by ~12%.
4. Attention Kernel    : FlashAttention-2 / PyTorch SDPA (scaled_dot_product_attention) — 2x-3x faster attention with O(N) GPU memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfigV2:
    vocab_size: int = 32768
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    d_ff: int = 1376
    block_size: int = 512
    dropout: float = 0.0
    tie_weights: bool = True
    bias: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    arch_version: str = "v2"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "GPTConfigV2":
        """Load a GPTConfigV2 from a YAML configuration file."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA/Mistral style).
    
    Eliminates mean-centering, speeding up forward/backward passes while maintaining stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine frequency tables for Rotary Position Embeddings (RoPE)."""
    assert dim % 2 == 0, "RoPE dimension must be even"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, dim)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to query and key tensors."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_k)
        sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class ModernCausalSelfAttention(nn.Module):
    """Multi-head Causal Attention with RoPE and FlashAttention-2 / SDPA backend."""

    def __init__(self, config: GPTConfigV2):
        super().__init__()
        assert config.d_model % config.n_head == 0, "d_model must be divisible by n_head"
        self.n_head = config.n_head
        self.d_k = config.d_model // config.n_head
        self.d_model = config.d_model
        self.dropout_p = config.dropout
        self.rope_theta = config.rope_theta

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute initial RoPE cache
        cos, sin = precompute_rope_frequencies(self.d_k, config.block_size * 4, theta=config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False, cache: Optional[Dict] = None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        # (B, T, d_model) -> (B, n_head, T, d_k)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # KV cache (generation): x holds only the tokens after the `past` cached ones,
        # so they are rotated by their absolute positions past .. past + T - 1.
        past = cache["k"].size(2) if cache is not None and "k" in cache else 0
        if past + T > self.rope_cos.size(0):
            cos, sin = precompute_rope_frequencies(self.d_k, (past + T) * 2, theta=self.rope_theta)
            self.rope_cos = cos.to(x.device)
            self.rope_sin = sin.to(x.device)

        cos = self.rope_cos[past:past + T].to(dtype=q.dtype, device=q.device)
        sin = self.rope_sin[past:past + T].to(dtype=q.dtype, device=q.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if cache is not None:
            if past:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"], cache["v"] = k, v
        S = past + T  # number of keys

        if return_attn:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
            causal_mask = torch.triu(torch.ones(T, S, device=x.device, dtype=torch.bool), diagonal=past + 1)
            att = att.masked_fill(causal_mask[None, None, :, :], float("-inf"))
            att = torch.softmax(att, dim=-1)
            if self.dropout_p > 0.0 and self.training:
                att = F.dropout(att, p=self.dropout_p)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.resid_dropout(self.c_proj(y))
            return y, att

        dropout_p = self.dropout_p if self.training else 0.0
        # SDPA's is_causal aligns the mask to the top-left, which is wrong once queries
        # start after cached keys; then pass the mask explicitly (True = may attend).
        attn_mask = (None if past == 0 else
                     torch.ones(T, S, device=x.device, dtype=torch.bool).tril(diagonal=past))
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=past == 0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network (LLaMA/Mistral style).
    
    SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """

    def __init__(self, config: GPTConfigV2):
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w_up = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class ModernTransformerBlock(nn.Module):
    """Pre-norm Transformer block with RMSNorm, RoPE Attention (SDPA), and SwiGLU MLP."""

    def __init__(self, config: GPTConfigV2):
        super().__init__()
        self.norm_1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = ModernCausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, return_attn: bool = False, cache: Optional[Dict] = None):
        if return_attn:
            y, att = self.attn(self.norm_1(x), return_attn=True, cache=cache)
            x = x + y
            x = x + self.mlp(self.norm_2(x))
            return x, att
        x = x + self.attn(self.norm_1(x), cache=cache)
        x = x + self.mlp(self.norm_2(x))
        return x


class GPTLanguageModelV2(nn.Module):
    """Modern GPT Version 2.0 with RoPE, SwiGLU, RMSNorm, and FlashAttention-2."""

    def __init__(self, config: GPTConfigV2):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([ModernTransformerBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w_down.weight"):
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
        kv_cache: Optional[List[Dict]] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """kv_cache: one dict per layer (start with empty dicts) holding the keys and
        values of earlier positions; idx then holds only the tokens after them."""
        B, T = idx.shape
        x = self.drop(self.token_embedding(idx))

        attn_weights: List[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            cache = kv_cache[i] if kv_cache is not None else None
            if return_attn:
                x, att = block(x, return_attn=True, cache=cache)
                attn_weights.append(att)
            else:
                x = block(x, cache=cache)
        x = self.norm_f(x)
        logits = self.lm_head(x)

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
        """Total parameter count. Tied weights are counted exactly once."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
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
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Autoregressively extend idx (B, T) with top-k, top-p, and repetition penalty.

        With eos_id set, generation stops once every sequence has emitted it.
        use_cache keeps each layer's keys and values, so every step runs only the new
        token (much faster on CPU). It applies when prompt + new tokens fit in block_size;
        logits can differ from the uncached path by float rounding, so evaluation keeps
        the default (False) for reproducible scores.
        """
        self.eval()
        finished = torch.zeros(idx.size(0), dtype=torch.bool, device=idx.device)
        cache = ([{} for _ in self.blocks]
                 if use_cache and idx.size(1) + max_new_tokens <= self.config.block_size else None)
        for step in range(max_new_tokens):
            if cache is not None:
                logits = self(idx if step == 0 else idx[:, -1:], kv_cache=cache)["logits"][:, -1, :].clone()
            else:
                # Crop to the trained context length: RoPE runs on longer inputs, but the
                # model never saw relative offsets beyond block_size during training.
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
                logits = self(idx_cond)["logits"][:, -1, :].clone()

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
                if temperature != 1.0:
                    logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    for b in range(logits.size(0)):
                        indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                        logits[b, indices_to_remove] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if eos_id is not None:
                finished |= idx_next.squeeze(1) == eos_id
                if finished.all():
                    break
        return idx
