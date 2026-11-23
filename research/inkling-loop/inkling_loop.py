"""Inkling-2-style recurrent loop retrofit for pretrained open-weight causal LMs.

Faithfully implements Thinking Machines' Inkling 2 Mini recurrent loop architecture
(prelude -> shared core x P passes with prelude reset, block MessageAttentionResidual,
and per-sublayer Q/K/V/MLP rank-64 gated controllers -> coda) as a zero-degradation
retrofit on pretrained Gemma 4 (dense & MoE), Qwen 3.8 / 3.5 (hybrid GatedDeltaNet +
Full Attention), Qwen 3 (uniform GQA), and Llama-family backbones.

At initialization (step 0):
  - forget_gate_proj and message_gate_proj are zero-initialized in FP32
  - tanh(0) == 0 => out = (1 - 0) * x + 0 * m == x identically
  - Every pass restarts from the prelude output, so P-pass output is mathematically
    identical to the stock 1-pass pretrained model before the first optimizer step.
  - Because down_proj is randomly initialized, dL/dW_gain and dL/dW_forget are
    non-zero on step 0, smoothly turning on cross-pass BPTT from step 1 onward.
"""

from __future__ import annotations

from collections import UserDict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from transformers.cache_utils import DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InklingLoopConfig:
    backbone_family: Literal["gemma4", "qwen3_5", "qwen3", "llama"]
    model_id: str
    num_hidden_layers: int
    hidden_size: int
    loop_start: int
    loop_end: int
    block_size: int
    num_passes: int = 3
    gate_rank: int = 64
    rms_norm_eps: float = 1e-6
    mode: Literal["inkling2", "naive_loop", "scalar_gate_loop", "lora_only"] = "inkling2"
    core_lora_rank: int = 0
    core_lora_alpha: float = 32.0
    scale_match_message: bool = True

    @property
    def num_loop_layers(self) -> int:
        return self.loop_end - self.loop_start

    @property
    def num_blocks(self) -> int:
        return math.ceil(self.num_loop_layers / self.block_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InklingLoopConfig":
        return cls(**data)


# ---------------------------------------------------------------------------
# Inkling 2 Mini Core Loop Modules
# ---------------------------------------------------------------------------

class UnweightedRMSNorm(nn.Module):
    """Exact match to Inkling2LoopUnweightedRMSNorm."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x.to(input_dtype)


class Inkling2MessageGate(nn.Module):
    """Rank-r cross-pass message controller from Inkling 2 Mini.

    h   = W_down @ [rms_norm(x), rms_norm(m)]
    out = (1 - tanh(W_forget @ h)) * x + tanh(W_gain @ h) * m_matched

    When scale_match_message=True, m_matched = rms_norm(m) * rms(x), which reduces
    identically to Inkling 2 Mini when ||x||_rms == ||m||_rms == 1, while preserving
    full [-1, +1] tanh authority on standard-residual models (Gemma 4, Qwen 3/3.8)
    where residual stream RMS can be 10x-100x larger than 1.
    """

    def __init__(
        self,
        hidden_size: int,
        gate_rank: int = 64,
        eps: float = 1e-6,
        scale_match_message: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.scale_match_message = scale_match_message
        self.norm = UnweightedRMSNorm(eps=eps)
        self.down_proj = nn.Linear(2 * hidden_size, gate_rank, bias=False, dtype=torch.float32)
        self.forget_gate_proj = nn.Linear(gate_rank, hidden_size, bias=False, dtype=torch.float32)
        self.message_gate_proj = nn.Linear(gate_rank, hidden_size, bias=False, dtype=torch.float32)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Width-invariant normal init so gate_h has unit variance across H=2560..5376
        in_dim = self.down_proj.in_features
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=1.0 / math.sqrt(float(in_dim)))
        # Zero-init forget & message gate projections so at step 0:
        #   tanh(0) = 0  =>  out = (1 - 0)*x + 0*m = x  (EXACT identity!)
        # And dL/dW_gain, dL/dW_forget are immediately non-zero on step 0!
        nn.init.zeros_(self.forget_gate_proj.weight)
        nn.init.zeros_(self.message_gate_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        message: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        out_dtype = hidden_states.dtype
        x = hidden_states.float()
        m = message.float()
        x_norm = self.norm(x)
        m_norm = self.norm(m)
        gate_h = self.down_proj(torch.cat([x_norm, m_norm], dim=-1))
        forget_gate = torch.tanh(self.forget_gate_proj(gate_h))
        message_gate = torch.tanh(self.message_gate_proj(gate_h))
        if self.scale_match_message:
            x_rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
            m_eff = m_norm * x_rms
        else:
            m_eff = m
        out = (1.0 - forget_gate) * x + message_gate * m_eff
        out = out.to(out_dtype)
        if return_stats:
            stats = {
                "forget_rms": float(forget_gate.detach().square().mean().sqrt().item()),
                "message_rms": float(message_gate.detach().square().mean().sqrt().item()),
                "forget_abs_max": float(forget_gate.detach().abs().max().item()),
                "message_abs_max": float(message_gate.detach().abs().max().item()),
            }
            return out, stats
        return out


class ScalarPassGate(nn.Module):
    """Simpler ablation baseline: per-channel scalar gate without input-dependent controller."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.norm = UnweightedRMSNorm(eps=eps)
        self.forget_logit = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.message_logit = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        message: torch.Tensor,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        out_dtype = hidden_states.dtype
        x = hidden_states.float()
        m_norm = self.norm(message.float())
        x_rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        f = torch.tanh(self.forget_logit)
        g = torch.tanh(self.message_logit)
        out = ((1.0 - f) * x + g * (m_norm * x_rms)).to(out_dtype)
        if return_stats:
            return out, {
                "forget_rms": float(f.detach().square().mean().sqrt().item()),
                "message_rms": float(g.detach().square().mean().sqrt().item()),
            }
        return out


class Inkling2MessageAttentionResidual(nn.Module):
    """Learned per-(loop_layer, sublayer_input) attention read over previous pass's block summaries.

    Exact mathematical match to Inkling2LoopMessageAttentionResidual:
      - Normalizes stacked previous-pass block summaries via UnweightedRMSNorm
      - Computes scaled dot-product attention with 4 learned FP32 queries (Q, K, V, MLP)
      - Zero-initialized queries start with uniform attention across previous-pass blocks
    """

    def __init__(self, num_loop_layers: int, hidden_size: int, num_messages: int = 4, eps: float = 1e-6):
        super().__init__()
        self.num_loop_layers = num_loop_layers
        self.num_messages = num_messages
        self.query = nn.Parameter(torch.zeros(num_loop_layers, num_messages, hidden_size, dtype=torch.float32))
        self.norm = UnweightedRMSNorm(eps=eps)

    def forward(
        self,
        prev_block_states: torch.Tensor,
        loop_layer_idx: int,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # prev_block_states: (B, S, K, H)
        stacked = self.norm(prev_block_states.float())
        q = self.query[loop_layer_idx].float()  # (4, H)
        scale = stacked.shape[-1] ** -0.5
        scores = torch.einsum("bskh,mh->bsmk", stacked, q) * scale
        weights = F.softmax(scores, dim=-1)  # (B, S, 4, K)
        messages = torch.einsum("bsmk,bskh->bsmh", weights, stacked)  # (B, S, 4, H)
        if return_weights:
            return messages, weights.detach()
        return messages


class LoRALinear(nn.Module):
    """Self-contained zero-init LoRA wrapper around a frozen nn.Linear."""

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float = 32.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scaling = alpha / float(rank)
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        device = base_linear.weight.device
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = self.lora_B(self.lora_A(x.float())) * self.scaling
        return base_out + delta.to(base_out.dtype)


# ---------------------------------------------------------------------------
# Architecture-specific single-layer forward with per-sublayer message gates
# ---------------------------------------------------------------------------

def _apply_rotary_gemma4(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Gemma4 rotates in (B, S, n_heads, head_dim) layout -> unsqueeze_dim=2
    cos_u = cos.unsqueeze(2)
    sin_u = sin.unsqueeze(2)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos_u) + (rot * sin_u)


def _apply_rotary_standard(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # Standard Qwen3 / Qwen3.5 / Llama rotates in (B, n_heads, S, head_dim) layout -> unsqueeze_dim=1
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    def _rot(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return (x * cos_u) + (torch.cat((-x2, x1), dim=-1) * sin_u)
    return _rot(q), _rot(k)


def _gemma4_layer_forward_gated(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    gates: nn.ModuleDict,
    messages: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    shared_kv_states: dict,
    position_ids: torch.LongTensor | None = None,
    past_key_values: DynamicCache | None = None,
    cache_layer_idx: int | None = None,
    per_layer_input: torch.Tensor | None = None,
    stats_out: dict | None = None,
) -> torch.Tensor:
    m_q, m_k, m_v, m_mlp = messages.unbind(-2)
    residual = hidden_states

    if stats_out is not None:
        q_h, stats_out["q"] = gates["q"](residual, m_q, return_stats=True)
        k_h, stats_out["k"] = gates["k"](residual, m_k, return_stats=True)
        v_h, stats_out["v"] = gates["v"](residual, m_v, return_stats=True)
    else:
        q_h = gates["q"](residual, m_q)
        k_h = gates["k"](residual, m_k)
        v_h = gates["v"](residual, m_v)

    q_in = layer.input_layernorm(q_h)
    k_in = layer.input_layernorm(k_h)
    v_in = layer.input_layernorm(v_h)

    attn = layer.self_attn
    input_shape = q_in.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    cos, sin = position_embeddings

    query_states = attn.q_proj(q_in).view(hidden_shape)
    query_states = attn.q_norm(query_states)
    query_states = _apply_rotary_gemma4(query_states, cos, sin).transpose(1, 2)

    if attn.is_kv_shared_layer:
        key_states, value_states = shared_kv_states[attn.layer_type]
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = attn.k_proj(k_in).view(hidden_shape)
        if attn.v_proj is not None:
            value_states = attn.v_proj(v_in).view(hidden_shape)
        else:
            # Gemma 4 full_attention uses k_eq_v=True (v_proj is None).
            # At step 0, v_in == k_in so k_proj(v_in) == k_proj(k_in) identically,
            # while passes >= 2 allow v_gate to specialize value representations.
            value_states = attn.k_proj(v_in).view(hidden_shape)

        key_states = attn.k_norm(key_states)
        key_states = _apply_rotary_gemma4(key_states, cos, sin).transpose(1, 2)

        value_states = attn.v_norm(value_states).transpose(1, 2)

    if past_key_values is not None and not attn.is_kv_shared_layer:
        idx = attn.layer_idx if cache_layer_idx is None else cache_layer_idx
        key_states, value_states = past_key_values.update(key_states, value_states, idx)
    if attn.store_full_length_kv:
        shared_kv_states[attn.layer_type] = (key_states, value_states)

    from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward as gemma4_eager
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, gemma4_eager
    )
    attn_output, _ = attention_interface(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=attn.attention_dropout if attn.training else 0.0,
        scaling=attn.scaling,
        sliding_window=attn.sliding_window,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn.o_proj(attn_output)
    hidden_states = residual + layer.post_attention_layernorm(attn_output)

    residual = hidden_states
    if stats_out is not None:
        mlp_base, stats_out["mlp"] = gates["mlp"](residual, m_mlp, return_stats=True)
    else:
        mlp_base = gates["mlp"](residual, m_mlp)

    h1 = layer.mlp(layer.pre_feedforward_layernorm(mlp_base))
    if layer.enable_moe_block:
        h1 = layer.post_feedforward_layernorm_1(h1)
        flat = mlp_base.reshape(-1, mlp_base.shape[-1])
        _, top_k_weights, top_k_index = layer.router(flat)
        h2 = layer.pre_feedforward_layernorm_2(flat)
        h2 = layer.experts(h2, top_k_index, top_k_weights).reshape(mlp_base.shape)
        h2 = layer.post_feedforward_layernorm_2(h2)
        h = h1 + h2
    else:
        h = h1

    hidden_states = residual + layer.post_feedforward_layernorm(h)

    if layer.hidden_size_per_layer_input and per_layer_input is not None:
        res2 = hidden_states
        h_ple = layer.act_fn(layer.per_layer_input_gate(hidden_states)) * per_layer_input
        h_ple = layer.post_per_layer_input_norm(layer.per_layer_projection(h_ple))
        hidden_states = res2 + h_ple

    hidden_states = hidden_states * layer.layer_scalar
    return hidden_states


def _qwen3_5_layer_forward_gated(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    gates: nn.ModuleDict,
    messages: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any = None,
    cache_layer_idx: int | None = None,
    stats_out: dict | None = None,
) -> torch.Tensor:
    m_q, m_k, m_v, m_mlp = messages.unbind(-2)
    residual = hidden_states

    if layer.block_type == "linear_attention":
        if stats_out is not None:
            attn_h, stats_out["q"] = gates["q"](residual, m_q, return_stats=True)
        else:
            attn_h = gates["q"](residual, m_q)
        attn_in = layer.input_layernorm(attn_h)
        if cache_layer_idx is not None and past_key_values is not None:
            orig_idx = layer.linear_attn.layer_idx
            layer.linear_attn.layer_idx = cache_layer_idx
            try:
                attn_out = layer.linear_attn(
                    hidden_states=attn_in,
                    cache_params=past_key_values,
                    attention_mask=attention_mask,
                )
            finally:
                layer.linear_attn.layer_idx = orig_idx
        else:
            attn_out = layer.linear_attn(
                hidden_states=attn_in,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
    else:
        if stats_out is not None:
            q_h, stats_out["q"] = gates["q"](residual, m_q, return_stats=True)
            k_h, stats_out["k"] = gates["k"](residual, m_k, return_stats=True)
            v_h, stats_out["v"] = gates["v"](residual, m_v, return_stats=True)
        else:
            q_h = gates["q"](residual, m_q)
            k_h = gates["k"](residual, m_k)
            v_h = gates["v"](residual, m_v)

        q_in = layer.input_layernorm(q_h)
        k_in = layer.input_layernorm(k_h)
        v_in = layer.input_layernorm(v_h)

        attn = layer.self_attn
        input_shape = q_in.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states, out_gate = torch.chunk(
            attn.q_proj(q_in).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1
        )
        out_gate = out_gate.reshape(*input_shape, -1)
        query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(k_in).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(v_in).view(hidden_shape).transpose(1, 2)

        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_rotary_pos_emb as q35_apply_rotary,
            eager_attention_forward as q35_eager,
        )
        cos, sin = position_embeddings
        query_states, key_states = q35_apply_rotary(query_states, key_states, cos, sin)

        if past_key_values is not None:
            idx = attn.layer_idx if cache_layer_idx is None else cache_layer_idx
            key_states, value_states = past_key_values.update(key_states, value_states, idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation, q35_eager
        )
        attn_out, _ = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
        )
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        attn_out = attn_out * torch.sigmoid(out_gate)
        attn_out = attn.o_proj(attn_out)

    hidden_states = residual + attn_out

    residual = hidden_states
    if stats_out is not None:
        mlp_h, stats_out["mlp"] = gates["mlp"](residual, m_mlp, return_stats=True)
    else:
        mlp_h = gates["mlp"](residual, m_mlp)
    hidden_states = residual + layer.mlp(layer.post_attention_layernorm(mlp_h))
    return hidden_states


def _qwen3_or_llama_layer_forward_gated(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    gates: nn.ModuleDict,
    messages: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: DynamicCache | None = None,
    cache_layer_idx: int | None = None,
    stats_out: dict | None = None,
) -> torch.Tensor:
    m_q, m_k, m_v, m_mlp = messages.unbind(-2)
    residual = hidden_states

    if stats_out is not None:
        q_h, stats_out["q"] = gates["q"](residual, m_q, return_stats=True)
        k_h, stats_out["k"] = gates["k"](residual, m_k, return_stats=True)
        v_h, stats_out["v"] = gates["v"](residual, m_v, return_stats=True)
    else:
        q_h = gates["q"](residual, m_q)
        k_h = gates["k"](residual, m_k)
        v_h = gates["v"](residual, m_v)

    q_in = layer.input_layernorm(q_h)
    k_in = layer.input_layernorm(k_h)
    v_in = layer.input_layernorm(v_h)

    attn = layer.self_attn
    input_shape = q_in.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)

    query_states = attn.q_proj(q_in).view(hidden_shape)
    key_states = attn.k_proj(k_in).view(hidden_shape)
    value_states = attn.v_proj(v_in).view(hidden_shape)
    if hasattr(attn, "q_norm"):
        query_states = attn.q_norm(query_states)
    if hasattr(attn, "k_norm"):
        key_states = attn.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = _apply_rotary_standard(query_states, key_states, cos, sin)

    if past_key_values is not None:
        idx = attn.layer_idx if cache_layer_idx is None else cache_layer_idx
        key_states, value_states = past_key_values.update(key_states, value_states, idx)

    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as q3_eager
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attn.config._attn_implementation, q3_eager
    )
    attn_kwargs = {}
    if getattr(attn, "sliding_window", None) is not None:
        attn_kwargs["sliding_window"] = attn.sliding_window
    attn_out, _ = attention_interface(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not attn.training else attn.attention_dropout,
        scaling=attn.scaling,
        **attn_kwargs,
    )
    attn_out = attn_out.reshape(*input_shape, -1).contiguous()
    attn_out = attn.o_proj(attn_out)
    hidden_states = residual + attn_out

    residual = hidden_states
    if stats_out is not None:
        mlp_h, stats_out["mlp"] = gates["mlp"](residual, m_mlp, return_stats=True)
    else:
        mlp_h = gates["mlp"](residual, m_mlp)
    hidden_states = residual + layer.mlp(layer.post_attention_layernorm(mlp_h))
    return hidden_states


# ---------------------------------------------------------------------------
# Unified Inkling-2 Looped Causal LM Wrapper
# ---------------------------------------------------------------------------

class InklingLoopedCausalLM(nn.Module):
    """Wraps a pretrained HF CausalLM with Inkling 2 Mini's recurrent loop architecture."""

    def __init__(self, base_model: nn.Module, loop_config: InklingLoopConfig):
        super().__init__()
        self.base_model = base_model
        self.loop_config = loop_config
        self.family = loop_config.backbone_family
        self.gradient_checkpointing = False

        # Freeze entire base model first
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        self.text_model = base_model.model
        self.lm_head = base_model.lm_head
        self.text_config = self.text_model.config
        self.final_logit_softcapping = getattr(self.text_config, "final_logit_softcapping", None)

        device = next(self.text_model.parameters()).device
        H = loop_config.hidden_size
        n_loop = loop_config.num_loop_layers

        # Build Inkling-2 loop controllers
        self.message_reader: Inkling2MessageAttentionResidual | None = None
        self.loop_gates: nn.ModuleList | None = None

        if loop_config.mode == "inkling2":
            self.message_reader = Inkling2MessageAttentionResidual(
                num_loop_layers=n_loop,
                hidden_size=H,
                num_messages=4,
                eps=loop_config.rms_norm_eps,
            ).to(device)
            gates_list = []
            for local_idx in range(n_loop):
                layer_idx = loop_config.loop_start + local_idx
                layer = self.text_model.layers[layer_idx]
                is_linear = (
                    self.family == "qwen3_5" and getattr(layer, "block_type", "") == "linear_attention"
                )
                if is_linear:
                    layer_gates = nn.ModuleDict({
                        "q": Inkling2MessageGate(H, loop_config.gate_rank, loop_config.rms_norm_eps, loop_config.scale_match_message),
                        "mlp": Inkling2MessageGate(H, loop_config.gate_rank, loop_config.rms_norm_eps, loop_config.scale_match_message),
                    })
                else:
                    layer_gates = nn.ModuleDict({
                        "q": Inkling2MessageGate(H, loop_config.gate_rank, loop_config.rms_norm_eps, loop_config.scale_match_message),
                        "k": Inkling2MessageGate(H, loop_config.gate_rank, loop_config.rms_norm_eps, loop_config.scale_match_message),
                        "v": Inkling2MessageGate(H, loop_config.gate_rank, loop_config.rms_norm_eps, loop_config.scale_match_message),
                        "mlp": Inkling2MessageGate(H, loop_config.gate_rank, loop_config.rms_norm_eps, loop_config.scale_match_message),
                    })
                gates_list.append(layer_gates)
            self.loop_gates = nn.ModuleList(gates_list).to(device)

        elif loop_config.mode == "scalar_gate_loop":
            self.message_reader = Inkling2MessageAttentionResidual(
                num_loop_layers=n_loop,
                hidden_size=H,
                num_messages=4,
                eps=loop_config.rms_norm_eps,
            ).to(device)
            gates_list = []
            for local_idx in range(n_loop):
                layer_idx = loop_config.loop_start + local_idx
                layer = self.text_model.layers[layer_idx]
                is_linear = (
                    self.family == "qwen3_5" and getattr(layer, "block_type", "") == "linear_attention"
                )
                if is_linear:
                    layer_gates = nn.ModuleDict({
                        "q": ScalarPassGate(H, loop_config.rms_norm_eps),
                        "mlp": ScalarPassGate(H, loop_config.rms_norm_eps),
                    })
                else:
                    layer_gates = nn.ModuleDict({
                        "q": ScalarPassGate(H, loop_config.rms_norm_eps),
                        "k": ScalarPassGate(H, loop_config.rms_norm_eps),
                        "v": ScalarPassGate(H, loop_config.rms_norm_eps),
                        "mlp": ScalarPassGate(H, loop_config.rms_norm_eps),
                    })
                gates_list.append(layer_gates)
            self.loop_gates = nn.ModuleList(gates_list).to(device)

        # Optional LoRA on core layers (for loop+LoRA, naive_loop+LoRA, or 1-pass LoRA control)
        self.has_core_lora = False
        if loop_config.core_lora_rank > 0:
            self._inject_core_lora(loop_config.core_lora_rank, loop_config.core_lora_alpha)
            self.has_core_lora = True

    def _inject_core_lora(self, rank: int, alpha: float) -> None:
        for layer_idx in range(self.loop_config.loop_start, self.loop_config.loop_end):
            layer = self.text_model.layers[layer_idx]
            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    mod = getattr(attn, name, None)
                    if isinstance(mod, nn.Linear):
                        setattr(attn, name, LoRALinear(mod, rank=rank, alpha=alpha))
            if hasattr(layer, "mlp"):
                mlp = layer.mlp
                for name in ("gate_proj", "up_proj", "down_proj"):
                    mod = getattr(mlp, name, None)
                    if isinstance(mod, nn.Linear):
                        setattr(mlp, name, LoRALinear(mod, rank=rank, alpha=alpha))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        loop_params = 0
        if self.message_reader is not None:
            loop_params += sum(p.numel() for p in self.message_reader.parameters())
        if self.loop_gates is not None:
            loop_params += sum(p.numel() for p in self.loop_gates.parameters())
        lora_params = trainable - loop_params
        return {
            "total_params": total,
            "trainable_params": trainable,
            "loop_controller_params": loop_params,
            "core_lora_params": lora_params,
            "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
        }

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: p.detach().cpu() for k, p in self.named_parameters() if p.requires_grad}

    def save_adapter(self, out_dir: Path) -> None:
        from safetensors.torch import save_file
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "loop_config.json").write_text(json.dumps(self.loop_config.to_dict(), indent=2))
        sd = self.adapter_state_dict()
        if sd:
            save_file(sd, str(out_dir / "loop_adapter.safetensors"))

    def load_adapter(self, in_dir: Path) -> None:
        from safetensors.torch import load_file
        sd = load_file(str(in_dir / "loop_adapter.safetensors"))
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected adapter keys: {unexpected[:5]}")

    # ------------------------------------------------------------------
    # Backbone prep & single-layer dispatch
    # ------------------------------------------------------------------

    def _prepare_backbone_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
    ) -> dict[str, Any]:
        tm = self.text_model
        inputs_embeds = tm.embed_tokens(input_ids)
        B, S, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if self.family == "gemma4":
            per_layer_inputs = None
            if getattr(tm, "hidden_size_per_layer_input", 0):
                per_layer_inputs = tm.get_per_layer_inputs(input_ids, inputs_embeds)
                per_layer_inputs = tm.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
            if position_ids is None:
                position_ids = torch.arange(S, device=device).unsqueeze(0)
            mask_kwargs = {
                "config": self.text_config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            masks = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
            pos_embs = {
                lt: tm.rotary_emb(inputs_embeds, position_ids, lt)
                for lt in tm.unique_layer_types
            }
            return {
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "masks": masks,
                "pos_embs": pos_embs,
                "per_layer_inputs": per_layer_inputs,
                "shared_kv_states": UserDict(),
            }

        elif self.family == "qwen3_5":
            if position_ids is None:
                position_ids = torch.arange(S, device=device).view(1, 1, -1).expand(4, B, -1)
            elif position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(4, B, -1)
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
            mask_kwargs = {
                "config": self.text_config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": text_position_ids,
            }
            masks = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }
            pos_embs = tm.rotary_emb(inputs_embeds, rope_position_ids)
            return {
                "inputs_embeds": inputs_embeds,
                "position_ids": text_position_ids,
                "masks": masks,
                "pos_embs": pos_embs,
            }

        else:  # qwen3 or llama
            if position_ids is None:
                position_ids = torch.arange(S, device=device).unsqueeze(0)
            mask_kwargs = {
                "config": self.text_config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if getattr(tm, "has_sliding_layers", False):
                masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
            pos_embs = tm.rotary_emb(inputs_embeds, position_ids)
            return {
                "inputs_embeds": inputs_embeds,
                "position_ids": position_ids,
                "masks": masks,
                "pos_embs": pos_embs,
            }

    def _run_plain_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        prep: dict[str, Any],
    ) -> torch.Tensor:
        layer = self.text_model.layers[layer_idx]
        if self.family == "gemma4":
            lt = self.text_config.layer_types[layer_idx]
            pli = prep["per_layer_inputs"][:, :, layer_idx, :] if prep["per_layer_inputs"] is not None else None
            return layer(
                hidden_states,
                pli,
                shared_kv_states=prep["shared_kv_states"],
                position_embeddings=prep["pos_embs"][lt],
                attention_mask=prep["masks"][lt],
                position_ids=prep["position_ids"],
                past_key_values=None,
            )
        elif self.family == "qwen3_5":
            lt = self.text_config.layer_types[layer_idx]
            return layer(
                hidden_states,
                position_embeddings=prep["pos_embs"],
                attention_mask=prep["masks"][lt],
                position_ids=prep["position_ids"],
                past_key_values=None,
                use_cache=False,
            )
        else:
            lt = getattr(self.text_config, "layer_types", ["full_attention"] * self.loop_config.num_hidden_layers)[layer_idx]
            mask = prep["masks"].get(lt, prep["masks"]["full_attention"])
            return layer(
                hidden_states,
                attention_mask=mask,
                position_ids=prep["position_ids"],
                past_key_values=None,
                use_cache=False,
                position_embeddings=prep["pos_embs"],
            )

    def _run_gated_layer(
        self,
        layer_idx: int,
        loop_layer_idx: int,
        hidden_states: torch.Tensor,
        messages: torch.Tensor,
        prep: dict[str, Any],
        stats_out: dict | None = None,
    ) -> torch.Tensor:
        layer = self.text_model.layers[layer_idx]
        gates = self.loop_gates[loop_layer_idx]
        if self.family == "gemma4":
            lt = self.text_config.layer_types[layer_idx]
            pli = prep["per_layer_inputs"][:, :, layer_idx, :] if prep["per_layer_inputs"] is not None else None
            return _gemma4_layer_forward_gated(
                layer=layer,
                hidden_states=hidden_states,
                gates=gates,
                messages=messages,
                position_embeddings=prep["pos_embs"][lt],
                attention_mask=prep["masks"][lt],
                shared_kv_states=prep["shared_kv_states"],
                position_ids=prep["position_ids"],
                past_key_values=None,
                per_layer_input=pli,
                stats_out=stats_out,
            )
        elif self.family == "qwen3_5":
            lt = self.text_config.layer_types[layer_idx]
            return _qwen3_5_layer_forward_gated(
                layer=layer,
                hidden_states=hidden_states,
                gates=gates,
                messages=messages,
                position_embeddings=prep["pos_embs"],
                attention_mask=prep["masks"][lt],
                position_ids=prep["position_ids"],
                past_key_values=None,
                stats_out=stats_out,
            )
        else:
            lt = getattr(self.text_config, "layer_types", ["full_attention"] * self.loop_config.num_hidden_layers)[layer_idx]
            mask = prep["masks"].get(lt, prep["masks"]["full_attention"])
            return _qwen3_or_llama_layer_forward_gated(
                layer=layer,
                hidden_states=hidden_states,
                gates=gates,
                messages=messages,
                position_embeddings=prep["pos_embs"],
                attention_mask=mask,
                position_ids=prep["position_ids"],
                past_key_values=None,
                stats_out=stats_out,
            )

    def _maybe_checkpoint(self, fn, *tensors):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return grad_checkpoint(fn, *tensors, use_reentrant=False)
        return fn(*tensors)

    # ------------------------------------------------------------------
    # Full Looped Forward Pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        num_passes: int | None = None,
        message_mode: Literal["normal", "zero", "shuffle_tokens", "freeze_pass1"] = "normal",
        logits_to_keep: int | torch.Tensor = 0,
        collect_diagnostics: bool = False,
    ) -> dict[str, Any]:
        cfg = self.loop_config
        P = cfg.num_passes if num_passes is None else int(num_passes)
        if cfg.mode == "lora_only":
            P = 1

        loop_start = cfg.loop_start
        loop_end = cfg.loop_end
        block_size = cfg.block_size
        n_layers = cfg.num_hidden_layers

        diagnostics: dict[str, Any] = {"passes": []} if collect_diagnostics else None

        # Prelude has NO trainable parameters (all base weights are frozen; LoRA only on core).
        with torch.no_grad():
            prep = self._prepare_backbone_inputs(input_ids, attention_mask, position_ids)
            hidden_states = prep["inputs_embeds"]
            for i in range(0, loop_start):
                hidden_states = self._run_plain_layer(i, hidden_states, prep)
            pre_loop_hidden = hidden_states

        # Determine whether Pass 1 needs gradients (only if core LoRA is active or naive_loop needs it)
        pass1_needs_grad = self.has_core_lora or (cfg.mode == "naive_loop" and P == 1)

        def _run_core_pass(
            h_in: torch.Tensor,
            prev_blocks: torch.Tensor | None,
            pass_idx: int,
            use_grad: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            block_summaries = []
            h = h_in
            block_start_h = h_in
            pass_diag = {"pass": pass_idx + 1, "layers": []} if collect_diagnostics else None

            ctx = torch.enable_grad() if use_grad else torch.no_grad()
            with ctx:
                for local_idx, layer_idx in enumerate(range(loop_start, loop_end)):
                    if prev_blocks is None or cfg.mode in ("naive_loop", "lora_only"):
                        def _fn_plain(x, _li=layer_idx):
                            return self._run_plain_layer(_li, x, prep)
                        h = self._maybe_checkpoint(_fn_plain, h) if use_grad else _fn_plain(h)
                    else:
                        if collect_diagnostics:
                            msgs, attn_w = self.message_reader(prev_blocks, local_idx, return_weights=True)
                        else:
                            msgs = self.message_reader(prev_blocks, local_idx)
                            attn_w = None

                        if message_mode == "zero":
                            msgs = torch.zeros_like(msgs)
                        elif message_mode == "shuffle_tokens" and msgs.shape[1] > 1:
                            # Deterministic circular shift across sequence positions to break token alignment
                            msgs = torch.roll(msgs, shifts=max(1, msgs.shape[1] // 3), dims=1)

                        layer_stats = {} if collect_diagnostics else None

                        def _fn_gated(x, m, _li=layer_idx, _lli=local_idx):
                            return self._run_gated_layer(_li, _lli, x, m, prep, stats_out=None)

                        if collect_diagnostics:
                            h = self._run_gated_layer(layer_idx, local_idx, h, msgs, prep, stats_out=layer_stats)
                            pass_diag["layers"].append({
                                "layer_idx": layer_idx,
                                "gates": layer_stats,
                                "block_attn_mean": attn_w.mean(dim=(0, 1)).tolist(),
                            })
                        elif use_grad:
                            h = self._maybe_checkpoint(_fn_gated, h, msgs)
                        else:
                            h = _fn_gated(h, msgs)

                    is_block_end = ((local_idx + 1) % block_size == 0) or (local_idx + 1 == cfg.num_loop_layers)
                    if is_block_end:
                        # Record per-block residual delta (analogous to Inkling 2's per-block sums)
                        block_summaries.append(h - block_start_h)
                        block_start_h = h

                # Append final cumulative pass output alongside block deltas
                block_summaries.append(h)
                stacked_blocks = torch.stack(block_summaries, dim=-2)  # (B, S, K+1, H)

            if collect_diagnostics:
                diagnostics["passes"].append(pass_diag)
            return h, stacked_blocks

        # Pass 1
        h_pass, prev_blocks = _run_core_pass(
            pre_loop_hidden,
            prev_blocks=None,
            pass_idx=0,
            use_grad=(pass1_needs_grad and torch.is_grad_enabled()),
        )
        pass1_blocks = prev_blocks

        # Passes 2..P
        for p in range(1, P):
            if cfg.mode == "naive_loop":
                # Plain recurrent feed-forward: pass p+1 starts from pass p's output (no prelude reset)
                next_in = h_pass
                msg_blocks = None
            else:
                # Inkling 2 Mini semantics: EVERY pass restarts from prelude output and reads prev pass messages
                next_in = pre_loop_hidden
                msg_blocks = pass1_blocks if (message_mode == "freeze_pass1") else prev_blocks

            h_pass, prev_blocks = _run_core_pass(
                next_in,
                prev_blocks=msg_blocks,
                pass_idx=p,
                use_grad=torch.is_grad_enabled(),
            )

        # Coda (layers loop_end .. n_layers - 1)
        hidden_states = h_pass
        coda_needs_grad = torch.is_grad_enabled() and hidden_states.requires_grad
        for i in range(loop_end, n_layers):
            def _fn_coda(x, _li=i):
                return self._run_plain_layer(_li, x, prep)
            if coda_needs_grad:
                hidden_states = self._maybe_checkpoint(_fn_coda, hidden_states)
            else:
                hidden_states = _fn_coda(hidden_states)

        hidden_states = self.text_model.norm(hidden_states)

        # Compute logits / loss memory-efficiently
        loss = None
        logits = None
        if labels is not None:
            # Shift for causal LM and compute CE ONLY on valid (non -100) label positions!
            shift_h = hidden_states[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_h = shift_h.view(-1, shift_h.shape[-1])
            flat_labels = shift_labels.view(-1)
            valid_mask = flat_labels != -100
            valid_h = flat_h[valid_mask]
            valid_labels = flat_labels[valid_mask]
            if valid_labels.numel() == 0:
                loss = hidden_states.sum() * 0.0
            else:
                # Chunk projection if many valid tokens (e.g. LM batch) to keep peak logit memory low
                chunk_size = 256
                total_loss = 0.0
                n_tok = valid_labels.numel()
                for start in range(0, n_tok, chunk_size):
                    end = min(start + chunk_size, n_tok)
                    chunk_logits = self.lm_head(valid_h[start:end]).float()
                    if self.final_logit_softcapping is not None:
                        cap = float(self.final_logit_softcapping)
                        chunk_logits = torch.tanh(chunk_logits / cap) * cap
                    total_loss = total_loss + F.cross_entropy(
                        chunk_logits, valid_labels[start:end], reduction="sum"
                    )
                loss = total_loss / n_tok
        else:
            if isinstance(logits_to_keep, int) and logits_to_keep > 0:
                h_slice = hidden_states[:, -logits_to_keep:, :]
            else:
                h_slice = hidden_states
            logits = self.lm_head(h_slice).float()
            if self.final_logit_softcapping is not None:
                cap = float(self.final_logit_softcapping)
                logits = torch.tanh(logits / cap) * cap

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": hidden_states,
            "diagnostics": diagnostics,
        }

    @torch.no_grad()
    def verify_step0_identity(self, input_ids: torch.LongTensor) -> dict[str, float]:
        """Verify that P=1 and P=3 looped outputs match stock base_model logits at step 0."""
        self.eval()
        stock_out = self.base_model(input_ids=input_ids, use_cache=False)
        stock_logits = stock_out.logits[:, -1, :].float()

        p1_logits = self.forward(input_ids=input_ids, num_passes=1, logits_to_keep=1)["logits"][:, -1, :]
        p3_logits = self.forward(input_ids=input_ids, num_passes=3, logits_to_keep=1)["logits"][:, -1, :]

        diff_p1_stock = float((p1_logits - stock_logits).abs().max().item())
        diff_p3_p1 = float((p3_logits - p1_logits).abs().max().item())
        diff_p3_stock = float((p3_logits - stock_logits).abs().max().item())
        return {
            "max_abs_diff_p1_vs_stock": diff_p1_stock,
            "max_abs_diff_p3_vs_p1": diff_p3_p1,
            "max_abs_diff_p3_vs_stock": diff_p3_stock,
        }

    @torch.no_grad()
    def measure_layer_rms_profile(self, input_ids: torch.LongTensor) -> list[dict[str, float]]:
        """Measure residual stream RMS norm at every layer (diagnostic for scale matching)."""
        self.eval()
        prep = self._prepare_backbone_inputs(input_ids, None, None)
        h = prep["inputs_embeds"]
        profile = [{"layer": -1, "rms": float(h.float().square().mean().sqrt().item())}]
        for i in range(self.loop_config.num_hidden_layers):
            h = self._run_plain_layer(i, h, prep)
            profile.append({"layer": i, "rms": float(h.float().square().mean().sqrt().item())})
        return profile

    @torch.no_grad()
    def greedy_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 5,
        num_passes: int = 3,
        message_mode: str = "normal",
        stop_token_ids: set[int] | None = None,
        tokenizer=None,
    ) -> torch.LongTensor:
        """Exact greedy decoding with immediate newline/EOS stopping."""
        self.eval()
        seq = input_ids
        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=seq,
                num_passes=num_passes,
                message_mode=message_mode,
                logits_to_keep=1,
            )
            next_tok = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            seq = torch.cat([seq, next_tok], dim=-1)
            tok_id = int(next_tok.item())
            if stop_token_ids and tok_id in stop_token_ids:
                break
            if tokenizer is not None:
                piece = tokenizer.decode([tok_id], skip_special_tokens=False)
                if "\n" in piece or "\r" in piece or "|" in piece:
                    break
        return seq[:, input_ids.shape[1]:]


# ---------------------------------------------------------------------------
# Model Loading Helper (text-only for multimodal checkpoints!)
# ---------------------------------------------------------------------------

def detect_family(config) -> Literal["gemma4", "qwen3_5", "qwen3", "llama"]:
    mt = getattr(config, "model_type", "")
    tc = getattr(config, "text_config", None)
    tmt = getattr(tc, "model_type", mt) if tc is not None else mt
    if "gemma4" in mt or "gemma4" in tmt:
        return "gemma4"
    if "qwen3_5" in mt or "qwen3_5" in tmt:
        return "qwen3_5"
    if "qwen3" in mt or "qwen3" in tmt:
        return "qwen3"
    return "llama"


def default_loop_layout(family: str, num_layers: int) -> tuple[int, int, int]:
    """Return (loop_start, loop_end, block_size) aligned with native attention blocks."""
    if family == "gemma4":
        if num_layers == 60:   # Gemma 4 31B: 10 blocks of 6 -> 3 prelude + 4 core + 3 coda
            return (18, 42, 6)
        if num_layers == 30:   # Gemma 4 26B-A4B: 5 blocks of 6 -> 1 prelude + 3 core + 1 coda
            return (6, 24, 6)
        if num_layers == 48:   # Gemma 4 12B: 8 blocks of 6 -> 2 prelude + 4 core + 2 coda
            return (12, 36, 6)
        if num_layers == 42:   # Gemma 4 E4B: 7 blocks of 6 -> 2 prelude + 3 core + 2 coda
            return (12, 30, 6)
    if family == "qwen3_5":
        if num_layers == 64:   # Qwen3.8-27B: 16 blocks of 4 -> 5 prelude + 6 core + 5 coda
            return (20, 44, 4)
    # Generic 1/4 prelude, 1/2 core, 1/4 coda
    block = 6 if num_layers % 6 == 0 else 4
    n_blocks = num_layers // block
    pre_b = max(1, n_blocks // 4)
    post_b = max(1, n_blocks // 4)
    return (pre_b * block, (n_blocks - post_b) * block, block)


def _fast_load_safetensors_strict(
    model_dir: Path,
    model: nn.Module,
    key_filter_and_map,
    tie_word_embeddings: bool,
    dtype=torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Fast direct-to-CUDA safetensors loader with strict missing/unexpected checks."""
    from safetensors.torch import safe_open

    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        weight_map = json.loads(index_file.read_text())["weight_map"]
    else:
        single = model_dir / "model.safetensors"
        with safe_open(str(single), framework="pt", device="cpu") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    by_shard: dict[str, list[tuple[str, str]]] = {}
    for raw_k, shard in weight_map.items():
        dst_k = key_filter_and_map(raw_k)
        if dst_k is not None:
            by_shard.setdefault(shard, []).append((raw_k, dst_k))

    state: dict[str, torch.Tensor] = {}
    for shard, pairs in sorted(by_shard.items()):
        with safe_open(str(model_dir / shard), framework="pt", device=device) as f:
            for raw_k, dst_k in pairs:
                t = f.get_tensor(raw_k)
                if t.is_floating_point() and t.dtype != dtype:
                    t = t.to(dtype=dtype)
                state[dst_k] = t

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if tie_word_embeddings and "lm_head.weight" in missing:
        model.lm_head.weight = model.model.embed_tokens.weight
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"Strict safetensors load mismatch! missing={missing[:10]}, unexpected={unexpected[:10]}")


def _load_gemma4_text_only_strict(model_dir: Path, cfg, dtype=torch.bfloat16, device: str = "cuda"):
    """Load Gemma4ForCausalLM directly from multimodal safetensors (model.language_model.* -> model.*)."""
    from transformers import Gemma4ForCausalLM
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tc._attn_implementation = "sdpa"

    with torch.device("meta"):
        model = Gemma4ForCausalLM(tc)

    def _map(raw_k: str) -> str | None:
        if raw_k.startswith("model.language_model."):
            return "model." + raw_k[len("model.language_model."):]
        if raw_k.startswith("model.") and not raw_k.startswith(("model.vision_tower.", "model.embed_vision.", "model.audio_tower.", "model.embed_audio.")):
            return raw_k
        if raw_k.startswith("lm_head."):
            return raw_k
        return None

    _fast_load_safetensors_strict(model_dir, model, _map, getattr(tc, "tie_word_embeddings", True), dtype, device)

    model.model.rotary_emb = Gemma4TextRotaryEmbedding(tc, device=device)
    model.model.embed_tokens.embed_scale = torch.tensor(
        model.model.embed_tokens.scalar_embed_scale, device=device
    )
    if getattr(model.model, "hidden_size_per_layer_input", 0):
        model.model.embed_tokens_per_layer.embed_scale = torch.tensor(
            model.model.embed_tokens_per_layer.scalar_embed_scale, device=device
        )

    meta_left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if meta_left:
        raise RuntimeError(f"Gemma4 still has meta tensors after load: {meta_left[:10]}")
    return model


def _load_qwen3_5_text_only_strict(model_dir: Path, cfg, dtype=torch.bfloat16, device: str = "cuda"):
    """Load Qwen3_5ForCausalLM directly from multimodal safetensors in ~8 seconds."""
    from transformers import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
    tc._attn_implementation = "sdpa"

    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(tc)

    def _map(raw_k: str) -> str | None:
        if raw_k.startswith("model.language_model."):
            return "model." + raw_k[len("model.language_model."):]
        if raw_k.startswith("model.") and not raw_k.startswith(("model.visual.", "model.vision")):
            return raw_k
        if raw_k.startswith("lm_head."):
            return raw_k
        return None

    _fast_load_safetensors_strict(model_dir, model, _map, getattr(tc, "tie_word_embeddings", False), dtype, device)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(tc).to(device)
    meta_left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if meta_left:
        raise RuntimeError(f"Qwen3_5 still has meta tensors after load: {meta_left[:10]}")
    return model


def _load_qwen3_strict(model_dir: Path, cfg, dtype=torch.bfloat16, device: str = "cuda"):
    """Load Qwen3ForCausalLM directly to CUDA in ~2 seconds."""
    from transformers import Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)

    def _map(raw_k: str) -> str | None:
        return raw_k if (raw_k.startswith("model.") or raw_k.startswith("lm_head.")) else None

    _fast_load_safetensors_strict(model_dir, model, _map, getattr(cfg, "tie_word_embeddings", True), dtype, device)
    model.model.rotary_emb = Qwen3RotaryEmbedding(cfg).to(device)
    meta_left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if meta_left:
        raise RuntimeError(f"Qwen3 still has meta tensors after load: {meta_left[:10]}")
    return model


def load_text_causal_lm(model_path_or_id: str, dtype=torch.bfloat16, device: str = "cuda"):
    """Load the text-only CausalLM cleanly (stripping vision towers on multimodal checkpoints)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_path_or_id)
    local_only = model_path.exists()
    cfg = AutoConfig.from_pretrained(model_path_or_id, local_files_only=local_only)
    family = detect_family(cfg)

    if family == "gemma4" and local_only:
        model = _load_gemma4_text_only_strict(model_path, cfg, dtype=dtype, device=device)
    elif family == "qwen3_5" and local_only:
        model = _load_qwen3_5_text_only_strict(model_path, cfg, dtype=dtype, device=device)
    elif family == "qwen3" and local_only:
        model = _load_qwen3_strict(model_path, cfg, dtype=dtype, device=device)
    else:
        model, load_info = AutoModelForCausalLM.from_pretrained(
            model_path_or_id,
            dtype=dtype,
            device_map=device,
            local_files_only=local_only,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        missing = [k for k in load_info.get("missing_keys", []) if "rotary" not in k]
        if missing:
            raise RuntimeError(f"CausalLM missing keys: {missing[:10]}")

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, family
