"""Hybrid dense / DeltaNet / top-k sparse attention language model.

This file is intentionally self-contained.  It implements a compact PyTorch
prototype for a decoder-only model with:

* inter-layer mixing: 1 dense full-attention layer followed by 3 hybrid layers;
* intra-layer mixing in hybrid layers: gated DeltaNet attention + top-k sparse
  attention;
* top-k sparse attention reusing the KV cache produced by the nearest previous
  dense full-attention layer.

The implementation favors clarity and experimental hackability over kernel
efficiency.  The top-k path still materializes dense scores before selecting the
top-k keys, so it should be replaced by a block/top-k kernel before scaling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
KVPair = Tuple[Tensor, Tensor]
DeltaState = Tuple[Tensor, Tensor]


@dataclass
class HybridDenseDeltaTopKConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    max_position_embeddings: int = 4096
    top_k: int = 128
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads


@dataclass
class HybridDenseDeltaTopKOutput:
    logits: Tensor
    loss: Optional[Tensor] = None
    cache: Optional["HybridDenseDeltaTopKCache"] = None


class HybridDenseDeltaTopKCache:
    """Autoregressive cache plus dense-layer memories for sparse reuse."""

    def __init__(self) -> None:
        self.layer_kv: Dict[int, KVPair] = {}
        self.dense_kv: Dict[int, KVPair] = {}
        self.delta_state: Dict[int, DeltaState] = {}

    def get_past_kv(self, layer_idx: int) -> Optional[KVPair]:
        return self.layer_kv.get(layer_idx)

    def set_past_kv(self, layer_idx: int, kv: KVPair) -> None:
        self.layer_kv[layer_idx] = kv

    def set_dense_kv(self, layer_idx: int, kv: KVPair) -> None:
        self.dense_kv[layer_idx] = kv

    def get_previous_dense_kv(self, layer_idx: int) -> Optional[KVPair]:
        previous_dense_layers = [idx for idx in self.dense_kv if idx < layer_idx]
        if not previous_dense_layers:
            return None
        return self.dense_kv[max(previous_dense_layers)]

    def get_delta_state(self, layer_idx: int) -> Optional[DeltaState]:
        return self.delta_state.get(layer_idx)

    def set_delta_state(self, layer_idx: int, state: DeltaState) -> None:
        self.delta_state[layer_idx] = state


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


class SwiGLUMLP(nn.Module):
    def __init__(self, config: HybridDenseDeltaTopKConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.dropout(self.down_proj(hidden_states))


def _shape_projection(hidden_states: Tensor, num_heads: int, head_dim: int) -> Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    return hidden_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype)[None, None, :, :]
    sin = emb.sin().to(dtype=dtype)[None, None, :, :]
    return cos, sin


def _rotate_half(hidden_states: Tensor) -> Tensor:
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def _apply_rope(hidden_states: Tensor, cos: Tensor, sin: Tensor, position_offset: int) -> Tensor:
    seq_len = hidden_states.shape[-2]
    cos = cos[:, :, position_offset : position_offset + seq_len, :]
    sin = sin[:, :, position_offset : position_offset + seq_len, :]
    return (hidden_states * cos) + (_rotate_half(hidden_states) * sin)


def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
    q_positions = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
    kv_positions = torch.arange(kv_len, device=device)[None, :]
    return kv_positions > q_positions


def _merge_masks(
    attention_mask: Optional[Tensor],
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> Tensor:
    mask = _causal_mask(q_len, kv_len, device=device)[None, None, :, :]
    if attention_mask is None:
        return mask

    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [batch_size, kv_len]")
    if attention_mask.shape[-1] != kv_len:
        raise ValueError("attention_mask length must match the current KV length")

    padding_mask = ~attention_mask.to(dtype=torch.bool)[:, None, None, :]
    return mask | padding_mask


def _scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor],
    dropout_p: float,
    training: bool,
) -> Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if attention_mask is not None:
        scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
    probs = F.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
    probs = F.dropout(probs, p=dropout_p, training=training)
    return torch.matmul(probs, value)


class DenseFullAttention(nn.Module):
    def __init__(self, config: HybridDenseDeltaTopKConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout_p = config.dropout

    def forward(
        self,
        hidden_states: Tensor,
        rope: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        cache: HybridDenseDeltaTopKCache,
        use_cache: bool,
    ) -> Tuple[Tensor, KVPair]:
        batch_size, q_len, _ = hidden_states.shape
        query = _shape_projection(self.q_proj(hidden_states), self.num_heads, self.head_dim)
        key = _shape_projection(self.k_proj(hidden_states), self.num_heads, self.head_dim)
        value = _shape_projection(self.v_proj(hidden_states), self.num_heads, self.head_dim)

        past_kv = cache.get_past_kv(self.layer_idx) if use_cache else None
        past_len = 0 if past_kv is None else past_kv[0].shape[-2]
        query = _apply_rope(query, rope[0], rope[1], past_len)
        key = _apply_rope(key, rope[0], rope[1], past_len)

        if past_kv is not None:
            key = torch.cat((past_kv[0], key), dim=-2)
            value = torch.cat((past_kv[1], value), dim=-2)
        if use_cache:
            cache.set_past_kv(self.layer_idx, (key, value))

        kv_len = key.shape[-2]
        merged_mask = _merge_masks(attention_mask, q_len, kv_len, hidden_states.device)
        attended = _scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=merged_mask,
            dropout_p=self.dropout_p,
            training=self.training,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        return self.o_proj(attended), (key, value)


class GatedDeltaNetAttention(nn.Module):
    """Simple recurrent gated DeltaNet attention path.

    It maintains a per-layer state for decoding and runs a Python loop over the
    current sequence.  This is fine for first experiments, but should be swapped
    for a fused scan kernel for real training runs.
    """

    def __init__(self, config: HybridDenseDeltaTopKConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.beta_proj = nn.Linear(config.hidden_size, self.num_heads, bias=True)
        self.gate_proj = nn.Linear(config.hidden_size, self.num_heads, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def _feature_map(hidden_states: Tensor) -> Tensor:
        return F.elu(hidden_states) + 1.0

    def forward(
        self,
        hidden_states: Tensor,
        cache: HybridDenseDeltaTopKCache,
        use_cache: bool,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query = self._feature_map(_shape_projection(self.q_proj(hidden_states), self.num_heads, self.head_dim))
        key = self._feature_map(_shape_projection(self.k_proj(hidden_states), self.num_heads, self.head_dim))
        value = _shape_projection(self.v_proj(hidden_states), self.num_heads, self.head_dim)
        beta = torch.sigmoid(self.beta_proj(hidden_states)).transpose(1, 2).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_proj(hidden_states)).transpose(1, 2).unsqueeze(-1)

        cached_state = cache.get_delta_state(self.layer_idx) if use_cache else None
        if cached_state is None:
            state = hidden_states.new_zeros(batch_size, self.num_heads, self.head_dim, self.head_dim)
            normalizer = hidden_states.new_zeros(batch_size, self.num_heads, self.head_dim)
        else:
            state, normalizer = cached_state

        outputs = []
        eps = 1e-6
        for idx in range(seq_len):
            k_t = key[:, :, idx, :]
            q_t = query[:, :, idx, :]
            v_t = value[:, :, idx, :]
            beta_t = beta[:, :, idx, :]

            predicted = torch.einsum("bhde,bhd->bhe", state, k_t)
            delta = (v_t - predicted) * beta_t
            state = state + torch.einsum("bhe,bhd->bhde", delta, k_t)
            normalizer = normalizer + beta_t * k_t

            y_t = torch.einsum("bhde,bhd->bhe", state, q_t)
            denom = torch.einsum("bhd,bhd->bh", normalizer, q_t).clamp_min(eps)
            y_t = y_t / denom.unsqueeze(-1)
            y_t = y_t * gate[:, :, idx, :]
            outputs.append(y_t)

        if use_cache:
            cache.set_delta_state(self.layer_idx, (state, normalizer))

        attended = torch.stack(outputs, dim=2).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.dropout(self.o_proj(attended))


class TopKSparseKVReuseAttention(nn.Module):
    """Top-k sparse attention over the nearest previous dense layer's KV cache."""

    def __init__(self, config: HybridDenseDeltaTopKConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.top_k = config.top_k
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout_p = config.dropout

    def forward(
        self,
        hidden_states: Tensor,
        dense_kv: KVPair,
        rope: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        batch_size, q_len, _ = hidden_states.shape
        key, value = dense_kv
        kv_len = key.shape[-2]
        query = _shape_projection(self.q_proj(hidden_states), self.num_heads, self.head_dim)
        query = _apply_rope(query, rope[0], rope[1], kv_len - q_len)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        merged_mask = _merge_masks(attention_mask, q_len, kv_len, hidden_states.device)
        scores = scores.masked_fill(merged_mask, torch.finfo(scores.dtype).min)

        effective_top_k = min(self.top_k, kv_len)
        topk_scores, topk_indices = torch.topk(scores, k=effective_top_k, dim=-1)
        probs = F.softmax(topk_scores.float(), dim=-1).to(dtype=hidden_states.dtype)
        probs = F.dropout(probs, p=self.dropout_p, training=self.training)

        value_expanded = value.unsqueeze(2).expand(-1, -1, q_len, -1, -1)
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        selected_values = torch.gather(value_expanded, dim=3, index=gather_indices)
        attended = torch.sum(probs.unsqueeze(-1) * selected_values, dim=3)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        return self.o_proj(attended)


class HybridDeltaTopKAttention(nn.Module):
    def __init__(self, config: HybridDenseDeltaTopKConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.delta_attn = GatedDeltaNetAttention(config, layer_idx)
        self.topk_attn = TopKSparseKVReuseAttention(config)
        self.mix_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Tensor,
        rope: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        cache: HybridDenseDeltaTopKCache,
        use_cache: bool,
    ) -> Tensor:
        dense_kv = cache.get_previous_dense_kv(self.layer_idx)
        if dense_kv is None:
            raise RuntimeError("Hybrid layers require a preceding dense full-attention layer")

        delta_out = self.delta_attn(hidden_states, cache=cache, use_cache=use_cache)
        topk_out = self.topk_attn(
            hidden_states=hidden_states,
            dense_kv=dense_kv,
            rope=rope,
            attention_mask=attention_mask,
        )
        mix_gate = torch.sigmoid(self.mix_proj(hidden_states))
        hidden_states = mix_gate * topk_out + (1.0 - mix_gate) * delta_out
        return self.dropout(hidden_states)


class HybridDenseDeltaTopKBlock(nn.Module):
    def __init__(self, config: HybridDenseDeltaTopKConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_dense_layer = layer_idx % 4 == 0
        self.input_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        if self.is_dense_layer:
            self.attention = DenseFullAttention(config, layer_idx)
        else:
            self.attention = HybridDeltaTopKAttention(config, layer_idx)
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        rope: Tuple[Tensor, Tensor],
        attention_mask: Optional[Tensor],
        cache: HybridDenseDeltaTopKCache,
        use_cache: bool,
    ) -> Tensor:
        residual = hidden_states
        normed = self.input_norm(hidden_states)
        if self.is_dense_layer:
            attn_out, dense_kv = self.attention(
                hidden_states=normed,
                rope=rope,
                attention_mask=attention_mask,
                cache=cache,
                use_cache=use_cache,
            )
            cache.set_dense_kv(self.layer_idx, dense_kv)
        else:
            attn_out = self.attention(
                hidden_states=normed,
                rope=rope,
                attention_mask=attention_mask,
                cache=cache,
                use_cache=use_cache,
            )
        hidden_states = residual + attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_norm(hidden_states))
        return hidden_states


class HybridDenseDeltaTopKForCausalLM(nn.Module):
    """Decoder-only LM with 1:3 dense-to-hybrid layer scheduling."""

    def __init__(self, config: HybridDenseDeltaTopKConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(
            [HybridDenseDeltaTopKBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        cache: Optional[HybridDenseDeltaTopKCache] = None,
        use_cache: bool = False,
    ) -> HybridDenseDeltaTopKOutput:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Only one of input_ids or inputs_embeds may be provided")

        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch_size, seq_len, _ = hidden_states.shape
        if attention_mask is None:
            attention_mask = hidden_states.new_ones(batch_size, seq_len, dtype=torch.bool)

        working_cache = cache if cache is not None else HybridDenseDeltaTopKCache()
        past_len = 0
        first_dense_past = working_cache.get_past_kv(0) if use_cache else None
        if first_dense_past is not None:
            past_len = first_dense_past[0].shape[-2]

        kv_len = past_len + seq_len
        if attention_mask.shape[-1] == seq_len and past_len > 0:
            past_mask = attention_mask.new_ones(batch_size, past_len)
            attention_mask = torch.cat((past_mask, attention_mask), dim=-1)
        if attention_mask.shape[-1] != kv_len:
            raise ValueError("attention_mask must cover the full KV length when cache is used")

        rope = _build_rope_cache(
            seq_len=kv_len,
            head_dim=self.config.head_dim,
            theta=self.config.rope_theta,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                rope=rope,
                attention_mask=attention_mask,
                cache=working_cache,
                use_cache=use_cache,
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return HybridDenseDeltaTopKOutput(
            logits=logits,
            loss=loss,
            cache=working_cache if use_cache else None,
        )


def tiny_debug_config(vocab_size: int = 1024) -> HybridDenseDeltaTopKConfig:
    """Small config useful for smoke tests."""

    return HybridDenseDeltaTopKConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=8,
        num_attention_heads=4,
        max_position_embeddings=512,
        top_k=32,
    )

