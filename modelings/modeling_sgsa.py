"""Sparse-Guided State Attention (SGSA) prototype.

This module prioritizes correctness and inspectability over speed:
- exact PyTorch sparse retrieval (token/block modes);
- explicit recurrent state update loop;
- switchable SGSA write modes for ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelings.sgsa_ops import chunk_sgsa, naive_recurrent_sgsa

_ROOT = Path(__file__).resolve().parents[1]
_FLA_LOCAL = _ROOT / "flash-linear-attention"
if _FLA_LOCAL.exists() and str(_FLA_LOCAL) not in sys.path:
    # Prefer the in-repo FLA implementation over an environment package named `fla`.
    sys.path.insert(0, str(_FLA_LOCAL))

try:
    from fla.ops.gated_delta_rule import chunk_gdn
except Exception:
    chunk_gdn = None

try:
    from flash_moba import flash_moba_attn_varlen_func, flash_topk_varlen_func
except Exception:
    flash_moba_attn_varlen_func = None
    flash_topk_varlen_func = None

from fla.modules.layernorm import RMSNorm
from fla.modules.mlp import GatedMLP

Tensor = torch.Tensor


@dataclass
class SGSAConfig:
    vocab_size: int = 32000
    hidden_size: int = 256
    intermediate_size: int = 512
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_kv_heads: int = 4
    max_position_embeddings: int = 2048
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 0
    top_k: int = 16
    block_size: int = 32
    top_blocks: int = 2
    retrieval_mode: str = "block_token"  # token_only | block_only | block_token
    sgsa_write_mode: str = "direct"  # none | direct | residual
    alpha_max: float = 0.25
    tau_c: float = 0.1
    local_window: int = 64
    sink_size: int = 8
    sparse_output_lambda: float = 0.0
    tie_word_embeddings: bool = True
    linear_backend: str = "auto"  # auto | python | torch_chunk | fla_gdn
    retrieval_backend: str = "auto"  # auto | torch | flash_moba
    chunk_size: int = 64

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads


def _to_heads(hidden_states: Tensor, num_heads: int, head_dim: int) -> Tensor:
    bsz, seq_len, _ = hidden_states.shape
    return hidden_states.view(bsz, seq_len, num_heads, head_dim)


def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
    positions = torch.arange(seq_len, device=device)
    return positions[None, :] <= positions[:, None]


class SparseRetriever(nn.Module):
    """Exact sparse retrieval with token- and block-level modes."""

    def __init__(self, config: SGSAConfig) -> None:
        super().__init__()
        self.top_k = config.top_k
        self.top_blocks = config.top_blocks
        self.block_size = config.block_size
        self.mode = config.retrieval_mode
        self.retrieval_backend = config.retrieval_backend
        self.use_flash_moba = (
            self.retrieval_backend in ("auto", "flash_moba")
            and flash_moba_attn_varlen_func is not None
            and flash_topk_varlen_func is not None
        )
        self.local_window = config.local_window
        self.sink_size = config.sink_size
        self.tau_c = config.tau_c
        self.eps = 1e-6

    def _build_token_mask(self, seq_len: int, device: torch.device) -> Tensor:
        return _causal_mask(seq_len, device=device)[None, None, :, :]

    def _token_only(
        self,
        scores: Tensor,
        valid_mask: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Dict[str, Tensor]:
        bsz, num_heads, q_len, kv_len = scores.shape
        topk_scores, topk_indices, topk_valid = self._masked_topk(
            scores=scores,
            valid_mask=valid_mask,
            topk=min(self.top_k, kv_len),
        )
        topk_weights = F.softmax(topk_scores.float(), dim=-1).to(dtype=scores.dtype)
        topk_weights = topk_weights * topk_valid.to(dtype=scores.dtype)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, k.shape[-1])
        selected_k = torch.gather(k.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, gather_indices)
        selected_v = torch.gather(v.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, gather_indices)
        k_hat = torch.sum(topk_weights.unsqueeze(-1) * selected_k, dim=3)
        v_hat = torch.sum(topk_weights.unsqueeze(-1) * selected_v, dim=3)
        return {
            "scores": topk_scores,
            "weights": topk_weights,
            "token_indices": topk_indices,
            "block_indices": None,
            "k_hat": k_hat,
            "v_hat": v_hat,
        }

    def _masked_topk(self, scores: Tensor, valid_mask: Tensor, topk: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Top-k with per-query valid-count handling.

        Returns fixed-size tensors [B, H, Q, topk], padding invalid slots.
        """
        bsz, num_heads, q_len, kv_len = scores.shape
        dtype_min = torch.finfo(scores.dtype).min
        top_scores = scores.new_full((bsz, num_heads, q_len, topk), dtype_min)
        top_indices = scores.new_zeros((bsz, num_heads, q_len, topk), dtype=torch.long)
        top_valid = torch.zeros((bsz, num_heads, q_len, topk), dtype=torch.bool, device=scores.device)

        for q_idx in range(q_len):
            row_scores = scores[:, :, q_idx, :]
            row_mask = valid_mask[:, :, q_idx, :]
            masked_row = row_scores.masked_fill(~row_mask, dtype_min)
            valid_count = int(row_mask.sum(dim=-1).max().item())
            k_q = min(topk, max(valid_count, 1))
            row_top_scores, row_top_indices = torch.topk(masked_row, k=k_q, dim=-1)
            top_scores[:, :, q_idx, :k_q] = row_top_scores
            top_indices[:, :, q_idx, :k_q] = row_top_indices
            top_valid[:, :, q_idx, :k_q] = row_mask.gather(dim=-1, index=row_top_indices)
        return top_scores, top_indices, top_valid

    def _build_block_id_map(self, seq_len: int, device: torch.device) -> Tuple[Tensor, int]:
        block_ids = torch.div(torch.arange(seq_len, device=device), self.block_size, rounding_mode="floor")
        num_blocks = int(block_ids.max().item()) + 1
        return block_ids, num_blocks

    def _select_blocks(
        self,
        scores: Tensor,
        valid_mask: Tensor,
        num_blocks: int,
        block_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        bsz, num_heads, q_len, _ = scores.shape
        block_scores = scores.new_full((bsz, num_heads, q_len, num_blocks), torch.finfo(scores.dtype).min)
        for blk in range(num_blocks):
            token_mask = block_ids == blk
            blk_valid = valid_mask[..., token_mask]
            blk_scores = scores[..., token_mask]
            blk_scores = blk_scores.masked_fill(~blk_valid, torch.finfo(scores.dtype).min)
            block_scores[..., blk] = blk_scores.max(dim=-1).values

        effective_blocks = min(self.top_blocks, num_blocks)
        block_valid = block_scores > torch.finfo(block_scores.dtype).min / 2
        top_block_scores, top_block_indices, _ = self._masked_topk(
            scores=block_scores,
            valid_mask=block_valid,
            topk=effective_blocks,
        )
        return top_block_scores, top_block_indices

    def _moba_aggregate_kv(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        # q, k, v: [B, H, T, D]
        if not self.use_flash_moba or not q.is_cuda:
            return None
        bsz, num_heads, seq_len, _ = q.shape
        if bsz == 0 or seq_len == 0:
            return None

        q_bt = q.permute(0, 2, 1, 3).contiguous()
        k_bt = k.permute(0, 2, 1, 3).contiguous()
        v_bt = v.permute(0, 2, 1, 3).contiguous()

        if attention_mask is None:
            token_mask = torch.ones((bsz, seq_len), device=q.device, dtype=torch.bool)
            lengths = torch.full((bsz,), seq_len, device=q.device, dtype=torch.int32)
        else:
            if attention_mask.shape != (bsz, seq_len):
                raise ValueError("attention_mask must have shape [batch, seq]")
            token_mask = attention_mask.to(dtype=torch.bool, device=q.device)
            lengths = token_mask.sum(dim=-1, dtype=torch.int32)

        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_len == 0:
            empty = q.new_zeros((bsz, num_heads, seq_len, q.shape[-1]))
            empty_idx = torch.full((bsz, num_heads, seq_len, 1), -1, device=q.device, dtype=torch.long)
            empty_valid = torch.zeros_like(empty_idx, dtype=torch.bool)
            return empty, empty, empty_idx, empty_valid

        cu_seqlens = torch.cat(
            [torch.zeros((1,), device=q.device, dtype=torch.int32), lengths.cumsum(dim=0)],
            dim=0,
        )
        q_flat = q_bt[token_mask]
        k_flat = k_bt[token_mask]
        v_flat = v_bt[token_mask]
        q_flat = q_flat.contiguous()
        k_flat = k_flat.contiguous()
        v_flat = v_flat.contiguous()

        col_offsets, col_nnz, row_indices = flash_topk_varlen_func(
            q=q_flat,
            k=k_flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            moba_topk=self.top_blocks,
            moba_chunk_size=self.block_size,
            causal=True,
        )
        topk_rounded = row_indices.numel() // (q_flat.shape[0] * num_heads)
        packed_block_indices = row_indices.view(q_flat.shape[0], num_heads, topk_rounded).long()

        # Reuse one sparse pattern to aggregate V and K (for v_hat / k_hat).
        moba_v_out = flash_moba_attn_varlen_func(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            moba_col_offsets=col_offsets,
            moba_col_nnz=col_nnz,
            moba_row_indices=row_indices,
            lg_block_n=self.block_size,
            dropout_p=0.0,
            causal=True,
        )
        moba_k_out = flash_moba_attn_varlen_func(
            q=q_flat,
            k=k_flat,
            v=k_flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_len,
            max_seqlen_k=max_len,
            moba_col_offsets=col_offsets,
            moba_col_nnz=col_nnz,
            moba_row_indices=row_indices,
            lg_block_n=self.block_size,
            dropout_p=0.0,
            causal=True,
        )
        v_out_bt = q_bt.new_zeros((bsz, seq_len, num_heads, q.shape[-1]))
        k_out_bt = q_bt.new_zeros((bsz, seq_len, num_heads, q.shape[-1]))
        block_idx_bt = torch.full((bsz, seq_len, num_heads, topk_rounded), -1, device=q.device, dtype=torch.long)
        v_out_bt[token_mask] = moba_v_out
        k_out_bt[token_mask] = moba_k_out
        block_idx_bt[token_mask] = packed_block_indices
        block_idx = block_idx_bt.permute(0, 2, 1, 3).contiguous()
        block_valid = block_idx >= 0
        return (
            k_out_bt.permute(0, 2, 1, 3).contiguous(),
            v_out_bt.permute(0, 2, 1, 3).contiguous(),
            block_idx.clamp_min(0),
            block_valid,
        )

    def _tokens_from_blocks(
        self,
        scores: Tensor,
        valid_mask: Tensor,
        k: Tensor,
        v: Tensor,
        block_ids: Tensor,
        top_block_indices: Tensor,
        top_block_valid: Optional[Tensor],
        refine_tokens: bool,
    ) -> Dict[str, Tensor]:
        bsz, num_heads, q_len, kv_len = scores.shape
        top_blocks_expanded = top_block_indices.unsqueeze(-2)
        block_ids_expanded = block_ids.view(1, 1, 1, kv_len, 1)
        in_selected_blocks = (block_ids_expanded == top_blocks_expanded).any(dim=-1)
        if top_block_valid is not None:
            selected_any_valid = top_block_valid.any(dim=-1, keepdim=True)
            in_selected_blocks = in_selected_blocks & selected_any_valid
        candidate_mask = valid_mask & in_selected_blocks
        candidate_scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)

        if refine_tokens:
            effective_k = min(self.top_k, kv_len)
            sel_scores, sel_indices, sel_valid = self._masked_topk(
                scores=candidate_scores,
                valid_mask=candidate_mask,
                topk=effective_k,
            )
            sel_weights = F.softmax(sel_scores.float(), dim=-1).to(dtype=scores.dtype)
            sel_weights = sel_weights * sel_valid.to(dtype=scores.dtype)
            sel_weights = sel_weights / sel_weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            gather_indices = sel_indices.unsqueeze(-1).expand(-1, -1, -1, -1, k.shape[-1])
            selected_k = torch.gather(k.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, gather_indices)
            selected_v = torch.gather(v.unsqueeze(2).expand(-1, -1, q_len, -1, -1), 3, gather_indices)
            k_hat = torch.sum(sel_weights.unsqueeze(-1) * selected_k, dim=3)
            v_hat = torch.sum(sel_weights.unsqueeze(-1) * selected_v, dim=3)
            return {
                "scores": sel_scores,
                "weights": sel_weights,
                "token_indices": sel_indices,
                "block_indices": top_block_indices,
                "k_hat": k_hat,
                "v_hat": v_hat,
            }

        token_weights = F.softmax(candidate_scores.float(), dim=-1).to(dtype=scores.dtype)
        token_weights = token_weights * candidate_mask
        token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        k_hat = torch.einsum("bhqk,bhkd->bhqd", token_weights, k)
        v_hat = torch.einsum("bhqk,bhkd->bhqd", token_weights, v)
        return {
            "scores": top_block_indices.float(),
            "weights": token_weights,
            "token_indices": None,
            "block_indices": top_block_indices,
            "k_hat": k_hat,
            "v_hat": v_hat,
        }

    def _diagnostics(
        self,
        selected_scores: Tensor,
        selected_weights: Tensor,
        token_indices: Optional[Tensor],
        k_hat: Tensor,
        self_key: Tensor,
    ) -> Dict[str, Tensor]:
        if selected_scores.shape[-1] >= 2:
            score_margin = selected_scores[..., 0] - selected_scores[..., 1]
        else:
            score_margin = torch.zeros_like(selected_scores[..., 0])
        concentration = torch.sigmoid(score_margin / max(self.tau_c, self.eps))
        max_confidence = selected_weights.max(dim=-1).values

        dot = torch.sum(k_hat * self_key, dim=-1)
        self_norm = self_key.norm(dim=-1).clamp_min(self.eps)
        k_hat_norm = k_hat.norm(dim=-1).clamp_min(self.eps)
        proj_coeff = dot / (self_norm * self_norm + self.eps)
        k_perp = k_hat - proj_coeff.unsqueeze(-1) * self_key
        novelty = (k_perp.norm(dim=-1) / (k_hat_norm + self.eps)).clamp(0.0, 1.0)
        write_conflict = (dot / (self_norm * k_hat_norm + self.eps)).clamp(-1.0, 1.0)

        entropy = -torch.sum(selected_weights.clamp_min(self.eps) * selected_weights.clamp_min(self.eps).log(), dim=-1)
        diagnostics: Dict[str, Tensor] = {
            "score_margin": score_margin,
            "concentration": concentration,
            "max_confidence": max_confidence,
            "novelty": novelty,
            "write_conflict": write_conflict,
            "entropy": entropy,
        }

        if token_indices is not None:
            seq_len = int(token_indices.max().item()) + 1 if token_indices.numel() > 0 else 1
            query_pos = torch.arange(token_indices.shape[2], device=token_indices.device).view(1, 1, -1, 1)
            distance = (query_pos - token_indices).clamp_min(0)
            local = (distance <= self.local_window).float().mean(dim=-1)
            sink = (token_indices < self.sink_size).float().mean(dim=-1)
            remote = ((distance > self.local_window) & (token_indices >= self.sink_size)).float().mean(dim=-1)
            diagnostics["avg_distance"] = distance.float().mean(dim=-1) / max(seq_len, 1)
            diagnostics["local_ratio"] = local
            diagnostics["sink_ratio"] = sink
            diagnostics["remote_ratio"] = remote
        return diagnostics

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool = True,
        attention_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # q, k, v: [batch, seq, heads, dim]
        bsz, seq_len, num_heads, _ = q.shape
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(kh.shape[-1])
        valid_mask = self._build_token_mask(seq_len, q.device).expand(bsz, num_heads, -1, -1)
        if not causal:
            valid_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        if attention_mask is not None:
            attn_bool = attention_mask.to(dtype=torch.bool, device=q.device)
            key_valid = attn_bool[:, None, None, :].expand(-1, num_heads, seq_len, -1)
            query_valid = attn_bool[:, None, :, None].expand(-1, num_heads, -1, seq_len)
            valid_mask = valid_mask & key_valid & query_valid

        if self.mode == "token_only":
            result = self._token_only(scores=scores, valid_mask=valid_mask, k=kh, v=vh)
        else:
            block_ids, num_blocks = self._build_block_id_map(seq_len=seq_len, device=q.device)
            top_block_valid = None
            moba_kv = self._moba_aggregate_kv(q=qh, k=kh, v=vh, attention_mask=attention_mask)
            if moba_kv is not None:
                moba_k_hat, moba_v_hat, top_block_indices, top_block_valid = moba_kv
            else:
                _, top_block_indices = self._select_blocks(
                    scores=scores,
                    valid_mask=valid_mask,
                    num_blocks=num_blocks,
                    block_ids=block_ids,
                )
            result = self._tokens_from_blocks(
                scores=scores,
                valid_mask=valid_mask,
                k=kh,
                v=vh,
                block_ids=block_ids,
                top_block_indices=top_block_indices,
                top_block_valid=top_block_valid,
                refine_tokens=self.mode == "block_token",
            )
            if moba_kv is not None:
                result["k_hat"] = moba_k_hat
                result["v_hat"] = moba_v_hat

        self_key = kh[:, :, torch.arange(seq_len, device=q.device), :]
        diagnostics = self._diagnostics(
            selected_scores=result["scores"],
            selected_weights=result["weights"],
            token_indices=result["token_indices"],
            k_hat=result["k_hat"],
            self_key=self_key,
        )
        return {
            "indices": result["token_indices"],
            "block_indices": result["block_indices"],
            "weights": result["weights"],
            "k_hat": result["k_hat"].transpose(1, 2),
            "v_hat": result["v_hat"].transpose(1, 2),
            "diagnostics": diagnostics,
        }


class SGSAStateLayer(nn.Module):
    """Recurrent state update with optional SGSA sparse write."""

    def __init__(self, config: SGSAConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.beta_proj = nn.Linear(config.hidden_size, self.num_heads, bias=True)
        self.gamma_proj = nn.Linear(config.hidden_size, self.num_heads, bias=True)
        self.budget_proj = nn.Linear(config.hidden_size, self.num_heads, bias=True)
        self.alpha_max = config.alpha_max
        self.write_mode = config.sgsa_write_mode
        self.linear_backend = config.linear_backend
        self.use_fla_gdn = (
            self.linear_backend in ("auto", "fla_gdn")
            and chunk_gdn is not None
        )
        self.eps = 1e-6

    def _compute_alpha(self, hidden_states: Tensor, diagnostics: Dict[str, Tensor], write_key: Tensor, v_hat: Tensor) -> Tensor:
        concentration = diagnostics["concentration"].transpose(1, 2)
        novelty = diagnostics["novelty"].transpose(1, 2).clamp(0.0, 1.0)
        budget = torch.sigmoid(self.budget_proj(hidden_states))
        alpha = self.alpha_max * concentration * novelty * budget
        proxy_norm = alpha * v_hat.norm(dim=-1) * write_key.norm(dim=-1)
        clip = torch.clamp(1.0 / proxy_norm.clamp_min(1.0), max=1.0)
        return alpha * clip

    def _fla_gdn_forward(self, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        # FLA expects g in log space as decay.
        g_log = torch.log(gamma.clamp_min(self.eps))
        # q,k,v: [B,T,H,D], beta/gamma: [B,T,H]
        out, _ = chunk_gdn(
            q=q,
            k=k,
            v=v,
            g=g_log,
            beta=beta,
            scale=1.0,
            initial_state=None,
            output_final_state=False,
        )
        return out

    def forward(
        self,
        hidden_states: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        k_hat: Tensor,
        v_hat: Tensor,
        diagnostics: Dict[str, Tensor],
        init_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        # all tensor inputs are [batch, seq, heads, dim]
        bsz, seq_len, num_heads, dim = q.shape
        if num_heads != self.num_heads or dim != self.head_dim:
            raise ValueError("input head shape does not match config")

        if init_state is None:
            state = hidden_states.new_zeros(bsz, num_heads, dim, dim)
        else:
            state = init_state

        beta = torch.sigmoid(self.beta_proj(hidden_states)).clamp_min(1e-4) # gain for state update
        gamma = torch.sigmoid(self.gamma_proj(hidden_states)) # decay for state update

        k_hat_perp = k_hat
        if self.write_mode == "residual":
            dot_k = (k_hat * k).sum(dim=-1)
            k_norm_sq = (k * k).sum(dim=-1).clamp_min(self.eps)
            k_hat_perp = k_hat - (dot_k / k_norm_sq).unsqueeze(-1) * k
        write_key = k_hat if self.write_mode == "direct" else k_hat_perp
        alpha = self._compute_alpha(hidden_states, diagnostics, write_key=write_key, v_hat=v_hat) # gain for sparse write

        use_fla_path = (
            self.use_fla_gdn
            and q.is_cuda
            and self.write_mode == "none"
            and init_state is None
        )
        use_torch_chunk_path = (
            self.linear_backend in ("auto", "torch_chunk")
            and (self.write_mode != "none" or not use_fla_path)
        )
        if use_fla_path:
            attended = self._fla_gdn_forward(q=q, k=k, v=v, beta=beta, gamma=gamma)
            # Keep state output for API compatibility when using kernel path.
            state = hidden_states.new_zeros(bsz, num_heads, dim, dim)
        elif use_torch_chunk_path:
            attended, state = chunk_sgsa(
                q=q,
                k=k,
                v=v,
                beta=beta,
                gamma=gamma,
                sparse_k=write_key if self.write_mode != "none" else None,
                sparse_v=v_hat if self.write_mode != "none" else None,
                alpha=alpha if self.write_mode != "none" else None,
                initial_state=state,
                chunk_size=self.config.chunk_size,
                output_final_state=True,
            )
        else:
            attended, state = naive_recurrent_sgsa(
                q=q,
                k=k,
                v=v,
                beta=beta,
                gamma=gamma,
                sparse_k=write_key if self.write_mode != "none" else None,
                sparse_v=v_hat if self.write_mode != "none" else None,
                alpha=alpha if self.write_mode != "none" else None,
                initial_state=state,
                output_final_state=True,
            )
        stats = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "k_hat_perp_norm_ratio": (k_hat_perp.norm(dim=-1) / k_hat.norm(dim=-1).clamp_min(self.eps)).clamp(0.0, 10.0) if self.write_mode == "residual" else None,
            "used_fla_gdn": torch.tensor(float(use_fla_path), device=hidden_states.device),
            "used_torch_chunk": torch.tensor(float(use_torch_chunk_path and not use_fla_path), device=hidden_states.device),
        }
        return attended, state, stats

class SwiGLUMLP(nn.Module):
    def __init__(self, config: SGSAConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.dropout(self.down_proj(hidden_states))


class SGSABlock(nn.Module):
    def __init__(self, config: SGSAConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.input_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_ret_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.sparse_retriever = SparseRetriever(config)
        self.state_layer = SGSAStateLayer(config)
        self.sparse_out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sparse_output_lambda = config.sparse_output_lambda
        self.dropout = nn.Dropout(config.dropout)
        # self.mlp = SwiGLUMLP(config)
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=4,
            intermediate_size=config.intermediate_size,
            hidden_act="swish",
            fuse_swiglu=True,
        )

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        residual = hidden_states
        normed = self.input_norm(hidden_states)
        q = _to_heads(self.q_proj(normed), self.num_heads, self.head_dim)
        k = _to_heads(self.k_proj(normed), self.num_kv_heads, self.head_dim)
        v = _to_heads(self.v_proj(normed), self.num_kv_heads, self.head_dim)
        q_ret = _to_heads(self.q_ret_proj(normed), self.num_heads, self.head_dim)

        retrieval = self.sparse_retriever(
            q=q_ret,
            k=k,
            v=v,
            causal=True,
            attention_mask=attention_mask,
        ) # [batch, seq, heads, dim]
        attended, _, state_stats = self.state_layer(
            hidden_states=normed,
            q=q,
            k=k,
            v=v,
            k_hat=retrieval["k_hat"],
            v_hat=retrieval["v_hat"],
            diagnostics=retrieval["diagnostics"],
        )

        attended = attended.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        attn_out = self.o_proj(attended)

        # TODO: replace constant output_lambda with learnable parameter
        if self.sparse_output_lambda > 0:
            sparse_out = retrieval["v_hat"].reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
            sparse_out = self.sparse_out_proj(sparse_out)
            attn_out = attn_out + self.sparse_output_lambda * sparse_out

        hidden_states = residual + self.dropout(attn_out)
        hidden_states = hidden_states + self.mlp(self.post_attn_norm(hidden_states))
        metrics = {
            "alpha_mean": state_stats["alpha"].mean(),
            "k_hat_perp_ratio_mean": (
                state_stats["k_hat_perp_norm_ratio"].mean()
                if state_stats["k_hat_perp_norm_ratio"] is not None
                else hidden_states.new_zeros(())
            ),
            "retrieval_confidence_mean": retrieval["diagnostics"]["max_confidence"].mean(),
            "used_fla_gdn": state_stats["used_fla_gdn"],
        }
        return hidden_states, metrics


class SGSAForCausalLM(nn.Module):
    def __init__(self, config: SGSAConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([SGSABlock(config) for _ in range(config.num_hidden_layers)])
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

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        hidden_states = self.embed_tokens(input_ids) + self.pos_embed(positions)
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1).to(hidden_states.dtype)

        all_metrics: Dict[str, Tensor] = {}
        for idx, layer in enumerate(self.layers):
            hidden_states, metrics = layer(hidden_states, attention_mask=attention_mask)
            all_metrics[f"layer_{idx}_alpha_mean"] = metrics["alpha_mean"]
            all_metrics[f"layer_{idx}_k_hat_perp_ratio_mean"] = metrics["k_hat_perp_ratio_mean"]
            all_metrics[f"layer_{idx}_retrieval_confidence_mean"] = metrics["retrieval_confidence_mean"]
            all_metrics[f"layer_{idx}_used_fla_gdn"] = metrics["used_fla_gdn"]

        logits = self.lm_head(self.norm(hidden_states))
        output: Dict[str, Tensor] = {"logits": logits, "metrics": all_metrics}
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            output["loss"] = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return output


def tiny_sgsa_config(vocab_size: int = 128) -> SGSAConfig:
    return SGSAConfig(
        vocab_size=vocab_size,
        hidden_size=96,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=256,
        top_k=8,
        block_size=16,
        top_blocks=2,
    )
