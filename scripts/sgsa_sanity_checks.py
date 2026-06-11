"""Sanity checks for SGSA prototype.

Usage:
  conda run -n zjh_dev python scripts/sgsa_sanity_checks.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modelings.modeling_sgsa import SGSAStateLayer, SparseRetriever, tiny_sgsa_config


def check_causal_topk_and_shapes() -> None:
    cfg = tiny_sgsa_config()
    cfg = replace(cfg, retrieval_mode="token_only", top_k=4)
    retriever = SparseRetriever(cfg)

    bsz, seq_len, num_heads, dim = 2, 12, cfg.num_attention_heads, cfg.head_dim
    q = torch.randn(bsz, seq_len, num_heads, dim)
    k = torch.randn(bsz, seq_len, num_heads, dim)
    v = torch.randn(bsz, seq_len, num_heads, dim)

    out = retriever(q=q, k=k, v=v, causal=True)
    indices = out["indices"]
    assert indices is not None
    query_positions = torch.arange(seq_len).view(1, 1, seq_len, 1)
    assert torch.all(indices <= query_positions), "causal top-k selected future token"
    assert out["k_hat"].shape == (bsz, seq_len, num_heads, dim)
    assert out["v_hat"].shape == (bsz, seq_len, num_heads, dim)
    assert out["diagnostics"]["score_margin"].shape == (bsz, num_heads, seq_len)


def check_residual_write_orthogonality() -> None:
    cfg = tiny_sgsa_config()
    cfg = replace(cfg, sgsa_write_mode="residual")
    layer = SGSAStateLayer(cfg)

    bsz, seq_len, num_heads, dim = 2, 10, cfg.num_attention_heads, cfg.head_dim
    x = torch.randn(bsz, seq_len, cfg.hidden_size)
    q = torch.randn(bsz, seq_len, num_heads, dim)
    k = torch.randn(bsz, seq_len, num_heads, dim)
    v = torch.randn(bsz, seq_len, num_heads, dim)
    k_hat = torch.randn_like(k)
    v_hat = torch.randn_like(v)
    diagnostics = {
        "concentration": torch.sigmoid(torch.randn(bsz, num_heads, seq_len)),
        "novelty": torch.sigmoid(torch.randn(bsz, num_heads, seq_len)),
    }

    _, _, stats = layer(
        hidden_states=x,
        q=q,
        k=k,
        v=v,
        k_hat=k_hat,
        v_hat=v_hat,
        diagnostics=diagnostics,
    )
    ratio = stats["k_hat_perp_norm_ratio"]
    assert torch.isfinite(ratio).all()
    assert torch.all(ratio >= 0)

    dot = (k_hat * k).sum(dim=-1)
    k_norm_sq = (k * k).sum(dim=-1).clamp_min(1e-6)
    k_hat_perp = k_hat - (dot / k_norm_sq).unsqueeze(-1) * k
    orth = (k_hat_perp * k).sum(dim=-1).abs().mean().item()
    assert orth < 1e-4, f"k_hat_perp not close to orthogonal, got {orth:.6f}"


def check_none_write_path_and_gate_range() -> None:
    cfg_none = replace(tiny_sgsa_config(), sgsa_write_mode="none")
    cfg_direct = replace(tiny_sgsa_config(), sgsa_write_mode="direct")
    layer_none = SGSAStateLayer(cfg_none)
    layer_direct = SGSAStateLayer(cfg_direct)

    layer_direct.load_state_dict(layer_none.state_dict(), strict=False)

    bsz, seq_len, num_heads, dim = 2, 8, cfg_none.num_attention_heads, cfg_none.head_dim
    x = torch.randn(bsz, seq_len, cfg_none.hidden_size)
    q = torch.randn(bsz, seq_len, num_heads, dim)
    k = torch.randn(bsz, seq_len, num_heads, dim)
    v = torch.randn(bsz, seq_len, num_heads, dim)
    k_hat = k.clone()
    v_hat = v.clone()
    diagnostics = {
        "concentration": torch.sigmoid(torch.randn(bsz, num_heads, seq_len)),
        "novelty": torch.sigmoid(torch.randn(bsz, num_heads, seq_len)),
    }

    out_none, _, stats_none = layer_none(x, q, k, v, k_hat, v_hat, diagnostics)
    out_direct, _, stats_direct = layer_direct(x, q, k, v, k_hat, v_hat, diagnostics)

    assert torch.isfinite(out_none).all()
    assert torch.isfinite(out_direct).all()
    assert torch.all((stats_none["alpha"] >= 0) & (stats_none["alpha"] <= cfg_none.alpha_max + 1e-6))
    assert torch.all((stats_direct["alpha"] >= 0) & (stats_direct["alpha"] <= cfg_direct.alpha_max + 1e-6))


def check_block_mode_causality() -> None:
    cfg = replace(tiny_sgsa_config(), retrieval_mode="block_token", top_blocks=2, block_size=4, top_k=5)
    retriever = SparseRetriever(cfg)
    bsz, seq_len, num_heads, dim = 1, 16, cfg.num_attention_heads, cfg.head_dim
    q = torch.randn(bsz, seq_len, num_heads, dim)
    k = torch.randn(bsz, seq_len, num_heads, dim)
    v = torch.randn(bsz, seq_len, num_heads, dim)

    out = retriever(q=q, k=k, v=v, causal=True)
    indices = out["indices"]
    assert indices is not None
    query_positions = torch.arange(seq_len).view(1, 1, seq_len, 1)
    assert torch.all(indices <= query_positions), "block-token refinement selected future token"


def main() -> None:
    check_causal_topk_and_shapes()
    check_residual_write_orthogonality()
    check_none_write_path_and_gate_range()
    check_block_mode_causality()
    print("SGSA sanity checks passed.")


if __name__ == "__main__":
    main()
