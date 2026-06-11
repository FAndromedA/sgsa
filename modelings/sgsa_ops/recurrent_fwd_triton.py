from __future__ import annotations

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None

Tensor = torch.Tensor


if triton is not None:

    @triton.heuristics({
        "HAS_SPARSE": lambda args: args["sparse_k"] is not None,
        "HAS_INITIAL_STATE": lambda args: args["initial_state"] is not None,
        "STORE_FINAL_STATE": lambda args: args["final_state"] is not None,
    })
    @triton.jit
    def _recurrent_sgsa_fwd_kernel(
        q,
        k,
        v,
        beta,
        gamma,
        sparse_k,
        sparse_v,
        alpha,
        out,
        initial_state,
        final_state,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BD: tl.constexpr,
        HAS_SPARSE: tl.constexpr,
        HAS_INITIAL_STATE: tl.constexpr,
        STORE_FINAL_STATE: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        i_b = pid_bh // H
        i_h = pid_bh % H
        offs_i = tl.arange(0, BD)
        offs_j = tl.arange(0, BD)
        mask_i = offs_i < D
        mask_j = offs_j < D
        mask_state = mask_i[:, None] & mask_j[None, :]

        state = tl.zeros((BD, BD), dtype=tl.float32)
        if HAS_INITIAL_STATE:
            init_ptrs = initial_state + ((i_b * H + i_h) * D * D + offs_i[:, None] * D + offs_j[None, :])
            state += tl.load(init_ptrs, mask=mask_state, other=0.0).to(tl.float32)

        base = (i_b * T * H + i_h) * D
        scalar_base = i_b * T * H + i_h
        for t in range(0, T):
            q_t = tl.load(q + base + t * H * D + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            k_t = tl.load(k + base + t * H * D + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            v_t = tl.load(v + base + t * H * D + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            beta_t = tl.load(beta + scalar_base + t * H).to(tl.float32)
            gamma_t = tl.load(gamma + scalar_base + t * H).to(tl.float32)

            base_state = state * gamma_t
            pred_local = tl.sum(base_state * k_t[:, None], axis=0)
            delta_local = (v_t - pred_local) * beta_t
            next_state = base_state + k_t[:, None] * delta_local[None, :]

            if HAS_SPARSE:
                sk_t = tl.load(sparse_k + base + t * H * D + offs_i, mask=mask_i, other=0.0).to(tl.float32)
                sv_t = tl.load(sparse_v + base + t * H * D + offs_j, mask=mask_j, other=0.0).to(tl.float32)
                alpha_t = tl.load(alpha + scalar_base + t * H).to(tl.float32)
                pred_sparse = tl.sum(base_state * sk_t[:, None], axis=0)
                delta_sparse = (sv_t - pred_sparse) * alpha_t
                next_state += sk_t[:, None] * delta_sparse[None, :]

            o_t = tl.sum(next_state * q_t[:, None], axis=0)
            tl.store(out + base + t * H * D + offs_j, o_t, mask=mask_j)
            state = next_state

        if STORE_FINAL_STATE:
            final_ptrs = final_state + ((i_b * H + i_h) * D * D + offs_i[:, None] * D + offs_j[None, :])
            tl.store(final_ptrs, state, mask=mask_state)


def recurrent_sgsa_fwd_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    gamma: Tensor,
    sparse_k: Optional[Tensor] = None,
    sparse_v: Optional[Tensor] = None,
    alpha: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
) -> Optional[Tuple[Tensor, Optional[Tensor]]]:
    """Triton recurrent SGSA forward, modeled after FLA fused recurrent GDN."""
    if triton is None or not q.is_cuda:
        return None
    if q.shape[-1] > 128:
        # Keep the first kernel conservative; larger D should use chunk mode.
        return None
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    gamma = gamma.contiguous()
    has_sparse = sparse_k is not None and sparse_v is not None and alpha is not None
    if has_sparse:
        sparse_k = sparse_k.contiguous()
        sparse_v = sparse_v.contiguous()
        alpha = alpha.contiguous()
    bsz, seq_len, num_heads, dim = q.shape
    block_dim = triton.next_power_of_2(dim)
    out = torch.empty_like(v)
    final_state = (
        torch.empty(bsz, num_heads, dim, dim, device=q.device, dtype=torch.float32)
        if output_final_state
        else None
    )
    _recurrent_sgsa_fwd_kernel[(bsz * num_heads,)](
        q=q,
        k=k,
        v=v,
        beta=beta,
        gamma=gamma,
        sparse_k=sparse_k,
        sparse_v=sparse_v,
        alpha=alpha,
        out=out,
        initial_state=initial_state,
        final_state=final_state,
        T=seq_len,
        H=num_heads,
        D=dim,
        BD=block_dim,
        num_warps=1,
        num_stages=3,
    )
    return out, final_state
