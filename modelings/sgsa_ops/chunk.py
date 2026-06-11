from __future__ import annotations

from typing import Optional, Tuple

import torch

from .chunk_bwd import chunk_sgsa_bwd
from .chunk_fwd import chunk_sgsa_fwd

Tensor = torch.Tensor


class ChunkSGSAFunction(torch.autograd.Function):
    """FLA-style autograd wrapper for the SGSA chunk operator."""

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        gamma: Tensor,
        sparse_k: Tensor,
        sparse_v: Tensor,
        alpha: Tensor,
        initial_state: Tensor,
        has_sparse: bool,
        has_initial_state: bool,
        chunk_size: int,
        output_final_state: bool,
        use_triton: bool,
    ):
        sparse_k_arg = sparse_k if has_sparse else None
        sparse_v_arg = sparse_v if has_sparse else None
        alpha_arg = alpha if has_sparse else None
        initial_state_arg = initial_state if has_initial_state else None
        out, final_state = chunk_sgsa_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            gamma=gamma,
            sparse_k=sparse_k_arg,
            sparse_v=sparse_v_arg,
            alpha=alpha_arg,
            initial_state=initial_state_arg,
            chunk_size=chunk_size,
            output_final_state=output_final_state,
            use_triton=use_triton,
        )
        ctx.save_for_backward(q, k, v, beta, gamma, sparse_k, sparse_v, alpha, initial_state)
        ctx.has_sparse = has_sparse
        ctx.has_initial_state = has_initial_state
        ctx.chunk_size = chunk_size
        return out, final_state

    @staticmethod
    def backward(ctx, do: Tensor, d_final_state: Optional[Tensor]):
        q, k, v, beta, gamma, sparse_k, sparse_v, alpha, initial_state = ctx.saved_tensors
        sparse_k_arg = sparse_k if ctx.has_sparse else None
        sparse_v_arg = sparse_v if ctx.has_sparse else None
        alpha_arg = alpha if ctx.has_sparse else None
        initial_state_arg = initial_state if ctx.has_initial_state else None
        grads = chunk_sgsa_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            gamma=gamma,
            sparse_k=sparse_k_arg,
            sparse_v=sparse_v_arg,
            alpha=alpha_arg,
            initial_state=initial_state_arg,
            do=do,
            d_final_state=d_final_state,
            chunk_size=ctx.chunk_size,
        )
        dq, dk, dv, dbeta, dgamma, dsparse_k, dsparse_v, dalpha, dinitial_state = grads
        if not ctx.has_sparse:
            dsparse_k = torch.zeros_like(sparse_k)
            dsparse_v = torch.zeros_like(sparse_v)
            dalpha = torch.zeros_like(alpha)
        if not ctx.has_initial_state:
            dinitial_state = torch.zeros_like(initial_state)
        return (
            dq,
            dk,
            dv,
            dbeta,
            dgamma,
            dsparse_k,
            dsparse_v,
            dalpha,
            dinitial_state,
            None,
            None,
            None,
            None,
            None,
        )


def chunk_sgsa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    gamma: Tensor,
    sparse_k: Optional[Tensor] = None,
    sparse_v: Optional[Tensor] = None,
    alpha: Optional[Tensor] = None,
    initial_state: Optional[Tensor] = None,
    chunk_size: int = 64,
    output_final_state: bool = False,
    use_triton: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Chunked Sparse-Guided State Attention core.

    Args follow `[B, T, H, D]` layout for q/k/v and sparse tensors, and
    `[B, T, H]` layout for beta/gamma/alpha.
    """
    has_sparse = sparse_k is not None and sparse_v is not None and alpha is not None
    has_initial_state = initial_state is not None
    sparse_k_arg = sparse_k if sparse_k is not None else torch.empty(0, device=q.device, dtype=q.dtype)
    sparse_v_arg = sparse_v if sparse_v is not None else torch.empty(0, device=q.device, dtype=q.dtype)
    alpha_arg = alpha if alpha is not None else torch.empty(0, device=q.device, dtype=beta.dtype)
    initial_state_arg = (
        initial_state
        if initial_state is not None
        else torch.empty(0, device=q.device, dtype=q.dtype)
    )
    return ChunkSGSAFunction.apply(
        q,
        k,
        v,
        beta,
        gamma,
        sparse_k_arg,
        sparse_v_arg,
        alpha_arg,
        initial_state_arg,
        has_sparse,
        has_initial_state,
        chunk_size,
        output_final_state,
        use_triton,
    )
