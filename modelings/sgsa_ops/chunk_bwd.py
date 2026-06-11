from __future__ import annotations

from typing import Optional, Tuple

import torch

from .chunk_fwd import chunk_sgsa_fwd

Tensor = torch.Tensor


def chunk_sgsa_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    gamma: Tensor,
    sparse_k: Optional[Tensor],
    sparse_v: Optional[Tensor],
    alpha: Optional[Tensor],
    initial_state: Optional[Tensor],
    do: Tensor,
    d_final_state: Optional[Tensor],
    chunk_size: int = 64,
) -> Tuple[Optional[Tensor], ...]:
    """Backward for SGSA chunk op.

    This is intentionally split from `chunk.py` to match FLA's operator layout.
    The current implementation recomputes forward under autograd. It is correct
    and keeps the public operator boundary stable while the Triton backward
    kernel is still pending.
    """
    inputs = [q, k, v, beta, gamma]
    optional_inputs = [sparse_k, sparse_v, alpha, initial_state]
    detached = [x.detach().requires_grad_(True) for x in inputs]
    detached_optional = [
        None if x is None else x.detach().requires_grad_(True)
        for x in optional_inputs
    ]
    q_, k_, v_, beta_, gamma_ = detached
    sparse_k_, sparse_v_, alpha_, initial_state_ = detached_optional

    with torch.enable_grad():
        out, final_state = chunk_sgsa_fwd(
            q=q_,
            k=k_,
            v=v_,
            beta=beta_,
            gamma=gamma_,
            sparse_k=sparse_k_,
            sparse_v=sparse_v_,
            alpha=alpha_,
            initial_state=initial_state_,
            chunk_size=chunk_size,
            output_final_state=True,
            use_triton=False,
        )
        objective = (out * do).sum()
        if d_final_state is not None and final_state is not None:
            objective = objective + (final_state * d_final_state).sum()

    grad_inputs = detached + [x for x in detached_optional if x is not None]
    grads = torch.autograd.grad(objective, grad_inputs, allow_unused=True)

    grad_iter = iter(grads)
    dq = next(grad_iter)
    dk = next(grad_iter)
    dv = next(grad_iter)
    dbeta = next(grad_iter)
    dgamma = next(grad_iter)

    dsparse_k = next(grad_iter) if sparse_k is not None else None
    dsparse_v = next(grad_iter) if sparse_v is not None else None
    dalpha = next(grad_iter) if alpha is not None else None
    dinitial_state = next(grad_iter) if initial_state is not None else None

    return dq, dk, dv, dbeta, dgamma, dsparse_k, dsparse_v, dalpha, dinitial_state
