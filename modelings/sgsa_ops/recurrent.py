from __future__ import annotations

from typing import Optional, Tuple

import torch

from .naive import naive_recurrent_sgsa
from .recurrent_fwd_triton import recurrent_sgsa_fwd_triton

Tensor = torch.Tensor


class FusedRecurrentSGSAFunction(torch.autograd.Function):
    """FLA-style recurrent SGSA autograd wrapper.

    Forward uses a Triton recurrent kernel when available. Backward recomputes
    the reference PyTorch recurrent path under autograd, keeping gradients
    correct while a hand-written Triton backward is pending.
    """

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
        output_final_state: bool,
        use_triton: bool,
    ):
        sparse_k_arg = sparse_k if has_sparse else None
        sparse_v_arg = sparse_v if has_sparse else None
        alpha_arg = alpha if has_sparse else None
        initial_state_arg = initial_state if has_initial_state else None
        result = (
            recurrent_sgsa_fwd_triton(
                q=q,
                k=k,
                v=v,
                beta=beta,
                gamma=gamma,
                sparse_k=sparse_k_arg,
                sparse_v=sparse_v_arg,
                alpha=alpha_arg,
                initial_state=initial_state_arg,
                output_final_state=output_final_state,
            )
            if use_triton
            else None
        )
        if result is None:
            result = naive_recurrent_sgsa(
                q=q,
                k=k,
                v=v,
                beta=beta,
                gamma=gamma,
                sparse_k=sparse_k_arg,
                sparse_v=sparse_v_arg,
                alpha=alpha_arg,
                initial_state=initial_state_arg,
                output_final_state=output_final_state,
            )
        out, final_state = result
        ctx.save_for_backward(q, k, v, beta, gamma, sparse_k, sparse_v, alpha, initial_state)
        ctx.has_sparse = has_sparse
        ctx.has_initial_state = has_initial_state
        return out, final_state

    @staticmethod
    def backward(ctx, do: Tensor, d_final_state: Optional[Tensor]):
        q, k, v, beta, gamma, sparse_k, sparse_v, alpha, initial_state = ctx.saved_tensors
        sparse_k_arg = sparse_k if ctx.has_sparse else None
        sparse_v_arg = sparse_v if ctx.has_sparse else None
        alpha_arg = alpha if ctx.has_sparse else None
        initial_state_arg = initial_state if ctx.has_initial_state else None

        detached_inputs = [x.detach().requires_grad_(True) for x in (q, k, v, beta, gamma)]
        q_, k_, v_, beta_, gamma_ = detached_inputs
        detached_optional = [
            None if x is None else x.detach().requires_grad_(True)
            for x in (sparse_k_arg, sparse_v_arg, alpha_arg, initial_state_arg)
        ]
        sparse_k_, sparse_v_, alpha_, initial_state_ = detached_optional
        with torch.enable_grad():
            out, final_state = naive_recurrent_sgsa(
                q=q_,
                k=k_,
                v=v_,
                beta=beta_,
                gamma=gamma_,
                sparse_k=sparse_k_,
                sparse_v=sparse_v_,
                alpha=alpha_,
                initial_state=initial_state_,
                output_final_state=True,
            )
            objective = (out * do).sum()
            if d_final_state is not None and final_state is not None:
                objective = objective + (final_state * d_final_state).sum()

        grad_inputs = detached_inputs + [x for x in detached_optional if x is not None]
        grads = torch.autograd.grad(objective, grad_inputs, allow_unused=True)
        grad_iter = iter(grads)
        dq = next(grad_iter)
        dk = next(grad_iter)
        dv = next(grad_iter)
        dbeta = next(grad_iter)
        dgamma = next(grad_iter)
        dsparse_k = next(grad_iter) if ctx.has_sparse else torch.zeros_like(sparse_k)
        dsparse_v = next(grad_iter) if ctx.has_sparse else torch.zeros_like(sparse_v)
        dalpha = next(grad_iter) if ctx.has_sparse else torch.zeros_like(alpha)
        dinitial_state = next(grad_iter) if ctx.has_initial_state else torch.zeros_like(initial_state)
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
        )


def fused_recurrent_sgsa(
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
    use_triton: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
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
    return FusedRecurrentSGSAFunction.apply(
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
        output_final_state,
        use_triton,
    )
