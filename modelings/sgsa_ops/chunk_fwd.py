from __future__ import annotations

from typing import Optional, Tuple

import torch

from .chunk_fwd_triton import build_sgsa_inverse_triton, build_sgsa_transition_triton, sgsa_readout_final_triton

Tensor = torch.Tensor


def chunk_sgsa_fwd(
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
    """Chunk SGSA forward with the same high-level structure as FLA GDN.

    This is a torch implementation of the forward operator boundary. It mirrors
    the GDN chunk algorithm: build virtual write samples, solve a triangular
    residual system per chunk, then compute readout and final state. The file is
    intentionally separated so a Triton kernel can replace this body later.
    """
    bsz, seq_len, num_heads, dim = q.shape
    state = q.new_zeros(bsz, num_heads, dim, dim) if initial_state is None else initial_state
    outputs = []
    has_sparse = sparse_k is not None and sparse_v is not None and alpha is not None
    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        length = end - start
        q_c = q[:, start:end].to(compute_dtype)
        k_c = k[:, start:end].to(compute_dtype)
        v_c = v[:, start:end].to(compute_dtype)
        beta_c = beta[:, start:end].to(compute_dtype)
        gamma_c = gamma[:, start:end].to(compute_dtype).clamp_min(1e-6)
        state_c = state.to(compute_dtype)

        if has_sparse:
            sparse_k_c = sparse_k[:, start:end].to(compute_dtype)
            sparse_v_c = sparse_v[:, start:end].to(compute_dtype)
            alpha_c = alpha[:, start:end].to(compute_dtype)
            kappa = torch.stack((k_c, sparse_k_c), dim=2).reshape(bsz, length * 2, num_heads, dim)
            nu = torch.stack((v_c, sparse_v_c), dim=2).reshape(bsz, length * 2, num_heads, dim)
            lam = torch.stack((beta_c, alpha_c), dim=2).reshape(bsz, length * 2, num_heads)
            tau = torch.arange(length, device=q.device).repeat_interleave(2)
        else:
            kappa = k_c
            nu = v_c
            lam = beta_c
            tau = torch.arange(length, device=q.device)

        kappa_h = kappa.transpose(1, 2).contiguous()
        nu_h = nu.transpose(1, 2).contiguous()
        lam_h = lam.transpose(1, 2).contiguous()
        q_h = q_c.transpose(1, 2).contiguous()
        gamma_h = gamma_c.transpose(1, 2).contiguous()
        virtual_len = kappa_h.shape[2]

        prefix = torch.cumprod(gamma_h, dim=-1)
        prefix_tau = prefix.index_select(dim=-1, index=tau)

        pred0 = torch.einsum("bhde,bhmd->bhme", state_c, kappa_h)
        rhs = nu_h - prefix_tau.unsqueeze(-1) * pred0

        inverse = build_sgsa_inverse_triton(kappa_h, lam_h, prefix_tau, tau) if use_triton else None
        if inverse is not None:
            residual = torch.matmul(inverse.to(rhs.dtype), rhs)
        else:
            transition = build_sgsa_transition_triton(kappa_h, lam_h, prefix_tau, tau) if use_triton else None
            if transition is None:
                gram = torch.einsum("bhid,bhjd->bhij", kappa_h, kappa_h)
                tau_i = tau.view(1, 1, virtual_len, 1)
                tau_j = tau.view(1, 1, 1, virtual_len)
                dep_mask = tau_j < tau_i
                decay = prefix_tau.unsqueeze(-1) / prefix_tau.unsqueeze(-2).clamp_min(1e-6)
                transition = dep_mask.to(compute_dtype) * lam_h.unsqueeze(-2) * decay * gram
            eye = torch.eye(virtual_len, dtype=compute_dtype, device=q.device).view(1, 1, virtual_len, virtual_len)
            residual = torch.linalg.solve_triangular(transition + eye, rhs, upper=False)

        triton_readout = (
            sgsa_readout_final_triton(
                q_h=q_h,
                kappa_h=kappa_h,
                residual=residual,
                lam_h=lam_h,
                prefix=prefix,
                prefix_tau=prefix_tau,
                tau=tau,
                state=state_c,
            )
            if use_triton
            else None
        )
        if triton_readout is not None:
            out_h, state_h = triton_readout
            outputs.append(out_h.transpose(1, 2).contiguous().to(q.dtype))
        else:
            final_decay = prefix[..., -1]
            tail_decay = final_decay.unsqueeze(-1) / prefix_tau.clamp_min(1e-6)
            weighted_residual = residual * (lam_h * tail_decay).unsqueeze(-1)
            state_h = (
                final_decay.unsqueeze(-1).unsqueeze(-1) * state_c
                + torch.einsum("bhmd,bhme->bhde", kappa_h, weighted_residual)
            )

            read_terms = []
            for local_t in range(length):
                token_decay = prefix[..., local_t]
                read_mask = (tau <= local_t).to(compute_dtype)
                read_tail = token_decay.unsqueeze(-1) / prefix_tau.clamp_min(1e-6)
                read_weight = lam_h * read_tail * read_mask.view(1, 1, virtual_len)
                qk = torch.einsum("bhd,bhmd->bhm", q_h[:, :, local_t], kappa_h)
                correction = torch.einsum("bhm,bhm,bhme->bhe", qk, read_weight, residual)
                base = token_decay.unsqueeze(-1) * torch.einsum("bhde,bhd->bhe", state_c, q_c[:, local_t])
                read_terms.append(base + correction)

            outputs.append(torch.stack(read_terms, dim=1).to(q.dtype))
        state = state_h.to(q.dtype)

    return torch.cat(outputs, dim=1), state if output_final_state else None
