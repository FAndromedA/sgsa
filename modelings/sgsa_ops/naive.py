from __future__ import annotations

from typing import Optional, Tuple

import torch

Tensor = torch.Tensor


def naive_recurrent_sgsa(
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
) -> Tuple[Tensor, Optional[Tensor]]:
    """Reference recurrent SGSA/GDN core.

    Shapes:
        q, k, v: [B, T, H, D]
        beta, gamma: [B, T, H]
        sparse_k, sparse_v: [B, T, H, D]
        alpha: [B, T, H]
        state: [B, H, D, D]

    Decay is applied before computing token residuals, matching FLA's GDN
    convention. Local and sparse writes both read the same decayed base state.
    """
    bsz, seq_len, num_heads, dim = q.shape
    state = q.new_zeros(bsz, num_heads, dim, dim) if initial_state is None else initial_state
    outputs = []
    has_sparse = sparse_k is not None and sparse_v is not None and alpha is not None

    for t in range(seq_len):
        base_state = gamma[:, t].unsqueeze(-1).unsqueeze(-1) * state

        k_t = k[:, t]
        v_t = v[:, t]
        predicted_local = torch.einsum("bhde,bhd->bhe", base_state, k_t)
        delta_local = (v_t - predicted_local) * beta[:, t].unsqueeze(-1)
        state = base_state + torch.einsum("bhd,bhe->bhde", k_t, delta_local)

        if has_sparse:
            sparse_k_t = sparse_k[:, t]
            sparse_v_t = sparse_v[:, t]
            predicted_sparse = torch.einsum("bhde,bhd->bhe", base_state, sparse_k_t)
            delta_sparse = (sparse_v_t - predicted_sparse) * alpha[:, t].unsqueeze(-1)
            state = state + torch.einsum("bhd,bhe->bhde", sparse_k_t, delta_sparse)

        outputs.append(torch.einsum("bhde,bhd->bhe", state, q[:, t]))

    return torch.stack(outputs, dim=1), state if output_final_state else None
