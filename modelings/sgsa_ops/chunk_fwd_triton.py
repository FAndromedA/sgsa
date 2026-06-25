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

    @triton.jit
    def _sgsa_kkt_transition_kernel(
        kappa,
        lam,
        prefix,
        tau,
        transition,
        B: tl.constexpr,
        H: tl.constexpr,
        M: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_bh = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        bh_base_k = pid_bh * M * D
        bh_base_t = pid_bh * M * M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d = d_start + offs_d
            row_ptrs = kappa + bh_base_k + offs_m[:, None] * D + d[None, :]
            col_ptrs = kappa + bh_base_k + offs_n[:, None] * D + d[None, :]
            row = tl.load(row_ptrs, mask=(offs_m[:, None] < M) & (d[None, :] < D), other=0.0)
            col = tl.load(col_ptrs, mask=(offs_n[:, None] < M) & (d[None, :] < D), other=0.0)
            acc += tl.dot(row, tl.trans(col))

        tau_m = tl.load(tau + offs_m, mask=offs_m < M, other=0)
        tau_n = tl.load(tau + offs_n, mask=offs_n < M, other=0)
        prefix_m = tl.load(prefix + pid_bh * M + offs_m, mask=offs_m < M, other=1.0)
        prefix_n = tl.load(prefix + pid_bh * M + offs_n, mask=offs_n < M, other=1.0)
        lam_n = tl.load(lam + pid_bh * M + offs_n, mask=offs_n < M, other=0.0)

        dep = tau_n[None, :] < tau_m[:, None]
        bounds = (offs_m[:, None] < M) & (offs_n[None, :] < M)
        decay = prefix_m[:, None] / tl.maximum(prefix_n[None, :], 1.0e-6)
        out = acc * decay * lam_n[None, :]
        out = tl.where(dep & bounds, out, 0.0)

        out_ptrs = transition + bh_base_t + offs_m[:, None] * M + offs_n[None, :]
        tl.store(out_ptrs, out, mask=bounds)


    @triton.jit
    def _sgsa_kkt_solve_kernel(
        kappa,
        lam,
        prefix,
        tau,
        inverse,
        M: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < M
        mask_n = offs_n < M

        bh_base_k = pid_bh * M * D
        bh_base_s = pid_bh * M
        bh_base_i = pid_bh * M * M

        gram = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d = d_start + offs_d
            row_ptrs = kappa + bh_base_k + offs_m[:, None] * D + d[None, :]
            col_ptrs = kappa + bh_base_k + offs_n[:, None] * D + d[None, :]
            row = tl.load(row_ptrs, mask=mask_m[:, None] & (d[None, :] < D), other=0.0)
            col = tl.load(col_ptrs, mask=mask_n[:, None] & (d[None, :] < D), other=0.0)
            gram += tl.dot(row, tl.trans(col))

        tau_m = tl.load(tau + offs_m, mask=mask_m, other=0)
        tau_n = tl.load(tau + offs_n, mask=mask_n, other=0)
        prefix_m = tl.load(prefix + bh_base_s + offs_m, mask=mask_m, other=1.0).to(tl.float32)
        prefix_n = tl.load(prefix + bh_base_s + offs_n, mask=mask_n, other=1.0).to(tl.float32)
        lam_n = tl.load(lam + bh_base_s + offs_n, mask=mask_n, other=0.0).to(tl.float32)

        bounds = mask_m[:, None] & mask_n[None, :]
        dep = tau_n[None, :] < tau_m[:, None]
        transition = gram * (prefix_m[:, None] / tl.maximum(prefix_n[None, :], 1.0e-6)) * lam_n[None, :]
        transition = tl.where(dep & bounds, transition, 0.0)

        eye = offs_m[:, None] == offs_n[None, :]
        inv = tl.where(eye & bounds, 1.0, 0.0)
        for i in range(0, BLOCK_M):
            active = i < M
            row_i = tl.sum(tl.where(offs_m[:, None] == i, transition, 0.0), axis=0)
            solved = -tl.sum(row_i[:, None] * inv, axis=0)
            new_row = tl.where(offs_n < i, solved, 0.0)
            new_row = tl.where(offs_n == i, 1.0, new_row)
            inv = tl.where((offs_m[:, None] == i) & active, new_row[None, :], inv)

        inv_ptrs = inverse + bh_base_i + offs_m[:, None] * M + offs_n[None, :]
        tl.store(inv_ptrs, inv, mask=bounds)


    @triton.jit
    def _sgsa_readout_kernel(
        q,
        kappa,
        residual,
        lam,
        prefix,
        prefix_tau,
        tau,
        state,
        out,
        L: tl.constexpr,
        M: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_v = tl.program_id(1)
        pid_bh = tl.program_id(2)

        offs_k = tl.arange(0, BLOCK_K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = offs_v < D

        q_base = (pid_bh * L + pid_t) * D
        kappa_base = pid_bh * M * D
        residual_base = pid_bh * M * D
        scalar_l_base = pid_bh * M
        state_base = pid_bh * D * D
        prefix_token = tl.load(prefix + pid_bh * L + pid_t).to(tl.float32)

        acc = tl.zeros((BLOCK_V,), dtype=tl.float32)

        # Baseline readout: prefix[t] * (state @ q_t).
        for k_start in range(0, D, BLOCK_K):
            k_idx = k_start + offs_k
            q_vals = tl.load(q + q_base + k_idx, mask=k_idx < D, other=0.0).to(tl.float32)
            state_vals = tl.load(
                state + state_base + k_idx[:, None] * D + offs_v[None, :],
                mask=(k_idx[:, None] < D) & mask_v[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(q_vals[:, None] * state_vals, axis=0) * prefix_token

        offs_m = tl.arange(0, BLOCK_M)
        for m_start in range(0, M, BLOCK_M):
            m_idx = m_start + offs_m
            mask_m = m_idx < M
            qk = tl.zeros((BLOCK_M,), dtype=tl.float32)
            for k_start in range(0, D, BLOCK_K):
                k_idx = k_start + offs_k
                q_vals = tl.load(q + q_base + k_idx, mask=k_idx < D, other=0.0).to(tl.float32)
                kappa_vals = tl.load(
                    kappa + kappa_base + m_idx[:, None] * D + k_idx[None, :],
                    mask=mask_m[:, None] & (k_idx[None, :] < D),
                    other=0.0,
                ).to(tl.float32)
                qk += tl.sum(kappa_vals * q_vals[None, :], axis=1)

            tau_m = tl.load(tau + m_idx, mask=mask_m, other=L).to(tl.int32)
            lam_m = tl.load(lam + scalar_l_base + m_idx, mask=mask_m, other=0.0).to(tl.float32)
            prefix_m = tl.load(prefix_tau + scalar_l_base + m_idx, mask=mask_m, other=1.0).to(tl.float32)
            read_mask = tau_m <= pid_t
            weight = qk * lam_m * prefix_token / tl.maximum(prefix_m, 1.0e-6)
            weight = tl.where(read_mask & mask_m, weight, 0.0)
            res_vals = tl.load(
                residual + residual_base + m_idx[:, None] * D + offs_v[None, :],
                mask=mask_m[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(weight[:, None] * res_vals, axis=0)

        tl.store(out + (pid_bh * L + pid_t) * D + offs_v, acc, mask=mask_v)


    @triton.jit
    def _sgsa_final_state_kernel(
        kappa,
        residual,
        lam,
        prefix_tau,
        state,
        final_state,
        M: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        pid_k = tl.program_id(0)
        pid_v = tl.program_id(1)
        pid_bh = tl.program_id(2)

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
        offs_m = tl.arange(0, BLOCK_M)
        mask_k = offs_k < D
        mask_v = offs_v < D

        kappa_base = pid_bh * M * D
        residual_base = pid_bh * M * D
        scalar_l_base = pid_bh * M
        state_base = pid_bh * D * D
        final_decay = tl.load(prefix_tau + scalar_l_base + (M - 1)).to(tl.float32)

        acc = tl.load(
            state + state_base + offs_k[:, None] * D + offs_v[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32) * final_decay

        for m_start in range(0, M, BLOCK_M):
            m_idx = m_start + offs_m
            mask_m = m_idx < M
            lam_m = tl.load(lam + scalar_l_base + m_idx, mask=mask_m, other=0.0).to(tl.float32)
            prefix_m = tl.load(prefix_tau + scalar_l_base + m_idx, mask=mask_m, other=1.0).to(tl.float32)
            tail_weight = lam_m * final_decay / tl.maximum(prefix_m, 1.0e-6)
            kappa_vals = tl.load(
                kappa + kappa_base + m_idx[:, None] * D + offs_k[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            res_vals = tl.load(
                residual + residual_base + m_idx[:, None] * D + offs_v[None, :],
                mask=mask_m[:, None] & mask_v[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(tl.trans(kappa_vals), res_vals * tail_weight[:, None])

        tl.store(
            final_state + state_base + offs_k[:, None] * D + offs_v[None, :],
            acc,
            mask=mask_k[:, None] & mask_v[None, :],
        )


    @triton.jit
    def _sgsa_transition_bwd_kernel(
        kappa,
        lam,
        prefix,
        tau,
        d_transition,
        d_kappa,
        d_lam,
        d_prefix,
        M: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_bh = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < M
        mask_n = offs_n < M

        bh_base_k = pid_bh * M * D
        bh_base_s = pid_bh * M
        bh_base_t = pid_bh * M * M

        gram = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d_start in range(0, D, BLOCK_D):
            d = d_start + offs_d
            row = tl.load(
                kappa + bh_base_k + offs_m[:, None] * D + d[None, :],
                mask=mask_m[:, None] & (d[None, :] < D),
                other=0.0,
            )
            col = tl.load(
                kappa + bh_base_k + offs_n[:, None] * D + d[None, :],
                mask=mask_n[:, None] & (d[None, :] < D),
                other=0.0,
            )
            gram += tl.dot(row, tl.trans(col))

        tau_m = tl.load(tau + offs_m, mask=mask_m, other=0)
        tau_n = tl.load(tau + offs_n, mask=mask_n, other=0)
        prefix_m = tl.load(prefix + bh_base_s + offs_m, mask=mask_m, other=1.0).to(tl.float32)
        prefix_n = tl.load(prefix + bh_base_s + offs_n, mask=mask_n, other=1.0).to(tl.float32)
        lam_n = tl.load(lam + bh_base_s + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        dep = tau_n[None, :] < tau_m[:, None]
        bounds = mask_m[:, None] & mask_n[None, :]
        valid = dep & bounds
        decay = prefix_m[:, None] / tl.maximum(prefix_n[None, :], 1.0e-6)
        d_t = tl.load(
            d_transition + bh_base_t + offs_m[:, None] * M + offs_n[None, :],
            mask=bounds,
            other=0.0,
        ).to(tl.float32)
        d_t = tl.where(valid, d_t, 0.0)

        d_lam_tile = tl.sum(d_t * decay * gram, axis=0)
        tl.atomic_add(d_lam + bh_base_s + offs_n, d_lam_tile, sem="relaxed", mask=mask_n)

        d_prefix_m = tl.sum(d_t * lam_n[None, :] * gram / tl.maximum(prefix_n[None, :], 1.0e-6), axis=1)
        prefix_n_safe = tl.maximum(prefix_n, 1.0e-6)
        prefix_n_grad_mask = prefix_n >= 1.0e-6
        d_prefix_n = -tl.sum(
            d_t * lam_n[None, :] * gram * prefix_m[:, None] / (prefix_n_safe[None, :] * prefix_n_safe[None, :]),
            axis=0,
        )
        d_prefix_n = tl.where(prefix_n_grad_mask, d_prefix_n, 0.0)
        tl.atomic_add(d_prefix + bh_base_s + offs_m, d_prefix_m, sem="relaxed", mask=mask_m)
        tl.atomic_add(d_prefix + bh_base_s + offs_n, d_prefix_n, sem="relaxed", mask=mask_n)

        coeff = d_t * lam_n[None, :] * decay
        for d_start in range(0, D, BLOCK_D):
            d = d_start + offs_d
            row = tl.load(
                kappa + bh_base_k + offs_m[:, None] * D + d[None, :],
                mask=mask_m[:, None] & (d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            col = tl.load(
                kappa + bh_base_k + offs_n[:, None] * D + d[None, :],
                mask=mask_n[:, None] & (d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            d_row = tl.dot(coeff.to(col.dtype), col)
            d_col = tl.dot(tl.trans(coeff).to(row.dtype), row)
            tl.atomic_add(
                d_kappa + bh_base_k + offs_m[:, None] * D + d[None, :],
                d_row,
                sem="relaxed",
                mask=mask_m[:, None] & (d[None, :] < D),
            )
            tl.atomic_add(
                d_kappa + bh_base_k + offs_n[:, None] * D + d[None, :],
                d_col,
                sem="relaxed",
                mask=mask_n[:, None] & (d[None, :] < D),
            )


def build_sgsa_transition_triton(
    kappa_h: Tensor,
    lam_h: Tensor,
    prefix_tau: Tensor,
    tau: Tensor,
    block_m: int = 16,
    block_n: int = 16,
    block_d: int = 32,
) -> Optional[Tensor]:
    """Build SGSA KKT/transition matrix with a Triton kernel.

    This is the SGSA analogue of the first part of FLA's GDN
    `chunk_gated_delta_rule_fwd_kkt_solve_kernel`: it computes the lower
    triangular transition coefficients

        T[i, j] = 1[tau_j < tau_i] * lambda_j
                  * prefix[tau_i] / prefix[tau_j]
                  * dot(kappa_i, kappa_j)

    The current kernel intentionally stops before the forward substitution
    solve. `chunk_fwd.py` applies `torch.linalg.solve_triangular` afterward.
    A later optimized version can fuse this kernel with the triangular solve,
    following FLA's KKT+solve pattern.

    Args:
        kappa_h: [B, H, M, D]
        lam_h: [B, H, M]
        prefix_tau: [B, H, M]
        tau: [M]

    Returns:
        transition: [B, H, M, M] in fp32, or None if Triton/CUDA is unavailable.
    """
    if triton is None or not kappa_h.is_cuda:
        return None
    if not (kappa_h.is_contiguous() and lam_h.is_contiguous() and prefix_tau.is_contiguous()):
        kappa_h = kappa_h.contiguous()
        lam_h = lam_h.contiguous()
        prefix_tau = prefix_tau.contiguous()
    tau = tau.to(device=kappa_h.device, dtype=torch.int32).contiguous()
    bsz, num_heads, virtual_len, dim = kappa_h.shape
    transition = torch.empty(
        bsz,
        num_heads,
        virtual_len,
        virtual_len,
        device=kappa_h.device,
        dtype=torch.float32,
    )
    grid = (
        triton.cdiv(virtual_len, block_m),
        triton.cdiv(virtual_len, block_n),
        bsz * num_heads,
    )
    _sgsa_kkt_transition_kernel[grid](
        kappa_h,
        lam_h,
        prefix_tau,
        tau,
        transition,
        B=bsz,
        H=num_heads,
        M=virtual_len,
        D=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )
    return transition


def build_sgsa_inverse_triton(
    kappa_h: Tensor,
    lam_h: Tensor,
    prefix_tau: Tensor,
    tau: Tensor,
    block_m: int = 32,
    block_d: int = 32,
) -> Optional[Tensor]:
    """Build `(I + T_sgsa)^{-1}` in one Triton KKT+solve kernel.

    This is a conservative first fused path: one program owns the whole
    virtual-sample matrix for one batch/head. It is intended for small chunks
    and falls back for larger chunks, where a GDN-style multi-block merge solve
    is needed.
    """
    if triton is None or not kappa_h.is_cuda:
        return None
    if not (kappa_h.is_contiguous() and lam_h.is_contiguous() and prefix_tau.is_contiguous()):
        kappa_h = kappa_h.contiguous()
        lam_h = lam_h.contiguous()
        prefix_tau = prefix_tau.contiguous()
    bsz, num_heads, virtual_len, dim = kappa_h.shape
    if virtual_len > block_m:
        return None
    tau = tau.to(device=kappa_h.device, dtype=torch.int32).contiguous()
    inverse = torch.empty(
        bsz,
        num_heads,
        virtual_len,
        virtual_len,
        device=kappa_h.device,
        dtype=torch.float32,
    )
    _sgsa_kkt_solve_kernel[(bsz * num_heads,)](
        kappa_h,
        lam_h,
        prefix_tau,
        tau,
        inverse,
        M=virtual_len,
        D=dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return inverse


def sgsa_readout_final_triton(
    q_h: Tensor,
    kappa_h: Tensor,
    residual: Tensor,
    lam_h: Tensor,
    prefix: Tensor,
    prefix_tau: Tensor,
    tau: Tensor,
    state: Tensor,
    block_m: int = 32,
    block_k: int = 32,
    block_v: int = 32,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Compute SGSA chunk readout and final state with Triton.

    Args use head-major chunk layout:
        q_h: [B, H, L, D]
        kappa_h/residual: [B, H, M, D]
        lam_h/prefix_tau: [B, H, M]
        prefix: [B, H, L]
        state: [B, H, D, D]
    """
    if triton is None or not q_h.is_cuda:
        return None
    if not (
        q_h.is_contiguous()
        and kappa_h.is_contiguous()
        and residual.is_contiguous()
        and lam_h.is_contiguous()
        and prefix.is_contiguous()
        and prefix_tau.is_contiguous()
        and state.is_contiguous()
    ):
        q_h = q_h.contiguous()
        kappa_h = kappa_h.contiguous()
        residual = residual.contiguous()
        lam_h = lam_h.contiguous()
        prefix = prefix.contiguous()
        prefix_tau = prefix_tau.contiguous()
        state = state.contiguous()

    bsz, num_heads, length, dim = q_h.shape
    virtual_len = kappa_h.shape[2]
    if dim > 128:
        return None

    block_k = min(block_k, triton.next_power_of_2(dim))
    block_v = min(block_v, triton.next_power_of_2(dim))
    tau = tau.to(device=q_h.device, dtype=torch.int32).contiguous()
    out_h = torch.empty_like(q_h)
    final_state = torch.empty_like(state)

    grid_read = (
        length,
        triton.cdiv(dim, block_v),
        bsz * num_heads,
    )
    _sgsa_readout_kernel[grid_read](
        q_h,
        kappa_h,
        residual,
        lam_h,
        prefix,
        prefix_tau,
        tau,
        state,
        out_h,
        L=length,
        M=virtual_len,
        D=dim,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=4,
        num_stages=3,
    )

    grid_state = (
        triton.cdiv(dim, block_k),
        triton.cdiv(dim, block_v),
        bsz * num_heads,
    )
    _sgsa_final_state_kernel[grid_state](
        kappa_h,
        residual,
        lam_h,
        prefix_tau,
        state,
        final_state,
        M=virtual_len,
        D=dim,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
        num_warps=4,
        num_stages=3,
    )
    return out_h, final_state


def sgsa_transition_bwd_triton(
    kappa_h: Tensor,
    lam_h: Tensor,
    prefix_tau: Tensor,
    tau: Tensor,
    d_transition: Tensor,
    block_m: int = 16,
    block_n: int = 16,
    block_d: int = 32,
) -> Optional[Tuple[Tensor, Tensor, Tensor]]:
    """Backward for the SGSA transition matrix.

    Returns gradients for `kappa_h`, `lam_h`, and `prefix_tau`. This primitive
    is meant to be used after triangular-solve backward has produced
    `d_transition`.
    """
    if triton is None or not kappa_h.is_cuda:
        return None
    if not (
        kappa_h.is_contiguous()
        and lam_h.is_contiguous()
        and prefix_tau.is_contiguous()
        and d_transition.is_contiguous()
    ):
        kappa_h = kappa_h.contiguous()
        lam_h = lam_h.contiguous()
        prefix_tau = prefix_tau.contiguous()
        d_transition = d_transition.contiguous()
    tau = tau.to(device=kappa_h.device, dtype=torch.int32).contiguous()
    bsz, num_heads, virtual_len, dim = kappa_h.shape
    d_kappa = torch.zeros_like(kappa_h, dtype=torch.float32)
    d_lam = torch.zeros_like(lam_h, dtype=torch.float32)
    d_prefix = torch.zeros_like(prefix_tau, dtype=torch.float32)
    grid = (
        triton.cdiv(virtual_len, block_m),
        triton.cdiv(virtual_len, block_n),
        bsz * num_heads,
    )
    _sgsa_transition_bwd_kernel[grid](
        kappa_h,
        lam_h,
        prefix_tau,
        tau,
        d_transition,
        d_kappa,
        d_lam,
        d_prefix,
        M=virtual_len,
        D=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return d_kappa, d_lam, d_prefix
