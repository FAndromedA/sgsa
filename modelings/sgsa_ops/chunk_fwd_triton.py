from __future__ import annotations

from typing import Optional

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
