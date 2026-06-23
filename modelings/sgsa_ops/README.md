# SGSA Ops Implementation Status

This directory contains the project-local SGSA operators. The layout intentionally mirrors FLA's `fla/ops/gated_delta_rule` structure, but the files live under `modelings/sgsa_ops/` and do not modify the vendored `flash-linear-attention` tree.

## Implemented

### Reference Recurrent Op

File: `naive.py`

- `naive_recurrent_sgsa`
- Pure PyTorch recurrent reference.
- Supports local write and sparse write.
- Local and sparse writes both read the same decayed base state for each token.
- Supports optional `initial_state` and optional final state output.

### Chunk Op API

Files:

- `chunk.py`
- `chunk_fwd.py`
- `chunk_bwd.py`

Implemented pieces:

- `chunk_sgsa` public API.
- `ChunkSGSAFunction` autograd wrapper.
- `chunk_sgsa_fwd` chunk forward using the SGSA virtual-sample formulation.
- `chunk_sgsa_bwd` recompute backward using PyTorch autograd.
- CPU fallback and non-Triton path.
- Optional Triton use through `use_triton=True`.

The chunk formulation supports:

- state-only GDN mode;
- direct sparse write;
- residual-subspace sparse write, supplied by the caller as `sparse_k`;
- same-token local/sparse dependency masking through virtual sample timestamps.

### Chunk Triton Forward Piece

File: `chunk_fwd_triton.py`

Implemented:

- `_sgsa_kkt_transition_kernel`
- `build_sgsa_transition_triton`

This kernel computes the SGSA chunk transition matrix:

```text
T[i, j] = 1[tau_j < tau_i]
        * lambda_j
        * prefix[tau_i] / prefix[tau_j]
        * dot(kappa_i, kappa_j)
```

This corresponds to the KKT/transition construction part of FLA's GDN chunk kernel.

### Recurrent Triton Forward

Files:

- `recurrent.py`
- `recurrent_fwd_triton.py`

Implemented:

- `fused_recurrent_sgsa`
- `FusedRecurrentSGSAFunction`
- `_recurrent_sgsa_fwd_kernel`
- `recurrent_sgsa_fwd_triton`

The recurrent Triton forward is modeled after FLA's `fused_recurrent_gated_delta_rule_fwd_kernel`. It updates the state token by token and supports:

- local write;
- sparse write;
- optional initial state;
- optional final state output;
- dense state layout `[B, H, D, D]`.

### Model Integration

File: `modelings/modeling_sgsa.py`

Implemented backend options:

- `linear_backend="python"`: uses `naive_recurrent_sgsa`.
- `linear_backend="torch_chunk"`: uses `chunk_sgsa`.
- `linear_backend="recurrent"`: uses `fused_recurrent_sgsa` with `use_triton=False`.
- `linear_backend="triton_recurrent"`: uses `fused_recurrent_sgsa` with Triton forward when CUDA is available.
- `linear_backend="fla_gdn"`: uses FLA `chunk_gdn` only for `sgsa_write_mode="none"`.
- `linear_backend="auto"`: keeps the previous chunk behavior.

Model metrics now expose:

- `used_fla_gdn`
- `used_torch_chunk`
- `used_recurrent`

## Partially Implemented

### Chunk Triton Forward

The current chunk Triton implementation only computes the KKT/transition matrix. It does **not** yet fuse triangular solve or recompute `w/u` the way FLA's GDN chunk implementation does.

Current flow:

1. Triton builds transition matrix.
2. PyTorch runs `torch.linalg.solve_triangular`.
3. PyTorch computes readout and final state.

Target FLA-style flow:

1. Triton builds KKT/transition.
2. Triton performs forward substitution / triangular solve.
3. Triton or a follow-up kernel recomputes the chunk write representation.
4. Triton computes chunk state/readout or calls a common state/readout kernel.

### Backward

Both chunk and recurrent public autograd wrappers currently use recompute backward:

- rerun PyTorch reference under `torch.enable_grad()`;
- call `torch.autograd.grad`;
- return gradients to the wrapper.

This is correct and useful for development, but it is not a production Triton backward.

## Not Implemented Yet

### Fused Chunk KKT + Solve Triton Kernel

Missing:

- in-register/blockwise forward substitution equivalent to FLA's `chunk_gated_delta_rule_fwd_kkt_solve_kernel`;
- solved inverse/triangular factor output;
- chunk-local write representation equivalent to GDN's `w, u, A`.

### Triton Chunk Readout Kernel

Missing:

- chunk readout kernel for `o_t`;
- final state update kernel;
- support for state carry across chunks in a fully fused path.

### Triton Backward Kernels

Missing:

- recurrent Triton backward;
- chunk Triton backward;
- gradients for q/k/v/beta/gamma/sparse_k/sparse_v/alpha without recompute autograd.

### Variable-length and Context Parallel Support

Missing:

- `cu_seqlens`;
- packed sequence support;
- context parallel support;
- FLA-compatible chunk index preparation.

### Production Kernel Features

Missing:

- autotune configs;
- block size specialization;
- mixed precision policy beyond basic fp32 accumulation;
- state layout variants;
- GQA/GVA-specific optimized layout;
- large head dimension support beyond the conservative recurrent kernel limit.

## Tests

File: `tests/test_sgsa_core.py`

Covered:

- chunk SGSA matches naive recurrent without sparse write;
- chunk SGSA matches naive recurrent with sparse write;
- recurrent wrapper matches naive recurrent and supports backward;
- CUDA-only transition Triton test, skipped when CUDA is unavailable;
- CUDA-only recurrent Triton test, skipped when CUDA is unavailable.

Current environment note:

- `pytest` may not be installed. Tests can be invoked directly from Python, as done during development.

## Next Implementation Steps

1. Fuse chunk transition and triangular solve in Triton.
2. Add a Triton chunk readout/final-state kernel.
3. Replace recompute backward with explicit recurrent backward first, since recurrent state order is easier to reason about than chunked scan.
4. Add packed sequence support only after dense fixed-length kernels are numerically stable.
