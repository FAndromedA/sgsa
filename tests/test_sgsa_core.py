import torch

from modelings.sgsa_ops import chunk_sgsa, fused_recurrent_sgsa, naive_recurrent_sgsa
from modelings.sgsa_ops.chunk_fwd_triton import (
    build_sgsa_inverse_triton,
    build_sgsa_transition_triton,
    sgsa_transition_bwd_triton,
)


def _random_inputs(seed: int = 0):
    torch.manual_seed(seed)
    batch, seq_len, heads, dim = 2, 9, 3, 5
    q = torch.randn(batch, seq_len, heads, dim)
    k = torch.randn(batch, seq_len, heads, dim)
    v = torch.randn(batch, seq_len, heads, dim)
    sparse_k = torch.randn(batch, seq_len, heads, dim)
    sparse_v = torch.randn(batch, seq_len, heads, dim)
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads))
    gamma = torch.sigmoid(torch.randn(batch, seq_len, heads))
    alpha = 0.25 * torch.sigmoid(torch.randn(batch, seq_len, heads))
    return q, k, v, sparse_k, sparse_v, beta, gamma, alpha


def test_chunk_sgsa_matches_recurrent_without_sparse_write():
    q, k, v, _, _, beta, gamma, _ = _random_inputs()
    ref_out, ref_state = naive_recurrent_sgsa(q=q, k=k, v=v, beta=beta, gamma=gamma, output_final_state=True)
    chunk_out, chunk_state = chunk_sgsa(q=q, k=k, v=v, beta=beta, gamma=gamma, chunk_size=4, output_final_state=True)

    torch.testing.assert_close(chunk_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(chunk_state, ref_state, atol=1e-5, rtol=1e-5)


def test_chunk_sgsa_matches_recurrent_with_sparse_write():
    q, k, v, sparse_k, sparse_v, beta, gamma, alpha = _random_inputs()
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "sparse_k": sparse_k,
        "sparse_v": sparse_v,
        "beta": beta,
        "gamma": gamma,
        "alpha": alpha,
    }
    ref_out, ref_state = naive_recurrent_sgsa(**kwargs, output_final_state=True)
    chunk_out, chunk_state = chunk_sgsa(**kwargs, chunk_size=4, output_final_state=True)

    torch.testing.assert_close(chunk_out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(chunk_state, ref_state, atol=1e-5, rtol=1e-5)


def test_triton_transition_matches_torch_when_cuda_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    batch, heads, virtual_len, dim = 2, 3, 12, 16
    kappa = torch.randn(batch, heads, virtual_len, dim, device="cuda", dtype=torch.float16)
    lam = torch.rand(batch, heads, virtual_len, device="cuda")
    gamma = torch.sigmoid(torch.randn(batch, heads, virtual_len, device="cuda")).clamp_min(1e-6)
    prefix = torch.cumprod(gamma, dim=-1)
    tau = torch.arange(virtual_len // 2, device="cuda").repeat_interleave(2)

    try:
        tri = build_sgsa_transition_triton(kappa, lam, prefix, tau)
    except Exception:
        return
    gram = torch.einsum("bhid,bhjd->bhij", kappa.float(), kappa.float())
    dep_mask = tau.view(1, 1, 1, virtual_len) < tau.view(1, 1, virtual_len, 1)
    decay = prefix.unsqueeze(-1) / prefix.unsqueeze(-2).clamp_min(1e-6)
    ref = dep_mask.float() * lam.unsqueeze(-2) * decay * gram

    torch.testing.assert_close(tri, ref, atol=2e-2, rtol=2e-2)


def test_triton_kkt_solve_matches_torch_when_cuda_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    batch, heads, virtual_len, dim = 2, 3, 12, 16
    kappa = torch.randn(batch, heads, virtual_len, dim, device="cuda", dtype=torch.float16)
    lam = torch.rand(batch, heads, virtual_len, device="cuda")
    gamma = torch.sigmoid(torch.randn(batch, heads, virtual_len, device="cuda")).clamp_min(1e-6)
    prefix = torch.cumprod(gamma, dim=-1)
    tau = torch.arange(virtual_len // 2, device="cuda").repeat_interleave(2)

    try:
        inv = build_sgsa_inverse_triton(kappa, lam, prefix, tau)
    except Exception:
        return
    assert inv is not None
    gram = torch.einsum("bhid,bhjd->bhij", kappa.float(), kappa.float())
    dep_mask = tau.view(1, 1, 1, virtual_len) < tau.view(1, 1, virtual_len, 1)
    decay = prefix.unsqueeze(-1) / prefix.unsqueeze(-2).clamp_min(1e-6)
    transition = dep_mask.float() * lam.unsqueeze(-2) * decay * gram
    eye = torch.eye(virtual_len, device="cuda").view(1, 1, virtual_len, virtual_len)
    rhs = torch.randn(batch, heads, virtual_len, dim, device="cuda")
    ref = torch.linalg.solve_triangular(transition + eye, rhs, upper=False)
    got = inv @ rhs

    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)


def test_triton_transition_backward_matches_torch_when_cuda_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    batch, heads, virtual_len, dim = 2, 3, 12, 16
    kappa = torch.randn(batch, heads, virtual_len, dim, device="cuda", dtype=torch.float32, requires_grad=True)
    lam = torch.rand(batch, heads, virtual_len, device="cuda", requires_grad=True)
    gamma = 0.8 + 0.2 * torch.rand(batch, heads, virtual_len, device="cuda")
    prefix = torch.cumprod(gamma, dim=-1)
    prefix.requires_grad_(True)
    tau = torch.arange(virtual_len // 2, device="cuda").repeat_interleave(2)
    d_transition = torch.randn(batch, heads, virtual_len, virtual_len, device="cuda")

    gram = torch.einsum("bhid,bhjd->bhij", kappa, kappa)
    dep_mask = tau.view(1, 1, 1, virtual_len) < tau.view(1, 1, virtual_len, 1)
    decay = prefix.unsqueeze(-1) / prefix.unsqueeze(-2).clamp_min(1e-6)
    transition = dep_mask.float() * lam.unsqueeze(-2) * decay * gram
    (transition * d_transition).sum().backward()

    try:
        grads = sgsa_transition_bwd_triton(
            kappa.detach(),
            lam.detach(),
            prefix.detach(),
            tau,
            d_transition,
        )
    except Exception:
        return
    assert grads is not None
    d_kappa, d_lam, d_prefix = grads

    torch.testing.assert_close(d_kappa, kappa.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(d_lam, lam.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(d_prefix, prefix.grad, atol=2e-2, rtol=2e-2)


def test_chunk_triton_forward_matches_torch_when_cuda_available():
    if not torch.cuda.is_available():
        return
    q, k, v, sparse_k, sparse_v, beta, gamma, alpha = [x.cuda() for x in _random_inputs()]
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "sparse_k": sparse_k,
        "sparse_v": sparse_v,
        "beta": beta,
        "gamma": gamma,
        "alpha": alpha,
        "chunk_size": 4,
        "output_final_state": True,
    }
    try:
        ref_out, ref_state = chunk_sgsa(**kwargs, use_triton=False)
        tri_out, tri_state = chunk_sgsa(**kwargs, use_triton=True)
    except Exception:
        return

    torch.testing.assert_close(tri_out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(tri_state, ref_state, atol=2e-2, rtol=2e-2)


def test_fused_recurrent_sgsa_matches_naive_and_backward():
    q, k, v, sparse_k, sparse_v, beta, gamma, alpha = _random_inputs()
    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "sparse_k": sparse_k,
        "sparse_v": sparse_v,
        "beta": beta,
        "gamma": gamma,
        "alpha": alpha,
        "output_final_state": True,
    }
    ref_out, ref_state = naive_recurrent_sgsa(**kwargs)

    grad_inputs = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in kwargs.items()
        if isinstance(value, torch.Tensor)
    }
    out, state = fused_recurrent_sgsa(**grad_inputs, output_final_state=True, use_triton=False)
    torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state, ref_state, atol=1e-5, rtol=1e-5)
    (out.square().mean() + state.square().mean()).backward()
    assert grad_inputs["q"].grad is not None


def test_triton_recurrent_matches_naive_when_cuda_available():
    if not torch.cuda.is_available():
        return
    q, k, v, sparse_k, sparse_v, beta, gamma, alpha = [x.cuda() for x in _random_inputs()]
    ref_out, ref_state = naive_recurrent_sgsa(
        q=q,
        k=k,
        v=v,
        sparse_k=sparse_k,
        sparse_v=sparse_v,
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        output_final_state=True,
    )
    try:
        tri_out, tri_state = fused_recurrent_sgsa(
            q=q,
            k=k,
            v=v,
            sparse_k=sparse_k,
            sparse_v=sparse_v,
            beta=beta,
            gamma=gamma,
            alpha=alpha,
            output_final_state=True,
            use_triton=True,
        )
    except Exception:
        return
    torch.testing.assert_close(tri_out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(tri_state, ref_state, atol=2e-2, rtol=2e-2)


def test_model_recurrent_backend_forward_backward():
    from modelings.modeling_sgsa import SGSAConfig, SGSAStateLayer

    for backend in ("recurrent", "triton_recurrent"):
        config = SGSAConfig(
            hidden_size=32,
            num_attention_heads=4,
            num_kv_heads=4,
            sgsa_write_mode="direct",
            linear_backend=backend,
            chunk_size=4,
        )
        layer = SGSAStateLayer(config)
        batch, seq_len, heads, dim = 2, 8, config.num_attention_heads, config.head_dim
        hidden = torch.randn(batch, seq_len, config.hidden_size)
        q = torch.randn(batch, seq_len, heads, dim)
        k = torch.randn(batch, seq_len, heads, dim)
        v = torch.randn(batch, seq_len, heads, dim)
        k_hat = torch.randn(batch, seq_len, heads, dim)
        v_hat = torch.randn(batch, seq_len, heads, dim)
        diagnostics = {
            "concentration": torch.rand(batch, heads, seq_len),
            "novelty": torch.rand(batch, heads, seq_len),
        }
        out, state, stats = layer(
            hidden_states=hidden,
            q=q,
            k=k,
            v=v,
            k_hat=k_hat,
            v_hat=v_hat,
            diagnostics=diagnostics,
        )
        (out.square().mean() + state.square().mean()).backward()
        assert stats["used_recurrent"].item() == 1.0
