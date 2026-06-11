import torch

from modelings.sgsa_ops import chunk_sgsa, fused_recurrent_sgsa, naive_recurrent_sgsa
from modelings.sgsa_ops.chunk_fwd_triton import build_sgsa_transition_triton


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

    tri = build_sgsa_transition_triton(kappa, lam, prefix, tau)
    gram = torch.einsum("bhid,bhjd->bhij", kappa.float(), kappa.float())
    dep_mask = tau.view(1, 1, 1, virtual_len) < tau.view(1, 1, virtual_len, 1)
    decay = prefix.unsqueeze(-1) / prefix.unsqueeze(-2).clamp_min(1e-6)
    ref = dep_mask.float() * lam.unsqueeze(-2) * decay * gram

    torch.testing.assert_close(tri, ref, atol=2e-2, rtol=2e-2)


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
    torch.testing.assert_close(tri_out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(tri_state, ref_state, atol=2e-2, rtol=2e-2)
