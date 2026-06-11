# SGSA Modeling and Experiment Agent Guide

## Goal

Implement a lightweight prototype and experiment suite for **Sparse-Guided State Attention (SGSA)** based on the research note in `modelings/idea.md`.

The target is not to train a full production model immediately. The first deliverable should make the core claim testable:

> sparse retrieval should guide recurrent state updates, not merely add a sparse attention output.

## Relevant Files

- `modelings/idea.md`: method proposal and math.
- `modelings/modeling_lali.py`: small Hugging Face-style causal LM; useful as a clean sandbox.
- `modelings/deepseek_v3_2.py`: larger MLA/indexer reference with `Indexer`, `topk_indices`, KV cache, and sparse mask logic.
- `experiments/plot_attention_topk_reuse_results.py`: plotting/reporting style for existing attention reuse experiments.
- `outputs/attention_topk_reuse_exclude_edges_plot_analysis.md`: current pilot result and limitations.
- 可以复用 [Flash Sparse Attention](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention)。

## Important Corrections

Do not assume local/sink retrieved keys are necessarily close to the current key. Sink tokens can occupy special global directions. If implementing residual subspace write, treat it only as removing the component already explained by the current write direction:

$$
\hat{k}_t^\perp = \hat{k}_t - \frac{k_t^\top \hat{k}_t}{||k_t||^2 + \epsilon} k_t
$$

Also do not claim ordinary hybrid attention cannot pass dense/sparse information forward. Residual streams let later layers consume previous mixed outputs. The more defensible SGSA claim is:

- ordinary hybrid passes retrieval evidence implicitly through layer outputs;
- SGSA writes retrieval evidence explicitly into a key-addressed recurrent state;
- the experiment should test whether explicit state writing gives better long-context reuse or efficiency.

## Prototype Scope

Prefer a minimal, inspectable implementation before touching `deepseek_v3_2.py`.

Create a new module such as `modelings/modeling_sgsa.py` unless there is a strong reason to modify `modeling_lali.py` directly.

The prototype should include:

1. `SGSAConfig`
2. `SparseRetriever`
3. `SGSAStateLayer`
4. `SGSAForCausalLM` or a standalone layer test harness

### SparseRetriever

Inputs:

- queries `q`: `(batch, seq, heads, dim)`
- keys `k`: `(batch, seq, heads, dim)` or shared-head keys
- values `v`: `(batch, seq, heads, dim)`
- causal mask
- `top_k` or `top_blocks`

Outputs:

- `indices`: top-k positions or selected block ids
- `weights`: softmax weights over selected positions/blocks
- `k_hat`
- `v_hat`
- diagnostics: score margin/concentration, max confidence, novelty, write conflict, and optional entropy/distance/local/sink/remote ratios for offline analysis

Prefer a block-level retriever first, because training and prefill are more hardware-friendly with contiguous blocks than with global token-level random gather. Implement modes:

- `block_only`: select blocks, then aggregate within selected blocks.
- `block_token`: select blocks, then optionally refine to token-level candidates inside selected blocks.
- `token_only`: global token top-k as a quality upper bound and efficiency lower bound.

Start with exact PyTorch scoring for correctness. Approximate/indexed retrieval can come later.

### SGSAStateLayer

Implement the state update:

$$
W_t=\gamma_t W_{t-1}
+\beta_t(v_t-W_{t-1}k_t)k_t^\top
+\alpha_t(\hat{v}_t-W_{t-1}\hat{k}_t)\hat{k}_t^\top
$$

Then add an option for residual subspace write:

$$
\hat{k}_t^\perp = \hat{k}_t - \frac{k_t^\top \hat{k}_t}{||k_t||^2+\epsilon}k_t
$$

and use:

$$
\alpha_t(\hat{v}_t-W_{t-1}\hat{k}_t^\perp)(\hat{k}_t^\perp)^\top
$$

Keep this switchable via config:

- `sgsa_write_mode="none"`: baseline linear/TTT-style state update
- `sgsa_write_mode="direct"`: sparse state write with `hat{k}`
- `sgsa_write_mode="residual"`: sparse state write with `hat{k_perp}`

### Gating

If the prototype is meant to support GDN-style chunked parallel training/prefill, the main gate must be state-independent. Do not gate on `e_t = v_t - W k_t` or `e_hat_t = v_hat_t - W k_hat_t`, because those residuals depend on the recurrent state and force a sequential recurrence.

Use cheap pre-update features:

$$
\alpha_t
:=\alpha_{\max}
\cdot c_t
\cdot \operatorname{clip}(n_t,0,1)
\cdot b_t
$$

where:

- `c_t`: retrieval concentration from top-1/top-2 score margin, e.g. `sigmoid((score1 - score2) / tau_c)`
- `n_t`: novelty, e.g. `||k_hat_perp|| / (||k_hat|| + eps)`
- `b_t`: optional learned/current-token budget from `x_t`, `q_t`, `v_hat_t`, key/value norms, etc.

Do not use entropy, distance buckets, sink ratio, or residual conflict in the online gate. They can be logged for analysis.

For write budget, use a state-independent proxy such as `alpha_t * ||v_hat_t|| * ||k_hat_t||`; do not use `||e_hat_t||` in the main chunk-parallel path.

### Chunked Parallel Requirement

Represent each token as two simultaneous virtual update samples:

- local sample: `(k_t, v_t, beta_t)`
- sparse sample: `(k_hat_t or k_hat_perp_t, v_hat_t, alpha_t)`

Both samples must read the same pre-token state `W_{t-1}`. In the chunk triangular solve, mask dependencies between samples with the same token timestamp. Only samples from strictly earlier tokens may influence the current sample.

The implementation should avoid materializing `d x d` state transitions. Within a chunk, form the virtual-sample Gram matrix and solve the lower-triangular residual system, following the derivation in `modelings/idea.md`.

## Baselines

At minimum compare:

- Linear/state-only update.
- Output-level sparse hybrid:

$$
o_t=o_t^{state}+\lambda o_t^{sparse}
$$

- Direct SGSA state write.
- Residual-subspace SGSA state write.
- SGSA state write without sparse output residual.

The key ablation is whether SGSA keeps gains when sparse output residual is removed.

## Experiments

### Unit Tests

Add small tests or scripts that verify:

- causal top-k never selects future tokens;
- `k_hat`, `v_hat`, score margin, and confidence have expected shapes;
- residual write produces a key direction nearly orthogonal to `k_t`;
- `sgsa_write_mode="none"` matches the baseline path;
- gates stay in a stable numeric range;
- block-level retrieval never selects future blocks or future tokens inside partially visible blocks.

### Synthetic Retrieval Task

Create a small synthetic task before expensive model training.

Suggested data:

- random token sequence with a key-value fact inserted early;
- later query token asks for that value;
- vary distance between fact and query;
- include distractor sink/local tokens.

Compare whether direct/residual SGSA improves retrieval accuracy over state-only and output-hybrid baselines.

### Top-k Reuse Analysis

Extend the current top-k reuse analysis using longer prompts:

- 1024 tokens
- 2048 tokens
- 4096 tokens if feasible

Compare:

- `reuse_kv`
- `local_window`
- `sink_local_window`
- `random`
- block-only retrieval
- block + token refinement
- token-only retrieval
- anchor-layer block/top-k reuse
- anchor reuse with per-layer reranking

Report:

- attention mass recall
- top-k overlap
- remote token recall
- score margin / concentration
- sink/local/remote composition
- layer-gap sensitivity
- prefill runtime and memory pattern regularity

### Model Ablation

When the prototype runs, compare:

- quality metric on synthetic retrieval;
- LM loss if training on text;
- prefill runtime;
- decode runtime;
- memory use;
- average `alpha_t`;
- average `||k_hat_perp|| / ||k_hat||`.

## Implementation Advice

Keep the first implementation simple and explicit. Prefer loops over time for the recurrent state update until correctness is clear. Optimize only after tests pass.

Avoid modifying `deepseek_v3_2.py` at first. Use it as a reference for how top-k indices and sparse masks are handled. Once the prototype behavior is clear, port only the retrieval/reuse idea into the MLA path.

Preserve all modes behind config flags. The ablation matrix is part of the research contribution, so do not hard-code SGSA as the only path.

## Expected Deliverables

1. A prototype SGSA module or layer.
2. A small synthetic retrieval experiment script.
3. A top-k reuse extension or clear instructions for running longer-prompt reuse analysis.
4. A markdown report under `outputs/` summarizing:
   - which variants were run;
   - what improved;
   - what failed;
   - whether state write beats output hybrid;
   - whether residual-subspace write helps or hurts.

## Success Criteria

The first milestone is successful if it can answer these questions:

1. Does explicit sparse-guided state write outperform output-level hybrid on a controlled retrieval task?
2. Does the gain remain when sparse output residual is disabled?
3. Does residual-subspace write reduce redundant writes without suppressing useful sink/global tokens?
4. Does top-k reuse remain competitive against local/sink baselines on long prompts?

If the answer is no, document the failure mode rather than forcing the method to look good.
