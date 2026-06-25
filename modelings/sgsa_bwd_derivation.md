# SGSA Backward Derivation from GDN

这份文档解释两件事：

1. GDN chunk backward 算子为什么不是一个简单的反向 kernel，而是一组围绕 WY representation 的分解算子。
2. SGSA 如果沿用 GDN 的 chunk 思路，backward 应该如何实现。

当前 `modelings/sgsa_ops/chunk_bwd.py` 还使用 recompute + PyTorch autograd。原因不是 backward 在数学上不可做，而是 GDN 的生产级 backward 本身就拆成多个依赖中间表示的 kernel；SGSA 又把 token 维度改成虚拟样本维度，并且 solve mask 从普通 causal mask 改成真实时间 block-causal mask。直接写一个完整 backward 很容易先把接口和中间量设计错，所以更合理的顺序是：

1. 先稳定 forward 的 `T / R / readout / final_state` 表示。
2. 再按 GDN backward 的模块边界逐个替换成 SGSA 版本。

## 1. GDN Forward 回顾

单 chunk 内，GDN 更新为：

$$
W_t=\gamma_t W_{t-1}+\beta_t r_t k_t^\top
$$

其中：

$$
r_t=v_t-W_{t-1}k_t
$$

把 chunk 内所有 residual 堆成矩阵：

$$
R\in\mathbb{R}^{L\times d_v}
$$

它满足：

$$
(I+T)R=B
$$

其中：

$$
T_{i,j}=
\begin{cases}
\beta_j \frac{D_{i-1}}{D_j}(k_i^\top k_j), & j<i \\
0, & j\ge i
\end{cases}
$$

GDN 的高性能实现不会长期保留原始 residual solve 的所有展开，而是构造 WY representation：

- `A`: 近似表示 `(I+T)^{-1}` 的 chunk-local triangular factor。
- `u`: solved value-side representation，等价于 `A @ (beta * v)` 再扣掉 state prediction 后用于 state update。
- `w`: solved key-side representation，用于和 chunk 初始 state 交互。

在代码里，这对应：

```text
chunk_gated_delta_rule_fwd_intra
  -> chunk_gated_delta_rule_fwd_kkt_solve_kernel
  -> recompute_w_u_fwd
```

随后：

```text
chunk_gated_delta_rule_fwd_h
  -> 用 k, w, u, g 扫 chunk 初始 state
  -> 得到每个 chunk 的 h、v_new、final_state
```

最后：

```text
chunk_fwd_o
  -> o = state_readout + local_readout
```

## 2. GDN Backward 的算子分解

GDN backward 在 `chunk_gated_delta_rule_bwd` 里大致是：

```python
w, u = recompute_w_u_fwd(k, v, beta, A, g)
h, v_new, _ = chunk_gated_delta_rule_fwd_h(k, w, u, g, initial_state)
dv_local = chunk_bwd_dv_local(q, k, g, do)
dh, dh0, dv_state = chunk_gated_delta_rule_bwd_dhu(q, k, w, g, do, dv_local)
dq, dk_read, dw, dg_read = chunk_bwd_dqkwg(q, k, v_new, w, g, h, dv_state, do, dh)
dk_wy, dv, dbeta, dg_wy = prepare_wy_repr_bwd(k, v, beta, g, A, dw, dv_state)
```

从数学上看，它拆成四块。

### 2.1 Readout 的梯度

GDN readout 可写成：

$$
O = O^{state} + A^{read}R
$$

其中：

$$
O^{state}_t = D_t W_0 q_t
$$

$$
A^{read}_{t,j} =
\mathbf{1}[j\le t]\beta_j\frac{D_t}{D_j}(q_t^\top k_j)
$$

给定输出梯度：

$$
\bar O=\frac{\partial \mathcal{L}}{\partial O}
$$

直接有：

$$
\bar R_{read} = (A^{read})^\top \bar O
$$

并且 readout 部分也贡献：

$$
\bar q_t += D_t W_0^\top \bar o_t
$$

$$
\bar k_j +=
\sum_{t\ge j}
\beta_j\frac{D_t}{D_j}
(\bar o_t^\top r_j) q_t
$$

$$
\bar \beta_j +=
\sum_{t\ge j}
\frac{D_t}{D_j}(q_t^\top k_j)(\bar o_t^\top r_j)
$$

decay 的梯度来自所有 `D_t/D_j` 和 `D_t` 项。

GDN 的 `chunk_bwd_dv_local` 与 `chunk_bwd_dqkwg` 就是在避免显式 materialize 大量中间张量的前提下，完成 readout 和 state 交互相关的 `dq/dk/dv/dw/dg`。

### 2.2 跨 chunk state scan 的梯度

chunk 间状态是 recurrent/scan 形式：

$$
W_{out}=D_L W_{in}+\Delta W
$$

forward 要保存或重算每个 chunk 开头的 state `h`。backward 从后往前传播：

$$
\bar W_{in} += D_L \bar W_{out} + \text{readout 对 }W_{in}\text{ 的贡献}
$$

同时 `final_state` 的梯度会进入最后一个 chunk 的 `\bar W_{out}`。

GDN 里对应的是：

```text
chunk_gated_delta_rule_bwd_dhu
```

它反向扫 chunk，输出：

- `dh`: 每个 chunk 初始 state 的梯度。
- `dh0`: initial_state 的梯度。
- 更新后的 `dv`，也就是 state path 对 solved value representation 的梯度贡献。

### 2.3 Triangular solve 的梯度

核心方程是：

$$
L R = B
$$

其中：

$$
L=I+T
$$

forward：

$$
R=L^{-1}B
$$

如果 backward 已经拿到：

$$
\bar R=\frac{\partial \mathcal{L}}{\partial R}
$$

则：

$$
\bar B = L^{-T}\bar R
$$

并且：

$$
\bar L = -\bar B R^\top
$$

因为：

$$
dR=L^{-1}dB-L^{-1}dL R
$$

所以：

$$
d\mathcal{L}
=\langle \bar R,L^{-1}dB\rangle
-\langle \bar R,L^{-1}dL R\rangle
=\langle L^{-T}\bar R,dB\rangle
-\langle (L^{-T}\bar R)R^\top,dL\rangle
$$

GDN 的 `prepare_wy_repr_bwd` 本质上就是把 `dw/du` 反传穿过 WY representation 和 triangular factor `A`，得到：

- `dk`
- `dv`
- `dbeta`
- `dg`

它没有显式写成上面的 `bar_L = -bar_B R^T`，但数学作用等价：把 solve 与 `K K^T`、`beta`、decay 的依赖关系反传回输入。

### 2.4 Decay / gate 梯度

GDN 的 `g` 是 log-space cumulative gate。forward 内部经常使用：

$$
\exp(g_i-g_j)
$$

所以 backward 里对 `g` 的梯度先以 prefix/cumsum 形式累积，然后再做 reverse cumsum：

```python
dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True)
```

如果 gate 是 kernel 内部从 raw gate 激活得到，还要再经过：

```text
gdn_gate_bwd
```

SGSA 里的 `gamma` 当前是 post-sigmoid decay，不是 GDN 的 log gate接口；实现 SGSA backward 时可以先对 `gamma` 直接求梯度，之后再让 PyTorch autograd 传回 `gamma_proj`。如果未来改成 log gate，则可以更接近 GDN 的 `dg` 路径。

## 3. SGSA Forward 的对应表示

SGSA 每个真实 token 有两个虚拟写入样本：

$$
(\kappa_{t,0},\nu_{t,0},\lambda_{t,0})=(k_t,v_t,\beta_t)
$$

$$
(\kappa_{t,1},\nu_{t,1},\lambda_{t,1})=(\tilde k_t,\hat v_t,\alpha_t)
$$

把 chunk 内 `L` 个真实 token 展成 `M=2L` 个虚拟样本，记真实时间为：

$$
\tau_i
$$

SGSA residual solve：

$$
L^{sgsa} R = B
$$

其中：

$$
L^{sgsa}=I+T^{sgsa}
$$

$$
T^{sgsa}_{i,j}
=
\mathbf{1}[\tau_j<\tau_i]
\lambda_j
\frac{D_{\tau_i}}{D_{\tau_j}}
(\kappa_i^\top \kappa_j)
$$

这里用 0-indexed chunk 代码口径时，`D_{\tau_i}` 对应 `prefix_tau[i]`，因为当前 implementation 在 token 开始先做 decay：

```python
base_state = gamma_t * state
```

同 token 的两个虚拟样本不能互相看见：

$$
\tau_j=\tau_i \Rightarrow T^{sgsa}_{i,j}=0
$$

readout 是真实 token 行、虚拟样本列：

$$
O_t
=D_t W_0 q_t
+\sum_{i:\tau_i\le t}
\lambda_i
\frac{D_t}{D_{\tau_i}}
(q_t^\top \kappa_i)
R_i
$$

final state：

$$
W_{out}
=D_L W_0
+\sum_i
\lambda_i
\frac{D_L}{D_{\tau_i}}
\kappa_i R_i^\top
$$

代码中 state layout 是 `[D, D]`，第一维是 key，第二维是 value，所以写入是：

```text
state += kappa_i[:, None] * weighted_residual_i[None, :]
```

## 4. SGSA Backward 的数学推导

SGSA backward 可以直接沿用 `L R = B` 的通用公式，只是 `L`、`B`、readout 和 final-state kernel 都换成虚拟样本版本。

### 4.1 Readout 对 R / q / kappa / lambda 的梯度

定义：

$$
A^{read}_{t,i}
=
\mathbf{1}[\tau_i\le t]
\lambda_i
\frac{D_t}{D_{\tau_i}}
(q_t^\top \kappa_i)
$$

则：

$$
O=O^{state}+A^{read}R
$$

给定 `do`：

$$
\bar R_{read}=(A^{read})^\top \bar O
$$

对每个真实 token：

$$
\bar q_t += D_t W_0^\top \bar o_t
$$

以及对虚拟样本 `i`：

$$
\bar \kappa_i^{read}
=
\sum_{t:\tau_i\le t}
\lambda_i
\frac{D_t}{D_{\tau_i}}
(\bar o_t^\top R_i)q_t
$$

$$
\bar \lambda_i^{read}
=
\sum_{t:\tau_i\le t}
\frac{D_t}{D_{\tau_i}}
(q_t^\top\kappa_i)
(\bar o_t^\top R_i)
$$

对 prefix decay 的梯度来自：

$$
D_t/D_{\tau_i}
$$

和 baseline：

$$
D_t W_0 q_t
$$

### 4.2 Final state 对 R / kappa / lambda / initial_state 的梯度

final state：

$$
W_{out}=D_LW_0+\sum_i c_i \kappa_i R_i^\top
$$

其中：

$$
c_i=\lambda_i\frac{D_L}{D_{\tau_i}}
$$

给定：

$$
\bar W_{out}
$$

有：

$$
\bar W_0 += D_L \bar W_{out}
$$

$$
\bar R_i^{state} += c_i \bar W_{out}^\top \kappa_i
$$

$$
\bar \kappa_i^{state} += c_i \bar W_{out} R_i
$$

$$
\bar \lambda_i^{state} +=
\frac{D_L}{D_{\tau_i}}
\langle \bar W_{out},\kappa_i R_i^\top\rangle
$$

decay 梯度同样来自：

$$
D_L
$$

和：

$$
D_L/D_{\tau_i}
$$

这部分就是我们刚加的 Triton forward readout/final-state kernel 的反向版本。

### 4.3 Triangular solve 梯度

SGSA solve：

$$
L^{sgsa}R=B
$$

设 readout 和 final state 累加后的 residual 梯度为：

$$
\bar R
$$

则：

$$
\bar B=(L^{sgsa})^{-T}\bar R
$$

$$
\bar L=-\bar B R^\top
$$

因为 `L = I + T`，对角线常数 `I` 没有梯度，只保留：

$$
\bar T_{i,j}=\bar L_{i,j}
\quad \text{where } \tau_j<\tau_i
$$

### 4.4 B 项的梯度

SGSA:

$$
B_i=\nu_i-D_{\tau_i}W_0\kappa_i
$$

所以：

$$
\bar \nu_i += \bar B_i
$$

$$
\bar W_0 += -D_{\tau_i}\bar B_i\kappa_i^\top
$$

$$
\bar \kappa_i^{B} += -D_{\tau_i}W_0^\top\bar B_i
$$

decay 也收到：

$$
-D_{\tau_i}(W_0\kappa_i)^\top\bar B_i
$$

的贡献。

### 4.5 T 项的梯度

SGSA transition：

$$
T_{i,j}
=m_{i,j}\lambda_j s_{i,j} g_{i,j}
$$

其中：

$$
m_{i,j}=\mathbf{1}[\tau_j<\tau_i]
$$

$$
s_{i,j}=\kappa_i^\top \kappa_j
$$

$$
g_{i,j}=\frac{D_{\tau_i}}{D_{\tau_j}}
$$

有：

$$
\bar \lambda_j^T +=
\sum_i
m_{i,j}
\bar T_{i,j}
g_{i,j}
s_{i,j}
$$

$$
\bar \kappa_i^T +=
\sum_j
m_{i,j}
\bar T_{i,j}
\lambda_j
g_{i,j}
\kappa_j
$$

$$
\bar \kappa_j^T +=
\sum_i
m_{i,j}
\bar T_{i,j}
\lambda_j
g_{i,j}
\kappa_i
$$

注意第二个式子里的 `j` 是列索引，所以同一个虚拟样本作为 row key 和 column key 都会收到梯度。实现时可以把 `bar_T * mask * lam[None, :] * decay` 当成一个矩阵 `G`：

```text
d_kappa_from_rows = G @ kappa
d_kappa_from_cols = G.T @ kappa
d_kappa += d_kappa_from_rows + d_kappa_from_cols
```

其中：

```text
G[i, j] = bar_T[i, j] * mask[i, j] * lam[j] * decay[i, j]
```

对 `lambda`：

```text
dlam[j] += sum_i bar_T[i, j] * mask[i, j] * decay[i, j] * dot(kappa_i, kappa_j)
```

对 decay/prefix：

$$
\bar g_{i,j}
=m_{i,j}\bar T_{i,j}\lambda_j(\kappa_i^\top \kappa_j)
$$

再由：

$$
g_{i,j}=D_{\tau_i}/D_{\tau_j}
$$

反传到 `D` 或 `log D`。实践上更建议在 kernel 内累积到 `dlog_prefix`：

$$
\frac{\partial g_{i,j}}{\partial \log D_{\tau_i}}=g_{i,j}
$$

$$
\frac{\partial g_{i,j}}{\partial \log D_{\tau_j}}=-g_{i,j}
$$

所以：

```text
dlogD[tau_i] += d_g[i,j] * g[i,j]
dlogD[tau_j] -= d_g[i,j] * g[i,j]
```

最后再把 `dlogD` 变成 `dgamma`。因为：

$$
\log D_t=\sum_{p\le t}\log\gamma_p
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial \log\gamma_p}
=\sum_{t\ge p}
\frac{\partial \mathcal{L}}{\partial \log D_t}
$$

再有：

$$
\frac{\partial \log\gamma_p}{\partial \gamma_p}
=1/\gamma_p
$$

因此：

```text
dgamma[p] = reverse_cumsum(dlogD)[p] / gamma[p]
```

这和 GDN 的 `chunk_local_cumsum(..., reverse=True)` 是同一个结构。

## 5. 虚拟样本梯度拆回 SGSA 输入

完成虚拟样本 backward 后，会得到：

- `d_kappa`: `[B, H, M, D]`
- `d_nu`: `[B, H, M, D]`
- `d_lam`: `[B, H, M]`

如果没有 sparse write：

```text
dk = d_kappa
dv = d_nu
dbeta = d_lam
```

如果有 sparse write，interleave 结构是：

```text
even virtual index: local
odd virtual index: sparse
```

所以：

```text
dk        = d_kappa[:, :, 0::2]
dsparse_k = d_kappa[:, :, 1::2]
dv        = d_nu[:, :, 0::2]
dsparse_v = d_nu[:, :, 1::2]
dbeta     = d_lam[:, :, 0::2]
dalpha    = d_lam[:, :, 1::2]
```

再 transpose 回 public API 的 `[B, T, H, D]` / `[B, T, H]`。

如果 sparse key 是 residual-subspace write：

```python
sparse_k = k_hat - dot(k_hat, k) / (||k||^2 + eps) * k
```

那么 `dsparse_k` 还要继续通过这个 projection 回传到 `k_hat` 和 `k`。目前 `SGSAStateLayer` 在 Python 里先构造 `write_key`，再传入 op；所以 SGSA op 只需要返回 `d_write_key`，projection 的梯度由 PyTorch autograd 负责。

## 6. SGSA Triton Backward 建议实现路线

不要一开始照搬 GDN 的完整 `chunk_gated_delta_rule_bwd`，建议按下面顺序做。

### Step 1: 保存必要中间量

forward autograd ctx 至少保存：

- `q, k, v, beta, gamma`
- optional `sparse_k, sparse_v, alpha`
- `initial_state`
- `tau`
- `prefix`
- `prefix_tau`
- solved `residual`
- transition factor 或 solved inverse representation

如果为了省显存，也可以保存 `A` 和重算 `residual`，但第一版 backward 建议先保存 `residual`，降低复杂度。

### Step 2: 写 readout/final-state backward Triton

输入：

- `do`
- `d_final_state`
- `q_h`
- `kappa_h`
- `residual`
- `lam_h`
- `prefix`
- `prefix_tau`
- `state_c`

输出：

- `dq_h`
- `d_kappa_h`
- `d_residual`
- `d_lam_h`
- `d_state_c`
- `dlogD`

这一步独立于 triangular solve，最适合作为第一个 bwd kernel。

### Step 3: 用 PyTorch triangular solve 做第一版 solve backward

先不要立刻写 fused solve backward。可以先在 CUDA backward 中使用：

```python
d_rhs = torch.linalg.solve_triangular(L.transpose(-1, -2), d_residual, upper=True)
d_transition = -torch.einsum("bhid,bhjd->bhij", d_rhs, residual)
```

然后写一个 Triton/PyTorch kernel 把 `d_transition` 反传到：

- `d_kappa`
- `d_lam`
- `dlogD`

这一步已经能摆脱 full recompute autograd，性能会明显好于当前 `chunk_bwd.py`。

### Step 4: 写 transition backward Triton

transition backward 是 SGSA 特有的核心 kernel：

```text
T[i,j] = mask(tau_j < tau_i) * lam[j] * prefix_tau[i]/prefix_tau[j] * dot(kappa[i], kappa[j])
```

它和 forward transition kernel 结构对称，可以按 `(BLOCK_M, BLOCK_N)` tile 做：

- 读取 `bar_T` tile。
- 重算 `dot(kappa_i, kappa_j)`。
- 累积 `dlam[j]`。
- 对 `d_kappa[i]` 和 `d_kappa[j]` 做 atomic add。
- 对 `dlogD[tau_i]` / `dlogD[tau_j]` 做 atomic add。

### Step 5: dB backward

根据：

```text
B = nu - prefix_tau * (state @ kappa)
```

写一个简单 kernel：

- `dnu += d_rhs`
- `dstate += -prefix_tau * outer(kappa, d_rhs)`
- `dkappa += -prefix_tau * state @ d_rhs`
- `dlogD[tau] += -prefix_tau * dot(state @ kappa, d_rhs)`

### Step 6: dlogD -> dgamma

把所有来源累加到 `dlogD` 后，对每个 chunk/head 做 reverse cumsum：

```text
dloggamma[t] = sum_{s >= t} dlogD[s]
dgamma[t] = dloggamma[t] / gamma[t]
```

这可以先用 PyTorch `flip(cumsum(flip()))` 实现，后面再换成 Triton scan。

### Step 7: 多 chunk backward

当前 SGSA chunk forward 是 Python for-loop over chunks，state carry 是顺序的。第一版 backward 可以反向 for-loop over chunks，逐 chunk 调用上面的 kernels，并把 `d_state_c` 传给前一个 chunk。

之后如果需要完全 FLA 风格的跨 chunk scan，再把 state carry 改成 associative scan。

## 7. 和 GDN Backward 的主要差异

SGSA 相比 GDN 有这些必须单独处理的点：

1. `M=2L` 虚拟样本长度，readout 仍是 `L` 个真实 token。
2. solve mask 是 `tau_j < tau_i`，不是普通 `j<i`。
3. readout mask 是 `tau_i <= t`，shape 是 `[L, M]`。
4. `lambda` 同时包含 `beta` 和 `alpha`，需要拆回两个梯度。
5. `kappa/nu` 同时包含 local 和 sparse write，需要拆回 `k/v/sparse_k/sparse_v`。
6. `gamma` 当前是 post-sigmoid decay，不是 GDN 的 log gate；建议先用 `dlogD -> dgamma`，而不是直接复用 GDN `dg`。
7. `direct` 和 `residual` sparse write 的 projection 不应放进 SGSA op 内部；op 只负责传入的 `sparse_k`。

## 8. 最短实现计划

第一版可用 SGSA backward 不需要一步到位复刻 GDN 的所有 fused kernel。建议：

1. 保存 `residual` 和 `transition`。
2. 写 readout/final-state backward Triton。
3. 用 PyTorch `solve_triangular` 做 solve backward。
4. 写 transition backward Triton。
5. 写 `B` backward Triton。
6. 用 PyTorch reverse cumsum 得到 `dgamma`。
7. 通过测试后，再把 solve backward 和 reverse cumsum fuse 掉。

这样实现出来的 backward 会比当前 recompute autograd 明确得多，也更容易逐步对齐 GDN 的高性能结构。
