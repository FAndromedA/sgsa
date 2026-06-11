# Sparse-Guided State Attention

## 1. Motivation

线性注意力、GDN、TTT 这类方法可以把历史 token 压缩进一个 recurrent state，因此在长上下文推理中有明显的计算和缓存优势。以 TTT 形式写，每个 token 都可以被看成对状态矩阵 $W_t$ 的一次在线更新：

$$
z_t=f(x_t; W_t) \\
W_t = W_{t-1} - \eta \nabla \mathcal{l}(W_{t-1};x_t) \\
\nabla \mathcal{l}(W_{t-1};x_t)=||W_{t-1}k_{t}-v_t||^2
$$

这类状态空间的弱点也很清楚：带衰减或低秩约束的状态更善于保留平滑、局部、频繁出现的信息，但对少数远距离关键 token 不够敏感。普通 hybrid attention 往往用一个 sparse/dense 分支在输出端做补丁：

$$
o_t = o_t^{\text{linear}} + \lambda_t o_t^{\text{sparse}}
$$

这种方法能改善当前 token 的输出，但没有改变线性/TTT 分支的记忆状态。下一步 token 仍然只能从原来的 $W_t$ 里读信息，远程证据不会真正进入 recurrent memory。因此，更有创新性的方向不是“输出级混合”，而是“状态级混合”：让 sparse retrieval 直接改变状态更新的方向、强度和记忆几何。

本文方法可以称为 **Sparse-Guided State Attention (SGSA)**。核心模块是 **Sparse-Guided State Update (SGU)**：

> sparse branch retrieves external memories; state branch treats them as additional online supervision and writes them into recurrent state through gated, confidence-aware updates.

## 2. Retrieval as State Supervision

对第 $t$ 个 token，先用 sparse branch 从历史 KV cache 中选择候选集合：

$$
S_t=\operatorname{TopK}(q_t,K_{\le t})
$$

在 $S_t$ 内重新归一化 attention 权重：

$$
a_{t,i}=\operatorname{softmax}_{i\in S_t}(q_t^\top k_i)
$$

然后构造 retrieval memory：

$$
\hat{k}_t=\sum_{i\in S_t}a_{t,i}k_i,\quad
\hat{v}_t=\sum_{i\in S_t}a_{t,i}v_i
$$

这里 $\hat{k}_t,\hat{v}_t$ 不应该被理解为普通 sparse attention 的最终输出，而应该被理解为一个外部记忆样本。于是状态更新目标从单样本拟合变成双样本拟合：

$$
\mathcal{l}_t(W)
= ||Wk_t-v_t||^2
+ \rho_t ||W\hat{k}_t-\hat{v}_t||^2
$$

其中 $\rho_t$ 是 retrieval memory 的可信度。对应的一阶更新为：

$$
W_t=\gamma_t W_{t-1}
+\beta_t(v_t-W_{t-1}k_t)k_t^\top
+\alpha_t(\hat{v}_t-W_{t-1}\hat{k}_t)\hat{k}_t^\top
$$

展开后得到：

$$
W_t=\gamma_t W_{t-1}
+\beta_t v_tk_t^\top
+\alpha_t \hat{v}_t\hat{k}_t^\top
-W_{t-1}(\beta_t k_tk_t^\top+\alpha_t \hat{k}_t\hat{k}_t^\top)
$$

这里：

- $\beta_t$ 控制当前 token 的局部写入；
- $\alpha_t$ 控制 sparse retrieved memory 的远程写入；
- $\gamma_t$ 控制遗忘或保留；
- $\rho_t$ 或 $\alpha_t$ 由 retrieval concentration、novelty、write conflict 和 write budget 决定。

这个公式比 $o_t=(o_t^{\text{sparse}}+o_t^{\text{linear}})/2$ 更关键，因为 sparse branch 的影响不只作用于当前输出，而是被写入 $W_t$，会影响之后所有 query 的读出。

## 3. Stronger Variant: Residual Subspace Sparse State Write

为了进一步增强创新性，可以把 sparse write 设计成 **residual subspace memory write**。这里的目标不是证明 local/sink token 的 key 一定和当前 $k_t$ 接近，而是更保守地避免 sparse update 在已经由当前 token 写入的方向上重复更新。

先把 retrieval key 分解为当前写入方向上的分量和残差分量：

$$
\hat{k}_t^\parallel
=\frac{k_t^\top\hat{k}_t}{||k_t||^2}k_t,\quad
\hat{k}_t^\perp
=\hat{k}_t-\hat{k}_t^\parallel
$$

只把 retrieval memory 中相对当前写入方向的新分量写入状态：

$$
W_t=\gamma_t W_{t-1}
+\beta_t(v_t-W_{t-1}k_t)k_t^\top
+\alpha_t(\hat{v}_t-W_{t-1}\hat{k}_t^\perp)(\hat{k}_t^\perp)^\top
$$

这个版本的直觉更严谨：

- 它不依赖“local/sink key 必然接近 $k_t$”这个强假设；如果 sink token 是特殊全局方向，$\hat{k}_t^\perp$ 仍然可能很大，模型不会仅因为它是 sink 就错误抑制。
- 它只去除和当前 token 写入方向共线的重复分量，因此更像一个 state-space residual correction。
- 如果 sparse branch 找到真正远程证据，且该证据对应当前状态尚未覆盖的方向，则 $\hat{k}_t^\perp$ 会捕获这个新增方向，状态容量被用在更有价值的信息上。
- 这让 sparse branch 的贡献从“多一个 attention 输出”变成“扩展 state 的可写入子空间”。

因此，方法可以主打：

> Sparse branch does not simply add retrieved values to the output; it proposes a retrieval-induced state correction, and the model writes the component that is not already explained by the current token update.

## 4. Mathematical Advantage Over Output-level Hybrid

先看单层 recurrent state 的抽象。普通输出混合可以写成：

$$
o_t^{\text{hyb}}=W_{t-1}q_t+\lambda_t\hat{v}_t
$$

其中 $\hat{v}_t$ 直接修正当前层当前 token 的输出。在单层 recurrent state 抽象中，如果不再次检索同一个远程 memory，则未来 token 的状态读出仍然是：

$$
o_u^{\text{hyb}}=W_{u-1}q_u
$$

也就是说，在同一个 recurrent state 内，输出混合的远程信息是 **transient correction**，不会显式变成未来 token 可按 key 访问的 memory。

需要注意，这并不意味着普通 layer-wise hybrid 完全不能把信息传给后续层。实际 Transformer 有 residual stream，当前层的 dense/sparse 混合输出会进入后面的层，因此后层确实可以利用前层已经融合过的信息。更准确的区别是：普通 hybrid 依赖 residual stream 和后续层的非线性变换来隐式传播远程证据；SGSA 则把远程证据显式写成当前层 state 的一个可寻址更新。

SGSA 则把 retrieval memory 写进状态：

$$
\Delta W_t^{\text{sparse}}
=\alpha_t(\hat{v}_t-W_{t-1}\hat{k}_t)\hat{k}_t^\top
$$

于是未来 query $q_u$ 会额外读到：

$$
\Delta o_{u|t}
=\Delta W_t^{\text{sparse}}q_u
=\alpha_t(\hat{v}_t-W_{t-1}\hat{k}_t)(\hat{k}_t^\top q_u)
$$

这给出了一个更准确的数学区别：

- 输出混合主要通过 residual stream 把 $\lambda_t\hat{v}_t$ 交给后续层处理，信息传递是隐式的、层间的；
- 状态写入在当前层形成 $\Delta W_t$，会对未来 query 产生按相似度 $\hat{k}_t^\top q_u$ 调制的显式影响。

如果未来 token $q_u$ 与 retrieval key $\hat{k}_t$ 相关，则 SGSA 自动复用远程证据；如果不相关，则影响接近 0。这相当于把 sparse retrieval 的结果变成一个可寻址的 associative memory。

更形式化地，设目标 dense attention 中某个远程 memory 对未来一组 query $Q_F=\{q_u\}_{u=t+1}^{T}$ 都有贡献。普通输出混合可以通过后续层间接传播这段信息，但在当前层没有一个显式的 key-addressed memory slot；若想在同层对多个未来 query 复用该 memory，通常需要每个 $u$ 再次 sparse retrieval。SGSA 只需在 $t$ 写入一次，其对未来窗口的累计贡献为：

$$
\sum_{u=t+1}^{T}\Delta o_{u|t}
=\alpha_t(\hat{v}_t-W_{t-1}\hat{k}_t)
\sum_{u=t+1}^{T}\hat{k}_t^\top q_u
$$

当未来 query 与同一远程 key 成簇相关时，SGSA 的单次写入可以服务多个未来 token。这是它相对普通 hybrid 的核心理论优势：**amortized retrieval**。

## 5. Error Reduction View

把理想 dense attention 的局部线性近似记作一个目标映射 $W^\star$，线性状态当前误差为：

$$
E_t(W)=||W-W^\star||_F^2
$$

在同一个 recurrent state 的视角下，普通输出混合不改变 $W$，因此对该层未来状态误差没有直接改进：

$$
E_t(W_t^{\text{hyb}})=E_t(W_{t-1})
$$

SGSA 的 sparse update 是沿着 retrieval residual 的梯度下降：

$$
\Delta W_t^{\text{sparse}}
=-\frac{\alpha_t}{2}\nabla_W ||W\hat{k}_t-\hat{v}_t||^2
$$

如果 retrieval memory 近似满足 $\hat{v}_t\approx W^\star\hat{k}_t$，则：

$$
\langle W_{t-1}-W^\star,\nabla_W ||W_{t-1}\hat{k}_t-\hat{v}_t||^2\rangle
=2|| (W_{t-1}-W^\star)\hat{k}_t ||^2 \ge 0
$$

因此在足够小的 $\alpha_t$ 下，sparse state write 会降低目标映射误差的一阶项：

$$
E_t(W_t) \le E_t(W_{t-1})
-2\alpha_t||(W_{t-1}-W^\star)\hat{k}_t||^2
+O(\alpha_t^2)
$$

这个不等式可以作为 proposal 中的理论卖点：当 sparse retrieval 找到的 memory 是可信的，SGSA 不只是补当前输出，而是在降低该层 state 对目标 dense attention 映射的近似误差。对完整深层网络来说，这不是证明普通 hybrid 一定劣于 SGSA，而是说明 SGSA 提供了普通输出混合没有显式建模的优化目标。

## 6. Write Gate and Conflict Control

如果目标是参考 GDN / DeltaNet 做 chunked parallel，gate 必须满足一个硬约束：$\alpha_t,\beta_t,\gamma_t$ 需要在进入 recurrent state update 之前就能确定，不能依赖 $W_{t-1}k_t$ 或 $W_{t-1}\hat{k}_t$ 这类在线 residual。否则每个 token 的 gate 都依赖前一步 state，chunk 内会变成非线性递推，无法化成一次 triangular solve。

因此主方法应采用 **state-independent write gate**：

$$
\alpha_t=\alpha_{\max}\cdot c_t\cdot n_t\cdot b_t
$$

其中：

- $c_t=\sigma((s_{t,(1)}-s_{t,(2)})/\tau_c)$，来自 top-1/top-2 block 或 token score margin，表示 retrieval 是否明确；
- $n_t=||\hat{k}_t^\perp||/(||\hat{k}_t||+\epsilon)$，只依赖当前 key 和 retrieval key，表示是否提供新 key 方向；
- $b_t$ 是可学习或手工的 write budget gate，只依赖 $x_t,q_t,\hat{v}_t,||\hat{k}_t||,||\hat{v}_t||$ 等当前 token 可见量。

不要把下面这些量作为主路径 gate：

- $h_t=-\sum a\log a$：需要额外 log/reduction，在 block sparse 或 token top-k 路径中增加开销；
- $e_t=v_t-W_{t-1}k_t$ 和 $\hat e_t=\hat v_t-W_{t-1}\hat k_t$：依赖在线 state，会破坏 chunked parallel；
- 显式 $u_t=e_tk_t^\top$、$\hat u_t=\hat e_t\hat k_t^\top$：即使可用 Frobenius 恒等式降成向量 dot product，也仍然依赖 $e_t,\hat e_t$，不适合作为主 gate。

这些 residual/conflict 指标可以保留为 **recurrent debug ablation** 或离线分析，用来解释失败样本，但不要进入需要高效训练/prefill 的主模型。

为了限制写入范数，主路径也不应依赖 $||\hat e_t||$。可以使用更保守但可并行的 proxy：

$$
\alpha_t \leftarrow \alpha_t\cdot
\min\left(1,\frac{\tau_w}{\alpha_t||\hat v_t||||\tilde k_t||+\epsilon}\right)
$$

其中 $\tilde k_t$ 表示实际 sparse write key，可以是 $\hat k_t$ 或 $\hat k_t^\perp$。这个 proxy 不如 residual norm 精确，但不依赖 $W_{t-1}$，因此能保留 chunked parallel 能力。

## 7. Chunked Parallel Form

修改为 state-independent gate 后，SGSA 可以化成 GDN/DeltaNet 类似的 chunked parallel 形式。关键是把每个 token 的 local write 和 sparse write 看成同一时刻的两个虚拟训练样本。

定义：

$$
(\kappa_{t,0},\nu_{t,0},\lambda_{t,0})=(k_t,v_t,\beta_t)
$$

$$
(\kappa_{t,1},\nu_{t,1},\lambda_{t,1})=(\tilde{k}_t,\hat{v}_t,\alpha_t)
$$

其中 $\tilde{k}_t$ 可以是 $\hat{k}_t$，也可以是 residual subspace 版本 $\hat{k}_t^\perp$。SGSA 更新写成：

$$
W_t=\gamma_t W_{t-1}
+\sum_{r\in\{0,1\}}\lambda_{t,r}
(\nu_{t,r}-W_{t-1}\kappa_{t,r})\kappa_{t,r}^\top
$$

注意两个虚拟样本都使用同一个 $W_{t-1}$，不是先 local write 再 sparse write。这样 sparse 分支不会在同一 token 内读到 local write 之后的 state。

展开为 affine recurrence：

$$
W_t=W_{t-1}A_t+B_t
$$

其中：

$$
A_t=\gamma_t I-\sum_r\lambda_{t,r}\kappa_{t,r}\kappa_{t,r}^\top,
\quad
B_t=\sum_r\lambda_{t,r}\nu_{t,r}\kappa_{t,r}^\top
$$

这说明它理论上一定可以做 associative scan：

$$
(A_2,B_2)\circ(A_1,B_1)=(A_1A_2, B_1A_2+B_2)
$$

直接 scan 会涉及 $d_k\times d_k$ 的 $A_t$，不够高效。GDN 的 chunk trick 是不显式形成 $A_t$，而是在 chunk 内用 kernel Gram matrix 和 lower-triangular solve 表示所有 residual。

考虑一个 chunk，初始 state 为 $W_0$。把 chunk 内所有虚拟样本展平成序列 $i=1,\dots,M$，其中 $M=2L$。每个虚拟样本有时间戳 $\tau(i)$，对应 $(\kappa_i,\nu_i,\lambda_i)$。为了保证同一 token 的 local/sparse write 都读 $W_{t-1}$，构造严格因果 mask：只有 $\tau(j)<\tau(i)$ 时，样本 $j$ 才影响样本 $i$。

令 decay product：

$$
G_{j\rightarrow i}=\prod_{p=\tau(j)+1}^{\tau(i)-1}\gamma_p
$$

定义初始 residual：

$$
b_i=\nu_i-\left(\prod_{p=1}^{\tau(i)-1}\gamma_p\right)W_0\kappa_i
$$

真实 residual 满足：

$$
r_i=b_i-
\sum_{j:\tau(j)<\tau(i)}
\lambda_j G_{j\rightarrow i}(\kappa_j^\top\kappa_i)r_j
$$

令 lower-triangular matrix $L$ 为：

$$
L_{i,j}=\mathbf{1}[\tau(j)<\tau(i)]\lambda_jG_{j\rightarrow i}(\kappa_j^\top\kappa_i)
$$

则整个 chunk 的 residual 可以并行求解：

$$
(I+L)R=B
$$

这里 $R,B\in\mathbb{R}^{M\times d_v}$，求解维度是 chunk 内虚拟样本数 $M$，不是 $d_k\times d_k$ state matrix。得到 $R$ 后，chunk 末状态为：

$$
W_{\text{out}}
=\left(\prod_{p=1}^{L}\gamma_p\right)W_0
+
\sum_{i=1}^{M}\lambda_i
\left(\prod_{p=\tau(i)+1}^{L}\gamma_p\right)
r_i\kappa_i^\top
$$

这就是 SGSA 的 chunked parallel 形式。它和 GDN 的关系是：GDN 每个 token 通常只有一个 update sample；SGSA 每个 token 有 local 和 sparse 两个 update sample，并且同 token 两个 sample 之间的 lower-triangular dependency 被 mask 掉。

因此，SGSA 可以 chunked parallel，但需要满足以下限制：

- $\alpha_t,\beta_t,\gamma_t$ 必须 state-independent；
- sparse retrieval 的 block/token selection 必须先于 state update 完成；
- residual/conflict gate 不能进入主路径；
- 同一 token 的 local/sparse writes 要作为 simultaneous multi-sample update，而不是顺序 update；
- chunk 内计算只显式形成 $M\times M$ Gram/triangular matrix，不能形成 $d\times d$ state transition。

如果坚持使用依赖 $e_t,\hat e_t$ 的 gate，那么它只能做 recurrent 版本或近似版本，不能保持 GDN 式精确 chunked parallel。

## 8. Readout

主输出应尽量来自状态读出：

$$
o_t^{\text{state}}=W_tq_t
$$

可以保留一个很小的 sparse readout 作为训练稳定项：

$$
o_t=o_t^{\text{state}}+\lambda_t o_t^{\text{sparse}}
$$

但关键 ablation 必须证明：只保留 sparse-guided state write、去掉 sparse output residual 时，仍然能获得长上下文收益。否则方法会退化成普通 hybrid attention。

## 9. Block-level Sparse Retrieval and Top-k Reuse

稀疏分支建议优先采用 **block-level sparse attention**，而不是一开始就做纯 token-level top-k。原因是 SGSA 的目标不是精确复刻 dense attention 的每个 token 权重，而是获得稳定、可写入 state 的 retrieval memory；block-level 选择在训练和 prefill 阶段更硬件友好。

block-level sparse retrieval 的优势：

- block 是连续内存，prefill 阶段可以用 block-sparse / FlashAttention 风格 kernel，避免 token-level random gather；
- 训练阶段 attention pattern 更规整，反向传播和 mask 处理更容易优化；
- block metadata 比 token top-k indices 小，适合跨层 anchor reuse；
- block summary 的 score margin、confidence、novelty 更稳定，不会被单个 sink token 或偶然高分 token 主导；
- 对 SGSA 来说，retrieval memory 可以先由 selected blocks 聚合，再写入 state，不必强依赖单 token 精度。

推荐路线是 coarse-to-fine：

1. 将历史序列划分为 block，计算 block score：

$$
s_{t,b}=\operatorname{score}(q_t, K_b)
$$

其中 $K_b$ 可以是 block key summary、max-pooled key、learned index key，或者 block 内 top score 的近似。

1. 选择 top block：

$$
B_t=\operatorname{TopKBlock}(q_t,\{K_b\})
$$

1. 在 selected blocks 内聚合 retrieval memory：

$$
\hat{k}_t=\sum_{b\in B_t}\sum_{i\in b}a_{t,i}k_i,\quad
\hat{v}_t=\sum_{b\in B_t}\sum_{i\in b}a_{t,i}v_i
$$

1. 如果质量不够，再在 selected blocks 内做 token-level refinement，而不是全局 token top-k。

这样可以得到三种可比较的 sparse branch：

- **block-only**：只选 block，在 block 内做完整局部 softmax 聚合；
- **block + token refinement**：先选 block，再在 block 内选 token；
- **token-only**：全局 token top-k，作为质量上界和硬件效率下界。

为了不抵消线性状态的效率优势，top-k/block 检索还可以做 anchor reuse：

1. 前几层保留 dense 或强 sparse 检索，保证早期表示充分混合。
2. 中后层每 4 层设置一个 anchor layer。
3. anchor layer 计算 $B_t$ 或 $S_t$。
4. 后续 3 层复用候选 block/token 集合，只重新打分、重新门控或轻量过滤。
5. decode 阶段缓存 candidate set 和 retrieval statistics，避免每层完整检索。

现有 MLA/indexer 原型中已经有 `Indexer`、`topk_indices`、KV cache 和 sparse mask，可以先用它验证 candidate set 复用。当前 pilot 还不能证明 `reuse_kv` 一定有效，因为短序列下 `k=64` 覆盖比例过大，且 `sink_local_window` 很强。因此 proposal 中应把 top-k reuse 写成待验证的效率假设，而不是既定结论。真正的实验应比较 block-only、block+token refinement、token-only 三条路线在质量、prefill 速度、训练可实现性上的 trade-off。

## 10. Relation to Existing Hybrid Attention

和已有 hybrid attention 相比，SGSA 的差异可以概括为：

- 普通 hybrid：sparse branch 是 output correction。
- SGSA：sparse branch 是 state supervision。
- 普通 hybrid：远程证据只服务当前 token。
- SGSA：远程证据被写入 associative memory，可被未来 query 复用。
- 普通 hybrid：融合发生在 value/output space。
- SGSA：融合发生在 state update geometry，即 $k$ 方向、$\hat{k}$ 方向和正交 residual 方向上。

更适合的论文定位是 **retrieval-augmented recurrent state**，而不是 sparse attention + linear attention。

## 11. Experiments

### Candidate Quality

用 1024/2048/4096+ token prompt 和 retrieval prompt 重跑 top-k reuse，比较：

- `reuse_kv`
- `local_window`
- `sink_local_window`
- `random`
- 每层独立 top-k
- anchor layer top-k reuse
- anchor reuse + per-layer rerank

指标包括 attention mass recall、top-k overlap、rank correlation、remote token recall、local/sink/remote token 占比。

### State Update Ablation

比较以下变体：

- 输出平均：$0.5(o_{\text{sparse}}+o_{\text{linear}})$；
- 只融合 $v_t$ 和 $\hat{v}_t$；
- 额外 TTT loss term，但固定 $\alpha_t,\beta_t,\gamma_t$；
- gated sparse state write；
- orthogonalized sparse state write；
- sparse branch 只读不写；
- sparse branch 只写不读；
- 有无 confidence gate / conflict control。

核心判断标准是：state write 版本是否在长上下文任务上稳定超过输出融合 baseline，并且 sparse output residual 去掉后收益是否仍然存在。

### Long-context Tasks

任务包括：

- needle-in-a-haystack / passkey retrieval；
- multi-hop QA；
- long summarization；
- repository-level code completion 或 code QA；
- 长文档中的实体一致性追踪。

同时记录 quality、prefill 开销、decode 开销、KV/cache 占用、index cache 占用，以及 anchor reuse 对吞吐的影响。

### Failure Analysis

按以下维度切分：

- token 距离：local、middle、remote；
- layer/head；
- retrieval concentration / score margin；
- top-k 中 sink token 比例；
- $\cos(k_t,\hat{k}_t)$；
- $||\hat{k}_t^\perp||/||\hat{k}_t||$；
- sparse write gate $\alpha_t$ 的分布。

如果 sparse branch 主要选中 sink/local token，或者 $\hat{k}_t^\perp$ 很小，说明它没有提供新的 state direction，需要改进 candidate selection 或 remote-aware gating。

## 12. Risks

1. **检索噪声污染状态**：错误 remote token 一旦写入 $W_t$，会影响后续多个 token。需要 concentration gate、write budget 和小 $\alpha_t$ warmup。
2. **重复写入局部信息**：如果 $\hat{k}_t\approx k_t$，额外写入只是放大局部模式。orthogonalized write 可以缓解。
3. **top-k reuse 不稳定**：相邻层 candidate set 未必一致。需要允许 per-layer rerank 和门控重算。
4. **训练不稳定**：$\alpha_t,\beta_t,\gamma_t$ 同时影响写入和遗忘。可以先固定 $\gamma_t$，只训练 $\alpha_t$，再逐步放开。
5. **贡献边界不清**：必须用 ablation 证明收益来自 sparse-guided state update，而不是额外 sparse readout。

## 13. Summary

最强的版本不是“线性注意力 + 稀疏注意力输出融合”，而是：

> sparse retrieval finds missing memory directions; SGSA writes the residual evidence into recurrent state, so the retrieved memory becomes reusable associative memory for future queries.

数学上，普通 hybrid 的 sparse 分支是一次性的输出修正；SGSA 的 sparse 分支产生 $\Delta W_t$，对未来 query 的影响为 $\Delta W_tq_u$。当 retrieval memory 近似来自目标 dense attention 映射时，SGSA 还能从一阶误差下降角度解释其优越性。这使得方法的创新点从工程混合提升为状态空间记忆机制的改造。
