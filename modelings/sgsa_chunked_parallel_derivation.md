# SGSA Chunked Parallel Derivation

这份文档只解释一个问题：**SGSA 的状态更新为什么可以像 GDN / DeltaNet 那样化成分块并行形式。**

我会先复习 GDN / delta rule 的顺序递推，再把它一步步改写成 chunk 内的 lower-triangular linear system，最后说明 SGSA 如何变成“每个 token 两个虚拟写入样本”的同一套求解问题。

## 0. 符号和矩阵方向

默认单 batch、单 head。多 batch、多 head 只是外面多几个 batch 维度。

- $W_t\in\mathbb{R}^{d_v\times d_k}$：状态矩阵，把 key 空间映射到 value 空间。
- $k_t\in\mathbb{R}^{d_k}$：当前写入 key。
- $v_t\in\mathbb{R}^{d_v}$：当前写入 value。
- $q_t\in\mathbb{R}^{d_k}$：读出 query。
- $o_t=W_tq_t\in\mathbb{R}^{d_v}$：状态读出。
- $\gamma_t\in[0,1]$：状态 decay。
- $\beta_t\ge0$：当前 token 的写入步长。

定义 prefix decay：

$$
D_t=\prod_{p=1}^{t}\gamma_p,\quad D_0=1
$$

定义从第 $j$ 个写入传播到第 $i$ 个 residual 之前的 decay：

$$
G_{j\rightarrow i}=\prod_{p=j+1}^{i-1}\gamma_p
$$

注意上式到 $i-1$ 为止，因为第 $i$ 个 residual 用的是 $W_{i-1}$。当 $j=i-1$ 时，这是空乘积，值为 1。

## 1. GDN / Delta Rule 的顺序形式

GDN / DeltaNet 一类状态更新可以写成：

$$
W_t=\gamma_t W_{t-1}+\beta_t(v_t-W_{t-1}k_t)k_t^\top
$$

定义 residual：

$$
r_t=v_t-W_{t-1}k_t
$$

那么更新就是：

$$
W_t=\gamma_t W_{t-1}+\beta_t r_tk_t^\top
$$

这个式子表示：

1. 用旧状态 $W_{t-1}$ 预测当前 value：$W_{t-1}k_t$。
2. 计算误差 $r_t$。
3. 把误差沿着 key 方向 $k_t$ 写进状态。

它也可以从一个 online regression loss 推出：

$$
\mathcal{l}_t(W)=\frac{1}{2}||Wk_t-v_t||^2
$$

梯度是：

$$
\nabla_W\mathcal{l}_t(W)=(Wk_t-v_t)k_t^\top
$$

在 $W_{t-1}$ 处做一步梯度下降：

$$
W_t=W_{t-1}-\beta_t(W_{t-1}k_t-v_t)k_t^\top
$$

再加上 decay，就得到：

$$
W_t=\gamma_t W_{t-1}+\beta_t(v_t-W_{t-1}k_t)k_t^\top
$$

## 2. 为什么顺序形式看起来不能并行

直接看：

$$
r_t=v_t-W_{t-1}k_t
$$

$r_t$ 依赖 $W_{t-1}$，而 $W_{t-1}$ 又依赖所有更早的 residual：

$$
r_1,r_2,\dots,r_{t-1}
$$

所以朴素算法必须按顺序执行：

1. 算 $r_1$，更新 $W_1$。
2. 用 $W_1$ 算 $r_2$，更新 $W_2$。
3. 用 $W_2$ 算 $r_3$，更新 $W_3$。
4. 一直递推。

GDN / DeltaNet 的 chunk trick 是：**在一个 chunk 内，不显式逐 token 更新 $W_t$，而是把所有 residual 之间的依赖写成一个 lower-triangular linear system。**

## 3. 展开一个 chunk 内的状态

考虑一个长度为 $L$ 的 chunk，初始状态为 $W_0$。

### Token 1

$$
W_1=\gamma_1W_0+\beta_1r_1k_1^\top
$$

其中：

$$
r_1=v_1-W_0k_1
$$

### Token 2

$$
W_2=\gamma_2W_1+\beta_2r_2k_2^\top
$$

代入 $W_1$：

$$
W_2=\gamma_2\gamma_1W_0+\gamma_2\beta_1r_1k_1^\top+\beta_2r_2k_2^\top
$$

第 2 个 residual 是：

$$
r_2=v_2-W_1k_2
$$

展开 $W_1k_2$：

$$
W_1k_2=\gamma_1W_0k_2+\beta_1r_1(k_1^\top k_2)
$$

所以：

$$
r_2=v_2-\gamma_1W_0k_2-\beta_1(k_1^\top k_2)r_1
$$

这说明 $r_2$ 对 $r_1$ 的依赖只通过一个标量系数：

$$
\beta_1(k_1^\top k_2)
$$

如果有 decay，则更早写入还会乘对应的 decay product。

### Token 3

先展开 $W_2$：

$$
W_2=\gamma_2\gamma_1W_0+\gamma_2\beta_1r_1k_1^\top+\beta_2r_2k_2^\top
$$

乘到 $k_3$ 上：

$$
W_2k_3
=\gamma_2\gamma_1W_0k_3
+\gamma_2\beta_1r_1(k_1^\top k_3)
+\beta_2r_2(k_2^\top k_3)
$$

所以：

$$
r_3
=v_3-\gamma_2\gamma_1W_0k_3
-\gamma_2\beta_1(k_1^\top k_3)r_1
-\beta_2(k_2^\top k_3)r_2
$$

这已经展示出一般规律：第 $t$ 个 residual 依赖所有过去 residual，依赖系数由 step size、decay 和 key Gram matrix 决定。

## 4. 一般 residual 方程

对任意 $t$，chunk 内旧状态可写成：

$$
W_{t-1}
=D_{t-1}W_0
+\sum_{j<t}\beta_jG_{j\rightarrow t}r_jk_j^\top
$$

其中：

$$
D_{t-1}=\prod_{p=1}^{t-1}\gamma_p
$$

$$
G_{j\rightarrow t}=\prod_{p=j+1}^{t-1}\gamma_p
$$

把它乘到 $k_t$ 上：

$$
W_{t-1}k_t
=D_{t-1}W_0k_t
+\sum_{j<t}\beta_jG_{j\rightarrow t}r_j(k_j^\top k_t)
$$

于是：

$$
r_t
=v_t-D_{t-1}W_0k_t
-\sum_{j<t}\beta_jG_{j\rightarrow t}(k_j^\top k_t)r_j
$$

定义不依赖未知 residual 的部分：

$$
b_t=v_t-D_{t-1}W_0k_t
$$

则：

$$
r_t=b_t-\sum_{j<t}\beta_jG_{j\rightarrow t}(k_j^\top k_t)r_j
$$

移项：

$$
r_t+\sum_{j<t}\beta_jG_{j\rightarrow t}(k_j^\top k_t)r_j=b_t
$$

这就是 lower-triangular system 的逐行形式。

## 5. GDN 的 chunk matrix form

把所有 residual 堆成：

$$
R=
\begin{bmatrix}
r_1^\top\\
r_2^\top\\
\vdots\\
r_L^\top
\end{bmatrix}
\in\mathbb{R}^{L\times d_v}
$$

把所有 $b_t$ 堆成：

$$
B=
\begin{bmatrix}
b_1^\top\\
b_2^\top\\
\vdots\\
b_L^\top
\end{bmatrix}
\in\mathbb{R}^{L\times d_v}
$$

定义 strictly lower-triangular matrix $T\in\mathbb{R}^{L\times L}$：

$$
T_{t,j}=
\begin{cases}
\beta_jG_{j\rightarrow t}(k_j^\top k_t), & j<t\\
0, & j\ge t
\end{cases}
$$

那么全部 residual 方程就是：

$$
(I+T)R=B
$$

因此：

$$
R=(I+T)^{-1}B
$$

这一步就是 chunk 内并行化的核心。我们不是顺序更新 $W_1,W_2,\dots,W_L$，而是一次构造 $T$ 和 $B$，然后做 triangular solve。

## 6. 如何并行构造 T 和 B

把 keys 堆成：

$$
K=
\begin{bmatrix}
k_1^\top\\
k_2^\top\\
\vdots\\
k_L^\top
\end{bmatrix}
\in\mathbb{R}^{L\times d_k}
$$

把 values 堆成：

$$
V=
\begin{bmatrix}
v_1^\top\\
v_2^\top\\
\vdots\\
v_L^\top
\end{bmatrix}
\in\mathbb{R}^{L\times d_v}
$$

先算 key Gram matrix：

$$
G^K=KK^\top
$$

其中：

$$
G^K_{t,j}=k_t^\top k_j
$$

再构造 decay matrix：

$$
G^\gamma_{t,j}=
\begin{cases}
\prod_{p=j+1}^{t-1}\gamma_p, & j<t\\
0, & j\ge t
\end{cases}
$$

于是：

$$
T_{t,j}=\mathbf{1}[j<t]\beta_jG^\gamma_{t,j}G^K_{t,j}
$$

再算 $B$。先计算所有初始预测：

$$
P_t=W_0k_t
$$

按行堆起来：

$$
P=KW_0^\top\in\mathbb{R}^{L\times d_v}
$$

所以：

$$
B=V-\operatorname{diag}(D_0,D_1,\dots,D_{L-1})P
$$

现在 $T$ 和 $B$ 都可以并行构造，接着求：

$$
(I+T)R=B
$$

## 7. chunk 末状态

得到 $R$ 后，chunk 末状态是：

$$
W_L
=D_LW_0
+\sum_{j=1}^{L}\beta_j
\left(\prod_{p=j+1}^{L}\gamma_p\right)
r_jk_j^\top
$$

定义：

$$
H_j=\prod_{p=j+1}^{L}\gamma_p
$$

则：

$$
W_L=D_LW_0+\sum_{j=1}^{L}\beta_jH_jr_jk_j^\top
$$

矩阵形式是：

$$
W_L=D_LW_0+R^\top\operatorname{diag}(\beta_1H_1,\dots,\beta_LH_L)K
$$

形状检查：

- $R^\top\in\mathbb{R}^{d_v\times L}$
- $\operatorname{diag}(\beta H)\in\mathbb{R}^{L\times L}$
- $K\in\mathbb{R}^{L\times d_k}$

所以输出是：

$$
d_v\times d_k
$$

和 $W_L$ 一致。

## 8. chunk 内 readout

前面第 5–7 节解决的是：**不显式顺序更新 $W_t$，一次性求出所有 residual $R$ 和 chunk 末状态 $W_L$。**

但模型真正要用的输出是：

$$
o_t=W_tq_t\in\mathbb{R}^{d_v}
$$

也就是每个 token 位置都要做一次 state readout。这一节说明：**在已经得到 $R=(I+T)^{-1}B$ 之后，chunk 内所有 $o_t$ 也可以并行算，不需要再顺序递推 $W_1,\dots,W_L$。**

### 8.1 从 $W_t$ 展开到 readout

和第 4 节对 $W_{t-1}$ 的展开完全同构，只是求和上限变成 $j\le t$，并且 decay product 要传播到第 $t$ 步本身：

$$
W_t
=D_tW_0
+\sum_{j\le t}\beta_j
\left(\prod_{p=j+1}^{t}\gamma_p\right)
r_jk_j^\top
$$

定义 readout 用的 decay product：

$$
H_{j\rightarrow t}^{\text{read}}
=\prod_{p=j+1}^{t}\gamma_p
$$

当 $j=t$ 时这是空乘积，值为 $1$。注意它和 residual 方程里的 $G_{j\rightarrow t}$ 差一个因子：

$$
G_{j\rightarrow t}=\prod_{p=j+1}^{t-1}\gamma_p
$$

$$
H_{j\rightarrow t}^{\text{read}}
=
\begin{cases}
G_{j\rightarrow t}\cdot\gamma_t, & j<t\\
1, & j=t
\end{cases}
$$

直觉上：$r_j$ 是在第 $j$ 步写入 state 的 correction；readout 用的是更新后的 $W_t$，所以 $j<t$ 的写入还要多经历一步 $\gamma_t$ decay。

两边右乘 $q_t$：

$$
o_t
=D_tW_0q_t
+\sum_{j\le t}\beta_j
H_{j\rightarrow t}^{\text{read}}
r_j(k_j^\top q_t)
$$

这就是 readout 的两部分：

1. **baseline**：$D_tW_0q_t$，只依赖初始状态 $W_0$ 和 prefix decay。
2. **memory**：对所有过去 residual 做 causal weighted sum，权重由 $\beta_j$、decay product 和 query-key 相似度 $k_j^\top q_t$ 共同决定。

### 8.2 矩阵形式：一次算出所有 $o_t$

把 queries 按行堆成：

$$
Q=
\begin{bmatrix}
q_1^\top\\
q_2^\top\\
\vdots\\
q_L^\top
\end{bmatrix}
\in\mathbb{R}^{L\times d_k}
$$

把所有输出按行堆成：

$$
O=
\begin{bmatrix}
o_1^\top\\
o_2^\top\\
\vdots\\
o_L^\top
\end{bmatrix}
\in\mathbb{R}^{L\times d_v}
$$

baseline 部分：

$$
O^{(0)}
=\operatorname{diag}(D_1,D_2,\dots,D_L)\,QW_0^\top
$$

其中 $(QW_0^\top)_t=W_0q_t$。

memory 部分定义 strictly lower-triangular（含对角）的 readout kernel：

$$
A_{t,j}^{\text{read}}
=
\begin{cases}
\beta_j
H_{j\rightarrow t}^{\text{read}}
(k_j^\top q_t), & j\le t\\
0, & j>t
\end{cases}
$$

则：

$$
O=O^{(0)}+A^{\text{read}}R
$$

形状检查：

- $A^{\text{read}}\in\mathbb{R}^{L\times L}$
- $R\in\mathbb{R}^{L\times d_v}$
- 所以 $A^{\text{read}}R\in\mathbb{R}^{L\times d_v}$

和 $O$ 一致。

### 8.3 用 prefix decay 并行构造 $A^{\text{read}}$

和第 6 节构造 $T$ 的方法平行。先 recall：

$$
D_t=\prod_{p=1}^{t}\gamma_p,\quad D_0=1
$$

则 readout decay product 可以写成比值：

$$
H_{j\rightarrow t}^{\text{read}}=\frac{D_t}{D_j}
\qquad (j\le t)
$$

验证：$j=t$ 时 $D_t/D_t=1$；$j<t$ 时 $D_t/D_j=\gamma_{j+1}\cdots\gamma_t$。

再算 query-key Gram：

$$
G^{QK}=QK^\top
$$

其中：

$$
G^{QK}_{t,j}=q_t^\top k_j
$$

于是：

$$
A_{t,j}^{\text{read}}
=\mathbf{1}[j\le t]\,
\beta_j
\frac{D_t}{D_j}
G^{QK}_{t,j}
$$

全部 readout：

$$
O
=
\operatorname{diag}(D_1,\dots,D_L)\,QW_0^\top
+
\left(
\mathbf{1}[j\le t]\odot
\operatorname{diag}(\beta)\,
(D_t/D_j)\odot
G^{QK}
\right)R
$$

这里 $\odot$ 表示按元素乘，$(D_t/D_j)$ 是一个 $L\times L$ 矩阵，第 $(t,j)$ 元素是 $D_t/D_j$。

**计算顺序**（在已有 $R$ 的前提下）：

1. 算 prefix decay $D_1,\dots,D_L$。
2. 算 $G^{QK}=QK^\top$ 和 baseline $O^{(0)}=\operatorname{diag}(D)QW_0^\top$。
3. 构造 $A^{\text{read}}$（只用 $\beta$、$D$、$G^{QK}$）。
4. $O=O^{(0)}+A^{\text{read}}R$。

第 1–3 步都不依赖 $R$，可以和 triangular solve 的前置构造部分重叠；第 4 步在得到 $R$ 后做一次矩阵乘即可。

### 8.4 和 $T$ 的对比：solve 与 readout 用不同的 kernel

容易混淆的是 $T$ 和 $A^{\text{read}}$。它们都是 lower-triangular causal kernel，但作用不同：

| | $T$（residual solve） | $A^{\text{read}}$（readout） |
| --- | --- | --- |
| 目的 | 解 $(I+T)R=B$ | 算 $O=O^{(0)}+A^{\text{read}}R$ |
| 内积 | $k_j^\top k_t$ | $k_j^\top q_t$ |
| decay | $G_{j\rightarrow t}=D_{t-1}/D_j$（到 $t-1$） | $H_{j\rightarrow t}^{\text{read}}=D_t/D_j$（到 $t$） |
| 对角 | 严格下三角，$T_{t,t}=0$ | 含对角，$A^{\text{read}}_{t,t}=\beta_t$ |

关键差异：

- **solve** 里的 $T_{t,j}$ 描述“第 $j$ 个 residual 通过旧状态 $W_{t-1}$ 影响第 $t$ 个 residual 的预测”。
- **readout** 里的 $A^{\text{read}}_{t,j}$ 描述“第 $j$ 个 residual 写入后，传播到 readout 时刻 $t$ 的贡献”。

所以 triangular solve 和 readout 共享同一个 $R$，但各自有独立的 $L\times L$ kernel matrix。

### 8.5 causal attention 视角

把 readout 的 memory 项单独写出来：

$$
o_t^{\text{mem}}
=\sum_{j\le t}
\underbrace{
\beta_j
H_{j\rightarrow t}^{\text{read}}
(k_j^\top q_t)
}_{\text{score }s_{t,j}}
r_j
$$

这就是一个 causal linear attention：

- **query**：$q_t$
- **key**：$k_j$
- **value**：residual $r_j$（不是原始 $v_j$）
- **score**：$s_{t,j}=\beta_j H_{j\rightarrow t}^{\text{read}}(k_j^\top q_t)$

和 softmax attention 的区别：

1. 没有 $\exp$ 和归一化；score 是线性的。
2. value 是 triangular solve 解出的 $r_j$，已经编码了“过去写入之间的互相修正”。
3. 额外乘了 step size $\beta_j$ 和 decay product。

因此 GDN chunk 的完整流程不是“先 attention 再 update”，而是：

1. 先通过 $(I+T)R=B$ 解出所有 correction $r_j$；
2. 再用 $A^{\text{read}}$ 把这些 correction 读出来。

### 8.6 概念伪代码

```python
# Inputs (same chunk as sections 5-7):
# W0: (dv, dk)
# gamma: (L,)
# K: (L, dk), V: (L, dv), Q: (L, dk)
# beta: (L,)
# R: (L, dv)   # already from (I+T)^{-1} B

# 1. Prefix decay: D[t] = prod_{p=1..t} gamma[p], D[0]=1  (1-indexed token id t)
D = prefix_prod(gamma)                 # length L

# 2. Baseline readout
O0 = diag(D) @ (Q @ W0.T)              # (L, dv)

# 3. Readout kernel
GQK = Q @ K.T                          # (L, L), GQK[t,j] = q_t · k_j
decay_ratio = D[:, None] / D[None, :]  # (L, L), [t,j] = D_t / D_j
A_read = tril(beta[None, :] * decay_ratio * GQK)  # lower incl. diagonal

# 4. Memory + total output
O = O0 + A_read @ R                    # (L, dv)
```

如果只需要 chunk 末状态 $W_L$ 而不需要中间 $o_t$，可以跳过整节 readout，直接用第 7 节的矩阵公式。实际训练中通常**两者都要**：$O$ 供当前层输出和反向传播，$W_L$ 供下一 chunk 递推。

### 8.7 小结

GDN 的分块并行可以概括为：

1. 并行构造 $T$ 和 $B$。
2. 用 triangular solve 得到所有 residual $R$。
3. 用 readout kernel $A^{\text{read}}$ 并行得到 chunk 内所有 $o_t$。
4. 用矩阵公式得到 chunk 末状态 $W_L$。

其中第 2 步和第 3 步共享 $R$，但 $T$ 与 $A^{\text{read}}$ 是两个不同的 causal kernel。

## 9. SGSA 的顺序形式

SGSA 的状态更新是：

$$
W_t=\gamma_t W_{t-1}
+\beta_t(v_t-W_{t-1}k_t)k_t^\top
+\alpha_t(\hat v_t-W_{t-1}\tilde k_t)\tilde k_t^\top
$$

其中：

- $(k_t,v_t,\beta_t)$ 是当前 token 的 local write。
- $(\tilde k_t,\hat v_t,\alpha_t)$ 是 sparse retrieval write。
- $\tilde k_t$ 可以等于 $\hat k_t$，也可以等于 residual subspace 版本 $\hat k_t^\perp$。

为了分块并行，必须采用一个关键约定：

> local write 和 sparse write 都读取同一个旧状态 $W_{t-1}$。

也就是说，我们不是做：

$$
W_{t-1}
\rightarrow \text{local write}
\rightarrow \text{sparse write}
\rightarrow W_t
$$

而是做：

$$
W_{t-1}
\rightarrow
\begin{cases}
r_{t,0}=v_t-W_{t-1}k_t\\
r_{t,1}=\hat v_t-W_{t-1}\tilde k_t
\end{cases}
\rightarrow W_t
$$

然后：

$$
W_t=\gamma_tW_{t-1}
+\beta_tr_{t,0}k_t^\top
+\alpha_tr_{t,1}\tilde k_t^\top
$$

这个“同时读取旧状态”的约定非常重要。如果 sparse write 读取的是 local write 之后的状态，那么同一个 token 内部也会形成顺序依赖，chunk matrix 需要额外处理同 token 的 ordering；而 SGSA 的设计目标是让两个写入都是同一时刻的两个监督样本。

## 10. 把 SGSA 看成虚拟样本序列

对每个真实 token $t$，创建两个虚拟写入样本：

$$
(\kappa_{t,0},\nu_{t,0},\lambda_{t,0})=(k_t,v_t,\beta_t)
$$

$$
(\kappa_{t,1},\nu_{t,1},\lambda_{t,1})=(\tilde k_t,\hat v_t,\alpha_t)
$$

其中：

- $\kappa$ 是写入 key。
- $\nu$ 是写入 target value。
- $\lambda$ 是写入步长。
- 下标 $0$ 表示 local write。
- 下标 $1$ 表示 sparse write。

SGSA 更新可以写成：

$$
W_t=\gamma_tW_{t-1}
+\sum_{r\in\{0,1\}}
\lambda_{t,r}
(\nu_{t,r}-W_{t-1}\kappa_{t,r})
\kappa_{t,r}^\top
$$

现在把一个 chunk 内 $L$ 个真实 token 展平成 $M=2L$ 个虚拟样本。

我们用 $i=1,\dots,M$ 表示虚拟样本索引。每个虚拟样本都有：

$$
(\kappa_i,\nu_i,\lambda_i,\tau_i)
$$

其中 $\tau_i$ 是它对应的真实 token 时间。

例如：

$$
i=1:(t=1,r=0)
$$

$$
i=2:(t=1,r=1)
$$

$$
i=3:(t=2,r=0)
$$

$$
i=4:(t=2,r=1)
$$

所以：

$$
\tau_1=\tau_2=1,\quad \tau_3=\tau_4=2
$$

## 11. SGSA residual 的定义

对虚拟样本 $i$，定义 residual：

$$
\rho_i=\nu_i-W_{\tau_i-1}\kappa_i
$$

注意这里是 $W_{\tau_i-1}$，不是“前一个虚拟样本之后的状态”。

这正是“同 token 的 local/sparse write 同时读取旧状态”的数学表达。

真实 token $t$ 的更新写成：

$$
W_t=\gamma_tW_{t-1}
+\sum_{i:\tau_i=t}\lambda_i\rho_i\kappa_i^\top
$$

因为每个 token 有两个虚拟样本，所以：

$$
\sum_{i:\tau_i=t}
=
\text{local sample}+\text{sparse sample}
$$

## 12. SGSA 的状态展开

和 GDN 一样，chunk 内任意 token $t$ 之前的状态可以展开成：

$$
W_{t-1}
=D_{t-1}W_0
+\sum_{j:\tau_j<t}
\lambda_jG_{\tau_j\rightarrow t}
\rho_j\kappa_j^\top
$$

这里求和只包括：

$$
\tau_j<t
$$

也就是只包括更早真实 token 的写入。

**不包括 $\tau_j=t$ 的同 token 写入。**

这是 SGSA 和“把两个虚拟样本简单顺序展开”之间最关键的区别。local 和 sparse 是 simultaneous multi-sample update，而不是同 token 内的 sequential update。

把上式乘到 $\kappa_i$ 上，其中虚拟样本 $i$ 的真实时间是 $\tau_i$：

$$
W_{\tau_i-1}\kappa_i
=D_{\tau_i-1}W_0\kappa_i
+\sum_{j:\tau_j<\tau_i}
\lambda_jG_{\tau_j\rightarrow \tau_i}
\rho_j(\kappa_j^\top\kappa_i)
$$

因此：

$$
\rho_i
=\nu_i-D_{\tau_i-1}W_0\kappa_i
-\sum_{j:\tau_j<\tau_i}
\lambda_jG_{\tau_j\rightarrow \tau_i}
(\kappa_j^\top\kappa_i)\rho_j
$$

定义：

$$
b_i=\nu_i-D_{\tau_i-1}W_0\kappa_i
$$

则：

$$
\rho_i
=b_i-\sum_{j:\tau_j<\tau_i}
\lambda_jG_{\tau_j\rightarrow \tau_i}
(\kappa_j^\top\kappa_i)\rho_j
$$

移项：

$$
\rho_i
+\sum_{j:\tau_j<\tau_i}
\lambda_jG_{\tau_j\rightarrow \tau_i}
(\kappa_j^\top\kappa_i)\rho_j
=b_i
$$

这和 GDN 完全同构，只是：

- GDN 的样本数是 $L$；
- SGSA 的虚拟样本数是 $M=2L$；
- SGSA 的 causal 条件不是 $j<i$，而是 $\tau_j<\tau_i$。

## 13. SGSA 的 chunk matrix form

堆叠所有虚拟 residual：

$$
R=
\begin{bmatrix}
\rho_1^\top\\
\rho_2^\top\\
\vdots\\
\rho_M^\top
\end{bmatrix}
\in\mathbb{R}^{M\times d_v}
$$

堆叠所有 $b_i$：

$$
B=
\begin{bmatrix}
b_1^\top\\
b_2^\top\\
\vdots\\
b_M^\top
\end{bmatrix}
\in\mathbb{R}^{M\times d_v}
$$

定义 $T^{\text{SGSA}}\in\mathbb{R}^{M\times M}$：

$$
T^{\text{SGSA}}_{i,j}
=
\begin{cases}
\lambda_jG_{\tau_j\rightarrow \tau_i}(\kappa_j^\top\kappa_i), & \tau_j<\tau_i\\
0, & \tau_j\ge \tau_i
\end{cases}
$$

那么：

$$
(I+T^{\text{SGSA}})R=B
$$

这就是 SGSA 的 chunked parallel residual solve。

注意它不是普通的严格 $j<i$ causal mask，而是严格真实时间 causal mask：

$$
\tau_j<\tau_i
$$

因此同一个真实 token 的 local sample 和 sparse sample 之间没有依赖：

$$
\tau_j=\tau_i\Rightarrow T^{\text{SGSA}}_{i,j}=0
$$

这保证两个写入都读同一个 $W_{t-1}$。

### 13.1 如何并行构造 $T^{\text{SGSA}}$ 和 $B$

SGSA 的构造方式和 GDN 第 6 节完全平行，只是所有矩阵都从真实 token 维度 $L$ 变成虚拟样本维度 $M=2L$，并且 causal mask 从 $j<i$ 变成 $\tau_j<\tau_i$。

把虚拟写入 key 堆成：

$$
\mathcal{K}=
\begin{bmatrix}
\kappa_1^\top\\
\kappa_2^\top\\
\vdots\\
\kappa_M^\top
\end{bmatrix}
\in\mathbb{R}^{M\times d_k}
$$

把虚拟 target value 堆成：

$$
\mathcal{V}=
\begin{bmatrix}
\nu_1^\top\\
\nu_2^\top\\
\vdots\\
\nu_M^\top
\end{bmatrix}
\in\mathbb{R}^{M\times d_v}
$$

先算虚拟 key 的 Gram matrix：

$$
G^{\kappa}=\mathcal{K}\mathcal{K}^\top
$$

其中：

$$
G^{\kappa}_{i,j}=\kappa_i^\top\kappa_j
$$

再构造真实时间 causal mask：

$$
M^{\tau}_{i,j}=\mathbf{1}[\tau_j<\tau_i]
$$

这个 mask 的形状是 $M\times M$。如果采用 interleave 顺序：

$$
(1,\text{local}),(1,\text{sparse}),(2,\text{local}),(2,\text{sparse}),\dots
$$

那么 $M^{\tau}$ 看起来像 block lower-triangular mask：每个真实 token 对应一个 $2\times2$ block，对角 block 全是 0，下面的 block 全是 1。也就是说，第 $t$ 个 token 的两个虚拟样本可以看见所有更早 token 的两个虚拟样本，但不能互相看见。

decay matrix 只依赖真实时间：

$$
G^{\gamma,\text{SGSA}}_{i,j}
=
\begin{cases}
\prod_{p=\tau_j+1}^{\tau_i-1}\gamma_p, & \tau_j<\tau_i\\
0, & \tau_j\ge\tau_i
\end{cases}
$$

也可以用 prefix decay 写成：

$$
G^{\gamma,\text{SGSA}}_{i,j}
=
\mathbf{1}[\tau_j<\tau_i]\frac{D_{\tau_i-1}}{D_{\tau_j}}
$$

因为：

$$
\frac{D_{\tau_i-1}}{D_{\tau_j}}
=
\gamma_{\tau_j+1}\gamma_{\tau_j+2}\cdots\gamma_{\tau_i-1}
$$

于是：

$$
T^{\text{SGSA}}_{i,j}
=
M^{\tau}_{i,j}
\lambda_j
G^{\gamma,\text{SGSA}}_{i,j}
G^{\kappa}_{i,j}
$$

注意这里 $G^{\kappa}_{i,j}=\kappa_i^\top\kappa_j$，而前面逐项公式写的是 $\kappa_j^\top\kappa_i$，两者是同一个标量。

再构造 $B$。先计算所有虚拟样本在初始状态下的 prediction：

$$
P_i=W_0\kappa_i
$$

按行堆起来：

$$
P=\mathcal{K}W_0^\top\in\mathbb{R}^{M\times d_v}
$$

第 $i$ 个虚拟样本读取的是真实时间 $\tau_i$ 的旧状态 $W_{\tau_i-1}$。只考虑初始状态项时，对应 decay 是 $D_{\tau_i-1}$，所以：

$$
B=\mathcal{V}
-\operatorname{diag}(D_{\tau_1-1},D_{\tau_2-1},\dots,D_{\tau_M-1})P
$$

现在 $T^{\text{SGSA}}$ 和 $B$ 都可以并行构造，然后一次 triangular solve：

$$
(I+T^{\text{SGSA}})R=B
$$

实现上不需要真的 materialize 每个 diagonal matrix。常见写法是：

```python
pred = kappa @ W0.T                         # (M, dv)
B = nu - D[tau - 1, None] * pred            # (M, dv)

gram = kappa @ kappa.T                      # (M, M)
mask = tau[None, :] < tau[:, None]          # mask[i,j] = tau[j] < tau[i]
decay = D[tau - 1, None] / D[tau[None, :]]  # D_{tau_i-1} / D_{tau_j}
T = mask * lam[None, :] * decay * gram

R = solve_triangular(I + T, B)              # (M, dv)
```

上面的 `decay` 在 `mask=False` 的位置没有数学意义，但最后会被 mask 掉；实际实现中可以先构造 mask 再安全填充，避免除以很小的 prefix decay。

## 14. SGSA chunk 末状态

chunk 末状态是：

$$
W_L
=D_LW_0
+\sum_{i=1}^{M}
\lambda_i
\left(\prod_{p=\tau_i+1}^{L}\gamma_p\right)
\rho_i\kappa_i^\top
$$

定义：

$$
H_i=\prod_{p=\tau_i+1}^{L}\gamma_p
$$

则：

$$
W_L=D_LW_0+\sum_{i=1}^{M}\lambda_iH_i\rho_i\kappa_i^\top
$$

把虚拟 keys 堆成：

$$
\mathcal{K}=
\begin{bmatrix}
\kappa_1^\top\\
\kappa_2^\top\\
\vdots\\
\kappa_M^\top
\end{bmatrix}
\in\mathbb{R}^{M\times d_k}
$$

矩阵形式：

$$
W_L
=D_LW_0
+R^\top\operatorname{diag}(\lambda_1H_1,\dots,\lambda_MH_M)\mathcal{K}
$$

形状：

- $R^\top\in\mathbb{R}^{d_v\times M}$
- diagonal matrix 是 $M\times M$
- $\mathcal{K}\in\mathbb{R}^{M\times d_k}$

结果仍然是：

$$
d_v\times d_k
$$

## 15. SGSA chunk 内 readout

SGSA 的 chunk 末状态只用于传给下一个 chunk。当前 chunk 内每个真实 token 的层输出仍然需要：

$$
o_t=W_tq_t
$$

这里的 $W_t$ 是真实 token $t$ 完成 local write 和 sparse write 之后的状态。因为 SGSA 里同 token 的两个虚拟样本都是同时写入，所以 readout 时应该包含：

$$
\tau_i\le t
$$

的所有虚拟写入。

### 15.1 从虚拟样本展开 $W_t$

由第 12 节的状态展开可得，更新到真实 token $t$ 之后：

$$
W_t
=D_tW_0
+\sum_{i:\tau_i\le t}
\lambda_i
\left(\prod_{p=\tau_i+1}^{t}\gamma_p\right)
\rho_i\kappa_i^\top
$$

定义 SGSA readout decay：

$$
H^{\text{read}}_{i\rightarrow t}
=
\prod_{p=\tau_i+1}^{t}\gamma_p
$$

当 $\tau_i=t$ 时，$H^{\text{read}}_{i\rightarrow t}=1$。这表示同一个真实 token 的 local/sparse 两个写入，都会直接进入该 token 的 readout，不再额外乘 $\gamma_t$。

右乘 $q_t$：

$$
o_t
=D_tW_0q_t
+\sum_{i:\tau_i\le t}
\lambda_i
H^{\text{read}}_{i\rightarrow t}
\rho_i(\kappa_i^\top q_t)
$$

这和 GDN 第 8 节的形式完全一致，只是求和对象从真实 token residual $r_j$ 变成虚拟 residual $\rho_i$。

### 15.2 矩阵形式：一次得到所有真实 token 输出

把真实 token 的 query 堆成：

$$
Q=
\begin{bmatrix}
q_1^\top\\
q_2^\top\\
\vdots\\
q_L^\top
\end{bmatrix}
\in\mathbb{R}^{L\times d_k}
$$

把输出堆成：

$$
O=
\begin{bmatrix}
o_1^\top\\
o_2^\top\\
\vdots\\
o_L^\top
\end{bmatrix}
\in\mathbb{R}^{L\times d_v}
$$

baseline 部分仍然是：

$$
O^{(0)}
=\operatorname{diag}(D_1,D_2,\dots,D_L)QW_0^\top
$$

memory 部分需要一个 $L\times M$ 的 readout kernel，而不是 $M\times M$：

$$
A^{\text{SGSA-read}}_{t,i}
=
\begin{cases}
\lambda_i
H^{\text{read}}_{i\rightarrow t}
(\kappa_i^\top q_t), & \tau_i\le t\\
0, & \tau_i>t
\end{cases}
$$

于是：

$$
O
=O^{(0)}+A^{\text{SGSA-read}}R
$$

形状检查：

- $A^{\text{SGSA-read}}\in\mathbb{R}^{L\times M}$
- $R\in\mathbb{R}^{M\times d_v}$
- $A^{\text{SGSA-read}}R\in\mathbb{R}^{L\times d_v}$

所以结果正好是所有真实 token 的输出 $O$。

### 15.3 如何并行构造 $A^{\text{SGSA-read}}$

先算 query 和虚拟 write key 的 Gram matrix：

$$
G^{Q\kappa}=Q\mathcal{K}^\top
$$

其中：

$$
G^{Q\kappa}_{t,i}=q_t^\top\kappa_i
$$

构造 readout mask：

$$
M^{\text{read}}_{t,i}=\mathbf{1}[\tau_i\le t]
$$

注意这里是 $\le$，不是 $<$。原因是 $o_t=W_tq_t$ 读取的是第 $t$ 个 token 写完之后的 state，所以同 token 的 local/sparse 两个写入都应该参与当前 token 的输出。

readout decay 可以用 prefix decay 比值表示：

$$
H^{\text{read}}_{i\rightarrow t}
=\frac{D_t}{D_{\tau_i}}
\qquad(\tau_i\le t)
$$

于是：

$$
A^{\text{SGSA-read}}_{t,i}
=
M^{\text{read}}_{t,i}
\lambda_i
\frac{D_t}{D_{\tau_i}}
G^{Q\kappa}_{t,i}
$$

这一步只依赖 $Q$、$\mathcal{K}$、$\lambda$、$\tau$、$\gamma$，不依赖未知 residual。因此可以和 $T^{\text{SGSA}}$、$B$ 的构造并行准备；真正需要等待 triangular solve 的只有最后一乘：

$$
A^{\text{SGSA-read}}R
$$

概念伪代码：

```python
# Q: (L, dk)
# kappa: (M, dk)
# lam: (M,)
# tau: (M,) with values in 1..L
# D: prefix decay with D[0]=1 and D[t]=prod_{p=1..t} gamma[p]
# R: (M, dv), from SGSA triangular solve

O0 = diag(D[1:]) @ (Q @ W0.T)                 # (L, dv)

gqk = Q @ kappa.T                              # (L, M)
t_real = arange(1, L + 1)                      # (L,)
read_mask = tau[None, :] <= t_real[:, None]    # (L, M)
read_decay = D[1:, None] / D[tau[None, :]]     # D_t / D_{tau_i}

A_read = read_mask * lam[None, :] * read_decay * gqk
O = O0 + A_read @ R                            # (L, dv)
```

和第 13 节一样，`read_decay` 在 mask 外的位置可以先算出来再 mask 掉；更稳健的实现会在 mask 外填 0，避免不必要的数值问题。

### 15.4 与 SGSA residual solve kernel 的区别

SGSA 里至少有两个 causal kernel：

| | $T^{\text{SGSA}}$ | $A^{\text{SGSA-read}}$ |
| --- | --- | --- |
| 形状 | $M\times M$ | $L\times M$ |
| 目的 | 解虚拟 residual $R$ | 从 $R$ 得到真实 token 输出 $O$ |
| 行索引 | 虚拟样本 $i$ | 真实 token $t$ |
| 列索引 | 虚拟样本 $j$ | 虚拟样本 $i$ |
| causal 条件 | $\tau_j<\tau_i$ | $\tau_i\le t$ |
| 内积 | $\kappa_j^\top\kappa_i$ | $\kappa_i^\top q_t$ |
| decay | $D_{\tau_i-1}/D_{\tau_j}$ | $D_t/D_{\tau_i}$ |
| 对同 token 写入 | 不可见 | 可见 |

这个差异很重要：

- residual solve 用的是旧状态 $W_{\tau_i-1}$，所以同 token 两个虚拟样本不能互相影响。
- readout 用的是更新后状态 $W_t$，所以当前 token 的 local/sparse 两个写入都应该出现在 $o_t$ 里。

### 15.5 完整 chunk 内流程

SGSA 单 chunk 的并行流程可以概括为：

1. 先用 local 和 sparse retrieval 结果构造虚拟样本 $(\kappa_i,\nu_i,\lambda_i,\tau_i)$。
2. 并行构造 $B$ 和 $T^{\text{SGSA}}$。
3. triangular solve 得到所有虚拟 residual $R$。
4. 用 $A^{\text{SGSA-read}}R$ 得到当前 chunk 的所有真实 token 输出 $O$。
5. 用第 14 节公式得到 chunk 末状态 $W_L$。

训练和推理通常都需要第 4 步的 $O$；跨 chunk 递推需要第 5 步的 $W_L$。两者共享同一个 $R$，但使用不同的投影 kernel。

## 16. 为什么 gate 必须 state-independent

上面的推导能成立，是因为 $\lambda_i$、$\kappa_i$、$\nu_i$、$\gamma_t$ 在求 residual 之前都已经确定。

对 SGSA 来说，这意味着：

- $\alpha_t$ 必须在 state update 之前确定；
- $\beta_t$ 必须在 state update 之前确定；
- $\gamma_t$ 必须在 state update 之前确定；
- sparse retrieval 的 $S_t,\hat k_t,\hat v_t$ 必须先算好。

如果 gate 依赖：

$$
v_t-W_{t-1}k_t
$$

或者：

$$
\hat v_t-W_{t-1}\hat k_t
$$

那 $\alpha_t$ 本身就依赖未知 residual。此时 residual 方程不再是：

$$
(I+T)R=B
$$

而会变成非线性递推，因为 $T$ 的系数也依赖 $R$。这样就不能做精确的 GDN 式 chunked parallel。

所以主路径 gate 应该只依赖 state-independent 信号，例如：

- retrieval score margin；
- block/token top-k confidence；
- normalized key novelty，比如 $||\hat k_t^\perp||/(||\hat k_t||+\epsilon)$；
- current token representation $x_t$；
- $q_t,k_t,\hat k_t,\hat v_t$ 的范数或点积；
- 距离桶、block id、是否 local/sink/remote。

不应该在主路径 gate 中使用：

- $W_{t-1}k_t$；
- $W_{t-1}\hat k_t$；
- $||v_t-W_{t-1}k_t||$；
- $||\hat v_t-W_{t-1}\hat k_t||$。

这些可以作为 recurrent ablation 或离线诊断，但会破坏精确 chunked parallel。

## 17. SGSA chunked parallel 的伪代码

下面是单 chunk、单 head 的概念伪代码。这里不是最终高性能实现，只是为了对应上面的公式。

```python
# Inputs:
# W0: (dv, dk)
# gamma: (L,)
# q: (L, dk)
# k: (L, dk)
# v: (L, dv)
# k_sparse: (L, dk)      # tilde k_t
# v_sparse: (L, dv)      # hat v_t
# beta: (L,)
# alpha: (L,)

# 1. Build virtual samples.
kappa = interleave(k, k_sparse)       # (M=2L, dk)
nu = interleave(v, v_sparse)          # (M=2L, dv)
lam = interleave(beta, alpha)         # (M,)
tau = tensor([1, 1, 2, 2, ..., L, L]) # (M,)
t_real = tensor([1, 2, ..., L])       # (L,)

# 2. Prefix decay.
# D[0] = 1, D[t] = prod_{p=1..t} gamma[p], with 1-indexed token id t.
D = prefix_prod_with_initial_one(gamma)  # (L+1,)

# 3. Initial prediction term for residual solve.
# b_i = nu_i - D_{tau_i-1} W0 kappa_i
pred = kappa @ W0.T                   # (M, dv)
b = nu - D[tau - 1, None] * pred      # (M, dv)

# 4. Build SGSA residual solve kernel T.
gram = kappa @ kappa.T                # (M, M)
solve_mask = tau[None, :] < tau[:, None]
solve_decay = D[tau - 1, None] / D[tau[None, :]]
T = solve_mask * lam[None, :] * solve_decay * gram

# 5. Solve virtual residuals.
R = solve_triangular(I + T, b)         # (M, dv)

# 6. Chunk readout: O = O0 + A_read @ R.
O0 = diag(D[1:]) @ (q @ W0.T)          # (L, dv)
gqk = q @ kappa.T                      # (L, M)
read_mask = tau[None, :] <= t_real[:, None]
read_decay = D[1:, None] / D[tau[None, :]]
A_read = read_mask * lam[None, :] * read_decay * gqk
O = O0 + A_read @ R                    # (L, dv)

# 7. Chunk final state for the next chunk.
H = D[L] / D[tau]                      # H_i = prod_{p=tau_i+1..L} gamma[p]
weighted_R = R * (lam * H)[:, None]
W_out = D[L] * W0 + weighted_R.T @ kappa
```

这里 `T` 和 `A_read` 是两个不同的 kernel：`T` 解 residual，`A_read` 读出真实 token 输出。两者都可以在 solve 前构造；只有 `R = solve_triangular(...)` 以及之后的 `O = O0 + A_read @ R`、`W_out` 需要依赖求解结果。

## 18. 和普通 GDN 相比到底多了什么

普通 GDN：

$$
M=L
$$

每个 token 一个 sample：

$$
(\kappa_t,\nu_t,\lambda_t)=(k_t,v_t,\beta_t)
$$

SGSA：

$$
M=2L
$$

每个 token 两个 sample：

$$
(k_t,v_t,\beta_t)
$$

和：

$$
(\tilde k_t,\hat v_t,\alpha_t)
$$

普通 GDN 的 mask 是：

$$
j<i
$$

SGSA 的 mask 是：

$$
\tau_j<\tau_i
$$

这就是最核心的变化。

如果错误地使用 $j<i$，那么同 token 的 local sample 会影响 sparse sample，或者 sparse sample 会影响 local sample，取决于 interleave 顺序。这会改变原始 SGSA 公式，因为原始公式要求它们都读取 $W_{t-1}$。

## 19. 关于 residual subspace write

如果使用 residual subspace 版本：

$$
\tilde k_t=\hat k_t^\perp
$$

其中：

$$
\hat k_t^\perp=\hat k_t-\frac{k_t^\top\hat k_t}{||k_t||^2+\epsilon}k_t
$$

这不会改变 chunked parallel 推导。它只改变虚拟样本里的 sparse key：

$$
(\kappa_{t,1},\nu_{t,1},\lambda_{t,1})
=(\hat k_t^\perp,\hat v_t,\alpha_t)
$$

只要 $\hat k_t^\perp$ 在 state update 前已经算好，就仍然能进入同一个 triangular solve。

注意不要把 residual subspace write 理解为“local/sink token 一定被抑制”。它只抑制与当前 $k_t$ 共线的写入分量。sink token 如果是独立全局方向，仍然可以写入。

## 20. 关于 sparse retrieval

SGSA 的 sparse retrieval 也必须在 state update 前完成。典型流程：

1. 用 $q_t$ 对历史 key 或 block summary 打分。
2. 选出 $S_t$ 或 block set。
3. 在候选集合内聚合：

$$
\hat k_t=\sum_{i\in S_t}a_{t,i}k_i
$$

$$
\hat v_t=\sum_{i\in S_t}a_{t,i}v_i
$$

1. 计算 gate $\alpha_t$。
2. 如果使用 residual subspace，计算 $\hat k_t^\perp$。
3. 把 $(\tilde k_t,\hat v_t,\alpha_t)$ 作为虚拟样本放进 chunk solve。

只要 sparse retrieval 不依赖 $W_{t-1}$，就不会破坏 chunked parallel。它可以依赖：

- 当前层输入 hidden state；
- $q_t,k_t,v_t$；
- KV cache；
- block index；
- score margin；
- distance bucket。

## 21. 跨 chunk 的并行 scan

上面讲的是一个 chunk 内怎么求 $W_{\text{out}}$。如果序列很长，会有多个 chunk。

每个 chunk 都定义一个 affine map：

$$
W_{\text{out}}=W_{\text{in}}A_{\text{chunk}}+B_{\text{chunk}}
$$

更准确地说，因为我们使用的是右乘 key-space transition，chunk 对初始 state 的作用可以看成：

$$
W_{\text{out}}=D_LW_{\text{in}}+\Delta W_{\text{chunk}}
$$

在只保留 scalar decay 的简化写法下，chunk 间就是：

$$
W^{(c)}_{\text{out}}
=D^{(c)}W^{(c)}_{\text{in}}+\Delta W^{(c)}
$$

两个 chunk 的组合是：

$$
D^{(2\circ1)}=D^{(2)}D^{(1)}
$$

$$
\Delta W^{(2\circ1)}
=D^{(2)}\Delta W^{(1)}+\Delta W^{(2)}
$$

这可以做 associative scan。

如果使用更一般的 matrix transition：

$$
W_{\text{out}}=W_{\text{in}}A+B
$$

两个 chunk 组合为：

$$
(A_2,B_2)\circ(A_1,B_1)
=(A_1A_2,\;B_1A_2+B_2)
$$

这解释了为什么理论上 affine recurrence 可以 scan。但实际高效实现通常避免显式形成大的 $d_k\times d_k$ transition，而是在 chunk 内用 Gram matrix 和 triangular solve。

## 22. 常见错误

### 错误 1：让 gate 依赖 online residual

如果：

$$
\alpha_t=f(||\hat v_t-W_{t-1}\hat k_t||)
$$

则 $\alpha_t$ 依赖未知 state，triangular matrix 的系数也依赖 residual。这样就不是线性系统。

### 错误 2：同 token 两个虚拟样本使用普通 causal mask

如果 flatten 后直接用 $j<i$，那么同 token 的两个 sample 之间会互相影响。这对应的是 sequential two-step update，不是 SGSA 的 simultaneous two-sample update。

正确 mask 是：

$$
\tau_j<\tau_i
$$

### 错误 3：把 sparse output hybrid 和 sparse state write 混为一谈

输出混合：

$$
o_t=o_t^{state}+\lambda o_t^{sparse}
$$

不会改变当前层 state update 的 triangular system。

SGSA state write：

$$
\alpha_t(\hat v_t-W_{t-1}\tilde k_t)\tilde k_t^\top
$$

会增加虚拟样本，改变 $T$、$B$ 和 $R$。

### 错误 4：认为 residual subspace write 会自动过滤 sink

不会。它只过滤和当前 $k_t$ 共线的部分。sink token 如果提供独立方向，仍然会保留。

## 23. 最短总结

GDN 能 chunked parallel，是因为 residual 满足：

$$
(I+T)R=B
$$

其中 $T$ 是由过去 key 与当前 key 的 Gram matrix、步长、decay 组成的 lower-triangular matrix。

SGSA 能用同一套方法，是因为它可以把每个 token 的 local write 和 sparse write 展平成两个虚拟样本：

$$
(k_t,v_t,\beta_t),\quad(\tilde k_t,\hat v_t,\alpha_t)
$$

然后求解：

$$
(I+T^{\text{SGSA}})R=B
$$

residual solve 唯一需要特别注意的是 mask：

$$
T^{\text{SGSA}}_{i,j}\neq0
\quad\text{only if}\quad
\tau_j<\tau_i
$$

也就是说，同一个真实 token 内的 local/sparse writes 不能互相依赖。这样它们才都对应原始公式里的 $W_{t-1}$。

得到 $R$ 后，chunk 内真实 token 输出也可以并行得到：

$$
O=O^{(0)}+A^{\text{SGSA-read}}R
$$

其中 readout mask 是：

$$
A^{\text{SGSA-read}}_{t,i}\neq0
\quad\text{only if}\quad
\tau_i\le t
$$

这里是 $\le$，因为 $o_t=W_tq_t$ 读取的是 token $t$ 写入后的状态，所以同 token 的 local/sparse writes 应该参与当前 token 的输出。
