# Attention Token Reuse Experiment

本文档说明如何验证大模型相邻层之间重要 attention tokens 的相似性，以及如何评估 sparse attention 复用前面 dense 层 KV cache 选择 top-k tokens 的可行性。

## 实验目标

目标不是直接证明某个 sparse kernel 加速，而是先验证选 token 的近似是否可靠：

1. 相邻层 `l` 与 `l-gap` 对同一 query token 关注的 key token 集合是否相似。
2. 使用当前层 query `Q_l` 与前层 key `K_{l-gap}` 计算 top-k，是否能覆盖当前层 full attention 的重要 token。
3. Qwen3-4B Thinking、Instruct、Base 三个变体上，这种相似性是否稳定。

## 已实现脚本

- `experiments/analyze_qwen_attention_topk_reuse.py`
  - 加载本地 Qwen3 模型。
  - 只请求 `output_hidden_states=True`，不请求完整 attention matrix。
  - 从每层输入 hidden states 重建 RoPE 后的 `Q/K`。
  - 支持 Qwen3 的 GQA：将 `num_key_value_heads` repeat 到 `num_attention_heads` 后再计算 attention score。
  - 分 chunk 计算 causal `QK^T`，只保留 top-k 与必要指标。
  - 输出 jsonl 结果。

- `experiments/summarize_attention_topk_reuse.py`
  - 读取 jsonl。
  - 按模型、长度桶、层、layer gap、candidate、k 聚合。
  - 输出 CSV，可选输出 per-head summary。

## 候选策略

脚本把当前层 `Q_l K_l^T` 的 top-k 作为 gold，并比较以下 candidate：

- `reuse_kv`：使用 `Q_l K_{l-gap}^T` 选择 top-k，直接对应“复用前层 KV cache 做 sparse token selection”。
- `reuse_qkv_previous`：使用 `Q_{l-gap} K_{l-gap}^T` 选择 top-k，测试更激进的前层全复用方案。
- `local_window`：只选最近 k 个历史 token。
- `sink_local_window`：保留少量 sink tokens，再补最近历史 token。
- `random`：随机历史 token baseline。

## 指标

- `topk_overlap`：`|TopK_gold ∩ TopK_candidate| / k`。
- `jaccard`：集合 Jaccard similarity。
- `attention_mass_recall`：gold attention softmax 概率在 candidate token 集合上的质量总和，是最重要的可行性指标。
- `rank_corr`：在 gold top-M token 上比较 candidate score 排名的 Spearman 相关。
- `distance_mean`：被选 token 到 query token 的平均距离。
- `local_128_frac`、`local_512_frac`：candidate 中局部 token 比例。
- `sink_4_frac`：candidate 中前 4 个 sink token 比例。

## Smoke Test

已跑通的最小验证命令：

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/analyze_qwen_attention_topk_reuse.py \
  --model-paths /mnt/jfzn/models/Qwen3-4B-Base \
  --max-samples 1 \
  --max-seq-len 256 \
  --k-values 16,32 \
  --layer-gaps 1 \
  --query-stride 32 \
  --tail-query-count 32 \
  --score-chunk-size 16 \
  --aggregate-heads \
  --output outputs/attention_topk_reuse_smoke.jsonl
```

汇总命令：

```bash
python experiments/summarize_attention_topk_reuse.py \
  --input outputs/attention_topk_reuse_smoke.jsonl \
  --output outputs/attention_topk_reuse_smoke_summary.csv \
  --also-head-summary
```

该 smoke test 已生成：

- `outputs/attention_topk_reuse_smoke.jsonl`
- `outputs/attention_topk_reuse_smoke_summary.csv`
- `outputs/attention_topk_reuse_smoke_summary_by_head.csv`

## Pilot Run

三模型 pilot 命令：

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/analyze_qwen_attention_topk_reuse.py \
  --model-paths \
    /mnt/jfzn/models/Qwen3-4B-Thinking-2507 \
    /mnt/jfzn/models/Qwen3-4B-Instruct-2507 \
    /mnt/jfzn/models/Qwen3-4B-Base \
  --prompts-file experiments/attention_reuse_pilot_prompts.jsonl \
  --max-samples 10 \
  --max-seq-len 1024 \
  --k-values 64,128 \
  --layer-gaps 1,2,4 \
  --query-stride 16 \
  --tail-query-count 256 \
  --score-chunk-size 64 \
  --aggregate-heads \
  --output outputs/attention_topk_reuse_pilot.jsonl
```

Pilot 汇总：

```bash
python experiments/summarize_attention_topk_reuse.py \
  --input outputs/attention_topk_reuse_pilot.jsonl \
  --output outputs/attention_topk_reuse_pilot_summary.csv \
  --also-head-summary
```

本次 pilot 已跑通并生成：

- `experiments/attention_reuse_pilot_prompts.jsonl`
- `outputs/attention_topk_reuse_pilot.jsonl`，15150 行
- `outputs/attention_topk_reuse_pilot_summary.csv`，1516 行
- `outputs/attention_topk_reuse_pilot_summary_by_head.csv`，1516 行

说明：本次使用 `--aggregate-heads`，因此 `by_head` 文件里的 head 字段仍是 `mean`。如果要真正按 head 展开，去掉分析脚本里的 `--aggregate-heads` 后重新运行。

本次 pilot 主要用于验证代码路径、显存行为和输出格式，不应作为最终科学结论。当前 10 条 prompt 大多落在 `512` 长度桶内，当 `k=64/128` 接近有效历史长度时，`random`、`sink_local_window` 等 baseline 会偏强；正式实验应使用更长的样本，重点比较 `1024/2048/4096` 长度桶。

## 扩展到完整实验

建议准备一个 jsonl prompt 文件，每行可以是：

```json
{"prompt": "plain text prompt"}
```

或 chat messages：

```json
{"messages": [{"role": "user", "content": "your prompt"}]}
```

完整实验示例：

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/analyze_qwen_attention_topk_reuse.py \
  --model-paths \
    /mnt/jfzn/models/Qwen3-4B-Thinking-2507 \
    /mnt/jfzn/models/Qwen3-4B-Instruct-2507 \
    /mnt/jfzn/models/Qwen3-4B-Base \
  --prompts-file data/attention_reuse_prompts.jsonl \
  --max-samples 500 \
  --max-seq-len 4096 \
  --k-values 32,64,128,256,512 \
  --layer-gaps 1,2,4 \
  --query-stride 16 \
  --tail-query-count 256 \
  --score-chunk-size 64 \
  --aggregate-heads \
  --output outputs/attention_topk_reuse_full.jsonl
```

## 结果解读

重点看同一组 `model/layer_gap/k` 下：

1. `reuse_kv` 的 `attention_mass_recall_mean` 是否显著高于 `local_window`、`sink_local_window` 和 `random`。
2. `reuse_kv` 从 `layer_gap=1` 到 `layer_gap=4` 下降是否平滑；如果下降很快，说明 dense anchor 间隔不能太大。
3. `topk_overlap_mean` 与 `attention_mass_recall_mean` 是否一致；如果 overlap 不高但 mass recall 高，说明候选集合抓住了高质量 token，但排序或长尾 top-k 不完全一致。
4. Thinking/Instruct/Base 是否存在系统差异；如果某个模型在推理类 prompt 上复用性更高，应单独报告。

## 注意事项

- 不要用 `output_attentions=True` 跑长上下文 4B 模型，完整 attention tensor 很容易造成显存峰值过高。
- 当前脚本分析的是 prefill/full sequence，不包含逐 token decode 阶段。
- 默认使用 `--aggregate-heads` 控制输出规模；需要分析 head 差异时去掉该参数，或使用汇总脚本的 `--also-head-summary`。
- 如果显存紧张，优先降低 `--max-seq-len`、`--tail-query-count` 和 `--score-chunk-size`。
