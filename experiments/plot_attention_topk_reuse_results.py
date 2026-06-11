#!/usr/bin/env python3
"""Plot and analyze the Qwen attention top-k reuse pilot results.

This script reads the CSV produced by `summarize_attention_topk_reuse.py`,
generates PNG figures, and writes a Markdown report with the plots embedded.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CANDIDATES = ["reuse_kv", "reuse_qkv_previous", "local_window", "sink_local_window", "random"]
CANDIDATE_LABELS = {
    "reuse_kv": "Reuse KV",
    "reuse_qkv_previous": "Prev QKV",
    "local_window": "Local",
    "sink_local_window": "Sink+Local",
    "random": "Random",
}
COLORS = {
    "reuse_kv": "#1f77b4",
    "reuse_qkv_previous": "#2ca02c",
    "local_window": "#7f7f7f",
    "sink_local_window": "#ff7f0e",
    "random": "#9467bd",
}


def to_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    number = float(value)
    if number != number:
        return None
    return number


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def load_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl_stats(path: Path) -> Dict[str, Any]:
    seen = set()
    seq_lens: List[int] = []
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            key = (record["model"], record["sample_id"])
            if key in seen:
                continue
            seen.add(key)
            seq_lens.append(int(record["seq_len"]))
    return {
        "jsonl_rows": rows,
        "samples": len(seen),
        "seq_min": min(seq_lens) if seq_lens else 0,
        "seq_max": max(seq_lens) if seq_lens else 0,
        "seq_avg": mean(seq_lens) or 0.0,
    }


def metric_average(
    rows: List[Dict[str, str]],
    metric: str,
    *,
    candidate: Optional[str] = None,
    model: Optional[str] = None,
    k: str = "256",
    gap: Optional[str] = None,
    layer: Optional[str] = None,
) -> Optional[float]:
    field = f"{metric}_mean"
    values = []
    for row in rows:
        if row["k"] != k:
            continue
        if candidate is not None and row["candidate"] != candidate:
            continue
        if model is not None and row["model"] != model:
            continue
        if gap is not None and row["layer_gap"] != gap:
            continue
        if layer is not None and row["layer"] != layer:
            continue
        values.append(to_float(row.get(field)))
    return mean(values)


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_candidate_quality(rows: List[Dict[str, str]], output_dir: Path) -> Path:
    labels = [CANDIDATE_LABELS[c] for c in CANDIDATES]
    mass = [metric_average(rows, "attention_mass_recall", candidate=c, gap="1") or 0.0 for c in CANDIDATES]
    overlap = [metric_average(rows, "topk_overlap", candidate=c, gap="1") or 0.0 for c in CANDIDATES]
    x = range(len(CANDIDATES))
    width = 0.36

    plt.figure(figsize=(10, 5))
    plt.bar([i - width / 2 for i in x], mass, width=width, label="attention mass recall", color="#1f77b4")
    plt.bar([i + width / 2 for i in x], overlap, width=width, label="top-k overlap", color="#9edae5")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("Candidate quality, k=64, layer_gap=1")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    path = output_dir / "candidate_quality_k64_gap1.png"
    savefig(path)
    return path


def plot_model_comparison(rows: List[Dict[str, str]], output_dir: Path) -> Path:
    models = sorted({row["model"] for row in rows})
    short_models = [model.replace("Qwen3-4B-", "").replace("-2507", "") for model in models]
    selected = ["reuse_kv", "reuse_qkv_previous", "sink_local_window", "random"]
    width = 0.18
    x = list(range(len(models)))

    plt.figure(figsize=(10, 5))
    for offset, candidate in enumerate(selected):
        values = [metric_average(rows, "attention_mass_recall", candidate=candidate, model=model, gap="1") or 0.0 for model in models]
        shifted = [i + (offset - 1.5) * width for i in x]
        plt.bar(shifted, values, width=width, label=CANDIDATE_LABELS[candidate], color=COLORS[candidate])
    plt.xticks(x, short_models)
    plt.ylim(0, 1.05)
    plt.ylabel("attention mass recall")
    plt.title("Model comparison, k=64, layer_gap=1")
    plt.legend(ncol=2)
    plt.grid(axis="y", alpha=0.25)
    path = output_dir / "model_comparison_k64_gap1.png"
    savefig(path)
    return path


def plot_gap_sensitivity(rows: List[Dict[str, str]], output_dir: Path) -> Path:
    gaps = sorted({int(row["layer_gap"]) for row in rows})
    selected = ["reuse_kv", "reuse_qkv_previous", "sink_local_window", "random", "local_window"]

    plt.figure(figsize=(9, 5))
    for candidate in selected:
        values = [
            metric_average(rows, "attention_mass_recall", candidate=candidate, gap=str(gap)) or 0.0
            for gap in gaps
        ]
        plt.plot(gaps, values, marker="o", label=CANDIDATE_LABELS[candidate], color=COLORS[candidate])
    plt.xticks(gaps)
    plt.ylim(0, 1.05)
    plt.xlabel("layer gap")
    plt.ylabel("attention mass recall")
    plt.title("Layer-gap sensitivity, k=64")
    plt.legend(ncol=2)
    plt.grid(alpha=0.25)
    path = output_dir / "gap_sensitivity_k64.png"
    savefig(path)
    return path


def plot_layerwise(rows: List[Dict[str, str]], output_dir: Path) -> Path:
    layers = sorted({int(row["layer"]) for row in rows if row["layer_gap"] == "1"})
    selected = ["reuse_kv", "reuse_qkv_previous", "sink_local_window", "local_window", "random"]

    plt.figure(figsize=(12, 5))
    for candidate in selected:
        values = [
            metric_average(rows, "attention_mass_recall", candidate=candidate, gap="1", layer=str(layer)) or 0.0
            for layer in layers
        ]
        plt.plot(layers, values, marker=".", linewidth=1.8, label=CANDIDATE_LABELS[candidate], color=COLORS[candidate])
    plt.ylim(0, 1.05)
    plt.xlabel("layer")
    plt.ylabel("attention mass recall")
    plt.title("Layer-wise mass recall, k=64, layer_gap=1")
    plt.legend(ncol=3)
    plt.grid(alpha=0.25)
    path = output_dir / "layerwise_mass_recall_k64_gap1.png"
    savefig(path)
    return path


def write_report(rows: List[Dict[str, str]], stats: Dict[str, Any], figures: Dict[str, Path], output_path: Path) -> None:
    reuse_kv = metric_average(rows, "attention_mass_recall", candidate="reuse_kv", gap="1")
    prev_qkv = metric_average(rows, "attention_mass_recall", candidate="reuse_qkv_previous", gap="1")
    sink_local = metric_average(rows, "attention_mass_recall", candidate="sink_local_window", gap="1")
    random = metric_average(rows, "attention_mass_recall", candidate="random", gap="1")
    local = metric_average(rows, "attention_mass_recall", candidate="local_window", gap="1")
    reuse_overlap = metric_average(rows, "topk_overlap", candidate="reuse_kv", gap="1")
    reuse_rank = metric_average(rows, "rank_corr", candidate="reuse_kv", gap="1")
    prev_rank = metric_average(rows, "rank_corr", candidate="reuse_qkv_previous", gap="1")

    rel = {key: path.relative_to(output_path.parent) for key, path in figures.items()}
    lines = [
        "# Attention Top-k Reuse Result Plots",
        "",
        "## 数据范围",
        "",
        f"- 原始指标行数：{stats['jsonl_rows']}，summary 行数：{len(rows)}。",
        f"- 覆盖 3 个 Qwen3-4B 变体，共 {stats['samples']} 个 model/sample 组合。",
        f"- 实际 token 长度为 {stats['seq_min']}-{stats['seq_max']}，平均 {stats['seq_avg']:.1f}。",
        "- 以下图表主要看 `k=64`；由于样本短，`k=128` 没有有效记录。",
        "",
        "## 图 1：候选策略质量",
        "",
        f"![Candidate quality]({rel['candidate']})",
        "",
        f"`reuse_kv` 的 attention mass recall 为 {fmt(reuse_kv)}，top-k overlap 为 {fmt(reuse_overlap)}。这说明当前层 query 打到前层 key 上能找回一部分 gold attention mass，但在这个短 prompt pilot 中没有压过 `random` ({fmt(random)}) 和 `sink_local_window` ({fmt(sink_local)})。",
        "",
        f"`reuse_qkv_previous` 的 mass recall 达到 {fmt(prev_qkv)}，rank corr 为 {fmt(prev_rank)}，说明相邻层自身的 attention token 选择非常相似；但这不是目标部署形态，因为它使用的是前层 query 和前层 key。",
        "",
        "## 图 2：模型间对比",
        "",
        f"![Model comparison]({rel['model']})",
        "",
        "Base、Instruct、Thinking 三个模型的曲线几乎重合。本次 pilot 不能支持模型差异结论，只能说明脚本在三个权重上都能稳定跑通。",
        "",
        "## 图 3：layer gap 敏感性",
        "",
        f"![Gap sensitivity]({rel['gap']})",
        "",
        "从 gap 1 到 gap 4 没有出现明显下降，甚至 `reuse_kv` 略有上升。这更可能是短序列和大 k 造成的饱和现象，不应据此认定 dense anchor 可以隔 4 层仍无损。",
        "",
        "## 图 4：逐层趋势",
        "",
        f"![Layer-wise mass recall]({rel['layer']})",
        "",
        f"`reuse_kv` 随层波动较明显，rank corr 接近 {fmt(reuse_rank)}，说明它抓到的集合有一定重叠，但排序结构和 gold attention 不一致。`sink_local_window` 在中后层很强，提示 sink token 和短距离 token 对当前 pilot 的 mass recall 贡献很大。",
        "",
        "## 结论",
        "",
        "1. 这个 pilot 已经验证了作图和分析链路：Q/K 重建、RoPE、GQA repeat、causal top-k、summary、plot 都能跑通。",
        "2. 当前结果不能直接证明 `reuse_kv` sparse attention 可行，因为样本长度太短，`k=64` 覆盖了过大的历史比例。",
        "3. 真正关键的下一步是用 1024/2048/4096 token 的长 prompt 重跑，并加入远距离 retrieval prompt；届时应重点看 `reuse_kv` 是否显著超过 `random`、`local_window`、`sink_local_window`。",
        "4. 如果长上下文下 `reuse_kv` 的 mass recall 仍稳定高于 baseline，再考虑把它用于 dense-layer KV cache 复用的 top-k sparse attention 设计。",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("outputs/attention_topk_reuse_pilot_summary.csv"))
    parser.add_argument("--jsonl", type=Path, default=Path("outputs/attention_topk_reuse_pilot.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attention_topk_reuse_figures"))
    parser.add_argument("--report", type=Path, default=Path("outputs/attention_topk_reuse_plot_analysis.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_summary(args.summary)
    stats = load_jsonl_stats(args.jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "candidate": plot_candidate_quality(rows, args.output_dir),
        "model": plot_model_comparison(rows, args.output_dir),
        "gap": plot_gap_sensitivity(rows, args.output_dir),
        "layer": plot_layerwise(rows, args.output_dir),
    }
    write_report(rows, stats, figures, args.report)


if __name__ == "__main__":
    main()
