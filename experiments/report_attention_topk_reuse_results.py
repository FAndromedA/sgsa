#!/usr/bin/env python3
"""Generate a Markdown analysis summary for the pilot top-k reuse experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CANDIDATES = ["reuse_kv", "reuse_qkv_previous", "local_window", "sink_local_window", "random"]


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


def load_dataset_stats(path: Path) -> Dict[str, Any]:
    seen = set()
    seq_lens = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["model"], record["sample_id"])
            if key in seen:
                continue
            seen.add(key)
            seq_lens.append(int(record["seq_len"]))
    return {
        "samples": len(seen),
        "seq_min": min(seq_lens) if seq_lens else 0,
        "seq_max": max(seq_lens) if seq_lens else 0,
        "seq_avg": sum(seq_lens) / len(seq_lens) if seq_lens else 0.0,
    }


def metric_average(
    rows: List[Dict[str, str]],
    metric: str,
    *,
    candidate: Optional[str] = None,
    model: Optional[str] = None,
    k: Optional[str] = None,
    gap: Optional[str] = None,
    layer: Optional[str] = None,
) -> Optional[float]:
    values = []
    field = f"{metric}_mean"
    for row in rows:
        if candidate is not None and row["candidate"] != candidate:
            continue
        if model is not None and row["model"] != model:
            continue
        if k is not None and row["k"] != k:
            continue
        if gap is not None and row["layer_gap"] != gap:
            continue
        if layer is not None and row["layer"] != layer:
            continue
        values.append(to_float(row.get(field)))
    return mean(values)


def top_candidates_table(rows: List[Dict[str, str]], k: str, gap: str) -> List[Tuple[str, Optional[float], Optional[float], Optional[float]]]:
    table = []
    for candidate in CANDIDATES:
        table.append(
            (
                candidate,
                metric_average(rows, "attention_mass_recall", candidate=candidate, k=k, gap=gap),
                metric_average(rows, "topk_overlap", candidate=candidate, k=k, gap=gap),
                metric_average(rows, "rank_corr", candidate=candidate, k=k, gap=gap),
            )
        )
    return table


def model_table(rows: List[Dict[str, str]], k: str, gap: str) -> List[Tuple[str, Dict[str, Optional[float]]]]:
    models = sorted({row["model"] for row in rows})
    table = []
    for model in models:
        table.append(
            (
                model,
                {
                    candidate: metric_average(
                        rows,
                        "attention_mass_recall",
                        candidate=candidate,
                        model=model,
                        k=k,
                        gap=gap,
                    )
                    for candidate in CANDIDATES
                },
            )
        )
    return table


def gap_table(rows: List[Dict[str, str]], k: str) -> List[Tuple[str, Dict[str, Optional[float]]]]:
    gaps = sorted({int(row["layer_gap"]) for row in rows})
    table = []
    for gap in gaps:
        gap_s = str(gap)
        table.append(
            (
                gap_s,
                {
                    candidate: metric_average(rows, "attention_mass_recall", candidate=candidate, k=k, gap=gap_s)
                    for candidate in CANDIDATES
                },
            )
        )
    return table


def layer_curve(rows: List[Dict[str, str]], k: str, gap: str) -> List[Tuple[str, Optional[float], Optional[float]]]:
    layers = sorted({int(row["layer"]) for row in rows if row["layer_gap"] == gap})
    return [
        (
            str(layer),
            metric_average(rows, "attention_mass_recall", candidate="reuse_kv", k=k, gap=gap, layer=str(layer)),
            metric_average(rows, "attention_mass_recall", candidate="reuse_qkv_previous", k=k, gap=gap, layer=str(layer)),
        )
        for layer in layers
    ]


def write_markdown(summary_path: Path, jsonl_path: Path, output_path: Path) -> None:
    rows = load_summary(summary_path)
    stats = load_dataset_stats(jsonl_path)
    overview = top_candidates_table(rows, k="64", gap="1")
    by_model = model_table(rows, k="64", gap="1")
    by_gap = gap_table(rows, k="64")
    curve = layer_curve(rows, k="64", gap="1")

    reuse_kv = metric_average(rows, "attention_mass_recall", candidate="reuse_kv", k="64", gap="1")
    previous = metric_average(rows, "attention_mass_recall", candidate="reuse_qkv_previous", k="64", gap="1")
    sink = metric_average(rows, "attention_mass_recall", candidate="sink_local_window", k="64", gap="1")
    random = metric_average(rows, "attention_mass_recall", candidate="random", k="64", gap="1")
    local = metric_average(rows, "attention_mass_recall", candidate="local_window", k="64", gap="1")

    lines = [
        "# Attention Top-k Reuse Pilot Results",
        "",
        "## 数据概况",
        "",
        f"- 输入结果：`{jsonl_path}`",
        f"- 汇总结果：`{summary_path}`",
        f"- 样本数：{stats['samples']} 个 model/sample 组合，序列长度 {stats['seq_min']}-{stats['seq_max']} tokens，平均 {stats['seq_avg']:.1f} tokens。",
        "- 本次 pilot 使用 `k=64,128`、`layer_gap=1,2,4`，并按 head 聚合输出。",
        "",
        "## 关键结论",
        "",
        f"- 在 `k=64, layer_gap=1` 的 pilot 平均上，`reuse_kv` 的 attention mass recall 为 {fmt(reuse_kv)}，低于 `reuse_qkv_previous` 的 {fmt(previous)}，也低于 `sink_local_window` 的 {fmt(sink)}。",
        f"- `local_window` 的平均 mass recall 为 {fmt(local)}，明显低于带 sink 的 baseline；说明当前短样本中 sink token 对覆盖 gold attention mass 很重要。",
        f"- `random` baseline 为 {fmt(random)}，偏高；这是因为本次 prompt 多落在 512 bucket，且 `k=64/128` 占有效历史 token 的比例较大。正式实验需要更长上下文来拉开 candidate 差距。",
        "- Thinking/Instruct/Base 三个模型在本次 pilot 中差异很小，说明脚本路径稳定，但当前样本不足以支持模型差异结论。",
        "- 结论上，当前 pilot 更像工程 smoke test：验证了 Q/K 重建、RoPE、GQA、causal top-k 和指标输出；科学判断应基于 1024/2048/4096 长度桶的完整实验。",
        "",
        "## Candidate 对比：k=64, layer_gap=1",
        "",
        "| candidate | mass recall | top-k overlap | rank corr |",
        "| --- | ---: | ---: | ---: |",
    ]
    for candidate, mass, overlap, rank in overview:
        lines.append(f"| `{candidate}` | {fmt(mass)} | {fmt(overlap)} | {fmt(rank)} |")

    lines.extend(["", "## 三个模型的 mass recall：k=64, layer_gap=1", ""])
    lines.append("| model | reuse_kv | reuse_qkv_previous | local | sink+local | random |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model, values in by_model:
        lines.append(
            f"| `{model}` | {fmt(values['reuse_kv'])} | {fmt(values['reuse_qkv_previous'])} | "
            f"{fmt(values['local_window'])} | {fmt(values['sink_local_window'])} | {fmt(values['random'])} |"
        )

    lines.extend(["", "## Layer gap 衰减：k=64", ""])
    lines.append("| layer_gap | reuse_kv | reuse_qkv_previous | local | sink+local | random |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for gap, values in by_gap:
        lines.append(
            f"| {gap} | {fmt(values['reuse_kv'])} | {fmt(values['reuse_qkv_previous'])} | "
            f"{fmt(values['local_window'])} | {fmt(values['sink_local_window'])} | {fmt(values['random'])} |"
        )

    lines.extend(["", "## 层级趋势摘录：k=64, layer_gap=1", ""])
    lines.append("| layer | reuse_kv | reuse_qkv_previous |")
    lines.append("| ---: | ---: | ---: |")
    for layer, reuse, prev in curve[::4]:
        lines.append(f"| {layer} | {fmt(reuse)} | {fmt(prev)} |")

    lines.extend(
        [
            "",
            "## 下一步建议",
            "",
            "1. 用更长 prompt 重跑，至少覆盖 `1024/2048/4096` bucket；当前短样本会让 random 与 sink baseline 偏强。",
            "2. 去掉 `--aggregate-heads` 做 per-head 分析，检查是否只有少数 head 支撑 reuse。",
            "3. 增加远距离 retrieval 任务，单独统计 gold top-k 中距离大于 512 的 token recall。",
            "4. 如果 `reuse_kv` 在长上下文上仍显著高于 local/random baseline，再把该结论用于 dense-anchor schedule 和 top-k sparse 层设计。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("outputs/attention_topk_reuse_pilot_summary.csv"))
    parser.add_argument("--jsonl", type=Path, default=Path("outputs/attention_topk_reuse_pilot.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/attention_topk_reuse_results.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_markdown(args.summary, args.jsonl, args.output)


if __name__ == "__main__":
    main()
