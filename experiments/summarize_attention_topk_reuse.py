#!/usr/bin/env python3
"""Summarize attention top-k reuse jsonl results into compact CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Tuple


METRIC_FIELDS = [
    "topk_overlap",
    "jaccard",
    "attention_mass_recall",
    "rank_corr",
    "gold_topk_mass",
    "distance_mean",
    "local_128_frac",
    "local_512_frac",
    "sink_4_frac",
]

GROUP_FIELDS = ["model", "seq_bucket", "layer", "layer_gap", "candidate", "k"]


def load_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def finite_metric(record: Dict[str, Any], field: str) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    return value


def group_key(record: Dict[str, Any], fields: List[str]) -> Tuple[Any, ...]:
    return tuple(record.get(field) for field in fields)


def summarize(records: Iterable[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    counts: Dict[Tuple[Any, ...], int] = defaultdict(int)
    for record in records:
        key = group_key(record, fields)
        counts[key] += 1
        for metric in METRIC_FIELDS:
            value = finite_metric(record, metric)
            if value is not None:
                grouped[key][metric].append(value)

    rows: List[Dict[str, Any]] = []
    for key, metric_values in sorted(grouped.items(), key=lambda item: item[0]):
        row = {field: value for field, value in zip(fields, key)}
        row["n"] = counts[key]
        for metric in METRIC_FIELDS:
            values = metric_values.get(metric, [])
            if not values:
                continue
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_median"] = median(values)
            row[f"{metric}_p10"] = percentile(values, 0.10)
            row[f"{metric}_p90"] = percentile(values, 0.90)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/attention_topk_reuse.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/attention_topk_reuse_summary.csv"))
    parser.add_argument("--group-fields", nargs="+", default=GROUP_FIELDS)
    parser.add_argument(
        "--also-head-summary",
        action="store_true",
        help="Also write a per-head summary next to the main output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(load_records(args.input))
    write_csv(args.output, summarize(records, args.group_fields))
    if args.also_head_summary:
        head_output = args.output.with_name(f"{args.output.stem}_by_head{args.output.suffix}")
        write_csv(head_output, summarize(records, args.group_fields + ["head"]))


if __name__ == "__main__":
    main()
