#!/usr/bin/env python3
"""Analyze attention top-k reuse after excluding sink and recent tokens.

This variant removes two high-signal but often trivial regions before computing
gold/candidate top-k metrics:

* absolute sink positions: key positions ``[0, exclude_sink_count)``;
* recent positions per query: key positions ``(query_pos - exclude_recent_count, query_pos]``.

With the default values this excludes the first 64 tokens and each query's
nearest 64 causal tokens, then computes the same overlap/mass-recall metrics on
the remaining key positions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from analyze_qwen_attention_topk_reuse import (
    DEFAULT_MODEL_PATHS,
    attention_mass_recall,
    distance_stats,
    filter_query_indices,
    filter_query_metric,
    get_decoder_layers,
    load_model,
    load_prompts,
    overlap_and_jaccard,
    parse_int_list,
    per_head_records,
    reconstruct_layer_qk,
    render_prompt,
    safe_mean,
    select_query_positions,
    seq_bucket,
    spearman_on_gold_topm,
    topk_indices,
)


@torch.no_grad()
def causal_scores_excluding_edges(
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: torch.Tensor,
    attention_mask: torch.Tensor,
    score_chunk_size: int,
    exclude_sink_count: int,
    exclude_recent_count: int,
) -> torch.Tensor:
    selected_query = query.index_select(dim=2, index=query_positions)
    key_t = key.transpose(-2, -1)
    chunks = []
    scale = 1.0 / math.sqrt(query.shape[-1])
    key_positions = torch.arange(key.shape[-2], device=query.device)
    padding_mask = ~attention_mask.to(dtype=torch.bool)[:, None, None, :]
    sink_mask = key_positions < exclude_sink_count

    for start in range(0, selected_query.shape[2], score_chunk_size):
        end = min(start + score_chunk_size, selected_query.shape[2])
        q_chunk = selected_query[:, :, start:end, :]
        q_positions = query_positions[start:end]
        scores = torch.matmul(q_chunk.to(torch.float32), key_t.to(torch.float32)) * scale

        future_mask = key_positions[None, :] > q_positions[:, None]
        recent_mask = key_positions[None, :] > (q_positions[:, None] - exclude_recent_count)
        excluded_mask = future_mask | recent_mask | sink_mask[None, :]
        scores = scores.masked_fill(excluded_mask[None, None, :, :], float("-inf"))
        scores = scores.masked_fill(padding_mask, float("-inf"))
        chunks.append(scores)

    return torch.cat(chunks, dim=2)


def allowed_local_indices(
    query_positions: torch.Tensor,
    max_k: int,
    sink_count: int,
    recent_count: int,
    device: torch.device,
) -> torch.Tensor:
    offsets = torch.arange(max_k, device=device)
    indices = query_positions[:, None] - recent_count - offsets[None, :]
    return indices.clamp_min(sink_count)


def allowed_random_indices(
    query_positions: torch.Tensor,
    max_k: int,
    sink_count: int,
    recent_count: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = random.Random(seed)
    rows = []
    for q_pos in query_positions.tolist():
        valid = list(range(sink_count, max(sink_count, q_pos - recent_count + 1)))
        generator.shuffle(valid)
        if len(valid) < max_k:
            valid.extend([sink_count] * (max_k - len(valid)))
        rows.append(valid[:max_k])
    return torch.tensor(rows, device=device, dtype=torch.long)


def expand_baseline_indices(indices: torch.Tensor, batch: int, heads: int) -> torch.Tensor:
    return indices[None, None, :, :].expand(batch, heads, -1, -1)


def eligible_queries_for_k(
    query_positions: torch.Tensor,
    k: int,
    sink_count: int,
    recent_count: int,
) -> torch.Tensor:
    # Allowed key positions are [sink_count, query_pos - recent_count].
    allowed_count = query_positions - recent_count - sink_count + 1
    return allowed_count >= k


@torch.no_grad()
def analyze_sample_excluding_edges(
    model: torch.nn.Module,
    tokenizer: Any,
    model_label: str,
    sample_id: int,
    prompt: Any,
    args: argparse.Namespace,
    output_handle: Any,
) -> None:
    text = render_prompt(tokenizer, prompt, args.apply_chat_template)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_seq_len,
    )
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(args.device)
    seq_len = int(input_ids.shape[1])
    valid_k_values = [k for k in args.k_values if k <= seq_len]
    if not valid_k_values:
        return

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.clamp_min(0)

    layers = get_decoder_layers(model)
    max_k = max(valid_k_values)
    query_positions = select_query_positions(
        seq_len,
        max_k,
        args.query_stride,
        args.tail_query_count,
        args.device,
        args.query_selection,
    )
    query_positions = query_positions[attention_mask[0, query_positions].bool()]
    if query_positions.numel() == 0:
        return

    qk_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_qk(layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in qk_cache:
            qk_cache[layer_idx] = reconstruct_layer_qk(model, layers[layer_idx], hidden_states[layer_idx], position_ids)
        return qk_cache[layer_idx]

    for layer_idx in range(1, len(layers)):
        q_cur, k_cur = get_qk(layer_idx)
        gold_scores = causal_scores_excluding_edges(
            q_cur,
            k_cur,
            query_positions,
            attention_mask,
            args.score_chunk_size,
            args.exclude_sink_count,
            args.exclude_recent_count,
        )
        gold_idx = topk_indices(gold_scores, max_k)

        for gap in args.layer_gaps:
            ref_idx = layer_idx - gap
            if ref_idx < 0:
                continue
            q_ref, k_ref = get_qk(ref_idx)
            candidates: List[Tuple[str, torch.Tensor, Optional[torch.Tensor]]] = []

            reuse_kv_scores = causal_scores_excluding_edges(
                q_cur,
                k_ref,
                query_positions,
                attention_mask,
                args.score_chunk_size,
                args.exclude_sink_count,
                args.exclude_recent_count,
            )
            candidates.append(("reuse_kv", topk_indices(reuse_kv_scores, max_k), reuse_kv_scores))

            prev_scores = causal_scores_excluding_edges(
                q_ref,
                k_ref,
                query_positions,
                attention_mask,
                args.score_chunk_size,
                args.exclude_sink_count,
                args.exclude_recent_count,
            )
            candidates.append(("reuse_qkv_previous", topk_indices(prev_scores, max_k), prev_scores))

            local_idx = expand_baseline_indices(
                allowed_local_indices(
                    query_positions,
                    max_k,
                    args.exclude_sink_count,
                    args.exclude_recent_count,
                    args.device,
                ),
                q_cur.shape[0],
                q_cur.shape[1],
            )
            candidates.append(("local_after_exclusion", local_idx, None))

            rand_idx = expand_baseline_indices(
                allowed_random_indices(
                    query_positions,
                    max_k,
                    args.exclude_sink_count,
                    args.exclude_recent_count,
                    args.device,
                    seed=args.seed + sample_id * 1000 + layer_idx * 10 + gap,
                ),
                q_cur.shape[0],
                q_cur.shape[1],
            )
            candidates.append(("random_after_exclusion", rand_idx, None))

            for candidate_name, cand_idx, cand_scores in candidates:
                for k in valid_k_values:
                    eligible_query_mask = eligible_queries_for_k(
                        query_positions,
                        k,
                        args.exclude_sink_count,
                        args.exclude_recent_count,
                    )
                    if not bool(eligible_query_mask.any()):
                        continue

                    gold_idx_for_k = filter_query_indices(gold_idx, eligible_query_mask)
                    cand_idx_for_k = filter_query_indices(cand_idx, eligible_query_mask)
                    gold_scores_for_k = filter_query_metric(gold_scores, eligible_query_mask)
                    overlap, jaccard = overlap_and_jaccard(gold_idx_for_k, cand_idx_for_k, k)
                    mass_recall = attention_mass_recall(gold_scores_for_k, cand_idx_for_k, k)
                    metrics = {
                        "topk_overlap": overlap,
                        "jaccard": jaccard,
                        "attention_mass_recall": mass_recall,
                        "gold_topk_mass": attention_mass_recall(gold_scores_for_k, gold_idx_for_k, k),
                    }
                    if cand_scores is not None:
                        cand_scores_for_k = filter_query_metric(cand_scores, eligible_query_mask)
                        metrics["rank_corr"] = spearman_on_gold_topm(
                            gold_idx_for_k,
                            cand_scores_for_k,
                            min(args.rank_top_m, k),
                        )

                    base_record = {
                        "model": model_label,
                        "model_path": args.current_model_path,
                        "sample_id": sample_id,
                        "seq_len": seq_len,
                        "seq_bucket": seq_bucket(seq_len, args.seq_buckets),
                        "layer": layer_idx,
                        "ref_layer": ref_idx,
                        "layer_gap": gap,
                        "k": k,
                        "candidate": candidate_name,
                        "query_selection": args.query_selection,
                        "exclude_sink_count": args.exclude_sink_count,
                        "exclude_recent_count": args.exclude_recent_count,
                        "num_query_positions": int(eligible_query_mask.sum().item()),
                    }
                    distance = distance_stats(cand_idx_for_k, query_positions[eligible_query_mask], k)
                    for record in per_head_records(base_record, metrics, distance, args.aggregate_heads):
                        output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        removable = [idx for idx in qk_cache if idx < layer_idx - max(args.layer_gaps)]
        for idx in removable:
            del qk_cache[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-paths", nargs="+", default=DEFAULT_MODEL_PATHS)
    parser.add_argument("--model-labels", nargs="*", default=None)
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/attention_topk_reuse_exclude_edges.jsonl"))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--k-values", type=parse_int_list, default=parse_int_list("64,128"))
    parser.add_argument("--layer-gaps", type=parse_int_list, default=parse_int_list("1,2,4"))
    parser.add_argument("--seq-buckets", type=parse_int_list, default=parse_int_list("512,1024,2048,4096"))
    parser.add_argument("--query-stride", type=int, default=16)
    parser.add_argument("--tail-query-count", type=int, default=256)
    parser.add_argument(
        "--query-selection",
        choices=["sampled", "all"],
        default="sampled",
        help="Use stride+tail query sampling, or evaluate every valid query position for each k.",
    )
    parser.add_argument("--score-chunk-size", type=int, default=64)
    parser.add_argument("--rank-top-m", type=int, default=64)
    parser.add_argument("--exclude-sink-count", type=int, default=64)
    parser.add_argument("--exclude-recent-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--apply-chat-template", action="store_true")
    parser.add_argument("--aggregate-heads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.prompts_file, args.max_samples)
    labels = args.model_labels or [Path(path).name for path in args.model_paths]
    if len(labels) != len(args.model_paths):
        raise ValueError("--model-labels must match --model-paths length")

    with args.output.open("w", encoding="utf-8") as output_handle:
        for model_path, label in zip(args.model_paths, labels):
            args.current_model_path = model_path
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
            model = load_model(model_path, args)
            for sample_id, prompt in enumerate(prompts):
                analyze_sample_excluding_edges(model, tokenizer, label, sample_id, prompt, args, output_handle)
                output_handle.flush()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
