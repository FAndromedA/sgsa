#!/usr/bin/env python3
"""Measure neighboring-layer attention top-k token reuse on local Qwen models.

The script runs a prefill-only analysis.  It does not request full attention
matrices from Transformers; instead it reconstructs each layer's RoPE-applied
Q/K tensors from hidden states and computes top-k scores in chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATHS = [
    "/mnt/jfzn/models/Qwen3-4B-Thinking-2507",
    "/mnt/jfzn/models/Qwen3-4B-Instruct-2507",
    "/mnt/jfzn/models/Qwen3-4B-Base",
]

DEFAULT_PROMPTS = [
    "Explain why sparse attention can reuse nearby-layer key/value caches. Give a concise technical answer.",
    "Solve step by step: A train travels 120 km in 1.5 hours, then 180 km in 2 hours. What is its average speed?",
    "Write a Python function that merges overlapping intervals and explain the complexity.",
    "Summarize the relationship between retrieval tokens, sink tokens, and local windows in long-context transformers.",
    "Given a long document, which tokens are likely to remain important across neighboring transformer layers?",
    "Prove or disprove: if two attention layers have similar key projections, their top-k attended tokens must be identical.",
    "Translate into Chinese and explain any ambiguity: The committee tabled the motion after a short debate.",
    "Design a small ablation study for comparing full attention with top-k sparse attention.",
    "What changes when a model is instruction tuned compared with a base language model?",
    "Implement binary search in C++ and list three common edge cases.",
]


def parse_int_list(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def load_prompts(path: Optional[Path], max_samples: Optional[int]) -> List[Any]:
    if path is None:
        prompts: List[Any] = list(DEFAULT_PROMPTS)
    else:
        prompts = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if path.suffix == ".jsonl":
                    item = json.loads(line)
                    prompts.append(item.get("messages", item.get("prompt", item.get("text", item))))
                else:
                    prompts.append(line)
    return prompts[:max_samples] if max_samples is not None else prompts


def render_prompt(tokenizer: Any, prompt: Any, apply_chat_template: bool) -> str:
    if isinstance(prompt, list):
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Prompt contains messages but tokenizer has no chat template")
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    if isinstance(prompt, dict):
        if "messages" in prompt:
            return render_prompt(tokenizer, prompt["messages"], apply_chat_template=True)
        return str(prompt.get("prompt", prompt.get("text", prompt)))
    if apply_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": str(prompt)}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return str(prompt)


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError("Could not find decoder layers at model.model.layers")
    return layers


def get_rotary_module(model: torch.nn.Module, layer: torch.nn.Module) -> Optional[torch.nn.Module]:
    base = getattr(model, "model", model)
    if hasattr(base, "rotary_emb"):
        return getattr(base, "rotary_emb")
    attn = getattr(layer, "self_attn", None)
    if attn is not None and hasattr(attn, "rotary_emb"):
        return getattr(attn, "rotary_emb")
    return None


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half = x[..., : x.shape[-1] // 2]
    second_half = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def fallback_rope(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.einsum("bs,d->bsd", position_ids.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def get_cos_sin(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotary = get_rotary_module(model, layer)
    if rotary is not None:
        try:
            cos, sin = rotary(hidden_states, position_ids)
            return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)
        except TypeError:
            try:
                cos, sin = rotary(hidden_states, seq_len=hidden_states.shape[1])
                if cos.dim() == 2:
                    cos = cos[position_ids[0]].unsqueeze(0)
                    sin = sin[position_ids[0]].unsqueeze(0)
                return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)
            except TypeError:
                pass

    theta = float(getattr(model.config, "rope_theta", 10000.0))
    return fallback_rope(position_ids, head_dim, theta, hidden_states.dtype, hidden_states.device)


def repeat_kv(hidden_states: torch.Tensor, num_groups: int) -> torch.Tensor:
    if num_groups == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, num_groups, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * num_groups, seq_len, head_dim)


@torch.no_grad()
def reconstruct_layer_qk(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    layer_input: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    attn = layer.self_attn
    normed = layer.input_layernorm(layer_input)

    num_heads = int(getattr(attn, "num_heads", getattr(model.config, "num_attention_heads")))
    num_key_value_heads = int(getattr(attn, "num_key_value_heads", getattr(model.config, "num_key_value_heads", num_heads)))
    query_states = attn.q_proj(normed)
    key_states = attn.k_proj(normed)
    head_dim = query_states.shape[-1] // num_heads
    batch_size, seq_len, _ = query_states.shape

    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim)
    key_states = key_states.view(batch_size, seq_len, num_key_value_heads, head_dim)
    if hasattr(attn, "q_norm"):
        query_states = attn.q_norm(query_states)
    if hasattr(attn, "k_norm"):
        key_states = attn.k_norm(key_states)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    cos, sin = get_cos_sin(model, layer, normed, position_ids, head_dim)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    groups = num_heads // num_key_value_heads
    key_states = repeat_kv(key_states, groups)
    return query_states, key_states


def select_query_positions(
    seq_len: int,
    max_k: int,
    stride: int,
    tail_count: int,
    device: torch.device,
    selection: str,
) -> torch.Tensor:
    if selection == "all":
        return torch.arange(seq_len, device=device, dtype=torch.long)

    min_pos = min(max(max_k - 1, 0), seq_len - 1)
    positions = set(range(min_pos, seq_len, max(1, stride)))
    tail_start = max(min_pos, seq_len - max(0, tail_count))
    positions.update(range(tail_start, seq_len))
    if not positions:
        positions.add(seq_len - 1)
    return torch.tensor(sorted(positions), device=device, dtype=torch.long)


def causal_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    query_positions: torch.Tensor,
    attention_mask: torch.Tensor,
    score_chunk_size: int,
) -> torch.Tensor:
    selected_query = query.index_select(dim=2, index=query_positions)
    key_t = key.transpose(-2, -1)
    chunks = []
    scale = 1.0 / math.sqrt(query.shape[-1])
    key_positions = torch.arange(key.shape[-2], device=query.device)
    padding_mask = ~attention_mask.to(dtype=torch.bool)[:, None, None, :]
    for start in range(0, selected_query.shape[2], score_chunk_size):
        end = min(start + score_chunk_size, selected_query.shape[2])
        q_chunk = selected_query[:, :, start:end, :]
        q_positions = query_positions[start:end]
        scores = torch.matmul(q_chunk.to(torch.float32), key_t.to(torch.float32)) * scale
        future_mask = key_positions[None, :] > q_positions[:, None]
        scores = scores.masked_fill(future_mask[None, None, :, :], float("-inf"))
        scores = scores.masked_fill(padding_mask, float("-inf"))
        chunks.append(scores)
    return torch.cat(chunks, dim=2)


def topk_indices(scores: torch.Tensor, max_k: int) -> torch.Tensor:
    effective_k = min(max_k, scores.shape[-1])
    return torch.topk(scores, k=effective_k, dim=-1).indices


def overlap_and_jaccard(gold_idx: torch.Tensor, cand_idx: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gold = gold_idx[..., :k]
    cand = cand_idx[..., :k]
    match = gold.unsqueeze(-1).eq(cand.unsqueeze(-2)).any(dim=-1)
    intersection = match.sum(dim=-1).to(torch.float32)
    overlap = intersection / float(k)
    jaccard = intersection / (float(2 * k) - intersection).clamp_min(1.0)
    return overlap, jaccard


def attention_mass_recall(gold_scores: torch.Tensor, cand_idx: torch.Tensor, k: int) -> torch.Tensor:
    probs = F.softmax(gold_scores, dim=-1)
    gathered = torch.gather(probs, dim=-1, index=cand_idx[..., :k])
    return gathered.sum(dim=-1)


def spearman_on_gold_topm(gold_idx: torch.Tensor, cand_scores: torch.Tensor, top_m: int) -> torch.Tensor:
    if top_m < 2:
        shape = gold_idx.shape[:-1]
        return torch.full(shape, float("nan"), device=gold_idx.device)
    gold_top = gold_idx[..., :top_m]
    cand_on_gold = torch.gather(cand_scores, dim=-1, index=gold_top)
    order = cand_on_gold.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    rank_values = torch.arange(top_m, device=gold_idx.device, dtype=torch.float32).expand_as(order)
    ranks.scatter_(dim=-1, index=order, src=rank_values)
    gold_ranks = torch.arange(top_m, device=gold_idx.device, dtype=torch.float32)
    gold_centered = gold_ranks - gold_ranks.mean()
    cand_centered = ranks - ranks.mean(dim=-1, keepdim=True)
    numerator = (cand_centered * gold_centered).sum(dim=-1)
    denominator = cand_centered.square().sum(dim=-1).sqrt() * gold_centered.square().sum().sqrt()
    return numerator / denominator.clamp_min(1e-6)


def local_indices(query_positions: torch.Tensor, max_k: int, device: torch.device) -> torch.Tensor:
    offsets = torch.arange(max_k, device=device)
    indices = query_positions[:, None] - offsets[None, :]
    return indices.clamp_min(0)


def sink_local_indices(query_positions: torch.Tensor, max_k: int, sink_count: int, device: torch.device) -> torch.Tensor:
    sink = torch.arange(min(sink_count, max_k), device=device)
    rows = []
    for q_pos in query_positions.tolist():
        values: List[int] = []
        values.extend(int(x) for x in sink.tolist() if int(x) <= q_pos)
        cur = q_pos
        while len(values) < max_k and cur >= 0:
            if cur not in values:
                values.append(cur)
            cur -= 1
        while len(values) < max_k:
            values.append(0)
        rows.append(values[:max_k])
    return torch.tensor(rows, device=device, dtype=torch.long)


def random_indices(query_positions: torch.Tensor, max_k: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = random.Random(seed)
    rows = []
    for q_pos in query_positions.tolist():
        valid = list(range(q_pos + 1))
        generator.shuffle(valid)
        if len(valid) < max_k:
            valid.extend([0] * (max_k - len(valid)))
        rows.append(valid[:max_k])
    return torch.tensor(rows, device=device, dtype=torch.long)


def expand_baseline_indices(indices: torch.Tensor, batch: int, heads: int) -> torch.Tensor:
    return indices[None, None, :, :].expand(batch, heads, -1, -1)


def distance_stats(indices: torch.Tensor, query_positions: torch.Tensor, k: int) -> Dict[str, float]:
    selected = indices[..., :k].to(torch.float32)
    q_pos = query_positions.to(torch.float32)[None, None, :, None]
    distance = q_pos - selected
    return {
        "distance_mean": safe_mean(distance),
        "local_128_frac": safe_mean((distance <= 128).to(torch.float32)),
        "local_512_frac": safe_mean((distance <= 512).to(torch.float32)),
        "sink_4_frac": safe_mean((selected < 4).to(torch.float32)),
    }


def safe_mean(tensor: torch.Tensor) -> float:
    value = tensor.detach().to(torch.float32).mean()
    return float(value.cpu().item())


def seq_bucket(seq_len: int, buckets: Sequence[int]) -> str:
    for bucket in sorted(buckets):
        if seq_len <= bucket:
            return str(bucket)
    return f">{max(buckets)}"


def per_head_records(
    base_record: Dict[str, Any],
    metrics: Dict[str, torch.Tensor],
    distance: Dict[str, float],
    aggregate_heads: bool,
) -> Iterable[Dict[str, Any]]:
    first_metric = next(iter(metrics.values()))
    if aggregate_heads:
        record = dict(base_record)
        record["head"] = "mean"
        for name, tensor in metrics.items():
            record[name] = safe_mean(tensor)
        record.update(distance)
        yield record
        return

    for head in range(first_metric.shape[1]):
        record = dict(base_record)
        record["head"] = head
        for name, tensor in metrics.items():
            record[name] = safe_mean(tensor[:, head, :])
        record.update(distance)
        yield record


def filter_query_metric(tensor: torch.Tensor, eligible_query_mask: torch.Tensor) -> torch.Tensor:
    return tensor.index_select(dim=2, index=eligible_query_mask.nonzero(as_tuple=False).squeeze(-1))


def filter_query_indices(indices: torch.Tensor, eligible_query_mask: torch.Tensor) -> torch.Tensor:
    return indices.index_select(dim=2, index=eligible_query_mask.nonzero(as_tuple=False).squeeze(-1))


@torch.no_grad()
def analyze_sample(
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
        gold_scores = causal_scores(q_cur, k_cur, query_positions, attention_mask, args.score_chunk_size)
        gold_idx = topk_indices(gold_scores, max_k)
        gold_probs_topk = attention_mass_recall(gold_scores, gold_idx, max_k)

        for gap in args.layer_gaps:
            ref_idx = layer_idx - gap
            if ref_idx < 0:
                continue
            q_ref, k_ref = get_qk(ref_idx)
            candidates: List[Tuple[str, torch.Tensor, Optional[torch.Tensor]]] = []

            reuse_kv_scores = causal_scores(q_cur, k_ref, query_positions, attention_mask, args.score_chunk_size)
            candidates.append(("reuse_kv", topk_indices(reuse_kv_scores, max_k), reuse_kv_scores))

            prev_scores = causal_scores(q_ref, k_ref, query_positions, attention_mask, args.score_chunk_size)
            candidates.append(("reuse_qkv_previous", topk_indices(prev_scores, max_k), prev_scores))

            local_idx = expand_baseline_indices(local_indices(query_positions, max_k, args.device), q_cur.shape[0], q_cur.shape[1])
            candidates.append(("local_window", local_idx, None))

            sink_idx = expand_baseline_indices(
                sink_local_indices(query_positions, max_k, args.sink_count, args.device),
                q_cur.shape[0],
                q_cur.shape[1],
            )
            candidates.append(("sink_local_window", sink_idx, None))

            rand_idx = expand_baseline_indices(
                random_indices(query_positions, max_k, args.device, seed=args.seed + sample_id * 1000 + layer_idx * 10 + gap),
                q_cur.shape[0],
                q_cur.shape[1],
            )
            candidates.append(("random", rand_idx, None))

            for candidate_name, cand_idx, cand_scores in candidates:
                for k in valid_k_values:
                    eligible_query_mask = query_positions >= (k - 1)
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
                        "num_query_positions": int(eligible_query_mask.sum().item()),
                    }
                    distance = distance_stats(cand_idx_for_k, query_positions[eligible_query_mask], k)
                    for record in per_head_records(base_record, metrics, distance, args.aggregate_heads):
                        output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        removable = [idx for idx in qk_cache if idx < layer_idx - max(args.layer_gaps)]
        for idx in removable:
            del qk_cache[idx]


def load_model(path: str, args: argparse.Namespace) -> torch.nn.Module:
    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    if args.device_map is None:
        model.to(args.device)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-paths", nargs="+", default=DEFAULT_MODEL_PATHS)
    parser.add_argument("--model-labels", nargs="*", default=None)
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/attention_topk_reuse.jsonl"))
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
    parser.add_argument("--sink-count", type=int, default=4)
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
                analyze_sample(model, tokenizer, label, sample_id, prompt, args, output_handle)
                output_handle.flush()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
