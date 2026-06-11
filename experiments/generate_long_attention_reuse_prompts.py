#!/usr/bin/env python3
"""Generate synthetic long-context prompts for attention top-k reuse analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ANCHOR_BLOCK = """
[ANCHOR DEFINITIONS]
alpha_dense_anchor means the nearest previous dense attention layer whose KV cache may be reused.
beta_topk_budget means the number of key tokens retained by a sparse selector.
gamma_layer_gap means the distance between the current layer and the reference layer.
delta_mass_recall means the gold attention probability mass recovered by the candidate token set.
epsilon_retrieval_key means a deliberately distant fact that must be retrieved near the end.
"""

LOCAL_BLOCK = """
[LOCAL ANALYSIS]
The sparse attention selector should keep recent tokens, but recent tokens alone are not enough.
Neighboring transformer layers may share important token identities because the same instruction,
variable definition, delimiter, or retrieval fact remains salient across multiple residual updates.
However, the current layer query may rotate or rescale the score geometry, so reusing the previous
key bank must be evaluated against the current layer's own full-attention top-k set.
"""

FILLER_BLOCK = """
This filler paragraph describes a monitoring log, a small code review, and a reasoning trace.
The engineer reads timestamps, request identifiers, function names, theorem statements, and notes.
Most details are intentionally irrelevant to the final answer, but they create realistic distractors.
Some sentences mention cache pressure, grouped-query attention, rotary position embeddings, and masks.
Other sentences mention deployment windows, dataset curation, and repeated diagnostic summaries.
The important part is that the final question will refer back to anchor definitions near the beginning.
"""

CODE_BLOCK = """
[CODE SNIPPET]
def select_candidate_tokens(query, reference_keys, k, causal_mask):
    scores = query @ reference_keys.transpose(-1, -2)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    return scores.topk(k, dim=-1).indices
"""

QUESTION_BLOCK = """
[FINAL QUESTION]
Using the definitions from the beginning, explain whether alpha_dense_anchor can provide a useful
KV cache for selecting beta_topk_budget tokens when gamma_layer_gap is greater than one. Discuss
how delta_mass_recall should be interpreted, why epsilon_retrieval_key is a stress test for sparse
attention reuse, and why sink tokens plus local windows are necessary baselines.
"""


def build_prompt(target_bucket: int, variant: int) -> str:
    blocks = [
        f"[TASK] Long-context attention reuse prompt. bucket={target_bucket}, variant={variant}.",
        ANCHOR_BLOCK,
        CODE_BLOCK,
    ]
    repeat_count = {1024: 12, 2048: 28, 4096: 62}[target_bucket]
    for idx in range(repeat_count):
        blocks.append(f"[DISTRACTOR {idx:03d}]")
        blocks.append(FILLER_BLOCK)
        if idx % 5 == 0:
            blocks.append(LOCAL_BLOCK)
        if idx % 11 == 0:
            blocks.append(
                f"[DISTANT FACT {idx:03d}] epsilon_retrieval_key remains tied to delta_mass_recall "
                "and must be recovered despite many intervening distractor paragraphs."
            )
    blocks.append(QUESTION_BLOCK)
    return "\n".join(block.strip() for block in blocks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments/attention_reuse_long_prompts.jsonl"))
    parser.add_argument("--variants-per-bucket", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for bucket in (1024, 2048, 4096):
            for variant in range(args.variants_per_bucket):
                prompt = build_prompt(bucket, variant)
                handle.write(json.dumps({"bucket": bucket, "variant": variant, "prompt": prompt}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
