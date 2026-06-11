"""Synthetic retrieval experiment for SGSA prototype.

Usage:
  conda run -n zjh_dev python experiments/run_sgsa_synthetic_retrieval.py --steps 120
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modelings.modeling_sgsa import SGSAConfig, SGSAForCausalLM, tiny_sgsa_config


FACT_TOKEN = 3
QUERY_TOKEN = 4
PAD_TOKEN = 0


def build_synthetic_batch(
    batch_size: int,
    seq_len: int,
    num_facts: int,
    distance_min: int,
    distance_max: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.full((batch_size, seq_len), PAD_TOKEN, dtype=torch.long, device=device)
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)

    for b in range(batch_size):
        key_id = random.randint(0, num_facts - 1)
        key_token = 10 + key_id
        value_token = 10 + num_facts + key_id
        distance = random.randint(distance_min, distance_max)

        query_pos = seq_len - 2
        fact_pos = max(1, query_pos - distance)
        fact_pos = min(fact_pos, query_pos - 2)

        filler_low = 10 + 2 * num_facts
        filler_high = filler_low + 80
        fillers = torch.randint(filler_low, filler_high, (seq_len,), device=device)
        input_ids[b] = fillers

        input_ids[b, fact_pos - 1] = FACT_TOKEN
        input_ids[b, fact_pos] = key_token
        input_ids[b, fact_pos + 1] = value_token

        input_ids[b, query_pos - 1] = QUERY_TOKEN
        input_ids[b, query_pos] = key_token
        labels[b, query_pos + 1] = value_token

    return input_ids, labels


@torch.no_grad()
def evaluate(model: SGSAForCausalLM, steps: int, batch_size: int, seq_len: int, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for _ in range(steps):
        input_ids, labels = build_synthetic_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_facts=16,
            distance_min=8,
            distance_max=32,
            device=device,
        )
        out = model(input_ids=input_ids, labels=labels)
        logits = out["logits"]
        pred = logits[:, -1, :].argmax(dim=-1)
        target = labels[:, -1]
        mask = target >= 0
        correct += (pred[mask] == target[mask]).sum().item()
        total += mask.sum().item()
    return correct / max(total, 1)


def train_variant(
    name: str,
    config: SGSAConfig,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    device: torch.device,
) -> Dict[str, float]:
    model = SGSAForCausalLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for _ in range(steps):
        input_ids, labels = build_synthetic_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            num_facts=16,
            distance_min=8,
            distance_max=32,
            device=device,
        )
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    acc = evaluate(model, steps=max(8, steps // 10), batch_size=batch_size, seq_len=seq_len, device=device)
    return {"name": name, "accuracy": acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SGSA synthetic retrieval ablations.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = tiny_sgsa_config(vocab_size=256)
    variants = [
        ("state_only", replace(base, sgsa_write_mode="none", sparse_output_lambda=0.0)),
        ("output_hybrid", replace(base, sgsa_write_mode="none", sparse_output_lambda=0.2)),
        ("sgsa_direct", replace(base, sgsa_write_mode="direct", sparse_output_lambda=0.2)),
        ("sgsa_residual", replace(base, sgsa_write_mode="residual", sparse_output_lambda=0.2)),
        ("sgsa_direct_no_sparse_out", replace(base, sgsa_write_mode="direct", sparse_output_lambda=0.0)),
    ]

    print(f"device={device}, steps={args.steps}, batch_size={args.batch_size}, seq_len={args.seq_len}")
    print("variant,accuracy")
    for name, cfg in variants:
        result = train_variant(
            name=name,
            config=cfg,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            device=device,
        )
        print(f"{result['name']},{result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
