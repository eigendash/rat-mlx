#!/usr/bin/env python
"""Second check: associative recall, which none of these models learn at this scale.

The task (`rat.task.make_recall_batch`) plants ``n_pairs`` key/value pairs at the
front of a sequence, fills the middle with distractor symbols, and asks for the
values of some of the keys at the end. Unlike the copy task, this needs *matching*
between a query symbol and a key symbol that may sit in a different chunk, so it is
the regime where a chunk summary has to act as a memory slot.

I ran this expecting the larger chunk sizes to struggle and the smaller ones to do
better. They do not: every configuration stays at the constant-answer baseline, so
the run is recorded as a negative result about the task, not as evidence about RAT.
It is here because a check that fails is still information.

Usage:
    python scripts/recall_check.py            # full check, a minute or two
    python scripts/recall_check.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from rat import RATLM, ModelConfig, count_parameters, make_recall_batch
from rat.task import random_baseline_accuracy

SEQ_LEN = 256
N_KEY = 16
N_PAIRS = 16
N_QUERIES = 8
D_MODEL = 64
N_LAYERS = 4
HEADS = 4
WINDOW = 32
BATCH = 32
STEPS = 400
EVAL_BATCH = 32
EVAL_ROUNDS = 6
LR = 3e-3
SEED = 0

CONFIGS = [
    ("c=1 (attention)", 1, False),
    ("c=4", 4, False),
    ("c=16", 16, False),
    ("c=64", 64, False),
    ("c=256 (=T, pure RNN)", 256, False),
    ("c=16 + local attention", 16, True),
]


def to_arrays(batch):
    return (
        mx.array(batch.tokens),
        mx.array(batch.targets),
        mx.array(batch.query_mask),
    )


def make_batch(rng, batch_size):
    return make_recall_batch(
        batch_size, SEQ_LEN, N_KEY, N_PAIRS, N_QUERIES, rng
    )


def train_one(label, chunk, hybrid, rng, steps, log_every):
    mx.random.seed(SEED)
    cfg = ModelConfig(
        vocab_size=1 + 2 * N_KEY,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        heads=HEADS,
        chunk=chunk,
        hybrid=hybrid,
        window=WINDOW,
        rope=False,
    )
    model = RATLM(cfg)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(
        learning_rate=optim.cosine_decay(LR, steps), weight_decay=0.01
    )
    loss_and_grad = nn.value_and_grad(model, model.masked_loss)

    losses = []
    t0 = time.time()
    for step in range(steps):
        tokens, targets, mask = to_arrays(make_batch(rng, BATCH))
        loss, grads = loss_and_grad(tokens, targets, mask)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss))
        if log_every and (step + 1) % log_every == 0:
            print(
                f"    step {step + 1:4d}/{steps}  train loss"
                f" {np.mean(losses[-log_every:]):.4f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    eval_losses = []
    correct = 0
    scored = 0
    for _ in range(EVAL_ROUNDS):
        batch = make_batch(rng, EVAL_BATCH)
        tokens, targets, mask = to_arrays(batch)
        loss = model.masked_loss(tokens, targets, mask)
        logits = model.forward(tokens)
        mx.eval(loss, logits)
        eval_losses.append(float(loss))
        pred = np.array(logits).argmax(axis=-1)
        correct += int((pred[batch.query_mask] == batch.targets[batch.query_mask]).sum())
        scored += batch.n_scored
    return {
        "label": label,
        "chunk": chunk,
        "hybrid": bool(hybrid),
        "params": count_parameters(model),
        "state_floats_per_token": model.chunk_state_per_token(SEQ_LEN, chunk),
        "train_loss_final": float(np.mean(losses[-20:])),
        "held_out_loss": float(np.mean(eval_losses)),
        "accuracy": correct / scored,
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="results/recall_check")
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else (20 if args.quick else STEPS)
    configs = CONFIGS[:2] if args.quick else CONFIGS
    log_every = 0 if args.quick else 100

    print("RAT chunk-size check on associative recall")
    print(
        f"  task: {N_PAIRS} planted key/value pairs, then filler, then"
        f" {N_QUERIES} queries asking for planted values; {1 + 2 * N_KEY}-symbol"
        " vocabulary"
    )
    print(
        f"  model: d_model={D_MODEL}, layers={N_LAYERS}, heads={HEADS};"
        f" training: {steps} steps, batch {BATCH}, seq len {SEQ_LEN},"
        f" AdamW lr {LR} cosine, seed {SEED}"
    )
    print()

    baseline_rng = np.random.default_rng(SEED + 1)
    constant = float(
        np.mean(
            [
                random_baseline_accuracy(make_batch(baseline_rng, EVAL_BATCH))
                for _ in range(EVAL_ROUNDS)
            ]
        )
    )
    print(f"  constant-answer baseline accuracy: {constant:.3f}")
    print()

    results = []
    for label, chunk, hybrid in configs:
        print(f"  {label}: chunk={chunk}, hybrid={hybrid}", flush=True)
        result = train_one(
            label, chunk, hybrid, np.random.default_rng(SEED + 7), steps, log_every
        )
        result["baseline_accuracy"] = constant
        results.append(result)
        print(
            f"    held-out loss {result['held_out_loss']:.4f}  accuracy"
            f" {result['accuracy']:.3f}  state/token"
            f" {result['state_floats_per_token']:.1f}  ({result['seconds']:.0f}s)",
            flush=True,
        )
        print()

    summary = {
        "task": {
            "name": "multi-pair associative recall",
            "seq_len": SEQ_LEN,
            "n_key": N_KEY,
            "n_pairs": N_PAIRS,
            "n_queries": N_QUERIES,
        },
        "setup": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "heads": HEADS,
            "batch": BATCH,
            "steps": steps,
            "lr": LR,
            "seed": SEED,
            "constant_answer_baseline_accuracy": constant,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as fh:
        json.dump(summary, fh, indent=2)

    header = (
        f"{'configuration':<24}{'c':>5}{'hybrid':>7}{'held-out loss':>15}"
        f"{'accuracy':>10}{'state/tok':>11}"
    )
    lines = [
        "RAT chunk-size check on associative recall",
        f"{N_PAIRS} planted key/value pairs, filler, {N_QUERIES} queries;"
        f" sequence length {SEQ_LEN}, vocabulary {1 + 2 * N_KEY}",
        f"model: {N_LAYERS} layers, d_model {D_MODEL}, {HEADS} heads;"
        f" {steps} steps, batch {BATCH}, AdamW lr {LR} cosine, seed {SEED}",
        f"constant-answer baseline accuracy: {constant:.3f}",
        "",
        header,
        "-" * len(header),
    ]
    for r in results:
        lines.append(
            f"{r['label']:<24}{r['chunk']:>5}{str(r['hybrid']):>7}"
            f"{r['held_out_loss']:>15.4f}{r['accuracy']:>10.3f}"
            f"{r['state_floats_per_token']:>11.1f}"
        )
    lines += [
        "",
        "Every configuration sits on the baseline: this task is not learnable by",
        "these models at this scale, so the numbers say nothing about the chunk-size",
        "trade-off. Recorded as a negative result.",
    ]
    with open(args.out + ".txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {args.out}.txt and {args.out}.json")


if __name__ == "__main__":
    main()
