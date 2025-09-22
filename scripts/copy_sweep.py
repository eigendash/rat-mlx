#!/usr/bin/env python
"""Sweep the RAT chunk size (and the hybrid) on a long-range copy task.

Task (`rat.task.make_copy_batch`): a sequence of ``T`` symbols whose second half
repeats the first half, scored on the second half only. Every scored token has to
be read from exactly ``T / 2`` positions earlier, the input distribution is
stationary, and no local statistic predicts the target, so a bounded-reach model
cannot do better than chance. The copy is content-based (an induction head), which
is the retrieval-style access long-context models are measured on.

Every configuration trains the same small decoder from scratch on the same stream
of freshly generated batches and is evaluated on a fixed held-out set, at the
training length and at twice the training length. Only the temporal mixer changes:
the RAT chunk size ``c``, or a hybrid that interleaves RAT with sliding-window
local attention.

Usage:
    python scripts/copy_sweep.py             # full sweep, a few minutes
    python scripts/copy_sweep.py --quick     # tiny run, for a smoke test
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

from rat import RATLM, ModelConfig, count_parameters, make_copy_batch
from rat.task import copy_accuracy_by_distance

SEQ_LEN = 512  # training length
EVAL_LEN = 1024  # held-out length-generalisation check
N_SYMBOLS = 8
D_MODEL = 64
N_LAYERS = 2
HEADS = 4
WINDOW = 32  # sliding window of the hybrid's local-attention layers
BATCH = 32
STEPS = 600
EVAL_BATCH = 32
EVAL_ROUNDS = 4
LR = 3e-3
SEED = 0

# (label, chunk size, hybrid flag)
CONFIGS = [
    ("c=1 (attention)", 1, False),
    ("c=4", 4, False),
    ("c=16", 16, False),
    ("c=64", 64, False),
    ("c=512 (=T, pure RNN)", 512, False),
    ("c=16 + local attention", 16, True),
]


def to_arrays(batch):
    return (
        mx.array(batch.tokens),
        mx.array(batch.targets),
        mx.array(batch.query_mask),
    )


def evaluate(model, batches, batch_len):
    """Held-out loss over scored positions, accuracy, and accuracy by distance."""
    losses = []
    correct = 0
    scored = 0
    by_distance: dict[str, list[float]] = {}
    for batch in batches:
        tokens, targets, mask = to_arrays(batch)
        loss = model.masked_loss(tokens, targets, mask)
        logits = model.forward(tokens)
        mx.eval(loss, logits)
        losses.append(float(loss))
        logits_np = np.array(logits)
        pred = logits_np.argmax(axis=-1)
        correct += int((pred[batch.query_mask] == batch.targets[batch.query_mask]).sum())
        scored += batch.n_scored
        for key, value in copy_accuracy_by_distance(logits_np, batch).items():
            by_distance.setdefault(key, []).append(value)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": correct / scored,
        "accuracy_by_source_distance": {
            key: float(np.mean(vals)) for key, vals in by_distance.items()
        },
    }


def train_one(label, chunk, hybrid, rng, steps, log_every):
    mx.random.seed(SEED)
    cfg = ModelConfig(
        vocab_size=N_SYMBOLS,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        heads=HEADS,
        chunk=chunk,
        hybrid=hybrid,
        window=WINDOW,
    )
    model = RATLM(cfg)
    mx.eval(model.parameters())
    params = count_parameters(model)

    schedule = optim.cosine_decay(LR, steps)
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, model.masked_loss)

    train_losses = []
    t0 = time.time()
    for step in range(steps):
        tokens, targets, mask = to_arrays(make_copy_batch(BATCH, SEQ_LEN, N_SYMBOLS, rng))
        loss, grads = loss_and_grad(tokens, targets, mask)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        train_losses.append(float(loss))
        if log_every and (step + 1) % log_every == 0:
            print(
                f"    step {step + 1:4d}/{steps}  train loss"
                f" {np.mean(train_losses[-log_every:]):.4f}"
                f"  ({time.time() - t0:.0f}s)",
                flush=True,
            )

    evals = {}
    for name, length, rounds in (
        ("train_len", SEQ_LEN, EVAL_ROUNDS),
        ("double_len", EVAL_LEN, max(EVAL_ROUNDS // 2, 1)),
    ):
        if length % chunk != 0:
            continue
        batches = [
            make_copy_batch(EVAL_BATCH, length, N_SYMBOLS, rng) for _ in range(rounds)
        ]
        evals[name] = evaluate(model, batches, length)

    return {
        "label": label,
        "chunk": chunk,
        "hybrid": bool(hybrid),
        "params": params,
        "state_floats_per_token": model.chunk_state_per_token(SEQ_LEN, chunk),
        "state_floats_per_token_long": model.chunk_state_per_token(EVAL_LEN, chunk),
        "train_loss_final": float(np.mean(train_losses[-20:])),
        "held_out": evals,
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="results/copy_sweep")
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else (30 if args.quick else STEPS)
    configs = CONFIGS[:2] if args.quick else CONFIGS
    log_every = 0 if args.quick else 100

    print("RAT chunk-size sweep on long-range copy")
    print(
        f"  task: second half of the sequence repeats the first half, converted to"
        f" {N_SYMBOLS} symbols, scored only on the second half"
    )
    print(
        f"  model: d_model={D_MODEL}, layers={N_LAYERS}, heads={HEADS},"
        f" {count_parameters(RATLM(ModelConfig(vocab_size=N_SYMBOLS, d_model=D_MODEL, n_layers=N_LAYERS, heads=HEADS, chunk=1)))}"
        " parameters (same for every configuration)"
    )
    print(
        f"  training: {steps} steps, batch {BATCH}, seq len {SEQ_LEN}, AdamW lr {LR}"
        f" cosine to 0, grad-norm clip 1.0, seed {SEED}"
    )
    print(
        f"  evaluation: {EVAL_ROUNDS} held-out batches at len {SEQ_LEN}"
        f" and {max(EVAL_ROUNDS // 2, 1)} at len {EVAL_LEN}"
    )
    print()

    baseline_rng = np.random.default_rng(SEED + 1)
    baseline = [
        make_copy_batch(EVAL_BATCH, SEQ_LEN, N_SYMBOLS, baseline_rng)
        for _ in range(EVAL_ROUNDS)
    ]
    from rat.task import random_baseline_accuracy

    constant = float(np.mean([random_baseline_accuracy(b) for b in baseline]))
    print(f"  constant-answer baseline accuracy: {constant:.3f}")
    print()

    results = []
    for label, chunk, hybrid in configs:
        print(f"  {label}: chunk={chunk}, hybrid={hybrid}", flush=True)
        # The same batch stream for every configuration, so differences are the
        # architecture and not the data.
        result = train_one(label, chunk, hybrid, np.random.default_rng(SEED + 7), steps, log_every)
        result["baseline_accuracy"] = constant
        results.append(result)
        at_len = result["held_out"].get("train_len")
        long_len = result["held_out"].get("double_len")
        tail = (
            f"  len{EVAL_LEN} acc {long_len['accuracy']:.3f}"
            if long_len is not None
            else f"  len{EVAL_LEN} not evaluated (not a multiple of {chunk})"
        )
        print(
            f"    held-out loss {at_len['loss']:.4f}  accuracy {at_len['accuracy']:.3f}"
            f"  state/token {result['state_floats_per_token']:.1f}  params"
            f" {result['params']}{tail}  ({result['seconds']:.0f}s)",
            flush=True,
        )
        print()

    summary = {
        "task": {
            "name": "long-range copy (second half repeats the first half)",
            "seq_len": SEQ_LEN,
            "eval_len": EVAL_LEN,
            "n_symbols": N_SYMBOLS,
        },
        "setup": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "heads": HEADS,
            "batch": BATCH,
            "steps": steps,
            "lr": LR,
            "seed": SEED,
            "window": WINDOW,
            "optimizer": "AdamW, cosine decay, grad-norm clip 1.0",
            "constant_answer_baseline_accuracy": constant,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + ".json", "w") as fh:
        json.dump(summary, fh, indent=2)

    header = (
        f"{'configuration':<24}{'c':>5}{'hybrid':>7}{'loss':>9}{'acc':>8}"
        f"{'acc@2T':>9}{'state/tok':>11}{'params':>8}"
    )
    lines = [
        "RAT chunk-size sweep on long-range copy",
        f"task: second half of a {SEQ_LEN}-symbol sequence repeats the first half;",
        f"scored on the second half only; vocabulary {N_SYMBOLS}",
        f"model: {N_LAYERS} layers, d_model {D_MODEL}, {HEADS} heads,"
        f" {results[0]['params']} parameters",
        f"training: {steps} steps, batch {BATCH}, AdamW lr {LR} cosine, seed {SEED}",
        f"constant-answer baseline accuracy: {constant:.3f}",
        "",
        header,
        "-" * len(header),
    ]
    for r in results:
        at_len = r["held_out"]["train_len"]
        long_len = r["held_out"].get("double_len")
        long_acc = f"{long_len['accuracy']:.3f}" if long_len is not None else "n/a"
        lines.append(
            f"{r['label']:<24}{r['chunk']:>5}{str(r['hybrid']):>7}"
            f"{at_len['loss']:>9.4f}{at_len['accuracy']:>8.3f}{long_acc:>9}"
            f"{r['state_floats_per_token']:>11.1f}{r['params']:>8}"
        )
    lines += [
        "",
        "loss is the mean cross-entropy over the scored (second-half) positions at",
        f"sequence length {SEQ_LEN}; acc is exact match at that length; acc@2T is",
        f"exact match at sequence length {EVAL_LEN} after training only at"
        f" {SEQ_LEN} (n/a means {EVAL_LEN} is not a multiple of that chunk size);",
        "state/tok is the generation-time key/value cache of the whole model divided",
        "by the sequence length: 2 * d_model / chunk per RAT layer, 2 * d_model / 32",
        "per local-attention layer (window 32).",
        "",
        "accuracy by source distance (every scored token reads exactly T/2 ="
        f" {SEQ_LEN // 2} positions back, so the 256+ bucket is the whole score):",
    ]
    for r in results:
        buckets = r["held_out"]["train_len"]["accuracy_by_source_distance"]
        rendered = "  ".join(
            f"{k}: {v:.3f}" for k, v in buckets.items() if v == v
        )
        lines.append(f"  {r['label']:<24}{rendered}")
    with open(args.out + ".txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nwrote {args.out}.txt and {args.out}.json")


if __name__ == "__main__":
    main()
