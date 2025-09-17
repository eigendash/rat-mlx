"""Synthetic tasks for the toy LM, with long-range access as the point.

Two tasks live here, both generated from a seeded ``numpy`` generator so runs are
reproducible:

``make_copy_batch``
    A sequence of ``T`` symbols whose second half repeats the first half. The
    model is scored on the second half, so predicting token ``T/2 + i`` requires
    reading prefix position ``i``. Every scored token is exactly ``T / 2``
    positions away from the token that determines it, and the input distribution
    is stationary, so no local statistic can do better than chance. This is the
    task the experiment trains on.

``make_recall_batch``
    The ``k``-pair associative recall used as a second check: a prefix plants
    ``k`` key/value pairs, filler follows, and the tail asks for the values of
    some of those keys. This one needs *matching* rather than copying, and at this
    scale none of the architectures learn it (see the README); it is kept because
    a negative result about the task is worth recording.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Batch",
    "make_copy_batch",
    "make_recall_batch",
    "source_offsets",
    "random_baseline_accuracy",
    "copy_accuracy_by_distance",
]


@dataclass
class Batch:
    """A batch of sequences.

    ``tokens`` is ``(B, T)`` int32, ``targets`` is ``(B, T)`` int32 with -1 at
    positions that are not scored, and ``query_mask`` marks the scored ones.
    """

    tokens: np.ndarray
    targets: np.ndarray
    query_mask: np.ndarray

    @property
    def n_scored(self) -> int:
        return int(self.query_mask.sum())


def make_copy_batch(
    batch_size: int,
    seq_len: int,
    n_symbols: int = 8,
    rng: np.random.Generator | None = None,
) -> Batch:
    """Copy the first half of the sequence into the second half.

    The vocabulary is ``1 .. n_symbols``; there is no separator, so every position
    of the prefix is a source and every position of the suffix a query. ``seq_len``
    must be even.
    """
    if seq_len % 2 != 0:
        raise ValueError("seq_len must be even for the copy task")
    rng = np.random.default_rng() if rng is None else rng
    half = seq_len // 2
    prefix = rng.integers(1, n_symbols + 1, size=(batch_size, half)).astype(np.int32)
    tokens = np.concatenate([prefix, prefix], axis=1)
    targets = np.full((batch_size, seq_len), -1, dtype=np.int32)
    mask = np.zeros((batch_size, seq_len), dtype=bool)
    targets[:, half:] = prefix
    mask[:, half:] = True
    return Batch(tokens=tokens, targets=targets, query_mask=mask)


def source_offsets(seq_len: int) -> np.ndarray:
    """For each position, how far back the token that determines it sits.

    -1 for unscored prefix positions. For the copy task this is ``seq_len // 2``
    everywhere in the suffix, which is what makes the "by distance" breakdown in
    the experiment a per-position split rather than a length split.
    """
    half = seq_len // 2
    offs = np.full(seq_len, -1, dtype=np.int32)
    offs[half:] = half
    return offs


def make_recall_batch(
    batch_size: int,
    seq_len: int,
    n_key: int = 8,
    n_pairs: int = 4,
    n_queries: int = 4,
    rng: np.random.Generator | None = None,
) -> Batch:
    """Associative recall: plant key/value pairs, then ask for some values.

    Layout::

        [k_1 v_1 ... k_j v_j | values used as filler | q_1 ... q_m]

    Query positions hold a key symbol that was planted in the prefix; the target is
    the value symbol that was paired with it. Filler is drawn from the value
    symbols only, so a query key can never be confused with filler.
    """
    rng = np.random.default_rng() if rng is None else rng
    need = 2 * n_pairs + n_queries
    if seq_len < need:
        raise ValueError(f"seq_len {seq_len} too short, need at least {need}")
    if n_queries > n_pairs:
        raise ValueError("cannot ask more queries than planted pairs")
    n_val = 1 + n_key
    first_key = 1 + n_key
    n_tok = 1 + 2 * n_key

    tokens = rng.integers(1, n_val, size=(batch_size, seq_len)).astype(np.int32)
    targets = np.full((batch_size, seq_len), -1, dtype=np.int32)
    mask = np.zeros((batch_size, seq_len), dtype=bool)

    for b in range(batch_size):
        keys = rng.choice(n_key, size=n_pairs, replace=False)
        vals = rng.choice(n_key, size=n_pairs, replace=False)
        for i in range(n_pairs):
            tokens[b, 2 * i] = first_key + keys[i]
            tokens[b, 2 * i + 1] = 1 + vals[i]
        tokens[b, 2 * n_pairs : seq_len - n_queries] = rng.integers(
            1, n_val, size=seq_len - n_queries - 2 * n_pairs
        )
        asked = rng.choice(n_pairs, size=n_queries, replace=False)
        for i, j in enumerate(asked):
            pos = seq_len - n_queries + i
            tokens[b, pos] = first_key + keys[j]
            targets[b, pos] = 1 + vals[j]
            mask[b, pos] = True
    del n_tok
    return Batch(tokens=tokens, targets=targets, query_mask=mask)


def random_baseline_accuracy(batch: Batch) -> float:
    """Accuracy of always answering the most common target symbol."""
    if batch.n_scored == 0:
        return float("nan")
    t = batch.targets[batch.query_mask]
    return float(max((t == v).mean() for v in np.unique(t)))


def copy_accuracy_by_distance(
    logits: np.ndarray, batch: Batch, edges: tuple[int, ...] = (0, 16, 64, 256)
) -> dict[str, float]:
    """Exact-match accuracy on the scored positions, bucketed by source distance.

    ``logits`` is ``(B, T, V)``. The buckets are ``[0, 16)``, ``[16, 64)``,
    ``[64, 256)`` and ``[256, inf)`` by default; a bucket with no scored position
    is reported as ``nan``.
    """
    offs = source_offsets(batch.tokens.shape[1])
    pred = logits.argmax(axis=-1)
    correct = (pred == batch.targets) & batch.query_mask
    out: dict[str, float] = {}
    bounds = list(edges) + [10**9]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = (offs >= lo) & (offs < hi) & batch.query_mask
        label = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
        out[label] = float(correct[sel].mean()) if sel.any() else float("nan")
    return out
