"""Synthetic long-range associative recall, the task the experiment trains on.

Layout of one sequence, with a vocabulary of ``2 * n_key`` symbols:

    [ k_1 v_1 k_2 v_2 ... k_n v_n | SEP | x_1 ... x_{T - 2n - 1} | q_1 ... q_m ]

The first ``2n`` positions plant ``n`` key/value symbol pairs. The middle is
filler drawn from the same symbol pool. The tail contains ``m`` query positions
holding a key symbol that also appeared in the prefix; the target at a query
position is the value symbol that was paired with it. Predicting the target
requires reaching back to a prefix position that is at least a filler length
away, and the filler is long enough that no bounded-window model can do it
without copying the prefix into its state.

Everything is generated from a seeded ``numpy`` generator, so runs are
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "InductionBatch",
    "n_symbols",
    "make_induction_batch",
    "retrieval_accuracy",
    "random_baseline_accuracy",
]

SEP = 0  # separator symbol
FIRST_VALUE = 1  # value symbols are FIRST_VALUE .. FIRST_VALUE + n_key - 1


def n_symbols(n_key: int) -> int:
    """Vocabulary size: separator, ``n_key`` values, ``n_key`` keys."""
    return 1 + 2 * n_key


@dataclass
class InductionBatch:
    """A batch of recall sequences.

    ``tokens`` is ``(B, T)`` int32, ``targets`` is ``(B, T)`` int32 with -1 at
    positions that are not scored, and ``query_mask`` marks the scored ones.
    """

    tokens: np.ndarray
    targets: np.ndarray
    query_mask: np.ndarray

    @property
    def n_scored(self) -> int:
        return int(self.query_mask.sum())


def make_induction_batch(
    batch_size: int,
    seq_len: int,
    n_key: int = 8,
    n_pairs: int = 4,
    n_queries: int = 8,
    rng: np.random.Generator | None = None,
) -> InductionBatch:
    """Build one batch of the recall task.

    ``n_pairs`` key/value pairs are planted, ``n_queries`` of them are asked
    again at the end, and the rest of the sequence is filler. The sequence must
    be long enough for ``2 * n_pairs + n_queries + 1`` tokens.
    """
    rng = np.random.default_rng() if rng is None else rng
    need = 2 * n_pairs + n_queries + 1
    if seq_len < need:
        raise ValueError(f"seq_len {seq_len} too short, need at least {need}")
    if n_queries > n_pairs:
        raise ValueError("cannot ask more queries than planted pairs")
    V = n_symbols(n_key)
    filler_pool = np.arange(1, V, dtype=np.int32)

    tokens = rng.integers(1, V, size=(batch_size, seq_len)).astype(np.int32)
    targets = np.full((batch_size, seq_len), -1, dtype=np.int32)
    query_mask = np.zeros((batch_size, seq_len), dtype=bool)

    for b in range(batch_size):
        keys = rng.choice(n_key, size=n_pairs, replace=False)
        vals = rng.choice(n_key, size=n_pairs, replace=False)
        planted = np.empty(2 * n_pairs, dtype=np.int32)
        planted[0::2] = FIRST_VALUE + n_key + keys
        planted[1::2] = FIRST_VALUE + vals
        tokens[b, : 2 * n_pairs] = planted
        tokens[b, 2 * n_pairs] = SEP
        # Filler is not scored and must not be a key symbol in a position that a
        # query could confuse with a planted key: use value symbols only.
        tokens[b, 2 * n_pairs + 1 : seq_len - n_queries] = rng.choice(
            FIRST_VALUE + np.arange(n_key), size=seq_len - n_queries - 2 * n_pairs - 1
        )
        asked = rng.choice(n_pairs, size=n_queries, replace=False)
        for i, p in enumerate(asked):
            pos = seq_len - n_queries + i
            tokens[b, pos] = FIRST_VALUE + n_key + keys[p]
            targets[b, pos] = FIRST_VALUE + vals[p]
            query_mask[b, pos] = True

    return InductionBatch(tokens=tokens, targets=targets, query_mask=query_mask)


def retrieval_accuracy(logits: np.ndarray, batch: InductionBatch) -> float:
    """Fraction of query positions whose argmax is the planted value."""
    if batch.n_scored == 0:
        return float("nan")
    pred = logits.argmax(axis=-1)
    return float((pred[batch.query_mask] == batch.targets[batch.query_mask]).mean())


def random_baseline_accuracy(batch: InductionBatch) -> float:
    """Accuracy of predicting the most common *value* symbol, no context used.

    A model that ignores the sequence and always answers the same value symbol
    gets this. It is the honest floor for the task: with ``n_key`` values, a
    uniform guesser would get ``1 / n_key``, and the constant answer gets
    ``max_p count(target == value_p) / n_scored``.
    """
    if batch.n_scored == 0:
        return float("nan")
    t = batch.targets[batch.query_mask]
    values = np.unique(t)
    return float(max((t == v).mean() for v in values))
