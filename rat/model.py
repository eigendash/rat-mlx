"""A small decoder-only LM built from RAT blocks.

Paper: *RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based
Sequence Modeling* (arXiv:2507.04416). The paper's hybrid interleaves RAT with
sliding-window attention; ``hybrid=True`` here puts ``SlidingWindowAttention`` on
layers where ``layer_index % interleave_step == 0`` and ``RATLayer`` elsewhere,
to match the released implementation's ``sequence_interleave`` backbone.

This is a toy: a few hundred thousand parameters, learned token embeddings, one
learned positional embedding over chunk indices for inter-chunk attention, and
no dropout or mixture-of-experts.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .attention import chunk_state_floats
from .block import RATLayer, SlidingWindowAttention

__all__ = ["ModelConfig", "RATLM", "build_layers", "count_parameters"]


@dataclass
class ModelConfig:
    """Everything needed to rebuild a model."""

    vocab_size: int
    d_model: int = 64
    n_layers: int = 4
    heads: int = 1
    chunk: int = 16
    max_chunks: int = 64
    hybrid: bool = False
    interleave_step: int = 2
    window: int = 32
    rope: bool = True

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def build_layers(cfg: ModelConfig) -> list[nn.Module]:
    """Layer stack; the hybrid alternates local attention and RAT."""
    layers: list[nn.Module] = []
    for i in range(cfg.n_layers):
        if cfg.hybrid and i % cfg.interleave_step == 0:
            layers.append(
                SlidingWindowAttention(
                    cfg.d_model, heads=cfg.heads, window=cfg.window
                )
            )
        else:
            layers.append(
                RATLayer(
                    cfg.d_model, heads=cfg.heads, chunk=cfg.chunk, rope=cfg.rope
                )
            )
    return layers


def count_parameters(model) -> int:
    from mlx.utils import tree_flatten

    return int(sum(v.size for _, v in tree_flatten(model.parameters())))


class RATLM(nn.Module):
    """Token embedding -> sequence-mixing layers -> tied LM head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = build_layers(cfg)
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-5)
        if not cfg.rope:
            # Learned chunk-index embedding for inter-chunk attention, the
            # alternative to the paper's chunk-level RoPE.
            self.chunk_pos = nn.Embedding(cfg.max_chunks, cfg.d_model)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.forward(tokens)

    def forward(self, tokens: mx.array) -> mx.array:
        h = self.tok_emb(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        # Weight tying: logits = h @ E^T.
        return h @ self.tok_emb.weight.T

    def loss(self, tokens: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
        """Cross-entropy over the masked positions only.

        ``targets`` is ``(B, T)`` with -1 where the position is not scored and
        ``mask`` is the boolean version of that.
        """
        logits = self.forward(tokens)
        safe = mx.where(mask, targets, 0)
        logp = nn.log_softmax(logits, axis=-1)
        picked = mx.take_along_axis(logp, safe[..., None], axis=-1)[..., 0]
        total = -(picked * mask).sum()
        return total / mx.maximum(mask.sum(), 1)

    def state_floats(self, seq_len: int) -> int:
        """Generation-time state (floats) held by all layers on one sequence."""
        return sum(layer.state_floats(seq_len) for layer in self.layers)

    def state_per_token(self, seq_len: int) -> float:
        return self.state_floats(seq_len) / seq_len

    def chunk_state_floats(self, seq_len: int, chunk: int) -> int:
        """Same count but with every RAT layer forced to ``chunk``."""
        total = 0
        for layer in self.layers:
            if isinstance(layer, RATLayer):
                total += chunk_state_floats(seq_len, chunk, self.cfg.d_model, layer.heads)
            else:
                total += layer.state_floats(seq_len)
        return total

    def chunk_state_per_token(self, seq_len: int, chunk: int) -> float:
        """Per-token generation state with all RAT layers at ``chunk``.

        The hybrid's local-attention layers are counted at their own window.
        """
        return self.chunk_state_floats(seq_len, chunk) / seq_len
