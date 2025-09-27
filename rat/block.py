"""The RAT temporal-mixing layer, plus sliding-window attention for the hybrid.

Paper: *RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based
Sequence Modeling* (arXiv:2507.04416), Subsec. 3.1 and Appendix A.1.

One RAT layer holds five ``D x D``-equivalent projections, which is the paper's
``4 D^2`` budget: output projection (``D^2``), shared query/key (``D^2``), value
(``D^2``), forget gate (``D^2``) and output gate (``D^2``). The paper proposes
either low-rank gates or sharing the query and key projection and reports that
sharing wins slightly; we share. With ``ngroups = 1`` the shared ``q``/``k``
projection has exactly the same width as the value projection, and the forget
gate keeps its per-dimension granularity, which the paper notes is what stops
the design from collapsing to single-head attention.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import chunk_state_floats, rat_attention

__all__ = ["RATLayer", "SlidingWindowAttention", "chunk_rope", "chunk_rope_cos_sin"]


def chunk_rope_cos_sin(n_chunks: int, dim: int, base: float = 10000.0):
    """Cos/sin tables of shape ``(n_chunks, dim // 2)`` for chunk-level RoPE."""
    half = dim // 2
    inv = base ** (-mx.arange(half, dtype=mx.float32) / half)
    ang = mx.arange(n_chunks, dtype=mx.float32)[:, None] * inv[None, :]
    return mx.cos(ang), mx.sin(ang)


def chunk_rope(x: mx.array, chunk: int) -> mx.array:
    """RoPE whose rotation is indexed by chunk, not by token.

    ``x`` is ``(B, H, T, P)`` with ``T`` a multiple of ``chunk``. Every token in a
    chunk gets the same rotation, which only makes sense for inter-chunk attention
    where all positions of a chunk share one summary; because the same rotation is
    applied to the queries and to the gated keys, only the *relative* chunk index
    matters, and the same-position diagonal term cancels it entirely.

    The paper says "we use RoPE based on chunk indices for inter-chunk attention"
    (Subsec. 3.1) without fixing a convention. This is half-split RoPE with base
    10000: the first ``P // 2`` features are rotated as 2-D pairs and the rest are
    left alone, which is why the norm is preserved.
    """
    B, H, T, P = x.shape
    n_chunks = T // chunk
    cos, sin = chunk_rope_cos_sin(n_chunks, P)
    # (n_chunks, P // 2) -> (B, H, T, P // 2) by repeating each chunk index.
    cos = mx.repeat(cos, chunk, axis=0)[None, None]
    sin = mx.repeat(sin, chunk, axis=0)[None, None]
    half = P // 2
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    return mx.concatenate([rot1, rot2, x[..., 2 * half :]], axis=-1)


def _l2_normalise(x: mx.array, heads: int) -> mx.array:
    """Unit-normalise each head's ``d_head`` vector, with a floor for safety."""
    B, T, D = x.shape
    P = D // heads
    xh = x.reshape(B, T, heads, P)
    norm = mx.sqrt(mx.maximum((xh * xh).sum(-1, keepdims=True), 1e-12))
    return (xh / norm).reshape(B, T, D)


class RATLayer(nn.Module):
    """One RAT block: intra-chunk recurrence + inter-chunk attention + gates.

    ``forward(x)`` takes ``(B, T, D)`` with ``T`` a multiple of ``chunk``.
    ``state_floats(T)`` reports the generation-time state this layer holds.
    """

    def __init__(
        self,
        d_model: int,
        heads: int = 1,
        chunk: int = 16,
        rope: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if chunk < 1:
            raise ValueError("chunk must be >= 1")
        self.d_model = d_model
        self.heads = heads
        self.chunk = chunk
        self.rope = rope
        d_head = d_model // heads
        self.scale = d_head**-0.5
        self.input_norm = nn.RMSNorm(d_model, eps=norm_eps)
        # z | x | g | q | k  (q and k share one projection, as in the paper)
        self.in_proj = nn.Linear(d_model, 4 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # Forget-gate bias, initialised so that a chunk's worth of gates decays
        # at a rate that matches the chunk length. The paper does not give this
        # initialisation; the released implementation draws log-uniform in
        # [1, chunk - 1] and we follow it (see README).
        if chunk == 1:
            self.gate_bias = mx.zeros((d_model,))
        else:
            u = mx.random.uniform(low=1.0, high=float(chunk - 1), shape=(d_model,))
            self.gate_bias = mx.log(u)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)

    def forward(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        if T % self.chunk != 0:
            raise ValueError(
                f"sequence length {T} is not a multiple of chunk size {self.chunk}"
            )
        h = self.input_norm(x)
        proj = self.in_proj(h)
        z, xv, g_pre, qk = mx.split(proj, 4, axis=-1)
        z = mx.sigmoid(z)
        g = mx.sigmoid(g_pre + self.gate_bias)
        q = qk
        k = qk

        # Normalise the query and key vectors per head before attention. This is
        # not in the paper (it does not need it at 1.3B with careful init), but the
        # toy runs push the raw dot products past the range where float32 softmax
        # keeps a usable backward pass, and normalising bounds every logit by the
        # head dimension regardless of how large the weights grow. It is a
        # monotone change to the ranking, so it cannot hurt retrieval.
        q = _l2_normalise(q, self.heads)
        k = _l2_normalise(k, self.heads)
        if self.rope:
            q = chunk_rope(q.reshape(B, T, self.heads, -1).transpose(0, 2, 1, 3), self.chunk)
            q = q.transpose(0, 2, 1, 3).reshape(B, T, self.heads * (D // self.heads))

        out, _ = rat_attention(
            z, g, q, k, xv, self.chunk, heads=self.heads, scale=self.scale
        )
        return x + self.out_proj(out)

    def state_floats(self, seq_len: int) -> int:
        """Generation state of this layer in floats, for ``seq_len`` tokens."""
        return chunk_state_floats(seq_len, self.chunk, self.d_model, self.heads)

    def chunk_summary_count(self, seq_len: int) -> int:
        return seq_len // self.chunk


class SlidingWindowAttention(nn.Module):
    """Causal attention restricted to the last ``window`` tokens.

    This is the local-attention half of the paper's hybrid. The hybrid
    interleaves these with RAT layers (Subsec. 3.1, "Hybrid design with Local
    Attention"); the released implementation places local attention on layers
    where ``layer_index % interleave_step == 0`` and RAT elsewhere, which is what
    ``rat.model.build_layers`` does with ``interleave_step = 2``.
    """

    def __init__(
        self,
        d_model: int,
        heads: int = 1,
        window: int = 32,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        self.d_model = d_model
        self.heads = heads
        self.window = window
        self.scale = (d_model // heads) ** -0.5
        self.input_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward(x)

    def forward(self, x: mx.array) -> mx.array:
        B, T, D = x.shape
        h = self.input_norm(x)
        q, k, v = mx.split(self.in_proj(h), 3, axis=-1)
        H, P = self.heads, D // self.heads
        q = q.reshape(B, T, H, P).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, P).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, P).transpose(0, 2, 1, 3)
        i = mx.arange(T)
        d = i[:, None] - i[None, :]
        mask = (d >= 0) & (d <= self.window - 1)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.heads * P)
        return x + self.out_proj(out)

    def state_floats(self, seq_len: int) -> int:
        """KV cache of the last ``window`` tokens."""
        return 2 * min(seq_len, self.window) * self.d_model
