"""Gated intra-chunk recurrence and chunk-based inter/intra attention.

The maths here is the temporal mixing of RAT: *RAT: Bridging RNN Efficiency and
Attention Accuracy via Chunk-based Sequence Modeling* (arXiv:2507.04416).
Appendix A.1 of that paper gives the training pseudocode this file follows.

Notation matches the paper. A sequence of length ``T`` is cut into ``C`` chunks
of length ``L`` (``T = C * L``); position ``t`` is ``(c, l)``. ``g`` is the
forget gate, ``z`` the output gate, both ``sigmoid`` of a linear projection.

Shapes are ``(B, T, D)`` for activations and ``(B, T,)`` for masks. All the
functions here are head-agnostic: heads are added by the block in ``rat/block.py``
by reshaping ``D`` into ``(H, P)``.
"""

from __future__ import annotations

import mlx.core as mx

__all__ = [
    "intra_chunk_scan",
    "intra_chunk_scan_recurrent",
    "inter_chunk_summary",
    "rat_attention",
    "full_causal_attention",
    "rat_attention_streaming",
    "chunk_state_floats",
]


def _check_divisible(T: int, chunk: int) -> int:
    if T % chunk != 0:
        raise ValueError(f"sequence length {T} is not a multiple of chunk size {chunk}")
    return T // chunk


def intra_chunk_scan(g: mx.array, x: mx.array, chunk: int) -> mx.array:
    """Gated linear recurrence inside each chunk (paper Eq. 3).

    ``out[b, c*L + l] = g_l * out[b, c*L + l - 1] + (1 - g_l) * x_l`` with the
    state reset to zero at every chunk boundary. ``g`` and ``x`` are
    ``(B, T, D)``; the result is ``(B, T, D)``.

    This is written as an explicit python loop over the chunk, one step at a
    time. The reference implementation uses a parallel associative scan over the
    chunk axis; for the chunk sizes that matter here (1 to 64) an unrolled loop
    is a handful of elementwise ops per step and stays differentiable through
    MLX's autodiff without a hand-written backward pass. ``T`` does not enter the
    loop, only ``L`` does.
    """
    B, T, D = g.shape
    C = _check_divisible(T, chunk)
    g = g.reshape(B, C, chunk, D)
    x = x.reshape(B, C, chunk, D)
    state = mx.zeros((B, C, D), dtype=x.dtype)
    steps = []
    for l in range(chunk):
        gl = g[:, :, l]
        state = gl * state + (1.0 - gl) * x[:, :, l]
        steps.append(state)
    return mx.stack(steps, axis=2).reshape(B, T, D)


def intra_chunk_scan_recurrent(g: mx.array, x: mx.array) -> mx.array:
    """The same recurrence run over the whole sequence, with no chunk resets.

    This is paper Eq. (2), the pure full-sequence RNN. Kept next to
    ``intra_chunk_scan`` so tests can check the chunked version against it and
    so the ``chunk = T`` limit has something to reduce to.
    """
    B, T, D = g.shape
    state = mx.zeros((B, D), dtype=x.dtype)
    steps = []
    for t in range(T):
        state = g[:, t] * state + (1.0 - g[:, t]) * x[:, t]
        steps.append(state)
    return mx.stack(steps, axis=1)


def inter_chunk_summary(g: mx.array, k: mx.array, v: mx.array, chunk: int):
    """End-of-chunk gated keys and values, the inter-chunk attention memory.

    Returns ``(kk, vv)`` of shape ``(B, C, D)``: the scan state of the last
    position of every chunk, which is the paper's ``K~_{:,-1}`` / ``V~_{:,-1}``.
    """
    B, T, D = g.shape
    C = _check_divisible(T, chunk)
    kk = intra_chunk_scan(g, k, chunk)
    vv = intra_chunk_scan(g, v, chunk)
    return kk.reshape(B, C, chunk, D)[:, :, chunk - 1, :], vv.reshape(B, C, chunk, D)[:, :, chunk - 1, :]


def full_causal_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    heads: int,
    scale: float | None = None,
) -> mx.array:
    """Plain causal softmax attention over tokens. Reference, not used by RAT.

    ``q``, ``k``, ``v`` are ``(B, T, D)``; the output is ``(B, T, D)``.
    """
    B, T, D = q.shape
    P = D // heads
    scale = P**-0.5 if scale is None else scale
    qh = q.reshape(B, T, heads, P).transpose(0, 2, 1, 3)
    kh = k.reshape(B, T, heads, P).transpose(0, 2, 1, 3)
    vh = v.reshape(B, T, heads, P).transpose(0, 2, 1, 3)
    out = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=scale, mask="causal")
    return out.transpose(0, 2, 1, 3).reshape(B, T, heads * P)


def rat_attention(
    z: mx.array,
    g: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    chunk: int,
    heads: int = 1,
    scale: float | None = None,
):
    """RAT temporal mixing. Returns ``(out, info)``.

    ``out[b, t] = z_t * sum_i softmax_i(q_t . kappa_i) * nu_i`` over the set of
    ``(kappa, nu)`` pairs available to query ``t = (c, l)``:

      * the end-of-chunk gated key/value of every chunk ``< c``, and
      * the gated key/value of the current chunk at its own position ``l``.

    The first set is the paper's inter-chunk attention over ``K~_{:,-1}``,
    ``V~_{:,-1}``. The second set is what the paper's Appendix A.1 pseudocode and
    the released implementation actually run: the per-chunk term is an
    elementwise ``einsum`` of the query with the gated key at the *same*
    position, merged with the inter-chunk softmax by adjusting the denominator.
    Treating the current chunk as a single entry with unnormalised weight 1
    reproduces the sequential (generation) path, where the cache slot for the
    chunk in progress holds the state at the token just processed.

    Two consequences, both checked in ``tests/test_attention.py``: at
    ``chunk = 1`` this is exactly causal softmax attention over
    ``((1 - g) * k, (1 - g) * v)``, because the end-of-chunk summary of chunk
    ``t`` *is* token ``t`` and the diagonal entry adds it back; at
    ``chunk = T`` there are no earlier chunks, the softmax is a single entry and
    the output is ``z * scan(g, v)``, the pure linear recurrence of Eq. (2).

    ``info`` carries the raw branches for tests and diagnostics: the inter-chunk
    log-sum-exp, the diagonal logits and the tensor of chunk summaries.
    """
    B, T, D = q.shape
    C = _check_divisible(T, chunk)
    P = D // heads
    scale = P**-0.5 if scale is None else scale

    kt = intra_chunk_scan(g, k, chunk)
    vt = intra_chunk_scan(g, v, chunk)
    kchunk = kt.reshape(B, C, chunk, D)[:, :, chunk - 1, :]
    vchunk = vt.reshape(B, C, chunk, D)[:, :, chunk - 1, :]

    qh = q.reshape(B, T, heads, P).transpose(0, 2, 1, 3)
    kch = kchunk.reshape(B, C, heads, P).transpose(0, 2, 1, 3)
    vch = vchunk.reshape(B, C, heads, P).transpose(0, 2, 1, 3)

    # keep[b] is the boolean "chunk b may be attended to" mask, evaluated per
    # query row. One extra always-masked wildcard slot (zero key and value) is
    # appended so that first-chunk queries, which have no visible chunk, still
    # see one entry; the fused kernel returns a non-finite row when nothing at all
    # is visible. The wildcard is never visible and its merge weight is clamped to
    # zero below, so it cannot change a result -- it only keeps values and
    # gradients finite. See the README.
    keep_prev = (mx.arange(T) // chunk)[:, None] > mx.arange(C)[None, :]  # (T, C)
    keep_all = mx.concatenate([keep_prev, mx.zeros((T, 1), dtype=mx.bool_)], axis=1)
    kch = mx.concatenate([kch, mx.zeros_like(kch[:, :, :1, :])], axis=2)
    vch = mx.concatenate([vch, mx.zeros_like(vch[:, :, :1, :])], axis=2)

    inter = mx.fast.scaled_dot_product_attention(
        qh, kch, vch, scale=scale, mask=keep_all[None, None, :, :]
    )  # (B, H, T, P)

    # Diagonal (same-position) contribution of the chunk in progress: query ``l``
    # of chunk ``c`` scores the gated key at that same position, ``k~_{c,l}``.
    kt_h = kt.reshape(B, T, heads, P).transpose(0, 2, 1, 3)
    vt_h = vt.reshape(B, T, heads, P).transpose(0, 2, 1, 3)
    diag_logits = (qh * kt_h).sum(-1) * scale  # (B, H, T)
    # The diagonal entry's numerator is the gated value itself; only the softmax
    # denominator is shared with the inter-chunk branch.
    diag = mx.broadcast_to(vt_h, qh.shape)

    # Online-softmax merge of the two branches (paper Appendix A.1,
    # merge_last_token_stable). mx.fast does not return a log-sum-exp, so it is
    # recomputed from the same logits. Masked entries use a large negative additive
    # constant rather than -inf so that the log-sum-exp and both branch weights
    # stay finite; a masked entry is then weighted by exp(-1e30), i.e. zero.
    add_mask = mx.where(keep_all, 0.0, -1e30)  # (T, C+1)
    inter_logits = (qh @ kch.transpose(0, 1, 3, 2)) * scale + add_mask
    inter_lse = mx.logsumexp(inter_logits, axis=-1)  # (B, H, T)

    m = mx.maximum(inter_lse, diag_logits)
    ei = mx.exp(inter_lse - m)
    ed = mx.exp(diag_logits - m)
    denom = ei + ed
    out = inter * (ei / denom)[..., None] + diag * (ed / denom)[..., None]
    out = out.transpose(0, 2, 1, 3).reshape(B, T, heads * P)
    info = {
        "chunk_summary_k": kchunk,
        "chunk_summary_v": vchunk,
        "inter_lse": inter_lse,
        "diag_logits": diag_logits,
        "chunks": C,
    }
    return z * out, info


def rat_attention_streaming(
    z: mx.array,
    g: mx.array,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    chunk: int,
    heads: int = 1,
    scale: float | None = None,
) -> mx.array:
    """Token-by-token RAT with a cache, the generation path the block mimics.

    ``rat_attention`` is the parallel training form. This is the sequential form
    the paper describes for decoding: keep the running gated key/value of the
    chunk in progress, promote them to a chunk summary when the chunk ends, and
    softmax over (all finished chunk summaries + the running gated pair). The two
    must agree, and a test pins that.

    At step ``t`` the running pair has been updated *before* attention is applied,
    which is what makes the diagonal entry in ``rat_attention`` a single
    same-position term rather than a full softmax inside the chunk.
    """
    B, T, D = q.shape
    C = _check_divisible(T, chunk)
    P = D // heads
    scale = P**-0.5 if scale is None else scale

    qh = q.reshape(B, T, heads, P).transpose(0, 2, 1, 3)
    kh = k.reshape(B, T, heads, P).transpose(0, 2, 1, 3)
    vh = v.reshape(B, T, heads, P).transpose(0, 2, 1, 3)

    running_k = mx.zeros((B, D))
    running_v = mx.zeros((B, D))
    summaries_k: list[mx.array] = []
    summaries_v: list[mx.array] = []
    outs = []
    for t in range(T):
        g_t = g[:, t]
        running_k = g_t * running_k + (1.0 - g_t) * k[:, t]
        running_v = g_t * running_v + (1.0 - g_t) * v[:, t]
        n_prev = len(summaries_k)
        if n_prev == 0:
            out_t = running_v[:, None, :]  # (B, 1, D)
        else:
            keys = mx.stack(summaries_k, axis=1).reshape(B, n_prev, heads, P)
            vals = mx.stack(summaries_v, axis=1).reshape(B, n_prev, heads, P)
            keys = keys.transpose(0, 2, 1, 3)
            vals = vals.transpose(0, 2, 1, 3)
            inter_logits = (qh[:, :, t : t + 1] @ keys.transpose(0, 1, 3, 2)) * scale
            inter = mx.softmax(inter_logits, axis=-1) @ vals
            diag_logits = (
                qh[:, :, t : t + 1] * running_k.reshape(B, 1, heads, P).transpose(0, 2, 1, 3)
            ).sum(-1) * scale
            m = mx.maximum(mx.logsumexp(inter_logits, axis=-1), diag_logits)
            ei = mx.exp(mx.logsumexp(inter_logits, axis=-1) - m)
            ed = mx.exp(diag_logits - m)
            denom = ei + ed
            diag = running_v.reshape(B, 1, heads, P).transpose(0, 2, 1, 3)
            out_t = inter * (ei / denom)[..., None] + diag * (ed / denom)[..., None]
        outs.append(out_t.reshape(B, 1, D))
        if (t + 1) % chunk == 0:
            summaries_k.append(running_k)
            summaries_v.append(running_v)
            running_k = mx.zeros((B, D))
            running_v = mx.zeros((B, D))
    return z * mx.concatenate(outs, axis=1)


def chunk_state_floats(seq_len: int, chunk: int, d_model: int, heads: int = 1) -> int:
    """Floats of generation state a RAT layer holds per sequence.

    RAT keeps one gated key and one gated value per *finished chunk* (the
    ``K~_{:,-1}`` / ``V~_{:,-1}`` cache), plus the running chunk's two gated
    vectors of size ``d_model``. Full attention keeps a key and a value per
    *token*, i.e. ``2 * d_model`` floats per token, which is the ``chunk = 1``
    value below. The factor 2 that appears twice is for key and value and for the
    two gated vectors of the current chunk.
    """
    chunks = seq_len // chunk
    prev = 2 * 2 * chunks * d_model  # finished chunks: k and v summaries
    running = 2 * d_model  # current chunk's running gated k and v
    return prev + running
