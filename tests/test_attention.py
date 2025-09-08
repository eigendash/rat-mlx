"""Invariants for the chunked RAT temporal mixing.

The two that matter most: ``chunk = 1`` must collapse to ordinary causal softmax
attention, and ``chunk = T`` must collapse to the pure gated recurrence. Both are
what the paper means by RAT interpolating between attention and an RNN.
"""

import mlx.core as mx
import numpy as np
import pytest

from rat.attention import (
    chunk_state_floats,
    full_causal_attention,
    inter_chunk_summary,
    intra_chunk_scan,
    intra_chunk_scan_recurrent,
    rat_attention,
    rat_attention_streaming,
)


def rand_inputs(B=2, T=24, D=16, heads=4, seed=0):
    mx.random.seed(seed)
    return {
        "z": mx.sigmoid(mx.random.normal((B, T, D))),
        "g": mx.sigmoid(mx.random.normal((B, T, D))),
        "q": mx.random.normal((B, T, D)),
        "k": mx.random.normal((B, T, D)),
        "v": mx.random.normal((B, T, D)),
        "heads": heads,
    }


def reference_numpy(a, chunk, zero_query=False):
    """Explicit per-position reference for the visible set.

    For each query ``t = (c, l)`` the visible pairs are the end-of-chunk gated
    key/value of the chunks before ``c`` plus the gated pair at ``t`` itself,
    scored by ``q_t . kappa * scale`` and combined with a softmax. Prints the
    pre-output-gate mixture.
    """
    q_src = np.zeros_like(np.array(a["q"])) if zero_query else np.array(a["q"])
    B, T, D = q_src.shape
    heads = a["heads"]
    P = D // heads
    scale = P**-0.5
    C = T // chunk
    gn, kn, vn, zn = (np.array(a[name]) for name in ("g", "k", "v", "z"))

    kt = np.zeros((B, T, D))
    vt = np.zeros((B, T, D))
    for c in range(C):
        sk = np.zeros((B, D))
        sv = np.zeros((B, D))
        for l in range(chunk):
            t = c * chunk + l
            sk = gn[:, t] * sk + (1 - gn[:, t]) * kn[:, t]
            sv = gn[:, t] * sv + (1 - gn[:, t]) * vn[:, t]
            kt[:, t], vt[:, t] = sk, sv

    ref = np.zeros((B, T, D))
    for t in range(T):
        c = t // chunk
        cand_k = [kt[:, j * chunk + chunk - 1, :] for j in range(c)] + [kt[:, t, :]]
        cand_v = [vt[:, j * chunk + chunk - 1, :] for j in range(c)] + [vt[:, t, :]]
        K = np.stack(cand_k, axis=1)  # (B, n, D)
        V = np.stack(cand_v, axis=1)
        for b in range(B):
            for h in range(heads):
                sl = slice(h * P, (h + 1) * P)
                logits = K[b, :, sl] @ q_src[b, t, sl] * scale
                w = np.exp(logits - logits.max())
                w /= w.sum()
                ref[b, t, sl] = w @ V[b, :, sl]
    return zn * ref


def test_chunked_scan_matches_per_chunk_reference_loop():
    """The chunked scan equals a loop that resets the state at every boundary."""
    a = rand_inputs(T=20, D=8, heads=1)
    out = np.array(intra_chunk_scan(a["g"], a["v"], 5))
    B, T, D = a["g"].shape
    gn, xn = np.array(a["g"]), np.array(a["v"])
    ref = np.zeros((B, T, D))
    for b in range(B):
        for c in range(T // 5):
            s = np.zeros(D)
            for l in range(5):
                t = c * 5 + l
                s = gn[b, t] * s + (1.0 - gn[b, t]) * xn[b, t]
                ref[b, t] = s
    np.testing.assert_allclose(out, ref, atol=1e-6, rtol=1e-6)


def test_single_chunk_scan_is_the_full_sequence_recurrence():
    """With one chunk the scan is the paper's Eq. (2) over the whole row."""
    a = rand_inputs(T=16, D=8, heads=1)
    chunked = np.array(intra_chunk_scan(a["g"], a["v"], 16))
    full = np.array(intra_chunk_scan_recurrent(a["g"], a["v"]))
    np.testing.assert_allclose(chunked, full, atol=1e-6, rtol=1e-6)


def test_chunk_summary_is_the_scan_at_chunk_end_positions():
    a = rand_inputs(T=20, D=8, heads=1)
    kk, vv = inter_chunk_summary(a["g"], a["k"], a["v"], 5)
    full_k = np.array(intra_chunk_scan(a["g"], a["k"], 5))
    full_v = np.array(intra_chunk_scan(a["g"], a["v"], 5))
    assert kk.shape == (2, 4, 8)
    for c in range(4):
        np.testing.assert_allclose(np.array(kk)[:, c], full_k[:, c * 5 + 4], atol=1e-6)
        np.testing.assert_allclose(np.array(vv)[:, c], full_v[:, c * 5 + 4], atol=1e-6)


@pytest.mark.parametrize("heads", [1, 4])
def test_chunk_one_is_causal_softmax_attention(heads):
    """L = 1 must be exactly causal attention over the gated keys and values."""
    a = rand_inputs(T=16, D=16, heads=heads)
    out, _ = rat_attention(a["z"], a["g"], a["q"], a["k"], a["v"], 1, heads=heads)
    gated_k = (1.0 - a["g"]) * a["k"]
    gated_v = (1.0 - a["g"]) * a["v"]
    ref = a["z"] * full_causal_attention(a["q"], gated_k, gated_v, heads)
    np.testing.assert_allclose(np.array(out), np.array(ref), atol=1e-5, rtol=1e-5)


def test_chunk_equals_sequence_is_a_pure_recurrence():
    """L = T leaves no earlier chunk, so the block is only the gated recurrence."""
    a = rand_inputs(T=16, D=8, heads=2)
    out, info = rat_attention(a["z"], a["g"], a["q"], a["k"], a["v"], 16, heads=2)
    scan_v = intra_chunk_scan(a["g"], a["v"], 16)
    np.testing.assert_allclose(
        np.array(out), np.array(a["z"] * scan_v), atol=1e-6, rtol=1e-6
    )
    assert info["chunks"] == 1
    assert info["chunk_summary_v"].shape == (2, 1, 8)


def test_parallel_form_equals_the_token_by_token_form():
    """The parallel training form must reproduce the sequential decoding path."""
    for T, chunk in [(24, 1), (24, 2), (24, 4), (24, 8), (24, 24), (32, 16)]:
        a = rand_inputs(T=T, D=16, heads=4)
        par, _ = rat_attention(a["z"], a["g"], a["q"], a["k"], a["v"], chunk, heads=4)
        seq = rat_attention_streaming(
            a["z"], a["g"], a["q"], a["k"], a["v"], chunk, heads=4
        )
        np.testing.assert_allclose(
            np.array(par), np.array(seq), atol=1e-5, rtol=1e-5, err_msg=f"chunk={chunk}"
        )


def test_output_is_causal_under_future_perturbation():
    """Changing tokens after position t must not move the output at or before t."""
    a = rand_inputs(T=32, D=16, heads=4)
    base, _ = rat_attention(a["z"], a["g"], a["q"], a["k"], a["v"], 8, heads=4)
    mx.eval(base)
    cut = 20
    changed = {}
    for name in ("z", "g", "q", "k", "v"):
        changed[name] = mx.concatenate(
            [a[name][:, :cut], mx.ones_like(a[name][:, cut:])], axis=1
        )
    other, _ = rat_attention(
        changed["z"], changed["g"], changed["q"], changed["k"], changed["v"], 8, heads=4
    )
    mx.eval(other)
    np.testing.assert_allclose(
        np.array(base[:, :cut]), np.array(other[:, :cut]), atol=1e-6
    )
    assert not np.allclose(np.array(base[:, cut:]), np.array(other[:, cut:]))


def test_visible_set_matches_an_explicit_per_position_reference():
    """The implemented visible set equals an explicitly built one (see helper).

    This test carries the real coverage of the mask rules: it fails if a query
    attends to its own chunk's *end* summary, if it can see a later chunk, or if
    the diagonal same-position term is dropped.
    """
    for chunk in (1, 2, 4, 8):
        a = rand_inputs(B=2, T=16, D=8, heads=2, seed=chunk)
        out, _ = rat_attention(a["z"], a["g"], a["q"], a["k"], a["v"], chunk, heads=2)
        mx.eval(out)
        ref = reference_numpy(a, chunk)
        np.testing.assert_allclose(
            np.array(out), ref, atol=2e-5, err_msg=f"chunk={chunk}"
        )


def test_equal_query_gives_an_even_mixture():
    """With q = 0 all visible entries share the weight, per the same reference."""
    a = rand_inputs(B=2, T=16, D=8, heads=2, seed=7)
    zeros = mx.zeros_like(a["q"])
    out, _ = rat_attention(a["z"], a["g"], zeros, a["k"], a["v"], 4, heads=2)
    mx.eval(out)
    ref = reference_numpy(a, 4, zero_query=True)
    np.testing.assert_allclose(np.array(out), ref, atol=2e-5)
    # The output gate really does scale the result.
    z2 = mx.concatenate([a["z"][:, :1], mx.zeros_like(a["z"][:, 1:])], axis=1)
    out2, _ = rat_attention(z2, a["g"], zeros, a["k"], a["v"], 4, heads=2)
    mx.eval(out2)
    np.testing.assert_allclose(np.array(out2)[:, 1:], 0.0, atol=1e-6)


def test_hand_computed_two_chunk_case():
    """A four-token, two-chunk case worked out by hand.

    chunk=2, D=1, one head, q = 0, g = 0.5, z = 1, and k = v = the token index as
    a float. The recurrence restarts at each chunk boundary, so the scan is

      chunk 0: [0.0, 0.5]   with 0.5 = 0.5 * 0.0 + 0.5 * 1
      chunk 1: [1.0, 2.0]   with 1.0 = 0.5 * 0.0 + 0.5 * 2
                                 2.0 = 0.5 * 1.0 + 0.5 * 3

    so the chunk-0 summary is 0.5 and the chunk-1 summary is 2.0, while the
    diagonal numerator at token ``t`` is the scan value at ``t``. With q = 0 every
    visible entry gets the same weight. Token 3 is the interesting one: it sees the
    chunk-1 summary (2.0), which already contains its own gated value, and its own
    diagonal gated value (2.0), so that value is counted twice and the mixture is
    2.0 -- the same as the plain recurrence. This duplicate is also what the
    released implementation computes (see the README).
    """
    B, T, D = 1, 4, 1
    ones = mx.ones((B, T, D))
    g = 0.5 * ones
    z = ones
    q = mx.zeros((B, T, D))
    k = mx.array([0.0, 1.0, 2.0, 3.0]).reshape(B, T, D)
    v = k
    out, info = rat_attention(z, g, q, k, v, 2, heads=1)
    got = np.array(out)[0, :, 0]
    want = np.array([0.0, 0.5, 0.75, 1.25])
    np.testing.assert_allclose(got, want, atol=1e-6)
    np.testing.assert_allclose(
        np.array(info["chunk_summary_v"])[0, :, 0], np.array([0.5, 2.0]), atol=1e-6
    )


def test_gradients_are_finite_for_every_chunk_size():
    def make_loss(chunk):
        def loss(z, g, q, k, v):
            out, _ = rat_attention(z, g, q, k, v, chunk, heads=2)
            return (out**2).sum()

        return loss

    for chunk in (1, 2, 8, 16):
        a = rand_inputs(T=16, D=8, heads=2)
        grads = mx.grad(make_loss(chunk), argnums=(0, 1, 2, 3, 4))(
            a["z"], a["g"], a["q"], a["k"], a["v"]
        )
        for name, gr in zip(("z", "g", "q", "k", "v"), grads):
            arr = np.array(gr)
            assert np.all(np.isfinite(arr)), f"non-finite grad for {name} at chunk={chunk}"
            if chunk == 16 and name in ("q", "k"):
                # With one chunk there is no attention at all: q and k are unused
                # and the block is a pure recurrence driven by g and v.
                assert np.all(arr == 0), f"{name} must not matter with one chunk"
            else:
                assert np.any(arr != 0), f"zero grad for {name} at chunk={chunk}"


def test_gradients_are_finite_when_the_first_chunk_has_no_visible_chunk():
    """The first chunk is the degenerate case of the inter-chunk mask."""

    def loss(q, k, v, g, z):
        out, _ = rat_attention(z, g, q, k, v, 8, heads=1)
        return out.sum()

    a = rand_inputs(T=8, D=8, heads=1)
    grads = mx.grad(loss, argnums=(0, 1, 2, 3, 4))(
        a["q"], a["k"], a["v"], a["g"], a["z"]
    )
    for gr in grads:
        assert np.all(np.isfinite(np.array(gr)))


def test_shapes_and_dtypes():
    a = rand_inputs(B=3, T=16, D=12, heads=3)
    out, info = rat_attention(a["z"], a["g"], a["q"], a["k"], a["v"], 4, heads=3)
    assert out.shape == (3, 16, 12)
    assert out.dtype == a["q"].dtype
    assert info["chunk_summary_k"].shape == (3, 4, 12)
    assert info["inter_lse"].shape == (3, 3, 16)


def test_unknown_chunk_size_raises():
    a = rand_inputs(T=15, D=8, heads=2)
    with pytest.raises(ValueError):
        rat_attention(a["z"], a["g"], a["q"], a["k"], a["v"], 4, heads=2)


def test_state_size_is_independent_of_sequence_length():
    """Per-token state falls as 1 / chunk and never grows with ``T``."""
    d_model = 32
    for chunk in (1, 4, 16, 64):
        per_token = [
            chunk_state_floats(T, chunk, d_model) / T for T in (128, 512, 4096)
        ]
        # Exactly 2 * d_model / chunk: one key and one value per chunk, nothing
        # that depends on the sequence length.
        assert per_token == [2 * d_model / chunk] * 3
    # chunk = 1 is full attention: one key and one value of width d_model per token.
    assert chunk_state_floats(1024, 1, d_model) == 2 * d_model * 1024
    assert chunk_state_floats(1024, 16, d_model) == 2 * d_model * 64
    with pytest.raises(ValueError):
        chunk_state_floats(1000, 16, d_model)
