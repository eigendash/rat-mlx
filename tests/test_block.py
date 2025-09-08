"""Invariants for the RAT layer, the sliding-window layer and RoPE."""

import mlx.core as mx
import numpy as np
import pytest

from rat.attention import chunk_state_floats
from rat.block import RATLayer, SlidingWindowAttention, chunk_rope
from rat.model import ModelConfig, RATLM, build_layers, count_parameters


def rand_x(B=2, T=32, D=16, seed=0):
    mx.random.seed(seed)
    return mx.random.normal((B, T, D))


def test_rat_layer_shapes_and_residual():
    layer = RATLayer(d_model=16, heads=2, chunk=8)
    x = rand_x(T=32, D=16)
    y = layer(x)
    mx.eval(y)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert np.all(np.isfinite(np.array(y)))
    # The residual is there: a zeroed output projection leaves the input alone.
    zeroed = RATLayer(d_model=16, heads=2, chunk=8)
    zeroed.update({"out_proj": {"weight": mx.zeros_like(zeroed.out_proj.weight)}})
    z = zeroed(x)
    mx.eval(z)
    np.testing.assert_allclose(np.array(z), np.array(x), atol=1e-6)


@pytest.mark.parametrize("chunk", [1, 4, 8, 32])
def test_rat_layer_gradients_are_finite(chunk):
    layer = RATLayer(d_model=16, heads=2, chunk=chunk)
    x = rand_x(T=32, D=16)

    def loss(layer, x):
        return (layer(x) ** 2).sum()

    from mlx.utils import tree_flatten

    # MLX returns one gradient per argument, keyed by argument name in a flat dict,
    # so the module's parameter gradients and the input gradient are all in there.
    grads = mx.grad(loss)(layer, x)
    mx.eval(grads)
    names = [name for name, _ in tree_flatten(grads)]
    assert any(name != "x" for name in names), "no parameter gradients were returned"
    for name, g in tree_flatten(grads):
        assert np.all(np.isfinite(np.array(g))), name


@pytest.mark.parametrize("chunk", [1, 4, 8])
def test_rat_layer_is_causal(chunk):
    layer = RATLayer(d_model=16, heads=2, chunk=chunk)
    x = rand_x(T=32, D=16)
    base = layer(x)
    mx.eval(base)
    cut = 20
    changed = mx.concatenate([x[:, :cut], mx.ones_like(x[:, cut:])], axis=1)
    other = layer(changed)
    mx.eval(other)
    np.testing.assert_allclose(np.array(base)[:, :cut], np.array(other)[:, :cut], atol=1e-6)


def test_chunk_size_changes_the_layer_output():
    x = rand_x(T=32, D=16, seed=5)
    a = RATLayer(d_model=16, heads=2, chunk=4)(x)
    b = RATLayer(d_model=16, heads=2, chunk=16)(x)
    mx.eval(a, b)
    assert not np.allclose(np.array(a), np.array(b))


def test_rat_layer_state_matches_the_counting_helper():
    for chunk in (1, 4, 16):
        layer = RATLayer(d_model=16, heads=2, chunk=chunk)
        for T in (64, 512):
            assert layer.state_floats(T) == chunk_state_floats(T, chunk, 16, 2)
            assert layer.chunk_summary_count(T) == T // chunk


def test_rat_layer_rejects_a_sequence_that_is_not_a_multiple_of_the_chunk():
    layer = RATLayer(d_model=16, heads=2, chunk=8)
    with pytest.raises(ValueError):
        layer(rand_x(T=30, D=16))


def test_rope_is_indexed_by_chunk_not_by_token():
    """Two tokens inside one chunk must get the same rotation."""
    B, H, chunk_size, P = 1, 1, 4, 8
    T = 8
    x = mx.zeros((B, H, T, P))
    x = x + mx.arange(P, dtype=x.dtype)[None, None, None, :]  # identical per token
    out = chunk_rope(x, chunk=chunk_size)
    mx.eval(out)
    arr = np.array(out)[0, 0]
    np.testing.assert_allclose(arr[0], arr[1], atol=1e-6)
    np.testing.assert_allclose(arr[2], arr[3], atol=1e-6)
    np.testing.assert_allclose(arr[4], arr[5], atol=1e-6)
    # And different chunks get different rotations.
    assert not np.allclose(arr[0], arr[4], atol=1e-6)


def test_rope_preserves_norm():
    """RoPE is a rotation, so it cannot change the vector length."""
    mx.random.seed(1)
    x = mx.random.normal((2, 3, 16, 8))
    out = chunk_rope(x, chunk=4)
    mx.eval(out)
    a = np.array(x)
    b = np.array(out)
    np.testing.assert_allclose(
        np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1), atol=1e-5
    )
    # Positions inside one chunk share a rotation, so feeding them the same vector
    # gives the same vector out; a later chunk gives a different one, and chunk 0
    # has angle zero so it is the identity.
    same = mx.zeros((1, 1, 16, 8)) + mx.arange(8, dtype=mx.float32)[None, None, None, :]
    rot = np.array(chunk_rope(same, 4))[0, 0]
    np.testing.assert_allclose(rot[0], rot[3], atol=1e-6)
    assert not np.allclose(rot[0], rot[4], atol=1e-6)
    np.testing.assert_allclose(rot[0], np.array(same)[0, 0, 0], atol=1e-6)


def test_sliding_window_attention_is_causal_and_windowed():
    layer = SlidingWindowAttention(d_model=16, heads=2, window=4)
    x = rand_x(T=24, D=16)
    out = layer(x)
    mx.eval(out)
    # Causality.
    changed = mx.concatenate([x[:, :12], mx.ones_like(x[:, 12:])], axis=1)
    other = layer(changed)
    mx.eval(other)
    np.testing.assert_allclose(np.array(out)[:, :12], np.array(other)[:, :12], atol=1e-6)
    # Window: a change at position 0 cannot reach position 8 (8 - 0 >= window).
    changed0 = mx.concatenate([mx.ones_like(x[:, :1]), x[:, 1:]], axis=1)
    out0 = layer(changed0)
    mx.eval(out0)
    np.testing.assert_allclose(np.array(out)[:, 5:], np.array(out0)[:, 5:], atol=1e-6)


def test_model_config_hybrid_interleaves_local_attention():
    cfg = ModelConfig(vocab_size=17, d_model=32, n_layers=6, heads=2, chunk=8, hybrid=True)
    layers = build_layers(cfg)
    kinds = [type(layer).__name__ for layer in layers]
    assert kinds == [
        "SlidingWindowAttention",
        "RATLayer",
        "SlidingWindowAttention",
        "RATLayer",
        "SlidingWindowAttention",
        "RATLayer",
    ]
    plain = ModelConfig(vocab_size=17, d_model=32, n_layers=6, heads=2, chunk=8)
    assert all(type(layer).__name__ == "RATLayer" for layer in build_layers(plain))


def test_model_forward_loss_and_initial_entropy():
    cfg = ModelConfig(vocab_size=17, d_model=64, n_layers=2, heads=2, chunk=8)
    model = RATLM(cfg)
    mx.random.seed(0)
    tokens = mx.random.randint(1, 17, (3, 32))
    targets = mx.random.randint(1, 17, (3, 32))
    mask = mx.ones((3, 32), dtype=mx.bool_)
    logits = model(tokens)
    loss = model.loss(tokens, targets, mask)
    mx.eval(logits, loss)
    assert logits.shape == (3, 32, 17)
    assert np.isfinite(float(loss))
    # Targets are independent of the input, so the loss is the cross-entropy of a
    # near-uniform distribution over 17 symbols: log(17) = 2.83 is the floor and a
    # badly-scaled initialisation would overshoot badly.
    assert 2.0 < float(loss) < 6.0


def test_model_is_causal():
    cfg = ModelConfig(vocab_size=17, d_model=32, n_layers=2, heads=2, chunk=8)
    model = RATLM(cfg)
    mx.random.seed(2)
    tokens = mx.random.randint(1, 17, (2, 32))
    base = model(tokens)
    mx.eval(base)
    changed = mx.concatenate([tokens[:, :20], mx.ones_like(tokens[:, 20:]) * 5], axis=1)
    other = model(changed)
    mx.eval(other)
    np.testing.assert_allclose(
        np.array(base)[:, :20], np.array(other)[:, :20], atol=1e-5
    )


def test_model_loss_ignores_unmasked_positions():
    cfg = ModelConfig(vocab_size=17, d_model=32, n_layers=1, heads=2, chunk=8)
    model = RATLM(cfg)
    tokens = mx.random.randint(1, 17, (2, 16))
    targets = mx.random.randint(1, 17, (2, 16))
    mask = mx.zeros((2, 16), dtype=mx.bool_)
    mask[:, 3] = True
    loss = model.loss(tokens, targets, mask)
    mx.eval(loss)
    logp = mx.log(mx.softmax(model(tokens), axis=-1))
    picked = np.array(logp)[np.arange(2), 3, np.array(targets)[:, 3]]
    assert float(loss) == pytest.approx(float(-picked.mean()), abs=1e-5)


def test_model_state_per_token_scales_as_one_over_chunk():
    cfg = ModelConfig(vocab_size=17, d_model=32, n_layers=3, heads=2, chunk=4)
    model = RATLM(cfg)
    for chunk in (1, 4, 16):
        for T in (128, 512):
            # Three layers, each caching one key and one value of width d_model
            # per chunk: 3 * 2 * d_model / chunk per token, with no T in sight.
            want = 3 * 2 * 32 / chunk
            assert model.chunk_state_per_token(T, chunk) == pytest.approx(want, rel=1e-6)
    assert model.chunk_state_per_token(512, 16) < model.chunk_state_per_token(512, 1)


def test_count_parameters_matches_tree_flatten():
    from mlx.utils import tree_flatten

    cfg = ModelConfig(vocab_size=17, d_model=32, n_layers=2, heads=2, chunk=8)
    model = RATLM(cfg)
    assert count_parameters(model) == sum(v.size for _, v in tree_flatten(model.parameters()))
    assert count_parameters(model) > 0
