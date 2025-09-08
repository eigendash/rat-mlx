"""The toy decoder-only language model built from RAT blocks."""

import mlx.core as mx
import numpy as np
import pytest

from rat.model import ModelConfig, RATLM, count_parameters


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
