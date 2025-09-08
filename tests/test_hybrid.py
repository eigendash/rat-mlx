"""The hybrid stack: sliding-window local attention interleaved with RAT."""

import mlx.core as mx
import numpy as np

from rat.block import SlidingWindowAttention
from rat.model import ModelConfig, build_layers


def rand_x(B=2, T=24, D=16, seed=0):
    mx.random.seed(seed)
    return mx.random.normal((B, T, D))


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
