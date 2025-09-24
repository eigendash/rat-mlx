# rat-mlx

Chunk-based RAT sequence mixing in plain MLX, plus a small decoder-only language
model to try it on. It runs on Apple silicon with no custom kernels.

The method is from *RAT: Bridging RNN Efficiency and Attention Accuracy via
Chunk-based Sequence Modeling* (arXiv:2507.04416). RAT cuts a sequence into
chunks, applies a gated linear recurrence **inside** each chunk to model local
dependencies, and softmax attention **across** the per-chunk representations to
reach distant tokens. With chunk size 16 the paper reports 7x faster training at
100K sequence length and 9x faster generation at position 4K, while matching
standard attention, and it proposes a hybrid that interleaves RAT with
sliding-window local attention.

This is a small independent implementation, **not** a reproduction of the paper's
results. The paper trains 1.3B-parameter models from scratch on web data; the
model here has 41,792 parameters and trains for 600 steps on a synthetic copy
task in under a minute. The numbers below are mine and are not comparable to the
paper's.

## What is in here

**`rat/attention.py` — the temporal mixing.** A sequence of length `T` is cut
into `C` chunks of length `L` (`T = C·L`); position `t` is `(c, l)`. Following the
paper's Eq. (3), a forget gate `g` (per dimension, `sigmoid` of a linear
projection of the input) runs a first-order recurrence inside each chunk, with its
state reset to zero at every chunk boundary:

```
v~_{c,l} = g_{c,l} ⊙ v~_{c,l-1} + (1 - g_{c,l}) ⊙ v_{c,l}
k~_{c,l} = g_{c,l} ⊙ k~_{c,l-1} + (1 - g_{c,l}) ⊙ k_{c,l}
```

The **value** is what carries local content forward; the same recurrence over the
**key** produces the summaries that inter-chunk attention reads. `intra_chunk_scan`
is written as an explicit loop over the chunk, so `T` never enters the loop
depth, only `L` does. The paper uses a parallel associative scan; MLX's autodiff
differentiates the unrolled loop without a hand-written backward pass, and for
`L ≤ 64` that is a handful of elementwise ops per step.

Inter-chunk attention over the end-of-chunk summaries `K~_{:,-1}`, `V~_{:,-1}`
(the paper's notation) is called `inter_chunk_summary`. For a query at `(c, l)`
the visible set is the summaries of chunks **strictly before** `c`, plus a
same-position term from the chunk in progress. The two branches are merged the
way the paper's Appendix A.1 (`merge_last_token_stable`) does it, as one softmax
over the union of their keys and a single denominator.

The same-position term follows the paper's pseudocode and its released
implementation literally: they compute the per-chunk contribution as an
elementwise `einsum` of the query with the gated key **at the same position**, so
a token in a chunk does *not* see its earlier neighbours inside that chunk — that
is the recurrence's job. `tests/test_attention.py` pins this down against an
explicit per-position reference, so an accidental change to the mask fails the
suite.

`rat_attention_streaming` is the token-by-token decoding path: keep the running
gated key/value of the chunk in progress, promote it to a chunk summary when the
chunk ends, and softmax over (finished summaries + the running pair). A test
requires the parallel training form and the sequential form to agree to 1e-5 for
every chunk size.

**`rat/block.py` — the layer.** `RATLayer` holds five `D×D`-equivalent
projections, which is the paper's `4D²` budget: output projection, a shared
query/key projection, the value projection, and the two gates. The paper reports
that sharing `q`/`k` slightly beats low-rank gates, so the projection is shared
(`in_proj` emits `z | x | g | qk`, and `q` and `k` are the same tensor). The
forget-gate bias is initialised log-uniform in `[1, L-1]`, following the released
implementation, so that a chunk's worth of gates decays at a rate matched to the
chunk length; the paper does not specify this initialisation.

Positional information is chunk-level RoPE (`chunk_rope`): every token in a chunk
gets the same rotation, so only the *relative* chunk index matters and the
same-position diagonal term cancels it exactly. The paper says it uses "RoPE
based on chunk indices for inter-chunk attention" without giving a convention;
this is half-split RoPE with base 10000. `SlidingWindowAttention` is the local
half of the hybrid: causal attention over the last `window` tokens.

Query and key vectors are L2-normalised per head before attention. That is **not**
in the paper, and it is the one deliberate addition of mine: at this toy scale the
raw dot products grew past the range where a float32 softmax keeps a usable
backward pass, and normalising bounds every logit by the head dimension no matter
how large the weights get. It is monotone, so it cannot change which entry wins.

**`rat/model.py` — the language model.** `RATLM` is token embedding → mixing
layers → tied LM head. `build_layers` places layers according to the config; with
`hybrid=True` it puts local attention where `layer_index % interleave_step == 0`
and RAT elsewhere, which mirrors the released implementation's
`sequence_interleave` backbone. `masked_loss` is cross-entropy over scored
positions only (the name is not `loss` because `mlx.nn.Module` reserves that
attribute).

**`rat/task.py` — the tasks.** `make_copy_batch` builds the copy task used in the
experiment: a sequence whose second half repeats the first half, scored only on
the second half, so every scored token has to be read from exactly `T/2`
positions earlier. `source_offsets` and `copy_accuracy_by_distance` produce the
per-distance breakdown. `make_recall_batch` is the associative-recall task used
for the negative result below.

## Checks

`pytest` runs **37 tests** across four files. The ones that carry the most
weight:

* **`c = 1` is ordinary causal attention.** With one token per chunk, the
  end-of-chunk summary of chunk `t` *is* token `t`, the diagonal entry adds it
  back, and the block reduces exactly to causal softmax attention over the gated
  keys and values `((1-g)·k, (1-g)·v)` — checked against a `mx.fast`
  implementation to 1e-5. The paper states this limit ("when `L = 1`, it resembles
  full attention", Subsec. 2.3); the check is against *gated* keys and values,
  which is what the architecture actually computes, so it is an implementer
  sanity check rather than the paper's exact claim.
* **`c = T` is the pure recurrence.** With one chunk there is no earlier chunk,
  the softmax is a single entry, and the output is `z ⊙ ṽ`, the paper's Eq. (2).
  Also stated in the paper ("when `L = T`, it reduces to a pure RNN").
* **The visible set is right.** A numpy reference builds the allowed pairs per
  query from the description above and the two agree to 2e-5 for `c ∈ {1,2,4,8}`.
  This is the test that fails if a query sees its own chunk's end summary or a
  later chunk.
* **Causality.** Changing tokens after position `t` leaves the output at or before
  `t` unchanged, for attention, for the layer and for the model.
* **Parallel equals sequential.** The chunked training form and the token-by-token
  decoding form agree to 1e-5 for six (length, chunk) combinations.
* **Hand-computed values.** A four-token, two-chunk case with `g = 0.5`,
  `q = 0`, `k = v =` the token index is worked out by hand in the test and
  compared to the implementation.
* **State size does not grow with sequence length.** Per-token generation state
  is exactly `2·d_model/L`, independent of `T`; the counting helper is checked
  against the layer's own method.
* **Numerical stability.** Scaling queries and keys until the inter-chunk
  log-sum-exp saturates exercises the merge, and the values and gradients must
  stay finite. An earlier version of the merge produced `nan` here, which is
  exactly how the chunk-size sweep first diverged.

## Experiment

`scripts/copy_sweep.py` trains a 41,792-parameter decoder (`d_model=64`, 2
layers, 4 heads) from scratch on the copy task for 600 steps at batch 32,
sequence length 512, AdamW with a cosine schedule from 1e-3 and grad-norm
clipping at 1.0, seed 0. Every configuration sees the same stream of
freshly generated batches. Held-out evaluation uses 4 batches at length 512 and 2
at length 1024, which the model never trained on.

| configuration | c | hybrid | held-out loss | accuracy | accuracy @ 2T | state/token | params |
|---|---:|---|---:|---:|---:|---:|---:|
| c=1 (attention) | 1 | no | 0.3413 | 0.875 | 0.880 | 256.0 | 41792 |
| c=4 | 4 | no | 0.3310 | 0.875 | 0.880 | 64.0 | 41792 |
| c=16 | 16 | no | 0.2454 | 0.855 | 0.876 | 16.0 | 41792 |
| c=64 | 64 | no | 0.1830 | 0.828 | 0.854 | 4.0 | 41792 |
| c=512 (=T, pure RNN) | 512 | no | 0.1562 | 0.828 | 0.793 | 0.5 | 41792 |
| c=16 + local attention | 16 | yes | 0.2985 | 0.869 | 0.880 | 16.0 | 37632 |

Mean held-out loss is over the scored (second-half) positions; accuracy is exact
match at those positions. The constant-answer baseline — always emitting the most
common value symbol — is 0.131, so every configuration is far above chance.

What the numbers say, and what they do not:

* **Long-range access works at every chunk size.** Every configuration copies
  tokens that sit 256 positions back. The per-distance breakdown collapses to a
  single populated bucket (`256+`) for this reason; the table above is therefore
  also the 256+-distance accuracy.
* **Smaller chunks are better here, and larger chunks are better on loss.** The
  loss falls monotonically with chunk size (0.3413 at `c=1` to 0.1562 at
  `c=512`), while exact-match accuracy falls from 0.875 to 0.828. Loss and
  accuracy disagree because most errors are close calls in the logits; the pure
  RNN is more confident and slightly less often exactly right. I did not run more
  seeds, so I cannot say how much of the 0.047 accuracy gap between `c=1` and
  `c=512` is noise.
* **The state comparison is the real result.** At `c=512` the model holds 0.5
  floats per token — 512x less than full attention — and still copies 82.8% of
  the tokens exactly. That is the trade the paper is about, and it is visible at
  this scale even though the accuracy difference is not statistically
  established.
* **The hybrid is in between.** `c=16 + local attention` gets 0.869 accuracy and
  0.880 at double length, the best generalisation of the `c=16` family, with an
  accuracy between `c=4` and `c=64`.
* **Length generalisation degrades with chunk size.** Trained at 512 and
  evaluated at 1024, the small chunks hold or improve (0.875 → 0.880, 0.855 →
  0.876) while the pure RNN drops (0.828 → 0.793). This is the memory-degradation
  effect the paper describes for full-sequence compression, visible here at toy
  scale.
* **The paper's 7x/9x speedups are not measured here.** No timing comparison is
  reported, because MLX's `mx.fast` attention is a fused kernel and this chunked
  path is a sequence of matmuls, so a timing table at this size would mostly
  measure kernel launch, not the algorithm.

### The recall task is a negative result

`scripts/recall_check.py` runs the same sweep on a different task: 16 key/value
pairs planted at the front of a 256-symbol sequence, filler, then 8 queries for
planted values. This needs *matching* rather than copying. Every configuration —
including full attention at `c=1` — sits on the constant-answer baseline
(accuracy 0.063–0.068 against a baseline of 0.081, loss 2.773 ≈ `ln(33)`).
The task is not learnable by these models at this scale, so those numbers say
nothing about the chunk-size trade-off. It is recorded because a check that fails
is still information.

### Divergence, reported rather than hidden

The sweep allows two attempts per configuration: the nominal learning rate
(1e-3), then a third of it if the first attempt's loss goes non-finite. Both
attempts see the same batch stream and the attempt that produced the reported
numbers is in `results/copy_sweep.json` (`"attempt": 1` for every row above, so
the fallback was not needed for the committed run). Earlier iterations of this
code diverged at 3e-3 for individual chunk sizes; the merge was rewritten to
clamp the logits and mask finitely, which is what
`tests/test_attention.py::test_merge_stays_finite_when_the_logits_are_large`
now protects.

## Where this may differ from the paper

* **Scale.** 41,792 parameters and 600 steps on a synthetic task. The paper's
  1.3B-parameter, long-context and LongBench numbers are not reproduced and are
  not comparable.
* **The intra-chunk reading.** The paper's Eq. (3) writes the inter-chunk
  attention over `[K~_{:,-1}; k~_{c,l}]`, which could be read as a full causal
  softmax inside the current chunk. Its Appendix A.1 pseudocode and the released
  implementation compute only the same-position term (an `einsum` of the query
  with the gated key at the same index) and merge it with the online softmax. I
  implemented the second reading, because it is what the released code runs and
  it is what the sequential decoding path does. The consequence is that `c = 1`
  reduces to causal attention over *gated* keys and values rather than the raw
  projections, and a token cannot attend to earlier positions within its own
  chunk. A full-causal-inside-the-chunk variant is a few lines away (the
  `mode="chunk"` branch of a scratch experiment used while writing this repo) and
  I did not find it materially better or worse on the copy task, but I did not
  run it as part of the committed sweep.
* **Query/key sharing and gate initialisation** are taken from the released
  implementation rather than the paper text, as described above.
* **L2 normalisation of queries and keys is mine.** The paper does not do this.
  It bounds the logits and is monotone, but it changes the geometry of the
  attention scores relative to what the paper would compute at scale.
* **RoPE convention.** Chunk-level RoPE is implemented as half-split RoPE with
  base 10000. The paper says it uses chunk-index RoPE but gives no convention.
  The experiment itself runs with `rope=False` (no positional encoding at all),
  because the copy task is content-based and the model does not need positions;
  `RATLayer(rope=True)` is the default and is tested separately.
* **No fused kernels, no tensor/context parallelism, no CUDA.** The paper's
  efficiency numbers come from `flex_attention` and an associative-scan kernel on
  H100s.
* **`c = L = 512` needs `T` to be a multiple of the chunk.** The implementation
  raises otherwise, and sequences shorter than one chunk are not supported.
* **The 2T evaluation is skipped when 1024 is not a multiple of the chunk size**
  (so `c = 512` has no `acc@2T` entry; `c = 64` does, since 1024 = 16·64).

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/copy_sweep.py
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/recall_check.py
```

The experiments write `results/copy_sweep.{txt,json}` and
`results/recall_check.{txt,json}`; the committed copies are the ones quoted
above. `--quick` runs a two-configuration smoke test in seconds.
