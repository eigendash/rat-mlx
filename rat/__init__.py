"""Chunk-based sequence mixers. See ``rat/attention.py`` for the maths."""

from .attention import (
    chunk_state_floats,
    full_causal_attention,
    inter_chunk_summary,
    intra_chunk_scan,
    intra_chunk_scan_recurrent,
    rat_attention,
    rat_attention_streaming,
)
from .block import RATLayer, SlidingWindowAttention, chunk_rope
from .model import RATLM, ModelConfig, build_layers, count_parameters
from .task import (
    InductionBatch,
    make_induction_batch,
    n_symbols,
    random_baseline_accuracy,
    retrieval_accuracy,
)

__all__ = [
    "chunk_state_floats",
    "full_causal_attention",
    "inter_chunk_summary",
    "intra_chunk_scan",
    "intra_chunk_scan_recurrent",
    "rat_attention",
    "rat_attention_streaming",
    "RATLayer",
    "SlidingWindowAttention",
    "chunk_rope",
    "RATLM",
    "ModelConfig",
    "build_layers",
    "count_parameters",
    "InductionBatch",
    "make_induction_batch",
    "n_symbols",
    "random_baseline_accuracy",
    "retrieval_accuracy",
]
