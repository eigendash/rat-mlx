"""Chunk-based RAT sequence mixing. See ``rat/attention.py`` for the maths."""

from .attention import (
    chunk_state_floats,
    full_causal_attention,
    inter_chunk_summary,
    intra_chunk_scan,
    intra_chunk_scan_recurrent,
    rat_attention,
    rat_attention_streaming,
)
from .block import (
    RATLayer,
    SlidingWindowAttention,
    chunk_rope,
    chunk_rope_cos_sin,
)
from .model import RATLM, ModelConfig, build_layers, count_parameters
from .task import (
    Batch,
    copy_accuracy_by_distance,
    make_copy_batch,
    make_recall_batch,
    random_baseline_accuracy,
    source_offsets,
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
    "chunk_rope_cos_sin",
    "RATLM",
    "ModelConfig",
    "build_layers",
    "count_parameters",
    "Batch",
    "copy_accuracy_by_distance",
    "make_copy_batch",
    "make_recall_batch",
    "random_baseline_accuracy",
    "source_offsets",
]
