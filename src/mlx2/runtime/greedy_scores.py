"""Opt-in greedy reduction with deferred full-vocabulary logprob storage.

The subtraction and argmax preserve the served FP32 normalized selection law;
logsumexp is deliberately retained because subtracting it can create new ties.
No probability is fabricated: a consumer requesting the full row gets the same
normalization, materialized then. See provenance/nanbeige43-generation.json.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import mlx.core as mx

SCHEMA = "mlx2.normalized-greedy-lazy-scores.v1"
_SOURCE = r"""
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_threadgroup;
    uint width = logits_shape[1];
    threadgroup float values[8];
    threadgroup uint indices[8];
    float best = -INFINITY;
    uint index = 0xffffffffu;
    for (uint col = lane; col < width; col += 256) {
        float value = logits[row * width + col] - normalizers[row];
        if (value > best || (value == best && col < index)) {
            best = value;
            index = col;
        }
    }
    float group_best = simd_max(best);
    uint group_index = simd_min(best == group_best ? index : 0xffffffffu);
    if (thread_index_in_simdgroup == 0) {
        values[simdgroup_index_in_threadgroup] = group_best;
        indices[simdgroup_index_in_threadgroup] = group_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32) {
        float value = lane < 8 ? values[lane] : -INFINITY;
        uint col = lane < 8 ? indices[lane] : 0xffffffffu;
        float result = simd_max(value);
        uint result_index = simd_min(value == result ? col : 0xffffffffu);
        if (lane == 0) selected[row] = result_index == 0xffffffffu ? 0u : result_index;
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="normalized_greedy_select", input_names=["logits", "normalizers"],
        output_names=["selected"], source=_SOURCE,
    )


class LazyLogprobs:
    """A normalized row that keeps only raw logits and a scalar normalizer."""
    def __init__(self, logits, normalizer):
        self.logits, self.normalizer = logits, normalizer
        self.shape, self.dtype, self.ndim = logits.shape, logits.dtype, logits.ndim

    def materialize_logprobs(self):
        return self.logits - self.normalizer

    def __getitem__(self, index):
        return self.logits[index] - self.normalizer

    def astype(self, dtype):
        return self.materialize_logprobs().astype(dtype)

    def tolist(self):
        return self.materialize_logprobs().tolist()


def materialize_logprobs(scores):
    method = getattr(scores, "materialize_logprobs", None)
    return method() if method is not None else scores


def score_eval_targets(scores):
    """Device arrays needed for scores, without forcing normalized row output."""
    outputs = []
    for row in scores:
        if isinstance(row, LazyLogprobs):
            outputs.extend((row.logits, row.normalizer))
        else:
            outputs.append(row)
    return outputs


def greedy_scores(logits):
    if logits.ndim != 2 or logits.shape[-1] < 1:
        raise ValueError("greedy scores require nonempty [batch,vocabulary] logits")
    logits = mx.contiguous(logits.astype(mx.float32))
    normalizers = mx.logsumexp(logits, axis=-1)
    if mx.default_device() == mx.cpu:
        selected = mx.argmax(logits - normalizers[:, None], axis=-1)
    else:
        selected = _kernel()(
            inputs=[logits, normalizers], grid=(256, logits.shape[0], 1),
            threadgroup=(256, 1, 1), output_shapes=[(logits.shape[0],)],
            output_dtypes=[mx.uint32],
        )[0]
    return selected, [LazyLogprobs(logits[i], normalizers[i]) for i in range(logits.shape[0])]


def admitted(model, samplers, fallback):
    return bool(getattr(model, "_mlx2_lean_greedy", False)) and all(
        bool(getattr(sampler or fallback, "greedy_argmax", False)) for sampler in samplers
    )


def receipt():
    return {"schema": SCHEMA, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selection_law": "FP32 logits minus served logsumexp; first-index argmax",
            "normalization": "full row lazy; scalar normalizer retained",
            "qualified": False}
