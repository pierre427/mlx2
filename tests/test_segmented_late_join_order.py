"""A lane parked by the segmented width lock must emit its first token first.

Regression: when the resident cohort drained, ``MTPGenerationBatch.next``
attached the parked arrival and immediately ran a proposal round, emitting the
lane's second token before its prepared first token.
"""

import mlx.core as mx
import pytest

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.sample_utils import LaneRNG
from mlx2.runtime.segmented_self_mtp import segmented_self_mtp_stats
from test_batched_mtp import _tiny_qwen4_model

PROMPTS = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19, 20, 21]]


def _run(model, config, *, join_step):
    batch = BatchGenerator(model, completion_batch_size=3, prefill_step_size=4, self_mtp=config)
    rows, lanes = {}, {}
    first = PROMPTS if join_step is None else PROMPTS[:2]
    uids = batch.insert(
        first, max_tokens=[10] * len(first),
        lane_rngs=[LaneRNG(i) for i in range(len(first))],
        self_mtp_configs=[{"sampling_temp": 0.0}] * len(first),
    )
    lanes.update({uid: index for index, uid in enumerate(uids)})
    try:
        for step in range(300):
            if step == join_step:
                (uid,) = batch.insert(
                    [PROMPTS[2]], max_tokens=[10], lane_rngs=[LaneRNG(9)],
                    self_mtp_configs=[{"sampling_temp": 0.0}],
                )
                lanes[uid] = 2
            _, responses = batch.next()
            for response in responses:
                rows.setdefault(lanes[response.uid], []).append(response.token)
            if len(rows) == 3 and all(len(row) >= 10 for row in rows.values()):
                return rows
    finally:
        batch.close()
    raise AssertionError("batch failed to complete")


def test_width_locked_late_joiner_emits_in_order(monkeypatch):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    monkeypatch.setenv("MLX_LM_SHARED_QSA_SUFFIX", "off")
    new_stream = mx.new_stream
    monkeypatch.setattr(mx, "new_stream", lambda device: new_stream(mx.cpu))
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(99)
        model = _tiny_qwen4_model()
        config = {"persistent": True, "num_draft": 2, "segment_aware_live_tip": True,
                  "segment_aware_cohort_size": 3}
        reference = _run(model, config, join_step=None)
        before = segmented_self_mtp_stats()["live_width_change_deferrals"]
        joined = _run(model, config, join_step=3)
        assert segmented_self_mtp_stats()["live_width_change_deferrals"] > before
        assert joined == reference
    finally:
        mx.set_default_device(previous)


def test_bucketed_attention_before_append_is_a_lifecycle_error():
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.segmented_plain_kv import SegmentedBatchKVCache

    cache = SegmentedBatchKVCache([KVCache(), KVCache()])
    with pytest.raises(RuntimeError, match="appended step"):
        cache.bucketed_attention(mx.zeros((2, 1, 1, 4)), 1.0, None)
