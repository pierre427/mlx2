"""mtp_cycle_state reads each lane's context length from the prefix array's
shape; it must not convert the whole context to Python (2026-09-23 audit:
~4 ms per conversion at 128K, up to 10 conversions per MTP cycle)."""
import mlx.core as mx
import pytest

from mlx2.runtime.generate import MTPGenerationBatch
from test_promotion_pressure import setup_batch


class _Lane:
    def __init__(self, prefix):
        self.token_prefix = prefix


@pytest.mark.parametrize("prefix", [mx.array([5, 6, 7], dtype=mx.uint32),
                                    mx.array([[5, 6, 7, 8]], dtype=mx.uint32),
                                    mx.array([], dtype=mx.uint32)])
def test_prefix_length_matches_token_list(prefix):
    lane = _Lane(prefix)
    assert MTPGenerationBatch._prefix_length(lane) == len(MTPGenerationBatch._prefix_tokens(lane))


def test_cycle_state_never_converts_the_context(monkeypatch):
    batch, _ = setup_batch(40.0, lanes=2)  # the segmented (default MTP) state
    try:
        old_contexts = [len(MTPGenerationBatch._prefix_tokens(lane)) + 1 for lane in batch.state.lanes]

        def forbidden(lane):
            raise AssertionError("mtp_cycle_state converted a lane's whole context")

        monkeypatch.setattr(MTPGenerationBatch, "_prefix_tokens", staticmethod(forbidden))
        assert [row[1] for row in batch.mtp_cycle_state()] == old_contexts
    finally:
        batch.close()
