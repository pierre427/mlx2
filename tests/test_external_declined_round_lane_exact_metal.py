"""Real-Metal gate: a progressive-declined B2 external round keeps B1 bits.

Before the fix, the declined round's standard verify ran the cohort as one
batched forward over segmented views (masked multi-row recurrent and SDPA
kernels), which on Metal is not bit-equal to each lane verifying alone.  The
round now runs the adapter-owned row-exact executor; every lane's committed
history and target cache must equal its own B1 run, bit for bit.
"""

import mlx.core as mx
import pytest

from tests.test_external_declined_round_lane_exact_cpu import (
    PROMPTS,
    _greedy_rows,
    _run,
    _selected_target,
)
from tests.test_nemotron35_progressive_cpu import exact_caches

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="needs Metal")


@pytest.fixture(autouse=True)
def gpu():
    previous = mx.default_device()
    try:
        yield
    finally:
        mx.set_default_device(previous)


def test_stop_declined_b2_round_is_bit_equal_to_each_lane_alone():
    stop = _greedy_rows(_selected_target(mx.gpu), PROMPTS)[0][5]
    batch, lanes, _ = _run(PROMPTS, stop=stop, device=mx.gpu)
    references = []
    try:
        for prompt, lane in zip(PROMPTS, lanes):
            ref_batch, (ref_lane,), _ = _run([prompt], stop=stop, device=mx.gpu)
            references.append(ref_batch)
            assert lane.history == ref_lane.history
            assert [item.token for item in lane.ready] == [
                item.token for item in ref_lane.ready
            ]
            exact_caches(lane.cache, ref_lane.cache)
        assert batch.scheduler_stats.get(
            "external_cohort_target_lane_exact_standard_verify_rounds"
        ) == 1
    finally:
        batch.close()
        for ref in references:
            ref.close()


def test_ordinary_b2_rounds_are_bit_equal_to_each_lane_alone():
    from tests.test_external_declined_round_lane_exact_cpu import _ordinary_steps

    batch, lanes = _ordinary_steps(PROMPTS, device=mx.gpu)
    references = []
    try:
        for prompt, lane in zip(PROMPTS, lanes):
            ref_batch, (ref_lane,) = _ordinary_steps([prompt], device=mx.gpu)
            references.append(ref_batch)
            assert lane.history == ref_lane.history
            assert [item.token for item in lane.ready] == [
                item.token for item in ref_lane.ready
            ]
            exact_caches(lane.cache, ref_lane.cache)
    finally:
        batch.close()
        for ref in references:
            ref.close()
