"""``inkling2_verify_deferred_commit`` on Metal: the shipped defaults, with
split-K reading the deferred view through its real kernel (``ring=`` args),
against the record-and-replay rollback path, bitwise.

Run under the GPU lock (lease queue), never as part of a plain pytest run on a
shared host. Tiny geometry (window 8, head_dim 32), FP32 and BF16 weights;
random verify widths, accepts, plain rounds and aborts lapping the ring.
"""

import mlx.core as mx
import pytest

import test_inkling2_deferred_commit_cpu as base

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("seed", range(3))
def test_default_policy_rounds_match_rollback_on_metal(dtype, seed):
    with mx.stream(mx.gpu):
        rng, arms, owners = base._arms({}, seed, dtype=dtype)
        for _ in range(14):
            base._round(rng, arms, owners)
        receipt = arms[1][0].kernel_receipt()
    assert receipt["verify_splitk_calls"] > 0
    assert receipt["verify_splitk_ring"].get("calls", 0) > 0
    assert receipt["verify_deferred_commit"]["ring_calls"] > 0
    assert receipt["verify_deferred_commit"]["ring_declines"] == {}
