"""F12: cohort lane-exact admission honours the executor's declared lane limit.

``Qwen38FastAST.execute`` (and the Flash-Next tree target) declare
``progressive_external_exact_max_lanes == 1`` and raise for wider slabs, but
the cohort forward admitted ``lane_exact`` for any 2..8-lane batch.
"""

from types import SimpleNamespace

import mlx.core as mx

from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.progressive_external_verify import (
    PROGRESSIVE_FIDELITY_EXACT,
    cohort_exact_lane_limit,
)


class _Target:
    def __init__(self, max_lanes):
        self.calls = []
        if max_lanes is not None:
            self.progressive_external_exact_max_lanes = max_lanes

    def progressive_external_target_rows(self, *args, **kwargs):
        raise AssertionError("executor launched outside its declared lane limit")

    def forward_with_taps(self, inputs, caches, layers):
        self.calls.append(tuple(inputs.shape))
        return mx.zeros((*inputs.shape, 4)), mx.zeros((*inputs.shape, 2))


def _forward(model, batch):
    fake = SimpleNamespace(
        ast_tree_tile_rows=0,
        model=model,
        progressive_verification_fidelity=PROGRESSIVE_FIDELITY_EXACT,
        scheduler_stats={},
        layers=(0,),
    )
    transaction = SimpleNamespace(batched=True, plain=False, caches=None)
    cohort = [SimpleNamespace(ordinary=False) for _ in range(batch)]
    ExternalDraftBatchGenerator._cohort_target_forward(
        fake, mx.zeros((batch, 3), mx.int32), transaction, cohort, site="t"
    )
    return fake.scheduler_stats


def test_single_lane_executor_is_not_launched_for_a_two_lane_cohort():
    model = _Target(max_lanes=1)
    stats = _forward(model, 2)
    assert model.calls == [(2, 3)]
    assert stats == {"external_cohort_target_batched_t_rounds": 1}


def test_lane_limit_helper():
    assert cohort_exact_lane_limit(_Target(None)) == 8
    assert cohort_exact_lane_limit(_Target(1)) == 1
    assert cohort_exact_lane_limit(_Target(4)) == 4
    assert cohort_exact_lane_limit(_Target(16)) == 8
    assert cohort_exact_lane_limit(_Target(True)) == 0
    assert cohort_exact_lane_limit(_Target(0)) == 0
