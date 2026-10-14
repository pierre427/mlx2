"""Copy-vs-head gate priced by measured verify-row costs (default off).

The gate compares emitted tokens per unit of round cost for copied and
head-drafted rounds.  Its default cost model is linear in verify rows
(``verify_row_cost`` a row).  On the lane kernels a verify window costs about
the same from one row to the tile edge, so the linear model overprices wide
copies and declines copies that pay.  ``verify_row_costs`` lets the policy
price a round from a measured cost per verify-row count instead (the
row-cost staircase the verify-topology probe records), as TensorFold #788's
CostDepth prices draft depth from each window width's measured round time.
"""

import math

import pytest

from mlx2.runtime.copy_draft import CopyDraftPolicy, CopyDraftState

CYCLE = [11, 23, 5, 42, 17, 30, 8, 51, 2, 36]
# M3 Pro lane-simd shape: flat to an 8-row tile, then a step (ms).
TILE = {1: 40.0, 2: 40.0, 4: 40.4, 8: 41.0, 9: 52.0, 16: 54.0}


def test_default_policy_is_unchanged_and_its_receipt_omits_the_table():
    policy = CopyDraftPolicy(enabled=True)
    assert policy.verify_row_costs is None
    assert "verify_row_costs" not in policy.as_dict()
    assert policy.head_cost(3) == pytest.approx(1.0 + 0.25 * 3)
    assert policy.copy_cost(7) == pytest.approx(1.0 + 0.1 * 7)


def test_a_linear_table_reproduces_the_default_cost_model():
    """Falsifier for the normalization: a table that encodes the default
    model's own line gives the default model's costs."""
    linear = {rows: 1.0 + 0.1 * (rows - 1) for rows in (1, 2, 5, 9)}
    measured = CopyDraftPolicy(enabled=True, verify_row_costs=linear)
    default = CopyDraftPolicy(enabled=True)
    for depth in range(0, 12):
        assert measured.head_cost(depth) == pytest.approx(default.head_cost(depth))
        assert measured.copy_cost(depth) == pytest.approx(default.copy_cost(depth))


def test_tile_table_prices_rows_by_the_staircase():
    policy = CopyDraftPolicy(enabled=True, verify_row_costs=TILE)
    assert policy.copy_cost(7) == pytest.approx(41.0 / 40.0)        # 8 rows, inside the tile
    assert policy.copy_cost(8) == pytest.approx(52.0 / 40.0)        # 9 rows, past the edge
    assert policy.copy_cost(2) == pytest.approx(40.2 / 40.0)        # 3 rows, interpolated
    assert policy.copy_cost(20) == pytest.approx((54.0 + 5 * 2 / 7) / 40.0)  # extrapolated
    assert policy.head_cost(3) == pytest.approx(40.4 / 40.0 + 0.15 * 3)


def _fill(state, *, copy_emitted, head_emitted):
    for _ in range(4):
        state.record(copy_span=7, head_depth=0, accepted=copy_emitted - 1,
                     emitted=copy_emitted, committed=[])
    for emitted in head_emitted:
        state.record(copy_span=0, head_depth=3, accepted=emitted - 1,
                     emitted=emitted, committed=[])


def test_measured_costs_copy_where_the_linear_model_declines():
    context = CYCLE * 4
    kwargs = dict(enabled=True, max_span=8, min_samples=3, probe_span=2)
    linear = CopyDraftState(CopyDraftPolicy(**kwargs), context)
    tiled = CopyDraftState(CopyDraftPolicy(**kwargs, verify_row_costs=TILE), context)
    for state in (linear, tiled):
        _fill(state, copy_emitted=2, head_emitted=[2, 3, 2, 3])
    # Linear: copy 2/1.7 = 1.18 per unit against head 2.5/1.75 = 1.43.
    assert linear.plan(head_depth=3, cap=8)[1] == "declined"
    # Measured: copy 2/1.025 = 1.95 against head 2.5/1.46 = 1.71.
    assert tiled.plan(head_depth=3, cap=8)[1] == "copy"


def test_the_table_round_trips_and_survives_the_adapter_clamp():
    policy = CopyDraftPolicy.from_value(
        {"enabled": True, "max_span": 16, "verify_row_costs": {"1": 2.0, "8": 2.0, "9": 3.0}}
    )
    data = policy.as_dict()
    assert data["verify_row_costs"] == {"1": 2.0, "8": 2.0, "9": 3.0}
    assert CopyDraftPolicy.from_value(data) == policy
    clamped = policy.clamped_to_self_mtp_proposer_depth(8)
    assert clamped.max_span == 8 and clamped.verify_row_costs == policy.verify_row_costs
    assert hash(policy) == hash(CopyDraftPolicy.from_value(data))


@pytest.mark.parametrize(
    "table",
    [
        {},
        {2: 1.0, 4: 1.2},                 # no one-row cost to normalize by
        {1: 0.0, 2: 1.0},
        {1: 1.0, 2: -1.0},
        {1: 1.0, 2: math.inf},
        {1: 1.0, 0: 1.0},
        {1: 1.0, "x": 1.0},
        {1: True},
        [1.0, 2.0],
    ],
)
def test_malformed_tables_are_refused(table):
    with pytest.raises(ValueError, match="verify_row_costs"):
        CopyDraftPolicy(enabled=True, verify_row_costs=table)


def test_a_decreasing_tail_extends_flat_never_below_its_last_measurement():
    """Codex review 2026-10-14: a measured table whose last segment falls
    (timing noise: {1: 2.0, 2: 1.0}) was extrapolated along that slope, so a
    wide copy cost -3.0 and the gate's ratio-of-sums turned off.  Beyond the
    widest measured count there is no evidence the cost keeps falling: the
    tail extends flat.  Measured points inside the table stay as measured."""
    policy = CopyDraftPolicy(enabled=True, max_span=8, verify_row_costs={1: 2.0, 2: 1.0})
    assert policy.copy_cost(1) == pytest.approx(0.5)            # measured, kept
    assert policy.copy_cost(8) == pytest.approx(0.5)            # flat past row 2
    assert policy.head_cost(3) == pytest.approx(0.5 + 0.15 * 3)
    for span in range(0, 64):
        assert policy.copy_cost(span) > 0 and policy.head_cost(span) > 0
    # A rising tail still extends along its own slope.
    rising = CopyDraftPolicy(enabled=True, verify_row_costs={1: 1.0, 2: 2.0})
    assert rising.copy_cost(3) == pytest.approx(4.0)


def test_a_decreasing_tail_does_not_switch_the_gate_off():
    context = CYCLE * 4
    policy = CopyDraftPolicy(enabled=True, max_span=8, min_samples=3, probe_span=2,
                             verify_row_costs={1: 2.0, 2: 1.0})
    state = CopyDraftState(policy, context)
    for _ in range(4):
        state.record(copy_span=8, head_depth=0, accepted=8, emitted=9, committed=[])
    for _ in range(4):
        state.record(copy_span=0, head_depth=3, accepted=1, emitted=2, committed=[])
    # Copies emit 9 tokens a round against the head's 2: they must keep copying.
    assert state.plan(head_depth=3, cap=8)[1] == "copy"


def test_row_cost_validation_does_not_walk_every_span_width():
    """Codex follow-up 2026-10-14: validation iterated every verify width up
    to max_span + 1, and max_span is unbounded before the adapter clamp, so a
    huge span stalled construction.  The cost is piecewise linear, so checking
    the endpoints and measured rows is exact and O(table)."""
    import time

    start = time.perf_counter()
    policy = CopyDraftPolicy(
        enabled=True, max_span=10**9, verify_row_costs={1: 2.0, 2: 1.0, 4: 3.0}
    )
    assert time.perf_counter() - start < 1.0
    assert policy.copy_cost(10**6) > 0
