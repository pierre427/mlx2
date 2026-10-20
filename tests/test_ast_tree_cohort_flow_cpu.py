"""CPU flow composition, not qualification of TensorFold's device arithmetic."""

import copy
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.target_tree_executor import TargetTreeDispatchMetrics
from test_ast_fast_tree_policy_cpu import _batch, _CPUProvider
from test_ast_terminal_ordinary_cpu import _committed_digest
from test_qwen38_dflash2_cpu import PROMPTS


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class _CohortProvider(_CPUProvider):
    """The real dispatch contract over a CPU stock-chain transaction oracle."""

    @property
    def ast_tree_max_sessions(self):
        return 4

    def ast_tree_cohort_input_widths(self, width):
        return tuple(range(1, 16 // width + 1)) if 2 <= width <= 4 else ()

    def validate_cohort(self, width, limit):
        assert 1 <= width <= limit == 4

    def dispatch_metrics(self, width, transaction, *, cached):
        return TargetTreeDispatchMetrics(
            width=width,
            physical_packed=width > 1,
            executor_cached=True,
            cached=cached,
            provider_protocol=self.protocol,
        )

    def forward_many(
        self, model, inputs, parents, caches, layers, *, cached, configured_limit
    ):
        assert cached is True and configured_limit == 4
        assert all(row == list(range(-1, len(row) - 1)) for row in parents)
        tx = HybridVerifyRows(caches).begin([len(row) for row in inputs])
        logits, features = model.forward_with_taps(mx.array(inputs), tx.caches, layers)
        provider = self

        class Transaction:
            closed = False
            provider_protocol = provider.protocol
            provider_revision_token = provider.source_revision_token

            def commit_paths(self, paths):
                assert all(path == list(range(len(path))) for path in paths)
                tx.commit([len(path) for path in paths])
                self.closed = True
                return caches

            def commit(self, *, accepted_lengths):
                return self.commit_paths(
                    [list(range(length)) for length in accepted_lengths]
                )

            def abort(self):
                tx.abort()
                self.closed = True

        return logits, features, Transaction()


def _cohort():
    draft, batch, first = _batch(maximum=32, num_draft=15, capacity=4)
    batch.minimum_draft_proposals = 3
    batch.ast_target_tree_executor_provider = _CohortProvider(
        (), "cpu-source", ast_fast_tree=True
    )
    ids = batch.insert(
        [PROMPTS[0]] * 3,
        max_tokens=[32] * 3,
        sampling_configs=[{"sampling_temp": 0}] * 3,
    )
    lanes = [first, *(batch.lanes[uid] for uid in ids)]
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
        lane.ready.clear()
    batch._ast_tree_ready_sessions = 4
    return draft, batch, lanes


def test_admitted_cohort_drafts_one_b4_depth7_group_and_dispatches_true_b4(monkeypatch):
    draft, batch, lanes = _cohort()
    calls = []
    original = draft.draft_distributions

    def observed(anchors, hidden, cache, depth, *args, **kwargs):
        calls.append((len(anchors), depth))
        return original(anchors, hidden, cache, depth, *args, **kwargs)

    monkeypatch.setattr(draft, "draft_distributions", observed)
    try:
        before = [lane.generated for lane in lanes]
        batch._run_round(lanes)
        assert calls == [(4, 7)]
        assert batch.scheduler_stats["external_ast_target_tree_cohort_launches"] == 1
        assert batch.scheduler_stats["external_ast_target_tree_cohort_sessions"] == 4
        for lane, generated in zip(lanes, before):
            assert lane.generated > generated
            assert lane.ast_tree_source_plan.max_tile_rows == 4
            assert lane.ast_tree_source_plan.target_sessions == 4
            assert lane.ast_tree_source_plan.branches == 1
            assert lane.ast_tree_tiles_receipt["physical_packed_observed"] is True
            assert not any(getattr(cache, "speculating", False) for cache in lane.cache)
        assert not batch._open
    finally:
        batch.close()


def test_decode_partition_keeps_final_anchor_and_ordinary_out_of_packed_turn(
    monkeypatch,
):
    _draft, batch, lanes = _cohort()
    calls = []
    monkeypatch.setattr(
        batch,
        "_round_at_batch_route",
        lambda group, **kw: calls.append([l.uid for l in group]),
    )
    try:
        batch._decode_phase(lanes)
        assert calls == [[lane.uid for lane in lanes]]
        calls.clear()
        lanes[1].maximum = lanes[1].generated + 1
        lanes[3].ordinary = True
        batch._decode_phase(lanes)
        assert sorted(uid for group in calls for uid in group) == sorted(
            l.uid for l in lanes
        )
        assert all(
            len(group) == 1
            for group in calls
            if lanes[1].uid in group or lanes[3].uid in group
        )
        batch.minimum_draft_proposals = 4
        assert batch._ast_tree_cohort_limit() == 3
    finally:
        batch.close()


def test_actual_cohort_dispatch_refuses_shape_before_target_and_abort_restores(
    monkeypatch,
):
    _draft, batch, lanes = _cohort()
    rows = [copy.deepcopy(lane.cache) for lane in lanes]
    before = [_committed_digest(row) for row in rows]
    inputs = [[lane.anchor, 1, 2, 3] for lane in lanes]
    parents = [list(range(-1, 3)) for _ in lanes]
    try:
        with pytest.raises(ValueError, match="physical geometry"):
            batch._ast_target_tree_forward_many(
                lanes,
                rows,
                [r + [4] for r in inputs],
                parents=[p + [3] for p in parents],
            )
        logits, features, tx = batch._ast_target_tree_forward_many(
            lanes, rows, inputs, parents=parents
        )
        mx.eval(logits, features)
        tx.abort()
        assert [_committed_digest(row) for row in rows] == before
        assert tx.closed
        assert batch.scheduler_stats["external_ast_target_tree_cohort_launches"] == 1
        provider = batch.ast_target_tree_executor_provider
        monkeypatch.setattr(
            type(provider),
            "dispatch_metrics",
            lambda self, width, transaction, **kw: SimpleNamespace(
                width=width,
                physical_packed=False,
                executor_cached=True,
                cached=True,
                provider_protocol=self.protocol,
            ),
        )
        with pytest.raises(TypeError, match="invalid dispatch metrics"):
            batch._ast_target_tree_forward_many(lanes, rows, inputs, parents=parents)
        assert [_committed_digest(row) for row in rows] == before
    finally:
        batch.close()


def test_two_tile_source_reservation_runs_second_cohort_only_for_surviving_lanes(
    monkeypatch,
):
    _draft, batch, lanes = _cohort()
    tokens = []
    original = batch._propose
    forward = type(batch.ast_target_tree_executor_provider).forward_many
    shapes = []

    def source(group, **kwargs):
        blocks = original(group, **kwargs)
        tokens.extend(list(block.tokens) for block in blocks)
        return blocks

    def target(provider, model, inputs, *args, **kwargs):
        # Deterministic target laws isolate the scheduling case without
        # forging source provenance. CPU target state still consumes the
        # actual source tokens through the real transaction below.
        second = bool(shapes)
        shapes.append((len(inputs), len(inputs[0])))
        logits, features, transaction = forward(
            provider, model, inputs, *args, **kwargs
        )
        owners = (0, 2, 3) if second else (0, 1, 2, 3)
        selected = []
        for owner in owners:
            positions = range(4, 8) if second else range(4)
            row = [
                tokens[owner][position] if position < 7 else 0 for position in positions
            ]
            if owner == 1:
                row[1] = (row[1] + 1) % logits.shape[-1]
            selected.append(row)
        laws = mx.full(logits.shape, -100.0)
        for owner, row in enumerate(selected):
            for position, token in enumerate(row):
                laws[owner, position, token] = 0
        return laws, features, transaction

    monkeypatch.setattr(batch, "_propose", source)
    monkeypatch.setattr(
        type(batch.ast_target_tree_executor_provider), "forward_many", target
    )
    try:
        before = [lane.generated for lane in lanes]
        batch._run_round(lanes)
        assert shapes == [(4, 4), (3, 4)]
        assert [lane.generated - count for lane, count in zip(lanes, before)] == [
            8,
            2,
            8,
            8,
        ]
        assert [lane.ast_tree_tiles_receipt["launches"] for lane in lanes] == [
            2,
            1,
            2,
            2,
        ]
        assert batch.scheduler_stats["external_ast_target_tree_cohort_launches"] == 2
        assert not batch._open
    finally:
        batch.close()


@pytest.mark.parametrize("failure", [False, True])
def test_cohort_round_stop_and_post_publication_recovery_preserve_private_lanes(
    monkeypatch, failure
):
    _draft, batch, lanes = _cohort()
    before = [
        (
            _committed_digest(lane.cache),
            lane.generated,
            list(lane.history),
            lane.rng.snapshot(),
        )
        for lane in lanes
    ]
    commit = batch._commit
    if failure:

        def broken(*args, **kwargs):
            commit(*args, **kwargs)
            raise RuntimeError("injected after cohort publication")

        monkeypatch.setattr(batch, "_commit", broken)
    else:
        batch.stops = frozenset(range(batch._target_args.vocab_size))
    try:
        if failure:
            with pytest.raises(RuntimeError, match="after cohort publication"):
                batch._run_round(lanes)
            assert [
                (
                    _committed_digest(lane.cache),
                    lane.generated,
                    list(lane.history),
                    lane.rng.snapshot(),
                )
                for lane in lanes
            ] == before
            assert all(not lane.ready for lane in lanes)
            monkeypatch.setattr(batch, "_commit", commit)
            batch._run_round(lanes)
            assert all(lane.generated > prior[1] for lane, prior in zip(lanes, before))
        else:
            batch._run_round(lanes)
            assert all(
                lane.generated == prior[1] + 1 for lane, prior in zip(lanes, before)
            )
            assert all(lane.ready[-1].finish_reason == "stop" for lane in lanes)
        assert not batch._open
    finally:
        batch.close()
