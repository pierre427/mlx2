"""Real generic tree-law/walk dispatch with a deterministic private host target.

These tests prove scheduler/state composition, not any model's tensor oracle.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.runtime.ast_tree_verify import verify_tree_tiles, verify_tree_tiles_many
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, TreeDraftRow
from mlx2.runtime.speculative_sampling import RequestRNG
from mlx2.runtime.speculative_topology import (
    ordered_alternative_winner,
    ordered_branch_commit_counts,
)
from mlx2.runtime.target_tree_executor import TargetTreeDispatchMetrics


class _HostArrays:
    float32 = np.float32
    int32 = np.int32
    array = staticmethod(np.array)
    take = staticmethod(np.take)
    concatenate = staticmethod(np.concatenate)
    zeros_like = staticmethod(np.zeros_like)
    broadcast_to = staticmethod(np.broadcast_to)
    take_along_axis = staticmethod(np.take_along_axis)
    any = staticmethod(np.any)
    isfinite = staticmethod(np.isfinite)
    isnan = staticmethod(np.isnan)
    argmax = staticmethod(np.argmax)
    logsumexp = staticmethod(np.logaddexp.reduce)
    eval = staticmethod(lambda *args: None)
    async_eval = staticmethod(lambda *args: None)


def _state(tokens):
    value = 0
    for token in tokens:
        value = value * 3 + token
    return {"tokens": list(tokens), "recurrent": value}


class _Generator:
    mx = _HostArrays
    ast_memo = None
    ast_target_tree_executor_provider = SimpleNamespace(protocol="host_private_tree_v1")
    _copy_reference_cache = staticmethod(copy.deepcopy)
    _copy_ast_private_cache = staticmethod(copy.deepcopy)
    _batched_law_contract = staticmethod(
        ExternalDraftBatchGenerator._batched_law_contract
    )
    _tree_paths = staticmethod(ExternalDraftBatchGenerator._tree_paths)
    _target_tree_parents = staticmethod(
        ExternalDraftBatchGenerator._target_tree_parents
    )
    _walk_tree = ExternalDraftBatchGenerator._walk_tree
    _launch_tree_laws = ExternalDraftBatchGenerator._launch_tree_laws
    _verify_tree_batched = ExternalDraftBatchGenerator._verify_tree_batched

    def __init__(self, target, *, tile_rows=16, stops=(), fail=None):
        self.target = (1001, 1002, 0, *target)
        self.ast_tree_tile_rows = tile_rows
        self.stops = frozenset(stops)
        self.tree_gates = {"logprobs_on_request": True}
        self.scheduler_stats = {}
        self.calls, self.transactions = [], []
        self.fail = fail

    def _ast_target_tree_forward(self, lane, cache, inputs, *, parents):
        assert tuple(lane.history) == tuple(cache["tokens"])
        assert inputs[0] == lane.anchor
        paths = self._tree_paths(parents)
        assert all(path[0] == 0 for path in paths)
        self.calls.append((tuple(inputs), tuple(parents), tuple(lane.history)))
        ordinal = len(self.calls)
        if self.fail == ("forward", ordinal):
            raise RuntimeError("injected target forward failure")
        before = copy.deepcopy(cache)
        logits = np.full((1, len(inputs), 1024), -100.0, dtype=np.float32)
        features = []
        for row, path in enumerate(paths):
            prefix = (*cache["tokens"], *(inputs[index] for index in path))
            selected = (
                self.target[len(prefix)]
                if prefix == self.target[: len(prefix)]
                and len(prefix) < len(self.target)
                else 999
            )
            logits[0, row, selected] = 0.0
            features.append((len(prefix), _state(prefix)["recurrent"]))
        owner = self

        class Transaction:
            closed = False
            aborted = False

            def commit_paths(self, chosen):
                assert len(chosen) == 1
                rows = chosen[0]
                assert rows and rows == paths[rows[-1]]
                after = _state((*before["tokens"], *(inputs[index] for index in rows)))
                cache.clear()
                cache.update(after)
                if owner.fail == ("commit", ordinal):
                    raise RuntimeError("injected partial commit failure")
                self.closed = True
                return [cache]

            def abort(self):
                cache.clear()
                cache.update(copy.deepcopy(before))
                self.closed = self.aborted = True

        transaction = Transaction()
        self.transactions.append(transaction)
        return logits, np.asarray([features], dtype=np.float64), transaction


def _lane():
    return SimpleNamespace(
        uid=9,
        anchor=0,
        cache=_state((1001, 1002)),
        history=[1001, 1002],
        rng=RequestRNG(7),
        sampling={"sampling_temp": 0, "emit_logprobs": False},
        maximum=64,
        generated=0,
        processors=(),
        session_scope_hash="host-session",
    )


def _blocks(paths, probabilities=None):
    selection = SimpleNamespace(
        paths=tuple(
            SimpleNamespace(
                tokens=tuple(path),
                conditional_acceptance=(
                    (0.5,) * len(path)
                    if probabilities is None
                    else probabilities[index]
                ),
                score_authority="host_fixture",
            )
            for index, path in enumerate(paths)
        )
    )
    return [
        SimpleNamespace(tokens=list(path), continuation_selection=selection)
        for path in paths
    ]


def _winner(result, paths):
    return ordered_alternative_winner(
        result[0], tuple(map(len, paths)), allow_partial_uplift=True
    )[0]


def test_dispatch_confident_fifteen_first_tile_and_no_later_alternatives():
    primary = tuple(range(1, 16))
    other = tuple(range(31, 46))
    paths = (primary, other)
    lane, generator = _lane(), _Generator(primary)
    original = copy.deepcopy(lane.cache), lane.rng.snapshot()
    result = verify_tree_tiles(
        generator, lane, _blocks(paths, ((0.99,) * 15, (0.01,) * 15))
    )
    assert len(generator.calls) == 1 and generator.calls[0][0] == (0, *primary)
    winner = _winner(result, paths)
    assert winner == 0
    assert result[0][winner].emitted == [*primary, 999]
    assert result[2][winner] == _state((1001, 1002, 0, *primary))
    assert result[3][winner].draws == 16
    assert result[2][1] is None and result[3][1] is None
    assert lane.cache == original[0] and lane.rng.snapshot() == original[1]
    assert lane.ast_tree_tiles_receipt["last"]["pending_nodes_skipped"] == 15


def test_branched_multiple_tiles_match_single_path_state_features_and_rng():
    paths = tuple(tuple(range(lane * 10 + 1, lane * 10 + 8)) for lane in range(4))
    target = paths[-1]
    lane, generator = _lane(), _Generator(target)
    result = verify_tree_tiles(generator, lane, _blocks(paths))
    winner = _winner(result, paths)
    assert winner == 3 and len(generator.calls) == 2
    assert generator.calls[0][0] != generator.calls[1][0]
    assert generator.calls[1][0] == target[3:]
    assert result[0][winner].emitted == [*target, 999]
    assert ordered_branch_commit_counts(result[0]) == (1, 1, 1, 8)
    reference = verify_tree_tiles(_Generator(target), _lane(), _blocks((target,)))
    assert result[2][winner] == reference[2][0]
    assert result[3][winner].snapshot() == reference[3][0].snapshot()
    np.testing.assert_array_equal(result[1][winner], reference[1][0])
    assert all(result[2][index] is None for index in range(3))
    assert result[5]["physical_launches"] == 2
    assert result[5]["physical_target_rows"] == 20


def test_shared_ten_prefix_continuation_commits_once_and_preserves_source_rank():
    prefix = tuple(range(1, 11))
    paths = ((*prefix, 11, 12, 13, 14, 15), (*prefix, 21, 22, 23, 24, 25))
    target = paths[1]
    generator = _Generator(target, tile_rows=11)
    result = verify_tree_tiles(generator, _lane(), _blocks(paths))
    winner = _winner(result, paths)
    assert winner == 1
    assert [call[0] for call in generator.calls] == [(0, *prefix), (21, 22, 23, 24, 25)]
    assert result[0][winner].emitted == [*target, 999]
    assert result[2][winner] == _state((1001, 1002, 0, *target))
    # The primary really verified its shared ten plus one correction; it did
    # not contribute zero work when the second branch won.
    assert ordered_branch_commit_counts(result[0]) == (11, 16)
    assert result[0][0].accepted == 10
    assert result[0][0].emitted == [*prefix, 21]


def test_full_short_branch_tie_uses_existing_arbiter_and_only_matching_state():
    paths = ((1, 2, 3, 4), (1, 2, 3))
    result = verify_tree_tiles(_Generator((1, 2, 3)), _lane(), _blocks(paths))
    winner = _winner(result, paths)
    assert winner == 1  # Existing full-branch tie law, not ad-hoc source rank.
    assert ordered_branch_commit_counts(result[0]) == (4, 4)
    assert result[2][0] is None
    assert result[2][1] == _state((1001, 1002, 0, 1, 2, 3))
    assert result[0][1].emitted == [1, 2, 3, 999]


@pytest.mark.parametrize("stop", [False, True])
def test_correction_or_stop_cancels_later_tiles_and_keeps_terminal_state(stop):
    path = tuple(range(1, 16))
    target = path if stop else ()
    generator = _Generator(target, tile_rows=5, stops=(3,) if stop else ())
    lane = _lane()
    result = verify_tree_tiles(generator, lane, _blocks((path,)))
    assert len(generator.calls) == 1
    expected = [1, 2, 3] if stop else [999]
    assert result[0][0].emitted == expected
    assert result[2][0] == _state((1001, 1002, 0, *expected[:-1]))
    assert result[3][0].draws == len(expected)
    assert lane.ast_tree_tiles_receipt["last"]["termination_reason"] == (
        "stop" if stop else "correction"
    )


@pytest.mark.parametrize("phase", ["forward", "commit"])
def test_later_failure_never_publishes_prior_private_tiles_or_rng(phase):
    path = tuple(range(1, 16))
    generator = _Generator(path, tile_rows=5, fail=(phase, 2))
    lane = _lane()
    before = copy.deepcopy(vars(lane))
    with pytest.raises(RuntimeError, match="injected"):
        verify_tree_tiles(generator, lane, _blocks((path,)))
    assert len(generator.calls) == 2
    assert lane.cache == before["cache"]
    assert lane.history == before["history"]
    assert lane.rng.snapshot() == before["rng"].snapshot()
    assert not hasattr(lane, "ast_tree_tiles_receipt")
    assert generator.transactions[0].closed
    if phase == "commit":
        assert generator.transactions[1].aborted


def _bound_dispatch_fixture(*, wrong_selection=False):
    batch = object.__new__(ExternalDraftBatchGenerator)
    batch.model, batch.layers = object(), (1,)
    batch.tree_gates = {"cache_executor": False}
    batch.scheduler_stats = {}
    lane, cache = SimpleNamespace(uid=1), object()
    events = []

    class Transaction:
        closed = False
        provider_protocol = "validated-host-provider"
        provider_revision_token = "immutable-host-revision"

        def commit(self, *, accepted_lengths):
            self.closed = True

        def abort(self):
            events.append(("abort",))
            self.closed = True

    transaction = Transaction()

    class Provider:
        protocol = transaction.provider_protocol
        source_revision_token = transaction.provider_revision_token

        def validate_cohort(self, width, limit):
            assert (width, limit) == (1, 1)

        def forward(self, model, inputs, parents, rows, layers, *, cached):
            assert model is batch.model and rows is cache and layers == batch.layers
            assert inputs == [0, 1] and parents == [-1, 0]
            events.append(("forward", cached))
            return "logits", "features", transaction

        def dispatch_metrics(self, width, txn, *, cached):
            assert width == 1 and txn is transaction
            events.append(("metrics", cached))
            return TargetTreeDispatchMetrics(
                width=1,
                physical_packed=False,
                executor_cached=True,
                cached=False if wrong_selection else cached,
                provider_protocol=self.protocol,
            )

    provider = Provider()
    batch.ast_target_tree_executor_provider = provider
    return batch, lane, cache, transaction, events


def test_ast_dispatch_uses_validated_cache_not_legacy_tree_gate():
    batch, lane, cache, transaction, events = _bound_dispatch_fixture()
    result = batch._ast_target_tree_forward(lane, cache, [0, 1])
    assert result == ("logits", "features", transaction)
    assert events == [("forward", True), ("metrics", True)]
    assert not transaction.closed
    assert batch.scheduler_stats["external_ast_target_tree_provider_launches"] == 1
    transaction.abort()


@pytest.mark.parametrize("mismatch", ["selection", "uncached_hit"])
def test_ast_dispatch_does_not_weaken_common_cache_metric_invariants(mismatch):
    batch, lane, cache, transaction, events = _bound_dispatch_fixture(
        wrong_selection=mismatch == "selection"
    )
    with pytest.raises(RuntimeError, match="cache selection|cache hit"):
        if mismatch == "selection":
            batch._ast_target_tree_forward(lane, cache, [0, 1])
        else:
            batch._finish_target_tree_dispatch(
                [lane],
                ("logits", "features", transaction),
                cached=False,
                provider=batch.ast_target_tree_executor_provider,
                publish_tree_metrics=False,
            )
    assert transaction.closed and events[-1] == ("abort",)
    assert batch.scheduler_stats == {}


@pytest.mark.parametrize("emit", [False, True])
def test_ast_tree_logprob_policy_does_not_inherit_legacy_gates(emit, monkeypatch):
    generator, lane = _Generator((1, 2)), _lane()
    generator.tree_gates["logprobs_on_request"] = False
    lane.sampling["emit_logprobs"] = emit
    if not emit:
        original_zeros = np.zeros

        def forbidden_dense_law(shape, *args, **kwargs):
            if shape == 1024:
                raise AssertionError(
                    "no-logprob AST request allocated a dense host law"
                )
            return original_zeros(shape, *args, **kwargs)

        monkeypatch.setattr(np, "zeros", forbidden_dense_law)
    result = verify_tree_tiles(generator, lane, _blocks(((1, 2),)))
    decision = result[0][0]
    assert decision.emitted == [1, 2, 999]
    assert result[3][0].draws == 3
    if emit:
        assert len(decision.target_laws) == len(decision.response_logprobs) == 3
        assert [int(np.argmax(law)) for law in decision.target_laws] == [1, 2, 999]
    else:
        assert decision.target_laws is decision.response_logprobs is None
        assert generator.scheduler_stats["external_tree_logprob_rows_skipped"] == 3


def test_batched_tree_legacy_logprob_gate_remains_unchanged():
    generator, lane = _Generator((1, 2)), _lane()
    generator.tree_gates["logprobs_on_request"] = False
    block = TreeDraftRow([1, 2], [-1, 0])
    logits = np.full((1, 3, 1024), -100.0, dtype=np.float32)
    logits[0, np.arange(3), [1, 2, 999]] = 0
    _fence, state = generator._launch_tree_laws(lane, block, logits)
    decision = generator._verify_tree_batched(lane, block, state)
    assert len(decision.target_laws) == len(decision.response_logprobs) == 3
    assert "external_tree_logprob_rows_skipped" not in generator.scheduler_stats


def test_measured_tile_budget_stops_at_exact_target_selected_anchor():
    from mlx2.runtime.ast_tree_tuning import ASTTreeSourcePlan

    paths = tuple(tuple(range(index * 20 + 1, index * 20 + 8)) for index in range(4))
    lane, generator = _lane(), _Generator(paths[-1])
    lane.ast_tree_source_plan = ASTTreeSourcePlan(
        7, 4, 16, 1, 2, 1, "host-budget", 0, False
    )
    result = verify_tree_tiles(generator, lane, _blocks(paths))
    winner = _winner(result, paths)
    chosen = result[0][winner]
    assert len(generator.calls) == 1
    assert chosen.emitted == list(paths[-1][:4])
    assert result[2][winner] == _state((1001, 1002, 0, *paths[-1][:3]))
    assert lane.ast_tree_tiles_receipt["last"]["termination_reason"] == "tile_budget"
    assert lane.ast_tree_tiles_receipt["pending_nodes_skipped"] == 13
    assert lane.cache == _state((1001, 1002))


class _CohortGenerator(_Generator):
    """One aggregate transaction over genuinely independent host state owners."""

    def __init__(self, lanes, targets, *, fail=None, maximum=4):
        super().__init__((), fail=fail)
        self.targets = {
            lane.uid: (*lane.history, lane.anchor, *target)
            for lane, target in zip(lanes, targets, strict=True)
        }
        self.ast_target_tree_executor_provider = SimpleNamespace(
            protocol="host_private_cohort_v1",
            ast_tree_max_sessions=maximum,
            ast_tree_cohort_input_widths=lambda width: tuple(range(1, 16 // width + 1)),
        )
        self.cohort_calls = []

    def _ast_target_tree_forward(self, lane, cache, inputs, *, parents):
        return self._forward([lane], [cache], [inputs], [parents])

    def _ast_target_tree_forward_many(self, lanes, caches, inputs, *, parents):
        assert 2 <= len(lanes) <= 4
        return self._forward(lanes, caches, inputs, parents)

    def _forward(self, lanes, caches, inputs, parents):
        width, count = len(lanes), len(inputs[0])
        assert all(len(row) == count for row in inputs)
        assert width * count <= 16
        assert len({id(cache) for cache in caches}) == width
        self.cohort_calls.append(
            (
                tuple(lane.uid for lane in lanes),
                copy.deepcopy(inputs),
                copy.deepcopy(parents),
            )
        )
        ordinal = len(self.cohort_calls)
        if self.fail == ("forward", ordinal):
            raise RuntimeError("injected cohort forward failure")
        logits = np.full((width, count, 1024), -100.0, dtype=np.float32)
        features, paths = [], []
        before = copy.deepcopy(caches)
        for owner, (lane, cache, tokens, parent_rows) in enumerate(
            zip(lanes, caches, inputs, parents, strict=True)
        ):
            assert lane.history == cache["tokens"] and tokens[0] == lane.anchor
            owner_paths = self._tree_paths(parent_rows)
            paths.append(owner_paths)
            owner_features = []
            for row, path in enumerate(owner_paths):
                prefix = (*cache["tokens"], *(tokens[index] for index in path))
                target = self.targets[lane.uid]
                selected = (
                    target[len(prefix)]
                    if prefix == target[: len(prefix)] and len(prefix) < len(target)
                    else 999
                )
                logits[owner, row, selected] = 0.0
                owner_features.append((len(prefix), _state(prefix)["recurrent"]))
            features.append(owner_features)
        generator = self

        class Transaction:
            closed = False
            aborted = False

            def commit_paths(self, chosen):
                assert len(chosen) == width
                for index, (cache, tokens, rows) in enumerate(
                    zip(caches, inputs, chosen, strict=True)
                ):
                    assert rows == paths[index][rows[-1]]
                    cache.clear()
                    cache.update(
                        _state(
                            (*before[index]["tokens"], *(tokens[row] for row in rows))
                        )
                    )
                    if generator.fail == ("commit", ordinal, index):
                        raise RuntimeError("injected cohort partial commit failure")
                self.closed = True
                return caches

            def abort(self):
                for cache, original in zip(caches, before, strict=True):
                    cache.clear()
                    cache.update(copy.deepcopy(original))
                self.closed = self.aborted = True

        transaction = Transaction()
        self.transactions.append(transaction)
        return logits, np.asarray(features, dtype=np.float64), transaction

    def _verify_tree_batched(self, lane, *args, **kwargs):
        if self.fail == ("verify", lane.uid):
            raise RuntimeError("injected cohort law failure")
        return super()._verify_tree_batched(lane, *args, **kwargs)


def _cohort_lanes():
    lanes = [_lane() for _ in range(4)]
    for index, lane in enumerate(lanes):
        lane.uid += index
        lane.history = [1001 + index] * (index + 1)
        lane.cache = _state(lane.history)
        lane.session_scope_hash = f"private-cohort-{index}"
    return lanes


def test_true_b4_one_launch_independent_offsets_and_no_double_quota():
    from mlx2.runtime.ast_tree_tuning import ASTTreeSourcePlan

    lanes = _cohort_lanes()
    targets = [tuple(range(10 * index + 1, 10 * index + 4)) for index in range(4)]
    before = copy.deepcopy(lanes)
    for lane in lanes:
        lane.ast_tree_source_plan = ASTTreeSourcePlan(
            3, 1, 4, 1, 1, 4, "host-pressure", 0, False
        )
    generator = _CohortGenerator(lanes, targets)
    results = verify_tree_tiles_many(
        generator, lanes, [_blocks((target,)) for target in targets]
    )
    assert [
        (len(uids), len(inputs[0])) for uids, inputs, _ in generator.cohort_calls
    ] == [(4, 4)]
    assert sum(result[5]["physical_launches"] for result in results) == 1
    assert sum(result[5]["physical_target_rows"] for result in results) == 16
    assert sum(result[5]["aligned_tile_launches"][4] for result in results) == 1
    for index, (lane, target, result) in enumerate(
        zip(lanes, targets, results, strict=True)
    ):
        reference = verify_tree_tiles(
            _CohortGenerator([before[index]], [target]),
            before[index],
            _blocks((target,)),
        )
        assert result[0][0].emitted == [*target, 999]
        assert result[2][0] == reference[2][0]
        assert result[3][0].snapshot() == reference[3][0].snapshot()
        np.testing.assert_array_equal(result[1], reference[1])
        assert lane.cache == before[index].cache
        receipt = lane.ast_tree_tiles_receipt
        assert receipt["max_physical_cohort_width"] == 4
        assert receipt["packed_launch_participations"] == 1
        assert receipt["physical_packed_observed"] is True


def test_cohort_shared_prefix_winners_continue_without_duplicate_inputs():
    lanes = _cohort_lanes()
    paths = ((1, 2, 3), (1, 12, 13))
    targets = [paths[index % 2] for index in range(4)]
    generator = _CohortGenerator(lanes, targets)
    results = verify_tree_tiles_many(generator, lanes, [_blocks(paths) for _ in lanes])
    assert [
        (len(uids), len(inputs[0])) for uids, inputs, _ in generator.cohort_calls
    ] == [(4, 4), (4, 1)]
    assert sum(result[5]["physical_launches"] for result in results) == 2
    for index, (lane, target, result) in enumerate(
        zip(lanes, targets, results, strict=True)
    ):
        winner = _winner(result, paths)
        assert winner == index % 2
        assert result[0][winner].emitted == [*target, 999]
        assert result[2][winner] == _state((*lane.history, 0, *target))
        assert result[2][1 - winner] is None
        reference = verify_tree_tiles(
            _CohortGenerator([lane], [target]), lane, _blocks((target,))
        )
        np.testing.assert_array_equal(result[1][winner], reference[1][0])
        assert result[3][winner].snapshot() == reference[3][0].snapshot()


def test_cohort_ragged_tiles_group_exact_lengths_without_padding():
    lanes = _cohort_lanes()
    targets = [(1,), (11, 12), (21, 22, 23), (31, 32, 33)]
    generator = _CohortGenerator(lanes, targets)
    results = verify_tree_tiles_many(
        generator, lanes, [_blocks((target,)) for target in targets]
    )
    assert [
        (len(uids), len(inputs[0])) for uids, inputs, _ in generator.cohort_calls
    ] == [(1, 2), (1, 3), (2, 4)]
    assert sum(result[5]["physical_target_rows"] for result in results) == 13
    assert all(result[5]["padding_rows"] == 0 for result in results)
    assert [
        lane.ast_tree_tiles_receipt["physical_packed_observed"] for lane in lanes
    ] == [False, False, True, True]


@pytest.mark.parametrize(
    "failure", [("forward", 2), ("verify", 11), ("commit", 1, 2), ("commit", 2, 1)]
)
def test_cohort_failure_never_publishes_any_owner_or_partial_receipt(failure):
    lanes = _cohort_lanes()
    before = copy.deepcopy(lanes)
    target = (1, 2, 3, 4, 5, 6)
    generator = _CohortGenerator(lanes, [target] * 4, fail=failure)
    with pytest.raises(RuntimeError, match="injected cohort"):
        verify_tree_tiles_many(generator, lanes, [_blocks((target,)) for _ in lanes])
    for lane, original in zip(lanes, before, strict=True):
        assert lane.cache == original.cache and lane.history == original.history
        assert lane.rng.snapshot() == original.rng.snapshot()
        assert not hasattr(lane, "ast_tree_tiles_receipt")
    if failure[0] in {"verify", "commit"}:
        assert generator.transactions[-1].aborted


def test_cohort_missing_capability_selects_private_singletons_before_dispatch():
    lanes = _cohort_lanes()
    targets = [(1, 2, 3)] * 4
    generator = _CohortGenerator(lanes, targets, maximum=1)
    results = verify_tree_tiles_many(
        generator, lanes, [_blocks((target,)) for target in targets]
    )
    assert all(len(uids) == 1 for uids, _inputs, _parents in generator.cohort_calls)
    assert sum(result[5]["physical_launches"] for result in results) == 4
    assert all(
        not lane.ast_tree_tiles_receipt["physical_packed_observed"] for lane in lanes
    )


def test_cohort_zero_partial_stop_and_full_acceptance_keep_distinct_states():
    lanes = _cohort_lanes()
    paths = [(1, 2, 3), (1, 2, 3), (1, 2, 3), (11, 12, 13)]
    targets = [(), (1, 8), (1, 2, 3), (11, 12, 13)]
    generator = _CohortGenerator(lanes, targets)
    generator.stops = frozenset((2,))
    results = verify_tree_tiles_many(
        generator, lanes, [_blocks((path,)) for path in paths]
    )
    assert len(generator.cohort_calls) == 1
    assert [result[0][0].accepted for result in results] == [0, 1, 2, 3]
    assert [result[0][0].emitted for result in results] == [
        [999],
        [1, 8],
        [1, 2],
        [11, 12, 13, 999],
    ]
    assert [result[0][0].commit_count for result in results] == [1, 2, 2, 4]
    for lane, target, path, result in zip(lanes, targets, paths, results, strict=True):
        reference_generator = _CohortGenerator([lane], [target])
        reference_generator.stops = generator.stops
        reference = verify_tree_tiles(reference_generator, lane, _blocks((path,)))
        assert result[2][0] == reference[2][0]
        assert result[3][0].snapshot() == reference[3][0].snapshot()
        np.testing.assert_array_equal(result[1], reference[1])


def test_cohort_final_result_failure_leaves_every_public_receipt_unpublished(
    monkeypatch,
):
    from mlx2.runtime import ast_tree_verify

    lanes = _cohort_lanes()
    targets = [(1, 2, 3)] * 4
    original = ast_tree_verify._finish_request

    def fail_last(work):
        result = original(work)
        if work.lane is lanes[-1]:
            raise RuntimeError("injected final cohort result failure")
        return result

    monkeypatch.setattr(ast_tree_verify, "_finish_request", fail_last)
    before = copy.deepcopy(lanes)
    generator = _CohortGenerator(lanes, targets)
    with pytest.raises(RuntimeError, match="final cohort result"):
        verify_tree_tiles_many(
            generator, lanes, [_blocks((target,)) for target in targets]
        )
    assert generator.transactions[0].closed
    for lane, prior in zip(lanes, before, strict=True):
        assert lane.cache == prior.cache
        assert lane.rng.snapshot() == prior.rng.snapshot()
        assert not hasattr(lane, "ast_tree_tiles_receipt")


def test_async_lookahead_prepares_bounded_frontiers_and_reuses_selected_plan(
    monkeypatch,
):
    from mlx2.runtime import ast_tree_verify

    events, prepared, works = [], {}, []
    original_plan = ast_tree_verify.plan_tree_tile
    original_prepare = ast_tree_verify._prepare_request

    class Arrays(_HostArrays):
        @staticmethod
        def async_eval(*arrays):
            events.append(("async", tuple(map(id, arrays))))

        @staticmethod
        def eval(*arrays):
            events.append(("fence", tuple(map(id, arrays))))

    def plan(trie, *, root=0, max_rows=16):
        events.append(("plan", root))
        tile = original_plan(trie, root=root, max_rows=max_rows)
        prepared[root] = tile
        return tile

    def prepare(*args):
        work = original_prepare(*args)
        works.append(work)
        return work

    monkeypatch.setattr(ast_tree_verify, "plan_tree_tile", plan)
    monkeypatch.setattr(ast_tree_verify, "_prepare_request", prepare)
    paths = tuple(tuple(range(index * 10 + 1, index * 10 + 8)) for index in range(4))
    generator, lane = _Generator(paths[-1]), _lane()
    generator.mx = Arrays
    before = copy.deepcopy(lane)
    result = verify_tree_tiles(generator, lane, _blocks(paths))
    first_async = next(
        index for index, event in enumerate(events) if event[0] == "async"
    )
    first_fence = next(
        index for index, event in enumerate(events) if event[0] == "fence"
    )
    between = events[first_async + 1 : first_fence]
    assert len(between) == 4 and all(event[0] == "plan" for event in between)
    assert events[first_async][1] == events[first_fence][1]
    # The survivor was already prepared inside the asynchronous window; there
    # is no second planner call at its actual next-tile dispatch.
    assert len([event for event in events if event[0] == "plan"]) == 5
    assert len([event for event in events if event[0] == "async"]) == 1
    selected_root = next(
        root for root, tile in prepared.items() if tile.tokens == paths[-1][3:]
    )
    assert sum(event == ("plan", selected_root) for event in events) == 1
    assert works[0].preplans == {}
    assert lane.cache == before.cache and lane.rng.snapshot() == before.rng.snapshot()
    reference = verify_tree_tiles(_Generator(paths[-1]), _lane(), _blocks((paths[-1],)))
    winner = _winner(result, paths)
    assert result[2][winner] == reference[2][0]
    assert result[3][winner].snapshot() == reference[3][0].snapshot()
    np.testing.assert_array_equal(result[1][winner], reference[1][0])
    receipt = lane.ast_tree_tiles_receipt
    assert receipt["prepared_successor_plans"] == 4
    assert receipt["consumed_successor_plans"] == 1
    assert receipt["async_lookahead_participations"] == 1


@pytest.mark.parametrize("cancel", ["stop", "correction", "tile_budget", "limit"])
def test_async_lookahead_cancellation_discards_all_unselected_plans(
    cancel, monkeypatch
):
    from mlx2.runtime import ast_tree_verify
    from mlx2.runtime.ast_tree_tuning import ASTTreeSourcePlan

    works, submitted = [], []
    original_prepare = ast_tree_verify._prepare_request

    def prepare(*args):
        work = original_prepare(*args)
        works.append(work)
        return work

    class Arrays(_HostArrays):
        async_eval = staticmethod(lambda *arrays: submitted.append(True))

    monkeypatch.setattr(ast_tree_verify, "_prepare_request", prepare)
    path = tuple(range(1, 16))
    lane = _lane()
    generator = _Generator(
        () if cancel in {"correction", "limit"} else path,
        tile_rows=5,
        stops=(3,) if cancel == "stop" else (),
    )
    generator.mx = Arrays
    if cancel == "tile_budget":
        lane.ast_tree_source_plan = ASTTreeSourcePlan(
            15, 1, 5, 1, 1, 1, "host-limit", 0, False
        )
    elif cancel == "limit":
        lane.maximum = 1
    verify_tree_tiles(generator, lane, _blocks((path,)))
    assert len(generator.calls) == 1 and works[0].preplans == {}
    assert lane.ast_tree_tiles_receipt["consumed_successor_plans"] == 0
    assert bool(submitted) == (cancel in {"stop", "correction"})
    assert lane.ast_tree_tiles_receipt["last"]["termination_reason"] == cancel


@pytest.mark.parametrize("foreign", ["trie", "root", "quota"])
def test_successor_preplan_requires_exact_trie_root_and_quota(foreign, monkeypatch):
    from mlx2.runtime import ast_tree_verify

    path = tuple(range(1, 16))
    generator, lane = _Generator(path, tile_rows=5), _lane()
    work = ast_tree_verify._prepare_request(generator, lane, _blocks((path,)))
    tile, _ = ast_tree_verify._plan_request(work, 5)
    plans = ast_tree_verify._preplan_successors(work.trie, tile, max_rows=5)
    key, prepared = next(iter(plans.items()))
    work.root = key[1]
    bad_key = list(key)
    bad_key[{"trie": 0, "root": 1, "quota": 2}[foreign]] += 1
    work.preplans = {tuple(bad_key): prepared}
    calls = []
    original = ast_tree_verify.plan_tree_tile

    def plan(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(ast_tree_verify, "plan_tree_tile", plan)
    actual, _ = ast_tree_verify._plan_request(work, 5)
    assert len(calls) == 1 and actual == prepared
    assert work.consumed_plans == 0 and work.preplans == {}


@pytest.mark.parametrize("fence_failure", [False, True])
def test_async_preplan_failure_fences_before_abort_and_preserves_error(
    monkeypatch, fence_failure
):
    from mlx2.runtime import ast_tree_verify

    events, works = [], []
    failure = RuntimeError("injected host successor preparation failure")
    original_prepare = ast_tree_verify._prepare_request

    def prepare(*args):
        work = original_prepare(*args)
        works.append(work)
        return work

    def fail_preplan(*args, **kwargs):
        events.append("prepare")
        # Submission may not alter the request's current anchor/cache/RNG.
        assert works[0].private.anchor == 0
        assert works[0].private.cache == _state((1001, 1002))
        assert works[0].private.rng.snapshot() == _lane().rng.snapshot()
        raise failure

    class Arrays(_HostArrays):
        @staticmethod
        def async_eval(*arrays):
            events.append("async")

        @staticmethod
        def eval(*arrays):
            events.append("fence")
            if fence_failure:
                raise ValueError("injected fence failure")

    class Generator(_Generator):
        def _ast_target_tree_forward(self, *args, **kwargs):
            logits, features, transaction = super()._ast_target_tree_forward(
                *args, **kwargs
            )
            abort = transaction.abort

            def abort_recorded():
                events.append("abort")
                abort()

            transaction.abort = abort_recorded
            return logits, features, transaction

    monkeypatch.setattr(ast_tree_verify, "_prepare_request", prepare)
    monkeypatch.setattr(ast_tree_verify, "_preplan_successors", fail_preplan)
    lane, generator = _lane(), Generator(tuple(range(1, 16)), tile_rows=5)
    generator.mx = Arrays
    before = copy.deepcopy(lane)
    with pytest.raises(RuntimeError, match="successor preparation") as caught:
        verify_tree_tiles(generator, lane, _blocks((tuple(range(1, 16)),)))
    assert caught.value is failure
    assert events == ["async", "prepare", "fence", "abort"]
    assert generator.transactions[0].aborted
    assert lane.cache == before.cache and lane.history == before.history
    assert lane.rng.snapshot() == before.rng.snapshot()
    assert not hasattr(lane, "ast_tree_tiles_receipt")
    if fence_failure:
        assert "fence also failed" in failure.__notes__[0]


def test_full_tiles_counts_continued_tiles_like_the_linear_path():
    """K4: a final 16-row tile is not a continued tile; quota tiles are."""
    primary = tuple(range(1, 16))
    single = verify_tree_tiles(_Generator(primary), _lane(), _blocks((primary,)))
    stat = single[4][0]
    assert stat["launches"] == 1
    assert stat["full_tiles"] == stat["rollback_base_tile"] == 0
    prefix = tuple(range(1, 11))
    paths = ((*prefix, 11, 12, 13, 14, 15), (*prefix, 21, 22, 23, 24, 25))
    generator = _Generator(paths[1], tile_rows=11)
    result = verify_tree_tiles(generator, _lane(), _blocks(paths))
    winner = _winner(result, paths)
    stat = result[4][winner]
    assert stat["launches"] == 2
    assert stat["full_tiles"] == stat["rollback_base_tile"] == 1
    assert result[1].shape[0] == 2 and single[1].shape[0] == 1


@pytest.mark.parametrize("values, valid", [
    ([0.0, -float("inf"), -1.0], True),
    ([-float("inf")] * 3, False),
    ([0.0, float("inf"), -1.0], False),
    ([0.0, float("nan"), -1.0], False),
])
def test_token_only_greedy_skips_normalization_preserving_invalid_law_rejection(values, valid):
    from mlx2.runtime.external_speculative import LaneFailure

    class TokenOnlyArrays(_HostArrays):
        @staticmethod
        def logsumexp(*args, **kwargs):
            raise AssertionError("token-only verification normalized vocabulary")

    generator, lane = _Generator(()), _lane()
    generator.mx = TokenOnlyArrays
    block = TreeDraftRow([], [])
    _, state = generator._launch_tree_laws(lane, block, np.asarray([[values]]))
    if valid:
        decision = generator._verify_tree_batched(lane, block, state)
        assert decision.emitted == [0]
        assert lane.rng.draws == 1
        assert decision.target_laws is decision.response_logprobs is None
    else:
        with pytest.raises(LaneFailure, match="not a probability"):
            generator._verify_tree_batched(lane, block, state)
        assert lane.rng.draws == 0


def test_winner_features_are_broadcast_without_allocating_loser_slabs():
    class Arrays(_HostArrays):
        @staticmethod
        def zeros_like(*args, **kwargs):
            raise AssertionError("allocated unused losing-branch features")

    paths = ((9, 8), (1, 2, 3), (4, 5))
    generator, lane = _Generator((1, 2, 3)), _lane()
    generator.mx = Arrays
    result = verify_tree_tiles(generator, lane, _blocks(paths))
    winner = _winner(result, paths)
    assert winner == 1
    assert result[1].shape[0] == len(paths)
    assert result[1].strides[0] == 0
    reference = verify_tree_tiles(_Generator((1, 2, 3)), _lane(), _blocks((paths[winner],)))
    np.testing.assert_array_equal(result[1][winner:winner + 1], reference[1])
    assert all(cache is None for i, cache in enumerate(result[2]) if i != winner)
    assert all(rng is None for i, rng in enumerate(result[3]) if i != winner)
