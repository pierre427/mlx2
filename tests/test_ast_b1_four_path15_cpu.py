"""Static source horizon and four-path tile contract; no GPU qualification."""

import copy

import mlx.core as mx
import pytest
from test_ast_fast_tree_policy_cpu import _batch, _CPUProvider, _policy

from mlx2.adapters.external_draft_policy import (
    normalize_progressive_external_policy,
    progressive_external_identity,
)
from mlx2.runtime.ast_tree_tiles import ASTProposalTrie, plan_tree_tile


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class _SerialTreeProvider(_CPUProvider):
    def __init__(self):
        super().__init__((), "cpu-source", ast_fast_tree=True)
        self.calls = []

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        assert len(inputs) == 16 and parents[0] == -1
        self.calls.append((tuple(inputs), tuple(parents)))
        states, logits, features = [], [], []
        for token, parent in zip(inputs, parents, strict=True):
            state = copy.deepcopy(cache if parent < 0 else states[parent])
            out, taps = model.forward_with_taps(mx.array([[token]]), state, layers)
            mx.eval(out, taps)
            states.append(state)
            logits.append(out)
            features.append(taps)
        provider = self

        class Transaction:
            closed = False
            provider_protocol = provider.protocol
            provider_revision_token = provider.source_revision_token

            def commit_paths(self, paths):
                assert len(paths) == 1 and paths[0][0] == 0
                self.closed = True
                return [states[paths[0][-1]]]

            def abort(self):
                self.closed = True

        return (
            mx.concatenate(logits, axis=1),
            mx.concatenate(features, axis=1),
            Transaction(),
        )


@pytest.mark.parametrize("remaining", [2, 8, 32])
def test_one_fixed15_backbone_four_real_roots_and_private_rng(monkeypatch, remaining):
    draft, batch, lane = _batch(
        maximum=remaining, num_draft=15, b1_tile_policy="four_path15"
    )
    calls = []
    original = draft._hidden

    def hidden(inputs, *args, **kwargs):
        calls.append(tuple(inputs.shape))
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(draft, "_hidden", hidden)
    before = lane.rng.snapshot()
    try:
        batch._propose([lane])
        assert calls == [(1, 16)]
        selection = batch.draft.last_continuation_selections[0]
        assert len(selection.paths) == 4
        assert all(len(path.tokens) == 15 for path in selection.paths)
        assert len({path.tokens[0] for path in selection.paths}) == 4
        assert lane.ast_tree_source_plan.depth == 15
        assert lane.ast_tree_source_plan.branches == 4
        assert lane.ast_tree_source_plan.max_tiles == 1
        assert lane.rng.snapshot() == before
        assert not batch._ast_tree_tuners
        diagnostics = batch.draft.last_continuation_source_diagnostics[0]["external"]
        assert diagnostics["private_head_calls"] == 0
        assert diagnostics["requested_depth"] == 15
    finally:
        batch.close()


def test_balanced_pack_drops_weakest_removable_tail_not_a_whole_branch():
    paths = [tuple(range(100 * row + 1, 100 * row + 16)) for row in range(4)]
    trie = ASTProposalTrie.from_paths(
        0, paths, conditional_acceptance=[(p,) * 15 for p in (0.9, 0.2, 0.8, 0.7)]
    )
    tile = plan_tree_tile(trie, balanced=True)
    assert tile.path_depths == (4, 3, 4, 4)
    assert len(tile.tokens) == 16
    assert tile.reason == "balanced_four_path_prefix"
    assert all(parent < index for index, parent in enumerate(tile.parents))


def test_complete_rounds_match_ordinary_and_every_ast_tile_is_full(monkeypatch):
    draft, batch, lane = _batch(maximum=40, num_draft=15, b1_tile_policy="four_path15")
    provider = _SerialTreeProvider()
    batch.ast_target_tree_executor_provider = provider
    original_history = [*lane.history, lane.anchor]
    state = copy.deepcopy(lane.cache)
    expected = []
    anchor = lane.anchor
    for _ in range(lane.maximum - lane.generated):
        logits = batch.model(mx.array([[anchor]]), cache=state)
        anchor = int(mx.argmax(logits[0, -1]).item())
        expected.append(anchor)
    seen = []
    original = draft.draft_distributions

    def produce(anchors, *args, **kwargs):
        seen.append((anchors[0], list(lane.history)))
        return original(anchors, *args, **kwargs)

    monkeypatch.setattr(draft, "draft_distributions", produce)
    try:
        for _ in range(50):
            if lane.generated >= lane.maximum:
                break
            batch._round([lane])
        else:
            raise AssertionError("B1 static tile scheduler stalled")
        assert [row.token for row in lane.ready] == expected
        assert provider.calls
        assert all(len(tokens) == 16 for tokens, _ in provider.calls)
        assert len(provider.calls) == len(seen)
        assert not batch.scheduler_stats.get("external_ast_anchor_target_launches", 0)
        assert not batch.draft.proposal_pool._pending
        assert len(seen) > 2
        assert all(
            history[: len(original_history) - 1] == original_history[:-1]
            for _, history in seen
        )
    finally:
        batch.close()


def test_anchor_only_provider_dispatch_is_refused():
    _draft, batch, lane = _batch(num_draft=15, b1_tile_policy="four_path15")
    try:
        with pytest.raises(ValueError, match="anchor-only"):
            batch._ast_target_tree_forward(
                lane, lane.cache, [lane.anchor], parents=[-1]
            )
    finally:
        batch.close()


@pytest.mark.parametrize(
    "change",
    [
        {"num_draft": 7},
        {"cost_per_commit_handoff": True},
        {"progressive_alternative_branches": 2},
        {"ast_target_tree_projection_law": None},
    ],
)
def test_static_policy_refuses_incompatible_geometry(change):
    policy = {
        **_policy(),
        "num_draft": 15,
        "ast_b1_tile_policy": "four_path15",
        "cost_per_commit_handoff": False,
        **change,
    }
    with pytest.raises(ValueError, match="four_path15"):
        normalize_progressive_external_policy(policy, num_draft=policy["num_draft"])


def test_static_policy_is_identity_bound():
    policy = {
        **_policy(),
        "num_draft": 15,
        "ast_b1_tile_policy": "four_path15",
        "cost_per_commit_handoff": False,
    }
    value = normalize_progressive_external_policy(policy, num_draft=15)
    assert "four_path15" in progressive_external_identity(value)


def test_failed_target_restores_draft_context_history_rng_and_tickets(monkeypatch):
    _draft, batch, lane = _batch(num_draft=15, b1_tile_policy="four_path15")
    before = (
        list(lane.history),
        lane.rng.snapshot(),
        lane.generated,
        [entry.offset for entry in lane.draft_cache],
    )

    def fail(*args, **kwargs):
        raise RuntimeError("target failure after source production")

    monkeypatch.setattr(batch, "_ast_target_tree_forward", fail)
    try:
        with pytest.raises(RuntimeError, match="target failure"):
            batch._round([lane])
        assert (
            list(lane.history),
            lane.rng.snapshot(),
            lane.generated,
            [entry.offset for entry in lane.draft_cache],
        ) == before
        assert not batch.draft.proposal_pool._pending
        assert not batch._open
    finally:
        batch.close()


def test_last_token_uses_explicit_ordinary_not_anchor_tile(monkeypatch):
    _draft, batch, lane = _batch(num_draft=15, b1_tile_policy="four_path15")
    lane.maximum = lane.generated + 1
    before = lane.generated

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal work entered AST TensorFold")

    monkeypatch.setattr(batch, "_ast_target_tree_forward", forbidden)
    try:
        batch._round([lane])
        assert lane.generated == before + 1
        assert (
            batch.scheduler_stats["external_ast_b1_explicit_ordinary_terminal_rounds"]
            == 1
        )
        assert not batch.scheduler_stats.get("external_ast_anchor_target_launches", 0)
    finally:
        batch.close()


def test_history_pure_eos_masks_stay_batched_and_apply_at_source_positions(monkeypatch):
    draft, batch, lane = _batch(num_draft=15, b1_tile_policy="four_path15")
    calls = []

    def mask(tokens, scores):
        raise AssertionError(
            "static source must use length masks, not token-wise tensor probes"
        )

    mask.history_pure = True
    mask.progressive_branch_safe = True
    mask.forbidden_token_ids_at_length = lambda length: (0,)
    lane.processors = [mask]
    original = draft._hidden

    def hidden(inputs, *args, **kwargs):
        calls.append(tuple(inputs.shape))
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(draft, "_hidden", hidden)
    try:
        batch._propose([lane])
        assert calls == [(1, 16)]
        assert all(
            0 not in path.tokens
            for path in batch.draft.last_continuation_selections[0].paths
        )
    finally:
        batch.close()


@pytest.mark.parametrize("remaining", [1, 2, 3, 4])
def test_full_matching_tile_respects_response_budget_before_rng_draws(remaining):
    from test_ast_tree_verify_host import _Generator, _blocks, _lane
    from mlx2.runtime.ast_tree_verify import verify_tree_tiles

    path = tuple(range(1, 16))
    generator, lane = _Generator(path), _lane()
    lane.maximum = lane.generated + remaining
    before = lane.rng.draws
    result = verify_tree_tiles(generator, lane, _blocks((path,)))
    assert result[0][0].emitted == list(path[:remaining])
    assert result[3][0].draws - before == remaining
    assert len(result[2][0]["tokens"]) - len(lane.cache["tokens"]) == remaining


def test_static_source_does_not_sample_unused_backbone_or_build_dense_laws(monkeypatch):
    draft, batch, lane = _batch(num_draft=15, b1_tile_policy="four_path15")

    def forbidden(*args, **kwargs):
        raise AssertionError("unused backbone pair-table sampling executed")

    monkeypatch.setattr(draft, "pairwise_score_table", forbidden)
    try:
        batch._propose([lane])
        assert len(batch.draft.last_continuation_selections[0].paths) == 4
        assert not lane.rng.draws
    finally:
        batch.close()


def test_unsubmitted_stop_candidates_do_not_disable_static_ast(monkeypatch):
    from dataclasses import replace

    _draft, batch, lane = _batch(num_draft=15, b1_tile_policy="four_path15")
    provider = _SerialTreeProvider()
    batch.ast_target_tree_executor_provider = provider
    stop = _draft.config.vocab_size - 1
    batch.stops = frozenset([stop])
    source = batch.draft.providers["external"]

    def propose(context, limit):
        # Synthetic source fixture: every offered path carries a stop only
        # beyond this tile's depth-four frontier.
        return [
            replace(path, tokens=(*path.tokens[:-1], stop))
            for path in source(context, limit)
        ]

    batch.draft.providers["external"] = propose
    try:
        batch._round([lane])
        assert len(provider.calls) == 1
        assert len(provider.calls[0][0]) == 16
        assert lane.ast_tree_tiles_receipt["observed_used"]
    finally:
        batch.close()


def test_target_selected_stop_ends_full_tile_at_exact_committed_prefix():
    _draft, batch, lane = _batch(num_draft=15, b1_tile_policy="four_path15")
    provider = _SerialTreeProvider()
    batch.ast_target_tree_executor_provider = provider
    logits = batch.model(mx.array([[lane.anchor]]), cache=copy.deepcopy(lane.cache))
    stop = int(mx.argmax(logits[0, -1]).item())
    batch.stops = frozenset([stop])
    before = lane.generated
    try:
        batch._round([lane])
        assert lane.generated == before + 1
        assert lane.ready[-1].token == stop and lane.ready[-1].finish_reason == "stop"
        assert len(provider.calls) == 1
        assert not batch.draft.proposal_pool._pending
    finally:
        batch.close()
