"""Real trained-block proposal wiring; CPU host admission is not MPP evidence."""

import copy
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.adapters.external_draft_policy import (
    normalize_progressive_external_policy,
    progressive_external_identity,
    progressive_external_terminal_input_widths,
    resolve_progressive_external_tile,
)
from mlx2.adapters.proposal_path_sources import build_continuation_drafter
from mlx2.adapters.qwen38_fast_ast import Qwen38FastAST
from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor
from mlx2.runtime.ast_planner import (
    AlternativeExecutionBackend,
    NumericalEquivalenceOracle,
)
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.progressive_external_verify import progressive_target_protocol_binding
from test_qwen38_dflash2_cpu import PROMPTS, tiny_pair


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture
def wide_startup(monkeypatch):
    """These tests exercise four-path lattice wiring at startup; production
    seeds a chain and widens only on measured throughput."""
    from mlx2.runtime.ast_tree_tuning import ASTTreeTuner

    monkeypatch.setattr(ASTTreeTuner, "SEED_BRANCHES", 4)


def _policy():
    return {
        "num_draft": 7,
        "progressive_verification_tile": "auto",
        "progressive_verification_fidelity": "exact",
        "progressive_alternative_branches": 4,
        "progressive_alternative_strategy": "confidence_tree",
        "ast_target_tree_projection_law": "lane_mpp_exact",
        "ast_target_tree_oracle": "/cpu-only/never-loaded.json",
        "target_verify_row_exact": False,
        "continuation_pool": {
            "sources": ["external"],
            "limit": 4,
            "ast_planner": True,
        },
    }


class _CPUProvider(Qwen38TargetTreeExecutor):
    @property
    def ast_tree_tile_rows(self):
        return 16

    def validate_revision(self):
        pass

    def ast_execution_backend(self, profile):
        widths = tuple(range(1, len(profile.row_costs) + 1))
        return AlternativeExecutionBackend(
            backend="target_tree_single_session",
            profile_binding=profile.binding,
            device=profile.device,
            build=profile.build,
            source_revision_token=self.source_revision_token,
            input_widths=widths,
            row_costs=profile.row_costs,
            oracle=NumericalEquivalenceOracle(
                profile.binding,
                profile.device,
                profile.build,
                self.source_revision_token,
                "CPU-host-fixture-not-GPU",
                True,
            ),
        )

    def forward(self, *args, **kwargs):
        raise AssertionError("proposal-only CPU test must not execute TensorFold")


def _batch(maximum=24, num_draft=7, *, capacity=1, b1_tile_policy=None):
    target, draft = tiny_pair()
    Qwen38FastAST(target)
    policy = normalize_progressive_external_policy(
        {**_policy(), "num_draft": num_draft,
         **({"ast_b1_tile_policy": b1_tile_policy, "cost_per_commit_handoff": False} if b1_tile_policy is not None else {})}, num_draft=num_draft
    )
    tile = resolve_progressive_external_tile(
        policy, num_draft=num_draft, target=target, max_lanes=1
    )
    wrapper = build_continuation_drafter(
        target,
        draft,
        policy["continuation_pool"],
        target_revision="a" * 64,
        draft_revision="b" * 64,
        tokenizer_revision="a" * 64,
        session_revision="c" * 64,
    )
    binding = progressive_target_protocol_binding(target, "exact")
    provider = _CPUProvider((), "cpu-source", ast_fast_tree=True)
    batch = ExternalDraftBatchGenerator(
        target,
        draft_model=wrapper,
        binding="cpu-fast-tree-host",
        num_draft=num_draft,
        completion_batch_size=capacity,
        ready_drain="all",
        continuation_pool=wrapper.policy,
        continuation_pool_ast_bridge=True,
        progressive_verification_tile=tile,
        progressive_verification_tile_policy="auto",
        progressive_terminal_input_widths=progressive_external_terminal_input_widths(
            target, max_lanes=1, num_draft=num_draft, fidelity="exact"
        ),
        progressive_verification_fidelity="exact",
        progressive_target_protocol=binding["target_protocol"],
        progressive_protocol_binding=binding,
        progressive_alternative_branches=4,
        progressive_alternative_strategy="confidence_tree",
        ast_target_tree_executor_provider=provider,
        ast_b1_tile_policy=b1_tile_policy,
        cost_per_commit_handoff=False,
    )
    uid = batch.insert(
        [PROMPTS[0]],
        max_tokens=[maximum],
        sampling_configs=[{"sampling_temp": 0}],
    )[0]
    lane = batch.lanes[uid]
    while lane.anchor is None:
        batch._prefill(lane)
    lane.ready.clear()
    # _round owns this publication list; these tests stop after _propose.
    batch._continuation_open_selections = []
    return draft, batch, lane


def test_fast_policy_remains_explicit_separate_and_source_bound():
    policy = normalize_progressive_external_policy(_policy(), num_draft=7)
    assert policy["target_verify_row_exact"] is False
    assert "lane_mpp_exact" in progressive_external_identity(policy)
    for change in (
        {"target_verify_row_exact": True},
        {"progressive_verification_fidelity": "near_tie_bounded"},
        {"ast_physical_target": True},
        {"progressive_alternative_strategy": "distinct_root"},
        {"progressive_cross_session_pack_rows": 4},
        {"ast_target_tree_projection_law": "invented"},
    ):
        with pytest.raises(ValueError):
            normalize_progressive_external_policy({**_policy(), **change}, num_draft=7)
    # Legacy REV oracle does not acquire tree permission from a pool alone.
    ordinary = copy.deepcopy(_policy())
    ordinary.pop("ast_target_tree_projection_law")
    with pytest.raises(ValueError, match="primary"):
        normalize_progressive_external_policy(ordinary, num_draft=7)


def test_actual_dflash_block7_reuses_lattice_for_four_path_node28_budget(
    monkeypatch, wide_startup
):
    draft, batch, lane = _batch()
    calls = []
    original = draft.draft_distributions

    def observed(anchors, hidden, cache, depth, *args, **kwargs):
        calls.append((len(anchors), depth, kwargs.get("capture_continuation_tree")))
        return original(anchors, hidden, cache, depth, *args, **kwargs)

    monkeypatch.setattr(draft, "draft_distributions", observed)
    try:
        blocks = batch._propose([lane])
        assert calls == [(1, 7, True)]
        assert draft.config.block_size == 8
        assert batch.ast_tree_tile_rows == 16
        profile = batch.ast_execution_planner.profile
        assert len(profile.row_costs) == 16
        assert max(profile.terminal_widths) <= 8
        assert profile.alternative_backends[0].input_widths == tuple(range(1, 17))
        assert lane.ast_source_budget["depth"] == 7
        assert lane.ast_source_budget["branches"] == 4
        assert lane.ast_source_budget["node_budget"] == 28
        assert lane.ast_source_budget["reason"] == "startup_measurement"
        source = lane.continuation_source_geometry_last["external"]
        assert source["node_budget"] == 28
        assert source["requested_depth"] == 7
        assert source["backbone_reused"] is True
        assert source["private_head_calls"] == 0
        assert blocks[0] is not None
        alternatives = batch._last_alternative_blocks[lane.uid]
        assert 1 < len(alternatives) <= 4
        assert alternatives[0].tokens == blocks[0].tokens
        assert all(1 <= len(block.tokens) <= 7 for block in alternatives)
        assert batch._last_alternative_prefix_tokens[lane.uid] == 0
        assert batch._last_alternative_topology[lane.uid] == "unique_prefix_tree_tiles"
        for rank, block in enumerate(alternatives):
            assert block.continuation_ast_branch_index == rank
            assert block.continuation_ast_provenance["rank"] == rank
            assert block.continuation_ast_provenance["contributors"]
            assert len(block.laws) == len(block.tokens)
    finally:
        batch.close()


def test_short_remaining_request_never_exceeds_trained_or_response_depth():
    _draft, batch, lane = _batch(maximum=4)
    try:
        blocks = batch._propose([lane])
        assert 0 < lane.ast_source_budget["depth"] <= 3
        assert blocks[0] is not None
        assert len(blocks[0].tokens) <= 3
        assert all(
            len(block.tokens) <= 3
            for block in batch._last_alternative_blocks.get(lane.uid, [blocks[0]])
        )
    finally:
        batch.close()


def test_fast_ast_ready_pressure_reallocates_source_work_without_changing_law(
    wide_startup,
):
    _draft, batch, lane = _batch(maximum=32, num_draft=15, capacity=4)
    batch._ast_tree_ready_sessions = 4
    try:
        blocks = batch._propose([lane])
        plan = lane.ast_tree_source_plan
        assert plan.ready_sessions == 4
        assert plan.depth == 15 and plan.branches == 2
        assert plan.max_tile_rows == 16 and plan.max_tiles == 2
        assert blocks[0] is not None and len(blocks[0].tokens) <= 15
        assert batch.ast_execution_planner.profile.alternative_backends
        assert batch.verify_rows_by_width(4) == {
            str(width): 16 for width in range(1, 5)
        }
        assert (
            batch._continuation_ast_bridge_geometry(4)["maximum_target_rows_per_launch"]
            == 16
        )
        resolved = resolve_progressive_external_tile(
            normalize_progressive_external_policy(_policy(), num_draft=15),
            num_draft=15,
            target=batch.model,
            max_lanes=4,
        )
        assert resolved == batch.progressive_verification_tile
    finally:
        batch.close()


def test_fast_ast_batch_services_each_private_request_in_fair_order(monkeypatch):
    _draft, batch, first = _batch(maximum=32, num_draft=15, capacity=4)
    try:
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
        calls = []

        def turn(group, **kwargs):
            calls.append(
                (tuple(lane.uid for lane in group), batch._ast_tree_ready_sessions)
            )

        monkeypatch.setattr(batch, "_round_at_batch_route", turn)
        batch._decode_phase(lanes)
        assert calls == [((lane.uid,), 4) for lane in lanes]
        assert len({lane.decode_service_epoch for lane in lanes}) == 4
    finally:
        batch.close()


def test_adapter_constructs_four_request_slots_with_one_private_tree_provider(
    monkeypatch,
):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime import external_speculative

    _draft, batch, _lane = _batch(maximum=32, num_draft=15, capacity=4)
    calls = []
    provider = batch.ast_target_tree_executor_provider

    def bind_oracle(path, **kwargs):
        calls.append((path, kwargs))
        return provider

    monkeypatch.setattr(
        Qwen38TargetTreeExecutor,
        "create",
        classmethod(lambda cls, **kwargs: SimpleNamespace(with_ast_oracle=bind_oracle)),
    )
    monkeypatch.setattr(
        external_speculative,
        "ExternalDraftBatchGenerator",
        lambda model, **kwargs: SimpleNamespace(model=model, kwargs=kwargs),
    )
    adapter = SimpleNamespace(
        model=batch.model,
        draft_model=batch.draft,
        external_policy={**_policy(), "ast_target_tree_oracle": "cpu-oracle"},
        identity={"fingerprint": "cpu-private-tree"},
        _initialize_external_feedback=lambda: None,
        _external_num_draft=lambda: 15,
        _progressive_external_binding=progressive_target_protocol_binding(
            batch.model, "exact"
        ),
    )
    try:
        created = Qwen3827BAdapter.create_external_batch(
            adapter, completion_batch_size=4
        )
        assert created.kwargs["completion_batch_size"] == 4
        assert created.kwargs["ast_target_tree_executor_provider"] is provider
        assert created.kwargs["ready_drain"] == "all"
        assert calls[0][0] == "cpu-oracle"
        assert calls[0][1]["capture_layers"] == batch.draft.config.target_layer_ids
    finally:
        batch.close()


def test_actual_dflash_block15_keeps_trained_metadata_and_reuses_one_lattice(
    monkeypatch, wide_startup,
):
    draft, batch, lane = _batch(maximum=32, num_draft=15)
    calls = []
    lattice_shapes = []
    original = draft.draft_distributions

    def observed(anchors, hidden, cache, depth, *args, **kwargs):
        calls.append((len(anchors), depth, kwargs.get("capture_continuation_tree")))
        result = original(anchors, hidden, cache, depth, *args, **kwargs)
        lattice = draft.last_continuation_lattice
        lattice_shapes.append((lattice["depth"], lattice["pending"][0].shape[1]))
        return result

    monkeypatch.setattr(draft, "draft_distributions", observed)
    try:
        blocks = batch._propose([lane])
        assert calls == [(1, 15, True)]
        assert draft.config.block_size == 8
        assert draft.receipt_settings["trained_block_size"] == 8
        assert draft.receipt_settings["maximum_proposal_length"] == 15
        assert batch.draft.source_records["external"].max_depth == 15
        assert lane.ast_source_budget["depth"] == 15
        branches = lane.ast_source_budget["branches"]
        assert 1 <= branches <= 4
        assert lane.ast_source_budget["node_budget"] == 15 * branches
        source = lane.continuation_source_geometry_last["external"]
        assert source["requested_depth"] == 15
        assert source["node_budget"] == 15 * branches
        assert source["backbone_reused"] is True
        assert source["private_head_calls"] == 0
        assert blocks[0] is not None
        assert lattice_shapes == [(15, 15)]
        assert all(
            1 <= len(block.tokens) <= 15 and len(block.laws) == len(block.tokens)
            for block in batch._last_alternative_blocks[lane.uid]
        )
    finally:
        batch.close()
