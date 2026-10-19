"""Phase-1 provider AST is exact, single-session and CPU-contract tested."""

from dataclasses import replace

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.ast_planner import (
    ASTExecutionPlanner,
    AlternativeExecutionBackend,
    ExecutionProfile,
    NumericalEquivalenceOracle,
)
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, HostDraftRow
from mlx2.runtime.sample_utils import LaneRNG
from mlx2.runtime.target_tree_executor import TargetTreeDispatchMetrics

from test_external_dflash2_cpu import _forced_target_proposal, tiny


class _BoundTransaction:
    def __init__(self, transaction, provider, events):
        self.transaction = transaction
        self.provider_protocol = provider.protocol
        self.provider_revision_token = provider.source_revision_token
        self.events = events

    @property
    def closed(self):
        return self.transaction.closed

    def commit(self, *, accepted_lengths):
        self.events.append(("commit", tuple(accepted_lengths)))
        return self.transaction.commit(accepted_lengths=accepted_lengths)

    def abort(self):
        self.events.append(("abort",))
        return self.transaction.abort()


class _ReferenceChainProvider:
    source_identity = (("revision", "cpu-fixture"),)
    source_revision_token = "cpu-fixture-source"
    protocol = "cpu-reference-chain-provider"
    maximum_cohort_width = 1

    def __init__(self, batch):
        self.batch = batch
        self.events = []

    def validate_revision(self):
        return None

    def validate_cohort(self, width, configured_limit):
        if width != 1 or configured_limit != 1:
            raise ValueError("fixture provider is single-session")

    def forward(self, model, inputs, parents, cache, layers, *, cached):
        assert parents == list(range(-1, len(inputs) - 1))
        owner = self.batch._target_owner([cache])
        transaction = owner.begin(lengths=[len(inputs)])
        logits, features = model.forward_with_taps(
            mx.array([inputs]), transaction.caches, layers
        )
        self.events.append(("forward", tuple(inputs), tuple(parents)))
        return logits, features, _BoundTransaction(transaction, self, self.events)

    def forward_many(self, *args, **kwargs):
        raise AssertionError("phase-1 AST must not call a cohort provider")

    def dispatch_metrics(self, width, transaction, *, cached):
        return TargetTreeDispatchMetrics(
            width=width,
            physical_packed=False,
            executor_cached=False,
            cached=cached,
            provider_protocol=self.protocol,
        )

    def receipt(self):
        return {"provider": self.protocol}

    def ast_execution_backend(self, profile):
        raise AssertionError("test attaches an already-bound profile")


def _provider_profile():
    primary = ExecutionProfile(
        binding="cpu-provider-profile",
        device="cpu",
        build="cpu-test",
        max_lanes=1,
        tile_widths=(2,),
        terminal_widths=(2, 3),
        packed_min_width=1,
        row_costs=(10.0, 10.0, 10.0),
        launch_cost=0,
        context_row_cost=0,
        draft_cost=0,
    )
    oracle = NumericalEquivalenceOracle(
        profile_binding=primary.binding,
        device=primary.device,
        build=primary.build,
        source_revision_token=_ReferenceChainProvider.source_revision_token,
        evidence="cpu-fixture-exact-oracle",
        equivalent=True,
    )
    backend = AlternativeExecutionBackend(
        backend="target_tree_single_session",
        profile_binding=primary.binding,
        device=primary.device,
        build=primary.build,
        source_revision_token=_ReferenceChainProvider.source_revision_token,
        input_widths=(2, 3),
        row_costs=(1.0, 1.0, 1.0),
        oracle=oracle,
    )
    return replace(primary, alternative_backends=(backend,))


def test_progressive_ast_uses_one_chain_transaction_and_exact_commit_length():
    model, draft = tiny()
    model.configure_target_verify_row_exact(True)
    prompt = [1, 2, 3, 4]
    anchor, _expected, proposal, laws = _forced_target_proposal(
        model, prompt, 3, None
    )
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="provider-progressive-test",
        num_draft=3,
        completion_batch_size=1,
        prefill_step_size=8,
        progressive_verification_tile=2,
    )
    provider = _ReferenceChainProvider(batch)
    batch.ast_execution_planner = ASTExecutionPlanner(
        _provider_profile(), tile_cap=2, source_limit=3, minimum_depth=3
    )
    batch.ast_target_tree_executor_provider = provider
    real = draft.draft_distributions

    def forced(anchors, hidden, caches, count, rngs, temps, **kwargs):
        kwargs.pop("ast_source_budgets", None)
        real(anchors, hidden, caches, count, rngs, temps, **kwargs)
        assert anchors == [anchor] and count == 3
        return [proposal], [laws]

    draft.draft_distributions = forced
    try:
        uid = batch.insert(
            [prompt], max_tokens=[8], lane_rngs=[LaneRNG(919)]
        )[0]
        lane = batch.lanes[uid]
        while lane.anchor is None:
            batch._prefill(lane)
        batch._round([lane])
        forwards = [event for event in provider.events if event[0] == "forward"]
        commits = [event for event in provider.events if event[0] == "commit"]
        assert len(forwards) == len(commits) == 2
        assert all(event[2] == tuple(range(-1, len(event[1]) - 1)) for event in forwards)
        assert commits == [("commit", (2,)), ("commit", (2,))]
        assert not [event for event in provider.events if event[0] == "abort"]
        backend = batch.ast_execution_planner.receipt()["alternative_backends"][0]
        assert backend["selected"] is True
        assert backend["observed_used"] is True
        assert backend["launches"] == 2
        assert backend["qualified"] is False
        assert lane.ready
    finally:
        batch.close()


def test_progressive_ast_aborts_open_provider_transaction_on_law_failure():
    model, draft = tiny()
    model.configure_target_verify_row_exact(True)
    prompt = [1, 2, 3, 4]
    _anchor, _expected, proposal, laws = _forced_target_proposal(
        model, prompt, 3, None
    )
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="provider-abort-test",
        num_draft=3,
        completion_batch_size=1,
        prefill_step_size=8,
        progressive_verification_tile=2,
    )
    provider = _ReferenceChainProvider(batch)
    batch.ast_execution_planner = ASTExecutionPlanner(
        _provider_profile(), tile_cap=2, source_limit=3, minimum_depth=3
    )
    batch.ast_target_tree_executor_provider = provider
    try:
        uid = batch.insert(
            [prompt],
            max_tokens=[8],
            lane_rngs=[LaneRNG(919)],
            sampling_configs=[{"sampling_temp": 0.8}],
        )[0]
        lane = batch.lanes[uid]
        while lane.anchor is None:
            batch._prefill(lane)

        def fail_law(*args, **kwargs):
            raise RuntimeError("injected law failure")

        batch._target_law = fail_law
        with pytest.raises(RuntimeError, match="injected law failure"):
            batch._progressive_verify_many(
                [lane], [HostDraftRow(proposal, laws)]
            )
        assert ("abort",) in provider.events
        assert not [event for event in provider.events if event[0] == "commit"]
    finally:
        batch.close()
