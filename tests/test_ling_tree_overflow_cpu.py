"""Synthetic UBTree overflow scheduling; CPU only, no checkpoint qualification."""

from contextlib import closing
from dataclasses import fields
from types import SimpleNamespace

import pytest
from test_dflash_pair_select import drain, generator, tiny

from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
from mlx2.adapters.ling3_flash import Ling3FlashAdapter
from mlx2.runtime.drafters.base_config import DFlashConfig
from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
from mlx2.runtime.drafters.ubtree import UBTreeDraftModel
from mlx2.runtime.drafters.ubtree_config import UBTreeConfig
from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor


def _tiny_ubtree():
    model, prototype = tiny(vocab=64, top_k=16, block_size=16)
    values = {field.name: getattr(prototype.config, field.name)
              for field in fields(DFlashConfig)}
    values.update(layer_types=["full_attention"] * 2, sliding_window=None)
    config = UBTreeConfig(
        **values, selector_rank=4, factorizer_hidden_width=8,
        selector_top_k=8, tree_supertree_width=4, tree_budget=15,
    )
    return model, UBTreeDraftModel(config).bind(model), prototype


def _batch(model, draft, width, **options):
    return generator(
        model, draft, completion_batch_size=width, ready_drain="all",
        dynamic_singleton_tree=True, dynamic_tree_max_width=4,
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3},
        target_tree_executor_provider=HybridTreeExecutor.create(),
        cost_per_commit_handoff=False, **options,
    )


@pytest.mark.parametrize("grafted", [False, True])
def test_ling_adapter_owns_cohort_overflow(monkeypatch, grafted):
    adapter = object.__new__(Ling3FlashAdapter)
    adapter.external_policy = {}
    base = SimpleNamespace(receipt_kind="external_ubtree")
    if grafted:
        adapter.draft_model = object.__new__(PromptLookupDraftModel)
        adapter.draft_model.base = base
    else:
        adapter.draft_model = base
    monkeypatch.setattr(
        ExternalDraftAdapterMixin, "create_external_batch", lambda self, **kw: kw
    )
    options = adapter.create_external_batch(completion_batch_size=8)
    assert options["dynamic_tree_overflow"] == "cohort"
    assert options["dynamic_tree_max_width"] == 4
    with pytest.raises(ValueError, match="bounded tree cohort overflow"):
        adapter.create_external_batch(dynamic_tree_overflow="chain")


@pytest.mark.parametrize("width", [5, 8])
@pytest.mark.parametrize("grafted", [False, True])
def test_ubtree_overflow_parity_fairness_and_row_bound(monkeypatch, width, grafted):
    model, base, prototype = _tiny_ubtree()
    assert not callable(getattr(base, "draft_distributions", None))
    draft = PromptLookupDraftModel(model, num_draft=15, base=base) if grafted else base
    # Repeated request-local spans ensure the graft itself is exercised.
    prompts = [[i + 1, 20, 21, 22, 23, i + 1, 20, 21] for i in range(width)]
    configs = [{"sampling_temp": 0.0, "emit_logprobs": False}] * width
    with closing(generator(model, prototype, completion_batch_size=width, ready_drain="all")) as ordinary:
        ordinary.insert(prompts, max_tokens=[20] * width, sampling_configs=configs)
        for lane in ordinary.lanes.values():
            lane.ordinary = True
        expected, _ = drain(ordinary)

    with closing(_batch(model, draft, width, dynamic_tree_overflow="cohort")) as batch:
        for live in range(5, width + 1):
            assert batch.verify_rows_by_width(width)[str(live)] == 16
            assert batch._chain_draft_cap(live) is None
        seen = []
        forward = batch._target_tree_forward_many

        def capture(cohort, inputs, parents):
            seen.append((batch._auto_active_width, tuple(lane.uid for lane in cohort),
                         sum(map(len, inputs))))
            return forward(cohort, inputs, parents)

        monkeypatch.setattr(batch, "_target_tree_forward_many", capture)
        uids = batch.insert(prompts, max_tokens=[20] * width, sampling_configs=configs)
        actual, receipts, overflow_receipts = {}, {}, []
        for _ in range(150):
            _, responses = batch.next()
            for response in responses:
                actual.setdefault(response.uid, []).append(response.token)
                receipts[response.uid] = response.speculative_receipt
                route = response.speculative_receipt.get("batch_size_route")
                if route and route["active_lanes"] > 4:
                    overflow_receipts.append(route)
            if not batch.lanes:
                break
        assert not batch.lanes, "overflow session starved"
        assert actual == expected
        assert set(actual) == set(uids)
        assert all(len(tokens) == 20 for tokens in actual.values())
        assert seen and all(len(cohort) <= 4 and rows <= 16 for _, cohort, rows in seen)
        overflow = [(cohort, rows) for active, cohort, rows in seen if active == width]
        assert overflow
        # With all sessions ready, age ordering serves all eight lanes within
        # two bounded cohorts rather than repeatedly serving the same four.
        first_two = overflow[:2]
        assert set().union(*(set(cohort) for cohort, _ in first_two)) == set(uids)
        assert batch.scheduler_stats["external_auto_chain_rounds"] == 0
        assert overflow_receipts
        assert all(route["policy"] == "tree15_bounded_cohorts_v1"
                   and route["overflow"] == "cohort"
                   and route["current"] == "tree15_provider"
                   and "chain_draft_cap" not in route for route in overflow_receipts)
        if grafted:
            assert draft.stats["copy_paths"] > 0
            assert draft.stats["tree_calls"] > 0


def test_dynamic_chain_default_and_invalid_overflow():
    model, draft = tiny(vocab=64, top_k=16, block_size=16)
    with closing(_batch(model, draft, 8)) as default:
        assert default.dynamic_tree_overflow == "chain"
        assert default.verify_rows_by_width(8)["8"] == 8 * (default._chain_draft_cap(8) + 1)
    with pytest.raises(ValueError, match="dynamic_tree_overflow"):
        _batch(model, draft, 8, dynamic_tree_overflow="unknown")
    with pytest.raises(ValueError, match="enabled dynamic tree"):
        generator(model, draft, dynamic_tree_overflow="cohort")


@pytest.mark.parametrize("route", [None, "tree15_b1_b4_chain_b5plus_v1", "tree15_bounded_cohorts_v1"])
def test_ling_canonical_route_is_selectable_and_reported(route):
    adapter = object.__new__(Ling3FlashAdapter)
    policy = {"draft_model": "/models/ubtree", "num_draft": 15, "batch_size_route": route}
    assert adapter._parse_external_policy(policy, family="Ling")
    assert adapter.external_policy["batch_size_route"] == "tree15_bounded_cohorts_v1"
    adapter.draft_model = SimpleNamespace()
    config = adapter.execution_config(max_lanes=8, prefill_step=512)
    assert config["batch_size_route"] == "tree15_bounded_cohorts_v1"
    assert config["dynamic_tree_overflow"] == "cohort"
    assert config["tree_node_budget_by_lanes"] == {1: 15, 2: 7, 3: 4, 4: 3}
    assert policy["batch_size_route"] == route
