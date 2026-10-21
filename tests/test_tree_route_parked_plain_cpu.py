"""Tree route (dynamic_singleton_tree, provider target): plain parked rounds.

h2h r2c (2026-10-21, Qwen3.6-35B-A3B + DFlash2): the tree route installs the
reversible cost-per-commit controller, but its ordinary probe and its parked
rounds ran a full depth-zero external round through the tree provider, about
twice the plain decode step, and parked cohorts were barred from the
pipelined plain path.  The controller therefore compared the tree against an
inflated ordinary baseline and never parked chat.

Here a tiny hybrid Qwen3.6 MoE target (GDN + attention) with a DFlash2
drafter runs the tree route (the generic exact tree stands in for the
TensorFold provider on CPU) through park -> pipelined plain parked rounds ->
re-probe -> re-engaged tree rounds.  Greedy output must equal plain greedy
decoding, the probes must be plain target steps, and the parked rounds must
use the pipeline.
"""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor
from mlx2.runtime import external_speculative as es
from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

from test_dflash_tree_bridge import _install_reference_cohort
from test_qwen38_dflash2_cpu import ordinary_greedy, tiny_draft_args
from test_qwen_progressive_external_portability_cpu import tiny_qwen36

PROMPT = [3, 9, 27, 81, 15, 44, 2]
MAX_TOKENS = 48
_ROUND = ExternalDraftBatchGenerator._round
_READ = ExternalDraftBatchGenerator._read_parked


@pytest.fixture(autouse=True)
def _tree_env(monkeypatch):
    for name in (
        "MLX2_DFLASH_TOPOLOGY",
        "MLX2_QWEN_TARGET_EXECUTION",
        "MLX2_EXTERNAL_ROUND_TIMING",
        "MLX2_TENSORFOLD_COHORT_LIMIT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused-by-cpu-oracle")


def _pair():
    target = tiny_qwen36()
    mx.random.seed(8)
    draft = DFlash2DraftModel(tiny_draft_args()).bind(target)
    mx.eval(draft.parameters())
    return target, draft


def _settings(**overrides):
    return {
        "enabled": True, "reversible": True, "ordinary_samples": 2,
        "speculative_samples": 2, "loss_decisions": 1, "window": 4,
        "reprobe_tokens": 6, "reprobe_max_tokens": 12, "settle_rounds": 0,
        "deferred_context_cap": 64,
        **overrides,
    }


def _run(monkeypatch, *, handoff, speculative_ns, prompts=(PROMPT,), **options):
    """One greedy tree-route lane to completion under a deterministic clock.

    A speculative (tree) round costs ``speculative_ns(index)``; every plain
    target step or depth-zero round costs 100 ns.
    """
    target, draft = _pair()
    clock = {"now": 1, "spec": 0, "zero": 0}

    def timed_round(self, cohort, *, forced_depth=None):
        result = _ROUND(self, cohort, forced_depth=forced_depth)
        if forced_depth == 0 or all(es._speculation_parked(l) for l in cohort):
            clock["now"] += 100
            clock["zero"] += 1
        else:
            clock["now"] += speculative_ns(clock["spec"])
            clock["spec"] += 1
        return result

    def timed_read(self, inflight, rows, following, **kwargs):
        clock["now"] += 100
        return _READ(self, inflight, rows, following, **kwargs)

    monkeypatch.setattr(es.time, "perf_counter_ns", lambda: clock["now"])
    monkeypatch.setattr(ExternalDraftBatchGenerator, "_round", timed_round)
    monkeypatch.setattr(ExternalDraftBatchGenerator, "_read_parked", timed_read)
    batch = ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="tree-parked-plain-cpu",
        num_draft=3, prefill_step_size=4, completion_batch_size=4,
        dynamic_singleton_tree=True,
        target_tree_executor_provider=Qwen38TargetTreeExecutor.create(),
        cost_per_commit_handoff=handoff,
        **options,
    )
    _install_reference_cohort(batch, monkeypatch, [])
    uids = batch.insert(
        [list(prompt) for prompt in prompts],
        max_tokens=[MAX_TOKENS] * len(prompts),
        sampling_configs=[{"sampling_temp": 0.0}] * len(prompts),
    )
    outputs = {uid: [] for uid in uids}
    finals = {}
    tree_rounds_after_reengage = None
    for _ in range(1200):
        _, responses = batch.next()
        for response in responses:
            outputs[response.uid].append(response.token)
            if response.finish_reason:
                finals[response.uid] = response
        states = [
            batch.lanes[uid].cost_handoff_state
            for uid in uids
            if uid in batch.lanes
        ]
        if tree_rounds_after_reengage is None and any(
            state is not None and state.receipt()["reversible"]["reengagements"]
            for state in states
        ):
            tree_rounds_after_reengage = batch.scheduler_stats[
                "external_auto_tree_rounds"
            ]
        if not batch.lanes:
            break
    else:
        raise AssertionError("scheduler stalled")
    if len(prompts) == 1:
        return outputs[uids[0]], finals[uids[0]], batch, tree_rounds_after_reengage, clock
    return [outputs[uid] for uid in uids], [finals[uid] for uid in uids], batch, tree_rounds_after_reengage, clock


def test_tree_route_parks_on_plain_pipelined_steps_and_reengages(monkeypatch):
    target, _draft = _pair()
    expected = ordinary_greedy(target, PROMPT, MAX_TOKENS)
    plain, _, _, _, _ = _run(
        monkeypatch, handoff=False, speculative_ns=lambda _i: 100_000
    )
    assert plain == expected

    price = lambda i: 100_000 if i < 2 else 10
    tokens, final, batch, reengaged_at, _clock = _run(
        monkeypatch, handoff=_settings(), speculative_ns=price
    )
    assert tokens == expected
    stats = batch.scheduler_stats
    receipt = final.speculative_receipt["cost_per_commit_handoff"]
    reversible = receipt["reversible"]
    assert reversible["parks"] >= 1
    assert reversible["reprobes"] >= 1
    assert reversible["reengagements"] >= 1
    # The parked cohort used the pipeline, and the ordinary probes were
    # plain target steps, not depth-zero rounds through the tree provider.
    assert stats.get("external_parked_pipelined_rounds", 0) > 0
    assert stats.get("external_cost_plain_probe_rounds", 0) >= 2
    assert stats.get("external_parked_pipeline_mismatches", 0) == 0
    # Every plain step costs 100 ns on this clock: the parked baseline the
    # re-probe is judged against counts one token per pipelined read.
    reengage = next(e for e in reversible["events"] if e["event"] == "reengage")
    assert reengage["ordinary_ns_per_committed_token"] == 100.0
    # Speculation resumed on the tree after the re-probe.
    assert reengaged_at is not None
    assert stats["external_auto_tree_rounds"] > reengaged_at
    # The paired draft plane was neither skipped nor double-appended.
    final.cache_sidecar.validate("tree-parked-plain-cpu", len(final.all_tokens))
    assert not batch._prelaunched
    assert batch._parked_inflight is None


def test_tree_route_probe_without_park_keeps_tree_output(monkeypatch):
    """A tree that keeps winning only pays the plain probes."""
    target, _draft = _pair()
    expected = ordinary_greedy(target, PROMPT, MAX_TOKENS)
    tokens, final, batch, _, _ = _run(
        monkeypatch, handoff=_settings(), speculative_ns=lambda _i: 10
    )
    assert tokens == expected
    stats = batch.scheduler_stats
    reversible = final.speculative_receipt["cost_per_commit_handoff"]["reversible"]
    assert reversible["parks"] == 0
    assert stats.get("external_cost_plain_probe_rounds", 0) == 2
    assert stats.get("external_parked_pipelined_rounds", 0) == 0
    final.cache_sidecar.validate("tree-parked-plain-cpu", len(final.all_tokens))


def test_tree_cohort_parks_and_reengages_at_b2(monkeypatch):
    """Two tree lanes park together, pipeline at B2 and return to the tree."""
    prompts = (PROMPT, [5, 6, 70, 71, 72])
    target, _draft = _pair()
    expected = [ordinary_greedy(target, prompt, MAX_TOKENS) for prompt in prompts]
    price = lambda i: 100_000 if i < 2 else 10
    outputs, finals, batch, reengaged_at, _clock = _run(
        monkeypatch, handoff=_settings(), speculative_ns=price, prompts=prompts,
        dynamic_tree_max_width=4,
        tree_node_budget_by_lanes={1: 15, 2: 7, 3: 4, 4: 3},
    )
    assert outputs == expected
    stats = batch.scheduler_stats
    assert stats.get("external_parked_pipelined_rounds", 0) > 0
    assert stats.get("external_cost_plain_probe_rounds", 0) >= 2
    assert stats["external_auto_tree_max_width"] == 2
    assert reengaged_at is not None
    assert stats["external_auto_tree_rounds"] > reengaged_at
    for final in finals:
        reversible = final.speculative_receipt["cost_per_commit_handoff"]["reversible"]
        assert reversible["parks"] >= 1 and reversible["reengagements"] >= 1
        final.cache_sidecar.validate("tree-parked-plain-cpu", len(final.all_tokens))
    assert not batch._prelaunched
