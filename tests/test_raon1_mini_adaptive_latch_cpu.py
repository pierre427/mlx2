"""CPU checks: do the external-route latches engage on a Raon-shaped route?

Raon-1-mini's one-lane default is the external DFlash2 nd5 chain (the official
4-stream drafter, wrapped in the default PLD proposal composition).  On prose
it commits ~1.9 tokens per round against 4.1 on GSM8K, and loses to ordinary
decode (docs/research/2026-11-26-raon-adaptive-latch-inventory.md).  These
tests build a tiny random Raon target (4 mHC streams, MLA + RDA layers, MoE)
with a 4-stream DFlash2 drafter bound to it, so the generic external executor
runs the same tap and verify contracts as the served route, and check that the
two opt-in mechanisms that can move such a lane off speculation engage on it,
keep output exact, and report their counters:

* ``cost_per_commit_handoff`` (reversible park), with the two presets the GPU
  unit runs (AST self-configuring, Hy4 ``hy4_park_20261121``) and the tree-tile
  runtime default;
* ``adaptive_verification`` (lagged cohort cost model), with and without the
  reject cooldown (scheduled solo rounds).

The random drafter accepts almost nothing, a harsher case than prose.  Timing
is a synthetic clock: CPU tests prove engagement and exactness, never speed.
They also pin how the operator reaches each option on the served route
(``--ast --ast-drafter`` plus ``--execution-policy``) through the Raon adapter.
"""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx.utils import tree_map

mx.set_default_device(mx.cpu)

from test_external_dflash2_cpu import drain
from test_raon1_mini_cpu import _model

from mlx2.runtime import external_speculative as es
from mlx2.runtime.cost_per_commit_handoff import (
    CostPerCommitHandoffSettings,
)

PROMPT = [5, 11, 17, 23, 29, 31, 37, 41]
NUM_DRAFT = 5
_ROUND = es.ExternalDraftBatchGenerator._round


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def ast_preset():
    """The AST self-configuring reversible preset, as a policy dictionary."""
    return CostPerCommitHandoffSettings.ast_self_configuring_defaults().as_dict()


def tree_tile_preset():
    """The tree-tile runtime default (settle 1 round, 2 speculative samples)."""
    return replace(
        CostPerCommitHandoffSettings.ast_self_configuring_defaults(),
        speculative_samples=2, settle_rounds=1, reprobe_max_tokens=256,
    ).as_dict()


def hy4_preset():
    """The slower-to-park Hy4 preset the GPU unit's park-hy4 arm runs."""
    from mlx2.adapters.hy_v4_flash import COST_HANDOFF_PRESETS

    return CostPerCommitHandoffSettings.from_value(
        COST_HANDOFF_PRESETS["hy4_park_20261121"]).as_dict()


def raon_route(*, composed=True):
    """Tiny Raon target + 4-stream DFlash2 (block 6, nd5), default composition."""
    from mlx2.adapters.external_draft_policy import (
        DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION,
    )
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config
    from mlx2.runtime.proposal_composition import ComposedDraftModel

    target = _model()
    assert target.tap_streams == 4
    args = DFlash2Config(
        hidden_size=64, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16, vocab_size=97, num_target_layers=5,
        target_layer_ids=[1, 3], target_streams_per_layer=4, conv_kernel_size=2,
        conv_group_size=16, selector_rank=4, selector_top_k=4, block_size=NUM_DRAFT + 1,
        mask_token_id=96, max_position_embeddings=4096, sliding_window=8,
        layer_types=["sliding_attention"] * 2,
    )
    draft = DFlash2DraftModel(args)
    mx.random.seed(3)
    draft.update(tree_map(lambda p: mx.random.normal(p.shape).astype(p.dtype) * 0.25,
                          draft.parameters()))
    draft = draft.bind(target)
    if composed:
        draft = ComposedDraftModel(draft, dict(DEFAULT_EXTERNAL_PROPOSAL_COMPOSITION))
    return target, draft


def run(monkeypatch, *, speculative_ns=None, max_tokens=48, **kwargs):
    """Greedy nd5 request; a synthetic clock prices rounds when ``speculative_ns``
    is given (an ordinary depth-0 round costs 100 ns, a speculative round
    ``speculative_ns``), so the cost controller sees a speculator that loses."""
    target, draft = raon_route()
    if speculative_ns is not None:
        clock = {"now": 1}
        monkeypatch.setattr(es.time, "perf_counter_ns", lambda: clock["now"])

        def timed(self, cohort, *, forced_depth=None):
            parked = all(es._speculation_parked(lane) for lane in cohort)
            result = _ROUND(self, cohort, forced_depth=forced_depth)
            clock["now"] += 100 if forced_depth == 0 or parked else speculative_ns
            return result

        monkeypatch.setattr(es.ExternalDraftBatchGenerator, "_round", timed)
    engine = es.ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="raon-test", num_draft=NUM_DRAFT,
        prefill_step_size=4, **kwargs,
    )
    uid = engine.insert([list(PROMPT)], max_tokens=[max_tokens],
                        sampling_configs=[{"sampling_temp": 0.0}])[0]
    outputs, finishes = drain(engine)
    return outputs[uid], finishes[uid], engine


def ordinary_reference(max_tokens=48):
    """Plain greedy decode on the same target (the ordinary-decode reference)."""
    target, _ = raon_route(composed=False)
    cache = target.make_cache()
    logits = target(mx.array([PROMPT]), cache=cache)
    tokens = []
    for _ in range(max_tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        logits = target(mx.array([[token]]), cache=cache)
    return tokens


def test_raon_shaped_default_route_is_exact_and_low_acceptance(monkeypatch):
    tokens, final, _engine = run(monkeypatch)
    assert len(tokens) == 48
    assert tokens == ordinary_reference()
    receipt = final.speculative_receipt
    assert receipt["proposed"] > 0
    # The random drafter accepts (almost) nothing: worse than Raon prose.
    assert receipt["accepted"] <= receipt["proposed"] // 10
    assert "cost_per_commit_handoff" not in receipt
    assert "adaptive_verification" not in receipt


@pytest.mark.parametrize("preset", [ast_preset, hy4_preset, tree_tile_preset],
                         ids=["ast", "hy4", "tree_tile"])
def test_reversible_cost_park_engages_on_raon_and_keeps_output_exact(monkeypatch, preset):
    baseline, _, _ = run(monkeypatch, speculative_ns=100_000)
    tokens, final, engine = run(
        monkeypatch, speculative_ns=100_000, cost_per_commit_handoff=preset())
    assert tokens == baseline
    receipt = final.speculative_receipt["cost_per_commit_handoff"]
    assert receipt["selection_authority"] == "explicit_policy"
    assert receipt["defaulted"] is False
    reversible = receipt["reversible"]
    assert reversible["parks"] >= 1
    assert reversible["reprobes"] >= 1
    assert reversible["reengagements"] == 0
    assert final.speculative_receipt["ordinary_fallback"] is False
    stats = engine.scheduler_stats
    assert stats.get("external_cost_park_events", 0) >= 1
    assert stats.get("external_cost_parked_rounds", 0) > 0
    final.cache_sidecar.validate("raon-test", len(final.all_tokens))


def test_cost_park_stays_off_when_speculation_wins_on_raon(monkeypatch):
    # A speculative round as cheap as an ordinary one commits more per second:
    # the controller must keep speculating (GSM8K-like economics).
    baseline, _, _ = run(monkeypatch, speculative_ns=100)
    tokens, final, engine = run(
        monkeypatch, speculative_ns=100, cost_per_commit_handoff=ast_preset())
    assert tokens == baseline
    reversible = final.speculative_receipt["cost_per_commit_handoff"]["reversible"]
    # Only the opening ordinary baseline rounds are target-only.
    assert reversible["parks"] == 0
    assert engine.scheduler_stats.get("external_cost_parked_rounds", 0) == 0


def adaptive_policy(**overrides):
    # Steep per-depth verify costs make shallow depths score higher once the
    # estimator has seen low acceptance.  The served arm derives its table
    # from measured fixed-depth rounds instead.
    return {
        "verification_costs": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "draft_cost": 0.0, "min_observations": 8, "full_depth_interval": 16,
        "cost_source": "raon_cpu_test_synthetic", **overrides,
    }


def test_adaptive_depth_trims_raon_rounds_and_keeps_output_exact(monkeypatch):
    baseline, _, _ = run(monkeypatch, max_tokens=64)
    tokens, final, engine = run(
        monkeypatch, max_tokens=64, adaptive_verification=adaptive_policy())
    assert tokens == baseline
    receipt = final.speculative_receipt["adaptive_verification"]
    assert receipt["selected"] is True and receipt["qualified"] is False
    assert receipt["policy"] == "lagged_cohort_cost_model"
    assert receipt["trimmed_rounds"] > 0
    assert receipt["observed_used"] is True
    assert engine.scheduler_stats["external_adaptive_trimmed_rounds"] > 0


def test_adaptive_depth_keeps_full_depth_when_depth_is_free_on_raon(monkeypatch):
    # Control: with a uniform per-depth cost a deeper round is never worse, so
    # the estimator keeps depth 5.  Engagement in the steep-table test is the
    # cost model's choice, not prompt-lookup copies shorter than the depth.
    baseline, _, _ = run(monkeypatch, max_tokens=64)
    tokens, final, _engine = run(
        monkeypatch, max_tokens=64,
        adaptive_verification=adaptive_policy(verification_costs=[1.0] * (NUM_DRAFT + 1)))
    assert tokens == baseline
    receipt = final.speculative_receipt["adaptive_verification"]
    assert receipt["trimmed_rounds"] == 0 and receipt["observed_used"] is False
    # Below depth 5 only the request's final rounds, capped by max_tokens.
    histogram = receipt["depth_histogram"]
    assert sum(c for d, c in histogram.items() if int(d) < NUM_DRAFT) <= NUM_DRAFT


def test_adaptive_reject_cooldown_schedules_solo_rounds_on_raon(monkeypatch):
    baseline, _, _ = run(monkeypatch, max_tokens=64)
    tokens, final, _engine = run(
        monkeypatch, max_tokens=64,
        adaptive_verification=adaptive_policy(reject_cooldown=8))
    assert tokens == baseline
    cooldown = final.speculative_receipt["adaptive_verification"]["reject_cooldown"]
    assert cooldown["cap"] == 8
    assert cooldown["started"] >= 1 and cooldown["solo_rounds"] >= 1


# ------------------------------------------------- operator plumbing ----

def _raon_resolution():
    from mlx2.adapters.raon1_mini import RaonAdapter

    return SimpleNamespace(adapter_type=RaonAdapter, artifact={}, default_route="ordinary",
                           descriptor=SimpleNamespace(metadata={}))


def test_ast_invocation_keeps_latch_policy_keys_on_raon():
    from mlx2.server import apply_ast_invocation

    args = SimpleNamespace(ast=True, ast_drafter="/models/Raon-1-mini-dflash2",
                           ast_topology=None, ast_graft_pld=False, ast_tiles=None,
                           ast_tile_rows=None, ast_tree_calibration=None,
                           ast_batch_row_sharing=False, draft_model=None, model="/models/raon")
    policy = {"cost_per_commit_handoff": ast_preset(),
              "adaptive_verification": adaptive_policy()}
    resolved = apply_ast_invocation(args, policy, _raon_resolution())
    assert args.external_draft is True
    assert args.ast_route["tree_tiles"] is False
    assert resolved["draft_model"] == "/models/Raon-1-mini-dflash2"
    assert resolved["cost_per_commit_handoff"] == ast_preset()
    assert resolved["adaptive_verification"] == adaptive_policy()


def test_raon_adapter_accepts_and_forwards_latch_policies(monkeypatch):
    from mlx2.adapters.raon1_mini import RaonAdapter

    adapter = RaonAdapter.__new__(RaonAdapter)
    policy = {"draft_model": "/draft", "cost_per_commit_handoff": ast_preset(),
              "adaptive_verification": adaptive_policy(reject_cooldown=8)}
    assert adapter._parse_external_policy(policy, family="Raon-1-mini")
    record = {"args": SimpleNamespace(block_size=8, minimum_proposal_length=1)}
    adapter._check_num_draft(record)
    assert adapter._external_num_draft() == 5
    adapter.model = None
    config = adapter._external_execution_config(max_lanes=1, prefill_step=2048)
    assert config["cost_per_commit_handoff"] == ast_preset()
    seen = {}

    class Recorder:
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(es, "ExternalDraftBatchGenerator", Recorder)
    adapter.draft_model = object()
    adapter._initialize_external_feedback = lambda: None
    adapter.identity = {"fingerprint": "raon-test"}
    adapter.create_external_batch(completion_batch_size=1)
    assert seen["cost_per_commit_handoff"] == ast_preset()
    assert seen["adaptive_verification"] == adaptive_policy(reject_cooldown=8)

