"""Hy4-flash DFly2 latches and latched prompt lookup on the tiny fixture (CPU).

* Adaptive depth (``adaptive_verification``): the engine's lagged cost model
  caps the DFly2 chain through the tree route's node budget.
* Park latch (``cost_per_commit_handoff``): the reversible park/reprobe
  controller the tree route already installs, driven here by a
  deterministic clock to park, re-probe and resume.
* Prompt lookup (``--prompt-lookup``): the generic latched PLD generator on
  the Hy4 layer caches (per-lane exact rollback, Engram token tails).

Every request must equal its own single-lane ordinary greedy decode.
"""

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import (  # noqa: E402
    LONG,
    _drain,
    _greedy,
    _tiny_artifact,
    build,
    checkpoint,
    tiny_config,
)

from mlx2.runtime import external_speculative as es  # noqa: E402
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, drafter = build(config, W)
    return config, W, model, drafter


PROMPTS = [LONG[9:21], LONG[10:24], LONG[50:58] + LONG[50:55], LONG[60:81]]
# A cheap-looking depth 1 against expensive deep chains: on random weights
# (low acceptance) the cost model trims as soon as it has observations.
STEEP = [1, 30, 60, 90, 120, 150]
ADAPTIVE = {"verification_costs": STEEP, "draft_cost": 0, "min_observations": 4,
            "full_depth_interval": 8}


def _adapter(tmp_path, tiny, policy):
    from mlx2.adapters import hy_v4_flash as A

    config, W, _, _ = tiny
    return A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)), execution_policy=policy)


def _run(adapter, width, n=40, max_tokens=None):
    batch = adapter.create_external_batch(completion_batch_size=width, prefill_step_size=4)
    prompts = PROMPTS[:width]
    limits = max_tokens or [n] * width
    uids = batch.insert(prompts, max_tokens=limits, sampling_configs=[{"sampling_temp": 0}] * width)
    out = {u: [] for u in uids}
    for _ in range(2000):
        _, responses = batch.next()
        for r in responses:
            out[r.uid].append(int(r.token))
        if not batch.lanes:
            break
    else:
        raise AssertionError("scheduler stalled")
    for uid, prompt, limit in zip(uids, prompts, limits):
        assert out[uid] == _greedy(adapter.model, prompt, limit), prompt
    return batch


# ----------------------------------------------------------------- policy --

def test_cost_table_formula_and_preset():
    from mlx2.adapters import hy_v4_flash as A

    table = A.hy4_dfly_cost_table(5)
    # cost[d] = 26.3 + 4.36 d + 1.82 d: 1 row = the ordinary step, and the
    # measured 6-row round (59.1 ms) = cost[5] + 1.9 ms host.
    assert table[0] == pytest.approx(26.3)
    assert table[5] + A.HY4_DFLY_HOST_MS == pytest.approx(59.1)
    assert all(b - a == pytest.approx(4.36 + 1.82) for a, b in zip(table, table[1:]))
    _, external, kind = A._parse_policy({"draft_model": "embedded_dfly",
                                         "adaptive_verification": "hy4_measured_20261121"})
    assert kind == "dfly"
    assert external["adaptive_verification"]["verification_costs"] == table
    assert external["adaptive_verification"]["draft_cost"] == pytest.approx(1.9)
    assert external["_presets"]["adaptive_verification"] == "hy4_measured_20261121"
    # num_draft sizes the table (depth 0..num_draft).
    _, external, _ = A._parse_policy({"draft_model": "embedded_dfly", "num_draft": 3,
                                      "adaptive_verification": "hy4_measured_20261121"})
    assert len(external["adaptive_verification"]["verification_costs"]) == 4
    # Default: no adaptive depth, no preset receipt.
    _, external, _ = A._parse_policy({"draft_model": "embedded_dfly"})
    assert "adaptive_verification" not in external and "_presets" not in external


def test_policy_fails_closed():
    from mlx2.adapters import hy_v4_flash as A

    with pytest.raises(ValueError, match="preset"):
        A._parse_policy({"draft_model": "embedded_dfly", "adaptive_verification": "fast"})
    with pytest.raises(ValueError, match="DFly2"):
        A._parse_policy({"draft_model": "prompt_lookup", "adaptive_verification": ADAPTIVE})
    with pytest.raises(ValueError, match="chain"):
        A._parse_policy({"draft_model": "embedded_dfly", "ast_topology": "3x5",
                         "adaptive_verification": ADAPTIVE})
    with pytest.raises(ValueError):
        A._parse_policy({"draft_model": "embedded_dfly",
                         "adaptive_verification": {"verification_costs": [1, 2]}})
    with pytest.raises(ValueError, match="preset"):
        A._parse_policy({"draft_model": "embedded_dfly", "cost_per_commit_handoff": "slow"})
    # Not route-neutral: it never keeps an ordinary best-route default.
    assert "adaptive_verification" not in A.HYV4FlashAdapter.route_neutral_policy_keys
    assert A.HYV4FlashAdapter.prompt_lookup_default is False


# --------------------------------------------------------- adaptive depth --

def test_adaptive_depth_trims_and_stays_exact_b1(tmp_path, tiny):
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly",
                                        "adaptive_verification": ADAPTIVE,
                                        "cost_per_commit_handoff": False})
    try:
        batch = _run(adapter, 1, n=48)
        assert batch.adaptive_tree_chain
        d = adapter.diagnostics()["dfly2"]["adaptive_depth"]
        assert d["selected"] and d["engaged"] and d["observed_used"] and not d["qualified"]
        assert d["trimmed_rounds"] > 0 and d["trimmed_target_rows"] > 0
        # Full-depth refresh rounds keep the estimator's deep positions live.
        assert max(int(k) for k in d["chain_depths"]) == 5
        assert min(int(k) for k in d["chain_depths"]) == 1
    finally:
        adapter.close()


def test_adaptive_depth_receipt_per_request(tmp_path, tiny):
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly",
                                        "adaptive_verification": ADAPTIVE,
                                        "cost_per_commit_handoff": False})
    try:
        batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=4)
        (_uid,) = batch.insert([PROMPTS[0]], max_tokens=[40], sampling_configs=[{"sampling_temp": 0}])
        final = None
        for _ in range(500):
            _, responses = batch.next()
            final = next((r for r in responses if r.finish_reason), final)
            if final is not None:
                break
        receipt = final.speculative_receipt["adaptive_verification"]
        assert receipt["selected"] and receipt["observed_used"]
        assert receipt["policy"] == "lagged_tree_chain_cost_model"
        assert receipt["current_policy"] == "lagged_tree_chain"
        assert receipt["request_trimmed_rounds"] > 0
        assert receipt["request_rounds"] >= receipt["request_trimmed_rounds"]
    finally:
        adapter.close()


def test_adaptive_depth_flat_costs_never_trim(tmp_path, tiny):
    # A flat table never pays for a shorter chain: output and depth unchanged.
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly",
                                        "adaptive_verification": {**ADAPTIVE,
                                                                  "verification_costs": [1] * 6},
                                        "cost_per_commit_handoff": False})
    try:
        batch = _run(adapter, 1, n=32)  # noqa: F841 - diagnostics read the live batch
        d = adapter.diagnostics()["dfly2"]["adaptive_depth"]
        assert d["rounds"] > 0 and d["trimmed_rounds"] == 0 and not d["observed_used"]
    finally:
        adapter.close()


@pytest.mark.parametrize("width", [2, 3, 4])
def test_adaptive_depth_multilane_mixed_lengths_exact(tmp_path, tiny, width):
    # Per-width tables trim inside the cohort; ragged max_tokens finish lanes
    # mid-run, so cohorts shrink through narrower widths and their tables.
    by_cohort = {str(w): STEEP for w in range(1, 5)}
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly", "ast_tile_rows": 16,
                                        "adaptive_verification": {
                                            **ADAPTIVE, "verification_costs_by_cohort": by_cohort},
                                        "cost_per_commit_handoff": False})
    try:
        limits = [40, 13, 27, 21][:width]
        batch = _run(adapter, width, max_tokens=limits)
        stats = batch.scheduler_stats
        assert stats["external_target_tree_cohort_max_width"] >= 2
        assert stats["external_adaptive_trimmed_rounds"] > 0
    finally:
        adapter.close()


def test_adaptive_depth_without_table_keeps_tile_depth_at_width_two(tmp_path, tiny):
    # The B1 table does not price a two-lane round: depth stays the budget's.
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly", "ast_tile_rows": 16,
                                        "ast_adaptive_rows": False,
                                        "adaptive_verification": ADAPTIVE,
                                        "cost_per_commit_handoff": False})
    try:
        batch = adapter.create_external_batch(completion_batch_size=2, prefill_step_size=4)
        uids = batch.insert(PROMPTS[:2], max_tokens=[30, 30],
                            sampling_configs=[{"sampling_temp": 0}] * 2)
        out = _drain(batch, uids, 30)
        for uid, prompt in zip(uids, PROMPTS[:2]):
            assert out[uid][:30] == _greedy(adapter.model, prompt, 30)
        assert batch._adaptive_tree_decisions[2][0] == batch._adaptive_tree_decisions[2][1]
    finally:
        adapter.close()


# -------------------------------------------------------------- park latch --

_ROUND = ExternalDraftBatchGenerator._round
_READ = ExternalDraftBatchGenerator._read_parked


def _clocked(monkeypatch, speculative_ns):
    """Deterministic clock: plain steps 100 ns, tree rounds ``speculative_ns(i)``."""
    clock = {"now": 1, "spec": 0}

    def timed_round(self, cohort, *, forced_depth=None):
        result = _ROUND(self, cohort, forced_depth=forced_depth)
        if forced_depth == 0 or all(es._speculation_parked(lane) for lane in cohort):
            clock["now"] += 100
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
    return clock


HANDOFF = {"enabled": True, "reversible": True, "ordinary_samples": 2, "speculative_samples": 2,
           "loss_decisions": 1, "window": 4, "reprobe_tokens": 6, "reprobe_max_tokens": 12,
           "settle_rounds": 0, "deferred_context_cap": 64}


def _final_receipts(batch, uids):
    finals = {}
    out = {u: [] for u in uids}
    for _ in range(3000):
        _, responses = batch.next()
        for r in responses:
            out[r.uid].append(int(r.token))
            if r.finish_reason:
                finals[r.uid] = r
        if not batch.lanes:
            break
    else:
        raise AssertionError("scheduler stalled")
    return out, finals


@pytest.mark.parametrize("adaptive", [False, True])
def test_park_latch_forced_to_park_stays_exact(tmp_path, tiny, monkeypatch, adaptive):
    _clocked(monkeypatch, lambda _i: 100_000)  # speculation always loses
    policy = {"draft_model": "embedded_dfly", "cost_per_commit_handoff": HANDOFF}
    if adaptive:
        policy["adaptive_verification"] = ADAPTIVE
    adapter = _adapter(tmp_path, tiny, policy)
    try:
        batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=4)
        uids = batch.insert([PROMPTS[0]], max_tokens=[48], sampling_configs=[{"sampling_temp": 0}])
        out, finals = _final_receipts(batch, uids)
        assert out[uids[0]] == _greedy(adapter.model, PROMPTS[0], 48)
        reversible = finals[uids[0]].speculative_receipt["cost_per_commit_handoff"]["reversible"]
        assert reversible["parks"] >= 1 and reversible["reengagements"] == 0
        assert reversible["reprobes"] >= 1 and reversible["failed_reprobes"] >= 1
        park = adapter.diagnostics()["dfly2"]["park_latch"]
        assert park["selected"] and park["observed_used"] and not park["qualified"]
        assert park["authority"] == "explicit_policy"
        assert park["parked_rounds"] > 0 and park["reprobes"] >= 1
    finally:
        adapter.close()


@pytest.mark.parametrize("adaptive", [False, True])
def test_park_latch_parks_reprobes_and_resumes_exact(tmp_path, tiny, monkeypatch, adaptive):
    # Two losing tree rounds park the lane; the re-probe wins and resumes.
    clock = _clocked(monkeypatch, lambda i: 100_000 if i < 2 else 10)
    policy = {"draft_model": "embedded_dfly", "cost_per_commit_handoff": HANDOFF}
    if adaptive:
        policy["adaptive_verification"] = ADAPTIVE
    adapter = _adapter(tmp_path, tiny, policy)
    try:
        batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=4)
        uids = batch.insert([PROMPTS[0]], max_tokens=[48], sampling_configs=[{"sampling_temp": 0}])
        out, finals = _final_receipts(batch, uids)
        assert out[uids[0]] == _greedy(adapter.model, PROMPTS[0], 48)
        reversible = finals[uids[0]].speculative_receipt["cost_per_commit_handoff"]["reversible"]
        assert reversible["parks"] >= 1 and reversible["reprobes"] >= 1
        assert reversible["reengagements"] >= 1
        assert clock["spec"] > 2  # tree rounds ran again after the resume
        park = adapter.diagnostics()["dfly2"]["park_latch"]
        assert park["reengagements"] >= 1
        if adaptive:
            assert adapter.diagnostics()["dfly2"]["adaptive_depth"]["rounds"] > 0
    finally:
        adapter.close()


def test_park_latch_multilane_mixed_park_and_adaptive_exact(tmp_path, tiny, monkeypatch):
    # Three lanes, ragged lengths: the cohort parks, re-probes and resumes
    # while adaptive depth trims the resumed tree rounds.
    _clocked(monkeypatch, lambda i: 100_000 if i < 2 else 10)
    by_cohort = {str(w): STEEP for w in range(1, 5)}
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly", "ast_tile_rows": 16,
                                        "cost_per_commit_handoff": HANDOFF,
                                        "adaptive_verification": {
                                            **ADAPTIVE, "verification_costs_by_cohort": by_cohort}})
    try:
        batch = adapter.create_external_batch(completion_batch_size=3, prefill_step_size=4)
        limits = [44, 19, 31]
        uids = batch.insert(PROMPTS[:3], max_tokens=limits,
                            sampling_configs=[{"sampling_temp": 0}] * 3)
        out, _finals = _final_receipts(batch, uids)
        for uid, prompt, limit in zip(uids, PROMPTS[:3], limits):
            assert out[uid] == _greedy(adapter.model, prompt, limit)
        latches = adapter.diagnostics()["dfly2"]
        assert latches["park_latch"]["park_events"] >= 1
        assert latches["adaptive_depth"]["rounds"] > 0
    finally:
        adapter.close()


def test_park_preset_resolves_to_explicit_reversible_settings(tmp_path, tiny):
    from mlx2.adapters import hy_v4_flash as A

    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly",
                                        "cost_per_commit_handoff": "hy4_park_20261121"})
    try:
        batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=4)
        assert batch.cost_per_commit_handoff_authority == "explicit_policy"
        settings = batch.cost_per_commit_handoff.as_dict()
        for key, value in A.COST_HANDOFF_PRESETS["hy4_park_20261121"].items():
            assert settings[key] == value
        park = adapter.diagnostics()["dfly2"]["park_latch"]
        assert park["preset"] == "hy4_park_20261121" and park["selected"]
    finally:
        adapter.close()


def test_default_route_reports_runtime_park_latch(tmp_path, tiny):
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly"})
    try:
        batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=4)  # noqa: F841 - kept alive for diagnostics
        d = adapter.diagnostics()["dfly2"]
        # Unchanged default: the tree route's own reversible latch, no adaptive depth.
        assert d["park_latch"]["selected"]
        assert d["park_latch"]["authority"] == "runtime_tree_tile_default"
        assert not d["adaptive_depth"]["selected"] and not d["adaptive_depth"]["engaged"]
    finally:
        adapter.close()


# ---------------------------------------------------- latched prompt lookup --

def _copy_prompt(model):
    seed = LONG[30:36]
    return seed + _greedy(model, seed, 12) + seed


def _pld(model, prompts, n, policy, width=None):
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    gen = PromptLookupBatchGenerator(model, completion_batch_size=width or len(prompts),
                                     prefill_step_size=4, prompt_lookup=policy)
    limits = n if isinstance(n, list) else [n] * len(prompts)
    uids = gen.insert(prompts, max_tokens=limits, caches=[model.make_cache() for _ in prompts])
    out = {u: [] for u in uids}
    finals = {}
    for _ in range(3000):
        _, responses = gen.next()
        for r in responses:
            out[r.uid].append(int(r.token))
            if r.finish_reason:
                finals[r.uid] = r
        if len(finals) == len(uids):
            break
    else:
        raise AssertionError("prompt lookup stalled")
    for uid, prompt, limit in zip(uids, prompts, limits):
        assert out[uid] == _greedy(model, prompt, limit), prompt
    return gen, [finals[u] for u in uids]


def test_hy4_declares_prompt_lookup_but_not_by_default():
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.contracts import Capability

    assert Capability.PROMPT_LOOKUP in A.descriptor_for().capabilities
    assert Capability.PROMPT_LOOKUP in A.descriptor_for(dfly_selected=True).capabilities
    assert A.HYV4FlashAdapter.prompt_lookup_default is False


@pytest.mark.parametrize("num_draft", [2, 4, 6])
def test_prompt_lookup_accepts_rejects_and_rolls_back_exactly(tiny, num_draft):
    _, _, model, _ = tiny
    prompt = _copy_prompt(model)
    _gen, (final,) = _pld(model, [prompt], 30, {"num_draft": num_draft, "ngram_min": 2,
                                               "ngram_max": 4})
    receipt = final.speculative_receipt
    assert receipt["proposed"] > 0 and receipt["accepted"] > 0
    # Rejected drafts rolled the layer caches (and Engram tails) back.
    assert receipt["proposed"] > receipt["accepted"]


@pytest.mark.parametrize("batched_verify", [False, True])
def test_prompt_lookup_multilane_keeps_per_lane_rollback(tiny, batched_verify):
    from mlx2.runtime.pld import PromptLookupBatchGenerator

    _, _, model, _ = tiny
    prompts = [_copy_prompt(model), PROMPTS[2], PROMPTS[3]]
    # The Hy4 layer cache has no segmented batch view: lanes keep their own
    # rollback epoch at every width instead of failing the first B3 verify.
    assert not PromptLookupBatchGenerator._batchable(model.make_cache())
    _pld(model, prompts, [30, 17, 24], {"num_draft": 4, "ngram_min": 2, "ngram_max": 4,
                                        "batched_verify": batched_verify})


def test_latched_default_prompt_lookup_policy_is_exact(tiny):
    from mlx2.server import DEFAULT_PROMPT_LOOKUP_POLICY

    _, _, model, _ = tiny
    prompts = [_copy_prompt(model), PROMPTS[0], PROMPTS[3]]
    gen, finals = _pld(model, prompts, [40, 26, 33], dict(DEFAULT_PROMPT_LOOKUP_POLICY))
    stats = gen.scheduler_stats
    # The cost latch ran (shadow lookups, plain/parked rounds) and the
    # receipts say whether speculation was admitted.
    assert stats.get("pld_cost_shadow_lookups", 0) > 0
    assert all("proposed" in f.speculative_receipt for f in finals)
