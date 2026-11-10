"""DSpark verify expert ledger, survival_miss_v1 and the lockstep MoE rows site.

CPU only: synthetic provider counters, clocks, kernels and the synthetic V4.1
target; no checkpoint, no Metal.
"""
from types import SimpleNamespace
import time

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_dspark_chain as chain
from mlx2.adapters import deepseek_v41_dspark_chain_moe as rows_moe
from mlx2.adapters import deepseek_v41_dspark_schedule as schedule
from mlx2.adapters import deepseek_v41_dspark_verify_ledger as ledger_mod
from mlx2.adapters.deepseek_v4_external import DeepSeekV4DSparkDraftModel
from mlx2.adapters.deepseek_v41_dspark_external import DeepSeekV41DSparkDraftModel
from mlx2.adapters.deepseek_v41_dspark_policy import split_chain_policy, validate_chain_policy
from mlx2.adapters.deepseek_v41_source_math import ordered_offsets

GIB = 1 << 30
COSTS = list(schedule.DEFAULT_COSTS["lockstep_multirow_v1"])


class Clock:
    def __init__(self):
        self.ns = 1_000_000_000

    def advance(self, seconds):
        self.ns += int(seconds * 1e9)


class Hidden:
    def __init__(self, rows):
        self.shape = (1, rows, 12)


# ------------------------------------------------------------------ ledger --

def test_ledger_attributes_provider_deltas_to_the_closed_verify_width():
    stats = {"cache_hits": 0, "cache_misses": 0, "bytes_read": 0, "demand_wait_ns": 0}
    ledger = ledger_mod.VerifyExpertLedger(stats)
    assert ledger.close(1) is None                     # nothing opened yet (prefill uncounted)
    ledger.open(4)
    stats.update(cache_hits=30, cache_misses=6, bytes_read=6 * 100, demand_wait_ns=50)
    delta = ledger.close(4)
    assert delta == {"width": 4, "expert_acquisitions": 36, "expert_loads": 6, "miss_bytes": 600,
                     "demand_wait_ns": 50}
    ledger.open(1)
    stats.update(cache_hits=40, cache_misses=7, bytes_read=700)
    ledger.close(1)
    ledger.open(3)                                      # final round: never closed, not counted
    receipt = ledger.receipt()
    assert receipt["closed_rounds"] == 2 and receipt["verify_rows"] == 5
    assert receipt["expert_loads"] == 7 and receipt["expert_loads_per_verify"] == 3.5
    assert receipt["by_width"]["4"]["committed_tokens"] == 4
    with pytest.raises(ValueError):
        ledger.open(7)
    # A missing provider (resident build) counts rounds with zero loads.
    empty = ledger_mod.VerifyExpertLedger(None)
    empty.open(2)
    assert empty.close(2)["expert_loads"] == 0


def test_miss_priced_costs_add_measured_extra_row_reads_only_with_evidence(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(time, "perf_counter_ns", lambda: clock.ns)
    stats = {"cache_hits": 0, "cache_misses": 0, "bytes_read": 0, "demand_wait_ns": 0}
    ledger = ledger_mod.VerifyExpertLedger(stats, timed=True)
    costs, receipt = ledger_mod.miss_priced_costs(COSTS, ledger)
    assert costs == COSTS and not receipt["priced"]
    for _ in range(2):                                  # plain rounds: 0.1 s, no reads
        ledger.open(1)
        clock.advance(0.1)
        ledger.close(1)
    assert ledger.slope_bytes() is None                 # no width>1 evidence yet
    ledger.open(4)
    stats.update(bytes_read=3 * GIB, demand_wait_ns=int(0.3e9))   # 1 GiB/extra row at 10 GiB/s
    ledger.close(4)
    costs, receipt = ledger_mod.miss_priced_costs(COSTS, ledger)
    assert receipt["priced"] and receipt["bandwidth_source"] == "request"
    assert receipt["per_extra_row_cost"] == pytest.approx(1.0)       # 0.1 s read / 0.1 s plain round
    assert costs == pytest.approx([c + (w - 1) for w, c in enumerate(COSTS, 1)])
    # No read bandwidth evidence but positive slope stays unpriced.
    other = ledger_mod.VerifyExpertLedger({"bytes_read": 0}, timed=True)
    other.plain_round_s, other.ewma_bytes = 0.1, {1: 0.0, 2: 1000.0}
    other.by_width = {1: ledger_mod._empty_width(), 2: {**ledger_mod._empty_width(), "rounds": 1}}
    assert ledger_mod.miss_priced_costs(COSTS, other)[1]["reason"] == "read_bandwidth_unknown"
    with pytest.raises(ValueError):
        ledger_mod.miss_priced_costs([1, 0.5, 1, 1, 1, 1], ledger)


# --------------------------------------------------------- survival_miss_v1 --

def _draft(monkeypatch, schedule_policy, clock, stats):
    monkeypatch.setattr(time, "perf_counter_ns", lambda: clock.ns)
    monkeypatch.setattr(time, "perf_counter", lambda: clock.ns / 1e9)
    model = DeepSeekV41DSparkDraftModel.__new__(DeepSeekV41DSparkDraftModel)
    model.admission, model.admission_stats, model.schedule_stats = "off", {}, {}
    model.verify_schedule, model._verify_costs, model._draft_cost = schedule_policy, None, None
    target = SimpleNamespace(verify_rows_policy="lockstep_multirow_v1")
    model.backbone = SimpleNamespace(target=target, target_weights=SimpleNamespace(
        expert_stream=SimpleNamespace(stats=stats)))
    model.adaptive_confidence_features = [[8.0] * 5]

    def proposals(self, anchors, hidden, cache, length, rngs, temperatures, **kwargs):
        return [[7] * length], [[None] * length]
    monkeypatch.setattr(DeepSeekV4DSparkDraftModel, "draft_distributions", proposals)
    return model


def _rounds(model, clock, stats, count, *, miss_bytes_per_extra_row):
    plane, committed, widths = SimpleNamespace(), 1, []
    for _ in range(count):
        tokens, _ = model.draft_distributions([5], Hidden(committed), [plane], 5, [None], [0.0])
        width = len(tokens[0]) + 1
        widths.append(width)
        clock.advance(0.1 * (1 + 0.45 * (width - 1)))
        stats["bytes_read"] += miss_bytes_per_extra_row * (width - 1)
        stats["demand_wait_ns"] += int(miss_bytes_per_extra_row * (width - 1) / (10 * GIB) * 1e9)
        stats["cache_misses"] += 6 * (width - 1)
        committed = width
    return widths, plane


def test_survival_miss_probes_plain_then_stops_verifying_rows_that_miss(monkeypatch):
    clock, stats = Clock(), {"cache_hits": 0, "cache_misses": 0, "bytes_read": 0, "demand_wait_ns": 0}
    model = _draft(monkeypatch, "survival_miss_v1", clock, stats)
    widths, plane = _rounds(model, clock, stats, 8, miss_bytes_per_extra_row=2 * GIB)
    # Two width-one probes, one unpriced survival round prices the misses,
    # then 0.2 s of reads per extra row against a 0.1 s plain round wins plain.
    assert widths[:3] == [1, 1, 6] and set(widths[3:]) == {1}
    assert model.schedule_stats["miss_probe_rounds"] >= 2
    assert model.schedule_stats["miss_priced_rounds"] >= 1
    assert model.schedule_stats["last_miss_pricing"]["per_extra_row_cost"] == pytest.approx(2.0, rel=1e-6)
    observed = model.verify_expert_stats
    assert observed["rounds_by_width"]["6"] == 1 and observed["expert_loads_by_width"]["6"] == 30
    assert plane._mlx2_dspark_verify_ledger.receipt()["closed_rounds"] == 7


def test_survival_miss_reprobes_speculation_after_a_priced_out_stretch(monkeypatch):
    clock, stats = Clock(), {"cache_hits": 0, "cache_misses": 0, "bytes_read": 0, "demand_wait_ns": 0}
    model = _draft(monkeypatch, "survival_miss_v1", clock, stats)
    widths, _ = _rounds(model, clock, stats, 3 + ledger_mod.REPROBE_INTERVAL + 2, miss_bytes_per_extra_row=2 * GIB)
    assert widths[:3] == [1, 1, 6]
    assert set(widths[3:3 + ledger_mod.REPROBE_INTERVAL - 1]) == {1}
    assert 6 in widths[3 + ledger_mod.REPROBE_INTERVAL - 1:]
    assert model.schedule_stats["miss_speculative_probe_rounds"] == 1


def test_survival_miss_keeps_full_verify_when_rows_hit_resident_experts(monkeypatch):
    clock, stats = Clock(), {"cache_hits": 0, "cache_misses": 0, "bytes_read": 0, "demand_wait_ns": 0}
    model = _draft(monkeypatch, "survival_miss_v1", clock, stats)
    widths, _ = _rounds(model, clock, stats, 8, miss_bytes_per_extra_row=0)
    assert widths[:2] == [1, 1] and set(widths[2:]) == {6}
    # survival_v1 itself never probes; the ledger still counts every round.
    clock, stats = Clock(), {"cache_hits": 0, "cache_misses": 0, "bytes_read": 0, "demand_wait_ns": 0}
    model = _draft(monkeypatch, "survival_v1", clock, stats)
    widths, plane = _rounds(model, clock, stats, 4, miss_bytes_per_extra_row=2 * GIB)
    assert set(widths) == {6} and "miss_probe_rounds" not in model.schedule_stats
    assert plane._mlx2_dspark_verify_ledger.timed is False
    assert model.verify_expert_stats["rounds"] == 3


def test_survival_miss_chain_commits_ordinary_greedy_tokens():
    pytest.importorskip("greenlet")
    from test_deepseek_v41_dspark_chain_schedule_cpu import _ordinary, _speculative
    from test_deepseek_v41_embedded_dspark_cpu import pair
    target, backbone, _ = pair()
    prompt = [1, 2, 3, 4, 5, 6, 7]
    expected = _ordinary(target, prompt, 14)
    target.verification_commit_policy = "local_frontier"
    target.verify_rows_policy = "lockstep_multirow_v1"
    target.verified_commit_stats = {}
    draft = DeepSeekV41DSparkDraftModel(backbone, verify_schedule="survival_miss_v1").bind(target)
    assert _speculative(target, draft, prompt, 14) == expected
    assert draft.schedule_stats["miss_probe_rounds"] >= 1
    assert "miss_bytes_per_extra_row" in draft.receipt_settings["verify_schedule_law"]
    plain = DeepSeekV41DSparkDraftModel(backbone, verify_schedule="survival_v1").bind(target)
    assert plain.execution_binding != draft.execution_binding


# ------------------------------------------------------ lockstep MoE rows --

def test_lockstep_moe_site_parks_all_rows_once_per_layer_and_stays_exact():
    pytest.importorskip("greenlet")
    from test_deepseek_v41_dspark_chain_cpu import RecordingExecutor, _model, _state
    for moe_rows in (False, True):
        model = _model()
        cache = model.make_cache()
        model(mx.array([[1, 2, 3, 4, 5]], mx.uint32), cache)
        base = [s.snapshot() for s in cache]
        tokens = mx.array([[1, 2, 3, 4]], mx.uint32)
        reference = mx.concatenate([model(tokens[:, i:i + 1], cache) for i in range(4)], 1)
        expected_state = _state(cache)
        for state, saved in zip(cache, base):
            state.restore(saved)
        if moe_rows:
            model.verify_moe_rows_policy = rows_moe.LAW
        recorder = RecordingExecutor()
        logits, _, _ = chain.verify(model, tokens, cache, executor=recorder)
        np.testing.assert_array_equal(np.array(reference), np.array(logits))
        for (meta_a, arrays_a), (meta_b, arrays_b) in zip(expected_state, _state(cache)):
            assert meta_a == meta_b
            for a, b in zip(arrays_a, arrays_b):
                if a is not None:
                    np.testing.assert_array_equal(a, b)
        moe_groups = [count for key, count in recorder.groups if key[0] == "moe"]
        assert (len(moe_groups) == len(model.layers)) is moe_rows
        assert all(count == 4 for count in moe_groups)
        assert "moe" not in vars(model.weights)


class _Store:
    def __init__(self, experts):
        self.experts, self.acquired, self._active, self.stats = experts, [], False, {}

    def acquire(self, prefix, expert):
        self.acquired.append((prefix, expert))
        return self.experts[expert]

    def window(self):
        from contextlib import nullcontext
        return nullcontext()


def _fake_kernels():
    # Row-invariant M1 law, like the real occurrence-row kernels: each
    # occurrence row is its own one-row product.
    def middle(prepared, rows, scores, data):
        out = []
        for i, row in enumerate(np.array(rows).tolist()):
            x = prepared[row:row + 1]
            gate, up = x @ data["gate"].T, x @ data["up"].T
            out.append(gate * mx.sigmoid(gate) * up * scores[i:i + 1, None])
        return mx.concatenate(out, 0)

    def down(activation, data):
        return mx.concatenate([activation[i:i + 1] @ data["down"].T for i in range(activation.shape[0])], 0)

    def combine(parts, values, order):
        packed = np.array(mx.concatenate(parts, 0))
        offsets = ordered_offsets(values, order)
        out = np.zeros((values.shape[0], packed.shape[1]), np.float32)
        for row in range(values.shape[0]):
            for slot in range(values.shape[1]):
                out[row] = out[row] + packed[offsets[row, slot]]
        return mx.array(out)
    return SimpleNamespace(prepare=lambda x: x, middle=middle, down=down, combine=combine)


def _expert_weights(count=12, hidden=8, inner=5, seed=7):
    rng = np.random.default_rng(seed)
    return {e: {k: mx.array(rng.normal(0, .3, shape).astype(np.float32))
                for k, shape in (("gate", (inner, hidden)), ("up", (inner, hidden)), ("down", (hidden, inner)))}
            for e in range(count)}


def test_rows_moe_acquires_each_distinct_expert_once_and_matches_per_row_reference():
    experts = _expert_weights()
    store = _Store(experts)
    weights = SimpleNamespace(expert_math_policy="routed_fused_fp32", expert_stream=store,
                              source_resident_bank_session=None, _ready=lambda: None, scheduled_expert_calls={})
    rng = np.random.default_rng(3)
    routes = [[0, 3, 5, 7, 9, 11], [3, 5, 1, 2, 9, 4], [11, 0, 6, 8, 10, 3]]
    parks = []
    for route in routes:
        x = mx.array(rng.normal(0, 1, (1, 8)).astype(np.float32))
        scores = mx.array(rng.uniform(0.1, 1, (1, 6)).astype(np.float32))
        parks.append(chain._Park("moe", "blk.2", (x, mx.array([route], mx.uint32), scores, "blk.2"), {}, None))
    stats, kernels = {}, _fake_kernels()
    outputs = rows_moe.rows_moe(weights, parks, stats, kernels=kernels)
    distinct = len({e for route in routes for e in route})
    assert len(store.acquired) == distinct and len(set(store.acquired)) == distinct
    assert stats == {"lockstep_moe_calls": 1, "lockstep_moe_rows": 3, "lockstep_moe_occurrences": 18,
                     "lockstep_moe_expert_jobs": distinct, "lockstep_moe_acquisitions_saved": 18 - distinct}
    for park, out, route in zip(parks, outputs, routes):
        x, _, scores, _ = park.args
        total = np.zeros((1, 8), np.float32)
        for slot in np.argsort(route, kind="stable"):            # ascending expert order
            data = experts[route[slot]]
            mid = kernels.middle(x, mx.array([0], mx.uint32), scores[0, slot:slot + 1], data)
            total = total + np.array(kernels.down(mid, data))
        assert out.shape == (1, 8)
        np.testing.assert_array_equal(np.array(out), total)


def test_rows_moe_declines_structural_mismatches_and_executor_falls_back():
    store = _Store(_expert_weights())
    weights = SimpleNamespace(expert_math_policy="routed_fused_fp32", expert_stream=store,
                              source_resident_bank_session=None, _ready=lambda: None)
    x, s = mx.zeros((1, 8), mx.float32), mx.ones((1, 6), mx.float32)

    def park(route, *, xx=x, kwargs=None, prefix="blk.0"):
        return chain._Park("moe", prefix, (xx, mx.array([route], mx.uint32), s, prefix), kwargs or {}, None)
    good = [0, 1, 2, 3, 4, 5]
    for parks in ([park([0, 0, 1, 2, 3, 4])], [park([0, 1, 2, 3, 4, 400])],
                  [park(good, xx=x.astype(mx.bfloat16))], [park(good, kwargs={"ticket": 1})],
                  [park(good), park(good, prefix="blk.1")]):
        with pytest.raises(rows_moe.Decline):
            rows_moe.rows_moe(weights, parks, {}, kernels=_fake_kernels())
    assert store.acquired == []
    weights.expert_math_policy = "reference"
    assert "routed_fused_fp32" in rows_moe.refusal(weights)
    stats = {}
    executor = chain.SourceRowsExecutor(weights, stats)
    with pytest.raises(chain._Decline):
        executor._moe([park(good), park(good)])
    assert stats["lockstep_moe_declines"] == 1


# ------------------------------------------------------------------ policy --

def test_policies_are_opt_in_and_fail_closed():
    value = {"draft_model": "embedded_dspark", "deepseek_verify_rows": "lockstep_multirow_v1",
             "deepseek_verify_moe_rows": "m1_rows_v1", "deepseek_dspark_verify_schedule": "survival_miss_v1"}
    chained = split_chain_policy(value, embedded=True)
    assert chained["deepseek_verify_moe_rows"] == "m1_rows_v1"
    assert split_chain_policy({"deepseek_verify_moe_rows": "reference"}, embedded=False) == {}
    with pytest.raises(ValueError, match="m1_rows_v1"):
        split_chain_policy({"deepseek_verify_moe_rows": "m16"}, embedded=True)
    with pytest.raises(ValueError, match="lockstep"):
        validate_chain_policy({"deepseek_verify_moe_rows": "m1_rows_v1"})
    from mlx2.adapters.deepseek_v41_source import source_policy
    plain = source_policy({"draft_model": "embedded_dspark", "num_draft": 3})
    assert "deepseek_verify_moe_rows" not in plain and "deepseek_dspark_verify_schedule" not in plain
    selected = source_policy({"draft_model": "embedded_dspark", "num_draft": 5,
                              "deepseek_verify_commit": "local_frontier",
                              "deepseek_verify_rows": "lockstep_multirow_v1",
                              "deepseek_verify_moe_rows": "m1_rows_v1",
                              "deepseek_source_expert_math": "routed_fused_fp32",
                              "deepseek_dspark_verify_schedule": "survival_miss_v1"})
    assert selected["deepseek_verify_moe_rows"] == "m1_rows_v1"
    assert selected["deepseek_dspark_verify_schedule"] == "survival_miss_v1"
    with pytest.raises(ValueError, match="routed_fused_fp32"):
        source_policy({"draft_model": "embedded_dspark", "deepseek_verify_commit": "local_frontier",
                       "deepseek_verify_rows": "lockstep_multirow_v1", "deepseek_verify_moe_rows": "m1_rows_v1",
                       "deepseek_source_expert_math": "reference"})
    with pytest.raises(ValueError, match="survival"):
        schedule.validate("fixed", [1, 2, 3, 4, 5, 6])
