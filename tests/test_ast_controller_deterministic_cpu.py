"""Deterministic AST controller mode (MLX2_AST_CONTROLLER_MODE=deterministic) on CPU.

The measured tile budget and cost-per-commit handoff decide verify widths from wall-clock time, so the same token
prefix could verify at different widths run to run (2026-11-23, Hy4 DFly2 cold replay drift). In deterministic mode
the same acceptance history gives the same decisions whatever the clock says:

* the tile budget's "model" cost mode chooses identical sizes under different injected round seconds (and the
  measured "ewma" mode does not, so the test can see a timing dependence);
* end to end on the tiny Hy4 DFly2 route, two engines with differently jittered clocks, each serving the same
  request twice in a row, verify the same width sequence for every request (the controller state resets on an idle
  engine), and the greedy tokens equal the ordinary decode;
* the mode refuses a missing or malformed cost table.
"""

import json
import random
import time as _time

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime import ast_tile_budget as T  # noqa: E402
from mlx2.runtime import external_speculative as ES  # noqa: E402
from test_hy_v4_flash_ast_multilane_cpu import PROMPTS, _adapter, _drain, _greedy, tiny  # noqa: E402,F401

COST = {"a_ms": 22.0, "b_ms": 6.18, "ordinary_ms": 26.3}
NEVER_PARK = {"a_ms": 5.0, "b_ms": 2.0, "ordinary_ms": 400.0}   # speculation always wins: the tile budget decides


def _rounds(controller, seconds, rounds=120, maximum=15):
    """Drive one width-1 lane: a fixed acceptance pattern, the given round seconds."""
    chosen = []
    for r in range(rounds):
        n = controller.choose(1, maximum)
        chosen.append(n)
        ranks = [i for i in range(min(n, 2 + (r * 7) % 5))]          # deterministic acceptance pattern
        controller.observe(1, n, ranks, seconds(r, n))
    return chosen


def test_model_cost_mode_ignores_measured_time():
    model = lambda: T.TileBudgetController(cost_mode="model", cost_model=(0.022, 0.00618))  # noqa: E731
    rng_a, rng_b = random.Random(1), random.Random(2)
    a = _rounds(model(), lambda r, n: 0.02 + 0.006 * n * rng_a.uniform(0.5, 1.5))
    b = _rounds(model(), lambda r, n: 0.05 + 0.001 * n * rng_b.uniform(0.2, 3.0))
    assert a == b and len(set(a)) > 1                    # same choices, and the controller does choose
    ewma = lambda: T.TileBudgetController(cost_mode="ewma")  # noqa: E731
    c = _rounds(ewma(), lambda r, n: 0.022 + 0.00618 * (n + 1))
    d = _rounds(ewma(), lambda r, n: 0.05 + 0.0001 * (n + 1))  # flat cost: always the maximum
    assert c != d                                        # the measured mode does depend on time
    ctl = model()
    _rounds(ctl, lambda r, n: 0.1)
    ctl.reset()
    assert ctl._widths == {} and ctl.decisions == {}
    with pytest.raises(ValueError, match="cost_model"):
        T.TileBudgetController(cost_mode="model")
    with pytest.raises(ValueError, match="cost_model"):
        T.TileBudgetController(cost_mode="ewma", cost_model=(0.02, 0.006))


class _JitterClock:
    """A monotonic clock that advances by a seeded random amount per read (the engine's ``time`` module)."""

    def __init__(self, seed):
        self.rng, self.t = random.Random(seed), 1000.0

    def _tick(self):
        self.t += self.rng.uniform(0.0005, 0.05)
        return self.t

    def perf_counter(self):
        return self._tick()

    def perf_counter_ns(self):
        return int(self._tick() * 1e9)

    def monotonic(self):
        return self._tick()

    def time(self):
        return _time.time()

    def sleep(self, s):
        return None


def _widths_per_request(tmp_path, tiny, monkeypatch, seed, n=40, num_draft=3):
    trace = tmp_path / f"trace-{seed}.jsonl"
    monkeypatch.setattr(ES, "_VERIFY_TRACE_PATH", str(trace))
    monkeypatch.setattr(ES, "_VERIFY_TRACE_STATE", {"forward": 0, "handle": None})
    monkeypatch.setattr(ES, "time", _JitterClock(seed))
    (tmp_path / f"a{seed}").mkdir(exist_ok=True)
    adapter = _adapter(tmp_path / f"a{seed}", tiny, {"draft_model": "embedded_dfly", "num_draft": num_draft})
    try:
        batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=4)
        per_request, tokens = [], []
        for _ in range(2):                                # the same request twice on one engine
            before = sum(1 for _ in open(trace)) if trace.exists() else 0
            uids = batch.insert([PROMPTS[0]], max_tokens=[n], sampling_configs=[{"sampling_temp": 0}])
            out = _drain(batch, uids, n)
            tokens.append(out[uids[0]][:n])
            ES._VERIFY_TRACE_STATE["handle"].flush()
            rows = [json.loads(line) for line in open(trace)][before:]
            per_request.append([len(r["inputs"]) for r in rows])
        engine = getattr(batch, "engine", batch)
        stats = dict(getattr(engine, "scheduler_stats", None) or getattr(batch, "scheduler_stats", {}) or {})
        return per_request, tokens, adapter, stats
    finally:
        h = ES._VERIFY_TRACE_STATE.get("handle")
        if h is not None:
            h.close()


def test_deterministic_mode_same_prefix_same_widths_under_jittered_clocks(tmp_path, tiny, monkeypatch):
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_COST", json.dumps(COST))
    w1, t1, ad1, _ = _widths_per_request(tmp_path, tiny, monkeypatch, seed=11)
    w2, t2, ad2, _ = _widths_per_request(tmp_path, tiny, monkeypatch, seed=97)
    try:
        assert w1[0] and w1[0] == w1[1] == w2[0] == w2[1], (w1, w2)   # same widths: across clocks and requests
        assert t1[0] == t1[1] == t2[0] == t2[1] == _greedy(ad1.model, PROMPTS[0], 40)
    finally:
        ad1.close()
        ad2.close()


def test_controller_settings_refuse_a_missing_or_bad_table(monkeypatch):
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
    monkeypatch.delenv("MLX2_AST_CONTROLLER_COST", raising=False)
    with pytest.raises(ValueError, match="needs MLX2_AST_CONTROLLER_COST"):
        ES.controller_settings()
    monkeypatch.setenv("MLX2_AST_CONTROLLER_COST", json.dumps({"a_ms": 0, "b_ms": 1, "ordinary_ms": 1}))
    with pytest.raises(ValueError, match="a_ms > 0"):
        ES.controller_settings()
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "sometimes")
    with pytest.raises(ValueError, match="must be one of"):
        ES.controller_settings()
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "measured")
    assert ES.controller_settings() == ("measured", None)


def test_measured_mode_is_clock_sensitive_on_the_same_route(tmp_path, tiny, monkeypatch):
    """Sensitivity check for the test above: the same seeded clocks change the measured mode's widths."""
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "measured")
    w1, _, ad1, _ = _widths_per_request(tmp_path, tiny, monkeypatch, seed=11)
    w2, _, ad2, _ = _widths_per_request(tmp_path, tiny, monkeypatch, seed=97)
    try:
        assert w1[0] != w1[1] or w1 != w2, (w1, w2)
    finally:
        ad1.close()
        ad2.close()


def test_deterministic_mode_drives_the_tile_budget_without_the_clock(tmp_path, tiny, monkeypatch):
    """With speculation always cheaper than the ordinary step, widths come from the tile budget's choices; they are
    the same under two different clocks and for both requests, and the mode's own receipts show it engaged."""
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_COST", json.dumps(NEVER_PARK))
    w1, t1, ad1, s1 = _widths_per_request(tmp_path, tiny, monkeypatch, seed=5, n=120, num_draft=7)
    w2, t2, ad2, s2 = _widths_per_request(tmp_path, tiny, monkeypatch, seed=61, n=120, num_draft=7)
    try:
        assert w1[0] == w1[1] == w2[0] == w2[1], (w1, w2)
        assert t1[0] == t2[0] == _greedy(ad1.model, PROMPTS[0], 120)
        for stats in (s1, s2):
            assert stats.get("external_ast_controller_mode") == "deterministic"
            assert stats.get("external_ast_controller_resets") == 2          # one per request on an idle engine
            assert stats.get("external_ast_controller_modelled_rounds", 0) > 0
            budget = (stats.get("external_tree_tile_budget") or {}).get("1", {})
            assert budget.get("cost_mode") == "model" and budget.get("rounds", 0) > 0
        assert len(set(w1[0])) > 1, w1[0]                                   # the controller does vary the width
    finally:
        ad1.close()
        ad2.close()
