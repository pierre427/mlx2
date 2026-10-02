"""Bench/check loops must not hang on a lane the generator drops (sweep H4).

BatchGenerator drops a lane whose sampled row is non-finite and records it in
``take_lane_failures()`` without ever emitting a finish_reason for it.  A loop
of the form ``while len(done) < batch: gen.next()`` then spins forever.  Each
script below now polls through ``_drain_poll``, which raises on a dropped lane
and on IDLE_POLL_LIMIT polls without a response (the scripts/paired_direct_ab.py
rule).
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FIXED = (
    "bench_8bit_hc_fold",
    "bench_fn_batch_verify",
    "bench_qwen4_hc_decode",
    "bench_tf_longctx",
    "measure_fn_mtp_padding",
    "bench_qwen4_attn_rows",
    "bench_row_exact_verify",
    "check_mtp_row_exact",
)


@pytest.fixture(scope="module", autouse=True)
def _scripts_on_path():
    sys.path.insert(0, str(SCRIPTS))
    yield
    sys.path.remove(str(SCRIPTS))


def _raw_next_calls_in_loops(source):
    """``<x>.next()`` calls made directly inside a while loop."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.While):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr == "next" and not inner.args):
                found.append(inner.lineno)
    return found


@pytest.mark.parametrize("name", FIXED)
def test_generator_loops_poll_through_the_drain_guard(name):
    source = (SCRIPTS / f"{name}.py").read_text()
    assert _raw_next_calls_in_loops(source) == [], (
        f"{name}: a while loop calls gen.next() without the dropped-lane guard"
    )
    module = importlib.import_module(name)
    assert module.IDLE_POLL_LIMIT == 4096


class _DroppingGen:
    """Two lanes; lane 1 is dropped after one token and never finishes."""

    def __init__(self):
        self.polls = 0
        self._failures = []

    def next(self):
        self.polls += 1
        if self.polls == 2:
            self._failures.append({"uid": 1, "reason": "non-finite logits"})
        return None, [type("R", (), {"uid": 0, "token": 1, "finish_reason": None})()]

    def take_lane_failures(self):
        failures, self._failures = self._failures, []
        return failures


class _SilentGen:
    def next(self):
        return None, []

    def take_lane_failures(self):
        return []


@pytest.mark.parametrize("name", FIXED)
def test_drain_poll_raises_on_dropped_lane_and_on_silence(name):
    module = importlib.import_module(name)
    gen, idle = _DroppingGen(), 0
    with pytest.raises(RuntimeError, match="dropped lane"):
        for _ in range(10):
            _responses, idle = module._drain_poll(gen, idle)
    with pytest.raises(RuntimeError, match="no response"):
        silent, idle = _SilentGen(), 0
        for _ in range(module.IDLE_POLL_LIMIT + 2):
            _responses, idle = module._drain_poll(silent, idle)


def test_real_generator_dropped_lane_raises_instead_of_hanging():
    """The repro: a tiny ordinary model whose lane 1 logits turn NaN."""
    mx = pytest.importorskip("mlx.core")
    from paired_direct_ab import tiny_ordinary
    from mlx2.runtime import generate as G
    from mlx2.runtime.sample_utils import LaneRNG

    module = importlib.import_module("bench_qwen4_hc_decode")
    model = tiny_ordinary()
    original = type(model).__call__
    calls = {"n": 0}

    class Poison(type(model)):
        def __call__(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            calls["n"] += 1
            if calls["n"] > 3 and out.shape[0] == 2:
                out = mx.concatenate(
                    [out[:1], mx.full(out[1:].shape, float("nan"), out.dtype)], axis=0
                )
            return out

    model.__class__ = Poison
    gen = G.BatchGenerator(model, completion_batch_size=2, prefill_batch_size=1,
                           prefill_step_size=64)
    gen.insert([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], max_tokens=[16, 16],
               lane_rngs=[LaneRNG(1), LaneRNG(2)])
    done, idle, polls = set(), 0, 0
    try:
        with pytest.raises(RuntimeError, match="dropped lane"):
            while len(done) < 2:
                polls += 1
                assert polls < 5000, "loop hung on the dropped lane"
                responses, idle = module._drain_poll(gen, idle)
                for response in responses:
                    if response.finish_reason:
                        done.add(response.uid)
    finally:
        gen.close()
