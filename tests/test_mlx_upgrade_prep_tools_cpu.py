"""Host-only checks of the MLX-upgrade prep tools (bench arm split, compile counter)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bench():
    spec = importlib.util.spec_from_file_location(
        "bench_q8a8_prefill_e2e", ROOT / "scripts" / "bench_q8a8_prefill_e2e.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arm_split_pins_every_arm_once_any_names_the_key():
    bench = _bench()
    arms = {"off": "off",
            "ours": {"enabled": True, "scope": "all", "mlx_int8_qmm": 0},
            "mlx": {"mlx_int8_qmm": 1}}
    policies, flags = bench.split_mlx_int8_qmm_arms(arms)
    assert policies == {"off": "off", "ours": {"enabled": True, "scope": "all"}, "mlx": "off"}
    assert flags == {"off": 0, "ours": 0, "mlx": 1}
    assert arms["mlx"] == {"mlx_int8_qmm": 1}  # the caller's mapping is not mutated


def test_arm_split_without_key():
    bench = _bench()
    arms = {"off": "off", "ours": "all"}
    assert bench.split_mlx_int8_qmm_arms(arms) == (arms, {})
    # On an MLX with #4802 (flag default on) every arm is pinned to 0.
    assert bench.split_mlx_int8_qmm_arms(arms, pin_default=True) == (arms, {"off": 0, "ours": 0})


def test_compile_counter_counts_without_changing_results(tmp_path):
    out = tmp_path / "counts.json"
    code = """
import mlx.core as mx
from functools import partial
mx.set_default_device(mx.cpu)
@partial(mx.compile, shapeless=True)
def f(x):
    return mx.exp(x) * x + 1
g = mx.compile(lambda x: x * 2)
class Holder:
    h = mx.compile(lambda x: x - 1)
x = mx.arange(4, dtype=mx.float32)
for _ in range(3):
    a, b, c = f(x), g(x), Holder.h(x)
ref = mx.exp(x) * x + 1
assert mx.allclose(a, ref).item() and mx.array_equal(b, x * 2).item()
assert mx.array_equal(c, x - 1).item() and mx.array_equal(Holder().h(x), x - 1).item()
"""
    env = dict(os.environ, MLX2_COMPILE_COUNTS_OUT=str(out),
               PYTHONPATH=str(ROOT / "scripts" / "research" / "compile_call_counter"))
    subprocess.run([sys.executable, "-c", code], check=True, env=env, timeout=120)
    sites = json.loads(out.read_text())["sites"]
    assert sites["__main__:f"] == {"compiled": 1, "calls": 3}
    lambdas = {k: v for k, v in sites.items() if "<lambda>" in k}
    assert len(lambdas) == 2 and sorted(v["calls"] for v in lambdas.values()) == [3, 4]
