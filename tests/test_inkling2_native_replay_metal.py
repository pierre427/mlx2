"""``inkling2_native_replay`` on Metal (GPU; run under the GPU lease only).

Checkpoint: synthetic tiny Inkling 2 (``test_inkling2_cpu.TINY``) in BF16,
plus the Mini-shaped attention visit. The fixed-capacity step on the GPU:

* same-state steps (each step from a copy of the ordinary arm's state, so
  every step is independent and every flip classified by the AGENTS.md
  near-tie rule, ``mlx2.runtime.replay.near_tie``) against the reference
  route (gated) and the default route (recorded);
* free-running drift against the reference route, first divergence only;
* a control: default ordinary vs reference ordinary, same state (recorded);
* one attention visit per kind at tiny (head_dim 32) and Mini (head_dim 128)
  shapes, identical inputs, fixed view vs ordinary cache, which localizes a
  gap to the attention kernels;
* pure against fused form;
* the engine declines every call as ``module_missing`` (no module is
  installed by this test) and the MLX step serves;
* when ``MLX2_STEP_REPLAY_SELFTEST=1`` and a staged module exists, the
  module self-test (hooks, census, recordings) runs in a subprocess.

Set ``INKLING2_NATIVE_REPLAY_METAL_RECEIPT`` to a path for a JSON summary.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")

RESULTS = {}
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def gpu(monkeypatch, tmp_path):
    from mlx2.runtime.replay import engine, loader

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    monkeypatch.setenv(loader.DIR_ENV, str(tmp_path / "no-module"))
    monkeypatch.setattr(engine, "_DEFAULT", None)
    monkeypatch.setattr(loader, "_result", None)
    try:
        yield
    finally:
        mx.clear_cache()
        mx.set_default_device(previous)


def teardown_module(_module):
    path = os.environ.get("INKLING2_NATIVE_REPLAY_METAL_RECEIPT")
    if path:
        with open(path, "w") as handle:
            json.dump(RESULTS, handle, indent=1, sort_keys=True, default=str)


def _bf16_model(seed=0):
    from test_inkling2_cpu import TINY

    from mlx2.runtime.models.inkling2 import Model, TextConfig

    mx.random.seed(seed)
    model = Model(TextConfig.from_dict(TINY))
    weights = []
    for key, value in tree_flatten(model.parameters()):
        noise = mx.random.normal(value.shape) * (0.05 if value.ndim > 1 else 0.3)
        weights.append((key, (noise + (1.0 if key.endswith("norm.weight") else 0.0))
                        .astype(mx.bfloat16)))
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


def _defaults(model, **extra):
    from mlx2.adapters.inkling2 import model_kernel_options, validate_policy

    policy, _ = validate_policy(extra)
    policy = model_kernel_options(policy)
    policy["moe_nax_gather"] = False
    model.configure_kernels(**policy)


NEAR_TIE_NATS = 0.5  # AGENTS.md near-tie ceiling


MINI = dict(hidden_size=2304, num_attention_heads=24, num_key_value_heads=8, head_dim=128,
            swa_num_attention_heads=24, swa_num_key_value_heads=8, swa_head_dim=128,
            sliding_window_size=1024, rel_extent=1024, d_rel=16)
# One-step BF16 noise bound for the same-state comparison against the
# reference route (CPU BF16 measured 0.01; the logit span is about 4.5).
SAME_STATE_BOUND = 0.25


def _tokens():
    return mx.random.randint(0, 500, (1, 80), key=mx.random.key(5))


@pytest.fixture
def small_steps(monkeypatch):
    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.inkling2 import capacity_state as cs

    monkeypatch.setattr(KVCache, "step", 16)
    monkeypatch.setattr(cs, "CAPACITY_STEP", 32)


def _arm(policy):
    model = _bf16_model()
    if policy == "defaults":
        _defaults(model)
    else:
        model.configure_kernels()  # the reference route: every switch off
    return model


@pytest.mark.parametrize("form", ["pure", "fused"])
@pytest.mark.parametrize("policy", ["reference", "defaults"])
def test_same_state_steps_on_gpu(policy, form, small_steps):
    """Every step of the fixed step starts from a copy of the ordinary arm's
    state: each step is an independent shared-prefix comparison, so every
    flip is classified. Against the reference route (the math the fixed step
    implements) the gap must stay at one-step BF16 noise and every flip must
    be a classified near tie. Against the default route the result is
    recorded only: it measures the default GPU kernels (split-K, fused short
    conv) on this geometry as much as the fixed step."""
    from _native_replay_probe import same_state_run

    out = same_state_run(_arm(policy), _tokens(), 9, form)
    RESULTS[f"same_state_{policy}_{form}"] = {"checkpoint": "synthetic", **out}
    assert out["site"]["counts"]["fixed_step"] == 71
    if policy == "reference":
        assert out["worst_abs_logit"] < SAME_STATE_BOUND
        assert [f for f in out["flips"] if f["verdict"] != "near_tie_equivalent"] == []


@pytest.mark.parametrize("form", ["pure", "fused"])
def test_drift_first_divergence_on_gpu(form, small_steps):
    """Both arms free-running from one prompt (teacher-forced tokens, each its
    own state): only the first divergence is classified (AGENTS.md); later
    flips describe downstream consequence and are not counted."""
    from _native_replay_probe import drift_run

    out = drift_run(_arm("reference"), _tokens(), 9, form)
    RESULTS[f"drift_reference_{form}"] = {"checkpoint": "synthetic", **out}
    first = out["first_divergence"]
    assert first is None or first["verdict"] == "near_tie_equivalent"


def test_default_route_control_same_state_on_gpu(small_steps):
    """Control without the fixed step: the default ordinary route against the
    reference ordinary route, same state per step. Recorded, not gated: it
    says how far the default GPU kernels themselves sit from the reference
    on this geometry, which bounds what the fixed step can match."""
    import copy

    from _native_replay_probe import _logp

    from mlx2.runtime.replay import near_tie

    default, reference = _arm("defaults"), _arm("reference")
    tokens = _tokens()
    cache = default.make_cache()
    default(tokens[:, :9], cache=cache)
    worst, flips = 0.0, []
    for i in range(9, tokens.shape[1]):
        side = copy.deepcopy(cache)
        a = reference(tokens[:, i:i + 1], cache=side)
        b = default(tokens[:, i:i + 1], cache=cache)
        mx.eval(a, b)
        worst = max(worst, mx.abs(a - b).max().item())
        if mx.argmax(a).item() != mx.argmax(b).item():
            flips.append({"step": i, **near_tie.classify(_logp(a), _logp(b))})
    RESULTS["control_default_vs_reference_same_state"] = {
        "checkpoint": "synthetic", "worst_abs_logit": worst, "flips": flips}


@pytest.mark.parametrize("shape", ["tiny", "mini"])
@pytest.mark.parametrize("layer", [0, 5])
@pytest.mark.parametrize("policy", ["reference", "defaults"])
def test_visit_attention_probe_on_gpu(shape, layer, policy):
    """One attention visit, identical BF16 inputs and state: the fixed view
    against the ordinary cache. tiny (head_dim 32): the fixed step's SDPA has
    no vector kernel for D=32 and runs MLX's composed fallback; the ordinary
    default route runs split-K. mini (head_dim 128, window 1024, 1500 rows):
    the fixed step runs MLX's sdpa_vector kernels (float mask + sinks). The
    reference arm is gated at BF16 output noise; the default arm is recorded."""
    from _native_replay_probe import attention, visit_probe

    over, pos = ({}, 30) if shape == "tiny" else (MINI, 1500)
    attn = attention(over, layer, mx.bfloat16)
    out = visit_probe(attn, pos, dtype=mx.bfloat16, policy=policy)
    RESULTS[f"visit_{shape}_{'sliding' if layer == 0 else 'global'}_{policy}"] = out
    if policy == "reference":
        assert out["max_abs_gap"] <= 0.02 * max(1.0, out["max_abs_output"])


def test_pure_and_fused_agree_on_gpu(monkeypatch):
    from mlx2.runtime.models.inkling2 import capacity_state as cs

    monkeypatch.setattr(cs, "CAPACITY_STEP", 32)
    model = _bf16_model()
    _defaults(model)
    cache = cs.synthetic_cache(model, 40)
    kinds, reason = cs.plan(model, cache)
    assert reason is None
    cs.normalize(model, cache, kinds, {})
    arrays, _ = cs.round_inputs(model, mx.array([[7]], mx.int32), cache, kinds)
    pure = cs.make_step(model, kinds, "pure")(arrays)[0]
    fused = cs.make_step(model, kinds, "fused")(arrays)[0]
    gap = mx.abs(pure - fused).max().item()
    RESULTS["pure_vs_fused"] = gap
    assert gap < 0.5


def test_step_function_is_deterministic_on_gpu(monkeypatch):
    """The engine's bit check compares a replay with the MLX step; the MLX
    step itself must be repeatable on the same inputs."""
    from mlx2.runtime.models.inkling2 import capacity_state as cs
    from mlx2.runtime.replay import bitwise_equal

    model = _bf16_model()
    _defaults(model, inkling2_native_replay=True)
    cache = cs.synthetic_cache(model, 30)
    kinds, _ = cs.plan(model, cache)
    cs.normalize(model, cache, kinds, {})
    arrays, _ = cs.round_inputs(model, mx.array([[3]], mx.int32), cache, kinds)
    step = cs.make_step(model, kinds)
    first = step(arrays)
    mx.eval(first)
    for _ in range(3):
        again = step(arrays)
        mx.eval(again)
        assert all(bitwise_equal(x, y) for x, y in zip(first, again))


def test_module_selftest_when_requested(tmp_path):
    staged = sorted((ROOT / "build" / "step_replay" / "stage").glob("_step_replay*.so"))
    if os.environ.get("MLX2_STEP_REPLAY_SELFTEST") != "1" or not staged:
        pytest.skip("set MLX2_STEP_REPLAY_SELFTEST=1 with a staged module")
    out = tmp_path / "selftest.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "step_replay_selftest.py"), "--out", str(out)]
    done = subprocess.run(cmd, text=True, capture_output=True, timeout=600, check=False,
                          env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    receipt = json.loads(out.read_text()) if out.exists() else {"stderr": done.stderr[-4000:]}
    RESULTS["selftest"] = receipt
    assert done.returncode == 0, done.stderr[-4000:]
    assert receipt["hooks_and_recording_ok"] is True
