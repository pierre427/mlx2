"""``inkling2_native_replay`` on Metal (GPU; run under the GPU lease only).

Checkpoint: synthetic tiny Inkling 2 (``test_inkling2_cpu.TINY``) in BF16,
plus the Mini-shaped attention visit. The fixed-capacity step on the GPU:

* against the ordinary decode with the default kernel set, teacher-forced
  (near-exact: the step reads masked columns). Every argmax flip is
  classified by the AGENTS.md near-tie rule from both arms' full
  log-probability rows at that step; any unclassified flip fails;
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
    from mlx2.adapters.inkling2 import validate_policy

    policy, _ = validate_policy(extra)
    for name in ("hard_no_thinking", "dspark_prefill_reclaim", "dspark_predraft"):
        policy.pop(name)
    policy["moe_nax_gather"] = False
    model.configure_kernels(**policy)


NEAR_TIE_NATS = 0.5  # AGENTS.md near-tie ceiling


def _exact_tie(logp_tied, tied_margin, other_selected, other_margin):
    """AGENTS.md exact-tie clause, bound to the arm whose margin is zero: the
    other arm selected one of that arm's tied tokens, with its own margin at
    most the ceiling."""
    if tied_margin != 0.0 or other_margin > NEAR_TIE_NATS:
        return False
    top = max(logp_tied)
    tied = {t for t in range(logp_tied.size) if logp_tied[t] == top}
    return other_selected in tied


def classify_flip(logp_ref, logp_fixed):
    """AGENTS.md near-tie classification of one argmax flip. The comparison
    is teacher-forced (both arms read the same token every step), so the
    prefix is shared and no continuation diverges; the functional oracle is
    not applicable to synthetic weights. Both arms' full log-probability rows
    are the evidence window, so no token is ever missing from it."""
    ref = sorted(range(logp_ref.size), key=lambda t: -logp_ref[t])[:2]
    fix = sorted(range(logp_fixed.size), key=lambda t: -logp_fixed[t])[:2]
    margins = (logp_ref[ref[0]] - logp_ref[ref[1]], logp_fixed[fix[0]] - logp_fixed[fix[1]])
    row = {"ref_top2": ref, "fixed_top2": fix, "margins": margins,
           "ref_logp": [logp_ref[t] for t in ref], "fixed_logp": [logp_fixed[t] for t in fix]}
    if set(ref) == set(fix) and max(margins) <= NEAR_TIE_NATS:
        row["verdict"] = "near_tie_equivalent"
    elif _exact_tie(logp_ref, margins[0], fix[0], margins[1]) or _exact_tie(
            logp_fixed, margins[1], ref[0], margins[0]):
        row["verdict"] = "near_tie_equivalent"  # exact-tie clause
    elif (logp_ref[ref[0]] - logp_ref[fix[0]] <= NEAR_TIE_NATS
          and logp_fixed[fix[0]] - logp_fixed[ref[0]] <= NEAR_TIE_NATS):
        row["verdict"] = "near_tie_equivalent"  # multi-way clause
    else:
        row["verdict"] = "unclassified"
    return row


@pytest.mark.parametrize("form", ["pure", "fused"])
def test_fixed_step_tracks_default_route_on_gpu(form, monkeypatch):
    """Every step's argmax agrees, or the flip is a classified near tie;
    any unclassified flip fails."""
    import numpy as np

    from mlx2.runtime.models.cache import KVCache
    from mlx2.runtime.models.inkling2 import capacity_state as cs

    monkeypatch.setattr(KVCache, "step", 16)
    monkeypatch.setattr(cs, "CAPACITY_STEP", 32)
    model = _bf16_model()
    _defaults(model)
    tokens = mx.random.randint(0, 500, (1, 80), key=mx.random.key(5))
    ref, fixed = model.make_cache(), model.make_cache()
    model(tokens[:, :9], cache=ref)
    model(tokens[:, :9], cache=fixed)
    site = cs.install(model, form=form)
    worst, flips = 0.0, []
    for i in range(9, 80):
        model.__dict__.pop("_native_replay")
        a = model(tokens[:, i:i + 1], cache=ref)
        object.__setattr__(model, "_native_replay", site)
        b = model(tokens[:, i:i + 1], cache=fixed)
        mx.eval(a, b)
        worst = max(worst, mx.abs(a - b).max().item())
        if mx.argmax(a).item() != mx.argmax(b).item():
            la = a[0, -1].astype(mx.float32)
            lb = b[0, -1].astype(mx.float32)
            la, lb = la - mx.logsumexp(la), lb - mx.logsumexp(lb)
            flips.append({"step": i, **classify_flip(np.array(la), np.array(lb))})
    receipt = cs.receipt(model)
    RESULTS[f"default_route_{form}"] = {"checkpoint": "synthetic", "worst_abs_logit": worst,
                                        "flips": flips, "steps": 71, "receipt": receipt}
    assert receipt["counts"]["fixed_step"] == 71
    assert receipt["engine"]["declines"] == {"module_missing": 71}
    assert [f for f in flips if f["verdict"] != "near_tie_equivalent"] == []


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
