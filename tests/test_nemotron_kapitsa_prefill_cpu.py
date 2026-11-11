"""CPU wiring for the opt-in Kapitsa-derived Nemotron prefill kernels.

``MLX2_NEMOTRON_SSD_MMA`` routes ``ssm.ssd_prefill`` segments to
``nemotron_ssd_mma.ssd_seg_mma``; ``MLX2_NEMOTRON_ATTN_STAGED`` routes
admitted Nemotron-H prefill attention to ``nemotron_attn_staged``.  Both are
off by default, recorded as serving switches and bound into the APCv2 key.
Numerics are gated in test_nemotron_kapitsa_prefill_metal.py.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx2.runtime import apc_numerics, env_switches
from mlx2.runtime.models import nemotron_attn_staged as staged
from mlx2.runtime.models import nemotron_ssd_mma as mma
from mlx2.runtime.models import ssm

SWITCHES = (mma.SSD_MMA_ENV, staged.ATTN_STAGED_ENV)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)


def test_switches_are_recorded_and_apc_bound():
    assert SWITCHES == ("MLX2_NEMOTRON_SSD_MMA", "MLX2_NEMOTRON_ATTN_STAGED")
    for name in SWITCHES:
        assert name in env_switches.SERVING_ENV_SWITCHES
        assert apc_numerics.CANDIDATE_ENV[name] == "flag"
    assert env_switches.serving_env_switches({}) == {}
    assert env_switches.serving_env_switches({SWITCHES[0]: "1"}) == {SWITCHES[0]: "1"}


@pytest.mark.parametrize("module", [mma, staged])
@pytest.mark.parametrize("value,on", [
    (None, False), ("", False), ("0", False), ("off", False), ("No", False),
    ("1", True), ("on", True), ("yes", True), ("2", True),
])
def test_enable_rule_matches_apc_flag_rule(module, value, on):
    name = mma.SSD_MMA_ENV if module is mma else staged.ATTN_STAGED_ENV
    env = {} if value is None else {name: value}
    assert module.enabled(env) is on
    # apc_numerics binds a "flag" switch exactly when it is on.
    assert (value is not None and value.strip().lower() not in apc_numerics._FALSE) is on


def _ssd_args(length=64, heads=8, head_dim=16, groups=2, state=32, dtype=mx.bfloat16):
    return (
        mx.zeros((1, length, heads, head_dim), dtype),
        mx.zeros((1, length, groups, state), dtype),
    )


def test_select_seg_off_by_default_and_refuses_without_metal(monkeypatch):
    assert mma.select_seg(*_ssd_args()) is None
    monkeypatch.setenv(mma.SSD_MMA_ENV, "1")
    before = dict(mma.status()["refusal_reasons"])
    with mx.stream(mx.cpu):
        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            assert mma.select_seg(*_ssd_args()) is None
        finally:
            mx.set_default_device(previous)
    after = mma.status()["refusal_reasons"]
    assert after.get("requires Metal", 0) == before.get("requires Metal", 0) + 1


@pytest.mark.parametrize("args,reason", [
    (_ssd_args(dtype=mx.float32), "requires matching 16-bit dtx/B/C"),
    (_ssd_args(head_dim=12), "head_dim 12"),
    (_ssd_args(state=20), "state_dim 20"),
    (_ssd_args(heads=6, groups=4), "heads 6 / groups 4"),
    (_ssd_args(head_dim=128, state=256), "threadgroup memory"),
])
def test_ssd_mma_geometry_refusals(args, reason):
    assert mma.refusal(*args) == reason


def test_super_geometry_fits_threadgroup_budget():
    reason = mma.refusal(*_ssd_args(heads=128, head_dim=64, groups=8, state=128))
    assert reason in (None, "requires Metal")


def test_ssd_prefill_runs_the_selected_segment(monkeypatch):
    calls = []

    def fake_seg(dtx, B, C, dtA, state):
        calls.append(dtx.shape[1])
        return mx.zeros(dtx.shape, dtx.dtype), state + 1

    monkeypatch.setattr(mma, "select_seg", lambda dtx, B, C=None: fake_seg)
    monkeypatch.setattr(ssm, "SSD_SEGMENT", 64)
    b, l, h, dh, g, ds = 1, 150, 4, 8, 2, 16
    x = mx.ones((b, l, h, dh), mx.bfloat16)
    B = mx.ones((b, l, g, ds), mx.bfloat16)
    y, state = ssm.ssd_prefill(
        x, mx.zeros((h,)), B, B, mx.ones((h,), mx.bfloat16),
        mx.zeros((b, l, h)), mx.zeros((h,)), None,
    )
    assert calls == [64, 64, 22]
    assert state.dtype == mx.float32 and float(state.max()) == 3.0
    assert y.shape == x.shape


class _Cache:
    pass


def _qkv(length=64, keys=None, hq=32, hk=2, d=128, dtype=mx.bfloat16):
    keys = length if keys is None else keys
    return (
        mx.zeros((1, hq, length, d), dtype),
        mx.zeros((1, hk, keys, d), dtype),
        mx.zeros((1, hk, keys, d), dtype),
    )


def test_staged_attention_off_by_default():
    assert staged.maybe_staged_attention(*_qkv(), cache=None, scale=1.0, mask="causal") is None
    assert staged.status()["enabled"] is False


@pytest.mark.parametrize("change,reason", [
    (dict(mask=mx.ones((64, 64), mx.bool_)), "mask is not the string causal"),
    (dict(mask=None), "mask is not the string causal"),
    (dict(cache="bits"), "quantized or bucketed cache"),
    (dict(qkv=_qkv(hq=8)), "GQA 8/2"),
    (dict(qkv=_qkv(d=64)), "head_dim 64"),
    (dict(qkv=_qkv(length=16)), "short query"),
    (dict(qkv=_qkv(length=64, keys=32)), "fewer keys than queries"),
    (dict(qkv=_qkv(dtype=mx.float32)), "requires matching 16-bit q/k/v"),
])
def test_staged_attention_refusals(change, reason):
    cache = None
    if change.get("cache") == "bits":
        cache = _Cache()
        cache.bits = 8
    q, k, v = change.get("qkv", _qkv())
    assert staged.refusal(q, k, v, cache, change.get("mask", "causal")) == reason


def _attention_module():
    from mlx2.runtime.models.nemotron_h import ModelArgs, NemotronHAttention

    args = ModelArgs.from_dict({
        "model_type": "nemotron_h", "vocab_size": 16, "hidden_size": 64,
        "intermediate_size": 64, "max_position_embeddings": 128,
        "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 8,
        "attention_bias": False, "mamba_num_heads": 4, "mamba_head_dim": 8,
        "mamba_proj_bias": False, "ssm_state_size": 16, "conv_kernel": 4,
        "n_groups": 1, "mlp_bias": False, "layer_norm_epsilon": 1e-5,
        "use_bias": False, "use_conv_bias": True, "layers_block_type": ["attention"],
        "num_hidden_layers": 1, "time_step_limit": (0.0, float("inf")),
    })
    return NemotronHAttention(args)


@pytest.mark.parametrize("length,consulted", [(1, False), (31, False), (32, True), (64, True)])
def test_nemotron_attention_consults_staged_only_for_prefill(monkeypatch, length, consulted):
    seen = []

    def fake(queries, keys, values, *, cache, scale, mask):
        seen.append((queries.shape, mask))
        return None  # refused: the module falls back to its SDPA

    monkeypatch.setattr(staged, "maybe_staged_attention", fake)
    with mx.stream(mx.cpu):
        attn = _attention_module()
        x = mx.ones((1, length, 64))
        mask = "causal" if length > 1 else None
        out = attn(x, mask=mask)
        mx.eval(out)
    assert out.shape == (1, length, 64)
    assert bool(seen) is consulted
    if consulted:
        assert seen[0] == ((1, 32, length, 8), "causal")


def test_nemotron_attention_uses_admitted_staged_output(monkeypatch):
    marker = []

    def fake(queries, keys, values, *, cache, scale, mask):
        marker.append(scale)
        return mx.zeros(queries.shape, queries.dtype)

    monkeypatch.setattr(staged, "maybe_staged_attention", fake)
    with mx.stream(mx.cpu):
        attn = _attention_module()
        out = attn(mx.ones((1, 40, 64)), mask="causal")
        mx.eval(out)
    assert marker == [pytest.approx(8 ** -0.5)]
    # o_proj has no bias: a zero attention output projects to zero.
    assert float(mx.abs(out).max()) == 0.0


def test_ssd_mma_refuses_a_mismatched_c():
    import mlx.core as mx

    from mlx2.runtime.models import nemotron_ssd_mma as ssd

    dtx = mx.zeros((1, 64, 8, 64), dtype=mx.bfloat16)
    B = mx.zeros((1, 64, 1, 128), dtype=mx.bfloat16)
    assert ssd.refusal(dtx, B, B.astype(mx.float32)) == "C dtype/shape differs from B"
    assert ssd.refusal(dtx, B, B[:, :32]) == "C dtype/shape differs from B"


def test_super_diagnostics_report_the_prefill_switches():
    from mlx2.adapters.nemotron35_super import _kapitsa_prefill_report

    report = _kapitsa_prefill_report()
    assert set(report) == {"ssd_mma", "attn_staged"}
    for item in report.values():
        assert item["implemented"] is True and item["qualified"] is False
        assert "counts" in item and "selected" in item and "observed_used" in item
