"""CPU tests for the two-launch Flash-Next HC decode (omlx #4038 port).

The kernels are Metal-only.  Here ``hc_decode_launch`` is replaced by the
composed ops it stands in for, so the tests pin the plumbing: the default-off
switch, admission and counted declines, the returned structure, the policy
field, and the layout cache.  Bit-exactness is a GPU claim checked by
``scripts/check_qwen4_hc_decode.py`` (real artifact weights) and by the
Metal-gated test at the bottom (``MLX2_METAL_TESTS=1``).
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime.models import qwen4_exp as Q
from mlx2.runtime.models import qwen4_hc_decode as HCD

H, R = 512, 320  # small stand-in with the admitted structure (FN: 2560, 320)


def _args(hidden=H, lowrank=R, hc=4):
    return SimpleNamespace(hc_count=hc, hidden_size=hidden, hc_lowrank=lowrank, rms_norm_eps=1e-6)


def _module(seed=0, combine=True, hidden=H, lowrank=R, bits=4, quantize=True):
    mx.random.seed(seed)
    m = Q.GatedResidual(_args(hidden, lowrank), use_combine=combine)
    m.hc_norm.weight = (1 + 0.1 * mx.random.normal(m.hc_norm.weight.shape)).astype(mx.bfloat16)
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        if name in m:
            m[name].weight = (0.05 * mx.random.normal(m[name].weight.shape)).astype(mx.bfloat16)
    if quantize:
        nn.quantize(m, group_size=64, bits=bits)
    m.eval()
    mx.eval(m.parameters())
    return m


def _x(seed, rows=1, hidden=H):
    return mx.random.normal((1, rows, 4 * hidden), key=mx.random.key(seed)).astype(mx.bfloat16)


@pytest.fixture(autouse=True)
def _clean():
    previous = HCD.set_hc_decode_enabled(False)
    HCD.reset_for_tests()
    yield
    HCD.set_hc_decode_enabled(previous)
    HCD.reset_for_tests()


@pytest.fixture
def reference_kernels(monkeypatch, served_exp_forms_match):
    """Swap the Metal launches for the composed body they transcribe."""
    calls = {"launch": 0}

    def launch(module, flat, *, debug=False, plan=None):
        calls["launch"] += 1
        was = HCD.set_hc_decode_enabled(False)
        try:
            out = module(flat[None])
        finally:
            HCD.set_hc_decode_enabled(was)
        if isinstance(out, tuple):
            return out[0].reshape(flat.shape[0], -1), out[2].reshape(flat.shape[0], -1)
        return out.reshape(flat.shape[0], -1), None

    monkeypatch.setattr(HCD, "hc_decode_launch", launch)
    monkeypatch.setattr(HCD, "runtime_supported", lambda: True)
    return calls


def _same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and mx.array_equal(a, b).item()


def test_env_parsing(monkeypatch):
    for raw, want in (("", False), ("0", False), ("off", False), ("1", True), ("on", True)):
        monkeypatch.setenv(HCD.HC_DECODE_ENV, raw)
        assert HCD.enabled_from_env() is want
    monkeypatch.setenv(HCD.HC_DECODE_ENV, "maybe")
    with pytest.raises(ValueError):
        HCD.enabled_from_env()
    monkeypatch.delenv(HCD.HC_DECODE_ENV)
    assert HCD.enabled_from_env() is False


def test_default_off_never_touches_the_kernels(reference_kernels):
    m = _module()
    m(_x(1))
    assert reference_kernels["launch"] == 0
    status = HCD.hc_decode_status()
    assert status["enabled"] is False and status["calls"] == 0 and status["declines"] == {}


def test_admission_accepts_flash_next_geometry():
    m = _module(hidden=2560, lowrank=320)
    assert HCD.static_admission(m) is None
    assert HCD.static_admission(_module(combine=False, hidden=2560, lowrank=320)) is None


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"bits": 8}, "bits"),
        ({"quantize": False}, "plain QuantizedLinear"),
        ({"lowrank": 512}, "lowrank"),
        ({"hidden": 320}, "hidden size"),
    ],
)
def test_admission_refuses_other_layouts(kwargs, needle):
    reason = HCD.static_admission(_module(**kwargs))
    assert reason is not None and needle in reason


def test_lane_owned_projection_is_refused():
    m = _module()
    m.input_mix_weight_up.__dict__["_lane_prepared"] = object()
    assert "lane" in HCD.static_admission(m)


@pytest.mark.parametrize("combine", [True, False])
def test_one_row_is_served_and_matches_the_composed_body(reference_kernels, combine):
    m = _module(combine=combine)
    x = _x(2)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    got = m(x)
    assert reference_kernels["launch"] == 1
    if combine:
        assert isinstance(got, tuple) and len(got) == 3
        assert got[1] is x
        assert _same(got[0], want[0]) and _same(got[2], want[2])
        assert got[2].shape == (1, 1, 4)
    else:
        assert _same(got, want)
    status = HCD.hc_decode_status()
    assert status["calls"] == 1 and status["launches"] == 2
    assert status["inject_calls"] == int(combine)


def test_projection_law_follows_mlx_dispatch(monkeypatch):
    monkeypatch.setattr(HCD, "_GPU_FAMILY", 17)
    monkeypatch.delenv("MLX_QMV_LIMIT", raising=False)
    assert HCD.projection_law(1) == 0
    assert [HCD.projection_law(r) for r in (2, 3, 8)] == [1, 1, 1]
    assert HCD.projection_law(9) is None
    monkeypatch.setenv("MLX_QMV_LIMIT", "4")
    assert HCD.projection_law(3) is None and HCD.projection_law(1) == 0
    monkeypatch.delenv("MLX_QMV_LIMIT")
    monkeypatch.setattr(HCD, "_GPU_FAMILY", 14)
    assert HCD.projection_law(3) is None and HCD.projection_law(1) == 0


def test_verify_window_is_served_on_wide_law(reference_kernels, monkeypatch):
    monkeypatch.setattr(HCD, "_GPU_FAMILY", 17)
    m = _module()
    x = _x(3, rows=3)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    got = m(x)
    assert reference_kernels["launch"] == 1
    assert got[1] is x and all(_same(a, b) for a, b in zip(got, want))
    assert got[0].shape == (1, 3, H) and got[2].shape == (1, 3, 4)


def test_wide_rows_beyond_the_cap_decline_and_are_counted(reference_kernels, monkeypatch):
    monkeypatch.setattr(HCD, "_GPU_FAMILY", 17)
    m = _module()
    x = _x(3, rows=9)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    got = m(x)
    assert reference_kernels["launch"] == 0
    assert all(_same(a, b) for a, b in zip(got, want))
    status = HCD.hc_decode_status()
    assert status["calls"] == 0
    assert sum(status["declines"].values()) == 1
    assert status["last_decline"].startswith("rows")


def _law_of(plan):
    return dict(plan["a_template"])["LAW"]


@pytest.fixture
def law_recorder(reference_kernels, monkeypatch):
    laws = []
    composed = HCD.hc_decode_launch

    def launch(module, flat, *, debug=False, plan=None):
        laws.append((flat.shape[0], _law_of(plan)))
        return composed(module, flat, debug=debug, plan=plan)

    monkeypatch.setattr(HCD, "hc_decode_launch", launch)
    monkeypatch.setattr(HCD, "_GPU_FAMILY", 17)
    monkeypatch.setattr(HCD, "_ROW_EXACT", True)
    return laws


@pytest.mark.parametrize("rows", [2, 3, 8, 9, 17])
def test_row_exact_window_runs_the_one_row_law(law_recorder, rows):
    from mlx2.runtime import row_exact_verify as REV

    m = _module()
    x = _x(11, rows=rows)
    # The route pins the one-token width for the norm gate (verify_backbone).
    with Q._declared_width(1):
        want = m(x)
        HCD.set_hc_decode_enabled(True)
        record = REV.Window(rows)
        with REV.window(record):
            got = m(x)
    assert law_recorder == [(rows, 0)]
    assert all(_same(a, b) for a, b in zip(got, want))
    assert record.stages == {"hc": {"kernel_one_row_law": 1}} and record.exact
    assert HCD.hc_decode_status()["row_exact_calls"] == 1


def test_outside_a_window_the_wide_law_is_unchanged(law_recorder):
    m = _module()
    HCD.set_hc_decode_enabled(True)
    m(_x(12, rows=3))
    m(_x(12, rows=1))
    assert law_recorder == [(3, 1), (1, 0)]
    assert HCD.hc_decode_status()["row_exact_calls"] == 0


def test_row_exact_window_beyond_the_guard_stays_composed(law_recorder):
    from mlx2.runtime import row_exact_verify as REV

    m = _module()
    rows = HCD.ROW_EXACT_MAX_ROWS + 1
    HCD.set_hc_decode_enabled(True)
    record = REV.Window(rows)
    with REV.window(record), Q._declared_width(1):
        m(_x(13, rows=rows))
    assert law_recorder == []
    assert record.stages == {"hc": {"composed": 1}}


def test_row_exact_mode_kill_switch_keeps_windows_composed(law_recorder):
    from mlx2.runtime import row_exact_verify as REV

    m = _module()
    HCD.set_hc_decode_enabled(True)
    previous = HCD.set_hc_row_exact_enabled(False)
    try:
        record = REV.Window(3)
        with REV.window(record), Q._declared_width(1):
            m(_x(14, rows=3))
        # Not the wide law either: that is not the one-token arithmetic.
        assert law_recorder == []
        assert record.stages == {"hc": {"composed": 1}}
        assert HCD.hc_decode_status()["last_decline"] == "row-exact window mode off"
    finally:
        HCD.set_hc_row_exact_enabled(previous)


def test_row_exact_mode_is_default_off(monkeypatch):
    monkeypatch.delenv(HCD.HC_ROW_EXACT_ENV, raising=False)
    assert HCD.row_exact_from_env() is False
    monkeypatch.setenv(HCD.HC_ROW_EXACT_ENV, "1")
    assert HCD.row_exact_from_env() is True
    monkeypatch.setenv(HCD.HC_ROW_EXACT_ENV, "maybe")
    with pytest.raises(ValueError):
        HCD.row_exact_from_env()


def test_row_exact_class_swapped_projections_are_admitted():
    from mlx2.runtime.models import qwen4_row_exact as RE

    m = _module()
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        layer = m[name]
        layer.__class__ = RE._subclass(RE._RowExactQuantizedLinear, type(layer))
    assert HCD._cached_static_admission(m) is None

    class Other(nn.QuantizedLinear):
        pass

    m.input_mix_weight_up.__class__ = Other
    assert "plain QuantizedLinear" in HCD._cached_static_admission(m)


def test_ineligible_layout_is_counted_and_composed(reference_kernels):
    m = _module(bits=8)
    x = _x(4)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    got = m(x)
    assert reference_kernels["launch"] == 0
    assert all(_same(a, b) for a, b in zip(got, want))
    assert "bits" in HCD.hc_decode_status()["last_decline"]


def test_non_eager_norm_and_glue_decline(reference_kernels, monkeypatch):
    m = _module()
    HCD.set_hc_decode_enabled(True)
    monkeypatch.setattr(Q, "fused_group_norm_enabled", lambda: True)
    m(_x(5))
    assert HCD.hc_decode_status()["last_decline"].startswith("norm")
    monkeypatch.setattr(Q, "fused_group_norm_enabled", lambda: False)
    monkeypatch.setattr(Q, "compile_glue_enabled", lambda: True)
    m(_x(5))
    assert HCD.hc_decode_status()["last_decline"] == "compiled glue enabled"
    assert reference_kernels["launch"] == 0


def test_without_metal_the_route_falls_back(monkeypatch):
    monkeypatch.setattr(HCD, "runtime_supported", lambda: False)
    m = _module()
    x = _x(6)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    got = m(x)
    assert all(_same(a, b) for a, b in zip(got, want))
    assert HCD.hc_decode_status()["last_decline"] == "Metal runtime unavailable"


def test_first_use_error_demotes_the_path(monkeypatch, served_exp_forms_match):
    def broken(module, flat, *, debug=False, plan=None):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(HCD, "hc_decode_launch", broken)
    monkeypatch.setattr(HCD, "runtime_supported", lambda: True)
    m = _module()
    x = _x(7)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    got = m(x)
    assert all(_same(a, b) for a, b in zip(got, want))
    status = HCD.hc_decode_status()
    assert status["broken"] is True and status["errors"] == 1
    m(x)
    assert HCD.hc_decode_status()["declines"]["demoted after an error"] == 2


def test_layout_cache_rechecks_replaced_tensors():
    m = _module()
    assert HCD._cached_static_admission(m) is None
    m.input_mix_weight_down.scales = m.input_mix_weight_down.scales.astype(mx.float16)
    assert "scales" in HCD._cached_static_admission(m)


def test_flash_next_policy_defaults_on_and_off_is_receipt_neutral():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default = FlashNextPolicy()
    assert default.as_dict()["hc_decode_kernels"] is True
    assert default.environment()[HCD.HC_DECODE_ENV] == "1"
    off = FlashNextPolicy.from_mapping({"hc_decode_kernels": False})
    assert "hc_decode_kernels" not in off.as_dict()
    assert HCD.HC_DECODE_ENV not in off.environment()
    with pytest.raises(ValueError, match="hc_decode_kernels"):
        FlashNextPolicy.from_mapping({"hc_decode_kernels": "yes"})


@pytest.mark.skipif(
    os.environ.get("MLX2_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="Metal oracle test; set MLX2_METAL_TESTS=1 on an idle GPU",
)
def test_metal_kernels_are_bit_identical_to_the_composed_body():
    with mx.stream(mx.gpu):
        for combine in (True, False):
            m = _module(combine=combine, hidden=2560, lowrank=320)
            for seed in range(6):
                x = _x(seed, hidden=2560) * (0.1 * (seed + 1))
                want = m(x)
                HCD.set_hc_decode_enabled(True)
                got = m(x)
                HCD.set_hc_decode_enabled(False)
                want = want if isinstance(want, tuple) else (want,)
                got = got if isinstance(got, tuple) else (got,)
                for a, b in zip(want, got):
                    assert mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()
        assert HCD.hc_decode_status()["calls"] == 12


def test_raw_gate_inject_mode_declines_the_hc_kernels(reference_kernels):
    # The kernels return the finished 2*sigmoid inject; main's fused
    # gate-inject path asks GatedResidual for the raw gate logit instead.
    # Serving both would apply the sigmoid twice (campaign merge, 2026-10-01).
    m = _module(seed=7, combine=True)
    x = _x(8)
    want = m(x, raw_inject=True)
    HCD.set_hc_decode_enabled(True)
    got = m(x, raw_inject=True)
    assert reference_kernels["launch"] == 0
    assert _same(got[0], want[0]) and _same(got[2], want[2])
    assert HCD.hc_decode_status()["declines"] == {"raw gate inject requested": 1}
