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
# A dense bf16 inject needs 8 norm_down simdgroups (hidden >= 1024): MLX's
# gemv spreads K over 8 simdgroups (sweep 2026-10-02 A2).
DH = 1024


def _args(hidden=H, lowrank=R, hc=4):
    return SimpleNamespace(hc_count=hc, hidden_size=hidden, hc_lowrank=lowrank, rms_norm_eps=1e-6)


def _module(seed=0, combine=True, hidden=H, lowrank=R, bits=4, quantize=True,
            inject_bits=None, dense_inject=False):
    """``inject_bits`` quantizes the inject at its own width; ``dense_inject``
    leaves it a bf16 ``nn.Linear`` (the uncensored artifact's layout)."""
    mx.random.seed(seed)
    m = Q.GatedResidual(_args(hidden, lowrank), use_combine=combine)
    m.hc_norm.weight = (1 + 0.1 * mx.random.normal(m.hc_norm.weight.shape)).astype(mx.bfloat16)
    for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
        if name in m:
            m[name].weight = (0.05 * mx.random.normal(m[name].weight.shape)).astype(mx.bfloat16)
    if quantize:
        def projection(path, module):
            if path == "block_inject_weight":
                if dense_inject:
                    return False
                if inject_bits is not None:
                    return {"group_size": 64, "bits": inject_bits}
            return isinstance(module, nn.Linear)

        nn.quantize(m, group_size=64, bits=bits, class_predicate=projection)
    m.eval()
    mx.eval(m.parameters())
    return m


def _x(seed, rows=1, hidden=H):
    return mx.random.normal((1, rows, 4 * hidden), key=mx.random.key(seed)).astype(mx.bfloat16)


@pytest.fixture(autouse=True)
def _clean():
    previous = HCD.set_hc_decode_enabled(False)
    multi_row = HCD.set_hc_multi_row_mode("off")
    HCD.reset_for_tests()
    yield
    HCD.set_hc_decode_enabled(previous)
    HCD.set_hc_multi_row_mode(multi_row)
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
        ({"bits": 6}, "bits"),
        ({"inject_bits": 6}, "inject: bits"),
        ({"quantize": False}, "plain QuantizedLinear"),
        ({"lowrank": 512}, "lowrank"),
        ({"hidden": 320}, "hidden size"),
    ],
)
def test_admission_refuses_other_layouts(kwargs, needle):
    reason = HCD.static_admission(_module(**kwargs))
    assert reason is not None and needle in reason


@pytest.mark.parametrize(
    "kwargs,inject",
    [
        ({"bits": 8}, "b8"),  # all-8-bit HC
        ({"bits": 8, "dense_inject": True}, "dense"),  # uncensored artifact
        ({"bits": 8, "inject_bits": 4}, "b4"),  # mixed formats in one module
        ({"bits": 4, "inject_bits": 8}, "b8"),
    ],
)
def test_admission_accepts_8bit_and_mixed_formats(kwargs, inject):
    m = _module(hidden=2560, lowrank=320, **kwargs)
    assert HCD.static_admission(m) is None
    plan = HCD._build_plan(m, 0)
    bits = kwargs["bits"]
    assert plan["format"] == f"down b{bits} up b{bits} inject {inject}"
    template = dict(plan["a_template"])
    assert template["DB"] == bits and dict(plan["b_template"])["UB"] == bits
    assert plan["has_inject"] is True
    assert template["INJ"] == (2 if inject == "dense" else 1)
    if inject == "dense":  # the bf16 weight is bound in the inject slot
        assert plan["a_inputs"][6] is m.block_inject_weight.weight


def test_dense_inject_width_must_be_whole_gemv_blocks():
    m = _module(bits=8, dense_inject=True, hidden=DH)  # width 4096: whole blocks
    assert HCD.static_admission(m) is None
    # width 4608: 4.5 gemv blocks (9 simdgroups; lowrank 576 tiles them)
    m = _module(bits=8, dense_inject=True, hidden=1152, lowrank=576)
    assert "gemv tiling" in HCD.static_admission(m)


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        # quantized inject at law 0 writes row sg for sg < 4: hidden/128 >= 4
        ({"hidden": 256, "lowrank": 64 * 5}, "inject rows"),
        # dense inject: MLX gemv spreads K over 8 simdgroups
        ({"bits": 8, "dense_inject": True, "hidden": 512}, "dense inject"),
        ({"bits": 8, "dense_inject": True, "hidden": 768, "lowrank": 384}, "dense inject"),
        # xs[4 * hidden] in bf16 alone fills the 32 KiB threadgroup memory
        ({"bits": 8, "hidden": 4096, "lowrank": 384}, "threadgroup memory"),
    ],
)
def test_admission_refuses_hidden_sizes_the_inject_cannot_compute(kwargs, needle):
    # Sweep 2026-10-02 A2: these layouts were admitted; the kernel left
    # inject rows unwritten, skipped K columns, or overflowed threadgroup
    # memory.
    reason = HCD.static_admission(_module(**kwargs))
    assert reason is not None and needle in reason, reason


def test_admission_threadgroup_memory_boundary():
    # 3968 is the widest multiple of 128 whose norm_down fits 32 KiB.
    assert HCD.static_admission(_module(bits=8, hidden=3968, lowrank=1984)) is None


def test_dense_inject_with_bias_or_other_dtype_is_refused():
    m = _module(bits=8, dense_inject=True, hidden=DH)
    m.block_inject_weight.bias = mx.zeros((4,), mx.bfloat16)
    assert "dense with bias" in HCD.static_admission(m)
    m = _module(bits=8, dense_inject=True, hidden=DH)
    m.block_inject_weight.weight = m.block_inject_weight.weight.astype(mx.float32)
    assert "dense weight dtype" in HCD.static_admission(m)


def test_8bit_alignment_follows_mlx_qmv_fast_block():
    assert HCD.qmv_fast_alignment(4) == 512 and HCD.qmv_fast_alignment(8) == 256
    # lowrank 256 is a whole 8-bit qmv_fast block: MLX would run qmv_fast
    # (not qmv) for the up rows, which the up launch does not transcribe.
    m = _module(bits=8, lowrank=256)
    assert "lowrank" in HCD.static_admission(m)


@pytest.fixture
def pair_reference(monkeypatch, served_exp_forms_match):
    """Replace only the compiled Metal pair by the composed body: the plan,
    admission, reshapes and counters around it run for real on CPU (the
    served-exp gates, which need Metal to probe, are taken as matching)."""

    def compiled_pair(plan, rows):
        def pair(flat, *_weights, module):
            was = HCD.set_hc_decode_enabled(False)
            try:
                out = module(flat[None])
            finally:
                HCD.set_hc_decode_enabled(was)
            if isinstance(out, tuple):
                return out[0].reshape(rows, -1), out[2].reshape(rows, -1)
            return out.reshape(rows, -1), mx.zeros((rows, 4), mx.bfloat16)

        return pair

    real = HCD.hc_decode_launch

    def launch(module, flat, *, debug=False, plan=None):
        monkeypatch.setattr(
            HCD, "_compiled_pair",
            lambda p, r: (lambda *args: compiled_pair(p, r)(*args, module=module)),
        )
        return real(module, flat, debug=debug, plan=plan)

    monkeypatch.setattr(HCD, "hc_decode_launch", launch)
    monkeypatch.setattr(HCD, "runtime_supported", lambda: True)
    monkeypatch.setattr(HCD, "_GPU_FAMILY", 17)


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("kwargs", [{"dense_inject": True}, {"inject_bits": 4}, {}])
def test_8bit_module_is_served_with_its_inject(pair_reference, rows, kwargs):
    HCD.set_hc_multi_row_mode("on")
    hidden = DH if kwargs.get("dense_inject") else H
    m = _module(seed=9, bits=8, hidden=hidden, **kwargs)
    x = _x(21, rows=rows, hidden=hidden)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    got = m(x)
    assert got[1] is x
    assert _same(got[0], want[0]) and _same(got[2], want[2])
    assert got[2].shape == (1, rows, 4)
    status = HCD.hc_decode_status()
    assert status["calls"] == 1 and status["inject_calls"] == 1
    inject = "dense" if kwargs.get("dense_inject") else f"b{kwargs.get('inject_bits', 8)}"
    assert status["calls_by_format"] == {f"down b8 up b8 inject {inject}": 1}


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


def test_multi_row_off_declines_counted(reference_kernels, monkeypatch):
    monkeypatch.setattr(HCD, "_GPU_FAMILY", 17)
    m = _module()
    x = _x(3, rows=3)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    got = m(x)
    assert reference_kernels["launch"] == 0
    assert all(_same(a, b) for a, b in zip(got, want))
    status = HCD.hc_decode_status()
    assert status["multi_row_mode"] == "off"
    assert status["declines"] == {HCD.MULTI_ROW_DECLINE: 1}
    m(_x(4, rows=1))  # one row is unaffected
    assert reference_kernels["launch"] == 1


def test_multi_row_env_parsing(monkeypatch):
    monkeypatch.delenv(HCD.HC_MULTI_ROW_ENV, raising=False)
    assert HCD.multi_row_from_env() == "auto"
    for raw, want in (("1", "on"), ("on", "on"), ("0", "off"), ("auto", "auto")):
        monkeypatch.setenv(HCD.HC_MULTI_ROW_ENV, raw)
        assert HCD.multi_row_from_env() == want
    monkeypatch.setenv(HCD.HC_MULTI_ROW_ENV, "maybe")
    with pytest.raises(ValueError):
        HCD.multi_row_from_env()
    with pytest.raises(ValueError):
        HCD.set_hc_multi_row_mode("maybe")


@pytest.mark.parametrize(
    "kwargs,served",
    [({}, True), ({"bits": 8}, False), ({"bits": 8, "dense_inject": True}, False),
     ({"inject_bits": 8}, False), ({"combine": False}, True)],
)
def test_multi_row_auto_serves_all_4bit_layouts_only(pair_reference, kwargs, served):
    HCD.set_hc_multi_row_mode("auto")
    hidden = DH if kwargs.get("dense_inject") else H
    m = _module(seed=12, hidden=hidden, **kwargs)
    x = _x(23, rows=3, hidden=hidden)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    got = m(x)
    want = want if isinstance(want, tuple) else (want,)
    got = got if isinstance(got, tuple) else (got,)
    assert _same(got[0], want[0])
    status = HCD.hc_decode_status()
    assert status["calls"] == int(served)
    if not served:
        assert status["declines"] == {HCD.MULTI_ROW_FORMAT_DECLINE: 1}
    m(_x(24, rows=1, hidden=hidden))  # one row is served whatever the layout
    assert HCD.hc_decode_status()["calls"] == int(served) + 1


def test_verify_window_is_served_on_wide_law(reference_kernels, monkeypatch):
    monkeypatch.setattr(HCD, "_GPU_FAMILY", 17)
    HCD.set_hc_multi_row_mode("on")
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
    HCD.set_hc_multi_row_mode("on")
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
    HCD.set_hc_multi_row_mode("on")
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
    m = _module(bits=6)
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


class _FaultyMx:
    """``mx`` for the HC module whose next ``eval`` raises ``text`` once."""

    def __init__(self, text):
        self.text = text

    def __getattr__(self, name):
        return getattr(mx, name)

    def eval(self, *args):
        if self.text is not None:
            (text, self.text) = (self.text, None)
            raise RuntimeError(text)
        return mx.eval(*args)


_OOM = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)
_TIMEOUT = (
    "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
    "(00000002:kIOGPUCommandBufferCallbackErrorTimeout)"
)


@pytest.mark.parametrize("fault", [_OOM, _TIMEOUT], ids=["oom", "gpu_timeout"])
def test_device_fault_in_validation_reraises_and_does_not_demote(
    reference_kernels, monkeypatch, fault
):
    # Serving recovers device faults by failing the lanes stepped in the
    # failed buffer and rebuilding; the first-use validation eval swallowed
    # one, set the process-wide _BROKEN and served the composed ops for the
    # rest of the process (sweep 2026-10-08, flashnext-kernels#2).
    monkeypatch.setattr(HCD, "mx", _FaultyMx(fault))
    m = _module()
    x = _x(3)
    HCD.set_hc_decode_enabled(True)
    with pytest.raises(RuntimeError, match="Command buffer execution failed"):
        m(x)
    status = HCD.hc_decode_status()
    assert status["broken"] is False, status["last_error"]
    # Serving recovered; the next step validates again and serves the kernel.
    m(x)
    status = HCD.hc_decode_status()
    assert status["broken"] is False
    assert status["calls"] == 1


def test_compile_failure_in_validation_still_demotes(reference_kernels, monkeypatch):
    monkeypatch.setattr(HCD, "mx", _FaultyMx("Unable to build metal library"))
    m = _module()
    HCD.set_hc_decode_enabled(True)
    m(_x(4))
    assert HCD.hc_decode_status()["broken"] is True


def test_layout_cache_rechecks_replaced_tensors():
    m = _module()
    assert HCD._cached_static_admission(m) is None
    m.input_mix_weight_down.scales = m.input_mix_weight_down.scales.astype(mx.float16)
    assert "scales" in HCD._cached_static_admission(m)


def test_multi_row_policy_auto_is_receipt_neutral():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default = FlashNextPolicy()
    assert default.hc_decode_multi_row == "auto"
    assert "hc_decode_multi_row" not in default.as_dict()
    assert HCD.HC_MULTI_ROW_ENV not in default.environment()
    for mode in ("on", "off"):
        chosen = FlashNextPolicy.from_mapping({"hc_decode_multi_row": mode})
        assert chosen.as_dict()["hc_decode_multi_row"] == mode
        assert chosen.environment()[HCD.HC_MULTI_ROW_ENV] == mode
    with pytest.raises(ValueError, match="requires hc_decode_kernels"):
        FlashNextPolicy.from_mapping({"hc_decode_multi_row": "on", "hc_decode_kernels": False})
    with pytest.raises(ValueError, match="auto, on, or off"):
        FlashNextPolicy.from_mapping({"hc_decode_multi_row": True})


def test_flash_next_policy_defaults_on_and_off_is_receipt_neutral():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default = FlashNextPolicy()
    assert "hc_decode_kernels" not in default.as_dict()  # omitted at its default
    assert default.environment()[HCD.HC_DECODE_ENV] == "1"
    off = FlashNextPolicy.from_mapping({"hc_decode_kernels": False})
    assert off.as_dict()["hc_decode_kernels"] is False  # an explicit off is recorded
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


def _stub_decoder_layer(seed=11):
    """A DecoderLayer with its two hyper connections and stand-in branches."""
    layer = Q.DecoderLayer.__new__(Q.DecoderLayer)
    nn.Module.__init__(layer)
    layer.is_linear = True
    layer.ple = None
    layer.attn_hyper_connection = _module(seed=seed, hidden=DH, dense_inject=True)
    layer.mlp_hyper_connection = _module(seed=seed + 1, hidden=DH, dense_inject=True)
    layer.linear_attn = lambda mixed, ssm_mask, cache: (mixed * 0.5).astype(mx.bfloat16)
    layer.mlp = lambda mixed: (mixed * 0.25).astype(mx.bfloat16)
    return layer


def test_gate_inject_yields_to_the_hc_decode_kernels(reference_kernels):
    # Options sweep 2026-10-01: with MLX_QWEN4_FUSED_GATE_INJECT on, every
    # dense-inject HC call declined the two-launch kernels ("raw gate inject
    # requested", 37,632 at B1) and ordinary decode lost 8.5%.  The candidate
    # now serves only the calls the HC kernels decline (counted).
    from mlx2.runtime.models import qwen4_gate_inject as GI

    layer = _stub_decoder_layer()
    x = _x(12, hidden=DH)
    want = layer(x, None)
    previous = GI.fused_gate_inject_enabled()
    GI.set_fused_gate_inject_enabled(True)
    GI.reset_qwen4_gate_inject_stats()
    try:
        HCD.set_hc_decode_enabled(True)
        got = layer(x, None)
        assert _same(got, want)
        assert reference_kernels["launch"] == 2
        assert "raw gate inject requested" not in HCD.hc_decode_status()["declines"]
        counts = GI.qwen4_gate_inject_stats()["counts"]
        assert counts["decline:hc decode kernels served the call"] == 2
        assert counts.get("calls", 0) == 0
        # A call the HC kernels decline (2 rows with multi-row off) still
        # reaches the gate-inject candidate with the raw gate.
        wide = _x(13, rows=2, hidden=DH)
        HCD.set_hc_decode_enabled(False)
        want_wide = layer(wide, None)
        HCD.set_hc_decode_enabled(True)
        GI.reset_qwen4_gate_inject_stats()
        assert _same(layer(wide, None), want_wide)
        assert reference_kernels["launch"] == 2
        counts = GI.qwen4_gate_inject_stats()["counts"]
        assert "decline:hc decode kernels served the call" not in counts
        assert counts["attempts"] == 2  # both reached admission (declined on CPU)
        assert HCD.hc_decode_status()["declines"] == {HCD.MULTI_ROW_DECLINE: 2}
    finally:
        GI.set_fused_gate_inject_enabled(previous)
        GI.reset_qwen4_gate_inject_stats()


def test_gate_inject_yields_per_token_under_shape_stable_projections(
    reference_kernels, monkeypatch
):
    # Sweep 2026-10-02 A3: with MLX_QWEN4_GDN_SHAPE_STABLE_PROJECTIONS on, a
    # multi-token call went raw as a whole; __call__ then split it per token
    # and every one-row call declined the HC kernels ("raw gate inject
    # requested").  Each token now yields to the kernels as at one token.
    from mlx2.runtime.models import qwen4_gate_inject as GI

    monkeypatch.setattr(Q, "_GDN_SHAPE_STABLE_PROJECTIONS", True)
    layer = _stub_decoder_layer()
    x = _x(13, rows=2, hidden=DH)
    HCD.set_hc_decode_enabled(True)
    previous = GI.fused_gate_inject_enabled()
    results = {}
    try:
        for gate_inject in (False, True):
            GI.set_fused_gate_inject_enabled(gate_inject)
            GI.reset_qwen4_gate_inject_stats()
            HCD.reset_for_tests()
            reference_kernels["launch"] = 0
            out = layer(x, None)
            results[gate_inject] = (
                reference_kernels["launch"], HCD.hc_decode_status()["declines"], out
            )
        counts = GI.qwen4_gate_inject_stats()["counts"]
    finally:
        GI.set_fused_gate_inject_enabled(previous)
        GI.reset_qwen4_gate_inject_stats()
    assert results[False][0] == 4  # two hyper connections x two tokens
    assert results[True][0] == results[False][0]
    assert "raw gate inject requested" not in results[True][1]
    assert counts["decline:hc decode kernels served the call"] == 4
    assert _same(results[True][2], results[False][2])


def test_shape_stable_gate_inject_mixed_tokens_finish_the_inject(monkeypatch):
    # A token the kernels decline carries the raw gate; mixed with a served
    # token, its inject is finished with the composed 2*sigmoid(raw / hc).
    monkeypatch.setattr(Q, "_GDN_SHAPE_STABLE_PROJECTIONS", True)
    m = _module(seed=5, hidden=DH, dense_inject=True)
    x = _x(31, rows=2, hidden=DH)
    want = m(x)
    HCD.set_hc_decode_enabled(True)
    served = iter([None, "fused"])

    def try_once(hyper_input, glue):
        if next(served) is None:
            return None
        was = HCD.set_hc_decode_enabled(False)
        try:
            return m(hyper_input)
        finally:
            HCD.set_hc_decode_enabled(was)

    monkeypatch.setattr(m, "_try_hc_decode", try_once)
    mixed, residual, inject, raw = m.split_for_gate_inject(x)
    assert raw is False
    assert _same(mixed, want[0]) and _same(inject, want[2])
