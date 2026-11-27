"""``fuse_kernels.RESIDUAL_MLX_ORDER`` (default on since 2026-11-26), CPU only.

The GPU question (is it bitwise?) is
``scripts/research/inkling2_residual_bitwise_probe.py``. Here: with the
switch off the shipped kernel's source and launch are byte-identical
whatever ``law`` the caller passes; the route plan restates MLX 0.32.5's
dispatch for the checkpoint shapes; the candidate's launch and declines; and
every generated Metal source (fused kernel, the probe's stage kernels)
compiles offline (``xcrun metal``, no GPU).
"""

import hashlib
import importlib.util
import py_compile
import types
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime.models.inkling2 import fuse_kernels as K

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "research" / "inkling2_residual_bitwise_probe.py"
# sha256[:16] of _residual_source(n) at origin/main 080c56000 (before the candidate).
SHIPPED = {1: "24a84c163824fec8", 2: "6d12e976d8fc5668", 5: "825604a150ffd6ef",
           16: "6657ffbd915b421c"}


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


class _Recorder:
    def __init__(self):
        self.built, self.calls = [], []

    def metal_kernel(self, **kwargs):
        self.built.append(kwargs)

        def run(**call):
            self.calls.append(call)
            return [mx.zeros(call["output_shapes"][0], call["output_dtypes"][0])]

        return run


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(mx.fast, "metal_kernel", rec.metal_kernel)
    monkeypatch.setattr(K, "_res_kernels", {})
    monkeypatch.setattr(K, "_mo_kernels", {})
    return rec


def _states(n, width=2304, dtype=mx.bfloat16):
    return [mx.zeros((1, 1, width), dtype) for _ in range(n - 1)] + [mx.zeros((1, 1, width))]


@pytest.mark.parametrize("n", sorted(SHIPPED))
def test_shipped_source_unchanged(n):
    assert hashlib.sha256(K._residual_source(n).encode()).hexdigest()[:16] == SHIPPED[n]


@pytest.mark.parametrize("n,nq", [(1, 1), (4, 1), (12, 3), (8, 4)])
def test_switch_off_dispatch_byte_identical(recorder, n, nq, monkeypatch):
    monkeypatch.setattr(K, "RESIDUAL_MLX_ORDER", False)
    query = mx.zeros((2304,) if nq == 1 else (nq, 2304))
    outs = []
    for law in ("residual", "message"):
        recorder.built.clear()
        recorder.calls.clear()
        K._res_kernels.clear()
        outs.append(K.softmax_mix(types.SimpleNamespace(), _states(n), query, law=law))
        (built,), (call,) = recorder.built, recorder.calls
        assert built == {"name": f"inkling2_fuse_residual_{n}",
                         "input_names": [f"b{i}" for i in range(n - 1)] + ["p", "q", "scale"],
                         "output_names": ["out"], "source": K._residual_source(n), "header": ""}
        assert call["template"] == [("N", n), ("D", 2304), ("TG", 256)]
        assert call["grid"] == (nq * 256, 1, 1) and call["threadgroup"] == (256, 1, 1)
        assert call["output_shapes"] == [(1, 1, nq, 2304)]
        assert [a.shape for a in call["inputs"]] == [(2304,)] * n + [(nq, 2304), (1,)]
    assert outs[0].shape == outs[1].shape == (1, 1, nq, 2304)


def test_plan_residual_one_query():
    for n in (2, 4, 8, 12, 16):
        plan, why = K.mlx_order_plan("residual", n, 1, 2304, tf32=False, arch="applegpu_g17s")
        assert why is None
        assert (plan["scores"], plan["softmax"], plan["mix"], plan["tg"]) == (
            "gemv", "row_small", "gemv_t", 576)
    plan, _ = K.mlx_order_plan("residual", 1, 1, 2304, tf32=False, arch="applegpu_g17s")
    assert plan["scores"] == "dot"
    # gemv routes do not depend on TF32
    assert K.mlx_order_plan("residual", 8, 1, 2304, tf32=True, arch="applegpu_g17s")[1] is None


@pytest.mark.parametrize("arch,bk", [("applegpu_g17s", 16), ("applegpu_g15d", 32),
                                     ("applegpu_g17g", 16)])
def test_plan_residual_three_queries(arch, bk):
    for n in (2, 4, 8, 12, 16):
        plan, why = K.mlx_order_plan("residual", n, 3, 2304, tf32=False, arch=arch)
        assert why is None
        assert (plan["scores"], plan["parts"], plan["iters"], plan["iters_last"]) == (
            "splitk", 32, 4, 20)
        assert (plan["softmax"], plan["col_rows"]) == ("col_small", min(8, n))
        assert (plan["mix"], plan["bk"]) == ("gemm", bk)
    assert K.mlx_order_plan("residual", 8, 3, 2304, tf32=True, arch="applegpu_g17s") == (
        None, "tf32")


def test_plan_message():
    plan, _ = K.mlx_order_plan("message", 1, 4, 2304, tf32=False, arch="applegpu_g17s")
    assert (plan["sa"], plan["sb"], plan["scores"], plan["softmax"], plan["mix"]) == (
        4, 1, "gemv", "block", "gemm")
    for n in (2, 8, 16):
        plan, _ = K.mlx_order_plan("message", n, 4, 2304, tf32=False, arch="applegpu_g17s")
        assert (plan["sa"], plan["sb"], plan["scores"], plan["parts"]) == (4, n, "splitk", 32)


def test_plan_declines():
    assert K.mlx_order_plan("residual", 17, 1, 2304, tf32=False, arch="applegpu_g17s")[1] == "states"
    assert K.mlx_order_plan("residual", 4, 1, 512, tf32=False, arch="applegpu_g17s")[1] == "width"
    assert K.mlx_order_plan("residual", 4, 5, 2304, tf32=False, arch="applegpu_g17s")[1] == "softmax_cols"
    assert K.mlx_order_plan("other", 4, 1, 2304, tf32=False, arch="applegpu_g17s")[1] == "law"


def test_switch_on_launch_and_decline(recorder, monkeypatch):
    monkeypatch.setattr(K, "RESIDUAL_MLX_ORDER", True)
    monkeypatch.setattr(K, "_ARCH", ["applegpu_g17s"])
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    module = types.SimpleNamespace()
    out = K.softmax_mix(module, _states(12), mx.zeros((3, 2304)))
    (built,), (call,) = recorder.built, recorder.calls
    assert built["name"] == "inkling2_fuse_residual_mlx_residual_12_3_source"
    assert built["input_names"][-4:] == ["p", "q", "scale", "eps"]
    assert call["grid"] == (576, 1, 1) and call["threadgroup"] == (576, 1, 1)
    assert out.shape == (1, 1, 3, 2304)
    assert module.fuse_counts == {"mlx_order_kernel:residual": 1}
    recorder.built.clear()
    recorder.calls.clear()
    small = K.softmax_mix(module, _states(4, 512), mx.zeros((512,)))
    # A declined shape keeps the earlier near-exact kernel (orchestrator
    # report 2026-11-26: tiny Metal-test widths ran the composed ops).
    assert module.fuse_counts["kernel_declined:mlx_order:width"] == 1
    (built,), _ = recorder.built, recorder.calls
    assert built["name"] == "inkling2_fuse_residual_4"
    assert small.shape == (1, 1, 1, 512)


def _compile_or_skip():
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from inkling2_metal_compile import compiler

    if compiler() is None:
        pytest.skip("no offline Metal compiler (xcrun metal)")


CASES = [("residual", n, nq) for n in (1, 2, 12, 16) for nq in (1, 3)] + [
    ("message", n, 4) for n in (1, 2, 12, 16)]


@pytest.mark.parametrize("law,n,nq", CASES)
@pytest.mark.parametrize("mac", ["source", "fma", "nocontract"])
def test_fused_source_compiles(law, n, nq, mac):
    _compile_or_skip()
    from inkling2_metal_compile import compile_unit, translation_unit

    plan, why = K.mlx_order_plan(law, n, nq, 2304, tf32=False, arch="applegpu_g17s")
    assert why is None
    defines = K._mo_defines(plan, K._mo_state_expr(n), "float(q[(j) * MO_D + (d)])",
                            "MO_X(i, d) * mo_inv[(i)]")
    header, source = K.mo_source(plan, ("rms", "scores", "softmax", "mix"), defines, mac=mac)
    inputs = ([(f"b{i}", mx.bfloat16, 2304) for i in range(n - 1)]
              + [("p", mx.float32, 2304), ("q", mx.float32, nq * 2304), ("scale", mx.float32, 1),
                 ("eps", mx.float32, 1)])
    err = compile_unit(translation_unit("mo", inputs, [("out", mx.float32)], source, header),
                       std="metal4.1")
    assert err is None, err


def _probe():
    spec = importlib.util.spec_from_file_location("inkling2_residual_bitwise_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_compiles_as_python():
    py_compile.compile(str(PROBE), doraise=True)


@pytest.mark.parametrize("law,n,nq", [("residual", 12, 3), ("residual", 4, 1), ("message", 8, 4)])
def test_probe_stage_sources_compile(law, n, nq):
    _compile_or_skip()
    from inkling2_metal_compile import compile_unit, translation_unit

    probe = _probe()
    f32 = mx.float32
    for name, inputs, source in (
            ("cur_rms", [("xs", f32, n * 2304)], probe._CUR_RMS),
            ("cur_scores", [("keys", f32, n * 2304), ("qs", f32, nq * 2304), ("scale", f32, 1)],
             probe._CUR_SCORES),
            ("cur_softmax", [("scores", f32, nq * n)], probe._CUR_SOFTMAX),
            ("cur_mix", [("w", f32, nq * n), ("keys", f32, n * 2304)], probe._CUR_MIX)):
        err = compile_unit(translation_unit(name, inputs, [("out", f32)], source,
                                            template=[("N", n), ("D", 2304), ("NQ", nq)]),
                           std="metal4.1")
        assert err is None, (name, err)
    plan, _ = K.mlx_order_plan(law, n, nq, 2304, tf32=False, arch="applegpu_g17s")
    sizes = {"xs": n * 2304, "eps": 1, "scale": 1, "keys": n * 2304, "qs": nq * 2304,
             "sc": nq * n, "w": nq * n}
    for stage in probe.STAGE_GLUE:
        for mac in ("source", "fma", "nocontract"):
            names, header, source = probe.stage_source(plan, stage, mac)
            inputs = [(name, f32, sizes[name]) for name in names]
            err = compile_unit(translation_unit(stage, inputs, [("out", f32)], source, header),
                               std="metal4.1")
            assert err is None, (stage, mac, err)


def test_mlx_order_is_the_default():
    """Pierre 2026-11-26: the bitwise MLX-order mix is the default."""
    assert K.RESIDUAL_MLX_ORDER is True


def test_plan_declines_unknown_release_device_and_wide_threadgroups(monkeypatch):
    """Review 2026-11-26: another MLX release, no device info and a
    threadgroup near MLX's eval-time limit decline (the reference runs)."""
    ok = K.mlx_order_plan("residual", 4, 1, 2304, tf32=False, arch="applegpu_g17s")
    assert ok[1] is None
    assert K.mlx_order_plan("residual", 4, 1, 2304, tf32=False, arch="")[1] == "architecture"
    assert K.mlx_order_plan("residual", 4, 1, 4096, tf32=False, arch="applegpu_g17s")[1] == "width"
    monkeypatch.setattr(K.mx, "__version__", "0.33.0")
    assert K.mlx_order_plan("residual", 4, 1, 2304, tf32=False, arch="applegpu_g17s")[1] == "mlx_version"
