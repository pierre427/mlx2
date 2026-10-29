"""CPU contracts for the opt-in metal_q4native TensorFold prefill backend.

No kernel runs here: sources are generated (and compiled offline with
``xcrun metal`` when the compiler is installed) and launches are intercepted.
GPU exactness/timing lives in scripts/check_prefill_q4native.py.
"""

import hashlib
import shutil
import subprocess
from collections import Counter

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import prefill_metal
from mlx2.runtime.models import prefill_metal_q4native as q4native
from mlx2.runtime.models import tensorfold_prefill as prefill
from mlx2.runtime.prefill_plan import apc_prefill_fingerprint, execution_identity


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def quantized(k=64, n=33, bias=False):
    layer = nn.Linear(k, n, bias=bias)
    layer.weight = layer.weight.astype(mx.bfloat16)
    if bias:
        layer.bias = layer.bias.astype(mx.bfloat16)
    return nn.QuantizedLinear.from_linear(layer, group_size=64, bits=4)


# Generated metal-backend sources recorded at 026a67023, before metal_q4native
# existed.  The default tile must keep producing exactly these bytes.
METAL_GOLDEN = {
    "header": "8eab3ec0f3d6e379fb194436d16e950801189c3f212028d8ead49238247fbad6",
    "project_5120": "dc7f60e4e1ca6957e74231d354e3458393fcae924caf0a58beed8006e9374e64",
    "project_group4": "46efeac8da9ae86855f70784cb21a2b25a895fcbb6c231915e65e1bba45fe908",
    "swiglu_17408": "760453363ecd7fe3873715876960f21c12ef023fde844352adb76786c73cd88d",
}


def test_metal_backend_sources_are_byte_identical():
    assert _sha(prefill_metal.HEADER) == METAL_GOLDEN["header"]
    assert _sha(prefill_metal.source_for((5120,), (False,))[0]) == METAL_GOLDEN["project_5120"]
    assert (
        _sha(prefill_metal.source_for((6144, 1024, 48, 48), (False, True, False, False))[0])
        == METAL_GOLDEN["project_group4"]
    )
    assert _sha(prefill_metal.swiglu_source(17408, (False, False))[0]) == METAL_GOLDEN["swiglu_17408"]


def test_metal_kernel_construction_is_unchanged(monkeypatch):
    seen = []
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda **kw: seen.append(kw) or kw)
    prefill._kernel.cache_clear()
    prefill._swiglu_kernel.cache_clear()
    try:
        prefill._kernel((5120,), (False,))
        prefill._swiglu_kernel(17408, (False, False))
        prefill._kernel((5120,), (False,), "metal_q4native")
        prefill._swiglu_kernel(17408, (False, False), "metal_q4native")
    finally:
        prefill._kernel.cache_clear()
        prefill._swiglu_kernel.cache_clear()
    metal_project, metal_swiglu, native_project, native_swiglu = seen
    assert metal_project["name"] == f"mlx2_tensorfold_prefill_{METAL_GOLDEN['project_5120'][:16]}"
    assert metal_swiglu["name"] == f"mlx2_prefill_swiglu_{METAL_GOLDEN['swiglu_17408'][:16]}"
    assert metal_project["header"] is prefill_metal.HEADER
    assert _sha(metal_project["source"]) == METAL_GOLDEN["project_5120"]
    assert native_project["name"].startswith("mlx2_tensorfold_prefill_q4native_")
    assert native_swiglu["name"].startswith("mlx2_prefill_swiglu_q4native_")
    assert native_project["source"] != metal_project["source"]
    # Same bindings, so project()/project_swiglu() launch both laws identically.
    assert native_project["input_names"] == metal_project["input_names"]
    assert native_swiglu["input_names"] == metal_swiglu["input_names"]


@pytest.mark.parametrize(
    "case",
    [((7,), (False,)), ((33, 48, 64, 7), (True, False, True, False))],
)
def test_q4native_source_reads_codes_in_place_and_applies_affine_law(case):
    source, inputs = q4native.source_for(*case)
    assert inputs == prefill_metal.source_for(*case)[1]
    blocks = len(case[0])
    assert source.count("tensor<device uint4b_format") == blocks
    # No threadgroup staging of weights or activations.
    assert "threadgroup bfloat" not in source
    assert "(K / 8) + k / 8]" not in source
    # Unscaled group product, affine applied per group in fixed order.
    assert source.count("matmul2d_descriptor::mode::multiply)") == blocks
    assert "multiply_accumulate" not in source
    assert source.count("matmul2d_descriptor(TM, TN, GS, false, true, false,") == blocks
    assert "fma(sb0[g & 1][n][0], P0[j], fma(sb0[g & 1][n][1], xs, c0[j]))" in source
    assert "simd_shuffle_xor(xq_sum, ushort(1))" in source
    assert source.count("threadgroup_barrier") == blocks


def test_q4native_swiglu_has_two_laws_and_the_same_epilogue():
    source, inputs = q4native.swiglu_source(33, (True, False))
    assert inputs == prefill_metal.swiglu_source(33, (True, False))[1]
    assert source.count("tensor<device uint4b_format") == 2
    assert "fma(sb1[g & 1][n][0], P1[j], fma(sb1[g & 1][n][1], xs, c1[j]))" in source
    metal = prefill_metal.swiglu_source(33, (True, False))[0]
    epilogue = metal[metal.index("bfloat value = bfloat(c0[j]);"):]
    assert epilogue.strip() in source


def test_policy_accepts_q4native_only_with_tensorfold_prefill():
    policy = FlashNextPolicy(tensorfold_prefill=True, tensorfold_prefill_backend="metal_q4native")
    assert policy.as_dict()["tensorfold_prefill_backend"] == "metal_q4native"
    assert FlashNextPolicy.from_mapping(policy.as_dict()) == policy
    with pytest.raises(ValueError, match="requires tensorfold_prefill"):
        FlashNextPolicy.from_mapping({"tensorfold_prefill_backend": "metal_q4native"})
    with pytest.raises(ValueError, match="metal_q4native"):
        FlashNextPolicy.from_mapping(
            {"tensorfold_prefill": True, "tensorfold_prefill_backend": "q4native"}
        )
    assert "tensorfold_prefill_backend" not in FlashNextPolicy(tensorfold_prefill=True).as_dict()


def test_install_receipt_identity_separates_the_law():
    receipts = {}
    for backend in ("metal", "metal_q4native"):
        model = nn.Module()
        model.dense = quantized()
        model.eval()
        receipts[backend] = prefill.install(model, backend=backend)
        assert model.dense._prefill_backend == backend
    metal, native = receipts["metal"], receipts["metal_q4native"]
    assert metal["kernel"] == "q4g64_prefill_m32n64k64_nax"
    assert native["kernel"] == q4native.KERNEL_IDENTITY
    assert native["tile"] == metal["tile"] == [32, 64, 64]
    assert native["names_sha256"] == metal["names_sha256"]
    plain = "text-token-v1"
    a = apc_prefill_fingerprint(plain, execution_identity(metal))
    b = apc_prefill_fingerprint(plain, execution_identity(native))
    assert a != b and a != plain and b != plain
    assert execution_identity(native)["projection"]["kernel"] == q4native.KERNEL_IDENTITY


def test_unknown_backend_is_refused():
    model = nn.Module()
    model.dense = quantized()
    with pytest.raises(ValueError, match="metal_q4native"):
        prefill.install(model, backend="q4native")
    with pytest.raises(ValueError, match="metal_q4native"):
        prefill.project(mx.ones((19, 64), dtype=mx.bfloat16), [model.dense], backend="native")


def test_q4native_launch_routes_its_own_kernel_with_the_metal_grid(monkeypatch):
    modules = [quantized(n=33, bias=True), quantized(n=7)]
    x = mx.random.normal((2, 37, 64)).astype(mx.bfloat16)
    expected = [m(x) for m in modules]
    seen = {}

    def kernel(widths, biases, backend="metal"):
        seen["backend"] = backend

        def call(**kwargs):
            seen.update(kwargs)
            inputs, i, ys = kwargs["inputs"], 2, []
            for biased in biases:
                w, s, b = inputs[i : i + 3]
                y = mx.quantized_matmul(
                    inputs[0], w, scales=s, biases=b, transpose=True, group_size=64, bits=4
                )
                i += 3
                if biased:
                    y = y + inputs[i]
                    i += 1
                ys.append(y)
            return ys

        return call

    monkeypatch.setattr(prefill, "_kernel", kernel)
    monkeypatch.setattr(mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)
    monkeypatch.setattr(mx, "device_info", lambda: {"device_name": "Apple M5 Max"})
    actual = prefill.project(x, modules, backend="metal_q4native")
    assert seen["backend"] == "metal_q4native"
    assert seen["grid"] == (128 * ((2 * 37 + 31) // 32), (33 + 63) // 64 + 1, 1)
    for a, b in zip(actual, expected):
        assert mx.array_equal(a, b).item()


def test_installed_linear_and_mlp_hooks_dispatch_the_selected_law(monkeypatch):
    from mlx2.runtime.models.qwen3_next import Qwen3NextMLP

    mlp = Qwen3NextMLP(64, 128)
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = quantized(n=128), quantized(n=128), quantized(k=128, n=64)
    mlp.eval()
    receipt = prefill.install(mlp, backend="metal_q4native")
    monkeypatch.setattr(prefill, "_admitted", lambda x: True)
    calls = Counter()

    def project(x, modules, backend="metal"):
        calls[("project", backend)] += 1
        return tuple(nn.QuantizedLinear.__call__(m, x) for m in modules)

    def swiglu(x, gate, up, backend="metal"):
        from mlx2.runtime.models.activations import swiglu as act

        calls[("swiglu", backend)] += 1
        return act(nn.QuantizedLinear.__call__(gate, x), nn.QuantizedLinear.__call__(up, x))

    monkeypatch.setattr(prefill, "project", project)
    monkeypatch.setattr(prefill, "project_swiglu", swiglu)
    x = mx.random.normal((1, 19, 64)).astype(mx.bfloat16)
    mlp(x)
    assert calls[("swiglu", "metal_q4native")] == 1
    assert calls[("project", "metal_q4native")] == 1      # down_proj via the linear hook
    assert not any(backend == "metal" for _, backend in calls)
    assert receipt["kernel"] == q4native.KERNEL_IDENTITY


def _metal_compiler():
    return shutil.which("xcrun") and not subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True, check=False
    ).returncode


@pytest.mark.parametrize("kind", ["project", "swiglu"])
def test_q4native_sources_compile_offline(tmp_path, kind):
    if not _metal_compiler():
        pytest.skip("Metal compiler unavailable")
    if kind == "project":
        source, names = q4native.source_for((33, 6144), (True, False))
        outputs = ["Y0", "Y1"]
    else:
        source, names = q4native.swiglu_source(17408, (False, True))
        outputs = ["OUT"]
    arguments = [
        "constant int& M" if name == "M"
        else f"const device {'uint' if name.startswith('W') else 'bfloat'}* {name}"
        for name in names
    ]
    arguments += [f"device bfloat* {name}" for name in outputs]
    arguments += [
        "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]",
        "uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]",
    ]
    path = tmp_path / "q4native.metal"
    path.write_text(
        "#include <metal_stdlib>\n" + q4native.HEADER + "\nkernel void check("
        + ",\n".join(arguments) + ") {\nconstexpr int K=5120;\n" + source + "\n}\n"
    )
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-c", str(path),
         "-o", str(tmp_path / "q4native.air")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
