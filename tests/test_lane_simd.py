"""CPU tests for the lane matmul's M1-M4 backend (``lane.simd``).

The Metal kernels run in scripts/lane_matmul_gate.py (``--backend simd``,
any Apple GPU).  Here: backend selection, preparation without a second copy
of the scales, kernel-family routing, the twin check's rerouting, and
installation under the simd law.
"""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.runtime import lane
from mlx2.runtime.lane import installer as inst
from mlx2.runtime.lane import matmul as lm
from mlx2.runtime.lane import simd


def _quantized(k, n, bits, gs, seed=0, dtype=mx.bfloat16):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(dtype)
    module = nn.QuantizedLinear(k, n, bias=False, group_size=gs, bits=bits)
    module.weight, module.scales, module.biases = mx.quantize(w, group_size=gs, bits=bits)
    return module


@pytest.fixture(autouse=True)
def _clean_simd_state():
    simd.mma_one_row.clear()
    simd.bits_fallback.clear()
    yield
    simd.mma_one_row.clear()
    simd.bits_fallback.clear()
    lm.force_backend(None)


def test_no_backend_off_the_gpu_even_when_forced():
    assert lm.backend() is None and not lane.available()
    lm.force_backend("simd")
    assert lm.backend() is None
    with pytest.raises(ValueError, match="backend"):
        lm.force_backend("cuda")


def test_simd_preparation_shares_mlx_scales_and_refuses_unquantized():
    module = _quantized(256, 64, 4, 64)
    lw = lm.prepare(module, "simd")
    assert lw.backend == "simd" and lw.scale_bias is None
    assert lw.scales is module["scales"] and lw.biases is module["biases"]   # no copy
    assert lw.split_k == simd.splits(64, 256) and lw.scales_dtype == mx.bfloat16
    mpp = lm.prepare(module, "mpp")
    assert mpp.backend == "mpp" and mpp.scale_bias is not None and mpp.scales is None
    dense = nn.Linear(128, 64, bias=False)
    with pytest.raises(lane.LaneUnsupported, match="M5 tensor units"):
        lm.prepare(dense, "simd")


@pytest.mark.parametrize("bits,gs,n,k,dtype,expected", [
    (4, 64, 64, 256, mx.bfloat16, "simd4"),
    (4, 32, 64, 256, mx.bfloat16, "simd4"),
    (4, 128, 64, 256, mx.bfloat16, "affine"),       # groups of 128: no matrix kernel
    (4, 64, 60, 256, mx.bfloat16, "affine"),        # N not a multiple of 8
    (4, 64, 64, 256, mx.float16, "affine"),         # fp16 scales
    (5, 64, 64, 256, mx.bfloat16, "simd_bits"),
    (6, 64, 64, 256, mx.bfloat16, "simd_bits"),
    (8, 64, 64, 256, mx.bfloat16, "simd_bits"),
    (8, 32, 64, 256, mx.bfloat16, "affine"),
    (2, 64, 64, 256, mx.bfloat16, "affine"),
    (3, 128, 64, 256, mx.bfloat16, "affine"),
])
def test_kernel_family_is_fixed_per_weight(bits, gs, n, k, dtype, expected):
    assert simd.path(bits, gs, n, k, dtype) == expected


def test_geometry_refusals():
    with pytest.raises(simd.SimdUnsupported, match="affine"):
        simd.check_geometry(bits=4, group_size=64, mode="mxfp4", n=64, k=256,
                            weight_dtype=mx.uint32, scales_dtype=mx.bfloat16)
    with pytest.raises(simd.SimdUnsupported, match="groups of 16"):
        simd.check_geometry(bits=4, group_size=16, mode="affine", n=64, k=256,
                            weight_dtype=mx.uint32, scales_dtype=mx.bfloat16)
    with pytest.raises(simd.SimdUnsupported, match="bf16"):
        simd.qmm(mx.zeros((1, 256), dtype=mx.float16), mx.zeros((64, 32), dtype=mx.uint32),
                 mx.zeros((64, 4), dtype=mx.bfloat16), mx.zeros((64, 4), dtype=mx.bfloat16), 64, 4)


def test_splits_depend_on_the_weight_shape_only():
    assert [simd.splits(n, 5120) for n in (64, 65, 6144, 6145, 248320)] == [32, 16, 16, 8, 8]


def test_fragment_kernel_reads_prepared_inputs():
    source = simd._fragment_source(simd._MMA)
    assert "XF2[" in source and "XS[size_t(xr0[rt])" in source
    assert "LOAD8(xr0[rt]" not in source


def _fake_qmm(differ):
    def qmm(x2, weight, scales, biases, group_size, bits, *, kind=None):
        rows, n = int(x2.shape[0]), int(weight.shape[0])
        value = 1.0 if (differ and kind == "scalar") else 0.0
        return mx.full((rows, n), value, dtype=mx.bfloat16)
    return qmm


def test_twin_check_reroutes_a_4bit_shape_that_differs(monkeypatch):
    module = _quantized(256, 64, 4, 64)
    monkeypatch.setattr(simd, "qmm", _fake_qmm(differ=False))
    assert simd.check(module["weight"], module["scales"], module["biases"], 64, 4)
    assert not simd.mma_one_row
    monkeypatch.setattr(simd, "qmm", _fake_qmm(differ=True))
    assert not simd.check(module["weight"], module["scales"], module["biases"], 64, 4)
    assert simd.mma_one_row == {(64, 256, 64)}
    assert not simd._scalar_kind(1, 64, 256, 64)       # 1-row calls now take the matrix kernel


def test_twin_check_sends_a_differing_high_bit_shape_to_affine_rows(monkeypatch):
    module = _quantized(256, 64, 6, 64)
    assert simd.path(6, 64, 64, 256, mx.bfloat16) == "simd_bits"
    monkeypatch.setattr(simd, "qmm", _fake_qmm(differ=True))
    assert not simd.check(module["weight"], module["scales"], module["biases"], 64, 6)
    assert simd.path(6, 64, 64, 256, mx.bfloat16) == "affine"


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = _quantized(256, 64, 4, 64, seed=1)
        self.k_proj = _quantized(256, 32, 4, 64, seed=2)
        self.v_proj = _quantized(256, 32, 4, 64, seed=3)
        self.o_proj = _quantized(64, 256, 4, 64, seed=4)
        self.head = nn.Linear(256, 128, bias=False)      # unquantized: stock under simd


def test_install_under_the_simd_law(monkeypatch):
    monkeypatch.setattr(lm, "backend", lambda: "simd")
    monkeypatch.setattr(inst, "backend", lambda: "simd")
    checked = []
    monkeypatch.setattr(simd, "check", lambda w, s, b, gs, bits: checked.append(
        (int(w.shape[0]), gs, bits)) or True)
    model = _Block()
    weights = {name: model[name]["weight"] for name in ("q_proj", "k_proj", "v_proj")}
    receipt = lane.install(model, min_rows=1)
    assert receipt["backend"] == "simd"
    assert receipt["law_id"] == "lane-simd-v1+grouped"
    assert receipt["covered"] == {"affine-q4-g64": 4}
    assert receipt["refused"] == {"unquantized weights need the M5 tensor units": 1}
    group = inst._group(model.q_proj)
    assert group is not None and group.lw.backend == "simd" and group.lw.n == 128
    assert group.lw.scale_bias is None and group.lw.scales.shape == (128, 4)
    for name, before in weights.items():                 # members are views of the stack
        assert mx.array_equal(model[name]["weight"], before)
        assert inst._prepared(model[name]).backend == "simd"
    # One twin check per launched shape: q (64), k/v (32) alone, the stack
    # (128) and o_proj (256).
    assert sorted(n for n, _gs, _bits in checked) == [32, 64, 128, 256]
    assert receipt["simd_twins"] == {"shapes": 4, "rerouted": 0}
    assert inst.law_id(4, "simd") == "lane-simd-v1+stock-below-4"
    assert inst.law_id(1, "mpp") == lane.LAW_ID
    assert lane.uninstall(model) == 4


def test_formats_of_different_backends_never_stack():
    a = lm.prepare(_quantized(256, 64, 4, 64), "simd")
    b = lm.prepare(_quantized(256, 64, 4, 64), "mpp")
    assert inst._format_key(a) != inst._format_key(b)


def test_auto_keeps_m1_m4_hosts_on_stock_until_opted_in():
    from mlx2.runtime.lane import policy

    detected = {"moe": False, "formats": {"q4": 10}, "backend": "simd"}
    auto = policy.resolve(detected, mode="auto")
    assert auto["mode"] == "off" and auto["sources"]["mode"] == "backend:simd"
    assert policy.resolve(detected, mode="exact")["mode"] == "exact"
    assert policy.resolve(detected, overrides={"mode": "crossover"})["mode"] == "crossover"
    m5 = policy.resolve({**detected, "backend": "mpp"}, mode="auto")
    assert m5["mode"] == "crossover" and m5["sources"]["mode"] == "builtin"


def test_reinstalling_under_the_other_backend_starts_over(monkeypatch):
    """A receipt must never name one law while prepared weights run another."""
    monkeypatch.setattr(simd, "check", lambda *a, **k: True)
    model = _Block()
    for name in ("simd", "mpp"):
        monkeypatch.setattr(lm, "backend", lambda name=name: name)
        monkeypatch.setattr(inst, "backend", lambda name=name: name)
        receipt = lane.install(model, min_rows=1)
        assert receipt["backend"] == name
        assert receipt["law_id"].startswith(inst.LAW_IDS[name])
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert inst._prepared(model[attr]).backend == name
        assert inst._group(model.q_proj).lw.backend == name
    lane.uninstall(model)


class _HighBit(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = _quantized(256, 64, 6, 64, seed=9)


def test_a_high_bit_affine_fallback_joins_the_law_on_every_install(monkeypatch):
    monkeypatch.setattr(lm, "backend", lambda: "simd")
    monkeypatch.setattr(inst, "backend", lambda: "simd")
    monkeypatch.setattr(simd, "qmm", _fake_qmm(differ=True))      # twins differ on this GPU
    model = _HighBit()
    first = lane.install(model, min_rows=1)
    assert first["simd_twins"] == {"shapes": 1, "rerouted": 1, "affine": ["64x256q6g64"]}
    assert "+simd-affine[" in first["law_id"]
    again = lane.install(model, min_rows=1)                        # route persists: still reported
    assert again["simd_twins"] == first["simd_twins"] and again["law_id"] == first["law_id"]
    lane.uninstall(model)


def test_a_4bit_one_row_reroute_is_reported_without_changing_the_law(monkeypatch):
    monkeypatch.setattr(lm, "backend", lambda: "simd")
    monkeypatch.setattr(inst, "backend", lambda: "simd")
    monkeypatch.setattr(simd, "qmm", _fake_qmm(differ=True))
    model = _Block()
    receipt = lane.install(model, min_rows=1, groups=())
    # q (64), k and v (32, one shape) and o (256): three shapes, all 4-bit.
    assert receipt["simd_twins"]["rerouted"] == 3 and "affine" not in receipt["simd_twins"]
    assert receipt["law_id"] == "lane-simd-v1"
    lane.uninstall(model)


def test_admission_matches_tensorfold_layout_checks():
    base = {"bits": 4, "group_size": 64, "mode": "affine", "n": 64, "k": 256,
            "weight_dtype": mx.uint32, "scales_dtype": mx.bfloat16}
    simd.check_geometry(**base, biases_dtype=mx.bfloat16)
    with pytest.raises(simd.SimdUnsupported, match="share one dtype"):
        simd.check_geometry(**base, biases_dtype=mx.float16)
    with pytest.raises(simd.SimdUnsupported, match="rank 2"):
        simd.check_geometry(**base, weight_ndim=3)
    # bf16 scales with non-bf16 biases never reach the bf16-only matrix kernels
    assert simd.path(4, 64, 64, 256, mx.bfloat16, mx.float16) == "affine"
    module = _quantized(256, 64, 4, 64)
    module.biases = module.biases.astype(mx.float16)
    with pytest.raises(lane.LaneUnsupported, match="share one dtype"):
        lm.prepare(module, "simd")
