"""CPU contracts for native 2-bit reads and MX formats in the lane matmul."""

import pytest
from mlx import nn

from mlx2.runtime.lane import matmul as lm
from mlx2.runtime.lane import policy


@pytest.fixture
def native_2bit():
    saved = lm.NATIVE_2BIT[0]
    yield lm.NATIVE_2BIT
    lm.NATIVE_2BIT[0] = saved


@pytest.mark.parametrize("group_size", lm.GROUP_SIZES)
def test_native_2bit_reads_in_place_and_staged_unpacks(native_2bit, group_size):
    native_2bit[0] = True
    assert "wbuf32" not in lm.main_source(2, group_size)
    native_2bit[0] = False
    assert "wbuf32" in lm.main_source(2, group_size)


@pytest.mark.parametrize("group_size", lm.GROUP_SIZES)
def test_native_2bit_keeps_the_staged_split_k_law(native_2bit, group_size):
    # The K slices fix the summation order; the read path must not move them.
    splits = []
    for flag in (True, False):
        native_2bit[0] = flag
        splits.append([lm.split_k(n, k, group_size, 2) for n, k in
                       ((1024, 2048), (4096, 1024), (6144, 5120), (248320, 5120))])
    assert splits[0] == splits[1]


def test_other_staged_formats_still_unpack(native_2bit):
    native_2bit[0] = True
    for bits in (3, 5, 6):
        assert "wbuf32" in lm.main_source(bits, 64)


@pytest.mark.parametrize("mode", sorted(lm.MX_MODES))
def test_mx_source_decodes_in_registers_with_e8m0_scales(mode):
    source = lm.main_source(lm.MX_MODES[mode][1], lm.MX_GROUP_SIZE, None, mode)
    assert "get_right_input_cooperative_tensor<XT, half, float>" in source
    assert "E8M0" in source and "0x1p-127f" in source
    assert "xs0" not in source          # no affine bias term
    assert "wbuf32" not in source


@pytest.mark.parametrize("kwargs, message", [
    ({"bits": 4, "group_size": 32, "mode": "mxfp8"}, "8-bit"),
    ({"bits": 8, "group_size": 32, "mode": "mxfp8", "weight_dtype": "uint8"}, "uint32"),
])
def test_mx_geometry_refusals(kwargs, message):
    import mlx.core as mx

    args = {"n": 64, "k": 128, "weight_dtype": mx.uint32, "scales_dtype": mx.uint8, **kwargs}
    if isinstance(args["weight_dtype"], str):
        args["weight_dtype"] = getattr(mx, args["weight_dtype"])
    with pytest.raises(lm.LaneUnsupported, match=message):
        lm.check_geometry(**args)


def test_mx_lane_weights_name_their_format():
    lw = lm.LaneWeights(8, 32, 64, 128, 1, None, None, mode="mxfp8")
    assert lw.format == "mxfp8-g32"
    assert lm.LaneWeights(4, 64, 64, 128, 1, None, None).format == "affine-q4-g64"


@pytest.mark.parametrize("mode, bits", [("mxfp8", 8), ("mxfp4", 4)])
def test_mx_modules_are_their_own_opt_in_format_class(mode, bits):
    module = nn.QuantizedLinear(64, 64, bias=False, group_size=32, bits=bits, mode=mode)
    assert policy.format_class(module) == mode
    # No builtin threshold: an MX projection is installed only on request.
    assert mode not in policy.BUILTIN["min_rows"]
    assert mode in policy.OPT_IN_FORMAT_CLASSES
    affine = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=bits)
    assert policy.format_class(affine) == f"q{bits}"


def test_mx_requires_tensor_units():
    module = nn.QuantizedLinear(64, 64, bias=False, group_size=32, bits=8, mode="mxfp8")
    with pytest.raises(lm.LaneUnsupported, match="tensor units"):
        lm.prepare(module, "simd")
