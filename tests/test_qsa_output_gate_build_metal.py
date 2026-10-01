"""Opt-in native bit gate for the two existing QSA output-gate epilogues.

Exclusive CPG GPU ownership and both GPU locks are externally mandatory before
setting MLX2_TEST_OPTIONAL_METAL=1. This test does NOT acquire GPU ownership,
locks or a queue slot itself.

Scope: epilogue build arithmetic only. Each native cell runs one actual,
unpatched production entry point twice on the same geometry, ungated then
gated, and requires the gated raw bits to equal the ungated output passed
through the ordinary ``mlx_apply_output_gate`` (standalone ``mx.sigmoid`` then
a storage-dtype multiply). That isolates the fused sigmoid and multiply from
the combine math; it does not qualify the ungated merge, attention, a model,
serving selection or performance.

Routes:
  sequential_fused_merge  merge.combine_indexed_partials, fused merge and gate
                          switches on.
  native_sdpa_merge       indexed._combine_sdpa_partials, fused merge off,
                          output_gate passed.

Host preparation (standard library only, importable without mlx): gate raw
encodings, the per-partial m/l tables, the attention amplitude grid and the
per-partial O scale. BF16 and FP16 gates exhaust every finite encoding; FP32 is
a bounded deterministic finite raw-bit sample plus explicit edges and is not
exhaustive. Each vector is padded to the cell size by cycling its edge
encodings. The native body only uploads these values, forms O as an exact
fp32 dyadic product cast to the storage dtype, views raw bits and calls the
production entry points. mlx, numpy and the project runtime are imported only
inside the native body, after the skip gate.

Geometry: B2 H2 L64 D256 with P128 partials (the SDPA combine's fixed block
count). This is an activation/epilogue sweep geometry, not a verify window.
"""
import os
import struct

import pytest

NATIVE_ENV = "MLX2_TEST_OPTIONAL_METAL"
MERGE_ENV = "MLX_QWEN4_QSA_INDEXED_FUSED_MERGE"
GATE_ENV = "MLX_QWEN4_QSA_INDEXED_FUSED_GATE"
BATCH, HEADS, LENGTH, DIM, PARTIALS = 2, 2, 64, 256, 128
ELEMENTS = BATCH * LENGTH * HEADS * DIM
ROUTES = ("sequential_fused_merge", "native_sdpa_merge")
DTYPES = ("bfloat16", "float16", "float32")
BF16_CORRECTION = -6.84375
NEIGHBOURS = 4
FP32_SEED = 0x51A0_6A7E
FP32_STRATIFIED_PER_EXPONENT = 64
FP32_DENSE_EXPONENTS = range(119, 133)
FP32_DENSE_PER_EXPONENT = 1160
# Dyadic values with at most four significant bits, so every O product below
# is exact in bf16 and fp16 and |O| stays far below the fp16 maximum.
AMPLITUDES = (1.0, -0.75, 0.0, 2.5, -1.5, 0.125, -64.0, 2.0**-10, -3.25, 48.0, -0.0, 0.3125, -6.0)
PARTIAL_SCALES = (1.0, 0.5, -0.75, 1.25)
_FORMATS = {"bfloat16": (16, 8, 7), "float16": (16, 5, 10), "float32": (32, 8, 23)}
_MASK64 = (1 << 64) - 1

pytestmark = pytest.mark.skipif(
    os.environ.get(NATIVE_ENV) != "1",
    reason="explicit exclusive GPU gate required (CPG ownership and both locks held externally)",
)


def encode(label, value):
    """Return the exact raw encoding of a Python float, refusing rounding."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if struct.unpack("<f", struct.pack("<I", bits))[0] != value:
        raise ValueError(f"{value!r} is not exact in float32")
    if label == "float32":
        return bits
    if label == "bfloat16":
        if bits & 0xFFFF:
            raise ValueError(f"{value!r} is not exact in bfloat16")
        return bits >> 16
    raw = struct.unpack("<H", struct.pack("<e", value))[0]
    if decode(label, raw) != value:
        raise ValueError(f"{value!r} is not exact in float16")
    return raw


def decode(label, raw):
    if label == "float16":
        return struct.unpack("<e", struct.pack("<H", raw))[0]
    bits = raw << 16 if label == "bfloat16" else raw
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def is_finite(label, raw):
    (_, exponent_bits, mantissa_bits) = _FORMATS[label]
    all_ones = (1 << exponent_bits) - 1
    return (raw >> mantissa_bits) & all_ones != all_ones


def edge_encodings(label):
    """Signed zeros, subnormal/normal/max limits, one, half and the correction neighbourhood."""
    (width, exponent_bits, mantissa_bits) = _FORMATS[label]
    sign = 1 << (width - 1)
    max_exponent = (1 << exponent_bits) - 2
    mantissa = (1 << mantissa_bits) - 1
    magnitudes = (
        0,
        1,
        mantissa,
        1 << mantissa_bits,
        (max_exponent << mantissa_bits) | mantissa,
        encode(label, 1.0),
        encode(label, 0.5),
    )
    edges = [raw for magnitude in magnitudes for raw in (magnitude, sign | magnitude)]
    correction = encode(label, BF16_CORRECTION)
    edges += [correction + step for step in range(-NEIGHBOURS, NEIGHBOURS + 1)]
    if label == "float32":
        widened = encode("bfloat16", BF16_CORRECTION)
        edges += [(widened + step) << 16 for step in range(-NEIGHBOURS, NEIGHBOURS + 1)]
    return list(dict.fromkeys(edges))


def _splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    word = state
    word = ((word ^ (word >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    word = ((word ^ (word >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (state, word ^ (word >> 31))


def fp32_sample():
    """Every finite exponent per sign, plus a dense band over |x| in [2**-8, 2**6)."""
    state = FP32_SEED
    sample = []
    plan = [(exponent, FP32_STRATIFIED_PER_EXPONENT) for exponent in range(255)]
    plan += [(exponent, FP32_DENSE_PER_EXPONENT) for exponent in FP32_DENSE_EXPONENTS]
    for (exponent, count) in plan:
        for sign in (0, 1):
            for _ in range(count):
                (state, word) = _splitmix64(state)
                sample.append((sign << 31) | (exponent << 23) | (word & 0x7FFFFF))
    return sample


def gate_vector(label):
    """Raw gate encodings for one native cell, labelled with scope and counts."""
    width = _FORMATS[label][0]
    edges = edge_encodings(label)
    if label == "float32":
        scope = "sampled-finite"
        unique = list(dict.fromkeys(edges + fp32_sample()))
    else:
        scope = "exhaustive-finite"
        unique = [raw for raw in range(1 << width) if is_finite(label, raw)]
    if len(unique) > ELEMENTS:
        raise ValueError(f"{label} gate vector exceeds the {ELEMENTS}-cell geometry")
    padding = [edges[index % len(edges)] for index in range(ELEMENTS - len(unique))]
    return {
        "dtype": label,
        "storage": f"uint{width}",
        "scope": scope,
        "unique_count": len(unique),
        "pad_count": len(padding),
        "pad_source": "edge encodings, cyclic",
        "count": ELEMENTS,
        "edges": edges,
        "raw": unique + padding,
    }


def gate_cell(index):
    """Map a flat (B, L, H*D) gate index to its (b, h, l, d) attention cell."""
    (b, rest) = divmod(index, LENGTH * HEADS * DIM)
    (l, rest) = divmod(rest, HEADS * DIM)
    (h, d) = divmod(rest, DIM)
    return (b, h, l, d)


def amplitude(b, h, l, d):
    return AMPLITUDES[(d + 3 * l + 5 * h + 7 * b) % len(AMPLITUDES)]


def partial_tables():
    """Safe finite partial state: m in [-1.75, 0], l in [1, 1.75], O = amplitude * scale."""
    return {
        "m": [-((partial * 5) % 8) * 0.25 for partial in range(PARTIALS)],
        "l": [1.0 + (partial % 4) * 0.25 for partial in range(PARTIALS)],
        "scale": [PARTIAL_SCALES[partial % len(PARTIAL_SCALES)] for partial in range(PARTIALS)],
        "amplitude": [
            amplitude(b, h, l, d)
            for b in range(BATCH)
            for h in range(HEADS)
            for l in range(LENGTH)
            for d in range(DIM)
        ],
    }


def _native_inputs(mx, dtype_label):
    dtype = getattr(mx, dtype_label)
    tables = partial_tables()
    shape = (BATCH, HEADS, LENGTH, PARTIALS)
    m = mx.contiguous(mx.broadcast_to(mx.array(tables["m"], dtype=mx.float32), shape))
    l = mx.contiguous(mx.broadcast_to(mx.array(tables["l"], dtype=mx.float32), shape))
    amp = mx.array(tables["amplitude"], dtype=mx.float32).reshape(BATCH, HEADS, LENGTH, 1, DIM)
    scale = mx.array(tables["scale"], dtype=mx.float32).reshape(1, 1, 1, PARTIALS, 1)
    o = (amp * scale).astype(dtype)
    vector = gate_vector(dtype_label)
    storage = getattr(mx, vector["storage"])
    raw = mx.array(vector["raw"], dtype=storage).reshape(BATCH, LENGTH, HEADS * DIM)
    gate = mx.view(raw, dtype)
    mx.eval(m, l, o, raw, gate)
    assert gate.dtype == dtype and gate.shape == (BATCH, LENGTH, HEADS * DIM)
    assert mx.view(gate, storage).reshape(-1).tolist() == vector["raw"]
    return (dtype, storage, m, l, o, gate)


def _check_output(value, dtype):
    assert value.dtype == dtype
    assert value.shape == (BATCH, HEADS, LENGTH, DIM)
    assert value.size == ELEMENTS


def _assert_bits_equal(mx, np, storage, actual, expected, ungated, gate, label):
    actual_bits = np.array(mx.view(actual, storage)).reshape(-1)
    expected_bits = np.array(mx.view(expected, storage)).reshape(-1)
    ungated_bits = np.array(mx.view(ungated, storage)).reshape(-1)
    gate_bits = np.array(
        mx.view(gate, storage).reshape(BATCH, LENGTH, HEADS, DIM).transpose(0, 2, 1, 3)
    ).reshape(-1)
    bad = np.flatnonzero(actual_bits != expected_bits)
    report = [
        (int(i), hex(gate_bits[i]), hex(ungated_bits[i]), hex(expected_bits[i]), hex(actual_bits[i]))
        for i in bad[:8]
    ]
    assert bad.size == 0, f"{label}: {bad.size} raw-bit mismatches (cell, gate, attention, expected, actual): {report}"
    # Falsifier: an ignored gate would leave the ungated bits in place.
    assert int(np.count_nonzero(actual_bits != ungated_bits)) > 0, f"{label}: gate had no effect"


def _sequential_cell(mx, merge, dtype, m, l, o, gate):
    fallbacks = []
    with pytest.MonkeyPatch.context() as env:
        env.setenv(MERGE_ENV, "1")
        env.setenv(GATE_ENV, "0")
        merge.fused_merge_status(reset=True)
        ungated = merge.combine_indexed_partials(
            m, l, o, output_dtype=dtype, on_fallback=lambda: fallbacks.append("ungated")
        )
        mx.eval(ungated)
        status = merge.fused_merge_status()
        assert status["engaged"] and status["fallbacks"] == 0, status
        assert not status["gate_engaged"] and status["gate_path"] is None, status
        expected = merge.mlx_apply_output_gate(ungated, gate)
        mx.eval(expected)
        env.setenv(GATE_ENV, "1")
        merge.fused_merge_status(reset=True)
        assert merge.fused_merge_status() == {
            "engaged": False, "fallbacks": 0, "candidate": None, "gate_engaged": False, "gate_path": None,
            "gate_refusals": 0, "gate_last_refusal": None,
        }
        actual = merge.combine_indexed_partials(
            m, l, o, output_dtype=dtype, output_gate=gate, on_fallback=lambda: fallbacks.append("gated")
        )
        mx.eval(actual)
        status = merge.fused_merge_status()
    assert not fallbacks, fallbacks
    assert status["engaged"] and status["fallbacks"] == 0, status
    assert status["gate_engaged"] is True and status["gate_path"] == "sequential_fused_merge", status
    return (ungated, expected, actual)


def _native_sdpa_cell(mx, merge, indexed, dtype, m, l, o, gate):
    sentinel = mx.array([1], dtype=mx.uint32)
    with pytest.MonkeyPatch.context() as env:
        env.setenv(MERGE_ENV, "0")
        env.setenv(GATE_ENV, "0")
        merge.fused_merge_status(reset=True)
        (ungated, returned) = indexed._combine_sdpa_partials(m, l, o, sentinel, output_dtype=dtype)
        assert returned is sentinel
        mx.eval(ungated, returned)
        status = merge.fused_merge_status()
        assert not status["engaged"] and status["fallbacks"] == 0, status
        assert not status["gate_engaged"] and status["gate_path"] is None, status
        expected = merge.mlx_apply_output_gate(ungated, gate)
        mx.eval(expected)
        env.setenv(GATE_ENV, "1")
        merge.fused_merge_status(reset=True)
        assert merge.fused_merge_status()["gate_path"] is None
        (actual, returned) = indexed._combine_sdpa_partials(
            m, l, o, sentinel, output_dtype=dtype, output_gate=gate
        )
        assert returned is sentinel
        mx.eval(actual, returned)
        assert returned.tolist() == [1]
        status = merge.fused_merge_status()
    assert not status["engaged"] and status["fallbacks"] == 0, status
    assert status["gate_engaged"] is True and status["gate_path"] == "native_sdpa_merge", status
    return (ungated, expected, actual)


@pytest.mark.parametrize("dtype_label", DTYPES)
@pytest.mark.parametrize("route", ROUTES)
def test_native_output_gate_epilogue_bits(route, dtype_label):
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import qwen4_qsa_indexed as indexed
    from mlx2.runtime.models import qwen4_qsa_indexed_merge as merge

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        (dtype, storage, m, l, o, gate) = _native_inputs(mx, dtype_label)
        if route == "sequential_fused_merge":
            cell = _sequential_cell(mx, merge, dtype, m, l, o, gate)
        else:
            cell = _native_sdpa_cell(mx, merge, indexed, dtype, m, l, o, gate)
        (ungated, expected, actual) = cell
        for value in cell:
            _check_output(value, dtype)
        _assert_bits_equal(mx, np, storage, actual, expected, ungated, gate, f"{route}/{dtype_label}")
    finally:
        # Status has no public restore; leave it cleared rather than holding this cell's evidence.
        merge.fused_merge_status(reset=True)
        mx.set_default_device(previous)
