"""omlx-layout V4.1 admission, affine Engram rows and expert slicing (CPU only)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from mlx2.adapters.deepseek_v41_omlx_artifact import (
    _affine_geometry,
    _module_spec,
    is_omlx_config,
)
from mlx2.runtime.models.deepseek_v41_engram import (
    EngramRows,
    decode_affine4_rows,
    file_stamp,
)

CHECKPOINT = Path.home() / ("mlx-models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-MLX")


def _bf16(value):
    bits = np.ascontiguousarray(value, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def _quantized(rows, seed):
    """Real MLX affine codes plus an independent single-rounding reference.

    The served law is Metal mx.dequantize (one F32 rounding to BF16); the GPU
    smoke checks it against real table rows. MLX's CPU kernel rounds twice
    and is deliberately not the reference here.
    """
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        mx.random.seed(seed)
        weight = (mx.random.normal((rows, 256))*0.05).astype(mx.bfloat16)
        q, s, b = mx.quantize(weight, group_size=32, bits=4)
        q, s, b = np.array(q), np.array(s.view(mx.uint16)), np.array(b.view(mx.uint16))
    codes = ((q[:, :, None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15).reshape(rows, 256).astype(np.float64)
    as_f32 = lambda v: np.repeat((v.astype(np.uint32) << 16).view(np.float32), 32, axis=1)
    reference = _bf16((codes*as_f32(s) + as_f32(b)).astype(np.float32))
    return q, s, b, reference, (reference.view(np.uint32) >> 16).astype(np.uint16)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_affine4_decode_matches_single_rounding_law(seed):
    q, s, b, reference, bits = _quantized(512, seed)
    assert np.array_equal(decode_affine4_rows(q, s, b), reference)
    assert np.array_equal(decode_affine4_rows(q, s, b, compact=True), bits)


def test_affine4_decode_refuses_wrong_layout():
    q, s, b, _, _ = _quantized(4, 0)
    with pytest.raises(ValueError, match="U32"):
        decode_affine4_rows(q[:, :16], s, b)
    with pytest.raises(ValueError, match="U32"):
        decode_affine4_rows(q, s.astype(np.float32), b)


@pytest.mark.parametrize("cache_format", ["fp32", "bf16_lut"])
def test_engram_rows_read_three_affine_planes(tmp_path, cache_format):
    q, s, b, reference, _ = _quantized(64, 3)
    path = tmp_path / "table.bin"
    header = b"\0"*416
    path.write_bytes(header + q.tobytes() + s.tobytes() + b.tobytes())
    offset = len(header)
    table = EngramRows(str(path), offset=offset, scale_offset=offset+q.nbytes,
                       bias_offset=offset+q.nbytes+s.nbytes, rows=64,
                       expected_stat=file_stamp(os.stat(path)), workers=2, cache_format=cache_format)
    try:
        ids = np.array([[5, 63, 5], [0, 17, 42]])
        assert np.array_equal(table.read(ids), reference[ids])
        assert table.stats["bytes_read"] == 5*160
        assert table.diagnostics()["storage"] == "pread-affine4-g32-bf16-three-planes"
        assert table.diagnostics()["table_bytes"] == 64*160
        assert np.array_equal(table.read(np.array([63])), reference[[63]])  # cache hit
        assert table.stats["cache_hits"] == 1
    finally:
        table.close()


def test_engram_affine_planes_must_not_overlap_or_exceed(tmp_path):
    path = tmp_path / "table.bin"
    path.write_bytes(b"\0"*(8*160))
    stamp = file_stamp(os.stat(path))
    with pytest.raises(ValueError, match="overlap"):
        EngramRows(str(path), offset=0, scale_offset=8, bias_offset=8*144, rows=8, expected_stat=stamp)
    with pytest.raises(ValueError, match="exceeds"):
        EngramRows(str(path), offset=0, scale_offset=8*128, bias_offset=8*160, rows=8, expected_stat=stamp)
    with pytest.raises(ValueError, match="bounds"):
        EngramRows(str(path), offset=0, bias_offset=8*144, rows=8, expected_stat=stamp)


def test_module_specs_fail_closed():
    assert _module_spec({"bits": 6, "group_size": 64, "mode": "affine", "quantize_input": True}, "m")["bits"] == 6
    assert _module_spec({"bits": 8, "mode": "mxfp8"}, "m")["mode"] == "mxfp8"
    for bad in ({"bits": 3, "group_size": 64, "mode": "affine", "quantize_input": True},
                {"bits": 6, "group_size": 64, "mode": "affine"},
                {"bits": 4, "mode": "mxfp8"},
                {"bits": 4, "mode": "nvfp4"}):
        with pytest.raises(ValueError):
            _module_spec(bad, "m")


def test_affine_geometry_checks_packed_width():
    spec = {"bits": 6, "group_size": 64}
    tensors = {"m.weight": {"dtype": "U32", "shape": (2304, 960)},
               "m.scales": {"dtype": "BF16", "shape": (2304, 80)},
               "m.biases": {"dtype": "BF16", "shape": (2304, 80)}}
    assert _affine_geometry(tensors, "m", spec) == (2304, 5120)
    with pytest.raises(ValueError, match="packed width"):
        _affine_geometry(tensors, "m", {"bits": 8, "group_size": 64})
    tensors["m.biases"] = {"dtype": "F32", "shape": (2304, 80)}
    with pytest.raises(ValueError, match="precision"):
        _affine_geometry(tensors, "m", spec)


def test_only_versioned_omlx_configs_take_the_omlx_route():
    assert is_omlx_config({"omlx_deepseek_v41": {"version": 1}})
    assert not is_omlx_config({"omlx_deepseek_v41": {"version": 2}})
    assert not is_omlx_config({"model_type": "deepseek_v41"})


requires_checkpoint = pytest.mark.skipif(
    not (CHECKPOINT / "mlx2-omlx-integrity.json").exists(), reason="sealed omlx V4.1 checkpoint absent")


@requires_checkpoint
def test_registry_dispatches_pinned_checkpoint_to_omlx_adapter():
    from mlx2.adapters.registry import inspect_model
    resolution = inspect_model(CHECKPOINT)
    assert resolution.adapter_type.__name__ == "DeepSeekV41OmlxAdapter"
    artifact = resolution.artifact
    assert artifact["args"].n_routed_experts == 192
    assert len(artifact["aliases"]) == len(artifact["quant"]) + len(artifact["dense"]) == 924
    assert sorted(artifact["engram"]) == [1, 14]
    assert artifact["engram"][14]["rows"] == 384016682
    assert artifact["inventory"]["expert_item_bytes"] == 11059200
    assert artifact["qualification"] == "unqualified"


@requires_checkpoint
def test_admission_fits_and_expert_slices_are_exact():
    from mlx2.adapters.deepseek_v41_omlx import admission_plan
    from mlx2.adapters.deepseek_v41_omlx_artifact import inspect_artifact
    from mlx2.adapters.deepseek_v41_omlx_weights import expert_extents, read_expert
    from mlx2.runtime.streamed_load import WeightStreamRequest
    artifact = inspect_artifact(CHECKPOINT)
    request = WeightStreamRequest(mode="moe_experts", budget_bytes=0, budget_mode="auto",
                                  admission_limit_bytes=92 << 30, read_workers=2, bypass_page_cache=True)
    plan = admission_plan(artifact, request)
    assert plan["required_bytes"] <= 92 << 30
    assert 0.9 < plan["expert_resident_fraction"] <= 1
    with pytest.raises(MemoryError):
        admission_plan(artifact, WeightStreamRequest(mode="moe_experts", budget_bytes=0, budget_mode="auto",
                                                     admission_limit_bytes=12 << 30))
    first, last = (read_expert(artifact["root"], artifact["seals"], expert_extents(artifact, 0, e))
                   for e in (0, 191))
    assert sum(v.nbytes for v in first.values()) == 11059200
    assert first["gate.qweight"].shape == (2304, 320) and first["down.scales"].dtype == np.uint16
    assert not np.array_equal(first["gate.qweight"], last["gate.qweight"])
    with pytest.raises(ValueError, match="outside"):
        expert_extents(artifact, 0, 192)


@requires_checkpoint
def test_pins_cover_every_served_file():
    pins = json.loads((Path(__file__).parents[1] / "src/mlx2/adapters/deepseek_v41_omlx_pins.json").read_text())
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index.values()) <= set(pins["files"])
    assert {"config.json", "tokenizer.json", "model.safetensors.index.json"} <= set(pins["files"])


def _synthetic_weights(rows=5, hidden=128, middle=64, experts=8, top=3, seed=0):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from mlx2.adapters.deepseek_v41_omlx_weights import OmlxV41Weights
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(seed)
    bank, specs = {}, {}
    for kind, shape in {"gate": (middle, hidden), "up": (middle, hidden), "down": (hidden, middle)}.items():
        w = (mx.random.normal((experts, *shape))*0.05).astype(mx.bfloat16)
        bank[kind] = mx.quantize(w, group_size=64 if kind != "down" else 32, bits=2)
        specs[kind] = {"bits": 2, "group_size": 64 if kind != "down" else 32, "quantize_input": True}
    weights = object.__new__(OmlxV41Weights)
    weights.artifact = {"args": SimpleNamespace(num_experts_per_tok=top, hidden_size=hidden, n_routed_experts=experts),
                        "experts": {0: specs}}
    weights._banks, weights._bank_views = {0: bank}, {}
    weights._parked = weights._closed = False
    weights._error = None
    weights.scheduled_expert_calls = {}
    weights.expert_math, weights.dense_qat, weights._last_input = "reference", "reference", None
    weights.bank_decode, weights.prefill_admission = "gather_qmm", "lru"

    class Stream:
        _active = False

        def __init__(self):
            self.tables, self.stats = {"blk.0": list(range(experts))}, {"lookups": 0}

        def window(self):
            return nullcontext()

        def acquire(self, prefix, expert):
            names = ("qweight", "scales", "biases")
            return {kind+"."+n: bank[kind][i][expert] for kind in bank for i, n in enumerate(names)}
    weights.expert_stream = Stream()
    x = mx.random.normal((rows, hidden))
    ids = mx.stack([mx.random.permutation(experts)[:top] for _ in range(rows)]).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((rows, top)), axis=-1)
    return weights, x, ids, scores


@pytest.mark.parametrize("rows", [1, 5])
def test_bank_gather_matches_per_expert_reference(rows):
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        weights, x, ids, scores = _synthetic_weights(rows=rows)
        banked = weights.moe(x, ids, scores, "blk.0")
        bank = weights._banks.pop(0)
        reference = weights.moe(x, ids, scores, "blk.0")
        weights._banks[0] = bank
        assert mx.array_equal(banked, reference).item()
        assert weights.scheduled_expert_calls == {"omlx_affine_experts_bank_gather_qmm": 1,
                                                  "omlx_affine_experts_act_fp8_per_expert": 1}


@pytest.mark.parametrize("rows", [1, 5])
def test_hoisted_expert_math_matches_reference(rows):
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        weights, x, ids, scores = _synthetic_weights(rows=rows, seed=3)
        bank = weights._banks.pop(0)
        weights.expert_math, weights.dense_qat = "reference", "reference"
        reference = weights.moe(x, ids, scores, "blk.0")
        weights.expert_math = "hoisted_v1"
        hoisted = weights.moe(x, ids, scores, "blk.0")
        weights._banks[0] = bank
        assert mx.array_equal(hoisted, reference).item()
        assert weights.scheduled_expert_calls["omlx_affine_experts_hoisted_v1"] == 1


def test_dense_input_quantize_is_reused_for_the_same_array():
    from mlx2.adapters.deepseek_v41_omlx_weights import OmlxV41Weights
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        weights = object.__new__(OmlxV41Weights)
        weights.dense_qat, weights._last_input, weights.scheduled_dense_calls = "reference", None, {}
        spec = {"quantize_input": True}
        x = mx.random.normal((1, 1, 64))
        first, second = weights._input(x, spec), weights._input(x, spec)
        assert first is second
        third = weights._input(x + 0, spec)
        assert third is not first and mx.array_equal(third, first).item()
        assert weights.scheduled_dense_calls == {"omlx_input_fp8_reference": 2, "omlx_input_fp8_reused": 1}


def test_woa_mxfp8_encoding_is_bitwise_or_refused():
    from mlx2.adapters.deepseek_v41_omlx_weights import OmlxV41Weights
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        rng = np.random.default_rng(5)
        weights = object.__new__(OmlxV41Weights)
        weights.woa_storage_receipt = {}
        # Values on the MXFP8 grid: E4M3 codes times a power-of-two scale per 32.
        codes = mx.from_fp8(mx.array(rng.integers(0, 256, size=(64, 256)).astype(np.uint8)), dtype=mx.float32)
        codes = mx.where(mx.isnan(codes), 0.0, codes)
        scale = mx.array(np.exp2(rng.integers(-12, -4, size=(64, 8))).astype(np.float32))
        on_grid = (codes.reshape(64, 8, 32) * scale[..., None]).reshape(64, 256).astype(mx.bfloat16)
        weights._dense = {"blk.0.attn_output_a.weight": on_grid}
        weights._encode_woa("blk.0.attn_output_a.weight")
        q, s = weights._dense["blk.0.attn_output_a.weight.q"], weights._dense["blk.0.attn_output_a.weight.s"]
        back = mx.dequantize(q, scales=s, mode="mxfp8", dtype=mx.bfloat16)
        assert mx.array_equal(back.view(mx.uint16), on_grid.view(mx.uint16)).item()
        assert "blk.0.attn_output_a.weight" not in weights._dense
        off_grid = (mx.random.normal((64, 256))*0.01).astype(mx.bfloat16)
        weights._dense = {"blk.1.attn_output_a.weight": off_grid}
        with pytest.raises(ValueError, match="MXFP8 grid"):
            weights._encode_woa("blk.1.attn_output_a.weight")


@pytest.mark.parametrize("rows", [2, 5])
def test_bank_routed_prefill_uses_exact_bank_slices(rows):
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        weights, x, ids, scores = _synthetic_weights(rows=rows, seed=9)
        weights.expert_math, weights.bank_decode = "hoisted_v1", "routed_v1"
        sliced = weights.moe(x, ids, scores, "blk.0")
        bank = weights._banks.pop(0)
        weights.expert_math = "reference"
        reference = weights.moe(x, ids, scores, "blk.0")
        weights._banks[0] = bank
        assert mx.array_equal(sliced, reference).item()
        assert "omlx_affine_experts_bank_gather_qmm" not in weights.scheduled_expert_calls


def test_routed_decode_headers_carry_two_bit_and_refuse_other_shapes():
    from mlx2.runtime.models import deepseek_v41_routed_decode as routed
    for fast in (True, False):
        assert "x_thread[i + 3] = x[i + 3] / 64.0f;" in routed.header(2, 64, fast)
    spec = {"bits": 2, "group_size": 64}
    specs = {"gate": spec, "up": spec, "down": spec}
    assert routed.supports(5120, 2304, 6, specs)
    assert not routed.supports(5120, 2048, 6, specs)  # MLX would pick qmv_fast for down
    assert not routed.supports(5000, 2304, 6, specs)


def test_bank_expert_views_share_memory_and_match_slices():
    from mlx2.adapters.deepseek_v41_omlx_weights import bank_expert_views
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        weights, _, _, _ = _synthetic_weights(seed=11)
        bank = weights._banks[0]
        views = bank_expert_views(bank, 8)
        for expert in (0, 3, 7):
            for kind, arrays in bank.items():
                for name, array in zip(("qweight", "scales", "biases"), arrays):
                    assert mx.array_equal(views[expert][kind+"."+name], array[expert]).item()


@pytest.mark.parametrize(("rows", "expected_admit"), [(1, True), (5, False)])
def test_scan_resistant_admission_only_bypasses_multi_row_misses(rows, expected_admit):
    mx = pytest.importorskip("mlx.core")
    with mx.stream(mx.cpu):
        weights, x, ids, scores = _synthetic_weights(rows=rows, seed=4)
        bank = weights._banks.pop(0)
        stream, seen = weights.expert_stream, []
        plain = type(stream).acquire

        def recording(prefix, expert, *, admit=True):
            seen.append(admit)
            return plain(stream, prefix, expert)
        stream.acquire = recording
        weights.prefill_admission, weights.expert_math = "scan_resistant_v1", "hoisted_v1"
        bypassed = weights.moe(x, ids, scores, "blk.0")
        weights.prefill_admission = "lru"
        seen_before = list(seen)
        reference = weights.moe(x, ids, scores, "blk.0")
        weights._banks[0] = bank
        assert mx.array_equal(bypassed, reference).item()
        assert set(seen_before) == {expected_admit}


def test_service_reserve_override_is_explicit_and_bounded(monkeypatch):
    from mlx2.runtime.memory_policy import SERVICE_RESERVE_ENV
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C
    monkeypatch.delenv(SERVICE_RESERVE_ENV, raising=False)
    assert C.host_scaled_reserves(128.0, 112.0) == (16.0, 4.0)
    monkeypatch.setenv(SERVICE_RESERVE_ENV, "8")
    assert C.host_scaled_reserves(128.0, 112.0) == (8.0, 4.0)
    for bad in ("1", "nan", "65", "x"):
        monkeypatch.setenv(SERVICE_RESERVE_ENV, bad)
        with pytest.raises(ValueError):
            C.host_scaled_reserves(128.0, 112.0)
