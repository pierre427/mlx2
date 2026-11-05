"""Synthetic CPU and offline-compiler checks; no checkpoint or MLX import.

Base source 96dfb0325c2a81098ed5fb02085ce3800f0963e0 plus task changes.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.runtime.models import qwen4_qsa_selective as candidate
from mlx2.runtime.models.qwen4_qsa_gvr_reference import selector_law

ARTIFACT = "synthetic-qsa@" + hashlib.sha256(b"no-checkpoint").hexdigest()
REVISION = "96dfb0325c2a81098ed5fb02085ce3800f0963e0"
ROOT = Path(__file__).resolve().parents[1]


def capture(q, p, pos, config):
    return candidate.ScoreSnapshot.capture(
        q, p, pos, artifact_revision=ARTIFACT, source_revision=REVISION, config=config)


def dense_scores(q, p):
    """Materialized independent FP32 oracle; dimension loop, vectorized rows."""
    result = np.zeros((q.shape[0], q.shape[1], p.shape[1]), dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        for batch in range(q.shape[0]):
            for row in range(q.shape[1]):
                heads = np.zeros((p.shape[1], 4), dtype=np.float32)
                for dim in range(128):
                    heads = heads + p[batch, :, dim, None] * q[batch, row, :, dim]
                total = np.zeros(p.shape[1], dtype=np.float32)
                for head in range(4):
                    total = total + np.maximum(heads[:, head], np.float32(0))
                result[batch, row] = total / np.float32(11.313708498984761)
    return result


def sorted_law(scores, positions, config):
    """Independent full-sort composite law, without candidate helper calls."""
    output = np.empty((*scores.shape[:2], config.topk), dtype=np.uint32)
    for batch in range(scores.shape[0]):
        for row in range(scores.shape[1]):
            valid = min(scores.shape[2], max(0, (int(positions[batch, row]) + 1) // config.ratio))
            def order(index, batch=batch, row=row):
                raw = int.from_bytes(scores[batch, row, index].tobytes(), "little")
                return ((0, 0xFFFFFFFF - raw, index) if raw >> 31 else (1, raw, index))
            ids = sorted(sorted(range(valid), key=order, reverse=True)[:config.topk])
            invalid = config.topk - len(ids)
            output[batch, row] = ids + list(range(scores.shape[2] - invalid, scores.shape[2]))
    return output


def test_cpu_producer_scores_and_selection_match_independent_dense_oracle():
    rng = np.random.default_rng(6741)
    q = rng.normal(size=(1, 2, 4, 128)).astype(np.float32)
    p = rng.normal(size=(1, 73, 128)).astype(np.float32)
    pos = np.array([[39, 291]], dtype=np.int32)
    config = candidate.SelectiveConfig(8, 4, capacity=32, samples=8)
    scores = dense_scores(q, p)
    snapshot = capture(q, p, pos, config)
    ids, diag, receipt = candidate.select_cpu(snapshot, config, expected_binding=snapshot.binding)
    np.testing.assert_array_equal(ids, sorted_law(scores, pos, config))
    np.testing.assert_array_equal(ids.reshape(2, 8), selector_law(
        scores.reshape(2, 73), pos.reshape(2), topk=8, compress_ratio=4))
    assert np.all(diag[..., 2] <= config.capacity)
    assert receipt["arithmetic"]["stock_gemm_parity"] == "unvalidated"
    assert not receipt["qualified"] and not receipt["gpu_used"]


@pytest.mark.parametrize("style,expected_path", [
    ("equal", 2), ("underfill", 4), ("bank", 3), ("empty", 0), ("fit", 1),
])
def test_recovery_branches_use_real_score_producer(style, expected_path):
    # Head 0's dim 0 dot is exactly the supplied score; all other terms zero.
    q = np.zeros((1, 1, 4, 128), dtype=np.float32); q[0, 0, 0, 0] = 1
    p = np.zeros((1, 256, 128), dtype=np.float32)
    config = candidate.SelectiveConfig(8, 1, capacity=128, samples=4)
    position = 255
    if style == "equal":
        p[:, :, 0] = 1
    elif style == "underfill":
        p[0, 0, 0] = 100
    elif style == "bank":
        p[0, ::4, 0] = 100
    elif style == "empty":
        position = -1
    else:
        p[0, :, 0] = np.arange(256, 0, -1)
        config = replace(config, samples=32)
    pos = np.array([[position]], dtype=np.int32)
    ids, diag, _ = candidate.select_cpu(capture(q, p, pos, config), config)
    assert diag[0, 0, 0] == expected_path
    np.testing.assert_array_equal(ids, sorted_law(dense_scores(q, p), pos, config))


def test_raw_bit_ties_zeros_nan_and_causal_invalid_fill_in_selector_core():
    rng = np.random.default_rng(12)
    for trial in range(100):
        bits = rng.integers(0, 2**32, size=257, dtype=np.uint32)
        bits[:8] = [0, 0x80000000, 0x7FC00001, 0x7FC00002,
                    0xFFC00001, 0x7F800000, 0xFF800000, 0x3F800000]
        row = bits.view(np.float32)
        config = candidate.SelectiveConfig(16, 4, capacity=32, samples=16)
        position = int(rng.integers(-8, 1030))
        got, diag = candidate._row_select(lambda i, row=row: row[i], 257, position, config)
        wanted = sorted_law(row.reshape(1, 1, 257), np.array([[position]]), config)
        np.testing.assert_array_equal(got, wanted[0, 0])
        assert diag[2] <= 32


def test_snapshot_is_byte_owned_and_survives_caller_changes():
    q = np.ones((1, 1, 4, 128), dtype=np.float32)
    p = np.ones((1, 16, 128), dtype=np.float32)
    pos = np.array([[63]], dtype=np.int32)
    config = candidate.SelectiveConfig(4, 4, capacity=8)
    snapshot = capture(q, p, pos, config)
    before = candidate.select_cpu(snapshot, config)[0]
    q[:] = 0; p[:] = 0; pos[:] = -1
    np.testing.assert_array_equal(before, candidate.select_cpu(snapshot, config)[0])
    for array in (snapshot.query, snapshot.pooled, snapshot.positions):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(ValueError, match="source identity"):
        candidate.select_cpu(replace(snapshot, source_revision="changed"), config)
    with pytest.raises(ValueError, match="expected source"):
        candidate.select_cpu(snapshot, config, expected_binding="foreign")
    forged = snapshot.query.copy(); forged.flags.writeable = False
    with pytest.raises(ValueError, match="immutable byte storage"):
        candidate.select_cpu(replace(snapshot, query=forged), config)


def test_input_capability_and_budget_refuse_before_array_protocol_execution():
    class Device:
        def __array__(self, *_):
            raise AssertionError("device conversion is forbidden")
    config = candidate.SelectiveConfig(4, 4)
    q = np.ones((1, 1, 4, 128), dtype=np.float32)
    p = np.ones((1, 16, 128), dtype=np.float32)
    pos = np.array([[63]], dtype=np.int32)
    with pytest.raises(TypeError, match="NumPy"):
        capture(Device(), p, pos, config)
    with pytest.raises(ValueError, match="budget"):
        capture(q, p, pos, replace(config, max_input_bytes=1))
    with pytest.raises(ValueError, match="workspace"):
        capture(q, p, pos, replace(config, max_workspace_bytes=1))
    with pytest.raises(ValueError, match="H4,D128"):
        capture(q[:, :, :2], p, pos, config)
    with pytest.raises(ValueError, match="float32"):
        capture(q.astype(np.float64), p, pos, config)
    with pytest.raises(ValueError, match="int32"):
        capture(q, p, np.array([[2**31]], dtype=np.int64), config)
    with pytest.raises(ValueError, match="artifact"):
        candidate.ScoreSnapshot.capture(q, p, pos, artifact_revision="", source_revision=REVISION,
                                         config=config)


def _owner_function(path, name, *, package, values, remove_mlx_import=False):
    tree = ast.parse(path.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    if remove_mlx_import:
        # Execute native host orchestration against an injected CPU-only ABI;
        # never resolve or import the accelerator library itself.
        fn.body = [node for node in fn.body if not (
            isinstance(node, ast.Import) and any(alias.name == "mlx.core" for alias in node.names))]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    namespace = {"__package__": package, **values}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)  # noqa: S102 -- execute only a reviewed local function AST
    return namespace[name]


def test_stage_owner_dispatches_producer_inputs_without_dense_scores(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(candidate, "select_metal", lambda *a, **kw: calls.append((a, kw)) or sentinel)
    path = ROOT / "src/mlx2/runtime/models/qwen4_qsa_stage1.py"
    fn = _owner_function(path, "qsa_stage1_select", package="mlx2.runtime.models", values={
        "_DIRECT_SELECTOR": "selective", "qsa_stage1_supported": lambda *a, **kw: True,
        "_record_candidate_dispatch": lambda name: calls.append(name),
    })
    q, p, pos = object(), object(), object()
    assert fn(q, p, pos, block_topk=8, compress_ratio=4) is sentinel
    assert calls[0][0] == (q, p, pos)
    assert calls[1] == "selective_scheduled_dispatches"


def test_long_prefill_metadata_has_bounded_row_windows_and_exact_workspace_guard():
    config = candidate.SelectiveConfig(512, 4)
    candidate.native_supported_metadata((1, 2048, 4, 128), (1, 16384, 128), (1, 2048), config)
    declared = candidate.declared_workspace_bytes((1, 2048, 4, 128), config)
    assert declared < config.max_workspace_bytes
    with pytest.raises(ValueError, match="workspace"):
        candidate.native_supported_metadata((1, 2048, 4, 128), (1, 16384, 128), (1, 2048),
                                             replace(config, max_workspace_bytes=declared - 1))
    with pytest.raises(ValueError, match="4096 rows"):
        candidate.native_supported_metadata((2, 4096, 4, 128), (2, 1024, 128), (2, 4096), config)
    assert config.native_window_rows == 32


def test_native_host_windows_retire_in_order_without_importing_mlx():
    events = []
    pending = False
    def kernel(**kwargs):
        nonlocal pending
        assert not pending, "next window dispatched before prior window retired"
        query, keys, positions, dims = kwargs["inputs"]
        assert query.shape[1] <= 32
        assert int(keys[0, 0, 0]) == int(query[0, 0, 0, 0])
        assert list(dims) == [1, query.shape[1], 256]
        events.append(("dispatch", query.shape[1]))
        pending = True
        ids = np.repeat(positions[..., None], 4, axis=2).astype(np.uint32)
        diagnostics = np.tile(np.array([1, 8, 8, 256], dtype=np.uint32), (*positions.shape, 1))
        return ids, diagnostics
    def retire(*outputs):
        nonlocal pending
        assert pending and outputs[0].shape[1] <= 32
        events.append(("retire", outputs[0].shape[1]))
        pending = False
    gpu = object()
    mock_mx = SimpleNamespace(
        float32=np.dtype("float32"), float16=np.dtype("float16"), bfloat16=object(),
        int32=np.dtype("int32"), uint32=np.dtype("uint32"), gpu=gpu,
        default_device=lambda: gpu, metal=SimpleNamespace(is_available=lambda: True),
        contiguous=np.ascontiguousarray, array=np.array, concatenate=np.concatenate, eval=retire)
    fn = _owner_function(ROOT / "src/mlx2/runtime/models/qwen4_qsa_selective.py", "select_metal",
        package="mlx2.runtime.models", remove_mlx_import=True, values={
            "mx": mock_mx, "SelectiveConfig": candidate.SelectiveConfig,
            "_metadata": candidate._metadata, "_metal_kernel": lambda _: kernel,
            "_LOCK": candidate._LOCK, "_COUNTS": candidate._COUNTS,
        })
    q = np.zeros((2, 65, 4, 128), dtype=np.float32); q[1] = 1
    p = np.zeros((2, 256, 128), dtype=np.float32); p[1] = 1
    positions = np.arange(130, dtype=np.int32).reshape(2, 65)
    candidate.status(reset=True)
    try:
        result = fn(q, p, positions, block_topk=4, compress_ratio=1)
        np.testing.assert_array_equal(result, np.repeat(positions[..., None], 4, axis=2))
        assert events == [(phase, length) for length in (32, 32, 1, 32, 32, 1)
                          for phase in ("dispatch", "retire")]
        counts = candidate.status()["counts"]
        assert counts["native_completed_windows"] == 6
        assert counts["native_bounded_rows"] == 130
        assert counts["native_max_stored_candidates"] == 8
    finally:
        candidate.status(reset=True)


def test_policy_opt_in_and_cache_arithmetic_contract_are_distinct():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    base = FlashNextPolicy()
    selected = FlashNextPolicy.from_mapping({"qsa_stage1_direct_selector": "selective"})
    assert "MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR" not in base.environment()
    assert selected.environment()["MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR"] == "selective"
    adapter = object.__new__(FlashNextAdapter)
    adapter._diagnostic_modules = [SimpleNamespace(fused_expert_kernel_mode="auto")]
    adapter.model = SimpleNamespace()
    adapter.policy = base
    ordinary = adapter.execution_numerics_contract()
    adapter.policy = selected
    alternate = adapter.execution_numerics_contract()
    assert "qsa_selective_score_law" not in ordinary
    assert alternate["qsa_selective_score_law"] == candidate.arithmetic_contract()
    assert json.dumps(ordinary, sort_keys=True) != json.dumps(alternate, sort_keys=True)


def test_import_and_cpu_execution_never_import_accelerator_libraries():
    assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-c",
        ("import sys; from mlx2.runtime.models.qwen4_qsa_selective import status; status(); "
         "assert not any(k=='mlx' or k.startswith('mlx.') or k=='torch' for k in sys.modules)")],
        env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind,topk,capacity", [("float", 8, 64), ("half", 8, 64),
                                                ("bfloat", 512, 2048)])
def test_metal_source_compiles_offline_without_device(tmp_path, kind, topk, capacity):
    config = candidate.SelectiveConfig(topk, 4, capacity=capacity)
    source = tmp_path / "selective.metal"
    source.write_text(candidate.metal_source(config, input_type=kind))
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2",
                             "-c", str(source), "-o", str(tmp_path / "selective.air")],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_mlx_kernel_wrapper_abi_compiles_without_importing_mlx(tmp_path):
    config = candidate.SelectiveConfig(8, 4, capacity=64)
    text = "#define MLX_WRAPPED 1\n" + candidate.metal_source(config)
    header, body = text.split("// MLX_BODY\n", 1)
    body = body.split("// MLX_END", 1)[0]
    wrapper = ("\nkernel void wrapper(const device float* q [[buffer(0)]], "
               "const device float* pooled [[buffer(1)]], "
               "const device int* positions [[buffer(2)]], "
               "const device uint* dims [[buffer(3)]], "
               "device uint* block_ids [[buffer(4)]], "
               "device uint* diagnostics [[buffer(5)]], "
               "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], "
               "uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {\n")
    source = tmp_path / "wrapped.metal"
    source.write_text(header + wrapper + body + "\n}\n")
    result = subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2", "-c",
                             str(source), "-o", str(tmp_path / "wrapped.air")],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
