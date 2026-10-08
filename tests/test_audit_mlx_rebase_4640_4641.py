import importlib.util
import json
import struct
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/audit_mlx_rebase_4640_4641.py"
SPEC = importlib.util.spec_from_file_location("audit_mlx_rebase", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write_shard(path: Path, tensors: dict) -> None:
    header = json.dumps(tensors).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


def test_model_audit_separates_dense_thin_n_from_quantized(tmp_path):
    shard = "model-00001-of-00001.safetensors"
    tensors = {
        "dense.weight": {"dtype": "BF16", "shape": [32, 128], "data_offsets": [0, 0]},
        "scalar.weight": {"dtype": "BF16", "shape": [1, 128], "data_offsets": [0, 0]},
        "fp32.weight": {"dtype": "F32", "shape": [16, 128], "data_offsets": [0, 0]},
        "wide.weight": {"dtype": "F16", "shape": [128, 128], "data_offsets": [0, 0]},
        "quant.weight": {"dtype": "U32", "shape": [64, 16], "data_offsets": [0, 0]},
        "quant.scales": {"dtype": "BF16", "shape": [64, 4], "data_offsets": [0, 0]},
    }
    _write_shard(tmp_path / shard, tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )

    result = audit.audit_model("fixture", tmp_path)

    assert result["dense_2d_count"] == 4
    assert result["quantized_2d_count"] == 1
    assert result["thin_n_dense_count"] == 1
    assert result["thin_n_dense"][0]["name"] == "dense.weight"


def test_source_audit_captures_route_boundaries(tmp_path):
    mlx_source = tmp_path / "mlx"
    metal = mlx_source / "mlx/backend/metal"
    metal.mkdir(parents=True)
    (metal / "matmul.cpp").write_text(
        "// Case 2: Few output columns\n"
        "if (use_nax && batch_size_out == 1 && !transpose_a && N <= 64 &&\n"
        "out.dtype() != float32 && (transpose_b || int64_t(K) * ldb <= INT_MAX)) {\n"
        "int sn = M < 2048 ? 1 : 2, ks = 4 / sn; steel_gemm_thin_nax; }\n"
        "// Case 3: Large K\n"
        "gemv_wide(); steel_matmul();"
    )
    (metal / "quantized.cpp").write_text(
        "array intermediate({split_k, M, N}, float32; "
        "array intermediate(temp_shape, x.dtype()"
    )
    invariant = tmp_path / "invariant_prefill.py"
    invariant.write_text("LINEAR_BATCHES = 2\nmx.contiguous(mx.stack([wt, wt]))")

    result = audit.audit_source(mlx_source, invariant)

    assert result["all_checks_pass"] is True
    assert "suppresses #4640" in result["route_conclusions"]["invariant_prefill_dense_lane"]


def test_build_audit_measures_nax_inputs_and_outputs(tmp_path):
    build_dir = tmp_path / "build"
    flags = build_dir / "CMakeFiles/mlx.dir"
    metal = build_dir / "mlx/backend/metal/kernels"
    make_dir = metal / "CMakeFiles/mlx-metallib.dir"
    flags.mkdir(parents=True)
    make_dir.mkdir(parents=True)
    (build_dir / "CMakeCache.txt").write_text(
        "CMAKE_OSX_DEPLOYMENT_TARGET:UNINITIALIZED=26.2\n"
    )
    (flags / "flags.make").write_text("CXX_DEFINES =\n")
    (metal / "steel_gemm_thin_nax.air").write_bytes(b"air")
    (make_dir / "build.make").write_text("mlx.metallib: steel_gemm_thin_nax.air\n")
    revision = "a" * 40
    wheel = tmp_path / "candidate-aaaaaaaaa.whl"
    wheel.write_bytes(b"wheel")

    result = audit.audit_build(
        wheel,
        build_dir,
        revision=revision,
        expected_revision=revision,
        source_tracked_clean=True,
    )

    assert result["all_checks_pass"] is True
    assert all(result["checks"].values())
