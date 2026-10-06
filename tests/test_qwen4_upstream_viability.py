import inspect
import json

import numpy as np
import pytest

from scripts.bench_qwen4_hc_gs32_candidate import (
    candidate_sources,
    require_matching_gpu_receipts,
)
from scripts.bench_qwen4_ple_reads import benchmark as benchmark_ple_reads
from scripts.qwen4_upstream_artifact_probe import inspect_mapping, trunk_blocks

PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


def _table(spelling: str, *, shards: int = 2, drop: tuple[int, str] | None = None):
    out = {}
    for index in range(shards):
        for part in ("weight", "scales", "biases"):
            if drop == (index, part):
                continue
            out[f"{PREFIX}{spelling}{index}.{part}"] = "model.safetensors"
    return out


def test_artifact_probe_distinguishes_both_ple_shard_spellings():
    native = inspect_mapping({"model_type": "qwen4_exp"}, _table(".shard_"))
    upstream = inspect_mapping({"model_type": "qwen4_exp"}, _table(".shards."))
    assert native["ple"]["embedded_tables"][0] == {
        "prefix": PREFIX,
        "spelling": "shard_N",
        "shards": 2,
        "contiguous_zero_based": True,
        "missing_parts": {},
        "current_mlx2_native_name": True,
        "tensorfold_330_name": False,
        "compatible_without_conversion": True,
    }
    table = upstream["ple"]["embedded_tables"][0]
    assert table["spelling"] == "shards.N"
    assert table["tensorfold_330_name"]
    assert not table["current_mlx2_native_name"]
    assert not table["compatible_without_conversion"]


def test_artifact_probe_refuses_to_call_partial_table_compatible():
    report = inspect_mapping(
        {"model_type": "qwen4_exp"}, _table(".shard_", drop=(1, "biases"))
    )
    table = report["ple"]["embedded_tables"][0]
    assert table["missing_parts"] == {"1": ["biases"]}
    assert not table["compatible_without_conversion"]


def test_artifact_probe_finds_only_declared_gs32_hc_projections():
    names = {
        "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down.weight": "a",
        "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up.weight": "a",
        "language_model.model.layers.0.attn_hyper_connection.block_inject_weight.weight": "a",
    }
    base = {"model_type": "qwen4_exp", "quantization": {"bits": 4, "group_size": 64}}
    report = inspect_mapping(base, names)
    assert report["hc"]["group_size_32_entries"] == 0
    down = next(iter(names)).removesuffix(".weight")
    base["quantization"][down] = {"bits": 4, "group_size": 32, "mode": "affine"}
    report = inspect_mapping(base, names)
    assert report["hc"]["group_size_32_entries"] == 1
    assert report["hc"]["synthetic_gs32_probe_worth_running"]


def test_strata_nextn_metadata_never_underflows_trunk_count():
    assert trunk_blocks(
        {"qwen4exp.block_count": 49, "qwen4exp.nextn_predict_layers": 1}
    ) == {"declared_blocks": 49, "nextn_blocks": 1, "trunk_blocks": 48}
    with pytest.raises(ValueError, match="exceeds"):
        trunk_blocks({"qwen4exp.block_count": 1, "qwen4exp.nextn_predict_layers": 2})


def test_current_mlx2_ple_loader_has_not_already_normalized_shards_dot_n():
    from mlx2.runtime.models import qwen4_exp, qwen4_ple_nvme

    predicate = inspect.getsource(qwen4_exp.TextModel.quant_predicate.fget)
    source_refs = inspect.getsource(qwen4_ple_nvme._source_shard_refs)
    assert "ngram_embedding.shard_" in predicate
    assert 'prefix + ".shard_"' in source_refs
    assert "shards." not in predicate
    assert "shards." not in source_refs


def test_gs32_candidate_parameterizes_every_projection_group_pointer():
    from mlx2.runtime.models import qwen4_hc_decode as hcd

    header, norm_down, up_mix = candidate_sources(
        hcd.HEADER, hcd.NORM_DOWN_SOURCE, hcd.UP_MIX_SOURCE
    )
    assert "int BITS, int GS" in header
    assert "g * GS + sc * 8" in header
    assert "sc < GS / 8" in header
    assert "K / GS" in norm_down
    assert "R / GS" in up_mix
    assert "hcd_wide_row<T, DB, GS>" in norm_down
    assert "hcd_wide_row<T, UB, GS>" in up_mix
    assert " / 64" not in norm_down + up_mix
    assert "64 / " not in norm_down + up_mix


def test_current_hc_route_still_refuses_group_size_32():
    from mlx2.runtime.models import qwen4_hc_decode as hcd

    assert hcd.GROUP_SIZE == 64


def test_gs32_gpu_probe_requires_two_matching_lease_receipts(tmp_path):
    first, second = tmp_path / "host.json", tmp_path / "local.json"
    with pytest.raises(RuntimeError, match="missing"):
        require_matching_gpu_receipts((first, second))
    first.write_text(json.dumps({"lease_id": "lease-a"}))
    second.write_text(json.dumps({"lease_id": "lease-b"}))
    with pytest.raises(RuntimeError, match="matching lease"):
        require_matching_gpu_receipts((first, second))
    second.write_text(first.read_text())
    assert require_matching_gpu_receipts((first, second))["lease_id"] == "lease-a"


def test_ple_read_benchmark_compares_identical_serial_and_parallel_bytes(tmp_path):
    dims = 32
    row_bytes = dims // 2 + 2 * (dims // 32) * 2
    total_rows = 64
    sidecar = tmp_path / "ple_rows.bin"
    sidecar.write_bytes(np.arange(total_rows * row_bytes, dtype=np.uint8).tobytes())
    manifest = {
        "total_rows": total_rows,
        "dims": dims,
        "num_shards": 2,
        "data_offset": 0,
    }
    report = benchmark_ple_reads(
        sidecar, manifest, rows=8, reps=2, workers=[1, 2], seed=687
    )
    assert report["page_cache_controlled"] is False
    assert set(report["workers"]) == {"1", "2"}
    assert all(
        len(values["first_pass_ms"]) == 2 for values in report["workers"].values()
    )


def test_local_artifact_probe_cli_shape(tmp_path):
    config = {"model_type": "qwen4_exp", "text_config": {"mtp_num_hidden_layers": 1}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": _table(".shards.")})
    )
    from scripts.qwen4_upstream_artifact_probe import inspect_artifact

    result = inspect_artifact(tmp_path)
    assert result["mtp"]["mlx_json_mtp_num_hidden_layers"] == 1
    assert result["ple"]["embedded_tables"][0]["spelling"] == "shards.N"
