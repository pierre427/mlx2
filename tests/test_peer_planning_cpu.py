"""Fit advice must stay metadata-only and must not become admission evidence."""

import json
import subprocess
import sys

import pytest

from mlx2.planning import artifact_payload, deployment_plan


def artifact(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"muse_glimmer"}')
    (tmp_path / "model.safetensors").write_bytes(b"not a tensor file")
    return tmp_path


def test_cli_uses_no_tensor_runtime_or_network(tmp_path):
    path = artifact(tmp_path)
    script = '''
import importlib.abc, sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'torch', 'requests', 'httpx'}:
            raise AssertionError('unexpected runtime/network import: ' + fullname)
sys.meta_path.insert(0, Refuse())
from mlx2.planning import main
main(sys.argv[1:])
'''
    result = subprocess.run([sys.executable, "-c", script, str(path),
                             "--host-gib", "32", "--apc-gib", "0"],
                             capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["assessment"] == "indeterminate"
    assert report["cache_geometry"] is None
    assert "adapter_cache_projection" in report["unknowns"]
    assert not any(report[name] for name in
                   ("qualified", "selected", "observed_used", "admission_authorized"))


def test_index_deduplicates_shards_and_counts_side_assets(tmp_path):
    path = artifact(tmp_path)
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "model.safetensors", "b": "model.safetensors"}}))
    (path / "ple_rows.bin").write_bytes(b"0123456789")
    report = artifact_payload(path)
    assert report["serialized_bytes"] == 27
    assert len(report["files"]) == 2
    assert "not-weight-content-hash" in report["identity_kind"]


@pytest.mark.parametrize("name", ["../foreign.safetensors", "/tmp/foreign.safetensors", None, []])
def test_bad_shard_metadata_refused(tmp_path, name):
    path = artifact(tmp_path)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": name}}))
    with pytest.raises(ValueError):
        artifact_payload(path)


def test_foreign_symlink_refused(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"0123456789")
    path = tmp_path / "model"
    path.mkdir()
    artifact(path)
    (path / "model.safetensors").unlink()
    (path / "model.safetensors").symlink_to(foreign)
    with pytest.raises(ValueError, match="foreign"):
        artifact_payload(path)


def test_duplicate_metadata_keys_refused(tmp_path):
    path = artifact(tmp_path)
    (path / "config.json").write_text('{"model_type":"a","model_type":"b"}')
    with pytest.raises(ValueError, match="duplicate"):
        artifact_payload(path)


def test_over_budget_is_advice_not_admission(tmp_path):
    report = deployment_plan(artifact(tmp_path), host_bytes=1024, apc_bytes=0)
    assert report["assessment"] == "over_planning_budget"
    assert report["estimated_headroom_bytes"] < 0
    assert not report["admission_authorized"]


@pytest.mark.parametrize("options", [
    {"context": 0}, {"lanes": True}, {"scratch_bytes": -1},
    {"host_bytes": 100, "advisory_bytes": 101}, {"route": "native_mtp"},
    {"route": "external_draft"}, {"draft_path": "unused"},
])
def test_invalid_or_unsupported_plan_refused(tmp_path, options):
    with pytest.raises(ValueError):
        deployment_plan(artifact(tmp_path), **options)


def test_adapter_projection_reuses_serving_geometry():
    from test_qwen38_memory import config as qwen_config

    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.flash_next_memory import FlashNextCacheBudget
    from mlx2.adapters.north_memory import NorthCacheBudget
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.adapters.qwen38_memory import Qwen38CacheBudget
    flash = dict(qwen_config(), layer_types=["linear_attention"] * 48 + ["full_attention"] * 16,
                 indexer_head_dim=128, indexer_compress_ratio=4,
                 ple_embed_dim=2560, ple_conv_kernel_size=4)
    north = {"num_hidden_layers": 2, "layer_types": ["full_attention", "sliding_attention"],
             "num_key_value_heads": 4, "head_dim": 128, "sliding_window": 4096}
    for adapter, budget, config, modes in (
        (FlashNextAdapter, FlashNextCacheBudget, flash, (False, True)),
        (NorthMiniCodeAdapter, NorthCacheBudget, north, (False,)),
        (Qwen3827BAdapter, Qwen38CacheBudget, qwen_config(), (False, True)),
    ):
        for mtp in modes:
            planned = adapter.planning_cache_budget(config, mtp=mtp)
            serving = budget.from_config(config, mtp=mtp)
            assert planned.as_dict() == serving.as_dict()
            assert planned.project(32768) == serving.project(32768)
