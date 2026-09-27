"""CPU-only metadata and fail-closed checks; run with --noconftest."""

import importlib.abc
import json
import os
import sys
from pathlib import Path

import pytest


class _BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError("real MLX import during CPU-only adapter test")
        return None


@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    assert "mlx" not in sys.modules and "mlx.core" not in sys.modules
    blocker = _BlockMLX()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])


AGNES = Path(os.environ.get("MLX2_AGNES_ARTIFACT", "~/mlx-models/Agnes-3.0-Flash-Preview-MLX-6bit"))
HY_FULL = Path(os.environ.get("MLX2_HY3_FULL_ARTIFACT", "/Volumes/T7/models/kernelpool/Hy3-6bit"))
HY_REAP = Path(os.environ.get("MLX2_HY3_REAP_ARTIFACT", "~/Desktop/mlx-uag/models/Hy3-REAP50-MLX-4bit"))


def _metadata_fixture(source: Path, destination: Path):
    if not source.is_dir():
        pytest.skip(f"local artifact absent: {source}")
    for name in ("config.json", "model.safetensors.index.json"):
        (destination / name).write_bytes((source / name).read_bytes())
    index = json.loads((destination / "model.safetensors.index.json").read_text())["weight_map"]
    for name in set(index.values()):
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(b"metadata")


def test_agnes_inspection_and_text_only_descriptor():
    if not AGNES.is_dir():
        pytest.skip("Agnes artifact absent")
    from mlx2.adapters.agnes_3_flash import DESCRIPTOR, Agnes3FlashAdapter, inspect_artifact
    from mlx2.contracts import Capability

    result = inspect_artifact(AGNES)
    assert result["text_weight_count"] > 2700
    assert result["has_mtp"] is False
    assert Capability.TEXT in DESCRIPTOR.capabilities
    assert Capability.MTP not in DESCRIPTOR.capabilities
    assert Capability.VISION not in DESCRIPTOR.capabilities
    assert Agnes3FlashAdapter.default_route == "ordinary"
    assert "mlx.core" not in sys.modules


@pytest.mark.parametrize("source,reap,experts,mtp_count", [
    (HY_FULL, False, 192, 0), (HY_REAP, True, 96, 44),
])
def test_hy_v3_full_and_reap_topology(source, reap, experts, mtp_count):
    if not source.is_dir():
        pytest.skip(f"HY artifact absent: {source}")
    from mlx2.adapters.hy_v3 import HYV3Adapter, descriptor_for, inspect_artifact
    from mlx2.contracts import Capability

    result = inspect_artifact(source)
    assert result["reap"] is reap
    assert result["config"]["num_experts"] == experts
    assert result["mtp_tensor_count"] == mtp_count
    assert result["has_mtp"] is False
    assert Capability.MTP not in descriptor_for(reap=reap).capabilities
    assert HYV3Adapter.default_route == "ordinary"
    assert "mlx.core" not in sys.modules


def test_agnes_wrong_layer_order_fails_before_model_import(tmp_path):
    _metadata_fixture(AGNES, tmp_path)
    from mlx2.adapters.agnes_3_flash import inspect_artifact

    config = json.loads((tmp_path / "config.json").read_text())
    config["text_config"]["layer_types"][0] = "agnes_global_attention"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="layer order"):
        inspect_artifact(tmp_path)


def test_hy_reap_expert_count_mismatch_fails(tmp_path):
    _metadata_fixture(HY_REAP, tmp_path)
    from mlx2.adapters.hy_v3 import inspect_artifact

    config = json.loads((tmp_path / "config.json").read_text())
    config["mtp_num_experts"] = 96
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="expert topology"):
        inspect_artifact(tmp_path)


def test_missing_indexed_shard_fails(tmp_path):
    _metadata_fixture(HY_FULL, tmp_path)
    from mlx2.adapters.hy_v3 import inspect_artifact

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())["weight_map"]
    (tmp_path / next(iter(index.values()))).unlink()
    with pytest.raises(ValueError, match="missing or escaped weight shard"):
        inspect_artifact(tmp_path)


def test_hy_embedded_mtp_candidate_is_gated_and_unselected(tmp_path):
    if not HY_REAP.is_dir():
        pytest.skip("HY REAP artifact absent")
    from mlx2.adapters.hy_v3_mtp import inspect_mtp_candidate

    record = inspect_mtp_candidate(HY_REAP)
    assert record["sidecar_tensors"] == 44
    assert record["config"]["mtp_num_experts"] == 192
    assert record["qualified"] is False and record["selected"] is False
    assert "mlx.core" not in sys.modules

    _metadata_fixture(HY_REAP, tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"].pop("mtp.eh_proj.weight")
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="MTP tensors are incomplete"):
        inspect_mtp_candidate(tmp_path)
