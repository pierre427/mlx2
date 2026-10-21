"""Separately shipped MTP heads (rapid-mlx / TensorFold layout) load as the native head."""

import mlx.core as mx

from mlx2.adapters import qwen38_27b


def test_sidecar_detection_and_prefixing(tmp_path, monkeypatch):
    (tmp_path / "mtp").mkdir()
    mx.save_safetensors(str(tmp_path / "mtp" / "model.safetensors"),
                        {"fc.weight": mx.zeros((2, 2)), "norm.weight": mx.ones((2,))})
    sidecar = qwen38_27b._mtp_sidecar(tmp_path)
    assert sidecar["file"] == "mtp/model.safetensors"
    assert sorted(sidecar["keys"]) == ["mtp.fc.weight", "mtp.norm.weight"]

    mx.save_safetensors(str(tmp_path / "trunk.safetensors"), {"model.norm.weight": mx.ones((2,))})
    monkeypatch.setattr(
        "mlx2.runtime.ubc_evict.load_shards_evicting",
        lambda files, **_: dict(mx.load(str(files[0]))),
    )
    weights = qwen38_27b._load_target_weights(
        [tmp_path / "trunk.safetensors"], external_draft=False, sanitize=lambda w: w,
        mtp_sidecar={**sidecar, "path": tmp_path / sidecar["file"]},
    )
    assert set(weights) == {"model.norm.weight", "mtp.fc.weight", "mtp.norm.weight"}
    external = qwen38_27b._load_target_weights(
        [tmp_path / "trunk.safetensors"], external_draft=True, sanitize=lambda w: w,
        mtp_sidecar={**sidecar, "path": tmp_path / sidecar["file"]},
    )
    assert set(external) == {"model.norm.weight"}      # an external drafter never loads it


def test_no_sidecar_means_no_head(tmp_path):
    assert qwen38_27b._mtp_sidecar(tmp_path) is None
