import hashlib
import json
from pathlib import Path

import pytest

from mlx2.runtime.models.qwen4_ple_nvme import verify_converted_sidecar
from scripts.convert_mlxserve_flash_next_artifact import runtime_weight_prefix


def _fixture(tmp_path: Path):
    sidecar = tmp_path / "ple_rows.bin"
    sidecar.write_bytes(b"converted-ple")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {}}))
    index_hash = hashlib.sha256(index.read_bytes()).hexdigest()
    sidecar_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    manifest = {
        "source_index_sha256": index_hash,
        "file_sha256": sidecar_hash,
        "conversion": {
            "schema": "mlx2.mlx-serve-ngram-import.v1",
            "receipt": "mlx2-conversion.json",
            "source_format": "mlx-serve-ngram",
        },
    }
    receipt = {
        "schema": "mlx2.artifact-conversion.v1",
        "target": str(tmp_path),
        "source_index_sha256": index_hash,
        "ple_output_sha256": sidecar_hash,
        "source_completion_marker": {
            "repo": "owner/model",
            "revision": "a" * 40,
            "completed_at": "2026-09-21T00:00:00+00:00",
            "files": 3,
            "total_bytes": 10,
        },
    }
    (tmp_path / "mlx2-conversion.json").write_text(json.dumps(receipt))
    return sidecar, manifest, receipt


def test_verified_conversion_accepts_exact_sidecar(tmp_path):
    sidecar, manifest, receipt = _fixture(tmp_path)
    assert verify_converted_sidecar(sidecar, tmp_path, manifest) == receipt


@pytest.mark.parametrize("mutation", ["sidecar", "manifest_hash", "receipt_hash", "index"])
def test_verified_conversion_rejects_mutation(tmp_path, mutation):
    sidecar, manifest, receipt = _fixture(tmp_path)
    if mutation == "sidecar":
        sidecar.write_bytes(b"mutated")
    elif mutation == "manifest_hash":
        manifest["file_sha256"] = "0" * 64
    elif mutation == "receipt_hash":
        receipt["ple_output_sha256"] = "0" * 64
        (tmp_path / "mlx2-conversion.json").write_text(json.dumps(receipt))
    else:
        (tmp_path / "model.safetensors.index.json").write_text("{}")
    with pytest.raises(ValueError):
        verify_converted_sidecar(sidecar, tmp_path, manifest)


def test_verified_conversion_rejects_missing_identity(tmp_path):
    sidecar, manifest, receipt = _fixture(tmp_path)
    del receipt["source_completion_marker"]["revision"]
    (tmp_path / "mlx2-conversion.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="exact source identity"):
        verify_converted_sidecar(sidecar, tmp_path, manifest)


@pytest.mark.parametrize(
    ("source", "runtime"),
    [
        ("language_model.mtp.fc_embedding", "mtp.fc_embedding"),
        ("model.language_model.mtp.fc_hidden", "mtp.fc_hidden"),
        ("model.mtp.layers.0.mlp.gate", "mtp.layers.0.mlp.gate"),
        (
            "language_model.model.layers.0.mlp.gate",
            "language_model.model.layers.0.mlp.gate",
        ),
    ],
)
def test_runtime_weight_prefix_matches_model_sanitize(source, runtime):
    assert runtime_weight_prefix(source) == runtime


# --- Load-path dispatch on synthetic artifacts -------------------------------
#
# The mlx-serve conversion (e.g. Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP)
# names shards ``model-00001.safetensors`` and carries no
# ``...ngram_embedding.shard_*`` tensors: the sidecar is the only copy of the
# PLE table. The native layout (``model-00001-of-00001.safetensors``) keeps
# the shard tensors and is spot-checked row by row against them.

import struct

import numpy as np

from mlx2.runtime.models.qwen4_ple_nvme import (
    spot_check_sidecar_rows,
    verify_sidecar_content,
)

PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
NUM_SHARDS = 2
ROWS_PER_SHARD = 3
PARTS = {"weight": 8, "scales": 2, "biases": 2}  # bytes per row


def _write_safetensors(path: Path, tensors: dict) -> None:
    header, blob = {}, b""
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [data.shape[0], data.shape[1]],
            "data_offsets": [len(blob), len(blob) + data.nbytes],
        }
        blob += data.tobytes()
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + blob)


def _write_index(root: Path, weight_map: dict) -> str:
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return hashlib.sha256(index.read_bytes()).hexdigest()


def _base_manifest(index_hash: str) -> dict:
    return {
        "tensor_prefix": PREFIX,
        "num_shards": NUM_SHARDS,
        "rows_per_shard": ROWS_PER_SHARD,
        "total_rows": NUM_SHARDS * ROWS_PER_SHARD,
        "row_bytes": sum(PARTS.values()),
        "data_offset": 0,
        "source_index_sha256": index_hash,
    }


def _native_artifact(root: Path, drop_shard=None):
    rng = np.random.default_rng(0)
    tensors, rows = {}, []
    shards = [
        {p: rng.integers(0, 256, (ROWS_PER_SHARD, n), dtype=np.uint8) for p, n in PARTS.items()}
        for _ in range(NUM_SHARDS)
    ]
    for i, shard in enumerate(shards):
        for part, data in shard.items():
            if i != drop_shard:
                tensors[f"{PREFIX}.shard_{i}.{part}"] = data
        for r in range(ROWS_PER_SHARD):
            rows.append(b"".join(shard[p][r].tobytes() for p in PARTS))
    name = "model-00001-of-00001.safetensors"
    _write_safetensors(root / name, tensors)
    index_hash = _write_index(root, {k: name for k in tensors})
    sidecar = root / "ple_rows.bin"
    sidecar.write_bytes(b"".join(rows))
    return sidecar, _base_manifest(index_hash)


def _converted_artifact(root: Path):
    # Mirrors the mlx-serve conversion: unsuffixed shard names, only the
    # non-table PLE tensors, a receipt, and a full-file-hashed sidecar.
    other = {
        f"{PREFIX.rsplit('.', 1)[0]}.layer_multipliers": np.ones((1, 4), np.uint8),
        "language_model.model.embed_tokens.weight": np.zeros((2, 4), np.uint8),
    }
    names = ["model-00001.safetensors", "model-00002.safetensors"]
    weight_map = {}
    for name, (key, data) in zip(names, other.items()):
        _write_safetensors(root / name, {key: data})
        weight_map[key] = name
    index_hash = _write_index(root, weight_map)
    sidecar = root / "ple_rows.bin"
    sidecar.write_bytes(bytes(range(NUM_SHARDS * ROWS_PER_SHARD * sum(PARTS.values()))))
    sidecar_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    manifest = _base_manifest(index_hash)
    manifest["file_sha256"] = sidecar_hash
    manifest["conversion"] = {
        "schema": "mlx2.mlx-serve-ngram-import.v1",
        "receipt": "mlx2-conversion.json",
        "source_format": "mlx-serve-ngram",
    }
    receipt = {
        "schema": "mlx2.artifact-conversion.v1",
        "target": str(root),
        "source_index_sha256": index_hash,
        "ple_output_sha256": sidecar_hash,
        "source_completion_marker": {
            "repo": "owner/model",
            "revision": "b" * 40,
            "completed_at": "2026-09-14T00:00:00+00:00",
            "files": 2,
            "total_bytes": 100,
        },
    }
    (root / "mlx2-conversion.json").write_text(json.dumps(receipt))
    return sidecar, manifest


def test_converted_layout_without_source_tensors_loads(tmp_path):
    sidecar, manifest = _converted_artifact(tmp_path)
    # The bare row spot check has nothing to sample here; this is the
    # production failure ("artifact has shard indices []...").
    with pytest.raises(ValueError, match=r"shard indices \[\]"):
        spot_check_sidecar_rows(sidecar, tmp_path, manifest, num_random=4)
    assert verify_sidecar_content(sidecar, tmp_path, manifest) == "converted"


def test_converted_layout_without_marker_is_refused(tmp_path):
    sidecar, manifest = _converted_artifact(tmp_path)
    del manifest["conversion"]
    with pytest.raises(ValueError, match="not an approved mlx2 conversion"):
        verify_sidecar_content(sidecar, tmp_path, manifest)


def test_converted_layout_with_flipped_sidecar_byte_is_refused(tmp_path):
    sidecar, manifest = _converted_artifact(tmp_path)
    data = bytearray(sidecar.read_bytes())
    data[-1] ^= 1
    sidecar.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="does not match manifest"):
        verify_sidecar_content(sidecar, tmp_path, manifest)


def test_native_layout_is_spot_checked(tmp_path):
    sidecar, manifest = _native_artifact(tmp_path)
    assert verify_sidecar_content(sidecar, tmp_path, manifest) == "spot_check"
    data = bytearray(sidecar.read_bytes())
    data[0] ^= 1  # row 0 is always sampled
    sidecar.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="does not match the artifact's shard tensors"):
        verify_sidecar_content(sidecar, tmp_path, manifest)


def test_native_layout_with_missing_shard_is_refused(tmp_path):
    # A partial set of source shards must not fall through to the
    # conversion path, even when the manifest carries a conversion marker.
    sidecar, manifest = _native_artifact(tmp_path, drop_shard=1)
    manifest["conversion"] = {
        "schema": "mlx2.mlx-serve-ngram-import.v1",
        "receipt": "mlx2-conversion.json",
        "source_format": "mlx-serve-ngram",
    }
    with pytest.raises(ValueError, match="shard indices"):
        verify_sidecar_content(sidecar, tmp_path, manifest)
