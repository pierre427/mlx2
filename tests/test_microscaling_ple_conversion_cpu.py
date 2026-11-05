"""Direct-table PLE sidecar rebinding during offline SC artifact conversion.

All tensors are tiny CPU fixtures; existing runtime verifiers are exercised.
"""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from mlx2.adapters.flash_next_conversion import finish_ple_rebind, prepare_ple_rebind
from mlx2.runtime.microscaling.conversion import ConversionError, Source, convert_artifact
from mlx2.runtime.models.qwen4_ple_nvme import verify_sidecar_against_artifact, verify_sidecar_content

PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


def fixture(root):
    root.mkdir()
    rng = np.random.default_rng(38)
    tensors = {"model.layers.0.mlp.up_proj.weight": ("F32", rng.normal(0, .02, (64, 128)).astype(np.float32))}
    sidecar_rows, shard_hashes = [], []
    for i in range(2):
        parts = {"weight": ("U32", rng.integers(0, 2**32, (3, 4), dtype=np.uint32)),
                 "scales": ("BF16", np.full((3, 1), 0x3C80, np.uint16)),
                 "biases": ("BF16", np.full((3, 1), 0xBC00, np.uint16))}
        rows = []
        for row in range(3):
            rows.append(b"".join(value[row].tobytes() for _, value in parts.values()))
        data = b"".join(rows)
        sidecar_rows.append(data); shard_hashes.append(hashlib.sha256(data).hexdigest())
        for part, value in parts.items():
            tensors[f"{PREFIX}.shard_{i}.{part}"] = value
    header, blob = {}, bytearray()
    for name, (dtype, value) in tensors.items():
        data = value.tobytes()
        header[name] = {"dtype": dtype, "shape": list(value.shape), "data_offsets": [len(blob), len(blob) + len(data)]}
        blob.extend(data)
    encoded = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)
    index = {"weight_map": {name: "model.safetensors" for name in tensors}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "config.json").write_text('{"model_type":"fixture"}')
    sidecar = b"PLE\0" + b"".join(sidecar_rows)
    (root / "ple_rows.bin").write_bytes(sidecar)
    manifest = {"format": "qwen4-ple-rows", "version": 1, "tensor_prefix": PREFIX,
                "bits": 4, "group_size": 32, "mode": "affine", "dims": 32,
                "weight_bytes": 16, "scales_bytes": 2, "biases_bytes": 2, "row_bytes": 20,
                "num_shards": 2, "rows_per_shard": 3, "total_rows": 6, "data_offset": 4,
                "source_model_path": str(root), "source_index_sha256": hashlib.sha256((root / "model.safetensors.index.json").read_bytes()).hexdigest(),
                "shard_sha256": shard_hashes, "file_sha256": hashlib.sha256(sidecar).hexdigest()}
    (root / "ple_rows.bin.manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_converted_reindexed_ple_passes_existing_runtime_verifiers(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    old = fixture(source)
    source_manifest = (source / "ple_rows.bin.manifest.json").read_bytes()
    # Scope all must retain embedding-table bytes, rather than treating
    # the table's rank-two packed tensors as dense projection weights.
    receipt = convert_artifact(source, target, scope="all")
    manifest = verify_sidecar_against_artifact(target / "ple_rows.bin", target)
    assert manifest["source_index_sha256"] != old["source_index_sha256"]
    assert manifest["source_model_path"] == str(target)
    assert manifest["file_sha256"] == old["file_sha256"]
    assert verify_sidecar_content(target / "ple_rows.bin", target, manifest) == "spot_check"
    assert len(receipt["ple_sidecar_rebind"]["table_tensor_sha256"]) == 6
    from mlx2.adapters.flash_next import artifact_identity
    assert receipt["output"]["scheme"] == "mlx2-flash-next"
    assert receipt["output"]["fingerprint"] == artifact_identity(target)["fingerprint"]
    assert receipt["ple_sidecar_rebind"]["source_index_sha256"] == old["source_index_sha256"]
    assert (source / "ple_rows.bin.manifest.json").read_bytes() == source_manifest
    assert (target / "ple_rows.bin").read_bytes() == (source / "ple_rows.bin").read_bytes()
    assert not (target / ".incomplete").exists()


@pytest.mark.parametrize("mutation", ["index", "table", "sidecar", "logical_digest"])
def test_stale_source_ple_refuses_before_destination_creation(tmp_path, mutation):
    source, target = tmp_path / "source", tmp_path / "target"
    manifest = fixture(source)
    if mutation == "index":
        (source / "model.safetensors.index.json").write_text((source / "model.safetensors.index.json").read_text() + "\n")
    elif mutation == "table":
        tensor = Source(source).tensors[f"{PREFIX}.shard_0.weight"]
        with tensor.path.open("r+b") as stream:
            stream.seek(tensor.offset); byte = stream.read(1)
            stream.seek(tensor.offset); stream.write(bytes([byte[0] ^ 1]))
    elif mutation == "sidecar":
        path = source / "ple_rows.bin"
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    else:
        manifest["shard_sha256"][0] = "0" * 64
        (source / "ple_rows.bin.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Rebuild the PLE sidecar"):
        convert_artifact(source, target)
    assert not target.exists()


def test_changed_output_table_cannot_authorize_rebind(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    fixture(source)
    original = Source(source)
    binding = prepare_ple_rebind(original)
    convert_artifact(source, target)
    output = Source(target)
    table = output.tensors[f"{PREFIX}.shard_1.biases"]
    with table.path.open("r+b") as stream:
        stream.seek(table.offset); value = stream.read(1)
        stream.seek(table.offset); stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match="tensor bytes changed"):
        finish_ple_rebind(original, output, binding, conversion_schema="fixture")


def test_tensor_free_converted_sidecar_and_bound_mtp_manifest_refuse(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    manifest = fixture(source)
    manifest["conversion"] = {"schema": "mlx2.mlx-serve-ngram-import.v1"}
    (source / "ple_rows.bin.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="already-converted"):
        convert_artifact(source, target)
    assert not target.exists()
    del manifest["conversion"]
    (source / "ple_rows.bin.manifest.json").write_text(json.dumps(manifest))
    (source / "mtp_draft_vocab.json").write_text('{"schema":"mlx2.mtp-draft-vocab.v1"}')
    with pytest.raises(ConversionError, match="build_mtp_draft_vocab_manifest.py"):
        convert_artifact(source, target)
    assert not target.exists()


def test_unbound_ids_assets_preserved_and_unknown_index_binding_refused(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    fixture(source)
    (source / "mtp_draft_vocab.ids").write_text("0\n1\n")
    (source / "mtp_draft_vocab.ids.LICENSE").write_text("Fixture IDs license\n")
    convert_artifact(source, target)
    for name in ("mtp_draft_vocab.ids", "mtp_draft_vocab.ids.LICENSE"):
        assert (target / name).read_bytes() == (source / name).read_bytes()
    (source / "other-manifest.json").write_text('{"schema":"unknown.v1","index_sha256":"abc"}')
    with pytest.raises(ConversionError, match="unsupported revision-bound"):
        convert_artifact(source, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
