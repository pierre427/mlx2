"""Import-free path and receipt checks for the OrcaSAQ affine repacker."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPT_DIR / "repack_orcasaq_qwen38_mlx_affine.py"
sys.path.insert(0, str(SCRIPT_DIR))
try:
    SPEC = importlib.util.spec_from_file_location("repack_orcasaq_path_safety", SCRIPT)
    assert SPEC and SPEC.loader
    repacker = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(repacker)
finally:
    sys.path.remove(str(SCRIPT_DIR))


def _source_files() -> dict:
    return {
        "plain": {
            "filename": "model-00001.safetensors",
            "key": "model.plain.weight",
            "source_type": "F32",
        },
        "packed": {
            "filename": "model-00002.safetensors",
            "key": "model.packed.weight",
            "source_type": "IQ4_XS",
        },
    }


def _records() -> dict:
    return {
        "plain": {
            "filename": "model-00001.safetensors",
            "source_type": "F32",
            "bits": None,
            "keys": ["model.plain.weight"],
            "size": 1,
            "data_bytes": 1,
            "sha256": "a" * 64,
        },
        "packed": {
            "filename": "model-00002.safetensors",
            "source_type": "IQ4_XS",
            "bits": 5,
            "keys": [
                "model.packed.biases",
                "model.packed.scales",
                "model.packed.weight",
            ],
            "size": 1,
            "data_bytes": 1,
            "sha256": "b" * 64,
        },
    }


@pytest.mark.parametrize("filename", ("../escape.safetensors", "/tmp/escape.safetensors", "nested/x.safetensors"))
def test_flat_child_rejects_traversal_and_absolute_paths(tmp_path, filename):
    with pytest.raises(ValueError, match="not flat"):
        repacker._flat_child(tmp_path, filename, suffix=".safetensors")


def test_flat_child_rejects_symlink_escape(tmp_path):
    outside = tmp_path.parent / "outside.safetensors"
    outside.write_bytes(b"outside")
    (tmp_path / "model.safetensors").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        repacker._flat_child(tmp_path, "model.safetensors", suffix=".safetensors")


def test_roots_reject_alias_and_nesting(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        repacker._resolved_roots(source, source)
    with pytest.raises(ValueError, match="disjoint"):
        repacker._resolved_roots(source, source / "output")
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        repacker._resolved_roots(alias, tmp_path / "output")


def test_valid_records_reconstruct_exact_weight_map(tmp_path):
    expected = {
        "model.plain.weight": "model-00001.safetensors",
        "model.packed.biases": "model-00002.safetensors",
        "model.packed.scales": "model-00002.safetensors",
        "model.packed.weight": "model-00002.safetensors",
    }
    assert repacker._validate_repack_records(
        tmp_path,
        _records(),
        _source_files(),
        repacker.DEFAULT_PROFILE,
        require_complete=True,
    ) == expected


@pytest.mark.parametrize("field,value,match", (
    ("filename", "../escape.safetensors", "not flat"),
    ("filename", "other.safetensors", "differs from source mapping"),
    ("bits", 4, "policy differs"),
    ("keys", ["model.packed.weight"], "tensor keys differ"),
    ("sha256", "not-a-digest", "invalid file metadata"),
))
def test_tampered_completion_record_fails_closed(tmp_path, field, value, match):
    records = _records()
    records["packed"][field] = value
    with pytest.raises(ValueError, match=match):
        repacker._validate_repack_records(
            tmp_path,
            records,
            _source_files(),
            repacker.DEFAULT_PROFILE,
            require_complete=True,
        )


def test_duplicate_shard_and_tensor_references_fail_closed(tmp_path):
    source_files = _source_files()
    records = _records()
    source_files["packed"]["filename"] = source_files["plain"]["filename"]
    records["packed"]["filename"] = records["plain"]["filename"]
    with pytest.raises(ValueError, match="reuse a shard filename"):
        repacker._validate_repack_records(
            tmp_path, records, source_files, repacker.DEFAULT_PROFILE,
            require_complete=True,
        )

    source_files = _source_files()
    records = _records()
    source_files["packed"]["key"] = source_files["plain"]["key"]
    records["packed"]["keys"] = [
        "model.plain.biases", "model.plain.scales", "model.plain.weight",
    ]
    with pytest.raises(ValueError, match="duplicate output tensor"):
        repacker._validate_repack_records(
            tmp_path, records, source_files, repacker.DEFAULT_PROFILE,
            require_complete=True,
        )


def test_complete_receipt_verification_needs_no_mlx_and_rejects_tampering(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    source_config = {"source": "config"}
    (source / "config.json").write_text(json.dumps(source_config))
    output_config = {"output": "config"}
    config_bytes = (json.dumps(output_config, sort_keys=True, indent=2) + "\n").encode()
    (output / "config.json").write_bytes(config_bytes)

    source_receipt = {
        "files": _source_files(),
        "support_files": {},
        "config_sha256": "c" * 64,
    }
    records = _records()
    for entry in records.values():
        path = output / entry["filename"]
        path.write_bytes(entry["filename"].encode())
        entry["size"] = path.stat().st_size
        entry["data_bytes"] = entry["size"]
        entry["sha256"] = repacker.sha256(path)
    weight_map = repacker._validate_repack_records(
        output, records, source_receipt["files"], repacker.DEFAULT_PROFILE,
        require_complete=True,
    )
    index = {"metadata": {"total_size": sum(item["data_bytes"] for item in records.values())},
             "weight_map": weight_map}
    repacker._write_json_atomic(output / "model.safetensors.index.json", index)
    identity = {
        "source_receipt_sha256": "d" * 64,
        "source_model_config_sha256": source_receipt["config_sha256"],
        "profile": repacker.DEFAULT_PROFILE,
        "output_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    receipt = {
        "schema": "mlx2.orcasaq-qwen38-affine-repack.v1",
        "qualification": "experimental-unqualified",
        "source_gguf_sha256": repacker.SOURCE_SHA256,
        "source_conversion_receipt": str(source / "mlx2-gguf-conversion.json"),
        "quantization": {
            "group_size": repacker.GROUP_SIZE,
            "mode": "affine",
            "bits_by_gguf_type": repacker.DEFAULT_PROFILE,
        },
        "identity": identity,
        "files": records,
        "config_sha256": identity["output_config_sha256"],
        "index_sha256": repacker.sha256(output / "model.safetensors.index.json"),
        "support_files": {},
        "added_quantization_error": "present-unmeasured",
        "notice": repacker.NOTICE,
    }
    receipt_path = output / "mlx2-affine-repack.json"
    repacker._write_json_atomic(receipt_path, receipt)
    monkeypatch.setattr(repacker, "_source_record", lambda _source: (source_receipt, "d" * 64))
    monkeypatch.setattr(repacker, "_quant_config", lambda *_args: output_config)

    assert repacker.repack(source, output) == {
        "state": "complete-verified",
        "receipt": str(receipt_path),
    }

    receipt["quantization"]["group_size"] = 32
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="invalid provenance or state"):
        repacker.repack(source, output)

    receipt["quantization"]["group_size"] = repacker.GROUP_SIZE
    receipt["files"]["packed"]["filename"] = "../escape.safetensors"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="not flat"):
        repacker.repack(source, output)
