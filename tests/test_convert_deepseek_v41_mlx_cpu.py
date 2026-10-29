"""Synthetic CPU artifact conversion, interruption and corruption boundaries."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def converter():
    directory = Path(__file__).resolve().parents[1] / "scripts/research"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("convert_v41_mlx_cpu_test", directory / "convert_deepseek_v41_mlx.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(directory))


def packed(kind, shape):
    width, size = (256, 50) if kind == 19 else (32, 34)
    blocks = np.random.default_rng(481).integers(0, 256, (int(np.prod(shape)) // width, size), np.uint8)
    scales = np.resize(np.array([0., -0., .5, -.125], "<f2"), len(blocks))
    blocks[:, :2] = scales.reshape(-1, 1).view(np.uint8)
    return blocks.tobytes()


@pytest.fixture
def model(tmp_path, monkeypatch):
    from mlx2.adapters import deepseek_v41_artifact as admission
    source = tmp_path / "source"
    source.mkdir()
    rows = [("token_embd.weight", 8, (3, 32), packed(8, (3, 32))),
            ("output_norm.weight", 0, (32,), np.arange(32, dtype="<f4").tobytes()),
            ("blk.0.example.weight", 1, (2, 32), np.arange(64, dtype="<f2").tobytes()),
            ("blk.0.grouped.weight", 8, (2, 2, 32), packed(8, (2, 2, 32))),
            ("blk.1.engram_embd.weight", 24, (3, 264), bytes(range(256)) * 3 + bytes(range(24)))]
    rows += [(f"blk.0.ffn_{kind}_exps.weight", 19, (2, 3, 256), packed(19, (2, 3, 256)))
             for kind in ("gate", "up", "down")]
    header = b"GGUF" + struct.pack("<IQQ", 3, len(rows), 0)
    body = bytearray()
    for name, kind, shape, raw in rows:
        body.extend(b"\0" * (-len(body) % 32))
        encoded = name.encode()
        header += struct.pack("<Q", len(encoded)) + encoded + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape[::-1]) + struct.pack("<IQ", kind, len(body))
        body.extend(raw)
    header += b"\0" * (-len(header) % 32)
    (source / admission.FILENAME).write_bytes(header + body)
    for name in ("config.json", "mlx2-v41-integrity.json", "tokenizer.reference.json"):
        (source / name).write_text("{}")
    monkeypatch.setattr(admission, "inspect_artifact", lambda path: dict(identity=dict(fingerprint="synthetic-cpu-fixture",
                                                                                     weight_sha256=hashlib.sha256(header + body).hexdigest())))
    return source, tmp_path / "native", rows


def test_exhaustive_algebra(converter):
    result = converter.algebra_selftest()
    assert result["passed"] and result["finite_half_scales"] == 63488
    assert result["iq1_codebook_entries"] == 2048
    assert result["scalar_cases"] > 19_000_000
    assert result["synthetic_signed_zero_differences"] > 0


def test_complete_standalone_conversion_preserves_geometry_and_raw_bytes(converter, model):
    from safetensors.numpy import load_file
    source, output, rows = model
    result = converter.convert(source, output, reserve_bytes=0, phase=lambda x: None)
    assert result["finished"] and result["all_file_hashes_checked"]
    manifest = json.loads((output / converter.MANIFEST).read_text())
    assert manifest["finished"] and len(manifest["tensors"]) == len(rows)
    assert manifest["numeric_profile"] == "native_f32_qmm"
    assert manifest["verification"]["source_tensor_bytes"] == sum(len(r[3]) for r in rows)
    arrays = {p: load_file(output / p) for p in manifest["files"] if p.endswith(".safetensors")}
    for name, kind, shape, raw in rows:
        storage = manifest["tensors"][name]["storage"]
        if kind == 24:
            assert (output / storage["file"]).read_bytes() == raw
        elif kind in (0, 1):
            assert arrays[storage["file"]][storage["key"]].tobytes() == raw
        elif kind == 8:
            converted = tuple(arrays[storage["file"]][storage["keys"][k]] for k in ("qweight", "scales", "biases"))
            assert converter.validate_affine(raw, kind, shape, converted)["value_exact"]
        else:
            size = len(raw) // shape[0]
            for i, path in enumerate(storage["expert_files"]):
                converted = tuple(arrays[path][storage["keys"][k]] for k in ("qweight", "scales", "biases"))
                assert converter.validate_affine(raw[i * size:(i + 1) * size], kind, shape[1:], converted)["value_exact"]
    assert sum(r.get("proof", {}).get("signed_zero_differences", 0) for r in manifest["files"].values()) > 0
    # No GGUF or source directory is needed for final integrity verification.
    moved = source.with_name("source-hidden")
    source.rename(moved)
    assert converter.verify_manifest(output, manifest)["passed"]


def test_resume_does_not_rewrite_completed_files(converter, model):
    source, output, _ = model
    first = converter.convert(source, output, reserve_bytes=0, max_files=1, phase=lambda x: None)
    assert not first["finished"] and first["newly_written"] == 1
    assert not json.loads((output / converter.MANIFEST).read_text())["finished"]
    saved = next(output.rglob("*.safetensors"))
    before = converter.stamp(saved)
    assert converter.convert(source, output, reserve_bytes=0, phase=lambda x: None)["finished"]
    assert converter.stamp(saved) == before


def test_resume_rejects_same_size_corruption(converter, model):
    source, output, _ = model
    converter.convert(source, output, reserve_bytes=0, max_files=1, phase=lambda x: None)
    saved = next(output.rglob("*.safetensors"))
    data = bytearray(saved.read_bytes())
    data[-1] ^= 1
    saved.write_bytes(data)
    with pytest.raises(ValueError, match="failed verification"):
        converter.convert(source, output, reserve_bytes=0, phase=lambda x: None)


def test_interrupted_partial_file_can_resume(converter, model, monkeypatch):
    source, output, _ = model
    original = converter.write_safetensors
    def fail(*args, **kwargs):
        Path(args[4]).write_bytes(b"interrupted")
        raise RuntimeError("injected stop")
    monkeypatch.setattr(converter, "write_safetensors", fail)
    with pytest.raises(RuntimeError, match="injected stop"):
        converter.convert(source, output, reserve_bytes=0, phase=lambda x: None)
    assert len(list(output.rglob("*.partial"))) == 1
    monkeypatch.setattr(converter, "write_safetensors", original)
    assert converter.convert(source, output, reserve_bytes=0, phase=lambda x: None)["finished"]
    assert not list(output.rglob("*.partial"))


def test_source_replacement_during_write_refuses_completion(converter, model, monkeypatch):
    source, output, _ = model
    original = converter.write_safetensors
    def replace_source(*args, **kwargs):
        result = original(*args, **kwargs)
        path = next(source.glob("*.gguf"))
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        return result
    monkeypatch.setattr(converter, "write_safetensors", replace_source)
    with pytest.raises(ValueError, match="GGUF changed"):
        converter.convert(source, output, reserve_bytes=0, phase=lambda x: None)
    assert not json.loads((output / converter.MANIFEST).read_text())["finished"]


def test_disk_reserve_and_output_contamination_fail_closed(converter, model, monkeypatch):
    source, output, _ = model
    monkeypatch.setattr(converter.shutil, "disk_usage", lambda path: SimpleNamespace(free=99))
    with pytest.raises(OSError, match="reserve"):
        converter.convert(source, output, reserve_bytes=100, phase=lambda x: None)
    assert not list(output.rglob("*.safetensors"))
    other = output.with_name("other")
    other.mkdir()
    (other / "unrelated").write_text("keep")
    with pytest.raises(ValueError, match="unrelated"):
        converter.convert(source, other, reserve_bytes=0, phase=lambda x: None)


def test_lock_and_symlink_paths_refuse_unsafe_ownership(converter, tmp_path):
    with converter.conversion_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another converter"):
            with converter.conversion_lock(tmp_path):
                pass
    (tmp_path / "alias").symlink_to(tmp_path / "other")
    for relative in ("../escape", "/absolute", "alias/data.bin"):
        with pytest.raises(ValueError):
            converter.safe_path(tmp_path, relative)


def test_short_reads_and_writes_are_completed(converter, tmp_path, monkeypatch):
    path = tmp_path / "bytes"
    path.write_bytes(bytes(range(64)))
    fd = os.open(path, os.O_RDWR)
    pread, pwrite = os.pread, os.pwrite
    try:
        monkeypatch.setattr(converter.os, "pread", lambda descriptor, size, offset: pread(descriptor, min(size, 7), offset))
        assert converter.read_exact(fd, 64, 0) == bytes(range(64))
        monkeypatch.setattr(converter.os, "pwrite", lambda descriptor, data, offset: pwrite(descriptor, data[:3], offset))
        converter.write_at(fd, bytes(range(63, -1, -1)), 0)
        assert converter.read_exact(fd, 64, 0) == bytes(range(63, -1, -1))
        with pytest.raises(EOFError):
            converter.read_exact(fd, 65, 0)
    finally:
        os.close(fd)


def test_manifest_inventory_and_recipe_corruption_detected(converter, model):
    source, output, _ = model
    converter.convert(source, output, reserve_bytes=0, phase=lambda x: None)
    original = json.loads((output / converter.MANIFEST).read_text())
    manifest = copy.deepcopy(original)
    manifest["files"].pop("weights.safetensors")
    with pytest.raises(ValueError, match="inventory"):
        converter.verify_manifest(output, manifest)
    manifest = copy.deepcopy(original)
    manifest["tensors"]["token_embd.weight"]["storage"]["bits"] = 4
    with pytest.raises(ValueError, match="recipe"):
        converter.verify_manifest(output, manifest)
    manifest = copy.deepcopy(original)
    key = manifest["tensors"]["token_embd.weight"]["storage"]["keys"]["qweight"]
    manifest["files"]["weights.safetensors"]["tensors"][key]["shape"][-1] *= 2
    with pytest.raises(ValueError, match="geometry"):
        converter.verify_manifest(output, manifest)
