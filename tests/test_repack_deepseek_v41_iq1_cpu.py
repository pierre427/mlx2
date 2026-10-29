"""CPU inverse algebra, persistent byte checks, COW and resume boundaries."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest


@pytest.fixture(scope="module")
def repacker():
    directory = Path(__file__).resolve().parents[1] / "scripts/research"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("repack_v41_iq1_cpu_test", directory / "repack_deepseek_v41_iq1.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(directory))


def raw_blocks(count, *, scales=None, unique=True):
    raw = np.random.default_rng(1421).integers(0, 256, (count, 50), np.uint8)
    scales = (np.resize(np.array([.5, -.125, 2**-24, -2**-24, 65504, -65504], "<f2"), count)
              if scales is None else np.asarray(scales, "<f2"))
    raw[:, :2] = scales.reshape(-1, 1).view(np.uint8)
    if unique:
        high = raw[:, 34:].view("<u2")
        high[:, 0] &= 0x8fff  # factor1
        high[:, 1] |= 0x7000  # factor15: together these force a unique base.
    return raw


@pytest.mark.parametrize("shape", [(256,), (7, 512), (2, 3, 256)])
def test_unique_encoding_recovers_original_bytes(repacker, shape):
    raw = raw_blocks(int(np.prod(shape)) // 256)
    arrays = repacker.convert_affine(raw, 19, shape)
    packed, proof = repacker.repack_iq1(*arrays, shape)
    assert packed.dtype == np.uint8 and packed.flags.c_contiguous
    assert packed.tobytes() == raw.tobytes()
    assert proof["forward_affine_bitwise_exact"] and proof["ambiguous_encoding_blocks"] == 0
    assert proof["unique_original_encoding_blocks"] == len(raw)


def test_nonunique_scale_encoding_is_canonical_and_not_original_byte_claim(repacker):
    raw = raw_blocks(1, scales=[.5])
    high = raw[:, 34:].view("<u2")
    high[:] = (high & 0x8fff) | 0x1000  # factor3 in all8 groups
    arrays = repacker.convert_affine(raw, 19, (256,))
    compact, proof = repacker.repack_iq1(*arrays, (256,))
    assert compact.tobytes() != raw.tobytes()
    assert proof["ambiguous_encoding_blocks"] == 1 and proof["unique_original_encoding_blocks"] == 0
    assert np.frombuffer(compact[:2], "<f2")[0] == 1.5
    again = repacker.convert_affine(compact, 19, (256,))
    assert all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(arrays, again))
    assert np.array_equal(repacker.repack_iq1(*again, (256,))[0], compact)


def test_all_finite_half_scales_and_subnormals_recover_exact(repacker):
    half = np.arange(65536, dtype=np.uint16).view(np.float16)
    half = half[np.isfinite(half)]
    totals = dict(blocks=0, zero_scale_blocks=0, half_d_nonzero_subnormal_blocks=0)
    for first in range(0, len(half), 2048):
        scales = half[first:first + 2048]
        raw = raw_blocks(len(scales), scales=scales)
        shape = (len(raw), 256)
        compact, proof = repacker.repack_iq1(*repacker.convert_affine(raw, 19, shape), shape)
        nonzero = scales != 0
        assert np.array_equal(compact.reshape(-1, 50)[nonzero], raw[nonzero])
        for key in totals:
            totals[key] += proof[key]
    assert totals == dict(blocks=63488, zero_scale_blocks=2, half_d_nonzero_subnormal_blocks=2046)


def test_exhaustive_grid_recovers_every_codebook_index(repacker):
    grid = repacker._iq1_affine_grid()
    assert len(np.unique(grid)) == 2048
    raw = raw_blocks(len(grid), scales=np.full(len(grid), .5, "<f2"))
    index = np.arange(len(grid), dtype=np.uint16)
    raw[:, 2:34] = (index & 255).astype(np.uint8)[:, None]
    hi = index >> 8
    upper = hi[:, None] | hi[:, None] << 3 | hi[:, None] << 6 | hi[:, None] << 9
    upper = upper | (np.arange(8, dtype=np.uint16)[None, :] << 12)
    upper[:, 1] |= 0x7000
    raw[:, 34:] = upper.astype("<u2").view(np.uint8)
    shape = (len(raw), 256)
    compact, proof = repacker.repack_iq1(*repacker.convert_affine(raw, 19, shape), shape)
    assert compact.tobytes() == raw.tobytes() and proof["ambiguous_encoding_blocks"] == 0


def test_gcd_inverse_matches_independent_eight_candidate_oracle(repacker):
    raw = raw_blocks(4096, unique=False)
    raw[:2048, 34:].view("<u2")[:] &= 0x8fff
    raw[:2048, 34:].view("<u2")[:] |= 0x1000
    arrays = repacker.convert_affine(raw, 19, (4096, 256))
    groups = arrays[1].reshape(-1, 8)
    expected = np.zeros(len(raw), np.float32)
    count = np.zeros(len(raw), np.uint8)
    candidate_bits = np.zeros((len(raw), 8), np.uint32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for first_factor in range(1, 16, 2):
            base = (groups[:, 0] / np.float32(first_factor)).astype(np.float16).astype(np.float32)
            ratio = np.rint(groups / base[:, None])
            valid = (np.isfinite(base) & (base != 0) & np.all((ratio >= 1) & (ratio <= 15)
                     & (np.remainder(ratio, 2) == 1) & (base[:, None] * ratio == groups), axis=1))
            first = valid & (count == 0)
            expected[first] = base[first]
            count += valid.astype(np.uint8)
            candidate_bits[valid, first_factor // 2] = base[valid].view(np.uint32)
    compact, proof = repacker.repack_iq1(*arrays, (4096, 256))
    actual = compact.reshape(-1, 50)[:, :2].view("<f2").reshape(-1).astype(np.float32)
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    ordered = np.sort(candidate_bits, axis=1)
    unique = np.count_nonzero((ordered != 0) & (ordered != np.pad(ordered[:, :-1], ((0, 0), (1, 0)))), axis=1)
    assert proof["ambiguous_encoding_blocks"] == int(np.count_nonzero(unique > 1))


def test_invalid_arrays_and_non_iq1_metadata_rejected(repacker):
    arrays = repacker.convert_affine(raw_blocks(1), 19, (256,))
    q, s, b = arrays
    invalid_q = q.copy()
    invalid_q[0] |= np.uint32(3)
    with pytest.raises(ValueError, match="codebook"):
        repacker.repack_iq1(invalid_q, s, b, (256,))
    invalid_b = b.copy()
    invalid_b[0] = np.nextafter(invalid_b[0], np.float32(np.inf))
    with pytest.raises(ValueError, match="quarter-delta"):
        repacker.repack_iq1(q, s, invalid_b, (256,))
    invalid_s = s.copy()
    invalid_s[0] *= 100
    invalid_b = invalid_s * np.float32(-.875)
    with pytest.raises(ValueError, match="common finite-half"):
        repacker.repack_iq1(q, invalid_s, invalid_b, (256,))
    invalid_s[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        repacker.repack_iq1(q, invalid_s, b, (256,))
    with pytest.raises(ValueError, match="budget"):
        repacker.repack_iq1(q, s, b, (256,), max_elements=255)
    with pytest.raises(ValueError, match="shape/dtype"):
        repacker.repack_iq1(q, s.astype(np.float16), b, (256,))


@pytest.mark.skipif(sys.platform != "darwin", reason="actual APFS clonefile check")
def test_cow_clone_preserves_source_stat_and_bytes(repacker, tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"source payload" * 500)
    before, original = repacker.stamp(source), source.read_bytes()
    repacker.clone_file(source, target)
    assert repacker.stamp(source) == before and target.read_bytes() == original
    assert source.stat().st_ino != target.stat().st_ino
    target.write_bytes(b"changed destination")
    assert repacker.stamp(source) == before and source.read_bytes() == original
    with pytest.raises(ValueError, match="absent destination"):
        repacker.clone_file(source, target)


@pytest.fixture
def affine_artifact(repacker, tmp_path, monkeypatch):
    from safetensors.numpy import save_file
    from mlx2.adapters import deepseek_v41_mlx_artifact as admission
    source, target = tmp_path / "affine", tmp_path / "compact"
    source.mkdir()
    tensors, files = {}, {}
    expert_paths = [f"experts/layer-000/expert-{i:03d}.safetensors" for i in range(2)]
    arrays = {}
    for plane in ("gate", "up", "down"):
        shape = (3, 256)
        raw = raw_blocks(3)
        q, s, b = repacker.convert_affine(raw, 19, shape)
        arrays.update({plane + ".qweight": q, plane + ".scales": s, plane + ".biases": b})
        name = "blk.0.ffn_" + plane + "_exps.weight"
        tensors[name] = dict(shape=[256, 3, 2], numpy_shape=[2, 3, 256], ggml_type=19,
                             offset=0, nbytes=2 * len(raw.tobytes()), storage=dict(kind="expert-affine", expert_files=expert_paths,
                             keys={k: plane + "." + k for k in ("qweight", "scales", "biases")}, bits=2, group_size=32, mode="affine"))
    for expert_path in expert_paths:
        path = source / expert_path
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(arrays, str(path))
        raw = path.read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + size])
        header.pop("__metadata__", None)
        files[expert_path] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest(), stat=repacker.stamp(path),
                                 tensors=header, header_bytes=8 + size)
    for name, data in (("config.json", b'{}'), ("tokenizer.reference.json", b'{}'), ("engram/layer-001.bin", bytes(range(256)) * 3)):
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files[name] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest(), stat=repacker.stamp(p), tensors={}, header_bytes=0)
    original = dict(tensors=tensors, files=files, source=dict(checkpoint="synthetic-cpu-fixture"), metadata={},
                    assets=["config.json", "tokenizer.reference.json"], derived_identity_sha256="a" * 64)
    manifest_path = source / "mlx2-v41-native-manifest.json"
    repacker.atomic_json(manifest_path, original)
    fingerprint = repacker.file_hash(manifest_path)
    def inspect(root):
        assert Path(root) == source
        for name, row in files.items():
            assert repacker.stamp(source / name) == row["stat"]
        return dict(root=source, manifest=original, identity=dict(fingerprint=fingerprint), weight_stat=repacker.stamp(manifest_path))
    monkeypatch.setattr(admission, "inspect_artifact", inspect)
    return source, target, original


@pytest.mark.skipif(sys.platform != "darwin", reason="standalone writer requires APFS clonefile")
def test_complete_artifact_clone_bytes_and_manifest_identity(repacker, affine_artifact):
    from safetensors.numpy import load_file
    source, target, original = affine_artifact
    before = {p: repacker.stamp(source / p) for p in original["files"]}
    proof = repacker.repack(source, target, reserve_bytes=0, phase=lambda _: None)
    assert proof["passed"] and proof["forward_affine_bitwise_exact"]
    result = json.loads((target / repacker.MANIFEST).read_text())
    assert result["finished"] and result["source_affine"]["derived_identity_sha256"] == "a" * 64
    assert result["source_affine"]["fingerprint"] == repacker.file_hash(source / "mlx2-v41-native-manifest.json")
    assert all(repacker.stamp(source / p) == seal for p, seal in before.items())
    for name, row in result["files"].items():
        assert repacker.verify_file(target, name, row) == row["stat"]
    path = "experts/layer-000/expert-000.safetensors"
    compact = load_file(target / path)
    affine = load_file(source / path)
    for plane in ("gate", "up", "down"):
        recon = repacker.convert_affine(compact[plane + ".data"], 19, (3, 256))
        assert all(np.array_equal(a.view(np.uint32), affine[plane + "." + k].view(np.uint32))
                   for a, k in zip(recon, ("qweight", "scales", "biases")))
    source.rename(source.with_name("affine-hidden"))
    assert (target / "engram/layer-001.bin").read_bytes() == bytes(range(256)) * 3


def test_zero_file_limit_and_completed_artifact_refusal(repacker, affine_artifact):
    source, target, original = affine_artifact
    # Zero file limit writes an unfinished, source-bound manifest only.
    proof = repacker.repack(source, target, reserve_bytes=0, max_files=0, phase=lambda _: None)
    assert not proof["finished"]
    assert not list(target.rglob("*.safetensors"))
    if sys.platform != "darwin":
        return
    repacker.repack(source, target, reserve_bytes=0, phase=lambda _: None)
    assert source.is_dir()
    with pytest.raises(ValueError, match="already complete"):
        repacker.repack(source, target, reserve_bytes=0, phase=lambda _: None)


@pytest.mark.skipif(sys.platform != "darwin", reason="standalone writer requires APFS clonefile")
def test_resume_reuses_verified_experts(repacker, affine_artifact):
    source, target, _ = affine_artifact
    proof = repacker.repack(source, target, reserve_bytes=0, max_files=1, phase=lambda _: None)
    assert not proof["finished"] and proof["completed_experts"] == 1
    first = next(target.rglob("*.safetensors"))
    before = repacker.stamp(first)
    assert repacker.repack(source, target, reserve_bytes=0, phase=lambda _: None)["passed"]
    assert repacker.stamp(first) == before


def test_resume_rejects_same_size_output_corruption(repacker, affine_artifact):
    source, target, _ = affine_artifact
    repacker.repack(source, target, reserve_bytes=0, max_files=1, phase=lambda _: None)
    first = next(target.rglob("*.safetensors"))
    raw = bytearray(first.read_bytes())
    raw[-1] ^= 1
    first.write_bytes(raw)
    with pytest.raises(ValueError, match="hash/extent"):
        repacker.repack(source, target, reserve_bytes=0, phase=lambda _: None)


def test_failed_stored_chunk_never_publishes_receipt(repacker, affine_artifact, monkeypatch):
    source, target, _ = affine_artifact
    original = repacker.write_at
    def corrupt(fd, data, offset):
        if offset > 0:
            data = bytearray(memoryview(data).cast("B"))
            data[0] ^= 1
        original(fd, data, offset)
    monkeypatch.setattr(repacker, "write_at", corrupt)
    with pytest.raises(ValueError, match="stored compact bytes"):
        repacker.repack(source, target, reserve_bytes=0, max_files=1, phase=lambda _: None)
    assert not list((target / "conversion-state").glob("*.json"))
    assert not json.loads((target / repacker.MANIFEST).read_text())["finished"]


def test_clone_failure_has_no_physical_copy_fallback(repacker, tmp_path, monkeypatch):
    class Missing:
        pass
    source, target = tmp_path / "source", tmp_path / "target"
    source.write_bytes(b"untouched")
    before = repacker.stamp(source)
    monkeypatch.setattr(repacker.ctypes, "CDLL", lambda *a, **k: Missing())
    with pytest.raises(OSError, match="fallback is forbidden"):
        repacker.clone_file(source, target)
    assert not target.exists() and repacker.stamp(source) == before
