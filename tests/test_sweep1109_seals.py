"""Bug sweep 2026-11-09, lane `seals`: persisted file stamps survive a reboot.

A volume's st_dev is renumbered at mount, so a seal written to disk must be
compared with ``same_stamp`` (device ignored). Each test seals synthetic files,
then rewrites only the saved stamp: a changed device id is accepted, a changed
size, mtime or inode is refused. CPU only; no checkpoint, no Metal kernel.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import struct
from pathlib import Path

import pytest

from mlx2.runtime.gguf import mlx_storage
from mlx2.runtime.gguf.mlx_storage import stamp

FIELDS = ("size", "mtime_ns", "inode")


def _bump(saved, field):
    saved = dict(saved)
    saved[field] = saved[field] + (7 if field != "device" else 4)
    return saved


def _rewrite(path, edit):
    value = json.loads(Path(path).read_text())
    edit(value)
    Path(path).write_text(json.dumps(value))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _FakeGGUF:
    def __init__(self, metadata=None, tensors=None, data_offset=0):
        self.metadata = metadata or {"general.name": "synthetic"}
        self.tensors = tensors or {}
        self.data_offset = data_offset

    def __call__(self, *_args, **_kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def inventory(self):
        return {"tensors": len(self.tensors)}


# --- deepseek_v4_gguf_artifact.inspect_artifact ---------------------------------

def _ds4_gguf_artifact(tmp_path, monkeypatch):
    from mlx2.adapters import deepseek_v4_gguf_artifact as m

    root = tmp_path / "ds4"
    root.mkdir()
    weight = root / m.FILENAME
    weight.write_bytes(b"GGUF synthetic packed payload \x00\x01\x02 for seal test")
    monkeypatch.setattr(m, "WEIGHT_BYTES", weight.stat().st_size)
    monkeypatch.setattr(m, "GGUFFile", _FakeGGUF())
    monkeypatch.setattr(m, "validate_header", lambda *_a: None)
    config = {"model_type": m.MODEL_TYPE, "mlx2_weight_format": "gguf-packed-v1",
              "gguf_file": m.FILENAME, "weight_sha256": m.WEIGHT_SHA256,
              "tokenizer_reference": m.REFERENCE_FILE}
    (root / "config.json").write_text(json.dumps(config))
    (root / m.REFERENCE_FILE).write_text(json.dumps({"model": {"type": "BPE"}}))
    seal = {"schema": m._SCHEMA, "filename": m.FILENAME, "sha256": m.WEIGHT_SHA256,
            "weight_stat": m._stamp(weight),
            "assets": {name: _sha(root / name) for name in m._ASSETS},
            "tokenizer_binding": {"aligned": True}}
    (root / m.INTEGRITY_FILE).write_text(json.dumps(seal))
    return m, root, root / m.INTEGRITY_FILE


def test_ds4_gguf_artifact_seal_ignores_device_only(tmp_path, monkeypatch):
    m, root, seal = _ds4_gguf_artifact(tmp_path, monkeypatch)
    _rewrite(seal, lambda v: v.update(weight_stat=_bump(v["weight_stat"], "device")))
    assert m.inspect_artifact(root)["identity"]["weight_sha256"] == m.WEIGHT_SHA256


@pytest.mark.parametrize("field", FIELDS)
def test_ds4_gguf_artifact_seal_refuses_real_change(tmp_path, monkeypatch, field):
    m, root, seal = _ds4_gguf_artifact(tmp_path, monkeypatch)
    _rewrite(seal, lambda v: v.update(weight_stat=_bump(v["weight_stat"], field)))
    with pytest.raises(ValueError, match="verified integrity seal"):
        m.inspect_artifact(root)


# --- deepseek_v4_dspark_artifact.inspect_sidecar -------------------------------

def _dspark_sidecar(tmp_path, monkeypatch):
    from mlx2.adapters import deepseek_v4_dspark_artifact as m

    path = tmp_path / m.FILENAME
    path.write_bytes(b"GGUF synthetic DSpark sidecar bytes \x05\x06 for seal test")
    monkeypatch.setattr(m, "SIZE", path.stat().st_size)
    monkeypatch.setattr(m, "_metadata", lambda _p: ({"block_size": 4}, 3))
    seal = {"schema": m.SCHEMA, "filename": m.FILENAME, "sha256": m.SIDECAR_SHA256,
            "weight_stat": m.file_stat(path), "config": {"block_size": 4},
            "tensor_count": 3,
            "verification": "actual_full_file_stream_sha256_before_after_stat"}
    m.seal_path(path).write_text(json.dumps(seal))
    return m, path, m.seal_path(path)


def test_dspark_seal_ignores_device_only(tmp_path, monkeypatch):
    m, path, seal = _dspark_sidecar(tmp_path, monkeypatch)
    _rewrite(seal, lambda v: v.update(weight_stat=_bump(v["weight_stat"], "device")))
    assert m.inspect_sidecar(path)["tensor_count"] == 3


@pytest.mark.parametrize("field", FIELDS)
def test_dspark_seal_refuses_real_change(tmp_path, monkeypatch, field):
    m, path, seal = _dspark_sidecar(tmp_path, monkeypatch)
    _rewrite(seal, lambda v: v.update(weight_stat=_bump(v["weight_stat"], field)))
    with pytest.raises(ValueError, match="missing, stale or incompatible"):
        m.inspect_sidecar(path)


# --- glm53_gguf_artifact.inspect_artifact -------------------------------------

def _glm53_artifact(tmp_path, monkeypatch):
    from mlx2.adapters import glm53_gguf_artifact as m

    root = tmp_path / "glm53"
    root.mkdir()
    (root / m.FILENAME).write_bytes(b"GGUF synthetic GLM payload \x09\x0a for seal test")
    config = {"model_type": m.MODEL_TYPE, "num_hidden_layers": 2}
    (root / "config.json").write_text(json.dumps(config))
    (root / m.REFERENCE).write_text(json.dumps({"added_tokens": []}))
    monkeypatch.setattr(m, "GGUFFile", _FakeGGUF())
    monkeypatch.setattr(m, "config_from_metadata", lambda _md: dict(config))
    monkeypatch.setattr(m, "validate_inventory", lambda *_a: None)
    seal = {"schema": "mlx2.glm53-gguf.v1", "weight_sha256": m.WEIGHT_SHA256,
            "weight_stat": stamp(root / m.FILENAME),
            "config_sha256": _sha(root / "config.json"),
            "reference_sha256": _sha(root / m.REFERENCE),
            "ordinary_payload_bytes": 1, "qualified": False}
    (root / m.SEAL).write_text(json.dumps(seal))
    return m, root, root / m.SEAL


def test_glm53_artifact_seal_ignores_device_only(tmp_path, monkeypatch):
    m, root, seal = _glm53_artifact(tmp_path, monkeypatch)
    _rewrite(seal, lambda v: v.update(weight_stat=_bump(v["weight_stat"], "device")))
    assert m.inspect_artifact(root)["qualified"] is False


@pytest.mark.parametrize("field", FIELDS)
def test_glm53_artifact_seal_refuses_real_change(tmp_path, monkeypatch, field):
    m, root, seal = _glm53_artifact(tmp_path, monkeypatch)
    _rewrite(seal, lambda v: v.update(weight_stat=_bump(v["weight_stat"], field)))
    with pytest.raises(ValueError, match="differs from CPU integrity preparation"):
        m.inspect_artifact(root)


# --- glm53_gguf load-time recheck against the sealed weight_stat ---------------

def test_glm53_load_recheck_uses_reboot_tolerant_compare(tmp_path):
    """The post-preload check compares a live stamp with the seal's weight_stat.

    That code runs only after a full model preload, so this test pins the
    compare itself and exercises the helper on the same stamp shapes.
    """
    from mlx2.runtime.immutable_weight_pages import same_stamp

    source = (Path(__file__).resolve().parents[1] / "src/mlx2/adapters/glm53_gguf.py").read_text()
    assert 'stamp(root / FILENAME) != artifact["weight_stat"]' not in source
    assert 'not same_stamp(stamp(root / FILENAME), artifact["weight_stat"])' in source
    weight = tmp_path / "w.gguf"
    weight.write_bytes(b"GGUF glm load recheck \x0b")
    saved = json.loads(json.dumps(stamp(weight)))
    assert same_stamp(stamp(weight), _bump(saved, "device"))
    for field in FIELDS:
        assert not same_stamp(stamp(weight), _bump(saved, field))


# --- native_read_view.validate_read_view ---------------------------------------

def _read_view(tmp_path, monkeypatch):
    from mlx2.runtime.gguf import native_read_view as m

    source = tmp_path / "source.gguf"
    source.write_bytes(b"GGUF synthetic source for read view \x0c\x0d")
    header = {"blk.0.w": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
    encoded = json.dumps(header).encode()
    view = tmp_path / "view.safetensors"
    view.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 4)
    monkeypatch.setattr(m, "GGUFFile", _FakeGGUF(data_offset=8 + len(encoded)))
    monkeypatch.setattr(m, "expected_header", lambda _f: header)
    receipt = view.with_suffix(view.suffix + ".json")
    receipt.write_text(json.dumps({"schema": m.SCHEMA, "source_sha256": "ab" * 32,
                                   "source_stat": stamp(source),
                                   "view_stat": stamp(view)}))
    return m, source, view, receipt


@pytest.mark.parametrize("key", ("source_stat", "view_stat"))
def test_read_view_receipt_ignores_device_only(tmp_path, monkeypatch, key):
    m, source, view, receipt = _read_view(tmp_path, monkeypatch)
    _rewrite(receipt, lambda v: v.update({key: _bump(v[key], "device")}))
    assert m.validate_read_view(source, view, source_sha256="ab" * 32)["schema"] == m.SCHEMA


@pytest.mark.parametrize("key", ("source_stat", "view_stat"))
@pytest.mark.parametrize("field", FIELDS)
def test_read_view_receipt_refuses_real_change(tmp_path, monkeypatch, key, field):
    from mlx2.runtime.gguf.reader import GGUFError

    m, source, view, receipt = _read_view(tmp_path, monkeypatch)
    _rewrite(receipt, lambda v: v.update({key: _bump(v[key], field)}))
    with pytest.raises(GGUFError, match="source/seal changed"):
        m.validate_read_view(source, view, source_sha256="ab" * 32)


# --- mlx_storage: inspect_storage, MLXPackedFile._get / packed_view ------------

SHARD = "model-00001.safetensors"


def _packed_storage(tmp_path):
    root = tmp_path / "packed"
    root.mkdir()
    header = {"token_embd.weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = struct.pack("<4f", 0.25, -1.5, 3.0, 7.75)
    (root / SHARD).write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    manifest = {"schema": mlx_storage.SCHEMA, "source_sha256": "cd" * 32, "metadata": {},
                "tensors": {"token_embd.weight": {
                    "name": "token_embd.weight", "shape": [4], "ggml_type": 0, "offset": 0,
                    "nbytes": 16, "shard": SHARD, "key": "token_embd.weight"}},
                "linear_fp32": {},
                "shards": {SHARD: {"bytes": (root / SHARD).stat().st_size,
                                   "sha256": _sha(root / SHARD)}}}
    mlx_storage.atomic_json(root / mlx_storage.MANIFEST, manifest)
    mlx_storage.atomic_json(root / mlx_storage.SEAL, {
        "schema": mlx_storage.SCHEMA,
        "manifest_sha256": mlx_storage.sha256(root / mlx_storage.MANIFEST),
        "shards": {SHARD: stamp(root / SHARD)}})
    return root, root / mlx_storage.SEAL


def _bump_shard(field):
    return lambda v: v["shards"].update({SHARD: _bump(v["shards"][SHARD], field)})


def test_inspect_storage_seal_ignores_device_only(tmp_path):
    root, seal = _packed_storage(tmp_path)
    _rewrite(seal, _bump_shard("device"))
    manifest, _, headers = mlx_storage.inspect_storage(root, verify_payload=True)
    assert set(headers) == {SHARD} and manifest["tensors"]


@pytest.mark.parametrize("field", FIELDS)
def test_inspect_storage_seal_refuses_real_change(tmp_path, field):
    root, seal = _packed_storage(tmp_path)
    _rewrite(seal, _bump_shard(field))
    with pytest.raises(mlx_storage.GGUFError, match="changed after verification"):
        mlx_storage.inspect_storage(root)


def _opened(tmp_path, field):
    """Open with a device-only drift on disk, then apply `field` to the held seal."""
    root, seal = _packed_storage(tmp_path)
    _rewrite(seal, _bump_shard("device"))
    packed = mlx_storage.MLXPackedFile(root)
    if field != "device":
        packed.seal["shards"][SHARD] = _bump(packed.seal["shards"][SHARD], field)
    return packed


def test_packed_view_ignores_device_only(tmp_path):
    packed = _opened(tmp_path, "device")
    try:
        view = packed.packed_view("token_embd.weight")
        assert struct.unpack("<4f", bytes(view)) == (0.25, -1.5, 3.0, 7.75)
        view.release()
    finally:
        packed.close()


@pytest.mark.parametrize("field", FIELDS)
def test_packed_view_refuses_real_change(tmp_path, field):
    packed = _opened(tmp_path, field)
    with contextlib.closing(packed), pytest.raises(mlx_storage.GGUFError,
                                                   match="before host read"):
        packed.packed_view("token_embd.weight")


def test_packed_get_ignores_device_only(tmp_path):
    packed = _opened(tmp_path, "device")
    sentinel = object()
    packed._loaded[SHARD] = {"token_embd.weight": sentinel}  # skip mx.load
    try:
        assert packed.device_array("token_embd.weight") is sentinel
    finally:
        packed.close()


@pytest.mark.parametrize("field", FIELDS)
def test_packed_get_refuses_real_change(tmp_path, field):
    packed = _opened(tmp_path, field)
    packed._loaded[SHARD] = {"token_embd.weight": object()}
    with contextlib.closing(packed), pytest.raises(mlx_storage.GGUFError,
                                                   match="changed before load"):
        packed.device_array("token_embd.weight")
