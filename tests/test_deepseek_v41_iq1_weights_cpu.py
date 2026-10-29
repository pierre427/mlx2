"""Standalone compact file/dispatch/ownership contracts; no GPU kernel run."""

from copy import deepcopy
import hashlib
import json
import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx2.adapters import deepseek_v41_iq1_weights as compact
from mlx2.adapters import deepseek_v41_mlx_weights as native
from mlx2.runtime.gguf.segmented_prefill import expert_prefill_context
from mlx2.runtime.io_ownership import ReadTimeout
from test_deepseek_v41_mlx_weights_cpu import matrix, write_native_artifact

mx.set_default_device(mx.cpu)


def write_compact_artifact(root, dense, packed, args):
    """Build complete compact synthetic files without a GGUF source."""
    rng = np.random.default_rng(4192)
    affine = {layer: [{kind: matrix(args.moe_intermediate_size, args.hidden_size, 2, rng)
                       if kind != "down" else matrix(args.hidden_size, args.moe_intermediate_size, 2, rng)
                       for kind in compact.FIELDS} for _ in rows] for layer, rows in packed.items()}
    art = write_native_artifact(root, dense, affine, args)
    manifest = art["manifest"]
    manifest.update(schema=compact.SCHEMA, numeric_profile=compact.PROFILE,
                    model_type="deepseek41_iq1", format="mlx2-v41-iq1s-v1")
    for layer, rows in packed.items():
        for expert, planes in enumerate(rows):
            relative = f"experts/layer-{layer:03d}/expert-{expert:03d}.safetensors"
            path = art["root"]/relative
            save_file({kind+".data": planes[kind].reshape(-1) for kind in compact.FIELDS}, str(path))
            data = path.read_bytes()
            base = 8+int.from_bytes(data[:8], "little")
            manifest["files"][relative] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                stat=native._stamp(path.stat()), header_bytes=base, tensors=json.loads(data[8:base]))
        for kind in compact.FIELDS:
            row = manifest["tensors"][f"blk.{layer}.ffn_{kind}_exps.weight"]
            row["storage"].update(kind="expert-iq1_s", keys={"data": kind+".data"})
            for key in ("bits", "group_size", "mode"):
                row["storage"].pop(key, None)
    (art["root"]/native.MANIFEST).unlink()
    (art["root"]/compact.MANIFEST).write_text(json.dumps(manifest))
    return art


@pytest.fixture
def artifact(tmp_path):
    rng = np.random.default_rng(4191)
    args = SimpleNamespace(hidden_size=256, moe_intermediate_size=256, num_experts_per_tok=2)
    dense = {"token_embd.weight": matrix(7, 256, 8, rng),
             "output_norm.weight": np.ones(256, np.float32)}
    packed = {}
    for layer in range(2):
        packed[layer] = []
        for expert in range(3):
            planes = {}
            for kind in compact.FIELDS:
                raw = rng.integers(0, 256, (256, 50), np.uint8)
                raw[:, :2] = np.frombuffer(np.float16(.03125).tobytes(), np.uint8)
                planes[kind] = raw.reshape(-1)
            packed[layer].append(planes)
    return write_compact_artifact(tmp_path/"compact", dense, packed, args), packed


def test_compact_plan_uses_packed_extents_and_shared_owned_manifest(artifact):
    art, _ = artifact
    row_bytes = 3*256*50
    plan = compact.compact_weight_plan(art, row_bytes)
    assert plan["minimum_expert_cache_bytes"] == row_bytes
    assert plan["expert_active_bound_bytes"] == 3*row_bytes
    assert plan["expert_table_bytes"] == 6*row_bytes
    weights = compact.CompactV41Weights(art, expert_cache_bytes=row_bytes)
    try:
        assert weights.resident_bytes() == 0
        assert weights.expert_stream.manifest is weights.artifact["manifest"]
        assert weights.expert_prefill_policy == "reference"
        assert weights.expert_stream.apc_revision() in weights.compute_policy
        assert weights.expert_stream.receipt()["numeric_profile"] == compact.PROFILE
        assert weights.expert_stream.receipt()["host_dequantization"] is False
    finally:
        weights.close()


def test_compact_rows_are_exact_and_remapped_without_weight_concat_or_gguf(artifact, monkeypatch):
    from mlx2.runtime.gguf import reader
    art, packed = artifact
    def forbidden(*args, **kwargs):
        raise AssertionError("provider must not open GGUF/dequantize/concatenate weights")
    monkeypatch.setattr(reader.GGUFFile, "__init__", forbidden)
    monkeypatch.setattr(mx, "concatenate", forbidden)
    monkeypatch.setattr(mx, "dequantize", forbidden)
    store = compact.CompactExpertStore(art, 3*256*50)
    try:
        routes = mx.array([[2, 0], [1, 2]], mx.uint32)
        with store.window():
            inverse, planes = store.gather("blk.0", routes)
            np.testing.assert_array_equal(inverse, [[2, 0], [1, 2]])
            for kind, values in zip(compact.FIELDS, planes):
                for expert, value in enumerate(values):
                    assert value.ggml_type == 19 and value.shape == (1, 256, 256)
                    assert value.data.dtype == mx.uint8 and value.data.ndim == 1
                    assert np.array(value.data).tobytes() == packed[0][expert][kind].tobytes()
            assert store.live_bytes() == 3*3*256*50
            assert store._cache_bytes == 3*256*50
        assert store.live_bytes() == 3*256*50
        assert store.stats["evictions"] == 2
    finally:
        store.close()


@pytest.mark.parametrize("rows,selected,scoped,expected", [
    (1, False, False, "exact"), (1, True, True, "exact"),
    (2, False, False, "exact"),
    (15, True, True, "exact"), (16, True, False, "exact"),
    (16, False, True, "exact"), (16, True, True, "prefill"),
    (1024, True, True, "prefill"), (1024, False, False, "exact"),
])
def test_dispatch_obeys_explicit_prefill_scope_and_preserves_route_slots(artifact, monkeypatch, rows, selected, scoped, expected):
    from contextlib import nullcontext
    from mlx2.runtime.gguf import v41_row_exact, v41_row_prefill
    art, packed = artifact
    weights = compact.CompactV41Weights(art, expert_cache_bytes=3*256*50,
        expert_prefill_policy="segmented_nax_fp16" if selected else "reference")
    x = mx.zeros((rows, 256), mx.float32)
    ids = (np.arange(rows*2, dtype=np.uint32).reshape(rows, 2)*2) % 3
    scores = mx.full(ids.shape, .5)
    calls = []
    def dispatch(kind, value, remapped, actual_scores, *planes, **kwargs):
        assert value is x and actual_scores is scores
        unique = np.unique(ids)
        np.testing.assert_array_equal(unique[remapped], ids)
        for projection, arrays in zip(compact.FIELDS, planes):
            for expert, item in zip(unique, arrays):
                assert np.array(item.data).tobytes() == packed[0][expert][projection].tobytes()
        if kind == "exact":
            assert {k: kwargs[k] for k in ("reuse", "broadcast", "group_experts", "simd_groups")} == {
                "reuse": 1 if rows == 1 else 4, "broadcast": False, "group_experts": 10, "simd_groups": 4}
        else:
            assert kwargs["group_experts"] == 12
        calls.append(kind)
        return mx.zeros(x.shape, mx.float32)
    monkeypatch.setattr(v41_row_exact, "moe", lambda *a, **kw: dispatch("exact", *a, **kw))
    monkeypatch.setattr(v41_row_prefill, "moe", lambda *a, **kw: dispatch("prefill", *a, **kw))
    try:
        context = expert_prefill_context("segmented_nax_fp16", tokens=rows) if scoped else nullcontext()
        with context:
            result = weights.moe(x, mx.array(ids), scores, "blk.0")
        assert result.shape == x.shape and calls == [expected]
        assert not weights.expert_stream._active and not weights.expert_stream._leased
    finally:
        weights.close()


def test_compact_dense_and_park_reload_remain_standalone(artifact):
    art, _ = artifact
    weights = compact.CompactV41Weights(art, expert_cache_bytes=3*256*50)
    try:
        expected = np.array(weights.embedding(mx.array([[6, 1]], mx.uint32)))
        weights.park()
        assert weights.resident_bytes() == weights.expert_stream.reserved_bytes() == 0
        weights.activate()
        np.testing.assert_array_equal(weights.embedding(mx.array([[6, 1]], mx.uint32)), expected)
        assert weights.verify_source()
        assert weights.dense_receipt()["numeric_profile"] == compact.PROFILE
    finally:
        weights.close()


def test_corrupted_compact_item_refuses_even_when_cached(artifact):
    art, _ = artifact
    store = compact.CompactExpertStore(art, 3*256*50)
    try:
        with store.window():
            store.acquire("blk.0", 0)
        path = art["root"]/store.tables["blk.0"][0]
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with pytest.raises(ValueError, match="changed after admission"):
            with store.window():
                store.acquire("blk.0", 0)
    finally:
        store.close()


def test_failed_compact_upload_retains_pending_owner_and_read_counters(artifact, monkeypatch):
    art, _ = artifact
    store = compact.CompactExpertStore(art, 3*256*50)
    def fail(*args):
        raise RuntimeError("compact upload/fence failed")
    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", fail)
        store._synchronize = fail
        with pytest.raises(RuntimeError, match="upload/fence"):
            with store.window():
                store.acquire("blk.0", 0)
        assert store._pending_host and store._pending_uploads and store._active
        assert store.stats["bytes_read"] > 3*256*50
        assert store.stats["uploaded_experts"] == 0
        assert id(store) in native._UPLOAD_QUARANTINE
    store._synchronize = lambda: None
    native.release_quarantined_native_uploads()
    assert store.live_bytes() == 0


def test_late_compact_read_blocks_park_until_owned_io_drains(artifact, monkeypatch):
    art, _ = artifact
    store = compact.CompactExpertStore(art, 3*256*50, timeout=.03)
    release, finished = threading.Event(), threading.Event()
    original = compact.read_native_file
    def delayed(*args):
        release.wait(2)
        try:
            return original(*args)
        finally:
            finished.set()
    monkeypatch.setattr(compact, "read_native_file", delayed)
    try:
        with pytest.raises(ReadTimeout):
            with store.window():
                store.acquire("blk.0", 0)
        assert store.pool.counters()["weight_read_inflight"] == 1
        with pytest.raises(ReadTimeout, match="prevent parking"):
            store.park()
    finally:
        release.set()
        assert finished.wait(2)
        store.close()


@pytest.mark.parametrize("change", ["profile", "missing_plane", "float_payload", "extent", "width"])
def test_compact_admission_rejects_unbound_format_or_geometry(artifact, change):
    art, _ = artifact
    art = {**art, "manifest": deepcopy(art["manifest"])}
    manifest = art["manifest"]
    path = "experts/layer-000/expert-000.safetensors"
    if change == "profile":
        manifest["numeric_profile"] = "native_f32_qmm"
    elif change == "missing_plane":
        manifest["files"][path]["tensors"].pop("up.data")
    elif change == "float_payload":
        manifest["files"][path]["tensors"]["gate.data"]["dtype"] = "F32"
    elif change == "extent":
        manifest["files"][path]["tensors"]["gate.data"]["shape"] = [1]
    else:
        manifest["tensors"]["blk.0.ffn_gate_exps.weight"]["numpy_shape"][-1] = 32
    with pytest.raises(ValueError):
        compact.compact_weight_plan(art, 3*256*50)
