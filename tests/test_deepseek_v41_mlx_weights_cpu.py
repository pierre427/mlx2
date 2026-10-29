"""Synthetic standalone native files, numerical laws and bounded ownership."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx2.adapters.deepseek_v41_mlx_weights import (
    MANIFEST, NativeExpertStore, NativeV41Weights, _stamp, native_weight_plan, read_native_file,
)
from mlx2.runtime.io_ownership import ReadTimeout
from mlx2.runtime.models.deepseek_v4 import limited_swiglu

mx.set_default_device(mx.cpu)


def matrix(rows, width, bits, rng):
    codes = rng.integers(0, 3 if bits == 2 else 256, (rows, width), np.uint32)
    group = 32//bits
    packed = np.zeros((rows, width//group), np.uint32)
    for i in range(group):
        packed |= codes[:, i::group] << (i*bits)
    scales = np.full((rows, width//32), .03125 if bits == 2 else .0078125, np.float32)
    biases = np.full_like(scales, -.03125 if bits == 2 else -1.)
    return packed, scales, biases


def write_native_artifact(root, dense, experts, args):
    """Reusable CPU fixture writer; no GGUF file is created or read."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    records, tensors = {}, {}
    def write(relative, arrays):
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(arrays, str(path))
        raw = path.read_bytes()
        base = 8+int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8:base])
        header.pop("__metadata__", None)
        records[relative] = dict(size=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                 header_bytes=base, tensors=header, stat=_stamp(path.stat()))
    arrays = {}
    for name, value in dense.items():
        if isinstance(value, tuple):
            q, s, b = value
            shape = [q.shape[0], q.shape[1]*4]
            keys = {f: name+"."+f for f in ("qweight", "scales", "biases")}
            arrays.update({keys[f]: v for f, v in zip(keys, value)})
            storage = dict(kind="affine", file="weights.safetensors", keys=keys,
                           bits=8, group_size=32, mode="affine")
            kind, size = 8, int(np.prod(shape))//32*34
        else:
            shape = list(value.shape)
            arrays[name] = value
            storage = dict(kind="raw", file="weights.safetensors", key=name)
            kind, size = (1 if value.dtype == np.float16 else 0), value.nbytes
        tensors[name] = dict(shape=shape[::-1], numpy_shape=shape, ggml_type=kind,
                             offset=0, nbytes=size, storage=storage)
    write("weights.safetensors", arrays)
    for layer, entries in experts.items():
        paths = [f"experts/layer-{layer:03d}/expert-{i:03d}.safetensors" for i in range(len(entries))]
        for path, entry in zip(paths, entries):
            write(path, {kind+"."+field: value for kind in ("gate", "up", "down")
                         for field, value in zip(("qweight", "scales", "biases"), entry[kind])})
        for kind in ("gate", "up", "down"):
            rows, width = ((args.hidden_size, args.moe_intermediate_size) if kind == "down"
                           else (args.moe_intermediate_size, args.hidden_size))
            name = f"blk.{layer}.ffn_{kind}_exps.weight"
            tensors[name] = dict(shape=[width, rows, len(entries)], numpy_shape=[len(entries), rows, width],
                ggml_type=19, offset=0, nbytes=len(entries)*rows*width//256*50,
                storage=dict(kind="expert-affine", expert_files=paths, bits=2, group_size=32, mode="affine",
                             keys={f: kind+"."+f for f in ("qweight", "scales", "biases")}))
    manifest = dict(schema="mlx2.deepseek-v41-native.v1", numeric_profile="native_f32_qmm",
                    finished=True, tensors=tensors, files=records,
                    source={"path": "/nonexistent/original.gguf", "fingerprint": "synthetic-only"})
    (root/MANIFEST).write_text(json.dumps(manifest))
    return dict(root=root, manifest=manifest, args=args, metadata={}, config={},
                tensors=tensors, files=records, identity={"fingerprint": "synthetic-native"})


@pytest.fixture
def artifact(tmp_path):
    rng = np.random.default_rng(4144)
    args = SimpleNamespace(hidden_size=32, moe_intermediate_size=32, num_experts_per_tok=2)
    dense = {"token_embd.weight": matrix(7, 32, 8, rng),
             "output.weight": matrix(7, 32, 8, rng),
             "output_norm.weight": np.ones(32, np.float32),
             "blk.0.example.weight": matrix(32, 32, 8, rng),
             "blk.0.grouped.weight": matrix(64, 32, 8, rng),
             "blk.0.raw.weight": rng.normal(size=(16, 32)).astype(np.float16)}
    experts = {layer: [{kind: matrix(32, 32, 2, rng) for kind in ("gate", "up", "down")}
                       for _ in range(4)] for layer in range(2)}
    return write_native_artifact(tmp_path/"native", dense, experts, args), dense, experts


def test_admission_is_cpu_only_and_reserves_full_live_layer_and_staging(artifact):
    art, _, _ = artifact
    plan = native_weight_plan(art, 8192)
    assert plan["minimum_expert_cache_bytes"] == 1536
    assert plan["expert_active_bound_bytes"] == 4*1536
    assert plan["expert_table_bytes"] == 8*1536
    assert plan["total_reservation_bytes"] == sum(plan[k] for k in (
        "dense_payload_bytes", "dense_prepared_bound_bytes", "dense_staging_bound_bytes",
        "expert_cache_bytes", "expert_active_bound_bytes", "expert_staging_bound_bytes"))
    with pytest.raises(ValueError, match="one complete expert"):
        native_weight_plan(art, 1535)


def test_native_dense_embedding_grouped_and_raw_match_direct_mlx(artifact):
    art, dense, _ = artifact
    weights = NativeV41Weights(art, expert_cache_bytes=4096)
    try:
        x = mx.arange(64).reshape(2, 32).astype(mx.float32)/64
        params = tuple(mx.array(v) for v in dense["blk.0.example.weight"])
        expected = mx.quantized_matmul(x, *params, group_size=32, bits=8)
        np.testing.assert_array_equal(weights.linear("blk.0.example.weight")(x), expected)
        raw = weights.linear("blk.0.raw.weight")(x)
        np.testing.assert_array_equal(raw, x @ mx.array(dense["blk.0.raw.weight"], mx.float32).T)
        ids = mx.array([[6, 1, 6], [0, 4, 2]], mx.uint32)
        table = mx.dequantize(*(mx.array(v) for v in dense["token_embd.weight"]), group_size=32, bits=8)
        np.testing.assert_array_equal(weights.embedding(ids), table[ids])
        grouped = mx.arange(2*3*2*32).reshape(2, 3, 2, 32).astype(mx.float32)/256
        full = mx.dequantize(*(mx.array(v) for v in dense["blk.0.grouped.weight"]), group_size=32, bits=8)
        expected = mx.einsum("btgi,goi->btgo", grouped, full.reshape(2, 32, 32))
        np.testing.assert_allclose(weights.grouped(grouped, "blk.0.grouped.weight", groups=2), expected, rtol=1e-6, atol=1e-6)
        assert weights.verify_source() and not weights.can_fuse_dense(x, "blk.0.example.weight")
        assert weights.dense_receipt()["source_gguf_required"] is False
    finally:
        weights.close()


def test_weights_and_experts_share_one_owned_manifest_with_bound_metadata(artifact):
    art, _, _ = artifact
    weights = NativeV41Weights(art, expert_cache_bytes=1536)
    try:
        assert weights.artifact["manifest"] is weights.expert_stream.manifest
        assert weights.artifact["manifest"] is not art["manifest"]
        path = weights.expert_stream.tables["blk.0"][0]
        art["files"][path]["sha256"] = "0"*64
        with weights.expert_stream.window():
            weights.expert_stream.acquire("blk.0", 0)
        assert weights.bounds["metadata_bound_bytes"] >= 16*weights.bounds["manifest_canonical_bytes"]
        counts = weights.expert_stream.counters()
        assert counts["v41_expert_bytes_read"] > 1536
        assert counts["v41_expert_upload_batches"] == 1
        assert counts["v41_expert_demand_wait_ns"] > 0
        assert weights.expert_stream.receipt()["source_read_byte_scope"].endswith("including headers")
        assert weights.expert_stream.apc_revision() in weights.compute_policy
    finally:
        weights.close()


def test_native_moe_keeps_route_slots_scores_rounding_and_only_concats_activations(artifact, monkeypatch):
    art, _, experts = artifact
    weights = NativeV41Weights(art, expert_cache_bytes=1536)
    ids = mx.array([[3, 1], [0, 3], [1, 0]], mx.uint32)
    scores = mx.array([[.75, .25], [.5, .5], [.125, .875]], mx.float32)
    x = mx.arange(96).reshape(3, 32).astype(mx.float32)/128
    rounded = lambda value: value.astype(mx.bfloat16).astype(mx.float32)
    expected = []
    for token, routes in enumerate(np.array(ids)):
        output = mx.zeros((1, 32), mx.float32)
        for slot, expert in enumerate(routes):
            entry = experts[0][int(expert)]
            def project(value, kind):
                return mx.quantized_matmul(value, *(mx.array(v) for v in entry[kind]), group_size=32, bits=2)
            gate, up = rounded(project(x[token:token+1], "gate")), rounded(project(x[token:token+1], "up"))
            middle = rounded(limited_swiglu(gate, up)*scores[token, slot])
            output = output+rounded(project(middle, "down"))
        expected.append(output)
    expected = mx.concatenate(expected, 0)
    mx.eval(expected)
    original = mx.concatenate
    calls = []
    def concatenate(values, axis=0, **kwargs):
        assert all(v.dtype == mx.float32 and v.shape[-1] == 32 for v in values)
        calls.append(len(values))
        return original(values, axis=axis, **kwargs)
    monkeypatch.setattr(mx, "concatenate", concatenate)
    try:
        actual = weights.moe(x, ids, scores, "blk.0")
        np.testing.assert_array_equal(actual, expected)
        assert calls == [3]
        assert weights.expert_stream.live_bytes() == 1536
        assert weights.expert_stream.stats["evictions"] == 2
        assert not weights.expert_stream._active and not weights.expert_stream._leased
    finally:
        weights.close()


def test_global_lru_leases_survive_cross_layer_eviction_and_release_at_fence(artifact):
    art, _, _ = artifact
    store = NativeExpertStore(art, 1536)
    try:
        with store.window():
            first = store.acquire("blk.0", 0)
            second = store.acquire("blk.1", 1)
            assert first is not second
            assert store._cache_bytes == 1536 and store.live_bytes() == 3072
            assert len(store._leased) == 2
            with pytest.raises(ValueError, match="idle"):
                store.resize_cache_budget(1536)
        assert store.live_bytes() == 1536 and not store._leased
        assert store.receipt()["allocation"] == "global_byte_lru"
        assert store.plan.as_dict()["resident_ceiling_bytes"] == 1536
        with store.window():
            store.acquire("blk.1", 1)
        assert store.stats["cache_hits"] == 1
    finally:
        store.close()


def test_source_mutation_rejected_on_resident_hit(artifact):
    art, _, _ = artifact
    store = NativeExpertStore(art, 1536)
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


def test_hash_mismatch_fails_even_when_admitted_stat_matches(artifact):
    art, _, _ = artifact
    path = "experts/layer-000/expert-000.safetensors"
    record = deepcopy(art["files"][path])
    record["sha256"] = "0"*64
    with pytest.raises(ValueError, match="hash"):
        read_native_file(art["root"], path, record)


def test_failed_fence_retains_leases_until_successful_close(artifact):
    art, _, _ = artifact
    store = NativeExpertStore(art, 1536)
    def fail():
        raise RuntimeError("synthetic completion failure")
    store._synchronize = fail
    with pytest.raises(RuntimeError, match="completion failure"):
        with store.window():
            store.acquire("blk.0", 0)
            store.acquire("blk.0", 1)
    assert store._active and len(store._leased) == 2 and store.live_bytes() == 3072
    with pytest.raises(ReadTimeout, match="prevent parking"):
        store.park()
    store._synchronize = lambda: None
    store.close()
    assert store.live_bytes() == 0


def test_failed_expert_upload_and_fence_keep_pending_arrays_and_host_owner(artifact, monkeypatch):
    from mlx2.adapters import deepseek_v41_mlx_weights as module
    art, _, _ = artifact
    store = NativeExpertStore(art, 1536)
    def fail(*args):
        raise RuntimeError("synthetic upload completion failure")
    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", fail)
        store._synchronize = fail
        with pytest.raises(RuntimeError, match="completion failure"):
            with store.window():
                store.acquire("blk.0", 0)
        assert store.live_bytes() == 1536
        assert store._pending_host and store._pending_uploads and store._active
        assert id(store) in module._UPLOAD_QUARANTINE
        with pytest.raises(RuntimeError, match="failed-completion owners"):
            NativeExpertStore(art, 1536)
    store._synchronize = lambda: None
    assert module.release_quarantined_native_uploads() == 1
    assert store.live_bytes() == 0 and not store._pending_host


def test_failed_dense_upload_stays_owned_after_constructor_until_fence(artifact, monkeypatch):
    from mlx2.adapters import deepseek_v41_mlx_weights as module
    art, _, _ = artifact
    weights = NativeV41Weights(art, expert_cache_bytes=1536)
    assert weights.resident_bytes() == 0  # Construction itself has no device work.
    def fail(*args):
        raise RuntimeError("synthetic dense upload failure")
    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", fail)
        patch.setattr(mx, "synchronize", fail)
        with pytest.raises(RuntimeError, match="dense upload failure"):
            weights.array("output_norm.weight")
        assert weights.resident_bytes() > 0 and weights._pending_dense_host
        assert weights.reserved_bytes() > 0
        assert id(weights) in module._UPLOAD_QUARANTINE
    assert module.release_quarantined_native_uploads() == 1
    assert weights.resident_bytes() == 0 and weights._pending_dense_host is None


def test_native_park_reload_uses_only_derived_files(artifact):
    art, _, _ = artifact
    weights = NativeV41Weights(art, expert_cache_bytes=1536)
    try:
        initial = np.array(weights.embedding(mx.array([[0]], mx.uint32)))
        weights.park()
        assert weights.resident_bytes() == 0 and weights.reserved_bytes() == 0
        assert weights.expert_stream.reserved_bytes() == 0
        with pytest.raises(ValueError, match="parked"):
            weights.array("output_norm.weight")
        weights.activate()
        np.testing.assert_array_equal(weights.embedding(mx.array([[0]], mx.uint32)), initial)
    finally:
        weights.close()


def test_timed_out_read_retains_ownership_and_refuses_park_until_drained(artifact, monkeypatch):
    from mlx2.adapters import deepseek_v41_mlx_weights as module
    art, _, _ = artifact
    store = NativeExpertStore(art, 1536, timeout=.03)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = module.read_native_file
    def delayed(*args):
        started.set()
        release.wait(2)
        try:
            return original(*args)
        finally:
            finished.set()
    monkeypatch.setattr(module, "read_native_file", delayed)
    try:
        with pytest.raises(ReadTimeout):
            with store.window():
                store.acquire("blk.0", 0)
        assert started.is_set() and store.pool.counters()["weight_read_inflight"] == 1
        assert store.reserved_bytes() > 0
        with pytest.raises(ReadTimeout, match="prevent parking"):
            store.park()
    finally:
        release.set()
        assert finished.wait(2)
        store.close()
