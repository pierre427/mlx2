"""glm53_kv_append=rotate_v1: the reference planes' values, bit for bit.

CPU only. Plane level (ring KVCache/BatchKVCache against the reference
classes through appends, catch-ups, growth, trims, ragged trims, restores and
metadata rewinds), the model's decode logits and committed state, APCv2
state/meta_state round trips and COW snapshots, and the serving merge
(generate._merge_caches -> BatchKVCache planes). Whether a pipelined append
still copies is a GPU property: scripts/research/glm53_kv_update_bench.py
--kv-append rotate_v1 --ring K.
"""

import json

import mlx.core as mx
import numpy as np
import pytest
from test_glm53_decode_levers_cpu import _adapter
from test_glm53_layer_tree_cpu import TreeWeights, live_cache, tree_model

from mlx2.runtime.models import glm53_kv_append as kv_store
from mlx2.runtime.models.cache import BatchKVCache, KVCache
from mlx2.runtime.models.glm53_gguf import (
    Glm53KDACache,
    Glm53MLACache,
    sparse_attention_phase,
)

GATHERED = {"indexer_pools": "incremental_v1", "indexer_select": "topk_v1",
            "sparse_attention": "gathered_v1", "sparse_prefill": "gathered_v1_all"}
MASKED = {"indexer_pools": "incremental_v1", "indexer_select": "topk_v1"}
REFERENCE = {}  # recompute pools, argsort, mask consumers


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(params=[2, 3])
def ring(request, monkeypatch):
    # Small blocks: many growths and catch-ups in few steps, same code paths.
    monkeypatch.setattr(kv_store, "RING_BUFFERS", request.param)
    monkeypatch.setattr(kv_store, "BASE_BLOCK", 32)
    return request.param


def bits(a):
    a = np.array(a)
    return (a.shape, str(a.dtype), a.tobytes())


def live_rows(plane, level):
    k, v = plane.state[:2]
    return bits(k[..., :level, :]), bits(v[..., :level, :])


def level_of(plane):
    return plane._idx if isinstance(plane, BatchKVCache) else plane.offset


def check_slots(ring_plane, ref):
    """Every slot's valid rows are the reference rows (the catch-up invariant)."""
    if ref.keys is None:
        return
    want_k, want_v = ref.keys_and_values()
    for k, v, valid in ring_plane.__dict__.get("_ring") or ():
        # Rows at or past the cursor are not committed in any law; a slot's
        # count is clamped to the cursor on its next append.
        valid = min(valid, level_of(ref))
        if valid:
            assert bits(k[..., :valid, :]) == bits(want_k[..., :valid, :])
            assert bits(v[..., :valid, :]) == bits(want_v[..., :valid, :])


def plane_ops(rng):
    ops = [("append", 37)] + [("append", 1)] * 40
    ops += [("trim", 3), ("append", 1), ("trim", 11), ("append", 5), ("append", 70)]
    ops += [("restore", 0), ("append", 1), ("append", 1), ("rewind", 4), ("append", 2)]
    for _ in range(40):
        ops.append(("append", int(rng.integers(1, 4))))
        if rng.random() < 0.15:
            ops.append(("trim", int(rng.integers(1, 9))))
    return ops + [("trim", 0), ("append", 2)]


def apply(op, arg, ref, ring_plane, rng, width):
    if op == "append":
        rows = mx.array(rng.normal(size=(1, 1, arg, width)).astype(np.float32))
        empty = mx.zeros((1, 1, arg, 0))
        want = ref.update_and_fetch(rows, empty)
        got = ring_plane.update_and_fetch(rows, empty)
        assert bits(got[0]) == bits(want[0]) and bits(got[1]) == bits(want[1])
    elif op == "trim":
        assert ring_plane.trim(arg) == ref.trim(arg)
    elif op == "restore":  # APCv2-style state set on the live object
        meta = ref.meta_state
        for plane in (ref, ring_plane):
            state = tuple(mx.array(np.array(a)) if isinstance(a, mx.array) else a for a in plane.state)
            plane.state = state
            plane.meta_state = meta
    elif op == "rewind":  # metadata-only rewind
        meta = (str(level_of(ref) - arg),)
        ref.meta_state = meta
        ring_plane.meta_state = meta
        if isinstance(ref, BatchKVCache):
            ref.offset -= arg
            ring_plane.offset -= arg


@pytest.mark.parametrize("width", [512, 256])
def test_ring_kvcache_equals_reference_kvcache(cpu, ring, width):
    rng = np.random.default_rng(width)
    ref, ring_plane = KVCache(), KVCache()
    kv_store.engage([ring_plane])
    assert type(ring_plane) is kv_store.Glm53RingKVCache
    for op, arg in plane_ops(rng):
        apply(op, arg, ref, ring_plane, rng, width)
        assert ring_plane.offset == ref.offset
        check_slots(ring_plane, ref)
    assert ring_plane.meta_state == ref.meta_state
    assert live_rows(ring_plane, ref.offset) == live_rows(ref, ref.offset)
    assert len({id(s[0]) for s in ring_plane._ring if s[0] is not None}) == ring
    assert ring_plane.nbytes >= ref.nbytes


def test_ring_batch_cache_equals_reference_batch_cache(cpu, ring):
    rng = np.random.default_rng(7)
    seed = KVCache()
    seed.update_and_fetch(mx.array(rng.normal(size=(1, 1, 50, 256)).astype(np.float32)),
                          mx.zeros((1, 1, 50, 0)))
    ref, ring_plane = BatchKVCache.merge([seed]), BatchKVCache.merge([seed])
    kv_store.engage([ring_plane])
    assert type(ring_plane) is kv_store.Glm53RingBatchKVCache
    for op, arg in plane_ops(rng):
        apply(op, arg, ref, ring_plane, rng, 256)
        assert ring_plane._idx == ref._idx and bits(ring_plane.offset) == bits(ref.offset)
        check_slots(ring_plane, ref)
    # Ragged trim (uniform at B1) through the declared empty aux ledger.
    assert ring_plane.trim_ragged([3]) == ref.trim_ragged([3])
    apply("append", 2, ref, ring_plane, rng, 256)
    assert ring_plane.meta_state == ref.meta_state
    got, want = ring_plane.extract(0), ref.extract(0)
    assert type(got) is KVCache and got.offset == want.offset
    assert live_rows(got, got.offset) == live_rows(want, want.offset)


def test_meta_state_names_reference_classes_and_restores_plain(cpu, ring):
    rng = np.random.default_rng(3)
    mla, ref = Glm53MLACache(), Glm53MLACache()
    kv_store.engage(mla.caches)
    for n in (20, 1, 1, 1, 5, 1):
        for a, b, width in zip(mla.caches, ref.caches, (512, 256)):
            rows = mx.array(rng.normal(size=(1, 1, n, width)).astype(np.float32))
            a.update_and_fetch(rows, mx.zeros((1, 1, n, 0)))
            b.update_and_fetch(rows, mx.zeros((1, 1, n, 0)))
    assert json.dumps(mla.meta_state) == json.dumps(ref.meta_state)
    restored = Glm53MLACache.from_state(mla.state, mla.meta_state)
    assert [type(c) for c in restored.caches] == [KVCache, KVCache]
    for a, b in zip(restored.caches, ref.caches):
        assert a.offset == b.offset and live_rows(a, a.offset) == live_rows(b, b.offset)
    for a, b in zip(mla.extract(0).caches, ref.extract(0).caches):
        assert a.offset == b.offset and live_rows(a, a.offset) == live_rows(b, b.offset)


# ---- model -----------------------------------------------------------------------


def mla_bits(cache):
    out = []
    for c in cache:
        if isinstance(c, Glm53KDACache):
            out.append(("kda", c.offset, [bits(a) for a in c.state]))
            continue
        pools = c.pools
        out.append((
            "mla", json.dumps(c.meta_state), c.offset,
            [live_rows(kv, kv.size()) for kv in c.caches],
            None if pools is None or pools.pooled is None
            else (pools.complete, bits(pools.pooled[:, : pools.complete])),
        ))
    return out


def decode(model, cache, tokens):
    out = []
    for token in tokens:
        logits = model(mx.array([[token]]), cache=cache)
        mx.eval(logits)
        out.append(bits(logits))
    return out


def twins(policy, tokens):
    weights = TreeWeights()
    ref = tree_model(weights, **policy)
    rot = tree_model(weights, kv_append="rotate_v1", **policy)
    return ref, rot, live_cache(ref, tokens), live_cache(rot, tokens)


STEPS = [(i * 7 + 3) % 16 for i in range(48)]


def mla_planes(cache):
    return [kv for c in cache if isinstance(c, Glm53MLACache) for kv in c.caches]


@pytest.mark.parametrize("policy", [GATHERED, MASKED, REFERENCE], ids=["gathered", "masked", "reference"])
def test_decode_logits_and_state_equal_the_reference(cpu, ring, policy):
    # 2040 committed rows: selection starts at 2049, inside the run.
    ref, rot, ref_cache, rot_cache = twins(policy, 2040)
    assert decode(rot, rot_cache, STEPS) == decode(ref, ref_cache, STEPS)
    assert mla_bits(rot_cache) == mla_bits(ref_cache)
    assert rot.calls["kv_append_ring"] == 2 * len(STEPS) and ref.calls["kv_append_ring"] == 0
    assert all(type(kv) is kv_store.Glm53RingKVCache for kv in mla_planes(rot_cache))
    assert all(type(kv) is KVCache for kv in mla_planes(ref_cache))
    with sparse_attention_phase("prompt"):
        got = rot(mx.array([[1, 5, 9]]), cache=rot_cache)
        want = ref(mx.array([[1, 5, 9]]), cache=ref_cache)
    assert bits(got) == bits(want)
    # APCv2 round trip: restore from state/meta_state, keep decoding.
    restored = [type(c).from_state(c.state, c.meta_state) for c in rot_cache]
    ref_restored = [type(c).from_state(c.state, c.meta_state) for c in ref_cache]
    assert mla_bits(restored) == mla_bits(ref_restored)
    assert decode(rot, restored, STEPS[:12]) == decode(ref, ref_restored, STEPS[:12])
    assert mla_bits(restored) == mla_bits(ref_restored)


def test_decode_crosses_real_capacity_and_256_boundaries(cpu):
    ref, rot, ref_cache, rot_cache = twins(GATHERED, 2040)
    steps = [(i * 5 + 1) % 16 for i in range(272)]  # 2040 -> 2312: crosses 2048 and 2304
    before = dict(kv_store._SCHEDULED)
    assert decode(rot, rot_cache, steps) == decode(ref, ref_cache, steps)
    assert mla_bits(rot_cache) == mla_bits(ref_cache)
    writes = kv_store._SCHEDULED["ring_writes"] - before["ring_writes"]
    grows = kv_store._SCHEDULED["ring_grows"] - before["ring_grows"]
    # 2 layers x 2 planes. A slot allocates on first use and when the
    # context crosses its 2048-row capacity block; every other append writes
    # an existing buffer in place.
    assert writes + grows == 4 * len(steps)
    assert grows <= 4 * (kv_store.RING_BUFFERS + 1)


def test_cow_snapshot_is_not_written_by_later_ring_appends(cpu, ring):
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors

    _, rot, _, rot_cache = twins(GATHERED, 2040)
    decode(rot, rot_cache, STEPS[:5])
    snapshot = snapshot_prompt_cache_descriptors(rot_cache, require_committed=False)[0]
    frozen = mla_bits(snapshot)
    decode(rot, rot_cache, STEPS[5:20])
    assert mla_bits(snapshot) == frozen


def test_serving_merge_runs_ring_batch_planes_bitwise(cpu, ring):
    from mlx2.runtime.generate import _merge_caches

    ref, rot, ref_cache, rot_cache = twins(GATHERED, 2040)
    ref_batch, rot_batch = _merge_caches([ref_cache]), _merge_caches([rot_cache])
    planes = mla_planes(rot_batch)
    assert planes and all(type(kv) is BatchKVCache for kv in planes)
    assert decode(rot, rot_batch, STEPS) == decode(ref, ref_batch, STEPS)
    assert all(type(kv) is kv_store.Glm53RingBatchKVCache for kv in planes)
    assert rot.calls["kv_append_ring"] == 2 * len(STEPS)
    for got, want in zip(rot_batch, ref_batch):
        if isinstance(got, Glm53MLACache):
            a, b = got.extract(0), want.extract(0)
            assert json.dumps(a.meta_state) == json.dumps(b.meta_state)
            assert mla_bits([a]) == mla_bits([b])
    assert mla_bits(rot_batch) == mla_bits(ref_batch)


def test_policy_is_validated_and_receipted(cpu):
    with pytest.raises(ValueError, match="KV append policy"):
        tree_model(TreeWeights(), kv_append="tail_v1")
    default, rot = _adapter({}), _adapter({"glm53_kv_append": "rotate_v1"})
    assert default.kernel_policy["glm53_kv_append"] == "reference"
    assert "kv_append" not in default.profile
    assert rot.kernel_policy["glm53_kv_append"] == "rotate_v1"
    assert ":kv_append=rotate_v1" in rot.profile
    assert rot.profile.replace(":kv_append=rotate_v1", "") == default.profile
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    with pytest.raises(ValueError, match="unsupported GLM-5.3 packed execution policy"):
        object.__new__(Glm53GGUFAdapter)._load("/nonexistent", {"glm53_kv_append": "tail_v1"})


def test_budget_books_the_ring_copies_and_block():
    from mlx2.adapters.glm53_gguf_memory import Glm53CacheBudget

    config = {"layer_types": ["linear_attention"] * 34 + ["deepseek_sparse_attention"] * 11}
    ref = Glm53CacheBudget.from_config(config, kernel_policy={})
    rot = Glm53CacheBudget.from_config(config, kernel_policy={"glm53_kv_append": "rotate_v1"})
    assert (ref.capacity_block, ref.indexed_plane_copies) == (256, 1)
    assert (rot.capacity_block, rot.indexed_plane_copies) == (kv_store.BASE_BLOCK, kv_store.RING_BUFFERS)
    assert ref.snapshot_bytes(32768) == ref.fixed_bytes + 11 * (32768 + 256) * 768 * 4
    assert rot.snapshot_bytes(32768) == (
        rot.fixed_bytes + 11 * (32768 + 2048) * 768 * kv_store.RING_BUFFERS * 4)
    with pytest.raises(ValueError, match="KV append"):
        Glm53CacheBudget.from_config(config, kernel_policy={"glm53_kv_append": "x"})


def test_engaging_a_cow_branch_rebinds_its_update_hook_to_the_ring(cpu):
    from mlx2.runtime.cow_cache import COWCacheTelemetry, _clone_graph

    rng = np.random.default_rng(11)
    source = KVCache()
    source.update_and_fetch(mx.array(rng.normal(size=(1, 1, 9, 256)).astype(np.float32)),
                            mx.zeros((1, 1, 9, 0)))
    branch = _clone_graph(source, telemetry=COWCacheTelemetry(), plane="attention_kv",
                          attach_tokens=True)
    assert "update_and_fetch" in branch.__dict__  # the COW instance hook
    kv_store.engage([branch])
    ref = KVCache.from_state(source.state, source.meta_state)
    rows = mx.array(rng.normal(size=(1, 1, 1, 256)).astype(np.float32))
    got = branch.update_and_fetch(rows, mx.zeros((1, 1, 1, 0)))
    want = ref.update_and_fetch(rows, mx.zeros((1, 1, 1, 0)))
    assert "_ring" in branch.__dict__  # the ring append ran, not the reference one
    assert bits(got[0]) == bits(want[0])
    token = branch._cow_segment_token
    assert token._materialized and token.reason == "Glm53RingKVCache.update_and_fetch"
    assert live_rows(source, 9) == live_rows(ref, 9)  # the source is untouched


def test_budget_planning_does_not_import_mlx():
    import subprocess
    import sys

    code = (
        "import sys; from mlx2.adapters.glm53_gguf_memory import Glm53CacheBudget as B;"
        "c={'layer_types':['linear_attention']*34+['deepseek_sparse_attention']*11};"
        "B.from_config(c, maximum=True); B.from_config(c, kernel_policy={'glm53_kv_append':'rotate_v1'});"
        "assert 'mlx.core' not in sys.modules, 'mlx imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
