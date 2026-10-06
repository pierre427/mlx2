"""CPU-only falsifiers for restored-cache joins and completion snapshots."""

from dataclasses import dataclass

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.cow_cache import freeze_prompt_cache
from mlx2.runtime.generate import PromptProcessingBatch, _merge_caches
from mlx2.runtime.models.cache import BatchKVCache, KVCache


def _kv(tokens: int, seed: int = 0) -> KVCache:
    cache = KVCache()
    values = mx.arange(
        seed, seed + tokens, dtype=mx.float32
    ).reshape(1, 1, tokens, 1)
    cache.update_and_fetch(values, values)
    mx.eval(cache.state)
    return cache


def test_restored_rows_release_apcv2_leases_at_physical_batch_handoff():
    """A long restored prefix must not stay pinned after its rows are merged."""
    sources = []
    branches = []
    logical_bytes = 0
    for row in range(2):
        source, _ = freeze_prompt_cache(
            [_kv(64, seed=row * 100)],
            key=f"row-{row}",
            tokens=range(64),
            cache_type="kv",
        )
        branch = source.branch()
        sources.append(source)
        branches.append(branch)
        logical_bytes += branch[0].nbytes
        assert source.cow_owner.pin_count == 1
        assert branch.cow_prep_telemetry["avoided_copy_bytes"] == branch[0].nbytes

    batch = PromptProcessingBatch(
        model=object(),
        uids=[1, 2],
        caches=branches,
        tokens=[[], []],
    )

    assert len(batch.prompt_cache) == 1
    assert isinstance(batch.prompt_cache[0], BatchKVCache)
    # The physical batch trims the source caches' spare allocation capacity;
    # it must never retain more logical K/V than the restored descriptors.
    assert batch.prompt_cache[0].nbytes == 2 * 64 * 2 * mx.float32.size
    assert batch.prompt_cache[0].nbytes <= logical_bytes
    assert all(source.cow_owner.pin_count == 0 for source in sources)
    assert all(branch.closed for branch in branches)


def test_restored_join_merges_one_layer_at_a_time_and_closes_after_success():
    events = []

    @dataclass
    class FakeLayer:
        row: int
        layer: int
        closed: bool = False

        @classmethod
        def merge(cls, rows):
            events.append(("merge", tuple((row.row, row.layer) for row in rows)))
            assert len({row.layer for row in rows}) == 1
            return cls(-1, rows[0].layer)

    class FakeBranch(list):
        cow_owner = object()
        cow_generation = 0

        def __init__(self, row):
            super().__init__([FakeLayer(row, layer) for layer in range(4)])
            self.row = row
            self.closed = False

        def close(self):
            self.closed = True
            events.append(("close", self.row))

    rows = [FakeBranch(0), FakeBranch(1)]
    merged = _merge_caches(rows)

    assert [layer.layer for layer in merged] == [0, 1, 2, 3]
    assert events[:4] == [
        ("merge", ((0, 0), (1, 0))),
        ("merge", ((0, 1), (1, 1))),
        ("merge", ((0, 2), (1, 2))),
        ("merge", ((0, 3), (1, 3))),
    ]
    assert events[4:] == [("close", 0), ("close", 1)]


def test_failed_restored_join_keeps_source_lease_for_caller_cleanup():
    class RefusingLayer:
        @classmethod
        def merge(cls, rows):
            raise RuntimeError("injected join failure")

    source, _ = freeze_prompt_cache(
        [_kv(8)], key="failed-row", tokens=range(8), cache_type="kv"
    )
    branch = source.branch()
    branch[0].merge = RefusingLayer.merge

    with pytest.raises(RuntimeError, match="injected join failure"):
        _merge_caches([branch])

    assert not branch.closed
    assert source.cow_owner.pin_count == 1
    branch.close()
    assert source.cow_owner.pin_count == 0


def test_merge_never_invalidates_apc_resident_frozen_source():
    source, _ = freeze_prompt_cache(
        [_kv(8)], key="resident-source", tokens=range(8), cache_type="kv"
    )
    generation = source.cow_owner.generation

    merged = _merge_caches([source])

    assert isinstance(merged[0], BatchKVCache)
    assert source.cow_owner.generation == generation
    branch = source.branch()
    assert branch.cow_generation == generation
    branch.close()
    source.close()


def test_completion_sidecar_contains_only_draft_state_not_target_kv():
    target = _kv(64, seed=0)
    draft = _kv(63, seed=100)
    tail = mx.ones((1, 1, 8), dtype=mx.float32)
    mx.eval(tail)
    sidecar = MTPAPCSidecar(([draft], tail), covered_tokens=64)
    expected_sidecar_bytes = draft.nbytes + tail.nbytes

    apc = APCv2(max_size=1, layout_name="completion-sidecar-falsifier-v1")
    key = APCKey("completion-sidecar")
    apc.store(key, list(range(64)), [target], sidecar=sidecar)
    hit = apc.lookup(key, list(range(65)))

    assert hit.hit
    assert hit.sidecar is not None
    assert hit.sidecar.covered_tokens == 64
    assert hit.sidecar.nbytes == expected_sidecar_bytes
    assert hit.sidecar.nbytes < hit.cache[0].nbytes + expected_sidecar_bytes
    assert len(hit.sidecar.state[0]) == 1
    assert hit.sidecar.state[0][0].offset == 63
    assert hit.cache[0].offset == 64
    hit.cache.close()
    apc.clear(release_memory=False)
