"""Synthetic CPU-only fixtures; no checkpoint/device load. Base 9fc9e848d."""

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mlx2.experimental.ironmlx_sparse_kv import (
    SparseKVConfig,
    SparseKVStore,
    bf16_round,
    partitioned_attention,
)

ARTIFACT = hashlib.sha256(b"synthetic-sparse-kv-v1:no-model").hexdigest()
SOURCE = "9fc9e848d8028874773c9245c67d8f2ad6c4e278"


def config(**kwargs):
    return replace(
        SparseKVConfig(ARTIFACT, SOURCE, heads=1, head_dim=8, page_tokens=4), **kwargs
    )


def fixture(tokens=28, seed=9):
    rng = np.random.default_rng(seed)
    return bf16_round(rng.normal(size=(tokens, 1, 8)).astype(np.float32)), bf16_round(
        rng.normal(size=(tokens, 1, 8)).astype(np.float32)
    )


def dense_mask_reference(query, keys, values, positions, blocks, scale):
    """Independent dense-addressed oracle; mask before the reference passes."""
    keep = set(map(int, positions))
    result = []
    for head, q in enumerate(query):
        states = []
        for block in range(blocks):
            maximum, denominator = np.float32(-np.inf), np.float32(0)
            numerator = np.zeros(q.shape, dtype=np.float32)
            for token in range(block, len(keys), blocks):
                if token not in keep:
                    continue
                kvhead = head // (len(query) // keys.shape[1])
                dot = np.float32(0)
                for dim in range(len(q)):
                    dot += np.float32(
                        np.float32(q[dim] * np.float32(scale))
                        * keys[token, kvhead, dim]
                    )
                updated_maximum = np.maximum(maximum, dot)
                old_weight = np.float32(np.exp(np.float32(maximum - updated_maximum)))
                new_weight = np.float32(np.exp(np.float32(dot - updated_maximum)))
                numerator = numerator * old_weight + values[token, kvhead] * new_weight
                denominator = np.float32(denominator * old_weight + new_weight)
                maximum = updated_maximum
            states.append((maximum, denominator, bf16_round(numerator)))
        maximum = max(row[0] for row in states)
        numerator = np.zeros(q.shape, dtype=np.float32)
        denominator = np.float32(0)
        for part_maximum, part_sum, part in states:
            weight = np.float32(np.exp(np.float32(part_maximum - maximum)))
            numerator += weight * part
            denominator += weight * part_sum
        result.append(numerator / denominator)
    return bf16_round(np.asarray(result, dtype=np.float32))


def test_import_does_not_import_mlx_or_initialize_device():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import mlx2.experimental.ironmlx_sparse_kv; assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_logical_partition_rounding_survives_sparse_gather_and_compaction_does_not(
    tmp_path,
):
    keys, values = fixture()
    query = bf16_round(np.random.default_rng(11).normal(size=(2, 8)).astype(np.float32))
    with (
        SparseKVStore(config(), directory=tmp_path) as store,
        store.history() as history,
    ):
        history.append(keys, values)
        selected = history.gather([0, 2, 5])
        expected = dense_mask_reference(
            query, keys, values, selected.positions, blocks=8, scale=0.3
        )
        actual = partitioned_attention(query, selected, blocks=8, scale=0.3)
        np.testing.assert_array_equal(actual, expected)
        compact = partitioned_attention(
            query, selected, blocks=8, scale=0.3, preserve_logical_positions=False
        )
        assert not np.array_equal(compact, expected), (
            "fixture must expose changed BF16 partial grouping"
        )
        assert np.array_equal(
            selected.positions, [*range(4), *range(8, 12), *range(20, 24)]
        )
        assert store.receipt()["gpu_used"] is False
        assert store.receipt()["qualified"] is False


def test_fork_restore_and_arbitrary_rollback_preserve_immutable_prefix(tmp_path):
    keys, values = fixture(13)
    with SparseKVStore(config(), directory=tmp_path) as store:
        with store.history() as first:
            first.append(keys[:9], values[:9])
            with first.snapshot() as snapshot, store.fork(snapshot) as second:
                first.append(keys[9:], values[9:])
                np.testing.assert_array_equal(second.gather([0, 1, 2]).keys, keys[:9])
                second.trim(5)
                second.append(keys[9:12], values[9:12])
                np.testing.assert_array_equal(
                    second.gather([0, 1]).keys, np.concatenate((keys[:5], keys[9:12]))
                )
                first.restore(snapshot)
                np.testing.assert_array_equal(
                    first.gather([0, 1, 2]).values, values[:9]
                )
                second.trim(0)
                assert second.gather([]).keys.shape == (0, 1, 8)
            assert store.receipt()["live_pages"] == 3
        assert store.receipt()["live_pages"] == 0


def test_snapshot_leases_cleanup_and_foreign_identity_refusal(tmp_path):
    keys, values = fixture(9)
    store = SparseKVStore(config(), directory=tmp_path)
    other = SparseKVStore(config(artifact_sha256="a" * 64), directory=tmp_path)
    root = store.root
    first = store.history()
    first.append(keys, values)
    snapshot = first.snapshot()
    second = store.fork(snapshot)
    first.close()
    snapshot.close()
    with pytest.raises(ValueError, match="live histories"):
        store.close()
    with pytest.raises(ValueError, match="foreign or closed"):
        other.fork(snapshot)
    second.close()
    assert store.receipt()["disk_bytes"] == 0
    assert store.receipt()["read_cache_bytes"] == 0
    store.close()
    assert not root.exists()
    other.close()


def test_atomic_append_releases_new_records_on_quota_or_write_failure(
    tmp_path, monkeypatch
):
    keys, values = fixture(20)
    with (
        SparseKVStore(config(max_disk_bytes=800), directory=tmp_path) as store,
        store.history() as history,
    ):
        history.append(keys[:4], values[:4])
        before = store.receipt()["disk_bytes"]
        # Quota preflight rejects before publishing or converting new pages.
        with pytest.raises(ValueError, match="disk quota"):
            history.append(keys[4:20], values[4:20])
        assert history.length == 4
        assert store.receipt()["disk_bytes"] == before
        assert len(list(store.root.iterdir())) == 1
        np.testing.assert_array_equal(history.gather([0]).keys, keys[:4])
        write = store._write
        count = 0

        def fail_second(k, v):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("injected disk failure")
            return write(k, v)

        monkeypatch.setattr(store, "_write", fail_second)
        with pytest.raises(OSError, match="injected"):
            history.append(keys[4:12], values[4:12])
        assert history.length == 4
        assert store.receipt()["disk_bytes"] == before


def test_partial_tail_append_failure_does_not_publish_replacement(tmp_path):
    keys, values = fixture(9)
    with (
        SparseKVStore(config(max_disk_bytes=245), directory=tmp_path) as store,
        store.history() as history,
    ):
        history.append(keys[:3], values[:3])
        with pytest.raises(ValueError, match="disk quota"):
            history.append(keys[3:6], values[3:6])
        assert history.length == 3
        np.testing.assert_array_equal(history.gather([0]).keys, keys[:3])


def test_checksums_and_gather_limits_fail_before_publication(tmp_path):
    keys, values = fixture(8)
    with (
        SparseKVStore(
            config(read_cache_bytes=0, max_gather_bytes=1000), directory=tmp_path
        ) as store,
        store.history() as history,
    ):
        history.append(keys, values)
        before = store.read_bytes
        with pytest.raises(ValueError, match="RAM budget"):
            history.gather([0, 1])
        assert store.read_bytes == before
        record = store._records[history._pages[0]]
        content = bytearray(record.path.read_bytes())
        content[-1] ^= 1
        record.path.write_bytes(content)
        with pytest.raises(ValueError, match="checksum"):
            history.gather([0])
        assert history.length == 8


def test_cache_bound_and_repeated_gather_no_extra_disk_reads(tmp_path):
    keys, values = fixture(12)
    with (
        SparseKVStore(config(read_cache_bytes=200), directory=tmp_path) as store,
        store.history() as history,
    ):
        history.append(keys, values)
        history.gather([0])
        before = store.read_bytes
        history.gather([0])
        assert store.read_bytes == before
        for index in (1, 2, 0):
            history.gather([index])
            assert store.receipt()["read_cache_bytes"] <= 200


@pytest.mark.parametrize("selection", [[-1], [1, 0], [0, 0], [True], [5]])
def test_invalid_page_selection_refuses_without_io(tmp_path, selection):
    keys, values = fixture(8)
    with (
        SparseKVStore(config(), directory=tmp_path) as store,
        store.history() as history,
    ):
        history.append(keys, values)
        with pytest.raises(ValueError):
            history.gather(selection)
        assert store.read_bytes == 0


def test_token_cap_nonfinite_shapes_and_reference_guards(tmp_path):
    keys, values = fixture(8)
    with (
        SparseKVStore(config(max_tokens=8), directory=tmp_path) as store,
        store.history() as history,
    ):
        for k, v in [(keys[:, :, :4], values), (keys * np.nan, values)]:
            with pytest.raises(ValueError):
                history.append(k, v)
        history.append(keys, values)
        with pytest.raises(ValueError, match="token cap"):
            history.append(keys[:1], values[:1])
        selected = history.gather([0])
        with pytest.raises(ValueError, match="positions"):
            partitioned_attention(
                np.ones((1, 8), dtype=np.float32),
                replace(selected, positions=np.array([0, 1, 1, 3])),
                blocks=4,
            )
        with pytest.raises(ValueError, match="workspace"):
            partitioned_attention(
                np.ones((1, 8), dtype=np.float32), selected, blocks=5000
            )
        with pytest.raises(ValueError, match="GQA"):
            partitioned_attention(np.ones((1, 3), dtype=np.float32), selected, blocks=4)
        with pytest.raises(ValueError, match="workspace budget"):
            partitioned_attention(
                np.ones((1, 8), dtype=np.float32),
                selected,
                blocks=4,
                max_workspace_bytes=8,
            )
        with pytest.raises(ValueError, match="positions"):
            partitioned_attention(
                np.ones((1, 8), dtype=np.float32),
                replace(selected, positions=np.array([3, 2, 1, 0], dtype=np.uint64)),
                blocks=4,
            )


def test_bf16_round_uses_ties_to_even_and_fails_on_overflow():
    values = np.array([0x3F808000, 0x3F818000, 0xBF808000], dtype=np.uint32).view(
        np.float32
    )
    np.testing.assert_array_equal(
        bf16_round(values).view(np.uint32), [0x3F800000, 0x3F820000, 0xBF800000]
    )
    with pytest.raises(ValueError, match="overflow"):
        bf16_round(np.array([np.finfo(np.float32).max], dtype=np.float32))


def test_device_array_protocols_are_refused_without_conversion(tmp_path):
    class DeviceLike:
        def __array__(self, *_args, **_kwargs):
            raise AssertionError("device conversion must never execute")

    device = DeviceLike()
    keys, values = fixture(4)
    with (
        SparseKVStore(config(), directory=tmp_path) as store,
        store.history() as history,
    ):
        with pytest.raises(ValueError, match="NumPy float32"):
            history.append(device, values)
        with pytest.raises(ValueError, match="NumPy float32"):
            bf16_round(device)
        history.append(keys, values)
        selected = history.gather([0])
        with pytest.raises(ValueError, match="NumPy float32"):
            partitioned_attention(device, selected, blocks=4)
        with pytest.raises(ValueError, match="NumPy float32"):
            partitioned_attention(
                np.ones((1, 8), dtype=np.float32),
                replace(selected, keys=device),
                blocks=4,
            )
        with pytest.raises(ValueError, match="positions"):
            partitioned_attention(
                np.ones((1, 8), dtype=np.float32),
                replace(selected, positions=device),
                blocks=4,
            )


@pytest.mark.parametrize("operation", ["trim", "close", "restore"])
def test_unlink_failure_keeps_cleanup_debt_retryable(tmp_path, monkeypatch, operation):
    keys, values = fixture(8)
    store = SparseKVStore(config(), directory=tmp_path)
    history = store.history()
    history.append(keys[:4], values[:4])
    snapshot = history.snapshot() if operation == "restore" else None
    if snapshot is not None:
        history.append(keys[4:], values[4:])
    original = Path.unlink
    failures = []

    def fail_once(path, *args, **kwargs):
        if path.parent == store.root and not failures:
            failures.append(str(path))
            raise OSError("injected unlink failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    if operation == "trim":
        history.trim(0)
        assert history.length == 0
    elif operation == "close":
        history.close()
        assert history.closed
    else:
        history.restore(snapshot)
        assert history.length == 4
        np.testing.assert_array_equal(history.gather([0]).keys, keys[:4])
        snapshot.close()
    assert store.receipt()["pending_delete_pages"] == 1
    assert all(record.references >= 0 for record in store._records.values())
    assert store.collect() == 0
    history.close()
    assert store.receipt()["disk_bytes"] == 0
    store.close()


def test_persistent_unlink_failure_refuses_store_close_then_recovers(
    tmp_path, monkeypatch
):
    keys, values = fixture(4)
    store = SparseKVStore(config(), directory=tmp_path)
    history = store.history()
    history.append(keys, values)
    with monkeypatch.context() as patch:

        def fail_unlink(*_args, **_kwargs):
            raise OSError("disk unavailable")

        patch.setattr(Path, "unlink", fail_unlink)
        history.close()
        with pytest.raises(OSError, match="cleanup pending"):
            store.close()
        assert store.receipt()["disk_bytes"] > 0
        assert store.receipt()["live_pages"] == 0
    store.close()
    assert not store.root.exists()
