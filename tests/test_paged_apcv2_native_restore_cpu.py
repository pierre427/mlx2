"""Host-only exact APCv2 to native Qwen3 geometry checks."""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.paged_apcv2_native_restore import (
    exact_qwen3_apc_planes, restore_exact_qwen3_apc_prefix,
    retire_failed_native_restore,
)


def _leaf(tokens=3, heads=2, dim=128, dtype=np.float16):
    leaf = KVCache()
    shape = (1, heads, tokens, dim)
    leaf.keys = mx.array(np.zeros(shape, dtype=dtype))
    leaf.values = mx.array(np.ones(shape, dtype=dtype))
    leaf.offset = tokens
    return leaf


def test_exact_apcv2_kv_validates_every_layer_before_native_write():
    leaves = [_leaf(), _leaf()]
    planes = exact_qwen3_apc_planes(
        leaves, layers=2, tokens=3, kv_heads=2, head_dim=128)
    assert len(planes) == 2
    with pytest.raises(ValueError, match="shape or dtype"):
        exact_qwen3_apc_planes(
            [leaves[0], _leaf(dtype=np.float32)], layers=2,
            tokens=3, kv_heads=2, head_dim=128)
    leaves[1].offset = 2
    with pytest.raises(ValueError, match="exact full-attention"):
        exact_qwen3_apc_planes(
            leaves, layers=2, tokens=3, kv_heads=2, head_dim=128)


def test_apcv2_native_restore_transposes_one_terminal_proved_layer_at_a_time():
    planes = exact_qwen3_apc_planes(
        [_leaf(), _leaf()], layers=2, tokens=3, kv_heads=2, head_dim=128)
    owners = [type("Owner", (), {"offset": 0})() for _ in range(2)]
    calls = []

    class TerminalBackend:
        def append_completed(self, targets, keys, values, counts):
            assert keys.shape == values.shape == (3, 2, 128)
            assert counts == (3,)
            targets[0].offset = 3
            calls.append(targets[0])

    restore_exact_qwen3_apc_prefix(
        planes, tuple(owners), TerminalBackend(), tokens=3)
    assert calls == owners


def test_apcv2_native_restore_refuses_missing_terminal_publication():
    planes = exact_qwen3_apc_planes(
        [_leaf()], layers=1, tokens=3, kv_heads=2, head_dim=128)
    owner = type("Owner", (), {"offset": 0})()

    class EarlyBackend:
        def append_completed(self, *_args):
            pass

    with pytest.raises(RuntimeError, match="before KV publication"):
        restore_exact_qwen3_apc_prefix(
            planes, (owner,), EarlyBackend(), tokens=3)


def test_failed_restore_keeps_ambiguous_owner_process_reachable():
    from mlx2.runtime import paged_apcv2_native_restore as bridge

    class Ambiguous:
        fully_retired = False

        def close(self):
            raise RuntimeError("native command still pending")

        def reap_retired(self):
            pass

        def reap_quarantine(self):
            pass

    owner = Ambiguous()
    with pytest.raises(RuntimeError, match="still pending"):
        retire_failed_native_restore(owner, object())
    assert any(item[0] is owner for item in bridge._RESTORE_ORPHANS)
    bridge._RESTORE_ORPHANS[:] = [
        item for item in bridge._RESTORE_ORPHANS if item[0] is not owner]


def test_failed_restore_reaps_only_after_terminal_and_one_way_teardown():
    from mlx2.runtime import paged_apcv2_native_restore as bridge

    class ClosedOwner:
        def __init__(self):
            self.fully_retired = False
            self.calls = []

        def close(self):
            self.calls.append("close")

        def reap_failed_after_teardown(self):
            self.calls.append("reap")
            self.fully_retired = True

    owner = ClosedOwner()
    writer = type("Writer", (), {})()
    writer.poisoned = True
    writer.pending_epochs = (1,)
    writer.ledger = type("Ledger", (), {"pending_count": 1})()
    writer.teardown_failed_arena = lambda: owner.calls.append("teardown")
    retire_failed_native_restore(owner, writer)
    assert owner.calls == ["close"]
    assert bridge.reap_failed_native_restores() >= 1
    assert owner.calls == ["close"]
    writer.pending_epochs = ()
    writer.ledger.pending_count = 0
    bridge.reap_failed_native_restores()
    assert owner.calls == ["close", "teardown", "reap"]
    assert not any(item[0] is owner for item in bridge._RESTORE_ORPHANS)
