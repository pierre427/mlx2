"""A leased deleted session must not return after persistent APCv2 restart."""

import pytest

from mlx2.runtime.apc_v2 import APCSessionNotFound
from test_apcv2_sessions_persistence import _identity, _persistent, _state


def test_deleted_leased_session_does_not_return_after_crash(tmp_path):
    key = _identity()
    tokens = list(range(8))
    tag = ("tenant-a", "deleted-while-leased")
    first = _persistent(tmp_path)
    second = None
    lease = None
    try:
        first.store(key, tokens, [_state(8)], session_tag=tag)
        first.park_session(*tag, ttl_seconds=600)
        lease = first.lookup(key, tokens + [99], session_tag=tag)
        assert lease.hit
        report = first.delete_session(*tag)
        assert report["deferred_leased_entries"] == 1
        with pytest.raises(APCSessionNotFound):
            first.session_state(*tag)
        # The live request retains its exact state despite the session delete.
        assert lease.cache[0].keys[0, 0, :8, 0].tolist() == list(range(8))
        # Simulate process loss before the lease closes or the idle scan retires
        # its entry. Do not run close(), which would rewrite the manifest.
        first._release_persist_lock()
        second = _persistent(tmp_path)
        with pytest.raises(APCSessionNotFound):
            second.session_state(*tag)
        assert second.apc_stats["persistence"]["rescan"]["registered"] == 0
        assert not list(tmp_path.glob("apc-idle-*"))
    finally:
        if second is not None:
            second.close()
        if lease is not None:
            lease.cache.close()
        first.close()


@pytest.mark.parametrize("shared", [False, True])
def test_session_delete_manifest_failure_is_retryable(tmp_path, monkeypatch, shared):
    from mlx2.runtime.apc_v2 import APCSessionUnavailable

    key = _identity()
    tokens = list(range(8))
    tag = ("tenant-a", "delete-retry")
    apc = _persistent(tmp_path)
    lease = None
    try:
        apc.store(key, tokens, [_state(8)], session_tag=tag)
        if shared:
            apc.store(key, tokens, [_state(8)], session_tag=("tenant-a", "kept"))
        apc.park_session(*tag, ttl_seconds=600)
        lease = apc.lookup(key, tokens + [99], session_tag=tag)
        before = apc.session_state(*tag)
        with monkeypatch.context() as patch:
            patch.setattr(apc, "_write_manifest_locked", lambda *_args: False)
            with pytest.raises(APCSessionUnavailable, match="could not be persisted"):
                apc.delete_session(*tag)
        assert apc.session_state(*tag) == before
        apc.delete_session(*tag)
        with pytest.raises(APCSessionNotFound):
            apc.session_state(*tag)
        if shared:
            assert apc.session_state("tenant-a", "kept")["entries"] == 1
    finally:
        if lease is not None:
            lease.cache.close()
        apc.close()


def test_shared_session_delete_survives_restart_without_deleting_other_tag(tmp_path):
    key = _identity()
    tokens = list(range(8))
    first = _persistent(tmp_path)
    first.store(key, tokens, [_state(8)], session_tag=("tenant-a", "deleted"))
    first.store(key, tokens, [_state(8)], session_tag=("tenant-a", "kept"))
    first.park_session("tenant-a", "kept", ttl_seconds=600)
    first.delete_session("tenant-a", "deleted")
    first._release_persist_lock()
    second = _persistent(tmp_path)
    try:
        with pytest.raises(APCSessionNotFound):
            second.session_state("tenant-a", "deleted")
        assert second.session_state("tenant-a", "kept")["entries"] == 1
        hit = second.lookup(key, tokens + [99], session_tag=("tenant-a", "kept"))
        assert hit.hit
        hit.cache.close()
    finally:
        second.close()
        first.close()


def test_manifest_without_deletion_marker_remains_readable(tmp_path):
    import json

    key = _identity()
    tokens = list(range(8))
    tag = ("tenant-a", "legacy-manifest")
    first = _persistent(tmp_path)
    first.store(key, tokens, [_state(8)], session_tag=tag)
    first.park_session(*tag, ttl_seconds=600)
    first.close()
    path = next(tmp_path.glob("apc-idle-*.manifest.json"))
    value = json.loads(path.read_text())
    value.pop("delete_pending", None)
    path.write_text(json.dumps(value))
    second = _persistent(tmp_path)
    try:
        assert second.session_state(*tag)["entries"] == 1
        hit = second.lookup(key, tokens + [99], session_tag=tag)
        assert hit.hit
        hit.cache.close()
    finally:
        second.close()


def test_republish_after_delete_starts_a_fresh_session_while_old_lease_lives(tmp_path):
    key = _identity()
    tokens = list(range(8))
    old_tag = ("tenant-a", "deleted")
    new_tag = ("tenant-a", "new-publication")
    first = _persistent(tmp_path)
    lease = None
    second = None
    try:
        first.store(key, tokens, [_state(8)], session_tag=old_tag)
        first.park_session(*old_tag, ttl_seconds=600)
        lease = first.lookup(key, tokens + [99], session_tag=old_tag)
        first.delete_session(*old_tag)
        first.store(key, tokens, [_state(8, seed=100)], session_tag=new_tag)
        first.park_session(*new_tag, ttl_seconds=600)
        assert lease.cache[0].keys[0, 0, :8, 0].tolist() == list(range(8))
        first._release_persist_lock()
        second = _persistent(tmp_path)
        with pytest.raises(APCSessionNotFound):
            second.session_state(*old_tag)
        assert second.session_state(*new_tag)["entries"] == 1
        hit = second.lookup(key, tokens + [99], session_tag=new_tag)
        assert hit.cache[0].keys[0, 0, :8, 0].tolist() == list(range(100, 108))
        hit.cache.close()
    finally:
        if second is not None:
            second.close()
        if lease is not None:
            lease.cache.close()
        first.close()
