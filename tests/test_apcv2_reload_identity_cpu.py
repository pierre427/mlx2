"""A resident kernel swap cannot publish under the old persistence identity."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from test_apcv2_sessions_persistence import _identity, _persistent, _state

from mlx2.runtime.apc_v2 import APCMetadataIncomplete, APCSessionNotFound


def test_rebind_requires_clear_and_preserves_artifact_and_layout(tmp_path):
    apc = _persistent(tmp_path)
    original = _identity()
    new = replace(original, revision="kernel-dev:new-source")
    try:
        with pytest.raises(RuntimeError, match="explicit clear"):
            apc.rebind_empty_identity(new)
        apc.clear()
        for changed in (
            replace(new, model="other"), replace(new, adapter="other"),
            replace(new, tokenizer_fingerprint="other"),
            replace(new, cache_layout_fingerprint="other"),
            replace(new, semantic_fingerprint="other"),
        ):
            with pytest.raises(ValueError, match="only the execution revision"):
                apc.rebind_empty_identity(changed)
            assert apc._persist_identity == original
        receipt = apc.rebind_empty_identity(new)
        assert receipt["revision"] == new.revision
        assert apc._identity_matches(new)
        assert not apc._identity_matches(original)
    finally:
        apc.close()


@pytest.mark.parametrize("owner", ["entry", "spill", "prefetch", "restore", "metadata", "retirement"])
def test_rebind_refuses_old_state_or_pending_owner(tmp_path, owner):
    apc = _persistent(tmp_path)
    original = _identity()
    apc.clear()
    if owner == "entry":
        apc.store(original, list(range(8)), [_state(8)])
    elif owner == "spill":
        apc._spill_jobs = [object()]
    elif owner == "prefetch":
        apc._pending_prefetch = object()
    elif owner == "restore":
        apc._restore_staging_bytes = 16
    elif owner == "metadata":
        apc._metadata_future = NS(done=lambda: False)
    else:
        apc._pending_retirements[(original, (1, 2))] = "old"
    try:
        with pytest.raises(RuntimeError, match="empty, idle"):
            apc.rebind_empty_identity(replace(original, revision="new"))
        assert apc._persist_identity == original
    finally:
        apc._spill_jobs = []
        apc._pending_prefetch = None
        apc._restore_staging_bytes = 0
        apc._metadata_future = None
        apc._pending_retirements.clear()
        apc.close()


def test_new_revision_persists_and_old_revision_cannot_restore_it(tmp_path):
    original = _identity()
    new = replace(original, revision="kernel-dev:unique-process-source")
    tokens = list(range(8))
    apc = _persistent(tmp_path)
    second = None
    try:
        apc.store(original, tokens, [_state(8)], session_tag=("tenant-a", "old"))
        apc.park_session("tenant-a", "old", ttl_seconds=600)
        apc.clear()
        apc.rebind_empty_identity(new)
        with pytest.raises(APCSessionNotFound):
            apc.session_state("tenant-a", "old")
        apc.store(new, tokens, [_state(8)], session_tag=("tenant-a", "new"))
        apc.park_session("tenant-a", "new", ttl_seconds=600)
        apc.close()
        second = _persistent(tmp_path, revision=new.revision)
        hit = second.lookup(new, tokens + [9])
        assert hit.hit and hit.cached_tokens == len(tokens)
        hit.cache.close()
        assert second.lookup(original, tokens + [9]).cached_tokens == 0
    finally:
        if second is not None:
            second.close()
        apc.close()


def test_metadata_failure_does_not_publish_a_new_identity(tmp_path):
    apc = _persistent(tmp_path)
    original = _identity()
    try:
        apc.clear()
        apc._metadata_failure = "failed purge"
        apc._metadata_completion = "incomplete"
        with pytest.raises(APCMetadataIncomplete, match="failed purge"):
            apc.rebind_empty_identity(replace(original, revision="new"))
        assert apc._persist_identity == original
    finally:
        apc._metadata_failure = None
        apc.close()
