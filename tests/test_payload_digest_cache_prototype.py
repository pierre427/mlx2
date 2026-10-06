"""Fail-closed tests for the default-off payload digest cache prototype."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prototype_payload_digest_cache.py"
SPEC = importlib.util.spec_from_file_location("prototype_payload_digest_cache", SCRIPT)
assert SPEC and SPEC.loader
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


def test_cache_hit_avoids_rehash(tmp_path, monkeypatch):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    cache_path = tmp_path / "cache.json"
    first = cache.digest_paths([payload], cache_path)
    assert first["misses"] == 1

    monkeypatch.setattr(cache, "_hash_fd", lambda _descriptor: pytest.fail("rehash"))
    second = cache.digest_paths([payload], cache_path)
    assert second["hits"] == 1
    assert second["files"][0]["sha256"] == first["files"][0]["sha256"]


@pytest.mark.parametrize("replacement", [b"PAYLOAD", b"short"])
def test_replacement_or_truncation_invalidates_cache(tmp_path, replacement):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    cache_path = tmp_path / "cache.json"
    first = cache.digest_paths([payload], cache_path)
    payload.write_bytes(replacement)
    second = cache.digest_paths([payload], cache_path)
    assert second["misses"] == 1
    assert second["files"][0]["sha256"] != first["files"][0]["sha256"]


def test_corrupt_or_unsafe_cache_fails_closed(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("not-json")
    with pytest.raises(ValueError, match="corrupt"):
        cache.digest_paths([payload], cache_path)

    cache_path.write_text(json.dumps({"schema": cache.SCHEMA, "entries": {}}))
    cache_path.chmod(0o666)
    with pytest.raises(ValueError, match="unsafe"):
        cache.digest_paths([payload], cache_path)


def test_symlink_target_change_invalidates_cache(tmp_path):
    first_target = tmp_path / "first.bin"
    second_target = tmp_path / "second.bin"
    first_target.write_bytes(b"first")
    second_target.write_bytes(b"second")
    payload = tmp_path / "payload.bin"
    payload.symlink_to(first_target)
    cache_path = tmp_path / "cache.json"
    first = cache.digest_paths([payload], cache_path)
    payload.unlink()
    payload.symlink_to(second_target)
    second = cache.digest_paths([payload], cache_path)
    assert second["misses"] == 1
    assert second["files"][0]["sha256"] != first["files"][0]["sha256"]


def test_stat_change_during_hash_fails_closed(tmp_path, monkeypatch):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    original = cache._hash_fd

    def mutate(descriptor):
        digest = original(descriptor)
        payload.write_bytes(b"changed")
        return digest

    monkeypatch.setattr(cache, "_hash_fd", mutate)
    with pytest.raises(ValueError, match="changed during"):
        cache.digest_paths([payload], tmp_path / "cache.json")
