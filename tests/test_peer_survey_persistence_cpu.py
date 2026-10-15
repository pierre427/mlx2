"""Representation and target/draft identity through real APC persistence."""

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.models.cache import KVCache, load_prompt_cache, save_prompt_cache


def _cache(keys, values):
    cache = KVCache()
    cache.state = (keys, values)
    mx.eval(cache.state)
    return cache


def _bits(array):
    return array.view(mx.uint16).flatten().tolist()


def test_prompt_codec_preserves_every_bf16_bit_pattern_and_strided_values(tmp_path):
    # Includes signed zeros, subnormals, infinities and every NaN payload;
    # this is a representation oracle, not permission to publish NaN state.
    bits = mx.arange(65536, dtype=mx.uint16).reshape(1, 1, 256, 256)
    raw = bits.view(mx.bfloat16)
    strided = mx.transpose(raw, (0, 1, 3, 2))
    half = mx.arange(32, dtype=mx.float16).reshape(1, 1, 8, 4)
    single = mx.arange(32, dtype=mx.float32).reshape(1, 1, 8, 4)
    path = tmp_path / "mixed.safetensors"
    save_prompt_cache(str(path), [_cache(raw, strided), _cache(half, single)])
    restored = load_prompt_cache(str(path))
    assert restored[0].keys.dtype == mx.bfloat16
    assert _bits(restored[0].keys) == list(range(65536))
    assert _bits(restored[0].values) == _bits(strided)
    assert restored[1].keys.dtype == mx.float16
    assert restored[1].values.dtype == mx.float32
    assert mx.array_equal(restored[1].keys, half)
    assert mx.array_equal(restored[1].values, single)


def _persistent(path, key):
    return APCv2(max_size=8, max_bytes=1 << 20, layout_name="sidecar-refault",
                idle_disk_seconds=3600, idle_disk_dir=str(path), persist_dir=str(path),
                persist_identity=key, persist_semantic_namespace="tenant")


def _fixture():
    key = APCKey("artifact", revision="exact-revision", adapter="test",
                 tokenizer_fingerprint="tokens", cache_layout_fingerprint="sidecar-refault",
                 semantic_fingerprint=("text-token-v1", "tenant", "a"))
    raw = mx.arange(32, dtype=mx.uint16).reshape(1, 1, 8, 4).view(mx.bfloat16)
    draft = mx.arange(100, 132, dtype=mx.float16).reshape(1, 1, 8, 4)
    hidden = mx.arange(8, dtype=mx.float32).reshape(1, 8)
    rng = mx.array([17, 39], dtype=mx.uint32)
    sidecar = MTPAPCSidecar(([_cache(draft, draft)], hidden), covered_tokens=8,
                            rng_key=rng, rng_draws=19)
    return key, raw, draft, hidden, sidecar


def test_target_draft_rng_and_hidden_survive_two_refaults_and_restart(tmp_path):
    key, raw, draft, hidden, sidecar = _fixture()
    tokens = list(range(8))
    apc = _persistent(tmp_path, key)
    try:
        assert apc.store(key, tokens, [_cache(raw, raw)], sidecar=sidecar).stored
        assert apc.park_all(time_budget_seconds=5)["spilled"] == 1
    finally:
        apc.close()
    restarted = _persistent(tmp_path, key)
    try:
        for _ in range(2):
            hit = restarted.lookup(key, tokens + [99], require_sidecar=True)
            assert hit.hit and hit.cached_tokens == 8
            assert _bits(hit.cache[0].keys) == _bits(raw)
            assert hit.sidecar.covered_tokens == 8
            assert hit.sidecar.rng_draws == 19
            assert hit.sidecar.rng_key.tolist() == [17, 39]
            draft_cache, restored_hidden = hit.sidecar.state
            assert mx.array_equal(draft_cache[0].keys, draft)
            assert mx.array_equal(restored_hidden, hidden)
            hit.cache.close()
            if hasattr(draft_cache, "close"):
                draft_cache.close()
            assert restarted.park_all(time_budget_seconds=5)["spilled"] == 1
    finally:
        restarted.close()


@pytest.mark.parametrize("kind", ["target", "draft", "aux"])
def test_corrupt_component_cannot_restore_a_partial_checkpoint(tmp_path, kind):
    key, raw, _draft, _hidden, sidecar = _fixture()
    tokens = list(range(8))
    first = _persistent(tmp_path, key)
    try:
        first.store(key, tokens, [_cache(raw, raw)], sidecar=sidecar)
        assert first.park_all(time_budget_seconds=5)["spilled"] == 1
    finally:
        first.close()
    path = next(tmp_path.glob(f"apc-idle-*.{kind}.safetensors"))
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    restarted = _persistent(tmp_path, key)
    try:
        hit = restarted.lookup(key, tokens + [99], require_sidecar=True)
        assert not hit.hit and hit.sidecar is None
        assert restarted.apc_stats["idle_disk"]["restore_failures"] == 1
    finally:
        restarted.close()


def test_refault_binds_digest_to_staged_bytes_not_a_prior_path_check(tmp_path, monkeypatch):
    from pathlib import Path

    key, raw, _draft, _hidden, _sidecar = _fixture()
    tokens = list(range(8))
    apc = _persistent(tmp_path, key)
    try:
        apc.store(key, tokens, [_cache(raw, raw)])
        assert apc.park_all(time_budget_seconds=5)["spilled"] == 1
        entry = apc._trie.get(key, tokens)
        disk = entry._apc_disk
        apc._verify_persisted_files(disk)
        old_digest = disk["files"]["target"]["sha256"]
        replacement = tmp_path / "replacement.safetensors"
        other = mx.full(raw.shape, 2, dtype=mx.bfloat16)
        save_prompt_cache(str(replacement), [_cache(other, other)])
        payload = replacement.read_bytes()
        assert len(payload) == disk["files"]["target"]["size"]
        Path(disk["target"]).write_bytes(payload)
        # Model a path verifier that observed the earlier valid inode. A
        # separate check/open sequence would accept the replacement payload.
        monkeypatch.setattr(apc, "_sha256_file", lambda _path: old_digest)
        assert not apc.lookup(key, tokens + [99]).hit
        assert apc.apc_stats["idle_disk"]["restore_digest_failures"] == 1
    finally:
        apc.close()
