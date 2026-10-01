"""APCv2 storage codec for recurrent state (approximate, default off).

Round-trip bounds, which leaves are touched, key separation between exact
and codec namespaces, persistent-tier refusal across a codec mismatch, and
fail-closed handling of corrupt payloads.  Live caches are never modified.
"""

import json
import time

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime import apc_v2 as apc_mod
from mlx2.runtime import recurrent_state_codec as rsc
from mlx2.runtime.apc_v2 import APCKey, APCv2
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.recurrent_state_codec import (
    RecurrentStateCodecPolicy,
    apc_state_codec_fingerprint,
    decode,
    encode,
    encode_prompt_cache,
    validate_prompt_cache,
)

POLICY = RecurrentStateCodecPolicy(enabled=True)
SEMANTIC = ("text-token-v1", "tenant", "tenant-a")


def _key(policy=None, semantic=SEMANTIC):
    return APCKey(
        "artifact-a",
        revision="source-a",
        adapter="artifact-a",
        tokenizer_fingerprint="tokenizer-a",
        cache_layout_fingerprint="layout-a",
        semantic_fingerprint=apc_state_codec_fingerprint(semantic, policy),
    )


def _recurrent_state(seed=0, shape=(1, 4, 8, 64)):
    rng = np.random.default_rng(seed)
    value = rng.standard_normal(shape).astype(np.float32)
    value[..., :2] *= 40.0  # outlier key channels, as in real GDN states
    return mx.array(value)


def _hybrid(length, seed=0, *, checkpoints=()):
    recurrent = ArraysCache(2)
    recurrent[0] = mx.ones((1, 3, 24), dtype=mx.bfloat16) * (seed + 1)
    recurrent[1] = _recurrent_state(seed)
    recurrent.lengths = mx.array([length], dtype=mx.int32)
    recurrent._host_lengths = (recurrent.lengths, [length])
    for position in checkpoints:
        recurrent.state_checkpoint([position], force=True)
    attention = KVCache()
    values = mx.arange(seed, seed + length, dtype=mx.float32).reshape(1, 1, length, 1)
    attention.update_and_fetch(values, values)
    mx.eval(recurrent.state, attention.state)
    return [recurrent, attention]


def _apc(directory=None, *, policy=None, persist=False):
    kwargs = {}
    if directory is not None:
        kwargs.update(idle_disk_seconds=180, idle_disk_dir=str(directory))
    if persist:
        kwargs.update(
            persist_dir=str(directory),
            persist_identity=_key(policy, ("text-token-v1", "tenant", "__template__")),
            persist_semantic_namespace="tenant",
        )
    return APCv2(
        max_size=16, max_bytes=1 << 30, layout_name="layout-a",
        state_codec=policy, **kwargs,
    )


# ------------------------------------------------------------------ codec
def test_round_trip_is_bounded_by_half_a_step_per_row():
    state = _recurrent_state(3)
    record = encode(state)
    restored = decode(record)
    assert restored.dtype == mx.float32 and restored.shape == state.shape
    step = mx.max(mx.abs(state), axis=-1, keepdims=True) / 127.0
    error = mx.abs(restored - state)
    assert bool(mx.all(error <= step * 0.5 + 1e-7))
    # The per-row max survives to within float32 rounding of 127 * (max / 127).
    assert np.allclose(
        np.array(mx.max(mx.abs(restored), axis=-1)),
        np.array(mx.max(mx.abs(state), axis=-1)),
        rtol=1e-6,
    )
    rel = float(mx.sqrt(mx.sum(error * error)) / mx.sqrt(mx.sum(state * state)))
    assert rel < 0.02
    # 8.25 bits/value plus an 11-byte codec tag.
    assert rsc.leaf_nbytes(record) == state.size + state.size // 64 * 4 + len(rsc.INT8_ROW_V1)


def test_zero_rows_are_exact_and_nonfinite_rows_decode_to_nan():
    state = _recurrent_state(1)
    zeros = mx.zeros((1, 1, 2, 64), dtype=mx.float32)
    assert bool(mx.all(decode(encode(zeros)) == 0))
    broken = mx.concatenate([state[:, :1, :1], mx.full((1, 1, 1, 64), np.inf)], axis=2)
    restored = decode(encode(broken))
    assert bool(mx.all(mx.isnan(restored[:, :, 1])))
    assert bool(mx.all(mx.isfinite(restored[:, :, 0])))


def test_encoding_is_deterministic():
    state = _recurrent_state(5)
    first, second = encode(state), encode(state)
    for name in ("q", "scale", rsc.CODEC_KEY):
        assert np.array_equal(np.array(first[name]), np.array(second[name]))


def test_only_rank4_float32_arrays_cache_slots_are_encoded_and_sources_untouched():
    caches = _hybrid(8, checkpoints=(4, 8))
    recurrent, attention = caches
    before = np.array(recurrent[1])
    conv = recurrent[0]
    encoded, counts = encode_prompt_cache(caches, POLICY)
    # Live slot + two checkpoint snapshots.
    assert counts["encoded_leaves"] == 3
    assert encoded[1] is attention
    assert encoded[0] is not recurrent
    assert encoded[0][0] is conv  # convolution window stays exact
    assert rsc.is_encoded(encoded[0][1])
    assert all(rsc.is_encoded(snap[1]) for _p, snap in encoded[0]._checkpoints[0])
    # The live cache keeps its float32 state and checkpoints.
    assert isinstance(recurrent[1], mx.array)
    assert np.array_equal(np.array(recurrent[1]), before)
    assert all(isinstance(snap[1], mx.array) for _p, snap in recurrent._checkpoints[0])
    assert encoded[0].nbytes < recurrent.nbytes * 0.4


def test_policy_parsing_and_disabled_namespace_is_identity():
    assert not RecurrentStateCodecPolicy.from_value(None).enabled
    assert not RecurrentStateCodecPolicy.from_value("off").enabled
    assert RecurrentStateCodecPolicy.from_value("int8-row-v1").enabled
    with pytest.raises(ValueError):
        RecurrentStateCodecPolicy.from_value("int4")
    with pytest.raises(ValueError):
        RecurrentStateCodecPolicy(enabled=True, qualified=True)
    assert apc_state_codec_fingerprint(SEMANTIC, RecurrentStateCodecPolicy()) == SEMANTIC
    wrapped = apc_state_codec_fingerprint(SEMANTIC, POLICY)
    assert wrapped != SEMANTIC
    assert rsc.key_codec_layer(wrapped) == f"int8-row-v1@{POLICY.revision}"
    status = POLICY.as_dict()
    assert status["fidelity"] == "approximate" and status["qualified"] is False
    assert status["state"] == "candidate"
    receipt = POLICY.receipt(restored_tokens=12)
    assert receipt["qualified"] is False and receipt["reason"] == "candidate_validation"
    assert receipt["restored_from_codec_state"] is True


def test_apc_unwraps_the_codec_layer_like_other_numerical_laws():
    base, layers = apc_mod._numerics_layers(apc_state_codec_fingerprint(SEMANTIC, POLICY))
    assert base == SEMANTIC
    assert layers == ((rsc.TAG, f"int8-row-v1@{POLICY.revision}"),)


# ------------------------------------------------------------- APCv2 memory
def test_codec_apc_stores_compressed_and_restores_float32():
    apc = _apc(policy=POLICY)
    key = _key(POLICY)
    source = _hybrid(8, seed=2, checkpoints=(4,))
    exact_bytes = sum(c.nbytes for c in source)
    reference = np.array(source[0][1])
    assert apc.store(key, list(range(8)), source).stored
    entry = apc._trie.get(key, list(range(8)))
    stored_recurrent = entry.prompt_cache[0]
    assert rsc.is_encoded(stored_recurrent[1])
    assert entry.nbytes < exact_bytes
    hit = apc.lookup(key, list(range(8)) + [99])
    assert hit.hit and hit.cached_tokens == 8
    restored = hit.cache[0][1]
    assert isinstance(restored, mx.array) and restored.dtype == mx.float32
    rel = np.linalg.norm(np.array(restored) - reference) / np.linalg.norm(reference)
    assert 0 < rel < 0.02
    # Restored checkpoints are decoded too, and the stored entry stays encoded.
    assert all(isinstance(s[1], mx.array) for _p, s in hit.cache[0]._checkpoints[0])
    assert rsc.is_encoded(entry.prompt_cache[0][1])
    # The caller's live cache was never touched.
    assert np.array_equal(np.array(source[0][1]), reference)
    stats = apc.apc_stats["recurrent_state_codec"]
    assert stats["enabled"] and stats["stores_encoded"] == 1
    assert stats["encoded_leaves"] == 2
    assert stats["encoded_bytes"] < 0.3 * stats["source_bytes"]
    hit.cache.close()
    apc.close()


def test_default_apc_is_exact_and_unchanged():
    apc = _apc()
    key = _key(None)
    source = _hybrid(8, seed=4)
    reference = np.array(source[0][1])
    assert apc.store(key, list(range(8)), source).stored
    hit = apc.lookup(key, list(range(8)) + [1])
    assert np.array_equal(np.array(hit.cache[0][1]), reference)
    assert apc.apc_stats["recurrent_state_codec"]["enabled"] is False
    assert apc.apc_stats["recurrent_state_codec"]["encoded_leaves"] == 0
    hit.cache.close()
    apc.close()


def test_key_separation_between_exact_and_codec_namespaces():
    exact_key, codec_key = _key(None), _key(POLICY)
    codec_apc = _apc(policy=POLICY)
    # A codec instance refuses to store or serve an exact namespace ...
    assert not codec_apc.store(exact_key, [1, 2, 3], _hybrid(3)).stored
    miss = codec_apc.lookup(exact_key, [1, 2, 3, 4])
    assert not miss.hit and miss.miss_reason == "state_codec_namespace_mismatch"
    # ... and an exact instance refuses a codec namespace.
    exact_apc = _apc()
    assert not exact_apc.store(codec_key, [1, 2, 3], _hybrid(3)).stored
    assert exact_apc.lookup(codec_key, [1, 2, 3, 4]).miss_reason == (
        "state_codec_namespace_mismatch"
    )
    # A different codec revision is a different namespace as well.
    other = ("text-token-v1", "tenant", "tenant-a")
    forged = APCKey(
        "artifact-a", revision="source-a", adapter="artifact-a",
        tokenizer_fingerprint="tokenizer-a", cache_layout_fingerprint="layout-a",
        semantic_fingerprint=(other, rsc.TAG, "int8-row-v1@0000000000000000"),
    )
    assert not codec_apc.store(forged, [1, 2, 3], _hybrid(3)).stored
    assert codec_apc.apc_stats["recurrent_state_codec"]["key_mismatch_refusals"] == 3
    assert exact_apc.apc_stats["recurrent_state_codec"]["key_mismatch_refusals"] == 2
    codec_apc.close()
    exact_apc.close()


# ------------------------------------------------------- APCv2 disk tiers
def _park(apc, key, tokens, seed=0):
    tag = ("tenant-a", f"conversation-{seed}")
    apc.store(key, tokens, _hybrid(len(tokens), seed, checkpoints=(len(tokens) // 2,)),
              session_tag=tag)
    assert apc.park_session(*tag, ttl_seconds=600)["state"] == "disk"
    return tag


def test_idle_disk_round_trip_keeps_the_payload_int8(tmp_path):
    apc = _apc(tmp_path, policy=POLICY)
    key = _key(POLICY)
    tokens = list(range(10))
    tag = _park(apc, key, tokens, seed=6)
    snapshot = next(tmp_path.glob("apc-idle-*.safetensors"))
    arrays = mx.load(str(snapshot))
    payloads = [name for name, value in arrays.items() if value.dtype == mx.int8]
    # Only int8 payloads and their [.., 1] row scales; no float32 state.
    assert payloads and not any(
        value.dtype == mx.float32 and value.ndim == 4 and value.shape[-1] > 1
        for value in arrays.values()
    )
    hit = apc.lookup(key, tokens + [99], session_tag=tag)
    assert hit.hit and hit.cached_tokens == 10
    assert hit.cache[0][1].dtype == mx.float32
    assert apc.apc_stats["recurrent_state_codec"]["disk_restore_validated"] == 1
    hit.cache.close()
    apc.close()


def test_persistent_tier_refuses_across_a_codec_mismatch(tmp_path):
    tokens = list(range(12))
    writer = _apc(tmp_path, policy=POLICY, persist=True)
    _park(writer, _key(POLICY), tokens, seed=7)
    writer.close()

    # An exact server never adopts the compressed snapshot ...
    exact = _apc(tmp_path, persist=True)
    rescan = exact.apc_stats["persistence"]["rescan"]
    assert rescan["registered"] == 0
    assert not exact.lookup(_key(None), tokens + [1]).hit
    exact.close()


def test_persistent_tier_codec_server_restores_its_own_snapshots(tmp_path):
    tokens = list(range(12))
    writer = _apc(tmp_path, policy=POLICY, persist=True)
    _park(writer, _key(POLICY), tokens, seed=8)
    writer.close()
    reader = _apc(tmp_path, policy=POLICY, persist=True)
    assert reader.apc_stats["persistence"]["rescan"]["registered"] == 1
    hit = reader.lookup(_key(POLICY), tokens + [1])
    assert hit.hit and hit.cached_tokens == 12
    assert hit.cache[0][1].dtype == mx.float32
    hit.cache.close()
    reader.close()


def test_exact_snapshots_are_not_adopted_by_a_codec_server(tmp_path):
    tokens = list(range(6))
    writer = _apc(tmp_path, persist=True)
    _park(writer, _key(None), tokens, seed=9)
    writer.close()
    reader = _apc(tmp_path, policy=POLICY, persist=True)
    assert reader.apc_stats["persistence"]["rescan"]["registered"] == 0
    reader.close()


# ------------------------------------------------------------ fail closed
def _tampered_loader(monkeypatch, mutate):
    real = apc_mod.load_prompt_cache

    def load(path, *args, **kwargs):
        cache = real(path, *args, **kwargs)
        mutate(cache)
        return cache

    monkeypatch.setattr(apc_mod, "load_prompt_cache", load)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda c: c[0].cache[1].update(q=mx.full(c[0].cache[1]["q"].shape, -128, dtype=mx.int8)), id="payload-out-of-range"),
        pytest.param(lambda c: c[0].cache[1].update(scale=c[0].cache[1]["scale"] * np.nan), id="nan-scale"),
        pytest.param(lambda c: c[0].cache[1].update(scale=-c[0].cache[1]["scale"]), id="negative-scale"),
        pytest.param(lambda c: c[0].cache[1].update(rsc_codec=mx.array(list(b"int8-row-v9"), dtype=mx.uint8)), id="foreign-codec"),
        pytest.param(lambda c: c[0].cache[1].pop("scale"), id="missing-scale"),
        pytest.param(lambda c: c[0].cache[1].update(q=c[0].cache[1]["q"].astype(mx.int16)), id="wrong-dtype"),
        pytest.param(lambda c: c[0].cache.__setitem__(1, mx.zeros((1, 4, 8, 64))), id="exact-leaf-in-codec-namespace"),
    ],
)
def test_corrupt_payloads_fail_closed_on_disk_restore(tmp_path, monkeypatch, mutate):
    apc = _apc(tmp_path, policy=POLICY)
    key = _key(POLICY)
    tokens = list(range(10))
    tag = _park(apc, key, tokens, seed=10)
    _tampered_loader(monkeypatch, mutate)
    miss = apc.lookup(key, tokens + [99], session_tag=tag)
    assert not miss.hit
    stats = apc.apc_stats
    assert stats["idle_disk"]["restore_failures"] == 1
    assert stats["recurrent_state_codec"]["disk_restore_validated"] == 0
    apc.close()


def test_encoded_record_in_an_exact_namespace_fails_closed():
    caches, _counts = encode_prompt_cache(_hybrid(4), POLICY)
    with pytest.raises(ValueError, match="exact APCv2 namespace"):
        validate_prompt_cache(caches, RecurrentStateCodecPolicy())
    assert validate_prompt_cache(caches, POLICY) == 1
    assert validate_prompt_cache(_hybrid(4), RecurrentStateCodecPolicy()) == 0


def test_decode_rejects_malformed_records():
    record = encode(_recurrent_state(0))
    bad_shape = dict(record, scale=record["scale"][..., :1, :])
    with pytest.raises(ValueError):
        decode(bad_shape)
    with pytest.raises(ValueError):
        decode({**record, "extra": mx.zeros(1)})


# ---------------------------------------------------------------- serving
def test_serving_gate_requires_qualification_mode():
    from mlx2.serving import recurrent_state_codec_policy

    assert not recurrent_state_codec_policy(None, qualification_mode=False, qualification=None).enabled
    with pytest.raises(ValueError, match="qualification mode"):
        recurrent_state_codec_policy("int8-row-v1", qualification_mode=False, qualification=None)
    policy = recurrent_state_codec_policy("int8-row-v1", qualification_mode=True, qualification=None)
    assert policy.enabled and not policy.qualified


def test_serving_cli_default_is_off():
    from mlx2.server import build_parser

    args = build_parser().parse_args(["--model", "dummy"])
    assert args.recurrent_state_codec == "off"


def test_codec_route_is_labelled_approximate():
    from mlx2 import qualification

    source = open(qualification.__file__).read()
    assert '(settings.get("recurrent_state_codec") or {}).get("enabled") is True' in source


def test_engine_argument_validation_gates_the_codec():
    from mlx2.serving import ServingEngine

    with pytest.raises(ValueError, match="qualification mode"):
        ServingEngine.validate_arguments("/nonexistent", recurrent_state_codec="int8-row-v1")
    ServingEngine.validate_arguments(
        "/nonexistent", recurrent_state_codec="int8-row-v1", qualification_mode=True
    )
    ServingEngine.validate_arguments("/nonexistent")


def test_selected_codec_requires_observed_encode_and_restore():
    from mlx2.qualification import required_feature_checks

    settings = {"mtp": False, "max_context": 4096, "execution_policy": {}, "environment": {}}
    assert "feature_recurrent_state_codec" not in required_feature_checks(settings)
    enabled = {**settings, "recurrent_state_codec": {"enabled": True, "codec": "int8-row-v1"}}
    assert "feature_recurrent_state_codec" in required_feature_checks(enabled)
