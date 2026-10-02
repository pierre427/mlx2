"""48-hour sweep regressions: persistence namespace and startup env bounds."""

import mlx.core as mx
import pytest

from mlx2.runtime import recurrent_state_codec as rsc
from mlx2.runtime.apc_v2 import APCv2
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.serving import (
    apc_request_semantic,
    apc_semantic_namespace,
    mlx_cache_limit_bytes,
)

LAW = {"schema": "selected-adapter-law.v1", "algorithm": "reference-v1"}


def key(tenant=None, scope=None, law=LAW):
    return APCv2.key(
        "artifact",
        revision="source",
        adapter="artifact",
        tokenizer_fingerprint="tokenizer",
        cache_layout_fingerprint="layout",
        semantic_fingerprint=apc_semantic_namespace(
            apc_request_semantic(tenant, scope),
            adapter_execution_numerics=law,
        ),
    )


def cache():
    item = KVCache()
    values = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
    item.update_and_fetch(values, values)
    mx.eval(item.state)
    return [item]


def persistent(directory, template, mode):
    return APCv2(
        max_size=8,
        max_bytes=1 << 20,
        layout_name="layout",
        idle_disk_seconds=180,
        idle_disk_dir=str(directory),
        persist_dir=str(directory),
        persist_identity=template,
        persist_semantic_namespace=mode,
    )


@pytest.mark.parametrize("tenant", [None, "tenant-a:media:literal"])
@pytest.mark.parametrize("scope", [None, "image-and-lora-revision"])
def test_selected_adapter_namespace_survives_real_restart(tmp_path, tenant, scope):
    mode = "shared" if tenant is None else "tenant"
    template = key(None if tenant is None else "__tenant_template__")
    request = key(tenant, scope)
    tokens, tag = [1, 2, 3, 4], (tenant or "default", "session")
    first = persistent(tmp_path, template, mode)
    try:
        assert first._identity_matches(request)
        assert not first._identity_matches(
            key(tenant, scope, {**LAW, "algorithm": "other"})
        )
        assert not first._identity_matches(key(tenant, scope, None))
        assert first.store(request, tokens, cache(), session_tag=tag).stored
        assert first.park_session(*tag, ttl_seconds=60)["state"] == "disk"
        assert list(tmp_path.glob("*.manifest.json"))
    finally:
        first.close()
    second = persistent(tmp_path, template, mode)
    try:
        assert second.apc_stats["persistence"]["rescan"]["registered"] == 1
        second.resume_session(*tag)
        # Lookup materializes the disk snapshot and retains its actual state.
        hit = second.lookup(request, tokens + [5])
        assert hit.cached_tokens == 4
        assert mx.array_equal(
            hit.cache[0].keys[..., :4, :], cache()[0].keys[..., :4, :]
        ).item()
        hit.cache.close()
    finally:
        second.close()


def test_adapter_wrapper_keeps_default_and_semantic_isolation():
    assert key(law=None).semantic_fingerprint == "text-token-v1"
    assert key("a") != key("b")
    assert key(None, "image-a") != key(None, "image-b")


@pytest.mark.parametrize(
    "raw", ["nan", "NaN", "inf", "-inf", "1e309", "1e308", "1e20", "oops", "-1"]
)
def test_invalid_cache_limit_env_cannot_abort_worker_startup(monkeypatch, raw):
    monkeypatch.setenv("MLX2_CACHE_LIMIT_GIB", raw)
    assert mlx_cache_limit_bytes() is None


@pytest.mark.parametrize("raw, expected", [("0", 0), ("0.5", 1 << 29), ("2", 2 << 30)])
def test_valid_cache_limit_env_keeps_byte_semantics(monkeypatch, raw, expected):
    monkeypatch.setenv("MLX2_CACHE_LIMIT_GIB", raw)
    assert mlx_cache_limit_bytes() == expected


@pytest.mark.parametrize(
    "kind",
    [
        "foreign-tag",
        "negative-scale",
        "nonfinite-scale",
        "out-of-range",
        "exact-namespace",
    ],
)
def test_malformed_resident_codec_state_is_refused_before_publication(kind):
    policy = rsc.RecurrentStateCodecPolicy(enabled=kind != "exact-namespace")
    apc = APCv2(layout_name="layout", max_bytes=1 << 20, state_codec=policy)
    state = ArraysCache(2)
    state[0] = mx.zeros((1, 3, 2), dtype=mx.float32)
    record = rsc.encode(mx.ones((1, 1, 1, 4), dtype=mx.float32))
    if kind == "foreign-tag":
        record[rsc.CODEC_KEY] = mx.array(list(b"unknown-codec"), mx.uint8)
    elif kind == "negative-scale":
        record["scale"] = -record["scale"]
    elif kind == "nonfinite-scale":
        record["scale"] = record["scale"] * float("nan")
    elif kind == "out-of-range":
        record["q"] = mx.full(record["q"].shape, -128, dtype=mx.int8)
    state[1] = record
    request_key = APCv2.key(
        "model",
        semantic_fingerprint=rsc.apc_state_codec_fingerprint("text-token-v1", policy),
    )
    try:
        assert not apc.store(request_key, [1, 2], [state]).stored
        assert not apc.lookup(request_key, [1, 2, 3]).hit
        assert apc.apc_stats["recurrent_state_codec"]["payload_refusals"] == 1
        assert state[1] is record
    finally:
        apc.close()


def test_unknown_codec_cannot_be_silently_decoded_as_current_codec():
    record = rsc.encode(mx.ones((1, 1, 1, 4), dtype=mx.float32))
    record[rsc.CODEC_KEY] = mx.array(list(b"unknown-codec"), mx.uint8)
    with pytest.raises(ValueError, match="unknown recurrent state codec"):
        rsc.decode(record)


@pytest.mark.parametrize("hybrid", [False, True])
def test_codec_observed_use_follows_actual_warm_request_planes(monkeypatch, hybrid):
    from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp
    from test_standard_xpress_serving_cpu import tiny

    patch_host(monkeypatch)
    if hybrid:
        model, vocab = tiny_qwen38_mtp()
    else:
        model, _draft = tiny()
        vocab = 9
        model.eval()
        mx.eval(model.parameters())
    engine = make_engine(model, vocab, mtp=False, recurrent_state_codec="int8-row-v1")
    body = {
        "tokens": [i % (vocab - 2) + 1 for i in range(24)],
        "max_tokens": 3,
        "temperature": 0,
        "top_k": 0,
    }
    try:
        cold = collect(engine.submit(dict(body)))
        warm = collect(engine.submit(dict(body)))
        assert "error" not in cold and "error" not in warm
        assert warm["receipt"]["cached_tokens"] > 0
        cold_codec = cold["receipt"]["recurrent_state_codec"]
        warm_codec = warm["receipt"]["recurrent_state_codec"]
        assert not cold_codec["observed_used"]
        assert warm_codec["observed_used"] is hybrid
        assert warm_codec["restored_from_codec_state"] is hybrid
        assert (warm_codec["restored_leaves"] > 0) is hybrid
        assert (warm_codec["restored_tokens"] > 0) is hybrid
    finally:
        engine.close()
