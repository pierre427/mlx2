"""Xing4.0 MLX port: parity with the HF torch reference and runtime contracts.

The fixture in tests/fixtures/xing4_0_tiny is produced by
scripts/xing4_0_reference_fixture.py from the upstream HF modeling code.
"""

import copy
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map_with_path

from mlx2.runtime.models import xing4_0
from mlx2.runtime.models.cache import BatchKVCache, KVCache
from mlx2.runtime.models.xing4_0 import HyperConnection, Model, ModelArgs

FIXTURE = Path(__file__).parent / "fixtures" / "xing4_0_tiny"
TIGHT = dict(rtol=1e-4, atol=2e-5)


def _config(**overrides):
    config = json.loads((FIXTURE / "config.json").read_text())
    config.update(overrides)
    return config


def _raw_weights():
    return mx.load(str(FIXTURE / "weights.safetensors"))


def _load(**overrides):
    model = Model(ModelArgs.from_dict(_config(**overrides)))
    weights = model.sanitize(_raw_weights())
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def model():
    return _load()


@pytest.fixture(scope="module")
def ref():
    return mx.load(str(FIXTURE / "reference.safetensors"))


@pytest.fixture(autouse=True)
def _restore_levers():
    yield
    xing4_0.set_compile_mhc(True)
    xing4_0.set_absorbed_max_query_override(None)
    xing4_0.set_mhc_kernel(True)


def close(a, b, **kw):
    np.testing.assert_allclose(np.array(a), np.array(b), **(kw or TIGHT))


# ----------------------------------------------------------------- parity


@pytest.mark.parametrize("seq", ["a", "b"])
def test_full_sequence_logits_match_hf_reference(model, ref, seq):
    close(model(ref[f"ids_{seq}"]), ref[f"logits_{seq}"])
    close(model.model(ref[f"ids_{seq}"]), ref[f"hidden_{seq}"])


def test_uncompiled_reference_path_matches_hf_and_compiled(model, ref):
    ids = ref["ids_a"]
    compiled = model(ids)
    eager = model.lm_head(model.model(ids, reference=True))
    close(eager, ref["logits_a"])
    close(eager, compiled, rtol=1e-5, atol=1e-6)
    xing4_0.set_compile_mhc(False)
    close(model(ids), eager, rtol=1e-6, atol=1e-6)


def test_mhc_matches_hf_and_compiled_equals_reference(model, ref):
    hc = model.model.layers[2].attn_hc
    post, comb, collapsed = hc.reference(ref["mhc_input"])
    close(post, ref["mhc_post"])
    close(comb, ref["mhc_comb"])
    close(collapsed, ref["mhc_collapsed"])
    # Doubly stochastic up to the Sinkhorn eps / iteration budget.
    close(comb.sum(-2), mx.ones(comb.shape[:-2] + (4,)), atol=1e-4)
    stats = xing4_0.mhc_stats(reset=True)
    c_post, c_comb, c_collapsed = hc(ref["mhc_input"])
    assert xing4_0.mhc_stats()["compiled_calls"] == 1
    assert stats["fallbacks"] == 0
    close(c_post, post, rtol=1e-6, atol=1e-7)
    close(c_comb, comb, rtol=1e-6, atol=1e-7)
    close(c_collapsed, collapsed, rtol=1e-6, atol=1e-6)
    out = mx.random.normal((2, 3, 64), key=mx.random.key(1))
    streams = ref["mhc_input"]
    close(
        HyperConnection.update(streams, out, post, comb),
        HyperConnection.update_reference(streams, out, post, comb),
        rtol=1e-6,
        atol=1e-6,
    )
    # HF orientation: rows of comb index the *output* stream.
    manual = mx.einsum("blij,bljd->blid", comb, streams) + post[..., None] * out[:, :, None, :]
    close(HyperConnection.update_reference(streams, out, post, comb), manual, rtol=1e-6, atol=1e-6)


def test_mtp_step_matches_deepseek_mtp_reference(model, ref):
    layer = model.mtp.layers[0]
    # Fixture: MTP embed duplicates the trunk (deduped), head differs (kept).
    assert "embed_tokens" not in layer
    assert "head" in layer.shared_head
    cache = model.make_mtp_cache()
    assert [type(c) for c in cache] == [KVCache]
    logits, post = model.mtp_step(ref["hidden_a"][:, :-1], ref["ids_a"][:, 1:], cache)
    assert logits.shape == (1, 10, 128) and post.shape == (1, 10, 64)
    close(logits, ref["mtp_logits_a"])
    close(post, ref["mtp_post_a"])
    assert cache[0].offset == 10


def test_absorbed_and_expanded_mla_branches_agree(model, ref):
    ids = ref["ids_a"]
    xing4_0.set_absorbed_max_query_override(0)
    expanded = model(ids)
    xing4_0.set_absorbed_max_query_override(1 << 20)
    absorbed = model(ids)
    close(expanded, ref["logits_a"])
    close(absorbed, ref["logits_a"])


# ----------------------------------------------------------- cache contracts


def test_make_cache_is_standard_kv(model):
    cache = model.make_cache()
    assert len(cache) == len(model.layers) == 4
    assert all(type(c) is KVCache for c in cache)
    assert Model.supports_speculative_rollback
    assert isinstance(Model.apc_v2_layout, str)


@pytest.mark.parametrize("chunks", [(11,), (5, 1, 1, 1, 1, 1, 1), (3, 4, 4), (1,) * 11])
def test_incremental_decode_matches_full_forward(model, ref, chunks):
    ids = ref["ids_a"]
    cache = model.make_cache()
    outs, start = [], 0
    for n in chunks:
        outs.append(model(ids[:, start : start + n], cache=cache))
        start += n
    assert cache[0].offset == 11
    # latent cache: keys = normed latent [B,1,S,r], values = roped key [B,1,S,d_rope]
    keys, values = cache[0].state
    assert keys.shape[1] == 1 and keys.shape[-1] == 32 and values.shape[-1] == 8
    close(mx.concatenate(outs, axis=1), ref["logits_a"])


def test_trim_rollback_is_exact(model, ref):
    ids = ref["ids_a"]
    cache = model.make_cache()
    model(ids[:, :6], cache=cache)
    speculative = mx.array([[5, 9, 17, 33]], mx.int32)
    model(speculative, cache=cache)
    assert all(c.is_trimmable() for c in cache)
    assert [c.trim(4) for c in cache] == [4] * 4
    got = model(ids[:, 6:], cache=cache)
    close(got, ref["logits_a"][:, 6:])


def _left_padded(ref):
    a, b = ref["ids_a"], ref["ids_b"]
    pad = a.shape[1] - b.shape[1]
    b_pad = mx.concatenate([mx.zeros((1, pad), a.dtype), b], axis=1)
    return mx.concatenate([a, b_pad], axis=0), [0, pad]


def test_batched_left_padded_prefill_matches_single(model, ref):
    ids, pads = _left_padded(ref)
    cache = [BatchKVCache(pads) for _ in model.layers]
    logits = model(ids, cache=cache)
    close(logits[0:1], ref["logits_a"])
    close(logits[1:2, pads[1] :], ref["logits_b"])
    # one decode step for both lanes vs single-lane decode
    nxt = mx.array([[3], [4]], mx.int32)
    step = model(nxt, cache=cache)
    for row, seq in enumerate("ab"):
        single = model.make_cache()
        model(ref[f"ids_{seq}"], cache=single)
        close(step[row : row + 1], model(nxt[row : row + 1], cache=single))


def test_merged_caches_decode_and_right_padded_verify(model, ref):
    singles = []
    for seq in "ab":
        c = model.make_cache()
        model(ref[f"ids_{seq}"], cache=c)
        singles.append(c)
    merged = [KVCache.merge([singles[0][i], singles[1][i]]) for i in range(4)]
    assert all(type(c) is BatchKVCache for c in merged)
    oracle = copy.deepcopy(singles)

    # ragged verify rows (self-MTP style): lane a 3 tokens, lane b 1 + 2 pad
    rows = mx.array([[7, 8, 9], [10, 0, 0]], mx.int32)
    lengths, right = [3, 1], [0, 2]
    for c in merged:
        c.prepare(lengths=lengths, right_padding=right)
    got = model(rows, cache=merged)
    for c in merged:
        c.finalize()
    want_a = model(rows[0:1], cache=oracle[0])
    want_b = model(rows[1:2, :1], cache=oracle[1])
    close(got[0:1], want_a)
    close(got[1:2, :1], want_b)

    # ragged rollback (reject last 2 of lane a), then continue together
    for c in merged:
        c.trim_ragged(mx.array([2, 0]))
    for c in oracle[0]:
        c.trim(2)
    nxt = mx.array([[11], [12]], mx.int32)
    step = model(nxt, cache=merged)
    close(step[0:1], model(nxt[0:1], cache=oracle[0]))
    close(step[1:2], model(nxt[1:2], cache=oracle[1]))


def test_mtp_incremental_and_batched_match_full(model, ref):
    hs, ts = ref["hidden_a"][:, :-1], ref["ids_a"][:, 1:]
    full_logits, full_post = model.mtp_step(hs, ts, model.make_mtp_cache())
    cache = model.make_mtp_cache()
    parts = [model.mtp_step(hs[:, :4], ts[:, :4], cache)]
    for i in range(4, 10):
        parts.append(model.mtp_step(hs[:, i : i + 1], ts[:, i : i + 1], cache))
    close(mx.concatenate([p[0] for p in parts], axis=1), full_logits)
    close(mx.concatenate([p[1] for p in parts], axis=1), full_post)
    # determinism
    again = model.mtp_step(hs, ts, model.make_mtp_cache())
    assert mx.array_equal(again[0], full_logits).item()
    # trim/rollback of the draft cache
    cache[0].trim(3)
    redo = model.mtp_step(hs[:, 7:], ts[:, 7:], cache)
    close(redo[0], full_logits[:, 7:])
    # two lanes, merged batched MTP cache, right-padded step
    c_a, c_b = model.make_mtp_cache(), model.make_mtp_cache()
    model.mtp_step(hs[:, :5], ts[:, :5], c_a)
    model.mtp_step(hs[:, :3], ts[:, :3], c_b)
    merged = [KVCache.merge([c_a[0], c_b[0]])]
    h2 = mx.concatenate([hs[:, 5:7], mx.concatenate([hs[:, 3:4], mx.zeros_like(hs[:, :1])], 1)])
    t2 = mx.concatenate([ts[:, 5:7], mx.concatenate([ts[:, 3:4], mx.zeros_like(ts[:, :1])], 1)])
    merged[0].prepare(lengths=[2, 1], right_padding=[0, 1])
    got_logits, _ = model.mtp_step(h2, t2, merged)
    merged[0].finalize()
    close(got_logits[0:1], full_logits[:, 5:7])
    close(got_logits[1:2, :1], full_logits[:, 3:4])


def test_greedy_self_mtp_runtime_matches_plain_greedy(model):
    """End-to-end through mlx2's native batched self-MTP scheduler."""
    from mlx2.runtime.hybrid_speculative import (
        attach_self_mtp_lanes,
        commit_batched_self_mtp,
        prepare_self_mtp_lane,
        propose_batched_self_mtp,
    )

    prompts = ([1, 5, 9, 13, 17, 21, 25], [2, 4, 6, 8, 10])
    steps = 10

    def plain(prompt):
        cache = model.make_cache()
        logits = model(mx.array([prompt], mx.int32), cache=cache)
        out = []
        for _ in range(steps):
            tok = int(mx.argmax(logits[0, -1]).item())
            out.append(tok)
            logits = model(mx.array([[tok]], mx.int32), cache=cache)
        return out

    lanes, emitted = [], []
    for uid, prompt in enumerate(prompts):
        detached, first = prepare_self_mtp_lane(
            mx.array(prompt, mx.uint32),
            model,
            uid=uid,
            max_tokens=steps,
            prompt_cache=None,
            mtp_state=None,
            lane_rng=None,
            num_draft=2,
            sampling_temp=0.0,
            sampling_top_p=1.0,
            sampling_top_k=0,
            sampling_min_p=0.0,
            accept_rule="exact",
            logits_processors=[],
            prefill_step_size=3,
            share_qsa_indices=False,
        )
        lanes.append(detached)
        emitted.append([int(first.token)])
    batch = attach_self_mtp_lanes(model, None, lanes)
    for _ in range(12):
        if all(len(e) >= steps for e in emitted):
            break
        proposal = propose_batched_self_mtp(model, batch)
        counts = [len(row) for row in proposal.outputs]
        for row, outputs in enumerate(proposal.outputs):
            emitted[row].extend(int(t.token) for t in outputs)
        commit_batched_self_mtp(
            batch, proposal, emitted_counts=counts, terminal=[False] * len(prompts)
        )
    for prompt, got in zip(prompts, emitted):
        assert got[:steps] == plain(prompt)


# ------------------------------------------------------------ weights / args


def test_sanitize_is_idempotent_and_reloads_strict(model):
    fresh = Model(ModelArgs.from_dict(_config()))
    once = fresh.sanitize(_raw_weights())
    twice = fresh.sanitize(dict(once))
    assert sorted(once) == sorted(twice)
    assert all(mx.array_equal(once[k], twice[k]).item() for k in once)
    assert not any("experts.0" in k or "kv_b_proj" in k or ".layers.4." in k for k in once)
    # Already-converted weights into a brand-new model (dedupe by absence).
    other = Model(ModelArgs.from_dict(_config()))
    reloaded = other.sanitize(dict(once))
    other.load_weights(list(reloaded.items()), strict=True)
    assert "embed_tokens" not in other.mtp.layers[0]
    ids = mx.array([[1, 2, 3, 4]], mx.int32)
    close(other(ids), model(ids), rtol=0, atol=0)


def test_sanitize_keeps_distinct_mtp_embedding(model):
    weights = _raw_weights()
    weights["model.layers.4.embed_tokens.weight"] = weights["model.layers.4.embed_tokens.weight"] + 1.0
    fresh = Model(ModelArgs.from_dict(_config()))
    sanitized = fresh.sanitize(weights)
    assert "mtp.layers.0.embed_tokens.weight" in sanitized
    fresh.load_weights(list(sanitized.items()), strict=True)
    assert "embed_tokens" in fresh.mtp.layers[0]


def test_mtp_disabled_or_absent_drops_layer(ref):
    off = _load(num_nextn_predict_layers=0)
    assert off.mtp is None
    close(off(ref["ids_a"]), ref["logits_a"])
    with pytest.raises(RuntimeError):
        off.make_mtp_cache()
    absent = Model(ModelArgs.from_dict(_config()))
    weights = {k: v for k, v in _raw_weights().items() if not k.startswith("model.layers.4.")}
    absent.load_weights(list(absent.sanitize(weights).items()), strict=True)
    assert absent.mtp is None


def test_model_args_fail_closed():
    with pytest.raises(ValueError):
        ModelArgs.from_dict(_config(topk_method="greedy"))
    with pytest.raises(ValueError):
        ModelArgs.from_dict(_config(scoring_func="softmax"))
    with pytest.raises(ValueError):
        ModelArgs.from_dict(_config(tie_word_embeddings=True))
    args = ModelArgs.from_dict(_config())
    assert args.rope_scaling["type"] == "yarn" and args.rope_interleave


def test_quant_and_cast_predicates_keep_mhc_and_router_full_precision():
    q = _load()
    pred = q.quant_predicate
    assert pred("model.layers.2.attn_hc", None) is False
    assert pred("model.layers.2.mlp.gate", None) is False
    assert pred("model.layers.2.self_attn.q_a_proj", None) is True
    cast = q.cast_predicate
    for path in (
        "model.layers.0.attn_hc.hc_fn",
        "model.layers.0.ffn_hc.hc_base",
        "model.layers.0.ffn_hc.hc_scale",
        "model.layers.2.mlp.gate.e_score_correction_bias",
        "model.layers.2.mlp.gate.weight",
    ):
        assert cast(path) is False
    assert cast("model.layers.0.self_attn.q_a_proj.weight") is True

    def class_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % 32:
            return False
        return pred(path, module)

    nn.quantize(q, group_size=32, bits=8, class_predicate=class_predicate)
    flat = dict(tree_flatten(q.parameters()))
    for key, value in flat.items():
        if any(s in key for s in ("hc_fn", "hc_base", "hc_scale", "mlp.gate.")):
            assert value.dtype == mx.float32, key
            assert not key.endswith(".scales"), key
    assert "model.layers.0.self_attn.q_a_proj.scales" in flat
    assert "model.layers.2.mlp.switch_mlp.gate_proj.scales" in flat
    ids = mx.array([[1, 2, 3, 4, 5]], mx.int32)
    assert q(ids).shape == (1, 5, 128)


def test_bf16_forward_tracks_fp32(model, ref):
    bf = _load()
    cast = bf.cast_predicate
    bf.update(
        tree_map_with_path(
            # CPU GatherMM is fp32-only, so routed experts stay fp32 here.
            lambda path, v: v.astype(mx.bfloat16)
            if cast(path) and "switch_mlp" not in path
            else v,
            bf.parameters(),
        )
    )
    assert bf.model.layers[0].self_attn.q_a_proj.weight.dtype == mx.bfloat16
    assert bf.model.layers[0].attn_hc.hc_fn.dtype == mx.float32
    logits = bf(ref["ids_a"]).astype(mx.float32)
    assert bool(mx.all(mx.isfinite(logits)).item())
    close(logits, ref["logits_a"], rtol=0, atol=0.25)


# ---------------------------------------------------------------- Metal mHC


def test_mhc_kernel_is_not_used_on_cpu(model):
    xing4_0.mhc_stats(reset=True)
    xing4_0.set_mhc_kernel(True)
    model(mx.array([[1, 2, 3]]))
    stats = xing4_0.mhc_stats()
    assert stats["kernel_calls"] == 0 and stats["kernel_fallbacks"] == 0


def _gpu():
    return mx.metal.is_available()


def _mhc_numpy(streams, hc_fn, scale, base, out, iters=20, eps=1e-6, lo=-30.0, hi=30.0):
    """float64 reference for the HF mHC coefficients, collapse and update."""
    s = np.asarray(streams, dtype=np.float64)
    flat = s.reshape(*s.shape[:-2], -1)
    flat = flat / np.sqrt((flat ** 2).mean(-1, keepdims=True) + 1e-6)
    mix = flat @ np.asarray(hc_fn, dtype=np.float64).T
    b = np.asarray(base, dtype=np.float64)
    sig = lambda v: 1.0 / (1.0 + np.exp(-v))
    pre = sig(mix[..., :4] * scale[0] + b[:4])
    post = 2.0 * sig(mix[..., 4:8] * scale[1] + b[4:8])
    comb = np.clip(mix[..., 8:].reshape(*mix.shape[:-1], 4, 4) * scale[2] + b[8:].reshape(4, 4), lo, hi)
    comb = np.exp(comb - comb.max(-1, keepdims=True))
    for _ in range(iters):
        comb = comb / (comb.sum(-1, keepdims=True) + eps)
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
    collapsed = (pre[..., None] * s).sum(-2)
    update = post[..., None] * np.asarray(out, dtype=np.float64)[..., None, :] + comb @ s
    return post, comb, collapsed, update


@pytest.mark.skipif(not _gpu(), reason="Metal is not available")
def test_mhc_kernels_match_float64_reference_on_gpu():
    from mlx2.runtime.models.xing4_0_mhc_metal import mhc_pre, mhc_update, pack_params

    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        H = 256
        rng = np.random.default_rng(3)
        hc_fn = mx.array(rng.normal(0, 0.05, (24, 4 * H)).astype(np.float32)).astype(mx.bfloat16)
        base = mx.array(rng.normal(0, 0.5, (24,)).astype(np.float32)).astype(mx.bfloat16)
        scale = [0.8, 1.1, 0.9]
        params = pack_params(1e-6, 1e-6, -30.0, 30.0, mx.array(scale), base)
        for shape in ((1, 1), (3, 1), (2, 37)):
            streams = mx.array(rng.normal(0, 3, (*shape, 4, H)).astype(np.float32))
            out = mx.array(rng.normal(0, 1, (*shape, H)).astype(np.float32))
            post, comb, collapsed, update = _mhc_numpy(
                streams, hc_fn.astype(mx.float32), scale, base.astype(mx.float32), out
            )
            got = mhc_pre(streams, hc_fn, params, iters=20)
            np.testing.assert_allclose(np.array(got[0]), post, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(np.array(got[1]), comb, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(np.array(got[2]), collapsed, rtol=1e-4, atol=1e-4)
            fused = mhc_update(streams, out, mx.array(post.astype(np.float32)), mx.array(comb.astype(np.float32)))
            np.testing.assert_allclose(np.array(fused), update, rtol=1e-4, atol=1e-4)
    finally:
        mx.set_default_device(previous)


def test_lazy_kernel_failure_falls_back_before_any_cache_write(model, monkeypatch):
    """A kernel whose failure only appears at evaluation must still fall back
    (the guard evaluates new specializations) and never touch the cache."""
    from mlx2.runtime.models import xing4_0_mhc_metal

    class Unevaluable:
        pass

    monkeypatch.setattr(xing4_0, "_kernel_usable", lambda array: xing4_0._MHC_KERNEL)
    monkeypatch.setattr(xing4_0_mhc_metal, "mhc_pre", lambda *a, **k: (Unevaluable(),) * 3)
    xing4_0.set_mhc_kernel(True)
    xing4_0.mhc_stats(reset=True)
    tokens = mx.array([[1, 2, 3, 4]])
    cache = model.make_cache()
    got = model(tokens, cache=cache)
    stats = xing4_0.mhc_stats()
    assert stats["kernel_fallbacks"] == 1 and not xing4_0.mhc_kernel_enabled()
    assert all(c.offset == 4 for c in cache)  # appended exactly once
    xing4_0.set_mhc_kernel(False)
    want = model(tokens, cache=model.make_cache())
    assert mx.allclose(got, want, rtol=1e-5, atol=1e-5).item()


def test_kernel_qualification_is_per_specialization(monkeypatch):
    evaluated = []
    real_eval = mx.eval
    monkeypatch.setattr(xing4_0.mx, "eval", lambda *a: (evaluated.append(1), real_eval(*a))[1])
    xing4_0.set_mhc_kernel(True)
    for key in (("k", 64, mx.float32), ("k", 64, mx.float32), ("k", 64, mx.bfloat16)):
        assert xing4_0._run_kernel(key, lambda: mx.zeros((2,))) is not None
    assert len(evaluated) == 2  # the repeated key is trusted lazily
    assert ("k", 64, mx.bfloat16) in xing4_0._MHC_QUALIFIED
    xing4_0.set_mhc_kernel(True)  # toggling clears qualification state
    assert not xing4_0._MHC_QUALIFIED
    xing4_0.set_mhc_kernel(False)
