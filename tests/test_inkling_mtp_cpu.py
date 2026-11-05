"""CPU contract checks for the Inkling native MTP head and cascaded drafter.

A tiny random Inkling target plus a tiny random MTP sidecar written in the
real q8 layout (``mx.quantize`` affine 8-bit, group 64, source tensor names,
interleaved ``w13``, PyTorch ``[C, 1, K]`` short-convolution weights).  These
establish wiring, alignment and cache bookkeeping on CPU arithmetic; they make
no acceptance, speed or qualification claim about the real checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import json

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.drafters.inkling_mtp import (
    CascadedMTPDraftCache,
    CascadedMTPDraftModel,
    inspect_cascaded_mtp_record,
)
from mlx2.runtime.models.inkling import Model, ModelArgs
from mlx2.runtime.models.inkling.mtp import build_mtp

PIN = "a" * 64
WIDTH, INTER, HEADS, KV, HEAD_DIM, D_REL = 64, 128, 2, 1, 32, 2
WINDOW, EXTENT, FLOOR, ALPHA, MUP = 4, 8, 3, 0.1, 4.0
DEPTHS, LOCAL = 3, [0, 2]
EPS = 1e-6


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def target_args():
    return ModelArgs(
        hidden_size=WIDTH,
        num_hidden_layers=2,
        vocab_size=48,
        unpadded_vocab_size=40,
        num_attention_heads=HEADS,
        num_key_value_heads=KV,
        head_dim=HEAD_DIM,
        swa_num_attention_heads=HEADS,
        swa_num_key_value_heads=KV,
        swa_head_dim=HEAD_DIM,
        moe_intermediate_size=64,
        intermediate_size=INTER,
        dense_intermediate_size=INTER,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=2,
        d_rel=D_REL,
        rel_extent=EXTENT,
        sliding_window_size=WINDOW,
        local_layer_ids=[0],
        dense_mlp_idx=1,
        log_scaling_n_floor=FLOOR,
        log_scaling_alpha=ALPHA,
        logits_mup_width_multiplier=MUP,
    )


def target():
    mx.random.seed(7)
    model = Model(target_args())
    for layer in model.layers:
        layer.self_attn.rel_proj = mx.random.normal(layer.self_attn.rel_proj.shape) * 0.1
    model.model.embed_norm.weight = 1 + 0.2 * mx.random.normal((WIDTH,))
    model.model.norm.weight = 1 + 0.2 * mx.random.normal((WIDTH,))
    model.eval()
    mx.eval(model.parameters())
    return model


def sidecar_weights(depths=DEPTHS, seed=3):
    """Source-named sidecar tensors: q8 projections, plain everything else."""
    mx.random.seed(seed)
    weights = {}
    for depth in range(depths):
        base = f"model.mtp.layers.{depth}."

        def quantized(name, shape, scale=0.08, base=base):
            q, s, b = mx.quantize(mx.random.normal(shape) * scale, group_size=64, bits=8)
            weights[base + name + ".weight"] = q
            weights[base + name + ".scales"] = s
            weights[base + name + ".biases"] = b

        quantized("input_proj", (WIDTH, 2 * WIDTH))
        quantized("transformer_block.attn.wq_du", (HEADS * HEAD_DIM, WIDTH))
        quantized("transformer_block.attn.wk_dv", (KV * HEAD_DIM, WIDTH))
        quantized("transformer_block.attn.wv_dv", (KV * HEAD_DIM, WIDTH))
        quantized("transformer_block.attn.wr_du", (HEADS * D_REL, WIDTH), 0.3)
        quantized("transformer_block.attn.wo_ud", (WIDTH, HEADS * HEAD_DIM))
        quantized("transformer_block.mlp.w13_dn", (2 * INTER, WIDTH))
        quantized("transformer_block.mlp.w2_md", (WIDTH, INTER))
        for name in (
            "hidden_norm.weight",
            "embed_norm.weight",
            "transformer_block.attn_norm.weight",
            "transformer_block.mlp_norm.weight",
        ):
            weights[base + name] = 1 + 0.2 * mx.random.normal((WIDTH,))
        for name in ("q_norm", "k_norm"):
            weights[base + f"transformer_block.attn.{name}.weight"] = (
                1 + 0.2 * mx.random.normal((HEAD_DIM,))
            )
        for name, channels in (
            ("attn.k_sconv", KV * HEAD_DIM),
            ("attn.v_sconv", KV * HEAD_DIM),
            ("attn_sconv", WIDTH),
            ("mlp_sconv", WIDTH),
        ):
            weights[base + f"transformer_block.{name}.weight"] = 0.4 * mx.random.normal(
                (channels, 1, 4)
            )
        extent = WINDOW if depth in LOCAL else EXTENT
        weights[base + "transformer_block.attn.rel_logits_proj.proj"] = (
            0.5 * mx.random.normal((D_REL, extent))
        )
        weights[base + "transformer_block.mlp.global_scale"] = mx.array([1.75])
    mx.eval(weights)
    return weights


QUANT = {"bits": 8, "group_size": 64, "mode": "affine"}


def attached(depths=DEPTHS, num_draft=None):
    model = target()
    head = build_mtp(
        model.config, sidecar_weights(), depths=depths, local_layer_ids=LOCAL,
        quantization=QUANT,
    )
    mx.eval(head.parameters())
    model.attach_mtp(head)
    model.select_external_feature_source("mtp_seed")
    draft = CascadedMTPDraftModel(
        model, num_draft=num_draft or depths, revision=PIN
    )
    return model, draft


# -- independent reference ---------------------------------------------------


def _np(value):
    return np.asarray(value.astype(mx.float32))


def _rms(x, weight):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS) * weight


def _dq(weights, name):
    return _np(
        mx.dequantize(
            weights[name + ".weight"], weights[name + ".scales"],
            weights[name + ".biases"], group_size=64, bits=8,
        )
    )


def _sconv(x, weight):
    """PyTorch depthwise causal conv on source ``[C, 1, K]`` plus residual."""
    kernel = weight[:, 0, :]  # [C, K]
    size = kernel.shape[1]
    padded = np.concatenate([np.zeros((size - 1, x.shape[1]), x.dtype), x], axis=0)
    out = np.stack(
        [(padded[t : t + size] * kernel.T).sum(axis=0) for t in range(x.shape[0])]
    )
    return out + x


def _attention(x, w, base, sliding):
    q = x @ _dq(w, base + "wq_du").T
    k = _sconv(x @ _dq(w, base + "wk_dv").T, _np(w[base + "k_sconv.weight"]))
    v = _sconv(x @ _dq(w, base + "wv_dv").T, _np(w[base + "v_sconv.weight"]))
    r = (x @ _dq(w, base + "wr_du").T).reshape(-1, HEADS, D_REL)
    q = _rms(q.reshape(-1, HEADS, HEAD_DIM), _np(w[base + "q_norm.weight"]))
    k = _rms(k.reshape(-1, KV, HEAD_DIM), _np(w[base + "k_norm.weight"]))
    v = v.reshape(-1, KV, HEAD_DIM)
    proj = _np(w[base + "rel_logits_proj.proj"])
    extent = proj.shape[1]
    rows = x.shape[0]
    out = np.zeros((rows, HEADS, HEAD_DIM), np.float32)
    for h in range(HEADS):
        g = h // (HEADS // KV)
        for i in range(rows):
            tau = 1.0 if sliding else 1 + ALPHA * np.log(max((i + 1) / FLOOR, 1.0))
            scores = np.full(rows, -np.inf, np.float32)
            for j in range(i + 1):
                dist = i - j
                if sliding and dist >= WINDOW:
                    continue
                bias = r[i, h] @ proj[:, dist] if dist < extent else 0.0
                scores[j] = tau * (q[i, h] @ k[j, g] / HEAD_DIM + bias)
            p = np.exp(scores - scores.max())
            out[i, h] = (p / p.sum()) @ v[:, g]
    return out.reshape(rows, -1) @ _dq(w, base + "wo_ud").T


def reference_depth(model, weights, depth, hidden, tokens):
    """Unfused, dequantized per-depth MTP law over positions ``0..P-1``."""
    base = f"model.mtp.layers.{depth}."
    block = base + "transformer_block."
    table = _np(model.model.embed_tokens.weight)
    embedded = _rms(table[np.asarray(tokens)], _np(model.model.embed_norm.weight))
    embedded = _rms(embedded, _np(weights[base + "embed_norm.weight"]))
    x = np.concatenate(
        [_rms(hidden, _np(weights[base + "hidden_norm.weight"])), embedded], axis=-1
    ) @ _dq(weights, base + "input_proj").T
    r = _attention(
        _rms(x, _np(weights[block + "attn_norm.weight"])), weights,
        block + "attn.", depth in LOCAL,
    )
    h = x + _sconv(r, _np(weights[block + "attn_sconv.weight"]))
    m = _rms(h, _np(weights[block + "mlp_norm.weight"]))
    w13 = _dq(weights, block + "mlp.w13_dn")
    gate, up = m @ w13[0::2].T, m @ w13[1::2].T
    mlp = (gate / (1 + np.exp(-gate)) * up) @ _dq(weights, block + "mlp.w2_md").T
    mlp = mlp * _np(weights[block + "mlp.global_scale"])
    out = h + _sconv(mlp, _np(weights[block + "mlp_sconv.weight"]))
    logits = (out / MUP) @ _np(model.lm_head.weight).T
    return logits[:, :40], out


def _features(model, prompt):
    cache = model.make_cache()
    seeds = model.prefill_body(mx.array([prompt]), cache, (len(model.layers) - 1,))
    mx.eval(seeds)
    return seeds, cache


# -- (a) per-depth forward ---------------------------------------------------


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_depth_step_matches_independent_reference(depth):
    model, _ = attached()
    weights = sidecar_weights()
    rows = 11  # beyond the 4-row window and the 8-entry relative extent
    hidden = mx.random.normal((1, rows, WIDTH))
    tokens = [3, 9, 1, 30, 7, 7, 12, 5, 39, 2, 18]
    expected_logits, expected = reference_depth(model, weights, depth, _np(hidden[0]), tokens)
    cache = model.make_mtp_cache()[depth]
    logits, raws = [], []
    for begin, end in ((0, 5), (5, 6), (6, 11)):  # chunked: cache continuity
        lg, raw = model.mtp_depth_step(
            depth, hidden[:, begin:end], mx.array([tokens[begin:end]]), cache
        )
        logits.append(lg)
        raws.append(raw)
    np.testing.assert_allclose(_np(mx.concatenate(raws, 1)[0]), expected, atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(
        _np(mx.concatenate(logits, 1)[0]), expected_logits, atol=2e-4, rtol=2e-4
    )
    assert cache[0].offset == rows


def test_sidecar_layout_mapping_is_the_converted_target_law():
    """Interleaved w13 split and [C,1,K] -> [C,K,1] conv law, on packed rows."""
    model, _ = attached()
    weights = sidecar_weights()
    block = model.mtp.layers[1].block
    np.testing.assert_array_equal(
        np.asarray(block.mlp.gate_proj.weight),
        np.asarray(weights["model.mtp.layers.1.transformer_block.mlp.w13_dn.weight"][0::2]),
    )
    np.testing.assert_array_equal(
        np.asarray(block.mlp.up_proj.scales),
        np.asarray(weights["model.mtp.layers.1.transformer_block.mlp.w13_dn.scales"][1::2]),
    )
    conv = weights["model.mtp.layers.1.transformer_block.attn_sconv.weight"]
    assert block.attn_sconv.conv.weight.shape == (WIDTH, 4, 1)
    np.testing.assert_array_equal(
        np.asarray(block.attn_sconv.conv.weight[:, :, 0]), np.asarray(conv[:, 0, :])
    )
    assert [layer.sliding for layer in model.mtp.layers] == [True, False, True]
    assert model.mtp.layers[1].block.self_attn.log_floor == FLOOR
    assert model.mtp.layers[0].block.self_attn.log_floor is None
    assert type(model.mtp.layers[0].block.self_attn.qkvr_proj).__name__ == "QuantizedLinear"


# -- (b) staircase chain -----------------------------------------------------


def _staircase(model, weights, features, history, anchor, count):
    """Draft tokens by running every depth over the whole sequence."""
    known = list(history[1:]) + [anchor]  # x[1..H]
    hidden = _np(features[0])
    drafts = []
    for depth in range(count):
        tokens = (known + drafts)[depth : depth + len(history)]
        logits, hidden = reference_depth(model, weights, depth, hidden, tokens)
        drafts.append(int(np.argmax(logits[-1])))
    return drafts


@pytest.mark.parametrize("prompt_length", [1, 2, 9])
def test_chain_draft_equals_the_reference_staircase(prompt_length):
    model, draft = attached()
    prompt = [5, 17, 2, 33, 8, 21, 4, 11, 26][:prompt_length]
    anchor = 13
    features, _ = _features(model, prompt)
    pairing = [prompt[1:] + [anchor]]
    tokens, laws = draft.draft_distributions(
        [anchor], features, draft.make_cache(), 8, [None], [0.0],
        context_tokens=pairing, processor_histories=[prompt],
    )
    expected = _staircase(model, sidecar_weights(), features, prompt, anchor, DEPTHS)
    assert tokens == [expected]  # capped at the three loaded depths
    assert [law.token for law in laws[0]] == expected
    assert draft.stats["chain_steps"] == DEPTHS - 1


def test_lazy_tree_primary_branch_equals_the_synced_chain():
    model, draft = attached()
    prompt = [5, 17, 2, 33, 8, 21, 4]
    features, _ = _features(model, prompt)
    pairing = [prompt[1:] + [13]]
    chain, _ = draft.draft_distributions(
        [13], features, draft.make_cache(), 3, [None], [0.0], context_tokens=pairing
    )
    trees = draft._lazy_tree_paths([13], features, draft.make_cache(), (3, 2, 1), pairing)
    assert trees[0][0] == tuple(chain[0])
    assert len({path[0] for path in trees[0]}) == 3
    packed = draft.draft_packed_paths(
        [13], features, draft.make_cache(), (3, 2), context_tokens=pairing
    )
    assert packed[0][0].tokens == tuple(chain[0])
    assert [len(path.tokens) for path in packed[0]] == [3, 2]
    tree = draft.propose_tree([13], features, draft.make_cache(), 6, context_tokens=pairing)
    assert tree[0][0], "propose_tree produced no nodes"


# -- (c) cascaded commit -----------------------------------------------------


def _assert_same_state(left, right):
    assert left.offset == right.offset
    for depth in range(left.depths):
        (lkv, lconv), (rkv, rconv) = left.depth_cache(depth), right.depth_cache(depth)
        assert lkv.offset == rkv.offset == max(0, left.offset - depth)
        if lkv.offset:
            if hasattr(lkv, "_temporal_order"):
                keep = min(lkv.offset, WINDOW - 1)
                pairs = [
                    (lkv._temporal_order(a)[..., -keep:, :], rkv._temporal_order(b)[..., -keep:, :])
                    for a, b in ((lkv.keys, rkv.keys), (lkv.values, rkv.values))
                ]
            else:
                pairs = [
                    (lkv.keys[..., : lkv.offset, :], rkv.keys[..., : rkv.offset, :]),
                    (lkv.values[..., : lkv.offset, :], rkv.values[..., : rkv.offset, :]),
                ]
            for a, b in pairs:
                np.testing.assert_allclose(_np(a), _np(b), atol=1e-5, rtol=1e-5)
        for a, b in zip(lconv.cache, rconv.cache, strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_allclose(_np(a), _np(b), atol=1e-5, rtol=1e-5)
    for depth in range(left.depths):
        a, b = left.pending[depth], right.pending[depth]
        assert (a is None) == (b is None)
        if a is not None:
            np.testing.assert_allclose(_np(a), _np(b), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_cascaded_commit_equals_building_from_scratch(accepted):
    model, draft = attached()
    prompt = [5, 17, 2, 33, 8, 21, 4]
    anchor = 13
    features, target_cache = _features(model, prompt)
    live = draft.make_cache()
    drafted, _ = draft.draft_distributions(
        [anchor], features, live, 3, [None], [0.0],
        context_tokens=[prompt[1:] + [anchor]],
    )
    # Verify: the target consumes the anchor and the accepted drafts, then a
    # correction/bonus token becomes the new anchor.
    consumed = [anchor] + drafted[0][:accepted]
    bonus = 37 if drafted[0][accepted:accepted + 1] != [37] else 38
    verified = model.prefill_body(mx.array([consumed]), target_cache, (len(model.layers) - 1,))
    following = consumed[1:] + [bonus]
    second, _ = draft.draft_distributions(
        [bonus], verified, live, 3, [None], [0.0], context_tokens=[following]
    )
    history = prompt + consumed
    scratch = draft.make_cache()
    all_features = mx.concatenate([features, verified], axis=1)
    again, _ = draft.draft_distributions(
        [bonus], all_features, scratch, 3, [None], [0.0],
        context_tokens=[history[1:] + [bonus]],
    )
    mx.eval(live[0].state, scratch[0].state)
    _assert_same_state(live[0], scratch[0])
    assert second == again
    assert live[0].offset == len(history)


def test_append_context_cascade_matches_one_shot_commit():
    model, draft = attached()
    prompt = [5, 17, 2, 33, 8, 21, 4, 11, 26, 9]
    features, _ = _features(model, prompt)
    stepwise, whole = draft.make_cache(), draft.make_cache()
    for begin, end in ((0, 1), (1, 2), (2, 6), (6, 9)):
        draft.append_context(
            features[:, begin:end], stepwise, context_tokens=[prompt[begin + 1 : end + 1]]
        )
    draft.append_context(features[:, :9], whole, context_tokens=[prompt[1:10]])
    mx.eval(stepwise[0].state, whole[0].state)
    _assert_same_state(stepwise[0], whole[0])
    assert draft.stats["depth_commit_rows"] == [18, 16, 14]


def test_draft_cache_is_a_publishable_committed_boundary():
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors

    model, draft = attached()
    prompt = [5, 17, 2, 33]
    features, _ = _features(model, prompt)
    cache = draft.make_cache()
    draft.append_context(features, cache, context_tokens=[prompt[1:] + [6]])
    mx.eval(cache[0].state)
    snapshot, _tail, _ = snapshot_prompt_cache_descriptors(cache, None)
    assert isinstance(snapshot[0], CascadedMTPDraftCache)
    assert snapshot[0].offset == cache[0].offset == 4
    before = _np(snapshot[0].pending[0])
    draft.append_context(features[:, :1], cache, context_tokens=[[6]])
    np.testing.assert_array_equal(_np(snapshot[0].pending[0]), before)
    assert snapshot[0].depth_cache(0)[0].offset == 4


def test_num_draft_is_capped_at_loaded_depths_and_never_reused():
    model, _ = attached(depths=2, num_draft=2)
    with pytest.raises(ValueError, match="never reused"):
        CascadedMTPDraftModel(model, num_draft=3, revision=PIN)
    draft = CascadedMTPDraftModel(model, num_draft=2, revision=PIN)
    prompt = [5, 17, 2, 33]
    features, _ = _features(model, prompt)
    tokens, _ = draft.draft_distributions(
        [6], features, draft.make_cache(), 5, [None], [0.0],
        context_tokens=[prompt[1:] + [6]],
    )
    assert len(tokens[0]) == 2
    with pytest.raises(ValueError, match="mtp_depths"):
        inspect_cascaded_mtp_record(
            num_draft=3, depths=2, sidecar_sha256="0" * 64, target_fingerprint=PIN
        )


def test_depths_beyond_num_draft_are_loaded_but_never_committed():
    model, full = attached()
    short = CascadedMTPDraftModel(model, num_draft=2, revision=PIN)
    prompt = [5, 17, 2, 33, 8, 21]
    features, _ = _features(model, prompt)
    pairing = [prompt[1:] + [6]]
    lanes = [full.make_cache(), short.make_cache()]
    tokens = [
        draft.draft_distributions(
            [6], features, lane, 8, [None], [0.0], context_tokens=pairing
        )[0][0]
        for draft, lane in zip((full, short), lanes)
    ]
    assert tokens[1] == tokens[0][:2]
    assert short.stats["depth_commit_rows"] == [6, 5, 0]
    assert lanes[1][0].depth_cache(2)[0].offset == 0
    assert lanes[1][0].offset == lanes[0][0].offset == 6


def test_generic_single_head_drafter_refuses_the_cascaded_head():
    from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel

    model, _ = attached()
    with pytest.raises(ValueError, match="mtp_step"):
        NativeMTPDraftModel(model, num_draft=3, revision=PIN)


# -- (d) target unchanged ------------------------------------------------------


def test_target_forward_and_taps_unchanged_by_the_head():
    plain = target()
    model, _ = attached()
    prompt = mx.array([[5, 17, 2, 33, 8, 21, 4, 11]])
    expected = plain(prompt, cache=plain.make_cache())
    model.select_external_feature_source(None)
    actual = model(prompt, cache=model.make_cache())
    np.testing.assert_array_equal(_np(actual), _np(expected))
    last = (len(model.layers) - 1,)
    logits, taps = model.forward_with_taps(prompt, model.make_cache(), last)
    plain_logits, plain_taps = plain.forward_with_taps(prompt, plain.make_cache(), last)
    np.testing.assert_array_equal(_np(logits), _np(plain_logits))
    np.testing.assert_array_equal(_np(taps), _np(plain_taps))
    assert set(dict(model.parameters())) == set(dict(plain.parameters()))
    model.select_external_feature_source("mtp_seed")
    logits, seeds = model.forward_with_taps(prompt, model.make_cache(), last)
    np.testing.assert_array_equal(_np(logits), _np(expected))
    pre = model.model(prompt, model.make_cache(), skip_final_norm=True)
    np.testing.assert_array_equal(_np(seeds), _np(model.model.norm(pre)))
    with pytest.raises(ValueError, match="last layer"):
        model.forward_with_taps(prompt, model.make_cache(), (0,))


def test_feature_source_and_head_fail_closed_without_a_head():
    model = target()
    assert model.mtp is None and model.mtp_depth_count == 0
    with pytest.raises(ValueError, match="attached MTP head"):
        model.select_external_feature_source("mtp_seed")
    with pytest.raises(ValueError, match="no attached MTP head"):
        model.make_mtp_cache()
    with pytest.raises(TypeError):
        model.attach_mtp(object())


def test_packed_tree_rows_use_the_selected_mtp_seed_law():
    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor

    model, _ = attached()
    live = model.make_cache()
    model(mx.array([[1, 2]]), cache=live)
    expected = copy.deepcopy(live)
    last = (len(model.layers) - 1,)
    _logits, features, transaction = HybridTreeExecutor.create().forward(
        model, [3, 4], [-1, 0], live, last, cached=False
    )
    transaction.abort()
    _, serial = model.forward_with_taps(mx.array([[3, 4]]), expected, last)
    np.testing.assert_allclose(_np(features), _np(serial), atol=1e-5)


# -- (e) adapter sidecar gate -------------------------------------------------


def _sidecar_dir(tmp_path, monkeypatch, *, depths=DEPTHS):
    from mlx2.adapters import inkling as adapter

    root = tmp_path / "mtp"
    root.mkdir()
    weights = sidecar_weights(depths)
    mx.save_safetensors(str(root / "mtp.safetensors"), weights, metadata={"format": "mlx"})
    digest = hashlib.sha256((root / "mtp.safetensors").read_bytes()).hexdigest()
    mtp_config = {
        "chain_hidden_post_norm": False,
        "local_layer_ids": LOCAL,
        "num_nextn_predict_layers": depths,
    }
    modules = sorted(
        f"model.mtp.layers.{d}.{name}" for d in range(depths) for name in adapter._MTP_PROJECTIONS
    )
    (root / "config.json").write_text(json.dumps({
        "mtp_config": mtp_config,
        "quantization": {**QUANT, "modules": modules},
    }))
    (root / "conversion.json").write_text(json.dumps({
        "schema": adapter.MTP_SIDECAR_SCHEMA,
        "source": {
            "repository": adapter.MTP_SOURCE_REPOSITORY,
            "revision": adapter.MTP_SOURCE_REVISION,
            "sha256": adapter.MTP_SOURCE_SHA256,
        },
        "conversion": dict(QUANT),
        "output_sha256": digest,
        "source_revision": "f" * 40,
    }))
    monkeypatch.setattr(adapter, "MTP_SIDECAR_SHA256", digest)
    return root, {"mtp_config": mtp_config}, digest


def test_adapter_accepts_and_pins_a_matching_sidecar(tmp_path, monkeypatch):
    from mlx2.adapters.inkling import inspect_mtp_sidecar

    root, config, digest = _sidecar_dir(tmp_path, monkeypatch)
    record = inspect_mtp_sidecar(str(root), depths=2, target_config=config)
    assert record["sha256"] == digest and record["depths"] == 2
    assert record["qualification"] == "unqualified"


@pytest.mark.parametrize("defect", ["sha", "missing", "depth", "mtp_config", "tensor"])
def test_adapter_refuses_a_bad_sidecar(tmp_path, monkeypatch, defect):
    from mlx2.adapters import inkling as adapter

    root, config, _digest = _sidecar_dir(tmp_path, monkeypatch)
    depths = 2
    if defect == "sha":
        monkeypatch.setattr(adapter, "MTP_SIDECAR_SHA256", "0" * 64)
        conversion = json.loads((root / "conversion.json").read_text())
        conversion["output_sha256"] = "0" * 64
        (root / "conversion.json").write_text(json.dumps(conversion))
        match = "SHA-256"
    elif defect == "missing":
        (root / "mtp.safetensors").unlink()
        match = "missing"
    elif defect == "depth":
        depths, match = 4, "mtp_depths"
    elif defect == "mtp_config":
        config = {"mtp_config": {**config["mtp_config"], "local_layer_ids": [0]}}
        match = "mtp_config"
    else:
        weights = sidecar_weights()
        weights.pop("model.mtp.layers.2.hidden_norm.weight")
        mx.save_safetensors(str(root / "mtp.safetensors"), weights)
        match = "tensor roles"
    with pytest.raises(ValueError, match=match):
        adapter.inspect_mtp_sidecar(str(root), depths=depths, target_config=config)


def test_adapter_refuses_a_payload_changed_after_its_record(tmp_path, monkeypatch):
    from mlx2.adapters import inkling as adapter

    root, config, _ = _sidecar_dir(tmp_path, monkeypatch)
    weights = sidecar_weights(seed=99)
    mx.save_safetensors(str(root / "mtp.safetensors"), weights, metadata={"format": "mlx"})
    with pytest.raises(ValueError, match="SHA-256"):
        adapter.inspect_mtp_sidecar(str(root), depths=3, target_config=config)


@pytest.mark.parametrize(
    ("policy", "match"),
    [
        ({"draft_model": "native_mtp"}, "mtp_sidecar"),
        ({"draft_model": "native_mtp", "mtp_sidecar": "/x", "mtp_depths": 9}, "mtp_depths"),
        ({"draft_model": "native_mtp", "mtp_sidecar": "/x", "mtp_depths": 0}, "mtp_depths"),
        (
            {"draft_model": "native_mtp", "mtp_sidecar": "/x", "mtp_depths": 4, "num_draft": 5},
            "never reused",
        ),
        ({"draft_model": "pld", "mtp_sidecar": "/x"}, "require draft_model native_mtp"),
        ({"draft_model": "other"}, "prompt-lookup"),
    ],
)
def test_policy_refuses_malformed_native_mtp_requests(policy, match):
    from mlx2.adapters.inkling import validate_policy

    with pytest.raises(ValueError, match=match):
        validate_policy(policy)


def test_policy_keeps_prompt_lookup_and_defaults_native_mtp():
    from mlx2.adapters.inkling import validate_policy

    _, pld = validate_policy({"draft_model": "pld"})
    assert "mtp_sidecar" not in pld and pld["ast_topology"] == "chain"
    _, native = validate_policy({"draft_model": "native_mtp", "mtp_sidecar": "/x"})
    assert native["mtp_depths"] == 8 and native["num_draft"] == 5
    assert native["mtp_sidecar"] == "/x"


def _bound_adapter(root, config, tmp_path, depths, num_draft=2, **policy):
    """An InklingAdapter shell around the tiny target, bound as in serving."""
    from mlx2.adapters.inkling import (
        DESCRIPTOR,
        InklingAdapter,
        inspect_mtp_sidecar,
        validate_policy,
    )

    adapter = object.__new__(InklingAdapter)
    adapter.model = target()
    adapter.config = config
    adapter.identity = {"fingerprint": "f" * 64, "path": str(tmp_path)}
    adapter.layout = "inkling-kv-four-shortconv-v1"
    adapter.descriptor = DESCRIPTOR
    adapter.kernel_policy, external = validate_policy(
        {"draft_model": "native_mtp", "mtp_sidecar": str(root), "mtp_depths": depths,
         "num_draft": num_draft, **policy}
    )
    sidecar = inspect_mtp_sidecar(str(root), depths=depths, target_config=config)
    adapter._bind_native_mtp(external, sidecar)
    return adapter


def test_sidecar_identity_enters_the_route_and_apc_namespace(tmp_path, monkeypatch):
    root, config, digest = _sidecar_dir(tmp_path, monkeypatch)
    two = _bound_adapter(root, config, tmp_path, 2)
    three = _bound_adapter(root, config, tmp_path, 3)
    assert two.identity["mtp_sidecar"]["sha256"] == digest
    assert two.identity["draft_fingerprint"] != three.identity["draft_fingerprint"]
    assert two.layout != three.layout
    assert two.draft_model.receipt_settings["qualified"] is False
    assert two.draft_model.receipt_settings["sidecar"]["sha256"] == digest
    assert two.model.external_feature_source == "mtp_seed"
    assert "mtp_sidecar" not in two.external_policy
    assert two.diagnostics()["native_mtp"]["selected"] is True


def test_serving_route_with_cascaded_head_emits_ordinary_greedy(tmp_path, monkeypatch):
    from test_standard_xpress_serving_cpu import drain, reference

    root, config, _ = _sidecar_dir(tmp_path, monkeypatch)
    adapter = _bound_adapter(root, config, tmp_path, 3, num_draft=3)
    batch = adapter.create_external_batch(
        completion_batch_size=1, prefill_step_size=64, memory_headroom=lambda: 1 << 40
    )
    prompt = [5, 17, 2, 33, 8, 21, 4, 11, 26, 9, 14, 3]
    batch.insert([prompt], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])
    output, _final = drain(batch)
    assert output[0] == reference(adapter.model, prompt, 14)
    stats = adapter.draft_model.stats
    assert stats["draft_calls"] > 0 and stats["chain_steps"] > 0


def test_acceptance_probe_case_runs_on_the_tiny_fixture():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "scripts/probe_inkling_mtp_acceptance.py"
    spec = importlib.util.spec_from_file_location("probe_inkling_mtp_acceptance", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    model, draft = attached()
    prompt = [5, 17, 2, 33, 8, 21, 4, 11, 26, 9, 14]
    result = probe._probe_case(SimpleNamespace(model=model), draft, prompt, 8, 4, set())
    assert result["generated_tokens"] == 8 and result["steps"] == 8
    assert [row["positions"] for row in result["teacher_forced"]] == [8, 8, 8]
    assert result["chain"]["rounds_scored"] == 8 - DEPTHS + 1
    # Ordinary greedy decode reproduces the probe's generated tokens.
    from test_standard_xpress_serving_cpu import reference

    assert result["generated_head"] == reference(model, prompt, 9)


def test_lazy_chain_equals_the_synced_ranked_chain_and_its_stash():
    prompt = [5, 17, 2, 33, 8, 21, 4, 11]
    results = {}
    for lazy in (True, False):
        model, draft = attached()
        features, _ = _features(model, prompt)
        processors = None if lazy else [[lambda tokens, logits: logits]]
        history = list(prompt)
        tokens, laws = draft.draft_distributions(
            [13], features, draft.make_cache(), 3, [None], [0.0],
            context_tokens=[prompt[1:] + [13]], logits_processors=processors,
            processor_histories=[history],
        )
        results[lazy] = (tokens, [law.token for law in laws[0]],
                         [entry[:2] for entry in draft._stash.values()],
                         draft.stats.get("lazy_chain_calls", 0))
    assert results[True][:3] == results[False][:3]
    assert results[True][3] == 1 and results[False][3] == 0


@pytest.mark.parametrize("pattern", [(3,), (3, 2), (2, 2, 1)])
def test_lazy_packed_paths_equal_the_synced_packed_paths(pattern):
    prompt = [5, 17, 2, 33, 8, 21, 4, 11]
    results = {}
    for lazy in (True, False):
        model, draft = attached()
        features, _ = _features(model, prompt)
        processors = None if lazy else [[lambda tokens, logits: logits]]
        paths = draft.draft_packed_paths(
            [13], features, draft.make_cache(), pattern,
            context_tokens=[prompt[1:] + [13]], logits_processors=processors,
            processor_histories=[list(prompt)],
        )
        results[lazy] = (
            [tuple(vars(p).values()) if hasattr(p, "__dict__") else tuple(p) for p in paths[0]],
            [entry[:2] for entry in draft._stash.values()],
            draft.stats.get("lazy_chain_calls", 0),
        )
    assert results[True][:2] == results[False][:2]
    assert results[True][2] == 1 and results[False][2] == 0


def test_quantized_draft_head_changes_only_draft_logits():
    from mlx2.adapters.inkling import validate_policy

    _, native = validate_policy({"draft_model": "native_mtp", "mtp_sidecar": "/x",
                                 "mtp_draft_head": "q4"})
    assert native["mtp_draft_head"] == "q4"
    assert validate_policy({"draft_model": "native_mtp", "mtp_sidecar": "/x"})[1]["mtp_draft_head"] == "bf16"
    with pytest.raises(ValueError):
        validate_policy({"draft_model": "native_mtp", "mtp_sidecar": "/x", "mtp_draft_head": "q3"})
    with pytest.raises(ValueError):
        validate_policy({"draft_model": "pld", "mtp_draft_head": "q4"})
    if getattr(target().config, "tie_word_embeddings", False):
        pytest.skip("tied embeddings keep the target head")
    model, draft = attached()
    prompt = [5, 17, 2, 33, 8, 21, 4]
    features, _ = _features(model, prompt)
    raw = mx.random.normal(features.shape).astype(features.dtype)
    before_target = model._logits_from_norm(model.model.norm(raw))
    before_draft = model.mtp_logits(raw)
    model.attach_mtp_draft_head((4, 64) if raw.shape[-1] % 64 == 0 else (4, 32))
    after_target = model._logits_from_norm(model.model.norm(raw))
    after_draft = model.mtp_logits(raw)
    mx.eval(before_target, after_target, before_draft, after_draft)
    assert bool(mx.array_equal(before_target, after_target).item())
    assert after_draft.shape == before_draft.shape
    assert not bool(mx.array_equal(before_draft, after_draft).item())
    tokens, _ = draft.draft_distributions(
        [13], features, draft.make_cache(), 3, [None], [0.0],
        context_tokens=[prompt[1:] + [13]],
    )
    assert len(tokens[0]) == 3


def _tied_first_logits(draft):
    """First logits whose raw argmax (token 1) rounds to a log-probability
    tie with token 0, which the host ranking orders first."""
    vocab = draft.config.vocab_size
    values = mx.full((vocab,), -30.0, mx.float32)
    values = mx.where(mx.arange(vocab) == 1, mx.array(1e-8, mx.float32), values)
    values = mx.where(mx.arange(vocab) == 0, mx.array(0.0, mx.float32), values)
    return values[None]


@pytest.mark.parametrize("packed", [False, True])
def test_lazy_drafting_never_reports_a_token_it_did_not_consume(packed):
    prompt = [5, 17, 2, 33, 8, 21, 4, 11]
    results = {}
    for lazy in (True, False):
        model, draft = attached()
        draft._first_logits = lambda state, draft=draft: _tied_first_logits(draft)
        features, _ = _features(model, prompt)
        processors = None if lazy else [[lambda tokens, logits: logits]]
        pairing = [prompt[1:] + [13]]
        if packed:
            paths = draft.draft_packed_paths(
                [13], features, draft.make_cache(), (3,), context_tokens=pairing,
                logits_processors=processors,
            )
            results[lazy] = paths[0][0].tokens
        else:
            tokens, _ = draft.draft_distributions(
                [13], features, draft.make_cache(), 3, [None], [0.0],
                context_tokens=pairing, logits_processors=processors,
            )
            results[lazy] = tuple(tokens[0])
            if lazy:
                assert draft.stats["lazy_chain_fallbacks"] == 1
    assert results[True][0] == 0
    assert results[True] == results[False]


def test_draft_revision_binds_num_draft_and_draft_head():
    def revision(**overrides):
        args = {"num_draft": 3, "depths": 3, "sidecar_sha256": "0" * 64,
                "target_fingerprint": PIN, **overrides}
        return inspect_cascaded_mtp_record(**args)["draft_revision"]

    base = revision()
    assert revision(num_draft=1) != base
    assert revision(draft_head="q4") != revision(draft_head="q8") != base
    assert revision(draft_head="bf16") == base


def test_cascaded_cache_round_trips_through_the_prompt_cache_file(tmp_path):
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    model, draft = attached()
    prompt = [5, 17, 2, 33, 8, 21, 4, 11]
    features, _ = _features(model, prompt)
    pairing = [prompt[1:] + [13]]
    lane = draft.make_cache()
    draft.draft_distributions(
        [13], features, lane, 2, [None], [0.0], context_tokens=pairing
    )
    mx.eval(lane[0].state)
    path = tmp_path / "draft.safetensors"
    save_prompt_cache(str(path), lane)
    restored = load_prompt_cache(str(path))
    assert type(restored[0]) is type(lane[0])
    _assert_same_state(lane[0], restored[0])
    assert restored[0].nbytes == lane[0].nbytes
    # Both lanes continue identically from the restored boundary.
    more = [13, 7, 9]
    seeds, _ = _features(model, prompt + more)
    tail = seeds[:, len(prompt):]
    outputs = [
        draft.draft_distributions(
            [9], tail, cache, 2, [None], [0.0], context_tokens=[more[1:] + [9]]
        )[0]
        for cache in (lane, restored)
    ]
    assert outputs[0] == outputs[1]


def test_started_tree_equals_propose_tree_and_leaves_the_live_plane():
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors

    model, draft = attached()
    prompt = [5, 17, 2, 33, 8, 21, 4]
    features, _ = _features(model, prompt)
    pairing = [prompt[1:] + [13]]
    expected = draft.propose_tree(
        [13], features, draft.make_cache(), 6, context_tokens=pairing,
        forbidden_token_ids=[(7,)],
    )
    live = draft.make_cache()
    copy_, _, _ = snapshot_prompt_cache_descriptors(live, None)
    state = draft.start_tree(
        [13], features, draft.batch_caches([copy_]), 6, context_tokens=pairing,
        forbidden_token_ids=[(7,)],
    )
    assert state["pending"] and live[0].offset == 0
    assert all(row is None for row in live[0].pending)
    assert draft.finish_tree(state) == expected
    assert copy_[0].offset == len(pairing[0])


def test_started_tree_refuses_processors_and_an_empty_seed():
    model, draft = attached()
    with pytest.raises(ValueError, match="processors"):
        draft.start_tree([1], mx.zeros((1, 1, WIDTH)), draft.make_cache(), 4,
                         logits_processors=[[object()]])
    with pytest.raises(ValueError, match="seed"):
        draft.start_tree([1], mx.zeros((1, 0, WIDTH)), draft.make_cache(), 4)


@pytest.mark.parametrize("pipeline", [True, False, None])
def test_served_route_adopts_prelaunched_mtp_trees(tmp_path, monkeypatch, pipeline):
    from test_standard_xpress_serving_cpu import drain, reference

    root, config, _ = _sidecar_dir(tmp_path, monkeypatch)
    policy = {} if pipeline is None else {"mtp_tree_pipeline": pipeline}
    adapter = _bound_adapter(root, config, tmp_path, 3, num_draft=3, **policy)
    assert adapter.draft_model.receipt_settings["tree_pipeline"] is bool(pipeline)
    batch = adapter.create_external_batch(
        completion_batch_size=1, prefill_step_size=64, memory_headroom=lambda: 1 << 40
    )
    prompt = [5, 17, 2, 33, 8, 21, 4, 11, 26, 9, 14, 3]
    batch.insert([prompt], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])
    output, _final = drain(batch)
    assert output[0] == reference(adapter.model, prompt, 14)
    stats = batch.scheduler_stats
    if pipeline:
        assert stats["external_tree_early_prelaunches"] > 0
        assert stats["external_tree_pipelined_drafts"] > 0
        assert adapter.draft_model.stats["pipelined_tree_starts"] > 0
    else:
        assert stats.get("external_tree_pipelined_drafts", 0) == 0
        assert "pipelined_tree_starts" not in adapter.draft_model.stats


def test_tree_pipeline_policy_is_a_boolean():
    from mlx2.adapters.inkling import validate_policy

    with pytest.raises(ValueError, match="mtp_tree_pipeline"):
        validate_policy({"draft_model": "native_mtp", "mtp_sidecar": "/x", "mtp_tree_pipeline": 1})
    with pytest.raises(ValueError, match="mtp_tree_pipeline"):
        validate_policy({"mtp_tree_pipeline": True})


def _buffer_address(value):
    """CPU buffer identity: a donated in-place write keeps the address."""
    mx.eval(value)
    view = np.asarray(value)
    try:
        return view.ctypes.data
    finally:
        del view


def test_chain_forks_depth_caches_as_descriptors_and_lends_the_global_kv(monkeypatch):
    """The chain's private depth copies alias the committed planes.

    The deep copy duplicated a global depth's whole KV every round. The
    descriptor fork must draft the same tokens, leave the same committed
    state after the next commit, and never pin the live buffer: the global
    depth's K/V keep their address through the chain and the following
    in-place commit.
    """
    prompt = [5, 17, 2, 33, 8, 21, 4, 11]
    anchor, bonus = 13, 37
    results = {}
    for mode in ("descriptor", "deepcopy"):
        model, draft = attached()
        if mode == "deepcopy":
            monkeypatch.setattr(
                CascadedMTPDraftModel,
                "_private_depth_cache",
                lambda self, state, depth: copy.deepcopy(state.depth_cache(depth)),
            )
        else:
            monkeypatch.undo()
        features, target_cache = _features(model, prompt)
        live = draft.make_cache()
        # Commit the prompt seeds first so the chain forks a populated,
        # unshared global KV buffer, as a decoding lane's is.
        draft.append_context(features, live, context_tokens=[prompt[1:] + [anchor]])
        mx.eval(live[0].state)
        depth_one = live[0].depth_cache(1)[0]
        assert not hasattr(depth_one, "_temporal_order"), "depth 1 must be global here"
        before = (_buffer_address(depth_one.keys), _buffer_address(depth_one.values))
        tokens, laws = draft.draft_distributions(
            [anchor], features[:, -1:], live, 3, [None], [0.0],
            context_tokens=[[anchor]],
        )
        mx.eval(live[0].state)
        after_chain = (_buffer_address(depth_one.keys), _buffer_address(depth_one.values))
        verified = model.prefill_body(
            mx.array([[anchor] + tokens[0][:1]]), target_cache, (len(model.layers) - 1,)
        )
        second, _ = draft.draft_distributions(
            [bonus], verified, live, 3, [None], [0.0],
            context_tokens=[tokens[0][:1] + [bonus]],
        )
        mx.eval(live[0].state)
        after_commit = (_buffer_address(depth_one.keys), _buffer_address(depth_one.values))
        results[mode] = (tokens, [law.token for law in laws[0]], second, live[0])
        if mode == "descriptor":
            assert draft.stats["chain_deepcopy_forks"] == 0
            assert draft.stats["chain_descriptor_forks"] == 2 * draft.stats["chain_steps"] > 0
            assert draft.stats["chain_kv_lends"] >= 1
            # Neither the lent chain append nor the commit after it copied
            # the global depth's buffers.
            assert after_chain == before
            assert after_commit == before
    assert results["descriptor"][:3] == results["deepcopy"][:3]
    _assert_same_state(results["descriptor"][3], results["deepcopy"][3])
