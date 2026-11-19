"""Hy4-flash image input: CPU tests on a tiny random vision + text checkpoint.

The oracles are independent float64 NumPy transcriptions of the reference
(transformers PR 50903 ``modeling_hy_v4_flash.py``: ``HYV4FlashVision*``,
``get_vision_position_ids``/``get_rope_index``, the Engram masking of
``build_engram_inputs``, the interleaved M-RoPE recomposition, and
``image_processing_hy_v4_flash.py``'s ``smart_resize``/``patchify``), run on
checkpoint-named ``vit.*`` weights in their convolution layouts.  No
reference code is executed.
"""

import base64
import io
import itertools
import json
import math

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_vision as V  # noqa: E402
from test_hy_v4_flash_cpu import (  # noqa: E402
    Oracle, build, checkpoint, rms, sigmoid, tiny_config,
)

IMG_START, IMG_END, IMG, VID = 90, 91, 92, 96
SECTION = [2, 1, 1]
VISION = dict(model_type="hy_v4_flash", hidden_size=16, intermediate_size=24, num_attention_heads=2,
              num_key_value_heads=2, num_hidden_layers=2, num_channels=3, patch_size=4,
              spatial_merge_size=2, out_hidden_size=32, max_image_size=32, min_image_size=16,
              rms_norm_eps=1e-5, hidden_act="gelu", interpolate_mode="bilinear",
              projector_type="rmsnorm_mlp_adapter", perceive_pre_norm=True,
              perceive_post_norm=False, cat_extra_token=0, temporal_patch_size=1,
              spatial_patch_size=1)


def vision_config():
    rope = {kind: {"rope_type": "default", "rope_theta": theta, "mrope_section": SECTION,
                   "mrope_interleaved": True}
            for kind, theta in (("deepseek_sparse_attention", 1e4), ("sliding_attention", 100.0))}
    config = tiny_config(qk_rope_head_dim=8, swa_qk_rope_head_dim=8, rope_parameters=rope,
                         compressed_indexer_head_dim=12)
    config.update(vision_config=dict(VISION), image_token_id=IMG, image_start_token_id=IMG_START,
                  image_end_token_id=IMG_END, image_newline_token_id=93, video_start_token_id=94,
                  video_end_token_id=95, video_token_id=VID, vision_start_token_id=IMG_START)
    return config


def vit_checkpoint(seed=1, s=0.3):
    v = VISION
    rng = np.random.default_rng(seed)
    W = {}
    D, I, p, C, m = v["hidden_size"], v["intermediate_size"], v["patch_size"], v["num_channels"], 2
    E = v["max_image_size"] // p

    def r(name, *shape, scale=s):
        W[name] = (rng.standard_normal(shape) * scale).astype(np.float32)

    def ones(name, n):
        W[name] = (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    r("vit.embeddings.patch_embedding.weight", D, C, p, p)
    r("vit.embeddings.patch_embedding.bias", D)
    r("vit.embeddings.position_embedding.weight", E * E + 1, D, scale=1.0)
    for i in range(v["num_hidden_layers"]):
        q = f"vit.layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            ones(q + norm + ".weight", D)
            r(q + norm + ".bias", D, scale=0.1)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            r(q + f"self_attn.{proj}.weight", D, D)
            r(q + f"self_attn.{proj}.bias", D, scale=0.1)
        r(q + "mlp.dense_h_to_4h.weight", I, D)
        r(q + "mlp.dense_h_to_4h.bias", I, scale=0.1)
        r(q + "mlp.dense_4h_to_h.weight", D, I)
        r(q + "mlp.dense_4h_to_h.bias", D, scale=0.1)
    ones("vit.perceive.norm.weight", D)
    r("vit.perceive.proj.0.weight", D * m * m, D, m, m)
    r("vit.perceive.proj.0.bias", D * m * m, scale=0.1)
    r("vit.perceive.proj.2.weight", v["out_hidden_size"], D * m * m, 1, 1)
    r("vit.perceive.proj.2.bias", v["out_hidden_size"], scale=0.1)
    r("vit.perceive.image_newline", v["out_hidden_size"], scale=1.0)
    return W


# ------------------------------------------------------- vision oracle --

_erf = np.vectorize(math.erf)


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def layer_norm(x, w, b, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def conv2d(x, w, b, stride):
    """NCHW convolution, no padding: out[n,o,i,j] = sum x[n,c,i*s+a,j*s+b] w[o,c,a,b] + b."""
    N, Ci, Hh, Ww = x.shape
    O, _, k, _ = w.shape
    Ho, Wo = (Hh - k) // stride + 1, (Ww - k) // stride + 1
    out = np.zeros((N, O, Ho, Wo))
    for a in range(k):
        for c in range(k):
            patch = x[:, :, a : a + stride * Ho : stride, c : c + stride * Wo : stride]
            out += np.einsum("nchw,oc->nohw", patch, w[:, :, a, c])
    return out + b[None, :, None, None]


def interpolate_bilinear(img, scale_h, scale_w):
    """``F.interpolate(mode="bilinear", align_corners=False, scale_factor=...)``, [C, h, w]."""
    C, Hin, Win = img.shape
    Ho, Wo = math.floor(Hin * scale_h), math.floor(Win * scale_w)

    def taps(n_out, n_in, scale):
        out = []
        for i in range(n_out):
            src = max((i + 0.5) / scale - 0.5, 0.0)
            i0 = min(int(math.floor(src)), n_in - 1)
            i1 = i0 + (1 if i0 < n_in - 1 else 0)
            lam = min(max(src - i0, 0.0), 1.0)
            out.append((i0, i1, lam))
        return out

    res = np.zeros((C, Ho, Wo))
    for i, (y0, y1, ly) in enumerate(taps(Ho, Hin, scale_h)):
        for j, (x0, x1, lx) in enumerate(taps(Wo, Win, scale_w)):
            res[:, i, j] = ((1 - ly) * ((1 - lx) * img[:, y0, x0] + lx * img[:, y0, x1])
                            + ly * ((1 - lx) * img[:, y1, x0] + lx * img[:, y1, x1]))
    return res


def vision_oracle(W, pixel_values, grids):
    v = VISION
    W = {k: x.astype(np.float64) for k, x in W.items() if k.startswith("vit.")}
    D, p, C, m = v["hidden_size"], v["patch_size"], v["num_channels"], v["spatial_merge_size"]
    E, heads, eps = v["max_image_size"] // p, v["num_attention_heads"], v["rms_norm_eps"]
    px = np.asarray(pixel_values, np.float64).reshape(-1, C, p, p)
    emb = conv2d(px, W["vit.embeddings.patch_embedding.weight"], W["vit.embeddings.patch_embedding.bias"], p)
    x = emb.reshape(-1, D)
    table = W["vit.embeddings.position_embedding.weight"][1:].reshape(1, E, E, D).transpose(0, 3, 1, 2)[0]
    pos = []
    for t, h, w in grids:
        grid = interpolate_bilinear(table, (h + 0.1) / E, (w + 0.1) / E)  # [D, h, w]
        pos.append(np.tile(grid.transpose(1, 2, 0).reshape(-1, D), (t, 1)))
    x = x + np.concatenate(pos)
    lengths = [h * w for t, h, w in grids for _ in range(t)]
    bounds = np.cumsum([0] + lengths)
    hd = D // heads
    for i in range(v["num_hidden_layers"]):
        q = f"vit.layers.{i}."
        y = layer_norm(x, W[q + "input_layernorm.weight"], W[q + "input_layernorm.bias"], eps)
        Q, K, Vv = (y @ W[q + f"self_attn.{n}_proj.weight"].T + W[q + f"self_attn.{n}_proj.bias"]
                    for n in ("q", "k", "v"))
        att = np.zeros_like(Q)
        for a, b in zip(bounds[:-1], bounds[1:]):
            qs, ks, vs = (z[a:b].reshape(b - a, heads, hd) for z in (Q, K, Vv))
            sc = np.einsum("ihd,jhd->hij", qs, ks) * hd ** -0.5
            pr = np.exp(sc - sc.max(-1, keepdims=True))
            pr /= pr.sum(-1, keepdims=True)
            att[a:b] = np.einsum("hij,jhd->ihd", pr, vs).reshape(b - a, D)
        x = x + att @ W[q + "self_attn.o_proj.weight"].T + W[q + "self_attn.o_proj.bias"]
        y = layer_norm(x, W[q + "post_attention_layernorm.weight"], W[q + "post_attention_layernorm.bias"], eps)
        y = gelu(y @ W[q + "mlp.dense_h_to_4h.weight"].T + W[q + "mlp.dense_h_to_4h.bias"])
        x = x + y @ W[q + "mlp.dense_4h_to_h.weight"].T + W[q + "mlp.dense_4h_to_h.bias"]
    out, cursor = [], 0
    for t, h, w in grids:
        item = x[cursor : cursor + t * h * w]
        cursor += t * h * w
        item = rms(item, W["vit.perceive.norm.weight"], eps)
        item = item.reshape(-1, h, w, D).transpose(0, 3, 1, 2)
        item = conv2d(item, W["vit.perceive.proj.0.weight"], W["vit.perceive.proj.0.bias"], m)
        item = conv2d(gelu(item), W["vit.perceive.proj.2.weight"], W["vit.perceive.proj.2.bias"], 1)
        frames, c, hh, _ = item.shape
        newline = np.broadcast_to(W["vit.perceive.image_newline"].reshape(1, c, 1, 1), (frames, c, hh, 1))
        item = np.concatenate([item, newline], -1).reshape(frames, c, -1).transpose(0, 2, 1)
        out.append(item.reshape(-1, c))
    return np.concatenate(out)


# --------------------------------------------------- position oracle --

def rope_index_ref(input_ids, grids, merge):
    """``get_rope_index`` for one row (image tokens type 1, text 0)."""
    types = [1 if t == IMG else 0 for t in input_ids]
    grid_iter = iter(grids)
    pieces, current = [], 0
    for key, group in itertools.groupby(enumerate(types), lambda x: x[1]):
        group = list(group)
        n = group[-1][0] + 1 - group[0][0]
        if key == 0:
            pieces.append(np.broadcast_to(np.arange(n) + current, (3, n)))
            current += n
        else:
            t, h, w = next(grid_iter)
            t, h, w = t, h // merge, w // merge
            pt = np.arange(t) * max(h, w + 1)
            T, Hh, Ww = np.meshgrid(pt, np.arange(h), np.arange(w + 1), indexing="ij")
            ids = np.stack([T, Hh + T, Ww + T]).reshape(3, -1) + current
            pieces.append(ids)
            current = ids.max() + 1
    pos = np.concatenate(pieces, 1)
    return pos, int(pos.max() + 1 - len(input_ids))


def mrope_ref(x, pos3, theta, section):
    """Reference rotary: per-plane freqs, ``recomposition_frequencies``, interleaved apply."""
    d = x.shape[-1]
    inv = 1.0 / theta ** (np.arange(0, d, 2) / d)
    freq = np.asarray(pos3, np.float64)[..., None] * inv  # [3, L, d/2]
    thw = freq[0].copy()
    for dim, offset in enumerate((1, 2), start=1):
        length = section[dim] * 3
        thw[..., offset:length:3] = freq[dim][..., offset:length:3]
    cos, sin = np.repeat(np.cos(thw), 2, -1), np.repeat(np.sin(thw), 2, -1)
    rot = np.stack([-x[..., 1::2], x[..., 0::2]], axis=-1).reshape(x.shape)
    shape = (pos3.shape[1],) + (1,) * (x.ndim - 2) + (d,)
    return x * cos.reshape(shape) + rot * sin.reshape(shape)


class VisionOracle(Oracle):
    """Text oracle with 3D positions, scattered image features and Engram masking."""

    def __init__(self, config, W, pos3, embeds):
        super().__init__(config, W)
        self.pos3 = np.asarray(pos3)
        self.embeds = embeds

    def indexer_mask(self, x, q_resid, p, L):
        t, W = self.t, self.W
        c = p + "self_attn.compressed_indexer."
        ratio, d, h = t["compressed_indexer_compress_ratio"], t["compressed_indexer_head_dim"], t["compressed_indexer_num_attention_heads"]
        rd = t["qk_rope_head_dim"]
        th = self.theta("deepseek_sparse_attention")
        tk = rms(x @ W[c + "token_key_projection.weight"].T, W[c + "token_key_norm.weight"])
        G = L // ratio
        grouped = tk[: G * ratio].reshape(G, ratio, d)
        values = grouped @ W[c + "key_value_projection.weight"].T
        gates = grouped @ W[c + "key_gate_projection.weight"].T + W[c + "intra_group_ape"]
        gates = np.exp(gates - gates.max(1, keepdims=True))
        gates /= gates.sum(1, keepdims=True)
        gk = rms((values * gates).sum(1), W[c + "key_norm.weight"])
        # A group carries the rotary phase of its first token.
        gpos = self.pos3[:, : G * ratio : ratio]
        gk = np.concatenate([gk[:, :-rd], mrope_ref(gk[:, -rd:], gpos, th, SECTION)], -1)
        q = (q_resid @ W[c + "query_projection.weight"].T).reshape(L, h, d)
        q = np.concatenate([q[..., :-rd], mrope_ref(q[..., -rd:], self.pos3, th, SECTION)], -1)
        hw = (x @ W[c + "head_weight_projection.weight"].T) * (h ** -0.5 * d ** -0.5)
        scores = np.maximum(np.einsum("lhd,gd->lhg", q, gk), 0)
        logits = np.einsum("lh,lhg->lg", hw, scores)
        own = np.arange(L) // ratio  # groups follow token slots
        logits = np.where(np.arange(G)[None] >= own[:, None], -np.inf, logits)
        k = min(t["compressed_indexer_group_topk"] - 1, G)
        recalled = np.argsort(-logits, axis=-1, kind="stable")[:, :k]
        recalled = np.concatenate([recalled, own[:, None]], -1)
        tokens = (recalled[..., None] * ratio + np.arange(ratio)).reshape(L, -1).clip(max=L - 1)
        keep = np.zeros((L, L), bool)
        np.put_along_axis(keep, tokens, True, axis=-1)
        return keep

    def attention(self, x, i, L):
        t, W = self.t, self.W
        p = f"model.layers.{i}."
        kind = t["layer_types"][i]
        sl = kind == "sliding_attention"
        Hh = t["swa_num_attention_heads"] if sl else t["num_attention_heads"]
        nope, rd, v, r = t["qk_nope_head_dim"], t["qk_rope_head_dim"], t["v_head_dim"], t["kv_lora_rank"]
        a = p + "self_attn."
        gate = (x @ W[a + "linear_gate.weight"].T).reshape(L, Hh, v)
        q_resid = rms(x @ W[a + "q_a_proj.weight"].T, W[a + "q_a_layernorm.weight"])
        q = (q_resid @ W[a + "q_b_proj.weight"].T).reshape(L, Hh, nope + rd)
        ckv = x @ W[a + "kv_a_proj_with_mqa.weight"].T
        k_pass = rms(ckv[:, :r], W[a + "kv_a_layernorm.weight"])
        th = self.theta(kind)
        q_rot = mrope_ref(q[..., nope:], self.pos3, th, SECTION)
        k_rot = mrope_ref(ckv[:, r:], self.pos3, th, SECTION)
        kv = (k_pass @ W[a + "kv_b_proj.weight"].T).reshape(L, Hh, nope + v)
        K = np.concatenate([kv[..., :nope], np.broadcast_to(k_rot[:, None], (L, Hh, rd))], -1)
        Q = np.concatenate([q[..., :nope], q_rot], -1)
        Vv = kv[..., nope:]
        s = np.einsum("ihd,jhd->hij", Q, K) * (nope + rd) ** -0.5
        # Causal and sliding masks follow cache slots, not positions.
        allowed = np.tril(np.ones((L, L), bool))
        if sl:
            allowed &= np.arange(L)[None] > np.arange(L)[:, None] - t["sliding_window"]
        elif L > t["compressed_indexer_group_topk"] * t["compressed_indexer_compress_ratio"]:
            allowed &= self.indexer_mask(x, q_resid, p, L)
        s = np.where(allowed[None], s, -np.inf)
        sink = W[a + "learnable_sink_param"][:, None, None]
        mx_ = np.maximum(s.max(-1, keepdims=True), sink)
        e = np.exp(s - mx_)
        probs = e / (e.sum(-1, keepdims=True) + np.exp(sink - mx_))
        out = np.einsum("hij,jhd->ihd", probs, Vv) * sigmoid(gate)
        return rms(out.reshape(L, -1) @ W[a + "o_proj.weight"].T, W[a + "post_proj_norm.weight"])

    def ngram_tokens(self, tokens):
        # ``build_ngram_tokens``: the leading (temporal) position plane is the
        # document offset.
        n, pad = self.t["engram_max_ngram_size"], self.t["engram_pad_id"]
        L = len(tokens)
        doc = self.pos3[0]
        cols = [tokens]
        for off in range(1, n):
            prev = np.concatenate([np.full(off, pad), tokens])[:L]
            cols.append(np.where(doc < off, pad, prev))
        return np.stack(cols, -1).astype(np.int64)

    def engram(self, streams, i, tokens):
        # ``engram_mask`` also excludes image/video positions.
        delta = super().engram(streams, i, tokens) - streams
        visual = np.isin(tokens, [IMG, VID])[:, None, None]
        return streams + np.where(visual, 0.0, delta)

    def hidden(self, tokens):
        t, W = self.t, self.W
        tokens = np.asarray(tokens)
        L, n = len(tokens), t["hc_mult"]
        streams = np.repeat(self.embeds[:, None], n, 1)
        for i in range(t["num_hidden_layers"]):
            p = f"model.layers.{i}."
            if i + 1 in t["engram_layer_ids"]:
                streams = self.engram(streams, i, tokens)
            post, x = self.hc(streams, p + "hc_attn_layer.hc_pre.")
            x = self.attention(rms(x, W[p + "input_layernorm.weight"]), i, L)
            streams = post[..., None] * x[:, None] + streams
            post, x = self.hc(streams, p + "hc_mlp_layer.hc_pre.")
            x = self.mlp(rms(x, W[p + "post_attention_layernorm.weight"]), i)
            streams = post[..., None] * x[:, None] + streams
        flat = streams.reshape(L, -1)
        mixes = (flat @ W["model.hc_head.hc_head_fn"].T) / np.sqrt(np.mean(flat ** 2, -1, keepdims=True) + t["rms_norm_eps"])
        pre = sigmoid(mixes * W["model.hc_head.hc_head_scale"] + W["model.hc_head.hc_head_base"]) + t["hc_eps"]
        return rms((pre[..., None] * streams).sum(1), W["model.norm.weight"])


# ------------------------------------------------------------ fixtures --

GRIDS = [(1, 4, 6), (1, 6, 4)]  # merged 2x3 -> 8 tokens, 3x2 -> 9 tokens


def image_prompt(grids=GRIDS, merge=2):
    """Text, framed image spans, text: long enough for the indexer and window."""
    prompt = [5, 7, 9, 11]
    texts = [[13, 4, 6, 8, 10], [12, 14, 16, 18]]
    for k, (t, h, w) in enumerate(grids):
        prompt += [IMG_START] + [IMG] * (t * (h // merge) * (w // merge + 1)) + [IMG_END]
        prompt += texts[k % 2]
    return prompt + [20, 22, 24]


def pixels_for(grids, seed=3):
    rng = np.random.default_rng(seed)
    n = sum(t * h * w for t, h, w in grids)
    return rng.standard_normal((n, VISION["num_channels"] * VISION["patch_size"] ** 2)).astype(np.float32)


@pytest.fixture(scope="module")
def vtiny():
    config = vision_config()
    W = checkpoint(config)
    VW = vit_checkpoint()
    model, _ = build(config, W)
    model.attach_vision(V.load_vision_tower(config, {k: mx.array(v) for k, v in VW.items()}))
    return config, W, VW, model


def oracle_logits(config, W, VW, tokens, grids, pixels):
    feats = vision_oracle(VW, pixels, grids)
    embeds = W["model.embed_tokens.weight"].astype(np.float64)[np.asarray(tokens)].copy()
    embeds[np.asarray(tokens) == IMG] = feats
    pos3, _ = rope_index_ref(tokens, grids, 2)
    return VisionOracle(config, W, pos3, embeds).hidden(tokens) @ W["lm_head.weight"].astype(np.float64).T


# --------------------------------------------------------------- tests --

def test_vision_tower_matches_numpy_oracle(vtiny):
    _, _, VW, model = vtiny
    # Two images and a two-frame grid (per-frame attention, frames merged apart).
    grids = [(1, 4, 6), (1, 6, 4), (2, 4, 4)]
    pixels = pixels_for(grids)
    got = np.array(model.vision_tower(mx.array(pixels), grids))
    ref = vision_oracle(VW, pixels, grids)
    assert got.shape == ref.shape == (8 + 9 + 2 * 6, 32)
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4)


def test_vision_position_grid_upsamples_like_interpolate(vtiny):
    _, _, VW, model = vtiny
    # A grid larger than the learned 8x8 table (12 > 8) resamples upward.
    got = np.array(model.vision_tower.position_grid(12, 6, mx.float32))
    E = 8
    table = VW["vit.embeddings.position_embedding.weight"][1:].astype(np.float64).reshape(E, E, -1).transpose(2, 0, 1)
    ref = interpolate_bilinear(table, (12 + 0.1) / E, (6 + 0.1) / E).transpose(1, 2, 0).reshape(72, -1)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("grids", [GRIDS, [(1, 2, 2)], [(2, 4, 6), (1, 2, 8)], [(3, 6, 2)]])
def test_3d_positions_match_get_rope_index(grids):
    tokens = image_prompt(grids)
    ref, delta = rope_index_ref(tokens, grids, 2)
    spans = V.find_spans(np.asarray(tokens), 0, IMG, grids, 2, ())
    got = V.slot_positions(spans, np.arange(len(tokens)))
    np.testing.assert_array_equal(got, ref)
    # Decode positions continue at slot + rope_delta.
    assert V.text_position(spans, len(tokens) + 5) == len(tokens) + 5 + delta
    assert V.chunk_positions(spans, len(tokens), 1) == len(tokens) + delta
    # Chunked recovery: spans found chunk by chunk give the same positions.
    cut = tokens.index(IMG_END) + 1
    first = V.find_spans(np.asarray(tokens[:cut]), 0, IMG, grids[:1], 2, ())
    rest = V.find_spans(np.asarray(tokens[cut:]), cut, IMG, grids[1:], 2, first)
    np.testing.assert_array_equal(V.slot_positions(first + rest, np.arange(len(tokens))), ref)


def test_mrope_with_equal_planes_is_fast_rope():
    x = mx.array(np.random.default_rng(0).standard_normal((1, 2, 7, 8)).astype(np.float32))
    pos = np.broadcast_to(np.arange(3, 10), (3, 7))
    got = H.mrope_rotate(x, pos, 1e4, SECTION)
    ref = mx.fast.rope(x, 8, traditional=True, base=1e4, scale=1.0, offset=3)
    np.testing.assert_allclose(np.array(got), np.array(ref), rtol=1e-5, atol=1e-5)


# Merged 2x3 / 3x2 spans, and a tall 4x1 span (frame block max(h, w + 1) = h).
@pytest.mark.parametrize("grids", [GRIDS, [(1, 8, 2), (1, 4, 6)]])
@pytest.mark.parametrize("step", [None, 7])
def test_text_image_prompt_matches_oracle_full_forward(vtiny, monkeypatch, grids, step):
    config, W, VW, model = vtiny
    if step is not None:  # the media chunk's layers run in slices
        monkeypatch.setattr(H, "MEDIA_PREFILL_STEP", step)
    tokens = image_prompt(grids)
    assert len(tokens) > 8 * 2 + 8  # indexer recall and sliding window both active
    pixels = pixels_for(grids)
    got = np.array(model(mx.array([tokens]), pixel_values=mx.array(pixels), image_grid_thw=grids)[0])
    ref = oracle_logits(config, W, VW, tokens, grids, pixels)
    np.testing.assert_allclose(got, ref, rtol=3e-4, atol=3e-4)


def test_media_prefill_then_decode_matches_full_forward(vtiny):
    config, W, VW, model = vtiny
    prompt = image_prompt()
    tail = [30, 31, 33, 35, 37, 39, 41, 43]
    pixels = pixels_for(GRIDS)
    cache = model.make_cache()
    # The serving rule: one media chunk of all but the final prompt token,
    # then ordinary one-token steps at slot + rope_delta.
    parts = [model(mx.array([prompt[:-1]]), cache, pixel_values=mx.array(pixels), image_grid_thw=GRIDS)]
    parts += [model(mx.array([[tok]]), cache) for tok in [prompt[-1]] + tail]
    got = np.array(mx.concatenate(parts, axis=1)[0])
    ref = oracle_logits(config, W, VW, prompt + tail, GRIDS, pixels)
    np.testing.assert_allclose(got, ref, rtol=3e-4, atol=3e-4)
    assert model.vision_stats["prefills"] >= 1


def test_batch_generator_media_lane_matches_single_lane(vtiny):
    from mlx2.runtime.generate import BatchGenerator
    from test_hy_v4_flash_cpu import _drain

    _, _, _, model = vtiny
    prompt = image_prompt()
    pixels = mx.array(pixels_for(GRIDS))
    cache = model.make_cache()
    model(mx.array([prompt[:-1]]), cache, pixel_values=pixels, image_grid_thw=GRIDS)
    out = [int(mx.argmax(model(mx.array([[prompt[-1]]]), cache)[0, -1]).item())]
    for _ in range(5):
        out.append(int(mx.argmax(model(mx.array([[out[-1]]]), cache)[0, -1]).item()))
    batch = BatchGenerator(model, max_tokens=6, completion_batch_size=2, prefill_batch_size=1)
    uids = batch.insert([prompt], max_tokens=[6],
                        prefill_inputs=[{"pixel_values": pixels, "image_grid_thw": GRIDS}])
    text = [5, 7, 9, 11, 13, 4, 6, 8, 10, 12]
    uids += batch.insert([text], max_tokens=[6])
    got = _drain(batch, uids, 6)
    assert got[uids[0]][:6] == out
    from test_hy_v4_flash_cpu import _greedy

    assert got[uids[1]][:6] == _greedy(model, text, 6)


def test_text_only_is_bit_identical_with_vision_attached(vtiny):
    config, W, _, model = vtiny
    plain, _ = build(config, W)
    tokens = [5, 7, 9, 11, 13, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31]
    np.testing.assert_array_equal(np.array(model(mx.array([tokens]))), np.array(plain(mx.array([tokens]))))
    a, b = model.make_cache(), plain.make_cache()
    model(mx.array([tokens[:11]]), a)
    plain(mx.array([tokens[:11]]), b)
    for tok in tokens[11:]:
        np.testing.assert_array_equal(np.array(model(mx.array([[tok]]), a)),
                                      np.array(plain(mx.array([[tok]]), b)))
    assert all(c.mrope_spans == () and len(c.meta_state) == 11 for c in a)


def test_image_paths_fail_closed(vtiny):
    config, W, _, model = vtiny
    prompt = image_prompt()
    pixels = mx.array(pixels_for(GRIDS))
    with pytest.raises(ValueError, match="pixel_values"):
        model(mx.array([prompt]))
    with pytest.raises(ValueError, match="video"):
        model(mx.array([[5, VID, 6]]))
    with pytest.raises(ValueError, match="video"):
        model(mx.array([[5, 6]]), pixel_values_videos=pixels)
    with pytest.raises(ValueError):  # one grid for two spans
        model(mx.array([prompt]), pixel_values=pixels[:24], image_grid_thw=GRIDS[:1])
    with pytest.raises(ValueError):  # grid disagrees with the span length
        model(mx.array([prompt]), pixel_values=pixels, image_grid_thw=[(1, 6, 4), (1, 6, 4)])
    with pytest.raises(ValueError, match="B=1"):
        model(mx.array([prompt, prompt]), pixel_values=pixels, image_grid_thw=GRIDS)
    plain, _ = build(config, W)
    with pytest.raises(ValueError, match="not loaded"):
        plain(mx.array([prompt]), pixel_values=pixels, image_grid_thw=GRIDS)


def test_vision_spans_survive_state_round_trip_and_refuse_split_trims(vtiny):
    _, _, _, model = vtiny
    prompt = image_prompt()
    pixels = mx.array(pixels_for(GRIDS))
    cache = model.make_cache()
    model(mx.array([prompt]), cache, pixel_values=pixels, image_grid_thw=GRIDS)
    spans = cache[0].mrope_spans
    assert len(spans) == 2 and len(cache[0].meta_state) == 12
    restored = [type(c).from_state(c.state, c.meta_state) for c in cache]
    assert all(c.mrope_spans == spans for c in restored)
    nxt = mx.array([[40]])
    np.testing.assert_array_equal(np.array(model(nxt, restored)), np.array(model(nxt, cache)))
    # Rolling back generated tokens is fine; cutting into a span is not.
    assert cache[0].can_trim(1)
    inside = len(prompt) + 1 - (V.span_end(spans[1]) - 3)
    assert not cache[0].can_trim(inside)
    with pytest.raises(ValueError, match="vision span"):
        cache[0].trim(inside)


def test_vision_sanitize_maps_every_checkpoint_name():
    VW = vit_checkpoint()
    names = V.VisionTower.sanitize({k: mx.array(v) for k, v in VW.items()})
    assert "layers.0.layer_norm1.bias" in names and "layers.1.mlp.fc2.weight" in names
    assert names["patch_merger.proj_conv.weight"].shape == (64, 64)
    assert names["patch_embedding.weight"].shape == (16, 48)
    with pytest.raises(ValueError):
        V.VisionArgs.from_config({**VISION, "projector_type": "mlp"})
    with pytest.raises(ValueError):
        V.VisionArgs.from_config({**VISION, "use_2d_rope": True})


# ------------------------------------------------------ preprocessing --

def smart_resize_ref(height, width, factor, min_pixels, max_pixels):
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def test_smart_resize_and_patchify_match_reference():
    from mlx2.adapters import hy_v4_flash_media as M

    for h, w in [(40, 56), (512, 512), (3000, 1200), (100, 7000), (17, 33)]:
        assert M.smart_resize(h, w) == smart_resize_ref(h, w, 32, 512 * 512, 2048 * 2048)
    rng = np.random.default_rng(0)
    p, m, C, gh, gw = 4, 2, 3, 6, 8
    image = rng.standard_normal((C, gh * p, gw * p)).astype(np.float32)
    got, a, b = M.patchify(image, p)
    # Reference ``patchify`` (batch 1, temporal 1).
    ref = image[None].reshape(1, C, gh // m, m, p, gw // m, m, p).transpose(0, 2, 3, 5, 6, 1, 4, 7)
    ref = ref[:, :, :, :, :, :, None].reshape(1, gh * gw, C * p * p)[0]
    assert (a, b) == (gh, gw)
    np.testing.assert_array_equal(got, ref)
    ids, end = M.expand_image_placeholders([1, IMG_START, IMG, IMG_END, 4, IMG_START, IMG, IMG_END, 5],
                                           [3, 2], image_token_id=IMG, start_token_id=IMG_START,
                                           end_token_id=IMG_END)
    assert ids == [1, IMG_START, IMG, IMG, IMG, IMG_END, 4, IMG_START, IMG, IMG, IMG_END, 5] and end == 10
    with pytest.raises(ValueError):
        M.expand_image_placeholders([1, IMG, 5], [3], image_token_id=IMG, start_token_id=IMG_START,
                                    end_token_id=IMG_END)
    with pytest.raises(ValueError):
        M.expand_image_placeholders([1, IMG_START, IMG, IMG_END, IMG_START, IMG, IMG_END], [3],
                                    image_token_id=IMG, start_token_id=IMG_START, end_token_id=IMG_END)


# ------------------------------------------------------ adapter route --

TEMPLATE = ("{% for m in messages %}t5 {% if m.content is string %}{{ m.content }} {% else %}"
            "{% for p in m.content %}{% if p.type == 'text' %}{{ p.text }} {% else %}"
            "t90 t92 t91 {% endif %}{% endfor %}{% endif %}{% endfor %}t6")


def _vision_artifact(tmp_path, config, W, VW):
    from tokenizers import Tokenizer, models, pre_tokenizers

    from test_hy_v4_flash_cpu import _write_artifact

    path = _write_artifact(tmp_path, config, {**W, **VW})
    vocab = {f"t{i}": i for i in range(config["text_config"]["vocab_size"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t2", "chat_template": TEMPLATE}))
    return path


def _png(width, height, seed=0):
    from PIL import Image

    rgb = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_adapter_image_request_end_to_end(tmp_path, vtiny):
    from mlx2.adapters import hy_v4_flash as A
    from mlx2.api_resources import CapabilityUnavailable
    from mlx2.contracts import Capability

    config, W, VW, _ = vtiny
    path = _vision_artifact(tmp_path, config, W, VW)
    adapter = A.HYV4FlashAdapter(str(path))
    try:
        assert Capability.VISION in adapter.descriptor.capabilities
        assert adapter.multimodal_routes == frozenset({"ordinary"})
        text = {"messages": [{"role": "user", "content": "t7 t8"}]}
        assert adapter.prepare_multimodal_request(text) is text
        request = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "t7 t8"},
            {"type": "image_url", "image_url": {"url": _png(56, 40)}},
            {"type": "text", "text": "t9 t10 t11"}]}]}
        prepared = adapter.prepare_multimodal_request(request)
        ids = prepared["_mlx2_prompt_tokens"]
        inputs = prepared["_mlx2_prefill_inputs"]
        # 40x56 under max 32*32 pixels at factor 8 -> 24x32 -> grid 6x8 -> 3*(4+1) tokens.
        assert inputs["image_grid_thw"] == ((1, 6, 8),)
        assert ids == [5, 7, 8, IMG_START] + [IMG] * 15 + [IMG_END, 9, 10, 11, 6]
        assert prepared["_mlx2_media_token_end"] == 19 and adapter.prompt_tokens(prepared) == ids
        assert adapter.validate_prefill_inputs(prepared, ids, dict(inputs)) == inputs
        with pytest.raises(ValueError):
            adapter.validate_prefill_inputs(prepared, ids[:10], dict(inputs))
        got = np.array(adapter.model(mx.array([ids]), **inputs)[0])
        ref = oracle_logits(config, W, VW, ids, [(1, 6, 8)], np.array(inputs["pixel_values"]))
        np.testing.assert_allclose(got, ref, rtol=3e-4, atol=3e-4)
        d = adapter.diagnostics()["vision"]
        assert d["loaded"] and not d["qualified"] and d["observed_used"]
        video = {"messages": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "data:video/mp4;base64,AAAA"}]}]}
        with pytest.raises(CapabilityUnavailable, match="video"):
            adapter.prepare_multimodal_request(video)
    finally:
        adapter.close()
    off = A.HYV4FlashAdapter(str(path), execution_policy={"vision": False})
    try:
        assert Capability.VISION not in off.descriptor.capabilities
        assert not off.model.has_vision
        with pytest.raises(CapabilityUnavailable):
            off.prepare_multimodal_request(request)
    finally:
        off.close()


def test_partial_vision_tower_refuses(tmp_path, vtiny):
    from mlx2.adapters import hy_v4_flash as A

    config, W, VW, _ = vtiny
    partial = {k: v for k, v in VW.items() if k != "vit.layers.1.mlp.dense_4h_to_h.bias"}
    path = _vision_artifact(tmp_path, config, W, partial)
    with pytest.raises(ValueError, match="vision tensors"):
        A.inspect_artifact(path)


def test_media_refused_on_external_draft_route():
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter
    from mlx2.serving import media_route_policy

    adapter = object.__new__(HYV4FlashAdapter)
    assert media_route_policy(adapter, "ordinary")["media"] == "route_native"
    assert media_route_policy(adapter, "external_draft")["media"] == "refused"
