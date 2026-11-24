"""Ling-3.2-air vision mode on CPU: vendor M-RoPE laws, request-owned positions,
bitwise text path, image prefill/decode consistency, plane state, refusals."""

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlx2.adapters import ling32_air_vl as vl
from mlx2.runtime.models import bailing_hybrid as bh
from mlx2.runtime.models import ling_mrope as lm

mx.set_default_device(mx.cpu)

IMAGE, START, END = 90, 91, 92
SECTION = [8, 12, 12]


# ---- independent numpy transcriptions of the vendor (MIT) laws -------------

def vendor_rope_index(tokens, grids, merge, image_id=IMAGE, start_id=START):
    """BailingMoeV3VLModel.get_rope_index for one row of still images."""
    tokens = list(tokens)
    starts = [i for i, t in enumerate(tokens) if t == start_id and i + 1 < len(tokens)
              and tokens[i + 1] == image_id]
    pos, st, index = [], 0, 0
    for _ in range(len(starts)):
        ed = tokens.index(image_id, st)
        t, h, w = grids[index]
        index += 1
        t, h, w = int(t), int(h) // merge, int(w) // merge
        text_len = ed - st
        st_idx = pos[-1][0].max() + 1 if pos else 0
        if text_len > 0:
            pos.append(np.broadcast_to(np.arange(text_len)[None] + st_idx, (3, text_len)))
        ti = np.repeat(np.arange(t), h * w) * 2.0 + text_len + st_idx
        hi = np.tile(np.repeat(np.arange(h), w), t) - (h - 1) // 2 + ti
        wi = np.tile(np.arange(w), t * h) - (w - 1) // 2 + ti
        pos.append(np.stack([ti, hi, wi]))
        st = ed + t * h * w
    if st < len(tokens):
        st_idx = pos[-1][0].max() + 1 if pos else 0
        n = len(tokens) - st
        pos.append(np.broadcast_to(np.arange(n)[None] + st_idx, (3, n)))
    out = np.concatenate(pos, axis=1).astype(np.int64)
    return out, int(out[0].max() + 1 - len(tokens))


def vendor_cos_sin(positions, base, dim, section):
    """BailingMoeV3VLRotaryEmbedding.forward + apply_video_rope (numpy; the vendor
    forms the angles in fp32, so the reference does too)."""
    inv = (1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))).astype(np.float32)
    freqs = positions[:, :, None].astype(np.float32) * inv[None, None, :]  # [3, L, d/2]
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = np.cos(emb), np.sin(emb)
    sec = [section[0], section[1] + section[2]] * 2
    sec = sec[::-1]
    index, rc, rs = 0, [], []
    for i, n in enumerate(sec):
        if i % 2 == 0:
            for j in range(n):
                row = 1 if j % 2 == 0 else 2
                rc.append(cos[row, :, index:index + 1])
                rs.append(sin[row, :, index:index + 1])
                index += 1
        else:
            rc.append(cos[0, :, index:index + n])
            rs.append(sin[0, :, index:index + n])
            index += n
    return np.concatenate(rc, -1), np.concatenate(rs, -1)


def vendor_rotate_interleave(x, cos, sin):
    """apply_rotary_pos_emb_interleave on [..., L, d] (numpy)."""
    half = x.shape[-1] // 2
    pairs = x.reshape(*x.shape[:-1], half, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    c, s = cos[..., :half], sin[..., :half]
    return np.stack([even * c - odd * s, odd * c + even * s], -1).reshape(x.shape)


def _prompt(grids, merge=2, text=(5, 6, 7)):
    tokens = [1, 2, 3]
    for g in grids:
        n = int(np.prod(g)) // merge ** 2
        tokens += [START] + [IMAGE] * n + [END, 4] + list(text)
    return tokens + [8, 9]


@pytest.mark.parametrize("grids", [[[1, 4, 6]], [[1, 6, 6], [1, 4, 4]], [[1, 2, 8]]])
def test_positions_match_vendor_rope_index(grids):
    tokens = _prompt(grids)
    spans = lm.find_spans(tokens, image_token_id=IMAGE, start_token_id=START, grids=grids, merge=2)
    ours = lm.slot_positions(spans, np.arange(len(tokens)))
    ref, delta = vendor_rope_index(tokens, grids, 2)
    assert np.array_equal(ours, ref)
    assert lm.text_delta(spans, len(tokens)) == delta
    # Decode positions after the prompt continue at slot + delta on all planes.
    assert lm.chunk_rope_positions(spans, len(tokens), 3) == len(tokens) + delta
    # A chunk that crosses an image gets 3D positions; a text-only chunk a scalar.
    first = spans[0]
    assert isinstance(lm.chunk_rope_positions(spans, first[0] - 1, 3), np.ndarray)
    assert lm.chunk_rope_positions(spans, 0, 2) == 0


def test_find_spans_refusals():
    tokens = _prompt([[1, 4, 4]])
    with pytest.raises(ValueError, match="image runs"):
        lm.find_spans(tokens, image_token_id=IMAGE, start_token_id=START, grids=[], merge=2)
    with pytest.raises(ValueError, match="run length"):
        lm.find_spans(tokens, image_token_id=IMAGE, start_token_id=START, grids=[[1, 4, 8]], merge=2)
    bad = [t if t != START else 4 for t in tokens]
    with pytest.raises(ValueError, match="vision-start"):
        lm.find_spans(bad, image_token_id=IMAGE, start_token_id=START, grids=[[1, 4, 4]], merge=2)


def test_frequency_planes_and_rotation_match_vendor():
    plane = lm.frequency_planes(32, SECTION)
    assert plane[:24].tolist() == [1, 2] * 12 and plane[24:].tolist() == [0] * 8
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 9, 64)).astype(np.float32)
    # Small positions keep every angle small: the plane/frequency law is tested tightly.
    # Up to 500 an fp32 angle carries ~3e-5 rad of rounding (MLX vs numpy pow/cos/sin).
    for high, tol in ((12, 2e-6), (500, 1e-4)):
        pos = np.stack([rng.integers(0, high, 9) for _ in range(3)])
        cos, sin = vendor_cos_sin(pos, 6e6, 64, SECTION)
        ref = vendor_rotate_interleave(x.astype(np.float64), cos, sin)
        got = np.array(lm.mrope_rotate(mx.array(x), pos, base=6e6, section=SECTION))
        np.testing.assert_allclose(got, ref, rtol=tol, atol=tol)
    # A wrong plane assignment (H/W swapped) is far outside that tolerance.
    pos = np.stack([np.arange(9), np.arange(9) * 3, np.arange(9) * 7])
    cos, sin = vendor_cos_sin(pos[[0, 2, 1]], 6e6, 64, SECTION)
    swapped = vendor_rotate_interleave(x.astype(np.float64), cos, sin)
    got = np.array(lm.mrope_rotate(mx.array(x), pos, base=6e6, section=SECTION))
    assert np.max(np.abs(got - swapped)) > 1e-2
    # Equal planes are 1D RoPE: the scalar fast.rope path the text keeps using.
    same = np.broadcast_to(np.arange(9)[None] + 40, (3, 9))
    flat = mx.fast.rope(mx.array(x), 64, traditional=True, base=6e6, scale=1.0, offset=40)
    np.testing.assert_allclose(np.array(lm.mrope_rotate(mx.array(x), same, base=6e6,
                                                        section=SECTION)),
                               np.array(flat), rtol=1e-4, atol=1e-4)


# ---- tiny bailing_hybrid + stand-in tower ----------------------------------

def _tiny():
    args = bh.ModelArgs(
        model_type="bailing_hybrid", vocab_size=96, hidden_size=64, intermediate_size=64,
        moe_intermediate_size=64, num_hidden_layers=6, num_attention_heads=2,
        num_key_value_heads=2, num_experts=8, num_experts_per_tok=2, n_group=2,
        topk_group=1, first_k_dense_replace=2, layer_group_size=6,
        max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=6e6,
        routed_scaling_factor=2.5, head_dim=32, kv_lora_rank=32, qk_rope_head_dim=16,
        qk_nope_head_dim=32, v_head_dim=32, router_dtype="fp32", vendor_v32_laws=True,
        no_kda_lora=True, kda_lower_bound=-5.0, kda_safe_gate=True,
        gated_attention_proj_granularity_type="head_wise", mrope_section=[2, 3, 3],
    )
    mx.random.seed(0)
    model = bh.Model(args)
    mx.eval(model.parameters())
    return model


class _Tower(nn.Module):
    """Stand-in tower with the ling32_vision call contract: [N, P] -> [N/4, 4P']."""

    def __init__(self, width):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(6, width)

    def __call__(self, pixels, grids):
        x = self.patch_embed.proj(pixels)
        return x.reshape(-1, x.shape[-1] * 4)


def _vision_model():
    model = _tiny()
    tower, projector = _Tower(16), nn.Linear(64, 64)
    mx.eval(tower.parameters(), projector.parameters())
    config = {"image_token_id": IMAGE, "vision_start_token_id": START,
              "vision_config": {"spatial_merge_size": 2}}
    return model, vl.LingVisionModel(model, tower, projector, config)


def test_text_only_vision_mode_is_bitwise_plain():
    model, wrapped = _vision_model()
    tokens = mx.array([[3, 7, 11, 2, 5, 9]])
    plain_cache, vl_cache = model.make_cache(), wrapped.make_cache()
    assert isinstance(vl_cache[-1], vl.LingMRoPECache) and len(vl_cache) == len(plain_cache) + 1
    a = model(tokens, plain_cache)
    b = wrapped(tokens, vl_cache)
    assert mx.array_equal(a, b)
    for step in (4, 13):
        a = model(mx.array([[step]]), plain_cache)
        b = wrapped(mx.array([[step]]), vl_cache)
        assert mx.array_equal(a, b)
    assert wrapped.vision_stats["plain_text_forwards"] == 3
    assert vl_cache[-1].rows[0] == {"length": 8, "delta": 0, "media": False, "media_end": 0}


def test_image_prefill_then_decode_matches_one_full_forward():
    model, wrapped = _vision_model()
    grids = [[1, 4, 6]]
    prompt = _prompt(grids)
    end = max(i + 1 for i, t in enumerate(prompt) if t == IMAGE)
    pixels = np.random.default_rng(1).standard_normal((24, 6)).astype(np.float32)
    cache = wrapped.make_cache()
    logits = wrapped(mx.array([prompt]), cache, pixel_values=pixels, image_grid_thw=grids,
                     _mlx2_rope_media_end=end)
    spans = lm.find_spans(prompt, image_token_id=IMAGE, start_token_id=START, grids=grids, merge=2)
    delta = lm.text_delta(spans, len(prompt))
    assert delta < 0 and cache[-1].rows[0] == {"length": len(prompt), "delta": delta,
                                               "media": True, "media_end": end}
    steps = [logits[:, -1]]
    generated = [17, 23]
    for token in generated:
        steps.append(wrapped(mx.array([[token]]), cache)[:, -1])
    assert wrapped.vision_stats["shifted_text_forwards"] == 2
    # Reference: one uncached forward over prompt + generated with explicit 3D positions.
    full = prompt + generated
    positions = lm.slot_positions(spans, np.arange(len(full)))
    embeds = model.model.word_embeddings(mx.array([full]))
    features = wrapped.projector(wrapped.vision_tower(mx.array(pixels), grids))
    where = mx.array([i for i, t in enumerate(full) if t == IMAGE], mx.int32)
    embeds[0, where] = features.astype(embeds.dtype)
    ref = model.lm_head(model.model(mx.array([full]), model.make_cache(), inputs_embeds=embeds,
                                    rope_positions=positions))
    for i, got in enumerate(steps):
        assert mx.allclose(got, ref[:, len(prompt) - 1 + i], atol=1e-4, rtol=1e-4)
    # The image actually moved positions: plain sequential RoPE gives other logits.
    plain = model.lm_head(model.model(mx.array([full]), model.make_cache(), inputs_embeds=embeds))
    assert not mx.allclose(plain[:, -1], ref[:, -1], atol=1e-4)


def test_media_prefill_refusals():
    _, wrapped = _vision_model()
    grids = [[1, 4, 4]]
    prompt = _prompt(grids)
    end = max(i + 1 for i, t in enumerate(prompt) if t == IMAGE)
    pixels = np.zeros((16, 6), np.float32)
    cache = wrapped.make_cache()
    wrapped(mx.array([[1, 2]]), cache)
    with pytest.raises(ValueError, match="cold isolated"):
        wrapped(mx.array([prompt]), cache, pixel_values=pixels, image_grid_thw=grids,
                _mlx2_rope_media_end=end)
    with pytest.raises(ValueError, match="without their image payload"):
        wrapped(mx.array([prompt]), wrapped.make_cache())
    with pytest.raises(ValueError, match="media boundary is not the end"):
        wrapped(mx.array([prompt]), wrapped.make_cache(), pixel_values=pixels,
                image_grid_thw=grids, _mlx2_rope_media_end=end - 1)
    with pytest.raises(ValueError, match="M-RoPE cache plane"):
        wrapped(mx.array([[1]]), _tiny().make_cache())
    with pytest.raises(ValueError, match="tree verification"):
        layer = next(l for l in wrapped.layers if l.is_global)
        layer.attention._rope_at(mx.zeros((1, 2, 1, 16)), mx.zeros((1, 1, 1, 16)), 3, object())


def test_plane_merge_extract_and_state_round_trip():
    a, b = vl.LingMRoPECache(), vl.LingMRoPECache()
    a.rows[0].update(length=40, delta=-11, media=True, media_end=30)
    b.rows[0].update(length=7)
    merged = vl.LingMRoPECache.merge([a, b])
    assert type(merged) is vl.LingMRoPECache and len(merged.rows) == 2
    assert merged.extract(0).rows == a.rows and type(merged.extract(1)) is vl.LingMRoPECache
    restored = vl.LingMRoPECache.from_state(merged.state, merged.meta_state)
    assert restored.rows == merged.rows and restored.meta_state[0] == vl.TAG
    with pytest.raises(ValueError, match="incompatible Ling"):
        vl.LingMRoPECache.from_state(merged.state, ("qwen25-mrope-v1", merged.meta_state[1]))
    with pytest.raises(ValueError, match="inside media span"):
        a.trim(20)
    a.trim(5)
    assert a.rows[0]["length"] == 35


def test_prompt_expansion_and_message_conversion():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "what is"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "?"}]}]
    converted, sources = vl.template_messages(messages)
    assert converted[0]["content"][1] == {"type": "image"} and len(sources) == 1
    with pytest.raises(ValueError, match="still images only"):
        vl.template_messages([{"role": "user", "content": [{"type": "video_url"}]}])
    prompt = "<|role_start|>user" + vl.TEMPLATE_IMAGE + "hi"
    expanded = vl.expand_image_tokens(prompt, [[1, 4, 6]], 2)
    assert expanded == ("<|role_start|>user" + vl.VISION_START + vl.IMAGE_PAD * 6
                        + vl.VISION_END + "\nhi")
    with pytest.raises(ValueError, match="placeholder"):
        vl.expand_image_tokens(prompt + vl.IMAGE_PAD, [[1, 4, 6]], 2)


def test_adapter_refuses_media_when_vision_mode_is_off(monkeypatch):
    from mlx2.adapters.ling32_air import Ling32AirAdapter
    from mlx2.api_resources import CapabilityUnavailable

    adapter = Ling32AirAdapter.__new__(Ling32AirAdapter)
    adapter.vision_enabled = False
    adapter.config = {"image_token_id": IMAGE}
    request = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    with pytest.raises(CapabilityUnavailable, match="vision mode is off"):
        adapter.prepare_multimodal_request(request)
    text = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert adapter.prepare_multimodal_request(text)["messages"][0]["content"] == [
        {"type": "text", "text": "hi"}]
    with pytest.raises(ValueError, match="without their image payload"):
        adapter.validate_prefill_inputs({}, [1, IMAGE, 2], None)
    with pytest.raises(ValueError, match="Unbound"):
        adapter.validate_prefill_inputs({}, [1], {"pixel_values": 0})
    with pytest.raises(ValueError, match="Unbound"):
        adapter.prompt_tokens({"_mlx2_prompt_tokens": [1, 2]})
    adapter.vision_enabled = True
    with pytest.raises(ValueError, match="ordinary route only"):
        adapter.profile_name(True)
    with pytest.raises(ValueError, match="ordinary route only"):
        Ling32AirAdapter("/nonexistent", execution_policy={"ling_vision": True, "num_draft": 2})


def test_mixed_chunk_rotates_text_runs_bitwise_like_text_mode():
    """Review 2026-11-24: text rows of an image prefill must not drift from text mode."""
    model = _tiny()
    attention = next(l for l in model.model.layers if l.is_global).attention
    grids = [[1, 4, 6]]
    prompt = _prompt(grids)
    spans = lm.find_spans(prompt, image_token_id=IMAGE, start_token_id=START, grids=grids, merge=2)
    positions = lm.slot_positions(spans, np.arange(len(prompt)))
    q = mx.random.normal((1, 2, len(prompt), 16)).astype(mx.bfloat16)
    k = mx.random.normal((1, 1, len(prompt), 16)).astype(mx.bfloat16)
    rq, rk = attention._rope_at(q, k, positions, None)
    first, end = spans[0][0], spans[0][0] + 6
    # Text before the image: exactly text mode's rope at offset 0.
    assert mx.array_equal(rq[..., :first, :], attention.rope(q[..., :first, :], offset=0))
    # Text after the image: fast.rope at its shifted offset, bitwise.
    after = int(positions[0, end])
    assert mx.array_equal(rk[..., end:, :], attention.rope(k[..., end:, :], offset=after))
    # Image rows with unequal planes: the 3D law. A centred token can have equal planes;
    # it takes fast.rope (the same law, its own rounding).
    image = positions[:, first:end]
    unequal = np.flatnonzero(~((image[0] == image[1]) & (image[1] == image[2])))
    assert unequal.size
    expect = lm.mrope_rotate(q[..., first:end, :], image, base=6e6, section=[2, 3, 3])
    pick = mx.array(unequal, mx.int32)
    assert mx.array_equal(rq[..., first:end, :][..., pick, :], expect[..., pick, :])
    assert mx.allclose(rq[..., first:end, :], expect, atol=2e-2)


ARTIFACT = __import__("pathlib").Path.home() / "mlx-models/Ling-3.2-air-mlx2-e4r8"


@pytest.mark.skipif(not (ARTIFACT / "tokenizer.json").exists(), reason="Ling-3.2-air artifact absent")
def test_prepare_with_real_template_image_and_tool_history():
    """Review 2026-11-24: prepare() read a missing attribute and skipped the tool-call
    history translation; exercise it with the real tokenizer and chat template."""
    import base64
    import io
    import json

    from PIL import Image
    from transformers import AutoTokenizer

    from mlx2.adapters.ling32_air import Ling32AirAdapter
    from mlx2.adapters.ling32_air_images import policy

    adapter = Ling32AirAdapter.__new__(Ling32AirAdapter)
    adapter.vision_enabled = True
    adapter.config = json.loads((ARTIFACT / "config.json").read_text())
    adapter.tokenizer = AutoTokenizer.from_pretrained(ARTIFACT)
    adapter.image_policy = policy(ARTIFACT)
    buffer = io.BytesIO()
    Image.new("RGB", (96, 64), "red").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    request = {"messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "user", "content": [{"type": "text", "text": "And this?"},
                                     {"type": "image_url", "image_url": {"url": url}}]}],
        "enable_thinking": False}
    out = adapter.prepare_multimodal_request(request)
    ids = out["_mlx2_prompt_tokens"]
    pad = adapter.config["image_token_id"]
    grid = out["_mlx2_prefill_inputs"]["image_grid_thw"][0]
    assert ids.count(pad) == grid[0] * grid[1] * grid[2] // 4
    end = out["_mlx2_media_token_end"]
    assert ids[end - 1] == pad and end < len(ids)
    tail = adapter.tokenizer.decode(ids[end:end + 2])
    assert tail.startswith("<|vision_end|>\n")
    assert adapter.prompt_tokens(out) == ids and out["_mlx2_media_fingerprint"]
    assert adapter.validate_prefill_inputs(out, ids, out["_mlx2_prefill_inputs"]) is not None
    # One image above the patch bound is refused before any model work.
    buffer = io.BytesIO()
    Image.new("RGB", (2048, 2048), "blue").save(buffer, format="PNG")
    big = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    with pytest.raises(ValueError, match="image patches"):
        adapter.prepare_multimodal_request({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": big}}]}]})
