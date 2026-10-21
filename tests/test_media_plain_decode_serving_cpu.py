"""Image requests on a prompt-lookup server: plain decode, with a receipt (CPU).

Pierre, 2026-10-21: keep prompt lookup as the Gemma 4 default for text, but
serve image/video requests on that server with plain decode (no speculation),
with a route receipt saying so.  These tests drive real ``ServingEngine``
instances over a tiny full Gemma 4 (mlx-vlm's vision tower, embedder and
language model) whose adapter declares the contract the Gemma 4 adapters
declare: media served natively on ``ordinary`` and with plain decode on
``prompt_lookup``.  An image request on the PLD server must equal the same
request on an ordinary server, token for token, and say ``media_decode``
plain decode; a concurrent text request on that server still speculates.
"""

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm")

from route_harness import collect, make_engine, patch_host  # noqa: E402

from mlx2.adapters.gemma4 import Gemma4A4BAdapter  # noqa: E402

IMAGE, VIDEO, V = 60, 61, 64
MEDIA_PROMPT = [1, 2, 3, 58, IMAGE, IMAGE, IMAGE, IMAGE, 59] + list(range(5, 22))
COPY_PROMPT = [7, 8, 9, 10, 11, 12] * 4 + [7, 8, 9]
LAYERS = ["sliding_attention", "sliding_attention", "full_attention"] * 2


def _tiny_gemma4():
    from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.gemma4.gemma4 import Model

    from mlx2.adapters.gemma4 import _Gemma4LogitsModel

    text = TextConfig(
        hidden_size=32, num_hidden_layers=len(LAYERS), intermediate_size=64,
        num_attention_heads=2, head_dim=8, global_head_dim=16, vocab_size=V,
        vocab_size_per_layer_input=V, num_key_value_heads=1,
        num_global_key_value_heads=1, attention_k_eq_v=True,
        num_kv_shared_layers=0, hidden_size_per_layer_input=0,
        sliding_window=8, layer_types=LAYERS, use_double_wide_mlp=False,
        max_position_embeddings=4096,
    )
    vision = VisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        global_head_dim=8, patch_size=4, pooling_kernel_size=2,
        default_output_length=4, position_embedding_size=64,
    )
    config = ModelConfig(
        text_config=text, vision_config=vision, vocab_size=V,
        image_token_id=IMAGE, video_token_id=VIDEO, audio_token_id=62,
        boi_token_id=58, eoi_token_id=59, boa_token_id=57, eoa_token_id=56,
    )
    mx.random.seed(3)
    model = Model(config)
    mx.eval(model.parameters())
    return _Gemma4LogitsModel(model)


PIXELS = {}


class MediaContract:
    """The Gemma 4 route contract over a fake media hook (no processor)."""

    multimodal_routes = Gemma4A4BAdapter.multimodal_routes
    multimodal_plain_decode_routes = Gemma4A4BAdapter.multimodal_plain_decode_routes

    def prepare_multimodal_request(self, request, *, file_loader=None):
        seed = int(request["messages"][0]["content"][0]["image_url"]["url"][-1])
        if seed not in PIXELS:
            # Materialized here, as an adapter's processor output is: a lazy
            # graph would belong to this (submitting) thread's stream.
            pixels = mx.random.uniform(shape=(1, 3, 16, 16), key=mx.random.key(seed))
            mx.eval(pixels)
            PIXELS[seed] = pixels
        pixels = PIXELS[seed]
        return {
            **request,
            "_mlx2_prompt_tokens": list(MEDIA_PROMPT),
            "_mlx2_prefill_inputs": {"pixel_values": pixels},
            "_mlx2_media_token_end": 9,
            "_mlx2_media_fingerprint": f"image-{seed}",
        }

    def prompt_tokens(self, request):
        if "_mlx2_prompt_tokens" in request:
            return list(request["_mlx2_prompt_tokens"])
        return list(request["tokens"])


def _image_request(seed=7, max_tokens=12):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;seed={seed}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def _engine(model, *, prompt_lookup, max_lanes=2):
    return make_engine(
        model, V, mtp=False, prompt_lookup=prompt_lookup, max_lanes=max_lanes,
        num_draft=4, adapter_mixin=MediaContract,
    )


@pytest.fixture(scope="module")
def tiny():
    return _tiny_gemma4()


def test_image_request_on_the_pld_server_equals_ordinary_with_a_plain_decode_receipt(
    monkeypatch, tiny
):
    patch_host(monkeypatch)
    ordinary = _engine(tiny, prompt_lookup=False)
    try:
        reference = collect(ordinary.submit(_image_request()))
        other = collect(ordinary.submit(_image_request(seed=8)))
    finally:
        ordinary.close()
    assert "error" not in reference, reference
    assert reference["receipt"]["route"] == "ordinary"
    # The ordinary route serves media natively: no plain-decode receipt.
    assert "media_decode" not in reference["receipt"]
    pld = _engine(tiny, prompt_lookup=True)
    try:
        assert pld.snapshot["settings"]["route"] == "prompt_lookup"
        assert pld.snapshot["media_route"]["media"] == "plain_decode"
        cold = collect(pld.submit(_image_request()))
        warm = collect(pld.submit(_image_request()))
        second = collect(pld.submit(_image_request(seed=8)))
        counts = dict(pld.counts)
    finally:
        pld.close()
    for result in (cold, warm, second):
        assert "error" not in result, result
    assert cold["tokens"] == reference["tokens"]
    assert warm["tokens"] == reference["tokens"]
    assert second["tokens"] == other["tokens"]
    receipt = cold["receipt"]
    assert receipt["route"] == "prompt_lookup"
    assert receipt["media_decode"] == {
        "schema": "mlx2.media-decode.v1",
        "route": "prompt_lookup",
        "speculation": "disabled_for_media",
        "decode": "ordinary",
        "reason": "media_request",
        "contract": "multimodal_plain_decode_routes",
        "media_prefill": "isolated_prefill_forward",
        "media_prefill_inputs": ["pixel_values"],
        "plain_only_rounds": receipt["speculation"]["plain_only_rounds"],
    }
    assert receipt["speculation"]["proposed"] == 0
    assert receipt["speculation"]["execution"] == "ordinary_target"
    # Warm: APCv2 restored the media prefix; the lane still never speculates.
    assert warm["receipt"]["media_decode"]["media_prefill"] == "apcv2_restored"
    assert warm["receipt"]["media_decode"]["decode"] == "ordinary"
    assert warm["receipt"]["speculation"]["speculation"] == "disabled_for_media"
    assert counts["media_plain_decode_requests"] == 3


def test_concurrent_text_on_the_pld_server_still_speculates(monkeypatch, tiny):
    patch_host(monkeypatch)
    ordinary = _engine(tiny, prompt_lookup=False)
    try:
        reference = collect(ordinary.submit(_image_request(max_tokens=16)))
        text_reference = collect(
            ordinary.submit({"tokens": COPY_PROMPT, "max_tokens": 16, "temperature": 0})
        )
    finally:
        ordinary.close()
    pld = _engine(tiny, prompt_lookup=True)
    try:
        text_job = pld.submit({"tokens": COPY_PROMPT, "max_tokens": 16, "temperature": 0})
        image_job = pld.submit(_image_request(max_tokens=16))
        text, image = collect(text_job), collect(image_job)
    finally:
        pld.close()
    assert "error" not in text and "error" not in image, (text, image)
    assert image["tokens"] == reference["tokens"]
    assert text["tokens"] == text_reference["tokens"]
    assert image["receipt"]["media_decode"]["speculation"] == "disabled_for_media"
    assert image["receipt"]["media_decode"]["decode"] == "ordinary"
    assert "media_decode" not in text["receipt"]
    assert text["receipt"]["route"] == "prompt_lookup"
    speculation = text["receipt"]["speculation"]
    assert "plain_decode_reason" not in speculation
    assert speculation["proposed"] > 0
