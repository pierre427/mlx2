"""Gemma 4 prompt lookup: segmented verify rows on mlx-vlm's attention, exactly.

model-matrix-20261021 found that PLD's batched verify hands mlx-vlm's Gemma 4
attention a ``SegmentedKVView`` (no batched keys) and an mlx2 ``_RowMask``,
which ``mx.fast.scaled_dot_product_attention`` rejects with a TypeError that
killed the worker (HTTP 503).  The adapter's ``pld_verify_forward`` lets the
view reduce its own rows.  These tests drive the real prompt-lookup generator
on a tiny random-weight Gemma 4 (mlx-vlm's language model, dense and sparse
MoE, a sliding window that wraps) and compare against ordinary greedy decode.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

mx.set_default_device(mx.cpu)

pytest.importorskip("mlx_vlm")

from mlx2.adapters.gemma4 import (  # noqa: E402
    GEMMA4_31B,
    GEMMA4_A4B,
    Gemma431BAdapter,
    Gemma4A4BAdapter,
    _Gemma4LogitsModel,
)
from mlx2.contracts import Capability  # noqa: E402
from mlx2.runtime.models.cache import KVCache, RotatingKVCache  # noqa: E402
from mlx2.runtime.pld import PromptLookupBatchGenerator  # noqa: E402

V, WINDOW = 64, 8
LAYERS = ["sliding_attention", "sliding_attention", "full_attention"] * 2
TEXT = dict(
    hidden_size=32,
    num_hidden_layers=len(LAYERS),
    intermediate_size=64,
    num_attention_heads=2,
    head_dim=8,
    global_head_dim=16,
    vocab_size=V,
    vocab_size_per_layer_input=V,
    num_key_value_heads=1,
    num_global_key_value_heads=1,
    attention_k_eq_v=True,
    num_kv_shared_layers=0,
    hidden_size_per_layer_input=0,
    sliding_window=WINDOW,
    layer_types=LAYERS,
    use_double_wide_mlp=False,
    max_position_embeddings=4096,
)
SPARSE = dict(
    enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=16
)
# Longer than the sliding window, so every ring has wrapped before decode.
PROMPTS = [
    [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3],
    [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8],
    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
]
EXPLORE = {
    "num_draft": 4,
    "ngram_min": 2,
    "ngram_max": 3,
    "adaptive": False,
    "deferred_admission": False,
}


class TinyGemma4(nn.Module):
    """mlx-vlm's Gemma 4 text model, as the adapter's wrapper serves it."""

    def __init__(self, seed=2, **extra):
        super().__init__()
        from mlx_vlm.models.gemma4.config import TextConfig
        from mlx_vlm.models.gemma4.language import LanguageModel

        config = TextConfig(**{**TEXT, **extra})
        mx.random.seed(seed)
        self.language_model = LanguageModel(config)
        self.config = SimpleNamespace(text_config=config)
        mx.eval(self.parameters())

    def __call__(self, inputs, cache=None, **kwargs):
        return self.language_model(inputs, cache=cache, **kwargs)


@pytest.fixture(scope="module", params=["dense", "sparse_moe"])
def gemma(request):
    extra = SPARSE if request.param == "sparse_moe" else {}
    return _Gemma4LogitsModel(TinyGemma4(**extra))


def _greedy(model, prompt, tokens):
    cache = model.make_cache()
    logits = model(mx.array([prompt]), cache=cache)
    produced = []
    for _ in range(tokens):
        token = int(mx.argmax(logits[0, -1]).item())
        produced.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    return produced


def _run(model, prompts, policy, tokens=24):
    generator = PromptLookupBatchGenerator(
        model,
        completion_batch_size=len(prompts),
        prefill_step_size=5,
        prompt_lookup=policy,
    )
    uids = generator.insert(prompts, max_tokens=[tokens] * len(prompts))
    out, finals = {uid: [] for uid in uids}, {}
    for _ in range(4000):
        _prompts, responses = generator.next()
        for response in responses:
            out[response.uid].append(response.token)
            if response.finish_reason:
                finals[response.uid] = response
        if len(finals) == len(uids):
            break
    generator.close()
    return [out[uid] for uid in uids], [finals[uid] for uid in uids], generator


def test_the_wrapper_owns_its_prompt_lookup_verify_forward(gemma):
    assert callable(gemma.pld_verify_forward)
    assert gemma.pld_verify_forward_kind == "gemma4_segmented_row_attention"
    # The generator reads both declarations off the wrapper, not mlx-vlm.
    assert getattr(gemma, "pld_verify_max_lanes", None) is None


def test_batched_verify_over_wrapped_rings_equals_ordinary_greedy(gemma):
    produced, finals, generator = _run(gemma, PROMPTS, EXPLORE)
    for prompt, tokens in zip(PROMPTS, produced):
        assert tokens == _greedy(gemma, prompt, 24)
    stats = generator.scheduler_stats
    # Shared segmented forwards, proposals, rejected tails: the crash path.
    assert stats["pld_batched_rounds"] > 0
    assert stats["pld_batched_max_width"] == len(PROMPTS)
    assert stats["pld_proposed"] > 0 and stats["pld_rollbacks"] > 0
    assert stats["pld_model_verify_forwards"] > 0
    for final in finals:
        live = final.prompt_cache
        assert {type(c) for c in live} == {KVCache, RotatingKVCache}
        assert all(int(c.offset) == len(final.all_tokens) for c in live)
        assert not any(getattr(c, "speculating", False) for c in live)


def test_per_lane_snapshot_rounds_equal_ordinary_greedy(gemma):
    # batched_verify off: each lane verifies on its own caches through the
    # snapshot/rewind/replay driver, wrapped rings included.
    produced, _finals, generator = _run(
        gemma, PROMPTS[:2], {**EXPLORE, "batched_verify": False}
    )
    for prompt, tokens in zip(PROMPTS, produced):
        assert tokens == _greedy(gemma, prompt, 24)
    assert generator.scheduler_stats["pld_batched_rounds"] == 0
    assert generator.scheduler_stats["pld_rollbacks"] > 0


def test_default_latched_policy_equals_ordinary_greedy(gemma):
    from mlx2.server import DEFAULT_PROMPT_LOOKUP_POLICY

    # A copy-heavy prompt: the model's own continuation of a prefix, then the
    # prefix again, so the latch's shadow probes match and it explores.
    proposed = 0
    for head in PROMPTS:
        prompt = head + _greedy(gemma, head, 30) + head
        produced, _finals, generator = _run(
            gemma, [prompt], dict(DEFAULT_PROMPT_LOOKUP_POLICY), tokens=40
        )
        assert produced[0] == _greedy(gemma, prompt, 40)
        assert generator.verify_forward == "gemma4_segmented_row_attention"
        proposed += generator.scheduler_stats["pld_proposed"]
    assert proposed > 0


def test_published_cache_continues_like_a_cold_prefill(gemma):
    _produced, finals, _generator = _run(gemma, PROMPTS[:1], EXPLORE, tokens=12)
    final = finals[0]
    follow = [4, 5, 6]
    logits = gemma(mx.array([follow]), cache=final.prompt_cache)
    reference = gemma(mx.array([final.all_tokens + follow]), cache=gemma.make_cache())
    assert int(mx.argmax(logits[0, -1]).item()) == int(
        mx.argmax(reference[0, -1]).item()
    )


def test_ordinary_forwards_keep_mlx_vlm_attention(gemma):
    # The installed router is exact outside segmented views: a plain cache
    # forward after a verify gives bit-identical logits to before it.
    prompt = mx.array([PROMPTS[2]])
    before = gemma(prompt, cache=gemma.make_cache())
    _run(gemma, PROMPTS[:1], EXPLORE, tokens=6)
    after = gemma(prompt, cache=gemma.make_cache())
    assert mx.array_equal(before, after).item()


def test_kv_shared_layers_fail_closed_on_a_segmented_verify():
    from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows

    model = _Gemma4LogitsModel(TinyGemma4(num_kv_shared_layers=3))
    cache = model.make_cache()
    model(mx.array([PROMPTS[1]]), cache=cache)
    transaction = SegmentedKVRows([cache]).begin(lengths=[2])
    with pytest.raises(ValueError, match="KV-shared"):
        model.pld_verify_forward(mx.array([[7, 8]]), cache=transaction.caches)
    transaction.abort()


@pytest.mark.parametrize(
    ("adapter_type", "descriptor"),
    [(Gemma4A4BAdapter, GEMMA4_A4B), (Gemma431BAdapter, GEMMA4_31B)],
)
def test_gemma4_defaults_to_latched_prompt_lookup_and_keeps_ordinary(
    adapter_type, descriptor
):
    from mlx2.adapters.registry import AdapterResolution
    from mlx2.server import (
        DEFAULT_PROMPT_LOOKUP_POLICY,
        build_parser,
        resolve_execution_policy_defaults,
        resolve_route_selection,
    )

    assert Capability.PROMPT_LOOKUP in descriptor.capabilities
    resolution = AdapterResolution(adapter_type, descriptor, {"fingerprint": "fixture"})
    assert resolution.default_route == "ordinary"
    parser = build_parser()
    selected = resolve_route_selection(
        parser.parse_args(["--model", "fixture"]), None, resolution
    )
    assert (selected.route, selected.source) == ("prompt_lookup", "capability_default")
    assert (
        resolve_execution_policy_defaults(None, selected, resolution)["prompt_lookup"]
        == DEFAULT_PROMPT_LOOKUP_POLICY
    )
    selected = resolve_route_selection(
        parser.parse_args(["--model", "fixture", "--ordinary"]), None, resolution
    )
    assert (selected.route, selected.source) == ("ordinary", "explicit_flag")
    # The ordinary route serves media natively; the PLD route serves it with
    # plain decode (no speculation for that request), with a receipt.
    assert adapter_type.multimodal_routes == frozenset({"ordinary"})
    assert adapter_type.multimodal_plain_decode_routes == frozenset({"prompt_lookup"})
    contract = object.__new__(adapter_type).route_capability_contract()
    assert contract["prompt_lookup"]["implemented"] is True
    assert contract["prompt_lookup"]["media_routes"] == ("ordinary",)
    assert contract["prompt_lookup"]["media_plain_decode_routes"] == ("prompt_lookup",)
    assert contract["prompt_lookup"]["media"] == "plain_decode"
    assert contract["prompt_lookup"]["media_speculation"] == "disabled_for_media"
    assert contract["prompt_lookup"]["media_receipt"] == "media_decode"
    assert contract["prompt_lookup"]["qualified"] is False


# ---------------------------------------------------------------------------
# Media on the prompt-lookup route: plain decode, no speculation (Pierre,
# 2026-10-21).  A tiny full Gemma 4 (mlx-vlm's vision tower, embedder and
# language model): the image changes the prompt embeddings.
# ---------------------------------------------------------------------------

IMAGE, VIDEO = 60, 61
MEDIA_PROMPT = [1, 2, 3, 58, IMAGE, IMAGE, IMAGE, IMAGE, 59] + list(range(5, 22))
# Copy-heavy text, so a concurrent PLD lane proposes and verifies drafts.
COPY_PROMPT = [1, 2, 3, 4, 5, 6] * 4 + [1, 2, 3]


@pytest.fixture(scope="module")
def vlm():
    from mlx_vlm.models.gemma4.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.gemma4.gemma4 import Model

    vision = VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        patch_size=4,
        pooling_kernel_size=2,
        default_output_length=4,
        position_embedding_size=64,
    )
    config = ModelConfig(
        text_config=TextConfig(**TEXT),
        vision_config=vision,
        vocab_size=V,
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        audio_token_id=62,
        boi_token_id=58,
        eoi_token_id=59,
        boa_token_id=57,
        eoa_token_id=56,
    )
    mx.random.seed(3)
    model = Model(config)
    mx.eval(model.parameters())
    return _Gemma4LogitsModel(model)


def _pixels(seed):
    # 16x16 at patch 4 and pooling 2: four soft tokens, one per IMAGE marker.
    pixels = mx.random.uniform(shape=(1, 3, 16, 16), key=mx.random.key(seed))
    mx.eval(pixels)
    return pixels


def _drive(generator, uids, limit=4000):
    tokens, rows, finals = {uid: [] for uid in uids}, {uid: [] for uid in uids}, {}
    for _ in range(limit):
        _prompts, responses = generator.next()
        for response in responses:
            tokens[response.uid].append(int(response.token))
            rows[response.uid].append(response.logprobs.astype(mx.float32))
            if response.finish_reason:
                finals[response.uid] = response
        if len(finals) == len(uids):
            break
    assert len(finals) == len(uids), "lanes did not finish"
    return tokens, rows, finals


def _ordinary(model, prompt, maximum, media=None):
    """The ordinary route's batch generator, the media reference."""
    from mlx2.runtime.generate import BatchGenerator

    generator = BatchGenerator(
        model, completion_batch_size=4, prefill_batch_size=2, prefill_step_size=5
    )
    uids = generator.insert(
        [prompt], max_tokens=[maximum], prefill_inputs=[media]
    )
    tokens, rows, _finals = _drive(generator, uids)
    generator.close()
    return tokens[uids[0]], rows[uids[0]]


def _pld(model, policy=EXPLORE, capacity=4):
    return PromptLookupBatchGenerator(
        model,
        completion_batch_size=capacity,
        prefill_step_size=5,
        prompt_lookup=dict(policy),
    )


def test_image_request_on_the_prompt_lookup_route_decodes_plain_like_ordinary(vlm):
    from mlx2.server import DEFAULT_PROMPT_LOOKUP_POLICY

    pixels = _pixels(7)
    reference, reference_rows = _ordinary(
        vlm, MEDIA_PROMPT, 16, {"pixel_values": pixels}
    )
    # The image is consumed: another image changes the ordinary output.
    other, other_rows = _ordinary(vlm, MEDIA_PROMPT, 16, {"pixel_values": _pixels(8)})
    assert other != reference or not all(
        mx.array_equal(a, b).item() for a, b in zip(other_rows, reference_rows)
    )
    for policy in (EXPLORE, DEFAULT_PROMPT_LOOKUP_POLICY):
        generator = _pld(vlm, policy)
        uids = generator.insert(
            [MEDIA_PROMPT], max_tokens=[16], prefill_inputs=[{"pixel_values": pixels}]
        )
        tokens, rows, finals = _drive(generator, uids)
        generator.close()
        assert tokens[uids[0]] == reference
        # The ordinary one-token step on the lane's own B1 caches: the
        # logprob rows are the ordinary route's, bit for bit.
        assert all(
            mx.array_equal(a, b).item() for a, b in zip(rows[uids[0]], reference_rows)
        )
        receipt = finals[uids[0]].speculative_receipt
        assert receipt["speculation"] == "disabled_for_media"
        assert receipt["decode"] == "ordinary"
        assert receipt["plain_decode_reason"] == "media_request"
        assert receipt["execution"] == "ordinary_target"
        assert receipt["media_prefill_inputs"] == ["pixel_values"]
        assert receipt["plain_only_rounds"] == receipt["cycles"] == 16
        assert receipt["proposed"] == 0 and receipt["verify_path_rounds"] == 0
        assert receipt["lookup_calls"] == 0
        stats = generator.scheduler_stats
        assert stats["pld_media_prefills"] == 1
        assert stats["pld_plain_only_lanes"] == 1
        assert stats["pld_proposed"] == 0


def test_video_request_on_the_prompt_lookup_route_decodes_plain_like_ordinary(vlm):
    prompt = [VIDEO if token == IMAGE else token for token in MEDIA_PROMPT]
    media = {"pixel_values_videos": _pixels(9)}
    reference, reference_rows = _ordinary(vlm, prompt, 12, media)
    other, other_rows = _ordinary(vlm, prompt, 12, {"pixel_values_videos": _pixels(10)})
    # The frames are consumed: other frames change the ordinary output.
    assert other != reference or not all(
        mx.array_equal(a, b).item() for a, b in zip(other_rows, reference_rows)
    )
    generator = _pld(vlm)
    uids = generator.insert([prompt], max_tokens=[12], prefill_inputs=[media])
    tokens, rows, finals = _drive(generator, uids)
    assert tokens[uids[0]] == reference
    assert all(
        mx.array_equal(a, b).item() for a, b in zip(rows[uids[0]], reference_rows)
    )
    receipt = finals[uids[0]].speculative_receipt
    assert receipt["speculation"] == "disabled_for_media"
    assert receipt["decode"] == "ordinary"
    assert receipt["media_prefill_inputs"] == ["pixel_values_videos"]
    assert receipt["proposed"] == 0


def test_media_lane_beside_speculating_text_lanes_keeps_each_reference(vlm):
    pixels = _pixels(7)
    media_reference, _rows = _ordinary(vlm, MEDIA_PROMPT, 20, {"pixel_values": pixels})
    text_reference = _greedy(vlm, COPY_PROMPT, 20)
    generator = _pld(vlm)
    uids = generator.insert(
        [COPY_PROMPT, MEDIA_PROMPT, PROMPTS[1]],
        max_tokens=[20] * 3,
        prefill_inputs=[None, {"pixel_values": pixels}, None],
    )
    tokens, _rows, finals = _drive(generator, uids)
    assert tokens[uids[1]] == media_reference
    assert tokens[uids[0]] == text_reference
    assert tokens[uids[2]] == _greedy(vlm, PROMPTS[1], 20)
    media = finals[uids[1]].speculative_receipt
    assert media["speculation"] == "disabled_for_media"
    assert media["decode"] == "ordinary" and media["proposed"] == 0
    # The text lanes still speculate: proposals verified in shared forwards.
    text = finals[uids[0]].speculative_receipt
    assert "speculation" not in text and "plain_decode_reason" not in text
    assert text["execution"] == "prompt_lookup_verify"
    assert text["proposed"] > 0 and text["accepted"] > 0
    stats = generator.scheduler_stats
    assert stats["pld_batched_rounds"] > 0 and stats["pld_proposed"] > 0
    # The media lane never joined a verify cohort.
    assert stats["pld_batched_max_width"] <= 2
    assert stats["pld_plain_only_rounds"] == 20


def test_warm_media_continuation_stays_plain_and_matches_cold(vlm):
    """APCv2 restores the media prefix, so no media inputs arrive; serving's
    plain-decode reason keeps the lane plain (no speculation) all the same."""
    pixels = _pixels(7)
    cold = _pld(vlm)
    uids = cold.insert(
        [MEDIA_PROMPT], max_tokens=[12], prefill_inputs=[{"pixel_values": pixels}]
    )
    for _ in range(50):
        cold.next()
        boundary = cold.boundaries.get(uids[0])
        if boundary is not None:
            break
    boundary = cold.pop_prompt_boundary(uids[0])
    cold.close()
    covered = boundary["covered_tokens"]
    assert boundary["tokens"] == MEDIA_PROMPT[:covered]
    reference, _rows = _ordinary(vlm, MEDIA_PROMPT, 12, {"pixel_values": pixels})
    warm = _pld(vlm)
    uids = warm.insert(
        [MEDIA_PROMPT[covered:]],
        max_tokens=[12],
        caches=[boundary["target_cache"]],
        all_tokens=[MEDIA_PROMPT[:covered]],
        prompt_lookup_configs=[{"plain_decode_reason": "media_request"}],
    )
    tokens, _rows, finals = _drive(warm, uids)
    assert tokens[uids[0]] == reference
    receipt = finals[uids[0]].speculative_receipt
    assert receipt["speculation"] == "disabled_for_media"
    assert receipt["decode"] == "ordinary"
    assert "media_prefill_inputs" not in receipt
    assert warm.scheduler_stats["pld_media_prefills"] == 0


def test_media_prefill_refusals_stay_precise(vlm):
    generator = _pld(vlm)
    for semantic in (
        {"_mlx2_persistent_decode_inputs": {"deep_concept_memory": {}}},
        {"deep_concept_memory": {"keys": 1}},
    ):
        with pytest.raises(ValueError, match="neural-concept and activation"):
            generator.insert([MEDIA_PROMPT], prefill_inputs=[semantic])
    with pytest.raises(ValueError, match="cold prompt"):
        generator.insert(
            [MEDIA_PROMPT[4:]],
            all_tokens=[MEDIA_PROMPT[:4]],
            prefill_inputs=[{"pixel_values": _pixels(7)}],
        )
    with pytest.raises(ValueError, match="plain_decode_reason"):
        generator.insert([MEDIA_PROMPT], prompt_lookup_configs=[{"plain_decode_reason": ""}])
    assert len(generator) == 0


def test_serving_media_route_policy_is_the_adapter_contract():
    from mlx2.serving import media_decode_receipt, media_route_policy

    adapter = object.__new__(Gemma4A4BAdapter)
    assert media_route_policy(adapter, "ordinary") == {
        "route": "ordinary", "media": "route_native"
    }
    plain = media_route_policy(adapter, "prompt_lookup")
    assert plain == {
        "route": "prompt_lookup",
        "media": "plain_decode",
        "speculation": "disabled_for_media",
        "decode": "ordinary",
        "reason": "media_request",
        "contract": "multimodal_plain_decode_routes",
    }
    refused = media_route_policy(adapter, "external_draft")
    assert refused["media"] == "refused"
    assert "no implemented multimodal prefill" in refused["reason"]
    # The terminal receipt reports what the lane ran, not the declaration.
    job = SimpleNamespace(media_decode=plain, cached_tokens=0)
    lane = {
        "speculation": "disabled_for_media",
        "decode": "ordinary",
        "media_prefill_inputs": ["pixel_values"],
        "plain_only_rounds": 9,
    }
    assert media_decode_receipt(job, lane) == {
        "schema": "mlx2.media-decode.v1",
        "route": "prompt_lookup",
        "speculation": "disabled_for_media",
        "decode": "ordinary",
        "reason": "media_request",
        "contract": "multimodal_plain_decode_routes",
        "media_prefill": "isolated_prefill_forward",
        "media_prefill_inputs": ["pixel_values"],
        "plain_only_rounds": 9,
    }
    warm = media_decode_receipt(
        SimpleNamespace(media_decode=plain, cached_tokens=40),
        {"speculation": "disabled_for_media", "decode": "ordinary"},
    )
    assert warm["media_prefill"] == "apcv2_restored"
    assert media_decode_receipt(SimpleNamespace(media_decode=None), lane) is None


@pytest.mark.parametrize(
    ("route", "admitted"),
    [("prompt_lookup", True), ("ordinary", True), ("external_draft", False)],
)
def test_serving_admits_media_on_the_prompt_lookup_route(route, admitted):
    from mlx2.api_resources import CapabilityUnavailable
    from mlx2.serving import ServingEngine

    class Prepared(Exception):
        pass

    def prepare(request, **_kwargs):
        raise Prepared

    engine = ServingEngine.__new__(ServingEngine)
    engine.route_capabilities = frozenset()
    engine.snapshot = {"settings": {"route": route}}
    engine.adapter = SimpleNamespace(
        multimodal_routes=Gemma4A4BAdapter.multimodal_routes,
        multimodal_plain_decode_routes=Gemma4A4BAdapter.multimodal_plain_decode_routes,
        prepare_multimodal_request=prepare,
        tokenizer=None,
    )
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
                ],
            }
        ]
    }
    # Admission reaches the adapter's media hook only on a route that
    # serves media; elsewhere it refuses before preprocessing.
    with pytest.raises(Prepared if admitted else CapabilityUnavailable):
        engine._prepare_job(request, tenant_id="test")
