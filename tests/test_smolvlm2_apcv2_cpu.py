"""SmolVLM2's source bridge and cache boundaries without a real MLX import."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from mlx_blocker import block_mlx_imports



@pytest.fixture(autouse=True)
def block_mlx(monkeypatch):
    block_mlx_imports(monkeypatch, __name__)


class _Embeds:
    def __init__(self, tokens):
        self.tokens = tuple(tokens)

    def astype(self, dtype):
        assert dtype == "bf16"
        return self


class _Cache:
    def __init__(self, offset=0, rows=(0,)):
        self.offset = offset
        self.rows = tuple(rows)

    def clone(self):
        return _Cache(self.offset, self.rows)

    def trim(self, count):
        self.offset -= count


class _Layer:
    def __init__(self):
        self.seen = []

    def __call__(self, h, mask, cache):
        self.seen.append((mask, cache.offset))
        cache.offset += len(h.tokens)
        return h


class _Model:
    def __init__(self):
        self.layers = [_Layer(), _Layer()]
        self.language_model = SimpleNamespace(
            layers=self.layers,
            norm=SimpleNamespace(weight=SimpleNamespace(dtype="bf16")),
            lm_head=lambda h: h.tokens,
        )
        self.language_model.norm = lambda h: h
        self.language_model.norm.weight = SimpleNamespace(dtype="bf16")
        self.media = []

    def get_input_embeddings(self, ids, pixel_values=None, **kwargs):
        self.media.append((pixel_values, kwargs))
        return SimpleNamespace(inputs_embeds=_Embeds(ids))


def _mask(h, cache):
    return (cache.offset, cache.rows, len(h.tokens))


def test_media_prefill_clone_trim_and_text_replay_use_request_cache():
    from mlx2.adapters.smolvlm2 import _forward_smol

    model = _Model()
    cache = [_Cache(), _Cache()]
    pixels = object()
    assert _forward_smol(
        model, [1, 2, 3, 4, 5], pixel_values=pixels, cache=cache,
        attention_mask_factory=_mask, pixel_attention_mask="media-mask",
    ) == (1, 2, 3, 4, 5)
    assert [c.offset for c in cache] == [5, 5]
    assert model.media[0] == (pixels, {"pixel_attention_mask": "media-mask"})

    branch = [c.clone() for c in cache]
    assert _forward_smol(
        model, [6, 7], cache=branch, attention_mask_factory=_mask,
    ) == (6, 7)
    assert model.layers[0].seen[-1] == ((5, (0,), 2), 5)
    assert [c.offset for c in cache] == [5, 5]
    assert [c.offset for c in branch] == [7, 7]

    for c in branch:
        c.trim(2)
    _forward_smol(model, [8], cache=branch, attention_mask_factory=_mask)
    assert model.layers[0].seen[-1] == ((5, (0,), 1), 5)
    assert [c.offset for c in branch] == [6, 6]
    assert model.media[1:] == [(None, {}), (None, {})]


def test_merged_rows_supply_mask_from_first_cache_not_cache_list():
    from mlx2.adapters.smolvlm2 import _forward_smol

    model = _Model()
    cache = [_Cache(7, rows=(7, 3)), _Cache(7, rows=(7, 3))]
    _forward_smol(model, [9], cache=cache, attention_mask_factory=_mask)
    assert model.layers[0].seen[-1] == ((7, (7, 3), 1), 7)
    assert model.layers[1].seen[-1] == ((7, (7, 3), 1), 7)


@pytest.mark.parametrize("ids,rejected", [([1, 42, 42, 2], False), ([1, 42], True)])
def test_prepared_image_sets_media_floor_for_apcv2_replay(monkeypatch, ids, rejected):
    from mlx2.adapters import pinned_vlm_candidate as pinned
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter

    # The processor and decoder are fake; no model weights or real MLX load.
    mlx = SimpleNamespace(core=SimpleNamespace(array=lambda value: value))
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx.core)
    monkeypatch.setattr(
        pinned, "resolve_media",
        lambda *_args, **_kwargs: SimpleNamespace(kind="image", value="pixels"),
    )
    monkeypatch.setattr(pinned, "media_fingerprint", lambda *_args, **_kwargs: "media-id")
    adapter = object.__new__(SmolVLM2CandidateAdapter)
    adapter.identity = {"config": {"image_token_id": 42}}
    adapter._media_proof_key = b"test-key"
    class Processor:
        image_token = "<image>"

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **_kwargs):
            return {"input_ids": [ids], "pixel_values": "pixels"}

    adapter.processor = Processor()
    request = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe"},
            {"type": "image_url", "image_url": "image-data"},
        ]}],
    }
    if rejected:
        with pytest.raises(ValueError, match="final prompt token"):
            adapter.prepare_multimodal_request(request)
        return
    prepared = adapter.prepare_multimodal_request(request)
    assert prepared["_mlx2_prompt_tokens"] == [1, 42, 42, 2]
    assert prepared["_mlx2_media_token_end"] == 3
    assert prepared["_mlx2_prefill_inputs"] == {"pixel_values": "pixels"}
    assert prepared["_mlx2_media_fingerprint"] == "media-id"
    assert adapter.has_trusted_media_preparation(prepared)


def test_wrong_cache_depth_fails_before_vision_or_text_forward():
    from mlx2.adapters.smolvlm2 import _forward_smol

    model = _Model()
    with pytest.raises(ValueError, match="wrong layer count"):
        _forward_smol(model, [1], cache=[_Cache()], attention_mask_factory=_mask)
    assert model.media == []


def test_media_request_refuses_processor_output_without_pixels(monkeypatch):
    from mlx2.adapters import pinned_vlm_candidate as pinned
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter

    mlx = SimpleNamespace(core=SimpleNamespace(array=lambda value: value))
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx.core)
    monkeypatch.setattr(
        pinned, "resolve_media",
        lambda *_args, **_kwargs: SimpleNamespace(kind="image", value="pixels"),
    )
    monkeypatch.setattr(pinned, "media_fingerprint", lambda *_args, **_kwargs: "media-id")
    adapter = object.__new__(SmolVLM2CandidateAdapter)
    adapter.identity = {"config": {"image_token_id": 42}}
    adapter._media_proof_key = b"test-key"

    class Processor:
        image_token = "<image>"

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **_kwargs):
            return {"input_ids": [[1, 42, 2]]}

    adapter.processor = Processor()
    request = {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": "image-data"},
    ]}]}
    with pytest.raises(ValueError, match="no image pixels"):
        adapter.prepare_multimodal_request(request)


def test_descriptor_declares_kv_reuse_but_keeps_qualification_pending():
    from mlx2.adapters.smolvlm2 import DESCRIPTOR
    from mlx2.contracts import Capability

    assert Capability.APC_V2 in DESCRIPTOR.capabilities
    assert Capability.PREFIX_REUSE in DESCRIPTOR.capabilities
    assert DESCRIPTOR.metadata["qualification"] == "pending"
    assert DESCRIPTOR.cache_layout == "smolvlm2-image-video-apcv2-kv-v1"


class _NumpyScatter:
    int32 = np.int32
    array = staticmethod(np.array)
    broadcast_to = staticmethod(np.broadcast_to)

    @staticmethod
    def put_along_axis(value, indices, updates, axis):
        result = value.copy()
        np.put_along_axis(result, indices, updates, axis=axis)
        return result


class _ScatterSource:
    def __init__(self):
        self.config = SimpleNamespace(image_token_index=42)
        self.vision_model = object()
        self.connector = object()
        self.source_scatter_calls = 0
        self.language_model = SimpleNamespace(
            layers=[lambda h, _mask, _cache: h],
            norm=lambda h: h,
            lm_head=lambda h: h,
        )
        self.language_model.norm = lambda h: h
        self.language_model.norm.weight = SimpleNamespace(dtype=np.float32)

    def _prepare_inputs_for_multimodal(self, features, embeds, ids):
        self.source_scatter_calls += 1
        result = embeds.copy()
        result[np.asarray(ids) == 42] = features
        return result

    def get_input_embeddings(self, ids, pixel_values=None, **_kwargs):
        embeds = np.arange(np.asarray(ids).size * 3, dtype=np.float32).reshape(
            1, -1, 3
        )
        if pixel_values is not None:
            embeds = self._prepare_inputs_for_multimodal(
                pixel_values, embeds, ids
            )
        return SimpleNamespace(inputs_embeds=embeds)


def test_indexed_image_scatter_matches_source_order_without_host_index_read():
    from mlx2.adapters.smolvlm2 import _forward_smol

    ids = np.array([[7, 42, 9, 42, 42, 11]])
    features = np.array([[90, 91, 92], [80, 81, 82], [70, 71, 72]], np.float32)
    reference = _forward_smol(
        _ScatterSource(), ids, pixel_values=features, cache=[_Cache()],
        attention_mask_factory=lambda *_: None,
    )
    model = _ScatterSource()
    stats = {"engagements": 0, "refusals": 0}
    candidate = _forward_smol(
        model, ids, pixel_values=features, cache=[_Cache()],
        image_token_positions=(1, 3, 4), indexed_scatter=True,
        _mlx2_smol_positions_verified=True,
        scatter_backend=_NumpyScatter,
        scatter_stats=stats, attention_mask_factory=lambda *_: None,
    )
    np.testing.assert_array_equal(candidate, reference)
    assert model.source_scatter_calls == 0
    assert stats == {"engagements": 1, "refusals": 0}


@pytest.mark.parametrize("positions", [(1, 3), (1, 3, 9), (3, 1, 4)])
def test_indexed_image_scatter_refuses_unproven_static_shape(positions):
    from mlx2.adapters.smolvlm2 import _forward_smol

    model = _ScatterSource()
    stats = {"engagements": 0, "refusals": 0}
    ids = np.array([[7, 42, 9, 42, 42, 11]])
    features = np.ones((3, 3), np.float32)
    _forward_smol(
        model, ids, pixel_values=features, cache=[_Cache()],
        image_token_positions=positions, indexed_scatter=True,
        _mlx2_smol_positions_verified=True,
        scatter_backend=_NumpyScatter,
        scatter_stats=stats, attention_mask_factory=lambda *_: None,
    )
    assert model.source_scatter_calls == 1
    assert stats == {"engagements": 0, "refusals": 1}


def test_indexed_image_positions_come_from_processed_prompt(monkeypatch):
    from mlx2.adapters import pinned_vlm_candidate as pinned
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter

    mlx = SimpleNamespace(core=SimpleNamespace(array=lambda value: value))
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx.core)
    monkeypatch.setattr(
        pinned, "resolve_media",
        lambda *_args, **_kwargs: SimpleNamespace(kind="image", value="pixels"),
    )
    monkeypatch.setattr(pinned, "media_fingerprint", lambda *_args, **_kwargs: "id")
    adapter = object.__new__(SmolVLM2CandidateAdapter)
    adapter.identity = {"config": {"image_token_id": 42}}
    adapter.model = SimpleNamespace(indexed_scatter=True)
    adapter._media_proof_key = b"test-key"
    class Processor:
        image_token = "<image>"

        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, **_kwargs):
            return {"input_ids": [[7, 42, 9, 42, 42, 11]], "pixel_values": "pixels"}

    adapter.processor = Processor()
    request = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe"},
        {"type": "image_url", "image_url": "image-data"},
    ]}]}
    prepared = adapter.prepare_multimodal_request(request)
    assert prepared["_mlx2_prefill_inputs"]["image_token_positions"] == (1, 3, 4)


def test_smol_template_preserves_typed_image_and_video_frame_order():
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter

    adapter = object.__new__(SmolVLM2CandidateAdapter)
    adapter.processor = SimpleNamespace(image_token="<image>")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Compare"},
        {"type": "image_url", "image_url": "ignored"},
        {"type": "input_video", "video_url": "ignored"},
    ]}]
    converted = adapter._template_messages(
        messages, ["<image>", "[video frame 1] <image>\n[video frame 2] <image>"],
    )
    assert converted[0]["content"] == [
        {"type": "text", "text": "Compare"},
        {"type": "image"},
        {"type": "text", "text": "[video frame 1] "},
        {"type": "image"},
        {"type": "text", "text": "\n[video frame 2] "},
        {"type": "image"},
    ]


def test_smol_text_chat_renders_distinct_typed_user_content():
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter

    class Processor:
        image_token = "<image>"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            assert len(messages) == 1 and messages[0]["role"] == "user"
            return "User: " + "".join(
                part["text"] for part in messages[0]["content"]
                if part["type"] == "text"
            ) + "\nAssistant:"

    adapter = object.__new__(SmolVLM2CandidateAdapter)
    adapter.processor = Processor()
    first = adapter.render_prompt({"messages": [
        {"role": "user", "content": "Reply with exactly HERMES_READY"},
    ]})
    second = adapter.render_prompt({"messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Reply with exactly MLX2_READY"},
        ]},
    ]})
    assert first == "User: Reply with exactly HERMES_READY\nAssistant:"
    assert second == "User: Reply with exactly MLX2_READY\nAssistant:"
    assert adapter.render_prompt({"prompt": "raw prompt"}) == "raw prompt"
    with pytest.raises(RuntimeError, match="replacements are fewer"):
        adapter.render_prompt({"messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": "x"}]},
        ]})


def test_prepared_video_frames_follow_still_image_in_processor_order(monkeypatch):
    from mlx2.adapters import pinned_vlm_candidate as pinned
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter
    from mlx2.multimodal import MediaValue

    mlx = SimpleNamespace(core=SimpleNamespace(array=lambda value: value))
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx.core)
    still = object()
    frame_a = np.ones((3, 4, 3), dtype=np.uint8)
    frame_b = np.full((3, 4, 3), 2, dtype=np.uint8)
    image = MediaValue("image", "image/png", "still-digest", 10, still, {})
    video = MediaValue(
        "video", "video/mp4", "video-digest", 20, [frame_a, frame_b],
        {"source_frames": 2, "source_fps": 1.0, "sampled_indices": (0, 1)},
    )
    monkeypatch.setattr(
        pinned, "resolve_media",
        lambda _source, *, kind, **_kwargs: image if kind == "image" else video,
    )
    seen = {}

    class Processor:
        image_token = "<image>"

        def apply_chat_template(self, messages, **_kwargs):
            seen["content"] = messages[0]["content"]
            return "prompt-with-three-markers"

        def __call__(self, *, text, images, videos):
            assert text == "prompt-with-three-markers"
            assert videos is None
            seen["images"] = images
            return {
                "input_ids": [[7, 42, 9, 42, 42, 11]],
                "pixel_values": "processed-pixels",
            }

    adapter = object.__new__(SmolVLM2CandidateAdapter)
    adapter.processor = Processor()
    adapter.model = SimpleNamespace(indexed_scatter=False)
    adapter.identity = {"config": {"image_token_id": 42}}
    adapter._media_proof_key = b"cpu-test"
    request = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Compare "},
        {"type": "image_url", "image_url": "image-data"},
        {"type": "input_video", "video_url": "video-data"},
    ]}]}
    prepared = adapter.prepare_multimodal_request(request)
    assert seen["images"][0] is still
    assert seen["images"][1] is frame_a
    assert seen["images"][2] is frame_b
    assert seen["content"] == [
        {"type": "text", "text": "Compare "},
        {"type": "image"},
        {"type": "text", "text": "[video frame 1] "},
        {"type": "image"},
        {"type": "text", "text": "\n[video frame 2] "},
        {"type": "image"},
    ]
    assert prepared["_mlx2_media_token_end"] == 5
    assert adapter.has_trusted_media_preparation(prepared)


def test_indexed_scatter_needs_serving_verified_cold_span():
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter, _forward_smol

    adapter = object.__new__(SmolVLM2CandidateAdapter)
    adapter.model = SimpleNamespace(indexed_scatter=True)
    adapter.identity = {"config": {"image_token_id": 42}}
    adapter._vision_feature_reuse_enabled = False
    adapter._media_proof_key = b"test-key"
    ids = [7, 42, 9, 42, 42, 11]
    request = {"_mlx2_prompt_tokens": ids,
               "_mlx2_media_token_end": 5,
               "_mlx2_media_fingerprint": "test-media"}
    request["_mlx2_media_proof"] = adapter._media_proof(ids, "test-media", 5)
    payload = {
        "image_token_positions": (1, 3, 4),
        "_mlx2_smol_positions_verified": True,
    }
    full = adapter.validate_prefill_inputs(request, ids, payload)
    assert full["_mlx2_smol_positions_verified"] is True
    assert full is not payload
    # A prefix hit before the media boundary shifts the remaining chunk.
    shifted = adapter.validate_prefill_inputs(request, ids[1:], payload)
    assert "_mlx2_smol_positions_verified" not in shifted
    wrong = adapter.validate_prefill_inputs(
        request, ids, {"image_token_positions": (1, 4, 3)}
    )
    assert "_mlx2_smol_positions_verified" not in wrong

    model = _ScatterSource()
    stats = {"engagements": 0, "refusals": 0}
    _forward_smol(
        model, np.array([ids]), pixel_values=np.ones((3, 3), np.float32),
        cache=[_Cache()], image_token_positions=shifted["image_token_positions"],
        indexed_scatter=True, scatter_stats=stats,
        attention_mask_factory=lambda *_: None,
    )
    assert model.source_scatter_calls == 1
    assert stats == {"engagements": 0, "refusals": 1}


def test_scatter_selection_is_in_execution_route_identity():
    from mlx2.adapters.smolvlm2 import SmolVLM2CandidateAdapter

    adapter = object.__new__(SmolVLM2CandidateAdapter)
    adapter.model = SimpleNamespace(indexed_scatter=True)
    adapter._vision_feature_reuse_enabled = False
    config = adapter.execution_config(max_lanes=1, prefill_step=128)
    assert config["smol_image_scatter"] == "indexed_v1"
    adapter.model.indexed_scatter = False
    config = adapter.execution_config(max_lanes=1, prefill_step=128)
    assert config["smol_image_scatter"] == "source"
