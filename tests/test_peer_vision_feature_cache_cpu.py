"""Exact-input media reuse, tenant isolation, and unchanged width-one calls."""

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.minicpmo_feature_reuse import cached_sequential_features, item_keys
from mlx2.adapters.mlx_vlm import MiniCPMOAdapter, _MLXVLMAdapter
from mlx2.adapters.multimodal import (
    MediaFeatureCache,
    MiniCPMOExecutionPolicy,
    install_media_feature_cache,
)
from mlx2.adapters.vision_feature_reuse import VisionFeatureCertificate, _key


class SequentialEncoder:
    """Pinned source-shaped encoder seam; scalar arithmetic is a CPU oracle."""

    def __init__(self):
        self.calls = []

    def get_vision_embedding(self, pixels, targets):
        self.calls.append(tuple(len(sample) for sample in pixels))
        result = []
        for sample, sizes in zip(pixels, targets):
            features = [
                mx.array([[float(value.sum()), float(size.prod())]])
                for value, size in zip(sample, sizes)
            ]
            result.append(mx.stack(features) if features else [])
        return result

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        assert not any(key.startswith("_mlx2_vision") for key in kwargs)
        features = kwargs.get("cached_image_features")
        if features is None:
            features = self.get_vision_embedding(pixel_values, kwargs["tgt_sizes"])
        # Preserve the source merge's ordering, rather than reconstructing
        # prompt placeholders in the cache implementation.
        return tuple(input_ids), [
            np.asarray(value).tolist() if len(value) else [] for value in features
        ]


def _inputs(values):
    pixels = [[np.full((3, 2, 2), value, dtype=np.float32) for value in values]]
    sizes = [np.tile(np.array([[1, 2]], dtype=np.int32), (len(values), 1))]
    return {
        "pixel_values": pixels,
        "tgt_sizes": sizes,
        "_mlx2_vision_cache_key": "whole-request",
    }


def _adapter(*, enabled=True):
    adapter = MiniCPMOAdapter.__new__(MiniCPMOAdapter)
    adapter.media_policy = MiniCPMOExecutionPolicy(per_item_feature_reuse=enabled)
    adapter.identity = {"fingerprint": "artifact-one"}
    adapter.mlx_vlm_runtime = {"source_sha256": "reviewed-source-one"}
    return adapter


def _validated(adapter, values, *, tenant="a", lora=None):
    request = {"_mlx2_feature_cache_tenant": tenant}
    if lora is not None:
        request["_mlx2_lora_fingerprint"] = lora
    return adapter.validate_prefill_inputs(request, [1, 2, 3], _inputs(values))


def test_append_and_reorder_reuse_only_exact_items_with_source_merge():
    model, reference = SequentialEncoder(), SequentialEncoder()
    cache = MediaFeatureCache(max_entries=256)
    install_media_feature_cache(model, cache, family="minicpmo", per_item_reuse=True)
    adapter = _adapter()
    for index, values in enumerate(([1, 2], [1, 2, 3], [3, 1, 2], [3, 4, 2])):
        inputs = _validated(adapter, values)
        actual = model.get_input_embeddings(input_ids=[10, 20], **inputs)
        raw = _inputs(values)
        raw.pop("_mlx2_vision_cache_key")
        expected = reference.get_input_embeddings(input_ids=[10, 20], **raw)
        assert actual == expected
        assert len(model.calls) == (2, 3, 3, 4)[index]
    assert model.calls == [(1,)] * 4
    assert cache.snapshot()["hits"] == 7
    assert cache.snapshot()["entries"] == 4


def test_repeated_item_is_encoded_once_and_retains_duplicate_positions():
    model = SequentialEncoder()
    install_media_feature_cache(
        model, MediaFeatureCache(), family="minicpmo", per_item_reuse=True
    )
    actual = model.get_input_embeddings(
        input_ids=[10], **_validated(_adapter(), [1, 1, 1])
    )
    assert len(model.calls) == 1
    assert actual[1][0] == [[[12.0, 2.0]]] * 3


def test_pinned_source_encoder_seam_has_exact_tensor_parity_and_width_one():
    module = pytest.importorskip("mlx_vlm.models.minicpmo.minicpmo")
    widths = []

    class Tower:
        embeddings = SimpleNamespace(
            patch_embedding=SimpleNamespace(weight=mx.array([1.0]))
        )

        def __call__(self, pixels, *, patch_attention_mask, tgt_sizes):
            widths.append(pixels.shape[0])
            assert tuple(patch_attention_mask.shape) == (1, 1, 2)
            return mx.sum(pixels, axis=(1, 2, 3)).reshape(1, 1, 1)

    source = SimpleNamespace(
        vision_tower=Tower(),
        resampler=lambda hidden, target: hidden,
        config=SimpleNamespace(patch_size=1),
    )
    source.get_vision_embedding = lambda pixels, sizes: (
        module.Model.get_vision_embedding(source, pixels, sizes)
    )
    data = _inputs([1, 2])
    expected = source.get_vision_embedding(data["pixel_values"], data["tgt_sizes"])
    keys = item_keys(
        data["pixel_values"], data["tgt_sizes"], namespace="reviewed-source"
    )
    cache = MediaFeatureCache()
    actual = cached_sequential_features(
        source, cache, data["pixel_values"], data["tgt_sizes"], keys, array_module=mx
    )
    assert np.array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
    assert widths == [1, 1, 1, 1]
    again = cached_sequential_features(
        source, cache, data["pixel_values"], data["tgt_sizes"], keys, array_module=mx
    )
    assert np.array_equal(np.asarray(again[0]), np.asarray(expected[0]))
    assert widths == [1, 1, 1, 1]


@pytest.mark.parametrize("change", ["tenant", "lora", "artifact", "source", "policy"])
def test_independent_identity_domains_do_not_share_features(change):
    adapter = _adapter()
    first = _validated(adapter, [1])["_mlx2_vision_item_keys"]
    args = {}
    if change == "tenant":
        args["tenant"] = "b"
    elif change == "lora":
        args["lora"] = "adaptation-two"
    elif change == "artifact":
        adapter.identity["fingerprint"] = "artifact-two"
    elif change == "source":
        adapter.mlx_vlm_runtime["source_sha256"] = "reviewed-source-two"
    else:
        adapter.media_policy = replace(adapter.media_policy, slice_mode=False)
    assert _validated(adapter, [1], **args)["_mlx2_vision_item_keys"] != first


@pytest.mark.parametrize("change", ["content", "target", "dtype", "shape"])
def test_exact_processor_content_and_geometry_bind_item_identity(change):
    original = _inputs([1])
    first = item_keys(
        original["pixel_values"], original["tgt_sizes"], namespace="source"
    )
    if change == "content":
        original["pixel_values"][0][0][0, 0, 0] = 2
    elif change == "target":
        original["tgt_sizes"][0][0] = [2, 1]
    elif change == "dtype":
        original["pixel_values"][0][0] = original["pixel_values"][0][0].astype(
            np.float64
        )
    else:
        original["pixel_values"][0][0] = original["pixel_values"][0][0].reshape(3, 1, 4)
    assert (
        item_keys(original["pixel_values"], original["tgt_sizes"], namespace="source")
        != first
    )


def test_hash_budget_and_foreign_representations_decline_reuse_without_conversion():
    data = _inputs([1])
    assert (
        item_keys(data["pixel_values"], data["tgt_sizes"], namespace="s", max_bytes=1)
        is None
    )

    class DeviceArray:
        def __array__(self, *args, **kwargs):
            pytest.fail("unsupported feature inputs must not materialize on host")

    assert item_keys([[DeviceArray()]], data["tgt_sizes"], namespace="s") is None


def test_adapter_replaces_forged_item_keys_and_declines_them_when_disabled():
    inputs = _inputs([1])
    inputs["_mlx2_vision_item_keys"] = (("forged",),)
    adapter = _adapter()
    result = adapter.validate_prefill_inputs({}, [1], inputs)
    assert "forged" not in repr(result["_mlx2_vision_item_keys"])
    inputs = _inputs([1])
    inputs["_mlx2_vision_item_keys"] = (("forged",),)
    assert "_mlx2_vision_item_keys" not in _adapter(
        enabled=False
    ).validate_prefill_inputs({}, [1], inputs)


def test_default_and_operator_policy_keep_reuse_sequential_and_explicit():
    default = MiniCPMOExecutionPolicy.from_config({"per_item_feature_reuse": True})
    assert default.per_item_feature_reuse is False
    assert "per_item_feature_reuse" not in default.receipt()
    enabled = MiniCPMOExecutionPolicy.from_config(
        {}, media_options={"per_item_feature_reuse": True}
    )
    assert enabled.receipt()["per_item_feature_reuse"] is True
    with pytest.raises(ValueError, match="sequential"):
        MiniCPMOExecutionPolicy.from_config(
            {},
            media_options={
                "per_item_feature_reuse": True,
                "batch_vision_input": True,
                "vision_batch_size": 2,
            },
        )
    with pytest.raises(ValueError, match="boolean"):
        MiniCPMOExecutionPolicy.from_config(
            {}, media_options={"per_item_feature_reuse": 1}
        )


def test_existing_whole_feature_keys_are_tenant_and_lora_bound():
    adapter = _MLXVLMAdapter.__new__(_MLXVLMAdapter)

    def scoped(tenant, lora=None):
        return adapter.validate_prefill_inputs(
            {"_mlx2_feature_cache_tenant": tenant, "_mlx2_lora_fingerprint": lora},
            [],
            {name: "same" for name in adapter.VISION_FEATURE_CACHE_KEYS},
        )

    assert scoped("a") == scoped("a")
    assert scoped("a") != scoped("b")
    assert scoped("a") != scoped("a", "adaptation")


@pytest.mark.parametrize("scope", ["prompt_v1", "tower_inputs_v1"])
def test_qwen_smol_certificates_bind_tenant_for_both_cache_scopes(scope):
    cert = VisionFeatureCertificate.from_prepared(
        family="qwen2_5_vl",
        artifact_fingerprint="artifact",
        source_revision="source",
        processor_fingerprint="processor",
        adaptation_fingerprint="base",
        media_fingerprint="media",
        prompt_tokens=[1, 2, 3],
        scope=scope,
        tower_sha256="a" * 64,
        tenant_scope="tenant-a",
    )
    args = {
        "family": "qwen2_5_vl",
        "artifact_fingerprint": "artifact",
        "source_revision": "source",
        "processor_fingerprint": "processor",
        "model_instance": 1,
        "verified": True,
    }
    assert _key(cert, **args) != _key(replace(cert, tenant_scope="tenant-b"), **args)
