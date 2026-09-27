"""Fake-source, import-guarded tests for exact-order vision feature reuse."""

import importlib.abc
import sys
import unittest

import numpy as np


class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError(f"real MLX import forbidden: {fullname}")
        return None


sys.meta_path.insert(0, BlockMLX())
from mlx2.adapters.multimodal import MediaFeatureCache  # noqa: E402
from mlx2.adapters.vision_feature_reuse import (  # noqa: E402
    VisionFeatureCertificate, certify_cold_prefill, install_qwen25, install_smol,
    tower_input_digest,
)


class Features:
    def __init__(self, rows):
        self.rows = tuple(rows)
        self.shape = (len(self.rows), 2)
        self.nbytes = len(self.rows) * 4


class Pixels:
    def __init__(self, rows):
        self.rows = tuple(rows)

    def astype(self, dtype):
        return Pixels(self.rows)


class QwenSource:
    def __init__(self):
        self.vision_calls = 0
        self.vision_tower = self
        self.patch_embed = self
        self.proj = self
        self.weight = self
        self.dtype = "bf16"

    def __call__(self, pixels, grid, *, output_hidden_states):
        self.vision_calls += 1
        assert output_hidden_states is False
        assert grid == (1, 2, 3)
        return Features(pixels.rows)

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        assert "_mlx2_vision_feature_certificate" not in kwargs
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos")
        if pixel_values is None:
            return (tuple(input_ids), None)
        features = kwargs.get("cached_image_features")
        if features is None:
            grid = kwargs.get("image_grid_thw")
            if grid is None:
                grid = kwargs["video_grid_thw"]
            features = self.vision_tower(
                pixel_values.astype(self.weight.dtype), grid,
                output_hidden_states=False,
            )
        return (tuple(input_ids), features.rows)


class SmolSource:
    def __init__(self):
        self.vision_calls = 0

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        assert "_mlx2_vision_feature_certificate" not in kwargs
        if pixel_values is None:
            return (tuple(input_ids), None)
        features = kwargs.get("cached_image_features")
        if features is None:
            self.vision_calls += 1
            # Fake source connector produces feature rows in its own order.
            features = Features(f"projected:{row}" for row in pixel_values.rows)
        return self._prepare_inputs_for_multimodal(features, tuple(input_ids), input_ids)

    def _prepare_inputs_for_multimodal(self, features, embeds, input_ids):
        return (embeds, features.rows)


def cert(family="qwen2_5_vl", *, media="media-1", adaptation="base", tokens=None):
    return VisionFeatureCertificate.from_prepared(
        family=family, artifact_fingerprint="artifact-1",
        source_revision="pinned-source", processor_fingerprint="processor-1",
        adaptation_fingerprint=adaptation, media_fingerprint=media,
        prompt_tokens=[10, 20, 30] if tokens is None else tokens,
    )


class VisionFeatureReuseTest(unittest.TestCase):
    def test_tower_digest_exact_inputs_and_cpu_budget(self):
        pixels = np.arange(12, dtype=np.float32).reshape(2, 6)
        qwen = {"pixel_values": pixels, "image_grid_thw": np.array([[1, 2, 3]])}
        base = tower_input_digest("qwen2_5_vl", qwen)
        self.assertEqual(len(base), 64)
        self.assertEqual(base, tower_input_digest("qwen2_5_vl", dict(qwen)))
        self.assertNotEqual(base, tower_input_digest("qwen2_5_vl", {
            **qwen, "image_grid_thw": np.array([[1, 3, 2]])}))
        self.assertNotEqual(base, tower_input_digest("qwen2_5_vl", {
            **qwen, "pixel_values": pixels[::-1]}))
        self.assertIsNone(tower_input_digest("qwen2_5_vl", qwen, max_bytes=16))
        self.assertIsNone(tower_input_digest("qwen2_5_vl", {
            **qwen, "pixel_values_videos": pixels}))
        self.assertIsNone(tower_input_digest("qwen2_5_vl", {
            **qwen, "image_grid_thw": object()}))
        video = {"pixel_values_videos": pixels, "video_grid_thw": np.array([[2, 2, 3]])}
        self.assertNotEqual(base, tower_input_digest("qwen2_5_vl", video))
        self.assertNotEqual(tower_input_digest("qwen2_5_vl", video),
                            tower_input_digest("qwen2_5_vl", {
                                **video, "video_grid_thw": np.array([[2, 3, 2]])}))
        smol = {"pixel_values": pixels,
                "pixel_attention_mask": np.array([[1, 0]], dtype=np.bool_)}
        self.assertNotEqual(tower_input_digest("smolvlm", smol),
                            tower_input_digest("smolvlm", {
                                **smol, "pixel_attention_mask": np.array([[0, 1]], dtype=np.bool_)}))

    def test_pinned_processor_mlx_arrays_are_bounded_before_host_sync(self):
        calls = []
        def make_mlx(array, *, reported_bytes=None, reported_shape=None):
            def host(value, dtype=None):
                calls.append("materialized")
                return array
            cls = type("array", (), {
                "__module__": "mlx.core",
                "__array__": host,
            })
            value = cls()
            value.shape = tuple(array.shape) if reported_shape is None else reported_shape
            value.dtype = str(array.dtype)
            value.nbytes = int(array.nbytes if reported_bytes is None else reported_bytes)
            return value

        pixels = np.arange(12, dtype=np.float32).reshape(2, 6)
        grid = np.array([[1, 2, 3]], dtype=np.int64)
        mlx_values = {"pixel_values": make_mlx(pixels),
                      "image_grid_thw": make_mlx(grid)}
        digest = tower_input_digest("qwen2_5_vl", mlx_values)
        self.assertEqual(len(digest), 64)
        self.assertEqual(calls, ["materialized", "materialized"])
        calls.clear()
        self.assertIsNone(tower_input_digest("qwen2_5_vl", mlx_values,
                                             max_bytes=pixels.nbytes - 1))
        self.assertEqual(calls, [])
        bad_shape = {**mlx_values, "pixel_values": make_mlx(
            pixels, reported_shape=(1, 12))}
        self.assertIsNone(tower_input_digest("qwen2_5_vl", bad_shape))
        self.assertEqual(calls, ["materialized"])
        calls.clear()
        bad_bytes = {**mlx_values, "pixel_values": make_mlx(
            pixels, reported_bytes=1)}
        self.assertIsNone(tower_input_digest("qwen2_5_vl", bad_bytes,
                                             max_bytes=40))
        self.assertEqual(calls, ["materialized"])

    def test_tower_only_reuses_across_trailing_prompts_but_remerges(self):
        model = QwenSource()
        cache = MediaFeatureCache(max_entries=1, max_bytes=16)
        counters = install_qwen25(model, cache, artifact_fingerprint="artifact-1",
                                  source_revision="pinned-source",
                                  processor_fingerprint="processor-1", evaluate=lambda _: None)
        digest = tower_input_digest("qwen2_5_vl", {
            "pixel_values": np.arange(4, dtype=np.float32),
            "image_grid_thw": np.array([[1, 2, 3]]),
        })
        def prepared(tokens, *, media="same-media", tower=digest, lora=None):
            request = {"_mlx2_prompt_tokens": tokens, "_mlx2_media_token_end": 2,
                       "_mlx2_media_fingerprint": media,
                       "_mlx2_tower_inputs_digest": tower,
                       "_mlx2_lora_fingerprint": lora}
            return certify_cold_prefill(
                request, tokens, {"pixel_values": Pixels(["frame-a"])},
                family="qwen2_5_vl", artifact_fingerprint="artifact-1",
                source_revision="pinned-source", processor_fingerprint="processor-1",
                attest_prepared=lambda value: value is request,
                scope="tower_inputs_v1",
            )
        first = prepared([10, 20, 30])
        second = prepared([10, 20, 31])
        kwargs = {"image_grid_thw": (1, 2, 3)}
        a = model.get_input_embeddings([10, 20, 30], **first, **kwargs)
        b = model.get_input_embeddings([10, 20, 31], **second, **kwargs)
        self.assertEqual(a[1], b[1])
        self.assertNotEqual(a[0], b[0])
        self.assertEqual(model.vision_calls, 1)
        self.assertEqual((counters.misses, counters.hits), (1, 1))
        changed = prepared([10, 20, 32], tower="a" * 64)
        model.get_input_embeddings([10, 20, 32], **changed, **kwargs)
        self.assertEqual(model.vision_calls, 2)
        self.assertEqual(cache.snapshot()["entries"], 1)
        self.assertEqual(cache.snapshot()["evictions"], 1)
        refused = prepared([10, 20, 33], lora="selected-lora")
        self.assertNotIn("_mlx2_vision_feature_certificate", refused)

    def test_smol_tower_only_reuses_post_connector_features(self):
        model = SmolSource()
        cache = MediaFeatureCache(max_entries=2, max_bytes=16)
        counters = install_smol(model, cache, artifact_fingerprint="artifact-1",
                                source_revision="pinned-source",
                                processor_fingerprint="processor-1", evaluate=lambda _: None)
        digest = tower_input_digest("smolvlm", {
            "pixel_values": np.arange(6, dtype=np.float32),
            "pixel_attention_mask": np.array([1, 1, 0], dtype=np.bool_),
        })
        def certificate(tokens):
            request = {"_mlx2_prompt_tokens": tokens, "_mlx2_media_token_end": 2,
                       "_mlx2_media_fingerprint": "same-media",
                       "_mlx2_tower_inputs_digest": digest}
            return certify_cold_prefill(
                request, tokens, {"pixel_values": Pixels(["tile-a"])},
                family="smolvlm", artifact_fingerprint="artifact-1",
                source_revision="pinned-source", processor_fingerprint="processor-1",
                attest_prepared=lambda value: value is request,
                scope="tower_inputs_v1",
            )
        first = model.get_input_embeddings([10, 20, 30], **certificate([10, 20, 30]))
        second = model.get_input_embeddings([10, 20, 31], **certificate([10, 20, 31]))
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(model.vision_calls, 1)
        self.assertEqual((counters.misses, counters.hits), (1, 1))

    def test_trusted_cold_prefill_certificate_and_partial_refusal(self):
        request = {"_mlx2_prompt_tokens": [10, 20, 30],
                   "_mlx2_media_token_end": 2,
                   "_mlx2_media_fingerprint": "ordered-media"}
        input_values = {"pixel_values": Pixels(["frame-a"]),
                        "_mlx2_vision_feature_verified": True}
        args = dict(family="smolvlm", artifact_fingerprint="artifact-1",
                    source_revision="pinned-source", processor_fingerprint="processor-1",
                    attest_prepared=lambda value: value is request)
        full = certify_cold_prefill(request, [10, 20, 30], input_values, **args)
        self.assertTrue(full["_mlx2_vision_feature_verified"])
        self.assertEqual(full["_mlx2_vision_feature_certificate"].adaptation_fingerprint,
                         "base")
        self.assertEqual(full["_mlx2_vision_feature_certificate"].media_fingerprint,
                         "ordered-media")
        partial = certify_cold_prefill(request, [20, 30], input_values, **args)
        self.assertNotIn("_mlx2_vision_feature_verified", partial)
        self.assertNotIn("_mlx2_vision_feature_certificate", partial)
        forged = certify_cold_prefill(request, [10, 20, 30], input_values,
                                     **{**args, "attest_prepared": None})
        self.assertNotIn("_mlx2_vision_feature_verified", forged)
        at_anchor = certify_cold_prefill(
            {**request, "_mlx2_media_token_end": 3}, [10, 20, 30], input_values,
            **{**args, "attest_prepared": lambda _: True})
        self.assertNotIn("_mlx2_vision_feature_verified", at_anchor)
        selected_lora = certify_cold_prefill(
            {**request, "_mlx2_lora_fingerprint": "lora-revision-1"},
            [10, 20, 30], input_values,
            **{**args, "attest_prepared": lambda _: True},
        )
        self.assertNotIn("_mlx2_vision_feature_verified", selected_lora)

    def test_certificate_rejects_ambiguous_identity(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            VisionFeatureCertificate.from_prepared(
                family="smolvlm", artifact_fingerprint="a", source_revision="r",
                processor_fingerprint="p", adaptation_fingerprint="",
                media_fingerprint="m", prompt_tokens=[1],
            )
        with self.assertRaisesRegex(ValueError, "tokens"):
            cert(tokens=[True])
        self.assertNotEqual(cert(tokens=[1, 2]), cert(tokens=[2, 1]))
        self.assertNotEqual(cert(adaptation="base"), cert(adaptation="lora-1"))

    def test_qwen_order_hit_and_identity_refusals(self):
        model = QwenSource()
        cache = MediaFeatureCache(max_entries=3, max_bytes=100)
        evaluated = []
        counters = install_qwen25(
            model, cache, artifact_fingerprint="artifact-1",
            source_revision="pinned-source", processor_fingerprint="processor-1",
            evaluate=lambda value: evaluated.append(value.rows),
        )
        pixels = Pixels(["frame-a", "frame-b"])
        kwargs = {"_mlx2_vision_feature_certificate": cert(),
                  "_mlx2_vision_feature_verified": True,
                  "image_grid_thw": (1, 2, 3)}
        first = model.get_input_embeddings([10, 20, 30], pixels, **kwargs)
        second = model.get_input_embeddings([10, 20, 30], pixels, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first[1], ("frame-a", "frame-b"))
        self.assertEqual(model.vision_calls, 1)
        self.assertEqual(evaluated, [("frame-a", "frame-b")])
        self.assertEqual((counters.misses, counters.hits, counters.stores), (1, 1, 1))

        # Warm APCv2 replay and ordinary batched decode are text-only calls.
        # They must not inflate media feature refusal telemetry.
        for _ in range(2):
            self.assertEqual(model.get_input_embeddings([31, 32]), ((31, 32), None))
        self.assertEqual(counters.refusals, 0)
        self.assertEqual(model.vision_calls, 1)

        # Missing trusted handoff and different ordered media both use source.
        model.get_input_embeddings([10, 20, 30], pixels,
                                   _mlx2_vision_feature_certificate=cert(),
                                   image_grid_thw=(1, 2, 3))
        self.assertEqual(model.vision_calls, 2)
        model.get_input_embeddings(
            [10, 20, 30], Pixels(["frame-b", "frame-a"]),
            **{**kwargs, "_mlx2_vision_feature_certificate": cert(media="media-2")},
        )
        self.assertEqual(model.vision_calls, 3)
        self.assertEqual(counters.refusals, 1)
        video_kwargs = {
            "_mlx2_vision_feature_certificate": cert(media="ordered-video"),
            "_mlx2_vision_feature_verified": True,
            "pixel_values_videos": Pixels(["video-frame-1", "video-frame-2"]),
            "video_grid_thw": (1, 2, 3),
        }
        video_first = model.get_input_embeddings([10, 20, 30], **video_kwargs)
        video_second = model.get_input_embeddings([10, 20, 30], **video_kwargs)
        self.assertEqual(video_first, video_second)
        self.assertEqual(video_first[1], ("video-frame-1", "video-frame-2"))
        self.assertEqual(model.vision_calls, 4)
        with self.assertRaisesRegex(ValueError, "already installed"):
            install_qwen25(model, cache, artifact_fingerprint="artifact-1",
                           source_revision="pinned-source",
                           processor_fingerprint="processor-1")

    def test_smol_captures_post_connector_features_in_source_order(self):
        model = SmolSource()
        cache = MediaFeatureCache(max_entries=3, max_bytes=100)
        evaluated = []
        counters = install_smol(
            model, cache, artifact_fingerprint="artifact-1",
            source_revision="pinned-source", processor_fingerprint="processor-1",
            evaluate=lambda value: evaluated.append(value.rows),
        )
        pixels = Pixels(["tile-2", "tile-1"])
        kwargs = {"_mlx2_vision_feature_certificate": cert("smolvlm"),
                  "_mlx2_vision_feature_verified": True}
        first = model.get_input_embeddings([10, 20, 30], pixels, **kwargs)
        second = model.get_input_embeddings([10, 20, 30], pixels, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first[1], ("projected:tile-2", "projected:tile-1"))
        self.assertEqual(model.vision_calls, 1)
        self.assertEqual(evaluated, [first[1]])
        self.assertEqual((counters.misses, counters.hits, counters.stores), (1, 1, 1))

        for _ in range(2):
            self.assertEqual(model.get_input_embeddings([31, 32]), ((31, 32), None))
        self.assertEqual(counters.refusals, 0)
        self.assertEqual(model.vision_calls, 1)

        # A failed source merge cannot publish features under this key.
        class FailingSource(SmolSource):
            def _prepare_inputs_for_multimodal(self, features, embeds, input_ids):
                raise ValueError("source placeholder mismatch")

        failing = FailingSource()
        failing_cache = MediaFeatureCache()
        install_smol(failing, failing_cache, artifact_fingerprint="artifact-1",
                     source_revision="pinned-source",
                     processor_fingerprint="processor-1", evaluate=lambda _: None)
        with self.assertRaisesRegex(ValueError, "placeholder"):
            failing.get_input_embeddings([10], pixels, **kwargs)
        self.assertEqual(failing_cache.snapshot()["entries"], 0)


if __name__ == "__main__":
    unittest.main()
