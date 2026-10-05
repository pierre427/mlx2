"""Import-guarded LFM2.5-VL target and DSpark artifact contracts."""

import json
import sys
import tempfile
import types
import unittest
from mlx_blocker import install_for_test_case
from pathlib import Path
from unittest.mock import patch



from mlx2.adapters.lfm25_vl import (  # noqa: E402
    LFM25_VL, LFM25VLAdapter, _LFMLogitsModel, generate_candidate_dspark,
    inspect_artifact, inspect_dspark_artifact,
)
from mlx2.adapters.lfm25_memory import LFM25CacheBudget  # noqa: E402
from mlx2.contracts import Capability  # noqa: E402
from mlx2.serving import adapter_media_checkpoint_position  # noqa: E402
import mlx2.adapters.lfm25_vl as lfm_module  # noqa: E402


def write_weights(path, shapes):
    header = {name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
              for name, shape in shapes.items()}
    raw = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw)


class LFM25VLCPUTest(unittest.TestCase):
    def setUp(self):
        install_for_test_case(self)

    def make_target(self, root):
        path = root / "target"
        path.mkdir()
        layers = ["full_attention" if i in {2, 5, 9, 13, 17, 21, 24, 27} else "conv" for i in range(30)]
        config = {
            "model_type": "lfm2_vl", "architectures": ["Lfm2VlForConditionalGeneration"],
            "image_token_id": 124907, "downsample_factor": 2,
            "projector_hidden_size": 2048, "eos_token_id": 124900,
            "text_config": {"num_hidden_layers": 30, "hidden_size": 2048,
                            "num_attention_heads": 32, "num_key_value_heads": 8,
                            "vocab_size": 128000, "conv_L_cache": 3,
                            "max_position_embeddings": 32768, "layer_types": layers},
            "vision_config": {"num_hidden_layers": 27, "hidden_size": 1152,
                              "num_attention_heads": 16, "patch_size": 16},
        }
        (path / "config.json").write_text(json.dumps(config))
        shapes = {
            "model.language_model.embed_tokens.weight": [128000, 2048],
            "model.multi_modal_projector.linear_1.weight": [2048, 4608],
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight": [1152, 768],
        }
        for i, kind in enumerate(layers):
            shapes[f"model.language_model.layers.{i}.{'self_attn.q_proj.weight' if kind == 'full_attention' else 'conv.conv.weight'}"] = [1]
        write_weights(path, shapes)
        return path, config

    def make_draft(self, root):
        path = root / "draft"
        path.mkdir()
        config = {
            "architectures": ["Lfm2DSparkDraftModel"], "model_type": "qwen3",
            "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 64, "vocab_size": 128000,
            "block_size": 9, "markov_rank": 256, "enable_confidence_head": True,
            "dflash_config": {"target_layer_ids": [2, 9, 17, 21, 27],
                              "num_target_layers": 30, "mask_token_id": 125017},
        }
        (path / "config.json").write_text(json.dumps(config))
        shapes = {
            "fc.weight": [2048, 10240], "markov_head.markov_w1.weight": [128000, 256],
            "markov_head.markov_w2.weight": [128000, 256],
            "confidence_head.proj.weight": [1, 2304],
        }
        for i in range(4):
            shapes[f"layers.{i}.self_attn.q_proj.weight"] = [1]
        write_weights(path, shapes)
        return path, config

    def test_target_and_draft_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, _ = self.make_target(root)
            draft, _ = self.make_draft(root)
            t = inspect_artifact(target)
            d = inspect_dspark_artifact(draft, target=t)
            self.assertEqual(d["target_fingerprint"], t["fingerprint"])
            self.assertTrue(d["auxiliary"])
            self.assertFalse(d["selected"])
            self.assertFalse(d["qualified"])
            self.assertNotIn(Capability.MTP, LFM25_VL.capabilities)
            self.assertIn(Capability.APC_V2, LFM25_VL.capabilities)
            self.assertIn(Capability.PREFIX_REUSE, LFM25_VL.capabilities)
            self.assertIn(Capability.VIDEO, LFM25_VL.capabilities)
            self.assertNotIn(Capability.CONTINUOUS_BATCH, LFM25_VL.capabilities)
            self.assertEqual(LFM25VLAdapter.default_route, "ordinary")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                generate_candidate_dspark(target, draft, " ")
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                generate_candidate_dspark(target, draft, "hello", max_tokens=513)
            self.assertNotIn("mlx", sys.modules)

    def test_unqualified_mixed_length_batch_is_rejected(self):
        adapter = object.__new__(LFM25VLAdapter)
        with self.assertRaisesRegex(ValueError, "max_lanes=1"):
            adapter.execution_config(max_lanes=2, prefill_step=2048)
        self.assertEqual(adapter.execution_config(max_lanes=1, prefill_step=2048)[
            "segment_aware_cohort_size"], 1)
        self.assertEqual(adapter.execution_config(max_lanes=1, prefill_step=2048)[
            "lfm_media_checkpoint"], "off")
        adapter.media_checkpoint_enabled = True
        self.assertEqual(adapter.execution_config(max_lanes=1, prefill_step=2048)[
            "lfm_media_checkpoint"], "candidate_v1")

    def test_default_route_and_unqualified_capabilities(self):
        adapter = object.__new__(LFM25VLAdapter)
        self.assertEqual(adapter.default_route, "ordinary")
        self.assertEqual(adapter.execution_config(max_lanes=1, prefill_step=256)[
            "lfm_shortconv"], "source")
        with self.assertRaisesRegex(ValueError, "speculative route"):
            adapter.profile_name(True)
        with self.assertRaisesRegex(ValueError, "tool calling"):
            adapter.output_parser({"tools": [{"type": "function"}]})
        self.assertEqual(adapter.profile_name(False), "lfm25-vl-ordinary-candidate")

    def test_generic_media_boundary_rejects_an_adapter_outside_media_end(self):
        tokens = [10, 124907, 11, 12]
        request = {"_mlx2_media_token_end": 2}
        bad = types.SimpleNamespace(
            apc_media_checkpoint_position=lambda *_args, **_kwargs: 3
        )
        with self.assertRaisesRegex(ValueError, "boundary is invalid"):
            adapter_media_checkpoint_position(
                bad, request, tokens, cached_tokens=0
            )
        self.assertIsNone(adapter_media_checkpoint_position(
            types.SimpleNamespace(), request, tokens, cached_tokens=0
        ))

    def test_media_prefill_boundary_excludes_decode_anchor(self):
        import numpy as np

        image_token = 124907

        class Processor:
            image_token = "<image>"
            tokenizer = property(lambda self: self)
            def apply_chat_template(self, messages, **kwargs):
                return "prompt"
            def __call__(self, **kwargs):
                return {"input_ids": np.asarray([self.ids]),
                        "pixel_values": np.zeros((1, 1), dtype=np.float32)}

        processor = Processor()
        processor.ids = [10, image_token, 11, 12]
        adapter = object.__new__(LFM25VLAdapter)
        adapter.identity = {"image_token_id": image_token}
        adapter.processor = processor
        adapter._media_proof_key = b"cpu-test-media-key"
        adapter.media_checkpoint_enabled = True
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.__path__ = []
        fake_core = types.ModuleType("mlx.core")
        fake_core.array = lambda value: value
        fake_mlx.core = fake_core
        request = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": "image"}]}]}
        media = types.SimpleNamespace(value="decoded")
        with patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}), \
                patch.object(lfm_module, "resolve_media", return_value=media), \
                patch.object(lfm_module, "media_fingerprint", return_value="media-key"):
            ready = adapter.prepare_multimodal_request(request)
            self.assertEqual(ready["_mlx2_media_token_end"], 2)
            self.assertEqual(ready["_mlx2_media_fingerprint"], "media-key")
            self.assertIn("pixel_values", ready["_mlx2_prefill_inputs"])
            tokens = ready["_mlx2_prompt_tokens"]
            self.assertEqual(adapter_media_checkpoint_position(
                adapter, ready, tokens, cached_tokens=0), 2)
            self.assertIsNone(adapter_media_checkpoint_position(
                adapter, ready, tokens, cached_tokens=2))
            no_suffix = {**ready, "_mlx2_prompt_tokens": tokens[:-1]}
            no_suffix["_mlx2_media_proof"] = adapter._media_checkpoint_proof(
                tokens[:-1], "media-key", 2
            )
            self.assertIsNone(adapter_media_checkpoint_position(
                adapter, no_suffix, tokens[:-1], cached_tokens=0))
            forged = {**ready, "_mlx2_media_token_end": 1}
            with self.assertRaisesRegex(ValueError, "not bound"):
                adapter_media_checkpoint_position(
                    adapter, forged, tokens, cached_tokens=0)
            forged = {**ready, "_mlx2_prompt_tokens": [10, image_token, 99, 12]}
            with self.assertRaisesRegex(ValueError, "not bound"):
                adapter_media_checkpoint_position(
                    adapter, forged, tokens, cached_tokens=0)
            processor.ids = [10, image_token]
            with self.assertRaisesRegex(ValueError, "final prompt token"):
                adapter.prepare_multimodal_request(request)
        self.assertNotIn("mlx", sys.modules)

    def test_checkpoint_chat_template_can_live_on_tokenizer(self):
        calls = []
        tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda messages, **kwargs: calls.append((messages, kwargs)) or "framed"
        )
        adapter = object.__new__(LFM25VLAdapter)
        adapter.processor = types.SimpleNamespace(tokenizer=tokenizer)
        messages = [{"role": "user", "content": "hello"}]
        self.assertEqual(adapter._apply_chat_template(messages), "framed")
        self.assertEqual(calls, [(messages, {"tokenize": False, "add_generation_prompt": True})])

    def test_video_frames_use_ordered_images_and_media_boundary(self):
        import numpy as np

        image_token = 124907
        frame_a = np.zeros((2, 3, 3), dtype=np.uint8)
        frame_b = np.ones((2, 3, 3), dtype=np.uint8)

        class Processor:
            image_token = "<image>"
            tokenizer = property(lambda self: self)

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return messages[0]["content"]

            def __call__(self, **kwargs):
                self.images = kwargs["images"]
                self.text = kwargs["text"]
                return {"input_ids": np.asarray([[10, image_token, image_token, 11, 12]]),
                        "pixel_values": np.zeros((2, 1), dtype=np.float32)}

        processor = Processor()
        adapter = object.__new__(LFM25VLAdapter)
        adapter.identity = {"image_token_id": image_token}
        adapter.processor = processor
        adapter._media_proof_key = b"cpu-test-media-key"
        adapter.media_checkpoint_enabled = True
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.__path__ = []
        fake_core = types.ModuleType("mlx.core")
        fake_core.array = lambda value: value
        fake_mlx.core = fake_core
        request = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Summarize the clip"},
            {"type": "input_video", "video_url": "video"},
        ]}]}
        media = types.SimpleNamespace(
            value=[frame_a, frame_b],
            metadata={"timestamps_seconds": (0.0, 1.0)},
        )
        with patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}), \
             patch.object(lfm_module, "resolve_media", return_value=media) as resolve, \
             patch.object(lfm_module, "media_fingerprint", return_value="video-key") as fingerprint:
            ready = adapter.prepare_multimodal_request(request)
            self.assertEqual(ready["_mlx2_media_token_end"], 3)
            self.assertEqual(ready["_mlx2_media_fingerprint"], "video-key")
            self.assertEqual(adapter_media_checkpoint_position(
                adapter, ready, ready["_mlx2_prompt_tokens"], cached_tokens=0), 3)
            self.assertIs(processor.images[0], frame_a)
            self.assertIs(processor.images[1], frame_b)
            self.assertIn("[video frame 1/2 at 0.000s] <image>", processor.text)
            self.assertIn("[video frame 2/2 at 1.000s] <image>", processor.text)
            self.assertEqual(processor.text.count("<image>"), 2)
            self.assertEqual(resolve.call_args.kwargs["fps"], 1)
            self.assertEqual(resolve.call_args.kwargs["max_frames"], 16)
            self.assertEqual(fingerprint.call_args.kwargs["policy"]["video_fps"], 1)
            original_call = Processor.__call__
            def missing_frame_placeholder(self, **kwargs):
                result = original_call(self, **kwargs)
                result["input_ids"] = np.asarray([[10, image_token, 11, 12]])
                return result
            with patch.object(Processor, "__call__", missing_frame_placeholder):
                with self.assertRaisesRegex(ValueError, "placeholders do not match"):
                    adapter.prepare_multimodal_request(request)
        self.assertNotIn("mlx", sys.modules)

    def test_video_without_frames_is_rejected(self):
        adapter = object.__new__(LFM25VLAdapter)
        adapter.processor = types.SimpleNamespace(image_token="<image>")
        request = {"messages": [{"role": "user", "content": [
            {"type": "input_video", "video_url": "video"}]}]}
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.__path__ = []
        fake_core = types.ModuleType("mlx.core")
        fake_mlx.core = fake_core
        with patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_core}), \
             patch.object(lfm_module, "resolve_media", return_value=types.SimpleNamespace(value=[])):
            with self.assertRaisesRegex(ValueError, "no sampled frames"):
                adapter.prepare_multimodal_request(request)

    def test_lfm_cache_uses_checkpoint_aware_mlx2_types(self):
        class KV:
            pass

        class Arrays:
            def __init__(self, size):
                self.size = size

        layers = [types.SimpleNamespace(is_attention_layer=i in {2, 5, 9, 13, 17, 21, 24, 27})
                  for i in range(30)]
        wrapped = _LFMLogitsModel(types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=layers),
            config=types.SimpleNamespace(text_config=types.SimpleNamespace(
                layer_types=["full_attention" if layer.is_attention_layer else "conv"
                             for layer in layers]))))
        with patch.dict(sys.modules, {"mlx2.runtime.models.cache":
                                      types.SimpleNamespace(ArraysCache=Arrays, KVCache=KV)}):
            caches = wrapped.make_cache()
        self.assertEqual(len(caches), 30)
        self.assertEqual(sum(isinstance(c, KV) for c in caches), 8)
        self.assertEqual(sum(isinstance(c, Arrays) and c.size == 1 for c in caches), 22)
        self.assertEqual(wrapped.apc_v2_layout, LFM25_VL.cache_layout)

    def test_lfm_budget_charges_conv_snapshots_and_media_prefill(self):
        with tempfile.TemporaryDirectory() as tmp:
            target, config = self.make_target(Path(tmp))
            budget = LFM25CacheBudget.from_config(config["text_config"], mtp=False)
            self.assertEqual(budget.attention_layers, 8)
            self.assertEqual(budget.conv_layers, 22)
            self.assertGreater(budget.project(4096), budget.project(2048))
            self.assertGreater(budget.prefill_transient_bytes(4096, 4096), 2 << 30)
            self.assertIn("provisional", budget.as_dict()["workspace"])
            with self.assertRaisesRegex(ValueError, "DSpark"):
                LFM25CacheBudget.from_config(config["text_config"], mtp=True)
            self.assertTrue(target.exists())

    def test_rejects_topology_and_draft_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, config = self.make_target(root)
            draft, dconfig = self.make_draft(root)
            config["text_config"]["layer_types"][2] = "conv"
            (target / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "hybrid layer"):
                inspect_artifact(target)
            config["text_config"]["layer_types"][2] = "full_attention"
            (target / "config.json").write_text(json.dumps(config))
            dconfig["dflash_config"]["num_target_layers"] = 29
            (draft / "config.json").write_text(json.dumps(dconfig))
            with self.assertRaisesRegex(ValueError, "target binding"):
                inspect_dspark_artifact(draft, target=inspect_artifact(target))

    def test_offline_dspark_dispatch_uses_dflash_round_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, _ = self.make_target(root)
            draft, _ = self.make_draft(root)
            calls = {}
            model = types.SimpleNamespace(config={})
            def generate(*args, **kwargs):
                calls.update(kwargs)
                return types.SimpleNamespace(text="ok", finish_reason="stop")
            modules = {
                "mlx_vlm": types.SimpleNamespace(load=lambda *args, **kwargs: (model, object())),
                "mlx_vlm.generate": types.SimpleNamespace(generate=generate),
                "mlx_vlm.prompt_utils": types.SimpleNamespace(
                    apply_chat_template=lambda *args, **kwargs: "formatted"),
            }
            with patch.dict(sys.modules, modules), \
                    patch.object(lfm_module, "_require_source_revision"), \
                    patch.object(lfm_module, "load_candidate_dspark",
                                 return_value=(object(), {"fingerprint": "draft"})):
                result = generate_candidate_dspark(target, draft, "hello")
            self.assertEqual(result["text"], "ok")
            self.assertEqual(calls["draft_kind"], "dflash")
            self.assertEqual(calls["draft_block_size"], 10)
            self.assertNotIn("mlx", sys.modules)


if __name__ == "__main__":
    unittest.main()
