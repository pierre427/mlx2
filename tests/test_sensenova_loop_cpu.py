"""Loop visits own independent state while sharing the physical parameters."""

import unittest
from types import SimpleNamespace

import mlx.core as mx

from mlx2.adapters.registry import AdapterResolution
from mlx2.adapters.sensenova_loop import (
    CACHE_LAYOUT,
    DESCRIPTOR,
    SenseNovaLoopAdapter,
    execution_plan,
)
from mlx2.runtime.models.cache import KVCache, make_prompt_cache
from mlx2.runtime.models.sensenova_loop import (
    Model,
    _sinkhorn_reference,
    _sinkhorn_two_streams,
)


def _config():
    return {
        "model_type": "sensenova_loop",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 32,
            "intermediate_size": 48,
            "vocab_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "num_hidden_layers": 6,
            "full_attention_interval": 2,
            "max_position_embeddings": 128,
            "layer_types": ["linear_attention", "full_attention"] * 3,
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 10000,
                "factor": 2,
                "original_max_position_embeddings": 64,
            },
        },
        "loop_settings": {"start": 2, "end": 4, "eval_count": 2, "counts": [2]},
        "mhc": {
            "start": 2,
            "end": 4,
            "streams": 2,
            "iterations": 20,
            "fixed_identity": False,
            "pre_scale": 1.0,
        },
    }


class SenseNovaLoopTest(unittest.TestCase):
    def test_server_defaults_to_compiled_mhc_and_preserves_reference_override(self):
        from mlx2.server import resolve_execution_policy_defaults

        route = SimpleNamespace(route="ordinary", native_mtp=False)
        adapter = AdapterResolution(SenseNovaLoopAdapter, DESCRIPTOR, {})
        default = resolve_execution_policy_defaults(None, route, adapter)
        override = resolve_execution_policy_defaults(
            {"compiled_mhc": False}, route, adapter
        )
        self.assertIs(default["compiled_mhc"], True)
        self.assertIs(override["compiled_mhc"], False)

    def test_loop_plan_refuses_mhc_crossing_a_visit_boundary(self):
        config = _config()
        self.assertEqual(execution_plan(config), (0, 1, 2, 3, 2, 3, 4, 5))
        config["mhc"]["end"] = 5
        with self.assertRaisesRegex(ValueError, "two-stream mHC"):
            execution_plan(config)

    def test_compiled_sinkhorn_matches_reference_across_shapes(self):
        with mx.stream(mx.cpu):
            for shape in ((1, 1, 4), (2, 7, 4)):
                logits = mx.random.normal(shape, key=mx.random.key(27)) * 20
                expected = _sinkhorn_reference(logits.reshape(*shape[:-1], 2, 2), 20)
                actual = _sinkhorn_two_streams(logits, 20)
                self.assertTrue(
                    mx.allclose(actual, expected, atol=2e-7, rtol=1e-6).item()
                )

    def test_split_prefill_matches_full_and_cache_visits_are_disjoint(self):
        with mx.stream(mx.cpu):
            config = _config()
            mx.random.seed(18)
            model = Model(config, execution_plan(config), CACHE_LAYOUT)
            tokens = mx.array([[5, 7, 11, 19, 31]])
            cache = make_prompt_cache(model)
            self.assertEqual(len(model.layers), 6)
            self.assertEqual(len(cache), 8)
            self.assertEqual(sum(isinstance(c, KVCache) for c in cache), 4)
            self.assertEqual(len({id(c) for c in cache}), len(cache))
            full = model(tokens)
            first = model(tokens[:, :3], cache)
            second = model(tokens[:, 3:], cache)
            self.assertTrue(
                mx.allclose(
                    mx.concatenate([first, second], axis=1), full, atol=3e-5, rtol=1e-4
                ).item()
            )
            self.assertEqual(cache[3].offset, 5)
            self.assertEqual(cache[5].offset, 5)
            self.assertFalse(
                mx.array_equal(cache[3].state[0], cache[5].state[0]).item()
            )
            self.assertFalse(mx.array_equal(cache[2][1], cache[4][1]).item())
            short = cache[:-1]
            with self.assertRaisesRegex(ValueError, "one cache slot per layer visit"):
                model(tokens, short)
