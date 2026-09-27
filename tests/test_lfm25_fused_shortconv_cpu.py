"""CPU-only structural tests for the default-off LFM ShortConv candidate."""

import importlib.abc
import sys
import types
import unittest
from unittest.mock import patch


class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError(f"real MLX import forbidden: {fullname}")
        return None


sys.meta_path.insert(0, BlockMLX())
from mlx2.adapters import lfm25_fused_shortconv as shortconv  # noqa: E402
from mlx2.adapters.lfm25_vl import LFM25VLAdapter  # noqa: E402


class Array:
    def __init__(self, shape, dtype="bf16"):
        self.shape = shape
        self.dtype = dtype


class Cache:
    def __init__(self):
        self.state = Array((1, 2, 2048))
        self.lengths = None
        self.left_padding = None
        self.advanced = 0

    def __getitem__(self, key):
        assert key == 0
        return self.state

    def __setitem__(self, key, value):
        assert key == 0
        self.state = value

    def advance(self, value):
        self.advanced += value


class ShortConvCandidateTest(unittest.TestCase):
    def setUp(self):
        self.projected = Array((1, 1, 6144))
        self.cache = Cache()
        self.weight = Array((2048, 3, 1))

    def admission(self, **changes):
        arguments = dict(projected=self.projected, state=self.cache[0],
                         weight=self.weight, bias=None, mask=None,
                         cache=self.cache, gdn_sink=None, dtype="bf16")
        arguments.update(changes)
        return shortconv.admit(**arguments)

    def test_exact_geometry_and_rejections(self):
        self.assertTrue(self.admission().accepted)
        variants = (
            {"projected": Array((2, 1, 6144))},
            {"projected": Array((1, 2, 6144))},
            {"state": Array((1, 1, 2048))},
            {"weight": Array((2048, 1, 3))},
            {"weight": Array((2048, 3, 1), "fp16")},
            {"bias": Array((2048,))},
            {"mask": object()},
            {"gdn_sink": []},
            {"state": None},
            {"groups": 1},
            {"stride": 2},
            {"dilation": 2},
            {"padding": 1},
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertFalse(self.admission(**variant).accepted)
        self.cache.lengths = object()
        self.assertFalse(self.admission().accepted)
        self.cache.lengths = None
        self.cache.left_padding = object()
        self.assertFalse(self.admission().accepted)

    def test_default_off_and_diagnostics_mark_candidate_unqualified(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(shortconv.enabled())
        adapter = object.__new__(LFM25VLAdapter)
        adapter.mlx_vlm_runtime = {"revision": "pinned"}
        adapter.fused_shortconv_opt_in_requested = True
        adapter.fused_shortconv_counters = shortconv.ShortConvCounters()
        status = adapter.diagnostics()["shortconv_fused_candidate"]
        self.assertEqual(status["qualification"], "pending")
        self.assertTrue(status["opt_in_requested"])
        self.assertEqual(status["engaged"], 0)
        self.assertEqual(adapter.execution_config(max_lanes=1, prefill_step=512)["lfm_shortconv"],
                         "source")
        adapter.fused_shortconv_counters.installed = 22
        self.assertEqual(adapter.execution_config(max_lanes=1, prefill_step=512)["lfm_shortconv"],
                         "fused-metal-candidate")

    def test_install_is_model_local_and_preserves_reference(self):
        calls = []

        class Conv:
            L_cache = 3
            def __init__(self):
                self.conv = types.SimpleNamespace(weight=Array((2048, 3, 1)), bias=None,
                                                  groups=2048, stride=1, dilation=1, padding=0)

            def _convolve_projected(self, projected, mask, cache, gdn_sink):
                calls.append("reference")
                return "reference"

        target = Conv()
        untouched = Conv()
        model = types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[
            types.SimpleNamespace(is_attention_layer=False, conv=target),
            types.SimpleNamespace(is_attention_layer=True),
        ]))
        fake_mx = types.ModuleType("mlx.core")
        fake_mx.bfloat16 = "bf16"
        fake_mx.fast = types.SimpleNamespace(metal_kernel=lambda **kwargs: None)
        fake_mx.metal = types.SimpleNamespace(is_available=lambda: True)
        fake_mx.gpu = object()
        fake_mx.default_device = lambda: fake_mx.gpu
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_mx
        new_state = Array((1, 2, 2048))
        with patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_mx}), \
             patch.dict("os.environ", {shortconv.ENV: "1"}), \
             patch.object(shortconv, "fused_projected", return_value=("fused", new_state)):
            counters = shortconv.ShortConvCounters()
            self.assertEqual(shortconv.install(model, counters), 1)
            self.assertEqual(target._convolve_projected(self.projected, None, self.cache, None), "fused")
            self.assertIs(self.cache[0], new_state)
            self.assertEqual(self.cache.advanced, 1)
            self.assertEqual(counters.engaged, 1)
            self.assertEqual(untouched._convolve_projected(self.projected, None, self.cache, None), "reference")
            self.assertEqual(target._convolve_projected(self.projected, object(), self.cache, None), "reference")
            self.assertEqual(counters.refused, 1)
            self.assertEqual(calls, ["reference", "reference"])
            with self.assertRaisesRegex(ValueError, "already installed"):
                shortconv.install(model, shortconv.ShortConvCounters())

    def test_build_failure_uses_reference_without_cache_write(self):
        calls = []

        class Conv:
            L_cache = 3
            def __init__(self):
                self.conv = types.SimpleNamespace(weight=Array((2048, 3, 1)), bias=None,
                                                  groups=2048, stride=1, dilation=1, padding=0)

            def _convolve_projected(self, projected, mask, cache, gdn_sink):
                calls.append("reference")
                return "reference"

        conv = Conv()
        model = types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[
            types.SimpleNamespace(is_attention_layer=False, conv=conv),
        ]))
        fake_mx = types.ModuleType("mlx.core")
        fake_mx.bfloat16 = "bf16"
        fake_mx.fast = types.SimpleNamespace(metal_kernel=lambda **kwargs: None)
        fake_mx.metal = types.SimpleNamespace(is_available=lambda: True)
        fake_mx.gpu = object()
        fake_mx.default_device = lambda: fake_mx.gpu
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_mx
        original = self.cache[0]
        with patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_mx}), \
             patch.dict("os.environ", {shortconv.ENV: "1"}), \
             patch.object(shortconv, "fused_projected", side_effect=ValueError("build")):
            counters = shortconv.ShortConvCounters()
            shortconv.install(model, counters)
            self.assertEqual(conv._convolve_projected(self.projected, None, self.cache, None), "reference")
            self.assertIs(self.cache[0], original)
            self.assertEqual(self.cache.advanced, 0)
            self.assertEqual(counters.build_failed, 1)
            self.assertEqual(calls, ["reference"])

    def test_kernel_dispatch_contract_without_mlx(self):
        calls = []

        def factory(**definition):
            calls.append(definition)
            return lambda **invocation: calls.append(invocation) or ("output", "state")

        fake_mx = types.ModuleType("mlx.core")
        fake_mx.fast = types.SimpleNamespace(metal_kernel=factory)
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_mx
        with patch.dict(sys.modules, {"mlx": fake_mlx, "mlx.core": fake_mx}):
            shortconv._kernel.cache_clear()
            self.assertEqual(shortconv.fused_projected(self.projected, self.cache[0], self.weight),
                             ("output", "state"))
            shortconv._kernel.cache_clear()
        self.assertEqual(calls[0]["input_names"], ["projected", "state", "weight"])
        self.assertEqual(calls[1]["template"], [("T", "bf16"), ("CHANNELS", 2048), ("TAPS", 3)])
        self.assertEqual(calls[1]["output_shapes"], [(1, 1, 2048), (1, 2, 2048)])
        self.assertEqual(calls[1]["grid"], (2048, 1, 1))
        self.assertIn("const T bx = T(b * x)", calls[0]["source"])


if __name__ == "__main__":
    unittest.main()
