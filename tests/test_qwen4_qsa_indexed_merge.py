# Mined from unified 1e2bc604, MIT; see docs/PROVENANCE.md.
import os
import unittest
from unittest import mock
import mlx.core as mx
import numpy as np
from mlx2.runtime.models import qwen4_qsa_indexed as indexed
from mlx2.runtime.models import qwen4_qsa_indexed_merge as merge

class TestQSAIndexedMerge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def setUp(self):
        merge.fused_merge_status(reset=True)

    def partials(self, count=8):
        rng = np.random.default_rng(91)
        m = rng.normal(0.0, 18.0, (1, 1, 1, count)).astype(np.float32)
        l = np.exp(rng.normal(0.0, 4.0, (1, 1, 1, count))).astype(np.float32)
        o = rng.normal(0.0, 1000.0, (1, 1, 1, count, 256)).astype(np.float32)
        return (mx.array(m), mx.array(l), mx.array(o))

    def test_metal_source_constructs_as_a_kernel_object(self):
        self.assertIsNotNone(merge._fused_merge_kernel())
        self.assertIsNotNone(merge._fused_merge_kernel(True))
        self.assertIn('metal::precise::exp', merge._SOURCE)
        self.assertIn('output_gate[', merge._SOURCE_GATED)
        self.assertIn('metal::exp(metal::abs(gate_x))', merge._SOURCE_GATED)
        self.assertEqual(merge._THREAD_CANDIDATES, (256, 128))

    def test_output_gate_preserves_attention_layout(self):
        out = mx.arange(2 * 3 * 2 * 4, dtype=mx.float32).reshape(2, 3, 2, 4)
        gate = mx.linspace(-3, 3, 2 * 2 * 3 * 4).reshape(2, 2, 12)
        expected = out.transpose(0, 2, 1, 3).reshape(2, 2, 12)
        expected = expected * mx.sigmoid(gate)
        expected = expected.reshape(2, 2, 3, 4).transpose(0, 2, 1, 3)
        actual = merge.mlx_apply_output_gate(out, gate)
        mx.eval(expected, actual)
        self.assertTrue(mx.array_equal(expected, actual).item())

    def test_sequential_order_is_independent_of_split_grouping(self):
        (m, l, o) = self.partials()
        one_split = merge.mlx_sequential_merge(m.reshape(1, 1, 1, 1, 8), l.reshape(1, 1, 1, 1, 8), o.reshape(1, 1, 1, 1, 8, 256), output_dtype=mx.float32)
        two_splits = merge.mlx_sequential_merge(m.reshape(1, 1, 1, 2, 4), l.reshape(1, 1, 1, 2, 4), o.reshape(1, 1, 1, 2, 4, 256), output_dtype=mx.float32)
        reversed_chunks = merge.mlx_sequential_merge(m[..., ::-1], l[..., ::-1], o[..., ::-1, :], output_dtype=mx.float32)
        mx.eval(one_split, two_splits, reversed_chunks)
        self.assertTrue(mx.array_equal(one_split, two_splits).item())
        self.assertFalse(mx.array_equal(one_split, reversed_chunks).item())

    def test_env_unset_keeps_the_fused_dispatch_inert_and_output_identical(self):
        (m, l, o) = self.partials()
        expected = merge.mlx_sequential_merge(m, l, o, output_dtype=mx.float32)
        env = dict(os.environ)
        env.pop('MLX_QWEN4_QSA_INDEXED_FUSED_MERGE', None)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(merge, '_fused_merge', side_effect=AssertionError('fused dispatch must stay inert')):
            actual = merge.combine_indexed_partials(m, l, o, output_dtype=mx.float32)
        mx.eval(expected, actual)
        self.assertTrue(mx.array_equal(expected, actual).item())
        self.assertEqual(merge.fused_merge_status(), {'engaged': False, 'fallbacks': 0, 'candidate': None, 'gate_engaged': False, 'gate_path': None})
        sentinel = mx.zeros((1, 1, 1, 256), dtype=mx.float32)
        old_kernel = mock.Mock(return_value=[sentinel])
        hook_m = mx.zeros((1, 1, 1, 128), dtype=mx.float32)
        hook_o = mx.zeros((1, 1, 1, 128, 256), dtype=mx.float32)
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(indexed, '_combine_kernel', return_value=old_kernel), mock.patch.object(indexed, 'combine_indexed_partials', side_effect=AssertionError('env-unset hook changed dispatch')):
            (hooked, counter) = indexed._combine_sdpa_partials(hook_m, hook_m, hook_o, mx.array([3], dtype=mx.uint32), output_dtype=mx.float32)
        self.assertIs(hooked, sentinel)
        self.assertEqual(int(counter.item()), 3)
        old_kernel.assert_called_once()

    def test_runtime_error_falls_back_and_records_merge_receipt(self):
        m = mx.full((1, 1, 1, 128), -mx.inf, dtype=mx.float32)
        l = mx.zeros_like(m)
        o = mx.zeros((1, 1, 1, 128, 256), dtype=mx.float32)
        expected = merge.mlx_sequential_merge(m, l, o, output_dtype=mx.float32)
        indexed.qsa_indexed_status(reset=True)
        with mock.patch.dict(os.environ, {'MLX_QWEN4_QSA_INDEXED_FUSED_MERGE': '1'}), mock.patch.object(merge, 'fused_merge_available', return_value=True), mock.patch.object(merge, '_fused_merge', side_effect=RuntimeError('compile')):
            (actual, counter) = indexed._combine_sdpa_partials(m, l, o, mx.array([7], dtype=mx.uint32), output_dtype=mx.float32)
        mx.eval(expected, actual)
        self.assertTrue(mx.array_equal(expected, actual).item())
        self.assertEqual(int(counter.item()), 7)
        status = indexed.qsa_indexed_status()
        self.assertEqual(status['counts']['merge_fallback'], 1)
        self.assertEqual(status['fused_merge'], {'engaged': False, 'fallbacks': 1, 'candidate': None, 'gate_engaged': False, 'gate_path': None})
if __name__ == '__main__':
    unittest.main()
