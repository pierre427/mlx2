# Mined from unified 1e2bc604, MIT; see docs/PROVENANCE.md.
import hashlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import mlx.core as mx
import numpy as np
from mlx2.runtime.models import qwen4_qsa_indexed as indexed
from mlx2.runtime.models import qwen4_exp as qwen4_exp
from mlx2.runtime.models.qwen4_exp import QSACompactBlocks, _gather_qsa_attention, _gather_qsa_quantized_attention
from mlx2.runtime.models.qwen4_qsa_nax import compact_token_validity

def _compact(batch, length, *, total=32, selected_width=7):
    left = np.arange(batch, dtype=np.int32) % 3
    ids = np.zeros((batch, length, selected_width), dtype=np.uint32)
    counts = np.zeros((batch, length), dtype=np.int32)
    tail_stop = np.zeros((batch, length), dtype=np.int32)
    for b in range(batch):
        logical_total = total - int(left[b])
        for row in range(length):
            q_pos = min(logical_total - 1, 7 + 2 * row)
            tail_stop[b, row] = q_pos + 1
            closed = (q_pos + 1) // 4
            count = min((b + row) % 5, closed)
            counts[b, row] = count
            if count:
                choices = np.arange(count, dtype=np.uint32)
                ids[b, row, :count] = choices
    tail_start = tail_stop // 4 * 4
    token_logical = np.arange(total)[None, :] - left[:, None]
    q_pos = tail_stop - 1
    causal = ((token_logical[:, None, :] >= 0) & (token_logical[:, None, :] <= q_pos[..., None]))[:, None]
    return QSACompactBlocks(block_ids=mx.array(ids), block_counts=mx.array(counts), tail_start=mx.array(tail_start), tail_stop=mx.array(tail_stop), left_padding=mx.array(left), block_size=4, physical_width=total, causal_mask=mx.array(causal))

def _arrays(batch, length, *, total=32, dtype=mx.float32):
    (heads, kv_heads, dim) = (4, 2, 8)
    q = mx.random.normal((batch, heads, length, dim)).astype(dtype)
    k = mx.random.normal((batch, kv_heads, total, dim)).astype(dtype)
    v = mx.random.normal((batch, kv_heads, total, dim)).astype(dtype)
    return (q, k, v)

def _wide_compact(length=2, *, selected_width=127):
    total = 512
    ids = mx.broadcast_to(mx.arange(selected_width, dtype=mx.uint32)[None, None], (1, length, selected_width))
    counts = mx.full((1, length), selected_width, dtype=mx.int32)
    tail_stop = mx.full((1, length), total, dtype=mx.int32)
    return QSACompactBlocks(block_ids=ids, block_counts=counts, tail_start=tail_stop, tail_stop=tail_stop, left_padding=None, block_size=4, physical_width=total, causal_mask=None)

def _real_bf16_fixture():
    path = Path(__file__).parent / 'fixtures' / 'qwen4_qsa_indexed_real_bf16_257.safetensors'
    arrays = mx.load(str(path))
    width = int(arrays['k'].shape[2])
    blocks = width // 4
    compact = QSACompactBlocks(block_ids=mx.arange(blocks, dtype=mx.uint32)[None, None], block_counts=mx.array([[blocks]], dtype=mx.int32), tail_start=mx.array([[width]], dtype=mx.int32), tail_stop=mx.array([[width]], dtype=mx.int32), left_padding=mx.array([0], dtype=mx.int32), block_size=4, physical_width=width, causal_mask=None)
    return (arrays['q'], arrays['k'], arrays['v'], compact)

def _real_bf16_m3_fixture():
    path = Path(__file__).parent / 'fixtures' / 'qwen4_qsa_indexed_real_bf16_m3.safetensors'
    arrays = mx.load(str(path))
    compact = QSACompactBlocks(block_ids=arrays['ids'], block_counts=arrays['n_sel'].astype(mx.int32), tail_start=arrays['tail_start'], tail_stop=arrays['tail_stop'], left_padding=arrays['left_pad'], block_size=4, physical_width=int(arrays['total'].item()), causal_mask=arrays['causal_mask'])
    return (arrays['q'], arrays['k'], arrays['v'], compact)

def _private_delta_m3_fixture():
    (q, k, v, compact) = _real_bf16_m3_fixture()
    base_tokens = 1028
    q = mx.concatenate([q, q], axis=0)
    delta_k = mx.concatenate([k[:, :, base_tokens:], k[:, :, base_tokens:] + 0.25], axis=0).astype(k.dtype)
    delta_v = mx.concatenate([v[:, :, base_tokens:], v[:, :, base_tokens:] - 0.25], axis=0).astype(v.dtype)
    block_ids = mx.concatenate([compact.block_ids, compact.block_ids], axis=0)
    block_counts = mx.concatenate([compact.block_counts, compact.block_counts], axis=0)
    tail_stop = mx.concatenate([compact.tail_stop, mx.array([[1029, 1030, 1030]])], axis=0)
    tail_start = tail_stop // 4 * 4
    q_pos = tail_stop - 1
    causal = (mx.arange(1031, dtype=mx.int32)[None, None, :] <= q_pos[..., None])[:, None]
    compact = QSACompactBlocks(block_ids=block_ids, block_counts=block_counts, tail_start=tail_start, tail_stop=tail_stop, left_padding=None, block_size=4, physical_width=1031, causal_mask=causal)
    return (q, k[:, :, :base_tokens], v[:, :, :base_tokens], delta_k, delta_v, mx.array([3, 2], dtype=mx.uint32), compact)

def _private_delta_randomized_fixture(length: int):
    """Two row-specific selections, causal masks and unequal suffixes."""
    (batch, nqh, nkh, dim) = (2, 12, 1, 256)
    base_tokens = 1028
    delta_width = int(length) + 2
    delta_lengths = np.array([delta_width, delta_width - 2], dtype=np.uint32)
    key = mx.random.key(9000 + int(length))
    q = mx.random.normal((batch, nqh, length, dim), key=key).astype(mx.bfloat16)
    base_k = mx.random.normal((1, nkh, base_tokens, dim), key=mx.random.key(9100 + int(length))).astype(mx.bfloat16)
    base_v = mx.random.normal((1, nkh, base_tokens, dim), key=mx.random.key(9200 + int(length))).astype(mx.bfloat16)
    delta_k = mx.random.normal((batch, nkh, delta_width, dim), key=mx.random.key(9300 + int(length))).astype(mx.bfloat16)
    delta_v = mx.random.normal((batch, nkh, delta_width, dim), key=mx.random.key(9400 + int(length))).astype(mx.bfloat16)
    total = base_tokens + delta_width
    selected_width = base_tokens // 4
    ids = np.zeros((batch, length, selected_width), dtype=np.uint32)
    counts = np.zeros((batch, length), dtype=np.int32)
    tail_stop = np.zeros((batch, length), dtype=np.int32)
    causal = np.zeros((batch, 1, length, total), dtype=np.bool_)
    base_pages = list(range(selected_width))
    for row in range(batch):
        prior = int(delta_lengths[row]) - length
        for query in range(length):
            qpos = base_tokens + prior + query
            tail_stop[row, query] = qpos + 1
            omitted = 1 + (row * length + query) % (selected_width - 1)
            selected = [page for page in base_pages if page != omitted]
            ids[row, query, :len(selected)] = selected
            counts[row, query] = len(selected)
            causal[row, 0, query, :qpos + 1] = True
    compact = QSACompactBlocks(block_ids=mx.array(ids), block_counts=mx.array(counts), tail_start=mx.array(tail_stop // 4 * 4), tail_stop=mx.array(tail_stop), left_padding=None, block_size=4, physical_width=total, causal_mask=mx.array(causal))
    return (q, base_k, base_v, delta_k, delta_v, mx.array(delta_lengths), compact)

def _private_delta_exact_set_fixture(length: int):
    """B2 fixture with shared base-page sets and row-private suffix state."""
    (q, base_k, base_v, delta_k, delta_v, lengths, compact) = _private_delta_randomized_fixture(length)
    shared_ids = mx.concatenate([compact.block_ids[:1], compact.block_ids[:1]], axis=0)
    shared_counts = mx.concatenate([compact.block_counts[:1], compact.block_counts[:1]], axis=0)
    return (q, base_k, base_v, delta_k, delta_v, lengths, replace(compact, block_ids=shared_ids, block_counts=shared_counts))

class TestQSAIndexedReference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = mx.default_device()
        mx.set_default_device(mx.cpu)

    @classmethod
    def tearDownClass(cls):
        mx.set_default_device(cls.device)

    def test_reference_matches_gather_random_and_adversarial_rows(self):
        mx.random.seed(17)
        for length in range(2, 9):
            batch = 1 + (length - 2) % 3
            with self.subTest(batch=batch, length=length):
                compact = _compact(batch, length)
                (q, k, v) = _arrays(batch, length)
                actual = indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=8 ** (-0.5), splits=4)
                expected = _gather_qsa_attention(q, k, v, compact, scale=8 ** (-0.5), tile_rows=2)
                mx.eval(actual, expected)
                np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-05, atol=1e-05)

    def test_quantized_reference_matches_one_token_gather_bit_exact(self):
        compact = QSACompactBlocks(block_ids=mx.zeros((1, 3, 1), dtype=mx.uint32), block_counts=mx.zeros((1, 3), dtype=mx.int32), tail_start=mx.zeros((1, 3), dtype=mx.int32), tail_stop=mx.ones((1, 3), dtype=mx.int32), left_padding=None, block_size=4, physical_width=4, causal_mask=None)
        for bits in (8, 4):
            with self.subTest(bits=bits):
                mx.random.seed(40 + bits)
                q = mx.random.normal((1, 4, 3, 32)).astype(mx.bfloat16)
                k = mx.random.normal((1, 2, 4, 32)).astype(mx.bfloat16)
                v = mx.random.normal((1, 2, 4, 32)).astype(mx.bfloat16)
                q_keys = mx.quantize(k, group_size=32, bits=bits)
                q_values = mx.quantize(v, group_size=32, bits=bits)
                mirror = indexed.qwen4_qsa_indexed_quantized_reference(q, q_keys, q_values, compact, scale=32 ** (-0.5), splits=1, group_size=32, key_bits=bits, value_bits=bits)
                gather = _gather_qsa_quantized_attention(q, q_keys, q_values, compact, scale=32 ** (-0.5), tile_rows=2, group_size=32, key_bits=bits, value_bits=bits)
                mx.eval(mirror, gather)
                self.assertTrue(np.array_equal(np.asarray(mirror.astype(mx.float32)), np.asarray(gather.astype(mx.float32))))

    def test_bf16_kernel_sources_are_pinned(self):
        pass_one = hashlib.sha256(indexed._SOURCE.encode()).hexdigest()
        private_delta = hashlib.sha256(indexed._PRIVATE_DELTA_SOURCE.encode()).hexdigest()
        pass_two = hashlib.sha256(indexed._COMBINE_SOURCE.encode()).hexdigest()
        self.assertEqual(pass_one, 'f5980991e8d5fb819a6f73bfb97c3444914cd236006911dee93577fe9eb262bf')
        self.assertEqual(private_delta, 'b0d3bcd76deb2bbbe7fe216806d46d0c800b215f18b16acb804a775ee1b46bdb')
        self.assertEqual(pass_two, '0ae2acf66aba304934b62f5e51015d4fc1fd4ad9532e2e8d182f5192a5197407')

    def test_split_count_is_bit_exact_in_fp32(self):
        mx.random.seed(23)
        compact = _wide_compact()
        q = mx.random.normal((1, 24, 2, 256)).astype(mx.float32)
        k = mx.random.normal((1, 2, 512, 256)).astype(mx.float32)
        v = mx.random.normal((1, 2, 512, 256)).astype(mx.float32)
        outputs = [indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=256 ** (-0.5), splits=splits) for splits in (1, 2, 4, 8)]
        mx.eval(*outputs)
        first = np.asarray(outputs[0])
        for (splits, output) in zip((1, 2, 4, 8), outputs):
            self.assertTrue(np.array_equal(first, np.asarray(output)), f'split count {splits} changed the fp32 result')

    def test_fixed_chunk_boundaries_do_not_depend_on_splits(self):
        expected = indexed.indexed_chunk_ranges(520)
        for splits in (1, 8):
            groups = indexed.indexed_split_chunk_ranges(520, splits)
            flattened = tuple((chunk for group in groups for chunk in group))
            self.assertEqual(flattened, expected)

    def test_query_rows_are_independent(self):
        mx.random.seed(29)
        compact = _compact(2, 3)
        (q, k, v) = _arrays(2, 3)
        baseline = indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=8 ** (-0.5), splits=4)
        changed_q = mx.concatenate([q[:, :, :1], q[:, :, 1:] * -31.0 + 19.0], axis=2)
        changed = indexed.qwen4_qsa_indexed_reference(changed_q, k, v, compact, scale=8 ** (-0.5), splits=4)
        mx.eval(baseline, changed)
        np.testing.assert_array_equal(np.asarray(baseline[:, :, 0]), np.asarray(changed[:, :, 0]))

    def test_shared_validity_matches_metal_slot_arithmetic_at_m3(self):
        compact = _compact(2, 3, total=32)
        (ids, counts, n_sel, u_width, q_pos, left_pad, total, physical, valid) = compact_token_validity(compact)
        mx.eval(ids, counts, n_sel, q_pos, left_pad, physical, valid)
        ids_np = np.asarray(ids)
        counts_np = np.asarray(counts)
        n_sel_np = np.asarray(n_sel)
        q_pos_np = np.asarray(q_pos)
        left_pad_np = np.asarray(left_pad)
        mask_np = np.asarray(compact.causal_mask)
        expected_physical = np.zeros(physical.shape, dtype=np.int32)
        expected_valid = np.zeros(valid.shape, dtype=bool)
        for batch in range(2):
            for row in range(3):
                complete = (int(q_pos_np[batch, row]) + 1) // 4 * 4
                for slot in range(u_width):
                    for tail in range(4):
                        logical = int(ids_np[batch, row, slot]) * 4 + tail
                        source = int(left_pad_np[batch]) + logical
                        expected_physical[batch, row, slot, tail] = np.clip(source, 0, total - 1)
                        live = slot < int(counts_np[batch, row])
                        live = live and 0 <= source < total
                        live = live and logical <= int(q_pos_np[batch, row])
                        live = live and (slot < int(n_sel_np[batch, row]) or logical >= complete)
                        if live:
                            live = bool(mask_np[batch, 0, row, source])
                        expected_valid[batch, row, slot, tail] = live
        np.testing.assert_array_equal(np.asarray(physical), expected_physical)
        np.testing.assert_array_equal(np.asarray(valid), expected_valid)
        self.assertGreater(int(left_pad_np.max()), 0)
        self.assertTrue(bool(expected_valid.any()))
        (q, k, v) = _arrays(2, 3, total=32)
        mirror = indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=8 ** (-0.5), splits=8)
        gather = _gather_qsa_attention(q, k, v, compact, scale=8 ** (-0.5), tile_rows=2)
        mx.eval(mirror, gather)
        np.testing.assert_allclose(np.asarray(mirror), np.asarray(gather), rtol=1e-05, atol=1e-05)

    def test_pld_verify_widths_mirror_gather_with_ragged_tails_and_masks(self):
        """The CPU mirror band for the widths adaptive PLD actually proposes.

        Kernel-vs-gather exactness is asserted on Metal by the gate; on CPU
        the contract is that the indexed reference and the gather reference
        agree at every width in the widened admission window, with ragged
        per-row block counts, non-zero left padding and a dense causal mask.
        """
        for length in (9, 12, 15, 16, 17):
            for splits in (1, 4, 8):
                with self.subTest(length=length, splits=splits):
                    mx.random.seed(1300 + length)
                    compact = _compact(2, length, total=64, selected_width=15)
                    counts = np.asarray(compact.block_counts)
                    self.assertGreater(int(counts.max()), int(counts.min()))
                    self.assertGreater(int(np.asarray(compact.left_padding).max()), 0)
                    self.assertIsNotNone(compact.causal_mask)
                    (q, k, v) = _arrays(2, length, total=64)
                    mirror = indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=8 ** (-0.5), splits=splits)
                    gather = _gather_qsa_attention(q, k, v, compact, scale=8 ** (-0.5), tile_rows=2)
                    mx.eval(mirror, gather)
                    np.testing.assert_allclose(np.asarray(mirror), np.asarray(gather), rtol=1e-05, atol=1e-05)

    def test_pld_verify_widths_are_split_invariant_without_a_mask(self):
        for length in (9, 12, 15, 16, 17):
            with self.subTest(length=length):
                mx.random.seed(2300 + length)
                compact = _wide_compact(length=length)
                (q, k, v) = _arrays(1, length, total=512)
                outputs = [indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=8 ** (-0.5), splits=splits) for splits in (1, 2, 8, 32, 127)]
                mx.eval(*outputs)
                for other in outputs[1:]:
                    self.assertTrue(bool(mx.array_equal(outputs[0], other).item()))

    def test_cache_prefix_views_pass_through_and_match_contiguous_mirror(self):
        mx.random.seed(30)
        total = 32
        compact = _compact(1, 3, total=total)
        q = mx.random.normal((1, 4, 3, 8))
        k_buffer = mx.random.normal((1, 2, 256, 8))
        v_buffer = mx.random.normal((1, 2, 256, 8))
        k_view = k_buffer[:, :, :total]
        v_view = v_buffer[:, :, :total]
        k_contiguous = mx.contiguous(k_view)
        v_contiguous = mx.contiguous(v_view)
        view_output = indexed.qwen4_qsa_indexed_reference(q, k_view, v_view, compact, scale=8 ** (-0.5), splits=4)
        contiguous_output = indexed.qwen4_qsa_indexed_reference(q, k_contiguous, v_contiguous, compact, scale=8 ** (-0.5), splits=4)
        mx.eval(view_output, contiguous_output)
        np.testing.assert_array_equal(np.asarray(view_output), np.asarray(contiguous_output))
        captured = {}

        def dispatch(**kwargs):
            captured['inputs'] = kwargs['inputs']
            return (mx.zeros((1,), dtype=mx.float32),)
        with mock.patch.object(indexed, '_partition_kernel', return_value=dispatch):
            indexed._partition_dispatch(q, k_view, v_view, compact, scale=8 ** (-0.5), threads=64, splits=4, hpt=2)
        self.assertIs(captured['inputs'][0], q)
        self.assertIs(captured['inputs'][1], k_view)
        self.assertIs(captured['inputs'][2], v_view)
        self.assertIs(captured['inputs'][8], compact.causal_mask)

    def test_partition_dispatch_declines_unsupported_mask_layouts(self):
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3)
        invalid_masks = (mx.ones((1, 3, 32), dtype=mx.bool_), mx.ones((1, 2, 3, 32), dtype=mx.bool_), mx.ones((1, 1, 3, 32), dtype=mx.float32), mx.ones((2, 1, 3, 32), dtype=mx.bool_))
        for mask in invalid_masks:
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with mock.patch.object(indexed, '_partition_kernel', side_effect=AssertionError('kernel must not run')), self.assertRaises(indexed.QSAIndexedProbeDeclined) as raised:
                    indexed._partition_dispatch(q, k, v, replace(compact, causal_mask=mask), scale=8 ** (-0.5), threads=64, splits=4, hpt=2)
                self.assertEqual(raised.exception.reason, 'unsupported_mask_layout')

    def test_duplicate_block_ids_fail_closed(self):
        compact = QSACompactBlocks(block_ids=mx.array([[[1, 1]]], dtype=mx.uint32), block_counts=mx.array([[2]], dtype=mx.int32), tail_start=mx.array([[4]], dtype=mx.int32), tail_stop=mx.array([[4]], dtype=mx.int32), left_padding=None, block_size=4, physical_width=16, causal_mask=None)
        (q, k, v) = _arrays(1, 1, total=16)
        with self.assertRaisesRegex(ValueError, 'unique'):
            indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=8 ** (-0.5), splits=1)

    def test_fully_masked_row_returns_zero_without_nan(self):
        compact = QSACompactBlocks(block_ids=mx.zeros((1, 2, 1), dtype=mx.uint32), block_counts=mx.zeros((1, 2), dtype=mx.int32), tail_start=mx.array([[4, 8]], dtype=mx.int32), tail_stop=mx.array([[4, 8]], dtype=mx.int32), left_padding=None, block_size=4, physical_width=16, causal_mask=None)
        (q, k, v) = _arrays(1, 2, total=16)
        output = indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=8 ** (-0.5), splits=4)
        mx.eval(output)
        np.testing.assert_array_equal(np.asarray(output), np.zeros(output.shape))

    def test_metal_kernel_object_construction_does_not_dispatch(self):
        self.assertIsNotNone(indexed._partition_kernel())
        self.assertIsNotNone(indexed._private_delta_partition_kernel())
        self.assertIsNotNone(indexed._quantized_partition_kernel())
        self.assertIsNotNone(indexed._combine_kernel())
        self.assertIsNotNone(indexed._combine_kernel(True))

    def test_real_capture_fixture_has_production_two_pass_geometry(self):
        (q, k, v, compact) = _real_bf16_fixture()
        self.assertEqual(q.shape, (1, 12, 1, 256))
        self.assertEqual(k.shape, (1, 1, 1028, 256))
        self.assertEqual(k.shape, v.shape)
        self.assertEqual(q.dtype, mx.bfloat16)
        self.assertEqual(indexed.indexed_splits_for(257), 128)
        reference = indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=256 ** (-0.5), splits=5)
        gather = _gather_qsa_attention(q, k, v, compact, scale=256 ** (-0.5), tile_rows=1)
        mx.eval(reference, gather)
        self.assertEqual(reference.shape, gather.shape)
        self.assertTrue(bool(mx.all(mx.isfinite(reference)).item()))

    def test_real_m3_fixture_mirror_matches_gather_on_cpu(self):
        (q, k, v, compact) = _real_bf16_m3_fixture()
        self.assertEqual(q.shape, (1, 12, 3, 256))
        self.assertEqual(k.shape, (1, 1, 1031, 256))
        mirror = indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=256 ** (-0.5), splits=8)
        gather = _gather_qsa_attention(q, k, v, compact, scale=256 ** (-0.5), tile_rows=1)
        mx.eval(mirror, gather)
        np.testing.assert_allclose(np.asarray(mirror.astype(mx.float32)), np.asarray(gather.astype(mx.float32)), rtol=0.0, atol=1.0 / 128.0)

    def test_private_delta_materialized_oracle_matches_physical_b2(self):
        (q, base_k, base_v, delta_k, delta_v, _lengths, compact) = _private_delta_m3_fixture()
        actual = indexed.qwen4_qsa_indexed_private_delta_reference(q, base_k, base_v, delta_k, delta_v, compact, scale=256 ** (-0.5), splits=8)
        physical_k = mx.concatenate([mx.broadcast_to(base_k, (2, *base_k.shape[1:])), delta_k], axis=2)
        physical_v = mx.concatenate([mx.broadcast_to(base_v, (2, *base_v.shape[1:])), delta_v], axis=2)
        expected = indexed.qwen4_qsa_indexed_reference(q, physical_k, physical_v, compact, scale=256 ** (-0.5), splits=8)
        mx.eval(actual, expected)
        self.assertTrue(mx.array_equal(actual, expected).item())

    def test_private_delta_preflight_is_width_bounded_and_side_effect_free(self):
        with mock.patch.object(indexed, 'indexed_kernel_available', return_value=True), mock.patch.object(indexed, '_sdpa_header_state', return_value=(True, 'digest', None)), mock.patch.object(indexed.mx, '__version__', next(iter(indexed._EXACT_MLX_BUILDS))):
            for width in range(1, 10):
                (admitted, reason) = indexed.qwen4_qsa_indexed_private_delta_preflight(length=width, base_tokens=16384, head_dim=256, num_query_heads=12, num_kv_heads=1, block_size=4, selected_blocks=512)
                self.assertTrue(admitted, (width, reason))
            (admitted, reason) = indexed.qwen4_qsa_indexed_private_delta_preflight(length=10, base_tokens=16384, head_dim=256, num_query_heads=12, num_kv_heads=1, block_size=4, selected_blocks=512)
            self.assertFalse(admitted)
            self.assertEqual(reason, 'width_out_of_range')

    def test_private_delta_width_specific_context_thresholds(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT', None)
            self.assertEqual(indexed.qwen4_qsa_private_delta_min_context(1), 2 ** 31 - 1)
            self.assertEqual(indexed.qwen4_qsa_private_delta_min_context(2), 65536)
            os.environ['MLX_LM_QSA_PRIVATE_DELTA_MIN_CONTEXT'] = '12000'
            self.assertEqual(indexed.qwen4_qsa_private_delta_min_context(1), 12000)
            self.assertEqual(indexed.qwen4_qsa_private_delta_min_context(9), 12000)

    def test_private_delta_exact_set_preflight_is_default_on_and_b2_gated(self):
        arguments = {'batch': 2, 'length': 3, 'base_tokens': 16384, 'head_dim': 256, 'num_query_heads': 12, 'num_kv_heads': 1, 'block_size': 4, 'selected_blocks': 512}
        with mock.patch.dict(os.environ, {'MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD': '0'}, clear=False):
            (admitted, reason) = indexed.qwen4_qsa_indexed_private_delta_exact_set_preflight(**arguments)
            self.assertFalse(admitted)
            self.assertEqual(reason, 'exact_set_fold_disabled')
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(indexed, 'indexed_kernel_available', return_value=True), mock.patch.object(indexed, '_sdpa_header_state', return_value=(True, 'digest', None)), mock.patch.object(indexed.mx, '__version__', next(iter(indexed._EXACT_MLX_BUILDS))):
            os.environ.pop('MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD', None)
            (admitted, reason) = indexed.qwen4_qsa_indexed_private_delta_exact_set_preflight(**arguments)
            self.assertTrue(admitted, reason)
            (admitted, reason) = indexed.qwen4_qsa_indexed_private_delta_exact_set_preflight(**{**arguments, 'batch': 3})
            self.assertFalse(admitted)
            self.assertEqual(reason, 'exact_set_fold_requires_b2')

    def test_private_delta_exact_set_proof_is_per_query_and_device_resident(self):
        (*_arrays, compact) = _private_delta_exact_set_fixture(3)
        proof = indexed.qwen4_qsa_indexed_private_delta_exact_set_proof(compact, base_tokens=1028)
        self.assertEqual(proof.shape, (3,))
        self.assertEqual(proof.dtype, mx.bool_)
        mx.eval(proof)
        self.assertEqual(np.asarray(proof).tolist(), [True, True, True])
        (*_arrays, compact) = _private_delta_randomized_fixture(3)
        proof = indexed.qwen4_qsa_indexed_private_delta_exact_set_proof(compact, base_tokens=1028)
        mx.eval(proof)
        self.assertEqual(np.asarray(proof).tolist(), [False, False, False])

    @unittest.skipUnless(os.environ.get('MLX_QWEN4_QSA_INDEXED_TEST_METAL') == '1', 'set MLX_QWEN4_QSA_INDEXED_TEST_METAL=1 for the real Metal fixture')
    def test_private_delta_kernel_is_bit_exact_against_physical_b2(self):
        device = mx.default_device()
        try:
            mx.set_default_device(mx.gpu)
            (q, base_k, base_v, delta_k, delta_v, lengths, compact) = _private_delta_m3_fixture()
            physical_k = mx.concatenate([mx.broadcast_to(base_k, (2, *base_k.shape[1:])), delta_k], axis=2)
            physical_v = mx.concatenate([mx.broadcast_to(base_v, (2, *base_v.shape[1:])), delta_v], axis=2)
            expected = indexed.qwen4_qsa_indexed_attention(q, physical_k, physical_v, compact, scale=256 ** (-0.5), splits=128, hpt=12)
            actual = indexed.qwen4_qsa_indexed_private_delta_attention(q, base_k, base_v, delta_k, delta_v, lengths, compact, scale=256 ** (-0.5), splits=128, hpt=12)
            mx.eval(expected, actual)
            self.assertTrue(mx.array_equal(actual, expected).item())
        finally:
            mx.set_default_device(device)

    @unittest.skipUnless(os.environ.get('MLX_QWEN4_QSA_INDEXED_TEST_METAL') == '1', 'set MLX_QWEN4_QSA_INDEXED_TEST_METAL=1 for the real Metal fixture')
    def test_private_delta_m1_to_m9_randomized_is_bit_exact_on_metal(self):
        device = mx.default_device()
        try:
            mx.set_default_device(mx.gpu)
            for width in range(1, 10):
                (q, base_k, base_v, delta_k, delta_v, lengths, compact) = _private_delta_randomized_fixture(width)
                physical_k = mx.concatenate([mx.broadcast_to(base_k, (2, *base_k.shape[1:])), delta_k], axis=2)
                physical_v = mx.concatenate([mx.broadcast_to(base_v, (2, *base_v.shape[1:])), delta_v], axis=2)
                expected = indexed.qwen4_qsa_indexed_attention(q, physical_k, physical_v, compact, scale=256 ** (-0.5), splits=128, hpt=12)
                actual = indexed.qwen4_qsa_indexed_private_delta_attention(q, base_k, base_v, delta_k, delta_v, lengths, compact, scale=256 ** (-0.5), splits=128, hpt=12)
                mx.eval(expected, actual)
                self.assertTrue(mx.array_equal(actual, expected).item(), f'M={width}')
        finally:
            mx.set_default_device(device)

    @unittest.skipUnless(os.environ.get('MLX_QWEN4_QSA_INDEXED_TEST_METAL') == '1', 'set MLX_QWEN4_QSA_INDEXED_TEST_METAL=1 for Metal exactness')
    def test_private_delta_exact_set_fold_m1_to_m4_is_bit_exact_on_metal(self):
        device = mx.default_device()
        try:
            mx.set_default_device(mx.gpu)
            with mock.patch.dict(os.environ, {'MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD': '1'}, clear=False):
                for width in range(1, 5):
                    (q, base_k, base_v, delta_k, delta_v, lengths, compact) = _private_delta_exact_set_fixture(width)
                    physical_k = mx.concatenate([mx.broadcast_to(base_k, (2, *base_k.shape[1:])), delta_k], axis=2)
                    physical_v = mx.concatenate([mx.broadcast_to(base_v, (2, *base_v.shape[1:])), delta_v], axis=2)
                    expected = indexed.qwen4_qsa_indexed_attention(q, physical_k, physical_v, compact, scale=256 ** (-0.5), splits=128, hpt=12)
                    actual = indexed.qwen4_qsa_indexed_private_delta_exact_set_attention(q, base_k, base_v, delta_k, delta_v, lengths, compact, scale=256 ** (-0.5), splits=128, hpt=12)
                    mx.eval(expected, actual)
                    self.assertTrue(mx.array_equal(actual, expected).item(), f'M={width}')
        finally:
            mx.set_default_device(device)

    @unittest.skipUnless(os.environ.get('MLX_QWEN4_QSA_INDEXED_TEST_METAL') == '1', 'set MLX_QWEN4_QSA_INDEXED_TEST_METAL=1 for Metal exactness')
    def test_private_delta_exact_set_fold_stale_proof_is_bit_exact_on_metal(self):
        device = mx.default_device()
        try:
            mx.set_default_device(mx.gpu)
            (q, base_k, base_v, delta_k, delta_v, lengths, compact) = _private_delta_randomized_fixture(3)
            physical_k = mx.concatenate([mx.broadcast_to(base_k, (2, *base_k.shape[1:])), delta_k], axis=2)
            physical_v = mx.concatenate([mx.broadcast_to(base_v, (2, *base_v.shape[1:])), delta_v], axis=2)
            expected = indexed.qwen4_qsa_indexed_attention(q, physical_k, physical_v, compact, scale=256 ** (-0.5), splits=128, hpt=12)
            with mock.patch.dict(os.environ, {'MLX_LM_QSA_PRIVATE_DELTA_EXACT_SET_FOLD': '1'}, clear=False):
                actual = indexed.qwen4_qsa_indexed_private_delta_exact_set_attention(q, base_k, base_v, delta_k, delta_v, lengths, compact, scale=256 ** (-0.5), splits=128, hpt=12)
            mx.eval(expected, actual)
            self.assertTrue(mx.array_equal(actual, expected).item())
        finally:
            mx.set_default_device(device)

    def test_reviewed_sdpa_header_hash_matches_installed_mlx(self):
        header = Path(mx.__file__).resolve().parent / 'include' / 'mlx' / 'backend' / 'metal' / 'kernels' / 'sdpa_vector.h'
        self.assertTrue(header.is_file(), f'MLX wheel does not ship {header}')
        digest = hashlib.sha256(header.read_bytes()).hexdigest()
        self.assertEqual(digest, indexed._SDPA_VECTOR_HEADER_SHA256)

    @unittest.skipUnless(os.environ.get('MLX_QWEN4_QSA_INDEXED_TEST_METAL') == '1', 'set MLX_QWEN4_QSA_INDEXED_TEST_METAL=1 for the real Metal fixture')
    def test_real_capture_fixture_is_bit_exact_on_metal(self):
        device = mx.default_device()
        try:
            mx.set_default_device(mx.gpu)
            (q, k, v, compact) = _real_bf16_fixture()
            gather = _gather_qsa_attention(q, k, v, compact, scale=256 ** (-0.5), tile_rows=1)
            outputs = [indexed.qwen4_qsa_indexed_attention(q, k, v, compact, scale=256 ** (-0.5), splits=splits) for splits in (8, 16, 32, 64, 128)]
            mx.eval(gather, *outputs)
            for (splits, output) in zip((8, 16, 32, 64, 128), outputs):
                self.assertTrue(bool(mx.array_equal(output, gather).item()), f'real fixture differs from gather at S={splits}')
        finally:
            mx.set_default_device(device)

    @unittest.skipUnless(os.environ.get('MLX_QWEN4_QSA_INDEXED_TEST_METAL') == '1', 'set MLX_QWEN4_QSA_INDEXED_TEST_METAL=1 for the real Metal fixture')
    def test_real_m3_fixture_kernel_mirror_and_gather_are_bit_exact(self):
        device = mx.default_device()
        try:
            mx.set_default_device(mx.gpu)
            (q, k, v, compact) = _real_bf16_m3_fixture()
            kernel = indexed.qwen4_qsa_indexed_attention(q, k, v, compact, scale=256 ** (-0.5), splits=8)
            mirror = indexed.qwen4_qsa_indexed_reference(q, k, v, compact, scale=256 ** (-0.5), splits=8)
            gather = _gather_qsa_attention(q, k, v, compact, scale=256 ** (-0.5), tile_rows=1)
            mx.eval(kernel, mirror, gather)
            self.assertTrue(bool(mx.array_equal(kernel, mirror).item()))
            self.assertTrue(bool(mx.array_equal(mirror, gather).item()))
        finally:
            mx.set_default_device(device)

    def test_synchronous_dispatch_failure_reuses_fetched_kv_for_gather(self):
        mx.random.seed(31)
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3)
        expected = _gather_qsa_attention(q, k, v, compact, scale=8 ** (-0.5), tile_rows=1)
        seen = []

        def gather_spy(gq, gk, gv, *args, **kwargs):
            seen.append((gk is k, gv is v))
            return _gather_qsa_attention(gq, gk, gv, *args, **kwargs)
        indexed.qsa_indexed_status(reset=True)
        with mock.patch.object(qwen4_exp, 'qwen4_qsa_indexed_attention', side_effect=RuntimeError('synthetic dispatch failure')), mock.patch.object(qwen4_exp, '_gather_qsa_attention', gather_spy):
            actual = qwen4_exp._indexed_qsa_attention_or_gather(q, k, v, compact, scale=8 ** (-0.5), splits=4, tile_rows=1)
        mx.eval(actual, expected)
        self.assertEqual(seen, [(True, True)])
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        self.assertEqual(indexed.qsa_indexed_status()['counts']['dispatch_raised'], 1)
        self.assertEqual(indexed.qsa_indexed_status()['last_decision']['exception_class'], 'RuntimeError')

    def test_synchronous_dispatch_failure_preserves_requested_output_gate(self):
        mx.random.seed(32)
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3)
        gate = mx.random.normal((1, 3, 32))
        gathered = _gather_qsa_attention(q, k, v, compact, scale=8 ** (-0.5), tile_rows=1)
        expected = qwen4_exp.mlx_apply_output_gate(gathered, gate)
        with mock.patch.object(qwen4_exp, 'qwen4_qsa_indexed_attention', side_effect=RuntimeError('synthetic dispatch failure')):
            actual = qwen4_exp._indexed_qsa_attention_or_gather(q, k, v, compact, scale=8 ** (-0.5), splits=4, tile_rows=1, output_gate=gate)
        mx.eval(actual, expected)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_contract_value_error_does_not_fall_back(self):
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3)
        with mock.patch.object(qwen4_exp, 'qwen4_qsa_indexed_attention', side_effect=ValueError('contract failure')), mock.patch.object(qwen4_exp, '_gather_qsa_attention', side_effect=AssertionError('gather must not run')):
            with self.assertRaisesRegex(ValueError, 'contract failure'):
                qwen4_exp._indexed_qsa_attention_or_gather(q, k, v, compact, scale=8 ** (-0.5), splits=4, tile_rows=1)

    def test_probe_ladder_does_not_swallow_contract_value_error(self):
        (q, k, v, compact) = _real_bf16_fixture()
        indexed._PROBE_RESULTS.clear()
        with mock.patch.dict(os.environ, {'MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX': '1'}), mock.patch.object(indexed, 'indexed_kernel_available', return_value=True), mock.patch.object(indexed.mx, 'device_info', return_value={'architecture': 'applegpu_g17s'}), mock.patch.object(indexed, '_partition_dispatch', side_effect=ValueError('contract failure')):
            with self.assertRaisesRegex(ValueError, 'contract failure'):
                indexed.qwen4_qsa_indexed_attention(q, k, v, compact, scale=256 ** (-0.5), splits=8)

    def test_default_ladder_is_timed_and_fastest_candidate_wins(self):
        candidates = indexed._candidate_ladder(None, 12, 12)
        self.assertEqual(candidates, ((384, 128, 12), (384, 64, 12), (384, 32, 12), (384, 16, 12), (384, 8, 12)))
        calls = []

        def dispatch(candidate):
            calls.append(candidate)
            return (mx.array(candidate[1], dtype=mx.int32), mx.array(1, dtype=mx.uint32))
        clock = iter((0, 500, 1000, 1200, 2000, 2100, 3000, 3400, 4000, 4300))
        with mock.patch.object(indexed.time, 'perf_counter_ns', side_effect=lambda : next(clock)):
            (selected, output, counter, timings) = indexed._measure_candidates(candidates, dispatch)
        self.assertEqual(selected, (384, 32, 12))
        self.assertEqual(int(output.item()), 32)
        self.assertEqual(int(counter.item()), 1)
        self.assertEqual(set(timings), {(8, 12), (16, 12), (32, 12), (64, 12), (128, 12)})
        self.assertEqual(calls, list(candidates) * 2)

    def test_device_engagement_is_credited_only_after_reconciliation(self):
        indexed.qsa_indexed_status(reset=True)
        output = mx.array([7], dtype=mx.int32)
        for context in (16384, 16385):
            output = indexed._device_attest_output(output, mx.array([1], dtype=mx.uint32), length=3, context=context, splits=32, hpt=12, candidate=(384, 32, 12), geometry_key='B1-L3-U520-dtypemlx.core.bfloat16-mask0', candidate_timings_ms={(32, 12): 0.4})
        self.assertEqual(indexed._STATUS_COUNTS.get('engaged', 0), 0)
        mx.eval(output)
        status = indexed.qsa_indexed_status()
        self.assertEqual(status['counts']['engaged'], 2)
        self.assertEqual(status['device_attestation'], {'expected': 2, 'observed': 2, 'mismatches': 0, 'pending': 0})
        self.assertTrue(status['last_decision']['device_attested'])
        self.assertEqual(status['last_decision']['device_counter_observed'], 2)

    def test_quantized_dispatch_failure_uses_dequantized_gather(self):
        mx.random.seed(35)
        compact = _compact(1, 3)
        q = mx.random.normal((1, 4, 3, 32)).astype(mx.bfloat16)
        k = mx.random.normal((1, 2, 32, 32)).astype(mx.bfloat16)
        v = mx.random.normal((1, 2, 32, 32)).astype(mx.bfloat16)
        q_keys = mx.quantize(k, group_size=32, bits=8)
        q_values = mx.quantize(v, group_size=32, bits=8)
        expected = _gather_qsa_quantized_attention(q, q_keys, q_values, compact, scale=32 ** (-0.5), tile_rows=1, group_size=32, key_bits=8, value_bits=8)
        indexed.qsa_indexed_status(reset=True)
        with mock.patch.object(qwen4_exp, 'qwen4_qsa_indexed_quantized_attention', side_effect=RuntimeError('synthetic quantized dispatch failure')):
            actual = qwen4_exp._indexed_qsa_quantized_attention_or_gather(q, q_keys, q_values, compact, scale=32 ** (-0.5), splits=4, tile_rows=1, group_size=32, key_bits=8, value_bits=8)
        mx.eval(actual, expected)
        np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32)))
        status = indexed.qsa_indexed_status()
        self.assertEqual(status['counts']['quantized_dispatch_raised'], 1)
        self.assertEqual(status['fallbacks'], 1)

    def test_capture_writes_mismatch_and_returns_gather(self):
        mx.random.seed(37)
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3, dtype=mx.bfloat16)
        gather = _gather_qsa_attention(q, k, v, compact, scale=8 ** (-0.5), tile_rows=1)
        perturb = mx.zeros_like(gather)
        perturb[..., 0] = 0.02
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {'MLX_QWEN4_QSA_INDEXED_CAPTURE_DIR': root}), mock.patch.object(qwen4_exp, 'qwen4_qsa_indexed_attention', return_value=gather + perturb):
                actual = qwen4_exp._dispatch_qsa_indexed_with_optional_capture(q, k, v, compact, scale=8 ** (-0.5), splits=4, tile_rows=1, layer_index=7, call_counter=11, gather_would_admit=False)
            mx.eval(actual, gather)
            np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), np.asarray(gather.astype(mx.float32)))
            rows = [json.loads(line) for line in Path(root, 'calls.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['layer_index'], 7)
            self.assertEqual(rows[0]['call_counter'], 11)
            self.assertTrue(rows[0]['indexed_only_admission'])
            self.assertGreater(rows[0]['indexed_vs_gather_max_abs_fp32'], 0.004)
            captures = list(Path(root).glob('mismatch-*.safetensors'))
            self.assertEqual(len(captures), 1)
            self.assertTrue(captures[0].with_suffix('.json').exists())
            saved = mx.load(str(captures[0]))
            self.assertIn('causal_per_slot', saved)
            self.assertIn('indexed_output', saved)
            self.assertIn('gather_output', saved)

    def test_capture_records_fallback_and_returns_gather(self):
        mx.random.seed(41)
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3)
        expected = _gather_qsa_attention(q, k, v, compact, scale=8 ** (-0.5), tile_rows=1)
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {'MLX_QWEN4_QSA_INDEXED_CAPTURE_DIR': root}), mock.patch.object(qwen4_exp, 'qwen4_qsa_indexed_attention', side_effect=RuntimeError('synthetic dispatch failure')):
                actual = qwen4_exp._dispatch_qsa_indexed_with_optional_capture(q, k, v, compact, scale=8 ** (-0.5), splits=4, tile_rows=1, layer_index=2, call_counter=3, gather_would_admit=True)
            mx.eval(actual, expected)
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
            row = json.loads(Path(root, 'calls.jsonl').read_text())
            self.assertEqual(row['fallback'], 'dispatch_raised')
            self.assertIsNone(row['indexed_vs_gather_max_abs_fp32'])
            self.assertFalse(list(Path(root).glob('mismatch-*.safetensors')))

    def test_unset_capture_env_uses_normal_dispatch(self):
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3)
        sentinel = mx.zeros_like(q)
        gate = mx.zeros((1, 3, 32))
        with mock.patch.dict(os.environ, {'MLX_QWEN4_QSA_INDEXED_CAPTURE_DIR': ''}), mock.patch.object(qwen4_exp, '_capture_qsa_indexed_comparison', side_effect=AssertionError('capture path ran')), mock.patch.object(qwen4_exp, '_indexed_qsa_attention_or_gather', return_value=sentinel) as normal:
            actual = qwen4_exp._dispatch_qsa_indexed_with_optional_capture(q, k, v, compact, scale=8 ** (-0.5), splits=4, tile_rows=1, layer_index=0, call_counter=0, gather_would_admit=True, output_gate=gate)
        self.assertIs(actual, sentinel)
        normal.assert_called_once()
        self.assertIs(normal.call_args.kwargs['output_gate'], gate)
_LEGACY_PASS1_PREAMBLE = '    const uint lane = thread_index_in_simdgroup;\n    const uint head = simdgroup_index_in_threadgroup;\n    const uint row = threadgroup_position_in_grid.y;\n    const uint unit = threadgroup_position_in_grid.z;\n    const uint split = unit % S;\n    const uint bkv = unit / S;\n'
_HPT_PASS1_PREAMBLE = '    const uint lane = thread_index_in_simdgroup;\n    const uint row = threadgroup_position_in_grid.y;\n    const uint unit = threadgroup_position_in_grid.z;\n    const uint slices = GQA / HPT;\n    const uint hslice = unit % slices;\n    const uint rest = unit / slices;\n    const uint head = hslice * HPT + simdgroup_index_in_threadgroup;\n    const uint split = rest % S;\n    const uint bkv = rest / S;\n'

class TestQSAIndexedHeadsPerThreadgroup(unittest.TestCase):
    """HPT must change only which threadgroup owns a head, never the math."""

    def test_pass1_source_is_legacy_modulo_head_indexing(self):
        legacy = (Path(__file__).parent / 'fixtures' / 'qwen4_qsa_indexed_pass1_legacy.metal').read_text()
        self.assertIn(_HPT_PASS1_PREAMBLE, indexed._SOURCE)
        rebuilt = indexed._SOURCE.replace(_HPT_PASS1_PREAMBLE, _LEGACY_PASS1_PREAMBLE)
        self.assertEqual(rebuilt.strip('\n'), legacy.strip('\n'))

    def test_quantized_pass1_uses_the_same_head_decomposition(self):
        self.assertIn(_HPT_PASS1_PREAMBLE, indexed._QUANTIZED_SOURCE)

    def test_default_ladder_keeps_one_threadgroup_per_gqa_fan_out(self):
        """The 2026-09-02 sweep left HPT=GQA as the shipped probe grid."""
        ladder = indexed._candidate_ladder(None, 12)
        self.assertEqual(len(ladder), len(indexed._SPLIT_CANDIDATES))
        self.assertEqual({hpt for (_, _, hpt) in ladder}, {12})

    def test_opt_in_ladder_covers_the_split_by_hpt_grid(self):
        with mock.patch.object(indexed, '_HPT_CANDIDATES', indexed._HPT_LADDER):
            ladder = indexed._candidate_ladder(None, 12)
        self.assertEqual(len(ladder), 5 * 4)
        self.assertEqual(ladder[0], (384, 128, 12))
        self.assertEqual(ladder[-1], (32, 8, 1))
        for (threads, splits, hpt) in ladder:
            self.assertEqual(threads, hpt * 32)
            self.assertEqual(12 % hpt, 0)
            self.assertIn(splits, indexed._SPLIT_CANDIDATES)

    def test_candidate_ladder_pins_a_requested_pair(self):
        self.assertEqual(indexed._candidate_ladder(32, 12, 6), ((192, 32, 6),))

    def test_validate_hpt_rejects_unlisted_and_non_divisors(self):
        with self.assertRaises(ValueError):
            indexed.validate_hpt(5, 12)
        with self.assertRaises(ValueError):
            indexed.validate_hpt(4, 6)
        self.assertEqual(indexed.validate_hpt(3, 12), 3)

    def test_env_override_pins_heads_per_threadgroup(self):
        with mock.patch.object(indexed, '_HPT_OVERRIDE', 6):
            self.assertEqual(indexed.indexed_hpt_for(12), 6)
        with mock.patch.object(indexed, '_HPT_OVERRIDE', 0):
            self.assertEqual(indexed.indexed_hpt_for(12), 12)

    def test_pass1_dispatch_rejects_a_thread_count_that_is_not_hpt_simds(self):
        (q, k, v, compact) = _real_bf16_fixture()
        with self.assertRaises(ValueError):
            indexed._partition_dispatch(q, k, v, compact, scale=256 ** (-0.5), threads=384, splits=8, hpt=6)
        with self.assertRaises(ValueError):
            indexed._partition_dispatch(q, k, v, compact, scale=256 ** (-0.5), threads=160, splits=8, hpt=5)

    def test_receipt_keys_timings_by_split_and_hpt(self):
        indexed.qsa_indexed_status(reset=True)
        indexed.record_qsa_indexed_receipt(engaged=True, reason='engaged', length=3, context=65536, splits=64, hpt=3, candidate=(96, 64, 3), geometry_key='B1-L3-U520-hpt', candidate_timings_ms={(64, 3): 0.31, (64, 12): 0.44})
        status = indexed.qsa_indexed_status(reset=True)
        geometry = status['geometry_candidates']['B1-L3-U520-hpt']
        self.assertEqual(geometry['candidate'], [96, 64, 3])
        self.assertEqual(geometry['candidate_timings_ms']['64x3'], 0.31)
        self.assertEqual(status['last_decision']['heads_per_threadgroup'], 3)

    @unittest.skipUnless(os.environ.get('MLX_QWEN4_QSA_INDEXED_TEST_METAL') == '1', 'set MLX_QWEN4_QSA_INDEXED_TEST_METAL=1 for the real Metal fixture')
    def test_real_m3_fixture_is_bit_exact_across_the_hpt_grid(self):
        device = mx.default_device()
        try:
            mx.set_default_device(mx.gpu)
            (q, k, v, compact) = _real_bf16_m3_fixture()
            gather = _gather_qsa_attention(q, k, v, compact, scale=256 ** (-0.5), tile_rows=1)
            grid = [(splits, hpt) for splits in (8, 32, 128) for hpt in (12, 6, 4, 3, 2, 1)]
            outputs = {(splits, hpt): indexed.qwen4_qsa_indexed_attention(q, k, v, compact, scale=256 ** (-0.5), splits=splits, hpt=hpt) for (splits, hpt) in grid}
            mx.eval(gather, *outputs.values())
            reference = outputs[8, 12]
            for (point, output) in outputs.items():
                self.assertTrue(bool(mx.array_equal(output, gather).item()), f'S/HPT {point} differs from gather')
                self.assertTrue(bool(mx.array_equal(output, reference).item()), f'S/HPT {point} differs from (8, 12)')
        finally:
            mx.set_default_device(device)

    @unittest.skipUnless(os.environ.get('MLX_QWEN4_QSA_INDEXED_TEST_METAL') == '1', 'set MLX_QWEN4_QSA_INDEXED_TEST_METAL=1 for the real Metal fixture')
    def test_probe_cache_admits_hpt_as_part_of_the_key(self):
        device = mx.default_device()
        try:
            mx.set_default_device(mx.gpu)
            (q, k, v, compact) = _real_bf16_m3_fixture()
            indexed._PROBE_RESULTS.clear()
            for hpt in (12, 3):
                mx.eval(indexed.qwen4_qsa_indexed_attention(q, k, v, compact, scale=256 ** (-0.5), splits=32, hpt=hpt))
            selected = sorted(indexed._PROBE_RESULTS.values())
            self.assertEqual(selected, [(96, 32, 3), (384, 32, 12)])
        finally:
            indexed._PROBE_RESULTS.clear()
            mx.set_default_device(device)

class TestQSAIndexedAdmission(unittest.TestCase):

    def selection(self, **overrides):
        values = dict(kind='explicit', n_blocks=600, raw_block_ids=SimpleNamespace(shape=(1, 3, 512)), physical_width=16384)
        values.update(overrides)
        return SimpleNamespace(**values)

    def decide(self, selection=None, **kwargs):
        values = dict(length=3, training=False, layout_ok=True)
        values.update(kwargs)
        return indexed.decide_qsa_indexed_admission(self.selection() if selection is None else selection, **values)

    def test_admission_reason_matrix(self):
        with mock.patch.object(indexed, '_QSA_INDEXED_ENABLED', False):
            self.assertEqual(self.decide(), (False, 'disabled'))
        with mock.patch.object(indexed, '_QSA_INDEXED_ENABLED', True), mock.patch.object(indexed, 'indexed_kernel_available', return_value=True):
            cases = [({'training': True}, self.selection(), 'training'), ({}, self.selection(kind='implicit_all'), 'selection_not_explicit'), ({}, self.selection(n_blocks=512), 'dense_by_construction'), ({'length': 1}, self.selection(), 'width_out_of_range'), ({}, self.selection(physical_width=8192), 'context_out_of_range'), ({'layout_ok': False}, self.selection(), 'unsupported_layout')]
            for (kwargs, selection, reason) in cases:
                with self.subTest(reason=reason):
                    self.assertEqual(self.decide(selection=selection, **kwargs), (False, reason))
            self.assertEqual(self.decide(), (True, 'engaged'))
        with mock.patch.object(indexed, '_QSA_INDEXED_ENABLED', True), mock.patch.object(indexed, 'indexed_kernel_available', return_value=False):
            self.assertEqual(self.decide(), (False, 'kernel_unavailable'))

    def test_auto_admission_mode_matrix(self):
        with mock.patch.object(indexed, '_QSA_INDEXED_ENABLED', None), mock.patch.object(indexed, 'indexed_kernel_available', return_value=True), mock.patch.object(indexed, '_MAX_QUERY', 17):
            cases = [(1, 65535, False, 'auto_context_out_of_range'), (1, 65536, True, 'engaged'), (2, 16383, False, 'auto_context_out_of_range'), (2, 16384, True, 'engaged'), (3, 16384, True, 'engaged'), (9, 16384, True, 'engaged'), (12, 16384, True, 'engaged'), (15, 16384, True, 'engaged'), (16, 16384, True, 'engaged'), (17, 16384, True, 'engaged'), (18, 65536, False, 'width_out_of_range')]
            for (length, context, engage, reason) in cases:
                with self.subTest(length=length, context=context):
                    self.assertEqual(self.decide(selection=self.selection(physical_width=context), length=length), (engage, reason))

    def test_unverified_mlx_build_falls_back_with_specific_receipt(self):
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3)
        expected = _gather_qsa_attention(q, k, v, compact, scale=8 ** (-0.5), tile_rows=1)
        indexed.qsa_indexed_status(reset=True)
        with mock.patch.object(indexed, '_EXACT_MLX_BUILDS', frozenset({'wrong'})), mock.patch.dict(os.environ, {'MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX': ''}):
            actual = qwen4_exp._indexed_qsa_attention_or_gather(q, k, v, compact, scale=8 ** (-0.5), splits=4, tile_rows=1)
        mx.eval(actual, expected)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        self.assertEqual(indexed.qsa_indexed_status()['counts']['mlx_build_unverified'], 1)
        self.assertEqual(indexed.qsa_indexed_status()['fallbacks'], 1)

    def _current_build_allowlist(self):
        return mock.patch.object(indexed, '_EXACT_MLX_BUILDS', frozenset({str(getattr(mx, '__version__', 'unknown'))}))

    def _header_digest(self, digest):
        return mock.patch.object(indexed, '_sdpa_vector_header_sha256', return_value=digest)

    def _no_escape_hatch(self):
        return mock.patch.dict(os.environ, {'MLX_QWEN4_QSA_INDEXED_ALLOW_UNVERIFIED_MLX': ''})

    def test_absent_sdpa_header_digests_to_none(self):
        missing = Path(tempfile.gettempdir()) / 'qsa-absent-sdpa_vector.h'
        self.assertFalse(missing.exists())
        with mock.patch.object(indexed, '_sdpa_vector_header_path', return_value=missing):
            indexed._sdpa_vector_header_sha256.cache_clear()
            try:
                self.assertIsNone(indexed._sdpa_vector_header_sha256())
                self.assertEqual(indexed._sdpa_header_state(), (False, None, 'sdpa_header_missing'))
            finally:
                indexed._sdpa_vector_header_sha256.cache_clear()
        self.assertIsNotNone(indexed._sdpa_vector_header_sha256())

    def test_pinned_sdpa_header_admits_past_the_exactness_gate(self):
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3)
        indexed.qsa_indexed_status(reset=True)
        with self._current_build_allowlist(), self._header_digest(indexed._SDPA_VECTOR_HEADER_SHA256), self._no_escape_hatch():
            status = indexed.qsa_indexed_status()
            self.assertTrue(status['mlx_build_verified'])
            self.assertTrue(status['sdpa_header_verified'])
            self.assertEqual(status['sdpa_header_sha256_prefix'], indexed._SDPA_VECTOR_HEADER_SHA256[:12])
            qwen4_exp._indexed_qsa_attention_or_gather(q, k, v, compact, scale=8 ** (-0.5), splits=4, tile_rows=1)
        counts = indexed.qsa_indexed_status()['counts']
        self.assertNotIn('mlx_build_unverified', counts)
        self.assertNotIn('sdpa_header_mismatch', counts)
        self.assertNotIn('sdpa_header_missing', counts)

    def _assert_header_decline(self, digest, reason, prefix):
        compact = _compact(1, 3)
        (q, k, v) = _arrays(1, 3)
        expected = _gather_qsa_attention(q, k, v, compact, scale=8 ** (-0.5), tile_rows=1)
        indexed.qsa_indexed_status(reset=True)
        with self._current_build_allowlist(), self._header_digest(digest), self._no_escape_hatch():
            status = indexed.qsa_indexed_status()
            self.assertTrue(status['mlx_build_verified'])
            self.assertFalse(status['sdpa_header_verified'])
            self.assertEqual(status['sdpa_header_sha256_prefix'], prefix)
            actual = qwen4_exp._indexed_qsa_attention_or_gather(q, k, v, compact, scale=8 ** (-0.5), splits=4, tile_rows=1)
        mx.eval(actual, expected)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        status = indexed.qsa_indexed_status()
        self.assertEqual(status['counts'][reason], 1)
        self.assertEqual(status['fallbacks'], 1)
        self.assertEqual(status['last_decision']['reason'], reason)
        self.assertFalse(status['last_decision']['engaged'])

    def test_mismatched_sdpa_header_falls_back_with_specific_receipt(self):
        self._assert_header_decline('0' * 64, 'sdpa_header_mismatch', '0' * 12)

    def test_missing_sdpa_header_falls_back_with_specific_receipt(self):
        self._assert_header_decline(None, 'sdpa_header_missing', None)

    def test_quantized_admission_reason_matrix(self):
        cache = SimpleNamespace(group_size=64, key_bits=8, value_bits=8, rotate=False, normalize=False)
        with mock.patch.object(indexed, '_QSA_INDEXED_ENABLED', True), mock.patch.object(indexed, 'indexed_kernel_available', return_value=True):
            self.assertEqual(self.decide(cache=cache), (True, 'engaged'))
            self.assertEqual(self.decide(cache=SimpleNamespace(**{**vars(cache), 'group_size': 16})), (False, 'quantized_group_size_unsupported'))
            self.assertEqual(self.decide(cache=SimpleNamespace(**{**vars(cache), 'key_bits': 3})), (False, 'quantized_bits_unsupported'))
            self.assertEqual(self.decide(cache=SimpleNamespace(**{**vars(cache), 'rotate': True})), (False, 'quantized_transform_unsupported'))

    def test_runtime_refusal_reasons_and_width_buckets_are_receipted(self):
        indexed.qsa_indexed_status(reset=True)
        for (reason, width) in (('nax_engaged', 1), ('probe_declined', 3), ('dispatch_raised', 12), ('width_out_of_range', 32)):
            indexed.record_qsa_indexed_receipt(engaged=False, reason=reason, length=width, context=32768, splits=4)
        indexed.record_qsa_indexed_receipt(engaged=True, reason='engaged', length=3, context=32768, splits=32, hpt=12, candidate=(384, 32, 12), geometry_key='B1-L3-T32768-U520-mask1', candidate_timings_ms={(128, 12): 0.7, (64, 12): 0.5, (32, 12): 0.4, (16, 12): 0.6, (8, 12): 0.8})
        status = indexed.qsa_indexed_status()
        self.assertEqual(status['counts']['nax_engaged'], 1)
        self.assertEqual(status['fallbacks'], 2)
        self.assertEqual(status['query_width_counts']['1']['declined'], 1)
        self.assertEqual(status['query_width_counts']['2-8']['declined'], 1)
        self.assertEqual(status['query_width_counts']['9-17']['declined'], 1)
        self.assertEqual(status['query_width_counts']['>17']['declined'], 1)
        self.assertEqual(status['split_candidates'], [128, 64, 32, 16, 8])
        self.assertEqual(status['hpt_candidates'], [12])
        self.assertFalse(status['hpt_ladder_enabled'])
        geometry = status['geometry_candidates']['B1-L3-T32768-U520-mask1']
        self.assertEqual(geometry['candidate'], [384, 32, 12])
        self.assertEqual(geometry['candidate_timings_ms']['32x12'], 0.4)

    def test_shipped_indexed_window_is_the_gated_default(self):
        """The gate measured no gain from widening, so the default stays 8."""
        self.assertEqual(indexed._MAX_QUERY, 8)
        with mock.patch.object(indexed, '_QSA_INDEXED_ENABLED', None), mock.patch.object(indexed, 'indexed_kernel_available', return_value=True):
            self.assertEqual(self.decide(selection=self.selection(physical_width=16384), length=17), (False, 'width_out_of_range'))

    def test_env_unset_selects_guarded_auto(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(indexed._env_mode('MLX_QWEN4_QSA_INDEXED'))

    def test_explicit_zero_keeps_dispatch_decisions_byte_identical(self):
        with mock.patch.dict(os.environ, {'MLX_QWEN4_QSA_INDEXED': '0'}, clear=False):
            hard_off = indexed._env_mode('MLX_QWEN4_QSA_INDEXED')
        self.assertFalse(hard_off)
        with mock.patch.object(indexed, '_QSA_INDEXED_ENABLED', hard_off):
            for use_nax in (False, True):
                for gather_enabled in (False, True):
                    (engage, reason) = self.decide()
                    self.assertFalse(engage)
                    self.assertEqual(reason, 'disabled')
                    old = (use_nax, gather_enabled and (not use_nax), not (use_nax or (gather_enabled and (not use_nax))))
                    new_gather = gather_enabled and (not use_nax) and (not engage)
                    new = (use_nax, new_gather, not (use_nax or engage or new_gather))
                    self.assertEqual(repr(old).encode(), repr(new).encode())

    def test_split_table_and_override(self):
        expected = {8: 128, 64: 128, 520: 128}
        with mock.patch.object(indexed, '_SPLITS_OVERRIDE', 0):
            self.assertEqual({width: indexed.indexed_splits_for(width) for width in expected}, expected)
        with mock.patch.object(indexed, '_SPLITS_OVERRIDE', 6):
            with self.assertRaisesRegex(ValueError, '8, 16, 32, 64, 128'):
                indexed.indexed_splits_for(520)
        with mock.patch.object(indexed, '_SPLITS_OVERRIDE', 64):
            self.assertEqual(indexed.indexed_splits_for(520), 64)
        with self.assertRaises(ValueError):
            indexed.indexed_splits_for(0)
if __name__ == '__main__':
    unittest.main()
