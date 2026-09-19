"""NumPy oracle for production KV/segmented logic; no MLX import or GPU calls.

Execute the actual production class bodies with NumPy array primitives and
an independent attention reduction. GPU/native behavior remains unqualified.
"""

import ast
from collections import Counter
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).parents[1] / "src/mlx2/runtime"


def attention(q, k, v, *, scale, mask=None, sinks=None):
    if sinks is not None:
        raise NotImplementedError("CPU oracle has no attention sinks")
    repeat = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, repeat, axis=1), np.repeat(v, repeat, axis=1)
    scores = q @ k.swapaxes(-1, -2) * scale
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs @ v


def load_cpu_classes():
    mx = SimpleNamespace(
        **{
            n: getattr(np, n)
            for n in (
                "array",
                "zeros",
                "arange",
                "concatenate",
                "pad",
                "expand_dims",
                "int32",
            )
        }
    )
    mx.contiguous = np.ascontiguousarray
    mx.fast = SimpleNamespace(scaled_dot_product_attention=attention)
    namespace = {"mx": mx, "Integral": Integral, "_BaseCache": object}
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    nodes = [future]
    base = ast.parse((ROOT / "models/base.py").read_text())
    nodes.extend(
        n
        for n in base.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_causal_mask"
    )
    cache = ast.parse((ROOT / "models/cache.py").read_text())
    nodes.extend(
        n for n in cache.body if isinstance(n, ast.ClassDef) and n.name == "KVCache"
    )
    segmented = ast.parse((ROOT / "segmented_plain_kv.py").read_text())
    nodes.extend(n for n in segmented.body if isinstance(n, ast.ClassDef))
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            "production-cache-cpu-oracle",
            "exec",
        ),
        namespace,
    )
    return namespace["KVCache"], namespace["SegmentedBatchKVCache"]


KVCache, SegmentedBatchKVCache = load_cpu_classes()


class SegmentedPlainKVTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(419)

    def arrays(self, batch, width):
        return (
            self.rng.normal(size=(batch, 2, width, 4)).astype(np.float32),
            self.rng.normal(size=(batch, 2, width, 3)).astype(np.float32),
        )

    def rows(self, lengths):
        result = []
        for length in lengths:
            row = KVCache()
            if length:
                row.update_and_fetch(*self.arrays(1, length))
            result.append(row)
        return result

    def assert_step(self, view, lengths):
        before = [r.offset for r in view.rows]
        width = max(lengths)
        padding = [width - n for n in lengths]
        view.prepare(lengths=lengths, right_padding=padding)
        np.testing.assert_array_equal(view.offset, before)
        mask = view.make_mask(width)
        keys, values = self.arrays(len(lengths), width)
        queries = self.rng.normal(size=(len(lengths), 4, width, 4)).astype(np.float32)
        self.assertEqual(view.update_and_fetch(keys, values), (None, None))
        actual = view.bucketed_attention(queries, 0.5, mask)
        for i, (row, valid, offset) in enumerate(zip(view.rows, lengths, before)):
            self.assertEqual(row.offset, offset + valid)
            if valid:
                rk, rv = row.keys_and_values()
                expected_mask = np.arange(offset + valid)[None, :] <= (
                    offset + np.arange(valid)[:, None]
                )
                expected = attention(
                    queries[i : i + 1, :, :valid], rk, rv, scale=0.5, mask=expected_mask
                )
                np.testing.assert_allclose(
                    actual[i : i + 1, :, :valid], expected, rtol=1e-6, atol=1e-6
                )
                np.testing.assert_array_equal(
                    rk[..., offset:, :], keys[i : i + 1, :, :valid]
                )
            np.testing.assert_array_equal(actual[i : i + 1, :, valid:], 0)
        self.assertIsNone(view.keys)
        self.assertIsNone(view.values)
        view.finalize()
        return actual

    def test_ragged_prefix_verify_decode_oracle(self):
        counts = Counter()
        view = SegmentedBatchKVCache(
            self.rows([3, 7, 1]), note=lambda k, n: counts.update({k: n})
        )
        self.assert_step(view, [3, 1, 2])
        self.assert_step(view, [1, 1, 1])
        self.assertEqual(counts["segmented_attention_calls"], 2)
        self.assertEqual(counts["independent_lineages_consumed"], 6)
        self.assertEqual(counts["full_prefix_materializations"], 0)

    def test_empty_prefix_and_zero_length_row(self):
        view = SegmentedBatchKVCache(self.rows([0, 4, 0]))
        self.assert_step(view, [2, 0, 1])
        self.assertEqual([r.offset for r in view.rows], [2, 4, 1])

    def test_ragged_trim_then_append_and_b2_to_b1(self):
        view = SegmentedBatchKVCache(self.rows([4, 7]))
        self.assert_step(view, [3, 3])
        originals = [tuple(a.copy() for a in r.keys_and_values()) for r in view.rows]
        self.assertEqual(view.trim_ragged([3, 1]), [3, 1])
        self.assertEqual([r.offset for r in view.rows], [4, 9])
        for row, original in zip(view.rows, originals):
            np.testing.assert_array_equal(
                row.keys_and_values()[0], original[0][..., : row.offset, :]
            )
        self.assert_step(view, [1, 2])
        remaining = view.rows[1]
        view.filter([1])
        self.assertIs(view.rows[0], remaining)
        self.assert_step(view, [1])

    def test_trim_validation_is_atomic(self):
        view = SegmentedBatchKVCache(self.rows([2, 5]))
        for counts in ([1, 6], [-1, 0], [1], [1.0, 0]):
            with self.assertRaises(ValueError):
                view.trim_ragged(counts, validate=False)
            self.assertEqual([r.offset for r in view.rows], [2, 5])

    def test_extract_independent_snapshot(self):
        view = SegmentedBatchKVCache(self.rows([3, 5]))
        snapshot = view.extract(0)
        expected = snapshot.keys_and_values()[0].copy()
        view.rows[0].trim(2)
        view.rows[0].update_and_fetch(*self.arrays(1, 2))
        np.testing.assert_array_equal(snapshot.keys_and_values()[0], expected)
        self.assertEqual(snapshot.offset, 3)

    def test_preparation_and_stale_row_rejected(self):
        view = SegmentedBatchKVCache(self.rows([2, 3]))
        with self.assertRaises(ValueError):
            view.prepare(lengths=[2, 1], right_padding=[0, 0])
        view.prepare(lengths=[1, 1])
        with self.assertRaises(RuntimeError):
            view.prepare(lengths=[1, 1])
        view.rows[0].trim(1)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            view.update_and_fetch(*self.arrays(2, 1))
        self.assertEqual([r.offset for r in view.rows], [1, 3])

    def test_geometry_checks_happen_before_any_write(self):
        view = SegmentedBatchKVCache(self.rows([2, 5]))
        view.prepare(lengths=[1, 1])
        k, v = self.arrays(2, 1)
        with self.assertRaises(ValueError):
            view.update_and_fetch(k[:, :1], v[:, :1])
        self.assertEqual([r.offset for r in view.rows], [2, 5])

    def test_ownership_and_membership_guards(self):
        row = self.rows([2])[0]
        with self.assertRaises(ValueError):
            SegmentedBatchKVCache([row, row])
        view = SegmentedBatchKVCache(self.rows([2, 4]))
        for indices in ([], [0, 0], [-1], [2]):
            with self.assertRaises(ValueError):
                view.filter(indices)
        view.prepare(lengths=[1, 1])
        with self.assertRaises(RuntimeError):
            view.filter([1])

    def test_window_mask_crop_matches_standalone(self):
        view = SegmentedBatchKVCache(self.rows([2, 9]))
        view.prepare(lengths=[2, 1], right_padding=[0, 1])
        mask = view.make_mask(2, window_size=3)
        q = self.rng.normal(size=(2, 4, 2, 4)).astype(np.float32)
        view.update_and_fetch(*self.arrays(2, 2))
        output = view.bucketed_attention(q, 0.5, mask)
        for i, (base, valid) in enumerate(zip([2, 9], [2, 1])):
            keys, values = view.rows[i].keys_and_values()
            pos = base + np.arange(valid)[:, None]
            kpos = np.arange(base + valid)[None, :]
            expected = attention(
                q[i : i + 1, :, :valid],
                keys,
                values,
                scale=0.5,
                mask=(pos >= kpos) & (pos < kpos + 3),
            )
            np.testing.assert_allclose(
                output[i : i + 1, :, :valid], expected, rtol=1e-6, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
