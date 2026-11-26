"""Tree-verify technology: exact parity with the copies it replaced (CPU).

Every ``_legacy_*`` function below is the pre-consolidation code, moved here
verbatim (only renamed, with ``self`` spelled as an argument where it was a
method).  The tests assert that the shared topology accepts and refuses the
same parent arrays with the same derived paths/depths/windows, that the
shared KV/recurrent/Mamba commit ops build bit-identical committed planes,
that the serial reference publishes bit-identical caches and logits, and
that the executor-choice helper picks what each adapter picked.
"""

from __future__ import annotations

import copy
import os
import random
from collections import deque

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.technologies import tree_verify as TV

mx.set_default_device(mx.cpu)


# ---------------------------------------------------------------------------
# Legacy topology copies
# ---------------------------------------------------------------------------


def _legacy_ling_topology(parents):
    # runtime/models/ling_tree.py LingTree.__init__
    if not 1 <= len(parents) <= 32:
        raise ValueError("Ling tree requires 1..32 rows")
    paths = []
    depths = []
    for row, parent in enumerate(parents):
        if type(parent) is not int or (
            (row == 0 and parent != -1) or (row and not 0 <= parent < row)
        ):
            raise ValueError("Ling tree requires one parent-first root")
        path = [] if parent < 0 else paths[parent]
        paths.append([*path, row])
        depths.append(len(path))
    levels = [
        [row for row, depth in enumerate(depths) if depth == level]
        for level in range(max(depths) + 1)
    ]
    return paths, depths, levels


def _legacy_nemotron_topology(parents):
    # runtime/models/nemotron_tree.py NemotronTree.__init__
    parents = tuple(parents)
    if not 1 <= len(parents) <= 16:
        raise ValueError("Nemotron tree requires 1..16 rows")
    paths = []
    depths = []
    for row, parent in enumerate(parents):
        if type(parent) is not int or (
            (row == 0 and parent != -1) or (row and not 0 <= parent < row)
        ):
            raise ValueError("Nemotron tree requires one parent-first root")
        path = [] if parent < 0 else paths[parent]
        paths.append([*path, row])
        depths.append(len(path))
    chain_prefix = [path == list(range(len(path))) for path in paths]
    return paths, depths, chain_prefix


def _legacy_inkling_topology(parents):
    # runtime/models/inkling/tree.py InklingTree.__init__
    if not 1 <= len(parents) <= 16:
        raise ValueError("Inkling trees require 1..16 rows")
    paths = []
    for row, parent in enumerate(parents):
        if type(parent) is not int or (
            (row == 0 and parent != -1) or (row > 0 and not 0 <= parent < row)
        ):
            raise ValueError("Inkling tree must have one parent-first root")
        paths.append([*(paths[parent] if parent >= 0 else []), row])
    return paths


def _legacy_hybrid_topology(tokens, parents):
    # runtime/hybrid_tree_executor.py tree_forward (validation only)
    if not 1 <= len(tokens) <= 16 or len(tokens) != len(parents):
        raise ValueError("hybrid AST tile must contain 1..16 matching rows")
    for row, (value, parent) in enumerate(zip(tokens, parents, strict=True)):
        if (
            type(parent) is not int
            or (row == 0 and parent != -1)
            or (row > 0 and not 0 <= parent < row)
        ):
            raise ValueError(
                "hybrid AST parents must form one parent-first rooted tree"
            )


def _legacy_attention_depths(parents):
    # runtime/attention_tree_executor.py _depths
    depths = []
    for row, parent in enumerate(parents):
        parent = int(parent)
        if parent < 0:
            depths.append(0)
        elif parent >= row:
            raise ValueError("tree parents must precede their children")
        else:
            depths.append(depths[parent] + 1)
    return depths


def _legacy_attention_ancestors(parents):
    # runtime/attention_tree_executor.py _ancestors
    width = len(parents)
    rows = [[False] * width for _ in range(width)]
    for row, parent in enumerate(parents):
        rows[row][row] = True
        node = int(parent)
        while node >= 0:
            rows[row][node] = True
            node = int(parents[node])
    return rows


def _legacy_attention_tile(tokens, parents):
    # runtime/attention_tree_executor.py tile_forward (validation) + _TileScope
    if not 1 <= len(tokens) <= 32 or len(tokens) != len(parents):
        raise ValueError("attention tree tile rows must match parents and fit 32 rows")
    if int(parents[0]) != -1 or any(int(p) < 0 for p in parents[1:]):
        raise ValueError("attention tree tile must be rooted at its anchor row")
    return _legacy_attention_depths(list(parents))


def _legacy_qwen_gdn_tree_paths(parents):
    # runtime/models/qwen38_tree_gdn.py tree_paths
    depths = []
    paths = []
    for row, parent in enumerate(parents):
        parent = int(parent)
        if parent < 0:
            path = [row]
        else:
            if parent >= row:
                raise ValueError("parents must come before their children")
            path = paths[parent] + [row]
        paths.append(path)
        depths.append(len(path) - 1)
    return depths, paths


def _legacy_dflash_tree_paths(parents):
    # runtime/drafters/dflash_tree.py tree_paths
    paths = []
    for row, parent in enumerate(parents):
        parent = int(parent)
        if parent < 0:
            path = [row]
        elif parent >= row:
            raise ValueError("tree parents must precede their children")
        else:
            path = paths[parent] + [row]
        paths.append(path)
    return paths


def _legacy_flash_ancestry(parents):
    # runtime/models/qwen4_tree_gdn.py ancestry
    parents = tuple(parents)
    if (not 1 <= len(parents) <= 16 or type(parents[0]) is not int
        or parents[0] != -1 or any(type(p) is not int or not 0 <= p < i
                                  for i, p in enumerate(parents[1:], 1))):
        raise ValueError("Flash tree requires one parent-first root and 1..16 rows")
    paths = []
    for row, parent in enumerate(parents):
        paths.append((row,) if parent < 0 else (*paths[parent], row))
    return tuple(paths)


def _legacy_qwen38_rooted_tree(inputs, parents):
    # adapters/qwen38_target_tree_executor.py _rooted_tree
    return (
        1 <= len(inputs) <= 16
        and len(parents) == len(inputs)
        and type(parents[0]) is int
        and parents[0] == -1
        and all(
            type(parent) is int and 0 <= parent < index
            for index, parent in enumerate(parents[1:], 1)
        )
    )


def _legacy_glm_rooted(inputs, parents):
    # adapters/glm53_ast_target.py Glm53TargetTreeExecutor.forward (topology)
    if (
        not 1 <= len(inputs) <= 32
        or len(parents) != len(inputs)
        or parents[0] != -1
        or any(
            type(p) is not int or not 0 <= p < i
            for i, p in enumerate(parents[1:], 1)
        )
    ):
        raise ValueError(
            "GLM-5.3 AST requires a parent-ordered rooted tree of at most32 rows"
        )


def _legacy_nanbeige_rooted(tree):
    # adapters/nanbeige43_tree.py NanbeigePackedTreeExecutor.forward_many
    if tree[0] != -1 or any(type(p) is not int or not 0 <= p < row for row, p in enumerate(tree[1:], start=1)):
        raise ValueError("packed tree parents must form a rooted topological tree")


def _legacy_ling_windows(paths, kernel_size):
    # runtime/models/ling_tree.py LingTree.convolution_windows (indices)
    count = kernel_size - 1
    windows = []
    for path in paths:
        history = [*range(count), *(count + row for row in path)]
        windows.append(history[-kernel_size:])
    return windows


def _legacy_nemotron_window_rows(paths, kernel_size):
    # runtime/models/nemotron_tree.py NemotronTree.window_rows
    count = kernel_size - 1
    windows = []
    for path in paths:
        history = [*range(count), *(count + row for row in path[:-1])]
        windows.append(history[-count:])
    return windows


def _legacy_kimi_merged_indices(paths, conv_kernel):
    # runtime/models/kimi_tree.py KimiTree._merged_convolution (indices)
    count = conv_kernel - 1
    indices = []
    for path in paths:
        window = [*range(count), *(count + row for row in path)]
        indices.append(window[-conv_kernel:])
    return indices


def _legacy_root_path(paths, parents, path):
    # Ling/Nemotron/Inkling validate_path (identical bodies)
    path = list(path)
    return not (
        not path or any(type(row) is not int or not 0 <= row < len(parents) for row in path)
        or path != paths[path[-1]]
    )


def _legacy_attention_path(parents, width, path):
    # runtime/attention_tree_executor.py AttentionTreeTransaction.validate_path
    path = tuple(path)
    return not (
        not path
        or any(type(row) is not int or not 0 <= row < width for row in path)
        or path[0] != 0
        or any(left >= right for left, right in zip(path, path[1:]))
        or (parents is not None and any(
            parents[row] != parent
            for parent, row in zip((-1, *path[:-1]), path)
        ))
    )


# ---------------------------------------------------------------------------
# Topology parity
# ---------------------------------------------------------------------------


def _outcome(function, *args):
    try:
        return ("ok", function(*args))
    except (TypeError, ValueError, IndexError) as error:
        return ("error", type(error).__name__, str(error))


def _random_parents(rng, width):
    kind = rng.random()
    parents = []
    for row in range(width):
        if row == 0:
            parents.append(-1 if kind < 0.85 else rng.choice([0, -2, 1, -1.0, True]))
            continue
        roll = rng.random()
        if roll < 0.80:
            parents.append(rng.randrange(row))
        elif roll < 0.86:
            parents.append(-1)
        elif roll < 0.90:
            parents.append(rng.randrange(row, row + 3))
        elif roll < 0.93:
            parents.append(np.int64(rng.randrange(row)))
        elif roll < 0.96:
            parents.append(float(rng.randrange(row)))
        elif roll < 0.98:
            parents.append(-3)
        else:
            parents.append(False)
    return parents


def _cases():
    rng = random.Random(1234)
    cases = [[-1], [-1, 0], [-1, 0, 1, 1, 0, 4], [-1, -1], [0], [-1, 1], [], [-1, 0, 2],
             list(range(-1, 15)), list(range(-1, 16)), list(range(-1, 31)), list(range(-1, 32)),
             [-1] + [0] * 31, [-1] + [0] * 32, [-1, np.int64(0)], [np.int64(-1), 0],
             [-1.0, 0], [-1, 0.0], [True], [-1, True], [-1, -2], [-1, 0, -1, 2]]
    for _ in range(1500):
        cases.append(_random_parents(rng, rng.randint(1, 36)))
    return cases


CASES = _cases()


def _new(name, parents):
    return TV.TreeTopology(parents, TV.legacy_topology(name))


@pytest.mark.parametrize("name,legacy,project", [
    ("ling_packed", _legacy_ling_topology,
     lambda t: (t.path_lists(), list(t.depths), t.levels)),
    ("nemotron_packed", _legacy_nemotron_topology,
     lambda t: (t.path_lists(), list(t.depths), t.chain_prefix())),
    ("inkling_packed", _legacy_inkling_topology, lambda t: t.path_lists()),
    ("flash_tree_ancestry", _legacy_flash_ancestry, lambda t: t.paths),
    ("qwen_tree_gdn_paths", _legacy_qwen_gdn_tree_paths,
     lambda t: (list(t.depths), t.path_lists())),
    ("dflash_tree_paths", _legacy_dflash_tree_paths, lambda t: t.path_lists()),
    ("attention_depths", _legacy_attention_depths, lambda t: list(t.depths)),
])
def test_topology_matches_each_legacy_copy(name, legacy, project):
    for parents in CASES:
        expected = _outcome(legacy, parents)
        actual = _outcome(lambda p: project(_new(name, p)), parents)
        assert actual == expected, parents


def test_attention_ancestors_match_legacy():
    for parents in CASES:
        try:
            _legacy_attention_depths(parents)
        except (TypeError, ValueError):
            continue
        assert _new("attention_depths", parents).ancestor_rows() == _legacy_attention_ancestors(parents)


def test_attention_tile_root_rule_matches_legacy():
    for parents in CASES:
        tokens = list(range(len(parents)))
        expected = _outcome(_legacy_attention_tile, tokens, parents)

        def new(tokens, parents):
            from mlx2.runtime.attention_tree_executor import TILE_TOPOLOGY, _depths

            if len(tokens) != len(parents):
                raise ValueError(TILE_TOPOLOGY.rows_error)
            TV.TreeTopology(parents, TILE_TOPOLOGY)
            return _depths(list(parents))

        assert _outcome(new, tokens, parents) == expected, parents


@pytest.mark.parametrize("name,legacy", [
    ("hybrid_serial", _legacy_hybrid_topology),
    ("glm_rooted_tree", _legacy_glm_rooted),
])
def test_tile_entry_points_accept_the_legacy_set(name, legacy):
    spec = TV.legacy_topology(name)
    for parents in CASES:
        tokens = list(range(len(parents)))
        expected = _outcome(legacy, tokens, parents)

        def new(tokens, parents):
            if len(tokens) != len(parents):
                raise ValueError(spec.rows_error)
            TV.TreeTopology(parents, spec)

        assert _outcome(new, tokens, parents) == expected, parents


def test_rooted_tree_predicate_matches_qwen38():
    from mlx2.adapters.qwen38_target_tree_executor import _rooted_tree

    for parents in CASES:
        if not parents:
            continue
        tokens = list(range(len(parents)))
        assert _rooted_tree(tokens, parents) == _legacy_qwen38_rooted_tree(tokens, parents), parents


@pytest.mark.parametrize("name", ["packed_attention_tile", "q8_attention_tile"])
def test_packed_attention_tile_rule_matches_legacy(name):
    spec = TV.legacy_topology(name)
    for parents in CASES:
        if not 1 <= len(parents) <= 32:
            continue  # the executors reject widths before parents
        expected = _outcome(_legacy_nanbeige_rooted, parents)
        actual = _outcome(lambda p: (TV.TreeTopology(p, spec), None)[1], parents)
        assert (actual[0], actual[1:2]) == (expected[0], expected[1:2]), parents


def test_history_windows_match_ling_nemotron_and_kimi():
    for parents in CASES:
        try:
            paths, _depths, _levels = _legacy_ling_topology(parents)
        except (TypeError, ValueError):
            continue
        topology = _new("ling_packed", parents)
        for kernel in (2, 3, 4, 5):
            ling = topology.history_array(kernel - 1, kernel, include_self=True)
            assert ling.tolist() == _legacy_ling_windows(paths, kernel)
            assert ling.tolist() == _legacy_kimi_merged_indices(paths, kernel)
            assert (topology.history_rows(kernel - 1, kernel - 1, include_self=False)
                    == _legacy_nemotron_window_rows(paths, kernel))


def test_root_path_validation_matches_legacy():
    rng = random.Random(7)
    for parents in CASES:
        try:
            paths, _depths, _levels = _legacy_ling_topology(parents)
        except (TypeError, ValueError):
            continue
        topology = _new("ling_packed", parents)
        width = len(parents)
        candidates = [[], *paths, [0, 0], [width], [-1], [0, width - 1], (0,), [0.0]]
        candidates += [rng.sample(range(width), rng.randint(1, width)) for _ in range(6)]
        candidates += [sorted(item) for item in candidates[-6:]]
        for path in candidates:
            assert topology.is_root_path(path) == _legacy_root_path(paths, parents, path)
            # The attention executor's publication rule (with parents bound).
            assert (_legacy_attention_path(tuple(parents), width, path)
                    == (bool(path) and path[0] == 0 and topology.is_root_path(tuple(path))))


# ---------------------------------------------------------------------------
# Commit parity: KV, recurrent terminal, Mamba replay
# ---------------------------------------------------------------------------


def _legacy_ling_prepare_commit(self, path):
    # runtime/models/ling_tree.py LingTree.prepare_commit (validate_path elided)
    path = list(path)
    indices = mx.array(path, dtype=mx.int32)
    prepared = []
    for live, private in zip(self.live, self.private, strict=True):
        if type(live) is KVCache:
            start = int(live.offset)
            for field in ("keys", "values"):
                values = getattr(private, field)
                kept = mx.take(values[..., start:start + len(self.parents), :], indices, axis=2)
                values[..., start:start + len(path), :] = kept
                setattr(private, field, values)
            private.offset = start + len(path)
        else:
            qs, ks, vs, states = self.recurrent[id(private)]
            terminal = path[-1]
            private.cache = [qs[terminal:terminal + 1], ks[terminal:terminal + 1],
                             vs[terminal:terminal + 1], states[terminal]]
            private.advance(len(path))
        prepared.append((live, private.__dict__.copy()))
    return prepared


def _legacy_nemotron_prepare_commit(self, path):
    # runtime/models/nemotron_tree.py NemotronTree.prepare_commit
    path = list(path)
    indices = mx.array(path, dtype=mx.int32)
    prepared = []
    for live in self.live:
        if type(live) is KVCache:
            entry = self._kv.get(id(live))
            if entry is None:
                raise RuntimeError("Nemotron tree attention layer did not run")
            private, start = entry["private"], entry["start"]
            if not self.chain_prefix[path[-1]]:
                for field, rows in (("keys", entry["keys"]), ("values", entry["values"])):
                    buffer = getattr(private, field)
                    buffer[..., start : start + len(path), :] = mx.take(
                        rows, indices, axis=2
                    )
                    setattr(private, field, buffer)
            private.offset = start + len(path)
            prepared.append((live, dict(private.__dict__)))
            continue
        material = self._mamba.get(id(live))
        if material is None:
            raise RuntimeError("Nemotron tree Mamba layer did not run")
        state = material["mixer"].tree_commit_state(material, path, indices)
        count = material["kernel"] - 1
        history = [*range(count), *(count + row for row in path)]
        conv = mx.take(
            material["source"], mx.array(history[-count:], dtype=mx.int32), axis=1
        )
        private = copy.copy(live)
        private.__dict__ = dict(live.__dict__)
        private.cache = [conv, state]
        private._rollbacks = deque()
        private.advance(len(path))
        prepared.append((live, dict(private.__dict__)))
    return prepared


def _legacy_inkling_global_kv(private_kv, live_offset, paths, raw_kv, path):
    # runtime/models/inkling/tree.py InklingTree.prepare_commit (global KV)
    ids = mx.array(path, mx.int32)
    start = int(live_offset)
    if path != list(range(len(path))):
        for field, raw in zip(("keys", "values"), raw_kv, strict=True):
            values = getattr(private_kv, field)
            source = (
                raw
                if raw.dtype == values.dtype
                else values[..., start : start + len(paths), :]
            )
            values[..., start : start + len(path), :] = mx.take(source, ids, axis=2)
            setattr(private_kv, field, values)
    private_kv.offset = start + len(path)


def _legacy_inkling_conv(conv, history, path):
    # runtime/models/inkling/tree.py InklingTree.prepare_commit (short-conv)
    conv.cache = [history[i][path[-1]] for i in range(4)]
    conv.advance(len(path))


def _arrays(fields):
    return {key: value for key, value in fields.items()}


def _assert_same_fields(left, right):
    assert set(left) == set(right)
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, mx.array):
            assert isinstance(b, mx.array) and a.dtype == b.dtype and mx.array_equal(a, b).item(), key
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                if x is None:
                    assert y is None
                else:
                    assert x.dtype == y.dtype and mx.array_equal(x, y).item(), key
        elif isinstance(a, deque):
            assert list(a) == list(b)
        else:
            assert a == b, key


def _kv(prefix, heads, dim, dtype, seed):
    mx.random.seed(seed)
    cache = KVCache()
    cache.update_and_fetch(mx.random.normal((1, heads, prefix, dim)).astype(dtype),
                           mx.random.normal((1, heads, prefix, dim)).astype(dtype))
    return cache


def _arrays_cache(dtype, seed):
    mx.random.seed(seed)
    cache = ArraysCache(4)
    for index in range(3):
        cache[index] = mx.random.normal((1, 3, 8)).astype(dtype)
    cache[3] = mx.random.normal((1, 2, 4, 4))
    return cache


TREES = [[-1], [-1, 0, 1, 2], [-1, 0, 0, 1, 2, 2, 5], [-1, 0, 1, 0, 3, 3, 1, 6]]


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("parents", TREES)
def test_ling_commit_is_bit_identical(dtype, parents):
    from mlx2.runtime.models.ling_tree import LingTree

    width = len(parents)

    def build():
        cache = [_kv(5, 2, 8, dtype, 1), _arrays_cache(dtype, 2), _kv(5, 2, 8, dtype, 3)]
        tree = LingTree(parents, cache)
        mx.random.seed(9)
        for item in tree.private:
            if type(item) is KVCache:
                item.update_and_fetch(mx.random.normal((1, 2, width, 8)).astype(dtype),
                                      mx.random.normal((1, 2, width, 8)).astype(dtype))
            else:
                planes = [mx.random.normal((width, 3, 8)).astype(dtype) for _ in range(3)]
                states = [mx.random.normal((1, 2, 4, 4)) for _ in range(width)]
                tree.recurrent[id(item)] = (*planes, states)
        return tree

    topology = TV.TreeTopology(parents, TV.legacy_topology("ling_packed"))
    for path in topology.path_lists():
        legacy_tree, new_tree = build(), build()
        legacy = _legacy_ling_prepare_commit(legacy_tree, path)
        new = new_tree.prepare_commit(path)
        assert len(legacy) == len(new)
        for (_, left), (_, right) in zip(legacy, new, strict=True):
            _assert_same_fields(left, right)


class _FakeMixer:
    conv_kernel_size = 4

    def tree_commit_state(self, material, path, indices):
        state = material["state"]
        for row in path:
            state = state * 0.5 + material["x"][:, row:row + 1].sum()
        return state + mx.take(material["dt"], indices, axis=1).sum()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("parents", TREES)
def test_nemotron_commit_is_bit_identical(dtype, parents):
    from mlx2.runtime.models.nemotron_tree import NemotronTree

    width = len(parents)

    def build():
        mamba = ArraysCache(2)
        mx.random.seed(4)
        mamba[0] = mx.random.normal((1, 3, 6)).astype(dtype)
        mamba[1] = mx.random.normal((1, 2, 4, 4))
        cache = [mamba, _kv(6, 2, 8, dtype, 5)]
        tree = NemotronTree(parents, cache)
        mx.random.seed(10)
        keys = mx.random.normal((1, 2, width, 8)).astype(dtype)
        values = mx.random.normal((1, 2, width, 8)).astype(dtype)
        tree.kv_append(cache[1], keys, values)
        source = mx.concatenate([mamba[0], mx.random.normal((1, width, 6)).astype(dtype)], axis=1)
        tree.record_mamba(cache[0], mixer=_FakeMixer(), source=source,
                          x=mx.random.normal((1, width, 6)), B=None, C=None,
                          dt=mx.random.normal((1, width, 2)), gate=None, state=mamba[1])
        return tree

    for path in TV.TreeTopology(parents, TV.legacy_topology("nemotron_packed")).path_lists():
        legacy_tree, new_tree = build(), build()
        legacy = _legacy_nemotron_prepare_commit(legacy_tree, path)
        new = new_tree.prepare_commit(path)
        for (_, left), (_, right) in zip(legacy, new, strict=True):
            _assert_same_fields(left, right)


@pytest.mark.parametrize("buffer_dtype,raw_dtype", [
    (mx.bfloat16, mx.bfloat16), (mx.float16, mx.float16), (mx.float32, mx.float32),
    (mx.bfloat16, mx.float32), (mx.float16, mx.float32),
])
@pytest.mark.parametrize("parents", TREES)
def test_inkling_global_kv_and_shortconv_commit_are_bit_identical(buffer_dtype, raw_dtype, parents):
    from mlx2.runtime.models.inkling.tree import _KV, _SHORTCONV

    width = len(parents)
    topology = TV.TreeTopology(parents, TV.legacy_topology("inkling_packed"))

    def build():
        kv = _kv(7, 2, 8, buffer_dtype, 6)
        live_offset = int(kv.offset)
        mx.random.seed(12)
        raw = (mx.random.normal((1, 2, width, 8)).astype(raw_dtype),
               mx.random.normal((1, 2, width, 8)).astype(raw_dtype))
        kv.update_and_fetch(raw[0], raw[1])
        conv = ArraysCache(4)
        history = [[mx.random.normal((1, 3, 5)) for _ in range(width)] for _ in range(4)]
        return kv, live_offset, raw, conv, history

    for path in topology.path_lists():
        (kv_a, offset, raw_a, conv_a, history_a), (kv_b, _, raw_b, conv_b, history_b) = build(), build()
        _legacy_inkling_global_kv(kv_a, offset, topology.path_lists(), raw_a, path)
        _KV.prepare_commit(kv_b, start=offset, path=path, width=width, raw=raw_b)
        _assert_same_fields(_arrays(kv_a.__dict__), _arrays(kv_b.__dict__))
        _legacy_inkling_conv(conv_a, history_a, path)
        _SHORTCONV.prepare_commit(conv_b, history_b, path)
        _assert_same_fields(_arrays(conv_a.__dict__), _arrays(conv_b.__dict__))


# ---------------------------------------------------------------------------
# Serial reference parity (legacy hybrid tree_forward + transaction)
# ---------------------------------------------------------------------------


def _legacy_hybrid_tree_forward(model, tokens, parents, live, layers):
    # runtime/hybrid_tree_executor.py tree_forward (+ HybridTreeTransaction state)
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors

    if not 1 <= len(tokens) <= 16 or len(tokens) != len(parents):
        raise ValueError("hybrid AST tile must contain 1..16 matching rows")
    branches, logits, features = [], [], []
    for row, (value, parent) in enumerate(zip(tokens, parents, strict=True)):
        if (
            type(parent) is not int
            or (row == 0 and parent != -1)
            or (row > 0 and not 0 <= parent < row)
        ):
            raise ValueError(
                "hybrid AST parents must form one parent-first rooted tree"
            )
        source = live if parent < 0 else branches[parent]
        private = snapshot_prompt_cache_descriptors(source, require_committed=False)[0]
        row_logits, row_features = model.forward_with_taps(
            mx.array([[int(value)]], dtype=mx.uint32), private, layers
        )
        branches.append(private)
        logits.append(row_logits)
        features.append(row_features)
    return mx.concatenate(logits, axis=1), mx.concatenate(features, axis=1), branches


def _legacy_hybrid_commit(live, branches, paths):
    # runtime/hybrid_tree_executor.py HybridTreeTransaction.commit_paths
    from mlx2.runtime.cow_cache import snapshot_prompt_cache_descriptors

    terminal = int(paths[0][-1])
    selected = snapshot_prompt_cache_descriptors(branches[terminal], require_committed=False)[0]
    for destination, source in zip(live, selected, strict=True):
        if type(destination) is not type(source):
            raise TypeError("hybrid tree cache type changed")
        destination.__dict__.clear()
        destination.__dict__.update(source.__dict__)
    return [live]


class _TinyModel:
    """One KV layer and one recurrent layer; ordinary one-token decode."""

    def __init__(self, dtype):
        mx.random.seed(21)
        self.embed = mx.random.normal((32, 8)).astype(dtype)
        self.mix = mx.random.normal((8, 8)).astype(dtype)

    def make_cache(self):
        return [KVCache(), ArraysCache(1)]

    def forward_with_taps(self, tokens, cache, layers):
        x = self.embed[tokens.reshape(-1)].reshape(1, -1, 8)
        keys, values = cache[0].update_and_fetch(
            x.reshape(1, 1, -1, 8), (x @ self.mix).reshape(1, 1, -1, 8))
        attended = mx.softmax((x.reshape(1, 1, -1, 8) @ keys.swapaxes(-1, -2)).astype(mx.float32),
                              axis=-1).astype(x.dtype) @ values
        state = cache[1][0]
        state = x[:, -1:] if state is None else state * 0.75 + x[:, -1:]
        cache[1][0] = state
        hidden = attended.reshape(1, -1, 8) + state
        return hidden @ self.mix.T, hidden


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("parents", TREES)
def test_hybrid_serial_reference_is_bit_identical(dtype, parents):
    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor

    model = _TinyModel(dtype)
    tokens = [(5 * row + 3) % 32 for row in range(len(parents))]

    def prefilled():
        cache = model.make_cache()
        model.forward_with_taps(mx.array([[1, 2, 3]], dtype=mx.uint32), cache, ())
        mx.eval([item.state for item in cache])
        return cache

    executor = HybridTreeExecutor()
    for path in TV.TreeTopology(parents, TV.legacy_topology("hybrid_serial")).path_lists():
        legacy_live, new_live = prefilled(), prefilled()
        legacy_logits, legacy_features, branches = _legacy_hybrid_tree_forward(
            model, tokens, parents, legacy_live, ())
        _legacy_hybrid_commit(legacy_live, branches, [path])
        logits, features, transaction = executor.forward(
            model, tokens, parents, new_live, (), cached=True)
        assert mx.array_equal(logits, legacy_logits).item()
        assert mx.array_equal(features, legacy_features).item()
        assert transaction.commit_paths([path]) == [new_live] and transaction.closed
        for left, right in zip(legacy_live, new_live, strict=True):
            _assert_same_fields(left.__dict__, right.__dict__)


def test_serial_tree_refuses_siblings_before_publishing_and_allows_retry():
    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor

    model = _TinyModel(mx.float32)
    cache = model.make_cache()
    executor = HybridTreeExecutor()
    _, _, transaction = executor.forward(model, [3, 4, 5], [-1, 0, 0], cache, (), cached=True)
    with pytest.raises(ValueError, match="root path"):
        transaction.commit_paths([[0, 1, 2]])
    assert not transaction.closed and cache[0].offset == 0
    transaction.commit_paths([[0, 2]])
    assert cache[0].offset == 2
    with pytest.raises(ValueError, match="independent cache descriptors"):
        executor.forward_many(model, [[3], [4]], [[-1], [-1]], [cache, list(cache)], (),
                              cached=True, configured_limit=2)


def test_hybrid_cohort_cannot_republish_an_aborted_lane():
    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor

    model = _TinyModel(mx.float32)
    caches = [model.make_cache(), model.make_cache()]
    _, _, transaction = HybridTreeExecutor().forward_many(
        model, [[3], [4]], [[-1], [-1]], caches, (), cached=True, configured_limit=2)
    transaction.transactions[1].abort()
    with pytest.raises(RuntimeError, match="closed transaction"):
        transaction.commit_paths([[0], [0]])
    assert transaction.closed and all(cache[0].offset == 0 for cache in caches)


def test_hybrid_cohort_prepares_every_lane_before_publishing():
    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor

    model = _TinyModel(mx.float32)
    caches = [model.make_cache(), model.make_cache()]
    for cache in caches:
        model.forward_with_taps(mx.array([[1, 2]], dtype=mx.uint32), cache, ())
    before = [[dict(item.__dict__) for item in cache] for cache in caches]
    executor = HybridTreeExecutor()
    _, _, transaction = executor.forward_many(
        model, [[3, 4], [5, 6]], [[-1, 0], [-1, 0]], caches, (), cached=True, configured_limit=2)
    with pytest.raises(ValueError, match="outside the tile"):
        transaction.commit_paths([[0, 1], [0, 7]])
    assert transaction.closed and all(item.closed for item in transaction.transactions)
    # Lane 0 was valid but nothing was published: the cohort is atomic.
    for cache, fields in zip(caches, before, strict=True):
        assert [item.offset if hasattr(item, "offset") else None for item in cache] == [
            entry.get("offset") for entry in fields]


# ---------------------------------------------------------------------------
# Revision binder
# ---------------------------------------------------------------------------


class _Bound(TV.SourceBoundRevision):
    _identity_style = "name"
    _revision_error = "probe source changed"


def test_revision_binder_rehashes_only_on_metadata_moves(tmp_path, monkeypatch):
    source = tmp_path / "a.py"
    source.write_bytes(b"one")
    bound = _Bound()
    bound._bind_sources((source,))
    reads = []
    original = TV.file_sha256
    monkeypatch.setattr(TV, "file_sha256", lambda path: reads.append(path) or original(path))
    bound.validate_revision()
    assert not reads
    saved = source.stat()
    source.write_bytes(b"two")
    os.utime(source, ns=(saved.st_atime_ns, saved.st_mtime_ns))  # same size, same mtime
    with pytest.raises(RuntimeError, match="probe source changed"):
        bound.validate_revision()
    assert reads


@pytest.mark.parametrize("cls_path,style", [
    ("mlx2.runtime.hybrid_tree_executor:HybridTreeExecutor", "module"),
    ("mlx2.runtime.attention_tree_executor:AttentionTreeExecutor", "module"),
    ("mlx2.runtime.ling_tree_executor:LingTreeExecutor", "name"),
    ("mlx2.runtime.kimi_tree_executor:KimiTreeExecutor", "name"),
    ("mlx2.runtime.nemotron_tree_executor:NemotronTreeExecutor", "relative"),
    ("mlx2.runtime.inkling_tree_executor:InklingTreeExecutor", "relative"),
])
def test_executors_bind_the_shared_source(cls_path, style):
    import importlib

    module, name = cls_path.split(":")
    executor = getattr(importlib.import_module(module), name)()
    assert executor._identity_style == style
    assert TV.TREE_VERIFY_SOURCE in executor._sources
    executor.validate_revision()
    identity = dict(executor.source_identity)
    assert len(identity) == len(executor.source_identity)
    if style == "module":
        assert identity["module"] == module and "tree_verify.py" in identity
    assert executor.receipt()["source_revision_token"] == executor.source_revision_token


def test_attention_executor_now_refuses_changed_source(tmp_path):
    from mlx2.runtime.attention_tree_executor import AttentionTreeExecutor

    executor = AttentionTreeExecutor()
    copy_path = tmp_path / "attention_tree_executor.py"
    copy_path.write_bytes(executor._sources[0].read_bytes())
    executor._sources = (copy_path, *executor._sources[1:])
    executor._source_stats = ()
    executor.validate_revision()          # same bytes, new metadata: re-armed
    copy_path.write_bytes(copy_path.read_bytes() + b"\n# edit\n")
    with pytest.raises(RuntimeError, match="source changed"):
        executor.forward(None, [1], [-1], [], (), cached=True)
    with pytest.raises(RuntimeError, match="source changed"):
        executor.forward_many(None, [[1]], [[-1]], [[]], (), cached=True, configured_limit=1)
    assert executor.dispatches == 0


# ---------------------------------------------------------------------------
# Serial / packed selection
# ---------------------------------------------------------------------------


def _legacy_choice_packed_default(policy):
    # kimi_linear / ling32_air / ling3_flash / nemotron35_super
    return "serial" if policy.get("ast_target_executor", "packed") == "serial" else "packed"


def _legacy_choice_inkling(policy):
    # adapters/inkling.py create_external_batch
    return "packed" if policy.get("inkling_tree_executor") == "packed" else "serial"


@pytest.mark.parametrize("value", [None, "serial", "packed", "other", "", 0, False])
def test_choice_matches_each_adapter(value):
    policies = [{}] if value is None else [{"ast_target_executor": value, "inkling_tree_executor": value}]
    for policy in policies:
        assert TV.tree_executor_choice(policy) == _legacy_choice_packed_default(policy)
        assert TV.tree_executor_choice(
            policy, key="inkling_tree_executor", default=None, explicit="packed"
        ) == _legacy_choice_inkling(policy)


def test_build_and_serial_row_cap():
    made = []
    assert TV.build_tree_executor("serial", serial=lambda: made.append("s") or "S",
                                  packed=lambda: made.append("p") or "P") == "S"
    assert TV.build_tree_executor("packed", serial=lambda: "S", packed=lambda: "P") == "P"
    assert made == ["s"]
    TV.check_serial_rows("packed", 32, serial_max_rows=16, error="cap")
    TV.check_serial_rows("serial", 16, serial_max_rows=16, error="cap")
    with pytest.raises(ValueError, match="cap"):
        TV.check_serial_rows("serial", 17, serial_max_rows=16, error="cap")


def test_executor_choice_is_counted_by_the_shared_dispatcher():
    from mlx2.runtime.technologies import dispatch

    dispatch.reset_counters()
    TV.build_tree_executor("serial", serial=lambda: "S", packed=lambda: "P")
    TV.build_tree_executor("packed", serial=lambda: "S", packed=lambda: "P")
    rows = {(row["operation"], row["path"]): row["count"]
            for row in dispatch.counters_snapshot()["selections"]
            if row["technology"] == "tree_verify"}
    assert rows == {("executor", "reference"): 1, ("executor", "packed"): 1}


def test_spec_is_frozen_and_from_config():
    spec = TV.TopologySpec.from_config({"max_tree_rows": 8})
    assert spec.max_rows == 8 and TV.TopologySpec.from_config(None).max_rows == 16
    with pytest.raises(Exception):
        spec.max_rows = 4
    with pytest.raises(ValueError):
        TV.TopologySpec(root_rule="forest")
    assert isinstance(TV.KVTreeOps(), TV.TreeMixerOps)
    assert isinstance(TV.MambaReplayOps(), TV.TreeMixerOps)
    assert isinstance(TV.SnapshotBranchOps(), TV.TreeMixerOps)
    assert isinstance(TV.RecurrentTerminalOps(TV.kda_terminal_planes), TV.TreeMixerOps)


def test_legacy_topology_caps_are_preserved():
    caps = {name: spec.max_rows for name, spec in TV.LEGACY_TOPOLOGIES.items()}
    assert caps["hybrid_serial"] == caps["nemotron_packed"] == caps["inkling_packed"] == 16
    assert caps["flash_tree_ancestry"] == caps["qwen38_rooted_tree"] == 16
    assert caps["ling_packed"] == caps["attention_tile"] == caps["glm_rooted_tree"] == 32
    assert caps["packed_attention_tile"] == caps["q8_attention_tile"] == 32
    assert all(TV.LEGACY_TOPOLOGIES[name].root_rule == TV.ROOT_ANY_NEGATIVE for name in (
        "attention_depths", "qwen_tree_gdn_paths", "dflash_tree_paths"))


def test_every_technology_module_is_bound_and_none_may_be_absent():
    from pathlib import Path

    on_disk = {p for p in Path(TV.__file__).parent.rglob("*.py") if "__pycache__" not in p.parts}
    assert set(TV.TECHNOLOGY_SOURCES) == on_disk
    assert len(TV.TECHNOLOGY_SOURCES) == len(set(TV.TECHNOLOGY_SOURCES))
    with pytest.raises(FileNotFoundError):
        TV.file_sha256(Path(TV.__file__).parent / "not_a_technology.py")
    with pytest.raises(FileNotFoundError):
        TV.source_stat(Path("/nonexistent/linear_recurrence.py"))
    assert set(TV.TECHNOLOGY_SOURCE_PATHS) >= {
        "src/mlx2/runtime/technologies/tree_verify.py",
        "src/mlx2/runtime/technologies/dispatch.py",
        "src/mlx2/runtime/technologies/linear_recurrence.py",
        "src/mlx2/runtime/technologies/mla.py",
        "src/mlx2/runtime/technologies/window_attention/__init__.py",
        "src/mlx2/runtime/technologies/window_attention/vector_sources.py",
        "src/mlx2/runtime/technologies/window_attention/vector_launch.py",
    }
