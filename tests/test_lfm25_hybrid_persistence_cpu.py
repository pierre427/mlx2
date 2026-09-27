"""Exercise the production recurrent cache with host arrays only.

This checks the LFM ShortConv state half of an APCv2 hybrid checkpoint.  The
real safetensors writer and mixed KV layout still need a device qualification.
"""

import ast
import copy
import importlib.abc
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise AssertionError(f"real MLX import forbidden: {fullname}")
        return None


_mlx_modules_before = {name for name in sys.modules
                       if name == "mlx" or name.startswith("mlx.")}
sys.meta_path.insert(0, BlockMLX())


def _arrays_cache():
    path = Path(__file__).parents[1] / "src/mlx2/runtime/models/cache.py"
    tree = ast.parse(path.read_text())
    names = {"ArraysCache", "_thin_checkpoints"}
    nodes = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *(node for node in tree.body
          if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names),
    ]

    class BaseCache:
        @classmethod
        def from_state(cls, state, meta_state):
            obj = cls.__new__(cls)
            obj.state = state
            obj.meta_state = meta_state
            return obj

    mx = SimpleNamespace(array=np.array, async_eval=lambda *_: None,
                         contiguous=np.ascontiguousarray)
    namespace = {
        "_BaseCache": BaseCache, "deque": deque, "mx": mx,
        "_state_checkpoint_max": lambda: 5,
        "_state_checkpoint_stride": lambda: 1,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(path), "exec"), namespace)  # noqa: S102 - production AST only
    return namespace["ArraysCache"]


def test_lfm_shortconv_checkpoint_roundtrip_and_long_branch_replay():
    ArraysCache = _arrays_cache()
    source = ArraysCache(1)
    # Many interior checkpoints force the production thinning policy; the
    # exact prompt boundary must remain available after serialization.
    for position in range(256, 2305, 256):
        source[0] = np.full((1, 3, 4), position, dtype=np.float32)
        source.state_checkpoint([position], force=True)
    retained = [p for p, _ in source._checkpoints[0]]
    assert len(retained) == 5 and retained[-1] == 2304

    state, meta = copy.deepcopy(source.state), copy.deepcopy(source.meta_state)
    restored = ArraysCache.from_state(state, meta)
    assert [p for p, _ in restored._checkpoints[0]] == retained
    assert restored.snap_trim_position(2000) == max(p for p in retained if p <= 2000)
    np.testing.assert_array_equal(restored[0], source[0])

    # APCv2 can restore the same persisted prefix into separate branches.
    first, second = copy.deepcopy(restored), copy.deepcopy(restored)
    landing = first.snap_trim_position(2000)
    first.trim_to_position(landing, 2304 - landing)
    second.trim_to_position(landing, 2304 - landing)
    np.testing.assert_array_equal(first[0], np.full((1, 3, 4), landing))
    first[0] = first[0] + 11
    second[0] = second[0] + 22
    np.testing.assert_array_equal(restored[0], np.full((1, 3, 4), 2304))
    np.testing.assert_array_equal(first[0], np.full((1, 3, 4), landing + 11))
    np.testing.assert_array_equal(second[0], np.full((1, 3, 4), landing + 22))
    assert first._checkpoints[0][-1][0] == landing
    assert second._checkpoints[0][-1][0] == landing
    assert {name for name in sys.modules
            if name == "mlx" or name.startswith("mlx.")} == _mlx_modules_before
