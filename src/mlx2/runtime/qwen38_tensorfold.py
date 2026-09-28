"""Default-off TensorFold Qwen3.8 target executor for lever measurement.

This module imports kernels from an explicitly pinned local TensorFold source
checkout. It is an experimental probe, not a qualified serving dependency.
See ``provenance/tensorfold-qwen38-lever-probe.json``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

EXPECTED_REVISION = "71377a5373ed7b394f1b480ba2a6a3986b03af1c"

# Every kernel module the tree forward and commit import, directly or lazily.
# The cached executor imports them all at validation so a later lazy import
# cannot read a checkout that changed after its revision was checked.
_KERNEL_MODULES = (
    "tensorfold.kernels.qwen.dense.v1.lane_tree",
    "tensorfold.kernels.qwen.dense.v1.lane_multi",
    "tensorfold.kernels.qwen.dense.v1.lane_fuse",
    "tensorfold.kernels.qwen.dense.v1.lane_glue",
    "tensorfold.kernels.qwen.dense.v1.stream_attention",
    "tensorfold.kernels.qwen.dense.v1.stream_gdn",
)

# Resolved source root -> validated lane_tree module (MLX2_TENSORFOLD_CACHE_EXECUTOR).
_EXECUTORS = {}


def _validate(root):
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"TensorFold source revision mismatch: {revision}, expected {EXPECTED_REVISION}"
        )


def _import(root, names):
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    modules = [importlib.import_module(name) for name in names]
    for module in modules:
        # sys.modules is keyed by name, not checkout: a module already
        # imported from another root must not pass as this root's.
        if not Path(module.__file__).resolve().is_relative_to(Path(source).resolve()):
            raise RuntimeError(
                f"TensorFold module {module.__name__} was imported from "
                f"{module.__file__}, not the validated source {source}"
            )
    return modules[0]


def _modules(source_root, *, cached=False):
    """Return ``(lane_tree, cache_hit)`` for a revision-checked source root.

    Uncached (the default) re-runs ``git rev-parse`` for every forward.
    Cached validates each resolved root once per process and then reuses the
    imported module: code already imported cannot change with the checkout,
    so a per-call revision check adds a subprocess without adding safety.
    """

    root = Path(source_root).resolve()
    if cached:
        module = _EXECUTORS.get(root)
        if module is not None:
            return module, True
    _validate(root)
    module = _import(root, _KERNEL_MODULES if cached else _KERNEL_MODULES[:1])
    if cached:
        _EXECUTORS[root] = module
    return module, False


def _offset(cache):
    offsets = [int(item.offset) for item in cache if hasattr(item, "offset")]
    if not offsets or len(set(offsets)) != 1:
        raise ValueError("TensorFold target execution requires one exact cache boundary")
    return offsets[0]


class TensorfoldTransaction:
    def __init__(self, lane_tree, cache, record, width, start):
        self.lane_tree = lane_tree
        self.cache = cache
        self.record = record
        self.width = int(width)
        self.start = int(start)
        self.closed = False
        self.executor_cached = False

    def commit_paths(self, paths):
        if self.closed or len(paths) != 1:
            raise RuntimeError("TensorFold probe transaction is single-lane")
        self.lane_tree.commit_tree(
            self.cache, self.record, paths[0], self.width, self.start
        )
        self.closed = True
        return [self.cache]

    def commit(self, *, accepted_lengths):
        """Adapt the chain verifier's prefix-length commit contract."""

        if len(accepted_lengths) != 1:
            raise RuntimeError("TensorFold probe transaction is single-lane")
        length = int(accepted_lengths[0])
        return self.commit_paths([list(range(length))])

    def abort(self):
        if not self.closed:
            self.lane_tree.commit_tree(
                self.cache, self.record, [], self.width, self.start
            )
            self.closed = True


def _capture_storage(model, capture_layers, cached):
    """Install the capture slots on the draft tap layers; return the storage.

    Cached mode installs one storage list per model and capture-layer tuple
    and clears its slots each forward; the slots are re-installed if another
    owner replaced them.
    """

    layers = list(model.model.layers)
    if cached:
        installed = getattr(model, "_mlx2_tensorfold_capture", None)
        if installed is not None and installed[0] == capture_layers:
            storage = installed[1]
            if all(
                getattr(layers[index], "_storage", None) is storage
                for index in capture_layers
            ):
                for index in capture_layers:
                    storage[index] = None
                return storage
    storage = [None] * len(layers)
    for index in capture_layers:
        layer = layers[index]
        object.__setattr__(layer, "_storage", storage)
        object.__setattr__(layer, "_idx", index)
    if cached:
        object.__setattr__(model, "_mlx2_tensorfold_capture", (capture_layers, storage))
    return storage


def forward(model, tokens, parents, cache, capture_layers, source_root, *, cached=False):
    """Return logits, target taps, and an exact arbitrary-path transaction."""

    lane_tree, hit = _modules(source_root, cached=cached)
    capture_layers = tuple(int(index) for index in capture_layers)
    storage = _capture_storage(model, capture_layers, cached)
    start = _offset(cache)
    logits, record = lane_tree.tree_forward(
        model.model,
        model.logits,
        [int(token) for token in tokens],
        [int(parent) for parent in parents],
        cache,
        start,
    )
    taps = [storage[index] for index in capture_layers]
    if any(value is None for value in taps):
        raise RuntimeError("TensorFold target executor did not capture every draft tap")
    features = mx.concatenate(taps, axis=-1)
    transaction = TensorfoldTransaction(lane_tree, cache, record, len(tokens), start)
    transaction.executor_cached = hit
    return logits, features, transaction


__all__ = ["EXPECTED_REVISION", "TensorfoldTransaction", "forward"]
