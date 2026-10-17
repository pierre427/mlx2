"""Temporary routed-input calibration wrappers for a dedicated model process.

Never install these in a serving process. They synchronize statistics and
would distort latency measurements. The original SwitchMLP implementation
still owns sorting, padded gathers, activation math and output layout.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_unflatten

from ..runtime.models.switch_layers import SwitchMLP


def reduced_routed_moments(inputs, indices, *, experts, real_assignments, sorted_indices):
    if (inputs.ndim < 2 or inputs.shape[-2] != 1
            or type(real_assignments) is not int or not 0 < real_assignments <= indices.size):
        raise ValueError("Routed calibration input geometry differs")
    values = mx.broadcast_to(inputs.squeeze(-2), (*indices.shape, inputs.shape[-1])).reshape(-1, inputs.shape[-1])
    ids = indices.reshape(-1)
    if sorted_indices:
        # _gather_sort can append repeated tail rows for a native-kernel
        # alignment workaround. They are compute padding, not observations.
        values, ids = values[:real_assignments], ids[:real_assignments]
    elif real_assignments != indices.size:
        raise ValueError("Unsorted calibration cannot discard routed assignments")
    host_ids = np.asarray(ids)
    if (not np.issubdtype(host_ids.dtype, np.integer)
            or np.any(host_ids < 0) or np.any(host_ids >= experts)):
        raise ValueError("Routed calibration expert id outside schema")
    values = values.astype(mx.float32)
    sums = mx.zeros((experts, inputs.shape[-1]), mx.float32).at[ids].add(values * values)
    counts = mx.zeros((experts,), mx.uint32).at[ids].add(mx.ones((ids.size,), mx.uint32))
    return sums, counts


class CapturedRoutedLinear(nn.Module):
    def __init__(self, inner, collector, name, assignment_count):
        super().__init__()
        self.inner = inner
        self._collector, self._name, self._assignment_count = collector, name, assignment_count

    def __call__(self, x, indices, sorted_indices=False):
        shape = self._collector.specs[self._name]
        if x.shape[-1] != shape[1]:
            raise ValueError("Captured projection input width differs from statistics")
        sums, counts = reduced_routed_moments(x, indices, experts=shape[0],
            real_assignments=self._assignment_count(), sorted_indices=sorted_indices)
        mx.eval(sums, counts)
        self._collector.add_reduced(self._name, np.asarray(sums), np.asarray(counts))
        return self.inner(x, indices, sorted_indices=sorted_indices)


class CapturedSwitchMLP(SwitchMLP):
    def __init__(self, inner, collector, prefix):
        nn.Module.__init__(self)
        # A callback closing over self makes a cycle through fc1/fc2 and
        # retains the sample's large moments until cyclic GC happens to run.
        # The shared scalar cell has no reference back to this module.
        assignments = [0]
        self._real_assignments = assignments
        count = lambda: assignments[0]
        self.fc1 = CapturedRoutedLinear(inner.fc1, collector, prefix + ".fc1", count)
        self.fc2 = CapturedRoutedLinear(inner.fc2, collector, prefix + ".fc2", count)
        self.activation = inner.activation
        self._sort_policy = inner.sort_status()["policy"]
        self._sort_counts = {
            "auto_sorted": 0,
            "auto_unsorted": 0,
            "invariant_sorted": 0,
            "unsorted_decode": 0,
        }
        self.train(inner.training)

    def __call__(self, x, indices):
        self._real_assignments[0] = indices.size
        return super().__call__(x, indices)


class RoutedCapture:
    """Install/restore only complete planned expert blocks, never global patches."""
    def __init__(self, model, collector):
        self.model, self.collector, self.originals = model, collector, None

    def __enter__(self):
        if self.originals is not None:
            raise ValueError("Routed calibration wrapper is already installed")
        modules = list(self.model.named_modules())
        if any(isinstance(module, CapturedSwitchMLP) for _, module in modules):
            raise ValueError("A routed calibration wrapper is already installed")
        originals, replacements, observed = [], [], set()
        for path, module in modules:
            if not isinstance(module, SwitchMLP):
                continue
            prefix = "language_model." + path
            names = {prefix + ".fc1", prefix + ".fc2"}
            if not names & self.collector.specs.keys():
                continue
            if not names <= self.collector.specs.keys():
                raise ValueError("Calibration must cover both expert projections")
            for name, child in ((prefix + ".fc1", module.fc1), (prefix + ".fc2", module.fc2)):
                if self.collector.specs[name] != (child.num_experts, child.input_dims):
                    raise ValueError("Calibration schema differs from the loaded expert bank")
            originals.append((path, module))
            replacements.append((path, CapturedSwitchMLP(module, self.collector, prefix)))
            observed.update(names)
        if observed != self.collector.specs.keys():
            raise ValueError("Calibration module schema has absent expert banks")
        self.model.update_modules(tree_unflatten(replacements))
        self.originals = originals
        return self

    def __exit__(self, *_):
        if self.originals is not None:
            self.model.update_modules(tree_unflatten(self.originals))
            self.originals = None
