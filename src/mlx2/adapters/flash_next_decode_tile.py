# SPDX-License-Identifier: Apache-2.0
"""Experimental HC/GDN/MoE decode block over one TensorFold row tile.

Rows are independent decode lanes, each with a private incoming checkpoint.
The tile reuses the existing Flash tree projection/HC/MoE laws. Its GDN
consumer receives [lanes,1,H], never a fabricated [1,lanes,H] time axis.
This is a shared tile traversal with several dependent Metal dispatches.
No APCv2 publication or default route is selected by this component.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
from mlx.utils import tree_flatten

from ..runtime.cow_cache import snapshot_prompt_cache_descriptors


def _binding(layer):
    return tuple((name, id(value)) for name, value in tree_flatten(layer.parameters()))


def _cache_binding(caches):
    return tuple((id(cache), tuple(id(value) for value in cache.cache)) for cache in caches)


@dataclass
class DecodeTileTransaction:
    layer: object
    revision: str
    layer_binding: tuple
    owners: tuple
    owner_binding: tuple
    successors: tuple
    metrics: dict
    closed: bool = False

    def commit(self, rows, *, revision):
        if self.closed:
            raise ValueError("decode tile transaction is closed")
        rows = tuple(rows)
        if revision != self.revision or _binding(self.layer) != self.layer_binding:
            raise ValueError("decode tile source/weight revision changed")
        if _cache_binding(self.owners) != self.owner_binding:
            raise ValueError("decode tile incoming checkpoint changed")
        if len(set(rows)) != len(rows) or any(type(i) is not int or not 0 <= i < len(self.successors) for i in rows):
            raise ValueError("decode tile commit rows must be unique valid lanes")
        self.closed = True
        # Return independent successor descriptors, never mutate live owners.
        return tuple(snapshot_prompt_cache_descriptors([self.successors[i]], require_committed=False)[0][0] for i in rows)

    def abort(self):
        self.closed = True


def prepare_gdn_decode_tile(layer, hidden, caches, token_ids, *, revision, cpu_reference=False):
    """Prepare one 1..16-row GDN decoder block with private successors.

    GPU callers cold-bind flash_next_tree.install_projection_law(model)
    before constructing the prefix/reference state. Calling this function
    does not change an existing model's arithmetic or reinterpret its cache.
    """
    from ..runtime.models.cache import ArraysCache
    from ..runtime.models.qwen4_exp import _apply_inject
    from .flash_next_tree import _hc, _moe
    if not isinstance(revision, str) or not revision:
        raise ValueError("decode tile requires a source/artifact revision")
    caches = tuple(caches)
    n = len(caches)
    if (not 1 <= n <= 16 or len({id(c) for c in caches}) != n or
            hidden.ndim != 3 or hidden.shape[:2] != (n, 1) or
            token_ids.shape != (n, 1) or token_ids.dtype not in (mx.int32, mx.int64)):
        raise ValueError("decode tile requires distinct lanes and one token per lane")
    if not layer.is_linear or layer.linear_attn is None or layer.training:
        raise ValueError("decode tile requires an inference GDN block")
    if not all(isinstance(c, ArraysCache) and not c.speculating for c in caches):
        raise ValueError("decode tile requires ordinary private recurrent checkpoints")
    if not cpu_reference and (mx.default_device() != mx.gpu or hidden.dtype != mx.bfloat16):
        raise ValueError("decode tile requires BF16 Metal; CPU oracle is explicit")
    binding, owners = _binding(layer), _cache_binding(caches)
    private = [snapshot_prompt_cache_descriptors([c], require_committed=False)[0][0] for c in caches]
    joined = type(private[0]).merge(private)
    if layer.ple is not None:
        hidden = hidden + layer.ple(hidden, token_ids, joined, None)
    mixed, residual, inject = _hc(layer.attn_hyper_connection, hidden, cpu_reference=cpu_reference)
    branch = layer.linear_attn(mixed, None, joined)
    hidden = _apply_inject(residual, branch, inject)
    mixed, residual, inject = _hc(layer.mlp_hyper_connection, hidden, cpu_reference=cpu_reference)
    branch = _moe(layer.mlp, mixed, cpu_reference=cpu_reference)
    hidden = _apply_inject(residual, branch, inject)
    successors = tuple(joined.extract(i) for i in range(n))
    metrics = {"rows": n, "tile_capacity": 16, "padding_rows_executed": 0,
               "hc_stages": 2, "gdn_stages": 1, "moe_stages": 1,
               "ordinary_model_replays": 0, "single_metal_dispatch": False,
               "qualified": False, "state_layout": "independent_Bx1_decode"}
    return hidden, DecodeTileTransaction(layer, revision, binding, caches, owners, successors, metrics)
