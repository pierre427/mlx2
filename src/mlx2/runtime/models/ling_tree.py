"""Ling's parent-indexed tree math; ordinary model modules own projections.

Projection/MLP work runs once per layer across all nodes. KDA recurrence runs
independent siblings in depth cohorts using the ordinary one-step kernel. MLA
keeps the ordinary absorbed one-query law and exact ancestor key lengths.
This is not a closed-form parallel recurrence or cross-request packed kernel.
Provenance: provenance/ling3-packed-tree.json.
"""

from __future__ import annotations

import copy

import mlx.core as mx

from ..technologies.tree_verify import (
    KVTreeOps,
    RecurrentTerminalOps,
    TreeTopology,
    kda_terminal_planes,
    legacy_topology,
    publish_descriptors,
)
from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update
from ..technologies.linear_recurrence import cohort_tree_recurrence, normalize_qk

TOPOLOGY = legacy_topology("ling_packed")
_KV = KVTreeOps()  # always gather from the private buffer's tile window
_KDA = RecurrentTerminalOps(kda_terminal_planes)


class LingTree:
    def __init__(self, parents, cache):
        self.topology = TreeTopology(parents, TOPOLOGY)
        self.parents = self.topology.parents
        self.paths = self.topology.path_lists()
        self.depths = list(self.topology.depths)
        self.levels = self.topology.levels
        if any(type(item) not in (ArraysCache, KVCache) for item in cache):
            raise ValueError("Ling packed trees require plain KV/Arrays caches")
        for item in cache:
            if type(item) is ArraysCache and (
                item.speculating or item.lengths is not None or item.left_padding is not None
            ):
                raise ValueError("Ling packed tree requires an unpadded committed cache")
        self.live = cache
        # One private descriptor set per request, never one full model cache
        # per node. MLX arrays preserve COW isolation on subsequent updates.
        self.private = copy.deepcopy(cache)
        self.recurrent = {}

    def rope(self, rope, values, offset):
        rows_first = values.transpose(2, 1, 0, 3)
        positions = mx.array(self.depths, dtype=mx.int32) + int(offset)
        return rope(rows_first, offset=positions).transpose(2, 1, 0, 3)

    def convolution(self, module, raw, state):
        windows = self.convolution_windows(module.kernel_size, raw, state)
        return module(windows[:, -1:], windows[:, :-1], None, None)

    def convolution_windows(self, kernel_size, raw, state):
        count = kernel_size - 1
        if state is None:
            state = mx.zeros((1, count, raw.shape[-1]), dtype=raw.dtype)
        # Each window is the last K-1 inputs on this node's true ancestry,
        # followed by this node's input, never an adjacent sibling's input.
        indices = self.convolution_indices(kernel_size)
        source = mx.concatenate([state, raw], axis=1)
        return mx.take(source, indices, axis=1)[0]

    def convolution_indices(self, kernel_size):
        """``[rows, K]`` ancestry window indices into ``[K-1 state; tile]``."""
        return self.topology.history_array(kernel_size - 1, kernel_size, include_self=True)

    def kda(self, module, x, cache):
        width = len(self.parents)
        raw = (module.q_proj(x), module.k_proj(x), module.v_proj(x))
        if module.ling_merged_conv:
            windows = [self.convolution_windows(module.conv_kernel, value, state)
                       for value, state in zip(raw, (cache[0], cache[1], cache[2]), strict=True)]
            (q, k, v), (qs, ks, vs) = module.convolve_qkv(
                [item[:, -1:] for item in windows], [item[:, :-1] for item in windows],
            )
        else:
            (q, qs), (k, ks), (v, vs) = [self.convolution(conv, value, state) for conv, value, state in
                zip((module.q_conv1d, module.k_conv1d, module.v_conv1d), raw,
                    (cache[0], cache[1], cache[2]), strict=True)]
        shape = (width, 1, module.num_heads, module.head_dim)
        q, k, v = (value.reshape(shape) for value in (q, k, v))
        q, k = normalize_qk(module.recurrence_spec, q, k)
        decay = (
            module.f_proj(x) if module.no_kda_lora
            else module.f_b_proj(module.f_a_proj(x))
        ).reshape(shape)
        beta = module.b_proj(x).reshape(width, 1, module.num_heads)
        initial = cache[3]
        if initial is None:
            initial = mx.zeros(
                (1, module.num_heads, module.head_dim, module.head_dim),
                dtype=mx.float32,
            )
        a_log = module.A_log.reshape(module.num_heads, 1)
        dt_bias = module.dt_bias.reshape(module.num_heads, module.head_dim)

        def step(q, k, v, decay, beta, parent_states):
            return gated_delta_update(
                q, k, v, decay, beta, a_log, dt_bias,
                state=parent_states, use_kernel=not module.training,
                lower_bound=module.lower_bound,
            )

        outputs, states = cohort_tree_recurrence(
            self.parents, self.levels, initial, (q, k, v, decay, beta), step)
        self.recurrent[id(cache)] = (qs, ks, vs, states)
        output = mx.concatenate(outputs, axis=1)
        gate = module.g_proj(x) if module.no_kda_lora else module.g_b_proj(module.g_a_proj(x))
        gate = gate.reshape(1, width, module.num_heads, module.head_dim)
        output = module.output_norm_gate(output, gate)
        return module.o_proj(output.reshape(1, width, -1))

    def mla(self, module, q_nope, q_pe, kv, pe, cache, start):
        from ..technologies import mla

        outputs = []
        for row, path in enumerate(self.paths):
            indices = mx.array(path, dtype=mx.int32)
            keys = mx.concatenate([kv[..., :start, :], mx.take(kv[..., start:, :], indices, axis=2)], axis=2)
            positional = mx.concatenate([pe[..., :start, :], mx.take(pe[..., start:, :], indices, axis=2)], axis=2)
            fused = module.fused_latent_output(q_nope[:, :, row:row + 1], q_pe[:, :, row:row + 1],
                                                keys, positional)
            if fused is not None:
                outputs.append(fused)
                continue
            query = module.embed_q(q_nope[:, :, row:row + 1])
            output = mla.absorbed_attention(
                module.mla_spec, query, q_pe[:, :, row:row + 1], keys, positional, None,
                scale=module.scale, cache=cache,
            )
            outputs.append(module.unembed_out(output))
        return mx.concatenate(outputs, axis=2)

    def validate_path(self, path):
        self.topology.validate_path(path)

    def prepare_commit(self, path):
        path = self.topology.validate_path(path)
        # Prepare every layer before publishing any descriptor.
        prepared = []
        for live, private in zip(self.live, self.private, strict=True):
            if type(live) is KVCache:
                # Preserve the cache's allocated tail. The private array
                # owns this COW update; accepted branches are compacted
                # without rebuilding the entire committed prefix.
                _KV.prepare_commit(private, start=int(live.offset), path=path,
                                   width=len(self.parents))
            else:
                _KDA.prepare_commit(private, self.recurrent[id(private)], path)
            prepared.append((live, private.__dict__.copy()))
        return prepared

    publish = staticmethod(publish_descriptors)

    def commit(self, path):
        self.publish(self.prepare_commit(path))
        return [self.live]
