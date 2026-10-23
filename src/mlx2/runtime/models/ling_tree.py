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

from .cache import ArraysCache, KVCache
from .gated_delta import gated_delta_update


class LingTree:
    def __init__(self, parents, cache):
        self.parents = tuple(parents)
        if not 1 <= len(parents) <= 32:
            raise ValueError("Ling tree requires 1..32 rows")
        self.paths = []
        self.depths = []
        for row, parent in enumerate(parents):
            if type(parent) is not int or (
                (row == 0 and parent != -1) or (row and not 0 <= parent < row)
            ):
                raise ValueError("Ling tree requires one parent-first root")
            path = [] if parent < 0 else self.paths[parent]
            self.paths.append([*path, row])
            self.depths.append(len(path))
        self.levels = [
            [row for row, depth in enumerate(self.depths) if depth == level]
            for level in range(max(self.depths) + 1)
        ]
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
        count = module.kernel_size - 1
        if state is None:
            state = mx.zeros((1, count, raw.shape[-1]), dtype=raw.dtype)
        # Each window is the last K-1 inputs on this node's true ancestry,
        # followed by this node's input, never an adjacent sibling's input.
        indices = []
        for path in self.paths:
            history = [*range(count), *(count + row for row in path)]
            indices.append(history[-module.kernel_size:])
        source = mx.concatenate([state, raw], axis=1)
        windows = mx.take(source, mx.array(indices, dtype=mx.int32), axis=1)[0]
        values, tails = module(windows[:, -1:], windows[:, :-1], None, None)
        return values, tails

    def kda(self, module, x, cache):
        from mlx import nn

        width = len(self.parents)
        q, qs = self.convolution(module.q_conv1d, module.q_proj(x), cache[0])
        k, ks = self.convolution(module.k_conv1d, module.k_proj(x), cache[1])
        v, vs = self.convolution(module.v_conv1d, module.v_proj(x), cache[2])
        shape = (width, 1, module.num_heads, module.head_dim)
        q, k, v = (value.reshape(shape) for value in (q, k, v))
        q = (module.scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = module.scale * mx.fast.rms_norm(k, None, 1e-6)
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
        outputs, states = [None] * width, [None] * width
        for level in self.levels:
            indices = mx.array(level, dtype=mx.int32)
            parent_states = mx.concatenate([
                initial if self.parents[row] < 0 else states[self.parents[row]]
                for row in level
            ])
            output, state = gated_delta_update(
                mx.take(q, indices, axis=0), mx.take(k, indices, axis=0),
                mx.take(v, indices, axis=0), mx.take(decay, indices, axis=0),
                mx.take(beta, indices, axis=0),
                module.A_log.reshape(module.num_heads, 1),
                module.dt_bias.reshape(module.num_heads, module.head_dim),
                state=parent_states, use_kernel=not module.training,
                lower_bound=module.lower_bound,
            )
            for index, row in enumerate(level):
                outputs[row], states[row] = output[index:index + 1], state[index:index + 1]
        self.recurrent[id(cache)] = (qs, ks, vs, states)
        output = mx.concatenate(outputs, axis=1)
        gate = module.g_proj(x) if module.no_kda_lora else module.g_b_proj(module.g_a_proj(x))
        gate = gate.reshape(1, width, module.num_heads, module.head_dim)
        output = module.o_norm(output) * nn.sigmoid(gate)
        return module.o_proj(output.reshape(1, width, -1))

    def mla(self, module, q_nope, q_pe, kv, pe, cache, start):
        from .base import scaled_dot_product_attention

        outputs = []
        for row, path in enumerate(self.paths):
            indices = mx.array(path, dtype=mx.int32)
            keys = mx.concatenate([kv[..., :start, :], mx.take(kv[..., start:, :], indices, axis=2)], axis=2)
            positional = mx.concatenate([pe[..., :start, :], mx.take(pe[..., start:, :], indices, axis=2)], axis=2)
            query = module.embed_q(q_nope[:, :, row:row + 1])
            scores = (q_pe[:, :, row:row + 1] * module.scale) @ positional.swapaxes(-1, -2)
            output = scaled_dot_product_attention(
                query, keys, keys, cache=cache, scale=module.scale, mask=scores,
            )
            outputs.append(module.unembed_out(output))
        return mx.concatenate(outputs, axis=2)

    def validate_path(self, path):
        path = list(path)
        if (
            not path or any(type(row) is not int or not 0 <= row < len(self.parents) for row in path)
            or path != self.paths[path[-1]]
        ):
            raise ValueError("Ling tree commit must follow one complete root path")

    def prepare_commit(self, path):
        self.validate_path(path)
        path = list(path)
        indices = mx.array(path, dtype=mx.int32)
        # Prepare every layer before publishing any descriptor.
        prepared = []
        for live, private in zip(self.live, self.private, strict=True):
            if type(live) is KVCache:
                start = int(live.offset)
                for field in ("keys", "values"):
                    values = getattr(private, field)
                    kept = mx.take(values[..., start:start + len(self.parents), :], indices, axis=2)
                    # Preserve the cache's allocated tail. The private array
                    # owns this COW update; accepted branches are compacted
                    # without rebuilding the entire committed prefix.
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

    @staticmethod
    def publish(prepared):
        for live, fields in prepared:
            live.__dict__.clear()
            live.__dict__.update(fields)

    def commit(self, path):
        self.publish(self.prepare_commit(path))
        return [self.live]
