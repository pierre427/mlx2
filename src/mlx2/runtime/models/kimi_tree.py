# SPDX-License-Identifier: MIT
"""Kimi Linear parent-indexed tree math; provenance/kimi-linear-speculative.json.

Same packing contract as Ling's tree (``ling_tree.LingTree``, whose node
paths, ancestry convolution windows and commit are reused): projections and
MLPs run once per layer across all tile rows; everything that carries state
reproduces Kimi's ordinary one-token arithmetic per node.

* KDA: each node's convolution window is the last K-1 inputs along its true
  ancestry; FP32 l2 q/k, the unbounded softplus log-decay gate, FP32 v and the
  vector delta recurrence run in depth cohorts from each node's own parent
  state with the module's ordinary one-token law (``_recurrence``); the FLA
  gated norm follows.  Siblings never see each other's inputs.
* MLA (NoPE): no rotary position enters, so a node's query attends over the
  committed latent/shared prefix plus exactly its ancestor rows, one query
  row at a time through the module's own ordinary ``attend`` policy law.

What is shared with sequential decode is the per-node law; the projection
GEMMs see the tile's row count (stock MLX kernels are not row-invariant), as
for every multi-row verify.  ``HybridTreeExecutor`` remains the bit-for-bit
one-token reference.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from .ling_tree import LingTree


class KimiTree(LingTree):
    def _merged_convolution(self, module, raw, cache):
        """Ancestry windows for the merged q/k/v depthwise convolution."""
        count = module.conv_kernel - 1
        p = module.num_heads * module.head_dim
        if cache[0] is None:
            history = mx.zeros((1, count, 3 * p), dtype=raw.dtype)
        else:
            history = mx.concatenate(list(cache)[:3], axis=-1)
        indices = []
        for path in self.paths:
            window = [*range(count), *(count + row for row in path)]
            indices.append(window[-module.conv_kernel:])
        source = mx.concatenate([history, raw], axis=1)
        windows = mx.take(source, mx.array(indices, dtype=mx.int32), axis=1)[0]
        values = nn.silu(mx.conv1d(windows, module._conv, groups=3 * p))
        tails = mx.contiguous(windows[:, 1:])
        return mx.split(values, 3, axis=-1), mx.split(tails, 3, axis=-1)

    def kimi_kda(self, module, x, cache):
        from .kimi_linear import kda_gates, l2_qk

        width = len(self.parents)
        h, d = module.num_heads, module.head_dim
        module.calls["tree_calls"] += 1
        module.calls["tree_nodes"] += width
        q, k, v, beta, a, gate = module.project(x)
        if module.fused_projections:
            (q, k, v), (qs, ks, vs) = self._merged_convolution(
                module, mx.concatenate([q, k, v], axis=-1), cache)
            module.calls["merged_conv"] += 1
        else:
            q, qs = self.convolution(module.q_conv1d, q, cache[0])
            k, ks = self.convolution(module.k_conv1d, k, cache[1])
            v, vs = self.convolution(module.v_conv1d, v, cache[2])
        shape = (width, 1, h, d)
        q, k = l2_qk(q.reshape(shape), k.reshape(shape))
        log_decay, beta = kda_gates(
            a.reshape(shape), beta.reshape(width, 1, h),
            module.A_log.reshape(h, 1), module.dt_bias.reshape(h, d))
        decay = mx.exp(log_decay)
        v = v.reshape(shape).astype(mx.float32)
        initial = cache[3]
        if initial is None:
            initial = mx.zeros((1, h, d, d), dtype=mx.float32)
        outputs, states = [None] * width, [None] * width
        for level in self.levels:
            indices = mx.array(level, dtype=mx.int32)
            parent_states = mx.concatenate([
                initial if self.parents[row] < 0 else states[self.parents[row]]
                for row in level
            ])
            output, state = module._recurrence(
                *(mx.take(value, indices, axis=0) for value in (q, k, v, decay, beta)),
                parent_states, None,
            )
            for index, row in enumerate(level):
                outputs[row], states[row] = output[index:index + 1], state[index:index + 1]
            module.calls["tree_levels"] += 1
        self.recurrent[id(cache)] = (qs, ks, vs, states)
        output = mx.concatenate(outputs, axis=1)
        out = module.norm_gate(output, gate, x.dtype)
        return module.o_proj(out.reshape(1, width, -1))

    def kimi_mla(self, module, x, cache):
        a = module.args
        width = len(self.parents)
        h, v = a.num_attention_heads, a.v_head_dim
        module.calls["tree_calls"] += 1
        module.calls["tree_nodes"] += width
        q, latent, shared = module.project(x)
        start = int(cache.offset)
        latent, shared = cache.update_and_fetch(latent, shared)
        outputs = []
        for row, path in enumerate(self.paths):
            indices = mx.array(path, dtype=mx.int32) + start
            keys = mx.concatenate([latent[..., :start, :], mx.take(latent, indices, axis=2)], axis=2)
            pieces = mx.concatenate([shared[..., :start, :], mx.take(shared, indices, axis=2)], axis=2)
            # One query row at a time: the ordinary one-token decode law,
            # with no mask (every key is on this node's own ancestry).
            outputs.append(module.attend(q[:, :, row:row + 1], keys, pieces, None, x.itemsize))
        out = mx.concatenate(outputs, axis=2)
        return module.o_proj(out.transpose(0, 2, 1, 3).reshape(1, width, h * v))

    def kda(self, module, x, cache):  # pragma: no cover - guard
        raise ValueError("Kimi trees run Kimi KDA math, not Ling's")

    def mla(self, *args, **kwargs):  # pragma: no cover - guard
        raise ValueError("Kimi trees run NoPE MLA math, not Ling's RoPE MLA")

    def rope(self, *args, **kwargs):  # pragma: no cover - guard
        raise ValueError("Kimi MLA is NoPE: no tree rotary positions")
