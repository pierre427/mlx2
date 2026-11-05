"""Parent-indexed Inkling tiles with packed projections and private branch state.

Original mlx2 glue. Four convolution histories follow true ancestry. Attention
keeps one query and its exact ancestor key count; projections use the installed
row-invariant lane law. Routed experts retain one-node gather arithmetic.
"""

from __future__ import annotations

import mlx.core as mx

from ...cow_cache import snapshot_prompt_cache_descriptors
from ..base import scaled_dot_product_attention
from ..cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from .cache import InklingConvolutionCache
from .language import banded_additive_mask


class InklingTree:
    def __init__(self, parents, cache):
        self.parents = tuple(parents)
        if not 1 <= len(parents) <= 16:
            raise ValueError("Inkling trees require 1..16 rows")
        self.paths = []
        for row, parent in enumerate(parents):
            if type(parent) is not int or (
                (row == 0 and parent != -1) or (row > 0 and not 0 <= parent < row)
            ):
                raise ValueError("Inkling tree must have one parent-first root")
            self.paths.append([*(self.paths[parent] if parent >= 0 else []), row])
        if len(cache) % 2:
            raise ValueError("Inkling cache must alternate KV and convolution planes")
        pairs = [CacheList(cache[i], cache[i + 1]) for i in range(0, len(cache), 2)]
        for item in pairs:
            if (
                type(item) is not CacheList
                or len(item.caches) != 2
                or type(item[0]) not in (KVCache, RotatingKVCache)
                or type(item[1]) is not InklingConvolutionCache
            ):
                raise ValueError("Inkling tree needs its KV/four-shortconv cache")
            if (
                item[1].speculating
                or item[1].left_padding is not None
                or item[1].lengths is not None
            ):
                raise ValueError(
                    "Inkling tree requires private unpadded committed state"
                )
        self.live = cache
        self.live_pairs = pairs
        private = snapshot_prompt_cache_descriptors(cache)[0]
        self.private = [
            CacheList(private[i], private[i + 1]) for i in range(0, len(private), 2)
        ]
        self.histories = {}
        self.raw_kv = {}

    def convolution(self, module, raw, conv):
        # Execute the ordinary one-row convolution for each node, staging only
        # tiny histories. Projection/MLP matrices are not replicated per node.
        count = module.kernel_size - 1
        state = conv[module.conv_idx]
        if state is None:
            state = mx.zeros((1, count, raw.shape[-1]), dtype=mx.float32)
        outputs, tails = [], []
        source = mx.concatenate([state, raw.astype(mx.float32)], axis=1)
        for row, path in enumerate(self.paths):
            history = [*range(count), *(count + ancestor for ancestor in path[:-1])][
                -count:
            ]
            previous = mx.take(source, mx.array(history, mx.int32), axis=1)
            cache = ArraysCache(4)
            cache[module.conv_idx] = previous
            outputs.append(module(raw[:, row : row + 1], cache=cache))
            tails.append(cache[module.conv_idx])
        self.histories.setdefault(id(conv), {})[module.conv_idx] = tails
        return mx.concatenate(outputs, axis=1)

    def attention(self, module, x, item):
        batch, width, _ = x.shape
        if batch != 1:
            raise ValueError("Inkling packed tree is per-request")
        dq, dk, dv, _ = module.qkvr_dims
        # Always call the installed projection: ordinary decoding uses the
        # same QKVR module once lane arithmetic is selected.
        packed = module.project_qkvr(x)
        k = self.convolution(module.k_sconv, packed[..., dq : dq + dk], item[1])
        v = self.convolution(
            module.v_sconv, packed[..., dq + dk : dq + dk + dv], item[1]
        )
        k = module.k_norm(k.reshape(1, width, module.n_kv, module.head_dim)).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(1, width, module.n_kv, module.head_dim).transpose(0, 2, 1, 3)
        q = module.q_norm(
            packed[..., :dq].reshape(1, width, module.n_heads, module.head_dim)
        ).transpose(0, 2, 1, 3)
        r = packed[..., dq + dk + dv :].reshape(1, width, module.n_heads, module.d_rel)
        start = int(item[0].offset)
        self.raw_kv[id(item)] = (k, v)
        keys, values = item[0].update_and_fetch(k, v)
        if type(item[0]) is RotatingKVCache:
            keys = item[0]._temporal_order(item[0].keys)
            values = item[0]._temporal_order(item[0].values)
        prefix = keys.shape[2] - width
        outputs = []
        for row, path in enumerate(self.paths):
            ids = mx.array(path, mx.int32)
            kr = mx.concatenate(
                [
                    keys[..., :prefix, :],
                    mx.take(keys[..., prefix : prefix + width, :], ids, axis=2),
                ],
                axis=2,
            )
            vr = mx.concatenate(
                [
                    values[..., :prefix, :],
                    mx.take(values[..., prefix : prefix + width, :], ids, axis=2),
                ],
                axis=2,
            )
            size = kr.shape[2]
            if module.sliding and size > module.sliding:
                kr, vr = kr[..., -module.sliding :, :], vr[..., -module.sliding :, :]
                size = kr.shape[2]
            mask = banded_additive_mask(
                r[:, row : row + 1],
                module.rel_proj.astype(x.dtype),
                size - 1,
                size,
                module.sliding,
                module.rel_extent,
                shape_ref=kr,
                fused=module.fused,
            )
            qr = q[:, :, row : row + 1]
            if module.log_floor is not None:
                pos = mx.array(start + len(path), mx.float32)
                tau = (
                    1 + module.log_alpha * mx.log(mx.maximum(pos / module.log_floor, 1))
                ).astype(x.dtype)
                qr = qr * tau
                mask = mx.where(mask > -1e29, mask * tau, mask)
            outputs.append(
                scaled_dot_product_attention(
                    qr, kr, vr, cache=None, scale=module.scale, mask=mask
                )
            )
        out = (
            mx.concatenate(outputs, axis=2).transpose(0, 2, 1, 3).reshape(1, width, -1)
        )
        return module.o_proj(out)

    def forward(self, model, tokens, layers):
        model.completed_forwards += 1
        taps = {}
        hidden = model.model.embed(tokens)
        for index, (layer, item) in enumerate(
            zip(model.layers, self.private, strict=True)
        ):
            raw = self.attention(layer.self_attn, layer.input_layernorm(hidden), item)
            hidden = hidden + self.convolution(layer.attn_sconv, raw, item[1])
            normalized = layer.post_attention_layernorm(hidden)
            # Keep routed gather selection/reduction identical to singleton
            # ordinary decode. Shared/dense projection tiling still engages
            # through the lane installer; a future expert tile needs its own gate.
            if hasattr(layer.mlp, "switch_mlp"):
                raw = layer.mlp.tree(normalized)
            else:
                raw = layer.mlp(normalized)
            hidden = hidden + self.convolution(layer.mlp_sconv, raw, item[1])
            if index in layers:
                taps[index] = hidden
        return model._logits_from_norm(model.model.norm(hidden)), mx.concatenate(
            [taps[i] for i in layers], axis=-1
        )

    def validate_path(self, path):
        path = list(path)
        if (
            not path
            or any(type(i) is not int or not 0 <= i < len(self.paths) for i in path)
            or path != self.paths[path[-1]]
        ):
            raise ValueError("Inkling commit must follow one complete root path")

    def prepare_commit(self, path):
        self.validate_path(path)
        prepared = []
        ids = mx.array(path, mx.int32)
        for live, private in zip(self.live_pairs, self.private, strict=True):
            if type(live[0]) is RotatingKVCache:
                chosen = snapshot_prompt_cache_descriptors([live[0]])[0][0]
                keys, values = self.raw_kv[id(private)]
                chosen.update_and_fetch(
                    mx.take(keys, ids, axis=2), mx.take(values, ids, axis=2)
                )
                private.caches = (chosen, private[1])
            else:
                start = int(live[0].offset)
                for field in ("keys", "values"):
                    values = getattr(private[0], field)
                    kept = mx.take(
                        values[..., start : start + len(self.paths), :], ids, axis=2
                    )
                    values[..., start : start + len(path), :] = kept
                    setattr(private[0], field, values)
                private[0].offset = start + len(path)
            history = self.histories[id(private[1])]
            private[1].cache = [history[i][path[-1]] for i in range(4)]
            private[1].advance(len(path))
            prepared.extend(
                (dst, src.__dict__.copy())
                for dst, src in zip(live.caches, private.caches, strict=True)
            )
        return prepared

    @staticmethod
    def publish(prepared):
        for live, fields in prepared:
            live.__dict__.clear()
            live.__dict__.update(fields)
