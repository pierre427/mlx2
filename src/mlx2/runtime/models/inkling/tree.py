"""Parent-indexed Inkling tiles with packed projections and private branch state.

Original mlx2 glue. Four convolution histories follow true ancestry. Attention
keeps singleton arithmetic and its exact ancestor key count; projections use the
installed row-invariant lane law. Grouped experts preserve private row reductions.
"""

from __future__ import annotations

import mlx.core as mx

from ...cow_cache import lend_append_only_kv, snapshot_prompt_cache_descriptors
from ..cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from .cache import InklingConvolutionCache


class InklingTree:
    def __init__(self, parents, cache, *, descriptor_plan=None, descriptor_scope=None):
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
        # Depth-d chain path [0, ..., d]; a node whose path equals it reads a
        # prefix slice of K/V in the local-attention loop.
        self._chain_paths = [list(range(depth + 1)) for depth in range(len(parents))]
        self.chain_slice_rows = 0
        self.chain_window_batches = 0
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
        captured = (
            None
            if descriptor_plan is None
            else descriptor_plan.capture(cache, descriptor_scope)
        )
        self.descriptor_plan_used = captured is not None
        private = (
            captured[0]
            if captured is not None
            else snapshot_prompt_cache_descriptors(cache, measure=False)[0]
        )
        self.private = [
            CacheList(private[i], private[i + 1]) for i in range(0, len(private), 2)
        ]
        self.histories = {}
        self.raw_kv = {}
        # Mapped-attention column maps, one per (prefix, window) per round.
        self.mapping_memo = {}
        self.lent_global_kv = 0
        self.anchor_bases = None

    def forward_anchor(self, model, tokens, layers):
        """One ordinary step on private KV rings and convolution histories.

        A root-only tree has no branches to gather or ancestry to evaluate.
        Keeping its private descriptor transaction lets parked AST use the
        ordinary tensor path without treating wrapped rings as trimmable KV.
        """
        if self.parents != (-1,) or tokens.shape != (1, 1):
            raise ValueError("Inkling ordinary anchor requires one root token")
        bases = [int(pair[0].offset) for pair in self.live_pairs]
        for live, private in zip(self.live_pairs, self.private, strict=True):
            # The existing contract lends only append-only planes. It refuses
            # local rings, which retain their private COW snapshots. Aborted
            # global suffix writes are outside the live committed fill level.
            lend_append_only_kv(live[0], private[0])
        flat = [plane for pair in self.private for plane in pair.caches]
        result = model.forward_with_taps(tokens, flat, layers)
        self.anchor_bases = bases
        return result

    @staticmethod
    def forward_anchor_cohort(model, trees, tokens, layers):
        """Batch ordinary token math, retaining private per-request KV planes.

        There is no ancestry to gather for a single root. Short convolutions
        share one launch across rows (including the ordinary residual fusion),
        while attention reads each request's KV without merging its prefix.
        Expert reductions retain the tree's singleton arithmetic contract.
        """
        if tokens.shape != (len(trees), 1) or any(
            tree.parents != (-1,) for tree in trees
        ):
            raise ValueError("Inkling ordinary cohort needs one root per lane")
        bases = [[int(pair[0].offset) for pair in tree.live_pairs] for tree in trees]
        for tree in trees:
            for live, private in zip(tree.live_pairs, tree.private, strict=True):
                lend_append_only_kv(live[0], private[0])
        hidden = model.model.embed(tokens)
        model.completed_forwards += 1
        taps = {}
        for index, layer in enumerate(model.layers):
            items = [tree.private[index] for tree in trees]
            convs = [item[1] for item in items]
            attention = layer.self_attn
            projected = attention.project_qkvr(layer.input_layernorm(hidden))
            dq, dk, dv, _ = attention.qkvr_dims
            k = InklingTree.anchor_convolution(
                attention.k_sconv, projected[..., dq : dq + dk], convs
            )
            v = InklingTree.anchor_convolution(
                attention.v_sconv, projected[..., dq + dk : dq + dk + dv], convs
            )
            shape = (len(trees), 1, attention.n_kv, attention.head_dim)
            k = attention.k_norm(k.reshape(shape)).transpose(0, 2, 1, 3)
            v = v.reshape(shape).transpose(0, 2, 1, 3)
            q = attention.q_norm(
                projected[..., :dq].reshape(
                    len(trees), 1, attention.n_heads, attention.head_dim
                )
            ).transpose(0, 2, 1, 3)
            r = projected[..., dq + dk + dv :].reshape(
                len(trees), 1, attention.n_heads, attention.d_rel
            )
            outputs = []
            for row, item in enumerate(items):
                kv = item[0]
                keys, values = kv.update_and_fetch(k[row : row + 1], v[row : row + 1])
                if type(kv) is RotatingKVCache:
                    keys = kv._temporal_order(kv.keys)
                    values = kv._temporal_order(kv.values)
                if attention.sliding and keys.shape[2] > attention.sliding:
                    keys = keys[..., -attention.sliding :, :]
                    values = values[..., -attention.sliding :, :]
                outputs.append(
                    attention.attend(
                        q[row : row + 1],
                        keys,
                        values,
                        r[row : row + 1],
                        keys.shape[2] - 1,
                    )
                )
            raw = attention.o_proj(
                mx.concatenate(outputs, axis=0)
                .transpose(0, 2, 1, 3)
                .reshape(len(trees), 1, -1)
            )
            hidden = InklingTree.anchor_convolution(
                layer.attn_sconv, raw, convs, residual=hidden
            )
            normalized = layer.post_attention_layernorm(hidden)
            raw = (
                layer.mlp.tree(normalized)
                if hasattr(layer.mlp, "switch_mlp")
                else layer.mlp(normalized)
            )
            hidden = InklingTree.anchor_convolution(
                layer.mlp_sconv, raw, convs, residual=hidden
            )
            for conv in convs:
                conv.advance(1)
            if index in layers:
                taps[index] = hidden
        for tree, offsets in zip(trees, bases, strict=True):
            tree.anchor_bases = offsets
        return model.tap_outputs(hidden, taps, layers)

    @staticmethod
    def anchor_convolution(module, raw, convs, *, residual=None):
        """Pack only the tiny histories; publish tails to private descriptors."""
        slot = module.conv_idx
        count = module.kernel_size - 1
        cache = ArraysCache(4)
        cache[slot] = mx.concatenate(
            [
                conv[slot]
                if conv[slot] is not None
                else mx.zeros((1, count, raw.shape[-1]), dtype=mx.float32)
                for conv in convs
            ],
            axis=0,
        )
        out = module(raw, cache=cache, residual=residual)
        for row, conv in enumerate(convs):
            conv[slot] = cache[slot][row : row + 1]
        return out

    def convolution(self, module, raw, conv):
        if (
            module.tree_fused
            and module.kernel_size == 4
            and mx.default_device() == mx.gpu
        ):
            from .ancestry import convolution

            out, tails = convolution(module, raw, conv[module.conv_idx], self.paths)
            module.tree_calls += 1
            self.histories.setdefault(id(conv), {})[module.conv_idx] = [
                tails[row : row + 1] for row in range(len(self.paths))
            ]
            return out
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

    def attention(self, module, x, item, *, packed=None, project_output=True):
        batch, width, _ = x.shape
        if batch != 1:
            raise ValueError("Inkling packed tree is per-request")
        dq, dk, dv, _ = module.qkvr_dims
        # Always call the installed projection: ordinary decoding uses the
        # same QKVR module once lane arithmetic is selected.
        if packed is None:
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
        # Match the ordinary decode law before packing queries. With eight KV
        # heads the ordinary singleton kernel uses the same sixteen splits as
        # mapped attention. Every ancestor must meet its global 16K threshold;
        # local/short queries retain ordinary SDPA instead of forcing the slower
        # mapped law on all 42 layers (including parked AST decoding).
        decode_mapped = (
            module.attention_policy == "decode"
            and not module.sliding
            and module.n_kv == 8
            and prefix + 1 >= 16384
        )
        from .attention import supported

        # Ordinary decode selects the kernel only where ``supported`` holds
        # (Metal, d128, GQA 1..4, one dtype) and keeps reference SDPA
        # otherwise; the tree follows the same law instead of raising.
        if (
            (module.attention_policy == "mapped" or decode_mapped)
            and module.head_dim == 128
            and supported(q, keys, values, r, module.rel_proj.astype(q.dtype))
        ):
            from .tree_attention import attend

            tau = None
            if module.log_floor is not None:
                positions = mx.array(
                    [[start + len(path) for path in self.paths]], mx.float32
                )
                tau = (
                    1
                    + module.log_alpha
                    * mx.log(mx.maximum(positions / module.log_floor, 1))
                ).astype(q.dtype)
                q = q * tau[:, None, :, None]
            out = attend(
                q,
                keys,
                values,
                r,
                module.rel_proj.astype(q.dtype),
                scale=module.scale,
                paths=self.paths,
                prefix=prefix,
                window=module.sliding,
                tau=tau,
                memo=self.mapping_memo,
            )
            module.mapped_attention_calls += 1
            out = out.transpose(0, 2, 1, 3).reshape(1, width, -1)
            return module.o_proj(out) if project_output else out
        batched = self._attend_chain_windows(module, q, keys, values, r, prefix, start)
        if batched is not None:
            out = batched.transpose(0, 2, 1, 3).reshape(1, width, -1)
            return module.o_proj(out) if project_output else out
        outputs = []
        for row, path in enumerate(self.paths):
            if path == self._chain_paths[len(path) - 1]:
                # Ancestors 0..row in order are the leading tree rows, so the
                # gathered K/V is exactly this prefix slice of the same
                # buffer: no per-node copy of the whole local window.
                kr = keys[..., : prefix + len(path), :]
                vr = values[..., : prefix + len(path), :]
                self.chain_slice_rows += 1
            else:
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
            outputs.append(
                module.attend(
                    q[:, :, row : row + 1],
                    kr,
                    vr,
                    r[:, row : row + 1],
                    size - 1,
                    query_position=start + len(path),
                )
            )
        out = (
            mx.concatenate(outputs, axis=2).transpose(0, 2, 1, 3).reshape(1, width, -1)
        )
        return module.o_proj(out) if project_output else out

    # Run a chain's full-window local attention as one batch of one-row
    # problems (see ``_attend_chain_windows``); False keeps the per-node loop.
    batch_chain_windows = True

    def _attend_chain_windows(self, module, q, keys, values, r, prefix, start):
        """A chain's local attention as one batch of the per-node one-row calls.

        Every node of a chain whose window is full attends the same number of
        keys at the same relative offset: node ``i`` reads the ``sliding``
        keys ending at its own row, a fixed-stride window of one buffer.
        Stacking those windows as batch items (an overlapping strided view)
        gives the stock kernels the same per-item inputs and shape as the
        per-node loop's one-row calls, so the mask, position scaling and SDPA
        run once per layer instead of once per node. Returns ``None`` (use the
        loop) for anything else: a branched tree, a window not yet full, or a
        policy whose one-row call is not the reference SDPA.
        """
        width = len(self.paths)
        window = int(module.sliding)
        if (
            not self.batch_chain_windows
            or width < 2
            or window <= 0
            or module.attention_policy not in ("reference", "decode")
            or self.paths[-1] != self._chain_paths[width - 1]
            or prefix + 1 < window
            or keys.shape[2] != prefix + width
        ):
            return None
        from ..base import scaled_dot_product_attention
        from .language import banded_additive_mask

        _, n_kv, total, head_dim = keys.shape
        value_dim = values.shape[3]
        first = prefix + 1 - window
        keys = mx.contiguous(keys)
        values = mx.contiguous(values)
        kb = mx.as_strided(
            keys, (width, n_kv, window, head_dim),
            (head_dim, total * head_dim, head_dim, 1), first * head_dim,
        )
        vb = mx.as_strided(
            values, (width, n_kv, window, value_dim),
            (value_dim, total * value_dim, value_dim, 1), first * value_dim,
        )
        qb = q.transpose(2, 1, 0, 3)  # [width, heads, 1, head_dim]
        rb = r.transpose(1, 0, 2, 3)  # [width, 1, heads, d_rel]
        mask = banded_additive_mask(
            rb,
            module.rel_proj.astype(q.dtype),
            window - 1,
            window,
            module.sliding,
            module.rel_extent,
            shape_ref=kb if module.fused else None,
            fused=module.fused,
        )
        # Receipts count launches: one mask and one SDPA per batched layer
        # (``chain_window_batches`` on the tree counts the batches).
        module.mask_tiles += 1
        module.fused_mask_calls += int(module.fused and mx.default_device() == mx.gpu)
        if module.log_floor is not None:
            qpos = mx.array(
                [[start + row + 1] for row in range(width)], mx.float32
            )
            tau = 1.0 + module.log_alpha * mx.log(
                mx.maximum(qpos / module.log_floor, 1.0)
            )
            tau = tau.reshape(width, 1, 1, 1).astype(q.dtype)
            qb = qb * tau
            mask = mx.where(mask > -1e29, mask * tau, mask)
        module.attention_reference_calls += 1
        self.chain_window_batches += 1
        out = scaled_dot_product_attention(
            qb, kb, vb, cache=None, scale=module.scale, mask=mask
        )
        return out.transpose(2, 1, 0, 3)  # [1, heads, width, value_dim]

    def lend_global_kv(self):
        """Append tree rows into the live global KV buffers, as anchors do.

        A private descriptor slices K/V to the fill level, so a tree's first
        append concat-grew every global layer's whole prefix into a new buffer
        each round (~0.5 GiB at 16K).  Lent buffers are written only past the
        lane's offset; MLX copies instead if another holder shares the buffer.
        Local rings keep their private snapshots (the contract refuses them).
        """
        for live, private in zip(self.live_pairs, self.private, strict=True):
            if lend_append_only_kv(live[0], private[0]):
                self.lent_global_kv += 1

    def forward(self, model, tokens, layers):
        model.completed_forwards += 1
        self.lend_global_kv()
        taps = {}
        hidden = model.model.embed(tokens)
        for index, (layer, item) in enumerate(
            zip(model.layers, self.private, strict=True)
        ):
            raw = self.attention(layer.self_attn, layer.input_layernorm(hidden), item)
            hidden = hidden + self.convolution(layer.attn_sconv, raw, item[1])
            normalized = layer.post_attention_layernorm(hidden)
            # Grouped expert tiles retain private singleton reductions; the
            # ordinary gather path remains available when the tile is disabled.
            if hasattr(layer.mlp, "switch_mlp"):
                raw = layer.mlp.tree(normalized)
            else:
                raw = layer.mlp(normalized)
            hidden = hidden + self.convolution(layer.mlp_sconv, raw, item[1])
            if index in layers:
                taps[index] = hidden
        return model.tap_outputs(hidden, taps, layers)

    def validate_path(self, path):
        path = list(path)
        if (
            not path
            or any(type(i) is not int or not 0 <= i < len(self.paths) for i in path)
            or path != self.paths[path[-1]]
        ):
            raise ValueError("Inkling commit must follow one complete root path")

    @staticmethod
    def forward_cohort(model, trees, tokens, layers):
        """Share dense/expert work across requests; retain independent ancestry.

        State operations stay in each private tree. A cohort has no cache writes
        to another request, even when the requests share an APCv2 backing slab.
        """
        for tree in trees:
            tree.lend_global_kv()
        widths = [len(tree.paths) for tree in trees]
        boundaries = []
        position = 0
        for width in widths:
            boundaries.append((position, position + width))
            position += width
        packed_tokens = (
            tokens.reshape(1, -1)
            if isinstance(tokens, mx.array)
            else mx.array([[t for row in tokens for t in row]], mx.uint32)
        )
        hidden = model.model.embed(packed_tokens)
        model.completed_forwards += 1
        taps = {}
        for index, layer in enumerate(model.layers):
            normalized = layer.input_layernorm(hidden)
            projections = layer.self_attn.project_qkvr(normalized)
            attention = []
            for tree, (begin, end) in zip(trees, boundaries, strict=True):
                attention.append(
                    tree.attention(
                        layer.self_attn,
                        normalized[:, begin:end],
                        tree.private[index],
                        packed=projections[:, begin:end],
                        project_output=False,
                    )
                )
            raw = layer.self_attn.o_proj(mx.concatenate(attention, axis=1))
            hidden = hidden + mx.concatenate(
                [
                    tree.convolution(
                        layer.attn_sconv, raw[:, begin:end], tree.private[index][1]
                    )
                    for tree, (begin, end) in zip(trees, boundaries, strict=True)
                ],
                axis=1,
            )
            normalized = layer.post_attention_layernorm(hidden)
            raw = (
                layer.mlp.tree(normalized)
                if hasattr(layer.mlp, "switch_mlp")
                else layer.mlp(normalized)
            )
            hidden = hidden + mx.concatenate(
                [
                    tree.convolution(
                        layer.mlp_sconv, raw[:, begin:end], tree.private[index][1]
                    )
                    for tree, (begin, end) in zip(trees, boundaries, strict=True)
                ],
                axis=1,
            )
            if index in layers:
                taps[index] = hidden
        logits, features = model.tap_outputs(hidden, taps, layers)
        return (
            mx.concatenate([logits[:, a:b] for a, b in boundaries], axis=0),
            mx.concatenate([features[:, a:b] for a, b in boundaries], axis=0),
        )

    def prepare_commit(self, path):
        self.validate_path(path)
        prepared = []
        if self.anchor_bases is not None:
            for live, private, base in zip(
                self.live_pairs, self.private, self.anchor_bases, strict=True
            ):
                if int(live[0].offset) != base or int(private[0].offset) != base + 1:
                    raise RuntimeError("Inkling ordinary anchor boundary changed")
                prepared.extend(
                    (dst, src.__dict__.copy())
                    for dst, src in zip(live.caches, private.caches, strict=True)
                )
            return prepared
        ids = mx.array(path, mx.int32)
        for live, private in zip(self.live_pairs, self.private, strict=True):
            if type(live[0]) is RotatingKVCache:
                chosen = snapshot_prompt_cache_descriptors([live[0]], measure=False)[0][0]
                keys, values = self.raw_kv[id(private)]
                chosen.update_and_fetch(
                    mx.take(keys, ids, axis=2), mx.take(values, ids, axis=2)
                )
                private.caches = (chosen, private[1])
            else:
                start = int(live[0].offset)
                # Rows of a root-prefix path (every chain) already sit in
                # order at start..start+len(path): nothing to write back.
                # Other paths gather from the rows this round appended, not
                # from a view of the lent buffer: a lazy view feeding the
                # write stops MLX from donating the buffer, so every global
                # layer copied its whole K/V prefix on each commit.
                if path != list(range(len(path))):
                    for field, raw in zip(
                        ("keys", "values"), self.raw_kv[id(private)], strict=True
                    ):
                        values = getattr(private[0], field)
                        source = (
                            raw
                            if raw.dtype == values.dtype
                            else values[..., start : start + len(self.paths), :]
                        )
                        values[..., start : start + len(path), :] = mx.take(
                            source, ids, axis=2
                        )
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
