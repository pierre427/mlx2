"""Qwen3.8 whole-layer ragged verification over private B1 lane state.

This module is an adapter-internal arithmetic primitive.  It packs every valid
row in descriptor order and calls each transformer layer once.  Projections,
normalization and the MLP therefore see one flat slab; attention and recurrent
cores split only at the stateful boundary and retain one private cache per lane.

It performs no route selection or qualification and never touches live state.
The caller must provide already-private caches whose rollback epochs are owned
by the surrounding transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .. import row_exact_verify as REV
from ..flat_segmented_ast import FlatSegmentDescriptor
from ..row_exact.canary import probe_policy
from . import qwen38_27b as qwen38
from .base import create_attention_mask
from .cache import ArraysCache, KVCache
from .moe_nax_gather import prefill_scope
from .qwen_hybrid_row_exact import _scratch, ordinary_one_token_step

EXECUTOR_PROTOCOL = "qwen38-whole-layer-ragged-v1"


@dataclass(frozen=True, slots=True)
class Qwen38WholeLayerExecution:
    outputs: tuple[tuple[mx.array, mx.array], ...]
    layer_invocations: int
    attention_segment_invocations: int
    recurrent_segment_invocations: int
    packed_rows: int


def _execute_gdn_flat(gdn, projected, descriptor, caches, dtype, record):
    """One packed GDN projection/output path with private lane recurrence."""

    qkv, z, b, a = projected
    values: list[tuple[str, mx.array]] = []
    staged = []
    routes = {}
    for lane, length in enumerate(descriptor.query_lengths):
        start = descriptor.cu_query_lengths[lane]
        cache = caches[lane]
        conv0, state0 = cache[0], cache[1]
        scratch = _scratch(conv0, state0)
        states = []
        for position in range(length):
            row = slice(start + position, start + position + 1)
            route, value = ordinary_one_token_step(
                gdn,
                qkv[:, row],
                z[:, row],
                b[:, row],
                a[:, row],
                scratch,
                dtype,
            )
            routes[route] = routes.get(route, 0) + 1
            values.append((route, value))
            states.append((scratch[0], scratch[1]))
        staged.append((cache, [conv0, state0], states))

    stock = [value for route, value in values if route == "stock_decode"]
    projected_stock = None
    if stock:
        projected_stock = gdn.out_proj(mx.concatenate(stock, axis=1))
    outputs = []
    stock_index = 0
    for route, value in values:
        if route == "stock_decode":
            outputs.append(projected_stock[:, stock_index : stock_index + 1])
            stock_index += 1
        else:
            outputs.append(value)
    for cache, snapshot, states in staged:

        def replay(count, _states=states):
            count = int(count)
            if not 1 <= count <= len(_states):
                raise RuntimeError(
                    f"whole-layer GDN replay outside staged rows: {count}"
                )
            return list(_states[count - 1])

        cache.record_rollback(len(states), replay, snapshot)
        cache[0], cache[1] = states[-1]
        cache.advance(len(states))
    for route, count in routes.items():
        record.note("gdn", f"whole_layer_serial_{route}", count)
    record.note("gdn_layers", "whole_layer_private_b1s1")
    return mx.concatenate(outputs, axis=1)


def _validate_private_rows(model, descriptor, lane_caches) -> tuple[tuple, ...]:
    rows = tuple(tuple(row) for row in lane_caches)
    layers = model.language_model.model.layers
    if len(rows) != descriptor.lane_count or any(
        len(row) != len(layers) for row in rows
    ):
        raise ValueError("whole-layer Qwen3.8 private cache geometry differs")
    if len({id(cache) for row in rows for cache in row}) != sum(map(len, rows)):
        raise ValueError("whole-layer Qwen3.8 caches must be lane-private")
    for layer_index, layer in enumerate(layers):
        for row in rows:
            cache = row[layer_index]
            if layer.is_linear:
                if (
                    type(cache) is not ArraysCache
                    or not cache.speculating
                    or cache.batch_size != 1
                    or cache.lengths is not None
                    or cache.left_padding is not None
                ):
                    raise TypeError(
                        "whole-layer Qwen3.8 GDN requires an unpadded private B1 "
                        "rollback epoch"
                    )
            elif type(cache) is not KVCache:
                raise TypeError(
                    "whole-layer Qwen3.8 attention requires a private B1 KV cache"
                )
    return rows


def execute_qwen38_whole_layer_flat(
    model,
    flat_tokens,
    descriptor,
    lane_caches,
    capture_layers,
    *,
    allow_numerical_canaries: bool = False,
) -> Qwen38WholeLayerExecution:
    """Execute exact lane-major rows with one packed call per transformer layer."""

    if type(model) is not qwen38.Model:
        raise TypeError("whole-layer Qwen3.8 executor requires the exact wrapper")
    if not isinstance(descriptor, FlatSegmentDescriptor):
        raise TypeError("whole-layer Qwen3.8 executor requires validated geometry")
    tokens = tuple(flat_tokens)
    if len(tokens) != descriptor.logical_rows or any(
        type(token) is not int or token < 0 for token in tokens
    ):
        raise ValueError("whole-layer Qwen3.8 inputs must cover logical rows")
    captures = tuple(capture_layers)
    trunk = model.language_model.model
    if (
        not captures
        or captures != tuple(sorted(set(captures)))
        or captures[0] < 0
        or captures[-1] >= len(trunk.layers)
    ):
        raise ValueError("whole-layer Qwen3.8 capture geometry differs")
    rows = _validate_private_rows(model, descriptor, lane_caches)

    route = model.__dict__.get("_mlx2_qwen_hybrid_row_exact")
    if route is None:
        raise ValueError("whole-layer Qwen3.8 row-exact arithmetic is unavailable")
    record = REV.Window(descriptor.logical_rows)
    for reason in route.static_refusals:
        record.fail(reason)
    route._launch_refusals(record)
    if not record.exact:
        raise ValueError("whole-layer Qwen3.8 row-exact binding differs")

    hidden = trunk.embed_tokens(mx.array([tokens], dtype=mx.int32))
    taps = []
    attention_segments = 0
    recurrent_segments = 0
    with (
        REV.window(record),
        prefill_scope(False),
        probe_policy(allow=allow_numerical_canaries),
    ):
        for layer_index, layer in enumerate(trunk.layers):
            normalized = layer.input_layernorm(hidden)
            if layer.is_linear:
                gdn = layer.linear_attn
                projected = gdn._input_projections(normalized)
                mixed = _execute_gdn_flat(
                    gdn,
                    projected,
                    descriptor,
                    tuple(rows[lane][layer_index] for lane in range(len(rows))),
                    normalized.dtype,
                    record,
                )
                recurrent_segments += descriptor.lane_count
            else:
                attention = layer.self_attn
                q_all = attention.q_proj(normalized)
                k_all = attention.k_proj(normalized)
                v_all = attention.v_proj(normalized)
                lane_outputs = []
                for lane, length in enumerate(descriptor.query_lengths):
                    start = descriptor.cu_query_lengths[lane]
                    cache = rows[lane][layer_index]
                    pieces = []
                    for position in range(length):
                        row = slice(start + position, start + position + 1)
                        mask = create_attention_mask(normalized[:, row], cache)
                        pieces.append(
                            attention._attend(
                                q_all[:, row],
                                k_all[:, row],
                                v_all[:, row],
                                1,
                                1,
                                mask,
                                cache,
                            )
                        )
                    lane_outputs.append(mx.concatenate(pieces, axis=1))
                    attention_segments += 1
                mixed = attention.o_proj(mx.concatenate(lane_outputs, axis=1))
                record.note(
                    "attention", "whole_layer_private_b1s1", descriptor.logical_rows
                )
                record.note("attention_layers", "whole_layer_private_b1s1")
            residual = hidden + mixed
            hidden = residual + layer.mlp(layer.post_attention_layernorm(residual))
            if layer_index in captures:
                taps.append(hidden)

        if len(taps) != len(captures):
            raise RuntimeError("whole-layer Qwen3.8 tap count differs")
        normalized = trunk.norm(hidden)
        logits = model.logits(normalized)
        features = mx.concatenate(taps, axis=-1)
    if not record.exact:
        raise RuntimeError(
            "whole-layer Qwen3.8 arithmetic lost row exactness: "
            + ",".join(sorted(record.failures))
        )
    mx.eval(logits, features)
    outputs = tuple(
        (logits[0, row], features[0, row]) for row in range(descriptor.logical_rows)
    )
    return Qwen38WholeLayerExecution(
        outputs=outputs,
        layer_invocations=len(trunk.layers),
        attention_segment_invocations=attention_segments,
        recurrent_segment_invocations=recurrent_segments,
        packed_rows=descriptor.logical_rows,
    )


__all__ = [
    "EXECUTOR_PROTOCOL",
    "Qwen38WholeLayerExecution",
    "execute_qwen38_whole_layer_flat",
]
