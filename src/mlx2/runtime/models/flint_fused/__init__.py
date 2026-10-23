# SPDX-License-Identifier: Apache-2.0
"""Explicit Flint Q4 decode candidate; ordinary prefill and APCv2 own state.

Shaders fuse QKV, routed/shared gate-up and down/combine around native attention.
New K/V are ordinary outputs applied through the existing cache update API.
No rec runtime or cache engine is imported. See provenance/rec-flint-fused.json.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MODE = "q4_fused_v1"
SCHEMA = "flint-functional-q4-v1"


def shader_identity():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for name in (
        "layer_header.metal",
        "qkv.metal",
        "gate_up.metal",
        "down_combine.metal",
    ):
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()


def validate_config(config):
    """Reject before tensor load; this first candidate is a pinned Q4 slice."""
    expected = {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 1024,
        "sliding_window": 513,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("fused Flint Q4 candidate: unsupported topology")
    types = config.get("layer_types", [])
    no_rope = set(config.get("no_rope_layer_indices", []))
    if (
        not types
        or len(types) != config.get("num_hidden_layers")
        or any(
            kind not in ("full_attention", "sliding_attention")
            or (kind == "full_attention") != (i in no_rope)
            for i, kind in enumerate(types)
        )
    ):
        raise ValueError(
            "fused Flint candidate requires full/no-RoPE and sliding/RoPE layers"
        )
    quant = config.get("quantization") or {}
    if quant.get("bits") != 4 or quant.get("group_size") != 64:
        raise ValueError("fused Flint candidate requires affine Q4 group64 layers")
    for key, value in quant.items():
        if (
            isinstance(value, dict)
            and not key.startswith(("model.embed_tokens", "lm_head"))
            and (value.get("bits", 4) != 4 or value.get("group_size", 64) != 64)
        ):
            raise ValueError("fused Flint Q4 candidate refuses mixed-bit layers")


def cache_geometry(cache):
    """Host-only physical read map before the canonical one-token update."""
    from ..cache import BatchKVCache, BatchRotatingKVCache, KVCache, RotatingKVCache

    if type(cache) not in (
        KVCache,
        RotatingKVCache,
        BatchKVCache,
        BatchRotatingKVCache,
    ):
        raise ValueError("fused Flint candidate: unsupported cache type")
    if getattr(cache, "speculating", False):
        raise ValueError(
            "fused Flint candidate does not support speculative cache updates"
        )
    if type(cache) is BatchKVCache:
        if cache._right_padding is not None or cache._padding_floor() != [0]:
            raise ValueError("fused Flint candidate requires an unpadded B1 cache")
        prior = cache._idx
    elif type(cache) is BatchRotatingKVCache:
        if cache._lengths is not None:
            raise ValueError("fused Flint candidate requires finalized sliding prefill")
        prior = cache._offset
    else:
        prior = cache.offset
    keys = cache.keys
    if keys is None:
        return (0, 0, 0, 0)
    if keys.shape[0] != 1 or keys.shape[1] != 4 or keys.shape[3] != 128:
        raise ValueError("fused Flint candidate cache geometry must be B1/H4/D128")
    cap = keys.shape[2]
    window = getattr(cache, "max_size", None)
    if window is None:
        return (cap, prior, 0, 0)
    if getattr(cache, "keep", 0):
        raise ValueError("fused Flint candidate refuses retained sink cache columns")
    if cap > window:
        return (cap, window - 1, cap - window + 1, 0)
    if prior < window:
        return (cap, prior, 0, 0)
    insert = cache._idx % window
    return (cap, window - 1, (insert + 1) % window, window)


_KERNELS = {}


def _kernel(kind):
    import mlx.core as mx

    if kind not in _KERNELS:
        root = Path(__file__).parent
        if kind == "qkv":
            names = ["hidden", "weights", "metadata", "parameters", "logical_position"]
            outputs = ["queries", "new_keys", "new_values"]
        elif kind == "gate_up":
            names = [
                "normalized",
                "weights",
                "metadata",
                "parameters",
                "indices",
                "silu_table",
            ]
            outputs = ["activation"]
        else:
            names = [
                "residual",
                "weights",
                "metadata",
                "parameters",
                "indices",
                "routing_weights",
                "activation",
            ]
            outputs = ["result"]
        _KERNELS[kind] = mx.fast.metal_kernel(
            name="mlx2_flint_" + kind + "_q4_" + shader_identity()[:12],
            input_names=names,
            output_names=outputs,
            header=(root / "layer_header.metal").read_text(),
            source=(root / (kind + ".metal")).read_text(),
            ensure_row_contiguous=True,
            compile_options={"math_mode": "safe"},
        )
    return _KERNELS[kind]


class Block:
    def __init__(self, layers, silu):
        import mlx.core as mx

        self.layers, self.silu = layers, silu
        self.calls = 0
        entries, metadata, offset = [], [], 0

        def add(obj, name, *, convert=False):
            nonlocal offset
            original = getattr(obj, name)
            x = original.astype(mx.bfloat16) if convert else original
            if x.dtype not in (mx.bfloat16, mx.uint32) and name != "expert_bias":
                raise ValueError(
                    f"fused Flint candidate refuses {name} dtype {x.dtype}"
                )
            pad = (-offset) % 256
            offset += pad
            at = offset
            offset += x.nbytes
            entries.append((obj, name, original, x, pad, at, convert))
            return at

        def linear(d, base, p):
            if p.bits != 4 or p.group_size != 64 or p.mode != "affine":
                raise ValueError(
                    "fused Flint candidate requires all layer projections Q4g64"
                )
            d[base : base + 5] = [
                add(p, "weight"),
                add(p, "scales"),
                add(p, "biases"),
                p.weight.shape[-2],
                p.scales.shape[-1] * 64,
            ]

        for layer in layers:
            a, m = layer.self_attn, layer.mlp
            sparse = hasattr(m, "switch_mlp")
            d = [0] * 64
            d[1] = int(not a.use_rope)
            d[2] = int(sparse)
            d[3] = add(layer.input_layernorm, "weight")
            d[4] = add(layer.post_attention_layernorm, "weight")
            d[5] = add(a.q_norm, "weight")
            d[6] = add(a.k_norm, "weight")
            d[7] = add(a, "softmax_offset")
            for base, p in zip(
                (10, 15, 20, 25), (a.q_proj, a.k_proj, a.v_proj, a.o_proj)
            ):
                linear(d, base, p)
            inner = m.switch_mlp if sparse else m
            for base, p in zip(
                (30, 35, 40), (inner.gate_proj, inner.up_proj, inner.down_proj)
            ):
                linear(d, base, p)
            if sparse:
                # The checkpoint router is BF16, including exports storing its
                # exact BF16 values in FP32. Preserve the ordinary parameter.
                router = m.gate.weight
                packed = router.astype(mx.bfloat16)
                if not bool(mx.array_equal(packed.astype(router.dtype), router).item()):
                    raise ValueError(
                        "fused Flint candidate router cannot be represented in BF16"
                    )
                d[60] = add(m.gate, "weight", convert=True)

                # Flint has no expert bias. Own a zero tensor in the bank.
                class Holder:
                    pass

                holder = Holder()
                holder.expert_bias = mx.zeros((256,), mx.float32)
                d[9] = add(holder, "expert_bias")
                for base, p in zip(
                    (45, 50, 55),
                    (
                        m.shared_expert.gate_proj,
                        m.shared_expert.up_proj,
                        m.shared_expert.down_proj,
                    ),
                ):
                    linear(d, base, p)
            metadata.append(d)
        pieces = []
        for obj, name, original, x, pad, at, convert in entries:
            if pad:
                pieces.append(mx.zeros((pad,), mx.uint8))
            pieces.append(x.reshape(-1).view(mx.uint8))
        self.bank = mx.concatenate(pieces)
        mx.eval(self.bank)
        for obj, name, original, x, pad, at, convert in entries:
            if not convert:
                setattr(
                    obj,
                    name,
                    self.bank[at : at + x.nbytes].view(x.dtype).reshape(x.shape),
                )
        self.metadata = mx.array(metadata, mx.uint32)
        mx.eval(self.metadata)
        mx.clear_cache()

    def __call__(self, hidden, caches):
        import mlx.core as mx

        from ..base import create_attention_mask, scaled_dot_product_attention

        if hidden.shape != (1, 1, 2048) or hidden.dtype != mx.bfloat16:
            raise ValueError(
                "fused Flint candidate supports BF16 B1 one-token decode only"
            )
        # Small views keep MLX resource accounting from forcing a command
        # buffer commit for each multi-gigabyte bank binding. The view aliases
        # the owned allocation; shader offsets remain within that allocation.
        bank = self.bank[:256]
        for index, (layer, cache) in enumerate(zip(self.layers, caches, strict=True)):
            cache_geometry(cache)  # Fail closed on unsupported padding/state.
            if cache.keys is not None and (
                cache.keys.dtype != mx.bfloat16 or cache.values.dtype != mx.bfloat16
            ):
                raise ValueError("fused Flint candidate requires BF16 K/V")
            position = cache.offset
            if isinstance(position, int):
                position = mx.array([position], mx.int32)
            params = mx.array([index], mx.uint32)
            q, k, v = _kernel("qkv")(
                inputs=[hidden, bank, self.metadata, params, position],
                grid=(36 * 256, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[(1, 32, 1, 128), (1, 4, 1, 128), (1, 4, 1, 128)],
                output_dtypes=[mx.bfloat16] * 3,
            )
            attn = layer.self_attn
            mask = (
                create_attention_mask(hidden, cache, window_size=513)
                if layer.layer_type == "sliding_attention"
                else create_attention_mask(hidden, cache)
            )
            k, v = cache.update_and_fetch(k, v)
            attended = (
                scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    cache=cache,
                    scale=attn.scale,
                    mask=mask,
                    sinks=attn.softmax_offset.astype(q.dtype),
                )
                .transpose(0, 2, 1, 3)
                .reshape(1, 1, 4096)
            )
            residual = hidden + attn.o_proj(attended)
            normalized = layer.post_attention_layernorm(residual)
            if not hasattr(layer.mlp, "switch_mlp"):
                hidden = residual + layer.mlp(normalized)
            else:
                mlp = layer.mlp
                scores = mx.sigmoid(mlp.gate(normalized).astype(mx.float32))
                indices = mx.argpartition(scores, kth=-6, axis=-1)[..., -6:]
                routing_weights = mx.take_along_axis(scores, indices, axis=-1)
                routing_weights = routing_weights / (
                    mx.sum(routing_weights, axis=-1, keepdims=True) + 1e-20
                )
                (activation,) = _kernel("gate_up")(
                    inputs=[
                        normalized,
                        bank,
                        self.metadata,
                        params,
                        indices,
                        self.silu,
                    ],
                    grid=(32 * 256, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(4096,)],
                    output_dtypes=[mx.bfloat16],
                )
                (hidden,) = _kernel("down_combine")(
                    inputs=[
                        residual,
                        bank,
                        self.metadata,
                        params,
                        indices,
                        routing_weights,
                        activation,
                    ],
                    grid=(64 * 256, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[hidden.shape],
                    output_dtypes=[mx.bfloat16],
                )
            self.calls += 1
        return hidden


def validate_device():
    import mlx.core as mx

    info = mx.device_info()
    if (info.get("architecture"), info.get("device_name")) != (
        "applegpu_g15s",
        "Apple M3 Pro",
    ):
        raise ValueError(
            "fused Flint candidate currently supports the tested M3 Pro worker envelope only"
        )


def install(model):
    import mlx.core as mx

    from ..activations import swiglu

    validate_device()
    gates = (
        (mx.arange(65536, dtype=mx.uint32) << 16).view(mx.float32).astype(mx.bfloat16)
    )
    silu = swiglu(gates, mx.ones_like(gates))
    mx.eval(silu)
    model.fused_blocks = [
        Block(model.layers[i : i + 4], silu) for i in range(0, len(model.layers), 4)
    ]
