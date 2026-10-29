"""Adapter-local packed projections and an original fused residual RMSNorm.

Ordinary standard-decoder math remains available without this opt-in install.
The same cache update/mask/RoPE interfaces serve ordinary, batched and AST rows.
See provenance/nanbeige43.json; external stacks supplied ideas, no shader code.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
from mlx import nn

from ..runtime.models.activations import swiglu
from ..runtime.models.base import scaled_dot_product_attention

SCHEMA = "mlx2.nanbeige43-packed-residual-rms-v2"
_SOURCE = r"""
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_threadgroup;
    uint width = residual_shape[residual_ndim - 1];
    threadgroup float partial[8];
    float sum = 0.0f;
    for (uint col = lane; col < width; col += 256) {
        // Round the residual addition before normalization, like the reference.
        T value = T(float(residual[row * width + col]) + float(delta[row * width + col]));
        summed[row * width + col] = value;
        float f = float(value);
        sum += f * f;
    }
    sum = simd_sum(sum);
    if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32) {
        float total = lane < 8 ? partial[lane] : 0.0f;
        total = simd_sum(total);
        if (lane == 0) partial[0] = metal::rsqrt(total / float(width) + NB_RMS_EPS);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col = lane; col < width; col += 256) {
        float value = float(summed[row * width + col]);
        normalized[row * width + col] = T((value * partial[0]) * float(weight[col]));
    }
"""


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def _kernel(eps):
    return mx.fast.metal_kernel(
        name="nanbeige43_residual_rms",
        input_names=["residual", "delta", "weight"],
        output_names=["summed", "normalized"],
        source=_SOURCE,
        header=f"constant constexpr float NB_RMS_EPS = {float(eps):.17g}f;",
    )


def residual_rms(residual, delta, weight, eps):
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("residual RMS epsilon must be positive and finite")
    if residual.shape != delta.shape or residual.dtype != delta.dtype:
        raise ValueError("residual RMS operands must match")
    if weight.shape != (residual.shape[-1],) or weight.dtype != residual.dtype:
        raise ValueError("residual RMS norm weight must match the hidden dtype")
    if residual.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        raise ValueError("residual RMS requires floating activations")
    # CPU is an independent arithmetic reference, never an implicit GPU backend.
    if mx.default_device() == mx.cpu:
        summed = residual + delta
        return summed, mx.fast.rms_norm(summed, weight, eps)
    rows = residual.size // residual.shape[-1]
    return _kernel(eps)(
        inputs=[residual, delta, weight],
        template=[("T", residual.dtype)],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[residual.shape, residual.shape],
        output_dtypes=[residual.dtype, residual.dtype],
    )


def _pack(members):
    """Merge compatible native projections; original members become stack views."""
    types = {type(member) for member in members}
    if len(types) != 1 or next(iter(types)) not in {nn.Linear, nn.QuantizedLinear}:
        raise ValueError("packed projections require uniform native linear modules")
    first = members[0]
    if any("bias" in member for member in members):
        raise ValueError("Nanbeige packed projections must be unbiased")
    sizes = tuple(member.weight.shape[0] for member in members)
    if isinstance(first, nn.QuantizedLinear):
        geometry = (first.bits, first.group_size, first.mode)
        if any((m.bits, m.group_size, m.mode) != geometry for m in members):
            raise ValueError("packed quantized projection formats differ")
        if first.bits != 8 or first.group_size != 64 or first.mode != "affine":
            raise ValueError("Nanbeige fused candidate requires affine Q8 group64")
        packed = nn.QuantizedLinear(
            first.weight.shape[1] * 4,
            sum(sizes),
            bias=False,
            bits=8,
            group_size=64,
            mode="affine",
        )
        keys = ("weight", "scales", "biases")
    else:
        packed = nn.Linear(first.weight.shape[1], sum(sizes), bias=False)
        keys = ("weight",)
    for key in keys:
        arrays = [m[key] for m in members]
        if any(
            a.shape[1:] != arrays[0].shape[1:] or a.dtype != arrays[0].dtype
            for a in arrays
        ):
            raise ValueError("packed projection shapes or dtypes differ")
        stack = mx.concatenate(arrays, axis=0)
        mx.eval(stack)
        packed[key] = stack
        start = 0
        for member, size in zip(members, sizes):
            member[key] = stack[start : start + size]
            start += size
    return packed, sizes


class PackedAttention(nn.Module):
    def __init__(self, original, kernel_policy=None):
        super().__init__()
        from .nanbeige43_kernels import validate_kernel_policy

        self.kernel_policy = validate_kernel_policy(kernel_policy)
        self.kernel_counts = {
            "lane_calls": 0,
            "reference_projection_calls": 0,
            "compiled_projection_calls": 0,
            "rope_append_calls": 0,
        }
        if hasattr(original, "q_norm"):
            raise ValueError("Nanbeige has no QK norm")
        self.qkv, self.sizes = _pack(
            (original.q_proj, original.k_proj, original.v_proj)
        )
        self.o_proj, self.rope = original.o_proj, original.rope
        self.n_heads, self.n_kv_heads = original.n_heads, original.n_kv_heads
        self.head_dim, self.scale = original.head_dim, original.scale
        self.calls = 0

    def __call__(self, x, mask=None, cache=None, *, row_exact=False):
        if row_exact:
            raise ValueError(
                "packed Nanbeige has a separate numerical contract from S1 row-exact"
            )
        batch, length, _ = x.shape
        from .nanbeige43_kernels import compiled_q8, project, quantized_q8, rope_append

        if self.kernel_policy["compiled_blocks"] and quantized_q8(self.qkv):
            qkv = compiled_q8(x, self.qkv.weight, self.qkv.scales, self.qkv.biases)
            self.kernel_counts["compiled_projection_calls"] += 1
        else:
            qkv = project(self.qkv, x, self.kernel_policy, self.kernel_counts)
        q, k, v = mx.split(qkv, [self.sizes[0], self.sizes[0] + self.sizes[1]], axis=-1)
        q = q.reshape(batch, length, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        v = v.reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        offset = cache.offset if cache is not None else 0
        appended = None
        if self.kernel_policy["rope_kv_fused"]:
            from .nanbeige43_kernels import physical_rope_append

            hook = getattr(cache, "try_fused_rope_append", None)
            appended = (
                hook(q, k, v, self.rope)
                if callable(hook)
                else physical_rope_append(q, k, v, self.rope, cache)
            )
            key = (
                "physical_rope_append_calls"
                if appended is not None
                else "physical_rope_append_refusals"
            )
            self.kernel_counts[key] = self.kernel_counts.get(key, 0) + 1
        if appended is None and self.kernel_policy["rope_kv_append"]:
            appended = rope_append(q, k, v, self.rope, cache)
        if appended is not None:
            q, k, v = appended
            self.kernel_counts["rope_append_calls"] += 1
        else:
            q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        self.calls += 1
        return project(
            self.o_proj,
            out.transpose(0, 2, 1, 3).reshape(batch, length, -1),
            self.kernel_policy,
            self.kernel_counts,
        )


class PackedMLP(nn.Module):
    def __init__(self, original, kernel_policy=None):
        super().__init__()
        from .nanbeige43_kernels import validate_kernel_policy

        self.kernel_policy = validate_kernel_policy(kernel_policy)
        self.kernel_counts = {
            "lane_calls": 0,
            "reference_projection_calls": 0,
            "compiled_mlp_calls": 0,
            "q8_swiglu_calls": 0,
        }
        self.gate_up, self.sizes = _pack((original.gate_proj, original.up_proj))
        self.down_proj = original.down_proj
        self.calls = 0

    def __call__(self, x, *, row_exact=False):
        if row_exact:
            raise ValueError("packed Nanbeige cannot claim row-exact verification")
        from .nanbeige43_kernels import compiled_mlp, project, q8_swiglu, quantized_q8

        policy, counts = self.kernel_policy, self.kernel_counts
        self.calls += 1
        if (
            policy["q8_swiglu"]
            and x.size // x.shape[-1] >= policy["q8_swiglu_min_rows"]
        ):
            # Explicit candidate: refuse unsupported devices/geometry/canary instead
            # of silently labelling ordinary activation as fused execution.
            hidden = q8_swiglu(x, self.gate_up, policy["q8_tile_rows"])
            counts["q8_swiglu_calls"] += 1
        elif (
            policy["compiled_blocks"]
            and quantized_q8(self.gate_up)
            and quantized_q8(self.down_proj)
        ):
            counts["compiled_mlp_calls"] += 1
            return compiled_mlp(
                x,
                self.gate_up.weight,
                self.gate_up.scales,
                self.gate_up.biases,
                self.down_proj.weight,
                self.down_proj.scales,
                self.down_proj.biases,
            )
        else:
            gate, up = mx.split(
                project(self.gate_up, x, policy, counts), [self.sizes[0]], axis=-1
            )
            hidden = swiglu(gate, up)
        return project(self.down_proj, hidden, policy, counts)


class FusedLayer(nn.Module):
    def __init__(self, original, kernel_policy=None):
        super().__init__()
        self.self_attn = PackedAttention(original.self_attn, kernel_policy)
        self.mlp = PackedMLP(original.mlp, kernel_policy)
        self.input_layernorm = original.input_layernorm
        self.post_attention_layernorm = original.post_attention_layernorm
        self.residual_rms_calls = 0

    def __call__(self, x, mask=None, cache=None, *, row_exact=False):
        if row_exact:
            raise ValueError("Nanbeige fused layer cannot claim row-exact verification")
        attended = self.self_attn(self.input_layernorm(x), mask, cache)
        h, normalized = residual_rms(
            x,
            attended,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps,
        )
        self.residual_rms_calls += 1
        return h + self.mlp(normalized)


def install(model, *, kernel_policy=None):
    from .nanbeige43_kernels import validate_kernel_policy

    kernel_policy = validate_kernel_policy(kernel_policy)
    # Build one layer at a time; no whole-model duplicate packed checkpoint.
    for index, layer in enumerate(model.model.layers):
        model.model.layers[index] = FusedLayer(layer, kernel_policy)
        mx.clear_cache()
    model.eval()
    mx.eval(model.parameters())
    return model


def status(model):
    from .nanbeige43_kernels import kernel_contract

    layers = [layer for layer in model.layers if isinstance(layer, FusedLayer)]
    calls = sum(layer.residual_rms_calls for layer in layers)
    return {
        "implemented": True,
        "qualified": False,
        "selected": bool(layers),
        "observed_used": False,
        "scheduled_layer_calls": calls,
        "observation_note": "scheduled counters are not completed GPU observations",
        "packed_layers": len(layers),
        "source_sha256": source_identity(),
        "kernel_candidates": (
            kernel_contract(layers[0].mlp.kernel_policy) if layers else None
        ),
        "scheduled_kernel_counts": {
            key: sum(
                module.kernel_counts.get(key, 0)
                for layer in layers
                for module in (layer.self_attn, layer.mlp)
            )
            for key in sorted(
                {
                    key
                    for layer in layers
                    for module in (layer.self_attn, layer.mlp)
                    for key in module.kernel_counts
                }
            )
        },
    }
