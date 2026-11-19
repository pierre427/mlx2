# SPDX-License-Identifier: Apache-2.0
# Ported from the Hy4-flash reference implementations (transformers PR 50903
# ``HYV4FlashVision*`` and ``HYV4FlashModel.get_rope_index``; vLLM PR 68296
# ``hy_v4/nvidia/vision.py``).  See provenance/hy-v4-flash-vision.json.
"""Tencent Hy4-flash vision tower, patch merger and 3D M-RoPE positions.

Tower (``vision_config`` of tencent/Hy4-flash@d88c9dfd): a 16x16 patch
convolution plus a learned 128x128 position grid, bilinearly resampled to
each image's patch grid with the reference's ``(size + 0.1) / edge`` scale
factor; 27 pre-LayerNorm blocks (biased q/k/v/o, exact GELU MLP) attending
within each frame, no 2D RoPE and no final norm.  The ``rmsnorm_mlp_adapter``
merger RMS-normalizes, merges 2x2 patches with a stride-2 convolution,
applies GELU and a 1x1 convolution to the text width, and appends the
learned ``image_newline`` feature after every merged row.

Positions: each frame is ``H`` merged rows of ``W + 1`` tokens (the newline
column); all three planes of a frame start at the same offset and frames
advance by ``max(H, W + 1)``.  Text after a span continues at the span's
largest position plus one, so decode positions are ``slot + delta``.

A span is ``(slot_start, start_position, t, h, w)`` with merged ``h``/``w``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

Span = Tuple[int, int, int, int, int]

# Bound on [heads, rows, keys] fp32 scores per vision attention chunk.
VISION_SCORE_BUDGET = 1 << 27


# ------------------------------------------------------------- positions --

def span_length(span: Span) -> int:
    _, _, t, h, w = span
    return t * h * (w + 1)


def span_end(span: Span) -> int:
    return span[0] + span_length(span)


def span_delta_after(span: Span) -> int:
    """``position - slot`` of the text that follows the span."""
    _, start_pos, t, h, w = span
    return start_pos + t * max(h, w + 1) - span_end(span)


def text_position(spans: Sequence[Span], slot: int) -> int:
    """Position of a text token at ``slot`` (all three planes equal)."""
    delta = 0
    for span in spans:
        if span_end(span) <= slot:
            delta = span_delta_after(span)
        elif span[0] <= slot:
            raise ValueError("slot lies inside a vision span")
    return slot + delta


def vision_position_ids(start_position: int, t: int, h: int, w: int) -> np.ndarray:
    """``[3, t*h*(w+1)]`` positions of one span (``get_vision_position_ids``)."""
    block = max(h, w + 1)
    frame = np.arange(t, dtype=np.int64)[:, None, None] * block
    rows = np.arange(h, dtype=np.int64)[None, :, None]
    cols = np.arange(w + 1, dtype=np.int64)[None, None, :]
    shape = (t, h, w + 1)
    planes = [np.broadcast_to(frame, shape), np.broadcast_to(frame + rows, shape),
              np.broadcast_to(frame + cols, shape)]
    return np.stack([p.reshape(-1) for p in planes]) + int(start_position)


def slot_positions(spans: Sequence[Span], slots) -> np.ndarray:
    """``[3, n]`` int64 positions of arbitrary token ``slots``."""
    slots = np.asarray(slots, dtype=np.int64).reshape(-1)
    out = np.broadcast_to(slots, (3, slots.size)).copy()
    for span in spans:
        start, start_pos, t, h, w = span
        end = span_end(span)
        after = slots >= end
        out[:, after] = slots[after] + span_delta_after(span)
        inside = (slots >= start) & (slots < end)
        if inside.any():
            k = slots[inside] - start
            per_frame = h * (w + 1)
            frame, rest = k // per_frame, k % per_frame
            base = start_pos + frame * max(h, w + 1)
            out[0, inside] = base
            out[1, inside] = base + rest // (w + 1)
            out[2, inside] = base + rest % (w + 1)
    return out


def chunk_positions(spans: Sequence[Span], start: int, length: int):
    """RoPE positions of slots ``[start, start + length)``.

    An ``int`` offset (``start + delta``) when no span intersects the chunk,
    so text-only chunks keep the scalar fast-RoPE path; otherwise ``[3, L]``.
    """
    stop = start + length
    if not any(s[0] < stop and span_end(s) > start for s in spans):
        return text_position(spans, start) if spans else start
    return slot_positions(spans, np.arange(start, stop))


def mrope_planes(half: int, section: Sequence[int]) -> np.ndarray:
    """Plane (0 T, 1 H, 2 W) of each rotary frequency (interleaved M-RoPE).

    ``recomposition_frequencies``: the T plane is overwritten at
    ``slice(dim, section[dim] * 3, 3)`` by plane ``dim`` for H and W.
    """
    if len(section) != 3 or sum(section) != half:
        raise ValueError("Hy4-flash M-RoPE section does not cover the rotary width")
    plane = np.zeros(half, dtype=np.int64)
    for dim in (1, 2):
        idx = np.arange(dim, int(section[dim]) * 3, 3)
        plane[idx[idx < half]] = dim
    return plane


def find_spans(tokens: np.ndarray, start: int, image_token_id: int, grids, merge: int,
               prior: Sequence[Span]) -> Tuple[Span, ...]:
    """New spans for the image-token runs of one chunk (``get_rope_index``).

    ``grids`` are the ``(t, h, w)`` patch grids in prompt order.  Every run
    must be exactly one image's token count: the reference groups
    consecutive image tokens, so adjacent images without a separator are
    refused rather than mis-assigned.
    """
    hits = np.flatnonzero(tokens == image_token_id)
    runs = []
    if hits.size:
        breaks = np.flatnonzero(np.diff(hits) != 1)
        starts = np.concatenate([[hits[0]], hits[breaks + 1]])
        ends = np.concatenate([hits[breaks], [hits[-1]]]) + 1
        runs = list(zip(starts.tolist(), ends.tolist()))
    grids = [tuple(int(v) for v in g) for g in grids]
    if len(runs) != len(grids):
        raise ValueError(f"Hy4-flash prompt has {len(runs)} image spans for {len(grids)} images")
    spans = list(prior)
    new = []
    for (a, b), (t, gh, gw) in zip(runs, grids):
        if t < 1 or gh % merge or gw % merge or gh < merge or gw < merge:
            raise ValueError("Hy4-flash image grid is not a multiple of the merge size")
        h, w = gh // merge, gw // merge
        if b - a != t * h * (w + 1):
            raise ValueError("Hy4-flash image span length disagrees with its grid")
        slot = start + a
        span = (slot, text_position(spans, slot), t, h, w)
        spans.append(span)
        new.append(span)
    return tuple(new)


# ------------------------------------------------------------------ tower --

@dataclass
class VisionArgs:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_attention_heads: int = 16
    num_hidden_layers: int = 27
    num_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    out_hidden_size: int = 3072
    max_image_size: int = 2048
    min_image_size: int = 512
    rms_norm_eps: float = 1e-5

    # Topology switches this port implements; anything else refuses.
    _EXPECTED = {"hidden_act": "gelu", "interpolate_mode": "bilinear",
                 "projector_type": "rmsnorm_mlp_adapter", "perceive_pre_norm": True,
                 "perceive_post_norm": False, "cat_extra_token": 0, "temporal_patch_size": 1,
                 "spatial_patch_size": 1, "use_2d_rope": False, "learnable_mlp_pooling_size": 0}

    @classmethod
    def from_config(cls, config: dict) -> "VisionArgs":
        for key, value in cls._EXPECTED.items():
            if config.get(key, value) != value:
                raise ValueError(f"Hy4-flash vision {key}={config.get(key)!r} is not implemented")
        heads = int(config.get("num_attention_heads", 16))
        if int(config.get("num_key_value_heads") or heads) != heads:
            raise ValueError("Hy4-flash vision attention must have equal q/kv heads")
        fields = {k: config[k] for k in cls.__dataclass_fields__ if k in config}
        args = cls(**fields)
        if args.max_image_size % args.patch_size or args.hidden_size % args.num_attention_heads:
            raise ValueError("Hy4-flash vision geometry is inconsistent")
        return args

    @property
    def position_edge(self) -> int:
        return self.max_image_size // self.patch_size

    @property
    def patch_dim(self) -> int:
        return self.num_channels * self.patch_size * self.patch_size


def bilinear_taps(out: int, edge: int, size: int):
    """Source rows/weights of ``F.interpolate(bilinear, align_corners=False)``.

    The reference passes ``scale_factor = (size + 0.1) / edge``, so PyTorch
    samples at ``(i + 0.5) / scale - 0.5`` (clamped at zero) rather than the
    size-ratio grid.
    """
    scale = (size + 0.1) / edge
    if math.floor(edge * scale) != out:
        raise ValueError("Hy4-flash position grid resample size mismatch")
    src = np.maximum((np.arange(out, dtype=np.float64) + 0.5) / scale - 0.5, 0.0)
    i0 = np.floor(src).astype(np.int64)
    i0 = np.minimum(i0, edge - 1)
    i1 = np.minimum(i0 + 1, edge - 1)
    lam = src - i0
    return i0, i1, (1.0 - lam).astype(np.float32), lam.astype(np.float32)


class VisionAttention(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        D = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = D // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(D, D, bias=True)
        self.k_proj = nn.Linear(D, D, bias=True)
        self.v_proj = nn.Linear(D, D, bias=True)
        self.o_proj = nn.Linear(D, D, bias=True)

    def __call__(self, x, segments):
        N = x.shape[0]
        H, d = self.num_heads, self.head_dim

        def heads(t):
            return t.reshape(N, H, d).transpose(1, 0, 2)[None]  # [1, H, N, d]

        q, k, v = heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))
        outs = []
        for a, b in segments:  # each frame attends to itself only
            T = b - a
            rows = max(1, VISION_SCORE_BUDGET // max(1, H * T))
            for r in range(a, b, rows):
                e = min(b, r + rows)
                outs.append(mx.fast.scaled_dot_product_attention(
                    q[:, :, r:e], k[:, :, a:b], v[:, :, a:b], scale=self.scale))
        out = mx.concatenate(outs, axis=2)[0].transpose(1, 0, 2).reshape(N, H * d)
        return self.o_proj(out)


class VisionMLP(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.intermediate_size, bias=True)
        self.fc2 = nn.Linear(args.intermediate_size, args.hidden_size, bias=True)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = VisionAttention(args)
        self.layer_norm2 = nn.LayerNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.mlp = VisionMLP(args)

    def __call__(self, x, segments):
        x = x + self.self_attn(self.layer_norm1(x), segments)
        return x + self.mlp(self.layer_norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, args: VisionArgs):
        super().__init__()
        D, m = args.hidden_size, args.spatial_merge_size
        self.merge = m
        self.before_rms = nn.RMSNorm(D, eps=args.rms_norm_eps)
        # Convolutions stored as linears: proj_conv over (channel, row, col)
        # of each m x m block, proj_out a 1x1 convolution.
        self.proj_conv = nn.Linear(D * m * m, D * m * m, bias=True)
        self.proj_out = nn.Linear(D * m * m, args.out_hidden_size, bias=True)
        self.image_newline = mx.zeros((args.out_hidden_size,))

    def __call__(self, x, t: int, h: int, w: int):
        """``x [t*h*w, D]`` (row-major patches) -> ``[t*(h/m)*(w/m+1), out]``."""
        m = self.merge
        D = x.shape[-1]
        x = self.before_rms(x)
        x = x.reshape(t, h // m, m, w // m, m, D).transpose(0, 1, 3, 5, 2, 4)
        x = x.reshape(t, h // m, w // m, D * m * m)
        x = self.proj_out(nn.gelu(self.proj_conv(x)))
        newline = mx.broadcast_to(self.image_newline.astype(x.dtype), (t, h // m, 1, x.shape[-1]))
        x = mx.concatenate([x, newline], axis=2)
        return x.reshape(-1, x.shape[-1])


class VisionTower(nn.Module):
    """``HYV4FlashVisionTransformer`` + merger: patches -> text-width features."""

    def __init__(self, args: VisionArgs):
        super().__init__()
        self.args = args
        self.patch_embedding = nn.Linear(args.patch_dim, args.hidden_size, bias=True)
        self.position_embedding = nn.Embedding(args.position_edge ** 2 + 1, args.hidden_size)
        self.layers = [VisionBlock(args) for _ in range(args.num_hidden_layers)]
        self.patch_merger = PatchMerger(args)

    def num_tokens(self, grid) -> int:
        t, h, w = (int(v) for v in grid)
        m = self.args.spatial_merge_size
        return t * (h // m) * (w // m + 1)

    def position_grid(self, h: int, w: int, dtype):
        """Learned positions resampled to an ``h x w`` patch grid ``[h*w, D]``."""
        E = self.args.position_edge
        table = self.position_embedding.weight[1:].astype(mx.float32).reshape(E, E, -1)
        y0, y1, wy0, wy1 = bilinear_taps(h, E, h)
        x0, x1, wx0, wx1 = bilinear_taps(w, E, w)
        rows = (table[mx.array(y0)] * mx.array(wy0)[:, None, None]
                + table[mx.array(y1)] * mx.array(wy1)[:, None, None])  # [h, E, D]
        grid = (rows[:, mx.array(x0)] * mx.array(wx0)[None, :, None]
                + rows[:, mx.array(x1)] * mx.array(wx1)[None, :, None])  # [h, w, D]
        return grid.reshape(h * w, -1).astype(dtype)

    def __call__(self, pixel_values, grids):
        grids = [tuple(int(v) for v in g) for g in grids]
        a = self.args
        m = a.spatial_merge_size
        total = sum(t * h * w for t, h, w in grids)
        if pixel_values.ndim != 2 or pixel_values.shape != (total, a.patch_dim):
            raise ValueError("Hy4-flash pixel_values do not match the image grids")
        for t, h, w in grids:
            if t < 1 or h < m or w < m or h % m or w % m:
                raise ValueError("Hy4-flash image grid is not a multiple of the merge size")
        dtype = self.patch_embedding.weight.dtype
        x = self.patch_embedding(pixel_values.astype(dtype))
        x = x + mx.concatenate([mx.tile(self.position_grid(h, w, x.dtype), (t, 1))
                                for t, h, w in grids], axis=0)
        segments, cursor = [], 0
        for t, h, w in grids:
            for _ in range(t):
                segments.append((cursor, cursor + h * w))
                cursor += h * w
        for layer in self.layers:
            x = layer(x, segments)
        out, cursor = [], 0
        for t, h, w in grids:
            out.append(self.patch_merger(x[cursor:cursor + t * h * w], t, h, w))
            cursor += t * h * w
        return mx.concatenate(out, axis=0)

    @staticmethod
    def sanitize(weights: dict) -> dict:
        """Checkpoint ``vit.*`` names (and conv layouts) -> module names."""
        out = {}
        for key, value in weights.items():
            if key.startswith("model.vision_tower."):
                key = "vit." + key[len("model.vision_tower."):]
            if not key.startswith("vit."):
                continue
            key = key[len("vit."):]
            key = (key.replace(".input_layernorm.", ".layer_norm1.")
                   .replace(".post_attention_layernorm.", ".layer_norm2.")
                   .replace("mlp.dense_h_to_4h.", "mlp.fc1.")
                   .replace("mlp.dense_4h_to_h.", "mlp.fc2."))
            for old, new in (("perceive.norm.", "patch_merger.before_rms."),
                             ("perceive.proj.0.", "patch_merger.proj_conv."),
                             ("perceive.proj.2.", "patch_merger.proj_out."),
                             ("perceive.", "patch_merger.")):
                if key.startswith(old):
                    key = new + key[len(old):]
                    break
            if key.startswith("embeddings."):
                key = key[len("embeddings."):]
            if key in ("patch_embedding.weight", "patch_merger.proj_conv.weight",
                       "patch_merger.proj_out.weight") and value.ndim == 4:
                value = value.reshape(value.shape[0], -1)  # (out, in, kh, kw) -> (out, in*kh*kw)
            out[key] = value
        return out


def load_vision_tower(config: dict, raw: dict) -> Optional[VisionTower]:
    """A loaded tower from ``vision_config`` and raw ``vit.*`` tensors, or None."""
    vision = config.get("vision_config")
    if not isinstance(vision, dict) or not any(k.startswith("vit.") for k in raw):
        return None
    args = VisionArgs.from_config(vision)
    tower = VisionTower(args)
    tower.load_weights(list(VisionTower.sanitize(raw).items()), strict=True)
    return tower
