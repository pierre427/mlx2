# SPDX-License-Identifier: MIT
"""Inkling-Small native multi-token-prediction head (sidecar weights).

Mechanism mined from vLLM ``vllm/models/inkling/nvidia/mtp.py`` and SGLang
``python/sglang/srt/models/inkling.py`` (``InklingMTPLayer``); see
``provenance/inkling-mtp-20261104.json``.  The released mlx2 conversion of
Inkling-Small carries no MTP tensors, so the head is loaded from a separate,
SHA-256-pinned q8 sidecar and attached to the target outside its module tree.

Per depth ``k`` (the spec in the provenance entry):

- hidden input: depth 0 takes the target's post-final-norm hidden (before the
  ``/mup`` logit scaling); depth ``k >= 1`` takes depth ``k - 1``'s raw block
  output.  ``chain_hidden_post_norm`` is false: there is no chain norm.
- token input: ``embed_norm_k(backbone_embed_norm(embed_tokens(token)))``,
  the target's own embedding table and embedding norm.
- ``x = input_proj(concat([hidden_norm_k(h), embedded], -1))`` (hidden first).
- one full Inkling decoder block with a dense SwiGLU MLP (output times its
  learned ``global_scale``, as the target's dense layers), sliding-window or
  global attention per ``mtp_config.local_layer_ids``.
- logits: the target's LM head law on ``raw / mup`` (unpadded-vocabulary
  crop), no final norm; the raw output chains to the next depth.

Each depth owns a KV cache plus four short-convolution histories at the
target's absolute positions (row ``p`` of every depth cache is position ``p``),
so relative bias, log-scaling ``tau`` and convolution history match the
reference engines' absolute-position MTP layers.
"""

from __future__ import annotations

import dataclasses

import mlx.core as mx
from mlx import nn

from ..cache import KVCache, RotatingKVCache
from .cache import InklingConvolutionCache
from .language import InklingAttention, InklingDecoderLayer

MAX_DEPTHS = 8
SIDECAR_PREFIX = "model.mtp.layers."
_ATTENTION_PARTS = ("wq_du", "wk_dv", "wv_dv", "wr_du")
_LEAVES = ("weight", "scales", "biases")


class InklingMTPAttention(InklingAttention):
    """Inkling attention whose stacked q/k/v/r projection may be quantized.

    The target keeps a BF16 ``qkvr_proj`` and slices its rows on the unfused
    path; a q8 ``QuantizedLinear`` cannot be row-sliced as a dense matrix, so
    the head always applies the stacked projection as one module call.
    """

    def project_qkvr(self, x):
        self.projection_calls += 1
        self.fused_projection_calls += 1
        return self.qkvr_proj(x)


def depth_config(config, sliding):
    """A target config copy for one MTP block: dense MLP forced, attention
    kind chosen by the depth, never by the backbone's layer order."""
    return dataclasses.replace(
        config,
        num_hidden_layers=1,
        local_layer_ids=[0] if sliding else [],
        layer_types=None,
        mlp_layer_types=None,
        dense_mlp_idx=1,
    )


class InklingMTPDepth(nn.Module):
    def __init__(self, config, *, sliding):
        super().__init__()
        width = config.hidden_size
        self.sliding = bool(sliding)
        self.hidden_norm = nn.RMSNorm(width, eps=config.rms_norm_eps)
        self.embed_norm = nn.RMSNorm(width, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(2 * width, width, bias=False)
        block_config = depth_config(config, self.sliding)
        self.block = InklingDecoderLayer(block_config, 0)
        self.block.self_attn = InklingMTPAttention(block_config, 0)
        self.window = block_config.sliding_window_size

    def __call__(self, hidden, embedded, cache):
        x = self.input_proj(
            mx.concatenate(
                [self.hidden_norm(hidden), self.embed_norm(embedded)], axis=-1
            )
        )
        return self.block(x, cache=cache)


class InklingMTP(nn.Module):
    """``depths`` loaded MTP depths (1..8), each a full Inkling block."""

    def __init__(self, config, *, depths, local_layer_ids):
        super().__init__()
        if type(depths) is not int or not 1 <= depths <= MAX_DEPTHS:
            raise ValueError(f"Inkling MTP depths must be an integer from 1 to {MAX_DEPTHS}")
        local = set(local_layer_ids)
        self.layers = [
            InklingMTPDepth(config, sliding=index in local) for index in range(depths)
        ]

    def make_cache(self):
        """Per depth ``[KV (ring for sliding depths), four shortconv]``."""
        caches = []
        for index, layer in enumerate(self.layers):
            conv = InklingConvolutionCache()
            conv.apcv2_layer_index = index
            kv = RotatingKVCache(layer.window) if layer.sliding else KVCache()
            caches.append([kv, conv])
        return caches

    def depth(self, index, hidden, embedded, cache):
        if type(index) is not int or not 0 <= index < len(self.layers):
            raise ValueError("Inkling MTP depth is not loaded")
        if hidden.shape[:2] != embedded.shape[:2]:
            raise ValueError("Inkling MTP hidden and token rows must pair")
        return self.layers[index](hidden, embedded, cache)


def _to_mlx_conv(value):
    """PyTorch depthwise ``[C, 1, K]`` to MLX ``[C, K, 1]`` (mlx-vlm's law)."""
    if value.ndim == 3 and value.shape[-1] == 1:
        return value
    if value.ndim != 3 or value.shape[1] != 1:
        raise ValueError("Inkling MTP short convolution weight is not depthwise")
    return value.transpose(0, 2, 1)


def _split_gate_up(value):
    """Interleaved ``w13`` rows: gate = even rows, up = odd rows.

    Valid on packed q8 rows and their scales/biases too: affine quantization
    groups run along the input axis, so row slicing commutes with it.
    """
    rows = value.shape[0]
    if rows % 2:
        raise ValueError("Inkling MTP w13 has an odd row count")
    pairs = value.reshape(rows // 2, 2, *value.shape[1:])
    return mx.contiguous(pairs[:, 0]), mx.contiguous(pairs[:, 1])


def sanitize_sidecar(weights, depths):
    """Map ``model.mtp.layers.{k}.*`` sidecar tensors to ``InklingMTP`` names.

    Every tensor of a loaded depth must be consumed; depths at or beyond
    ``depths`` are dropped.  Unknown names fail closed.
    """
    out, fused = {}, {}
    for name, value in weights.items():
        if not name.startswith(SIDECAR_PREFIX):
            raise ValueError(f"unexpected Inkling MTP sidecar tensor: {name}")
        index_text, sub = name[len(SIDECAR_PREFIX) :].split(".", 1)
        index = int(index_text)
        if not 0 <= index < MAX_DEPTHS:
            raise ValueError(f"Inkling MTP sidecar depth out of range: {name}")
        if index >= depths:
            continue
        base = f"layers.{index}."
        block = base + "block."
        attn = block + "self_attn."
        leaf = sub.rsplit(".", 1)[-1]
        if sub in ("hidden_norm.weight", "embed_norm.weight") or (
            sub.startswith("input_proj.") and leaf in _LEAVES
        ):
            out[base + sub] = value
        elif sub == "transformer_block.attn_norm.weight":
            out[block + "input_layernorm.weight"] = value
        elif sub == "transformer_block.mlp_norm.weight":
            out[block + "post_attention_layernorm.weight"] = value
        elif sub in ("transformer_block.attn_sconv.weight", "transformer_block.mlp_sconv.weight"):
            out[block + sub.split(".")[1] + ".conv.weight"] = _to_mlx_conv(value)
        elif sub.startswith("transformer_block.attn."):
            part = sub[len("transformer_block.attn.") :]
            head = part.split(".", 1)[0]
            if head in _ATTENTION_PARTS and leaf in _LEAVES:
                fused.setdefault((index, leaf), {})[head] = value
            elif head == "wo_ud" and leaf in _LEAVES:
                out[attn + "o_proj." + leaf] = value
            elif part in ("q_norm.weight", "k_norm.weight"):
                out[attn + part] = value
            elif part in ("k_sconv.weight", "v_sconv.weight"):
                out[attn + part.split(".")[0] + ".conv.weight"] = _to_mlx_conv(value)
            elif part == "rel_logits_proj.proj":
                out[attn + "rel_proj"] = value
            else:
                raise ValueError(f"unexpected Inkling MTP attention tensor: {name}")
        elif sub.startswith("transformer_block.mlp.w13_dn.") and leaf in _LEAVES:
            gate, up = _split_gate_up(value)
            out[block + "mlp.gate_proj." + leaf] = gate
            out[block + "mlp.up_proj." + leaf] = up
        elif sub.startswith("transformer_block.mlp.w2_md.") and leaf in _LEAVES:
            out[block + "mlp.down_proj." + leaf] = value
        elif sub == "transformer_block.mlp.global_scale":
            out[block + "mlp.global_scale"] = value
        else:
            raise ValueError(f"unexpected Inkling MTP sidecar tensor: {name}")
    for (index, leaf), parts in fused.items():
        if set(parts) != set(_ATTENTION_PARTS):
            raise ValueError(f"partial q/k/v/r {leaf} set at MTP depth {index}")
        out[f"layers.{index}.block.self_attn.qkvr_proj.{leaf}"] = mx.concatenate(
            [parts[head] for head in _ATTENTION_PARTS], axis=0
        )
    return out


def build_mtp(config, weights, *, depths, local_layer_ids, quantization):
    """Construct, quantize and strictly load an ``InklingMTP``.

    ``weights`` are raw sidecar tensors; ``quantization`` is the sidecar's
    ``{"bits", "group_size", "mode"}``.  Exactly the modules with packed
    ``.scales`` are quantized; everything else stays at its stored dtype.
    """
    mapped = sanitize_sidecar(weights, depths)
    head = InklingMTP(config, depths=depths, local_layer_ids=local_layer_ids)
    nn.quantize(
        head,
        group_size=int(quantization["group_size"]),
        bits=int(quantization["bits"]),
        mode=quantization.get("mode", "affine"),
        class_predicate=lambda name, _module: f"{name}.scales" in mapped,
    )
    head.load_weights(list(mapped.items()), strict=True)
    head.eval()
    return head


__all__ = [
    "MAX_DEPTHS",
    "InklingMTP",
    "InklingMTPAttention",
    "InklingMTPDepth",
    "build_mtp",
    "depth_config",
    "sanitize_sidecar",
]
