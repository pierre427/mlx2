# SPDX-License-Identifier: MIT
"""Kimi Linear NoPE MLA/KDA; provenance/kimi-linear-reap.json.

Reuses Ling's hybrid traversal, convolution, caches and weight stacking.
The checkpoint's normalization, unbounded gate and NoPE math stay explicit.
Performance candidates are per-model policies, never scheduler model branches.
"""
from collections import Counter
from dataclasses import dataclass, replace
import os

import mlx.core as mx
from mlx import nn

from ..segmented_plain_kv import SegmentedBatchKVCache
from . import bailing_hybrid as ling
from . import moe_nax_gather
from .base import scaled_dot_product_attention
from .cache import BatchKVCache, KVCache
from .gated_delta import gated_delta_kernel, gated_delta_ops


@dataclass
class ModelArgs(ling.ModelArgs):
    linear_attn_config: dict | None = None
    mla_use_nope: bool = True
    moe_layer_freq: int = 1

    @classmethod
    def from_dict(cls, config):
        linear = config["linear_attn_config"]
        return super().from_dict({
            **config,
            "num_experts_per_tok": config["num_experts_per_token"],
            "n_group": config["num_expert_group"],
            "layer_group_size": 4,
            "max_position_embeddings": config["model_max_length"],
            "norm_topk_prob": config["moe_renormalize"],
            "score_function": config["moe_router_activation_func"],
            "short_conv_kernel_size": linear["short_conv_kernel_size"],
        })


@dataclass(frozen=True)
class LeftPaddedCausal:
    """A left-padded BatchKVCache's own decode/verify mask with its lane starts.

    ``array`` is exactly ``create_causal_mask(N, offset=_idx,
    left_padding=starts)``: key k is visible to lane b, row i iff
    ``starts[b] <= k <= _idx + i``. Latent flash reads ``starts``; every
    other law receives ``array`` unchanged.
    """

    array: mx.array
    starts: mx.array


@dataclass(frozen=True)
class SegmentedCausal:
    """A plain SegmentedBatchKVCache's own unrestricted mask (no window).

    ``row_views`` slices each lane's mask out of ``array``; built by
    ``LanguageModel.attention_mask``, that slice is exactly bottom-right
    causal over the lane's keys, so MLA may attend with "causal".
    """

    array: mx.array


def l2_qk(q, k):
    # FLA normalizes in fp32, epsilon on SUM, with no bf16 round before scan.
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    return (q * mx.rsqrt(mx.sum(q * q, -1, keepdims=True) + 1e-6) * q.shape[-1]**-0.5,
            k * mx.rsqrt(mx.sum(k * k, -1, keepdims=True) + 1e-6))


class RMSNorm(nn.RMSNorm):
    """``law="reference"`` is KimiRMSNorm's two roundings: FP32 normalize,
    cast to the activation dtype, then the weight product in that dtype.
    ``"fused"`` (MLX convention) rounds once after the weight product."""

    law = "fused"

    def __call__(self, x):
        if self.law == "reference":
            return self.weight * mx.fast.rms_norm(x, None, self.eps)
        return super().__call__(x)


def dense_weight(linear, dtype=None):
    """A Linear's dense weight; affine-quantized banks dequantize once.

    The absorbed MLA law and the FP32 router need the matrix itself, not a
    quantized product. The copy is cached beside the module, never loaded
    as a parameter, and dropped if the weights are rebound.
    """
    if not isinstance(linear, nn.QuantizedLinear):
        return linear.weight if dtype is None else linear.weight.astype(dtype)
    # Bound to every input of the dequantization, not the packed bank alone:
    # rebinding scales/biases or the quantization law also drops the copy.
    biases = getattr(linear, "biases", None)
    binding = (id(linear.weight), id(linear.scales), None if biases is None else id(biases),
               linear.group_size, linear.bits, getattr(linear, "mode", "affine"),
               bool(getattr(linear, "_lane_tiled", False)), getattr(linear, "_lane_nt", 32))
    key = (binding, str(dtype))
    cache = linear.__dict__.setdefault("_kimi_dense", {})
    if not isinstance(cache, dict):  # an older single-entry cache
        cache = linear.__dict__["_kimi_dense"] = {}
    if cache.get("key") != key[0]:
        cache.clear()
        cache["key"] = key[0]
    if key[1] not in cache:
        weight = linear.weight
        if getattr(linear, "_lane_tiled", False):
            # Raw TensorFold (its lane_qmm installer) re-tiles the packed bank in
            # place for its kernel; dequantizing that layout as MLX's row
            # layout scrambled the absorbed kv_b and the router (2026-11-02,
            # scripts/research/kimi_tensorfold_lane_repro.py).
            from ..tensorfold_qwen38.lane_qmm import untile_weight

            weight = untile_weight(weight, getattr(linear, "_lane_nt", 32), linear.group_size,
                                   bits=linear.bits)
        dense = mx.dequantize(weight, linear.scales, linear.biases,
                              group_size=linear.group_size, bits=linear.bits, mode=linear.mode)
        cache[key[1]] = dense if dtype is None else dense.astype(dtype)
    return cache[key[1]]


@mx.compile
def kda_gates(a, beta, a_log, dt_bias):
    shifted = a.astype(mx.float32) + dt_bias.astype(mx.float32)
    # Stable softplus; checkpoint FLA gate returns log-decay in fp32.
    softplus = mx.maximum(shifted, 0) + mx.log1p(mx.exp(-mx.abs(shifted)))
    # Log decay stays finite where exp() of it underflows to zero.
    return (-mx.exp(a_log.astype(mx.float32)) * softplus,
            mx.sigmoid(beta.astype(mx.float32)))


class KimiDeltaAttention(ling.KimiDeltaAttention):
    def __init__(self, args):
        linear = args.linear_attn_config
        super().__init__(replace(args, num_attention_heads=linear["num_heads"],
                                 head_dim=linear["head_dim"]))
        self.calls = Counter()
        self.reference = False
        self.fused_projections = False
        self.fused_norm = False
        self.chunk32 = False
        self.chunk_wy = False
        self.norm_rounding = "fla_fp32"

    def prepare_projections(self):
        # SGLang's qkv/beta/f_a/g_a GEMM and batched f_b/g_b mechanism.
        # Views are installed only after strict loading; never replace weights.
        names = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")
        object.__setattr__(self, "_joined", mx.concatenate([self[n].weight for n in names]))
        object.__setattr__(self, "_second", mx.stack([self.f_b_proj.weight, self.g_b_proj.weight]))
        object.__setattr__(self, "_conv", mx.concatenate([
            self[n].conv.weight for n in ("q_conv1d", "k_conv1d", "v_conv1d")]))

    def __call__(self, x, mask=None, cache=None, *, tree=None):
        if tree is not None:
            # Parent-indexed tree math (kimi_tree.KimiTree) reuses this
            # module's projections and reproduces the ordinary arithmetic.
            kda = getattr(tree, "kimi_kda", None)
            if kda is None:
                raise ValueError("Kimi KDA requires the Kimi tree contract")
            return kda(self, x, cache)
        b, t, _ = x.shape
        shape_kind = "multirow" if t > 1 else "single_row"
        self.calls[shape_kind+"_calls"] += 1
        self.calls[shape_kind+"_tokens"] += b*t
        h, d = self.num_heads, self.head_dim
        p = h * d
        q, k, v, beta, a, gate = self.project(x)
        states = [None] * 4 if cache is None else list(cache)
        lengths = None if cache is None else cache.lengths
        # Exact speculative rollback (see _record_rollback): only staged when
        # the cache is speculating and its row geometry is describable.
        spans = None
        if cache is not None and getattr(cache, "speculating", False):
            describe = getattr(cache, "rollback_spans", None)
            spans = describe(t, mask) if describe is not None else None
            if spans is None:
                self.calls["rollback_unstaged"] += 1
        snapshot = list(states) if spans is not None else None
        raw = (q, k, v) if spans is not None else None
        if self.fused_projections and mask is None and lengths is None:
            # GLM/vLLM's merged depthwise convolution: each channel remains
            # independent, while the three launches become one.
            history = (mx.zeros((b,self.conv_kernel-1,3*p),dtype=x.dtype) if states[0] is None
                       else mx.concatenate(states[:3],axis=-1))
            conv_input = mx.concatenate([history,mx.concatenate([q,k,v],axis=-1)],axis=1)
            q,k,v = mx.split(nn.silu(mx.conv1d(conv_input,self._conv,groups=3*p)),3,axis=-1)
            states[:3] = mx.split(mx.contiguous(conv_input[:,-(self.conv_kernel-1):]),3,axis=-1)
            self.calls["merged_conv"] += 1
        else:
            q, states[0] = self.q_conv1d(q, states[0], mask, lengths)
            k, states[1] = self.k_conv1d(k, states[1], mask, lengths)
            v, states[2] = self.v_conv1d(v, states[2], mask, lengths)
        q, k, v = [z.reshape(b, t, h, d) for z in (q, k, v)]
        q, k = l2_qk(q, k)
        log_decay, beta = kda_gates(a.reshape(b, t, h, d), beta,
                                    self.A_log.reshape(h, 1), self.dt_bias.reshape(h, d))
        decay = mx.exp(log_decay)
        state = states[3]
        if state is None:
            state = mx.zeros((b, h, d, d), dtype=mx.float32)
        # Widen v: the existing vector Metal kernel shares one operand dtype.
        v = v.astype(mx.float32)
        if spans is not None:
            self._record_rollback(cache, t, spans, snapshot, raw, mask,
                                  (q, k, v, decay, beta), state, b)
        chunked = t >= 17 and mask is None and not getattr(cache, "speculating", False)
        if self.chunk_wy and chunked:
            from .kda_chunk_wy import chunk_wy
            out, state = chunk_wy(q, k, v, log_decay, beta, state)
            self.calls["chunk_wy"] += 1
        elif self.chunk32 and chunked:
            from .gdn_chunk32 import chunk32
            out, state = chunk32(q, k, v, decay, beta, state)
            self.calls["chunk32"] += 1
        elif not self.reference and mx.default_device() == mx.gpu:
            out, state = gated_delta_kernel(q, k, v, decay, beta, state, mask)
            self.calls["vector_kernel"] += 1
        else:
            out, state = gated_delta_ops(q, k, v, decay, beta, state, mask)
            self.calls["reference"] += 1
        if cache is not None:
            for i in range(3):
                cache[i] = states[i]
            cache[3] = state
            cache.advance(t)
        return self.o_proj(self.norm_gate(out, gate, x.dtype).reshape(b, t, p))

    _PROJECTIONS = ("q_proj", "k_proj", "v_proj", "b_proj",
                    "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj")

    def _projections_unwrapped(self):
        # The joined views hold the loaded weights only; a served LoRA wraps
        # the modules, so its delta exists only on the per-module path.
        if all(type(self[n]) is nn.Linear for n in self._PROJECTIONS):
            return True
        self.calls["fused_projections_declined_wrapped"] += 1
        return False

    def project(self, x):
        """Row-wise q/k/v/beta/decay-logit/gate projections (pre-convolution)."""
        h, d = self.num_heads, self.head_dim
        p = h * d
        if self.fused_projections and self._projections_unwrapped():
            q, k, v, beta, fa, ga = mx.split(x @ self._joined.T,
                                            [p, 2*p, 3*p, 3*p+h, 3*p+h+d], -1)
            fg = mx.stack([fa, ga], axis=0) @ self._second[:, None].swapaxes(-1, -2)
            self.calls["fused_projections"] += 1
            return q, k, v, beta, fg[0], fg[1]
        return (self.q_proj(x), self.k_proj(x), self.v_proj(x), self.b_proj(x),
                self.f_b_proj(self.f_a_proj(x)), self.g_b_proj(self.g_a_proj(x)))

    def norm_gate(self, out, gate, dtype):
        """FLA gated output norm on [B, T, H, D] recurrence output."""
        h, d = self.num_heads, self.head_dim
        b, t = out.shape[:2]
        # FLA writes recurrence output in the value dtype before gated norm.
        out = out.astype(dtype)
        gate = gate.reshape(b, t, h, d)
        if self.fused_norm and mx.default_device() == mx.gpu and out.dtype == mx.bfloat16:
            if self.norm_rounding == "qwen_bf16":
                from .qwen4_fused_gdn_prefill import qwen4_gdn_prefill_norm_gate
                out = qwen4_gdn_prefill_norm_gate(out,gate,self.o_norm.weight,self.o_norm.eps,
                                                 num_value_heads=h,value_head_dim=d)
            else:
                from .kimi_linear_fused import norm_gate
                out = norm_gate(out, gate, self.o_norm.weight, self.o_norm.eps)
            self.calls["fused_norm"] += 1
        else:
            # FLA RMSNormGated widens both norm and sigmoid before final cast.
            if self.norm_rounding == "qwen_bf16":
                normed = mx.fast.rms_norm(out,self.o_norm.weight,self.o_norm.eps).astype(mx.float32)
            else:
                normed = mx.fast.rms_norm(out.astype(mx.float32),self.o_norm.weight.astype(mx.float32),
                                         self.o_norm.eps)
            out = (normed * mx.sigmoid(gate.astype(mx.float32))).astype(dtype)
        self.calls["norm_"+self.norm_rounding] += 1
        return out

    def _recurrence(self, q, k, v, decay, beta, state, mask):
        """The ordinary one-token-law recurrence (never a chunked candidate)."""
        if not self.reference and mx.default_device() == mx.gpu:
            return gated_delta_kernel(q, k, v, decay, beta, state, mask)
        return gated_delta_ops(q, k, v, decay, beta, state, mask)

    def _record_rollback(self, cache, t, spans, snapshot, raw, mask, scan, state, b):
        """Stage an exact ``ArraysCache`` rollback for this forward.

        Convolution histories are closed form: the state after the first
        ``m`` tokens is rows ``m:m+K-1`` of ``[history, masked raw input]``.
        The FP32 recurrent matrix is replayed from the pre-forward state over
        the stashed per-token q/k/v/decay/beta with the same one-token law
        (vector kernel or ops reference) the forward itself uses, so a
        rewound state equals the state the forward held after ``m`` tokens.
        The snapshot is the pre-forward cache list (``m == 0``).
        """
        keep = self.conv_kernel-1
        p = self.num_heads*self.head_dim
        padded = []
        for index, value in enumerate(raw):
            if mask is not None:
                value = mx.where(mask[..., None], value, 0)
            history = snapshot[index]
            if history is None:
                history = mx.zeros((b, keep, p), dtype=value.dtype)
            padded.append(mx.concatenate([history, value], axis=1))
        q, k, v, decay, beta = scan

        def rollback(m, padded=padded, scan=(q, k, v, decay, beta), initial=state, mask=mask):
            m = int(m)
            self.calls["rollback_replays"] += 1
            convs = [mx.contiguous(item[:, m:m+keep]) for item in padded]
            if m == 0:
                return [*convs, initial]
            rows = [item[:, :m] for item in scan]
            _, replayed = self._recurrence(*rows, initial, None if mask is None else mask[:, :m])
            return [*convs, replayed]

        cache.record_rollback(t, rollback, snapshot, depths=spans or None)
        self.calls["rollback_records"] += 1


class MultiLatentAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        h, n, r, v, c = (args.num_attention_heads, args.qk_nope_head_dim,
                          args.qk_rope_head_dim, args.v_head_dim, args.kv_lora_rank)
        self.q_proj = nn.Linear(args.hidden_size, h * (n+r), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(args.hidden_size, c+r, bias=False)
        # Released KimiMLAAttention constructs this norm at its default 1e-6.
        self.kv_a_layernorm = RMSNorm(c, eps=1e-6)
        self.kv_b_proj = nn.Linear(c, h * (n+v), bias=False)
        self.o_proj = nn.Linear(h * v, args.hidden_size, bias=False)
        self.scale = (n+r)**-0.5
        self.policy = "expanded"
        # Opt-in training law; the session restores this separately from the
        # ordinary serving policy. See provenance/kimi-mla-training-padded.json.
        self.training_mla_policy = "expanded"
        self.calls = Counter()
        self.key_tile = 2048
        # latent_single bound on the FP32 score block; larger shapes tile.
        self.single_score_bytes = 256 << 20
        # auto: prompt chunks wider than auto_rows use expanded SDPA while
        # its transient expanded prefix stays under expanded_bytes.
        self.auto_rows = 16
        self.expanded_bytes = 4 << 30
        # auto sub-routes on the NAX flash kernels (kimi_linear_mla_flash,
        # unqualified): "off", "expanded" (qk192/v128 prompt chunks, before
        # padded D256) or "all" (also absorbed latent rows). Each declines
        # (counter flash_declined) and falls back when not admissible.
        self.flash = "off"
        # With flash="all", chunks of at most flash_rows rows stay on the
        # latent flash: it beat flash192 + kv_b re-expansion through 128 rows
        # and lost at 512 at 8K/32K/128K (real-weight layer A/B, 2026-11-02).
        self.flash_rows = 128

    def __call__(self, x, mask=None, cache=None, *, tree=None):
        if self.training_mla_policy != "expanded":
            self._validate_training_padded(mask, cache=cache, tree=tree)
        if tree is not None:
            mla = getattr(tree, "kimi_mla", None)
            if mla is None:
                raise ValueError("Kimi MLA requires the Kimi tree contract")
            return mla(self, x, cache)
        b, t, _ = x.shape
        h, v = self.args.num_attention_heads, self.args.v_head_dim
        starts, causal_rows = None, False
        if isinstance(mask, LeftPaddedCausal):
            mask, starts = mask.array, mask.starts
        elif isinstance(mask, SegmentedCausal):
            mask, causal_rows = mask.array, True
        shape_kind = "multirow" if t > 1 else "single_row"
        self.calls[shape_kind+"_calls"] += 1
        self.calls[shape_kind+"_tokens"] += b*t
        q, latent, shared, q_rows = self._project_rows(x)
        if cache is not None and hasattr(cache, "row_views"):
            # Segmented verify batch (multi-lane PLD/external rounds): history
            # stays in each lane's own KV cache; every row attends its own
            # latent/shared rows through the same policy law.
            cache.update_and_fetch(latent, shared)
            # A plain segmented batch's own mask (or None) gives each lane
            # exactly bottom-right causal over its keys (create_causal_mask(
            # valid, offset=base), or the prepared mask with its padding
            # sliced off); "causal" is the same law and admits the latent
            # flash. Any other explicit mask (a window) is kept as is.
            causal_rows = (type(cache) is SegmentedBatchKVCache
                           and (causal_rows or mask is None))
            pieces = []
            for index, valid, row_latent, row_shared, row_mask in cache.row_views(mask):
                if not valid:
                    pieces.append(mx.zeros((1, h, t, v), dtype=x.dtype))
                    continue
                piece = self.attend(q[index:index+1, :, :valid], row_latent, row_shared,
                                    "causal" if causal_rows else row_mask, x.itemsize)
                if valid < t:
                    piece = mx.pad(piece, [(0, 0), (0, 0), (0, t-valid), (0, 0)])
                pieces.append(piece)
            cache.note_attention()
            self.calls["segmented_rows"] += b
            out = mx.concatenate(pieces, axis=0)
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, t, h*v))
        if cache is not None:
            latent, shared = cache.update_and_fetch(latent, shared)
        # The flash kernels can read a plain or batch KVCache's buffers in
        # place: a B > 1 prefix slice is not row contiguous and would be
        # copied whole.
        filled = (cache.offset if type(cache) is KVCache
                  else cache._idx if type(cache) is BatchKVCache else None)
        buffers = ((cache.keys, cache.values) if filled is not None and b > 1
                   and filled == latent.shape[2] < cache.keys.shape[2] else None)
        out = self.attend(q, latent, shared, mask, x.itemsize, q_rows=q_rows, buffers=buffers,
                          starts=starts)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(b, t, h*v))

    def project(self, x):
        """Queries [B,H,T,N+R], normalized latent [B,1,T,C], shared key [B,1,T,R]."""
        return self._project_rows(x)[:3]

    def _project_rows(self, x):
        """``project`` plus the row-major query projection [B,T,H*(N+R)].

        The row-major projection feeds the flash kernels without a copy; it
        is returned, never kept on the module, so no forward's [B,T,H*192]
        query block outlives that forward.
        """
        a = self.args
        b, t, _ = x.shape
        h, n, r, c = a.num_attention_heads, a.qk_nope_head_dim, a.qk_rope_head_dim, a.kv_lora_rank
        q_rows = self.q_proj(x)
        q = q_rows.reshape(b, t, h, n+r).transpose(0, 2, 1, 3)
        latent, shared = mx.split(self.kv_a_proj_with_mqa(x), [c], -1)
        # "rope" names in the config describe a shared key slice. No rotation.
        return q, self.kv_a_layernorm(latent)[:, None], shared[:, None], q_rows

    def attend(self, q, latent, shared, mask, itemsize, *, q_rows=None, buffers=None, starts=None):
        """The policy's attention law over fetched latent/shared keys -> [B,H,T,V].

        ``starts`` (with an array ``mask``) marks a left-padded batch cache:
        only the latent flash consumes it; other laws read ``mask``.
        """
        a = self.args
        b, _, t, _ = q.shape
        h, n, r, v = a.num_attention_heads, a.qk_nope_head_dim, a.qk_rope_head_dim, a.v_head_dim
        if self.training_mla_policy != "expanded":
            self._validate_training_padded(mask, cache=buffers, tree=starts)
            if ((n, r, v) != (128, 64, 128) or q.shape != (b, h, t, 192)
                    or t <= 8 or b < 1 or h < 1
                    or latent.shape != (b, 1, t, a.kv_lora_rank)
                    or shared.shape != (b, 1, t, 64)
                    or q.dtype not in (mx.float32, mx.float16, mx.bfloat16)):
                raise ValueError("padded training MLA requires full self-attention qk192/v128, T>8")
            kv = self.kv_b_proj(latent[:, 0]).reshape(b, t, h, n+v).transpose(0, 2, 1, 3)
            keys, values = mx.split(kv, [n], -1)
            keys = mx.concatenate([keys, mx.broadcast_to(shared, (b, h, t, r))], -1)
            # Keep the original 192-dimensional dot product and scale. Only
            # zero-extend V, then discard its identically zero output channels.
            values = mx.pad(values, [(0, 0), (0, 0), (0, 0), (0, 64)])
            out = mx.fast.scaled_dot_product_attention(
                q, keys, values, scale=self.scale, mask=mask, force_fused=True)[..., :v]
            self.calls["training_padded_sdpa"] += 1
            self.calls["expanded_sdpa"] += 1
            return out
        if q_rows is None or q_rows.shape[:2] != (b, t):
            q_rows = q.transpose(0, 2, 1, 3).reshape(b, t, h*(n+r))
        policy = self.policy
        flash = None
        if policy == "auto":
            # DeepSeek/Ling phase split: expanded K/V for prompt chunks while
            # the expanded prefix fits its bound, one latent per token for
            # decode/verify rows and for longer prefixes.
            width = latent.shape[2]
            wide = t > self.auto_rows
            if (wide and self.flash == "all" and t <= self.flash_rows
                    and self._flash_admits("absorbed", q_rows, mask, starts)):
                wide = False
            if wide and self._flash_admits("expanded", q_rows, mask):
                # Flash192 reads the kv_b_proj output in place: no padded,
                # transposed or concatenated K/V copies, no score block.
                # Over the bound it streams the prefix in key windows.
                flash, transient = "expanded", b*width*h*(n+v)*itemsize
                if transient > self.expanded_bytes:
                    flash, transient = "windows", 0
            elif self._fused_d256(q, mask):
                # Fused D256 holds padded K/V only; the unfused fallback also
                # materializes the FP32 score block, which dominates at depth.
                transient = b*width*h*512*itemsize
            else:
                transient = b*width*h*((n+r+v)*itemsize + 4*t)
            expanded = wide and transient <= self.expanded_bytes
            policy = "expanded" if expanded else "latent_single"
            if not expanded:
                flash = "absorbed" if self._flash_admits("absorbed", q_rows, mask, starts) else None
            self.calls["auto_expanded" if expanded else "auto_latent"] += 1
        if policy in {"latent_tiled", "latent_fused", "latent_single"}:
            out = self._latent(q, latent, shared, mask, "latent_flash" if flash else policy, buffers,
                               starts=starts if flash else None)
        elif flash == "expanded":
            if type(self.kv_b_proj) is nn.Linear:
                # Head-major [B, H, S, 256] from the same product: each head
                # then reads one contiguous block (measured faster).
                w = self.kv_b_proj.weight.reshape(h, n+v, a.kv_lora_rank)
                kv = latent @ w.swapaxes(-1, -2)
            else:
                kv = self.kv_b_proj(latent[:, 0])
            from .kimi_linear_mla_flash import expanded as flash_expanded
            out = flash_expanded(q_rows, kv, shared if buffers is None else buffers[1], mask,
                                 scale=self.scale, heads=h)
            self.calls["expanded_flash192"] += 1
            self.calls["expanded_sdpa"] += 1
            return out.reshape(b, t, h, v).transpose(0, 2, 1, 3)
        elif flash == "windows":
            from .kimi_linear_mla_flash import expanded_windows
            # Each window's re-expansion stays under the expanded bound.
            window = max(1024, self.expanded_bytes//(b*h*(n+v)*itemsize)//1024*1024)
            out = expanded_windows(q_rows, latent, shared, self.kv_b_proj, mask,
                                   scale=self.scale, window=window, heads=h)
            self.calls["expanded_flash192_windows"] += 1
            self.calls["expanded_sdpa"] += 1
            return out.reshape(b, t, h, v).transpose(0, 2, 1, 3)
        else:
            kv = self.kv_b_proj(latent[:, 0]).reshape(b, -1, h, n+v).transpose(0, 2, 1, 3)
            keys, values = mx.split(kv, [n], -1)
            shared = mx.broadcast_to(shared, (b, h, shared.shape[2], r))
            if self.policy == "auto" and self._fused_d256(q, mask):
                # MLX fuses no 192/128 prefill; the NAX D256 kernel is exact
                # for zero-padded columns and never materializes the scores.
                def pad(z, d):
                    return mx.concatenate([z, mx.zeros((*z.shape[:-1], d), z.dtype)], -1)
                keys = mx.concatenate([keys, shared, mx.zeros((*keys.shape[:-1], 256-n-r), keys.dtype)], -1)
                out = mx.fast.scaled_dot_product_attention(
                    pad(q, 256-n-r), keys, pad(values, 256-v), scale=self.scale, mask=mask,
                    force_fused=True)[..., :v]
                self.calls["expanded_fused_d256"] += 1
            else:
                keys = mx.concatenate([keys, shared], -1)
                out = scaled_dot_product_attention(q, keys, values, cache=None, scale=self.scale, mask=mask)
            self.calls["expanded_sdpa"] += 1
        return out

    def _validate_training_padded(self, mask, *, cache=None, tree=None):
        if self.training_mla_policy != "padded_sdpa":
            raise ValueError("Unknown training MLA policy")
        if not self.training or self.policy != "expanded" or self.flash != "off":
            raise ValueError("padded training MLA requires training mode and expanded/flash-off policy")
        if cache is not None or tree is not None:
            raise ValueError("padded training MLA does not support cache, tree, or segmented state")
        if mask is not None and (not isinstance(mask, str) or mask != "causal"):
            raise ValueError("padded training MLA supports only no mask or causal masking")
        if os.environ.get("MLX_SDPA_VJP_FALLBACK", "0").strip() not in {"", "0"}:
            raise ValueError("padded training MLA requires MLX_SDPA_VJP_FALLBACK=0")

    _d256_available = None  # per process: probed once on first use

    def _flash_admits(self, kind, q_rows, mask, starts=None):
        """auto's flash sub-route gate; a decline is counted, never raised.

        A left-padded batch cache's mask (``starts`` given) is admitted by the
        absorbed kernel, which hides each lane's leading keys in-kernel.
        """
        if self.flash == "off" or (kind == "absorbed" and self.flash != "all"):
            return False
        a = self.args
        from . import kimi_linear_mla_flash as fl
        padded = starts is not None and kind == "absorbed"
        ok = ((mask is None or (isinstance(mask, str) and mask == "causal")
               or (padded and fl.available("absorbed_padded", q_rows.dtype)))
              and q_rows.dtype in (mx.bfloat16, mx.float16)
              and (a.num_attention_heads, a.qk_nope_head_dim, a.qk_rope_head_dim,
                   a.v_head_dim, a.kv_lora_rank) == (fl.HEADS, fl.NOPE, fl.SHARED, fl.VDIM, fl.LATENT)
              and fl.available(kind, q_rows.dtype))
        if not ok:
            self.calls["flash_declined"] += 1
        return ok

    def _fused_d256(self, q, mask):
        a = self.args
        if (mx.default_device() != mx.gpu or q.dtype not in (mx.bfloat16, mx.float16)
                or mask != "causal" or a.qk_nope_head_dim + a.qk_rope_head_dim > 256
                or a.v_head_dim > 256):
            return False
        cls = MultiLatentAttention
        if cls._d256_available is None:
            probe = mx.zeros((1, 1, 32, 256), q.dtype)
            try:
                mx.eval(mx.fast.scaled_dot_product_attention(probe, probe, probe, scale=1.0,
                                                             mask="causal", force_fused=True))
                cls._d256_available = True
            except ValueError:
                # No fused D256 kernel here (CPU, a pre-NAX GPU).
                cls._d256_available = False
        return cls._d256_available

    def _latent(self, q, latent, shared, mask, policy, buffers=None, *, starts=None):
        if policy == "latent_flash":
            # Absorbed NAX flash over the latent cache (qk 576 / v 512):
            # bottom-right causal offset in-kernel, split partials combined.
            from .kimi_linear_mla_flash import absorbed
            a = self.args
            h, n = a.num_attention_heads, a.qk_nope_head_dim
            w = dense_weight(self.kv_b_proj).reshape(h, n+a.v_head_dim, a.kv_lora_rank)
            qn, qs = mx.split(q, [n], -1)
            if starts is not None:
                # The padded batch cache's mask is bottom-right causal after
                # each lane's start: the kernel's causal law plus starts.
                mask = "causal"
                self.calls["latent_flash_padded"] += 1
            if buffers is None:
                result = absorbed(qn @ w[:, :n], qs, latent, shared, mask, scale=self.scale, heads=h,
                                  starts=starts)
            else:
                result = absorbed(qn @ w[:, :n], qs, *buffers, mask, scale=self.scale, heads=h,
                                  width=latent.shape[2], starts=starts)
            self.calls["latent_flash"] += 1
            return result @ w[:, n:].swapaxes(-1, -2)
        # A string mask is bottom-right causal over [prefix + rows]. It is
        # built from positions for the block each pass reads, never whole:
        # a [rows, context] bool mask is 8 GiB at 8192 x 1M. Values equal
        # the explicit mask's slices, so results are bit-identical.
        causal = isinstance(mask, str)
        offset = latent.shape[2] - q.shape[2]

        def visible(r0, r1, k0, k1):
            return mx.arange(offset+r0, offset+r1)[:, None] >= mx.arange(k0, k1)[None, :]
        # GLM/DS phase split: absorb K projection, attend on one latent per
        # token, then unabsorb V. Bounded online softmax avoids O(context*rows)
        # score storage and full-prefix expanded K/V, including at 1M.
        a = self.args
        h, n, v = a.num_attention_heads, a.qk_nope_head_dim, a.v_head_dim
        w = dense_weight(self.kv_b_proj).reshape(h, n+v, a.kv_lora_rank)
        qn, qs = mx.split(q, [n], -1)
        qn = qn @ w[:, :n]
        if policy == "latent_fused":
            from .kimi_linear_mla_fused import admits, attend
            # Unmasked rows only: a causal string declines, as its
            # materialized array did.
            if admits(qn,qs,latent,shared,mask):
                self.calls["fused_mla_decode"] += 1
                result = attend(qn,qs,latent,shared).astype(q.dtype)
                return result @ w[:,n:].swapaxes(-1,-2)
            self.calls["fused_mla_declined"] += 1
        if policy == "latent_single" and q.size // q.shape[-1] * latent.shape[2] * 4 <= self.single_score_bytes:
            # One pass over the latent cache: the same FP32 score/softmax law
            # as one key tile, without O(context/key_tile) launches per row.
            score = ((qn @ latent.swapaxes(-1, -2)).astype(mx.float32)
                     + (qs @ shared.swapaxes(-1, -2)).astype(mx.float32)) * self.scale
            if causal:
                # [rows, width] bool: at most single_score_bytes / (4*B*H),
                # a quarter of one head's share of the bounded score block.
                score = mx.where(visible(0, q.shape[2], 0, latent.shape[2]), score, -float("inf"))
            elif mask is not None:
                score = mx.where(mask, score, -float("inf"))
            # A fully masked (padding) row yields zeros, as in the tiled law.
            top = score.max(-1, keepdims=True)
            prob = mx.exp(score - mx.where(mx.isfinite(top), top, 0))
            total = mx.maximum(prob.sum(-1, keepdims=True), 1e-30)
            self.calls["latent_single"] += 1
            return ((prob @ latent.astype(mx.float32)) / total).astype(q.dtype) @ w[:, n:].swapaxes(-1, -2)
        if policy == "latent_single":
            # Over the score bound: same law in key tiles, with its own receipt.
            self.calls["latent_single_tiled"] += 1
        else:
            # The tiled loop proper: latent_tiled, or latent_fused declined.
            self.calls["latent_tiled"] += 1
        pieces = []
        for start in range(0, q.shape[2], 64):
            end = min(start+64, q.shape[2])
            qn_tile, qs_tile = qn[:, :, start:end], qs[:, :, start:end]
            maximum = mx.full((*qn_tile.shape[:-1], 1), -float("inf"), dtype=mx.float32)
            total = mx.zeros_like(maximum)
            accum = mx.zeros((*qn_tile.shape[:-1], a.kv_lora_rank), dtype=mx.float32)
            for k0 in range(0, latent.shape[2], self.key_tile):
                k1 = min(k0+self.key_tile, latent.shape[2])
                score = ((qn_tile @ latent[:, :, k0:k1].swapaxes(-1, -2)).astype(mx.float32)
                         + (qs_tile @ shared[:, :, k0:k1].swapaxes(-1, -2)).astype(mx.float32)) * self.scale
                if causal:
                    score = mx.where(visible(start, end, k0, k1), score, -float("inf"))
                elif mask is not None:
                    score = mx.where(mask[..., start:end, k0:k1], score, -float("inf"))
                new_max = mx.maximum(maximum, score.max(-1, keepdims=True))
                safe_max = mx.where(mx.isfinite(new_max), new_max, 0)
                alpha = mx.exp(maximum-safe_max)
                prob = mx.exp(score-safe_max)
                accum = accum * alpha + prob @ latent[:, :, k0:k1].astype(mx.float32)
                total = total * alpha + prob.sum(-1, keepdims=True)
                maximum = new_max
            result = (accum / mx.maximum(total, 1e-30)).astype(q.dtype) @ w[:, n:].swapaxes(-1, -2)
            pieces.append(result)
        return mx.concatenate(pieces, axis=2)


class Gate(ling.Gate):
    # Opted in only by the bounded training helper. Module.training defaults
    # True at construction, so it is deliberately not the inference gate.
    training_indices = False

    def __init__(self, args):
        super().__init__(args)
        # Uncorrected combine weights, as Transformers v5.17, vLLM, SGLang and
        # mlx-lm gather them (Pierre, 2026-11-02, after the loopback A/B).
        # "checkpoint" keeps the hub modeling_kimi.py in-place alias law.
        self.combine = "unbiased"
        self.calls = Counter()

    def __call__(self, x):
        self.calls["router_calls"] += 1
        self.calls["router_tokens"] += x.size // x.shape[-1]
        # Source F.linear widens operands BEFORE GEMM, not just logits.
        logits = x.astype(mx.float32) @ dense_weight(self.gate_proj, mx.float32).T
        scores = mx.sigmoid(logits)
        choice = scores + self.expert_bias
        if self.n_group != 1:
            raise ValueError("staged Kimi router supports the checkpoint's single expert group")
        ids = mx.argpartition(-choice, kth=self.top_k-1, axis=-1)[..., :self.top_k]
        if self.training_indices:
            # Expert selection is discrete; combine scores remain differentiable.
            ids = mx.stop_gradient(ids)
        # modeling_kimi.py aliases scores.view() then += correction bias;
        # its gather therefore sees corrected scores. SGLang/vLLM use the
        # conventional uncorrected combine scores. Keep the discrepancy named.
        source = choice if self.combine == "checkpoint" else scores
        weights = mx.take_along_axis(source, ids, -1)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdims=True) + 1e-20)
        return ids, weights * self.routed_scaling_factor


class SparseMoeBlock(ling.SparseMoeBlock):
    def __init__(self, args):
        super().__init__(args)
        self.gate = Gate(args)

    def __call__(self, x):
        ids, weights = self.gate(x)
        expert = self.switch_mlp(x, ids)
        out = (expert.astype(mx.float32) * weights[..., None]).sum(-2).astype(x.dtype)
        return out + self.shared_experts(x) if self.shared_experts is not None else out


class DecoderLayer(ling.DecoderLayer):
    def __init__(self, args, index):
        nn.Module.__init__(self)
        self.is_global = index+1 in args.linear_attn_config["full_attn_layers"]
        self.attention = MultiLatentAttention(args) if self.is_global else KimiDeltaAttention(args)
        self.mlp = (SparseMoeBlock(args) if index >= args.first_k_dense_replace
                    and index % args.moe_layer_freq == 0 else ling.MLP(args))
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)


def _proves_unpadded(cache):
    """Host proof that a cache's decode mask is all-true.

    BatchKVCache records it when no row is left-padded (cache.py). The PLD
    runtime gives its request-private B1 KVCache the ordinary mask with
    padding mx.array([0]) by construction (pld._set_ordinary_b1_mask).
    """
    if type(cache).__name__ == "KVCache" and getattr(cache, "_pld_ordinary_mask_padding", None) is not None:
        return True
    ref = getattr(cache, "_unpadded_ref", None)
    return ref is not None and ref is getattr(cache, "left_padding", None)


class LanguageModel(ling.LanguageModel):
    @staticmethod
    def attention_mask(hidden, cache):
        """The MLA layers' mask (ordinary and tapped forwards build it alike)."""
        # Unpadded caches give the "causal" offset mode, so expanded prompt
        # chunks use fused causal SDPA instead of an O(context*rows) array.
        mask = ling.create_attention_mask(hidden, cache)
        if hidden.shape[1] == 1 and mask is not None and _proves_unpadded(cache):
            # A batch cache's decode mask is all-true when no lane is left-
            # padded; dropping it is exact and admits the latent flash kernel.
            return None
        if isinstance(mask, mx.array) and type(cache) is BatchKVCache:
            # Left-padded lanes: the mask is that cache's own
            # create_causal_mask(N, offset=_idx, left_padding); carry the lane
            # starts so latent flash can hide the padding in-kernel.
            return LeftPaddedCausal(mask, cache.left_padding)
        if isinstance(mask, mx.array) and type(cache) is SegmentedBatchKVCache:
            return SegmentedCausal(mask)
        return mask

    def __init__(self, args):
        nn.Module.__init__(self)
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._attention_idx = next(i for i, l in enumerate(self.layers) if l.is_global)
        self._kda_idx = next(i for i, l in enumerate(self.layers) if not l.is_global)

    def __call__(self, inputs, cache=None):
        hidden = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        attention_mask = self.attention_mask(hidden, cache[self._attention_idx])
        kda_mask = ling.create_ssm_mask(hidden, cache[self._kda_idx])
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(hidden, mask=attention_mask if layer.is_global else kda_mask, cache=layer_cache)
        return self.norm(hidden)


class Model(ling.Model):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        self.model_type = args.model_type
        self.model = LanguageModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None):
        from ..prefill_checkpoint_rows import select_checkpoint_rows

        with moe_nax_gather.forward_scope(inputs, cache):
            # Prompt chunks project only each lane's checkpoint row through
            # the 163,840-wide head (identity outside a prompt-chunk request).
            return self.lm_head(select_checkpoint_rows(self.model(inputs, cache)))

    # Draft/verify seams inherited from Ling (post-layer residual taps, as HF
    # ``hidden_states[layer + 1]``) run under the same expert phase scope as
    # the ordinary forward.
    # The masks follow LanguageModel.__call__ (causal offset mode for unpadded
    # caches), so external prefill and verify rows see the ordinary mask law.
    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False,
                          last_logits_only=False):
        capture_layers = tuple(int(value) for value in capture_layers)
        if (not capture_layers or tuple(sorted(set(capture_layers))) != capture_layers
                or capture_layers[0] < 0 or capture_layers[-1] >= len(self.layers)):
            raise ValueError("invalid Kimi target capture layers")
        if body_only and last_logits_only:
            raise ValueError("body_only and last_logits_only are mutually exclusive")
        with moe_nax_gather.forward_scope(inputs, cache):
            inner = self.model
            hidden = inner.word_embeddings(inputs)
            cache = [None] * len(self.layers) if cache is None else cache
            attention_mask = inner.attention_mask(hidden, cache[inner._attention_idx])
            kda_mask = ling.create_ssm_mask(hidden, cache[inner._kda_idx])
            wanted, taps = frozenset(capture_layers), []
            for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
                hidden = layer(hidden, mask=attention_mask if layer.is_global else kda_mask,
                               cache=layer_cache)
                if index in wanted:
                    taps.append(hidden)
            features = mx.concatenate(taps, axis=-1)
            if body_only:
                return None, features
            hidden = inner.norm(hidden)
            return self.lm_head(hidden[:, -1:] if last_logits_only else hidden), features

    # Tree forwards stay outside the prefill phase scope (stock expert law).

    def sanitize(self, weights):
        renamed = {}
        for key, value in weights.items():
            key = key.replace("model.embed_tokens.", "model.word_embeddings.")
            key = key.replace(".self_attn.", ".attention.").replace(".block_sparse_moe.", ".mlp.")
            key = key.replace(".gate.e_score_correction_bias", ".gate.expert_bias")
            # mlx-lm conversions: router bias beside the gate, short convs
            # named q_conv/k_conv/v_conv, quantized router under mlp.gate.
            key = key.replace(".mlp.e_score_correction_bias", ".mlp.gate.expert_bias")
            for name in ("q", "k", "v"):
                key = key.replace(f".attention.{name}_conv.conv.", f".attention.{name}_conv1d.conv.")
            for part in ("weight", "scales", "biases"):
                if key.endswith(f".mlp.gate.{part}"):
                    key = key[:-len(part)] + "gate_proj." + part
            for old, new in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
                if ".experts." in key:
                    key = key.replace(f".{old}.", f".{new}.")
            if key.endswith(".A_log"):
                value = value.reshape(-1)
            renamed[key] = value
        # Drop the caller's references so each per-expert source frees as soon
        # as its stacked table is materialised below.  Stacking every layer
        # lazily and evaluating once held the materialised per-expert sources
        # and the stacked tables together (~2x the 67 GB of experts) and
        # swapped 6-10 GB on the 128 GB M5 Max at every load (2026-11-02).
        # A stack is a copy: values are unchanged.
        weights.clear()
        # Ling's MLA transforms are intentionally bypassed: Kimi retains the
        # original kv_b projection for its ordinary-reference arithmetic.
        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}"
            if isinstance(layer.mlp, SparseMoeBlock):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    key = f"{prefix}.mlp.experts.0.{name}.weight"
                    if key in renamed:
                        stacked = mx.stack([
                            renamed.pop(f"{prefix}.mlp.experts.{e}.{name}.weight")
                            for e in range(self.args.num_experts)])
                        mx.eval(stacked)
                        renamed[f"{prefix}.mlp.switch_mlp.{name}.weight"] = stacked
            if not layer.is_global:
                for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    key = f"{prefix}.attention.{name}.weight"
                    if key in renamed:
                        # HF depthwise [C, 1, K] -> MLX [C, K, 1]; mlx-lm
                        # conversions already store the MLX layout.
                        renamed[f"{prefix}.attention.{name}.conv.weight"] = renamed.pop(key).moveaxis(2, 1)
        return renamed

    def configure(self, policy):
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.law = policy["rms_law"]
        for layer in self.layers:
            attn = layer.attention
            if layer.is_global:
                attn.policy = policy["mla"]
                attn.flash = policy["mla_flash"]
            else:
                attn.reference = policy["kda"] == "reference"
                attn.chunk32 = policy["kda"] == "chunk32"
                attn.chunk_wy = policy["kda"] == "chunk_wy"
                attn.fused_projections = policy["fused_projections"]
                attn.fused_norm = policy["fused_norm"]
                attn.norm_rounding = policy["norm_rounding"]
                if attn.fused_projections:
                    attn.prepare_projections()
                    mx.eval(attn._joined, attn._second, attn._conv)
            if isinstance(layer.mlp, SparseMoeBlock):
                layer.mlp.gate.combine = policy["router_combine"]

    def route_status(self):
        counters = Counter()
        for layer in self.layers:
            counters.update(layer.attention.calls)
        return dict(counters)

    def moe_status(self):
        """Router calls and SwitchGLU sort decisions summed over MoE layers."""
        router, sort = Counter(), Counter()
        for layer in self.layers:
            if isinstance(layer.mlp, SparseMoeBlock):
                router.update(layer.mlp.gate.calls)
                sort.update(layer.mlp.switch_mlp.sort_status()["counts"])
        return {"router_calls": router["router_calls"], "router_tokens": router["router_tokens"],
                "sort": dict(sort)}

    def mechanism_status(self):
        kda,mla = Counter(),Counter()
        for layer in self.layers:
            (mla if layer.is_global else kda).update(layer.attention.calls)
        return {"kda":dict(kda),"mla":dict(mla)}
