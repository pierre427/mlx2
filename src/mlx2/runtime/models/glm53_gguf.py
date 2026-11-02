"""Adapter-owned GLM-5.3 ordinary text forward over exact GGUF blocks.

See provenance/glm53-gguf-forward.json. The checkpoint's positive exp(A_log)
is consumed directly, without a log/exp round trip or affine requantization.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
from mlx import nn

from ..cache_operations import CacheOperationsContract, CachePlaneDeclaration
from ..prefill_checkpoint_rows import select_checkpoint_rows
from .cache import CacheList, KVCache, register_cache_class
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .glm53_gguf_indexer import pooled_selection
from .precise_ops import gate_sigmoid


class Glm53KDACache:
    """Exact conv + vector-decay recurrent state; restore only at checkpoints."""

    cache_operations_contract = CacheOperationsContract(
        family="glm53-kda",
        topology="checkpointed_hybrid",
        state_revision="glm53-kda-conv3-fp32-vector-decay-v1",
        cache_layout="glm53-gguf-unpadded-b1-v1",
        numerics="fp32-positive-exp-a",
        planes=(CachePlaneDeclaration("gdn_recurrent", ("state",), ("meta_state",)),),
        exact_prefix=True,
        restores_at_checkpoints=True,
        rollback=None,
    )

    def __init__(self, identity):
        self.identity, self.offset = identity, 0
        self.conv = self.recurrent = None

    @property
    def state(self):
        return self.conv, self.recurrent

    @property
    def meta_state(self):
        return "glm53-kda-v1", self.identity, str(self.offset)

    @classmethod
    def from_state(cls, state, meta_state):
        if (
            len(meta_state) != 3
            or meta_state[0] != "glm53-kda-v1"
            or not isinstance(meta_state[1], str)
            or not meta_state[1]
            or not str(meta_state[2]).isascii()
            or not str(meta_state[2]).isdecimal()
            or len(state) != 2
        ):
            raise ValueError("invalid GLM-5.3 recurrent state identity")
        obj = cls(meta_state[1])
        obj.offset = int(meta_state[2])
        obj.conv, obj.recurrent = state
        if obj.offset and (obj.conv is None or obj.recurrent is None):
            raise ValueError("nonempty GLM-5.3 state lacks a state plane")
        if obj.recurrent is not None and obj.recurrent.dtype != mx.float32:
            raise ValueError("GLM-5.3 recurrent state must remain FP32")
        if (
            obj.offset > 65536
            or (obj.conv is not None and obj.conv.shape != (1, 3, 24576))
            or (obj.recurrent is not None and obj.recurrent.shape != (1, 64, 128, 128))
        ):
            raise ValueError("GLM-5.3 recurrent state geometry/frontier mismatch")
        return obj

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.state if a is not None)

    def is_trimmable(self):
        return False

    def empty(self):
        return self.offset == 0

    def size(self):
        return self.offset

    def trim(self, count):
        if count:
            raise ValueError(
                "GLM-5.3 KDA can restore checkpoints but cannot trim state"
            )
        return 0

    def supports_ragged_trim(self):
        return False

    def extract(self, index):
        if index != 0:
            raise ValueError("GLM-5.3 KDA only owns one row")
        return self.from_state(
            tuple(None if a is None else mx.stop_gradient(a) for a in self.state),
            self.meta_state,
        )

    def filter(self, keep):
        if list(keep) != [0]:
            raise ValueError("GLM-5.3 KDA only supports its single live row")

    @classmethod
    def merge(cls, caches):
        if len(caches) != 1:
            raise ValueError("GLM-5.3 KDA cannot merge multiple rows")
        return caches[0].extract(0)

    def exact_decode_checkpoint_contract(self):
        return ("glm53-kda-v1", self.identity)

    def restore_exact_decode_checkpoint(self, contract):
        if contract != self.exact_decode_checkpoint_contract():
            raise ValueError("GLM-5.3 recurrent checkpoint identity changed")


register_cache_class(Glm53KDACache)


class Glm53MLACache(CacheList):
    """One exact latent/indexer checkpoint, with an explicit APCv2 ABI."""

    cache_operations_contract = CacheOperationsContract(
        family="glm53-latent-and-raw-indexer",
        topology="append_only",
        state_revision="glm53-latent512-indexer256-fp32-v1",
        cache_layout="glm53-gguf-unpadded-b1-v1",
        numerics="fp32-nope-latent-raw-indexer",
        planes=(CachePlaneDeclaration("attention_kv", ("state",), ("meta_state",)),),
        exact_prefix=True,
        restores_at_checkpoints=True,
        rollback="kv_trim",
    )

    def __init__(self):
        super().__init__(KVCache(), KVCache())

    @classmethod
    def from_state(cls, state, meta_state):
        if (
            len(state) != 2
            or len(meta_state) != 2
            or any(len(v) != 2 for v in meta_state)
        ):
            raise ValueError("GLM-5.3 latent/indexer checkpoint topology changed")
        result = super().from_state(state, meta_state)
        offsets = []
        for cache, width in zip(result.caches, (512, 256), strict=True):
            offsets.append(cache.offset)
            k, v = cache.state
            if k is None and v is None and cache.offset == 0:
                continue
            if (
                k is None
                or v is None
                or k.dtype != mx.float32
                or v.dtype != mx.float32
                or k.ndim != 4
                or k.shape[:2] != (1, 1)
                or k.shape[3] != width
                or v.shape != (*k.shape[:3], 0)
                or not 0 <= cache.offset <= min(k.shape[2], 65536)
            ):
                raise ValueError("GLM-5.3 latent/indexer checkpoint geometry changed")
        if offsets[0] != offsets[1]:
            raise ValueError("GLM-5.3 latent/indexer checkpoint frontiers differ")
        return result

    @property
    def offset(self):
        return self.caches[0].offset

    def extract(self, index):
        if index != 0:
            raise ValueError("GLM-5.3 MLA/indexer only owns one row")
        result = type(self)()
        result.caches = tuple(
            c.extract(index)
            if callable(getattr(c, "extract", None))
            else type(c).from_state(c.state, c.meta_state)
            for c in self.caches
        )
        return result

    def exact_decode_checkpoint_contract(self):
        return ("glm53-latent-and-indexer-v1",)

    def restore_exact_decode_checkpoint(self, contract):
        if contract != self.exact_decode_checkpoint_contract():
            raise ValueError("GLM-5.3 MLA/indexer checkpoint identity changed")


register_cache_class(Glm53MLACache)


def rms(x, weight, eps):
    return mx.fast.rms_norm(x.astype(mx.float32), weight, eps)


def hc_collapse(x, weights, prefix, eps, *, fused):
    from .deepseek_v4_hc import split_sinkhorn, weighted_sum

    z = rms(x.reshape(*x.shape[:-2], -1), None, eps)
    mixes = weights.linear(prefix + "_fn")(z)
    scale, base = weights.array(prefix + "_scale"), weights.array(prefix + "_base")
    if fused:
        pre, post, comb = split_sinkhorn(mixes, scale, base, iterations=20, eps=1e-6)
        return weighted_sum(x, pre), post, comb
    pre = gate_sigmoid(mixes[..., :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * gate_sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    comb = (
        mx.softmax(
            (mixes[..., 8:] * scale[2] + base[8:]).reshape(*mixes.shape[:-1], 4, 4),
            axis=-1,
            precise=True,
        )
        + 1e-6
    )
    comb = comb / (comb.sum(-2, keepdims=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdims=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdims=True) + 1e-6)
    return (x * pre[..., None]).sum(-2), post, comb


def hc_expand(branch, residual, post, comb, *, fused):
    if fused:
        from .deepseek_v4_hc import expand

        return expand(branch, residual, post, comb)
    return branch[..., None, :] * post[..., None] + comb.swapaxes(-1, -2) @ residual


class Glm53GGUFModel:
    """No placeholder tensors; the weight provider owns all model storage."""

    def __init__(
        self, config, weights, *, identity, math_policy="ordinary", hc_fused=False
    ):
        if math_policy not in ("ordinary", "reference"):
            raise ValueError("unknown GLM-5.3 math policy")
        self.args = SimpleNamespace(**config)
        self.config = self.args
        self.weights = weights
        self.identity = identity
        self.math_policy = math_policy
        self.hc_fused = hc_fused
        self.layers = [
            SimpleNamespace(is_linear=kind == "linear_attention")
            for kind in config["layer_types"]
        ]
        self._conv_weights = {}
        self.embed_tokens = self._embed_tokens
        self.lm_head = weights.linear("output.weight")
        self.external_feature_source = "layer_hc_mean"
        self.external_feature_width = (
            self.args.hidden_size if hasattr(self.args, "hidden_size") else 4096
        )
        self.mtp = None
        self.load_receipt = {}
        self.calls = {
            "kda_kernel": 0,
            "kda_reference": 0,
            "mla": 0,
            "pooled_indexer": 0,
        }

    def _embed_tokens(self, inputs):
        # External draft blocks use signed token IDs; packed lookup owns the
        # uint32 shader ABI. Keep that adaptation at the target boundary.
        return self.weights.embedding(inputs.astype(mx.uint32))

    def make_cache(self):
        return [
            Glm53KDACache(self.identity) if layer.is_linear else Glm53MLACache()
            for layer in self.layers
        ]

    def _linear(self, x, prefix, suffix):
        return self.weights.linear(prefix + "." + suffix)(x)

    def _kda(self, x, cache, prefix, *, _projected=None, _return_pre_output=False):
        b, length, _ = x.shape
        heads, dim = 64, 128
        inner = heads * dim
        project = lambda value, name: self._linear(value, prefix, name)
        mixed = (
            mx.concatenate(
                [project(x, "attn_" + p + ".weight") for p in ("q", "k", "v")], axis=-1
            )
            if _projected is None
            else _projected["qkv"]
        )
        conv = (
            cache.conv
            if cache.conv is not None
            else mx.zeros((b, 3, 3 * inner), dtype=mx.float32)
        )
        conv_input = mx.concatenate((conv, mixed), axis=1)
        cache.conv = mx.contiguous(conv_input[:, -3:])
        if prefix not in self._conv_weights:
            self._conv_weights[prefix] = mx.concatenate(
                [
                    self.weights.array(prefix + ".ssm_conv1d_" + p + ".weight").reshape(
                        inner, 4, 1
                    )
                    for p in ("q", "k", "v")
                ],
                axis=0,
            )
        # GGUF stores [channels,1,kernel]; transpose before the MLX convolution.
        filters = self._conv_weights[prefix]
        out = nn.silu(mx.conv1d(conv_input, filters, groups=3 * inner))
        q, k, v = [a.reshape(b, length, heads, dim) for a in mx.split(out, 3, axis=-1)]
        # GGML/FLA L2 norm: epsilon floors the norm itself.
        q = q / mx.maximum(mx.sqrt((q * q).sum(-1, keepdims=True)), 1e-6) * dim**-0.5
        k = k / mx.maximum(mx.sqrt((k * k).sum(-1, keepdims=True)), 1e-6)
        a = (
            project(project(x, "ssm_f_a.weight"), "ssm_f_b.weight")
            if _projected is None
            else _projected["a"]
        ).reshape(b, length, heads, dim)
        a = a + self.weights.array(prefix + ".ssm_dt.bias").reshape(heads, dim)
        stored_a = self.weights.array(prefix + ".ssm_a").reshape(heads, 1)
        decay = mx.exp(-5 * gate_sigmoid(a * stored_a))
        beta = gate_sigmoid(
            project(x, "ssm_beta.weight") if _projected is None else _projected["beta"]
        )
        state = cache.recurrent
        if state is None:
            state = mx.zeros((b, heads, dim, dim), dtype=mx.float32)
        use_kernel = self.math_policy == "ordinary" and mx.default_device() == mx.gpu
        function = gated_delta_kernel if use_kernel else gated_delta_ops
        self.calls["kda_kernel" if use_kernel else "kda_reference"] += 1
        out, cache.recurrent = function(q, k, v, decay, beta, state)
        cache.offset += length
        gate = (
            project(project(x, "ssm_g_a.weight"), "ssm_g_b.weight")
            if _projected is None
            else _projected["gate"]
        ).reshape(b, length, heads, dim)
        out = rms(
            out, self.weights.array(prefix + ".ssm_norm.weight"), self.args.rms_norm_eps
        ) * gate_sigmoid(gate)
        out = out.reshape(b, length, inner)
        return out if _return_pre_output else project(out, "attn_output.weight")

    def _mla(self, x, cache, prefix, *, _return_pre_output=False):
        b, length, _ = x.shape
        project = lambda value, name: self._linear(value, prefix, name)
        qr = rms(
            project(x, "attn_q_a.weight"),
            self.weights.array(prefix + ".attn_q_a_norm.weight"),
            self.args.rms_norm_eps,
        )
        q = project(qr, "attn_q_b.weight").reshape(b, length, 64, 256)
        latent = rms(
            project(x, "attn_kv_a_mqa.weight"),
            self.weights.array(prefix + ".attn_kv_a_norm.weight"),
            self.args.rms_norm_eps,
        )[:, None]
        latent, _ = cache[0].update_and_fetch(
            latent, mx.zeros((b, 1, length, 0), dtype=x.dtype)
        )
        selected = pooled_selection(x, qr, cache[1], self.weights, prefix)
        self.calls["pooled_indexer"] += int(selected is not None)
        q = self.weights.grouped(q, prefix + ".attn_k_b.weight", groups=64).transpose(
            0, 2, 1, 3
        )
        total = latent.shape[2]
        pieces = []
        for start in range(0, length, 128):
            stop = min(start + 128, length)
            if selected is None:
                pos = total - length + mx.arange(start, stop)
                mask = mx.arange(total)[None, :] <= pos[:, None]
                mask = mask[None, None]
            else:
                ids = selected[:, start:stop]
                # Extra dump column keeps -1 padding outside the visible keys.
                safe = mx.where(ids >= 0, ids, total)
                mask = mx.zeros((b, stop - start, total + 1), dtype=mx.bool_)
                mask = mx.put_along_axis(mask, safe, mx.array(True), axis=-1)[
                    ..., :total
                ][:, None]
            pieces.append(
                mx.fast.scaled_dot_product_attention(
                    q[:, :, start:stop], latent, latent, scale=256**-0.5, mask=mask
                )
            )
        out = mx.concatenate(pieces, axis=2).transpose(0, 2, 1, 3)
        out = self.weights.grouped(out, prefix + ".attn_v_b.weight", groups=64)
        self.calls["mla"] += 1
        out = out.reshape(b, length, -1)
        return out if _return_pre_output else project(out, "attn_output.weight")

    def _mlp(self, x, prefix, index):
        project = lambda value, name: self._linear(value, prefix, name)

        def shared(suffix):
            gate = mx.minimum(project(x, "ffn_gate" + suffix + ".weight"), 10)
            up = mx.clip(project(x, "ffn_up" + suffix + ".weight"), -10, 10)
            return project(nn.silu(gate) * up, "ffn_down" + suffix + ".weight")

        if index < 3:
            return shared("")
        scores = gate_sigmoid(project(x, "ffn_gate_inp.weight"))
        choice = scores + self.weights.array(prefix + ".exp_probs_b.bias")
        ids = mx.argsort(-choice, axis=-1)[..., :8].astype(mx.uint32)
        selected = mx.take_along_axis(scores, ids, axis=-1)
        selected = selected / selected.sum(-1, keepdims=True) * 2.5
        routed = self.weights.moe(
            x.reshape(-1, x.shape[-1]),
            ids.reshape(-1, 8),
            selected.reshape(-1, 8),
            prefix,
        ).reshape(x.shape)
        return routed + shared("_shexp")

    def __call__(
        self, inputs, cache=None, *, logits_to_keep=None, _taps=None, _body_only=False
    ):
        if inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] < 1:
            raise ValueError("GLM-5.3 packed ordinary text currently requires B1")
        if cache is None:
            cache = self.make_cache()
        if len(cache) != len(self.layers):
            raise ValueError("GLM-5.3 cache layer count differs")
        offset = cache[3][0].offset
        if offset + inputs.shape[1] > self.args.max_position_embeddings:
            raise ValueError("GLM-5.3 request exceeds route context")
        for c, layer in zip(cache, self.layers):
            if layer.is_linear and (c.identity != self.identity or c.offset != offset):
                raise ValueError("GLM-5.3 recurrent cache identity/frontier mismatch")
            if not layer.is_linear and (
                len(c.caches) != 2 or any(kv.offset != offset for kv in c.caches)
            ):
                raise ValueError("GLM-5.3 latent/indexer cache frontier mismatch")
        h = self.weights.embedding(inputs.astype(mx.uint32))
        h = mx.contiguous(
            mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], 4, h.shape[-1]))
        )
        for index, (layer, c) in enumerate(zip(self.layers, cache)):
            prefix = f"blk.{index}"
            x, post, comb = hc_collapse(
                h,
                self.weights,
                prefix + ".hc_attn",
                self.args.rms_norm_eps,
                fused=self.hc_fused,
            )
            x = rms(
                x,
                self.weights.array(prefix + ".attn_norm.weight"),
                self.args.rms_norm_eps,
            )
            branch = (
                self._kda(x, c, prefix) if layer.is_linear else self._mla(x, c, prefix)
            )
            h = hc_expand(branch, h, post, comb, fused=self.hc_fused)
            x, post, comb = hc_collapse(
                h,
                self.weights,
                prefix + ".hc_ffn",
                self.args.rms_norm_eps,
                fused=self.hc_fused,
            )
            x = rms(
                x,
                self.weights.array(prefix + ".ffn_norm.weight"),
                self.args.rms_norm_eps,
            )
            h = hc_expand(
                self._mlp(x, prefix, index), h, post, comb, fused=self.hc_fused
            )
            if _taps is not None and index in _taps:
                _taps[index] = h.mean(-2)
        h = rms(
            h.mean(-2), self.weights.array("output_norm.weight"), self.args.rms_norm_eps
        )
        if _taps is not None and self.external_feature_source == "mtp_seed":
            _taps[len(self.layers) - 1] = h
        if _body_only:
            return None
        if logits_to_keep is None:
            h = select_checkpoint_rows(h)
        elif logits_to_keep:
            h = h[:, -logits_to_keep:]
        return self.weights.linear("output.weight")(h)

    def prefill_forward(self, inputs, cache=None):
        return self(inputs, cache)

    def forward_with_taps(self, inputs, cache, capture_layers, *, body_only=False):
        layers = tuple(capture_layers)
        if (
            not layers
            or tuple(sorted(set(layers))) != layers
            or layers[0] < 0
            or layers[-1] >= len(self.layers)
        ):
            raise ValueError("GLM-5.3 invalid feature capture layers")
        taps = {i: None for i in layers}
        logits = self(inputs, cache, logits_to_keep=0, _taps=taps, _body_only=body_only)
        return logits, mx.concatenate([taps[i] for i in layers], axis=-1)

    def prefill_body(self, inputs, cache, capture_layers):
        return self.forward_with_taps(inputs, cache, capture_layers, body_only=True)[1]

    def enable_mtp(self):
        from .glm53_gguf_mtp import Glm53MTPHead

        self.mtp = Glm53MTPHead(self)
        self.external_feature_source = "mtp_seed"
        self.external_feature_width = 4096
        return self

    def make_mtp_cache(self):
        if self.mtp is None:
            raise ValueError("GLM-5.3 MTP head is not loaded")
        return [Glm53MLACache()]

    def mtp_step(self, hidden, tokens, cache):
        if self.mtp is None:
            raise ValueError("GLM-5.3 MTP head is not loaded")
        return self.mtp(hidden, tokens, cache)

    def preload(self):
        import logging
        import time

        started = time.monotonic()
        completed = count = 0
        boundary = 8 << 30
        for name, t in self.weights.tensors.items():
            if name.startswith("blk.45.") and self.mtp is None:
                continue
            value = (
                self.weights.array(name)
                if t.ggml_type == 0
                else self.weights.packed(name).data
            )
            mx.eval(value)
            completed += t.nbytes
            count += 1
            if completed >= boundary:
                logging.getLogger(__name__).info(
                    "GLM packed load: %.1f GiB, %.1fs",
                    completed / (1 << 30),
                    time.monotonic() - started,
                )
                boundary += 8 << 30
        self.weights.verify_storage()
        self.load_receipt = {
            "seconds": time.monotonic() - started,
            "payload_bytes": completed,
            "tensors": count,
            "native_read_view": hasattr(self.weights.file, "view"),
        }
        logging.getLogger(__name__).info(
            "GLM packed load complete: %s", self.load_receipt
        )
        return self

    def parameters(self):
        return {
            "native": dict(self.weights._dense),
            "packed": {n: w.data for n, w in self.weights._packed.items()},
            "conv": dict(self._conv_weights),
        }

    def eval(self):
        mx.eval(self.parameters())
        return self

    def named_modules(self):
        yield "", self
