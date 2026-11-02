"""Adapter-owned GLM-5.3 ordinary text forward over exact GGUF blocks.

See provenance/glm53-gguf-forward.json. The checkpoint's positive exp(A_log)
is consumed directly, without a log/exp round trip or affine requantization.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn

from ..cache_operations import CacheOperationsContract, CachePlaneDeclaration
from ..prefill_checkpoint_rows import select_checkpoint_rows
from . import glm53_gguf_indexer as indexer
from . import glm53_kv_append as kv_store
from . import glm53_sparse_attention as sparse_attention
from .cache import CacheList, KVCache, _cache_class_token, register_cache_class
from .gated_delta import gated_delta_kernel, gated_delta_ops
from .glm53_gguf_indexer import (
    TOPK,
    Glm53PoolCache,
    audit_pools,
    pack_indexer_rows,
    pooled_selection,
    select_from_rows,
)
from .precise_ops import gate_sigmoid

# Sparse DSA execution policies (docs/design/glm53-sparse-dsa.md section 12).
# The first value of each is today's law and the default; every other value
# is opt-in and changes the APCv2 profile.
SPARSE_POLICIES = {
    "indexer_pools": ("recompute", "recompute_fixed_order", "incremental_v1", "incremental_v1_audit"),
    "indexer_select": ("argsort", "topk_v1", "topk_v1_kernel"),
    # fused_v2 and gathered_v2_from_<n> (docs/design/glm53-speed-round2.md
    # 1.1, 2.1): the v2 kernel law for decode rows and for prompt rows from
    # position n (dense causal rows below 2049 included); opt-in, unqualified.
    "sparse_attention": sparse_attention.POLICIES_ATTENTION[:1] + sparse_attention.DECODE_POLICIES,
    "sparse_prefill": ("dense_reference", "gathered_v1_8k", "gathered_v1_16k", "gathered_v1_all",
                       *(f"gathered_v2_from_{n}" for n in sparse_attention.PREFILL_POSITIONS_V2)),
    "tree_rows_attention": ("per_row", "batched_v1"),
    "tree_frontier": ("clone", "tails_v1"),
}
INCREMENTAL_POOLS = ("incremental_v1", "incremental_v1_audit")
# Decode glue launch policies (glm53_glue_metal). The first value is today's
# op chain and the default; fused_v1 is the same law bit for bit in one launch
# (Metal only; elsewhere the reference chain runs and is counted as declined).
GLUE_POLICIES = {
    # fused_v2 also folds the ffn_gate_inp F32 gemv into the router launch
    # (B1 single-token rows; other rows run fused_v1 and count a decline).
    "router": ("reference", "fused_v1", "fused_v2"),
    "kda_small": ("reference", "fused_v1"),
    # One launch per KDA layer per B1 decode token for conv, SiLU, q/k norms,
    # gates, delta rule and norm x gate (glm53_kda_fused); same bits.
    "kda_fused": ("reference", "decode_v1"),
}
GLUE_COUNTERS = (
    "router_fused",
    "router_fused_declined",
    "kda_small_fused",
    "kda_small_fused_declined",
    "kda_fused",
    "kda_fused_declined",
    "router_fused_v2",
    "router_fused_v2_declined",
)
# glm53_hc_fused (glm53_hc_fused.py): HC boundaries with the flat norm and the
# F32 mixing gemv (and the previous half's expand) in one launch; requires
# glm53_hc=mega_v1. "tail_v1" also runs the mega post step in that launch.
HC_FUSED_POLICIES = ("reference", "v1", "tail_v1")
HC_FUSED_COUNTERS = ("hc_fused_boundary", "hc_fused_tail", "hc_fused_declined")
# glm53_f32_weights: the F32 gemv operands read by the fused HC and router
# kernels (hc_*_fn, ffn_gate_inp) hold BF16-representable values in this
# checkpoint; "bf16_v1" hands those kernels an exact BF16 copy (half the bytes,
# the same float operand per element, so the same bits). A tensor whose copy
# is not exact keeps its F32 operand and counts as declined.
F32_WEIGHT_POLICIES = ("reference", "bf16_v1")
F32_WEIGHT_COUNTERS = ("f32_bf16_operand", "f32_bf16_declined")
# Prompt-phase selection scores pools in blocks of this many (design 5.2).
SELECT_BLOCK = 4096
# Decode-phase gathered query rows live at once (Glm53CacheBudget books 8).
DECODE_ROW_GROUP = 8
Q8_REFUSAL = (
    "q8 latent is a dense-mask research knob; the gathered sparse paths require the fp32 latent plane"
)
# indexer route counters -> Glm53GGUFModel.calls
_INDEXER_COUNTERS = (
    ("select_argsort", "indexer_select_argsort"),
    ("select_topk", "indexer_select_topk"),
    ("select_kernel", "indexer_select_kernel"),
    ("select_kernel_declined", "indexer_select_kernel_declined"),
    ("select_blocked", "indexer_select_blocked"),
)
SPARSE_COUNTERS = (
    "indexer_pool_recompute_full",
    "indexer_pool_extend",
    "indexer_pools_computed",
    "indexer_pool_audit",
    "indexer_pool_kernel",
    "indexer_pool_kernel_declined",
    *(name for _, name in _INDEXER_COUNTERS),
    "sparse_attention_mask",
    "sparse_attention_gathered",
    "sparse_attention_fused",
    "sparse_attention_fused_launches",
    "sparse_gathered_rows",
    "sparse_prefill_gathered_blocks",
    "sparse_prefill_dense_pieces",
    "sparse_attention_fused_v2",
    "sparse_attention_fused_v2_rows",
    "sparse_attention_fused_v2_per_row_fallback",
    "sparse_attention_fused_v2_dense_rows",
    "sparse_prefill_fused_v2_blocks",
    "tree_rows_attention_batched",
    "tree_rows_attention_per_row_fallback",
    "tree_frontier_materialized",
)
# glm53_kv_append (glm53_kv_append.py): MLA appends on rotate_v1 ring planes.
KV_APPEND_COUNTERS = ("kv_append_ring",)
# glm53_mla_fused (glm53_mla_fused.py): B1 decode MLA glue (norms, raw indexer
# row, pool completion, pool scores, gather + mask) in four kernels around the
# unchanged projections, select kernel, q x pooled matmul and SDPA; same bits.
MLA_FUSED_POLICIES = ("reference", "decode_v1")
MLA_FUSED_COUNTERS = ("mla_fused", "mla_fused_dense", "mla_fused_pool", "mla_fused_declined")


def check_sparse_policy(policy):
    """Raise unless ``policy`` (SPARSE_POLICIES keys) is a supported combination.

    Model-level rules only; the adapter adds the AST-target, tree and Metal
    requirements it alone can see.
    """
    for key, value in policy.items():
        if key not in SPARSE_POLICIES or value not in SPARSE_POLICIES[key]:
            raise ValueError(f"unsupported GLM-5.3 sparse policy {key}={value!r}")
    get = lambda key: policy.get(key, SPARSE_POLICIES[key][0])
    if get("indexer_select") != "argsort" and get("indexer_pools") == "recompute":
        raise ValueError(
            "GLM-5.3 reference pooling (recompute) runs only with argsort selection; "
            "topk_v1 needs a fixed-order pool policy"
        )
    if get("indexer_select") == "argsort" and (
        get("sparse_attention") != "mask_reference" or get("sparse_prefill") != "dense_reference"
    ):
        raise ValueError(
            "GLM-5.3 gathered attention requires chronological ids (glm53_indexer_select != argsort)"
        )


_SPARSE_PHASE = ContextVar("mlx2_glm53_sparse_attention_phase", default=None)


@contextmanager
def sparse_attention_phase(phase):
    """Bind the sparse-attention phase of one explicit prompt forward.

    ``prompt`` lets glm53_sparse_prefill decide dense pieces vs gathered
    blocks; every other forward (decode, verify rows, MTP head, draft steps)
    uses glm53_sparse_attention. Width never decides the phase.
    """
    if phase != "prompt":
        raise ValueError("GLM-5.3 sparse attention phase must be 'prompt'")
    token = _SPARSE_PHASE.set(phase)
    try:
        yield
    finally:
        _SPARSE_PHASE.reset(token)


def sparse_phase():
    return _SPARSE_PHASE.get()


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


class Glm53LatentQ8Cache:
    """Research (opt-in, approximate): the MLA latent as 8-bit affine groups of 64.

    Replaces the exact fp32 latent plane of ``Glm53MLACache`` when the model's
    ``mla_kv_q8`` knob is set; attention then reads the latent through
    ``quantized_matmul`` (scores and values both come from the latent).  Not
    an APCv2/approximate-operation citizen: no state/from_state, ordinary
    route only (the layered tree clones exact planes).

    Left off: not helpful (GLM-5.3-Flash-REAP50-Q3_K_M.gguf@5a865918, mlx2
    ed4a08fd, 2026-10-31). The 11 latents are ~0.4 GB at 16K of an ~87 GB
    peak, so q8 saves nothing visible; prefill is 5-11% slower (1K/8K/16K
    544/410/315 -> 510/393/281 tok/s) and decode is no faster. Greedy output
    diverges at 1K/8K/16K tokens 34/48/4, each a near tie (same top two,
    margins <= 0.15 nat). Superseded by the sparse DSA design.
    """

    step = 256
    offset = 0
    parts = None

    def update_and_fetch(self, latent, _values=None):
        prev, length = self.offset, latent.shape[2]
        new = mx.quantize(latent, group_size=64, bits=8)
        if self.parts is None or prev + length > self.parts[0].shape[2]:
            size = (prev + length + self.step - 1) // self.step * self.step
            grown = []
            for index, part in enumerate(new):
                buffer = mx.zeros((*part.shape[:2], size, part.shape[3]), part.dtype)
                if self.parts is not None and prev:
                    buffer[:, :, :prev] = self.parts[index][:, :, :prev]
                grown.append(buffer)
            self.parts = grown
        for buffer, part in zip(self.parts, new):
            buffer[:, :, prev:prev + length] = part
        self.offset = prev + length
        return self.state, None

    @property
    def state(self):
        # Read-only (for mx.eval of the cache); there is no restore path.
        return tuple(part[:, :, :self.offset] for part in self.parts) if self.parts else ()


def _latent_q8_attention(q, keys, scale, mask):
    """SDPA over a q8 latent used as both keys and values (one shared head)."""
    b, heads, length, width = q.shape
    keys = [mx.expand_dims(part, -3) for part in keys]
    scores = mx.quantized_matmul(
        (q * scale).reshape(b, 1, heads, length, width), *keys,
        transpose=True, group_size=64, bits=8,
    )
    if mask is not None:
        scores = mx.where(mask[:, :, None], scores, mx.array(-mx.inf, scores.dtype))
    weights = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(weights, *keys, transpose=False, group_size=64, bits=8)
    return out.reshape(b, heads, length, width)


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

    # Derived pooled-key plane (Glm53PoolCache), attached lazily by _mla under
    # an incremental pool policy. Never in `state`: serialized restores come
    # back with None and recompute once; every rewind cuts it (design 6.5).
    pools = None

    def __init__(self):
        super().__init__(KVCache(), KVCache())

    def _cut_pools(self):
        if self.pools is not None:
            # The indexer plane is what pools derive from. A scalar array
            # offset is converted by cut (one sync, rewinds only).
            self.pools.cut(self.caches[1].offset)

    def trim(self, n):
        trimmed = super().trim(n)
        self._cut_pools()
        return trimmed

    def trim_to_position(self, position, num_tokens):
        trimmed = super().trim_to_position(position, num_tokens)
        self._cut_pools()
        return trimmed

    def trim_ragged(self, n, *, validate=True):
        trimmed = super().trim_ragged(n, validate=validate)
        self._cut_pools()
        return trimmed

    @property
    def state(self):
        return CacheList.state.fget(self)

    @state.setter
    def state(self, value):
        CacheList.state.fset(self, value)
        self.pools = None

    @property
    def meta_state(self):
        # A rotate_v1 plane serializes as its reference class (same state, same
        # payload): restores come back plain and re-engage on their next append.
        _, metas = CacheList.meta_state.fget(self)
        return [_cache_class_token(kv_store.canonical_type(c)) for c in self.caches], metas

    @meta_state.setter
    def meta_state(self, value):
        # Rewrites child offsets: a metadata-only rewind must not keep pools.
        CacheList.meta_state.fset(self, value)
        self.pools = None

    @property
    def nbytes(self):
        pooled = 0 if self.pools is None else self.pools.nbytes
        return CacheList.nbytes.fget(self) + pooled

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
        # New array descriptors: a later append to either cache must not
        # write into the other's buffer object (its cloned pools would then
        # describe rows it no longer holds).
        result.caches = tuple(
            c.extract(index)
            if callable(getattr(c, "extract", None))
            else type(c).from_state(
                tuple(None if a is None else mx.stop_gradient(a) for a in c.state), c.meta_state
            )
            for c in self.caches
        )
        if self.pools is not None:
            result.pools = self.pools.clone()
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
        self, config, weights, *, identity, math_policy="ordinary", hc_fused=False, hc_mega=False,
        shared_fold=False, kda_group=False, mla_group=False, indexer_pools="recompute",
        indexer_select="argsort", sparse_attention="mask_reference", sparse_prefill="dense_reference",
        tree_rows_attention="per_row", tree_frontier="clone", router="reference",
        kda_small="reference", kv_append="reference", kda_fused="reference", hc_fused_policy="reference",
        mla_fused="reference", f32_weights="reference",
    ):
        if math_policy not in ("ordinary", "reference"):
            raise ValueError("unknown GLM-5.3 math policy")
        if kv_append not in kv_store.KV_APPEND_POLICIES:
            raise ValueError(f"unsupported GLM-5.3 KV append policy {kv_append!r}")
        for key, value in (("router", router), ("kda_small", kda_small), ("kda_fused", kda_fused)):
            if value not in GLUE_POLICIES[key]:
                raise ValueError(f"unsupported GLM-5.3 glue policy {key}={value!r}")
        if hc_fused_policy not in HC_FUSED_POLICIES:
            raise ValueError(f"unsupported GLM-5.3 HC fused policy {hc_fused_policy!r}")
        if hc_fused_policy != "reference" and not (hc_mega and hc_fused):
            # mega_v1 sets both; the fused expand is the fused_fp32 expand law.
            raise ValueError("glm53_hc_fused requires glm53_hc=mega_v1")
        if mla_fused not in MLA_FUSED_POLICIES:
            raise ValueError(f"unsupported GLM-5.3 MLA fused policy {mla_fused!r}")
        if f32_weights not in F32_WEIGHT_POLICIES:
            raise ValueError(f"unsupported GLM-5.3 F32 weight policy {f32_weights!r}")
        sparse = {
            "indexer_pools": indexer_pools,
            "indexer_select": indexer_select,
            "sparse_attention": sparse_attention,
            "sparse_prefill": sparse_prefill,
            "tree_rows_attention": tree_rows_attention,
            "tree_frontier": tree_frontier,
        }
        check_sparse_policy(sparse)
        self.args = SimpleNamespace(**config)
        self.config = self.args
        self.weights = weights
        self.identity = identity
        self.math_policy = math_policy
        self.hc_fused = hc_fused
        self.hc_mega = hc_mega
        self.shared_fold = shared_fold
        self.kda_group = kda_group
        self.mla_group = mla_group
        self.router = router
        self.kda_small = kda_small
        self.kda_fused = kda_fused
        self.hc_fused_policy = hc_fused_policy
        self.mla_fused = mla_fused
        self.f32_weights = f32_weights
        # Storage law of the MLA latent/raw planes: "reference" appends in
        # place, which under pipelined decode copies the whole plane every
        # token (the step in flight still holds it); "rotate_v1" rotates
        # RING_BUFFERS full buffers so the written one is free. Same values.
        self.kv_append = kv_append
        # Sparse DSA policies; the defaults are today's laws bit for bit. The
        # tree values are read by the layered tree (glm53_layer_tree).
        for key, value in sparse.items():
            setattr(self, key, value)
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
            **{name: 0 for name in SPARSE_COUNTERS},
            **{name: 0 for name in GLUE_COUNTERS},
            **{name: 0 for name in HC_FUSED_COUNTERS},
            **{name: 0 for name in F32_WEIGHT_COUNTERS},
            "prompt_layer_evals": 0,
            **{name: 0 for name in KV_APPEND_COUNTERS},
            **{name: 0 for name in MLA_FUSED_COUNTERS},
        }

    def _embed_tokens(self, inputs):
        # External draft blocks use signed token IDs; packed lookup owns the
        # uint32 shader ABI. Keep that adaptation at the target boundary.
        return self.weights.embedding(inputs.astype(mx.uint32))

    def make_cache(self):
        return [
            Glm53KDACache(self.identity) if layer.is_linear else self._mla_cache()
            for layer in self.layers
        ]

    def _mla_cache(self):
        cache = Glm53MLACache()
        if getattr(self, "mla_kv_q8", False):
            if self._gathered_policy():
                raise ValueError(Q8_REFUSAL)
            cache.caches = (Glm53LatentQ8Cache(), cache.caches[1])
        return cache

    def _linear(self, x, prefix, suffix):
        return self.weights.linear(prefix + "." + suffix)(x)

    def _kda_static(self, prefix):
        """Per-layer KDA host constants, built once per (weights, prefix).

        The small vectors are the provider's own arrays (and their reshaped
        views); a reshape moves no value, so reading them from here instead of
        rebuilding the view every step changes no bit.
        """
        memo = self.__dict__.get("_kda_static_memo")
        if memo is None or memo[0] is not self.weights:
            memo = self.__dict__["_kda_static_memo"] = (self.weights, {})
        static = memo[1].get(prefix)
        if static is None:
            from .glm53_glue_metal import fused_kda_gate, kda_gate_admits
            from .glm53_kda_fused import fused_kda_decode, kda_fused_admits

            heads, dim = 64, 128
            weights = self.weights
            dt_bias = weights.array(prefix + ".ssm_dt.bias")
            stored_a = weights.array(prefix + ".ssm_a")
            static = memo[1][prefix] = SimpleNamespace(
                dt_bias=dt_bias, stored_a=stored_a,
                dt_bias_hd=dt_bias.reshape(heads, dim), stored_a_h1=stored_a.reshape(heads, 1),
                norm=weights.array(prefix + ".ssm_norm.weight"),
                group_names=[prefix + "." + n + ".weight" for n in
                             ("attn_q", "attn_k", "attn_v", "ssm_f_a", "ssm_g_a", "ssm_beta")],
                second_names=[prefix + ".ssm_f_b.weight", prefix + ".ssm_g_b.weight"],
                fused_kda_gate=fused_kda_gate, kda_gate_admits=kda_gate_admits,
                fused_kda_decode=fused_kda_decode, kda_fused_admits=kda_fused_admits,
            )
        return static

    def _kda(self, x, cache, prefix, *, _projected=None, _return_pre_output=False):
        b, length, _ = x.shape
        heads, dim = 64, 128
        inner = heads * dim
        static = self._kda_static(prefix)
        project = lambda value, name: self._linear(value, prefix, name)
        if _projected is None and getattr(self, "kda_group", False):
            _projected = self._kda_grouped_projections(x, prefix)
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
        filters = self._kda_filters(prefix, inner)
        if getattr(self, "kda_fused", "reference") == "decode_v1":
            fused, _projected = self._kda_fused_decode(
                x, cache, prefix, static, mixed, conv, filters, _projected)
            if fused is not None:
                return fused if _return_pre_output else project(fused, "attn_output.weight")
        conv_input = mx.concatenate((conv, mixed), axis=1)
        cache.conv = mx.contiguous(conv_input[:, -3:])
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
        dt_bias = static.dt_bias
        stored_a = static.stored_a
        beta = project(x, "ssm_beta.weight") if _projected is None else _projected["beta"]
        gated = None
        if getattr(self, "kda_small", "reference") == "fused_v1":
            # One launch for decay, beta and sigmoid(gate); same law bit for bit.
            fused_kda_gate, kda_gate_admits = static.fused_kda_gate, static.kda_gate_admits
            gate = (
                project(project(x, "ssm_g_a.weight"), "ssm_g_b.weight")
                if _projected is None
                else _projected["gate"]
            ).reshape(b, length, heads, dim)
            flat_a = a.reshape(b, length, heads * dim)
            flat_gate = gate.reshape(b, length, heads * dim)
            if kda_gate_admits(flat_a, dt_bias, stored_a, beta, flat_gate, heads=heads, dim=dim):
                self.calls["kda_small_fused"] += 1
                decay, beta, gated = fused_kda_gate(
                    flat_a, dt_bias, stored_a, beta, flat_gate, heads=heads, dim=dim
                )
                decay = decay.reshape(b, length, heads, dim)
                gated = gated.reshape(b, length, heads, dim)
            else:
                self.calls["kda_small_fused_declined"] += 1
                _projected = {**(_projected or {}), "gate": gate}
        if gated is None:
            a = a + static.dt_bias_hd
            decay = mx.exp(-5 * gate_sigmoid(a * static.stored_a_h1))
            beta = gate_sigmoid(beta)
        state = cache.recurrent
        if state is None:
            state = mx.zeros((b, heads, dim, dim), dtype=mx.float32)
        use_kernel = self.math_policy == "ordinary" and mx.default_device() == mx.gpu
        function = gated_delta_kernel if use_kernel else gated_delta_ops
        self.calls["kda_kernel" if use_kernel else "kda_reference"] += 1
        out, cache.recurrent = function(q, k, v, decay, beta, state)
        cache.offset += length
        if gated is None:
            gate = (
                project(project(x, "ssm_g_a.weight"), "ssm_g_b.weight")
                if _projected is None or "gate" not in _projected
                else _projected["gate"]
            ).reshape(b, length, heads, dim)
            gated = gate_sigmoid(gate)
        out = rms(out, static.norm, self.args.rms_norm_eps) * gated
        out = out.reshape(b, length, inner)
        return out if _return_pre_output else project(out, "attn_output.weight")

    def _kda_filters(self, prefix, inner):
        # GGUF stores [channels,1,kernel]; transpose before the MLX convolution.
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
        return self._conv_weights[prefix]

    def _kda_fused_decode(self, x, cache, prefix, static, mixed, conv, filters, projected):
        """glm53_kda_fused=decode_v1: the whole post-projection KDA body in one
        launch (glm53_kda_fused), or None to run the reference chain; plus the
        small projections it built, so a decline never projects twice.

        Admitted only for one B1 token on the ordinary Metal route (the
        reference runs gated_delta_kernel there); prefill, B > 1, the reference
        math policy and CPU decline and are counted. Writes the cache exactly
        as the reference does.
        """
        heads, dim = 64, 128
        project = lambda value, name: self._linear(value, prefix, name)
        b, length = x.shape[:2]
        state = cache.recurrent
        admitted = (
            b == 1 and length == 1 and self.math_policy == "ordinary"
            and mx.default_device() == mx.gpu
        )
        if admitted:
            projected = dict(projected or {})
            if projected.get("a") is None:
                projected["a"] = project(project(x, "ssm_f_a.weight"), "ssm_f_b.weight")
            if projected.get("beta") is None:
                projected["beta"] = project(x, "ssm_beta.weight")
            if projected.get("gate") is None:
                projected["gate"] = project(project(x, "ssm_g_a.weight"), "ssm_g_b.weight")
            a, beta, gate = projected["a"], projected["beta"], projected["gate"]
            if state is None:
                state = mx.zeros((b, heads, dim, dim), dtype=mx.float32)
            admitted = static.kda_fused_admits(
                mixed, conv, filters, a, static.dt_bias, static.stored_a, beta, gate,
                state, static.norm,
            )
        if not admitted:
            self.calls["kda_fused_declined"] += 1
            return None, projected
        self.calls["kda_fused"] += 1
        out, cache.conv, cache.recurrent = static.fused_kda_decode(
            mixed, conv, filters, a, static.dt_bias, static.stored_a, beta, gate, state,
            static.norm, rms_eps=self.args.rms_norm_eps,
        )
        cache.offset += length
        return out, projected

    def _kda_grouped_projections(self, x, prefix):
        """KDA input projections in two K-block launches, or None if not admitted."""
        static = self._kda_static(prefix)
        first = self.weights.multi_linear([x] * 6, static.group_names, joined=3)
        if first is None:
            return None
        qkv, f_a, g_a, beta = first
        second = self.weights.multi_linear([f_a, g_a], static.second_names)
        if second is None:
            return None
        return {"qkv": qkv, "a": second[0],
                "beta": beta, "gate": second[1]}

    def _gathered_policy(self):
        """True when any gathered consumer (decode, prompt or tree rows) is selected."""
        return (
            self.sparse_attention != "mask_reference"
            or self.sparse_prefill != "dense_reference"
            or self.tree_rows_attention != "per_row"
        )

    def _mla_grouped_projections(self, x, selecting, prefix):
        """MLA input projections that read x, in one K-block launch, or None.

        q_a, kv_a_mqa and the indexer key always run; the indexer score
        projection joins only once the indexer selects (``selecting``: more
        than TOPK keys after this call), matching pooled_selection's own skip.
        Outputs equal separate calls.
        """
        names = [prefix + ".attn_q_a.weight", prefix + ".attn_kv_a_mqa.weight",
                 prefix + ".indexer.attn_k.weight"]
        if selecting:
            names.append(prefix + ".indexer.proj.weight")
        out = self.weights.multi_linear([x] * len(names), names)
        if out is None:
            return None
        result = {"q_a": out[0], "kv_a": out[1], "key": out[2]}
        if selecting:
            result["proj"] = out[3]
        return result

    def _mla_prepare(self, x, state, prefix, *, frontier=None):
        """Projections, the one append, and selection: (q_lat, ids, source).

        ``state`` is a Glm53MLACache (appended through its two KVCaches) or a
        Glm53MLAFrontier (duck-typed on ``latent_source``; appended once with
        ``append``). ``frontier`` is the integer token count before this call
        (the KDA frontier in the trunk, the head cache's own offset for NextN);
        None reads the state's own offset, as before the split. ``ids`` is
        (1, L, WIDTH) or None when every key fits; ``source`` is the latent
        the consumer reads ((1,1,T,512), a q8 tuple, or a TreeLatentView).
        """
        b, length, _ = x.shape
        tree = hasattr(state, "latent_source")
        if frontier is None:
            frontier = state.offset if tree else state[1].offset
        project = lambda value, name: self._linear(value, prefix, name)
        grouped = (
            self._mla_grouped_projections(x, frontier + length > TOPK, prefix)
            if getattr(self, "mla_group", False)
            else None
        )
        qa = grouped["q_a"] if grouped else project(x, "attn_q_a.weight")
        qr = rms(
            qa,
            self.weights.array(prefix + ".attn_q_a_norm.weight"),
            self.args.rms_norm_eps,
        )
        indexer_projected = {}
        if grouped:
            indexer_projected["key"] = grouped["key"]
            if "proj" in grouped:
                indexer_projected["proj"] = grouped["proj"]
                second = self.weights.multi_linear(
                    [qr, qr], [prefix + ".attn_q_b.weight", prefix + ".indexer.attn_q_b.weight"])
                if second is not None:
                    q_full, indexer_projected["q"] = second
                else:
                    q_full = project(qr, "attn_q_b.weight")
            else:
                q_full = project(qr, "attn_q_b.weight")
        else:
            q_full = project(qr, "attn_q_b.weight")
        q = q_full.reshape(b, length, 64, 256)
        latent = rms(
            grouped["kv_a"] if grouped else project(x, "attn_kv_a_mqa.weight"),
            self.weights.array(prefix + ".attn_kv_a_norm.weight"),
            self.args.rms_norm_eps,
        )[:, None]
        if tree:
            selected, latent, total = self._mla_select_frontier(
                x, qr, latent, state, prefix, indexer_projected
            )
        else:
            if getattr(self, "kv_append", "reference") != "reference":
                kv_store.engage(state.caches, self.kv_append)
            self.calls["kv_append_ring"] += int(kv_store.is_ring(state[0]))
            latent, _ = state[0].update_and_fetch(
                latent, mx.zeros((b, 1, length, 0), dtype=x.dtype)
            )
            total = latent[0].shape[2] if isinstance(latent, tuple) else latent.shape[2]
            selected = self._mla_select_cache(x, qr, state, prefix, indexer_projected, total)
        if type(frontier) is int and frontier + length != total:
            raise ValueError("GLM-5.3 MLA frontier differs from its latent plane")
        self.calls["pooled_indexer"] += int(selected is not None)
        q = self.weights.grouped(q, prefix + ".attn_k_b.weight", groups=64).transpose(
            0, 2, 1, 3
        )
        return q, selected, latent

    def _mla_select_cache(self, x, qr, cache, prefix, projected, total):
        """Append the raw indexer rows to ``cache[1]`` once and select."""
        if self.indexer_pools == "recompute":
            # Today's law, verbatim (pooled_selection_reference).
            selected = pooled_selection(x, qr, cache[1], self.weights, prefix, projected=projected)
            if selected is not None:
                self.calls["indexer_pool_recompute_full"] += 1
                self.calls["indexer_select_argsort"] += 1
            return selected
        pools = None
        if self.indexer_pools in INCREMENTAL_POOLS:
            # Fresh and serialized-restored caches alike: the first extend
            # pools every committed raw row once, then only the suffix.
            if cache.pools is None:
                cache.pools = Glm53PoolCache()
            pools = cache.pools
        before = self._selection_snapshot()
        selected = pooled_selection(
            x, qr, cache[1], self.weights, prefix, projected=projected, pools=pools,
            pool_law="fixed_order", select=self.indexer_select, block=self._select_block(),
        )
        self._count_selection(before, selected, pools)
        if selected is not None and self.indexer_pools == "incremental_v1_audit":
            raw = cache[1].keys
            self._audit(x, qr, prefix, projected, pools, lambda a, z: raw[0, 0, a:z], total)
        return selected

    def _mla_select_frontier(self, x, qr, latent, frontier, prefix, projected):
        """Tree frontier: append latent and raw rows once, then select over them."""
        if self.indexer_pools not in INCREMENTAL_POOLS or self.indexer_select == "argsort":
            raise ValueError("GLM-5.3 MLA frontiers require incremental pools and topk_v1 selection")
        packed = pack_indexer_rows(x, self.weights, prefix, projected=projected)
        frontier.append(latent, packed)
        total = frontier.offset
        before = self._selection_snapshot()
        selected = select_from_rows(
            x, qr, self.weights, prefix, total=total, raw_rows_fn=frontier.raw_rows,
            pools=frontier.pools, pool_law="fixed_order", select=self.indexer_select,
            block=None, projected=projected,
        )
        self._count_selection(before, selected, frontier.pools)
        if selected is not None and self.indexer_pools == "incremental_v1_audit":
            self._audit(x, qr, prefix, projected, frontier.pools, frontier.raw_rows, total)
        source = frontier.latent_source() if selected is not None else frontier.dense_keys()
        return selected, source, total

    def _audit(self, x, qr, prefix, projected, pools, raw_rows_fn, total):
        """incremental_v1_audit (tests and GPU checks only; syncs): pools equal
        a full fixed-order recompute bit for bit, and no pool score is
        nonfinite (a model fault the selection would otherwise mask, 4.1)."""
        pooled = pools.pooled[:, : pools.complete]
        audit_pools(pooled, raw_rows_fn, total, self.weights.array(prefix + ".indexer.kpool_ape"))
        b, length, _ = x.shape
        q = (projected["q"] if "q" in projected
             else self._linear(qr, prefix, "indexer.attn_q_b.weight")).reshape(b, length, 32, 128)
        weights = (projected["proj"] if "proj" in projected
                   else self._linear(x, prefix, "indexer.proj.weight")) * 32**-0.5
        if not mx.all(mx.isfinite(indexer.pool_scores(q, weights, pooled))).item():
            raise ValueError("GLM-5.3 indexer produced a nonfinite pool score")
        self.calls["indexer_pool_audit"] += 1

    def _selection_snapshot(self):
        # Run the select kernel's first-use gate before the snapshot: its
        # synthetic launches are not this call's selection.
        if self.indexer_select == "topk_v1_kernel":
            indexer.select_kernel_ready()
        return dict(indexer._SCHEDULED)

    def _select_block(self):
        # Blocked running top-k only bounds prompt-phase score memory; it is
        # the topk_v1 law (argsort selects globally, design 5.2).
        if sparse_phase() == "prompt" and self.indexer_select != "argsort":
            return SELECT_BLOCK
        return None

    def _count_selection(self, before, selected, pools):
        if selected is None:
            return
        after = indexer._SCHEDULED
        for source, name in _INDEXER_COUNTERS:
            self.calls[name] += after.get(source, 0) - before.get(source, 0)
        if pools is None:
            self.calls["indexer_pool_recompute_full"] += 1
        elif pools.last_computed:
            # From pool 0: a fresh cache crossing TOPK or a restored cache.
            self.calls["indexer_pool_recompute_full" if pools.last_from == 0 else "indexer_pool_extend"] += 1
            self.calls["indexer_pools_computed"] += pools.last_computed

    def _mla_masked(self, q, latent, selected, stop_query, total, start_query=0):
        """Today's consumer for queries [start_query, stop_query): 128-query
        pieces, key trim, boolean put_along_axis mask (or the causal mask when
        ``selected`` is None) and dense SDPA. A query with no valid id gets a
        finite row here (MLX's SDPA maps a boolean mask's false entries to
        finfo.min), not NaN; it is the reference law and stays as it is.
        ``start_query`` (a multiple of 128, so the pieces keep their places)
        skips leading rows another consumer computes."""
        if start_query % 128:
            raise ValueError("GLM-5.3 masked pieces start on a 128-query boundary")
        b, length = q.shape[0], q.shape[2]
        q8 = isinstance(latent, tuple)
        pieces = []
        for start in range(start_query, stop_query, 128):
            stop = min(start + 128, stop_query)
            # A piece reads only the keys its last query can see: the later
            # keys were fully masked, and scoring them made prefill attention
            # work grow with chunk x context. Bit-identical (w1-1k/w1-2048).
            seen = total - length + stop
            keys = tuple(p[:, :, :seen] for p in latent) if q8 else latent[:, :, :seen]
            if selected is None:
                pos = total - length + mx.arange(start, stop)
                mask = mx.arange(seen)[None, :] <= pos[:, None]
                mask = mask[None, None]
            else:
                ids = selected[:, start:stop]
                # Extra dump column keeps -1 padding outside the visible keys.
                safe = mx.where(ids >= 0, ids, seen)
                mask = mx.zeros((b, stop - start, seen + 1), dtype=mx.bool_)
                mask = mx.put_along_axis(mask, safe, mx.array(True), axis=-1)[
                    ..., :seen
                ][:, None]
            q_piece = q[:, :, start:stop]
            if pieces and getattr(self, "mla_serial_pieces", False):
                # Research: order the pieces so one piece's unfused 512-wide
                # score matrix is alive at a time. Same arithmetic.
                q_piece = mx.depends(q_piece, pieces[-1])
            pieces.append(
                _latent_q8_attention(q_piece, keys, 256**-0.5, mask)
                if q8
                else mx.fast.scaled_dot_product_attention(
                    q_piece, keys, keys, scale=256**-0.5, mask=mask
                )
            )
        if selected is not None:
            self.calls["sparse_attention_mask"] += 1
            if sparse_phase() == "prompt":
                self.calls["sparse_prefill_dense_pieces"] += len(pieces)
        return mx.concatenate(pieces, axis=2)

    def _mla_attend(self, q, selected, source):
        """The consumer of the phase contract: q (1, 64, L, 512) -> (1, 64, L, 512)."""
        length = q.shape[2]
        if isinstance(source, sparse_attention.TreeLatentView):
            if selected is None:
                raise ValueError("GLM-5.3 tree latent view without ids; use the frontier's dense keys")
            if sparse_phase() == "prompt":
                raise ValueError("GLM-5.3 MLA frontiers are verify-only")
            return self._sparse_decode(q, source, selected)
        q8 = isinstance(source, tuple)
        total = source[0].shape[2] if q8 else source.shape[2]
        if sparse_phase() == "prompt":
            if sparse_attention.prefill_law(self.sparse_prefill) == "fused_v2":
                return self._mla_prefill_v2(q, selected, source, total)
            if selected is None:
                return self._mla_masked(q, source, None, length, total)
            start = sparse_attention.prefill_gathered_start(
                total, length, sparse_attention.PREFILL_THRESHOLDS[self.sparse_prefill]
            )
            if start == length:
                return self._mla_masked(q, source, selected, length, total)
            if q8:
                raise ValueError(Q8_REFUSAL)
            gathered = sparse_attention.sparse_prefill(
                q[:, :, start:], source, selected[:, start:], counters=self.calls
            )
            if start == 0:
                return gathered
            dense = self._mla_masked(q, source, selected, start, total)
            return mx.concatenate([dense, gathered], axis=2)
        if selected is None:
            if self.sparse_attention == "fused_v2":
                # Dense rows (T <= 2048) under the v2 decode law: causal ids
                # through the same kernel, as the prompt rows below 2049.
                if q8:
                    raise ValueError(Q8_REFUSAL)
                self.calls["sparse_attention_fused_v2_dense_rows"] += length
                return self._sparse_decode(q, source, None)
            return self._mla_masked(q, source, None, length, total)
        if self.sparse_attention == "mask_reference":
            return self._mla_masked(q, source, selected, length, total)
        if q8:
            raise ValueError(Q8_REFUSAL)
        return self._sparse_decode(q, source, selected)

    def _mla_prefill_v2(self, q, selected, source, total):
        """Prompt rows under ``gathered_v2_from_<n>``, by query position alone:

        * ``pos < 2048`` (``pos + 1 < 2049``; every visible key fits): the
          fused_v2 kernel with dense causal ids (``dense_row_ids``), whether or
          not this chunk selects - a chunk that crosses 2048 keys has ids for
          these rows, a chunk below it has none, and both must give the same
          bits, so cold and warm chunkings agree;
        * ``2048 <= pos < n - 1``: the dense ``_mla_masked`` pieces (round 1,
          5.1), as under the v1 thresholds;
        * ``pos >= n - 1``: fused_v2 blocks of PREFILL_BLOCK_V2 on the selected ids.
        """
        q8 = isinstance(source, tuple)
        if q8:  # the gathered sparse paths require the fp32 latent plane
            raise ValueError(Q8_REFUSAL)
        length = q.shape[2]
        first = total - length
        dense_stop = min(length, max(0, TOPK - first))
        start = (length if selected is None else sparse_attention.prefill_gathered_start(
            total, length, sparse_attention.PREFILL_THRESHOLDS[self.sparse_prefill]))
        start = max(start, dense_stop)
        pieces = []
        if dense_stop:
            # The source's first ``first + dense_stop <= 2048`` rows: the
            # query rows are its last rows, as the dense-ids path requires.
            self.calls["sparse_attention_fused_v2_dense_rows"] += dense_stop
            pieces.append(sparse_attention.sparse_prefill(
                q[:, :, :dense_stop], source[:, :, : first + dense_stop], None,
                block=sparse_attention.PREFILL_BLOCK_V2, counters=self.calls, policy="fused_v2"))
        if start > dense_stop:
            aligned = dense_stop - dense_stop % 128
            masked = self._mla_masked(q, source, selected, start, total, start_query=aligned)
            pieces.append(masked[:, :, dense_stop - aligned :])
        if start < length:
            pieces.append(sparse_attention.sparse_prefill(
                q[:, :, start:], source, selected[:, start:],
                block=sparse_attention.PREFILL_BLOCK_V2, counters=self.calls, policy="fused_v2"))
        return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=2)

    def _sparse_decode(self, q, source, selected):
        """Decode-phase gathered consumer in groups of at most 8 query rows.

        gathered_v1 computes each row by the same one-row arithmetic whatever
        its companions, so grouping is bitwise neutral; group g+1's ids wait
        on group g's output so one group's gather is live at a time (the
        workspace Glm53CacheBudget books). fused_decode_v1 groups itself.
        """
        length, group = q.shape[2], DECODE_ROW_GROUP
        if self.sparse_attention != "gathered_v1" or length <= group:
            return sparse_attention.sparse_decode(
                q, source, selected, policy=self.sparse_attention, counters=self.calls
            )
        out = []
        for start in range(0, length, group):
            ids = selected[:, start : start + group]
            if out:
                ids = mx.depends(ids, out[-1])
            out.append(sparse_attention.sparse_decode(
                q[:, :, start : start + group], source, ids, policy="gathered_v1", counters=self.calls
            ))
        return mx.concatenate(out, axis=2)

    def _mla_finish(self, out, prefix, *, _return_pre_output=False):
        """(1, 64, L, 512) -> grouped v_b -> (1, L, 16384) -> attn_output."""
        b, length = out.shape[0], out.shape[2]
        out = out.transpose(0, 2, 1, 3)
        out = self.weights.grouped(out, prefix + ".attn_v_b.weight", groups=64)
        self.calls["mla"] += 1
        out = out.reshape(b, length, -1)
        return out if _return_pre_output else self._linear(out, prefix, "attn_output.weight")

    def _mla(self, x, state, prefix, *, frontier=None, _return_pre_output=False):
        if getattr(self, "mla_fused", "reference") == "decode_v1":
            out = self._mla_fused_decode(x, state, prefix, frontier)
            if out is not None:
                return self._mla_finish(out, prefix, _return_pre_output=_return_pre_output)
        q, selected, source = self._mla_prepare(x, state, prefix, frontier=frontier)
        out = self._mla_attend(q, selected, source)
        return self._mla_finish(out, prefix, _return_pre_output=_return_pre_output)

    def _mla_fused_admits(self, x, state, frontier):
        """glm53_mla_fused=decode_v1 admission before any projection is built."""
        from . import glm53_mla_fused as fused

        return (
            x.shape[:2] == (1, 1)
            and isinstance(state, Glm53MLACache)
            and not isinstance(state.caches[0], Glm53LatentQ8Cache)
            and type(state.caches[1].offset if frontier is None else frontier) is int
            and sparse_phase() != "prompt"
            and self.indexer_pools == "incremental_v1"
            and self.indexer_select == "topk_v1_kernel"
            and self.sparse_attention in ("gathered_v1", "fused_decode_v1", "fused_v2")
            and fused.ready(self.args.rms_norm_eps)
            and indexer.select_kernel_ready()
        )

    def _mla_fused_decode(self, x, state, prefix, frontier):
        """glm53_mla_fused=decode_v1: one B1 decode token's MLA body up to the
        attention output (1, 64, 1, 512), or None to run the reference chain.

        The reference's projections, cache appends, pool bookkeeping, select
        kernel, q x pooled matmul and SDPA call, in the same order on the same
        operands; the glue between them runs in glm53_mla_fused's four
        kernels, each gated bitwise against that glue. Admitted only under the
        arm4 sparse laws (incremental_v1, topk_v1_kernel, gathered_v1) on the
        Metal GPU; everything else declines (counted) and runs the reference.
        Under glm53_sparse_attention=fused_decode_v1 or fused_v2 the selected ids
        feed that consumer's own kernel instead of gather + SDPA (its law, not
        gathered_v1's); a dense step (T <= 2048) takes _mla_attend's dense rows.
        """
        from . import glm53_mla_fused as fused

        if not self._mla_fused_admits(x, state, frontier):
            self.calls["mla_fused_declined"] += 1
            return None
        b, length = 1, 1
        if frontier is None:
            frontier = state[1].offset
        weights = self.weights
        project = lambda value, name: self._linear(value, prefix, name)
        grouped = (
            self._mla_grouped_projections(x, frontier + length > TOPK, prefix)
            if getattr(self, "mla_group", False)
            else None
        )
        qa = grouped["q_a"] if grouped else project(x, "attn_q_a.weight")
        kv_a = grouped["kv_a"] if grouped else project(x, "attn_kv_a_mqa.weight")
        key = grouped["key"] if grouped else weights.linear(prefix + ".indexer.attn_k.weight")(x)
        gate = weights.linear(prefix + ".indexer.kpool_gate")(x)
        if not fused.prep_admits(qa, kv_a, key, gate):
            # Unevaluated projections are dropped; the reference rebuilds them.
            self.calls["mla_fused_declined"] += 1
            return None
        qr, latent_row, raw_row = fused.prep(
            qa, kv_a, key, gate,
            weights.array(prefix + ".attn_q_a_norm.weight"),
            weights.array(prefix + ".attn_kv_a_norm.weight"),
            weights.array(prefix + ".indexer.k_norm.weight"),
            weights.array(prefix + ".indexer.k_norm.bias"),
            rms_eps=self.args.rms_norm_eps,
        )
        # _mla_prepare's q path, verbatim.
        projected = {}
        if grouped and "proj" in grouped:
            projected["proj"] = grouped["proj"]
            second = weights.multi_linear(
                [qr, qr], [prefix + ".attn_q_b.weight", prefix + ".indexer.attn_q_b.weight"])
            if second is not None:
                q_full, projected["q"] = second
            else:
                q_full = project(qr, "attn_q_b.weight")
        else:
            q_full = project(qr, "attn_q_b.weight")
        q = q_full.reshape(b, length, 64, 256)
        # The two appends, as _mla_prepare and pooled_selection make them.
        if getattr(self, "kv_append", "reference") != "reference":
            kv_store.engage(state.caches, self.kv_append)
        self.calls["kv_append_ring"] += int(kv_store.is_ring(state[0]))
        latent, _ = state[0].update_and_fetch(latent_row, mx.zeros((b, 1, length, 0), dtype=x.dtype))
        full, _ = state[1].update_and_fetch(raw_row, mx.zeros((b, 1, length, 0), dtype=x.dtype))
        total = latent.shape[2]
        if frontier + length != total or full.shape[2] != total:
            raise ValueError("GLM-5.3 MLA frontier differs from its latent plane")
        q = weights.grouped(q, prefix + ".attn_k_b.weight", groups=64).transpose(0, 2, 1, 3)
        if total <= TOPK:
            # No ids exist yet: the reference's dense causal consumer. The raw
            # append joins this step's graph instead of waiting for a read.
            self.calls["mla_fused_dense"] += 1
            return self._mla_attend(q, None, mx.depends(latent, full))
        # select_from_rows at L = 1 under incremental_v1 + topk_v1_kernel.
        if state.pools is None:
            state.pools = Glm53PoolCache()
        pools = state.pools
        before = self._selection_snapshot()
        iq = (projected["q"] if "q" in projected
              else weights.linear(prefix + ".indexer.attn_q_b.weight")(qr)).reshape(b, length, 32, 128)
        proj = projected["proj"] if "proj" in projected else weights.linear(prefix + ".indexer.proj.weight")(x)
        ape = weights.array(prefix + ".indexer.kpool_ape")
        scheduled = fused._SCHEDULED["pool"]
        pooled = pools.extend(
            lambda a, z: full[0, 0, a:z], total, ape,
            pool_fn=lambda raw, ape_: fused.pool_one(raw, ape_) if raw.shape == (4, 256)
            else indexer.pool_fixed_order(raw, ape_),
        )
        self.calls["mla_fused_pool"] += fused._SCHEDULED["pool"] - scheduled
        count = min(TOPK // indexer.KPOOL, pooled.shape[1])
        mm = iq[:, 0:1] @ pooled[:, None].swapaxes(-1, -2)
        scores = fused.pool_scores(mm, proj)
        ids = indexer.select_expand(scores, mx.array([total - 1], dtype=mx.int32), count)
        self._count_selection(before, ids, pools)
        self.calls["pooled_indexer"] += 1
        # Tie the raw append into this step (a later pool read would otherwise
        # pull several steps' deferred appends into one step); no value changes.
        ids = mx.depends(ids, full)
        if self.sparse_attention in ("fused_decode_v1", "fused_v2"):
            # The consumer's own kernel on the selected ids (its law).
            self.calls["mla_fused"] += 1
            return self._sparse_decode(q, latent, ids[None])
        rows, mask = fused.gather(latent, ids)
        self.calls["sparse_attention_gathered"] += 1
        self.calls["sparse_gathered_rows"] += ids.shape[-1]
        sparse_attention._SCHEDULED["gathered"] += 1
        self.calls["mla_fused"] += 1
        return sparse_attention.gathered_attention_masked(q, rows, mask)

    def _mlp(self, x, prefix, index):
        # Host memo: this layer's tensor names, derived once per prefix.
        memo = self.__dict__.get("_mlp_names")
        if memo is None:
            memo = self.__dict__["_mlp_names"] = {}
        names = memo.get(prefix)
        if names is None:
            names = memo[prefix] = {
                suffix: tuple(prefix + ".ffn_" + p + suffix + ".weight" for p in ("gate", "up", "down"))
                for suffix in ("", "_shexp")
            }
            names["router"] = prefix + ".ffn_gate_inp.weight"
            names["fold"] = prefix + ".ffn_{}_shexp.weight"
        linear = self.weights.linear

        def shared(suffix):
            gate_name, up_name, down_name = names[suffix]
            gate = mx.minimum(linear(gate_name)(x), 10)
            up = mx.clip(linear(up_name)(x), -10, 10)
            return linear(down_name)(nn.silu(gate) * up)

        if index < 3:
            return shared("")
        routed_ids = self._route_gemv(x, prefix, names["router"])
        ids, selected = (
            routed_ids if routed_ids is not None else self._route(linear(names["router"])(x), prefix)
        )
        if getattr(self, "shared_fold", False):
            folded = self.weights.moe_with_shared(
                x.reshape(-1, x.shape[-1]),
                ids.reshape(-1, 8),
                selected.reshape(-1, 8),
                prefix,
                names["fold"],
            )
            if folded is not None:
                return folded.reshape(x.shape)
        routed = self.weights.moe(
            x.reshape(-1, x.shape[-1]),
            ids.reshape(-1, 8),
            selected.reshape(-1, 8),
            prefix,
        ).reshape(x.shape)
        return routed + shared("_shexp")

    def _f32_rows(self, name):
        """The native F32 [N, K] row-major matrix behind ``weights.linear(name)``'s
        operand (its transposed view), or None for any other storage."""
        memo = self.__dict__.get("_f32_rows_memo")
        if memo is None:
            memo = self.__dict__["_f32_rows_memo"] = {}
        if name not in memo:
            weights = self.weights
            tensor = getattr(weights, "tensors", {}).get(name)
            rows = None
            if tensor is not None and tensor.ggml_type == 0:
                weights.linear(name)  # registers the reference's F32 operand
                operand = getattr(weights, "_native_linear_fp32", {}).get(name)
                if operand is not None and operand.dtype == mx.float32 and operand.ndim == 2:
                    rows = operand.T
            memo[name] = rows
        return memo[name]

    def _f32_operand(self, name):
        """``_f32_rows(name)``, or under glm53_f32_weights=bf16_v1 its exact
        BF16 copy (built and bit-checked once, eagerly; compile regions warm
        every policy with one eager step before tracing). A copy that is not
        exact is refused and the F32 rows are returned."""
        rows = self._f32_rows(name)
        if rows is None or getattr(self, "f32_weights", "reference") != "bf16_v1":
            return rows
        memo = self.__dict__.get("_f32_bf16_memo")
        if memo is None:
            memo = self.__dict__["_f32_bf16_memo"] = {}
        if name not in memo:
            half = mx.contiguous(rows.astype(mx.bfloat16))
            back = mx.view(half.astype(mx.float32), mx.uint32)
            exact = mx.array_equal(back, mx.view(mx.contiguous(rows), mx.uint32))
            mx.eval(half, exact)
            memo[name] = half if bool(exact.item()) else None
        half = memo[name]
        if half is None:
            self.calls["f32_bf16_declined"] += 1
            return rows
        self.calls["f32_bf16_operand"] += 1
        return half

    def _route_gemv(self, x, prefix, router_name):
        """glm53_router=fused_v2: (ids, weights) from x in one launch, or None."""
        if getattr(self, "router", "reference") != "fused_v2":
            return None
        from . import glm53_glue_metal as glue

        weight = self._f32_operand(router_name)
        bias = self.weights.array(prefix + ".exp_probs_b.bias")
        if x.shape[:-1] == (1, 1) and glue.router_gemv_admits(x, weight, bias, 8):
            self.calls["router_fused_v2"] += 1
            # A fresh counter per launch (glue.fused_router_gemv default).
            return glue.fused_router_gemv(x, weight, bias, topk=8, scale=2.5)
        self.calls["router_fused_v2_declined"] += 1
        return None

    def _route(self, logits, prefix):
        """Top-8 expert ids and their normalized x2.5 unbiased scores per row."""
        bias = self.weights.array(prefix + ".exp_probs_b.bias")
        if getattr(self, "router", "reference") in ("fused_v1", "fused_v2"):
            from .glm53_glue_metal import fused_router, router_admits

            if router_admits(logits, bias, 8):
                self.calls["router_fused"] += 1
                return fused_router(logits, bias, topk=8, scale=2.5)
            self.calls["router_fused_declined"] += 1
        scores = gate_sigmoid(logits)
        choice = scores + bias
        ids = mx.argsort(-choice, axis=-1)[..., :8].astype(mx.uint32)
        selected = mx.take_along_axis(scores, ids, axis=-1)
        selected = selected / selected.sum(-1, keepdims=True) * 2.5
        return ids, selected

    def __call__(
        self, inputs, cache=None, *, logits_to_keep=None, _taps=None, _body_only=False
    ):
        if inputs.ndim != 2 or inputs.shape[0] != 1 or inputs.shape[1] < 1:
            raise ValueError("GLM-5.3 packed ordinary text currently requires B1")
        if cache is None:
            cache = self.make_cache()
        if len(cache) != len(self.layers):
            raise ValueError("GLM-5.3 cache layer count differs")
        # Frontier checks use Python ints only: a serving KV cache may hold its
        # offset as an mx.array, and every truth test of one forces a GPU sync
        # in the middle of graph build (~110 per decode step serialized build
        # and execution: server decode 24 vs 32 tok/s direct). The recurrent
        # KDA offsets are always ints; array-valued MLA offsets are not tested.
        offset = next(c.offset for c, layer in zip(cache, self.layers) if layer.is_linear)
        if offset + inputs.shape[1] > self.args.max_position_embeddings:
            raise ValueError("GLM-5.3 request exceeds route context")
        for c, layer in zip(cache, self.layers):
            if layer.is_linear and (c.identity != self.identity or c.offset != offset):
                raise ValueError("GLM-5.3 recurrent cache identity/frontier mismatch")
            if not layer.is_linear and (
                len(c.caches) != 2
                or any(isinstance(kv.offset, int) and kv.offset != offset for kv in c.caches)
            ):
                raise ValueError("GLM-5.3 latent/indexer cache frontier mismatch")
        h = self.weights.embedding(inputs.astype(mx.uint32))
        h = mx.contiguous(
            mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], 4, h.shape[-1]))
        )
        # glm53_compile_regions=v1: B1 decode layer bodies replay a traced graph
        # (glm53_compile_regions); None keeps the eager body below.
        regions = None
        if getattr(self, "compile_regions", "off") != "off":
            from . import glm53_compile_regions

            regions = glm53_compile_regions.active(
                self, inputs, cache, plain=_taps is None and not _body_only
            )
        prompt_eval_every = (getattr(self, "prompt_eval_every", 0)
                             if sparse_phase() == "prompt" and _taps is None and not _body_only else 0)
        for index, (layer, c) in enumerate(zip(self.layers, cache)):
            if regions is not None:
                h = regions.layer(index, layer, c, h, offset)
                if _taps is not None and index in _taps:
                    _taps[index] = h.mean(-2)
                continue
            prefix = f"blk.{index}"
            x, post, comb = self._hc_in(h, prefix + ".hc_attn", prefix + ".attn_norm.weight")
            branch = (
                self._kda(x, c, prefix) if layer.is_linear
                else self._mla(x, c, prefix, frontier=offset)
            )
            h, x, post, comb = self._hc_mid(
                branch, h, post, comb, prefix + ".hc_ffn", prefix + ".ffn_norm.weight"
            )
            h = hc_expand(
                self._mlp(x, prefix, index), h, post, comb, fused=self.hc_fused
            )
            if _taps is not None and index in _taps:
                _taps[index] = h.mean(-2)
            if prompt_eval_every and (index + 1) % prompt_eval_every == 0:
                # glm53_prompt_eval: evaluate each layer of a prompt chunk on
                # its own (same kernels, same bits). With a large command-
                # buffer op cap and no byte trigger (MLX2_COMMAND_BUFFERS=
                # ops100_v1) a chunk's buffers otherwise hold ~100 kernels'
                # prefill temporaries alive at once (+24 GB peak at 2048-row
                # chunks); a per-layer evaluation commits and retires them.
                # A cache without serializable state (research q8 latent)
                # contributes only h; its writes stay lazy.
                mx.eval(h, getattr(c, "state", []) if not isinstance(c, Glm53MLACache) or not getattr(
                    self, "mla_kv_q8", False) else [])
                self.calls["prompt_layer_evals"] += 1
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

    def _hc_in(self, h, half, norm, _fused=True):
        """HC collapse and the half's norm: (normalized x, post, comb).

        hc_mega runs the split, weighted sum and norm as one kernel per row
        after the stock flat norm and F32 mixing matmul; equal to the fused
        composition bit for bit (glm53_hc_mega.self_check, first use).
        """
        eps = self.args.rms_norm_eps
        if _fused and getattr(self, "hc_fused_policy", "reference") != "reference":
            if self.hc_mega and h.shape[:-2] == (1, 1):
                fused = self._hc_fused(h, half, norm, None)
                if fused is not None:
                    return fused[1:]
            else:
                self.calls["hc_fused_declined"] += 1  # multi-row boundary
        if self.hc_mega and h.shape[:-2] == (1, 1):
            # Host memo (per half): the module and the derived tensor names are
            # fixed; the weights are still read through the provider each step.
            memo = self.__dict__.get("_hc_in_memo")
            if memo is None:
                from . import glm53_hc_mega

                memo = self.__dict__["_hc_in_memo"] = {None: glm53_hc_mega}
            names = memo.get(half)
            if names is None:
                names = memo[half] = (half + "_fn", half + "_scale", half + "_base")
            mega = memo[None]
            mega.self_check()
            weights = self.weights
            z = rms(h.reshape(1, 1, -1), None, eps)
            mixes = weights.linear(names[0])(z)
            x, _, post, comb = mega.post(
                h, mixes, weights.array(names[1]), weights.array(names[2]),
                weights.array(norm), rms_eps=eps)
            return x, post, comb
        x, post, comb = hc_collapse(h, self.weights, half, eps, fused=self.hc_fused)
        return rms(x, self.weights.array(norm), eps), post, comb

    def _hc_mid(self, branch, h, post, comb, half, norm):
        """The previous half's HC expand, then this half's HC-in: (h, x, post, comb).

        glm53_hc_fused folds the expand, the flat norm and the mixing gemv
        (and under tail_v1 the mega post step) into one launch, same bits.
        """
        if getattr(self, "hc_fused_policy", "reference") != "reference":
            if h.shape[:-2] == (1, 1):
                fused = self._hc_fused(h, half, norm, (branch, h, post, comb))
                if fused is not None:
                    return fused
            else:
                self.calls["hc_fused_declined"] += 1  # multi-row boundary
            h = hc_expand(branch, h, post, comb, fused=self.hc_fused)
            return (h, *self._hc_in(h, half, norm, _fused=False))
        h = hc_expand(branch, h, post, comb, fused=self.hc_fused)
        return (h, *self._hc_in(h, half, norm))

    def _hc_fused(self, h, half, norm, expand):
        """One glm53_hc_fused boundary (h, x, post, comb), or None when declined."""
        from . import glm53_hc_fused as fused

        memo = self.__dict__.get("_hc_fused_memo")
        if memo is None:
            memo = self.__dict__["_hc_fused_memo"] = {}
        entry = memo.get(half)
        if entry is None:
            entry = memo[half] = (half + "_fn", (half + "_scale", half + "_base"))
        fn_name, (scale_name, base_name) = entry
        fn = self._f32_operand(fn_name)
        stream = h if expand is None else expand[1]
        if not (
            self.hc_mega
            and fused.admits(stream, fn)
            and (expand is None or (
                tuple(expand[0].shape) == (1, 1, fused.DIM)
                and tuple(expand[2].shape) == (1, 1, 4)
                and tuple(expand[3].shape) == (1, 1, 4, 4)
            ))
        ):
            self.calls["hc_fused_declined"] += 1
            return None
        fused.self_check()
        eps, weights = self.args.rms_norm_eps, self.weights
        scale, base, weight = weights.array(scale_name), weights.array(base_name), weights.array(norm)
        if self.hc_fused_policy == "tail_v1":
            # A fresh arrival counter per launch (fused.boundary default).
            h, x, _, post, comb = fused.boundary(h, fn, scale, base, weight, rms_eps=eps, expand=expand)
            self.calls["hc_fused_tail"] += 1
            return h, x, post, comb
        h, mixes = fused.project(h, fn, rms_eps=eps, expand=expand)
        x, _, post, comb = fused.mega.post(h, mixes, scale, base, weight, rms_eps=eps)
        self.calls["hc_fused_boundary"] += 1
        return h, x, post, comb

    def _prompt_scope(self, tokens):
        """Bind the weights' prefill laws to one explicit prompt forward.

        Decode, verification and AST rows never inherit them by width alone.
        """
        from contextlib import ExitStack

        from ..gguf.prefill_scope import dense_prefill_context
        from ..gguf.segmented_prefill import expert_prefill_context

        stack = ExitStack()
        stack.enter_context(
            expert_prefill_context(self.weights.expert_prefill_policy, tokens=tokens)
        )
        stack.enter_context(
            dense_prefill_context(self.weights.dense_prefill_policy, tokens=tokens)
        )
        stack.enter_context(sparse_attention_phase("prompt"))
        return stack

    def prefill_forward(self, inputs, cache=None):
        with self._prompt_scope(inputs.shape[1]):
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
        with self._prompt_scope(inputs.shape[1]):
            return self.forward_with_taps(
                inputs, cache, capture_layers, body_only=True
            )[1]

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
