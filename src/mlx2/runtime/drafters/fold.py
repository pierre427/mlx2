"""MLX FOLD attention for block-parallel external drafters.

The target and verifier are unchanged.  Draft attention retains exact local
positions and reads older committed target features as position-free memory,
with separately normalized mass (FOLD equations 1--4).
"""

from __future__ import annotations

import mlx.core as mx

from ..cache_operations import CacheOperationsContract, CachePlaneDeclaration
from ..models.cache import KVCache
from .fold_reference import FOLDPolicy

FOLD_NUMERICS_VERSION = "mlx2.fold-drafter.v2"

FOLD_KV_OPERATIONS = CacheOperationsContract(
    family="fold_kv",
    topology="kv",
    state_revision="mlx2.fold-kv.v1",
    cache_layout="rotated-kv-plus-unrotated-keys-bhsd",
    numerics="stored_dtype",
    planes=(
        CachePlaneDeclaration(
            "attention_kv", ("keys", "values", "raw_keys")
        ),
    ),
    exact_prefix=True,
    restores_at_checkpoints=False,
    rollback="kv_trim",
)


class FOLDKVCache(KVCache):
    """Full rotated K/V plus the unrotated keys required by far memory."""

    cache_operations_contract = FOLD_KV_OPERATIONS
    _RECOVERY_APPEND_ONLY_FIELDS = (
        ("keys", -2, "offset"),
        ("values", -2, "offset"),
        ("raw_keys", -2, "offset"),
    )

    def __init__(self):
        super().__init__()
        self.raw_keys = None

    def append_raw_keys(self, keys):
        previous = self.offset
        # As append_context_kv: an empty serializable seed (float32) must not
        # promote the first real keys' dtype for the life of the plane.
        if previous == 0 and self.raw_keys is not None and self.raw_keys.shape[2] == 0:
            self.raw_keys = None
        if self.raw_keys is None or previous + keys.shape[2] > self.raw_keys.shape[2]:
            batch, heads, _, dim = keys.shape
            steps = (self.step + keys.shape[2] - 1) // self.step
            fresh = mx.zeros((batch, heads, steps * self.step, dim), keys.dtype)
            if self.raw_keys is None:
                self.raw_keys = fresh
            else:
                if previous % self.step:
                    self.raw_keys = self.raw_keys[..., :previous, :]
                self.raw_keys = mx.concatenate((self.raw_keys, fresh), axis=2)
        self.raw_keys[..., previous : previous + keys.shape[2], :] = keys

    def raw_keys_and_values(self):
        if self.raw_keys is None:
            return self.raw_keys, None
        end = self.offset
        return self.raw_keys[..., :end, :], self.values[..., :end, :]

    @property
    def state(self):
        return self.keys, self.values, self.raw_keys

    @state.setter
    def state(self, value):
        same_storage = getattr(self, "keys", None) is value[0]
        previous_offset = getattr(self, "offset", None) if same_storage else None
        self.keys, self.values, self.raw_keys = value
        self.offset = (
            previous_offset
            if previous_offset is not None
            else 0
            if self.keys is None
            else self.keys.shape[2]
        )

    @property
    def nbytes(self):
        return sum(
            int(value.nbytes)
            for value in (self.keys, self.values, self.raw_keys)
            if value is not None
        )


def _expand_gqa(value, heads):
    if value.shape[1] == heads:
        return value
    if heads % value.shape[1]:
        raise ValueError("FOLD query heads must be divisible by KV heads")
    return mx.repeat(value, heads // value.shape[1], axis=1)


def _branch(query, keys, values, scale, concentration=1.0):
    """Return branch output and log mass in float32."""
    heads = query.shape[1]
    keys = _expand_gqa(keys, heads).astype(mx.float32)
    values = _expand_gqa(values, heads).astype(mx.float32)
    logits = mx.matmul(query.astype(mx.float32), keys.transpose(0, 1, 3, 2)) * float(
        scale
    )
    weighted = logits * float(concentration)
    log_mass = mx.logsumexp(weighted, axis=-1) / float(concentration)
    output = mx.matmul(mx.softmax(weighted, axis=-1), values)
    return output, log_mass


def fold_attention(
    queries,
    prop_keys,
    prop_values,
    *,
    rope,
    cache: FOLDKVCache,
    policy: FOLDPolicy,
    scale: float,
    local_context: int | None = None,
):
    """Read one proposal block from a committed FOLD cache."""
    context = int(cache.offset)
    block = int(queries.shape[2])
    if local_context is not None and (
        type(local_context) is not int or local_context < 1
    ):
        raise ValueError("FOLD local context must be a positive integer")
    exact_context = policy.window if local_context is None else local_context
    if context <= exact_context:
        q = rope(queries, offset=context)
        pk = rope(prop_keys, offset=context)
        keys, values = cache.keys_and_values()
        return mx.fast.scaled_dot_product_attention(
            q,
            mx.concatenate((keys, pk), axis=2),
            mx.concatenate((values, prop_values), axis=2),
            scale=scale,
            mask=None,
        ), False

    rotated_keys, context_values = cache.keys_and_values()
    raw_keys, _ = cache.raw_keys_and_values()
    if raw_keys is None or raw_keys.shape[2] != context:
        raise ValueError("FOLD unrotated-key cache is not aligned with draft context")
    q_local = rope(queries, offset=context)
    pk = rope(prop_keys, offset=context)
    distance = policy.window + block
    rows = []
    used_far = False
    for row in range(block):
        # far iff p-j >= W+B.  All proposal positions remain in the local
        # branch; the number of exact context entries varies across the block.
        far_count = (
            max(0, context - distance + row + 1)
            if local_context is None
            else max(0, context - local_context)
        )
        near_start = far_count
        local_keys = mx.concatenate((rotated_keys[..., near_start:, :], pk), axis=2)
        local_values = mx.concatenate(
            (context_values[..., near_start:, :], prop_values), axis=2
        )
        local, local_mass = _branch(
            q_local[..., row : row + 1, :], local_keys, local_values, scale
        )
        if far_count == 0:
            rows.append(local)
            continue
        used_far = True
        q_far = rope(queries[..., row : row + 1, :], offset=distance + row)
        far, far_mass = _branch(
            q_far,
            raw_keys[..., :far_count, :],
            context_values[..., :far_count, :],
            scale,
            concentration=policy.alpha,
        )
        maximum = mx.maximum(local_mass, far_mass)
        local_weight = mx.exp(local_mass - maximum)
        far_weight = mx.exp(far_mass - maximum)
        norm = local_weight + far_weight
        rows.append(
            (local * local_weight[..., None] + far * far_weight[..., None])
            / norm[..., None]
        )
    return mx.concatenate(rows, axis=2).astype(queries.dtype), used_far


def configure(model, value):
    """Install one validated FOLD policy on a DFlash-family model."""
    policy = FOLDPolicy.from_value(value)
    installed = getattr(model, "fold_policy", None)
    if installed is not None:
        if policy != installed:
            raise ValueError("FOLD policy changes require a fresh drafter and cache identity")
        return installed
    if policy is None:
        return None
    # The sliding attention layers alone are not enough: the drafter must
    # build the three-plane cache, append raw keys and reserve for them.
    if getattr(model, "supports_fold", False) is not True:
        raise ValueError(f"{type(model).__name__} does not support FOLD")
    if getattr(model, "draft_attention_windows", None) is not None:
        raise ValueError("FOLD conflicts with draft_attention_windows")
    selected = []
    for layer in model.layers:
        attention = layer.self_attn
        if attention.is_sliding and policy.sliding_layers == "window":
            continue
        if attention.is_sliding and int(attention.sliding_window) > policy.window:
            raise ValueError(
                "FOLD window must cover every selected native sliding window"
            )
        selected.append(attention)
    if not selected:
        raise ValueError("FOLD selected no draft attention layers")
    # A later incompatible layer must not leave a partly enabled drafter.
    for attention in selected:
        attention.fold_policy = policy
        attention.fold_calls = 0
        attention.fold_far_calls = 0
        attention.fold_max_context = 0
    model.fold_policy = policy
    return policy


def extra_memory_reservation(model, caches, *, append, pending_tokens, prefill):
    """Host-only incremental bound beyond the executor's ordinary two-plane KV.

    Reserve allocation-step rounding and unrotated K before the first append,
    including the previous pending context. The current eager-shaped lazy graph
    also retains per-row GQA repeats, FP32 branch arrays and attention scores;
    conservatively count every selected layer/row without assuming allocator
    reuse. This is an admission bound, not a measured device peak.
    """
    if any(type(value) is not int or value < 0 for value in (append, pending_tokens)):
        raise ValueError("invalid FOLD reservation geometry")
    if len(caches) != len(model.layers):
        raise ValueError("FOLD reservation cache topology mismatch")
    required = 0
    for layer, cache in zip(model.layers, caches):
        attention = layer.self_attn
        if getattr(attention, "fold_policy", None) is None:
            continue
        if not isinstance(cache, FOLDKVCache):
            raise TypeError("FOLD reservation requires paired three-plane cache")
        context = int(cache.offset) + pending_tokens + append
        capacity = ((context + cache.step - 1) // cache.step) * cache.step
        elements = int(attention.n_kv_heads) * int(attention.head_dim)
        # Four bytes follows the executor's conservative FP32/BF16 bound.
        growth = sum(max(0, capacity - (0 if value is None else int(value.shape[2])))
                     for value in cache.state) * elements * 4
        required += max(0, growth - 2 * append * elements * 4)
        if not prefill:
            # ``append`` is the target verification width: one committed
            # anchor row plus at most ``append - 1`` draft proposal rows.
            # FOLD runs only for those proposal rows.  Reserving the trained
            # checkpoint's maximum block here made a short final-budget round
            # look as expensive as a full-width round and could strand an
            # exact rollback retry after the failed round's buffers reduced
            # measured headroom.  The admitted append width is already an
            # upper bound when adaptive depth selects fewer proposals.
            block = max(0, append - 1)
            heads, dim = int(attention.n_heads), int(attention.head_dim)
            # K/V repeats plus FP32 casts, logits/concentration/softmax and
            # branch outputs/normalizers for local and far branches together.
            required += 4 * block * heads * (
                4 * (context + block) * dim + 6 * (context + block) + 8 * dim
            )
    return required


def status(model):
    policy = getattr(model, "fold_policy", None)
    if policy is None:
        return None
    layers = [
        layer.self_attn
        for layer in model.layers
        if getattr(layer.self_attn, "fold_policy", None) is not None
    ]
    return {
        "implemented": True,
        "qualified": False,
        "selected": True,
        "observed_used": any(layer.fold_far_calls for layer in layers),
        "policy": policy.as_dict(),
        "numerics": FOLD_NUMERICS_VERSION,
        "layers": len(layers),
        "calls": sum(layer.fold_calls for layer in layers),
        "far_memory_calls": sum(layer.fold_far_calls for layer in layers),
        "max_context": max((layer.fold_max_context for layer in layers), default=0),
        "cache_values_per_token_per_layer": 3,
    }


__all__ = [
    "FOLD_KV_OPERATIONS",
    "FOLD_NUMERICS_VERSION",
    "FOLDKVCache",
    "configure",
    "extra_memory_reservation",
    "fold_attention",
    "status",
]
