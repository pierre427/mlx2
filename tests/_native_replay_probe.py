"""Shared probes for ``inkling2_native_replay`` (CPU and Metal tests).

* ``same_state_run``: every decode step of the fixed step starts from a
  copy of the ordinary arm's state, so each step is an independent,
  shared-prefix comparison and every flip may be classified.
* ``drift_run``: both arms run free from the same prompt (teacher-forced
  tokens, their own state); only the first divergence is classified
  (AGENTS.md: later positions describe downstream consequence).
* ``visit_probe``: one attention visit, identical inputs and state, the
  fixed view against the ordinary cache: localizes a gap to the attention
  kernels of one visit.
"""

import copy

import mlx.core as mx
import numpy as np

from mlx2.runtime.models.cache import CacheList, KVCache, RotatingKVCache
from mlx2.runtime.models.inkling.cache import InklingConvolutionCache
from mlx2.runtime.models.inkling2 import capacity_state as cs
from mlx2.runtime.replay import near_tie


def _logp(logits):
    row = logits[0, -1].astype(mx.float32)
    return np.array(row - mx.logsumexp(row), dtype=np.float64)


def _ordinary(model, tokens, cache):
    site = model.__dict__.pop("_native_replay", None)
    try:
        return model(tokens, cache=cache)
    finally:
        if site is not None:
            object.__setattr__(model, "_native_replay", site)


def same_state_run(model, tokens, prompt, form):
    ref = model.make_cache()
    model(tokens[:, :prompt], cache=ref)
    site = cs.install(model, form=form)
    worst, flips = 0.0, []
    for i in range(prompt, tokens.shape[1]):
        side = copy.deepcopy(ref)
        a = _ordinary(model, tokens[:, i:i + 1], ref)
        b = model(tokens[:, i:i + 1], cache=side)
        mx.eval(a, b)
        worst = max(worst, mx.abs(a - b).max().item())
        if mx.argmax(a).item() != mx.argmax(b).item():
            flips.append({"step": i, **near_tie.classify(_logp(a), _logp(b))})
    return {"worst_abs_logit": worst, "flips": flips, "site": site.receipt()}


def drift_run(model, tokens, prompt, form):
    ref, fixed = model.make_cache(), model.make_cache()
    model(tokens[:, :prompt], cache=ref)
    model(tokens[:, :prompt], cache=fixed)
    site = cs.install(model, form=form)
    worst, first = 0.0, None
    for i in range(prompt, tokens.shape[1]):
        a = _ordinary(model, tokens[:, i:i + 1], ref)
        b = model(tokens[:, i:i + 1], cache=fixed)
        mx.eval(a, b)
        worst = max(worst, mx.abs(a - b).max().item())
        if first is None and mx.argmax(a).item() != mx.argmax(b).item():
            first = {"step": i, **near_tie.classify(_logp(a), _logp(b))}
    return {"worst_abs_logit": worst, "first_divergence": first, "site": site.receipt()}


def _planes(attn, pos, dtype, key):
    keys = mx.random.normal((1, attn.n_kv, pos, attn.head_dim), key=key[0]).astype(dtype)
    values = mx.random.normal((1, attn.n_kv, pos, attn.head_dim), key=key[1]).astype(dtype)
    if attn.is_sliding:
        kv = RotatingKVCache(attn.sliding)
        kv.update_and_fetch(keys, values)
        if kv.keys.shape[2] > kv.max_size:
            kv.compact_to_window()
        elif kv.keys.shape[2] < kv.max_size:
            kv.keys = cs._pad_rows(kv.keys, kv.max_size - kv.keys.shape[2])
            kv.values = cs._pad_rows(kv.values, kv.max_size - kv.values.shape[2])
    else:
        kv = KVCache()
        kv.update_and_fetch(keys, values)
        cs.grow_to_capacity(kv, cs.CAPACITY_STEP)
    conv = InklingConvolutionCache()
    channels = attn.n_kv * attn.head_dim
    hidden = attn.config.hidden_size
    for idx, width in enumerate((channels, channels, hidden, hidden)):
        conv[idx] = mx.random.normal((1, 3, width), key=key[2 + idx])
    conv.advance(pos)
    mx.eval(kv.keys, kv.values, conv.cache)
    return kv, conv


DEFAULT_FUSE = frozenset({"fuse_attention", "fuse_sconv", "fuse_sconv_kernel"})


def configure_attention(attn, policy):
    """Set one standalone ``Attention``'s switches as ``configure_kernels``
    would: ``"reference"`` (all off) or ``"defaults"`` (split-K, sink SDPA,
    sink slot and the fused attention/short-conv set)."""
    on = policy == "defaults"
    holder = {"memo": None} if on else None
    for module in (attn, attn.k_sconv, attn.v_sconv):
        object.__setattr__(module, "_fuse", DEFAULT_FUSE if on else frozenset())
        object.__setattr__(module, "_fuse_holder", holder)
    for name in ("_verify_splitk", "_sink_sdpa", "_sink_slot_on"):
        object.__setattr__(attn, name, on)
    return holder


def visit_probe(attn, pos, *, dtype, policy="reference", seed=0, form="fused"):
    """Max |difference| of one visit's attention output, fixed view vs the
    ordinary cache, from identical inputs and state."""
    holder = configure_attention(attn, policy)
    keys = mx.random.split(mx.random.key(seed), 7)
    kv, conv = _planes(attn, pos, dtype, keys[:6])
    x = mx.random.normal((1, 1, attn.config.hidden_size), key=keys[6]).astype(dtype)
    if holder is not None:
        holder["memo"] = {}
    try:
        ordinary = attn(x, cache=CacheList(copy.deepcopy(kv), copy.deepcopy(conv)))
        u32 = lambda v: mx.array([v], mx.uint32)
        if attn.is_sliding:
            newest, count = cs._ring_phase(kv)
            view = cs.RingView(kv.keys, kv.values, u32(pos), u32(newest), u32(count),
                               form == "fused")
        else:
            view = cs.GlobalView(kv.keys, kv.values, u32(pos), form == "fused")
        fixed = attn(x, cache=CacheList(view, cs.ConvView(conv.cache)))
    finally:
        if holder is not None:
            holder["memo"] = None
    mx.eval(ordinary, fixed)
    gap = mx.abs(ordinary.astype(mx.float32) - fixed.astype(mx.float32)).max().item()
    scale = mx.abs(ordinary.astype(mx.float32)).max().item()
    return {"sliding": attn.is_sliding, "pos": pos, "head_dim": attn.head_dim,
            "dtype": str(dtype), "policy": policy, "max_abs_gap": gap, "max_abs_output": scale}


def attention(config_overrides, layer_index, dtype, seed=0):
    """A standalone ``Attention`` of the tiny config with overrides, random
    weights (the per-visit probe needs no other layer)."""
    from mlx.utils import tree_flatten
    from test_inkling2_cpu import TINY

    from mlx2.runtime.models.inkling2 import TextConfig
    from mlx2.runtime.models.inkling2.language import Attention

    mx.random.seed(seed)
    attn = Attention(TextConfig.from_dict({**TINY, **config_overrides}), layer_index)
    weights = [(k, (mx.random.normal(v.shape) * (0.05 if v.ndim > 1 else 0.3)
                    + (1.0 if k.endswith("scale") else 0.0)).astype(dtype))
               for k, v in tree_flatten(attn.parameters())]
    attn.load_weights(weights, strict=True)
    mx.eval(attn.parameters())
    return attn
