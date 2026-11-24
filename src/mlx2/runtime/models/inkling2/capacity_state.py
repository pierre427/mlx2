"""Fixed-capacity decode state for the Inkling 2 one-row ordinary step.

``inkling2_native_replay`` (default off, unqualified). A recorded-step replay
engine only pays off when the decode step keeps its shapes from one token to
the next. The ordinary caches do not: a global ``KVCache`` returns
``keys[..., :offset]`` (one more row every token), a sliding
``RotatingKVCache`` returns a chronological copy whose length grows until
the window fills, and the attention mask is built from Python integers.

This module gives that step a constant geometry without a second copy of
the state:

* **Layout, in place.** The planes stay the model's own cache objects, so
  APCv2, snapshots and rollback see ordinary caches. ``normalize`` changes a
  buffer only where the ordinary path would also copy it:
  - a global ``KVCache`` that cannot take the next row grows (zero rows
    appended) to ``ceil((offset + 1) / CAPACITY_STEP) * CAPACITY_STEP`` rows,
    one copy per capacity step instead of one per 256 rows, and is marked
    ``_fixed_capacity_step``. The marker (not ``KVCache.reserve``, whose
    reservation batch merges carry to every lane and which other options
    use) says the spare rows are capacity, not state: ``state`` and the
    capsule capture expose only the committed rows, batch merges copy only
    committed rows and carry no plan, and the lane budget
    (``Inkling2CacheBudget.fixed_capacity_step``) charges the spare rows;
  - a ring shorter than its window is padded once to ``max_size`` rows (the
    ordinary path grows it in 256-row copies to the same width); a
    post-chunk ring longer than its window is compacted
    (``compact_to_window``, the trim its next in-place update performs).
  Each grown visit is evaluated before the next grows (bounded peak).
* **Round integers as arrays.** The committed length ``pos``, the ring's
  newest slot and its valid row count enter the step as ``uint32`` arrays;
  the step derives every distance, mask and log-tau scale from them on the
  device, so no per-round host value is baked into the graph.
* **Constant reads.** A global visit attends over all ``cap`` rows (rows past
  the new one masked), a sliding visit over all ``W`` ring slots (slots
  outside the window masked). The sink is SDPA's ``sinks`` operand (a
  per-head logit with a zero value), so no zero row is concatenated.

Two step forms:

* ``fused`` (default): the step writes its own row into the K/V buffers
  (``mx.slice_update`` at the device index) and attends over them; ``commit``
  installs the new buffers. When the cache is the only holder of a buffer
  (nothing else references it at evaluation time), MLX donates it and the
  write is in place; an APCv2 copy-on-write alias makes MLX copy instead,
  which is the correct COW behaviour. Those inputs are the written (state)
  inputs of a replay program.
* ``pure``: the step reads the state and returns per-visit deltas (the new
  K/V row and the four convolution histories); ``commit`` applies them through
  the caches' own update methods. No input is written, but attending to the
  new row concatenates it onto the plane: one full copy of every K/V plane
  per step, counted as ``attend["copies:pure_concat_rows"]`` (rows copied) so
  an A/B of the pure form is read with that cost in view.

Exactness. The step uses the reference attention math (banded bias, -1e30
mask, one SDPA) with the sink as SDPA ``sinks`` (as ``inkling2_sink_sdpa``).
It reads more columns than the ordinary route (masked ones), so the softmax
and value reductions run over a different column count: near-exact, not
bitwise. The committed rows are exactly the rows the step computed. Kernel
options that take the attention front end or the int8 K/V planes
(``inkling2_attn_decode_kernel``, ``inkling2_kv_q8``) are refused together
with this one.

Counters: ``Model._native_replay`` (a ``DecodeSite``) ``counts``,
``declines`` and ``layout``; per attention ``native_replay_attend``;
``receipt(model)``.
"""

from __future__ import annotations

import mlx.core as mx

from ...prefill_checkpoint_rows import select_checkpoint_rows
from ..cache import KVCache, RotatingKVCache
from ..inkling.cache import InklingConvolutionCache
from . import fuse as _fuse

# Global K/V capacity granularity in rows (a module switch for tests; the
# engine sees one program per (capacity step, width)).
CAPACITY_STEP = 4096
FORMS = ("pure", "fused")
DEFAULT_FORM = "fused"
# Options that take the attention front end before the hook (attn_decode) or
# store the global planes as int8 rows (kv_q8): the site declines them.
REFUSED_KERNELS = ("attn_decode_kernel", "kv_q8")
_MASKED = -1e30


class ConvView:
    """The four convolution histories of one visit, as the step sees them.

    Reads come from the step's input arrays; writes are captured as the
    visit's deltas (the conv modules assign ``cache[idx] = new`` exactly as
    they do on ``InklingConvolutionCache``)."""

    speculating = False
    left_padding = None
    lengths = None

    def __init__(self, states):
        self.cache = list(states)
        self.written = [False] * len(self.cache)
        self.advanced = 0

    def __getitem__(self, idx):
        return self.cache[idx]

    def __setitem__(self, idx, value):
        self.cache[idx] = value
        self.written[idx] = True

    def advance(self, n):
        self.advanced += n

    def outputs(self):
        if not all(self.written) or self.advanced != 1:
            raise RuntimeError("Inkling 2 fixed step: a visit did not write its four "
                               "convolution histories exactly once")
        return list(self.cache)


class _KVView:
    """Base of the two fixed K/V views (read by ``attend``)."""

    speculating = False
    left_padding = None
    deferred_rows = 0

    def __init__(self, keys, values, pos, fused):
        self.keys, self.values, self.pos, self.fused = keys, values, pos, fused
        self.result = None

    def outputs(self):
        if self.result is None:
            raise RuntimeError("Inkling 2 fixed step: a visit's attention did not run")
        return list(self.result)


class GlobalView(_KVView):
    """Global visit: ``(1, Hkv, cap, D)`` with rows ``0 .. pos-1`` valid."""


class RingView(_KVView):
    """Sliding visit: ``(1, Hkv, W, D)`` ring; the row of age ``a`` (0 =
    newest) lives at slot ``(newest - a) mod W``; ``count`` rows are valid."""

    def __init__(self, keys, values, pos, newest, count, fused):
        super().__init__(keys, values, pos, fused)
        self.newest, self.count = newest, count


FIXED_VIEWS = (GlobalView, RingView)


# ----- the attention over a fixed view (hook target in Attention) -----


def _bias_mask(attn, r, rel_proj, dist, valid):
    """Reference banded bias for one query row over columns with the given
    distances; ``valid`` False columns get the -1e30 mask."""
    rl = (r @ rel_proj).transpose(0, 2, 1, 3)  # (1, H, 1, E)
    extent = attn.rel_extent
    columns = dist.shape[-1]
    gidx = mx.broadcast_to(mx.clip(dist, 0, extent - 1)[None, None, None, :],
                           (rl.shape[0], rl.shape[1], 1, columns))
    pb = mx.take_along_axis(rl, gidx, axis=-1)
    dtype = rl.dtype
    pb = mx.where((dist >= extent)[None, None, None, :], mx.array(0.0, dtype), pb)
    masked = ~valid
    if attn.sliding > 0:
        masked = masked | (dist >= attn.sliding)
    return mx.where(masked[None, None, None, :], mx.array(_MASKED, dtype), pb).astype(dtype)


def attend(attn, q, r, gate, sink, k, v, view, dtype):
    """One-row attention of ``attn`` over a fixed view, the reference math.

    ``q``/``k``/``v`` are the visit's transposed ``(1, heads, 1, D)`` rows,
    ``r`` the clipped relative query ``(1, 1, H, d_rel)``. Records the
    visit's delta (pure) or its new buffers (fused) on the view."""
    cfg = attn.config
    b, length = q.shape[0], q.shape[2]
    if b != 1 or length != 1:
        raise ValueError("Inkling 2 fixed step serves one row of one token")
    pos = view.pos.astype(mx.int32)  # (1,)
    if attn.log_floor is not None:
        # Reference: positions = arange(1) + offset + 1 with offset = pos.
        positions = (pos + 1).astype(mx.float32)[None, :]
        tau = 1 + cfg.log_scaling_alpha * mx.log(mx.maximum(positions / attn.log_floor, 1))
        tau = tau[:, :, None, None]
        q = (q.astype(mx.float32) * tau.transpose(0, 2, 1, 3)).astype(dtype)
        r = (r.astype(mx.float32) * tau).astype(dtype)
    rel_proj = attn.rel_proj.astype(dtype)
    keys, values = view.keys, view.values
    rows = keys.shape[2]
    j = mx.arange(rows, dtype=mx.int32)
    counts = attn.__dict__.setdefault("native_replay_attend", {})
    if type(view) is GlobalView:
        if view.fused:
            keys = mx.slice_update(keys, k, view.pos, axes=(2,))
            values = mx.slice_update(values, v, view.pos, axes=(2,))
            dist = pos - j
            valid = j <= pos
            view.result = (keys, values)
        else:
            dist = mx.concatenate([pos - j, mx.zeros((1,), mx.int32)])
            valid = mx.concatenate([j < pos, mx.ones((1,), mx.bool_)])
            keys = mx.concatenate([keys, k], axis=2)
            values = mx.concatenate([values, v], axis=2)
            view.result = (k, v)
    else:
        width = rows
        newest = view.newest.astype(mx.int32)
        count = view.count.astype(mx.int32)
        if view.fused:
            slot = (newest + 1) % width
            keys = mx.slice_update(keys, k, slot.astype(mx.uint32), axes=(2,))
            values = mx.slice_update(values, v, slot.astype(mx.uint32), axes=(2,))
            age = (slot - j + width) % width
            dist = age
            valid = age < mx.minimum(count + 1, width)
            view.result = (keys, values)
        else:
            age = (newest - j + width) % width
            dist = mx.concatenate([age + 1, mx.zeros((1,), mx.int32)])
            valid = mx.concatenate([age < count, mx.ones((1,), mx.bool_)])
            keys = mx.concatenate([keys, k], axis=2)
            values = mx.concatenate([values, v], axis=2)
            view.result = (k, v)
    if not view.fused:
        # The pure form's honest cost: the plane is copied to append the row.
        counts["copies:pure_concat_rows"] = counts.get("copies:pure_concat_rows", 0) + rows
    mask = _bias_mask(attn, r, rel_proj, dist, valid)
    # The query-dependent sink: a per-head logit with a zero value (one row).
    sinks = sink[0, 0].astype(mask.dtype)
    out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attn.scale, mask=mask,
                                               sinks=sinks)
    key = ("global_" if type(view) is GlobalView else "ring_") + ("fused" if view.fused else "pure")
    counts[key] = counts.get(key, 0) + 1
    return attn._finish(out, gate, b, length, dtype)


# ----- layout normalization (host integers only, in place) -----


def _pad_rows(buf, rows):
    shape = (*buf.shape[:2], rows, buf.shape[3])
    return mx.concatenate([buf, mx.zeros(shape, buf.dtype)], axis=2)


def _global_reason(kv):
    if type(kv) is not KVCache:
        return "cache_type:" + type(kv).__name__
    if kv.keys is None:
        return "empty_cache"
    if kv.keys.shape[0] != 1 or not isinstance(kv.offset, int):
        return "rows"
    if kv.keys.dtype == mx.int8:
        return "kv_q8_plane"
    return None


def _ring_reason(kv):
    if type(kv) is not RotatingKVCache:
        return "cache_type:" + type(kv).__name__
    if kv.speculating or kv.__dict__.get("_deferred") is not None:
        return "speculating"
    if kv.keep:
        return "ring_keep"
    if kv.keys is None:
        return "empty_cache"
    if kv.keys.shape[0] != 1:
        return "rows"
    width = kv.keys.shape[2]
    if width > kv.max_size and kv._idx != width:
        return "ring_layout"
    if width < kv.max_size and (kv._idx != kv.offset or kv.offset > width):
        return "ring_layout"
    return None


def _conv_reason(conv):
    if type(conv) is not InklingConvolutionCache:
        return "conv_type:" + type(conv).__name__
    if conv.speculating or conv._rollbacks or conv.left_padding is not None \
            or conv.lengths is not None:
        return "conv_state"
    for state in conv.cache:
        if not isinstance(state, mx.array) or state.ndim != 3 or state.shape[0] != 1 \
                or state.shape[1] != 3 or state.dtype != mx.float32:
            return "conv_state"
    return None


def plan(model, cache):
    """``(kinds, reason)``: the visit kinds (``"global"``/``"ring"``) when
    every plane is fit for the fixed step, else the first decline reason."""
    policy = getattr(model, "kernel_policy", {}) or {}
    for name in REFUSED_KERNELS:
        if policy.get(name):
            return None, "kernel_policy:" + name
    visits = len(model.model.layer_indices)
    if cache is None or len(cache) != 2 * visits:
        return None, "cache_planes"
    kinds = []
    offsets, rings = set(), set()
    for visit, index in enumerate(model.model.layer_indices):
        kv, conv = cache[2 * visit], cache[2 * visit + 1]
        sliding = model.model.layers[index].self_attn.is_sliding
        reason = _ring_reason(kv) if sliding else _global_reason(kv)
        if reason is None:
            reason = _conv_reason(conv)
        if reason is None and getattr(kv, "speculating", False):
            reason = "speculating"
        if reason is not None:
            return None, reason
        offsets.add(kv.offset)
        if sliding:
            rings.add((kv.offset, kv._idx, kv.keys.shape[2]))
        kinds.append("ring" if sliding else "global")
    if len(offsets) != 1:
        return None, "visit_offsets"
    if len(rings) > 1:
        return None, "ring_phase"
    return tuple(kinds), None


def grow_to_capacity(kv, step):
    """Grow a global ``KVCache`` to ``ceil((offset + 1) / step) * step`` rows
    (zeros appended, one copy) and mark the spare rows as fixed capacity."""
    target = -(-(kv.offset + 1) // step) * step
    if kv.keys.shape[2] < target:
        kv.keys = _pad_rows(kv.keys, target - kv.keys.shape[2])
        kv.values = _pad_rows(kv.values, target - kv.values.shape[2])
    kv._fixed_capacity_step = int(step)


def normalize(model, cache, kinds, counts, step=None):
    """Bring the planes into the fixed layout, in place. Host integers only.

    A buffer is copied only where the ordinary path would also copy it (see
    the module docstring): a global plane that cannot take the next row grows
    to whole ``step`` multiples (marked ``_fixed_capacity_step``); a ring is padded
    or compacted to exactly ``max_size`` rows once."""
    step = CAPACITY_STEP if step is None else step
    for visit, kind in enumerate(kinds):
        kv = cache[2 * visit]
        if kind == "global":
            need = kv.offset + 1
            if kv.keys.shape[2] >= need:
                continue
            grow_to_capacity(kv, step)
            mx.eval(kv.keys, kv.values)
            counts["capacity_grows"] = counts.get("capacity_grows", 0) + 1
        else:
            width = kv.keys.shape[2]
            if width == kv.max_size:
                continue
            if width > kv.max_size:
                if not kv.compact_to_window():
                    raise RuntimeError("Inkling 2 fixed layout: ring compaction refused")
                counts["ring_compactions"] = counts.get("ring_compactions", 0) + 1
            else:
                kv.keys = _pad_rows(kv.keys, kv.max_size - width)
                kv.values = _pad_rows(kv.values, kv.max_size - width)
                counts["ring_pads"] = counts.get("ring_pads", 0) + 1
    # The convolution histories need no change: they are (1, 3, C) FP32 at
    # every length (after a decode step, views of the step's (1, 4, C)
    # context, a geometry that is stable from round to round).


def _ring_phase(kv):
    """``(newest, count)`` of a normalized ring (``Attention._ring_layout``
    convention: chronological key j at slot ``(j + rot) % W`` for j < size)."""
    width = kv.keys.shape[2]
    idx = kv._idx
    if idx == width:
        rot, size = 0, width
    elif idx < kv.offset:
        rot, size = idx, width
    else:
        rot, size = 0, idx
    return (rot + size - 1) % width, size


def round_inputs(model, tokens, cache, kinds):
    """The step's flat input list: ``[tokens, pos, newest, count]`` then, per
    visit, ``[keys, values, conv0, conv1, conv2, conv3]``."""
    first = cache[0]
    pos = first.offset
    newest, count = 0, 0
    for visit, kind in enumerate(kinds):
        if kind == "ring":
            newest, count = _ring_phase(cache[2 * visit])
            break
    arrays = [tokens, mx.array([pos], mx.uint32), mx.array([newest], mx.uint32),
              mx.array([count], mx.uint32)]
    for visit in range(len(kinds)):
        kv, conv = cache[2 * visit], cache[2 * visit + 1]
        arrays += [kv.keys, kv.values, *conv.cache]
    return arrays, pos


def make_step(model, kinds, form=DEFAULT_FORM):
    """``f(arrays) -> outputs``: a pure function of MLX arrays whose launches
    depend only on the input shapes. Outputs: ``[logits]`` then per visit
    ``[k_new, v_new]`` (pure) or ``[keys', values']`` (fused), then the four
    convolution histories."""
    if form not in FORMS:
        raise ValueError(f"Inkling 2 fixed step form must be one of {FORMS}")
    fused = form == "fused"
    visits = len(kinds)

    def step(arrays):
        tokens, pos, newest, count = arrays[:4]
        planes, views = [], []
        for visit, kind in enumerate(kinds):
            keys, values, *convs = arrays[4 + 6 * visit: 10 + 6 * visit]
            if kind == "global":
                kvv = GlobalView(keys, values, pos, fused)
            else:
                kvv = RingView(keys, values, pos, newest, count, fused)
            cv = ConvView(convs)
            planes += [kvv, cv]
            views.append((kvv, cv))
        if len(planes) != 2 * visits:
            raise ValueError("Inkling 2 fixed step input count")
        pre = model.model(tokens, planes, None)
        logits = model._project_logits(
            _fuse.rms(model.model, model.model.norm, select_checkpoint_rows(pre)))
        outputs = [logits]
        for kvv, cv in views:
            outputs += kvv.outputs() + cv.outputs()
        return outputs

    return step


def commit(cache, kinds, outputs, form=DEFAULT_FORM):
    """Apply a step's outputs to the caches (host bookkeeping + the caches'
    own update for the pure form). Returns the logits."""
    logits = outputs[0]
    for visit, kind in enumerate(kinds):
        kv, conv = cache[2 * visit], cache[2 * visit + 1]
        a, b, *convs = outputs[1 + 6 * visit: 7 + 6 * visit]
        if form == "pure":
            kv.update_and_fetch(a, b)
        elif kind == "global":
            kv.keys, kv.values = a, b
            kv.offset += 1
        else:
            # RotatingKVCache._update_in_place for one row on a full-width
            # ring with keep=0, whose slot the step already wrote.
            if kv._idx == kv.max_size:
                kv._idx = 0
            kv.keys, kv.values = a, b
            kv.offset += 1
            kv._idx += 1
        for idx, state in enumerate(convs):
            conv[idx] = state
        conv.advance(1)
    return logits


# ----- the decode site (Model.__call__ hook target) -----


class DecodeSite:
    """Routes eligible one-row ordinary decode calls through the fixed step
    and offers that step to the replay engine (``mlx2.runtime.replay``).

    Every call that the site does not take is counted with its reason and
    served by the ordinary path; every call it takes is counted by the
    engine's route (native, recording, checking, or mlx with the engine's
    decline reason). An error inside the site is a counted decline
    (``error:<type>``): before the step it falls back to the ordinary path,
    in the engine it falls back to the MLX step. ``ReplayLost`` is never
    absorbed: the state cannot be trusted, so the request must end."""

    def __init__(self, model, *, form=DEFAULT_FORM, engine=None, weights_identity=None):
        from ...replay import StepSite

        if form not in FORMS:
            raise ValueError(f"Inkling 2 native replay form must be one of {FORMS}")
        self.form = form
        self.counts = {}
        self.declines = {}
        self.layout_counts = {}
        self._steps = {}
        # Committed length of the last step per cache list (bounded LRU).
        self._lanes = {}
        self.replay = StepSite("inkling2_decode", engine=engine,
                               weights_identity=weights_identity or f"model@{id(model):x}")

    def _decline(self, reason):
        self.declines[reason] = self.declines.get(reason, 0) + 1

    def decline_reason(self, model, inputs, cache, inputs_embeds, skip_logits, kwargs):
        if inputs_embeds is not None:
            return "input_embeddings"
        if skip_logits:
            return "skip_logits"
        if kwargs:
            return "kwargs"
        if not isinstance(inputs, mx.array) or inputs.ndim != 2 or inputs.shape != (1, 1):
            return "not_one_row_decode"
        if cache is None:
            return "no_cache"
        return None

    _LANES = 64

    def _boundary(self, cache, pos):
        """A new request: a cache list this site has not stepped yet (a
        request that starts with a one-row decode, e.g. after a full APCv2
        hit, has no prefill call to mark it), or a known list whose length
        went back (a rollback or a reused list). Lanes that alternate, and a
        list that moved forward through calls the site declined, are not
        boundaries."""
        key = id(cache)
        last = self._lanes.pop(key, None)
        if last is None or pos <= last:
            self.replay.note_request_boundary()
        self._lanes[key] = pos
        while len(self._lanes) > self._LANES:
            self._lanes.pop(next(iter(self._lanes)))

    def decode(self, model, inputs, cache, inputs_embeds=None, skip_logits=False, kwargs=None):
        from ...replay import ReplayLost

        reason = self.decline_reason(model, inputs, cache, inputs_embeds, skip_logits, kwargs)
        if reason is not None:
            if reason == "not_one_row_decode":
                self.replay.note_request_boundary()
                if cache is not None:
                    self._lanes.pop(id(cache), None)
            return self._decline(reason)
        try:
            kinds, reason = plan(model, cache)
            if reason is not None:
                return self._decline(reason)
            normalize(model, cache, kinds, self.layout_counts)
            arrays, pos = round_inputs(model, inputs, cache, kinds)
            step = self._steps.get(kinds)
            if step is None:
                step = self._steps[kinds] = make_step(model, kinds, self.form)
            policy = tuple(sorted((getattr(model, "kernel_policy", {}) or {}).items()))
            geometry = ("inkling2_decode", 1, self.form, kinds, policy)
        except Exception as exc:  # noqa: BLE001 - counted; the ordinary path serves
            return self._decline("error:" + type(exc).__name__)
        self._boundary(cache, pos)
        try:
            outputs = self.replay.call(step, arrays, geometry, written=self._written(kinds))
        except ReplayLost:
            raise
        except Exception as exc:  # noqa: BLE001 - counted; the MLX step serves
            self._decline("error:" + type(exc).__name__)
            outputs = step(arrays)
        del arrays  # the step's inputs: no host reference left at evaluation
        self.counts["fixed_step"] = self.counts.get("fixed_step", 0) + 1
        return commit(cache, kinds, outputs, self.form)

    def _written(self, kinds):
        if self.form == "pure":
            return ()
        written = []
        for visit in range(len(kinds)):
            written += [4 + 6 * visit, 5 + 6 * visit]
        return tuple(written)

    def receipt(self):
        return {"form": self.form, "counts": dict(self.counts),
                "declines": dict(self.declines), "layout": dict(self.layout_counts),
                "engine": self.replay.receipt()}


def install(model, *, enabled=True, form=DEFAULT_FORM, engine=None, weights_identity=None):
    """Attach a fresh decode site to an Inkling 2 ``Model`` (or remove it).

    ``configure_kernels`` calls this on every reconfiguration, so programs
    and counters never outlive the kernel set they were made under."""
    model.__dict__.pop("_native_replay", None)
    if not enabled:
        return None
    site = DecodeSite(model, form=form, engine=engine, weights_identity=weights_identity)
    object.__setattr__(model, "_native_replay", site)
    return site


def receipt(model):
    site = model.__dict__.get("_native_replay")
    attend_counts = {}
    for layer in model.model.layers:
        for key, value in layer.self_attn.__dict__.get("native_replay_attend", {}).items():
            attend_counts[key] = attend_counts.get(key, 0) + value
    if site is None:
        return {"enabled": False, "attend": attend_counts}
    return {"enabled": True, "attend": attend_counts, **site.receipt()}


def synthetic_cache(model, pos, *, dtype=None, seed=0):
    """A random cache in the decode layout at length ``pos`` (warm-up and
    the GPU smoke; never a real request's state). Uses the model's own
    cache types, filled through their update methods."""
    cfg = model.config
    dtype = dtype or model.model.embed_norm.weight.dtype
    key = mx.random.key(seed)
    cache = model.make_cache()
    for visit, index in enumerate(model.model.layer_indices):
        attn = model.model.layers[index].self_attn
        kv, conv = cache[2 * visit], cache[2 * visit + 1]
        key, k1, k2, k3 = mx.random.split(key, 4)
        shape = (1, attn.n_kv, pos, attn.head_dim)
        kv.update_and_fetch(mx.random.normal(shape, key=k1).astype(dtype),
                            mx.random.normal(shape, key=k2).astype(dtype))
        widths = (attn.n_kv * attn.head_dim,) * 2 + (cfg.hidden_size,) * 2
        for idx, width in enumerate(widths):
            k3, sub = mx.random.split(k3)
            conv[idx] = mx.random.normal((1, cfg.sconv_kernel_size - 1, width), key=sub)
        conv.advance(pos)
    mx.eval([c.keys for c in cache[0::2]], [c.values for c in cache[0::2]],
            [c.cache for c in cache[1::2]])
    return cache
