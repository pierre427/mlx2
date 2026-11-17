"""A donor loaded through its own mlx2 adapter, with every unit site patched (units.py).

Generalises scripts/amoeba27b_lib.py's Host (claude/amoeba-27b-80bc6a@37d75f144) to any donor
whose adapter loads it: nothing here names a model. What the 27B Host hardcoded is found from the
module structure and then *checked*, never assumed:

  logits module   the first of model / model.language_model that maps ids [B, T] to logits
  inner model     the module that owns ``layers`` and whose output is the post-norm hidden state
  head            lm_head, or the tied embedding's as_linear, with final logit soft-capping when
                  the config declares it

``check_paths`` compares head(inner(ids)) and the layer-replay path (record every layer call, then
replay ``layer(h, *recorded args)`` and ``norm`` + head) against the logits module on a probe. A
path that does not match is disabled and the exact (slower) path is used instead; the receipt says
which. Layer-wise backprop (one layer's tape alive at a time, the 27B recipe) runs only after its
gradients have passed a gate on a short probe: whole_train, matching whole-graph backprop
(``check_layerwise_grad``); layer_ref, a directional-derivative check against central differences
of the masked fused forward (``check_directional_grad``).

Gradient modes (``Host.grad_mode``):

  layer_ref    (default) the model stays in eval mode, so every forward -- the carve's target logits
               and the layer-wise residual pass -- runs the production fast path (fused kernels).
               Only the unit being backpropagated (one layer inside the layer-wise backward, or the
               final norm + head for the head vjp) is switched to its reference mode for its
               ``mx.vjp``: ``train()`` on that module tree plus the per-module kernel flags in
               ``REFERENCE_FLAGS``, then restored exactly. ``check_layer_reference`` measures, per
               layer, how far the eval output is from the reference output on the same input and
               which modules changed path; the driver refuses a run whose gap exceeds bf16 noise.
  whole_train  the old behaviour: the whole model in training mode for the whole gradient
               computation (every forward on the reference path). Kept for comparison.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from contextlib import contextmanager

import numpy as np

from . import units as U


def grad_compare(a, b):
    """Compare two gradient sets (array or per-layer list): relative L2, relative max element, cosine."""
    def rows(g):
        out = []
        import mlx.core as mx
        if isinstance(g, np.ndarray):
            g = [g] if g.ndim <= 1 else list(g)
        for x in (g if isinstance(g, (list, tuple)) else list(g)):
            if isinstance(x, mx.array):
                x = x.astype(mx.float32)
            out.append(np.asarray(x, dtype=np.float64).ravel())
        return out
    ra, rb = rows(a), rows(b)
    a, b = np.concatenate(ra), np.concatenate(rb)
    nb = max(1e-30, float(np.linalg.norm(b)))

    def cos(x, y):
        return float(x @ y / max(1e-30, float(np.linalg.norm(x)) * float(np.linalg.norm(y))))
    per = [cos(x, y) for x, y in zip(ra, rb) if np.linalg.norm(y) > 0]
    return {"rel_l2": float(np.linalg.norm(a - b) / nb),
            "rel_max": float(np.abs(a - b).max() / max(1e-30, float(np.abs(b).max()))),
            "cosine": cos(a, b), "min_layer_cosine": min(per) if per else 1.0}


GRAD_MODES = ("layer_ref", "whole_train")

# Per-module flags (besides ``training``) that select a kernel with no VJP, and their reference value.
# Applied wherever a module has the attribute as a bool; nothing here names a model.
REFERENCE_FLAGS = {"use_kernel": False, "training_indices": True}

# Process-global fast-path flags a model reads when it builds a forward op, so a per-module
# ``training`` cannot reach them: (model module, flag module, global, reference value). A flag is
# switched while a module tree containing a class from the MODEL module is in reference mode; the
# flag module may be a helper the model calls (it has no class in the tree). Neither is imported here:
# an entry applies only when both are already loaded.
#   qwen4_exp._QSA_SCATTER_CHOSEN: True builds the QSA block mask with put_along_axis on the
#     indexer's argpartition ids, whose VJP raises ("with respect to indices"); False builds the same
#     mask with any(block_ids == selected) (bit-identical, tested), no VJP through the ids. Read at
#     QSASelection construction, inside the layer call.
#   qwen4_qsa_scores._ENABLED: a Metal kernel for the QSA block scores feeding argpartition, gated by
#     neither ``training`` nor a VJP (past ~2048 tokens the vjp would raise); False = the ops scores.
#   qwen4_fused_group_norm._ENABLE: a Metal GroupRMSNorm with no VJP and no training gate (default off).
# The model file is on a hashed path list (Flash-Next decode-tile law), so the switches live here.
_Q4 = "mlx2.runtime.models.qwen4_exp"
REFERENCE_GLOBALS = (
    (_Q4, _Q4, "_QSA_SCATTER_CHOSEN", False),
    (_Q4, "mlx2.runtime.models.qwen4_qsa_scores", "_ENABLED", False),
    (_Q4, "mlx2.runtime.models.qwen4_fused_group_norm", "_ENABLE", False),
    # Dispatch scheduling only (no value change): the trunk forward calls mx.async_eval mid-graph for
    # eager dispatch and PLE early dispatch, which MLX refuses inside a graph transformation
    # ("[async_eval] Not allowed inside a graph transformation"; FN smoke 2026-11-15, whole-graph grad gate).
    (_Q4, _Q4, "_EAGER_DISPATCH", False),
    (_Q4, _Q4, "_PLE_EARLY_DISPATCH", False),
)

# Module-global receipts (module, status function) snapshotted around check_layer_reference, so a
# fused route with process-level counters shows up as engaged or declined in the receipt.
STATUS_PROBES = (
    ("mlx2.runtime.models.qwen_hc_prefill", "status"),
    (_Q4, "qwen4_ple_compile_status"),
    ("mlx2.runtime.models.qwen4_qsa_scores", "status"),
    ("mlx2.runtime.models.moe_nax_gather", "status"),
    ("mlx2.runtime.models.qwen4_fused_group_norm", "fused_group_norm_status"),
)


def _numeric_leaves(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = d
    return out


def status_snapshot():
    """Numeric leaves of every loaded STATUS_PROBES receipt, as {module.fn.path: value}."""
    import sys
    out = {}
    for mod, fn in STATUS_PROBES:
        g = sys.modules.get(mod)
        f = getattr(g, fn, None) if g is not None else None
        if callable(f):
            try:
                out.update(_numeric_leaves(f(), f"{mod.rsplit('.', 1)[-1]}.{fn}"))
            except Exception:                              # noqa: BLE001 - a receipt, never a gate
                pass
    return out


def status_delta(before, after):
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(set(before) | set(after))
            if after.get(k, 0) != before.get(k, 0)}


FINAL_NORM_ATTRS = ("norm", "final_layernorm", "ln_f", "hyper_connection_mixer")


def _walk(module, prefix=""):
    """(name, module) for module and every submodule (mlx ``named_modules`` plus the root)."""
    return [(prefix, module)] + [(f"{prefix}.{n}" if prefix else n, m) for n, m in module.named_modules() if m is not module]


def _counters(module):
    """Scalar and counter-dict state of one module (not arrays, not children): the receipts fused
    paths keep (calls, fallbacks, last variant)."""
    out = {}
    for k, v in vars(module).items():
        if k.startswith("__") or k == "_training":
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values()):
            out[k] = dict(v)
    for k, v in module.items():                       # mlx keeps dict-valued attributes as items
        if isinstance(v, dict) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                             for x in v.values()):
            out[k] = dict(v)
    return out


def _delta(before, after):
    d = {}
    for k in set(before) | set(after):
        a, b = before.get(k), after.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            d[k] = b - a
        elif isinstance(a, dict) or isinstance(b, dict):
            a, b = a or {}, b or {}
            d[k] = {kk: b.get(kk, 0) - a.get(kk, 0) for kk in set(a) | set(b) if b.get(kk, 0) != a.get(kk, 0)}
        else:
            d[k] = b if a != b else "unchanged"
    return d


def reference_globals(modules):
    """[(flag module object, global name, reference value)] of REFERENCE_GLOBALS that apply to these
    module trees: some class in the trees comes from the entry's model module, and the flag module
    is loaded and has the global."""
    import sys
    used = set()
    for root in modules:
        if root is not None:
            used |= {c.__module__ for _, m in _walk(root) for c in type(m).__mro__}
    out = []
    for model_mod, mod, name, ref in REFERENCE_GLOBALS:
        g = sys.modules.get(mod)
        if g is not None and model_mod in used and hasattr(g, name):
            out.append((g, name, ref))
    return out


@contextmanager
def reference_mode(modules, module_flags=True):
    """Switch each module tree to its reference (differentiable) path: ``train()``, every
    ``REFERENCE_FLAGS`` attribute it has, and the applicable ``REFERENCE_GLOBALS``; restore each
    previous value on exit, also when the body raises. ``module_flags=False``: the globals only
    (whole_train, where the model's own training mode already stands)."""
    saved = []
    for root in (modules if module_flags else ()):
        if root is None:
            continue
        for _, m in _walk(root):
            saved.append((m, "_training", m._training))
            for k, v in REFERENCE_FLAGS.items():
                if isinstance(getattr(m, k, None), bool):
                    saved.append((m, k, getattr(m, k)))
    globs = [(g, name, getattr(g, name), ref) for g, name, ref in reference_globals(modules)]
    try:
        for g, name, _, ref in globs:
            setattr(g, name, ref)
        for m, k, _ in saved:
            if k == "_training":
                m._set_training_mode(True)
            else:
                setattr(m, k, REFERENCE_FLAGS[k])
        yield
    finally:
        for m, k, v in reversed(saved):
            if k == "_training":
                m._set_training_mode(v)
            else:
                setattr(m, k, v)
        for g, name, v, _ in globs:
            setattr(g, name, v)


@contextmanager
def trace_training_reads(roots):
    """Record which modules under ``roots`` ([(prefix, module)]) read ``training`` while the
    context is open (their forward branches on the mode). Patches ``nn.Module.training``."""
    import mlx.nn as nn
    names = {id(m): n for p, root in roots for n, m in _walk(root, p)}
    reads = set()
    orig = nn.Module.__dict__["training"]

    def getter(self):
        n = names.get(id(self))
        if n is not None:
            reads.add(n)
        return self._training
    nn.Module.training = property(getter, doc=orig.__doc__)
    try:
        yield reads
    finally:
        nn.Module.training = orig


def _first_attr(obj, names):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


EMBED_NAMES = ("embed_tokens", "wte", "tok_embeddings", "embeddings")      # embeddings: Nemotron-H backbone
NORM_NAMES = ("norm", "final_layernorm", "ln_f", "norm_f")


def find_inner(model):
    """(logits_module, inner) by structure: inner owns ``layers`` and ``embed_tokens``."""
    cands = []
    for top in (getattr(model, "language_model", None), model):
        if top is None:
            continue
        for inner in (getattr(top, "model", None), getattr(top, "language_model", None),
                      getattr(top, "backbone", None), top):
            if inner is not None and getattr(inner, "layers", None) is not None and \
                    _first_attr(inner, EMBED_NAMES) is not None:
                cands.append((top, inner))
    if not cands:
        raise SystemExit("carve host: no inner model with layers and embed_tokens found")
    return cands[0]


class Host:
    def __init__(self, path=None, log=print, policy=None, model=None, tokenizer=None, fingerprint=None,
                 checkpoint=None, moe_units="expert"):
        """Load ``path`` through ``inspect_model`` (or wrap a given ``model``/``tokenizer``, for tests).
        ``moe_units``: "expert" (whole routed experts) or "neuron" ((expert, neuron) units; units.py)."""
        import mlx.core as mx
        t0 = time.time()
        self.log = log
        self.adapter = None
        if model is None:
            from mlx2.adapters.registry import inspect_model
            self.path = Path(path)
            kw = {"execution_policy": policy} if policy else {}
            self.adapter = inspect_model(self.path).adapter_type(str(self.path), **kw)
            model, tokenizer = self.adapter.model, self.adapter.tokenizer
            fingerprint = self.adapter.identity["fingerprint"]
            self.adapter_name = type(self.adapter).__name__
        else:
            self.path = Path(path) if path else None
            self.adapter_name = None
        self.model, self.tok = model, tokenizer
        self.fingerprint = fingerprint or "unknown"
        name = checkpoint or (self.path.name if self.path else type(model).__name__)
        self.checkpoint = f"{name}@{self.fingerprint.split(':')[-1][:8]}"
        self.lm, self.inner = find_inner(model)
        self.layers = list(self.inner.layers)
        self.L = len(self.layers)
        self.moe_units = moe_units
        self.sites, self.ctl = U.install(self.layers, moe_units=moe_units)
        self.N = U.uniform_width(self.sites)                 # None when site widths differ
        self.widths = [s.n for s in self.sites]
        self.cfg = model_config(self.lm, self.inner, model)
        norm = _first_attr(self.inner, NORM_NAMES)
        self.d = int(self.cfg.get("hidden_size") or norm.weight.shape[-1])
        self.softcap = self.cfg.get("final_logit_softcapping")
        # the model's own head method when it has one (it knows tying and soft-capping); else lm_head or
        # the tied embedding. check_paths refuses the run if this does not reproduce the logits module.
        self.head_fn = next((getattr(m, n) for m in (self.lm, model) for n in ("logits_from_hidden", "logits")
                             if callable(getattr(m, n, None))), None)
        self.tied = getattr(self.lm, "lm_head", None) is None
        self.fast_head = False
        self.replay_ok = False
        self.layerwise_ok = False
        self.grad_mode = "layer_ref"
        self.paths = {}
        self.model.freeze()
        self.model.eval()
        mx.eval(self.model.parameters())
        log(f"loaded {self.checkpoint} via {self.adapter_name or 'given model'}: {self.L} layers, sites "
            f"{sorted({s.kind for s in self.sites if s.kind}, key=str)}, MoE units {moe_units}, units {sum(self.widths)}, d={self.d} "
            f"in {time.time() - t0:.0f}s, MLX active {mx.get_active_memory() / 2**30:.1f} GB")

    # ------------------------------------------------------------------ text

    def encode(self, text: str) -> list[int]:
        for t in (self.tok, getattr(self.tok, "_tokenizer", None)):
            if t is None or not hasattr(t, "encode"):
                continue
            try:
                out = t.encode(text, add_special_tokens=False)
            except TypeError:
                out = t.encode(text)
            return list(getattr(out, "ids", out))
        return list(self.tok(text)["input_ids"])

    def decode(self, ids) -> str:
        t = getattr(self.tok, "_tokenizer", None) or self.tok
        return t.decode(list(ids))

    # ------------------------------------------------------------------ forward paths

    def _embed(self):
        return _first_attr(self.inner, EMBED_NAMES)

    def head(self, h):
        import mlx.core as mx
        if self.head_fn is not None:
            return self.head_fn(h).astype(mx.float32)
        if self.tied:
            out = self._embed().as_linear(h)
        else:
            out = self.lm.lm_head(h)
        out = out.astype(mx.float32)
        if self.softcap:
            out = mx.tanh(out / self.softcap) * self.softcap
        return out

    def full_logits(self, ids):
        import mlx.core as mx
        return self.lm(ids).astype(mx.float32)

    def hidden(self, ids):
        return self.inner(ids)

    def logits_at(self, ids, pos=None):
        """Float32 logits [B, P, V] at positions pos (list, same per row; or [B, P] int array)."""
        import mlx.core as mx
        h = self.hidden(ids) if self.fast_head else None
        if h is None:
            lg = self.full_logits(ids)
            return lg if pos is None else _take(mx, lg, pos)
        if pos is not None:
            h = _take(mx, h, pos)
        return self.head(h)

    # ------------------------------------------------------------------ layer record / replay

    def record(self, ids, before=None, outputs=None):
        """One forward through the model's own trunk that records every layer's input and call
        arguments: [(h_in, args, kwargs)]. Returns (records, inner_output).

        Runs inside whatever scopes the trunk opens around its layers (e.g. a prefill phase that
        admits a fused MoE route), which a bare ``layer(...)`` replay would not. ``before(l)`` runs
        just before layer l (to set that layer's masks); ``outputs`` (a list) collects each layer's
        residual output."""
        recs = []
        patched = []
        for li, layer in enumerate(self.layers):
            base = type(layer)

            def call(self_, x, *a, _base=base, _l=li, **k):
                recs.append((x, a, k))
                if before is not None:
                    before(_l)
                out = _base.__call__(self_, x, *a, **k)
                if outputs is not None:
                    outputs.append(_hid(out))
                return out
            layer.__class__ = type(f"Rec{base.__name__}", (base,), {"__call__": call})
            patched.append((layer, base))
        try:
            out = self.inner(ids)
        finally:
            for layer, base in patched:
                layer.__class__ = base
        if len(recs) != self.L:
            raise RuntimeError(f"record: {len(recs)} layer calls for {self.L} layers")
        return recs, out

    def _final_norm_module(self):
        return _first_attr(self.inner, FINAL_NORM_ATTRS)

    def final_norm(self, h):
        return self._final_norm_module()(h)

    def _tail_modules(self):
        """What the head vjp backpropagates through: the final norm and the head's own modules."""
        return [self._final_norm_module(), getattr(self.lm, "lm_head", None) if not self.tied else self._embed()]

    def replay_hidden(self, recs, upto=None):
        h = recs[0][0]
        for l, (_, a, k) in enumerate(recs[:upto]):
            h = _hid(self.layers[l](h, *a, **k))
        return h

    def check_paths(self, ids, tol=1e-3):
        """Enable head(inner(ids)) and the replay path only where they match the logits module."""
        import mlx.core as mx
        self.ctl.reset()
        ref = self.full_logits(ids)
        fast = self.head(self.hidden(ids))
        mx.eval(ref, fast)
        d_fast = float(mx.abs(fast - ref).max())
        self.fast_head = d_fast <= tol * max(1.0, float(mx.abs(ref).max()))
        try:
            recs, _ = self.record(ids)
            rep = self.head(self.final_norm(self.replay_hidden(recs)))
            mx.eval(rep)
            d_rep = float(mx.abs(rep - ref).max())
            self.replay_ok = d_rep <= tol * max(1.0, float(mx.abs(ref).max()))
        except Exception as e:                                # noqa: BLE001 - recorded, path disabled
            d_rep, self.replay_ok = f"{type(e).__name__}: {e}"[:200], False
        self.paths = {"fast_head": self.fast_head, "fast_head_max_abs": d_fast,
                      "head": "model head method" if self.head_fn is not None else ("tied embedding" if self.tied else "lm_head"),
                      "softcap": self.softcap, "replay": self.replay_ok, "replay_max_abs": d_rep}
        self.log(f"paths: {json.dumps(self.paths, default=str)}")
        if not self.fast_head:
            raise SystemExit(f"carve host: head(inner(ids)) differs from the model's logits by {d_fast}; every carve "
                             f"loss uses head(), so the run is refused ({self.paths['head']})")
        return self.paths

    def train_mode(self, on: bool):
        """Enter or leave gradient computation.

        Some mixers (GDN) take a differentiable ops path only in training mode, and routers stop
        gradients on their (discrete) expert ids only when asked (Kimi ``training_indices``;
        Qwen3-Next checks ``training``). ``whole_train`` sets both on the whole model for the whole
        computation (the old behaviour). ``layer_ref`` keeps the model in eval mode (fused forward)
        and sets them only on the module being backpropagated (``reference_mode`` in ``_layerwise``
        and ``_whole``); the routers' input stop-gradient (``router_sg``) is value-neutral, so it is
        on for the whole computation in both modes."""
        if self.grad_mode not in GRAD_MODES:
            raise ValueError(f"grad_mode {self.grad_mode!r}: expected one of {GRAD_MODES}")
        self.ctl.router_sg = bool(on)
        whole = on and self.grad_mode == "whole_train"
        self.model.train() if whole else self.model.eval()
        for _, m in self.model.named_modules():
            if hasattr(m, "training_indices"):
                m.training_indices = bool(whole)

    def _ref(self, modules):
        """Reference mode for ``modules`` in layer_ref mode; in whole_train the model's own mode
        stands and only the process-global flags (REFERENCE_GLOBALS) are switched."""
        return reference_mode(modules, module_flags=self.grad_mode == "layer_ref")

    def check_layer_reference(self, ids, tol=2e-2):
        """Per layer, on the same input: eval-mode (fused) output vs reference-mode output.

        The eval arm is the trunk's own forward (``record``: every layer inside the scopes the trunk
        opens, e.g. the prefill phase that admits a fused MoE route), on the carve's own path (core =
        ones at every site, so MoE sites take the carve's re-implemented routing). These are the
        residuals ``_layerwise`` linearises around. The reference arm calls each layer alone, on its
        recorded input, in ``reference_mode`` (as the layer-wise backward does).

        The gate is the relative L2 of the layer's UPDATE, ||(y_e - x) - (y_r - x)|| / ||y_r - x||
        with the shared input x: a layer's output is mostly the pass-through residual (several
        hyper-connection streams), so an output-relative figure hides a badly wrong branch; it is
        reported too (``rel_l2_output``). The final norm + head is checked as layer "head" (its
        output is not a residual, so its gate is the output figure). Also reported per layer:
        ``reads_training`` (modules that branched on the mode), the flags flipped, ``went_eager``
        (modules whose call/fallback receipts moved differently in the two arms); and for the
        whole check ``global_receipts``, the module-global route counters (STATUS_PROBES) each arm
        moved. ok iff every gated rel L2 <= tol."""
        import mlx.core as mx
        c = self.ctl
        c.reset()
        c.core = [mx.ones((n,)) for n in self.widths]
        rows = []
        try:
            layer_roots = [[(f"layers.{l}", self.layers[l])] for l in range(self.L)]
            tail_roots = [(f"tail.{i}", m) for i, m in enumerate(self._tail_modules()) if m is not None]
            walk = {i: [w for p, root in roots for w in _walk(root, p)] for i, roots in enumerate(layer_roots + [tail_roots])}
            # ---- eval arm: one trunk forward for every layer, then the head
            cnt0 = {i: {n: _counters(m) for n, m in w} for i, w in walk.items()}
            g0 = status_snapshot()
            outs = []
            with trace_training_reads([r for roots in layer_roots for r in roots]) as reads_e:
                recs, _ = self.record(ids, outputs=outs)
                mx.eval([r[0] for r in recs], outs)
            with trace_training_reads(tail_roots) as reads_te:
                head_e = self.head(self.final_norm(outs[-1])).astype(mx.float32)
                mx.eval(head_e)
            g1 = status_snapshot()
            d_e = {i: {n: _delta(cnt0[i][n], _counters(m)) for n, m in w} for i, w in walk.items()}
            reads_e = set(reads_e) | set(reads_te)
            # ---- reference arm: each unit alone, in reference mode
            units = [(str(l), layer_roots[l], recs[l][0], outs[l],
                      lambda x=x, a=a, k=k, l=l: _hid(self.layers[l](x, *a, **k)))
                     for l, (x, a, k) in enumerate(recs)]
            units.append(("head", tail_roots, None, head_e, lambda x=outs[-1]: self.head(self.final_norm(x))))
            for i, (name, roots, x, e, run) in enumerate(units):
                w = walk[i]
                mods = [m for _, m in roots]
                flags = sorted(f"{n}.{k}" for n, m in w for k in REFERENCE_FLAGS
                               if isinstance(getattr(m, k, None), bool) and getattr(m, k) != REFERENCE_FLAGS[k])
                flags += sorted(f"{g.__name__}.{gn}" for g, gn, ref in reference_globals(mods) if getattr(g, gn) != ref)
                before = {n: _counters(m) for n, m in w}
                with reference_mode(mods), trace_training_reads(roots) as reads_r:
                    r = run().astype(mx.float32)
                    mx.eval(r)
                d_r = {n: _delta(before[n], _counters(m)) for n, m in w}
                e = e.astype(mx.float32)
                names = {n for n, _ in w}
                reads = sorted((reads_e & names) | set(reads_r))

                def rel(a, b):
                    return float(mx.sqrt(((a - b) ** 2).sum()) / mx.maximum(mx.sqrt((b ** 2).sum()), 1e-30))
                rel_out = rel(e, r)
                if x is not None and tuple(x.shape) == tuple(e.shape):
                    xf = x.astype(mx.float32)
                    rel_upd, gate = rel(e - xf, r - xf), "update"
                else:
                    rel_upd, gate = rel_out, "output"
                eager = sorted(f"{n}.{k}" for n in d_e[i] for k in d_e[i][n] if d_e[i][n][k] != d_r[n].get(k))
                rows.append({"layer": name, "gate": gate, "rel_l2": rel_upd, "rel_l2_output": rel_out,
                             "max_abs": float(mx.abs(e - r).max()), "reads_training": reads,
                             "flags_flipped": flags, "went_eager": eager})
            g2 = status_snapshot()
        finally:
            c.reset()
        bad = [r["layer"] for r in rows if not r["rel_l2"] <= tol]
        worst = max(rows, key=lambda r: r["rel_l2"]) if rows else {"layer": None, "rel_l2": 0.0}
        rep = {"ok": not bad, "tol_rel_l2": tol, "gate": "rel L2 of the layer update (y - x)",
               "failing_layers": bad, "worst_layer": worst["layer"], "worst_rel_l2": worst["rel_l2"],
               "worst_rel_l2_output": max((r["rel_l2_output"] for r in rows), default=0.0),
               "grad_mode": self.grad_mode, "probe_shape": list(ids.shape),
               "mode_dependent_layers": sum(bool(r["reads_training"] or r["flags_flipped"]) for r in rows),
               "global_receipts": {"eval": status_delta(g0, g1), "reference": status_delta(g1, g2)},
               "layers": rows}
        self.paths["layer_reference"] = rep
        self.log(f"layer reference check: ok={rep['ok']} worst layer {worst['layer']} update rel L2 "
                 f"{worst['rel_l2']:.3g}" + (f"; failing {bad}" if bad else ""))
        return rep

    # ------------------------------------------------------------------ gradients

    def set_layer_masks(self, l, sl, mean=None, sink_layers=None, boxes=()):
        c = self.ctl
        c.reset()
        c.mean = mean
        c.sink_layers = self.L if sink_layers is None else sink_layers
        c.core = {l: sl["core"]} if "core" in sl else None
        if boxes:
            c.boxes = list(boxes)

    def value_and_grad(self, ids, m, head_loss, mean=None, layerwise=None):
        """Loss and d loss / d m for a core mask m (per-layer list or [L, N]); head_loss(post-norm
        hidden) -> scalar. Layer-wise when checked (``check_layerwise_grad``), else whole graph."""
        layerwise = self.layerwise_ok if layerwise is None else layerwise
        if layerwise:
            return self._layerwise(ids, {"core": m}, lambda l, sl: self.set_layer_masks(l, sl, mean), head_loss)
        return self._whole(ids, m, head_loss, mean)

    def _whole(self, ids, m, head_loss, mean):
        import mlx.core as mx
        c = self.ctl
        ms = list(m) if not isinstance(m, mx.array) else m

        def f(mm):
            c.reset()
            c.mean, c.sink_layers, c.core = mean, self.L, mm
            try:
                return head_loss(self.hidden(ids))
            finally:
                c.reset()
        # one graph: forward and backward are the same trace, so in layer_ref mode every
        # backpropagated module (all layers, final norm, head) runs its reference path here
        with self._ref([*self.layers, *self._tail_modules()]):
            loss, g = mx.value_and_grad(f)(ms)
        return loss, g

    def _layerwise(self, ids, trainables, apply, head_loss):
        """The 27B layerwise_value_and_grad, with recorded layer arguments instead of rebuilt masks."""
        import mlx.core as mx
        keys = sorted(trainables)
        self.ctl.reset()
        # the residual forward runs through the trunk (its scopes, its fused routes), each layer with
        # its own mask applied as a constant; the backward linearises around exactly these residuals
        outs = []
        stopped = {kk: [mx.stop_gradient(v) for v in trainables[kk]] for kk in keys}
        # MoE sites: record the routing of the forward the backward linearises around (eval: the fused routers) and
        # replay it inside each layer's reference vjp, so a near-tie that the reference router resolves differently
        # cannot change which experts the gradient is taken through (review 2026-11-15; neuron and expert units)
        routes = {}

        def cap(l, kind, p):
            if kind in ("qwen_next", "gemma4_experts") and "inds" in p:
                routes[l] = (mx.stop_gradient(p["inds"]), mx.stop_gradient(p["w"]))

        def before(l):
            apply(l, {kk: stopped[kk][l] for kk in keys})
            self.ctl.stats = cap
        recs, _ = self.record(ids, before=before, outputs=outs)
        if len(outs) != self.L:
            raise RuntimeError(f"_layerwise: {len(outs)} layer outputs for {self.L} layers")
        hs = [r[0] for r in recs] + [outs[-1]]
        mx.eval(hs)
        del outs
        self.ctl.reset()
        with self._ref(self._tail_modules()):
            (loss,), (gh,) = mx.vjp(lambda x: head_loss(self.final_norm(x)), [hs[-1]], [mx.array(1.0)])
            mx.eval(loss, gh)
        grads = {kk: [None] * self.L for kk in keys}
        for l in reversed(range(self.L)):
            _, a, k = recs[l]

            def fl(x, *vals, l=l, a=a, k=k):
                apply(l, dict(zip(keys, vals)))
                if l in routes:
                    self.ctl.replay = {l: routes[l]}
                return _hid(self.layers[l](x, *a, **k))
            with self._ref([self.layers[l]]):           # only layer l leaves the fused path
                _, cots = mx.vjp(fl, [hs[l], *[trainables[kk][l] for kk in keys]], [gh])
                self.ctl.reset()
                gh = cots[0]
                for i, kk in enumerate(keys):
                    grads[kk][l] = cots[1 + i]
                mx.eval(gh, *cots[1:])
            hs[l + 1] = None
            mx.clear_cache()
        out = {}
        for kk in keys:
            v = grads[kk]
            out[kk] = mx.stack(v) if len({tuple(x.shape) for x in v}) == 1 else v
        return loss, out

    def check_layerwise_grad(self, ids, mean=None, tol=0.15, seed=0, min_layer_cos=0.95, min_cos=0.99):
        """Layer-wise gradients must equal whole-graph gradients at a random 70% mask.

        The two paths are the same graph in a different reduction order, so in bf16 on the GPU they
        differ by noise that grows with depth. Measured on Qwen3.8-27B-BF16-q8-mlx2@2dd8d304
        (2026-11-15):
        - 4-layer slice: relative L2 about 0.01;
        - all 64 layers: relative L2 0.081, pooled cosine 0.9969, worst layer cosine 0.9895, the
          loss identical to 1e-4, and the difference spread over every layer.
        A real replay defect (a dropped argument, a wrong layer) gives a layer cosine far from 1.
        The guards are therefore the cosines (pooled >= min_cos, every layer >= min_layer_cos).
        Relative L2 <= tol (0.15, about 2x the 64-layer noise) only catches gross scale errors.
        Pass --grad-tol to change it for a donor.

        In layer_ref mode the two sides are: layer-wise = residuals from the eval (fused) forward,
        each layer's vjp on its reference path; whole-graph = one value_and_grad with every layer
        (and the final norm + head) on its reference path, so its forward is the reference forward.
        They differ by (a) reduction order, as above, and (b) the fused-vs-reference gap of the
        residuals the layer-wise backward linearises around. (b) is bounded per layer by
        ``check_layer_reference`` and is zero where the two paths are the same arithmetic (the CPU
        toys). So this gate keeps testing the replay (arguments, layer order, masks); the
        fused/reference agreement has its own gate. The target logits are the eval-mode ones on
        both sides."""
        if not self.replay_ok:
            self.layerwise_ok = False
            self.paths["layerwise_grad"] = "replay path disabled"
            return False
        pr = self._grad_probe(ids, mean, seed)
        cmp = pr["whole_graph"]
        self.layerwise_ok = self._whole_graph_ok(cmp, tol, min_cos, min_layer_cos)
        self.paths["layerwise_grad"] = {**cmp, "gate": "whole_graph", "grad_mode": self.grad_mode, "tol_rel_l2": tol,
                                        "min_cos": min_cos, "min_layer_cos": min_layer_cos,
                                        "loss_layerwise": pr["loss_layerwise"], "loss_whole": pr["loss_whole"],
                                        "ok": self.layerwise_ok}
        self.log(f"layerwise grad check: {self.paths['layerwise_grad']}")
        return self.layerwise_ok

    @staticmethod
    def _whole_graph_ok(cmp, tol, min_cos, min_layer_cos):
        return cmp["rel_l2"] <= tol and cmp["cosine"] >= min_cos and cmp["min_layer_cosine"] >= min_layer_cos

    def _probe_mask(self, seed):
        rng = np.random.default_rng(seed)
        return self._layout([(rng.random(n) > 0.3).astype(np.float32) for n in self.widths])

    def _layout(self, rows):
        """Per-layer numpy rows -> the mask layout the carve uses ([L, N] when uniform, else a list)."""
        import mlx.core as mx
        rows = [mx.array(np.asarray(r, dtype=np.float32)) for r in rows]
        return mx.stack(rows) if self.N is not None else rows

    def _probe_loss(self, ids):
        """The gate's target logits (eval mode) and head loss: KL to the unmasked model."""
        import mlx.core as mx
        from .carving import kl_rows
        self.ctl.reset()
        pos = list(range(1, ids.shape[1]))
        lc = mx.stop_gradient(self.logits_at(ids, pos))
        mx.eval(lc)

        def hl(hid):
            return kl_rows(lc, self.head(hid[:, mx.array(pos)])).mean()
        return hl

    def _grad_probe(self, ids, mean, seed):
        """Layer-wise and whole-graph gradients at the gate's random 70% mask (train_mode on)."""
        m = self._probe_mask(seed)
        self.train_mode(True)
        try:
            hl = self._probe_loss(ids)
            lw_loss, gl = self._layerwise(ids, {"core": m}, lambda l, sl: self.set_layer_masks(l, sl, mean), hl)
            wg_loss, gw = self._whole(ids, m, hl, mean)
        finally:
            self.train_mode(False)
        return {"mask": m, "hl": hl, "grad": gl["core"], "whole_graph": grad_compare(gl["core"], gw),
                "loss_layerwise": float(lw_loss), "loss_whole": float(wg_loss)}

    def _masked_loss(self, ids, m, hl, mean, stats=None, replay=None):
        """L(m) on the forward the model is in (eval = fused in layer_ref): no gradient. replay
        ({layer: (inds, w)}) freezes the MoE sites' routing to those routes and weights."""
        import mlx.core as mx
        c = self.ctl
        c.reset()
        c.mean, c.sink_layers, c.core = mean, self.L, m
        c.stats = stats
        c.replay = replay
        try:
            v = hl(self.hidden(ids))
            mx.eval(v)
            return float(v)
        finally:
            c.reset()

    def _fd_loss(self, ids, m, hl, mean, stats=None, replay=None):
        """L(m) as ``_layerwise`` computes its loss: the trunk's own forward (no reference scopes,
        ``record``), then the final norm + head on their reference path. replay freezes MoE routing."""
        import mlx.core as mx
        c = self.ctl
        c.reset()
        c.mean, c.sink_layers, c.core = mean, self.L, m
        c.stats, c.replay = stats, replay
        try:
            outs = []
            self.record(ids, outputs=outs)
            h = outs[-1]
            mx.eval(h)
        finally:
            c.reset()
        with self._ref(self._tail_modules()):
            v = hl(self.final_norm(h))
            mx.eval(v)
        return float(v)

    def _routes(self, ids, m, hl, mean, reference=False):
        """{layer: top-k expert ids [..., k] (numpy)} of every MoE site on the masked forward, read
        from the sites' own routing (the ``stats`` hook), on the fused or the reference forward."""
        routes, self._last_route_w = {}, {}

        def st(l, kind, p):
            if kind != "dense" and "inds" in p:
                routes[l] = p["inds"]
                self._last_route_w[l] = (p["inds"], p["w"])      # for routing replay (frozen-routing FD)
        ctx = reference_mode([*self.layers, *self._tail_modules()]) if reference else _null()
        with ctx:
            loss = self._masked_loss(ids, m, hl, mean, stats=st)
        return loss, {l: np.array(v) for l, v in routes.items()}

    def _routed_units(self, routes):
        """Per layer, the unit ids routed on the probe (MoE sites; neuron sites expand each expert
        to its F neurons), or None for a layer with no routing (dense)."""
        out = []
        for l, s in enumerate(self.sites):
            if l not in routes:
                out.append(None)
                continue
            e = np.unique(routes[l].astype(np.int64).ravel())
            out.append((e[:, None] * s.F + np.arange(s.F)[None]).ravel() if s.units == "neuron" else e)
        return out

    DIRECTIONAL_EPS = (2.0 ** -7, 2.0 ** -6, 2.0 ** -5, 2.0 ** -3)   # exact in bf16 at 1 +- e and 0 +- e
    DIRECTIONAL_PREF_EPS = 2.0 ** -6

    def check_directional_grad(self, ids, mean=None, seed=0, k=8, n_sparse=2, eps=None,
                               tol=0.10, min_corr=0.99, outlier_x=3.0, same_fn_tol=1e-3, pref_eps=None,
                               min_clean=None, norm_ratio=None, min_pair_corr=None):
        """layer_ref gate: the layer-wise gradient is the derivative of the function the carve
        optimises, the masked FUSED (eval) forward.

        A whole-graph comparison is the wrong gate here: the all-reference forward routes some
        tokens to other experts (residual drift of ~1% flips near-tie top-k picks), and an
        expert-mask gradient is zero for an unrouted expert, so the two gradients are each correct
        for a different function (FN smoke 2: rel L2 0.247 with the per-layer update check at
        0.011). Instead, at the gate's random 70% mask m (not all-ones: 30% of units dropped, so
        both kept and dropped units are probed; the mask is a continuous multiplier in the forward, on MoE sites of the routed
        weight), for k directions delta (k - n_sparse dense Rademacher over every unit, n_sparse
        Rademacher over only the units routed on the probe -- or a random 5% of units on a donor
        with no routed sites) compare g . delta with the central difference
        [L(m + e delta) - L(m - e delta)] / 2e on the fused forward, for every e in ``eps``.

        The steps are powers of two, so 1 +- e and 0 +- e are exact in bf16 (a 3e-3 step rounds to
        0.65 e at 1 in bf16). bf16 forward noise swamps small steps and large steps cross routing
        boundaries, so the gate uses two ADJACENT e on the plateau. Each +-e pair of forwards also
        records how many routed (token, slot) experts differ between L(m + e d) and L(m - e d)
        (``pm_flips``). L is only piecewise smooth in m on MoE sites, and a flipped cell's central
        difference measures the jump (~jump/2e), not the derivative, so flipped cells are excluded
        and listed; on the CPU toys every g/fd disagreement is a flipped cell. Pair choice, never
        consulting g: among adjacent pairs with >= min_clean (default max(4, k/2)) directions
        flip-free at both e, the best agreement of the two steps' central differences (mean
        relative difference over those directions); pref_eps (2^-6) wins only a near tie (within
        2x of the best). Pass: (a) the gradient's forward and the finite differences' forward are
        the same function: |loss_layerwise - loss_fused| / |loss_fused| <= same_fn_tol; (b) not
        vacuous: max |fd| > 0; (c) a usable pair exists; (d) at both e of the pair, over its
        flip-free directions, median relative error |g.d - fd| / max(|fd|, floor) <= tol and
        Pearson correlation of g.d against fd >= min_corr; floor = 0.1 x the RMS of fd there (a
        direction with a ~zero derivative is neither a 0/0 nor a pair-selection outlier).
        Directions whose error exceeds outlier_x x the median at a gate e are listed (most likely
        a step that crossed a routing boundary); every per-direction, per-e value is recorded.

        Diagnostics (not gated): the whole-graph comparison, the fused-vs-reference loss gap, and
        the share of routed (token, slot) experts that differ between the fused and the reference
        forward at m, per MoE layer and in total."""
        if not self.replay_ok:
            self.layerwise_ok = False
            self.paths["layerwise_grad"] = "replay path disabled"
            return False
        eps = tuple(eps or self.DIRECTIONAL_EPS)
        pref_eps = self.DIRECTIONAL_PREF_EPS if pref_eps is None else pref_eps
        pr = self._grad_probe(ids, mean, seed)
        m, hl = pr["mask"], pr["hl"]
        g = [np.asarray(pr["grad"][l], dtype=np.float64) for l in range(self.L)]
        mrows = [np.asarray(m[l], dtype=np.float64) for l in range(self.L)]
        self.train_mode(True)                         # the fused forward (layer_ref keeps eval)
        try:
            loss_f, routes_f = self._routes(ids, m, hl, mean)
            import mlx.core as mx
            replay = {l: (mx.stop_gradient(i), mx.stop_gradient(w)) for l, (i, w) in self._last_route_w.items()}
            moe_kinds = {s_.kind for s_ in self.sites if s_.is_moe}
            frozen = bool(replay) and moe_kinds <= {"qwen_next", "gemma4_experts"}
            # the function the layer-wise tape differentiates: the fused trunk, the final norm + head on
            # their reference path (``_layerwise``), routing ids frozen at m (frozen mode). The FD forwards
            # use the same tail, so the same-function check compares like with like; the all-fused loss
            # is kept as a diagnostic.
            loss_fn = self._fd_loss(ids, m, hl, mean)
            loss_replay = self._fd_loss(ids, m, hl, mean, replay=replay) if frozen else None
            loss_r, routes_r = self._routes(ids, m, hl, mean, reference=True)
            routed = self._routed_units(routes_f)
            rng = np.random.default_rng(seed + 1)
            dirs, kinds = [], []
            has_routes = any(r is not None for r in routed)
            for j in range(k):
                sparse = j >= k - n_sparse
                d = []
                for l, n in enumerate(self.widths):
                    row = np.zeros(n)
                    if not sparse:
                        row = rng.choice([-1.0, 1.0], n)
                    elif has_routes:
                        if routed[l] is not None and len(routed[l]):
                            row[routed[l]] = rng.choice([-1.0, 1.0], len(routed[l]))
                    elif n:
                        idx = rng.choice(n, max(1, n // 20), replace=False)
                        row[idx] = rng.choice([-1.0, 1.0], len(idx))
                    d.append(row)
                dirs.append(d)
                kinds.append(("sparse_routed" if has_routes else "sparse_random") if sparse else "dense")
            gd = np.array([sum(float(gl @ dl) for gl, dl in zip(g, d)) for d in dirs])
            fd, pm_flips, fd_free, free_flips, rmis = {}, {}, {}, {}, []

            def routed_loss(mm, rp=None):
                rs = {}

                def st(l, kind, p):
                    if kind != "dense" and "inds" in p:
                        rs[l] = np.array(p["inds"])
                return self._fd_loss(ids, self._layout(mm), hl, mean, stats=st, replay=rp), rs

            def n_flips(a, b):
                n = 0
                for l, x in a.items():
                    y = b.get(l)
                    if y is not None and x.shape == y.shape:
                        n += int((~(x[..., :, None] == y[..., None, :]).any(-1)).sum())
                return n
            for e in eps:
                vals, fl, fv = [], [], []
                for d in dirs:
                    mp = [mr + e * dl for mr, dl in zip(mrows, d)]
                    mn = [mr - e * dl for mr, dl in zip(mrows, d)]
                    up, ru = routed_loss(mp)
                    dn, rd = routed_loss(mn)
                    fv.append((up - dn) / (2 * e))
                    fl.append(n_flips(ru, rd))
                    if frozen:     # the function g differentiates: expert ids frozen at m
                        (fu, rpu), (fdn, rpd) = routed_loss(mp, replay), routed_loss(mn, replay)
                        vals.append((fu - fdn) / (2 * e))
                        # replay must engage: the replayed forwards route exactly as at m
                        rmis.append(n_flips(routes_f, rpu) + n_flips(routes_f, rpd))
                fd_free[e] = np.array(fv)
                fd[e] = np.array(vals) if frozen else fd_free[e]
                pm_flips[e] = rmis[-len(dirs):] if frozen else fl
                free_flips[e] = fl
        finally:
            self.train_mode(False)

        # A (direction, e) cell whose +e and -e forwards route some (token, slot) differently is a
        # central difference across a routing boundary (L is only piecewise smooth in m on MoE
        # sites): it measures the jump, not the derivative, so it is excluded from the gate. The
        # gate needs at least min_clean directions without a +- flip at both e of its pair.
        clean = {e: np.array([f == 0 for f in pm_flips[e]]) for e in eps}
        min_clean = max(4, (k + 1) // 2) if min_clean is None else min_clean

        def floor_of(*vs):
            v = np.concatenate([np.asarray(x) for x in vs])
            return max(1e-30, 0.1 * float(np.sqrt(np.mean(v ** 2)))) if v.size else 1e-30

        def rel_err(e, sel=None):
            sel = np.ones(k, bool) if sel is None else sel
            return np.abs(gd[sel] - fd[e][sel]) / np.maximum(np.abs(fd[e][sel]), floor_of(fd[e][sel]))

        def corr(e, sel):
            a, b = gd[sel] - gd[sel].mean(), fd[e][sel] - fd[e][sel].mean()
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(a @ b / den) if den > 0 else (1.0 if np.allclose(gd[sel], fd[e][sel]) else 0.0)

        pairs = list(itertools.pairwise(eps))

        def both(p):
            return clean[p[0]] & clean[p[1]]

        def agree(p):
            # mean (not median) relative difference of the two steps' central differences over the
            # directions clean at both: residual curvature or noise moves the pair off the plateau.
            # The choice never looks at g.
            sel = both(p)
            if not sel.any():
                return float("inf")
            a, b = fd[p[0]][sel], fd[p[1]][sel]
            return float(np.mean(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), floor_of(a, b))))
        def pair_corr(p):
            # Pearson correlation of the two steps' FD vectors over the directions clean at both: a pair
            # that agrees with itself measures a stable derivative (not noise, not curvature)
            sel_ = both(p)
            if sel_.sum() < 2:
                return 0.0
            a, b = fd[p[0]][sel_] - fd[p[0]][sel_].mean(), fd[p[1]][sel_] - fd[p[1]][sel_].mean()
            den = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(a @ b / den) if den > 0 else 0.0
        usable = [p for p in pairs if int(both(p).sum()) >= min_clean
                  and (min_pair_corr is None or pair_corr(p) >= min_pair_corr)]
        if usable:
            best = min(usable, key=agree)
            near = [p for p in usable if pref_eps in p and agree(p) <= 2 * agree(best)]
            pair = min(near, key=agree) if near else best
        else:
            pair = max(pairs, key=lambda p: (int(both(p).sum()), -agree(p)))
        sel = both(pair)
        per_eps = {f"{e:g}": {"fd": fd[e].tolist(), "pm_flips": pm_flips[e], "clean": int(clean[e].sum()),
                              "rel_err_all": rel_err(e).tolist(),
                              "median_rel_err_clean": float(np.median(rel_err(e, clean[e]))) if clean[e].any() else None}
                   for e in eps}
        def nratio(e, sel):
            nf = float(np.linalg.norm(fd[e][sel]))
            return float(np.linalg.norm(gd[sel])) / nf if nf > 0 else float("nan")
        gate = {f"{e:g}": {"directions": int(sel.sum()), "norm_ratio": nratio(e, sel) if sel.any() else None,
                           "median_rel_err": float(np.median(rel_err(e, sel))) if sel.any() else 1e30,
                           "corr": corr(e, sel) if sel.sum() > 1 else 0.0} for e in pair}
        fd_max = max(float(np.abs(fd[e]).max()) for e in eps)
        same_fn_gap = abs(pr["loss_layerwise"] - loss_fn) / max(1e-30, abs(loss_fn))
        replay_gap = abs(loss_replay - loss_fn) / max(1e-30, abs(loss_fn)) if frozen else None
        fails = []
        if not same_fn_gap <= same_fn_tol:
            fails.append(f"gradient forward and FD forward differ (loss rel gap {same_fn_gap:.3g} > {same_fn_tol})")
        replay_mismatch = int(sum(rmis)) if frozen else None
        if frozen and replay_mismatch:
            fails.append(f"routing replay did not engage: {replay_mismatch} routed slots differ from m's routing")
        if frozen and not replay_gap <= same_fn_tol:
            fails.append(f"routing replay changes L(m) (rel gap {replay_gap:.3g} > {same_fn_tol})")
        if not fd_max > 0:
            fails.append("vacuous: every central difference is 0 (masks not wired into the forward?)")
        if not usable:
            fails.append(f"no adjacent eps pair has {min_clean} directions free of +- routing flips"
                         + (f" and FD self-agreement corr >= {min_pair_corr}" if min_pair_corr is not None else "")
                         + f" (best {int(sel.sum())}/{k}; pair corr "
                         + ", ".join(f"{a:g}/{b:g} {pair_corr((a, b)):.3f}" for a, b in pairs) + ")")
        for e, v in gate.items():
            if not (v["median_rel_err"] <= tol and v["corr"] >= min_corr):
                fails.append(f"eps {e}: median rel err {v['median_rel_err']:.3g} (tol {tol}), corr {v['corr']:.4g} "
                             f"(min {min_corr}) over {v['directions']} flip-free directions")
        if norm_ratio is not None:      # |g.d| / |fd| over the gated directions: catches a scale defect directly
            lo, hi = norm_ratio
            for e, v in gate.items():
                if not (v["norm_ratio"] is not None and lo <= v["norm_ratio"] <= hi):
                    fails.append(f"eps {e}: norm ratio |g.d|/|fd| {v['norm_ratio']} outside [{lo}, {hi}]")
        ok = not fails
        outliers = sorted({int(j) for e in pair for j, r in zip(np.flatnonzero(sel), rel_err(e, sel))
                           if r > outlier_x * max(gate[f"{e:g}"]["median_rel_err"], 1e-12) and r > tol})
        flips = {}
        slots = flipped = 0
        for l in sorted(routes_f):
            a, b = routes_f[l], routes_r.get(l)
            if b is None or a.shape != b.shape:
                continue
            match = (a[..., :, None] == b[..., None, :]).any(-1)
            flips[str(l)] = float(1.0 - match.mean())
            slots += match.size
            flipped += int((~match).sum())
        self.layerwise_ok = bool(ok)
        rep = {"gate": "directional", "grad_mode": self.grad_mode, "ok": self.layerwise_ok,
               "fails": fails, "eps": list(eps), "pref_eps": pref_eps, "gate_eps": list(pair),
               "pair_agreement": {f"{a:g}/{b:g}": (None if agree((a, b)) == float("inf") else agree((a, b)))
                                  for a, b in pairs},
               "pair_pm_flips": {f"{a:g}/{b:g}": sum(pm_flips[a]) + sum(pm_flips[b]) for a, b in pairs},
               "pair_fd_corr": {f"{a:g}/{b:g}": pair_corr((a, b)) for a, b in pairs},
               "min_pair_corr": min_pair_corr,
               "pair_clean_directions": {f"{a:g}/{b:g}": int(both((a, b)).sum()) for a, b in pairs},
               "min_clean": min_clean, "gate_directions": np.flatnonzero(sel).tolist(),
               "excluded_directions": np.flatnonzero(~sel).tolist(),
               "fd_routing": "frozen at m (replayed expert ids; qwen_next weights recomputed)" if frozen else "free",
               "replay_rel_gap": replay_gap, "replay_route_mismatch": replay_mismatch,
               "free_flips_total": {f"{e:g}": int(sum(free_flips[e])) for e in eps},
               "replay_note": ("qwen_next: ids frozen, weights live (as the gradient); gemma4_experts: ids and "
                               "weights frozen (router_sg), so a missing router-weight gradient is not tested"),
               "free_routing_diag": {f"{e:g}": {"fd": fd_free[e].tolist(), "pm_flips": free_flips[e],
                                                "median_rel_err": float(np.median(np.abs(gd - fd_free[e]) /
                                                    np.maximum(np.abs(fd_free[e]), floor_of(fd_free[e]))))}
                                     for e in eps},
               "same_fn_rel_gap": same_fn_gap, "loss_fd_function": loss_fn,
               "fd_tail": "reference (as the layer-wise loss)",
               "objective_note": ("on MoE sites the carve objective is discontinuous across top-k routing flips; the "
                                  "gate checks the gradient the carve computes (expert ids frozen at m, routing "
                                  "weights live). The exact masked eval on the free-routing fused forward "
                                  "(bisect, ladder) is the final word."), "same_fn_tol": same_fn_tol, "fd_max_abs": fd_max,
               "norm_ratio_bounds": list(norm_ratio) if norm_ratio else None, "tol_median_rel_err": tol, "min_corr": min_corr, "k": k, "n_sparse": n_sparse,
               "directions": [{"kind": kinds[j], "nnz": int(sum(int((d != 0).sum()) for d in dirs[j])),
                               "g_dot_d": float(gd[j]),
                               **{f"fd@{e:g}": float(fd[e][j]) for e in eps},
                               **{f"pm_flips@{e:g}": pm_flips[e][j] for e in eps},
                               **{f"rel_err@{e:g}": float(rel_err(e)[j]) for e in eps}} for j in range(k)],
               "per_eps": per_eps, "gate_values": gate, "outlier_directions": outliers,
               "loss_layerwise": pr["loss_layerwise"], "loss_fused": loss_f, "loss_reference": loss_r,
               "loss_gap": loss_f - loss_r,
               "routing_flips": {"per_layer": flips, "total": flipped / slots if slots else 0.0, "slots": slots},
               "whole_graph": {**pr["whole_graph"], "loss_whole": pr["loss_whole"], "diagnostic": True}}
        self.paths["layerwise_grad"] = rep
        self.log(f"directional grad check: ok={rep['ok']} {fails} gate eps {pair} "
                 f"{ {e: (round(v['median_rel_err'], 4), round(v['corr'], 4)) for e, v in gate.items()} } "
                 f"outliers {outliers} loss gap {rep['loss_gap']:.3g} routing flips {rep['routing_flips']['total']:.3g}")
        return self.layerwise_ok

    @contextmanager
    def fp32_activations(self):
        """Run the residual stream in float32 (the embedding's output is cast; quantized and bf16
        weights stay as they are, and every op downstream promotes). For gradient references only."""
        emb = self._embed()
        base = type(emb)

        def call(self_, *a, _base=base, **k):
            import mlx.core as mx
            return _base.__call__(self_, *a, **k).astype(mx.float32)
        emb.__class__ = type(f"F32{base.__name__}", (base,), {"__call__": call})
        try:
            import mlx.core as mx
            probe = self.hidden(mx.array([[0, 1]]))
            if probe.dtype != mx.float32:
                raise RuntimeError(f"fp32_activations: the residual stream is {probe.dtype}, not float32 "
                                   f"(the model casts back after the embedding); no float32 reference")
            yield
        finally:
            emb.__class__ = base

    def check_ones_parity(self, ids):
        """core = all ones must reproduce the base logits (exact for dense sites; MoE sites are
        re-implemented, so report the difference and top-1 agreement)."""
        import mlx.core as mx
        self.ctl.reset()
        base = self.logits_at(ids)
        self.ctl.core = [mx.ones((n,)) for n in self.widths]
        ones = self.logits_at(ids)
        self.ctl.reset()
        mx.eval(base, ones)
        lb = base.astype(mx.float32) - mx.logsumexp(base.astype(mx.float32), axis=-1, keepdims=True)
        lo = ones.astype(mx.float32) - mx.logsumexp(ones.astype(mx.float32), axis=-1, keepdims=True)
        kl = (mx.exp(lb) * (lb - lo)).sum(-1)
        r = {"max_abs": float(mx.abs(ones - base).max()),
             "top1_agree": float((ones.argmax(-1) == base.argmax(-1)).mean()),
             "positions": int(base.size // base.shape[-1]),
             "kl_mean": float(kl.mean()), "kl_max": float(kl.max())}
        self.paths["ones_parity"] = r
        return r

    def check_site_parity(self, ids, tol=2e-2):
        """Every unit site engages on a forward, and every MoE site's re-implemented routing
        reproduces its block's own output (relative L2 <= tol; dense sites mask the exact
        down_proj input and report 0). A wrong router law or a fused path that skips a site
        fails here, before a capture or a carve is read. Returns the report (``ok``)."""
        import mlx.core as mx
        self.ctl.reset()
        self.ctl.verify = {}
        try:
            mx.eval(self.full_logits(ids), *[v for v in self.ctl.verify.values() if isinstance(v, mx.array)])
            got = {l: float(v) for l, v in self.ctl.verify.items()}
        finally:
            self.ctl.reset()
        want = [s.layer for s in self.sites if s.kind is not None]
        missing = [l for l in want if l not in got]
        bad = {l: v for l, v in got.items() if not v <= tol}
        r = {"ok": not missing and not bad, "tol": tol, "max_rel_l2": max(got.values(), default=0.0),
             "not_engaged": missing, "over_tol": bad,
             "per_layer": {str(l): got.get(l) for l in want}}
        self.paths["site_parity"] = r
        return r

    # ------------------------------------------------------------------ masks

    def ones(self):
        import mlx.core as mx
        return mx.ones((self.L, self.N)) if self.N is not None else [mx.ones((n,)) for n in self.widths]

    def zeros(self):
        import mlx.core as mx
        return mx.zeros((self.L, self.N)) if self.N is not None else [mx.zeros((n,)) for n in self.widths]


@contextmanager
def _null():
    yield


def _hid(out):
    """Layers may return (h, kv, offset) (Gemma4): the residual stream is the first element."""
    return out[0] if isinstance(out, tuple) else out


def model_config(*mods) -> dict:
    """Flat text config from ``args``/``config`` of the given modules (text_config merged in)."""
    out = {}
    for m in mods:
        for attr in ("args", "config"):
            c = getattr(m, attr, None)
            if c is None:
                continue
            d = dict(c) if isinstance(c, dict) else dict(getattr(c, "__dict__", {}))
            t = d.get("text_config")
            if t is not None:
                d.update(dict(t) if isinstance(t, dict) else dict(getattr(t, "__dict__", {})))
            for k, v in d.items():
                out.setdefault(k, v)
    return out


def _take(mx, h, pos):
    if isinstance(pos, mx.array) and pos.ndim == 2:
        return mx.take_along_axis(h, pos[..., None], axis=1)
    return h[:, mx.array(pos)]


class TopicsBackend:
    """The amoeba-topics backend interface (encode / set_mask / stats / nll / mean_act, as
    scripts/amoeba_topics_lib.py's Mlx27bBackend on claude/amoeba-topics-20261114) over any donor,
    including MoE donors (units = experts). stats returns per-layer mean |unit signal - mu| over
    positions >= 1: |act - mu| for neurons, routed weight x ||expert row|| for experts."""

    def __init__(self, model_path, log=print, policy=None, host=None):
        self.h = host or Host(model_path, log=log, policy=policy)
        self.L, self.N = self.h.L, self.h.N
        self.widths = list(self.h.widths)      # per layer; 0 where a layer has no unit site
        self.checkpoint, self.fingerprint = self.h.checkpoint, self.h.fingerprint
        self.tok = self.h.tok
        self._mask = None

    def encode(self, text):
        return self.h.encode(text)

    def set_mask(self, mask):
        import mlx.core as mx
        self._mask = None if mask is None else (mx.array(np.asarray(mask, dtype=np.float32))
                                                if self.N is not None else [mx.array(np.asarray(r, dtype=np.float32)) for r in mask])

    def _cfg(self):
        self.h.ctl.reset()
        self.h.ctl.core = self._mask

    def stats(self, ids, mu=None):
        import mlx.core as mx
        if U.neuron_sites(self.h.sites):
            raise SystemExit("TopicsBackend.stats: not implemented for --moe-units neuron (its per-expert one-hot "
                             "would be tokens x E*F); use expert units")
        out = [None] * self.L

        def st(l, kind, p):
            if kind == "dense":
                a = p["act"][:, 1:].astype(mx.float32)
                if mu is not None:
                    a = a - mx.array(np.asarray(mu[l], dtype=np.float32)[:a.shape[-1]])
                out[l] = mx.abs(a).mean(1)
            else:
                ye = mx.linalg.norm(p["rows"].astype(mx.float32), axis=-1) * p["w"]      # B,T,k
                n = self.h.widths[l]
                acc = mx.zeros((ye.shape[0], n))
                for j in range(ye.shape[-1]):
                    oh = (p["inds"][:, 1:, j][..., None] == mx.arange(n)).astype(mx.float32)
                    acc = acc + (oh * ye[:, 1:, j][..., None]).mean(1)
                out[l] = acc
        self._cfg()
        self.h.ctl.stats = st
        mx.eval(self.h.hidden(mx.array(ids)), *[o for o in out if o is not None])
        self.h.ctl.reset()
        return [np.array(o) if o is not None else None for o in out]

    def mean_act(self, seqs, bs=8):
        """Per-layer mean pre-down_proj activation over positions >= 1 (dense sites), as a padded
        [L, max width] float32 array. Expert sites and layers without a site stay 0: their stats
        signal (routed weight x row norm) is not mean-centred."""
        import mlx.core as mx
        W = max(self.widths) if self.widths else 0
        acc = np.zeros((self.L, W), dtype=np.float64)
        cnt = 0
        sums = [None] * self.L

        def st(l, kind, p):
            if kind == "dense":
                sums[l] = p["act"][:, 1:].astype(mx.float32).sum((0, 1))
        by_len = {}
        for j, q in enumerate(seqs):
            by_len.setdefault(len(q), []).append(j)
        for js in by_len.values():
            for b in range(0, len(js), bs):
                ids = mx.array([seqs[j] for j in js[b:b + bs]])
                for l in range(self.L):
                    sums[l] = None
                self.h.ctl.reset()
                self.h.ctl.stats = st
                mx.eval(self.h.hidden(ids), *[x for x in sums if x is not None])
                self.h.ctl.reset()
                for l, x in enumerate(sums):
                    if x is not None:
                        acc[l, :x.shape[0]] += np.array(x)
                cnt += ids.shape[0] * (ids.shape[1] - 1)
                mx.clear_cache()
        return (acc / max(cnt, 1)).astype(np.float32)

    def nll(self, ids):
        import mlx.core as mx
        from .carving import ce_rows
        ids = mx.array(ids)
        self._cfg()
        # per text: mean NLL of targets 2..T-1 from positions 1..T-2, as Mlx27bBackend / TorchBackend
        # (amoeba_topics_capture, _function and _stub index one value per text)
        lg = self.h.logits_at(ids, list(range(1, ids.shape[1] - 1)))
        self.h.ctl.reset()
        ce = ce_rows(lg, ids[:, 2:]).mean(1)
        mx.eval(ce)
        return np.array(ce)
