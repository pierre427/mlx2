"""Carvable unit sites of any donor, found from the module structure (never from a model name).

One site per decoder layer, found in the layer's FFN slot (``experts``, ``mlp``, ``feed_forward``,
``block_sparse_moe``, or ``mixer`` in hybrid blocks whose token mixer and FFN share that slot; a
``mixer`` with ``o_proj``/``out_proj`` and no ``down_proj`` is attention or an SSM and has no site):

  dense   the MLP's intermediate neurons, i.e. the input columns of ``down_proj``. Any gated or
          plain MLP whose output is ``down_proj(act)`` qualifies (SwiGLU, GeGLU, ReLU^2 ...).
          Masking acts on ``act`` (the down_proj input), so it is exact for every activation.
          The width is the down_proj input width in any storage format (bf16, affine 2-8 bit,
          mxfp4, mxfp8, nvfp4, owned NVFP4/FP8 modules, EXL3) and must equal up_proj's output.
  moe     routed experts of a sparse block. Block shapes (``classify``, by structure):
            qwen_next       gate -> softmax top-k -> switch_mlp (+ sigmoid-gated shared expert);
                            Flash Next (qwen4_exp), Qwen3.5/3.6 MoE (qwen3_5_moe)
            nemotron        gate(x) -> (ids, scores) [grouped sigmoid + correction bias];
                            optional latent fc1 -> switch_mlp (SwitchMLP fc1/fc2, ReLU^2) -> sum
                            -> fc2; scores x folded per-expert NVFP4 scale (_routing_weights);
                            + shared_experts(x) on the block input; Nemotron-H / 3.5 Super, Bolt,
                            Lightning ('E' blocks; 'M' Mamba and '*' attention blocks have no site,
                            '-' blocks are dense)
            laguna          gate(x) -> (ids, weights), routed * scale + ungated shared expert
            gemma4_experts  Experts(x, indices, weights) beside Gemma4's parallel dense MLP
            ling            gate(x) -> (ids, weights) -> switch_mlp (+ shared_experts); Kimi, Ling,
                            Xing4.0
            routing         routing(x) -> (ids, weights) -> switch_mlp, no shared; LFM2.5 MoE
            softmax_topk    softmax(gate(x)) -> top_k (/ sum when norm_topk_prob) -> switch_mlp,
                            no shared; Mellum, Qwen3-MoE
            sigmoid_shared  sigmoid(gate(x)) fp32 -> top_k -> / (sum + 1e-20) -> switch_mlp
                            + ungated shared_expert (no scale, no shared gate); Flint
            router_experts  router(x) fp32 -> top_k logits -> softmax over the k -> experts;
                            gpt-oss, GraniteMoE
          Shared experts and parallel dense MLPs always run. Masking is the drop rule: a dropped
          expert's routed weight is multiplied by 0 after selection (no renormalisation, the slot
          is not refilled), so a zeroed-expert checkpoint reproduces it exactly. Selection ids
          carry no gradient. A MoE site re-implements its block's router from these shapes; with
          ``Control.verify`` set, every site records how far that re-implementation is from the
          block's own forward (``Host.check_site_parity``), so a wrong law fails before a map or
          a carve is read, and a site that never ran (a fused path around down_proj) is reported.
          A block whose shape matches none of these refuses at install.
  not covered: DeepSeek-V4.1 (runtime/models/deepseek_v41.py) layers are plain objects over a
          weight store (Block.router(x) -> (ids, scores), inline routed experts, shared_expert(x)),
          not nn.Modules; its drop rule belongs on Block.router's scores, in its own layer stream.

Masks are per layer (sites may differ in width: dense first layers in an MoE model). The serving
form of a mask is ``m = core + sum_b box_b * g_b(x)``, with mean ablation for dense neurons
``used = mu + m * (act - mu)`` (zero ablation when mu is None) and the first-position rule
(position 0 exact in layers < sink_layers). With no core set every site is the base module,
bit for bit (the patched __call__ returns the base class's __call__ unchanged).

``Control.stats`` is a capture hook ``stats(layer, kind, payload)``: dense payload
``{"act": act, "x": mlp_input}``; moe payload ``{"inds", "w", "rows", "x"}`` (``x`` the block input,
``rows`` the switch output per selected expert; for a latent block, the latent-space rows; neuron
units add ``"h"``, the activated intermediate [..., k, F] before masking, and ``"F"``). ``Control.box_out``
(a dict) collects each box's per-layer write (hub-align phase 1 wants the donor's box write).

``Control.replay`` ({layer: (inds, w)}) freezes MoE routing (frozen-routing finite differences and the
layer-reference vjp): qwen_next sites route to the replayed ids and recompute their weights there,
gemma4_experts sites take both; any other MoE kind raises rather than routing freely.
``Control.verify`` measures each MoE site's re-implementation against its block's own forward on the
free (unreplayed) route: a replayed site records nothing (``Host.check_site_parity`` resets the
Control, so replay is never set there). Neuron sites record it too, from an all-open (unmasked) pass
of the switch in pieces against the block's forward.

MoE unit granularity (``install(..., moe_units=)``):

  expert  (default) a unit is a routed expert, as above.
  neuron  a unit is (routed expert e, intermediate neuron f), flat id ``e * F + f``, so a MoE
          site has E*F units (Flash Next: 512 * 640 = 327,680 per layer). Ported from
          scripts/probe_amoeba_unit_box.py@16a1bbe88 (``_neuron_switch_types``, ``_switch_layout``,
          ``_switch_masked``, ``_moe_neuron_call``; see NEURON_PROVENANCE). The switch GLU runs in
          pieces through the module's own (quantized) projections: gate/up -> activation
          h = act(gate) * up -> h[..., e, f] * keep -> down_proj, for the routed experts only,
          following the stock sorted-gather rule of that switch type. Router weights are untouched
          (no renormalisation) and shared experts / parallel dense MLPs always run. qwen_next sites
          route and combine with the stock block's arithmetic (precise softmax in the gate dtype,
          ``(rows * scores).sum`` in the score dtype, then ``+ gate * shared``; in eval mode through
          the block's own routing helpers, in training/reference mode through ``_stock_routing``
          with the expert ids stop-gradiented, the same values), so an all-open neuron mask equals
          the stock block bit for bit wherever the stock block runs its plain switch tail (CPU;
          Metal without the fused-down / weighted-sum / NAX / window kernels). Other kinds keep the
          carve's expert-mode routing and combine, so there an all-open neuron mask equals expert
          mode with every expert kept, bit for bit, and stock as closely as expert mode does (ling:
          bit for bit on CPU; laguna: to rounding, the carve folds the routed scale into the weights
          before the sum). In both, all neurons of the experts in S kept equals the
          drop rule keeping S (down(0) = 0). On qwen_next, expert mode routes with a float32
          softmax and combines in float32, the stock block (and neuron mode) in the gate / score
          dtype: identical in float32, but in bf16 a neuron-vs-expert comparison is close, not
          bitwise (compare neuron masks against expert_to_neuron masks in neuron mode for an exact
          check). Since fnsmoke3c (2026-11-15) expert-mode qwen_next also routes and combines through
          the stock block's helpers (``_qwen_next_neuron_rows`` with the plain switch,
          ``_qwen_next_combine``; the keep weight cast to the score dtype), so expert and neuron mode
          share the stock arithmetic. Only SwitchGLU, FusedDownSwitchGLU and
          FusedGateUpSwitchGLU are accepted and a biased down_proj is refused (a fully masked
          expert would still write its bias): any other MoE site raises, never falling back.
          Ablation: always the drop rule (zero ablation of h[e, f]), also for --ablations mean (the
          mean stage reports the conditional mean of h per neuron over the tokens routed to e).
          Memory: masks are per (e, f) [E*F] vectors; per call only the routed slots' mask rows
          [..., k, F] (same size as h) are gathered, never tokens x E x F, also for gated boxes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Site:
    layer: int
    kind: str | None            # "dense", a MoE kind (see the module docstring), or None
    n: int                      # units at this site
    params_per_unit: int
    module: object = field(repr=False, default=None)     # the patched module (down_proj or MoE block)
    parent: object = field(repr=False, default=None)     # the MLP holding down_proj (dense)
    units: str = "expert"       # MoE sites: "expert" (unit e) | "neuron" (unit e * F + f); dense sites keep the
                                # default, unused (their units are always down_proj input neurons)
    E: int = 0                  # neuron MoE sites: routed experts
    F: int = 0                  # neuron MoE sites: intermediate neurons per expert

    @property
    def is_moe(self) -> bool:
        return self.kind not in (None, "dense")


class Control:
    """Mutable state shared by every patched site of one model (set per forward)."""

    def __init__(self, sites):
        self.sites = sites
        self.reset()

    def reset(self):
        self.core = None            # list per layer of [n_l] masks, or an [L, N] array (uniform), or None
        self.mean = None            # same layout, dense sites only (zero ablation when None)
        self.sink_layers = 0        # position 0 exact in layers < sink_layers
        self.dense_layers = ()      # layers that run whole at every position
        self.boxes = []             # [{"name", "mask", "gate": (W [L,d], b [L]) | None, "hard": bool,
                                    #   "token_gate": [B, T, 1] float | None (precomputed, wins over "gate")}]
        self.stats = None           # callback(layer, kind, payload) or None
        self.stats_fast = False     # with stats: return the captured rows' output, no second MoE call
        self.gate_log = None        # list to append (box, layer, g) or None
        self.box_out = None         # dict (box, layer) -> write, or None
        self.verify = None          # dict layer -> rel L2 of re-implemented vs block output (0.0 dense)
        self.replay = None          # {layer: (inds, w)}: MoE sites use these routes and weights (frozen routing)
        # opt-in, neuron MoE sites only (docs/research/2026-11-20-amoeba-next-experiments.md 1.3 / 4.3); None = unchanged
        self.moe_mean = None        # per layer [E*F] conditional mean of h: a closed slot gets mu instead of 0 (mean
                                    # ablation = the servable per-expert bias sum_f mu W_down[:, f], down_proj is linear)
        self.moe_thresh = None      # {layer: t}: per token, a routed slot's neuron with |h| < t is closed too (TEAL-style
                                    # reference; position 0 stays exact in layers < sink_layers)
        self.moe_open = None        # dict layer -> [open, total] mx scalars, accumulated when moe_thresh is set
        self.router_sg = getattr(self, "router_sg", False)   # set by Host.train_mode, survives reset

    def active(self, layer) -> bool:
        if self.core is None or layer in self.dense_layers:
            return False
        return layer in self.core if isinstance(self.core, dict) else True


def _row(a, l):
    return None if a is None else a[l]


def _check_width(c, l, n, units):
    """Every mask row reaching site l has the site's unit count n (an [E*F] neuron mask must never
    index an expert site, nor an [E] expert mask a neuron site, on a mixed donor)."""
    rows = [("core", c.core[l])] + [(f"box {b.get('name')}", b["mask"][l]) for b in c.boxes]
    for what, r in rows:
        if r.shape[-1] != n:
            raise ValueError(f"units: layer {l} ({units} units) has {n} units but its {what} mask row has "
                             f"{r.shape[-1]}")


def _box_mask(mx, c, l, x):
    """m for layer l: core + sum of (gated) boxes. x = the site's input (for gates)."""
    m = c.core[l]
    for box in c.boxes:
        bm = box["mask"][l]
        if box.get("token_gate") is not None:      # precomputed per-token open bits [B, T, 1] (next-token gating)
            m = m + bm * box["token_gate"]
        elif box.get("gate") is not None:
            W, b = box["gate"]
            g = mx.sigmoid((x.astype(mx.float32) @ W[l][:, None]) + b[l])      # B,T,1
            if box.get("hard"):
                g = (g > 0.5).astype(mx.float32)
            if c.gate_log is not None:
                c.gate_log.append((box.get("name"), l, g))
            m = m + bm * g
        else:
            m = m + bm
    return m


def _first_position(mx, c, l, exact, used):
    if l < c.sink_layers and used.ndim >= 2 and used.shape[1] > 1:
        return mx.concatenate([exact[:, :1], used[:, 1:]], axis=1)
    return used


def _replayed(c, l, kind):
    """The routing recorded for layer l when the Control replays it, else None: (inds, w) for Gemma4
    experts (the router runs outside the site, stop-gradiented in training mode, so ids and weights are
    both frozen), inds for qwen_next (the ids are stop-gradiented, the weights are differentiable, so
    they are recomputed at the replayed ids)."""
    if c.replay is None or l not in c.replay:
        return None
    if kind not in ("qwen_next", "gemma4_experts"):
        raise NotImplementedError(f"routing replay is not implemented for {kind} sites")
    inds, w = c.replay[l]
    return inds if kind == "qwen_next" else (inds, w)


def _gather_scores(mx, probs, inds, norm):
    w = mx.take_along_axis(probs, inds, axis=-1)
    return w / w.sum(axis=-1, keepdims=True) if norm else w


def _ling_rows(mx, self, x, router_sg=False, switch=None):
    # routers whose gather still carries gradients on its indices (Ling's group_expert_select) get a
    # constant input in training mode: routing weights are then treated as constants for mask
    # gradients (Kimi's router stops gradients on its ids itself, and keeps the exact path)
    gx = mx.stop_gradient(x) if router_sg and not hasattr(self.gate, "training_indices") else x
    ids, w = self.gate(gx)
    ids = mx.stop_gradient(ids)
    rows = (switch or self.switch_mlp)(x, ids)
    shared = self.shared_experts(x) if getattr(self, "shared_experts", None) is not None else 0
    return ids, w.astype(mx.float32), rows, shared


def _laguna_rows(mx, self, x, switch=None):
    ids, w = self.gate(x)                 # Laguna's router stops gradients on its ids itself
    rows = (switch or self.switch_mlp)(x, mx.stop_gradient(ids))
    return ids, w.astype(mx.float32) * self.scale, rows, self.shared_expert(x)


def _nemotron_rows(mx, self, x, router_sg=False, switch=None):
    ids, scores = self.gate(mx.stop_gradient(x) if router_sg else x)
    ids = mx.stop_gradient(ids)
    latent = getattr(self, "moe_latent_size", None) is not None
    xl = self.fc1_latent_proj(x) if latent else x
    rows = (switch or self.switch_mlp)(xl, ids)
    w = self._routing_weights(ids, scores).astype(mx.float32)
    shared = self.shared_experts(x) if getattr(self, "shared_experts", None) is not None else 0
    return ids, w, rows, shared, (self.fc2_latent_proj if latent else None)


def _routing_rows(mx, self, x, router_sg=False, switch=None):
    ids, w = self.routing(mx.stop_gradient(x) if router_sg else x)
    ids = mx.stop_gradient(ids)
    return ids, w.astype(mx.float32), (switch or self.switch_mlp)(x, ids), 0


def _topk(mx, scores, k):
    return mx.stop_gradient(mx.argpartition(scores, kth=-k, axis=-1)[..., -k:])


def _softmax_topk_rows(mx, self, x, switch=None):
    # exactly the block's own law, in the gate's dtype (Mellum, standard_decoder MoE): a float32
    # upcast changes which experts win bf16 probability ties
    probs = mx.softmax(self.gate(x), axis=-1, precise=True)
    inds = _topk(mx, probs, self.top_k)
    w = mx.take_along_axis(probs, inds, axis=-1)
    if self.norm_topk_prob:
        w = w / mx.sum(w, axis=-1, keepdims=True)
    return inds, w.astype(mx.float32), (switch or self.switch_mlp)(x, inds), 0


def _sigmoid_shared_rows(mx, self, x, switch=None):
    scores = mx.sigmoid(self.gate(x).astype(mx.float32))
    inds = _topk(mx, scores, self.top_k)
    w = mx.take_along_axis(scores, inds, axis=-1)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    return inds, w, (switch or self.switch_mlp)(x, inds), self.shared_expert(x)


def _router_experts_rows(mx, self, x, switch=None):
    logits = self.router(x).astype(mx.float32)
    inds = _topk(mx, logits, _top_k(self))
    w = mx.softmax(mx.take_along_axis(logits, inds, axis=-1), axis=-1)
    return inds, w, (switch or self.experts)(x, inds), 0


def _top_k(m):
    k = getattr(m, "top_k", None) or getattr(m, "num_experts_per_tok", None)
    return int(k) if k else None


def _has(m, *names):
    return all(getattr(m, n, None) is not None for n in names)


def classify(module):
    """The MoE block kind of ``module`` from its own structure (see the module docstring), or None."""
    names = {c.__name__ for c in type(module).__mro__}
    shared = _has(module, "shared_expert") or _has(module, "shared_experts")
    if "Qwen3NextSparseMoeBlock" in names:
        return "qwen_next"
    if _has(module, "gate", "switch_mlp") and hasattr(module, "moe_latent_size") \
            and callable(getattr(module, "_routing_weights", None)):
        return "nemotron"         # grouped router, optional latent projections, folded expert scales
    if (_has(module, "gate", "switch_mlp", "shared_expert")
            and not hasattr(module, "shared_expert_gate") and isinstance(getattr(module, "scale", None), (int, float))):
        return "laguna"           # routed * scale + an ungated shared expert (Laguna XS/S)
    if "Experts" in names and hasattr(module, "switch_glu"):
        return "gemma4_experts"
    if _has(module, "gate", "switch_mlp") and hasattr(module, "shared_experts"):
        return "ling"             # shared_experts may be None (num_shared_experts == 0); _ling_rows handles it
    if _has(module, "switch_mlp") and callable(getattr(module, "routing", None)) and not shared:
        return "routing"          # the block's own routing(x) -> (ids, weights)
    if _has(module, "gate", "switch_mlp", "top_k", "norm_topk_prob") and not shared:
        return "softmax_topk"
    if (_has(module, "gate", "switch_mlp", "top_k", "shared_expert") and not hasattr(module, "shared_expert_gate")
            and not hasattr(module, "scale") and not hasattr(module, "norm_topk_prob")):
        return "sigmoid_shared"
    if (_has(module, "router", "experts") and not hasattr(module, "gate") and _top_k(module)
            and _has(module.experts, "down_proj") and not shared):
        return "router_experts"
    return None


SITE_ATTRS = ("experts", "mlp", "feed_forward", "block_sparse_moe", "mixer")
TOKEN_MIXER_SLOTS = ("mixer",)      # hybrid blocks (Nemotron-H) put attention, SSM or FFN here


def _token_mixer(m):
    return not hasattr(m, "down_proj") and (hasattr(m, "o_proj") or hasattr(m, "out_proj"))


def _dims(p):
    """(input, output) features of a dense projection, or of one expert of a switch projection, in
    any storage format: plain, MLX quantized (affine / mxfp4 / mxfp8 / nvfp4, from ``bits``), owned
    modules exposing ``input_dims``/``output_dims`` (EXL3, NVFP4 switch), and owned microscaling
    modules with neither (ScaledNVFP4Linear, ScaledFP8Linear, GlobalScaledNVFP4Linear): their uint8
    group scales fix the group (packed bits x group = 64 for nvfp4 g16, 128 mxfp4 g32, 256 mxfp8 g32)."""
    import mlx.core as mx
    i, o = getattr(p, "input_dims", None), getattr(p, "output_dims", None)
    if isinstance(i, int) and isinstance(o, int):
        return i, o
    w = getattr(p, "weight", None)
    if w is None:
        raise SystemExit(f"units: cannot size projection {type(p).__name__} (no weight, no input_dims)")
    out = int(w.shape[-2])
    sc = p["scales"] if "scales" in p else None
    if w.dtype != mx.uint32 or sc is None:
        return int(w.shape[-1]), out
    bits = getattr(p, "bits", None)
    if bits:
        return int(w.shape[-1]) * 32 // int(bits), out
    gs = getattr(p, "group_size", None)
    if not gs and sc.dtype == mx.uint8:
        gs = {64: 16, 128: 32, 256: 32}.get(int(w.shape[-1]) * 32 // int(sc.shape[-1]))
    if not gs:
        raise SystemExit(f"units: cannot size packed projection {type(p).__name__} (no bits, no group size)")
    return int(sc.shape[-1]) * int(gs), out


PROJ = ("gate_proj", "up_proj", "down_proj", "gate_up_proj", "fc1", "fc2")


def _params(m):
    return sum(int(np.prod(_dims(getattr(m, n)))) for n in PROJ if getattr(m, n, None) is not None)


def _switch(m, kind):
    return m.switch_glu if kind == "gemma4_experts" else m.experts if kind == "router_experts" else m.switch_mlp


def _n_experts(m, sw):
    for v in (getattr(m, "num_experts", None), getattr(getattr(m, "gate", None), "n_routed_experts", None),
              getattr(m, "num_local_experts", None)):
        if isinstance(v, int) and v > 0:
            return v
    for n in ("down_proj", "fc2"):
        q = getattr(sw, n, None)
        if q is not None:
            v = getattr(q, "num_experts", None)
            return int(v) if v is not None else int(q.weight.shape[0])
    raise SystemExit(f"units: cannot count experts of {type(m).__name__}")


def _router_width(m, kind):
    """Expert count the block's router scores, when the router is a plain projection."""
    r = getattr(m, "router" if kind == "router_experts" else "gate", None)
    if kind in ("softmax_topk", "sigmoid_shared", "router_experts", "qwen_next") and r is not None:
        return _dims(r)[1]
    return None


MOE_UNITS = ("expert", "neuron")

NEURON_PROVENANCE = {
    "source": "scripts/probe_amoeba_unit_box.py@16a1bbe88 (mlx2, private; with scripts/amoeba_cwe_donor.py@16a1bbe88)",
    "functions": ["_neuron_switch_types", "_switch_layout", "_switch_masked", "_moe_neuron_call",
                  "_qwen_next_route", "_combine", "install_units(moe_units=)"],
    "modified": [
        "a mask is a per-layer [E*F] carve core (differentiable, straight-through) instead of a 0/1 allow table",
        "mask rows gathered per routed slot ([..., k, F]) in slot order, then permuted with the sort, so "
        "gated boxes and the position-0 rule apply without a tokens x E x F tensor",
        "sort decision per switch type (SwitchGLU sort policy + invariant lane; fused switches _sort_min + invariant)",
        "qwen_next routing: block helpers in eval mode only; in training/reference mode the stock composed "
        "routing with stop-gradiented ids (MLX refuses a VJP through gather indices)",
        "ling / laguna / gemma4_experts keep the carve's expert-mode routing and combine",
        "not ported: contribution-mass recording (_neuron_record, _down_colnorm) and the box-finding arms",
    ],
}


def expert_to_neuron(mask, F):
    """An expert mask [..., E] as the neuron mask [..., E*F] that keeps every neuron of each kept expert."""
    a = np.asarray(mask, dtype=np.float32)
    return np.repeat(a, F, axis=-1)


def _neuron_switch_types():
    from mlx2.runtime.models.qwen3_next import FusedDownSwitchGLU, FusedGateUpSwitchGLU
    from mlx2.runtime.models.switch_layers import SwitchGLU
    return (SwitchGLU, FusedDownSwitchGLU, FusedGateUpSwitchGLU)


def _proj_params(p):
    return int(np.prod(_dims(p)))          # one expert's (input x output), any storage format


def _switch_layout(sw, where):
    """(F, params per expert) of a switch GLU, or raise: neuron mode never falls back to expert units."""
    ok = _neuron_switch_types()
    if type(sw) not in ok:
        raise SystemExit(f"units: {where}: --moe-units neuron supports only {', '.join(t.__name__ for t in ok)} switch "
                         f"modules (their gate/up -> activation -> down body is the one re-run in pieces); found "
                         f"{type(sw).__module__}.{type(sw).__name__}")
    if "bias" in sw.down_proj:
        raise SystemExit(f"units: {where}: --moe-units neuron refuses a biased down_proj (a fully masked expert "
                         f"would still write its bias, so it would not equal the expert drop rule)")
    projs = [p for p in ("gate_proj", "up_proj", "gate_up_proj", "down_proj") if hasattr(sw, p)]
    F = int(_dims(sw.down_proj)[0])
    total = sum(_proj_params(getattr(sw, p)) for p in projs)
    if total % F:
        raise SystemExit(f"units: {where}: switch params per expert {total} not divisible by F={F}")
    return F, total


def install(layers, ctl_factory=Control, moe_units="expert"):
    """Patch one unit site per layer. Returns (sites, ctl). Idempotent per module object.

    moe_units="neuron": MoE sites carve (expert, intermediate neuron) units (module docstring)."""
    import mlx.core as mx

    if moe_units not in MOE_UNITS:
        raise SystemExit(f"units: moe_units must be one of {MOE_UNITS}, not {moe_units!r}")
    sites = []
    for li, layer in enumerate(layers):
        cands = [m for n in SITE_ATTRS if (m := getattr(layer, n, None)) is not None
                 and not (n in TOKEN_MIXER_SLOTS and _token_mixer(m))]
        moe = next(((classify(m), m) for m in cands if classify(m)), None)
        if moe is None:
            mlp = next((m for m in cands if hasattr(m, "down_proj")), None)
            if mlp is None:
                odd = [type(m).__name__ for m in cands]
                if odd:
                    raise SystemExit(f"units: layer {li} has an MLP/MoE module of unknown shape {odd}; refusing "
                                     f"rather than carving it as zero units")
                sites.append(Site(li, None, 0, 0))
                continue
            dp = mlp.down_proj
            n, d = _dims(dp)
            for nm in ("up_proj", "gate_proj", "fc1"):
                q = getattr(mlp, nm, None)
                if q is not None and _dims(q)[1] != n:
                    raise SystemExit(f"units: layer {li} down_proj takes {n} inputs but {nm} gives {_dims(q)[1]}")
            sites.append(Site(li, "dense", n, _params(mlp) // n, dp, mlp))
        else:
            kind, m = moe
            sw = _switch(m, kind)
            n = _n_experts(m, sw)
            rw = _router_width(m, kind)
            if rw is not None and rw != n:
                raise SystemExit(f"units: layer {li} {kind} router scores {rw} experts, the block has {n}")
            if moe_units == "neuron":
                where = f"layer {li} ({kind}, {type(m).__name__})"
                F, per_expert = _switch_layout(sw, where)
                swE = int(sw.down_proj.weight.shape[0])
                if swE < int(n):
                    raise SystemExit(f"units: {where}: switch holds {swE} experts < {n} routed")
                if getattr(m, "sharding_group", None) is not None:
                    raise SystemExit(f"units: {where}: a sharded MoE block is not carved in neuron units")
                sites.append(Site(li, kind, int(n) * F, per_expert // F, m, None, "neuron", int(n), F))
                continue
            sites.append(Site(li, kind, int(n), _params(sw), m, None))
    ctl = ctl_factory(sites)
    for li, layer in enumerate(layers):
        if sites[li].kind == "gemma4_experts":
            r = getattr(layer, "router", None)
            if r is None:
                raise SystemExit(f"units: layer {li} Gemma4 experts without a router module")
            _patch_router(mx, r, ctl)
    for s in sites:
        if s.kind == "dense":
            _patch_dense(mx, s, ctl)
        elif s.kind is not None and s.units == "neuron":
            _patch_moe_neuron(mx, s, ctl)
        elif s.kind is not None:
            _patch_moe(mx, s, ctl)
    return sites, ctl


def _swap(module, name, call):
    base = type(module)
    if getattr(base, "_amoeba_carve_patched", False):
        base = base.__mro__[1]
    module.__class__ = type(f"Carve{name}{base.__name__}", (base,), {"__call__": call, "_amoeba_carve_patched": True})
    return base


def _patch_dense(mx, site, ctl):
    l = site.layer
    mlp_base = type(site.parent)
    if getattr(mlp_base, "_amoeba_carve_patched", False):
        mlp_base = mlp_base.__mro__[1]

    def mlp_call(self, x, *a, _base=mlp_base, **k):
        ctl._x = x                     # the MLP input, for gates and stats
        return _base.__call__(self, x, *a, **k)

    def down_call(self, act, _l=l):
        c = ctl
        if c.stats is not None:
            c.stats(_l, "dense", {"act": act, "x": getattr(c, "_x", None)})
        if c.verify is not None:
            c.verify[_l] = 0.0            # engaged; masking acts on the exact down_proj input
        if not c.active(_l):
            return _base_down.__call__(self, act)
        x = getattr(c, "_x", None)
        m = _box_mask(mx, c, _l, x).astype(act.dtype)
        mu = _row(c.mean, _l)
        if mu is not None:
            mu = mu.astype(act.dtype)
            used = mu + m * (act - mu)
        else:
            used = m * act
        used = _first_position(mx, c, _l, act, used)
        if c.box_out is not None:
            for box in c.boxes:
                bm = box["mask"][_l].astype(act.dtype)
                c.box_out[(box.get("name"), _l)] = _base_down.__call__(self, bm * (act if mu is None else act - mu))
        return _base_down.__call__(self, used)

    _base_down = _swap(site.module, "Down", down_call)
    site.parent.__class__ = type(f"CarveMLP{mlp_base.__name__}", (mlp_base,),
                                 {"__call__": mlp_call, "_amoeba_carve_patched": True})


def _patch_router(mx, router, ctl):
    def call(self, x, *a, **k):
        return _base.__call__(self, mx.stop_gradient(x) if ctl.router_sg else x, *a, **k)
    _base = _swap(router, "Router", call)


_ROWS = {"laguna": _laguna_rows, "softmax_topk": _softmax_topk_rows,
         "sigmoid_shared": _sigmoid_shared_rows, "router_experts": _router_experts_rows}
_ROWS_SG = {"ling": _ling_rows, "nemotron": _nemotron_rows, "routing": _routing_rows}


def _route(mx, self, kind, a, router_sg, switch=None, force=None):
    """(x, inds, w, rows, shared, post): the block's routing, re-implemented; post maps the routed
    sum back (latent blocks) or is None. ``switch`` replaces the block's switch call (neuron units run
    it in pieces); ``force`` is the replayed routing from ``_replayed`` (qwen_next: ids; gemma4_experts:
    (ids, w)). qwen_next routes through the stock block's own helpers (``_qwen_next_neuron_rows``):
    its ``shared`` is the pair (shared gate, shared rows) and ``_combine`` takes the stock tail."""
    if kind == "gemma4_experts":
        x, inds, w = a
        if force is not None:
            inds, w = force
        inds = mx.stop_gradient(inds)
        return x, inds, w.astype(mx.float32), (switch or self.switch_glu)(x, inds), 0, None
    (x,) = a
    if kind == "qwen_next":
        # the stock block's own routing helpers (fused routers in eval, the composed routing in
        # training / reference mode) and its own combine tail, in expert and neuron mode alike, so
        # core = ones tracks the stock forward (fnsmoke3c: the fp32-softmax re-implementation agreed
        # on top-1 at only 0.953 on FN)
        sw = switch or (lambda xx, ii: self.switch_mlp(xx, ii))
        return (x, *_qwen_next_neuron_rows(mx, self, x, sw, force), None)
    kw = {} if switch is None else {"switch": switch}
    out = _ROWS[kind](mx, self, x, **kw) if kind in _ROWS else _ROWS_SG[kind](mx, self, x, router_sg, **kw)
    return (x, *out) if len(out) == 5 else (x, *out, None)


def _combine(mx, self, kind, rows, w, shared, post, dtype):
    """The routed sum (+ post) + shared. qwen_next: the stock tail in the score dtype, then cast to the
    residual dtype (bf16 on FN; CPU GatherMM is fp32-only, so the toys cannot show it); dtype=None keeps
    the stock tail's own dtype (neuron mode, which equals the stock block bit for bit)."""
    if kind == "qwen_next":
        y = _qwen_next_combine(mx, self, rows, w, shared)
        return y if dtype is None else y.astype(dtype)
    y = (rows.astype(mx.float32) * w[..., None]).sum(axis=-2).astype(dtype)
    if post is not None:
        y = post(y)
    return y + shared


def _rel(mx, a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return mx.sqrt(mx.sum(mx.square(a - b)) / mx.maximum(mx.sum(mx.square(b)), 1e-30))


def _record_verify(mx, c, l, y, base_out):
    r = _rel(mx, y, base_out)
    prev = c.verify.get(l)            # a block may run more than once per forward: keep the worst
    c.verify[l] = r if prev is None else mx.maximum(prev, r)


def _patch_moe(mx, site, ctl):
    l, kind = site.layer, site.kind

    def call(self, *a, _l=l, _kind=kind, _n=site.n):
        c = ctl
        if not c.active(_l) and c.stats is None and c.verify is None:
            return _base.__call__(self, *a)
        force = _replayed(c, _l, _kind)
        x, inds, w, rows, shared, post = _route(mx, self, _kind, a, c.router_sg, force=force)
        if c.stats is not None:
            c.stats(_l, _kind, {"inds": inds, "w": w, "rows": rows, "x": x})
        base_out = None
        if c.verify is not None and force is None:      # parity of the free route only (module docstring)
            base_out = _base.__call__(self, *a)
            _record_verify(mx, c, _l, _combine(mx, self, _kind, rows, w, shared, post, x.dtype), base_out)
        if not c.active(_l):
            return base_out if base_out is not None else _base.__call__(self, *a)
        _check_width(c, _l, _n, "expert")
        m = _box_mask(mx, c, _l, x)                   # [n] or [B,T,n] when gated
        keep = mx.take_along_axis(m, inds, axis=-1) if m.ndim > 1 else m[inds]
        if _kind == "qwen_next":           # keep the stock score dtype (a float32 mask would promote the
            keep = keep.astype(w.dtype)     # residual stream to float32 downstream; review 2026-11-15)
        wk = _first_position(mx, c, _l, w, w * keep)
        if c.box_out is not None:
            for box in c.boxes:
                bk = box["mask"][_l][inds]
                bw = (rows.astype(mx.float32) * (w * bk)[..., None]).sum(axis=-2)
                c.box_out[(box.get("name"), _l)] = bw if post is None else post(bw.astype(x.dtype))
        return _combine(mx, self, _kind, rows, wk, shared, post, x.dtype)

    _base = _swap(site.module, "MoE", call)

# ------------------------------------------------------------------ neuron units (MoE)

def _qn(block):
    import sys
    cls = next(c for c in type(block).__mro__ if c.__name__ == "Qwen3NextSparseMoeBlock")
    return sys.modules[cls.__module__]


def _qwen_next_route(mx, block, x):
    """(inds, scores) with the stock block's values. Eval mode: the block's own routing helpers, as
    Qwen3NextSparseMoeBlock.__call__ picks them. Training / reference mode: the stock composed routing
    (``_stock_routing``) with the ids stop-gradiented before the score gather (a VJP through gather
    indices raises), the same values; ``check_layer_reference`` measures any eval/reference gap."""
    qn = _qn(block)
    gates = block.gate(x)
    if not block.training:
        launched = qn._try_topk_launch(block, x, gates)
        if launched is not None:
            return mx.stop_gradient(launched[0]), launched[1]
        if block.moe_router_mode == "fused":
            adm = qn.admit_qwen4_moe_router(gates, top_k=block.top_k, norm_topk_prob=bool(block.norm_topk_prob))
            if adm.accepted and qn.probe_qwen4_moe_router(gates.dtype):
                inds, sc = qn.qwen4_moe_router(gates)
                return mx.stop_gradient(inds), sc
        if qn._MOE_GATE_COMPILE and gates.size // gates.shape[-1] <= qn._MOE_GATE_COMPILE_MAX_TOKENS:
            inds, sc = qn._select_experts(gates, block.top_k, bool(block.norm_topk_prob))
            return mx.stop_gradient(inds), sc
    probs = mx.softmax(gates, axis=-1, precise=True)
    k = block.top_k
    inds = mx.stop_gradient(mx.argpartition(probs, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(probs, inds, axis=-1)
    if block.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


def _qwen_next_neuron_rows(mx, block, x, switch, force=None):
    """inds, scores, routed rows [..., k, D], (shared gate, shared rows): the stock block's plain tail."""
    from mlx2.runtime.models.precise_ops import gate_sigmoid
    if force is not None:        # replayed ids; the weights at them, as the reference routing gathers them
        inds = mx.stop_gradient(force)
        w = _gather_scores(mx, mx.softmax(block.gate(x), axis=-1, precise=True), inds, block.norm_topk_prob)
    else:
        inds, w = _qwen_next_route(mx, block, x)
    k = block.top_k
    if getattr(block, "shared_folded", False):
        col = mx.full(inds.shape[:-1] + (1,), block.num_experts, dtype=inds.dtype)
        rows = switch(x, mx.concatenate([inds, col], axis=-1))
        rows, shared = rows[..., :k, :], rows[..., k, :]
    else:
        rows, shared = switch(x, inds), block.shared_expert(x)
    return inds, w, rows, (gate_sigmoid(block.shared_expert_gate(x)), shared)


def _qwen_next_combine(mx, block, rows, w, shared):
    """Stock tail: weighted routed sum in the score dtype, then the gated shared term."""
    y = (rows * w[..., None]).sum(axis=-2)
    gate, shared_y = shared
    qn = _qn(block)
    if qn._COMPILE_GLUE:
        c = qn._run_glue(("moe_combine",), qn._build_moe_combine, y, gate, shared_y)
        if c is not None:
            return c
    return y + gate * shared_y


def _sort_decision(sw, inds):
    """The stock sorted-gather decision of this switch type (same gather order as its __call__)."""
    from mlx2.runtime.models import invariant_prefill as inv
    from mlx2.runtime.models import switch_layers as sl
    if type(sw) is sl.SwitchGLU:
        tl = int(inds.shape[-2]) if inds.ndim >= 2 else 1
        return sl._switch_sort_decision(getattr(sw, "_sort_policy", "auto"), assignments=int(inds.size),
                                        token_length=tl, invariant=inv.active())[0]
    from mlx2.runtime.models import qwen3_next as q3
    return inds.size >= q3._sort_min(sw) or inv.active()


def _slot_order(mx, v, inds, h, inv, do_sort, pad_value):
    """A routed-slot-order row array [*inds.shape, F'] in the layout of h (the sort's order, tail pad rows)."""
    F = v.shape[-1]
    vv = v.reshape(-1, F)
    if do_sort:
        order = mx.argsort(inv)                       # inv = argsort(order), a permutation
        vv = vv[order]
        if h.shape[0] > vv.shape[0]:                  # _gather_sort's tail pad rows (sliced off later)
            vv = mx.concatenate([vv, mx.full((h.shape[0] - vv.shape[0], F), pad_value, vv.dtype)], axis=0)
        return vv[:, None, :]
    return vv.reshape(*inds.shape, 1, F)


def _switch_masked(mx, sw, x, inds, keep=None, want_hidden=False, fill=None, thresh=None, exact=None, open_acc=None):
    """The switch GLU in pieces: gate/up -> activation -> (* keep) -> down, through the module's own
    (quantized) switch projections. ``keep`` is [*inds.shape, F] in routed-slot order (or None).
    Opt-in (None = unchanged): ``fill`` [*inds.shape, F] is the value a closed neuron takes instead of 0
    (h -> keep * h + (1 - keep) * fill, the dense mean-ablation rule); ``thresh`` (float) also closes, per token,
    every neuron with |h| < thresh, except at slots where ``exact`` [*inds.shape, 1] is 1; ``open_acc``
    (list [open, total]) accumulates the open count of those thresholded slots.
    Returns (rows [..., k, D], activated hidden [..., k, F] before masking, or None)."""
    from mlx2.runtime.models import switch_layers as sl
    xe = mx.expand_dims(x, (-2, -3))
    do_sort = _sort_decision(sw, inds)
    idx, inv = inds, None
    if do_sort:
        xe, idx, inv = sl._gather_sort(xe, inds)
    idx = mx.stop_gradient(idx)
    if hasattr(sw, "gate_up_proj"):
        gu = sw.gate_up_proj(xe, idx, sorted_indices=do_sort)
        half = sw.hidden_dims
        h = sw.activation(gu[..., half:], gu[..., :half])
    else:
        h = sw.activation(sw.up_proj(xe, idx, sorted_indices=do_sort), sw.gate_proj(xe, idx, sorted_indices=do_sort))
    h0 = h
    kk = None
    if keep is not None:
        F = keep.shape[-1]
        kk = keep.reshape(-1, F)
        if do_sort:
            order = mx.argsort(inv)                       # inv = argsort(order), a permutation
            kk = kk[order]
            if h.shape[0] > kk.shape[0]:                  # _gather_sort's tail pad rows (sliced off later)
                kk = mx.concatenate([kk, mx.ones((h.shape[0] - kk.shape[0], F), kk.dtype)], axis=0)
            kk = kk[:, None, :]
        else:
            kk = kk.reshape(*inds.shape, 1, F)
    if thresh is not None:                                # per-token |h| threshold (TEAL-style reference)
        dyn = (mx.abs(h) >= thresh).astype(h.dtype)
        if exact is not None:
            ex = _slot_order(mx, exact.astype(h.dtype), inds, h, inv, do_sort, 1.0)
            dyn = mx.maximum(dyn, ex)
        if open_acc is not None:
            ex_ = 0.0 if exact is None else ex
            free = (1.0 - ex_) * mx.ones_like(dyn)
            open_acc[0] = open_acc[0] + (dyn * free).astype(mx.float32).sum()
            open_acc[1] = open_acc[1] + free.astype(mx.float32).sum()
            mx.eval(open_acc)                             # a measurement path (no gradient): never hold h alive
        kk = dyn if kk is None else kk.astype(h.dtype) * dyn
    if kk is not None:
        if fill is not None:
            ff = _slot_order(mx, fill, inds, h, inv, do_sort, 0.0).astype(h.dtype)
            kd = kk.astype(h.dtype)
            h = h * kd + ff * (1 - kd)                    # an open neuron (kd = 1) keeps h exactly
        else:
            h = h * kk.astype(h.dtype)
    y = sw.down_proj(h, idx, sorted_indices=do_sort)
    hid = None
    if do_sort:
        y = sl._scatter_unsort(y, inv, inds.shape)
        if want_hidden:
            hid = sl._scatter_unsort(h0, inv, inds.shape).squeeze(-2)
    elif want_hidden:
        hid = h0.squeeze(-2)
    return y.squeeze(-2), hid


def _slot_rows(v, inds, E, F):
    """A per-layer [E*F] vector as the routed slots' rows [*inds.shape, F] (gather, never one-hot)."""
    return v.reshape(E, F)[inds]


def _neuron_keep(mx, c, l, x, inds, E, F):
    """core + sum of (gated) boxes, per routed slot: [*inds.shape, F] (tokens x k x F at most)."""
    K = _slot_rows(c.core[l], inds, E, F)
    for box in c.boxes:
        bk = _slot_rows(box["mask"][l], inds, E, F)
        if box.get("token_gate") is not None:      # precomputed per-token open bits [B, T, 1] (next-token gating)
            K = K + bk * box["token_gate"][..., None]
        elif box.get("gate") is not None:
            W, b = box["gate"]
            g = mx.sigmoid((x.astype(mx.float32) @ W[l][:, None]) + b[l])      # [..., 1]
            if box.get("hard"):
                g = (g > 0.5).astype(mx.float32)
            if c.gate_log is not None:
                c.gate_log.append((box.get("name"), l, g))
            K = K + bk * g[..., None]
        else:
            K = K + bk
    return _first_position(mx, c, l, mx.ones_like(K), K)


def _patch_moe_neuron(mx, site, ctl):
    l, kind, E, F = site.layer, site.kind, site.E, site.F

    def call(self, *a, _l=l, _kind=kind):
        c = ctl
        if not c.active(_l) and c.stats is None and c.verify is None:
            return _base.__call__(self, *a)
        active = c.active(_l)
        if active:
            _check_width(c, _l, E * F, "neuron")
        want = c.stats is not None
        force = _replayed(c, _l, _kind)

        def route(masked):
            """One pass through the block's routing with the switch run in pieces; masked=False is the
            all-open pass (keep None). Returns (x, inds, w, rows, shared, post, h)."""
            cap = {"k": self.top_k} if _kind == "qwen_next" else {}

            def padded(v, i_, value):
                if v is not None and i_.shape[-1] > v.shape[-2]:           # folded shared row(s): always on
                    pad = mx.full((*v.shape[:-2], i_.shape[-1] - v.shape[-2], v.shape[-1]), value, v.dtype)
                    v = mx.concatenate([v, pad], axis=-2)
                return v

            def run_switch(sw, x_, i_, keep, fill=None, thresh=None, exact=None, acc=None):
                return _switch_masked(mx, sw, x_, i_, padded(keep, i_, 1.0), want, fill=padded(fill, i_, 0.0),
                                      thresh=thresh, exact=padded(exact, i_, 1.0), open_acc=acc)

            def switch_fn(sw, x_):
                def f(xx, ii):
                    routed = ii[..., : cap["k"]] if "k" in cap else ii        # the folded shared column is not a unit
                    rr = mx.stop_gradient(routed)
                    keep = _neuron_keep(mx, c, _l, x_, rr, E, F) if masked else None
                    fill = thr = exact = acc = None
                    if masked and c.moe_mean is not None and c.moe_mean[_l] is not None:
                        fill = _slot_rows(c.moe_mean[_l], rr, E, F)
                    if masked and c.moe_thresh is not None and _l in c.moe_thresh:
                        thr = float(c.moe_thresh[_l])
                        exact = mx.zeros((*rr.shape, 1), mx.float32)
                        if _l < c.sink_layers and rr.ndim >= 3 and rr.shape[1] > 1:
                            exact = mx.concatenate([mx.ones_like(exact[:, :1]), exact[:, 1:]], axis=1)
                        if c.moe_open is not None:
                            acc = c.moe_open.setdefault(_l, [mx.array(0.0), mx.array(0.0)])
                    rows_, cap["h"] = run_switch(sw, xx, ii, keep, fill, thr, exact, acc)
                    return rows_
                return f

            x0 = a[0]
            out = _route(mx, self, _kind, a, c.router_sg, switch=switch_fn(_switch(self, _kind), x0), force=force)
            return (*out, cap.get("h"))

        res = base_out = None
        if c.verify is not None and force is None:      # parity of the free route, all-open (module docstring)
            res = route(False)
            x, inds, w, rows, shared, post, _h = res
            base_out = _base.__call__(self, *a)
            _record_verify(mx, c, _l, _combine(mx, self, _kind, rows, w, shared, post,
                                                None if _kind == "qwen_next" else x.dtype), base_out)
            if not active and not want:
                return base_out
        if active or res is None:
            res = route(active)
        x, inds, w, rows, shared, post, h = res
        if want:
            c.stats(_l, _kind, {"inds": inds, "w": w, "rows": rows, "x": x, "F": F,
                                "h": None if h is None else h[..., : inds.shape[-1], :]})
        if active and c.box_out is not None:
            if post is not None:
                raise NotImplementedError(f"units: layer {_l} ({_kind}): neuron box writes of a latent block")
            sw_mod = _switch(self, _kind)
            for box in c.boxes:
                bk = _slot_rows(box["mask"][_l], inds, E, F)
                rb, _ = _switch_masked(mx, sw_mod, x, inds, bk, False)
                c.box_out[(box.get("name"), _l)] = (rb.astype(mx.float32) * w.astype(mx.float32)[..., None]).sum(axis=-2)
        if not active and base_out is not None:
            return base_out                 # verify already ran the stock block: its output, no extra call
        if not active and not getattr(c, "stats_fast", False):
            return _base.__call__(self, *a)
        # stats_fast (firing maps, forward only): the output is the routed rows just computed for the capture, through
        # the stock tail, instead of a second full MoE call (equal to the fused path to bf16 rounding: layer_ref gate).
        # qwen_next: the stock tail in its own dtype (dtype None), equal to the stock block bit for bit
        return _combine(mx, self, _kind, rows, w, shared, post, None if _kind == "qwen_next" else x.dtype)

    _base = _swap(site.module, "MoENeuron", call)


def site_summary(sites) -> list[dict]:
    out = []
    for s in sites:
        r = {"layer": s.layer, "kind": s.kind, "units": s.n, "params_per_unit": s.params_per_unit}
        if s.units == "neuron" and s.is_moe:
            r.update(moe_units="neuron", E=s.E, F=s.F, unit_id="expert * F + neuron")
        out.append(r)
    return out


def neuron_sites(sites):
    return [s for s in sites if s.is_moe and s.units == "neuron"]


def uniform_width(sites) -> int | None:
    ns = {s.n for s in sites if s.kind is not None}
    return ns.pop() if len(ns) == 1 else None


def pad_stack(rows, width, fill=0.0, dtype=np.float32):
    """Per-layer vectors (ragged) -> [L, width] array plus per-layer widths (npz layout)."""
    out = np.full((len(rows), width), fill, dtype=dtype)
    for i, r in enumerate(rows):
        if r is not None:
            r = np.asarray(r)
            out[i, :r.shape[0]] = r
    return out
