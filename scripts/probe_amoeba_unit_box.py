#!/usr/bin/env python3
"""Per-donor box for one prompt, comparable across architectures (amoeba transplant, step 1).

Run inside an owned GPU window, in-process, one donor per process. For one candidate prompt
from amoeba_transplant_screen.py:

1. Traces. Chat thinking mode rendered exactly as the server renders it
   (serving.render_prompt_tokens; Flash Next gets its default medium effort). One greedy run
   plus --samples runs at --temperature (seeded), each scored by the oracle. The box is
   built from the CORRECT traces only (the solution space); wrong traces are kept and
   reported.
2. Units, by architecture (detected per layer, never by model name):
     routed experts  qwen3-next-style blocks (gate -> softmax top-k -> switch_mlp, shared
                     expert with sigmoid gate: Flash Next, Qwen3.5/3.6 MoE), ling/Kimi
                     blocks (gate -> ids, weights -> switch_mlp, shared experts), Gemma4
                     Experts(x, indices, weights). Shared experts and Gemma4's parallel
                     dense MLP always run (they are the host-side "core" here).
     dense neurons   every other layer's MLP down_proj input (intermediate neurons).
   A unit's contribution at a token is the norm of what it writes: ||w_e * y_e|| for an
   expert, |a_i| * ||W_down[:, i]|| for a neuron.
3. Capture. Teacher-forced forward per trace: per token and layer the routed expert ids,
   weights, contribution norms (MoE) or the top-32 neuron ids and contributions (dense);
   per-layer contribution sums per unit; layer outputs at --capture-layers (fp16; hyper-
   connection streams averaged) for the later entry map.
4. Box arms. Per layer, units ranked by contribution mass pooled over correct traces:
   union (every unit that contributed), and mass coverage at each --coverage level; for each,
   a random twin with the same per-layer counts. Drop rule (non-box units zeroed, no
   renormalisation). Teacher-forced KL(full || box) and top-1 agreement on every trace's
   generated tokens. Greedy regeneration + oracle for the real arms from the largest down,
   stopping after two consecutive failures, and for the random twins of the union and of
   the smallest passing arm. "Units needed" = the smallest passing arm whose random twin
   fails.

Before any arm, the patched reference path is checked against the unpatched forward on
the greedy trace (top-1 agreement >= 0.98); a parity failure stops the run.
Writes traces.jsonl, units/*.npz, hidden/*.npz, box.json, arms.jsonl, run.json.
--moe-units neuron carves MoE layers into (expert, intermediate neuron) units (see amoeba_cwe_donor.py);
box.json then records moe_units="neuron" and keeps the [L, E*F] per-layer mass and the box masks in
per_layer_mass.npy and boxes.npz instead of JSON lists. Box consumers (amoeba_hub_transfer.py) refuse a
box whose moe_units differs from the units they install.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import amoeba_transplant_screen as screen  # noqa: E402

ST = {"active": False, "record": None, "allow": None, "moe_units_override": None}
MOE_UNITS = ("expert", "neuron")


# ------------------------------------------------------------------ unit adapters
# The patched MoE paths reproduce the stock block's routing (same router dtype, precise softmax, top-k and
# norm_topk; qwen_next through the block module's own routing helpers) and its plain combine arithmetic
# ((rows * scores).sum in the router-score dtype, then the shared term), so an all-allowed patch equals the
# stock module bit for bit wherever the stock block runs its plain switch tail (CPU; Metal without the
# fused-down / sorted weighted-sum kernels). Expert and neuron modes share that arithmetic exactly.

def _qn(self):
    cls = next(c for c in type(self).__mro__ if c.__name__ == "Qwen3NextSparseMoeBlock")
    return sys.modules[cls.__module__]


def _qwen_next_route(mx, self, x):
    """(inds, scores) exactly as Qwen3NextSparseMoeBlock.__call__ picks them."""
    qn = _qn(self)
    gates = self.gate(x)
    launched = qn._try_topk_launch(self, x, gates)
    if launched is not None:
        return launched
    if self.moe_router_mode == "fused":
        adm = qn.admit_qwen4_moe_router(gates, top_k=self.top_k, norm_topk_prob=bool(self.norm_topk_prob))
        if adm.accepted and qn.probe_qwen4_moe_router(gates.dtype):
            return qn.qwen4_moe_router(gates)
    if qn._MOE_GATE_COMPILE and gates.size // gates.shape[-1] <= qn._MOE_GATE_COMPILE_MAX_TOKENS:
        return qn._select_experts(gates, self.top_k, bool(self.norm_topk_prob))
    return qn._stock_routing(self, gates)


def _qwen_next_rows(mx, self, x, switch=None):
    """inds, scores, routed rows [..., k, D], (shared gate, shared rows)."""
    from mlx2.runtime.models.precise_ops import gate_sigmoid
    switch = switch or self.switch_mlp
    inds, w = _qwen_next_route(mx, self, x)
    k = self.top_k
    if getattr(self, "shared_folded", False):
        col = mx.full(inds.shape[:-1] + (1,), self.num_experts, dtype=inds.dtype)
        rows = switch(x, mx.concatenate([inds, col], axis=-1))
        rows, shared = rows[..., :k, :], rows[..., k, :]
    else:
        rows, shared = switch(x, inds), self.shared_expert(x)
    return inds, w, rows, (gate_sigmoid(self.shared_expert_gate(x)), shared)


def _ling_rows(mx, self, x, switch=None):
    ids, w = self.gate(x)
    rows = (switch or self.switch_mlp)(x, ids)
    shared = self.shared_experts(x) if getattr(self, "shared_experts", None) is not None else None
    return ids, w, rows, shared


def _combine(mx, self, kind, rows, w, shared):
    """The stock block's tail: weighted routed sum in the score dtype, then the shared term."""
    y = (rows * w[..., None]).sum(axis=-2)
    if kind == "qwen_next":
        gate, shared_y = shared
        qn = _qn(self)
        if qn._COMPILE_GLUE:
            c = qn._run_glue(("moe_combine",), qn._build_moe_combine, y, gate, shared_y)
            if c is not None:
                return c
        return y + gate * shared_y
    return y if shared is None else y + shared


def classify(module):
    """'qwen_next' | 'ling' | 'gemma4_experts' | None, from the block's own structure."""
    names = {c.__name__ for c in type(module).__mro__}
    if "Qwen3NextSparseMoeBlock" in names:
        return "qwen_next"
    if "Experts" in names and hasattr(module, "switch_glu"):
        return "gemma4_experts"
    if hasattr(module, "gate") and hasattr(module, "switch_mlp") and hasattr(module, "shared_experts"):
        return "ling"
    return None


def _proj_params(p):
    """Logical parameters of one switch projection, per expert (input_dims * output_dims; any bit width)."""
    return int(p.input_dims) * int(p.output_dims)


def _neuron_switch_types():
    from mlx2.runtime.models.qwen3_next import FusedDownSwitchGLU, FusedGateUpSwitchGLU
    from mlx2.runtime.models.switch_layers import SwitchGLU
    return (SwitchGLU, FusedDownSwitchGLU, FusedGateUpSwitchGLU)


def _switch_layout(sw, where):
    """(F, params per expert) of a switch GLU, or raise: neuron mode never falls back to expert units."""
    ok = _neuron_switch_types()
    if type(sw) not in ok:
        raise ValueError(f"{where}: --moe-units neuron supports only {', '.join(t.__name__ for t in ok)} switch "
                         f"modules (their gate/up -> activation -> down body is the one re-run in pieces); found "
                         f"{type(sw).__module__}.{type(sw).__name__}")
    projs = [p for p in ("gate_proj", "up_proj", "gate_up_proj", "down_proj") if hasattr(sw, p)]
    if "bias" in sw.down_proj:
        raise ValueError(f"{where}: --moe-units neuron refuses a biased down_proj (a fully masked expert would "
                         f"still write its bias, so it would not equal expert-mode drop)")
    F = int(sw.down_proj.input_dims)
    total = sum(_proj_params(getattr(sw, p)) for p in projs)
    if total % F:
        raise ValueError(f"{where}: switch params per expert {total} not divisible by F={F}")
    return F, total


def _down_colnorm(mx, dp, E, chunk=16):
    """||W_down[e][:, j]|| as [E, F] float32, dequantizing at most `chunk` experts at a time."""
    out = []
    for e0 in range(0, E, chunk):
        e1 = min(E, e0 + chunk)
        if hasattr(dp, "scales"):
            b = getattr(dp, "biases", None)
            kw = {"mode": dp.mode} if hasattr(dp, "mode") else {}
            W = mx.dequantize(dp.weight[e0:e1], dp.scales[e0:e1], None if b is None else b[e0:e1],
                              dp.group_size, dp.bits, **kw)
        else:
            W = dp.weight[e0:e1]
        n = mx.sqrt(mx.square(W.astype(mx.float32)).sum(axis=1))  # weight is [e, D, F]
        mx.eval(n)
        out.append(n)
    return mx.concatenate(out, axis=0)


def _switch_masked(mx, sw, x, inds, table=None, want_hidden=False):
    """The switch GLU in pieces: gate/up -> activation -> (* table[e, j]) -> down, through the module's own
    (quantized) switch projections. Returns (rows [..., k, D], activated hidden [..., k, F] or None), both
    in routed-slot order. Sorting follows the stock assignment threshold (same gather order as SwitchGLU)."""
    from mlx2.runtime.models import switch_layers as sl
    xe = mx.expand_dims(x, (-2, -3))
    thr = sw.__dict__.get("moe_sort_min_assignments")
    do_sort = inds.size >= (sl._GATHER_SORT_MIN_ASSIGNMENTS if thr is None else thr)
    idx, inv = inds, None
    if do_sort:
        xe, idx, inv = sl._gather_sort(xe, inds)
    if hasattr(sw, "gate_up_proj"):
        gu = sw.gate_up_proj(xe, idx, sorted_indices=do_sort)
        half = sw.hidden_dims
        h = sw.activation(gu[..., half:], gu[..., :half])
    else:
        h = sw.activation(sw.up_proj(xe, idx, sorted_indices=do_sort), sw.gate_proj(xe, idx, sorted_indices=do_sort))
    if table is not None:
        h = h * mx.expand_dims(table[idx], -2).astype(h.dtype)
    y = sw.down_proj(h, idx, sorted_indices=do_sort)
    if do_sort:
        y = sl._scatter_unsort(y, inv, inds.shape)
        h = sl._scatter_unsort(h, inv, inds.shape) if want_hidden else None
    return y.squeeze(-2), (h.squeeze(-2) if want_hidden else None)


def _neuron_record(mx, self, inds, w, h):
    """Per-layer [E*F] contribution sums (scatter-add over tokens, on device) plus per-token top-32 flat ids."""
    E, F = self._amoeba_E, self._amoeba_F
    c = mx.abs(h.astype(mx.float32)) * w.astype(mx.float32)[..., None] * self._amoeba_colnorm[inds]  # [..., k, F]
    flat = inds.astype(mx.int32)[..., None] * F + mx.arange(F, dtype=mx.int32)
    tot = mx.zeros((E * F,), mx.float32).at[flat.reshape(-1)].add(c.reshape(-1))
    cc = c.reshape(*c.shape[:-2], -1)
    nt = min(32, cc.shape[-1])
    top = mx.argpartition(-cc, kth=nt - 1, axis=-1)[..., :nt]
    ids = mx.take_along_axis(inds.astype(mx.int32), top // F, axis=-1) * F + top % F
    return ("moe_neuron", tot, ids.astype(mx.int32), mx.take_along_axis(cc, top, axis=-1).astype(mx.float16))


def _moe_rows(mx, self, kind, a, switch=None):
    """(inds, scores, rows, shared) for one MoE site call, routed exactly as the stock block."""
    if kind == "gemma4_experts":
        x, inds, w = a
        return inds, w, (self.switch_glu(x, inds) if switch is None else switch(x, inds, self.switch_glu)), None
    (x,) = a
    sw = self.switch_mlp
    fn = _qwen_next_rows if kind == "qwen_next" else _ling_rows
    return fn(mx, self, x, None if switch is None else (lambda x_, i_: switch(x_, i_, sw)))


def _moe_expert_call(mx, self, kind, a):
    li_ = self._amoeba_layer
    inds, w, rows, shared = _moe_rows(mx, self, kind, a)
    if ST["allow"] is not None:
        w = w * ST["allow"][li_][inds].astype(w.dtype)
    if ST["record"] is not None:
        ye = rows.astype(mx.float32) * w.astype(mx.float32)[..., None]
        ST["record"][li_] = ("moe", inds.astype(mx.int16), w.astype(mx.float16),
                             mx.linalg.norm(ye, axis=-1).astype(mx.float16))
    return _combine(mx, self, kind, rows, w, shared)


def _moe_neuron_call(mx, self, kind, a):
    li_ = self._amoeba_layer
    E_, F_ = self._amoeba_E, self._amoeba_F
    table = None
    if ST["allow"] is not None:
        table = ST["allow"][li_][: E_ * F_].reshape(E_, F_)
        if self._amoeba_sw_E > E_:  # folded shared expert row(s): always on
            table = mx.concatenate([table, mx.ones((self._amoeba_sw_E - E_, F_), table.dtype)], 0)
    want = ST["record"] is not None
    cap = {}

    def switch(x_, i_, sw):
        rows_, cap["h"] = _switch_masked(mx, sw, x_, i_, table, want)
        return rows_

    inds, w, rows, shared = _moe_rows(mx, self, kind, a, switch)
    out = _combine(mx, self, kind, rows, w, shared)
    if want:
        rec = _neuron_record(mx, self, inds, w, cap.pop("h")[..., : inds.shape[-1], :])
        # Materialise this layer's sums and output now, so its routed hidden [T, k, F] and contribution
        # tensor are released before the next layer builds its own (one layer's worth alive at a time).
        mx.eval(out, *rec[1:])
        ST["record"][li_] = rec
    return out


def install_units(mx, layers, moe_units="expert"):
    """Patch one unit site per layer; returns [(layer, kind, n_units, params_per_unit, module)].

    moe_units="expert": a MoE layer's unit is a routed expert. "neuron": a unit is (expert e, intermediate
    neuron j), flat id e*F + j; see the module docstring of amoeba_cwe_donor.py. Any MoE site the neuron path
    cannot carve exactly raises instead of falling back to expert units. A neuron-installed site can run the
    expert rule for a check (ST["moe_units_override"] = "expert", allow indexed by expert id)."""
    from mlx2.experimental.weight_space import residual_ablation as ra

    if moe_units not in MOE_UNITS:
        raise ValueError(f"moe_units must be expert or neuron, not {moe_units!r}")

    sites = []
    for li, layer in enumerate(layers):
        cands = [getattr(layer, n, None) for n in ("experts", "mlp", "feed_forward", "block_sparse_moe")]
        site = None
        for m in cands:
            if m is not None and classify(m):
                site = (classify(m), m)
                break
        if site is None:
            mlp = next((m for m in cands if m is not None and hasattr(m, "down_proj")), None)
            if mlp is None:
                sites.append((li, None, 0, 0, None))
                continue
            site = ("dense", mlp.down_proj)
        kind, m = site
        base = type(m)
        if kind == "dense":
            W = ra.dense_weight(m)            # [D, F]
            m._amoeba_colnorm = mx.array(np.linalg.norm(W, axis=0).astype(np.float32))
            n, ppu = W.shape[1], 3 * W.shape[0]
            del W

            def call(self, x, _base=base):
                if not ST["active"]:
                    return _base.__call__(self, x)
                li_ = self._amoeba_layer
                if ST["allow"] is not None:
                    x = x * ST["allow"][li_][: x.shape[-1]].astype(x.dtype)
                if ST["record"] is not None:
                    c = mx.abs(x.astype(mx.float32)) * self._amoeba_colnorm
                    top = mx.argpartition(-c, kth=32, axis=-1)[..., :32]
                    ST["record"][li_] = ("dense", c.reshape(-1, c.shape[-1]).sum(0),
                                         top.astype(mx.int32), mx.take_along_axis(c, top, axis=-1).astype(mx.float16))
                return _base.__call__(self, x)
        else:
            if kind == "gemma4_experts":
                n = m.switch_glu.num_experts if hasattr(m.switch_glu, "num_experts") else None
            else:
                n = getattr(m, "num_experts", None) or getattr(getattr(m, "gate", None), "n_routed_experts", None)
            sw = m.switch_glu if kind == "gemma4_experts" else m.switch_mlp
            ppu = sum(_proj_params(getattr(sw, p)) for p in ("gate_proj", "up_proj", "gate_up_proj", "down_proj")
                      if hasattr(sw, p))
            if n is None:
                n = getattr(sw, "num_experts", None) or sw.down_proj.weight.shape[0]
            m._amoeba_units = moe_units
            if moe_units == "neuron":
                F, per_expert = _switch_layout(sw, f"layer {li} ({kind}, {type(m).__name__})")
                E = int(n)
                if sw.down_proj.weight.shape[0] < E:
                    raise ValueError(f"layer {li}: switch holds {sw.down_proj.weight.shape[0]} experts < {E} routed")
                m._amoeba_E, m._amoeba_F, m._amoeba_sw_E = E, F, int(sw.down_proj.weight.shape[0])
                m._amoeba_colnorm = _down_colnorm(mx, sw.down_proj, E)
                n, ppu = E * F, per_expert // F

            def call(self, *a, _base=base, _kind=kind):
                if not ST["active"]:
                    return _base.__call__(self, *a)
                units = ST.get("moe_units_override") or self._amoeba_units
                if units == "neuron":
                    if self._amoeba_units != "neuron":
                        raise RuntimeError("neuron units requested on a site installed with expert units")
                    return _moe_neuron_call(mx, self, _kind, a)
                return _moe_expert_call(mx, self, _kind, a)

        m.__class__ = type(f"Amoeba{base.__name__}", (base,), {"__call__": call})
        m._amoeba_layer = li
        sites.append((li, kind, int(n), int(ppu), m))
    return sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--capture-every", type=int, default=4, help="layer outputs kept every N layers plus layers 0-5")
    ap.add_argument("--coverage", default="0.99,0.95,0.9,0.8,0.7,0.6,0.5,0.4,0.3")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--no-thinking", action="store_true", help="plain chat, for routes without reasoning (match the screen)")
    ap.add_argument("--device", default="mps", help="accepted for the queue driver; MLX owns Metal")
    ap.add_argument("--cpu", action="store_true", help="smoke tests only: run MLX on the CPU")
    ap.add_argument("--moe-units", choices=("expert", "neuron"), default="expert",
                    help="MoE unit granularity: whole routed experts (default) or (expert, neuron) pairs")
    ap.add_argument("--smoke-accept-all", action="store_true",
                    help="smoke tests only: count every trace and regeneration as correct (exercises the pipeline)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty dir {out}")
    for d in ("units", "hidden"):
        (out / d).mkdir(parents=True, exist_ok=True)
    import amoeba_gsm549 as g549  # GSM8K idx 549 and its variants (numeric oracle)
    cand = next((c for c in list(screen.candidates()) + g549.items() if c["id"] == args.candidate), None)
    if cand is None:
        sys.exit(f"unknown candidate {args.candidate}")

    import mlx.core as mx
    if args.cpu:
        mx.set_default_device(mx.cpu)
    from mlx2.adapters.registry import inspect_model
    from mlx2.experimental.weight_space import residual_ablation as ra
    from mlx2.serving import render_prompt_tokens

    t_start = time.time()
    adapter = inspect_model(args.model).adapter_type(str(args.model))
    load_s = time.time() - t_start
    model, tok = adapter.model, adapter.tokenizer
    lm = getattr(model, "language_model", model)
    inner = lm.model
    layers = list(inner.layers)
    L = len(layers)
    cfg = json.loads((Path(args.model) / "config.json").read_text())
    hidden_size = (cfg.get("text_config") or cfg).get("hidden_size")
    checkpoint = f"{Path(args.model).name}@{adapter.identity['fingerprint']}"
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                           text=True).stdout.strip()
    mx.set_cache_limit(4 << 30)
    sites = install_units(mx, layers, moe_units=args.moe_units)
    unit_layers = [s for s in sites if s[1] is not None]
    Nmax = max(s[2] for s in unit_layers)
    kinds = sorted({s[1] for s in unit_layers})
    cap_layers = sorted(set(list(range(min(6, L))) + list(range(args.capture_every - 1, L, args.capture_every)) + [L - 1]))
    eos = set(tok.eos_token_ids)
    request = {"messages": [{"role": "user", "content": cand["prompt"]}], "enable_thinking": not args.no_thinking, "max_tokens": args.max_tokens}
    prompt = [int(t) for t in render_prompt_tokens(adapter, request)]
    T0 = len(prompt)
    print(f"{checkpoint} rev {rev} load {load_s:.0f}s layers {L} unit kinds {kinds} max units/layer {Nmax} prompt {T0}",
          flush=True)

    def step_logits(seq, cache):
        return model(mx.array([seq]), cache=cache)[0, -1].astype(mx.float32)

    def generate(temp, seed):
        cache = model.make_cache()
        logits = step_logits(prompt, cache)
        key = mx.random.key(seed)
        gen = []
        for _ in range(args.max_tokens):
            if temp <= 0:
                t = int(mx.argmax(logits).item())
            else:
                key, sub = mx.random.split(key)
                t = int(mx.random.categorical(logits / temp, key=sub).item())
            if t in eos:
                break
            gen.append(t)
            logits = step_logits([t], cache)
        del cache
        text = tok.decode(gen)
        pred = screen.extract_final(text)
        return {"gen": gen, "tokens": len(gen), "pred": pred, "correct": g549.correct(pred, cand) or args.smoke_accept_all,
                "hit_cap": len(gen) >= args.max_tokens, "tail": text[-400:]}

    def forward(seq, record=False, hidden=False):
        """Teacher-forced; float32 logits at the positions that predict generated tokens."""
        ST["record"] = [None] * L if record else None
        keep = {}

        def hook(i):
            def h(o):
                keep[i] = (o[0] if isinstance(o, tuple) else o)[0]
                return o
            return h

        pairs = [(layers[i], hook(i)) for i in cap_layers] if hidden else []
        with ra._hooked(pairs):
            lg = model(mx.array([seq[:-1]]), cache=model.make_cache())
        if lg.shape[1] != len(seq) - 1:
            raise RuntimeError(f"forward returned {lg.shape[1]} rows for {len(seq) - 1} inputs")
        lg = lg[0, T0 - 1:].astype(mx.float32)
        mx.eval(lg)
        rec, ST["record"] = ST["record"], None
        return lg, rec, keep

    def ref_summary(lg, k=256):
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        top = mx.argpartition(-lp, kth=k, axis=-1)[:, :k]
        r = {"top": top, "top_lp": mx.take_along_axis(lp, top, axis=-1), "argmax": mx.argmax(lg, -1)}
        mx.eval(r)
        return r

    def kl_from_ref(ref, lg):
        lq = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        lq_top = mx.take_along_axis(lq, ref["top"], axis=-1)
        p_top = mx.exp(ref["top_lp"])
        p_tail = mx.maximum(1 - p_top.sum(-1), 1e-12)
        q_tail = mx.maximum(1 - mx.exp(lq_top).sum(-1), 1e-12)
        kl = (p_top * (ref["top_lp"] - lq_top)).sum(-1) + p_tail * (mx.log(p_tail) - mx.log(q_tail))
        return np.asarray(kl), np.asarray(mx.argmax(lg, -1) == ref["argmax"])

    # 1. traces
    ST["active"] = False
    traces = []
    for s in [-1] + list(range(args.samples)):
        temp = 0.0 if s < 0 else args.temperature
        tg = time.time()
        tr = generate(temp, args.seed + max(s, 0))
        tr.update({"sample": s, "temperature": temp, "gen_s": round(time.time() - tg, 1)})
        traces.append(tr)
        print(f"{time.strftime('%H:%M:%S')} trace s{s} T={temp} tokens {tr['tokens']} pred {tr['pred']!r} "
              f"correct {tr['correct']} ({tr['gen_s']}s)", flush=True)
    with open(out / "traces.jsonl", "w") as f:
        for tr in traces:
            f.write(json.dumps({k: v for k, v in tr.items() if k != "gen"} | {"ids": prompt + tr["gen"]}) + "\n")
    good = [i for i, t in enumerate(traces) if t["correct"]]
    if not good:
        (out / "run.json").write_text(json.dumps({"checkpoint": checkpoint, "error": "no correct trace"}, indent=1))
        sys.exit("no correct trace")

    # parity of the patched reference path (unrestricted) against the stock forward
    seq0 = prompt + traces[good[0]]["gen"]
    lg_a, _, _ = forward(seq0)
    ST["active"] = True
    lg_b, _, _ = forward(seq0)
    parity = {"max_abs_logit_diff": float(mx.abs(lg_a - lg_b).max().item()),
              "top1_agree": float((mx.argmax(lg_a, -1) == mx.argmax(lg_b, -1)).astype(mx.float32).mean().item())}
    print(f"parity {parity}", flush=True)
    del lg_a, lg_b
    if parity["top1_agree"] < 0.98:
        (out / "run.json").write_text(json.dumps({"checkpoint": checkpoint, "parity": parity, "error": "parity"}, indent=1))
        sys.exit("reference path parity failed")

    # 2. capture (ST active, no mask)
    mass = np.zeros((L, Nmax), np.float64)
    refs = {}
    for ti, tr in enumerate(traces):
        lg, rec, keep = forward(prompt + tr["gen"], record=True, hidden=True)
        refs[ti] = ref_summary(lg)
        save = {"prompt_len": T0, "correct": tr["correct"]}
        for li, r in enumerate(rec):
            if r is None:
                continue
            if r[0] == "moe":
                inds, w, c = (np.asarray(x)[0] for x in r[1:])
                save[f"L{li}_inds"], save[f"L{li}_w"], save[f"L{li}_c"] = inds, w, c
                if tr["correct"]:
                    np.add.at(mass[li], inds.reshape(-1).astype(np.int64), c.reshape(-1).astype(np.float64))
            else:
                tot, top, topc = (np.asarray(x) for x in r[1:])
                save[f"L{li}_top"], save[f"L{li}_topc"] = top[0], topc[0]
                save[f"L{li}_sum"] = tot.astype(np.float32)
                if tr["correct"]:
                    mass[li, :tot.shape[0]] += tot
        np.savez_compressed(out / "units" / f"trace{ti}.npz", **save)
        hid = {}
        for li, g in keep.items():
            g = g.astype(mx.float32)
            H = hidden_size or g.shape[-1]
            if g.shape[-1] != H and g.shape[-1] % H == 0:
                g = g.reshape(g.shape[0], -1, H).mean(1)
            hid[f"L{li}"] = np.asarray(g.astype(mx.float16))
        np.savez_compressed(out / "hidden" / f"trace{ti}.npz", **hid)
        del lg, rec, keep
        print(f"{time.strftime('%H:%M:%S')} captured trace {ti} ({len(tr['gen'])} gen tokens)", flush=True)

    # 3. boxes
    rng = np.random.default_rng(args.seed)
    nunits = np.array([s[2] for s in sites])
    ppu = np.array([s[3] for s in sites], np.float64)
    valid = np.zeros((L, Nmax), bool)
    for li, n in enumerate(nunits):
        valid[li, :n] = True
    boxes = {"union": (mass > 0) & valid}
    covs = [float(c) for c in args.coverage.split(",")]
    for cov in covs:
        b = np.zeros((L, Nmax), bool)
        for li in range(L):
            if mass[li].sum() <= 0:
                continue
            order = np.argsort(-mass[li])
            cum = np.cumsum(mass[li][order]) / mass[li].sum()
            b[li, order[:int(np.searchsorted(cum, cov) + 1)]] = True
        boxes[f"mass{cov:g}"] = b

    def twin(b):
        r = np.zeros_like(b)
        for li in range(L):
            k = int(b[li].sum())
            if k:
                r[li, rng.choice(nunits[li], k, replace=False)] = True
        return r

    def allow_of(b):
        a = b.astype(np.float32)
        for li, s in enumerate(sites):  # layers without units: nothing to mask
            if s[1] is None:
                a[li] = 1.0
        return mx.array(a)

    def summarize(b):
        per = b.sum(1)
        return {"units_total": int(per.sum()), "units_per_layer": per.tolist(),
                "fraction_of_units": round(float(per.sum() / max(1, nunits.sum())), 5),
                "params": float((per * ppu).sum()), "params_fraction": round(float((per * ppu).sum() / (nunits * ppu).sum()), 5)}

    def run_arm(name, b, regen):
        ST["allow"] = allow_of(b)
        row = {"arm": name, **summarize(b), "traces": []}
        for ti in range(len(traces)):
            lg, _, _ = forward(prompt + traces[ti]["gen"])
            kl, agree = kl_from_ref(refs[ti], lg)
            row["traces"].append({"trace": ti, "correct_trace": traces[ti]["correct"], "kl_mean": float(kl.mean()),
                                  "kl_p95": float(np.quantile(kl, 0.95)), "top1_agree": float(agree.mean())})
            del lg
        if regen:
            g = generate(0.0, args.seed)
            row["regen"] = {k: g[k] for k in ("tokens", "pred", "correct", "hit_cap", "tail")}
        ST["allow"] = None
        good_t = [t for t in row["traces"] if t["correct_trace"]]
        row["kl_mean_correct"] = float(np.mean([t["kl_mean"] for t in good_t]))
        row["top1_agree_correct"] = float(np.mean([t["top1_agree"] for t in good_t]))
        print(f"{time.strftime('%H:%M:%S')} {name}: units {row['units_total']} ({row['fraction_of_units']:.2%}, "
              f"params {row['params_fraction']:.2%}) KL {row['kl_mean_correct']:.4f} agree {row['top1_agree_correct']:.3f} "
              f"regen {row.get('regen', {}).get('correct')}", flush=True)
        with open(out / "arms.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    arms = {}
    arms["union"] = run_arm("union", boxes["union"], True)
    fails, smallest_pass = 0, None
    for cov in covs:
        name = f"mass{cov:g}"
        regen = fails < 2
        arms[name] = run_arm(name, boxes[name], regen)
        if regen:
            if arms[name]["regen"]["correct"]:
                smallest_pass, fails = name, 0
            else:
                fails += 1
    arms["random_union"] = run_arm("random_union", twin(boxes["union"]), True)
    needed = None
    if smallest_pass:
        tw = run_arm(f"random_{smallest_pass}", twin(boxes[smallest_pass]), True)
        arms[f"random_{smallest_pass}"] = tw
        needed = {"arm": smallest_pass, **summarize(boxes[smallest_pass]), "random_twin_correct": tw["regen"]["correct"]}

    box_json = {"checkpoint": checkpoint, "mlx2_source_revision": rev, "candidate": cand["id"], "layers": L,
                "moe_units": args.moe_units,
                "sites": [{"layer": s[0], "kind": s[1], "units": s[2], "params_per_unit": s[3]} for s in sites]}
    if args.moe_units == "expert":
        box_json |= {"per_layer_mass": [mass[li, :nunits[li]].round(5).tolist() for li in range(L)],
                     "boxes": {n: {**summarize(b), "units": [np.flatnonzero(b[li]).tolist() for li in range(L)]}
                               for n, b in boxes.items()}}
    else:  # E*F units per MoE layer: arrays beside box.json, not JSON lists
        np.save(out / "per_layer_mass.npy", mass.astype(np.float32))
        np.savez_compressed(out / "boxes.npz", **{n: b for n, b in boxes.items()})
        box_json |= {"per_layer_mass_file": "per_layer_mass.npy", "boxes_file": "boxes.npz",
                     "unit_id": "MoE layers: expert * F + neuron; per_layer_mass_file is [L, Nmax] float32, "
                                "boxes_file holds one [L, Nmax] bool mask per box",
                     "moe_layout": {str(s[0]): [s[4]._amoeba_E, s[4]._amoeba_F] for s in sites
                                    if s[1] not in (None, "dense")},
                     "boxes": {n: summarize(b) for n, b in boxes.items()}}
    (out / "box.json").write_text(json.dumps(box_json))
    rep = {"schema": "mlx2.amoeba-unit-box.v1", "checkpoint": checkpoint, "mlx2_source_revision": rev, "tree_dirty": bool(dirty),
           "candidate": cand, "unit_kinds": kinds, "moe_units": args.moe_units, "units_total": int(nunits.sum()), "args": vars(args),
           "load_s": round(load_s, 1), "prompt_tokens": T0, "parity": parity,
           "traces": [{k: v for k, v in t.items() if k != "gen"} for t in traces], "correct_traces": good,
           "units_needed": needed,
           "arms": {n: {k: a.get(k) for k in ("units_total", "fraction_of_units", "params_fraction", "kl_mean_correct",
                                               "top1_agree_correct")} | {"regen_correct": a.get("regen", {}).get("correct")}
                    for n, a in arms.items()},
           "wall_s": round(time.time() - t_start, 1)}
    (out / "run.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({"units_needed": needed}, indent=1))


if __name__ == "__main__":
    main()
