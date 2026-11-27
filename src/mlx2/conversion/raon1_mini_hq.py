"""Calibrated mixed-precision affine conversion for Raon-1-mini (recipes ``raon1-mini-affine-mix-v2/v3``).

Original mlx2 code following the method of Litke & Lamy, "Crossing the 384 GB
Cliff: Sub-4-Bit Mixed Trellis-NVFP4 MoE Compaction" (2026): layer-uniform
precision for the routed experts chosen by a distortion knapsack under
a byte budget, error-feedback (GPTQ/LDLQ-family) rounding against full-rank
Hessians calibrated on many more tokens than the hidden size (N_tok >> H, the
rank-deficiency trap), and the boundary-heavy sensitivity that motivates
keeping sensitive layers at higher precision.  Adapted to MLX affine planes
(the format menu below), no trellis codes.  Provenance: provenance/raon1-mini.json.

Pipeline (GPU device, the exact mxfp8e artifact as the source of weights):

1. calibrate: forward ``tokens`` (default >= 32 * hidden) through the model
   and accumulate fp32 ``H = sum x x^T`` for every shared linear input:
   RDA projections (q/k/v/b/f_a/g/w share one input), f_b, o_proj; MLA
   q_a/kv_a/g (shared), q_b, o_proj; the MoE input (router, routed gate/up and
   the shared expert's gate/up share it: the paper's shared full-rank FC1
   Hessian); shared and routed down inputs (routed rows pooled over experts);
   dense layer gate/up and down.  Expert routing frequencies are counted.
2. sensitivity: per MoE layer and format, a layer cost.  Scorer ``kl``
   (default, recipe v3): KL(source || variant) on held-out probe rows (after
   the calibration rows), where the variant is the source with only that
   layer's routed experts on that format's real GPTQ planes; the source's own
   fp16 log-probs are the reference.  Scorer ``hessian`` (recipe v2): the
   routing-frequency-weighted Hessian error ``sum_e p_e tr(dW_e H dW_e^T)``
   of round-to-nearest planes (gate/up against the FC1 Hessian, down against
   the pooled down Hessian).  The Hessian error is recorded either way.  On
   Raon the Hessian error chose a worse schedule than greedy did (v2 vs v1,
   oracle KL 0.075 vs 0.045), which is why KL is the default.
3. knapsack: one format per MoE layer minimising the summed layer cost with
   the average expert bits/weight within ``budget_bpw``, solved exactly by
   ``mlx2.conversion.knapsack.solve`` (v1 used greedy marginal-per-bit).
   Costs are assumed additive; at these widths (4-bit and up) the Mellum
   check found single-layer KL nearly additive.
4. quantize: expert banks with GPTQ at their layer's format (a bank is one
   ``[E*O, I]`` matrix against its shared Hessian); dense linears and the
   head at ``dense_bits`` / ``head_bits`` (GPTQ where a Hessian exists).
   Embedding, kv_b_proj, router, mHC, norms, convs, A_log/dt_bias, expert
   bias and steering keep their stored dtype.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import mlx.core as mx

# v1 (2026-11-25): greedy marginal-per-bit schedule over the Hessian-weighted error.
# v2 (2026-11-26): exact DP schedule over the same error (worse end-to-end than v1: the
#     proxy error is not a faithful objective; qualification/runs/raon1-mini-20261126/hq-v2-5282c7ca7).
# v3 (2026-11-26): exact DP schedule over a per-layer end-to-end KL probe (scorer "kl", default).
RECIPE_IDS = {"hessian": "raon1-mini-affine-mix-v2", "kl": "raon1-mini-affine-mix-v3"}
RECIPE_ID = RECIPE_IDS["kl"]
# (name, bits, group_size, bits per weight incl. bf16 scale + bias)
FORMATS = {"a4g64": (4, 64, 4.5), "a4g32": (4, 32, 5.0), "a8g64": (8, 64, 8.5)}


@dataclass
class HQSettings:
    tokens: int = 98304
    seq: int = 2048
    budget_bpw: float = 5.0
    formats: tuple = ("a4g64", "a4g32", "a8g64")
    dense_bits: int = 8
    dense_group: int = 64
    head_bits: int = 8
    damp: float = 0.01
    min_first_layers_bits: dict = field(default_factory=dict)  # optional floors {layer: format}
    # Layer cost for the knapsack: "kl" (default) = KL(source || variant) on held-out probe rows
    # when that layer's routed experts alone take a format's real GPTQ planes; "hessian" = the
    # routing-weighted Hessian error of RTN planes (recipe v2).
    scorer: str = "kl"
    kl_rows: int = 4        # probe rows of ``seq`` tokens, after (disjoint from) the calibration rows
    kl_skip: int = 1024     # positions scored per row: skip .. seq-1

    def __post_init__(self):
        if self.scorer not in RECIPE_IDS:
            raise ValueError(f"scorer must be one of {sorted(RECIPE_IDS)}")
        if self.scorer == "kl" and (self.kl_rows < 1 or not 0 <= self.kl_skip < self.seq - 1):
            raise ValueError("kl_rows must be >= 1 and kl_skip within the row length")

    @property
    def recipe(self) -> str:
        return RECIPE_IDS[self.scorer]


# --------------------------------------------------------------- recording --

class _Recorder:
    """Callable stand-in for a module: accumulates ``x^T x`` then calls it."""

    def __init__(self, module, store, key, *, counts=None, experts=0):
        self.module, self.store, self.key = module, store, key
        self.counts, self.experts = counts, experts

    def __call__(self, x, *args, **kwargs):
        rows = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        update = rows.T @ rows
        prev = self.store.get(self.key)
        self.store[self.key] = update if prev is None else prev + update
        self.store[self.key + "#rows"] = self.store.get(self.key + "#rows", 0) + int(rows.shape[0])
        out = self.module(x, *args, **kwargs)
        if self.counts is not None:  # a router: (ids, weights)
            ids = out[0].reshape(-1)
            hits = mx.zeros((self.experts,), dtype=mx.int32).at[ids].add(1)
            prev = self.counts.get(self.key)
            self.counts[self.key] = hits if prev is None else prev + hits
        return out

    def __getattr__(self, name):
        return getattr(self.module, name)


def _targets(model):
    """``(parent, name, key, router)`` for every recorded input (one per shared input)."""
    out = []
    for i, layer in enumerate(model.layers):
        a = layer.self_attn
        if layer.is_global:
            out += [(a, "q_a_proj", f"{i}.attn_in", False), (a, "q_b_proj", f"{i}.q_b", False),
                    (a, "o_proj", f"{i}.o", False)]
        else:
            out += [(a, "q_proj", f"{i}.attn_in", False), (a, "f_b_proj", f"{i}.f_b", False),
                    (a, "o_proj", f"{i}.o", False)]
        if layer.is_moe:
            m = layer.moe
            out += [(m, "gate", f"{i}.moe_in", True), (m.shared_experts, "down_proj", f"{i}.shared_down", False),
                    (m.switch_mlp, "down_proj", f"{i}.routed_down", False)]
        else:
            out += [(layer.mlp, "gate_proj", f"{i}.mlp_in", False), (layer.mlp, "down_proj", f"{i}.mlp_down", False)]
    return out


def calibrate(model, sequences, *, log=print):
    """Accumulate the shared-input Hessians over ``sequences`` (lists of token ids)."""
    store, counts = {}, {}
    installed = []
    experts = int(model.args.num_experts)
    try:
        for parent, name, key, router in _targets(model):
            module = parent[name]
            rec = _Recorder(module, store, key, counts=counts if router else None, experts=experts)
            parent[name] = rec
            installed.append((parent, name, module))
        started = time.time()
        for index, ids in enumerate(sequences):
            model(mx.array([ids]))
            mx.eval([v for v in store.values() if isinstance(v, mx.array)] + list(counts.values()))
            if index % 8 == 0:
                log(f"calibration {index + 1}/{len(sequences)} ({time.time() - started:.0f}s)")
    finally:
        for parent, name, module in installed:
            parent[name] = module
    hessians = {k: v for k, v in store.items() if not k.endswith("#rows")}
    rows = {k[:-5]: v for k, v in store.items() if k.endswith("#rows")}
    return hessians, rows, counts


# ------------------------------------------------------------- sensitivity --

def _rtn_dequant(W2, bits, group):
    q, s, b = mx.quantize(W2, group_size=group, bits=bits)
    return mx.dequantize(q, s, b, group_size=group, bits=bits).astype(mx.float32)


def weighted_error(bank, H, weights, bits, group, *, chunk=16):
    """``sum_e weights[e] * tr(dW_e H dW_e^T)`` for a bank ``[E, O, I]`` (RTN planes)."""
    total = mx.array(0.0, dtype=mx.float32)
    E = bank.shape[0]
    for s in range(0, E, chunk):
        Wc = bank[s:s + chunk].astype(mx.float32)
        n, O, I = Wc.shape
        W2 = Wc.reshape(n * O, I)
        D = (_rtn_dequant(W2, bits, group) - W2).reshape(n, O, I)
        per = ((D @ H) * D).sum(axis=(1, 2))
        total = total + (per * weights[s:s + chunk]).sum()
        mx.eval(total)
    return float(total)


def knapsack(distortion: dict, budget_bpw: float, formats=("a4g64", "a4g32", "a8g64"), floors=None):
    """Exact per-layer format choice (the paper's eq. 4) via ``knapsack.solve``.

    ``distortion[layer][fmt]`` -> chosen ``{layer: fmt}`` minimising total
    distortion with mean bpw <= budget.  ``floors[layer]`` is a minimum
    format: formats with fewer bits/weight are forbidden for that layer.
    The paper's greedy marginal-per-bit allocation is not optimal for a
    multiple-choice knapsack (a cheap early upgrade can block a better large
    one); the DP is exact and costs nothing at these sizes.  Routed-expert
    banks have the same shape in every MoE layer, so bits/weight is a
    faithful size unit.
    """
    from fractions import Fraction

    from mlx2.conversion.knapsack import solve

    floors = floors or {}
    for layer, fmt in floors.items():
        if fmt not in formats:
            raise ValueError(f"floor {fmt!r} for layer {layer} is not in the menu {list(formats)}")
    bpw = {f: Fraction(FORMATS[f][2]).limit_denominator(1000) for f in formats}
    den = math.lcm(*(b.denominator for b in bpw.values()))
    unit = {f: int(b * den) for f, b in bpw.items()}         # integer size per layer
    layers = list(distortion)
    budget = math.floor(Fraction(budget_bpw).limit_denominator(10**6) * len(layers) * den)
    costs, sizes = [], []
    for layer in layers:
        lo = bpw[floors[layer]] if layer in floors else min(bpw.values())
        ok = [bpw[f] >= lo for f in formats]
        costs.append([float(distortion[layer][f]) if a else None for f, a in zip(formats, ok)])
        sizes.append([unit[f] if a else None for f, a in zip(formats, ok)])
    try:
        sol = solve(costs, sizes, budget, resolution=1)
    except ValueError as exc:
        raise ValueError(f"no schedule fits the {budget_bpw} bpw budget with these floors") from exc
    return {layer: formats[j] for layer, j in zip(layers, sol.choice)}


# ---------------------------------------------------------------- quantize --

def gptq_bank(bank, H, bits, group, *, damp, dtype=mx.bfloat16):
    """GPTQ a bank ``[E, O, I]`` as one ``[E*O, I]`` matrix against a shared Hessian."""
    from ..gptq import GPTQSettings, quantize_bank

    E, O, I = bank.shape
    settings = GPTQSettings(bits=bits, group_size=group, damp=damp)
    (w, s, b), receipt = quantize_bank(bank.reshape(E * O, I), lambda a, z: H[None], settings, dtype=dtype)
    return (w.reshape(E, O, -1), s.reshape(E, O, -1), b.reshape(E, O, -1)), receipt


def gptq_linear(W, H, bits, group, *, damp, dtype=mx.bfloat16):
    from ..gptq import GPTQSettings, quantize_bank

    settings = GPTQSettings(bits=bits, group_size=group, damp=damp)
    return quantize_bank(W, lambda a, z: H[None], settings, dtype=dtype)[0]


def _dense_bank(module):
    """The bf16 ``[E, O, I]`` expert weights of a (quantized) switch linear."""
    if hasattr(module, "scales"):
        return mx.dequantize(module["weight"], module["scales"], module.get("biases"),
                             group_size=module.group_size, bits=module.bits, mode=module.mode,
                             dtype=mx.bfloat16)
    return module.weight


def _dense_weight(linear):
    if "scales" in linear:
        return mx.dequantize(linear["weight"], linear["scales"], linear.get("biases"),
                             group_size=linear.group_size, bits=linear.bits, mode=linear.mode,
                             dtype=mx.bfloat16)
    return linear.weight


_BANKS = (("gate_proj", "moe_in"), ("up_proj", "moe_in"), ("down_proj", "routed_down"))
_MISSING = object()


def _expert_identity(layer):
    """What a restore must give back exactly: each bank module and its arrays, by identity."""
    glu = layer.moe.switch_mlp
    return [(glu[n], tuple(glu[n].get(k) for k in _PLANE_KEYS), glu[n].__dict__.get("biases", _MISSING),
             tuple(getattr(glu[n], k, None) for k in ("bits", "group_size", "mode"))) for n, _ in _BANKS]


def _same_identity(a, b) -> bool:
    return all(m1 is m2 and all(x is y for x, y in zip(p1, p2)) and b1 is b2 and t1 == t2
               for (m1, p1, b1, t1), (m2, p2, b2, t2) in zip(a, b))
_PLANE_KEYS = ("weight", "scales", "biases")


def apply_expert_format(layer, index, hessians, bits, group, *, damp):
    """Put one MoE layer's routed expert banks on GPTQ affine planes in place; returns what
    :func:`restore_experts` needs.  The planes are the ones the export would write."""
    glu = layer.moe.switch_mlp
    saved = []
    try:
        for name, key in _BANKS:
            module = glu[name]
            (w, sc, bi), _ = gptq_bank(_dense_bank(module), hessians[f"{index}.{key}"], bits, group, damp=damp)
            mx.eval(w, sc, bi)
            if "scales" not in module:
                # A dense bank: swap in a quantized one (its forward reads the planes), put back on restore.
                replacement = module.to_quantized(group_size=group, bits=bits, mode="affine")
                replacement["weight"], replacement["scales"], replacement["biases"] = w, sc, bi
                setattr(glu, name, replacement)
                saved.append((glu, name, module))
                continue
            # An mxfp8 bank keeps ``biases = None`` as a plain attribute, not a parameter key;
            # record which, so restore puts back exactly that.  ``setattr`` (MLX Module) drops
            # the shadowing attribute when it stores the array, so ``.biases`` reads the planes.
            attrs = {k: getattr(module, k) for k in ("bits", "group_size", "mode")}
            attrs["_biases_attr"] = module.__dict__.get("biases", _MISSING)
            saved.append((module, {k: module[k] for k in _PLANE_KEYS if k in module}, attrs))
            module["weight"], module["scales"] = w, sc
            setattr(module, "biases", bi)
            module.bits, module.group_size, module.mode = bits, group, "affine"
    except BaseException:
        restore_experts(saved)
        raise
    return saved


def restore_experts(saved) -> None:
    for entry in reversed(saved):
        if len(entry) == 3 and isinstance(entry[1], str):
            parent, name, original = entry
            setattr(parent, name, original)
            continue
        module, arrays, attrs = entry
        for k in _PLANE_KEYS:
            if k in arrays:
                module[k] = arrays[k]
            elif k in module:
                del module[k]
        attrs = dict(attrs)
        biases_attr = attrs.pop("_biases_attr", _MISSING)
        if biases_attr is not _MISSING and "biases" not in arrays:
            setattr(module, "biases", biases_attr)
        for k, v in attrs.items():
            setattr(module, k, v)
    mx.clear_cache()


def kl_costs(model, hessians, probe, settings: HQSettings, workdir: Path, *, checkpoint: str, log=print):
    """``{layer: {format: KL}}``: KL(source || source with only that layer's experts at the format),
    mean over the scored positions of the probe rows, plus the probe receipt."""
    from .. import layer_kl

    texts = [{"id": f"probe-{i}", "split": "dev", "ids": list(map(int, ids))} for i, ids in enumerate(probe)]
    ref = Path(workdir) / "kl-reference"
    meta = layer_kl.write_reference(ref, texts, layer_kl.reference_from_model(model), skip=settings.kl_skip,
                                    checkpoint=checkpoint, source={"recipe": settings.recipe}, dtype="float16")
    base = layer_kl.score(model, ref, meta, texts)
    log(f"kl probe: {meta['positions']} positions, source vs its fp16 reference KL {base.kl_mean:.3g}")
    costs, raw, started = {}, {}, time.time()
    for i, layer in enumerate(model.layers):
        if not layer.is_moe:
            continue
        costs[i], raw[i] = {}, {}
        identity = _expert_identity(layer)
        for fmt in settings.formats:
            bits, group, _ = FORMATS[fmt]
            saved = apply_expert_format(layer, i, hessians, bits, group, damp=settings.damp)
            try:
                got = layer_kl.score(model, ref, meta, texts)
            finally:
                restore_experts(saved)
            if not _same_identity(identity, _expert_identity(layer)):
                raise RuntimeError(f"kl probe: layer {i} experts not restored after {fmt}")
            raw[i][fmt] = {"kl_mean": got.kl_mean, "top1_agreement": got.top1}
            costs[i][fmt] = max(got.kl_mean - base.kl_mean, 0.0)
        log(f"kl layer {i}: " + ", ".join(f"{f} {costs[i][f]:.4g}" for f in settings.formats)
            + f" ({time.time() - started:.0f}s)")
    again = layer_kl.score(model, ref, meta, texts)
    if again.kl_mean != base.kl_mean or again.top1 != base.top1:
        raise RuntimeError(f"kl probe restore check failed: baseline {base.kl_mean!r} -> {again.kl_mean!r}")
    return costs, {"positions": meta["positions"], "rows": len(texts), "skip": settings.kl_skip,
                   "texts_sha256": meta["texts_sha256"], "baseline_kl": base.kl_mean,
                   "baseline_top1": base.top1, "per_layer": {str(k): v for k, v in raw.items()},
                   # fp16 reference: costs are KL minus this floor, clamped at 0; 8-bit costs sit
                   # near it (ties among them are noise, 4-bit vs 8-bit decisions are not).
                   "cost_floor": base.kl_mean,
                   "restore_check": "bit-equal", "seconds": round(time.time() - started, 1)}


def plan_and_quantize(model, hessians, counts, settings: HQSettings, *, log=print, costs=None):
    """Sensitivity -> knapsack -> quantized planes ``{tensor name: array}`` and a receipt.

    ``costs`` (``{layer: {format: cost}}``, e.g. from :func:`kl_costs`) drive the knapsack when
    given; the Hessian-weighted error is always computed and recorded for comparison."""
    distortion, moe_layers = {}, []
    for i, layer in enumerate(model.layers):
        if not layer.is_moe:
            continue
        moe_layers.append(i)
        glu = layer.moe.switch_mlp
        freq = counts[f"{i}.moe_in"].astype(mx.float32)
        p = freq / mx.maximum(freq.sum(), 1.0)
        H1, H2 = hessians[f"{i}.moe_in"], hessians[f"{i}.routed_down"]
        distortion[i] = {}
        for fmt in settings.formats:
            bits, group, _ = FORMATS[fmt]
            d = sum(weighted_error(_dense_bank(glu[n]), H1, p, bits, group) for n in ("gate_proj", "up_proj"))
            d += weighted_error(_dense_bank(glu["down_proj"]), H2, p, bits, group)
            distortion[i][fmt] = d
        log(f"sensitivity layer {i}: " + ", ".join(f"{f} {distortion[i][f]:.4g}" for f in settings.formats))
    floors = {int(k): v for k, v in (settings.min_first_layers_bits or {}).items()}
    if settings.scorer == "kl" and costs is None:
        raise ValueError("scorer 'kl' needs the probe costs (kl_costs)")
    objective = costs if settings.scorer == "kl" else distortion
    schedule = knapsack(objective, settings.budget_bpw, settings.formats, floors)
    planes, quant, receipts = {}, {}, {}
    for i, layer in enumerate(model.layers):
        p = f"model.layers.{i}."
        a = layer.self_attn
        dense = []
        if layer.is_global:
            dense += [(a, "q_a_proj", f"{i}.attn_in"), (a, "kv_a_proj_with_mqa", f"{i}.attn_in"),
                      (a, "g_proj", f"{i}.attn_in"), (a, "q_b_proj", f"{i}.q_b"), (a, "o_proj", f"{i}.o")]
        else:
            dense += [(a, n, f"{i}.attn_in") for n in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj",
                                                         "g_proj", "rda_w_proj")]
            dense += [(a, "f_b_proj", f"{i}.f_b"), (a, "o_proj", f"{i}.o")]
        if layer.is_moe:
            s = layer.moe.shared_experts
            dense += [(s, "gate_proj", f"{i}.moe_in"), (s, "up_proj", f"{i}.moe_in"),
                      (s, "down_proj", f"{i}.shared_down")]
        else:
            m = layer.mlp
            dense += [(m, "gate_proj", f"{i}.mlp_in"), (m, "up_proj", f"{i}.mlp_in"), (m, "down_proj", f"{i}.mlp_down")]
        for parent, name, key in dense:
            path = p + ("self_attn." if parent is a else ("moe.shared_experts." if layer.is_moe and parent is layer.moe.shared_experts else "mlp.")) + name
            W = _dense_weight(parent[name])
            if W.shape[-1] % settings.dense_group:
                continue  # input width not a whole number of groups: stays as stored
            w, sc, bi = gptq_linear(W, hessians[key], settings.dense_bits, settings.dense_group, damp=settings.damp)
            planes[path + ".weight"], planes[path + ".scales"], planes[path + ".biases"] = w, sc, bi
            mx.eval(w, sc, bi)
        if layer.is_moe:
            fmt = schedule[i]
            bits, group, _ = FORMATS[fmt]
            glu = layer.moe.switch_mlp
            for n, key in (("gate_proj", f"{i}.moe_in"), ("up_proj", f"{i}.moe_in"), ("down_proj", f"{i}.routed_down")):
                (w, sc, bi), rec = gptq_bank(_dense_bank(glu[n]), hessians[key], bits, group, damp=settings.damp)
                path = f"{p}moe.switch_mlp.{n}"
                planes[path + ".weight"], planes[path + ".scales"], planes[path + ".biases"] = w, sc, bi
                quant[path] = {"bits": bits, "group_size": group}
                receipts[path] = rec
                mx.eval(w, sc, bi)
            log(f"layer {i}: experts {fmt} quantized")
    head = _dense_weight(model.lm_head)
    w, sc, bi = mx.quantize(head, group_size=64, bits=settings.head_bits)
    planes["lm_head.weight"], planes["lm_head.scales"], planes["lm_head.biases"] = w, sc, bi
    quant["lm_head"] = {"bits": settings.head_bits, "group_size": 64}
    moe_bpw = sum(FORMATS[schedule[i]][2] for i in moe_layers) / max(len(moe_layers), 1)
    receipt = {"schedule": {str(k): v for k, v in schedule.items()}, "moe_mean_bpw": round(moe_bpw, 4),
               "budget_bpw": settings.budget_bpw,
               "scorer": settings.scorer,
               "distortion": {str(k): v for k, v in distortion.items()},
               **({"kl_costs": {str(k): v for k, v in costs.items()}} if costs is not None else {}),
               "gptq": receipts}
    return planes, quant, receipt


def _untouched(model, planes):
    """Every model tensor not replaced by a quantized plane, as stored."""
    from mlx.utils import tree_flatten

    replaced = {k.rsplit(".", 1)[0] for k in planes}
    out = {}
    for key, value in tree_flatten(model.parameters()):
        module = key.rsplit(".", 1)[0]
        if module in replaced:
            continue
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            # Checkpoint names (sanitize maps them back to the conv module).
            key = key.replace(f".{name}.conv.weight", f".{name}.weight")
        out[key] = value
    return out


def convert_hq(source: Path, dest: Path, settings: HQSettings | None = None, *, log=print) -> dict:
    """Calibrate on the exact mxfp8e artifact at ``source`` and write the mixed affine artifact."""
    from .raon1_mini import _publish

    return _publish(source, dest, partial(_write_hq, settings=settings or HQSettings()), log)


def _write_hq(source, dest, *, settings, log):
    from ..adapters import raon1_mini as A
    from ..runtime.models import raon1_mini as R
    from ..runtime.ubc_evict import load_shards_evicting
    from transformers import AutoTokenizer

    artifact = A.inspect_artifact(source)
    if artifact["layout"] != "mlx2_mxfp8e":
        raise ValueError("the HQ recipe calibrates on the exact mxfp8e artifact")
    started = time.time()
    model = R.Model(R.ModelArgs.from_dict(artifact["config"]))
    R.install_for_layout(model, artifact["layout"], artifact["config"], artifact["weight_map"].keys())
    weights = model.sanitize(load_shards_evicting([source / n for n in sorted(set(artifact["weight_map"].values()))]))
    model.load_weights(list(weights.items()), strict=True)
    weights.clear()
    model.eval()
    # Calibration runs the reference laws (no merged projections, ops mHC).
    model.configure(A.parse_policy({"merged_projections": False, "mhc": "ops"}))
    model.set_reasoning_effort(None)
    tok = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    n_calib = -(-settings.tokens // settings.seq)
    extra = settings.kl_rows * settings.seq if settings.scorer == "kl" else 0
    rows_all = calibration_sequences(tok, settings.tokens + extra, settings.seq)
    # Probe rows come after the calibration rows (the generator is sequential, so the
    # calibration rows are the same as without them): GPTQ never sees the probe text.
    sequences, probe = rows_all[:n_calib], rows_all[n_calib:]
    hessians, rows, counts = calibrate(model, sequences, log=log)
    hidden = int(model.args.hidden_size)
    ratio = rows[f"{next(i for i, l in enumerate(model.layers) if l.is_moe)}.moe_in"] / hidden
    log(f"calibration rows / hidden = {ratio:.1f}")
    costs, probe_receipt = None, None
    if settings.scorer == "kl":
        costs, probe_receipt = kl_costs(model, hessians, probe, settings, Path(dest).parent,
                                        checkpoint=artifact["checkpoint"], log=log)
    planes, quant, receipt = plan_and_quantize(model, hessians, counts, settings, log=log, costs=costs)
    if probe_receipt is not None:
        receipt["kl_probe"] = probe_receipt
    keep = _untouched(model, planes)
    index, files = {}, []
    tensors = {**keep, **planes}
    chunk, size = {}, 0
    for key in sorted(tensors):
        chunk[key] = tensors[key]
        size += tensors[key].nbytes
        if size >= 4 << 30:
            name = f"model-{len(files):02d}.safetensors"
            mx.save_safetensors(str(dest / name), chunk, metadata={"format": "mlx"})
            files.append(name); index.update({k: name for k in chunk}); chunk, size = {}, 0
    if chunk:
        name = f"model-{len(files):02d}.safetensors"
        mx.save_safetensors(str(dest / name), chunk, metadata={"format": "mlx"})
        files.append(name); index.update({k: name for k in chunk})
    config = dict(artifact["config"])
    config["quantization"] = {"group_size": settings.dense_group, "bits": settings.dense_bits, "mode": "affine", **quant}
    config["mlx2_conversion"] = {**config["mlx2_conversion"], "recipe": settings.recipe,
                                 "experts": "GPTQ affine by layer (quantization module overrides)",
                                 "dense": f"GPTQ affine {settings.dense_bits}-bit g{settings.dense_group} where aligned",
                                 "head": f"affine {settings.head_bits}-bit g64",
                                 "calibrated_from": artifact["checkpoint"],
                                 "method": ("per-layer KL-probe knapsack + GPTQ" if settings.scorer == "kl" else
                                            "Hessian-weighted layer knapsack + GPTQ (Litke & Lamy 2026 method on MLX affine)")}
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (dest / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"format": "mlx"}, "weight_map": dict(sorted(index.items()))}, indent=1) + "\n")
    for name in ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "LICENSE", "README.md"):
        if (source / name).is_file():
            shutil.copy2(source / name, dest / name)
    out = {"schema": "mlx2.raon1-mini-conversion.v1", "recipe": settings.recipe, "source": artifact["checkpoint"],
           "settings": {k: v for k, v in vars(settings).items()}, "calibration": {
               "sequences": len(sequences), "rows_moe_in": rows, "rows_over_hidden": round(ratio, 2)},
           **receipt, "outputs": files, "seconds": round(time.time() - started, 1)}
    out["calibration"]["rows_moe_in"] = {k: v for k, v in rows.items() if k.endswith("moe_in")}
    (dest / "mlx2-conversion-receipt.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def calibration_sequences(tok, tokens: int, seq: int):
    """Varied calibration text: repository prose/code (realistic_haystack) and
    chat-formatted GSM8K training problems with solutions, cut to ``seq``-token rows."""
    import sys

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "scripts"))
    from realistic_haystack import haystack

    rows, offset = [], 0
    n_rows = -(-tokens // seq)
    # The vendored full GSM8K training split (disjoint from the test questions
    # the acceptance runs score); each problem is used once.
    path = root / "benchmarks/gsm8k/train.jsonl"
    train = ([json.loads(line) for line in path.read_text().split("\n") if line.strip()]
             if path.is_file() else [])
    chat, k = [], 0
    while len(rows) < n_rows:
        if train and len(rows) % 3 == 2:
            msgs = []
            while True:
                if k >= len(train):
                    raise ValueError("GSM8K training split exhausted for calibration")
                item = train[k]; k += 1
                msgs += [{"role": "user", "content": item["question"]},
                         {"role": "assistant", "content": item["answer"]}]
                ids = tok.apply_chat_template(msgs, return_dict=True)["input_ids"]
                if len(ids) >= seq:
                    rows.append(list(ids)[:seq]); break
        else:
            text = haystack(seq * 6, offset=offset)
            offset += len(text)
            rows.append(tok.encode(text, add_special_tokens=False)[:seq])
    return rows
