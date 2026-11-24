"""Layer-major removal scoring: the removal check's KL(full || full - box) without the whole model resident.

``child_removal``'s resident ``Scorer`` runs every window through the whole loaded model once per arm. For a donor
that does not fit beside the desktop (Hy4-flash-MLX-4.6bit-mlx2, ~85 GB: five eager loads tripped the carve memory
watchdog on 2026-11-22), ``LayerMajorScorer`` computes the same numbers layer-major through a sweep family
(research/amoeba/sweep, interface v1): for each chunk of windows it walks the layers once, loads layer l's weights
(``family._load_layer``), installs the carve's own unit sites on that one layer (``units.install([layer])``, the same
code and mask law the resident host installs on every layer), pushes every arm's carried state through it with the
arm's keep row as ``ctl.core`` (zero ablation, no sink layers: the removal check's setting), and drops the layer.
After the last layer each arm's final logits are scored against the full arm's with the carve's ``kl_rows``, per
window, over the same scored positions. Nothing but the windows' carries (one per arm) and one layer are resident.

Same interface as ``child_removal.Scorer``: ``kl_many(wins, keeps) -> {name: per-window mean KL}``. The numbers are
the resident scorer's up to the family's numerical tolerance (its layers are the model's own DecoderLayer outside
the adapter's trunk scopes); ``parity`` measures that on real windows over the first K layers against a resident
K-layer model carrying the same carve sites, and the CPU tests check it at full depth on a tiny Hy4 artifact.

Families: any family with ``_load_layer(l)``, ``embed(ids)``, ``final_logits(carry)``, a per-layer call
(``layer_call(l, layer, carry) -> carry``) and ``reference_model(layers)``. Hy4-flash provides them
(``HyV4FlashLayerCalls`` below wraps the family's own per-layer arguments: fresh layer cache, Engram rows).
"""
from __future__ import annotations

import math
import time

import numpy as np

# The family's layers run without the adapter's long-context paths (band attention, DEFAULT 512 blocks on Hy4-flash,
# engages past 512 rows; sparse prefill attention past the indexer's dense length). Windows up to this length run the
# same numerics on both paths; a longer --seq-len refuses rather than scoring a different forward.
MAX_SEQ_LEN = 512


class HyV4FlashLayerCalls:
    """The Hy4-flash family's per-layer forward on a carry {"ids", "h"}, exactly as its run_layer calls the layer
    (fresh ``make_cache()[l]``, the layer's Engram inputs), without the payload taps."""

    def __init__(self, family):
        self.f = family
        self.L = family.L

    def inputs(self, l, carry):
        """The layer's arm-independent inputs for a batch (its Engram rows: read once, shared by every arm)."""
        f = self.f
        tokens = f._check_ids(carry["ids"])
        return f._engram(l, tokens) if l in f.args.engram_layers else None

    def call(self, l, layer, carry, engram=None):
        cache = self.f.model.make_cache()[l]                  # fresh per call, as run_layer
        return {"ids": carry["ids"], "h": layer(carry["h"], cache, engram)}

    def reference_model(self, layers):
        """A resident ordinary Model of the first ``layers`` blocks (the family's parity reference), returned
        instead of run, so the caller can install carve sites on its layers."""
        import dataclasses

        import mlx.core as mx
        f, H, a = self.f, self.f.H, self.f.args
        K = int(layers)
        need = sum(f.tensors[n].end - f.tensors[n].begin for l in range(min(K, a.num_hidden_layers))
                   for n in f._names[l])
        if need > f.reference_gb * 2 ** 30:
            raise SystemExit(f"layer-major: a resident reference of {K} layers needs {need / 2 ** 30:.0f} GiB, over "
                             f"the family's reference_gb {f.reference_gb:g}")
        args = a if K >= a.num_hidden_layers else dataclasses.replace(
            a, num_hidden_layers=K, layer_types=list(a.layer_types[:K]), mlp_layer_types=list(a.mlp_layer_types[:K]),
            engram_layer_ids=[lid for lid in (a.engram_layer_ids or []) if lid - 1 < K])
        ref = H.Model(args)
        weights = dict(f._top_items)
        for l in range(K):
            weights.update(f._layer_weights(l))
        f._quantize(ref, "", weights)
        ref.load_weights(list(weights.items()), strict=True)
        weights.clear()
        ref.bind_engram({i: f.readers[i] for i in args.engram_layers})
        ref.eval()
        mx.eval(ref.parameters())
        return ref


class LayerMajorScorer:
    """``kl_many`` computed layer-major (module docstring). ``widths``: the carve host's per-layer unit counts (the
    keep rows' widths). ``carry_gb``: budget for all arms' carried states of one chunk. ``layers``: walk only the
    first K layers and score their logits (the parity setting); None = every layer."""

    def __init__(self, family, calls, widths, *, moe_units="neuron", bs=4, carry_gb=24.0, layers=None, log=print):
        self.f, self.calls, self.widths = family, calls, list(widths)
        self.moe_units, self.bs, self.carry_gb = moe_units, int(bs), float(carry_gb)
        self.K = int(layers) if layers else calls.L
        self.log = log
        self.peak_live_batches = 0
        self.passes = 0                   # chunk walks (one weight read of K layers each)
        self.layer_seconds = 0.0

    def _rows(self, keep):
        import mlx.core as mx
        return [mx.array(keep[l, :n].astype(np.float32)) for l, n in enumerate(self.widths)]

    def _chunk_windows(self, wins, arms):
        """Windows per chunk so every arm's carry fits carry_gb."""
        if not wins:
            return 1
        a = self.f.args
        T = max(len(w[1]) for w in wins)
        per = arms * T * a.hc_mult * a.hidden_size * 2          # bf16 streams per window and arm
        n = int(self.carry_gb * 2 ** 30 // per)
        return max(self.bs, n - n % self.bs)                     # whole batches: the resident scorer's grouping

    def _walk(self, batches, rows_by_arm):
        """{arm: [carry per batch]} after the first K layers; rows_by_arm[arm] None = the full model."""
        import mlx.core as mx

        from . import units as U
        carries = {arm: [self.f.embed(mx.array([x[1] for x in items])) for items in batches] for arm in rows_by_arm}
        for l in range(self.K):
            t0 = time.time()
            layer = self.f._load_layer(l)
            sites, ctl = U.install([layer], moe_units=self.moe_units)
            if self.widths[l] and sites[0].n != self.widths[l]:
                raise SystemExit(f"layer-major: layer {l} site has {sites[0].n} units, the host {self.widths[l]}")
            try:
                for b in range(len(batches)):
                    engram = self.calls.inputs(l, carries["__full__" if "__full__" in carries else
                                                           next(iter(carries))][b])
                    for arm, rows in rows_by_arm.items():
                        ctl.reset()
                        if rows is not None and sites[0].kind is not None:
                            ctl.core, ctl.sink_layers = [rows[l]], 0
                        carries[arm][b] = self.calls.call(l, layer, carries[arm][b], engram)
                        mx.eval(carries[arm][b]["h"])
                        ctl.reset()
                    del engram
            finally:
                self.f.release_layer(l)
                del layer, sites, ctl
                mx.clear_cache()
            self.layer_seconds += time.time() - t0
        self.passes += 1
        return carries

    def logits(self, carry, T):
        import mlx.core as mx
        if self.K >= self.calls.L:
            lg = self.f.final_logits(carry)
        else:                               # the parity setting: the head on the K-th layer's streams
            m = self.f.model
            lg = m.logits(m.model.norm(m.model.hc_head(carry["h"])))
        return lg[:, : T - 1].astype(mx.float32)

    def kl_many(self, wins, keeps):
        import mlx.core as mx
        if wins and max(len(w[1]) for w in wins) > MAX_SEQ_LEN:
            raise SystemExit(f"layer-major: windows over {MAX_SEQ_LEN} tokens run the adapter's band / sparse "
                             "attention on the resident path, which the family's layers do not")

        from .carving import kl_rows
        names = list(keeps)
        rows_by_arm = {"__full__": None, **{n: self._rows(k) for n, k in keeps.items()}}
        out = {n: np.zeros(len(wins)) for n in names}
        per_chunk = self._chunk_windows(wins, len(rows_by_arm))
        for c0 in range(0, len(wins), per_chunk):
            chunk = list(enumerate(wins))[c0:c0 + per_chunk]
            batches = []
            for i in range(0, len(chunk), self.bs):
                by_len = {}
                for idx, w in chunk[i:i + self.bs]:
                    by_len.setdefault(len(w[1]), []).append((idx, w[1], w[2]))
                batches += list(by_len.values())
            self.peak_live_batches = max(self.peak_live_batches, len(batches))
            carries = self._walk(batches, rows_by_arm)
            for b, items in enumerate(batches):
                T = len(items[0][1])
                full = self.logits(carries["__full__"][b], T)
                mx.eval(full)
                for n in names:
                    k = np.array(kl_rows(full, self.logits(carries[n][b], T)))
                    for j, (idx, _, lf) in enumerate(items):
                        out[n][idx] = float(k[j, max(0, lf - 1):].mean())
                del full
            del carries
            mx.clear_cache()
            self.log(f"layer-major: chunk {c0 // per_chunk + 1} ({len(chunk)} windows, {len(rows_by_arm)} arms) "
                     f"scored; {self.passes} walks, {self.layer_seconds:.0f} s in layers")
        return out

    def kl(self, wins, keep, tag=None):
        return self.kl_many(wins, {"x": keep})["x"]


def parity(scorer, wins, keeps, *, layers, log=print):
    """The layer-major scorer against a resident model of the first ``layers`` blocks carrying the carve's sites
    (``units.install`` on its layers: the resident host's law), on ``wins`` for the full arm and every keep in
    ``keeps``: per arm, max |dKL| over windows, and for the full arm the max relative logit difference and top-1
    agreement. Returns the record; the caller decides the gate."""
    import mlx.core as mx

    from . import units as U
    from .carving import kl_rows
    lm = LayerMajorScorer(scorer.f, scorer.calls, scorer.widths, moe_units=scorer.moe_units, bs=scorer.bs,
                          carry_gb=scorer.carry_gb, layers=layers, log=log)
    got = lm.kl_many(wins, keeps)
    ref = scorer.calls.reference_model(layers)
    sites, ctl = U.install(list(ref.model.layers), moe_units=scorer.moe_units)
    rows = {n: lm._rows(k) for n, k in keeps.items()}
    want = {n: np.zeros(len(wins)) for n in keeps}
    full_rel, top1 = 0.0, []
    try:
        for i in range(0, len(wins), scorer.bs):
            items = wins[i:i + scorer.bs]
            by_len = {}
            for j, w in enumerate(items):
                by_len.setdefault(len(w[1]), []).append((i + j, w[1], w[2]))
            for T, grp in by_len.items():
                ids = mx.array([g[1] for g in grp])
                ctl.reset()
                full = ref(ids)[:, : T - 1].astype(mx.float32)
                # the layer-major full arm on the same windows
                lm_full = lm.logits(lm._walk([grp], {"__full__": None})["__full__"][0], T)
                d = float(mx.abs(full - lm_full).max() / mx.maximum(mx.abs(full).max(), 1e-6))
                full_rel = full_rel if math.isnan(full_rel) else (d if math.isnan(d) else max(full_rel, d))
                top1.append(float((full.argmax(-1) == lm_full.argmax(-1)).astype(mx.float32).mean()))
                for n in keeps:
                    ctl.reset()
                    ctl.core, ctl.sink_layers = [rows[n][l] for l in range(len(sites))], 0
                    k = np.array(kl_rows(full, ref(ids)[:, : T - 1].astype(mx.float32)))
                    ctl.reset()
                    for b, (idx, _, lf) in enumerate(grp):
                        want[n][idx] = float(k[b, max(0, lf - 1):].mean())
    finally:
        del ref
        mx.clear_cache()
    rec = {"layers": int(layers), "windows": len(wins), "full_logits_max_rel": full_rel,
           "full_top1_agreement": float(np.mean(top1)) if top1 else None,
           "arms": {n: {"max_abs_dkl": float(np.max(np.abs(got[n] - want[n]))) if len(wins) else None,
                        "max_rel_dkl": float(np.max(np.abs(got[n] - want[n]) / np.maximum(want[n], 1e-12)))
                        if len(wins) else None,
                        "mean_kl_layer_major": float(got[n].mean()) if len(wins) else None,
                        "per_window": {"layer_major": got[n].tolist(), "resident": want[n].tolist()},
                        "mean_kl_resident": float(want[n].mean()) if len(wins) else None} for n in keeps}}
    log(f"layer-major parity over {layers} layers, {len(wins)} windows: full rel {full_rel:.2e}, "
        f"top-1 {rec['full_top1_agreement']}, " + ", ".join(f"{n} |dKL| {v['max_abs_dkl']:.2e}"
                                                            for n, v in rec["arms"].items()))
    return rec
