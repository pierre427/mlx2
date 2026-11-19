#!/usr/bin/env python3
"""amoeba carve: one resumable pipeline that carves a core (and tiers, residues, boxes) out of any donor.

    python -m research.amoeba.carve --model DIR --out RUN --stages inspect,mean,rank,bisect,sink,atlas,export

Stages (run in this order; --resume skips a stage whose receipt is in RUN/carve.json):
  inspect  unit sites from the module structure, forward-path checks against the adapter's own
           logits, core=ones parity, per-layer eval (fused) vs reference check (--grad-mode layer_ref),
           layer-wise gradient check (fail closed to whole-graph backprop)
  mean     mean unit signal on the mean split (mean.npz: ``mean``)
  rank     straight-through e2e carve per ablation (ranking_<abl>.npz: ``Z``, ``mean``), the 27B
           recipe; over the free units only: --fixed keeps a lower tier, --pool limits the candidates
  fisher   capture per-sequence mask gradients (fisher/U_L*.npy), then carve from them with no
           forward passes (ranking_fisher_<abl>.npz); --init-from fisher warm-starts rank. Opt-in,
           unmeasured: see fisher.py. Runs before rank so rank can warm-start from it
  bisect   per ranking: Z>0 core; cut to each --targets English cost (held) and each --size-targets
           count of free units; random same-counts core and random ranking controls; export core
           cut on the spare split (export_core.npz: ``core``, ``Z``, ``frac``)
  sink     position-0 scan on the export core (rule depths, dense stand-ins, add-back); the chosen
           stand-in is recorded, --stand-in none is the 27B decision
  atlas    per-sequence peak |unit - mean| capture in the format scripts/amoeba27b_atlas.py reads
           (atlas/peak_L*.npy, atlas_stats.npz, meta.json); clustering is CPU work for later
  export   dense masked checkpoint (dropped neurons' up rows zeroed / dropped experts' down
           weights zeroed, so the drop rule is exact), verified on the saved files

--moe-units neuron carves MoE sites in (routed expert e, intermediate neuron f) units, flat id e*F + f
(units.py; ported from scripts/probe_amoeba_unit_box.py@16a1bbe88). Every stage takes the [L, E*F]
layout (carve.json shape.widths; npz rows padded to the widest site as before) except atlas, which
refuses. MoE neuron sites use the drop rule in every ablation (mean only applies to dense sites).

Every number names the checkpoint (<artifact>@<fingerprint>) and the mlx2 revision. Settings are
chosen on fit/spare splits and reported on held. Carving = choosing units (no weight changes).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from . import corpus as C
from . import evals as E
from . import fisher as F
from . import ops as O
from . import tasks as T
from .carving import CleanStop, kl_rows, masked_kl, random_same_counts, topk_mask, train_mask
from .host import Host
from .units import site_summary

STAGES = ["inspect", "mean", "fisher", "rank", "bisect", "sink", "atlas", "export"]
SCHEMA = "mlx2.amoeba-carve.v1"


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="inspect,mean,rank,bisect,sink,atlas")
    ap.add_argument("--resume", action="store_true", help="reuse RUN and skip stages already in carve.json")
    ap.add_argument("--corpus", default="legacy", help="legacy | jsonl:PATH")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--windows", type=int, default=0,
                    help="windowed jsonl loader: seeded random-offset windows per document (0 = document starts)")
    ap.add_argument("--window-seed", type=int, default=0)
    ap.add_argument("--val-windows", type=int, default=0,
                    help="windows per --val-file document (0 = --windows); a thin val split picks a noisy export cut")
    ap.add_argument("--strip-prefix", type=float, default=0.0,
                    help="strip a line/sentence-aligned prefix shared by more than this share of a domain's rows")
    ap.add_argument("--held-file", default="", help="fixed held-out jsonl: the held split of every domain")
    ap.add_argument("--held-len", type=int, default=0, help="held-file rows cropped to this many tokens (0 = seq-len)")
    ap.add_argument("--val-file", default="",
                    help="validation jsonl of whole documents (neither carve nor held documents): the export cut is "
                         "chosen on it instead of the carve documents' spare windows")
    ap.add_argument("--bisect-subsample", type=float, default=1.0,
                    help="bisect steps score a fixed seeded share of each domain's held/val sequences; the final cut "
                         "is scored on the full set (1.0 = every sequence, the original behaviour)")
    ap.add_argument("--bisect-tol", type=float, default=0.0,
                    help="stop a bisect once its kept-fraction bracket is narrower than this (0 = all --bisect-iters)")
    ap.add_argument("--code-cut", action="store_true",
                    help="windowed loader: code domains start at their first code line (drop the English task statement)")
    ap.add_argument("--pool-domains", action="store_true",
                    help="held / val means pooled over sequences (weighted by each domain's row count) instead of the "
                         "unweighted mean over domains (FN boxes: 1 greedy + 3 T=0.7 count 1:3)")
    ap.add_argument("--keep-prompt", type=int, default=32,
                    help="on-policy rows (a 'prompt' field): also score the last N prompt tokens")
    ap.add_argument("--n-mean", type=int, default=240)
    ap.add_argument("--n-fit", type=int, default=900)
    ap.add_argument("--n-held", type=int, default=60)
    ap.add_argument("--ablations", default="zero,mean")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--init-from", default="", help="'fisher' or NPZ[:key]: warm-start rank's Z")
    ap.add_argument("--init-raw", action="store_true",
                    help="with --init-from NPZ: use its Z as given (clipped to +-3) instead of rescaling to the z0=3 "
                         "scale, so a seed (e.g. map clusters of a topic at +3, every other unit at +1) survives")
    ap.add_argument("--extra-ranking", action="append", default=[],
                    help="NAME=NPZ: import an earlier ranking's Z (e.g. e2e150_zero=core-696591553/ranking_zero.npz) "
                         "so bisect scores it beside this run's rankings on the same splits and controls")
    ap.add_argument("--fixed", default="", help="NPZ[:key] mask kept fixed (a lower tier, e.g. T1 export_core.npz:core)")
    ap.add_argument("--pool", default="", help="NPZ[:key] mask of units allowed to be carved")
    ap.add_argument("--fisher-seqs", type=int, default=512)
    ap.add_argument("--fisher-lams", default="1.0")
    ap.add_argument("--fisher-alpha", type=float, default=0.0)
    ap.add_argument("--fisher-steps", type=int, default=300)
    ap.add_argument("--targets", default="0.18,0.05", help="English-cost targets (nats, held)")
    ap.add_argument("--size-targets", default="", help="fractions of ALL units to add beyond --fixed (T3: 0.01,0.02,0.03)")
    ap.add_argument("--export-target", type=float, default=0.18)
    ap.add_argument("--export-ablation", default="zero")
    ap.add_argument("--bisect-iters", type=int, default=9)
    ap.add_argument("--stand-in", default="none", help="none | chosen (use the sink stage's choice)")
    ap.add_argument("--atlas-seqs", type=int, default=2048)
    ap.add_argument("--atlas-dir", default="")
    ap.add_argument("--export-dir", default="", help="where the dense masked checkpoint goes (default under ~/mlx-models)")
    ap.add_argument("--memory-gb", type=float, default=56)
    ap.add_argument("--cache-gb", type=float, default=4)
    ap.add_argument("--assert-ple-offload", action="store_true",
                    help="Flash-Next (qwen4): fail closed unless the adapter installed file-backed PLE from "
                         "<model>/ple_rows.bin (Pierre 2026-11-15: 'Use ple offload when loading qwen fn to save ram')")
    ap.add_argument("--watchdog-gb", type=float, default=80, help="own physical footprint cap (Pierre 2026-11-14: 80 GB)")
    ap.add_argument("--need-gb", type=float, default=45)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0,
                    help="abort when swap grows more than this over its level at start (coordinator's waiver terms)")
    ap.add_argument("--swap-waiver-note", default="", help="why --swap-growth-gb differs from 2 (recorded)")
    ap.add_argument("--mem-policy", default="auto:waived:60:80:15")
    ap.add_argument("--train-peak-cap-gb", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-expert-units", action="store_true",
                    help="--fixed is an expert-unit mask: in --moe-units neuron it is expanded to every neuron of each "
                         "kept expert (the same computation: exact on CPU; on Metal neuron mode recomputes the kept "
                         "experts piecewise, so equal to rounding and near-tie routing flips; do not compare T2-alone "
                         "costs across unit modes)")
    ap.add_argument("--no-admit", action="store_true",
                    help="skip the memcheck admission (a box batch: the model is already loaded in this process)")
    ap.add_argument("--stop-at", type=float, default=None,
                    help="clean stop (epoch seconds): no stage starts after it, rank checkpoints and stops after "
                         "the step that passes it; exit code 75; rerun the same command with --resume")
    ap.add_argument("--allow-whole-graph", action="store_true",
                    help="if the layer-wise gradient check fails, carve with whole-graph backprop instead of "
                         "refusing (all layers' tapes at once: ~110 GB on the 27B, over the 80 GB watchdog)")
    ap.add_argument("--grad-tol", type=float, default=0.15,
                    help="layer-wise vs whole-graph relative L2 gate; the cosine gates are the real guard "
                         "(27B, 64 layers: 0.081 measured noise)")
    ap.add_argument("--dir-grad-tol", type=float, default=0.10,
                    help="layer_ref directional gate: max median relative error of g.d vs central differences")
    ap.add_argument("--dir-grad-min-corr", type=float, default=0.99,
                    help="layer_ref directional gate: min Pearson correlation of g.d vs central differences")
    ap.add_argument("--dir-grad-k", type=int, default=8, help="layer_ref directional gate: directions (2 sparse)")
    ap.add_argument("--dir-grad-eps", default=None,
                    help="layer_ref directional gate: comma-separated steps (default 2^-7,2^-6,2^-5,2^-3; "
                         "powers of two are exact in bf16)")
    ap.add_argument("--dir-grad-min-clean", type=int, default=None,
                    help="layer_ref directional gate: min directions free of +- routing flips at both gate eps "
                         "(default max(4, k/2))")
    ap.add_argument("--dir-grad-norm-ratio", default=None,
                    help="layer_ref directional gate: LO,HI bounds on |g.d| / |fd| over the gated directions (off by "
                         "default)")
    ap.add_argument("--dir-grad-min-pair-corr", type=float, default=None,
                    help="layer_ref directional gate: a gate pair is usable only if its two FD vectors correlate "
                         ">= this (off by default)")
    ap.add_argument("--dir-grad-note", default="",
                    help="layer_ref directional gate: why non-default thresholds are used (recorded in the receipt)")
    ap.add_argument("--dir-grad-same-fn-tol", type=float, default=1e-3,
                    help="layer_ref directional gate: max relative loss gap between the gradient's forward "
                         "and the finite differences' forward (both must be the masked fused forward)")
    ap.add_argument("--grad-mode", default="layer_ref", choices=["layer_ref", "whole_train"],
                    help="layer_ref: eval-mode (fused) forwards, only the backpropagated layer on its reference "
                         "path (gated by inspect's per-layer check); whole_train: the whole model in training "
                         "mode for the whole gradient computation (the old behaviour)")
    ap.add_argument("--layer-ref-tol", type=float, default=2e-2,
                    help="per-layer relative L2 between a layer's eval (fused) and reference outputs (bf16 noise)")
    ap.add_argument("--moe-units", default="expert", choices=["expert", "neuron"],
                    help="MoE unit granularity: whole routed experts (default) or (expert, intermediate neuron) "
                         "pairs, E*F units per MoE layer (units.py)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny splits and steps: a code test, never a result")
    return ap.parse_args(argv)


def expand_expert_mask(a, host):
    """An expert-unit mask ([L, max over sites of (E for MoE, n for dense)]) in this neuron-unit host's layout: every
    neuron of each kept expert kept (units.expert_to_neuron per MoE site), dense rows unchanged, padded to the widest
    site (FN pilot, coordinator 2026-11-16: T2 was carved in expert units)."""
    from .units import expert_to_neuron
    rows = []
    for l, s in enumerate(host.sites):
        if s.is_moe and s.units == "neuron":
            rows.append(expert_to_neuron(a[l, :s.E], s.F))
        else:
            rows.append(a[l, :host.widths[l]])
    w = max(host.widths)
    out = np.zeros((host.L, w), np.float32)
    for l, r in enumerate(rows):
        if r.shape[0] != host.widths[l]:
            raise SystemExit(f"expert mask row {l}: {r.shape[0]} units after expansion, the site has {host.widths[l]}")
        out[l, :r.shape[0]] = r
    return out


def load_mask(spec, host, expert_units=False):
    import mlx.core as mx
    if not spec:
        return None
    path, _, key = spec.partition(":")
    z = np.load(Path(path).expanduser())
    a = z[key or z.files[0]].astype(np.float32)
    want = (host.L, max(host.widths))
    if expert_units and a.shape != want and getattr(host, "moe_units", "expert") == "neuron":
        a = expand_expert_mask(a, host)
    if a.shape != want:
        raise SystemExit(f"mask {spec}: shape {a.shape} does not match this donor's [L, max width] {want} "
                         f"(--moe-units {host.moe_units}; an expert mask needs units.expert_to_neuron first)")
    if host.N is not None:
        return mx.array(a)
    return [mx.array(a[l, :n]) for l, n in enumerate(host.widths)]


def as_np(mask, host):
    """Mask (array or ragged list) -> padded [L, Nmax] float32 (npz layout; widths in carve.json)."""
    if mask is None:
        return None
    if host.N is not None or (not isinstance(mask, (list, tuple)) and np.ndim(np.asarray(mask)) == 2):
        return np.array(mask, dtype=np.float32)
    w = max(host.widths)
    out = np.zeros((host.L, w), np.float32)
    for l, r in enumerate(mask):
        out[l, :host.widths[l]] = np.array(r)
    return out


def from_np(a, host):
    import mlx.core as mx
    if host.N is not None:
        return mx.array(a)
    return [mx.array(a[l, :n]) for l, n in enumerate(host.widths)]


def main(argv=None, host_factory=None):
    """host_factory(log) -> Host replaces adapter loading (tests)."""
    args = parse(argv)
    out = Path(args.out)
    if args.resume and out.exists():
        rep = json.loads((out / "carve.json").read_text()) if (out / "carve.json").exists() else {}
    else:
        out = O.fresh_dir(out)
        rep = {}
    log = O.make_log(out)
    mem = (O.admit(args.need_gb, args.mem_policy, log) if not (args.smoke or args.no_admit) else
           {"policy": "smoke: not admitted" if args.smoke else "batch: model already loaded in this process"})
    O.footprint_watchdog(args.watchdog_gb, log, swap_growth_gb=args.swap_growth_gb)
    if args.swap_growth_gb != 2.0:                # a waived guard is recorded with its reason
        rep["watchdog_waiver"] = {"swap_growth_gb": args.swap_growth_gb, "reason": args.swap_waiver_note}
    O.setup_mlx(args.memory_gb, args.cache_gb)
    import mlx.core as mx
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)
    stages = [s for s in STAGES if s in args.stages.split(",")]
    unknown = set(args.stages.split(",")) - set(STAGES)
    if unknown:
        raise SystemExit(f"unknown stages {sorted(unknown)}")
    t_start = time.time()
    C.LOSS_FROM.clear()                       # per-run registries (a batch runs many carves in one process)
    host = host_factory(log) if host_factory else Host(args.model, log=log, moe_units=args.moe_units)
    if getattr(host, "moe_units", "expert") != args.moe_units:
        raise SystemExit(f"host installed {getattr(host, 'moe_units', 'expert')} MoE units, --moe-units {args.moe_units}")
    host.grad_mode = args.grad_mode
    if args.assert_ple_offload:
        # Same check as scripts/probe_flash_next_packed_ast.py: the adapter path must have installed the
        # file-backed PLE tables (MLX_QWEN4_PLE_NVME set by configure_environment); never a bypassing load.
        import os
        sidecar = Path(args.model).expanduser().resolve() / "ple_rows.bin"      # the adapter resolves its path
        tables = getattr(getattr(host, "adapter", None), "_tables", None)
        env = os.environ.get("MLX_QWEN4_PLE_NVME")
        if not tables or not env or Path(env).expanduser().resolve() != sidecar:
            raise SystemExit(f"--assert-ple-offload: PLE offload is not active (tables={bool(tables)}, "
                             f"MLX_QWEN4_PLE_NVME={os.environ.get('MLX_QWEN4_PLE_NVME')!r}, expected {str(sidecar)!r})")
        rep["memory"] = {"ple_offload": True, "sidecar_bytes": sidecar.stat().st_size,
                         "watchdog_gb": args.watchdog_gb, "memory_gb": args.memory_gb}
        log(f"PLE offload active: {sidecar} ({sidecar.stat().st_size / 2**30:.1f} GiB on NVMe)")
    L = host.L
    done = set(rep.get("stages_done", []))
    ident = {k: getattr(args, k) for k in ("corpus", "seq_len", "windows", "window_seed", "strip_prefix", "held_file",
                                          "val_file", "val_windows", "code_cut", "fixed_expert_units",
                                          "held_len", "keep_prompt", "n_mean", "n_fit", "n_held", "ablations", "lam",
                                          "steps", "lr", "bs", "init_from", "init_raw", "fixed", "pool", "fisher_seqs",
                                          "fisher_lams", "fisher_alpha", "fisher_steps", "seed", "smoke")}
    ident["fingerprint"] = host.fingerprint
    # gradients depend on the mode (and, in layer_ref, on the per-layer gate); a run from before the
    # modes existed carved with the whole model in training mode
    ident["grad_mode"] = args.grad_mode
    ident["layer_ref_tol"] = args.layer_ref_tol if args.grad_mode == "layer_ref" else None
    ident["moe_units"] = args.moe_units
    if rep.get("identity"):
        rep["identity"].setdefault("grad_mode", "whole_train")
        rep["identity"].setdefault("layer_ref_tol", None)
        rep["identity"].setdefault("moe_units", "expert")
        for k_, v_ in (("windows", 0), ("window_seed", 0), ("strip_prefix", 0.0), ("held_file", ""), ("held_len", 0),
                       ("keep_prompt", 32), ("val_file", ""), ("val_windows", 0), ("code_cut", False),
                       ("fixed_expert_units", False), ("init_raw", False)):
            rep["identity"].setdefault(k_, v_)      # runs from before the windowed loader
    if rep.get("identity") and rep["identity"] != ident:
        diff = {k: (rep["identity"].get(k), v) for k, v in ident.items() if rep["identity"].get(k) != v}
        raise SystemExit(f"--resume: run identity differs from the earlier run {diff}; use a fresh --out")
    rep["identity"] = ident
    rep.update({"schema": SCHEMA, "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
                "adapter": host.adapter_name, "mlx2_source_revision": O.git_rev(),
                "args": vars(args), "memcheck": mem,
                "shape": {"layers": L, "intermediate": host.N, "widths": host.widths, "hidden": host.d,
                          "units": int(sum(host.widths)), "moe_units": args.moe_units}})
    nsites = [s_ for s_ in host.sites if s_.is_moe and s_.units == "neuron"]
    if nsites:
        from .units import NEURON_PROVENANCE
        rep["shape"]["moe_layout"] = {str(s_.layer): {"E": s_.E, "F": s_.F} for s_ in nsites}
        rep["shape"]["unit_id"] = "MoE sites: expert * F + neuron; dense sites: neuron"
        rep["neuron_units"] = {"provenance": NEURON_PROVENANCE, "memory": _neuron_memory(host, args, log)}
    rep.setdefault("runs", []).append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": stages,
                                       "mlx2_source_revision": O.git_rev()})

    def save():
        rep["wall_s"] = round(time.time() - t_start, 1)
        rep["mlx_peak_gb"] = O.peak_gb()
        rep["watch"] = dict(O.WATCH)
        rep["stages_done"] = sorted(done)
        O.write_json(out / "carve.json", rep)
        if host.N is not None and all(s.kind == "dense" for s in host.sites):
            O.write_json(out / "core27b.json", {**rep, "schema": "mlx2.amoeba27b-core.v1",
                                                "carve_schema": SCHEMA})   # read by the 27B scripts

    def skip(stage):
        if stage in done and args.resume:
            log(f"stage {stage}: done in an earlier run, skipped")
            return True
        if args.stop_at is not None and time.time() >= args.stop_at:
            stop(f"--stop-at deadline before stage {stage}")
        log(f"stage {stage}")

    def stop(msg):
        rep["clean_stop"] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "why": msg}
        save()
        log(f"clean stop: {msg}; rerun with --resume")
        raise CleanStop(msg)
        return False

    sm = args.smoke
    split, crec = C.load(args.corpus, host, seq_len=args.seq_len, n_mean=12 if sm else args.n_mean,
                         n_fit=12 if sm else args.n_fit, n_held=4 if sm else args.n_held, seed=args.seed, log=log,
                         windows=args.windows, window_seed=args.window_seed, strip_prefix=args.strip_prefix,
                         held_file=args.held_file or None, held_len=args.held_len, keep_prompt=args.keep_prompt,
                         val_file=args.val_file or None, code_cut=args.code_cut,
                         val_windows=args.val_windows)
    if crec.get("offsets"):                       # window offsets: recorded beside the receipt (seeded, reproducible)
        O.write_json(out / "windows.json", {"offsets": crec.pop("offsets")})
    rep["corpus"] = crec                         # (crec gains "inputs" below, once the fit order is known)
    domains = C.domains(split)
    tasks, tsplits = T.build_tasks(host, seed=args.seed)
    rep["tasks"] = {k: {"n": len(v), "fit": len(tsplits[k]["fit"]), "held": len(tsplits[k]["held"])} for k, v in tasks.items()}
    caps, caps_idx = tasks["capitals"], tsplits["capitals"]["held"][: 8 if sm else None]
    fit = [q for dn in domains for q in split[dn]["fit"]]
    random.Random(args.seed).shuffle(fit)
    if C.SEQ_SRC:                                 # every sequence this run used and its exact source row (prompt set)
        n_trained = (3 if sm else args.steps) * args.bs if {"rank", "fisher"} & set(stages) else 0
        trained = {tuple(q) for q in (fit + fit)[:n_trained]} if fit else set()   # batches(step) walks fit in order
        use = {"held": "test", "val": "export_select", "mean": "mean_stats", "fit": "carve",
               "spare": "unused" if args.val_file else "export_select"}
        with open(out / "inputs.jsonl", "w") as fh:
            parts = [(dn, k, v) for dn, sp in split.items() for k, v in sp.items()] + \
                    [(dn, "val", v) for dn, v in C.VAL.items()]
            for dn, k, seqs in parts:
                for q in seqs:
                    fh.write(json.dumps({"domain": dn, "split": k, "use": use.get(k, k),
                                         "trained": (tuple(q) in trained) if k == "fit" else None,
                                         "tokens": len(q), "loss_from": C.loss_from(q),
                                         "sources": C.SEQ_SRC.get(tuple(q), [])}) + "\n")
        crec["inputs"] = {"file": "inputs.jsonl", "corpus": args.corpus, "held_file": args.held_file or None,
                          "val_file": args.val_file or None, "trained_sequences": n_trained,
                          "note": "rows: one per sequence (split, use: test / export_select / mean_stats / carve / "
                                  "unused, trained flag for fit, loss mask) with its source rows (file corpus/held/val, "
                                  "row key = probe_id, doc id or gen:<prompt_key>:<mode>:<seed>, token offset into the "
                                  "text after text_cut_chars, window length); scripts/amoeba_prompt_set.py joins them"}

    if crec.get("val_file"):                      # export point on whole validation documents (coordinator 2026-11-16)
        spare = {dn: {"spare": v} for dn, v in C.VAL.items() if v}
    else:
        spare = {dn: {"spare": split[dn]["spare"][:(4 if sm else 60)]} for dn in domains}
    def _sub(d):                                  # fixed seeded subsample for bisect steps
        if args.bisect_subsample >= 1.0:
            return d
        r = random.Random(args.seed + 99)
        return {dn: {k: (r.sample(v, max(1, int(round(len(v) * args.bisect_subsample)))) if v else v)
                     for k, v in parts.items()} for dn, parts in d.items()}
    split_sub, spare_sub = _sub({dn: {"held": split[dn]["held"]} for dn in split}), _sub(spare)
    clean_held_sub, clean_spare_sub = {}, {}
    clean_held, clean_spare = {}, {}
    fixed = load_mask(args.fixed, host, expert_units=args.fixed_expert_units)
    pool = load_mask(args.pool, host)
    zeros = host.zeros()
    fixed_or_zero = fixed if fixed is not None else zeros
    rep["tiers"] = {"fixed": args.fixed or None, "pool": args.pool or None,
                    "fixed_frac": float(np.mean(as_np(fixed_or_zero, host))) if fixed is not None else 0.0}

    def held(core, mean=None, sink=L, dense=()):
        return E.english_cost(host, split, clean_held, E.configure(host, core, E.dense_mean(host, mean), sink, dense),
                              pooled=args.pool_domains)

    def fit_eval(core, mean=None, sink=L, dense=()):
        return E.english_cost(host, spare, clean_spare, E.configure(host, core, E.dense_mean(host, mean), sink, dense),
                              part="spare", pooled=args.pool_domains)

    def held_sub(core, mean=None, sink=L, dense=()):
        if args.bisect_subsample >= 1.0:
            return held(core, mean, sink, dense)
        return E.english_cost(host, split_sub, clean_held_sub, E.configure(host, core, E.dense_mean(host, mean), sink, dense),
                              pooled=args.pool_domains)

    def fit_eval_sub(core, mean=None, sink=L, dense=()):
        if args.bisect_subsample >= 1.0:
            return fit_eval(core, mean, sink, dense)
        return E.english_cost(host, spare_sub, clean_spare_sub,
                              E.configure(host, core, E.dense_mean(host, mean), sink, dense), part="spare",
                              pooled=args.pool_domains)

    def capitals(core, mean=None, sink=L, dense=()):
        return T.task_eval(host, caps, caps_idx, E.configure(host, core, E.dense_mean(host, mean), sink, dense))[0]

    def layer_ref_gate(h, ids, refuse):
        """layer_ref backprops layer l on its reference path from residuals the fused path made: the
        two paths must agree per layer to bf16 noise, or the gradients are of a different function."""
        if h.grad_mode != "layer_ref":
            return
        r = h.check_layer_reference(ids, tol=args.layer_ref_tol)
        if refuse and not r["ok"]:
            worst = {x["layer"]: x for x in r["layers"]}
            raise SystemExit(
                "inspect: eval (fused) and reference outputs differ beyond --layer-ref-tol "
                f"{args.layer_ref_tol} at layer(s) {r['failing_layers']} (worst {r['worst_layer']}: rel L2 "
                f"{r['worst_rel_l2']:.3g}, went eager {worst[r['worst_layer']]['went_eager']}); refusing the "
                f"layer_ref carve (pass --grad-mode whole_train, or --layer-ref-tol after measuring)")

    def grad_gate(h, ids, ref_ids):
        backprop = bool({"rank", "fisher"} & set(stages))
        layer_ref_gate(h, ref_ids, refuse=backprop)
        if not backprop:
            return                                   # no stage of this run backprops
        # layer_ref: the derivative of the masked fused forward (the function carved) is checked
        # against central differences on it; whole_train: layer-wise vs whole-graph gradients
        if h.grad_mode == "layer_ref":
            if args.dir_grad_note:
                h.paths["dir_grad_note"] = args.dir_grad_note
            ok = h.check_directional_grad(
                ids, k=args.dir_grad_k, tol=args.dir_grad_tol, min_corr=args.dir_grad_min_corr,
                same_fn_tol=args.dir_grad_same_fn_tol, min_clean=args.dir_grad_min_clean,
                min_pair_corr=args.dir_grad_min_pair_corr,
                norm_ratio=tuple(float(x) for x in args.dir_grad_norm_ratio.split(",")) if args.dir_grad_norm_ratio else None,
                eps=tuple(float(x) for x in args.dir_grad_eps.split(",")) if args.dir_grad_eps else None)
        else:
            ok = h.check_layerwise_grad(ids, tol=args.grad_tol)
        if ok or args.allow_whole_graph:
            return
        why = h.paths.get("layerwise_grad")
        what = "the layer replay path is disabled" if isinstance(why, str) else \
            (f"layer-wise gradients fail the directional-derivative check on the fused forward "
             f"({why.get('fails') if isinstance(why, dict) else why})"
             if h.grad_mode == "layer_ref" else f"layer-wise gradients do not match whole-graph ones ({why})")
        rep["inspect"] = {"refused": (why.get("fails") if isinstance(why, dict) and "fails" in why else
                                      "gate failed" if isinstance(why, dict) else why),
                          "sites": site_summary(h.sites), "paths": h.paths}
        save()                                       # the refusal's receipt (gate values) is evidence
        knobs = ("--dir-grad-tol / --dir-grad-min-corr / --dir-grad-eps / --dir-grad-min-clean / --dir-grad-same-fn-tol"
                 if h.grad_mode == "layer_ref" else "--grad-tol")
        raise SystemExit(f"inspect: {what}; refusing the whole-graph fallback "
                         f"(pass --allow-whole-graph for a donor small enough for it, or {knobs} "
                         f"after measuring the noise at this depth)")

    # ------------------------------------------------------------------ inspect
    if "inspect" in stages and not skip("inspect"):
        ids = mx.array(fit[:2])
        host.check_paths(ids)
        par = host.check_ones_parity(ids)
        grad_gate(host, mx.array(fit[:1])[:, :16], ids)       # layer check at the carve's sequence length
        rep["inspect"] = {"sites": site_summary(host.sites), "paths": host.paths, "ones_parity": par}
        dense = all(s.kind == "dense" for s in host.sites if s.kind)
        if dense and par["max_abs"] != 0.0:
            raise SystemExit(f"inspect: core=ones changed dense logits by {par['max_abs']} (must be exact)")
        if not dense and par["top1_agree"] < 0.98:
            rep["inspect"]["refused"] = f"MoE re-implementation top-1 agreement {par['top1_agree']} < 0.98"
            save()
            raise SystemExit(f"inspect: MoE re-implementation top-1 agreement {par['top1_agree']} < 0.98")
        done.add("inspect")
        save()
    else:
        host.check_paths(mx.array(fit[:2]))
        grad_gate(host, mx.array(fit[:1])[:, :16], mx.array(fit[:2]))

    # ------------------------------------------------------------------ mean
    if "mean" in stages and not skip("mean"):
        t0 = time.time()
        mean = E.mean_acts(host, [q for dn in domains for q in split[dn]["mean"]])
        O.save_npz(out / "mean.npz", mean=as_np(mean, host))
        rep["mean"] = {"s": round(time.time() - t0, 1)}
        done.add("mean")
        save()
    mean = from_np(np.load(out / "mean.npz")["mean"], host) if (out / "mean.npz").exists() else None

    def mean_for(abl):
        if abl == "mean" and mean is None:
            raise SystemExit("mean ablation needs the mean stage")
        return E.dense_mean(host, mean) if abl == "mean" else None

    def batches(step):
        i = (step * args.bs) % len(fit)
        return [mx.array((fit + fit)[i:i + args.bs])]

    # ------------------------------------------------------------------ fisher
    if "fisher" in stages and not skip("fisher"):
        fdir = out / "fisher"
        frep = rep.setdefault("fisher", {})
        for abl in args.ablations.split(","):
            udir = fdir / abl
            if not (udir / "U_L00.npy").exists():
                seqs = fit[: 4 if sm else args.fisher_seqs]
                frep.setdefault("capture_disk_bytes", {})[abl] = int(len(seqs) * sum(host.widths) * 4)
                log(f"fisher capture {abl}: {len(seqs)} vectors x {sum(host.widths)} units = "
                    f"{len(seqs) * sum(host.widths) * 4 / 2**30:.2f} GiB of float32 U on disk")
                frep.setdefault("capture", {})[abl] = F.capture(host, seqs, udir, mean=mean_for(abl), seed=args.seed, log=log)
                save()
            Us = F.load(udir, L)
            fx = [np.array(r) for r in (fixed_or_zero if host.N is None else list(fixed_or_zero))]
            pl = None if pool is None else [np.array(r) for r in (pool if host.N is None else list(pool))]
            for lam in [float(x) for x in args.fisher_lams.split(",")]:
                Zs, crep = F.carve(Us, fx, pl, lam, args.seq_len - 1, steps=40 if sm else args.fisher_steps,
                                   alpha=args.fisher_alpha, seed=args.seed, log=log)
                Z = np.stack(Zs) if host.N is not None else as_np(Zs, host)
                tag = f"fisher_{abl}" + ("" if len(args.fisher_lams.split(",")) == 1 else f"_lam{lam:g}")
                O.save_npz(out / f"ranking_{tag}.npz", Z=Z, mean=as_np(mean, host) if mean is not None else np.zeros(1))
                frep.setdefault("carve", {})[tag] = crep
                save()
        done.add("fisher")
        save()

    # ------------------------------------------------------------------ rank (e2e)
    if "rank" in stages and not skip("rank"):
        for abl in args.ablations.split(","):
            p = out / f"ranking_{abl}.npz"
            if p.exists():
                continue
            mu = mean_for(abl)
            log(f"carving the {abl}-ablation core e2e: lam={args.lam} steps={args.steps} lr={args.lr} bs={args.bs} "
                f"({'layer-wise' if host.layerwise_ok else 'whole-graph'} backprop)")

            def core_grad(m, ids, mu=mu):
                host.train_mode(False)
                host.ctl.reset()
                pos = list(range(1, ids.shape[1]))
                lc = host.logits_at(ids, pos)
                mx.eval(lc)
                host.train_mode(True)
                # loss masks (on-policy rows: continuation + last prompt tokens): position p predicts token p + 1
                rows = np.array(ids).tolist()
                wm = np.array([[1.0 if p + 1 >= max(1, C.loss_from(q)) else 0.0 for p in pos] for q in rows], np.float32)
                wmx = None if wm.all() else mx.array(wm)      # all scored: exactly the unmasked mean
                loss, g = host.value_and_grad(
                    ids, m, lambda hid: masked_kl(lc, host.head(hid[:, mx.array(pos)]), wmx), mean=mu)
                return loss, (g["core"] if isinstance(g, dict) else g)
            Z0 = _init_z(args.init_from, abl, out, host, raw=args.init_raw)
            fx = fixed_or_zero
            if pool is not None:
                # units outside the pool are frozen: kept if --fixed keeps them, dropped otherwise.
                # train_mask freezes "fixed" units at 1, so frozen-dropped units get Z = -inf instead.
                Z0 = _freeze_outside(Z0, pool, fixed, host)
            t0 = time.time()
            grad = core_grad
            if host.N is None:
                # ragged sites: carve on the padded [L, max width] layout; padding is fixed (never free)
                pad = mx.array(1.0 - _valid(host))
                fx = mx.maximum(mx.array(as_np(fx, host)), pad)
                Z0 = mx.array(as_np(Z0, host))

                def grad(m, ids, _g=core_grad):
                    loss, g = _g(from_np(np.array(m), host), ids)
                    return loss, mx.array(as_np(g, host))
            try:
                Z = train_mask(Z0, fx, args.lam, 3 if sm else args.steps, args.lr, grad, batches, log,
                               f"core-{abl}", peak_cap_gb=args.train_peak_cap_gb,
                               ckpt=out / f"ckpt_rank_{abl}.npz", deadline=args.stop_at)
            except CleanStop as e:
                host.train_mode(False)
                stop(e.msg)
            host.train_mode(False)
            O.save_npz(p, Z=as_np(Z, host), mean=as_np(mean, host) if mean is not None else np.zeros(1))
            rep.pop("clean_stop", None)
            rep.setdefault("rank", {})[abl] = {"train_s": round(time.time() - t0, 1), "init_from": args.init_from or "z0=3",
                                               "backprop": "layerwise" if host.layerwise_ok else "whole-graph"}
            save()
        done.add("rank")
        save()

    for spec in args.extra_ranking:
        name, _, src = spec.partition("=")
        dst = out / f"ranking_{name}.npz"
        if not dst.exists():
            z = np.load(Path(src).expanduser())
            Z = z["Z"].astype(np.float32)
            if Z.shape != as_np(host.zeros(), host).shape:
                raise SystemExit(f"--extra-ranking {name}: shape {Z.shape} does not match this donor")
            O.save_npz(dst, Z=Z, mean=z["mean"] if "mean" in z.files else np.zeros(1))
            rep.setdefault("imported_rankings", {})[name] = {"source": str(Path(src).expanduser()),
                                                            "sha256_12": C.sha12(Path(src).expanduser())}
            save()
    rankings = {}
    for p in sorted(out.glob("ranking_*.npz")):
        rankings[p.stem[len("ranking_"):]] = from_np(np.load(p)["Z"], host)

    # ------------------------------------------------------------------ bisect
    g_seed = args.seed + 1234
    if "bisect" in stages and not skip("bisect"):
        rnd = np.random.default_rng(g_seed)
        rnd_score = from_np(rnd.random((L, max(host.widths))).astype(np.float32), host)
        rep["full_capitals_acc"] = capitals(None)
        rep.setdefault("bisect", {})
        if fixed is not None and "fixed_only_held" not in rep:
            rep["fixed_only_held"] = held(fixed)          # the lower tier alone: the gap a box/tier closes
        fx_np = as_np(fixed_or_zero, host)

        def cut(score, frac):
            """Keep --fixed plus the top `frac` of ALL units by score among the free units."""
            s = as_np(score, host).copy()
            s[fx_np > 0] = np.inf
            if pool is not None:
                s[as_np(pool, host) == 0] = np.where(fx_np[as_np(pool, host) == 0] > 0, np.inf, -np.inf)
            valid = np.zeros_like(s, bool)
            for l, n in enumerate(host.widths):
                valid[l, :n] = True
            s[~valid] = -np.inf
            allowed = valid & (s > -np.inf)          # out-of-pool, non-fixed units can never be kept
            flat = s[allowed]
            k = min(flat.size, max(1, int(round(frac * int(sum(host.widths))))))
            thr = np.sort(flat)[-k]
            M = ((s >= thr) & allowed).astype(np.float32)
            return from_np(M, host)

        def bisect(score, target, mu, evalf, tag):
            lo, hi, evals = max(0.02, float(fx_np.mean())), 1.0, []
            for _ in range(2 if sm else args.bisect_iters):
                if args.bisect_tol and hi - lo < args.bisect_tol:
                    break
                mid = (lo + hi) / 2
                d = evalf(cut(score, mid), mu)["mean_d_ce"]
                if d is None:
                    raise SystemExit(f"{tag}: the evaluation set is empty (nothing scored); refusing the bisect")
                evals.append((round(mid, 5), round(d, 4)))
                if d > target:
                    lo = mid
                else:
                    hi = mid
            log(f"  [{tag} +{target}] kept in ({lo:.4f}, {hi:.4f}]; evals {evals}")
            return hi, evals

        for name, score in rankings.items():
            abl = "mean" if "mean" in name.split("_") else "zero"
            mu = mean_for(abl)
            trained = from_np(np.maximum(fx_np, (as_np(score, host) > 0).astype(np.float32)), host)
            r = {"ablation": abl, "trained_core": {"kept_frac": float(as_np(trained, host).sum() / sum(host.widths)),
                                                   "held": held(trained, mu), "capitals_acc": capitals(trained, mu)}}
            for t in [float(x) for x in args.targets.split(",") if x]:
                frac, ev = bisect(score, t, mu, held_sub, f"{name} learned")
                M = cut(score, frac)
                R = _random_like(M, fixed_or_zero, pool, host, g_seed + int(t * 1000))
                rfrac, rev = bisect(rnd_score, t, mu, held_sub, f"{name} random ranking")
                lh = held(M, mu)
                r[str(t)] = {"learned_kept_frac": frac, "learned_held": lh, "learned_capitals_acc": capitals(M, mu),
                             "learned_meets_target_full": lh["mean_d_ce"] <= t,
                             "bisect_subsample": args.bisect_subsample, "bisect_tol": args.bisect_tol,
                             "learned_per_layer_kept": [float(x) for x in (as_np(M, host).sum(1) / np.array(host.widths))],
                             "bisect_evals": ev, "random_same_counts_held": held(R, mu),
                             "random_same_counts_capitals_acc": capitals(R, mu),
                             "random_ranking_kept_frac_at_target": rfrac, "random_bisect_evals": rev}
                save()
            for s in [float(x) for x in args.size_targets.split(",") if x]:
                frac = float(fx_np.sum() / sum(host.widths)) + s
                M = cut(score, frac)
                R = _random_like(M, fixed_or_zero, pool, host, g_seed + int(s * 10000))
                r[f"size+{s}"] = {"kept_frac": frac, "held": held(M, mu), "capitals_acc": capitals(M, mu),
                                  "random_same_counts_held": held(R, mu), "random_same_counts_capitals_acc": capitals(R, mu)}
                O.save_npz(out / f"tier_{name}_size{s:g}.npz", core=as_np(M, host), Z=as_np(score, host), frac=np.array(frac))
                save()
            rep["bisect"][name] = r
            save()
        ex = args.export_ablation if args.export_ablation in rankings else None
        if ex and "mean" in ex.split("_"):
            raise SystemExit("export core: a mean-ablation ranking cannot be exported by zeroing; use a zero ranking")
        if ex:
            frac, ev = bisect(rankings[ex], args.export_target, mean_for(ex if ex in ("zero", "mean") else "zero"),
                              fit_eval_sub, "export core on " + ("val documents" if crec.get("val_file") else
                                                                 "spare (fit-eval)"))
            M = cut(rankings[ex], frac)
            O.save_npz(out / "export_core.npz", core=as_np(M, host), Z=as_np(rankings[ex], host), frac=np.array(frac))
            rep["export_core"] = {"ranking": ex, "kept_frac": frac,
                                  "chosen_on": "val documents (--val-file)" if crec.get("val_file") else "spare (fit-eval) split",
                                  "bisect_subsample": args.bisect_subsample, "bisect_tol": args.bisect_tol, "evals": ev,
                                  "held": held(M), "spare": fit_eval(M), "capitals_acc": capitals(M),
                                  "export_target": args.export_target,
                                  "held_as_exported_sink0": held(M, None, 0),
                                  "note": "chosen with position 0 exact in every layer (the 27B recipe); a zeroed "
                                          "checkpoint cannot do that, so held_as_exported_sink0 is what it serves"}
            ec = rep["export_core"]
            Rx = _random_like(M, fixed_or_zero, pool, host, g_seed + 77)     # random box at the EXPORT count, held once
            ec["random_same_count_held"] = held(Rx)
            hv, sv = ec["held"]["mean_d_ce"], ec["spare"]["mean_d_ce"]
            ec["meets_export_target_full"] = sv is not None and sv <= args.export_target   # on the full val/spare set
            ec["held_meets_0.05"] = hv is not None and hv <= 0.05                          # Pierre's hard target
            log(f"export core: kept {frac:.4f} held dCE {hv} ({'meets' if ec['held_meets_0.05'] else 'MISSES'} +0.05 on held)")
        done.add("bisect")
        save()

    # ------------------------------------------------------------------ sink
    if "sink" in stages and not skip("sink"):
        M = from_np(np.load(out / "export_core.npz")["core"], host)
        Mn = as_np(M, host)
        p0 = [mx.zeros((n,)) for n in host.widths]

        def st(l, kind, p):
            if kind == "dense":
                p0[l] = p0[l] + mx.abs(p["act"][:, 0].astype(mx.float32)).sum(0)
        n0 = min(48, len(fit))
        for i in range(0, n0, 8):
            host.ctl.reset()
            host.ctl.stats = st
            mx.eval(host.hidden(mx.array(fit[i:min(i + 8, n0)])), *p0)
        host.ctl.reset()
        P0 = as_np([x / float(n0) for x in p0], host) if host.N is None else np.stack([np.array(x) / n0 for x in p0])
        O.save_npz(out / "pos0_abs_act.npz", p0=P0)
        arms = {"none": dict(sink=0), "full_rule_all_layers": dict(sink=L)}
        for k in (1, 2, 3, 4, 6, 8, 16):
            if k < L:
                arms[f"rule_k{k}"] = dict(sink=k)
        for k in (1, 2, 3, 4, 6, 8):
            if k < L:
                arms[f"dense_k{k}"] = dict(sink=0, dense=tuple(range(k)))
        res = {}
        for name, a in arms.items():
            res[name] = {"spare": fit_eval(M, None, a["sink"], a.get("dense", ()))["mean_d_ce"],
                         "held": held(M, None, a["sink"], a.get("dense", ()))}
            log(f"sink {name}: spare {res[name]['spare']:.3f} held {res[name]['held']['mean_d_ce']:.3f}")
        for k in (2, 3, 4):
            for n in (1, 3, 8):
                Mb = Mn.copy()
                for l in range(min(k, L)):
                    top = np.argsort(-(P0[l] * (1 - Mn[l])))[:n]
                    Mb[l, top] = 1.0
                nm = f"addback_k{k}_n{n}"
                Mbm = from_np(Mb, host)
                res[nm] = {"spare": fit_eval(Mbm, None, 0)["mean_d_ce"], "held": held(Mbm, None, 0),
                           "kept_frac": float(Mb.sum() / sum(host.widths))}
        stand = {k: v for k, v in res.items() if k.startswith(("dense_", "addback_"))}
        best = min(stand, key=lambda k: stand[k]["spare"])
        rep["sink"] = {"arms": res, "chosen_stand_in": best, "chosen_on": "spare (fit-eval) split",
                       "used": best if args.stand_in == "chosen" else "none",
                       "note": "core = export core (zero ablation); rule_kK = position 0 exact in layers < K"}
        done.add("sink")
        save()

    # ------------------------------------------------------------------ atlas
    if "atlas" in stages and not skip("atlas"):
        if nsites:
            raise SystemExit("atlas: not implemented for --moe-units neuron (its per-expert one-hot capture would be "
                             "tokens x E*F); run atlas with expert units")
        if host.N is None or mean is None:
            raise SystemExit("atlas capture needs uniform site widths and the mean stage")
        rep["atlas"] = _atlas(host, split, tasks, mean, out, args, log, domains)
        done.add("atlas")
        save()

    # ------------------------------------------------------------------ export
    if "export" in stages and not skip("export"):
        from .export import export_dense_core
        M = np.load(out / "export_core.npz")["core"]
        stand = (rep.get("sink") or {}).get("used", "none")
        if args.stand_in == "chosen":
            stand = (rep.get("sink") or {}).get("chosen_stand_in") or "none"
        if stand.startswith("dense_k"):
            M = M.copy()
            for l in range(int(stand[len("dense_k"):])):
                M[l, :host.widths[l]] = 1.0
        elif stand.startswith("addback_"):
            raise SystemExit("export: add-back stand-ins are not reconstructed here; use --stand-in none or a dense_k one")
        dst = Path(args.export_dir) if args.export_dir else \
            Path.home() / "mlx-models" / f"{Path(args.model).name}-carve-{O.git_rev()}-core"
        rep["export"] = export_dense_core(host, Path(args.model), M, dst, log, verify_ids=fit[:2], dense_layers=())
        rep["export"]["stand_in"] = stand
        done.add("export")
        save()
    save()
    log(f"done in {time.time() - t_start:.0f}s, MLX peak {O.peak_gb()} GB")


def _init_z(spec, abl, out, host, raw=False):
    import mlx.core as mx
    if not spec:
        return mx.full((host.L, host.N), 3.0) if host.N is not None else [mx.full((n,), 3.0) for n in host.widths]
    if spec == "fisher":
        p = out / f"ranking_fisher_{abl}.npz"
    else:
        p = Path(spec.split(":")[0])
    Z = np.load(p)["Z"]
    if Z.shape != (host.L, max(host.widths)):
        raise SystemExit(f"--init-from {p}: Z shape {Z.shape} does not match this donor {(host.L, max(host.widths))}")
    if raw:                               # a seed: its relative scale is the point (mlx2 amoeba extract)
        return from_np(np.clip(Z, -3.0, 3.0).astype(np.float32), host)
    # rescale so the carve starts from the same |Z| scale as z0=3 (sign = the fisher decision)
    Z = 3.0 * np.sign(Z) * np.minimum(1.0, np.abs(Z) / max(1e-6, np.median(np.abs(Z))))
    return from_np(Z.astype(np.float32), host)


def _neuron_memory(host, args, log):
    """Bytes of the carve's per-unit state in neuron units (float32), and the largest per-call MoE
    tensors: a carve never builds tokens x E x F (masks are gathered per routed slot)."""
    U_ = int(sum(host.widths))
    f32 = 4 * U_
    ns = [s for s in host.sites if s.is_moe and s.units == "neuron"]
    k = max((int(getattr(s.module, "top_k", 0) or getattr(getattr(s.module, "gate", None), "top_k", 0) or 0)
             for s in ns), default=0)
    T = args.bs * args.seq_len
    Fm = max(s.F for s in ns)
    Em = max(s.E for s in ns)
    r = {"units": U_, "dtype": "float32",
         "mask_bytes": f32, "grad_bytes": f32, "Z_bytes": f32, "adam_bytes": 2 * f32,
         "train_mask_total_bytes": 6 * f32,            # Z, m, grad sum, sigmoid/gZ, Adam m and v
         "per_call": {"tokens": T, "top_k": k, "F": Fm,
                      "routed_hidden_and_mask_rows_bytes_per_layer": 2 * T * k * Fm * 4,
                      "never_materialised_tokens_x_E_x_F_bytes": T * Em * Fm * 4}}
    log(f"neuron units: {U_} units; mask/grad/Z {f32 / 2**20:.1f} MiB each, Adam {2 * f32 / 2**20:.1f} MiB; "
        f"per MoE call (T={T}, k={k}) hidden + mask rows {r['per_call']['routed_hidden_and_mask_rows_bytes_per_layer'] / 2**20:.1f} MiB")
    return r


def _valid(host):
    v = np.zeros((host.L, max(host.widths)), np.float32)
    for l, n in enumerate(host.widths):
        v[l, :n] = 1.0
    return v


def _freeze_outside(Z0, pool, fixed, host):
    Zn = as_np(Z0, host)
    pl = as_np(pool, host)
    Zn[pl == 0] = -1e9            # dropped unless --fixed keeps it (fixed units are added back by train_mask)
    return from_np(Zn, host)


def _random_like(M, fixed, pool, host, seed):
    """Random core with M's per-layer counts of FREE units, drawn from the free units only."""
    rng = np.random.default_rng(seed)
    Mn, fx = as_np(M, host), as_np(fixed, host)
    pl = as_np(pool, host) if pool is not None else None
    R = fx.copy()
    for l, n in enumerate(host.widths):
        free = np.where((fx[l, :n] == 0) & ((pl[l, :n] > 0) if pl is not None else True))[0]
        k = min(len(free), int(((Mn[l, :n] > 0) & (fx[l, :n] == 0)).sum()))
        if k:
            R[l, rng.choice(free, k, replace=False)] = 1.0
    return from_np(R, host)


def _atlas(host, split, tasks, mean, out, args, log, domains):
    """scripts/amoeba27b_core.py's atlas phase (claude/amoeba-27b-80bc6a@37d75f144), site-generic."""
    import mlx.core as mx
    L, N = host.L, host.N
    adir = Path(args.atlas_dir) if args.atlas_dir else out / "atlas"
    if adir.exists() and not (adir / "meta.json").exists():   # a capture that crashed part-way: start it again
        import shutil
        shutil.rmtree(adir)
    adir.mkdir(parents=True, exist_ok=False)
    seqs, src = [], []
    per = args.atlas_seqs // (len(domains) + 1)
    for dn in domains:
        pool = split[dn]["fit"] + split[dn]["spare"]
        seqs += pool[:per]
        src += [dn] * len(pool[:per])
    tl = []
    for name in ("capitals", "arithmetic", "french", "recall"):
        for it in tasks[name]:
            tl.append((it["ids"], "task_" + name))
    random.Random(args.seed).shuffle(tl)
    tl = tl[:max(0, args.atlas_seqs - len(seqs))]
    S = len(seqs) + len(tl)
    mm = [np.lib.format.open_memmap(adir / f"peak_L{l:02d}.npy", mode="w+", dtype=np.float16, shape=(S, N)) for l in range(L)]
    names = sorted(set(src + [s for _, s in tl]))
    dsum = np.zeros((len(names), L, N), dtype=np.float64)
    dcnt = np.zeros(len(names))
    topk = 8
    topv = [mx.full((topk, N), -1.0) for _ in range(L)]
    topt = [mx.zeros((topk, N), dtype=mx.int32) for _ in range(L)]
    state = {}

    def st(l, kind, p):
        if kind == "dense":
            sig = p["act"][:, 1:].astype(mx.float32)
        else:
            inds, w = p["inds"][:, 1:], p["w"][:, 1:]
            sig = ((inds[..., None] == mx.arange(N)).astype(mx.float32) * w[..., None]).sum(-2)
        dev = mx.abs(sig - mean[l])
        state["peak"][l] = dev.max(1)
        state["dsum"][l] = dev.sum(1)
        flat = dev.reshape(-1, N)
        k = min(topk, flat.shape[0])
        idx = mx.argpartition(-flat, k - 1, axis=0)[:k]
        v = mx.take_along_axis(flat, idx, axis=0)
        t = state["tok"][idx]
        cv, ct = mx.concatenate([topv[l], v]), mx.concatenate([topt[l], t])
        o = mx.argsort(-cv, axis=0)[:topk]
        topv[l], topt[l] = mx.take_along_axis(cv, o, axis=0), mx.take_along_axis(ct, o, axis=0)

    order = [(q, s) for q, s in zip(seqs, src)] + tl
    by_len = {}
    for j, (q, s) in enumerate(order):
        by_len.setdefault(len(q), []).append(j)
    t0, nd = time.time(), 0
    for ln, js in by_len.items():
        for b in range(0, len(js), 8):
            grp = js[b:b + 8]
            ids = mx.array([order[j][0] for j in grp])
            state.update(peak=[None] * L, dsum=[None] * L, tok=ids[:, 1:].reshape(-1).astype(mx.int32))
            host.ctl.reset()
            host.ctl.stats = st
            mx.eval(host.hidden(ids), *state["peak"], *state["dsum"], *topv, *topt)
            host.ctl.reset()
            di = [names.index(order[j][1]) for j in grp]
            for l in range(L):
                mm[l][grp] = np.array(state["peak"][l]).astype(np.float16)
                rows = np.array(state["dsum"][l])
                for r in range(len(grp)):
                    dsum[di[r], l] += rows[r]
            for r in range(len(grp)):
                dcnt[di[r]] += ln - 1
            nd += len(grp)
            mx.clear_cache()
            if nd % 256 < 8:
                log(f"  atlas capture {nd}/{S} ({time.time() - t0:.0f}s, peak {O.peak_gb()} GB)")
    for m in mm:
        m.flush()
    prof = dsum / np.maximum(dcnt, 1)[:, None, None]
    np.savez(adir / "atlas_stats.npz", domain_names=np.array(names), domain_profile=prof.astype(np.float32),
             top_vals=np.stack([np.array(v) for v in topv]), top_toks=np.stack([np.array(t) for t in topt]),
             seq_source=np.array([s for _, s in order]), mean=np.array(mean))
    O.write_json(adir / "meta.json", {"checkpoint": host.checkpoint, "sequences": S, "layers": L, "neurons": N,
                                      "unit_kinds": sorted({s.kind for s in host.sites if s.kind}),
                                      "seq_sources": {n: int(sum(1 for _, s in order if s == n)) for n in names},
                                      # windowed loader: each sequence's source document and token offset, in order
                                      # every claimant (identical windows of different documents collide);
                                      # the part-2 build prefers a non-held claimant
                                      "seq_docs": [[list(c) for c in C.SEQ_DOC.get(tuple(q), [])] for q, _ in order]
                                                  if C.SEQ_DOC else None,
                                      "corpus": args.corpus, "code_cut": args.code_cut,
                                      "unit": "per-sequence peak |unit signal - mean| over positions >= 1, fp16",
                                      "mlx2_source_revision": O.git_rev()})
    return {"dir": str(adir), "sequences": S, "capture_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    main()
