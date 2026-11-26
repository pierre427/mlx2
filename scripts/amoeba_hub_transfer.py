#!/usr/bin/env python3
"""Amoeba hub phase 1: does a donor's box, mapped into the host's residual stream, help the host
*while it generates*? One donor, one host, one process, one owned GPU window (or --cpu for smokes).

Mapping-only: the only fitted objects are linear ridge maps between frozen residual streams; no
weights are updated (recorded in run.json "mapping_only").

Items: decimal-period variants "first 16 digits after the decimal point of a/b" with exact oracles,
same prompt template as amoeba_transplant_screen.decimal_17; train/test split by denominator, so
test denominators are unseen when the maps are fit.

1. Donor stage (plain chat, greedy): every item's trace + oracle -> donor ceiling. Train items are
   teacher-forced again with every donor layer's output kept at generated positions.
2. Host stage: the host is teacher-forced on each train trace's text under its own plain-chat prompt,
   every host layer kept; positions paired where both tokenizations end at the same character
   (as in amoeba_hub_align). Linear CKA (on a position subsample) -> monotone DTW depth warp.
3. Maps per --host-layers: PCA_d(k) -> ridge (penalty relative to mean feature variance) -> host
   residual; only the linear part M (D_donor x D_host) is used, since a box write is a difference.
   Control map M_shuf: the same fit on row-shuffled pairs.
4. Box write in residual space: delta(t) = h_full[l_d](t) - h_box_off[l_d](t), the donor's layer-l_d
   output with every unit versus with the box's units zeroed (drop rule, all layers). Random-box
   control: the same with a random twin of the box (same per-layer counts).
5. Test arms, greedy, per test item:
     host            host alone
     box@L           host + M @ delta_box injected at host layer L
     shuf@L          host + M_shuf @ delta_box
     rand@L          host + M @ delta_randbox (adapter-capacity control: same map, random units)
   Lockstep: after each host token the donor (two caches: full, box-off) is fed the host's text
   re-tokenized by the donor, holding back the last donor token (it may still merge); when new donor
   tokens were fed, the delta at the newest one (its residual integrates the earlier ones, matching the
   same-character pairing of the fit) is injected once, at the next host position. A re-segmentation of
   already-fed donor tokens is a desync: coupling stops for that item (counted). Steps whose host text
   ends in a partial UTF-8 character skip the donor update.
   Controls are norm-matched per step to the real arm's write: shuf uses M_shuf @ delta_box rescaled to
   |M @ delta_box|; rand uses M @ delta_rand rescaled to |M @ delta_box| (a third donor cache runs the
   box-off pass alongside), so they test direction/specificity, not magnitude.
--budget-s stops new work cleanly and writes what is done.

v4 additions (2026-11-15):
--task gsm549: GSM8K test idx 549 plus 30 structural variants (amoeba_gsm549.items; test = original + 10,
   train = 20, numeric answer compare) instead of the decimal items.
Norm budget (--norm-budget, default 0.25; 0 = off): every injected write v at host layer L (point, span band,
   quiet channel and sink) is scaled to at most budget x |h_L - h_{L-1}| at that token, the host layer's own
   increment (layer 0: the residual norm). Controls are norm-matched before alpha and the budget, so the clamp
   treats real and control writes alike. rows carry norm_ratio (|v| / host increment, before the clamp) per
   layer and the clamped fraction; summary aggregates them.
Side stream (Pierre's "untouched box with its own attention"): the donor, or a carved copy of it, runs as its
   own model with its own attention and KV on the same text (its own chat template + the host's generated text
   re-tokenized), and only its output distribution meets the host's, through a gated logit hitch:
       p = (1 - a g) p_host + a g p_side,   g = [max p_side >= tau]
   p_side = the donor's top --side-k next-token probabilities mapped to host ids by identical decoded string,
   renormalised over the mapped mass. Arms (name@a<a>t<tau> per --side-grid entry):
     side_full   whole donor                       (ceiling; resident = the donor)
     side_box    keep ONLY the box's units (every other unit zeroed, drop rule; attention/embeddings intact)
     side_rand   keep only a random twin of the box (same per-layer counts)
     side_shuf   side_box with its mapped probabilities permuted over the same host ids (shuffled hitch)
   The side stream feeds every donor token (no hold-back); a re-segmentation of already-fed donor tokens rebuilds
   its cache from the prompt (counted as rebuilds), so side arms never desync. --side-select-train N chooses one
   grid entry on the first N train items (side_box accuracy, then lower alpha) before the test items.
   Bytes: donor_bytes_on_disk and, per keep set, an estimate of what a carved copy keeps resident
   (disk bytes x (1 - unit_param_share x dropped fraction of unit params)); mapped arms keep the whole donor
   resident (full + box-off passes). Per-arm tok/s in the summary (one lane, plain MLX loop, not mlx2 serving).
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import amoeba_gsm549 as g549  # noqa: E402
import amoeba_hub_align as ha  # noqa: E402
import probe_amoeba_unit_box as pb  # noqa: E402

STATE = {}  # receipt written on any failure (see __main__)

TEMPLATE = "Give the first 16 digits after the decimal point of {a}/{b}, as one digit string."


TRACE_KEYS = ("task", "prompt_format", "max_tokens")


def traces_mismatch(prev_args, cur_args):
    """--donor-traces: the fields a stored donor trace depends on besides the items and the donor checkpoint.
    A trace generated after another prompt format or cut at another token cap is not this run's donor trace.
    Returns the differing fields as {field: (previous, current)}; a field the earlier run.json lacks differs."""
    prev_args = prev_args or {}
    return {k: (prev_args.get(k, "<missing>"), cur_args[k]) for k in TRACE_KEYS
            if k not in prev_args or prev_args[k] != cur_args[k]}



def items(n_items, n_test, seed):
    dens = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    pool = [(a, b) for b in dens for a in (1, 2, 3, 5) if a < b]
    rng = np.random.default_rng(seed)
    pick = [pool[i] for i in rng.permutation(len(pool))[:n_items]]
    test_dens = set()
    for b in rng.permutation(sorted({b for _, b in pick})):  # whole denominators go to test
        if sum(1 for _, bb in pick if bb in test_dens) >= n_test:
            break
        test_dens.add(int(b))
    out = []
    for a, b in pick:
        out.append({"id": f"dec_{a}_{b}", "a": a, "b": b, "split": "test" if b in test_dens else "train",
                    "prompt": TEMPLATE.format(a=a, b=b) + "\nEnd your reply with a line of the form FINAL: <answer>.",
                    "answer": f"{(a * 10 ** 16) // b:016d}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True)
    ap.add_argument("--task", choices=("decimal", "gsm549"), default="decimal",
                    help="gsm549: GSM8K idx 549 + variants (--items/--test ignored)")
    ap.add_argument("--prompt-format", choices=("chat", "gsm8k5shot"), default="chat",
                    help="gsm8k5shot (with --task gsm549): the GSM8K harness completion shape, both models")
    ap.add_argument("--norm-budget", type=float, default=0.25,
                    help="max |write| / |host layer increment| per token at the injection layer (0 = off)")
    ap.add_argument("--side-grid", default="1:0.5,0.5:0", help="side arms: alpha:tau entries")
    ap.add_argument("--side-k", type=int, default=64, help="side stream: donor top-k mapped into the hitch")
    ap.add_argument("--side-select-train", type=int, default=0,
                    help="choose one --side-grid entry on the first N train items (0 = run every entry on test)")
    ap.add_argument("--donor-box-run", help="probe_amoeba_unit_box run dir (box.json); omit only with --smoke-box-frac")
    ap.add_argument("--box-arm", default="mass0.9")
    ap.add_argument("--moe-units", choices=pb.MOE_UNITS, default="expert",
                    help="donor MoE unit granularity; must match the box.json's moe_units (a neuron box is refused "
                         "unless neuron units are installed)")
    ap.add_argument("--host", default=str(Path.home() / "mlx-models/Qwen3.5-4B-MLX-4bit"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", type=int, default=50)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--host-layers", default="16,24", help="single-point arms (box@L, shuf@L, rand@L)")
    ap.add_argument("--span", default="", help="host layer range a-b for span arms (span_box, span_shuf, span_rand)")
    ap.add_argument("--arms", default="", help="comma list of arm names to run (default: host + all defined arms)")
    ap.add_argument("--sink-k", type=int, default=32, help="sink target: union of donor top-k tokens (full, box-off)")
    ap.add_argument("--sink-lam", type=float, default=1e-2, help="sink ridge, relative to mean Jacobian Gram")
    ap.add_argument("--alphas", default="", help="sweep: each injected arm runs once per strength (span and sink)")
    ap.add_argument("--sink-alpha", type=float, default=1.0, help="scale of the sink correction (target = full box effect)")
    ap.add_argument("--english-arms", default="", help="arms for the English-cost KL (default: the test arms)")
    ap.add_argument("--quiet-k", type=int, default=256)
    ap.add_argument("--quiet-fit", type=int, default=48, help="ordinary-text fit texts for the quiet basis")
    ap.add_argument("--quiet-corpus", default=str(Path.home() / "Library/Application Support/mlx2/research-artifacts/"
                                "amoeba/core-corpus-v2-9b64b3791/corpus.jsonl"))
    ap.add_argument("--english-texts", type=int, default=0, help="held ordinary texts for the English-cost KL (0 = off)")
    ap.add_argument("--english-tokens", type=int, default=192)
    ap.add_argument("--test-limit", type=int, default=0, help="first N test items only (0 = all)")
    ap.add_argument("--donor-traces", default="", help="reuse a previous run dir's items.json + donor_traces.jsonl "
                    "(same donor checkpoint and items) instead of regenerating them")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--alpha", type=float, default=1.0, help="scale of the injected mapped write")
    ap.add_argument("--cka-rows", type=int, default=1500)
    ap.add_argument("--max-pairs", type=int, default=6000, help="paired train positions kept (even subsample)")
    ap.add_argument("--budget-s", type=float, default=6600)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--cpu", action="store_true", help="smoke: MLX on CPU, models dequantized to float32")
    ap.add_argument("--mem-cap-gb", type=float, default=0, help="hard MLX memory limit (0 = none)")
    ap.add_argument("--smoke-box-frac", type=float, default=0.0,
                    help="smoke only: synthetic box = this fraction of units per layer (no box.json)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty dir {out}")
    bj = None
    if args.donor_box_run:  # checked before any model loads
        bj = json.loads((Path(args.donor_box_run) / "box.json").read_text())
        box_units = bj.get("moe_units", "expert")  # boxes written before the field existed are expert boxes
        if box_units != args.moe_units:
            sys.exit(f"box.json was built with moe_units={box_units!r}; this run installs {args.moe_units!r} "
                     f"units (pass --moe-units {box_units})")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    over = lambda: time.time() - t_start > args.budget_s  # noqa: E731

    import mlx.core as mx
    if args.cpu:
        mx.set_default_device(mx.cpu)
    if args.mem_cap_gb:
        mx.set_memory_limit(int(args.mem_cap_gb * 1e9))
    mx.set_cache_limit(2 << 30)
    from mlx2.adapters.registry import inspect_model
    from mlx2.experimental.weight_space import residual_ablation as ra
    from mlx2.serving import render_prompt_tokens

    def load(path):
        ad = inspect_model(path).adapter_type(str(path))
        if args.cpu:
            ha.dequantize_model(mx, ad.model, mx.float32)
        lm = getattr(ad.model, "language_model", ad.model)
        return ad, ad.model, ad.tokenizer, list(lm.model.layers), f"{Path(path).name}@{ad.identity['fingerprint']}"

    # host first: Qwen-family adapters pin their import-time environment and refuse if mlx2.runtime
    # model modules were imported before them (Gemma4 has no such guard)
    # Two adapters in one process. The host loads first, so mlx2.runtime is imported under the host's
    # profile (its GDN kernel flags). The donor's adapter then re-pins its own profile; its import-order
    # guard is waived for MLX_GDN_* only (GDN = the host's linear-attention kernels; the donor families
    # used here have no GDN layers), and the host's MLX_GDN_* values are restored afterwards. Any other
    # differing variable still refuses. Recorded in run.json "profile_waiver".
    import os
    from mlx2.runtime.models import import_env
    had, hmodel, htok, hlayers, h_ck = load(args.host)
    host_gdn = {k: v for k, v in os.environ.items() if k.startswith("MLX_GDN_")}
    orig_assert, waived = import_env.assert_profile_applied, set()

    def gdn_tolerant(owner, environ=None, *, live=()):
        names = {k for k in import_env._scoped(os.environ if environ is None else environ, import_env.PREFIXES)}
        names |= {k for snap in import_env._SNAPSHOTS.values() for k in snap}
        gdn = {k for k in names if k.startswith("MLX_GDN_")}
        waived.update(gdn)
        return orig_assert(owner, environ, live=frozenset(live) | gdn)

    import_env.assert_profile_applied = gdn_tolerant
    try:
        dad, dmodel, dtok, dlayers, d_ck = load(args.donor)
    finally:
        import_env.assert_profile_applied = orig_assert
        for k in [k for k in os.environ if k.startswith("MLX_GDN_")]:
            del os.environ[k]
        os.environ.update(host_gdn)
    if any("GatedDelta" in type(m).__name__ for l in dlayers for _, m in l.named_modules()):
        sys.exit("donor has GDN layers; the MLX_GDN_* profile waiver does not apply")
    LD, LH = len(dlayers), len(hlayers)
    sites = pb.install_units(mx, dlayers, moe_units=args.moe_units)
    nunits = np.array([s[2] for s in sites])
    Nmax = int(nunits.max())
    rng = np.random.default_rng(args.seed)
    if bj is not None:
        if bj["checkpoint"] != d_ck:
            sys.exit(f"box.json is for {bj['checkpoint']}, not {d_ck}")
        if [s["units"] for s in bj.get("sites", [])] not in ([], nunits.tolist()):
            sys.exit("box.json unit counts per layer differ from the installed sites")
        box = np.zeros((LD, Nmax), bool)
        if "units" in bj["boxes"][args.box_arm]:
            for li, u in enumerate(bj["boxes"][args.box_arm]["units"]):
                box[li, u] = True
        else:  # neuron-mode probe runs keep their masks in boxes_file
            mask = np.load(Path(args.donor_box_run) / bj["boxes_file"])[args.box_arm]
            if mask.shape != box.shape:
                sys.exit(f"box mask shape {mask.shape} != installed {box.shape}")
            box[:] = mask
    elif args.smoke_box_frac > 0:
        box = np.zeros((LD, Nmax), bool)
        for li, n in enumerate(nunits):
            if n:
                box[li, rng.choice(n, max(1, int(n * args.smoke_box_frac)), replace=False)] = True
    else:
        sys.exit("need --donor-box-run (or --smoke-box-frac for a smoke)")
    twin = np.zeros_like(box)
    for li in range(LD):
        k = int(box[li].sum())
        if k:
            twin[li, rng.choice(nunits[li], k, replace=False)] = True

    def off_mask(b):  # every unit kept except b's; layers without units untouched
        a = (~b).astype(np.float32)
        for li, s in enumerate(sites):
            if s[1] is None:
                a[li] = 1.0
        return mx.array(a)

    OFF = {"box": off_mask(box), "rand": off_mask(twin)}
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    its = items(args.items, args.test, args.seed) if args.task == "decimal" else g549.items()
    ok_of = lambda pred, it: g549.correct(pred, it)  # noqa: E731  (exact string for decimal items)

    def keep_mask(b):  # ONLY b's units kept; layers without units untouched
        a = b.astype(np.float32)
        for li, s_ in enumerate(sites):
            if s_[1] is None:
                a[li] = 1.0
        return mx.array(a)

    KEEP = {"full": None, "box": keep_mask(box), "rand": keep_mask(twin)}
    ppu = np.array([s_[3] for s_ in sites], np.float64)
    unit_params = float((nunits * ppu).sum())

    def logical_params(model):
        n = 0
        for _, m in model.named_modules():
            for k, v in m.items():
                if isinstance(v, mx.array) and k not in ("scales", "biases"):
                    n += v.size * (32 // m.bits if k == "weight" and "scales" in m and hasattr(m, "bits") else 1)
        return float(n)

    d_total = logical_params(dmodel)
    disk = float(sum(f.stat().st_size for f in Path(args.donor).glob("*.safetensors")))
    share = min(1.0, unit_params / max(d_total, 1.0))
    kept = {"full": 1.0, "box": float((box.sum(1) * ppu).sum() / unit_params), "rand": float((twin.sum(1) * ppu).sum() / unit_params)}
    bytes_info = {"donor_bytes_on_disk": disk, "unit_param_share": round(share, 4),
                  "unit_params_kept": {k: round(v, 4) for k, v in kept.items()},
                  "carved_resident_bytes_est": {k: round(disk * (1 - share * (1 - v))) for k, v in kept.items()},
                  "mapped_arms_resident_bytes": disk}
    print(f"bytes {bytes_info}", flush=True)
    STATE.update(out=str(out), receipt={"schema": "amoeba-hub-transfer/v4", "donor_checkpoint": d_ck,
                 "host_checkpoint": h_ck, "mlx2_source_revision": rev, "tree_dirty": dirty, "args": vars(args),
                 "mapping_only": "ridge maps fit on frozen residual streams; no weight updates",
                 "profile_waiver": {"waived_on_donor_load": sorted(waived), "host_gdn_env_restored": host_gdn}},
                 stage="donor stage", t_start=t_start)
    print(f"donor {d_ck} host {h_ck} rev {rev} dirty {dirty} box {args.box_arm} units {int(box.sum())} "
          f"items {len(its)} test {sum(i['split'] == 'test' for i in its)} load {time.time() - t_start:.0f}s", flush=True)

    FMT = args.prompt_format

    def prompt_of(ad, it):
        render = lambda c, th: [int(t) for t in render_prompt_tokens(  # noqa: E731
            ad, {"messages": [{"role": "user", "content": c}], "enable_thinking": th})]
        return g549.prompt_ids(ad.tokenizer, render, it, FMT)

    def pred_text(text):
        return g549.pred_of(text, FMT)

    d_eos, h_eos = set(dtok.eos_token_ids), set(htok.eos_token_ids)

    def out_of(o):
        x = o[0] if isinstance(o, tuple) else o
        return x

    # ---------------------------------------------------------------- 1. donor stage
    pb.ST.update(active=True, allow=None, record=None)
    dtr = {}
    if args.donor_traces:
        prev = Path(args.donor_traces)
        if json.loads((prev / "items.json").read_text()) != its:
            sys.exit("--donor-traces items differ (seed/items/test args must match)")
        prun = json.loads((prev / "run.json").read_text())
        if prun.get("donor_checkpoint") != d_ck:
            sys.exit(f"--donor-traces donor {prun.get('donor_checkpoint')} != {d_ck}")
        if bad := traces_mismatch(prun.get("args"), vars(args)):
            sys.exit("--donor-traces were generated with other settings (previous, current): "
                     + ", ".join(f"{k} {v[0]!r} != {v[1]!r}" for k, v in bad.items()))
        for l in (prev / "donor_traces.jsonl").read_text().split("\n"):
            if not l.strip():
                continue
            r = json.loads(l)
            it = next(i for i in its if i["id"] == r["item"])
            dtr[r["item"]] = {k: v for k, v in r.items() if k != "item"} | {"prompt": prompt_of(dad, it)}
        print(f"reused {len(dtr)} donor traces from {prev}", flush=True)
    for it in its:
        if over():
            break
        if it["id"] in dtr:
            continue
        dp = prompt_of(dad, it)
        cache, gen = dmodel.make_cache(), []
        lg = dmodel(mx.array([dp]), cache=cache)[0, -1]
        tg = time.time()
        for _ in range(args.max_tokens):
            t = int(mx.argmax(lg).item())
            if t in d_eos:
                break
            gen.append(t)
            if FMT != "chat" and len(gen) % 4 == 0 and g549.stopped(dtok.decode(gen), FMT):
                break
            lg = dmodel(mx.array([[t]]), cache=cache)[0, -1]
        del cache
        text = dtok.decode(gen)
        pred = pred_text(text)
        dtr[it["id"]] = {"prompt": dp, "gen": gen, "text": text, "pred": pred, "correct": ok_of(pred, it),
                         "tokens": len(gen), "gen_s": round(time.time() - tg, 1)}
        print(f"{time.strftime('%H:%M:%S')} donor {it['id']} {it['split']} tokens {len(gen)} pred {pred!r} "
              f"correct {ok_of(pred, it)}", flush=True)
    mx.clear_cache()
    (out / "items.json").write_text(json.dumps(its, indent=1))
    with open(out / "donor_traces.jsonl", "w") as f:
        for iid, tr in dtr.items():
            f.write(json.dumps({"item": iid, **{k: v for k, v in tr.items() if k not in ("prompt",)}}) + "\n")
    STATE["stage"] = "host stage"

    def tf_layers(model, layers, seq, gen0):
        """Teacher-forced forward; every layer's output at rows gen0-1.. (inputs that predict generated tokens)."""
        keep = {}

        def hook(i):
            def h(o):
                x = out_of(o)[0, gen0 - 1:].astype(mx.float32)
                if x.ndim == 3:  # hyper-connection streams
                    x = x.mean(1)
                keep[i] = x  # float32: Gemma-style residuals can overflow float16
                mx.eval(keep[i])
                return o
            return h

        with ra._hooked([(layers[i], hook(i)) for i in range(len(layers))]):
            mx.eval(model(mx.array([seq[:-1]]), cache=model.make_cache()))
        res = {i: np.array(v) for i, v in keep.items()}
        mx.clear_cache()
        return res

    # ---------------------------------------------------------------- 2. host stage on train traces
    train = [it for it in its if it["split"] == "train" and it["id"] in dtr and dtr[it["id"]]["tokens"] > 1]
    DX = {l: [] for l in range(LD)}
    HY = {l: [] for l in range(LH)}
    for it in train:
        if over():
            break
        tr = dtr[it["id"]]
        dends, gtext = ha.char_ends_from_ids(dtok, tr["gen"])
        hp = prompt_of(had, it)
        enc = htok.encode(gtext, add_special_tokens=False)
        hends, _ = ha.char_ends_from_ids(htok, enc)
        hpos = {e: j for j, e in enumerate(hends)}
        # generated token i is predicted from input row i (rows start at gen0-1); drop the last token
        pairs = [(i, hpos[e]) for i, e in enumerate(dends) if e in hpos and i < len(dends) - 1 and hpos[e] < len(enc) - 1]
        if not pairs:
            continue
        cap_per = max(1, args.max_pairs // max(1, len(train)))
        if len(pairs) > cap_per:
            pairs = [pairs[j] for j in np.linspace(0, len(pairs) - 1, cap_per).astype(int)]
        dh = tf_layers(dmodel, dlayers, tr["prompt"] + tr["gen"], len(tr["prompt"]))
        hh = tf_layers(hmodel, hlayers, hp + enc, len(hp))
        di, hj = np.array([p[0] + 1 for p in pairs]), np.array([p[1] + 1 for p in pairs])
        for l in range(LD):
            DX[l].append(dh[l][di])
        for l in range(LH):
            HY[l].append(hh[l][hj])
        print(f"{time.strftime('%H:%M:%S')} paired {it['id']}: {len(pairs)} positions", flush=True)
    if not DX[0]:
        sys.exit("no paired train positions")
    DX = {l: np.concatenate(v) for l, v in DX.items()}
    HY = {l: np.concatenate(v) for l, v in HY.items()}
    P = len(DX[0])
    if not all(np.isfinite(v).all() for v in list(DX.values()) + list(HY.values())):
        raise RuntimeError("non-finite residuals in train captures")
    sub = rng.choice(P, min(P, args.cka_rows), replace=False)

    def gram(X):  # centred linear Gram, unit Frobenius norm: CKA(X, Y) = <Kx, Ky>
        X = X[sub] - X[sub].mean(0)
        K = X @ X.T
        return K / (np.linalg.norm(K) + 1e-12)

    Kd = [gram(DX[a]) for a in range(LD)]
    Kh = [gram(HY[b]) for b in range(LH)]
    C = np.array([[float((Kd[a] * Kh[b]).sum()) for b in range(LH)] for a in range(LD)])
    del Kd, Kh
    warp = ha.dtw(1 - C)
    np.save(out / "cka.npy", C)
    (out / "warp.json").write_text(json.dumps({"donor_for_host": warp, "cka_on_warp": [float(C[w, j]) for j, w in enumerate(warp)]}))
    print(f"pairs {P} warp {warp}", flush=True)
    STATE["stage"] = "maps + test arms"

    # ---------------------------------------------------------------- 3. maps
    maps = {}
    point_layers = [int(x) for x in args.host_layers.split(",") if x.strip()]
    span_layers = []
    if args.span:
        a_, b_ = (int(x) for x in args.span.split("-"))
        span_layers = list(range(a_, b_ + 1))
    # Span bands from the dense warp (every donor layer is captured here): walk the span's host layers; a
    # host layer whose donor layer equals the previous band's is a plateau and gets no band (its increment
    # would be zero); a host step that jumps several donor layers takes their summed (telescoped) increment.
    # The first band's base is None: it carries the box's cumulative write up to its donor layer, so box
    # writes below the span enter at the span's first layer and nothing is dropped. Sum over bands of the
    # increments = the cumulative write at the last band's donor layer.
    bands, prev_dl = [], None
    for h in span_layers:
        if warp[h] != prev_dl:
            bands.append({"hl": h, "dl": warp[h], "base": prev_dl})
            prev_dl = warp[h]
    for hl in sorted(set(point_layers) | {b["hl"] for b in bands}):
        dl = warp[hl]
        X, Y = DX[dl], HY[hl]
        k = min(args.k, len(X) - 1, X.shape[1])
        pca = ha.PCA(X, k)
        Z = pca(X)

        def lin(Zf):
            Zc = Zf - Zf.mean(0)
            G = Zc.T @ Zc
            W = np.linalg.solve(G + args.lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), Zc.T @ (Y - Y.mean(0)))
            return (pca.V @ W).astype(np.float32)  # D_donor x D_host

        M, Msh = lin(Z), lin(Z[rng.permutation(len(Z))])
        fit_r2 = ha.r2((X - X.mean(0)) @ M + Y.mean(0), Y)
        maps[hl] = {"dl": dl, "M": mx.array(M), "Msh": mx.array(Msh), "fit_r2": fit_r2,
                    "host_rms": float(np.sqrt((Y ** 2).mean()))}
        print(f"map host L{hl} <- donor L{dl}: fit R2 {fit_r2:.3f} (train, in-sample)", flush=True)
    del DX, HY

    # ---------------------------------------------------------------- 4/5. arms
    LAST = LH - 1
    hlm = getattr(hmodel, "language_model", hmodel)

    head_mod = hlm.model.embed_tokens if getattr(hlm.args, "tie_word_embeddings", False) else hlm.lm_head

    def head_rows(T):  # unembedding rows for host token ids T, float32 [|T|, D] (dequantized if quantized)
        idx = mx.array(T)
        if "scales" in head_mod:
            kw = {"mode": head_mod.mode} if hasattr(head_mod, "mode") else {}
            b_ = head_mod["biases"][idx] if "biases" in head_mod else None
            w = mx.dequantize(head_mod["weight"][idx], head_mod["scales"][idx], b_, head_mod.group_size,
                              head_mod.bits, **kw)
        else:
            w = head_mod["weight"][idx]
        return w.astype(mx.float32)

    # Cross-tokenizer token map for the sink: donor id -> host id where both decode to the same string.
    h_str2id, d2h = {}, {}

    def host_of(did):
        if did not in d2h:
            if not h_str2id:
                for i in range(len(htok)):
                    h_str2id.setdefault(htok.decode([i]), i)
            d2h[did] = h_str2id.get(dtok.decode([did]))
        return d2h[did]

    # Quiet subspace: host residual PCA on ordinary text (core corpus fit split) at every injection layer.
    quiet = {}

    def quiet_basis(layers_needed):
        if quiet or not layers_needed:
            return
        recs = [json.loads(l) for l in Path(args.quiet_corpus).read_text().split("\n") if l.strip()]
        fit = [r for r in recs if r["split"] == "fit"]
        fit = [fit[i] for i in rng.permutation(len(fit))[:args.quiet_fit]]
        S, mu, n = {l: 0 for l in layers_needed}, {l: 0 for l in layers_needed}, 0
        for r in fit:
            ids = htok.encode(r["text"], add_special_tokens=False)[:256]
            keep = {}

            def hk(i):
                def h(o):
                    keep[i] = out_of(o)[0].astype(mx.float32)
                    return o
                return h

            with ra._hooked([(hlayers[i], hk(i)) for i in layers_needed]):
                mx.eval(hmodel(mx.array([ids]), cache=hmodel.make_cache()))
            for l in layers_needed:
                X = np.array(keep[l], dtype=np.float64)
                S[l] = S[l] + X.T @ X
                mu[l] = mu[l] + X.sum(0)
            n += len(ids)
            mx.clear_cache()
        for l in layers_needed:
            m = mu[l] / n
            w, V = np.linalg.eigh(S[l] / n - np.outer(m, m))
            Vb = V[:, np.argsort(w)[:args.quiet_k]].astype(np.float32)  # lowest-variance directions
            Q, _ = np.linalg.qr(rng.standard_normal((V.shape[0], args.quiet_k)))
            quiet[l] = {"bottom": mx.array(Vb), "random": mx.array(Q.astype(np.float32))}
        print(f"quiet bases k={args.quiet_k} at host layers {layers_needed} from {n} ordinary-text positions", flush=True)

    def sink_delta(x, tgt_h, tgt_v):
        """Least-squares host residual correction d (ridge) whose linearised effect through the host's final
        norm + unembedding reproduces the centred target logit delta on host tokens tgt_h."""
        W = head_rows(tgt_h)  # |T| x D
        h0 = x.reshape(-1).astype(mx.float32)
        H0 = mx.broadcast_to(x.reshape(1, -1).astype(mx.float32), W.shape)
        # row i of d/dH sum_i <norm(H_i), W_i> is J_i = d logit_{T_i} / dh (the norm is row-wise), taken
        # through the host's own final norm module
        J = mx.grad(lambda H: (hlm.model.norm(H[None].astype(x.dtype))[0].astype(mx.float32) * W).sum())(H0)
        Jc = J - J.mean(0, keepdims=True)
        t = tgt_v - tgt_v.mean()
        G = Jc @ Jc.T
        lam = args.sink_lam * mx.trace(G) / len(tgt_h)
        a = mx.linalg.solve(G + lam * mx.eye(len(tgt_h)), t[:, None], stream=mx.cpu)
        d = (Jc.T @ a)[:, 0]
        # realised (nonlinear) logit change on T, centred, vs the centred target
        nrm = lambda h: hlm.model.norm(h.reshape(1, 1, -1).astype(x.dtype)).reshape(-1).astype(mx.float32)  # noqa: E731
        got = W @ (nrm(h0 + d) - nrm(h0))
        got = got - got.mean()
        cos = (got * t).sum() / (mx.sqrt((got * got).sum() * (t * t).sum()) + 1e-12)
        return d, cos

    def run_arm(it, arm, forced=None, ref_lps=None):
        """arm: dict(bands, mkey, off, match, sink, quiet). forced: host token ids to teacher-force (English
        cost); ref_lps: host-alone log-probs at those steps (returned when arm is host and ref_lps is None)."""
        bands_, mkey, off, match = arm["bands"], arm["mkey"], OFF.get(arm["off"]), arm["match"]
        sink, qk, side = arm["sink"], arm["quiet"], arm.get("side")
        if forced is None:
            hp, steps = prompt_of(had, it), args.max_tokens
        else:
            hp, steps = forced[:8], len(forced) - 8
        hc = hmodel.make_cache()
        inj, sink_now = {}, {}
        stats = {"injections": 0, "desync": False, "delta_rms": [], "sink_rms": [], "lag_chars": [], "sink_tokens": [],
                 "sink_cos": [], "norm_ratio": {}, "clamped": 0, "writes": 0, "rebuilds": 0, "side_steps": 0,
                 "side_open": 0, "side_mapped": []}
        prev_out = {}

        def budget(v, x0, hl):
            """v scaled to at most norm_budget x the host layer's own increment at this token."""
            ref = prev_out.get(hl - 1)
            a_ = x0[0, -1].astype(mx.float32)
            loc = mx.sqrt((((a_ - ref) if ref is not None else a_) ** 2).sum())
            r_ = mx.sqrt((v.astype(mx.float32) ** 2).sum()) / (loc + 1e-12)
            rv = float(r_.item())
            stats["norm_ratio"].setdefault(hl, []).append(rv)
            stats["writes"] += 1
            if args.norm_budget > 0 and rv > args.norm_budget:
                stats["clamped"] += 1
                v = v * (args.norm_budget / rv)
            return v

        def hhook(hl):
            def h(o):
                v = inj.get(hl)
                st = sink_now.get("t") if hl == LAST else None
                if v is None and st is None:
                    if hl + 1 in hook_set:
                        prev_out[hl] = out_of(o)[0, -1].astype(mx.float32)
                    return o
                x = out_of(o)  # injection happens only on single-token steps: [1, 1, D]
                x0 = x
                if v is not None:
                    x = x + budget(v, x0, hl).astype(x.dtype)
                if st is not None:
                    d, cos = sink_delta(x, *st["real"])
                    stats["sink_cos"].append(float(cos.item()))
                    if st.get("ctrl") is not None:  # control target, rescaled to the real correction's norm
                        dc_, _ = sink_delta(x, *st["ctrl"])
                        d = dc_ * (mx.sqrt((d * d).sum()) / (mx.sqrt((dc_ * dc_).sum()) + 1e-12))
                    d = budget(arm.get("alpha", args.sink_alpha) * d, x0, hl)
                    mx.eval(d)
                    stats["sink_rms"].append(float(mx.sqrt((d * d).mean()).item()))
                    x = x + d.astype(x.dtype)[None, None]
                if hl + 1 in hook_set:
                    prev_out[hl] = x[0, -1].astype(mx.float32)
                return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x
            return h

        coupled = bands_ is not None or sink is not None or side is not None
        inj_layers = {b["hl"] for b in (bands_ or [])} | ({LAST} if sink else set())
        hook_set = inj_layers | {l - 1 for l in inj_layers if l > 0}
        hook_layers = sorted(hook_set)
        if coupled:
            dls = sorted({b["dl"] for b in (bands_ or [])} | {b["base"] for b in (bands_ or []) if b["base"] is not None})
            if forced is None:
                dp = prompt_of(dad, it)
            else:
                bos = getattr(dtok, "bos_token_id", None)
                dp = ([bos] if bos is not None else []) + dtok.encode(htok.decode(hp), add_special_tokens=False)
            if side is not None:
                passes = [("side", KEEP[side["keep"]])]
            else:
                passes = [("full", None), ("off", off)] + ([("ref", OFF["box"])] if match == "off" else [])
            dc = {nm: dmodel.make_cache() for nm, _ in passes}
            cap = {}

            def dhook(l):
                def h(o):
                    x = out_of(o)
                    if x.ndim == 4:
                        x = x.mean(2)
                    cap[l] = x[0, -1].astype(mx.float32)
                    return o
                return h

            def dfeed(toks):
                outs = {}
                for nm, allow in passes:
                    pb.ST["allow"] = allow
                    with ra._hooked([(dlayers[l], dhook(l)) for l in dls]):
                        lgd = dmodel(mx.array([toks]), cache=dc[nm])[0, -1].astype(mx.float32)
                    outs[nm] = dict(cap) | {"lp": lgd - mx.logsumexp(lgd)}
                    mx.eval(outs[nm]["lp"])
                pb.ST["allow"] = None
                return outs

            o0 = dfeed(dp)
            fed = []
            ctx = ra._hooked([(hlayers[l], hhook(l)) for l in hook_layers])
        else:
            ctx = ra._hooked([])

        def sink_target(lp_a, lp_b):
            """donor logp delta lp_a - lp_b on the union of both top-k, mapped to host ids (string-identical)."""
            cand = set(np.array(mx.argpartition(-lp_a, kth=args.sink_k)[:args.sink_k]).tolist()) | \
                set(np.array(mx.argpartition(-lp_b, kth=args.sink_k)[:args.sink_k]).tolist())
            dl_ = np.array(lp_a - lp_b)
            hid, val = [], []
            for c in sorted(cand):
                h_ = host_of(c)
                if h_ is not None and h_ not in hid:
                    hid.append(h_)
                    val.append(dl_[c])
            return (hid, mx.array(np.array(val, np.float32))) if len(hid) >= 4 else None

        def side_dist(lpd):
            """donor top-k -> host ids (identical string), renormalised over the mapped mass; gate on max prob."""
            k_ = min(args.side_k, lpd.shape[-1] - 1)
            top = np.array(mx.argpartition(-lpd, kth=k_)[:k_])
            pv = np.exp(np.array(lpd[mx.array(top)], dtype=np.float64))
            hid, pr = [], []
            for c, p_ in sorted(zip(top.tolist(), pv.tolist()), key=lambda z: -z[1]):
                h_ = host_of(int(c))
                if h_ is not None and h_ not in hid:
                    hid.append(h_)
                    pr.append(p_)
            if not hid:
                return None
            pr = np.array(pr)
            stats["side_mapped"].append(float(pr.sum()))
            pr = pr / pr.sum()
            if side.get("shuf"):
                pr = pr[rng.permutation(len(pr))]
            return {"hid": mx.array(hid), "p": mx.array(pr.astype(np.float32)), "open": float(pv.max()) >= side["tau"]}

        side_now = side_dist(o0["side"]["lp"]) if side is not None else None
        gen, kls = [], []
        lps = [] if (forced is not None and ref_lps is None) else None
        with ctx:
            lg = hmodel(mx.array([hp]), cache=hc)[0, -1]
            for step in range(steps):
                if side is not None:
                    stats["side_steps"] += 1
                    if side_now is not None and side_now["open"]:
                        stats["side_open"] += 1
                        a_ = side["alpha"]
                        ph = (1 - a_) * mx.exp(lg.astype(mx.float32) - mx.logsumexp(lg.astype(mx.float32)))
                        lg = mx.log(ph.at[side_now["hid"]].add(a_ * side_now["p"]) + 1e-30)
                if forced is not None:
                    lp = lg.astype(mx.float32) - mx.logsumexp(lg.astype(mx.float32))
                    if lps is not None:
                        mx.eval(lp)
                        lps.append(lp)
                    else:
                        pr = mx.exp(ref_lps[step])
                        kls.append(float((pr * (ref_lps[step] - lp)).sum().item()))
                    t = forced[8 + step]
                else:
                    t = int(mx.argmax(lg).item())
                    if t in h_eos:
                        break
                    if FMT != "chat" and g549.stopped(htok.decode(gen), FMT):
                        break
                gen.append(t)
                inj.clear()
                sink_now.clear()
                if side is not None:
                    text = htok.decode(gen)
                    d = dtok.encode(text, add_special_tokens=False) if not text.endswith("�") else None
                    if d is None:
                        side_now = None  # partial character: no hitch this step
                    else:
                        if d[:len(fed)] != fed:  # re-segmentation: rebuild the side stream from its prompt
                            dc["side"] = dmodel.make_cache()
                            o_ = dfeed(dp + d)
                            fed = list(d)
                            stats["rebuilds"] += 1
                        elif len(d) > len(fed):
                            o_ = dfeed(d[len(fed):])
                            fed = list(d)
                        else:
                            o_ = None
                        if o_ is not None:
                            side_now = side_dist(o_["side"]["lp"])
                            stats["injections"] += 1
                elif coupled and not stats["desync"]:
                    text = htok.decode(gen)
                    d = dtok.encode(text, add_special_tokens=False) if not text.endswith("�") else None
                    if d is None:
                        pass
                    elif d[:len(fed)] != fed:
                        stats["desync"] = True
                    else:
                        new = d[len(fed):len(d) - 1]
                        if new:
                            o_ = dfeed(new)  # newest fed donor position
                            fed += new
                            rms = []
                            for b in (bands_ or []):
                                def inc_of(nm):
                                    a_, b_ = o_["full"][b["dl"]] - o_[nm][b["dl"]], 0
                                    if b["base"] is not None:
                                        b_ = o_["full"][b["base"]] - o_[nm][b["base"]]
                                    return a_ - b_
                                inc = inc_of("off")
                                mp = maps[b["hl"]]
                                v = inc @ mp[mkey]
                                if match:  # per band, per step: rescale the control to the real write's norm
                                    ref = inc @ mp["M"] if match == "map" else inc_of("ref") @ mp["M"]
                                    v = v * (mx.sqrt((ref * ref).sum()) / (mx.sqrt((v * v).sum()) + 1e-12))
                                if qk:  # quiet channel: project onto the k-dim subspace, keep the write's size
                                    B = quiet[b["hl"]][qk]
                                    vq = (v @ B) @ B.T
                                    v = vq * (mx.sqrt((v * v).sum()) / (mx.sqrt((vq * vq).sum()) + 1e-12))
                                v = arm.get("alpha", args.alpha) * v
                                inj[b["hl"]] = inj[b["hl"]] + v if b["hl"] in inj else v
                                rms.append(mx.sqrt((v * v).mean()))
                            if sink:
                                real = sink_target(o_["full"]["lp"], o_["ref" if match == "off" else "off"]["lp"])
                                if real is not None:
                                    st = {"real": real}
                                    if match == "off":  # random-box target, rescaled to the real box correction
                                        st["ctrl"] = sink_target(o_["full"]["lp"], o_["off"]["lp"])
                                    elif match == "map":  # shuffled target over the same tokens
                                        st["ctrl"] = (real[0], real[1][mx.array(rng.permutation(len(real[0])))])
                                    if st.get("ctrl", 0) is not None:
                                        sink_now["t"] = st
                                        stats["sink_tokens"].append(len(real[0]))
                            if inj:
                                mx.eval(list(inj.values()))
                            stats["injections"] += 1
                            if rms:
                                stats["delta_rms"].append(float(mx.stack(rms).mean().item()))
                            stats["lag_chars"].append(len(text) - len(dtok.decode(fed)))
                lg = hmodel(mx.array([[t]]), cache=hc)[0, -1]
        text = htok.decode(gen)
        del hc
        if coupled:
            del dc
        mx.clear_cache()
        r = {"injections": stats["injections"], "desync": stats["desync"],
             "inj_rms_mean": float(np.mean(stats["delta_rms"])) if stats["delta_rms"] else 0.0,
             "sink_rms_mean": float(np.mean(stats["sink_rms"])) if stats["sink_rms"] else 0.0,
             "sink_tokens_mean": float(np.mean(stats["sink_tokens"])) if stats["sink_tokens"] else 0.0,
             "sink_fit_cos_mean": float(np.mean(stats["sink_cos"])) if stats["sink_cos"] else None,
             "lag_chars_mean": float(np.mean(stats["lag_chars"])) if stats["lag_chars"] else 0.0,
             "norm_ratio": {int(k): round(float(np.mean(v)), 4) for k, v in stats["norm_ratio"].items()},
             "norm_ratio_max": {int(k): round(float(np.max(v)), 4) for k, v in stats["norm_ratio"].items()},
             "clamped_frac": round(stats["clamped"] / stats["writes"], 4) if stats["writes"] else 0.0,
             "rebuilds": stats["rebuilds"],
             "side_open_frac": round(stats["side_open"] / stats["side_steps"], 4) if stats["side_steps"] else None,
             "side_mapped_mass": round(float(np.mean(stats["side_mapped"])), 4) if stats["side_mapped"] else None}
        if forced is not None:
            return (lps if lps is not None else r | {"kl_vs_host": float(np.mean(kls)) if kls else 0.0})
        pred = pred_text(text)
        # answer-in-trace: a slower trajectory that reaches the digits but not a FINAL line still counts here
        return r | {"tokens": len(gen), "pred": pred, "correct": ok_of(pred, it),
                    "answer_in_text": it["answer"] in text, "answer_in_digits": it["answer"] in re.sub(r"\D", "", text),
                    "hit_cap": len(gen) >= args.max_tokens, "tail": text[-300:]}

    A = lambda bands_=None, mkey="M", off=None, match=None, sink=None, quiet_=None, side=None: \
        {"bands": bands_, "mkey": mkey, "off": off, "match": match, "sink": sink, "quiet": quiet_, "side": side}  # noqa: E731
    arms = {"host": A()}
    grid = [tuple(float(y) for y in x.split(":")) for x in args.side_grid.split(",") if x.strip()]
    gname = lambda a_, t_: f"a{a_:g}t{t_:g}"  # noqa: E731
    for a_, t_ in grid:
        for nm, kp, sh in (("side_full", "full", False), ("side_box", "box", False), ("side_rand", "rand", False),
                           ("side_shuf", "box", True)):
            arms[f"{nm}@{gname(a_, t_)}"] = A(side={"keep": kp, "shuf": sh, "alpha": a_, "tau": t_})
    for hl in point_layers:
        pt = [{"hl": hl, "dl": maps[hl]["dl"], "base": None}]
        arms |= {f"box@{hl}": A(pt, "M", "box"), f"shuf@{hl}": A(pt, "Msh", "box", "map"),
                 f"rand@{hl}": A(pt, "M", "rand", "off")}
    if bands:
        arms |= {"span_box": A(bands, "M", "box"), "span_shuf": A(bands, "Msh", "box", "map"),
                 "span_rand": A(bands, "M", "rand", "off"),
                 "spansink_box": A(bands, "M", "box", None, "sink"),
                 "spansink_shuf": A(bands, "Msh", "box", "map", "sink"),
                 "spansink_rand": A(bands, "M", "rand", "off", "sink"),
                 "quietsink_box": A(bands, "M", "box", None, "sink", "bottom"),
                 "quietsink_shuf": A(bands, "Msh", "box", "map", "sink", "bottom"),
                 "quietsink_rand": A(bands, "M", "rand", "off", "sink", "bottom"),
                 "quietsink_randsub": A(bands, "M", "box", None, "sink", "random")}
    arms |= {"sink_box": A(None, "M", "box", None, "sink"), "sink_shuf": A(None, "M", "box", "map", "sink"),
             "sink_rand": A(None, "M", "rand", "off", "sink")}
    # --arms names a side arm by its base name (side_box) to take every grid entry
    def pick(names):
        outn = []
        for a_ in names:
            outn += [n for n in arms if n.split("@a")[0] == a_ and n.startswith("side_")] if a_.startswith("side_") and "@" not in a_ else [a_]
        return outn
    want = pick([a.strip() for a in args.arms.split(",")]) if args.arms else \
        [a for a in arms if not a.startswith(("spansink", "quietsink", "sink", "side_"))]
    unknown = set(want) - set(arms)
    if unknown:
        sys.exit(f"unknown arms {sorted(unknown)}; defined: {sorted(arms)}")
    eng_want = pick([a.strip() for a in args.english_arms.split(",")]) if args.english_arms else want
    unknown = set(eng_want) - set(arms)
    if unknown:
        sys.exit(f"unknown english arms {sorted(unknown)}")
    eng_arms = {a: arms[a] for a in eng_want}
    arms = {a: arms[a] for a in want}
    if args.alphas:  # one variant per strength; controls stay norm-matched to the real write before alpha
        al = [float(x) for x in args.alphas.split(",")]
        expand = lambda d: {n: a for n, a in d.items() if n == "host"} | {  # noqa: E731
            f"{n}@a{x:g}": dict(a, alpha=x) for n, a in d.items() if n != "host" and not a["side"] for x in al} | {
            n: a for n, a in d.items() if a["side"]}
        arms, eng_arms = expand(arms), expand(eng_arms)
    quiet_basis(sorted({b["hl"] for a in list(arms.values()) + list(eng_arms.values()) if a["quiet"] for b in a["bands"]}))
    side_sel = None
    if args.side_select_train and any(a["side"] for a in arms.values()) and len(grid) > 1:
        STATE["stage"] = "side selection"
        sel_items = [it for it in its if it["split"] == "train"][:args.side_select_train]
        scores = {}
        for a_, t_ in grid:
            rs = [run_arm(it, A(side={"keep": "box", "shuf": False, "alpha": a_, "tau": t_})) for it in sel_items]
            scores[gname(a_, t_)] = sum(r["correct"] for r in rs)
            print(f"{time.strftime('%H:%M:%S')} side selection {gname(a_, t_)}: {scores[gname(a_, t_)]}/{len(rs)}", flush=True)
        best = max(grid, key=lambda g_: (scores[gname(*g_)], -g_[0]))
        side_sel = {"items": [it["id"] for it in sel_items], "scores": scores, "chosen": gname(*best)}
        keepn = lambda n: not n.startswith("side_") or n.endswith("@" + gname(*best))  # noqa: E731
        arms = {n: a for n, a in arms.items() if keepn(n)}
        eng_arms = {n: a for n, a in eng_arms.items() if keepn(n)}
    print(f"arms {list(arms)} span bands {[(b['hl'], b['dl'], b['base']) for b in bands]}", flush=True)

    # English cost: teacher-forced KL(host || arm) on held ordinary text, same injection code
    eng = {}
    if args.english_texts:
        STATE["stage"] = "english cost"
        recs = [json.loads(l) for l in Path(args.quiet_corpus).read_text().split("\n") if l.strip()]
        held = [r for r in recs if r["split"] == "held"]
        held = [held[i] for i in np.random.default_rng(args.seed + 1).permutation(len(held))[:args.english_texts]]
        with open(out / "english.jsonl", "w") as f:
            for r in held:
                ids = htok.encode(r["text"], add_special_tokens=False)[:args.english_tokens + 8]
                pseudo = {"id": r["id"], "prompt": "", "answer": None}
                ref = run_arm(pseudo, arms.get("host", A()), forced=ids)
                for name, arm in eng_arms.items():
                    if name == "host":
                        continue
                    e = run_arm(pseudo, arm, forced=ids, ref_lps=ref)
                    eng.setdefault(name, []).append(e["kl_vs_host"])
                    f.write(json.dumps({"text": r["id"], "register": r["register"], "arm": name, **e}) + "\n")
                    f.flush()
                del ref
                mx.clear_cache()
                print(f"{time.strftime('%H:%M:%S')} english {r['id']} done", flush=True)
        STATE["stage"] = "test arms"

    test = [it for it in its if it["split"] == "test"][: args.test_limit or None]
    rows_f = open(out / "rows.jsonl", "w")
    done = {a: [] for a in arms}
    stopped = None
    item_s = []
    for it in test:  # item-major, so a budget stop leaves every arm on the same items
        need = max(item_s) if item_s else 0.0  # do not start an item that would not finish in budget
        if time.time() - t_start + need > args.budget_s:
            stopped = f"budget: {len(item_s)} test items done, next needs ~{need:.0f}s"
            break
        ti0 = time.time()
        for name, arm in arms.items():
            tg = time.time()
            r = run_arm(it, arm)
            r.update({"item": it["id"], "arm": name, "gen_s": round(time.time() - tg, 1),
                      "donor_correct": dtr.get(it["id"], {}).get("correct")})
            done[name].append(r)
            rows_f.write(json.dumps(r) + "\n")
            rows_f.flush()
            print(f"{time.strftime('%H:%M:%S')} {it['id']} {name}: pred {r['pred']!r} correct {r['correct']} "
                  f"tokens {r['tokens']} inj {r['injections']} sink_rms {r['sink_rms_mean']:.3g} "
                  f"desync {r['desync']} ({r['gen_s']}s)", flush=True)
        item_s.append(time.time() - ti0)
    rows_f.close()

    def acc(rs):
        if not rs:
            return {"n": 0}
        a = {"n": len(rs), "correct": sum(r["correct"] for r in rs), "acc": round(sum(r["correct"] for r in rs) / len(rs), 4)}
        if "tokens" in rs[0]:
            a |= {"answer_in_trace": sum(r["answer_in_digits"] for r in rs),
                  "tokens_mean": round(float(np.mean([r["tokens"] for r in rs])), 1), "hit_cap": sum(r["hit_cap"] for r in rs)}
        return a

    tested = {r["item"] for v in done.values() for r in v}
    summary = {"donor_ceiling_test": acc([{"correct": dtr[i]["correct"]} for i in tested if i in dtr]),
               "donor_train": acc([{"correct": dtr[it["id"]]["correct"]} for it in train]),
               "arms": {a: acc(v) for a, v in done.items()},
               "english_kl_vs_host": {a: round(float(np.mean(v)), 5) for a, v in eng.items()},
               "maps": {hl: {"donor_layer": m["dl"], "fit_r2_train": round(m["fit_r2"], 4), "host_rms": m["host_rms"]}
                        for hl, m in maps.items()},
               "span_bands": bands,
               "inj_rms_mean": {a: round(float(np.mean([r["inj_rms_mean"] for r in v])), 4) if v else None
                                for a, v in done.items() if a != "host"},
               "sink_rms_mean": {a: round(float(np.mean([r["sink_rms_mean"] for r in v])), 4) if v else None
                                 for a, v in done.items() if arms[a]["sink"]},
               "english_tokens_per_text": args.english_tokens, "english_texts": args.english_texts,
               "desync_items": {a: sum(r["desync"] for r in v) for a, v in done.items() if a != "host"},
               "tok_s": {a: round(sum(r["tokens"] for r in v) / max(sum(r["gen_s"] for r in v), 1e-9), 2)
                         for a, v in done.items() if v},
               "norm_budget": args.norm_budget,
               "norm_ratio_mean": {a: {l: round(float(np.mean([r["norm_ratio"][l] for r in v if l in r["norm_ratio"]])), 4)
                                       for l in sorted({l for r in v for l in r["norm_ratio"]})}
                                   for a, v in done.items() if v and any(r["norm_ratio"] for r in v)},
               "clamped_frac": {a: round(float(np.mean([r["clamped_frac"] for r in v])), 4)
                                for a, v in done.items() if v and a != "host" and not arms[a]["side"]},
               "side": {a: {"open_frac": round(float(np.mean([r["side_open_frac"] for r in v])), 4),
                            "mapped_mass": round(float(np.mean([r["side_mapped_mass"] or 0 for r in v])), 4),
                            "rebuilds": sum(r["rebuilds"] for r in v),
                            "resident_bytes_est": bytes_info["carved_resident_bytes_est"][arms[a]["side"]["keep"]]}
                        for a, v in done.items() if v and arms[a]["side"]},
               "side_selection": side_sel, "bytes": bytes_info, "task": args.task,
               "item_s": [round(x, 1) for x in item_s]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    STATE["stage"] = "done"
    (out / "run.json").write_text(json.dumps({
        "schema": "amoeba-hub-transfer/v4", "donor_checkpoint": d_ck, "host_checkpoint": h_ck,
        "donor_box_run": args.donor_box_run, "box_arm": args.box_arm, "box_units": int(box.sum()),
        "mlx2_source_revision": rev, "tree_dirty": dirty, "args": vars(args), "stopped": stopped,
        "mapping_only": "ridge maps fit on frozen residual streams; sink = closed-form ridge least squares on the "
                        "host's linearised final norm + unembedding; no weight updates",
        "profile_waiver": STATE["receipt"]["profile_waiver"], "wall_s": round(time.time() - t_start),
        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2)}, indent=1))
    print(json.dumps(summary["arms"]), flush=True)


def _sigterm(signum, frame):  # make SIGTERM an exception so the failure receipt below is written
    raise SystemExit(f"terminated by signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    try:
        main()
    except BaseException as e:  # keep a receipt for failed / interrupted runs (not for refusals before setup)
        if STATE.get("out") and STATE.get("stage") != "done":
            rec = dict(STATE["receipt"], error=f"{type(e).__name__}: {e}"[:2000], failed_stage=STATE["stage"],
                       wall_s=round(time.time() - STATE["t_start"]))
            Path(STATE["out"], "run.json").write_text(json.dumps(rec, indent=1))
        raise
