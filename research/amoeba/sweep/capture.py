"""Layer-streaming (sweep) topic capture driver. Contract: README.md in this directory.

  python -m research.amoeba.sweep.capture --family deepseek_v41 --model DIR --corpus corpus.jsonl \
      --out OUT --state-dir SCRATCH [--resume] [--limit N] [--parity N]

For each layer in order: (dense layers: a mu pass on the fit split), then every text's carried
state through the layer in same-length batches, resp rows written, next state stored on disk;
the layer is released before the next one. Resumable per layer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src", ROOT / "scripts", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import amoeba_topics_lib as T  # noqa: E402

SCHEMA = "mlx2.amoeba-topic-capture.v1"
SWEEP_SCHEMA = "mlx2.amoeba-sweep-capture.v1"


# ----------------------------------------------------------------------------- host signals
def swapouts_bytes():
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None
    page = int(re.search(r"page size of (\d+)", out).group(1)) if "page size of" in out else 16384
    m = re.search(r"Swapouts:\s+(\d+)", out)
    return int(m.group(1)) * page if m else None


def rss_bytes():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    except Exception:
        return None


# ----------------------------------------------------------------------------- carried state on disk
def _to_np(a):
    import mlx.core as mx
    if a.dtype == mx.bfloat16:
        return np.array(a.view(mx.uint16)), "bfloat16"
    data = np.array(a)
    return data, str(data.dtype)


def _from_np(a, dtype):
    import mlx.core as mx
    if dtype == "bfloat16":
        return mx.array(a).view(mx.bfloat16)
    return mx.array(a)


class LayerState:
    """Disk-backed carry for one layer boundary: L<l>/<key>.npy [S, *max tail], <key>.shape.npy."""

    def __init__(self, root: Path, S: int):
        self.root, self.S = Path(root), S
        self.arrays, self.shapes, self.dtypes = {}, {}, {}

    @property
    def done(self):
        return (self.root / "DONE.json").exists()

    def write(self, rows, carry):
        if not isinstance(carry, dict):
            carry = {"h": carry}
        self.root.mkdir(parents=True, exist_ok=True)
        for key, value in carry.items():
            data, dtype = _to_np(value)
            if data.shape[0] != len(rows):
                raise ValueError(f"carry {key!r}: leading axis {data.shape[0]} != batch {len(rows)}")
            if key not in self.arrays:
                np_dtype = np.uint16 if dtype == "bfloat16" else data.dtype
                self.arrays[key] = np.lib.format.open_memmap(self.root / f"{key}.npy", mode="w+", dtype=np_dtype,
                                                             shape=(self.S, *data.shape[1:]))
                self.shapes[key] = np.lib.format.open_memmap(self.root / f"{key}.shape.npy", mode="w+",
                                                             dtype=np.int32, shape=(self.S, data.ndim - 1))
                self.dtypes[key] = dtype
            mm = self.arrays[key]
            if dtype != self.dtypes[key] or data.ndim != mm.ndim or any(
                    a > b for a, b in zip(data.shape[1:], mm.shape[1:])):
                raise ValueError(f"carry {key!r}: shape {data.shape} / {dtype} does not fit {mm.shape} / "
                                 f"{self.dtypes[key]} (trailing dims must be non-decreasing in T; batches run "
                                 "longest first)")
            sl = (rows,) + tuple(slice(0, n) for n in data.shape[1:])
            mm[sl] = data
            self.shapes[key][rows] = np.array(data.shape[1:], np.int32)

    def seal(self, extra=None):
        for mm in list(self.arrays.values()) + list(self.shapes.values()):
            mm.flush()
        T.write_json(self.root / "DONE.json", {"keys": {k: self.dtypes[k] for k in self.arrays}, **(extra or {})})

    def open(self):
        meta = json.loads((self.root / "DONE.json").read_text())
        for key, dtype in meta["keys"].items():
            self.arrays[key] = np.load(self.root / f"{key}.npy", mmap_mode="r")
            self.shapes[key] = np.load(self.root / f"{key}.shape.npy", mmap_mode="r")
            self.dtypes[key] = dtype
        return self

    def read(self, rows):
        out = {}
        for key, mm in self.arrays.items():
            shape = self.shapes[key][rows[0]]
            if any((self.shapes[key][r] != shape).any() for r in rows):
                raise ValueError(f"carry {key!r}: mixed shapes inside one batch")
            sl = (np.asarray(rows),) + tuple(slice(0, int(n)) for n in shape)
            out[key] = _from_np(np.ascontiguousarray(mm[sl]), self.dtypes[key])
        return out

    def remove(self):
        self.arrays.clear()
        self.shapes.clear()
        shutil.rmtree(self.root, ignore_errors=True)


# ----------------------------------------------------------------------------- unit statistics
def unit_rows(kind, payload, width, mu_l=None):
    """[B, width] f32: per-text mean over positions >= 1 of the unit signal."""
    import mlx.core as mx
    if kind == "dense":
        a = payload["act"][:, 1:].astype(mx.float32)
        if mu_l is not None:
            a = a - mx.array(np.asarray(mu_l, np.float32)[:a.shape[-1]])
        return mx.abs(a).mean(1)
    inds = payload["inds"][:, 1:]
    if "signal" in payload:
        sig = payload["signal"][:, 1:].astype(mx.float32)
    else:
        sig = mx.linalg.norm(payload["rows"][:, 1:].astype(mx.float32), axis=-1) * payload["w"][:, 1:].astype(mx.float32)
    B, Tm1, k = sig.shape
    acc = mx.zeros((B, width), mx.float32)
    for j in range(k):
        oh = (inds[..., j][..., None] == mx.arange(width)).astype(mx.float32)
        acc = acc + (oh * sig[..., j][..., None]).sum(1)
    return acc / Tm1


def batches_longest_first(seqs, rows_cap, max_texts):
    by = {}
    for j, s in enumerate(seqs):
        by.setdefault(len(s), []).append(j)
    for ln in sorted(by, reverse=True):
        js = by[ln]
        bs = max(1, min(max_texts, rows_cap // ln))
        for b in range(0, len(js), bs):
            yield js[b:b + bs]


NLL_CONVENTION = "targets 2..T-1"      # recorded in meta.json (nll_full); captures without the key scored 1..T-1


def nll_rows(logits, ids):
    """Per-text mean NLL of targets 2..T-1 from positions 1..T-2: the resident capture's convention
    (carve.host.TopicsBackend.nll, child_rolling), so nll_full.npy compares across writers. Position 0's
    prediction (the unconditioned first-token loss) is not scored."""
    import mlx.core as mx
    lg = logits[:, 1:-1].astype(mx.float32)
    tgt = ids[:, 2:]
    lse = mx.logsumexp(lg, axis=-1)
    pick = mx.take_along_axis(lg, tgt[..., None].astype(mx.int32), axis=-1)[..., 0]
    return (lse - pick).mean(1)


def nll_meta(receipt: dict) -> dict:
    """meta.json's ``nll_full`` entry, only when the receipt says a sweep at this convention saved nll_full.npy:
    an older capture resumed after its last layer keeps its array and must not get the new label."""
    conv = ((receipt or {}).get("nll_full") or {}).get("convention")
    if not conv:
        return {}
    return {"nll_full": {"convention": conv,
                         "note": "per-text mean NLL, as the resident capture (TopicsBackend.nll); a meta.json "
                                 "without this key is from a capture that scored targets 1..T-1 (position 0's "
                                 "prediction included): do not compare the two"}}


# ----------------------------------------------------------------------------- driver
def sweep(fam, seqs, items_split, out: Path, state_dir: Path, *, log=print, resume=False, max_texts=64,
          max_layers=None, swap_abort_bytes=512 << 20, receipt=None, parity=False):
    """Run the layer-major sweep. Returns the final carry LayerState (parity) or None."""
    import mlx.core as mx
    S, L = len(seqs), fam.L
    widths = [int(w) for w in fam.widths]
    W = max(widths)
    rows_cap = int(getattr(fam, "batch_rows", 1024) or 1024)
    batches = list(batches_longest_first(seqs, rows_cap, max_texts))
    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if (resume and progress_path.exists()) else {"done": []}
    mu = np.load(out / "mu.npy") if (resume and (out / "mu.npy").exists()) else np.zeros((L, W), np.float32)
    receipt = receipt if receipt is not None else {}
    receipt.setdefault("layers", {})
    fit = [j for j in range(S) if items_split[j] == "fit"]
    fit_batches = list(batches_longest_first([seqs[j] for j in fit], rows_cap, max_texts))
    done = set(progress["done"])
    start = 0
    while start in done:
        start += 1
    if start:
        log(f"resume: layers 0..{start - 1} complete")
    if resume and state_dir.exists():
        # a kill between "layer done" and the removal of its input leaves that input behind
        for d in state_dir.glob("L[0-9][0-9]"):
            if int(d.name[1:]) != start:
                shutil.rmtree(d, ignore_errors=True)

    def state_at(l):
        return LayerState(state_dir / f"L{l:02d}", S)

    import hashlib
    run_id = hashlib.sha256(json.dumps([[int(t) for t in q] for q in seqs]).encode()).hexdigest()[:16]

    def check(state):
        meta = json.loads((state.root / "DONE.json").read_text())
        if meta.get("run_id") != run_id or meta.get("sequences") != S:
            raise SystemExit(f"carried state {state.root} belongs to another run "
                             f"({meta.get('run_id')}, {meta.get('sequences')} texts); not resuming it")

    cur = state_at(start)
    if start == 0:
        if resume and cur.done:
            check(cur)
            cur.open()
        else:
            cur.remove()                      # never reuse a state a different run left behind
            cur = state_at(0)
            for g in batches:
                ids = mx.array(np.array([seqs[j] for j in g], np.int32))
                c = fam.embed(ids)
                mx.eval(c)
                cur.write(np.array(g), c)
            cur.seal({"layer_input": 0, "run_id": run_id, "sequences": S})
            log(f"embedded {S} texts")
    else:
        if not cur.done:
            raise SystemExit(f"resume: carried state {cur.root} missing or incomplete")
        check(cur)
        cur.open()
    stop = L if max_layers is None else min(L, start + max_layers)
    swap0 = swapouts_bytes()
    for l in range(start, stop):
        t0 = time.time()
        sw_a = swapouts_bytes()
        mx.reset_peak_memory()
        kind = fam.kinds[l]
        prep = getattr(fam, "prepare_layer", None)
        prepared = prep(l, seqs, state_dir) if prep is not None else None
        if prepared:
            log(f"  L{l:02d} prepared in {time.time() - t0:.0f}s: {prepared}")
        if kind == "dense":
            acc, cnt = None, 0
            for g in fit_batches:
                rows = np.array([fit[j] for j in g])
                _, p = fam.run_layer(l, cur.read(rows))
                s = p["act"][:, 1:].astype(mx.float32).sum((0, 1))
                acc = s if acc is None else acc + s
                cnt += p["act"].shape[0] * (p["act"].shape[1] - 1)
            mu[l, :widths[l]] = np.array(acc / max(cnt, 1))
            np.save(out / "mu.npy", mu)
        resp = np.lib.format.open_memmap(out / f"resp_L{l:02d}.npy", mode="w+", dtype=np.float16, shape=(S, W))
        nxt = state_at(l + 1)
        nxt.remove()
        nxt = state_at(l + 1)
        nll = np.zeros(S, np.float32) if (l == L - 1 and hasattr(fam, "final_logits")) else None
        n_done = 0
        for g in batches:
            rows = np.array(g)
            carry, payload = fam.run_layer(l, cur.read(rows))
            if kind is not None:
                r = unit_rows(kind, payload, widths[l], mu[l] if kind == "dense" else None)
                resp[rows, :widths[l]] = np.array(r).astype(np.float16)
            if nll is not None:
                lg = fam.final_logits(carry)
                nll[rows] = np.array(nll_rows(lg, carry["ids"] if isinstance(carry, dict) else
                                              mx.array(np.array([seqs[j] for j in g], np.int32))))
            nxt.write(rows, carry)
            n_done += len(g)
            if n_done % 1024 < len(g):
                sw = swapouts_bytes()
                log(f"  L{l:02d} {n_done}/{S} ({time.time() - t0:.0f}s, peak {mx.get_peak_memory() / 2**30:.1f} GB)")
                if sw is not None and sw_a is not None and sw - sw_a > swap_abort_bytes:
                    raise SystemExit(f"swap grew {(sw - sw_a) / 2**20:.0f} MB during layer {l}; aborting")
        resp.flush()
        del resp
        nxt.seal({"layer_input": l + 1, "run_id": run_id, "sequences": S})
        if not parity:
            fam.release_layer(l)
        mx.clear_cache()
        if nll is not None:
            np.save(out / "nll_full.npy", nll)
            # on the receipt, where the array is written: meta.json gets the label only for numbers this code made
            receipt["nll_full"] = {"convention": NLL_CONVENTION}
        cur = nxt.open() if not nxt.arrays else nxt
        sw_b = swapouts_bytes()
        rec = {"seconds": round(time.time() - t0, 1), "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2),
               "swapouts_start": sw_a, "swapouts_end": sw_b}
        sr = getattr(fam, "stream_receipt", None)
        if sr is not None:
            rec["experts"] = sr()
        if prepared:
            rec["prepared"] = prepared
        receipt["layers"][str(l)] = rec
        progress["done"] = sorted(set(progress["done"]) | {l})
        T.write_json(progress_path, progress)
        T.write_json(out / "receipt.json", receipt)
        if not parity:
            state_at(l).remove()              # only after progress names layer l done
        log(f"layer {l} done in {rec['seconds']}s (peak {rec['mlx_peak_gb']} GB)")
        if sw_b is not None and sw_a is not None and sw_b - sw_a > swap_abort_bytes:
            raise SystemExit(f"swap grew {(sw_b - sw_a) / 2**20:.0f} MB during layer {l}; aborting")
    receipt["swapouts_run_start"] = swap0
    receipt["swapouts_run_end"] = swapouts_bytes()
    return cur if stop == L else None


def run_parity(fam, seqs, out: Path, state_dir: Path, log=print, layers=None):
    """Sweep N texts and compare final logits to the ordinary forward (of the first ``layers``
    blocks when given: the family's L is set to it for the sweep)."""
    import mlx.core as mx
    split = ["fit"] * len(seqs)
    full_L = fam.L
    if layers:
        fam.L = int(layers)
    np.save(out / "mu.npy", np.zeros((fam.L, max(fam.widths)), np.float32))
    try:
        final = sweep(fam, seqs, split, out, state_dir, log=log, max_texts=len(seqs), parity=True)
    finally:
        fam.L = full_L
    rows = []
    for j, s in enumerate(seqs):
        ids = mx.array(np.array([s], np.int32))
        carry = final.read(np.array([j]))
        got = fam.final_logits(carry).astype(mx.float32)
        ref = (fam.reference_logits(ids, layers=layers) if layers else fam.reference_logits(ids)).astype(mx.float32)
        mx.eval(got, ref)
        g, r = np.array(got[0]), np.array(ref[0])
        d = np.abs(g - r)
        rows.append({"text": j, "tokens": len(s), "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                     "ref_max_abs": float(np.abs(r).max()), "bitwise_equal": bool(np.array_equal(g.view(np.uint32), r.view(np.uint32))),
                     "top1_agree": float((g.argmax(-1) == r.argmax(-1)).mean())})
        log(f"parity text {j}: max|d| {rows[-1]['max_abs']:.3g} top1 {rows[-1]['top1_agree']:.3f} "
            f"bitwise {rows[-1]['bitwise_equal']}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-dir", required=True, help="scratch for carried states (<= two layers)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--min-len", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-texts", type=int, default=64, help="texts per run_layer call (also capped by batch_rows)")
    ap.add_argument("--max-layers", type=int, default=None, help="stop after this many layers (resume later)")
    ap.add_argument("--parity", type=int, default=0, help="parity check on the first N texts instead of a capture")
    ap.add_argument("--parity-layers", type=int, default=0, help="parity on the first K layers only")
    ap.add_argument("--check-signal", action="store_true", help="family: compare routed signal to a per-slot probe")
    ap.add_argument("--ffn-texts", type=int, default=None, help="family: texts per FFN call (1 = ordinary shapes)")
    ap.add_argument("--expert-budget-gb", type=float, default=24)
    ap.add_argument("--memory-gb", type=float, default=96, help="mx.set_memory_limit")
    ap.add_argument("--cache-gb", type=float, default=4, help="mx.set_cache_limit")
    ap.add_argument("--swap-abort-mb", type=float, default=512)
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.resume:
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = T.fresh_dir(out)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    log = T.make_log(out)
    import signal

    def _term(signum, frame):
        # SIGTERM/SIGHUP: unwind through finally (receipt, adapter close); the current layer's
        # partial resp/state are redone on --resume, finished layers are kept
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGHUP, _term)
    (out / "capture.pid").write_text(f"{os.getpid()}\n")
    import mlx.core as mx
    mx.set_memory_limit(int(args.memory_gb * 2**30))
    mx.set_cache_limit(int(args.cache_gb * 2**30))
    rec = {"schema": SCHEMA, "sweep_schema": SWEEP_SCHEMA, "args": vars(args),
           "mlx2_source_revision": T.git_rev(), "corpus_sha256": T.sha_file(args.corpus)}
    receipt = json.loads((out / "receipt.json").read_text()) if (args.resume and (out / "receipt.json").exists()) else {}
    receipt.setdefault("runs", []).append({"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": os.getpid(),
                                           "swapouts_start": swapouts_bytes(), "rev": T.git_rev()})
    t_load = time.time()
    from research.amoeba.sweep.family import HOST_UNIT_FAMILIES, family_options, load_family
    if args.family in HOST_UNIT_FAMILIES:
        raise SystemExit(f"--family {args.family} needs the begin_layer driver: run it through "
                         f"mlx2.experimental.amoeba.child_rolling (mlx2 amoeba), not this capture driver")
    opts = {"expert_budget_gb": args.expert_budget_gb}
    if args.ffn_texts or args.parity:
        opts["ffn_texts"] = args.ffn_texts or 1
    opts, dropped = family_options(args.family, opts)      # only the options this family's loader declares
    if dropped:
        log(f"--family {args.family} does not take {sorted(dropped)}: not passed")
    rec["family_options"] = {"passed": opts, "not_taken": sorted(dropped)}
    fam = load_family(args.family, args.model, **opts)
    if hasattr(fam, "begin_layer"):                    # this driver never calls it (README: child_rolling extension)
        close = getattr(fam, "close", None)
        if close is not None:
            close()
        raise SystemExit(f"--family {args.family} defines begin_layer, which this capture driver never calls: run "
                         f"it through mlx2.experimental.amoeba.child_rolling (mlx2 amoeba)")
    receipt["runs"][-1]["load_s"] = round(time.time() - t_load, 1)
    log(f"loaded {fam.checkpoint} in {receipt['runs'][-1]['load_s']}s: {fam.L} layers, widths {sorted(set(fam.widths))}")
    rec.update(checkpoint=fam.checkpoint, fingerprint=fam.fingerprint, policy=getattr(fam, "policy", None))
    items = T.load_items(args.corpus)
    try:
        if args.parity:
            toks = T.tokenize_items(fam.tok.encode, items[: max(args.parity * 4, 16)], args.seq_len, args.min_len)
            seqs = [ids for _, ids in toks][: args.parity]
            if args.check_signal:
                fam.check_signal = True
            rows = run_parity(fam, seqs, out, state_dir, log=log, layers=args.parity_layers or None)
            T.write_json(out / "parity.json", {**rec, "texts": rows, "counters": getattr(fam, "counters", None),
                                               "ffn_texts": opts.get("ffn_texts"), "layers": args.parity_layers or fam.L,
                                               "receipt": receipt,
                                               "max_abs": max(r["max_abs"] for r in rows),
                                               "all_bitwise": all(r["bitwise_equal"] for r in rows)})
            return
        if args.limit:
            items = items[: args.limit]
        toks = T.tokenize_items(fam.tok.encode, items, args.seq_len, args.min_len)
        seqs = [ids for _, ids in toks]
        log(f"{len(seqs)}/{len(items)} items with >= {args.min_len} tokens")
        widths = [int(w) for w in fam.widths]
        rec.update(sequences=len(seqs), layers=fam.L, neurons=max(widths), widths=widths, unit_kinds=list(fam.kinds),
                   unit=getattr(fam, "unit", "per-text mean over positions >= 1 of the unit signal, fp16"))
        rec.update(nll_meta(receipt))                  # a resumed capture whose last layer already wrote it
        if args.resume and (out / "meta.json").exists():
            old = json.loads((out / "meta.json").read_text())
            keys = ("corpus_sha256", "checkpoint", "fingerprint", "sequences")
            if any(old.get(k) != rec.get(k) for k in keys) or any(
                    old.get("args", {}).get(k) != getattr(args, k.replace("-", "_"))
                    for k in ("seq_len", "min_len", "limit", "family")):
                raise SystemExit("--resume: OUT/meta.json describes a different run; refusing to mix outputs")
            if old.get("complete"):
                # A completed capture has already released its final scratch state.
                log("capture already complete; preserving outputs")
                return
        T.write_json(out / "items.json", [{"id": it["id"], "topics": it["topics"], "split": it["split"],
                                           "n_tokens": len(s)} for it, s in toks])
        if not (out / "mu.npy").exists():
            np.save(out / "mu.npy", np.zeros((fam.L, max(widths)), np.float32))
        T.write_json(out / "meta.json", {**rec, "complete": False})
        t0 = time.time()
        final = sweep(fam, seqs, [it["split"] for it, _ in toks], out, state_dir, log=log, resume=args.resume,
                      max_texts=args.max_texts, max_layers=args.max_layers,
                      swap_abort_bytes=int(args.swap_abort_mb * 2**20), receipt=receipt)
        complete = final is not None
        if complete:
            final.remove()
        rec.update(nll_meta(receipt))
        rec["mu"] = {"source": "computed on the corpus fit split, per dense layer, before its resp pass",
                     "dense_layers": [l for l, k in enumerate(fam.kinds) if k == "dense"]}
        rec["capture_s"] = round(sum(v["seconds"] for v in receipt["layers"].values()), 1)
        rec["mlx_peak_gb"] = max(v["mlx_peak_gb"] for v in receipt["layers"].values())
        T.write_json(out / "meta.json", {**rec, "complete": complete})
        log(f"{'done' if complete else 'stopped'}: run {time.time() - t0:.0f}s")
    finally:
        receipt["runs"][-1]["swapouts_end"] = swapouts_bytes()
        receipt["runs"][-1]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        receipt["runs"][-1]["rss_max_bytes"] = rss_bytes()
        T.write_json(out / "receipt.json", receipt)
        (out / "capture.pid").unlink(missing_ok=True)
        close = getattr(fam, "close", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
