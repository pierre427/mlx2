"""Rolling layer-window capture (child process): the map capture for models that do not fit in memory.

mapping-workflow s8, "Models that don't fit in memory" (Pierre 2026-11-16), as a driver over the sweep family
interface (research/amoeba/sweep/README.md, v1): ``--family host`` (any adapter that accepts a weight-streaming
request: research/amoeba/sweep/host_family.py) or a model family of the maps session's (deepseek_v41, mimo_v2).
Same output as child_capture (the topic-capture format), computed layer-major:

1. The corpus state (the family's carry: one row per token, every residual stream it needs) stays resident for a
   chunk of texts (``--carry resident``), or lives in the sweep's disk-backed LayerState between layers
   (``--carry memmap``: one weight read however large the state; the plan picks it when chunking would cost more
   than two weight reads).
2. For each layer l: wait for layer l's read-ahead (the prefetch stall), run it over EVERY batch, updating the state,
   while the next K layers' checkpoint bytes are read into the page cache (families that expose
   ``prefetch_ranges``); every weight is read once per chunk, and the reads overlap compute.
3. Per-text unit PEAKS over positions >= 1 (mapping-workflow s5.1.1) from the family's carve-style payload: dense
   |act - mu| finished exactly after the pass from per-text max / min and the fit mean (one pass); MoE neuron units
   (host family) |w * h|; MoE expert units w * ||row|| (or the family's reduced ``signal``).
4. Parity gate, every run (a fresh pass, never a stored one): the first --parity-texts texts through this schedule
   and through the family's ordinary ``reference_logits`` (the first --parity-layers blocks when given); refuses
   unless the logits are bitwise equal or within rel 1e-3 with top-1 agreement 1.0 (the sweep's parity_check exact
   criteria). Recorded in meta.json and parity.json.
5. Per-layer metrics as progress events (``layer``): load s, compute s, prefetch stall s, state bytes, chunk.

    python -m mlx2.experimental.amoeba.child_rolling --family host --model M --corpus C --out DIR --window K \
        --chunks N --carry resident|memmap --stream-mode dense_layers|moe_experts --ceiling-gb G [--cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .progress import child_event


class Prefetcher:
    """Reads the byte ranges of layers ahead of the one running, so its load is a copy from the page cache."""

    def __init__(self, ranges: dict, chunk: int = 16 << 20):
        self.ranges = ranges                       # layer -> [(file, begin, end)]
        self.chunk = chunk
        self.q = queue.Queue()
        self.done = {}
        self.want = set()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.bytes = 0
        self.error = None
        self.closed = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def reset(self):
        """A new chunk reads every layer again: forget what was read ahead for the previous one."""
        with self.lock:
            self.want.clear()
            self.done.clear()

    def request(self, l):
        with self.lock:
            if l in self.ranges and l not in self.want:
                self.want.add(l)
                self.q.put(l)

    def _run(self):
        try:
            self._read_ranges()
        except Exception as exc:  # noqa: BLE001 - deliver every worker failure to the waiting capture
            with self.cv:
                self.error = exc
                self.cv.notify_all()

    def _read_ranges(self):
        buf = bytearray(self.chunk)
        while not self.closed.is_set():
            l = self.q.get()
            if l is None:
                return
            for f, b, e in self.ranges.get(l, []):
                with open(f, "rb", buffering=0) as shard:
                    pos = b
                    while pos < e:
                        if self.closed.is_set():
                            return
                        n = os.preadv(shard.fileno(), [memoryview(buf)[:min(self.chunk, e - pos)]], pos)
                        if n <= 0:
                            raise OSError(f"unexpected end of {f} at {pos}, prefetch range ends at {e}")
                        pos += n
                        self.bytes += n
            with self.cv:
                self.done[l] = time.time()
                self.cv.notify_all()

    def wait(self, l, timeout=3600.0) -> float:
        """Seconds the caller waits for layer l's read-ahead (0 when it is done or was never requested)."""
        t0 = time.monotonic()
        with self.cv:
            if l not in self.want:
                return 0.0
            if not self.cv.wait_for(lambda: l in self.done or self.error is not None or self.closed.is_set(), timeout):
                raise TimeoutError(f"prefetch timed out for layer {l}")
            if self.error is not None:
                raise self.error
            if l not in self.done:
                raise RuntimeError(f"prefetch closed before layer {l} was ready")
        return round(time.monotonic() - t0, 4)

    def close(self):
        with self.cv:
            self.closed.set()
            self.cv.notify_all()
        self.q.put(None)
        self.th.join()


def layer_ranges(model_path: Path, L: int, experts_only: bool) -> dict:
    from .inspection import _LAYER_KEY, _NOT_DECODER, safetensors_index
    out = {}
    for k, f, b, e in safetensors_index(model_path):
        m = None if _NOT_DECODER.search(k) else _LAYER_KEY.search(k)
        if not m or "mtp" in k:
            continue
        if experts_only and not any(t in k for t in (".experts.", ".switch_mlp.", ".switch_glu.")):
            continue
        out.setdefault(int(m.group(1)), []).append((f, b, e))
    return out


# ------------------------------------------------------------------ layer-major schedule

def batches_by_len(idx_seqs, bs):
    by = {}
    for j, s in idx_seqs:
        by.setdefault(len(s), []).append(j)
    out = []
    for ln, js in sorted(by.items(), reverse=True):       # longest first: the memmap carry sizes from the first write
        for b in range(0, len(js), bs):
            out.append(js[b:b + bs])
    return out


def resident_batches(seqs, bs):
    """The resident capture's batches exactly (scripts/amoeba_topics_lib.batches_by_len: equal-length groups in index
    order, ascending length, bs at a time). A text's batch neighbours and the batch size choose the kernels' tiling,
    so rolling keeps the same batches: the 27B parity of 2026-11-16 showed interleaved chunks change the numbers."""
    by = {}
    for j, s in enumerate(seqs):
        by.setdefault(len(s), []).append(j)
    out = []
    for ln in sorted(by):
        js = by[ln]
        for b in range(0, len(js), bs):
            out.append(js[b:b + bs])
    return out


def chunk_batches(seqs, batches, chunks):
    """Whole batches into ``chunks`` groups of about equal token count (largest batches placed first); within a
    group, longest batches first (the memmap carry sizes from the first write)."""
    order = sorted(range(len(batches)), key=lambda b: -sum(len(seqs[j]) for j in batches[b]))
    groups = [[] for _ in range(max(1, chunks))]
    tok = [0] * len(groups)
    for b in order:
        g = tok.index(min(tok))
        groups[g].append(batches[b])
        tok[g] += sum(len(seqs[j]) for j in batches[b])
    return [sorted(g, key=lambda js: (-len(seqs[js[0]]), js[0])) for g in groups if g]


def contiguous_groups(seqs, batches, chunks):
    """``batches`` (in resident order) cut into at most ``chunks`` contiguous runs of about equal token count, each
    run longest batch first for the carry. Processing the runs in order adds the per-batch sums in the resident
    order exactly, across chunks too (the accumulator carries over)."""
    if not batches:
        return []
    tok = [sum(len(seqs[j]) for j in b) for b in batches]
    total, n = sum(tok), max(1, min(chunks, len(batches)))
    runs, cur, acc_t, k = [], [], 0, 1
    for b, t in zip(batches, tok):
        cur.append(b)
        acc_t += t
        if k < n and acc_t >= total * k / n:
            runs.append(cur)
            cur, k = [], k + 1
    if cur:
        runs.append(cur)
    return [sorted(r, key=lambda js: (-len(seqs[js[0]]), js[0])) for r in runs]


def chunk_texts(seqs, chunks, bs=8):
    """Text indices per chunk, built from whole resident batches (kept for callers that need index groups)."""
    return [sorted(j for js in g for j in js) for g in chunk_batches(seqs, resident_batches(seqs, bs), chunks)]


def payload_signal(kind, payload, width):
    """Per text [B, width] float32 peaks over positions >= 1; for dense, (max act, min act, sum act) instead."""
    import mlx.core as mx
    if kind == "dense":
        x = payload["act"][:, 1:].astype(mx.float32)
        return x.max(1), x.min(1), x.sum(1)
    inds = payload["inds"][:, 1:]
    B = inds.shape[0]
    if kind == "neurons":
        F = int(payload["F"])
        w = payload["w"][:, 1:].astype(mx.float32)
        act = mx.abs(payload["h"][:, 1:].astype(mx.float32) * w[..., None])           # B, T-1, k, F
        idx = inds[..., None] * F + mx.arange(F) + (mx.arange(B) * width)[:, None, None, None]
        return mx.zeros((B * width,), mx.float32).at[idx.reshape(-1)].maximum(act.reshape(-1)).reshape(B, width)
    if "signal" in payload:
        sig = payload["signal"][:, 1:].astype(mx.float32)
    else:
        sig = mx.linalg.norm(payload["rows"][:, 1:].astype(mx.float32), axis=-1) * payload["w"][:, 1:].astype(mx.float32)
    idx = inds + (mx.arange(B) * width)[:, None, None]
    return mx.zeros((B * width,), mx.float32).at[idx.reshape(-1)].maximum(sig.reshape(-1)).reshape(B, width)


def _ids_of(seqs, js):
    import mlx.core as mx
    return mx.array(np.array([seqs[j] for j in js], np.int32))


def parity(fam, seqs, layers, log=print):
    """Fresh parity, every run: this schedule's logits vs the family's ordinary forward on ONE batch of the given texts
    (cropped to their common length), so each side reads the weights once."""
    import mlx.core as mx
    n = min(len(s) for s in seqs)
    batch = [s[:n] for s in seqs]
    full_L = fam.L
    if layers:
        fam.L = int(layers)
    try:
        ids = mx.array(np.array(batch, np.int32))
        carry = fam.embed(ids)
        for l in range(fam.L):
            if hasattr(fam, "begin_layer"):
                fam.begin_layer(l)
            try:
                carry, _ = fam.run_layer(l, carry)
            finally:
                fam.release_layer(l)
        got = fam.final_logits(carry).astype(mx.float32)
        ref = (fam.reference_logits(ids, layers=layers) if layers else fam.reference_logits(ids)).astype(mx.float32)
        mx.eval(got, ref)
    finally:
        fam.L = full_L
    if got.shape != ref.shape or got.shape[0] != len(batch):
        raise SystemExit(f"rolling parity: logit shapes {got.shape} and {ref.shape} for batch {len(batch)}")
    rows = []
    for j in range(len(batch)):
        g, r = np.array(got[j]), np.array(ref[j])
        # A shared -inf can mask a vocabulary entry. Each position still needs a
        # finite candidate, and NaN/+inf cannot define a softmax distribution.
        finite = np.isfinite(r)
        usable = bool(np.all(finite.any(axis=-1)) and np.all(np.isfinite(g) == finite)
                      and np.all(np.isneginf(g[~finite])) and np.all(np.isneginf(r[~finite])))
        max_abs = float(np.abs(g[finite] - r[finite]).max()) if usable else None
        ref_max_abs = float(np.abs(r[finite]).max()) if usable else None
        bitwise = bool(np.array_equal(g.view(np.uint32), r.view(np.uint32)))
        top1 = float((g.argmax(-1) == r.argmax(-1)).mean())
        ok = usable and (bitwise or (max_abs <= 1e-3 * max(ref_max_abs, 1e-30) and top1 == 1.0))
        rows.append({"text": j, "tokens": n, "usable_logits": usable, "max_abs": max_abs, "ref_max_abs": ref_max_abs,
                     "bitwise_equal": bitwise, "top1_agree": top1, "ok": ok})
        log(f"parity text {j}: max|d| {max_abs} top1 {rows[-1]['top1_agree']:.3f} "
            f"bitwise {rows[-1]['bitwise_equal']}")
    return {"texts": rows, "layers": layers or full_L, "batch": len(batch), "ok": bool(rows) and all(t["ok"] for t in rows),
            "criteria": "usable logits, matching vocabulary masks, and bitwise equality or rel <= 1e-3 with top-1 agreement 1.0"}


class _Carry:
    """The corpus state between layers: resident per batch, or the sweep's disk-backed LayerState (two boundaries)."""

    def __init__(self, mode, S, root):
        self.mode, self.S, self.root = mode, S, Path(root)
        self.mem = {}
        self.cur = self.nxt = None
        self.l = 0
        self.rows = 0

    def begin_layer(self, l):
        if self.mode == "memmap":
            from research.amoeba.sweep.capture import LayerState
            if self.nxt is not None:
                self.nxt.seal()
                if self.cur is not None:
                    self.cur.remove()
                self.cur = self.nxt.open() if hasattr(self.nxt, "open") else self.nxt
            self.nxt = LayerState(self.root / f"L{l + 1}", self.S)
        self.l = l

    def get(self, bi, js):
        if self.mode == "resident":
            return self.mem[bi]
        return self.cur.read(np.asarray(js))

    def put(self, bi, js, carry):
        if self.mode == "resident":
            self.mem[bi] = carry
        else:
            self.nxt.write(np.asarray(js), carry)

    def first(self, bi, js, carry):
        self.rows = getattr(self, "rows", 0) + len(js)
        if self.mode == "resident":
            self.mem[bi] = carry
            return
        if self.cur is None:
            from research.amoeba.sweep.capture import LayerState
            self.cur = LayerState(self.root / "L0", self.S)
        self.cur.write(np.asarray(js), carry)

    def seal_first(self):
        if self.mode == "memmap":
            self.cur.seal()
            self.cur = self.cur.open()

    def nbytes(self):
        if self.mode == "resident":
            return int(sum(sum(v.nbytes for v in (c.values() if isinstance(c, dict) else [c])) for c in self.mem.values()))
        total = sum(mm.nbytes for mm in (self.cur.arrays.values() if self.cur else []))
        return int(total * self.rows / max(1, self.S))   # the store is sized to S rows; only these were written

    def close(self):
        if self.mode == "memmap":
            import shutil
            shutil.rmtree(self.root, ignore_errors=True)
        self.mem.clear()


def _layer_batches(fam, l, batches, store, kinds, widths, seqs, fit, mx_t, mn_t, fsum, fcnt, resp, times):
    import mlx.core as mx
    for bi, js in enumerate(batches):
        t0 = time.time()
        carry, pay = fam.run_layer(l, store.get(bi, js))
        if kinds[l] is not None and pay is not None:
            sig = payload_signal(kinds[l], pay, widths[l])
            if kinds[l] == "dense":
                mx.eval(*sig)
                mxa, mna, sm = (np.array(x) for x in sig)
                mx_t[l][js] = mxa
                mn_t[l][js] = mna
                r_ = [r for r, j in enumerate(js) if fit[j]] if fit is not None else []
                if r_:
                    fsum[l, :widths[l]] += sm[r_].sum(0)
                    fcnt[l] += len(r_) * (len(seqs[js[0]]) - 1)
            else:
                mx.eval(sig)
                resp[l][js, :sig.shape[-1]] = np.array(sig).astype(np.float16)
        store.put(bi, js, carry)
        times.append(time.time() - t0)


def _fit_layer(fam, l, fbatches, forder, fstore, kinds, acc, cnt, times):
    """The fit-only pass at layer l, under the same loaded weights: the fit texts in the resident capture's own
    fit-only batches, the dense mean accumulated as child_capture does (a float32 sum over batch and positions >= 1
    per batch, the batch sums added in float32 in the resident order). Batches run in carry order (longest first:
    the memmap carry sizes from its first write); only the additions follow the resident order."""
    import mlx.core as mx
    vs = {}
    for bi, js in enumerate(fbatches):
        t0 = time.time()
        carry, pay = fam.run_layer(l, fstore.get(bi, js))
        if kinds[l] == "dense" and pay is not None:
            vs[bi] = pay["act"][:, 1:].astype(mx.float32).sum((0, 1))
            mx.eval(vs[bi])
            cnt[l] += pay["act"].shape[0] * (pay["act"].shape[1] - 1)
        fstore.put(bi, js, carry)
        times.append(time.time() - t0)
    for bi in forder:
        if bi in vs:
            acc[l] = vs[bi] if acc[l] is None else acc[l] + vs[bi]
    if acc[l] is not None:
        mx.eval(acc[l])


def run(fam, seqs, fit, out: Path, *, window, chunks, carry_mode="resident", prefetch=None, max_texts=64, log=print,
        mu_pass="fit"):
    """The layer-major pass over a sweep family. Writes resp_L*.npy; returns (receipt, mu [L, W], nll [S] or None).

    mu_pass "fit" (default): the dense fit means come from a second carry of the fit texts in the resident capture's
    fit-only batches, run under each layer's weights as they are loaded (no extra weight read; extra compute for the
    fit split). "inline": from the fit rows of the main batches (one pass; its batching differs from the resident
    reference's, ~4e-3 of a layer's largest mean on the 27B, 2026-11-16)."""
    import mlx.core as mx
    S, L = len(seqs), fam.L
    widths = [int(w) for w in fam.widths]
    W = max(widths)
    kinds = list(fam.kinds)
    resp = [np.lib.format.open_memmap(out / f"resp_L{l:02d}.npy", mode="w+", dtype=np.float16, shape=(S, widths[l]))
            for l in range(L)]
    tmp = out / ".rolling"
    tmp.mkdir(exist_ok=True)
    mx_t = {l: np.lib.format.open_memmap(tmp / f"max_L{l:02d}.npy", mode="w+", dtype=np.float32, shape=(S, widths[l]))
            for l in range(L) if kinds[l] == "dense"}
    mn_t = {l: np.lib.format.open_memmap(tmp / f"min_L{l:02d}.npy", mode="w+", dtype=np.float32, shape=(S, widths[l]))
            for l in range(L) if kinds[l] == "dense"}
    fsum = np.zeros((L, W), np.float64)
    fcnt = np.zeros(L, np.float64)
    has_logits = callable(getattr(fam, "final_logits", None))
    nll = np.zeros(S, np.float32) if has_logits else None
    rows_cap = int(getattr(fam, "batch_rows", 4096))
    bs = max(1, min(max_texts, rows_cap // max(1, max(len(x) for x in seqs))))
    if bs < max_texts:     # child_capture uses --bs unclipped: the batches (and so the numbers) differ from resident
        log(f"warning: batch {max_texts} clipped to {bs} by the family's {rows_cap}-row cap: batches differ from the "
            "resident capture's (parity is not expected to be exact)")
    groups = chunk_batches(seqs, resident_batches(seqs, bs), chunks)
    dense = [l for l in range(L) if kinds[l] == "dense"]
    fit_pass = mu_pass == "fit" and bool(dense)
    last_dense = max(dense) if dense else -1
    fgroups, fpos = [], {}
    if fit_pass:
        fidx = [j for j in range(S) if fit[j]]
        fglob = [[fidx[i] for i in b] for b in resident_batches([seqs[j] for j in fidx], bs)]
        fpos = {tuple(b): n for n, b in enumerate(fglob)}               # the resident accumulation order
        fgroups = contiguous_groups(seqs, fglob, len(groups))
    acc, cnt = [None] * L, np.zeros(L, np.float64)
    rec = {"chunks": len(groups), "window": window, "carry": carry_mode, "layers": [], "batch": bs,
           "batch_matches_resident": bs == max_texts,
           "batching": "the resident capture's length batches, whole batches per chunk",
           "mu_pass": (f"fit (resident fit-only batches, same weight load, layers 0..{last_dense})" if fit_pass
                       else "inline")}
    child_event("steps", steps=L * len(groups))
    step = 0
    for ci, batches in enumerate(groups):
        g = sorted(j for js in batches for j in js)
        store = _Carry(carry_mode, S, out / ".carry" / f"c{ci}")
        fb = fgroups[ci] if ci < len(fgroups) else []
        fstore = _Carry(carry_mode, S, out / ".carry" / f"f{ci}") if fb else None
        try:
            if prefetch is not None:
                prefetch.reset()
            for bi, js in enumerate(batches):
                store.first(bi, js, fam.embed(_ids_of(seqs, js)))
            store.seal_first()
            forder = sorted(range(len(fb)), key=lambda b: fpos[tuple(fb[b])])
            if fstore is not None:
                for bi, js in enumerate(fb):
                    fstore.first(bi, js, fam.embed(_ids_of(seqs, js)))
                fstore.seal_first()
            fg = sorted(j for js in fb for j in js)
            for l in range(L):
                store.begin_layer(l)
                if fstore is not None:
                    fstore.begin_layer(l)
                if prefetch is not None:
                    for nxt in range(l + 1, min(L, l + 1 + window)):
                        prefetch.request(nxt)
                stall = prefetch.wait(l) if prefetch is not None else 0.0
                if hasattr(fam, "prepare_layer"):     # the disk floor covers both carries' texts
                    fam.prepare_layer(l, [seqs[j] for j in sorted(set(g) | set(fg))], out / ".prepare")
                t_load = time.time()
                if hasattr(fam, "begin_layer"):
                    fam.begin_layer(l)                        # the layer's weights, read once for all its batches
                load = time.time() - t_load
                times = []
                try:
                    _layer_batches(fam, l, batches, store, kinds, widths, seqs, None if fit_pass else fit, mx_t,
                                   mn_t, fsum, fcnt, resp, times)
                    if fstore is not None:
                        _fit_layer(fam, l, fb, forder, fstore, kinds, acc, cnt, times)
                except BaseException as e:
                    try:
                        fam.release_layer(l, e)
                    except TypeError:                          # a family whose release_layer takes only l
                        fam.release_layer(l)
                    raise
                fam.release_layer(l)
                if fstore is not None and l >= last_dense:    # mu needs no layer past the last dense one
                    fstore.close()
                    fstore = None
                ev = {"layer": l, "chunk": ci, "chunks": len(groups), "load_s": round(load, 3),
                      "compute_s": round(float(sum(times)), 3), "stall_s": stall,
                      "state_bytes": store.nbytes() + (fstore.nbytes() if fstore is not None else 0),
                      "prefetched_bytes": prefetch.bytes if prefetch is not None else 0}
                rec["layers"].append(ev)
                child_event("layer", **ev)
                child_event("step", step=step, s_per_step=round(float(sum(times)), 3))
                step += 1
                log(f"chunk {ci + 1}/{len(groups)} layer {l}: load {load:.2f}s compute {sum(times):.2f}s "
                    f"stall {stall:.2f}s")
            if carry_mode == "memmap":
                store.begin_layer(L)                          # seal the last boundary for reading
            if has_logits:
                from research.amoeba.carve.carving import ce_rows
                for bi, js in enumerate(batches):         # full-model NLL per text: targets 2..T-1 (TopicsBackend.nll)
                    ids = _ids_of(seqs, js)
                    lg = fam.final_logits(store.get(bi, js))[:, 1:-1].astype(mx.float32)
                    ce = ce_rows(lg, ids[:, 2:])
                    mx.eval(ce)
                    nll[js] = np.array(ce).mean(-1)
        finally:                                          # no carry directory outlives its chunk
            store.close()
            if fstore is not None:
                fstore.close()
    mu = np.zeros((L, W), np.float32)
    for l in range(L):                            # finish the dense peaks: max(max - mu, mu - min), exact
        if kinds[l] != "dense":
            continue
        if fit_pass:                              # as child_capture: np.array(acc) / cnt
            if acc[l] is not None and cnt[l]:
                mu[l, :widths[l]] = np.array(acc[l]) / cnt[l]
        else:
            mu[l, :widths[l]] = (fsum[l, :widths[l]] / max(1.0, fcnt[l])).astype(np.float32)
        m = mu[l, :widths[l]][None]
        blk = max(1, int((256 << 20) // max(1, 4 * widths[l])))   # rows per block: about 256 MB per array in flight
        for a0 in range(0, S, blk):
            resp[l][a0:a0 + blk] = np.maximum(mx_t[l][a0:a0 + blk] - m, m - mn_t[l][a0:a0 + blk]).astype(np.float16)
    for r in resp:
        r.flush()
    return rec, mu, nll


def _get_weight(module, key):
    """The array at a dotted parameter path of a module (None when absent or unloaded)."""
    obj = module
    for part in key.split("."):
        try:
            obj = obj[int(part)] if part.isdigit() and isinstance(obj, (list, tuple)) else obj[part]
        except (KeyError, IndexError, TypeError):
            obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def load_streamed_host(path: Path, *, stream_mode, ceiling_bytes, resident, moe_units, log, admission_gb=None):
    """The model through its adapter with weight streaming (or resident, for parity runs), as a carve Host."""
    from mlx2.adapters.registry import inspect_model
    from research.amoeba.carve.host import Host, find_inner
    cls = inspect_model(path).adapter_type
    if resident:
        adapter = cls(str(path))
        stream = None
    else:
        from mlx2.runtime.streamed_load import WeightStreamRequest, accepts_streaming_request
        if not accepts_streaming_request(cls):
            raise SystemExit(f"{cls.__name__} does not accept a weight-streaming request: no rolling capture for this "
                             "adapter (never a silent resident load); add streaming to the adapter or a sweep family")
        if stream_mode != "dense_layers":
            # moe_experts is an on-demand per-expert cache sized for decode: a capture batch touches nearly every
            # expert, so it would re-read most of each layer per batch, and its streamed switch modules hold no weights
            # for the carve host's neuron sites. Whole-layer MoE streaming is not in mlx2's loaders yet.
            raise SystemExit("rolling capture of a streamed MoE model needs whole-layer expert streaming, which mlx2's "
                             "loaders do not have (dense_layers refuses routed experts; moe_experts re-reads per batch): "
                             "use a sweep family for this model, or --resident when it fits")
        req = WeightStreamRequest(mode=stream_mode, budget_bytes=int(ceiling_bytes),
                                  admission_limit_bytes=int(admission_gb * 2 ** 30) if admission_gb else None)
        adapter = cls(str(path), weight_streaming=req)
        stream = getattr(adapter, "weight_stream", None)
        if stream is None or getattr(stream, "mode", None) != stream_mode:
            raise SystemExit(f"the adapter accepted the streaming request but installed "
                             f"{getattr(stream, 'mode', None)!r}, not {stream_mode!r}")
    import mlx.core as mx
    from mlx2.runtime.layer_stream import _set_weights
    from mlx2.runtime.weight_stream import _SAFETENSORS_DTYPES
    lm, inner = find_inner(adapter.model)
    wrapped = {}
    for i, layer in enumerate(list(inner.layers)):  # the carve host introspects the real blocks, not the wrappers
        slot = getattr(type(layer), "__name__", "") == "StreamedDecoderLayer" and object.__getattribute__(layer, "_stream_slot")
        if slot:
            # shape-true placeholders (a broadcast scalar: evaluating it allocates nothing) so the unit sites can size
            # their projections; the stream manager gets the blocks back empty right after
            specs = stream.readers[slot.path].specs
            # what the block holds NOW for each streamed tensor: None, or the real arrays of a layer the loader has
            # already cached (the 27B adapter warms some up at load); restored exactly after the host is built
            saved = {k: _get_weight(slot.block, k) for k in specs}
            loaded = [v is not None for v in saved.values()]
            cached = slot.path in getattr(stream, "_cache", {})
            if (any(loaded) and not all(loaded)) or (all(loaded) != cached and loaded):
                raise SystemExit(f"rolling: {slot.path} is {'partly ' if any(loaded) and not all(loaded) else ''}"
                                 f"loaded={all(loaded)} but the stream manager has cached={cached}: refusing (the "
                                 "placeholder restore would not match the manager's bookkeeping)")
            ph = {k: mx.broadcast_to(mx.zeros((1,) * len(loc.shape), getattr(mx, _SAFETENSORS_DTYPES[loc.dtype][1])),
                                     tuple(loc.shape)) for k, loc in specs.items() if saved[k] is None}
            _set_weights(slot.block, ph)
            wrapped[i] = (layer, slot.block, saved)
            inner.layers[i] = slot.block
    from .inspection import artifact_name
    try:
        host = Host(str(path), log=log, model=adapter.model, tokenizer=adapter.tokenizer,
                    fingerprint=adapter.identity["fingerprint"], checkpoint=artifact_name(path).split("/")[-1],
                    moe_units=moe_units)
    finally:
        for i, (w, block, saved) in wrapped.items():
            _set_weights(block, saved)                    # None stays None, a cached layer keeps its arrays
            inner.layers[i] = w
    host.layers = list(inner.layers)
    host.adapter = adapter
    return host, stream


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="host", help="a research/amoeba/sweep FAMILIES name")
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--min-len", type=int, default=48)
    ap.add_argument("--bs", type=int, default=8, help="texts per run_layer call (also capped by the family's batch_rows)")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--mu-pass", choices=["fit", "inline"], default="fit",
                    help="fit (default): dense means from the resident capture's fit-only batches, under the same weight "
                         "load; inline: from the main batches' fit rows (one pass, batching differs from resident)")
    ap.add_argument("--carry", choices=["resident", "memmap"], default="resident")
    ap.add_argument("--stream-mode", choices=["dense_layers", "moe_experts"], default="dense_layers")
    ap.add_argument("--ceiling-gb", type=float, default=0.0, help="host family: the adapter's streaming cache ceiling")
    ap.add_argument("--resident", action="store_true", help="host family, parity runs only: load resident")
    ap.add_argument("--parity-texts", type=int, default=3)
    ap.add_argument("--family-opt", action="append", default=[], metavar="KEY=VALUE",
                    help="passed to a sweep family's loader (e.g. expert_budget_gb=24)")
    ap.add_argument("--parity-layers", type=int, default=0, help="0 = all layers")
    ap.add_argument("--moe-units", default="neuron", choices=["neuron", "expert"])
    ap.add_argument("--memory-gb", type=float, default=56)
    ap.add_argument("--watchdog-gb", type=float, default=80)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    import amoeba_topics_lib as T
    from research.amoeba.carve import ops as O
    from research.amoeba.sweep.family import load_family
    out = T.fresh_dir(a.out)
    log = T.make_log(out)
    O.footprint_watchdog(a.watchdog_gb, log)
    O.setup_mlx(a.memory_gb, 2)
    import mlx.core as mx
    if a.cpu:
        mx.set_default_device(mx.cpu)
    t0 = time.time()
    from .config import parse_overrides
    opts = ({"stream_mode": a.stream_mode, "ceiling_bytes": int(a.ceiling_gb * 2 ** 30), "resident": a.resident,
             "moe_units": a.moe_units, "log": log, "admission_gb": a.memory_gb} if a.family == "host"
            else parse_overrides(a.family_opt))
    if a.family == "host":
        from research.amoeba.sweep.host_family import load as host_load
        fam = host_load(a.model, **opts)
    else:
        fam = load_family(a.family, a.model, **opts)
    rec = {"schema": "mlx2.amoeba-topic-capture.v1", "writer": "mlx2.experimental.amoeba.child_rolling",
           "family": a.family, "args": vars(a), "mlx2_source_revision": T.git_rev(), "corpus_sha256": T.sha_file(a.corpus),
           "schedule": "rolling layer-window (layer-major), mapping-workflow s8",
           "load": {"seconds": round(time.time() - t0, 1), "resident": a.resident,
                    "stream_mode": None if a.resident else a.stream_mode, "ceiling_gb": a.ceiling_gb,
                    "stream_receipt": fam.stream_receipt() if hasattr(fam, "stream_receipt") else None}}
    items = T.load_items(a.corpus)
    toks = T.tokenize_items(fam.tok.encode, items, a.seq_len, a.min_len)
    log(f"{len(toks)}/{len(items)} items with >= {a.min_len} tokens")
    seqs = [ids for _, ids in toks]
    fit = [it["split"] == "fit" for it, _ in toks]
    par = parity(fam, seqs[:a.parity_texts], a.parity_layers or None, log=log)
    rec["parity"] = par
    T.write_json(out / "parity.json", {"checkpoint": fam.checkpoint, "fingerprint": fam.fingerprint, **par})
    if not par["ok"]:
        T.write_json(out / "meta.failed.json", rec)
        raise SystemExit(f"rolling parity failed ({par['criteria']}): {par['texts']}")
    ranges = {} if a.resident else (getattr(fam, "prefetch_ranges", None) or {})
    pre = Prefetcher(ranges) if ranges else None
    import shutil
    pins0 = (fam.stream_counters() or {}).get("page_ins") if hasattr(fam, "stream_counters") else None
    try:
        r, mu, nll = run(fam, seqs, fit, out, window=a.window, chunks=a.chunks, carry_mode=a.carry, prefetch=pre,
                         max_texts=a.bs, log=log, mu_pass=a.mu_pass)
    finally:
        if pre is not None:
            pre.close()
        shutil.rmtree(out / ".rolling", ignore_errors=True)
        shutil.rmtree(out / ".carry", ignore_errors=True)
    rec["stream_counters"] = fam.stream_counters() if hasattr(fam, "stream_counters") else None
    if pins0 is not None and rec["stream_counters"]:     # MEASURED weight reads of the capture pass (not parity)
        streamed = len(getattr(fam, "_blocks", {}) or {})
        loads = rec["stream_counters"]["page_ins"] - pins0
        rec["weight_reads_observed"] = {"layer_loads": loads, "streamed_layers": streamed, "chunks": r["chunks"],
                                        "reads_per_layer": round(loads / max(1, streamed), 3),
                                        "once_per_chunk": loads <= streamed * r["chunks"]}
    np.save(out / "mu.npy", mu)
    if nll is not None:
        np.save(out / "nll_full.npy", nll)
    np.save(out / "n_bytes.npy", np.array([len(fam.tok.decode(s_[2:]).encode("utf-8")) for s_ in seqs], np.int32))
    T.write_json(out / "items.json", [{"id": it["id"], "topics": it["topics"], "split": it["split"],
                                       "n_tokens": len(s_)} for it, s_ in toks])
    lt = r["layers"]
    rec.update({"checkpoint": fam.checkpoint, "fingerprint": fam.fingerprint, "sequences": len(seqs),
                "layers": fam.L, "neurons": max(fam.widths), "widths": [int(w) for w in fam.widths],
                "unit_kinds": list(fam.kinds), "moe_units": a.moe_units if a.family == "host" else "expert",
                "rolling": {"window": a.window, "chunks": r["chunks"], "carry": a.carry, "weight_reads": r["chunks"],
                            "load_s": round(sum(x["load_s"] for x in lt), 2),
                            "compute_s": round(sum(x["compute_s"] for x in lt), 2),
                            "stall_s": round(sum(x["stall_s"] for x in lt), 2),
                            "state_bytes_max": max((x["state_bytes"] for x in lt), default=0),
                            "prefetched_bytes": pre.bytes if pre is not None else 0, "per_layer": lt},
                "mlx_peak_gb": O.peak_gb(), "watch": dict(O.WATCH),
                "unit": "per-text peak over positions >= 1 of |act - mu| (dense), of |w * h| (MoE neuron units), of "
                        "w * ||row|| or the family's routed signal (MoE expert units), fp16",
                "window": "the first seq_len tokens of each probe text"})
    T.write_json(out / "meta.json", rec)
    log(f"done: {len(seqs)} texts, {r['chunks']} chunk(s), window {a.window}, carry {a.carry}")


if __name__ == "__main__":
    main()
