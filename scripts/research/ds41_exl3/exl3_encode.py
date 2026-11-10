"""EXL3 (exllamav3 trellis / QTIP-style) encoder for Apple Silicon, CPU.

Original mlx2 code. Format compatibility target: the upstream EXL3 tensor
layout mlx2 already decodes (src/mlx2/runtime/trellis/upstream.py,
src/mlx2/runtime/models/exl3_reference.py). Algorithm references: QTIP
(arXiv 2406.11235: trellis codes, tail-biting approximation, incoherence),
QuIP# (arXiv 2402.04396: random-sign Hadamard incoherence), QuIP/LDLQ
(arXiv 2307.13304) and GPTQ (arXiv 2210.17323) error feedback, and the
exllamav3 quantizer's published order of operations (turboderp-org/exllamav3,
MIT; read for the regularization order and constants, no code copied). See
provenance/exl3-encoder-20261109.json.

Stored linear (weight as [input I, output O], x @ W):
  W_eff = diag(suh) H inner H diag(svh), H = orthonormal block-128 Hadamard,
  inner = 16x16 tiles of codebook values, each tile one circular bitstream of
  16*n bits (n = tile words = 16 * bits/weight), slot s's codeword = the 16
  bits ending at bit floor((s+1)*n/16), MSB first within little-endian uint32
  words, tile elements permuted to the tensor-core slot order.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mlx2.runtime.models.exl3_reference import codebook, hadamard128  # noqa: E402

# --- tile layout -------------------------------------------------------------


def slot_positions():
    """slot -> flat index row*16+col inside the 16x16 tile (decode_inner's map)."""
    pos = np.empty(256, np.int64)
    for slot in range(256):
        lane, s = divmod(slot, 8)
        row = (lane & 3) * 2 + (s & 1) + (8 if s in (2, 3, 6, 7) else 0)
        col = (lane >> 2) + (8 if s >= 4 else 0)
        pos[slot] = row * 16 + col
    return pos


SLOT_POS = slot_positions()


def ends(words):
    """E(s): stream bit where slot s's codeword ends."""
    return (np.arange(1, 257) * words) >> 4


def shifts(words):
    e = ends(words)
    return np.diff(np.concatenate([[0], e])).astype(np.uint8)


def tiles_from_matrix(m):
    """[I, O] -> [I/16, O/16, 256] in slot order."""
    i, o = m.shape
    t = m.reshape(i // 16, 16, o // 16, 16).transpose(0, 2, 1, 3).reshape(i // 16, o // 16, 256)
    return t[..., SLOT_POS]


def matrix_from_tiles(t):
    a, b, _ = t.shape
    flat = np.empty_like(t)
    flat[..., SLOT_POS] = t
    return flat.reshape(a, b, 16, 16).transpose(0, 2, 1, 3).reshape(a * 16, b * 16)


# --- bitstream ---------------------------------------------------------------


def pack_states(states, words):
    """Per-slot states [..., 256] -> trellis U16 [..., words].

    Slot s contributes its low k_s bits at stream bits [E(s-1), E(s)).
    """
    states = np.asarray(states, np.uint64)
    k = shifts(words).astype(np.uint64)
    e = ends(words)
    lead = states.shape[:-1]
    nwords32 = words // 2
    acc = np.zeros((*lead, nwords32), np.uint64)
    start = np.concatenate([[0], e[:-1]])
    for s in range(256):
        chunk = states[..., s] & ((np.uint64(1) << k[s]) - np.uint64(1))
        for bit in range(int(k[s])):
            pos = int(start[s]) + bit  # MSB-first stream position
            v = (chunk >> (k[s] - np.uint64(1 + bit))) & np.uint64(1)
            j, r = divmod(pos, 32)
            acc[..., j] |= v << np.uint64(31 - r)
    w32 = acc.astype(np.uint32)
    return np.ascontiguousarray(w32).view(np.uint16).reshape(*lead, words)


def unpack_codewords(trellis):
    """Trellis U16 [..., words] -> the 16-bit codeword per slot [..., 256]."""
    t = np.ascontiguousarray(trellis).view(np.uint16)
    words = t.shape[-1]
    w32 = t.view(np.uint32).astype(np.uint64)
    total = 16 * words
    e = ends(words)
    out = np.zeros((*t.shape[:-1], 256), np.uint64)
    for s in range(256):
        for b in range(16):
            pos = (int(e[s]) - 16 + b) % total
            j, r = divmod(pos, 32)
            out[..., s] = (out[..., s] << np.uint64(1)) | ((w32[..., j] >> np.uint64(31 - r)) & np.uint64(1))
    return out.astype(np.uint16)


# --- Viterbi -----------------------------------------------------------------

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        src = HERE / "viterbi.c"
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        cache = Path(os.environ.get("EXL3_ENC_BUILD", HERE / ".build"))
        cache.mkdir(exist_ok=True)
        lib = cache / f"viterbi-{digest}.dylib"
        if not lib.exists():
            tmp = lib.with_suffix(f".{os.getpid()}.tmp")
            subprocess.run(["clang", "-O3", "-mcpu=native", "-ffast-math", "-shared", "-fPIC",
                            "-o", str(tmp), str(src)], check=True)
            tmp.rename(lib)
        _LIB = ctypes.CDLL(str(lib))
        _LIB.viterbi_tiles.restype = ctypes.c_int
    return _LIB


def codebook_values(cb="mcg", window=16):
    return codebook(cb, window)[: 1 << window].astype(np.float32)


def viterbi(targets, words, cb="mcg", window=16, probe=32, threads=None):
    """targets [N, 256] (slot order, codebook units) -> states [N, 256] U16."""
    targets = np.ascontiguousarray(targets, np.float32)
    n = targets.shape[0]
    states = np.zeros((n, 256), np.uint16)
    table = np.ascontiguousarray(codebook_values(cb, window))
    sh = np.ascontiguousarray(shifts(words))
    lib = _lib()
    threads = threads or int(os.environ.get("EXL3_ENC_THREADS", os.cpu_count() or 1))
    step = max(1, -(-n // (threads * 4)))

    def run(a):
        b = min(n, a + step)
        rc = lib.viterbi_tiles(
            targets[a:b].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(b - a),
            table.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(window),
            sh.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(probe),
            states[a:b].ctypes.data_as(ctypes.c_void_p))
        if rc:
            raise RuntimeError(f"viterbi_tiles failed: {rc}")

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(run, range(0, n, step)))
    return states


def viterbi_reference(target, words, cb="mcg", window=16, probe=32):
    """Pure numpy single-tile oracle of viterbi.c (same tie-breaking)."""
    table = codebook_values(cb, window).astype(np.float64)
    table = table.astype(np.float32)
    sh = shifts(words).astype(int)
    nstate = 1 << window
    idx = np.arange(nstate)

    def run_pass(order, start_o, end_o):
        k0 = sh[order[0]]
        c = (table - np.float32(target[order[0]])) ** 2
        if start_o >= 0:
            c = np.where((idx >> k0) == start_o, c, np.inf).astype(np.float32)
        bps = []
        for s in order[1:]:
            k = sh[s]
            cand = c.reshape(1 << k, nstate >> k)
            arg = np.argmin(cand, axis=0)
            m = cand[arg, np.arange(nstate >> k)]
            bps.append(arg)
            c = (np.repeat(m, 1 << k) + (table - np.float32(target[s])) ** 2).astype(np.float32)
        omask = (1 << (window - sh[order[0]])) - 1
        if end_o >= 0:
            c = np.where((idx & omask) == end_o, c, np.inf)
        S = int(np.argmin(c))
        out = [S]
        for i in range(len(order) - 1, 0, -1):
            k = sh[order[i]]
            g = S >> k
            S = ((int(bps[i - 1][g]) << (window - k)) | g) & (nstate - 1)
            out.append(S)
        return out[::-1]

    order1 = list(range(256 - probe, 256)) + list(range(probe))
    st1 = run_pass(order1, -1, -1)
    o = st1[probe - 1] & ((1 << (window - sh[0])) - 1)
    return np.array(run_pass(list(range(256)), o, o), np.uint16)


def decode_states(states, cb="mcg", window=16):
    return codebook_values(cb, window)[np.asarray(states) & ((1 << window) - 1)]


# --- Hessian / LDL -----------------------------------------------------------


def block_ldl(h, block=16, sigma=0.025):
    """H = L D L^T with unit block-lower L (16x16 blocks). Damped like exllamav3."""
    h = np.array(h, np.float64)
    n = h.shape[0]
    damp = sigma * float(np.mean(np.diag(h)))
    for attempt in range(10):
        try:
            c = np.linalg.cholesky(h + damp * np.eye(n))
            break
        except np.linalg.LinAlgError:
            damp *= 2
    else:
        raise np.linalg.LinAlgError("Hessian not positive definite")
    L = np.empty_like(c)
    for b in range(0, n, block):
        inv = np.linalg.inv(c[b:b + block, b:b + block])
        L[:, b:b + block] = c[:, b:b + block] @ inv
    return L


_SQRT128 = np.sqrt(128)
_HAD_POOL = None


def _had_pool():
    global _HAD_POOL
    if _HAD_POOL is None:
        _HAD_POOL = ThreadPoolExecutor(int(os.environ.get("EXL3_ENC_HAD_THREADS", min(8, os.cpu_count() or 1))))
    return _HAD_POOL


def _had_chunk(x, y, axis):
    """In-place block-128 butterfly of x (rows of 128 on ``axis``), y scratch.

    The same additions as exl3_reference.hadamard128 in the same order, so the
    float64 result is bit-identical; ping-pong buffers instead of copies.
    """
    for bit in (1, 2, 4, 8, 16, 32, 64):
        if axis == 1:
            xb = x.reshape(x.shape[0], -1, 2, bit)
            yb = y.reshape(xb.shape)
            lo, hi, olo, ohi = xb[:, :, 0, :], xb[:, :, 1, :], yb[:, :, 0, :], yb[:, :, 1, :]
        else:
            xb = x.reshape(-1, 2, bit, x.shape[1])
            yb = y.reshape(xb.shape)
            lo, hi, olo, ohi = xb[:, 0], xb[:, 1], yb[:, 0], yb[:, 1]
        np.add(lo, hi, out=olo)
        np.subtract(lo, hi, out=ohi)
        x, y = y, x
    np.divide(x, _SQRT128, out=y)
    return y


def _hadamard(a, axis):
    a = np.asarray(a, np.float64)
    if a.ndim != 2:
        return hadamard128(a) if axis == 1 else hadamard128(a.T).T
    n0, n1 = a.shape
    if (n1 if axis == 1 else n0) % 128:
        raise ValueError("Hadamard axis must be a multiple of 128")
    out = np.empty((n0, n1))
    buf = np.empty((n0, n1))
    if axis == 1:
        step = max(1, min(n0, (1 << 21) // n1))  # rows per chunk (~16 MB)
        spans = [(r, min(n0, r + step)) for r in range(0, n0, step)]

        def run(sp):
            r0, r1 = sp
            x = out[r0:r1]
            x[...] = a[r0:r1]
            res = _had_chunk(x, buf[r0:r1], 1)
            if res is not x:
                x[...] = res
    else:
        step = max(128, (min(n1, (1 << 21) // 128) // 128) * 128)  # 128-row blocks x column span
        spans = [(c, min(n1, c + step)) for c in range(0, n1, step)]

        def run(sp):
            c0, c1 = sp
            x = np.array(a[:, c0:c1], order="C", copy=True)  # never alias the input
            y = np.empty_like(x)
            out[:, c0:c1] = _had_chunk(x, y, 0)
    list(_had_pool().map(run, spans))
    return out


def hadamard_cols(a):
    """a @ H over the last axis (block 128); bit-identical to hadamard128(a)."""
    return _hadamard(a, 1)


def hadamard_rows(a):
    """H @ a over the first axis (block 128); bit-identical to hadamard128(a.T).T."""
    return _hadamard(a, 0)


# --- encoder -----------------------------------------------------------------


def codebook_rms(cb="mcg", window=16):
    v = codebook_values(cb, window).astype(np.float64)
    return float(np.sqrt(np.mean(v * v)))


class _Job:
    """One linear being encoded (prepared incoherent target + Hessian factor)."""

    def __init__(self, w, hx, words, cb, window, seed, ldlq, sigma):
        w = np.asarray(w, np.float64)
        I, O = w.shape
        if I % 128 or O % 128:
            raise ValueError("EXL3 dims must be multiples of 128")
        hx = np.eye(I) if hx is None else np.asarray(hx, np.float64)
        t0 = time.time()
        rng = np.random.default_rng(seed)
        su = np.where(rng.random(I) < 0.5, -1.0, 1.0)
        sv = np.where(rng.random(O) < 0.5, -1.0, 1.0)
        # Output-channel scale (column RMS over its mean) folded into sv.
        col = np.sqrt(np.mean(w * w, axis=0)) + 1e-12
        sv = (sv * (col / col.mean())).astype(np.float16).astype(np.float64)
        a = hadamard_cols(w / sv)
        # Input-channel scale: rows of the incoherent matrix match the codebook RMS.
        row = np.sqrt(np.mean(a * a, axis=1)) + 1e-12
        su = (su * row / codebook_rms(cb, window)).astype(np.float16).astype(np.float64)
        if np.any(su == 0) or np.any(sv == 0):
            raise ValueError("EXL3 scale underflowed fp16")
        self.inner = hadamard_rows(a / su[:, None])  # codebook units
        self.L = None
        if ldlq:
            # Hessian of the inner rows (x_inner = H diag(su) x), block LDL.
            hs = hadamard_rows(hadamard_cols(hx * su[None, :] * su[:, None]))
            self.L = block_ldl(hs, sigma=sigma)
        self.w, self.hx, self.su, self.sv = w, hx, su, sv
        self.words, self.cb, self.window = words, cb, window
        self.blocks = I // 16
        self.q = np.empty_like(self.inner)
        self.err = np.zeros_like(self.inner)
        self.states = np.empty((I // 16, O // 16, 256), np.uint16)
        self.prep_s = time.time() - t0

    def target(self, b):
        r0, r1 = 16 * b, 16 * b + 16
        t = self.inner[r0:r1].copy()
        if self.L is not None and r1 < self.inner.shape[0]:
            t += self.L[r1:, r0:r1].T @ self.err[r1:]
        return tiles_from_matrix(t)[0]

    def accept(self, b, states):
        r0, r1 = 16 * b, 16 * b + 16
        self.states[b] = states
        self.q[r0:r1] = matrix_from_tiles(decode_states(states, self.cb, self.window)[None].astype(np.float64))
        self.err[r0:r1] = self.inner[r0:r1] - self.q[r0:r1]

    def finish(self, refit=True):
        w, hx, su = self.w, self.hx, self.su
        weff = hadamard_cols(hadamard_rows(self.q) * su[:, None])  # diag(su) H q H
        hw = hx @ weff
        if refit:
            # Per-output-column least squares under hx (exllamav3 refits scales too).
            svh = np.sum(w * hw, axis=0) / (np.sum(weff * hw, axis=0) + 1e-30)
        else:
            svh = self.sv
        svh16 = svh.astype(np.float16)
        weff = weff * svh16.astype(np.float64)
        d = weff - w
        # Proxy statistic without a third product: hx @ d = (hx @ weff) * svh - hx @ w
        # (equal up to float64 rounding; stats only, outputs are unaffected).
        hxw = hx @ w
        hxd = hw * svh16.astype(np.float64) - hxw
        stats = {
            "words": self.words, "bits_per_weight": self.words / 16, "codebook": self.cb,
            "window": self.window, "ldlq": self.L is not None, "refit": bool(refit),
            "weight_rel_err": float(np.linalg.norm(d) / np.linalg.norm(w)),
            "proxy_rel_err": float(np.sqrt(max(np.sum(d * hxd), 0.0) / max(np.sum(w * hxw), 1e-30))),
            "inner_rel_err": float(np.linalg.norm(self.q - self.inner) / np.linalg.norm(self.inner)),
            "prep_s": round(self.prep_s, 3),
        }
        return {"trellis": pack_states(self.states, self.words), "suh": su.astype(np.float16),
                "svh": svh16, "stats": stats, "weight": weff}


def _metal_submit(words, window, probe):
    """Async GPU search when the v2 kernel covers this rate, else None."""
    import viterbi_metal as VM

    if VM.KERNEL_VERSION == "v2" and VM.v2_supported(words, window):
        return lambda t: VM.viterbi_metal_submit(t, words, window, probe)
    return None


class _Done:
    def __init__(self, st):
        self.st = st

    def result(self):
        return self.st


def _ldlq_run(jobs, submit, groups, clock):
    """Bottom-up LDLQ over ``jobs``; each group's tiles go to one search call
    per block step. Groups are independent, so with an async ``submit`` the
    host target/accept work of one group runs while the GPU searches the
    others (results are identical to lockstep: every matrix still walks its own
    blocks in order with its own error feedback)."""
    from collections import deque

    gl = [[j for j, g in zip(jobs, groups) if g == k] for k in sorted(set(groups))]
    step = [0] * len(gl)
    inflight = deque()

    def launch(k):
        active = [j for j in gl[k] if step[k] < j.blocks]
        if not active:
            return
        targets = [j.target(j.blocks - 1 - step[k]) for j in active]
        tv = time.time()
        h = submit(np.concatenate(targets))
        clock["submit_s"] += time.time() - tv
        clock["calls"] += 1
        inflight.append((k, active, targets, h))

    for k in range(len(gl)):
        launch(k)
    while inflight:
        k, active, targets, h = inflight.popleft()
        tv = time.time()
        st = h.result()
        clock["wait_s"] += time.time() - tv
        at = 0
        for j, t in zip(active, targets):
            j.accept(j.blocks - 1 - step[k], st[at:at + len(t)])
            at += len(t)
        step[k] += 1
        launch(k)


def encode_many(items, words=32, *, cb="mcg", window=16, seed=0, probe=32, threads=None,
                ldlq=True, refit=True, sigma=0.025, backend="cpu", groups=None):
    """Encode several linears together: ``items`` = [(w [I, O], hx [I, I] or None), ...].

    LDLQ walks each matrix's 16-row blocks bottom-up. ``groups`` (one id per
    item) decides which matrices share a search call per block step: by default
    all of them on the CPU and with the v1 Metal kernel (one lockstep call per
    step, as before), and one group per matrix with the async v2 Metal kernel
    so host LDLQ work overlaps GPU search (``groups=None`` there streams each
    matrix as its own encode_experts unit). Output does not depend on grouping.
    Seeds are ``seed + index``.
    """
    submit = None
    if backend == "metal":
        from viterbi_metal import viterbi_metal

        if cb != "mcg":
            raise ValueError("the Metal Viterbi computes the MCG codebook inline")
        submit = _metal_submit(words, window, probe)

        def vit(t):
            return viterbi_metal(t, words, window, probe)
    else:
        def vit(t):
            return viterbi(t, words, cb, window, probe, threads)
    if submit is None:
        def submit(t):
            return _Done(vit(t))
        if groups is None:
            groups = [0] * len(items)
    elif groups is None and ldlq:
        # Each matrix as its own streamed unit: the first matrix's search starts
        # while the others are still being prepared, and each is refit/packed as
        # soon as its last block is searched.
        units = [([it], seed + n) for n, it in enumerate(items)]
        return [o[0] for o in encode_experts(units, words, inflight=len(units), group="unit", cb=cb, window=window,
                                             probe=probe, ldlq=ldlq, refit=refit, sigma=sigma, finish_workers=2)]
    elif groups is None:
        groups = list(range(len(items)))
    t0 = time.time()
    jobs = [_Job(w, hx, words, cb, window, seed + n, ldlq, sigma) for n, (w, hx) in enumerate(items)]
    t1 = time.time()
    clock = {"submit_s": 0.0, "wait_s": 0.0, "calls": 0}
    if ldlq:
        _ldlq_run(jobs, submit, groups, clock)
    else:
        targets = [tiles_from_matrix(j.inner).reshape(-1, 256) for j in jobs]
        tv = time.time()
        st = vit(np.concatenate(targets))
        clock["wait_s"] += time.time() - tv
        clock["calls"] += 1
        at = 0
        for j, t in zip(jobs, targets):
            n = len(t)
            j.states[:] = st[at:at + n].reshape(j.states.shape)
            j.q[:] = matrix_from_tiles(decode_states(j.states, cb, window).astype(np.float64))
            at += n
    t_quant = time.time() - t1
    t2 = time.time()
    out = [j.finish(refit) for j in jobs]
    timing = {"prep_s": round(t1 - t0, 3), "search_wait_s": round(clock["wait_s"], 3),
              "search_submit_s": round(clock["submit_s"], 3), "search_calls": clock["calls"],
              "host_ldlq_s": round(t_quant - clock["wait_s"] - clock["submit_s"], 3),
              "quant_s": round(t_quant, 3), "finish_s": round(time.time() - t2, 3)}
    for o in out:
        o["stats"].update(probe=probe, backend=backend, quant_s_shared=round(t_quant, 3),
                          total_s_shared=round(time.time() - t0, 3), timing=timing)
    return out


def encode_experts(experts, words=32, *, inflight=2, group="unit", cb="mcg", window=16, probe=32,
                   ldlq=True, refit=True, sigma=0.025, clock=None, finish_workers=1):
    """Stream many encodes through the async v2 Metal search; yields in input order.

    ``experts`` yields ``(items, seed)`` or ``(items, seed, words)`` (a per-unit
    bit rate; ``words`` is the default) or a zero-argument callable returning
    one; callables run on the prep thread in input order, so loading and
    Hessian building overlap the GPU too. Up to ``inflight`` units share the
    search queue (one call per block step per unit with ``group="unit"``, or per
    matrix with ``group="job"``), the next unit is prepared
    on a background thread, and finished units are refit and packed on another,
    so the GPU stays fed. Each unit's output equals encode_many(items, words,
    seed=seed, ...) bit for bit (same per-matrix order, seeds and arithmetic).
    """
    from collections import deque

    submits = {}

    def submit_for(wd):
        if wd not in submits:
            submits[wd] = _metal_submit(wd, window, probe)
            if submits[wd] is None:
                raise ValueError(f"words={wd}: encode_experts needs the v2 Metal kernel (window 16, k in 2..4)")
        return submits[wd]

    submit_for(words)
    if cb != "mcg" or not ldlq:
        raise ValueError("encode_experts needs the MCG codebook and LDLQ")
    if inflight < 1:
        raise ValueError("inflight must be >= 1")
    prep_pool, fin_pool = ThreadPoolExecutor(1), ThreadPoolExecutor(finish_workers)
    src = iter(experts)
    preps = deque()

    def prep(e):
        t0 = time.time()
        unit = e() if callable(e) else e
        items, seed = unit[0], unit[1]
        wd = unit[2] if len(unit) > 2 else words
        if not items:
            raise ValueError("empty unit")
        t1 = time.time()
        jobs = [_Job(w, hx, wd, cb, window, seed + n, ldlq, sigma) for n, (w, hx) in enumerate(items)]
        return jobs, {"load_s": round(t1 - t0, 3), "prep_s": round(time.time() - t1, 3), "words": wd}

    def feed():
        for e in src:
            preps.append(prep_pool.submit(prep, e))
            return

    def finish(jobs, info, t_start):
        t2 = time.time()
        out = [j.finish(refit) for j in jobs]
        info = dict(info, finish_s=round(time.time() - t2, 3), unit_wall_s=round(time.time() - t_start, 3))
        for o in out:
            o["stats"].update(probe=probe, backend="metal", timing=info)
        return out

    queue = deque()   # (unit, group, active jobs, blocks, handle) in GPU submission order
    units, finished = {}, {}
    clock = {} if clock is None else clock  # caller-visible: main-thread wait vs busy seconds
    clock.update(wait_s=0.0, prep_wait_s=0.0, calls=0, t0=time.time())
    next_uid = 0

    def launch(uid, gi):
        u = units[uid]
        active = [j for j in u["groups"][gi] if u["step"][id(j)] < j.blocks]
        if not active:
            return
        blocks = [j.blocks - 1 - u["step"][id(j)] for j in active]
        targets = [j.target(b) for j, b in zip(active, blocks)]
        queue.append((uid, gi, active, blocks, [len(t) for t in targets], u["submit"](np.concatenate(targets))))
        clock["calls"] += 1

    def admit():
        nonlocal next_uid
        if not preps:
            return False
        tw = time.time()
        jobs, info = preps.popleft().result()
        info["prep_wait_s"] = round(time.time() - tw, 3)
        clock["prep_wait_s"] += time.time() - tw
        feed()
        uid, next_uid = next_uid, next_uid + 1
        groups = [jobs] if group == "unit" else [[j] for j in jobs]
        units[uid] = {"jobs": jobs, "info": info, "groups": groups, "step": {id(j): 0 for j in jobs},
                      "submit": submit_for(info["words"]),
                      "left": len(jobs), "t0": time.time()}
        for gi in range(len(groups)):
            launch(uid, gi)
        return True

    def admit_ready():
        # Admit prepared units while slots are free; block on a prep only when
        # the GPU queue would otherwise run dry.
        while len(units) < inflight and preps and (preps[0].done() or not queue):
            admit()

    for _ in range(inflight + 1):
        feed()
    next_out = 0
    try:
        admit_ready()
        while queue or finished or preps:
            if queue:
                uid, gi, active, blocks, sizes, h = queue.popleft()
                tv = time.time()
                st = h.result()
                clock["wait_s"] += time.time() - tv
                u = units[uid]
                at = 0
                for j, b, n in zip(active, blocks, sizes):
                    j.accept(b, st[at:at + n])
                    at += n
                    u["step"][id(j)] += 1
                    if u["step"][id(j)] == j.blocks:
                        u["left"] -= 1
                if u["left"]:
                    launch(uid, gi)
                else:
                    u["info"]["quant_s"] = round(time.time() - u["t0"], 3)
                    finished[uid] = fin_pool.submit(finish, u["jobs"], u["info"], u["t0"])
                    del units[uid]
            admit_ready()
            # Yield in input order; block on a finish only when nothing else can run.
            while next_out in finished and (finished[next_out].done() or not queue):
                yield finished.pop(next_out).result()
                next_out += 1
    finally:
        prep_pool.shutdown(wait=False, cancel_futures=True)
        fin_pool.shutdown(wait=False, cancel_futures=True)


def encode_linear(w, hx=None, words=32, **kw):
    """Encode one linear. ``w`` is [I, O] (x @ w); ``hx`` is E[x^T x] [I, I].

    Returns dict(trellis U16 [I/16, O/16, words], suh F16 [I], svh F16 [O],
    weight = the dense effective weight, stats). Deterministic per seed.
    """
    return encode_many([(w, hx)], words, **kw)[0]
