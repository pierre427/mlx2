#!/usr/bin/env python3
"""Real-checkpoint GPU gates for the GLM-5.3 sparse DSA (docs/design/glm53-sparse-dsa.md, P5).

One subcommand per gate; each writes one atomic ``receipt.json`` into a new or
empty ``--out`` directory (a nonempty one is refused, so retries stay
distinguishable) naming the checkpoint, the mlx2 commit (``--source-revision``
must be HEAD and ``src/`` clean unless explicitly overridden), host, MLX
version, the policies run, every check with its numbers, and a verdict.
GPU only: pass ``--i-own-the-gpu`` and run under the GPU lease. Correctness
and sanity checks, not publishable statistics: no thermal gate, and every
timing recorded is an exploratory signal.

Every gate first proves its comparator can fail (``comparator_detects_
difference``: a one-ulp / one-id perturbation must be reported). A gate fails
when any check fails; the R-numbered design falsifiers (section 13) are
recorded under ``falsifiers`` and reported, but a falsified performance claim
is a design finding, not a correctness failure.

Gates and what makes each fail:

``select``   (P1 kernel, 9.6, 9.7, R3, R9b/c) the select+expand kernel's
             first-use gate does not pass on Metal; kernel ids differ in any
             word from the topk_v1 ops law on synthetic cases (ties, shared
             top-16-bit scores, signed zeros, nonfinite, invisible pools,
             P in 600..16384) or on real pool scores captured from a 16K
             prefill and a decode step; blocked_select ids differ from the
             global select on real data (differences are classified with the
             512th/513th gap and the inter-arm score difference, never
             accepted); any real pool score is nonfinite; either pooling law's
             fp32 scores fall outside 8 eps32 sum|terms| of float64 (R9b).
``pools``    (R5, 9.3, R9a) a full prefill + decode under
             ``incremental_v1_audit`` raises (incremental pools differ from a
             full recompute bit for bit, or a nonfinite pool score); the
             offline extends (one-shot, 2048-row chunks, random 1-9 / 1-300 row
             steps, cut-then-re-extend) of every MLA layer's real raw rows at
             T in {2049, 4096, 16384} differ from pool_fixed_order in any word;
             a pool computed among fewer pools differs from the same pool
             among more (shape invariance); either law leaves the 8 eps32 M
             bound or the laws differ by more than 16 eps32 M.
``attention``  (P2, 9.5, 9.7, R4, 6.1) the fused first-use gate fails;
             gathered or mask-path attention exceeds 1e-5 abs / 1e-6 rel RMS of
             float64 on unit-scale real-geometry data (64 heads, 512 dims,
             W = 2051, T up to 16384); fused exceeds 1e-5 abs of gathered; an
             empty selection is not NaN in the gathered/fused consumers;
             sparse_prefill at blocks 1/4/32/L or one-row decode differ in any
             word (synthetic and real captured operands); the peak of 8 live
             blocks exceeds one block's by more than half a block workspace.
``rows``     (V1, R7) rows_attention over R in {2, 4, 8, 16} ragged plain and
             tree sources differs in any word from one-row decode, from a
             permuted batch, or (fused) from one-row fused launches.
``decode``   (9.1, 9.10) at 4K and 16K: the first decode step's selected ids
             differ between mask_reference, gathered_v1 and fused_decode_v1 at
             equal pooling/selection; any cross-policy token comparison is
             ``failed`` or ``unclassified`` (AGENTS.md near-tie contract,
             functional oracle = the buried code is answered); an arm's
             counters show the policy did not engage; the reference fails its
             oracle or is nonfinite.
``tree``     (9.8, 9.9, 9.13) glm53_state_oracle under each tree policy
             (per_row/batched_v1 x clone/tails_v1, plus fused) with fused HC
             (mega_v1) and reference HC, at 2046 (a depth-3 path crosses
             TOPK) and 4100 tokens: tree logits/features differ from one-row
             decode, any prepared/committed state array or meta differs, the
             continuation differs, the live cache changes, or the policy's
             counters are zero.
``replay``   (6.5, 9.11, APCv2) under the reference and a gathered candidate:
             after a serialized restore at 8191 / 8193 (pools dropped and
             recomputed once) followed by the same suffix chunks, any committed
             plane, pooled key or continuation token differs from the
             unrestored run, or the pool rebuild is not observed. The
             unaligned comparison with a uniform cold chunking is recorded
             as evidence (no bitwise claim, design 5.1).
``memory``   (R6, R8, 6.1) a gathered prefill chunk's measured MLX peak
             exceeds the adapter's booked prefill transient, or gathered decode
             peaks above the mask path; layered verify logits differ between
             clone and tails_v1 frontiers.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from glm53_policy_ab import (
    CHECKPOINT,
    MODEL,
    apply_sparse_policy,
    atomic_write,
    claim_out_dir,
    classify_divergence,
    comparison_passes,
    decode_text,
    engagement,
    host_record,
    load_adapter,
    needle_found,
    needle_prompt,
    pool_law_audit,
    source_record,
)

BASE_POLICY = ROOT / "docs/research/glm53-sparse-dsa-20261031/arm0-reference.json"
REFERENCE = {
    "glm53_indexer_pools": "recompute", "glm53_indexer_select": "argsort",
    "glm53_sparse_attention": "mask_reference", "glm53_sparse_prefill": "dense_reference",
    "glm53_tree_rows_attention": "per_row", "glm53_tree_frontier": "clone",
}
SELECTION = {**REFERENCE, "glm53_indexer_pools": "incremental_v1", "glm53_indexer_select": "topk_v1"}
GATHERED = {**SELECTION, "glm53_sparse_attention": "gathered_v1"}
TOPK, WIDTH, LATENT, HEADS = 2048, 2051, 512, 64


# ---------------------------------------------------------------------------
# Receipts and comparators.

class Gate:
    """One gate's receipt: checks (pass/fail), falsifiers (reported), atomic writes."""

    def __init__(self, name, args):
        self.out = claim_out_dir(args.out)
        self.path = self.out / "receipt.json"
        self.started = time.time()
        self.receipt = {
            "schema": "mlx2.glm53-sparse-dsa-gpu-check.v1", "gate": name, "checkpoint": CHECKPOINT,
            **source_record(args.source_revision, allow_mismatch=args.allow_revision_mismatch,
                            allow_dirty=args.allow_dirty_src),
            "argv": sys.argv, "status": "running", "checks": {}, "falsifiers": {}, "policies": {},
            "qualified": False, "thermal": "not gated (correctness/sanity check)",
        }
        self.flush()

    def flush(self):
        atomic_write(self.path, self.receipt)

    def check(self, name, passed, **data):
        self.receipt["checks"][name] = {"passed": bool(passed), **data}
        self.flush()
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {json.dumps(data, default=str)[:300]}", flush=True)
        return passed

    def falsifier(self, name, falsified, **data):
        self.receipt["falsifiers"][name] = {"falsified": falsified, **data}
        self.flush()

    def finish(self, mx, error=None):
        checks = self.receipt["checks"]
        if error is not None:
            self.receipt.update(status="error", error=repr(error))
        elif not checks:
            self.receipt.update(status="failed", error="no checks ran")
        else:
            self.receipt["status"] = "passed" if all(c["passed"] for c in checks.values()) else "failed"
        self.receipt["elapsed_seconds"] = time.time() - self.started
        self.receipt["mlx_peak_bytes"] = mx.get_peak_memory()
        self.receipt["host_after"] = host_record(mx)
        self.flush()
        return 0 if self.receipt["status"] == "passed" else 1


def bitwise(a, b):
    """(equal, differing words); -1 words when shape or dtype differ. NaN bits compare as bits."""
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, -1
    if a.dtype.kind == "f":
        a, b = a.view(f"u{a.itemsize}"), b.view(f"u{b.itemsize}")
    words = int(np.count_nonzero(a != b))
    return words == 0, words


def comparator_selftest(gate):
    """The comparator must report a one-ulp float change and a one-id int change."""
    import numpy as np

    x = np.linspace(-3, 3, 64, dtype=np.float32)
    y = x.copy()
    y[17] = np.nextafter(y[17], np.float32(10))
    i = np.arange(64, dtype=np.int32)
    j = i.copy()
    j[5] = -1
    ok = bitwise(x, x)[0] and not bitwise(x, y)[0] and bitwise(x, y)[1] == 1 and not bitwise(i, j)[0]
    ok = ok and not bitwise(x, x[:-1])[0]
    return gate.check("comparator_detects_difference", ok)


def np_of(mx, value):
    import numpy as np

    mx.eval(value)
    return np.array(value)


# ---------------------------------------------------------------------------
# Real-model helpers.

def mla_layers(model):
    return [i for i, layer in enumerate(model.layers) if not layer.is_linear]


def prefill(model, ids, chunk, mx, *, cache=None, start=0, stop=None):
    """Chunked prompt phase over ids[start:stop] (``chunk`` rows each), as served."""
    cache = model.make_cache() if cache is None else cache
    stop = len(ids) if stop is None else stop
    logits = None
    for begin in range(start, stop, chunk):
        piece = mx.array([ids[begin : min(begin + chunk, stop)]])
        with model._prompt_scope(piece.shape[1]):
            logits = model(piece, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    return cache, logits


def greedy(model, cache, logits, steps, mx):
    """``steps`` greedy token ids (ids only; no per-step window readback) and
    whether every emitted logits row was finite (full rows, reduced lazily)."""
    ids, finite = [], mx.array(True)
    for step in range(steps):
        row = logits[0, -1]
        finite = finite & mx.all(mx.isfinite(row))
        ids.append(int(mx.argmax(row).item()))
        if step + 1 < steps:
            logits = model(mx.array([[ids[-1]]]), cache, logits_to_keep=1)
    return ids, bool(finite.item())


def window_from(model, start, first_logits, emitted, index, mx, k=16):
    """Replay from a decode-start snapshot to ``emitted[index]``'s distribution and
    return its bounded top-``k`` window and argmax (first divergence only, AGENTS.md)."""
    from glm53_ordinary_profile import top_window

    from mlx2.adapters.glm53_ast_target import clone

    cache, logits = clone(start), first_logits
    for token in emitted[:index]:
        logits = model(mx.array([[token]]), cache, logits_to_keep=1)
    row = logits[0, -1]
    return top_window(row, mx, k), int(mx.argmax(row).item())


def served_ids(adapter, ids):
    """Ids through the first stop token (inclusive): what serving would emit."""
    text, cut = decode_text(adapter, ids)
    return list(ids[: cut + 1]), text


def compare_runs(gate, adapter, ref, cand, mx):
    """Classify ``cand`` against ``ref`` (dicts with policy, start, first, served, needle_found)."""
    a, b = ref["served"], cand["served"]
    from glm53_policy_ab import first_divergence

    index = first_divergence(a, b)
    if index is None:
        return classify_divergence(a, b, [], [], ref["needle_found"], cand["needle_found"])
    windows, consistent = {}, True
    for key, run, ids in (("a", ref, a), ("b", cand, b)):
        if index >= len(ids):
            windows[key] = []
            continue
        use(gate, adapter, "replay", run["policy"])
        window, argmax = window_from(adapter.model, run["start"], run["first"], ids, index, mx)
        consistent &= argmax == ids[index]
        windows[key] = [None] * index + [window]
    result = classify_divergence(a, b, windows["a"], windows["b"], ref["needle_found"], cand["needle_found"])
    if not consistent:
        result = {**result, "verdict": "unclassified", "reason": "the window replay did not reproduce the token"}
    return {**result, "replay_consistent": consistent}


def load(gate, args, mx):
    if not mx.metal.is_available():
        raise SystemExit("Metal is not available")
    policy = json.loads(Path(args.base_policy).read_text())
    adapter, seconds = load_adapter(policy, args.model)
    gate.receipt.update(base_policy=str(args.base_policy), loaded_policy=dict(adapter.kernel_policy),
                        load_seconds=seconds)
    gate.flush()
    return adapter


def use(gate, adapter, name, overrides):
    merged = apply_sparse_policy(adapter, {**adapter.kernel_policy, **overrides})
    gate.receipt["policies"][name] = {k: merged[k] for k in REFERENCE}
    return merged


# ---------------------------------------------------------------------------
# select

def synthetic_select_cases(seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    cases = []
    for pools in (600, 1024, 4096, 16384):
        rows = 128
        s = rng.normal(size=(rows, pools)).astype(np.float32)
        s[: rows // 2] = np.round(s[: rows // 2] * 4) / 4  # heavy ties
        s[:, ::5] = np.maximum(s[:, ::5], 0)  # ReLU zeros
        s[3, :] = 0.0  # one row all tied
        decode = np.full(rows, 4 * pools + 2, np.int32)
        prefill = np.sort(rng.integers(4 * min(pools, 512), 4 * pools + 3, size=rows)).astype(np.int32)
        count = min(512, pools)
        cases += [(f"P{pools}-decode", s, decode, count), (f"P{pools}-prefill", s, prefill, count)]
    return cases


def time_ms(mx, fn, reps=40):
    out = fn()
    mx.eval(out)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(fn())
        samples.append(time.perf_counter() - t0)
    return 1e3 * sorted(samples)[len(samples) // 2]


def capture_selection(model, layers, mx):
    """Wrap indexer.select_from_rows: keep, per target layer, the last prompt
    call and the first decode call's real (q, weights, raw rows, positions)."""
    import numpy as np

    from mlx2.runtime.models import glm53_gguf_indexer as ix

    captured = {}
    original = ix.select_from_rows
    targets = {f"blk.{i}" for i in layers}

    def capture(x, qr, weights, prefix, *, total, raw_rows_fn, projected=None, **kwargs):
        length = x.shape[1]
        key = (prefix, "decode" if length == 1 else "prompt")
        if prefix in targets and int(total) > TOPK and not (key[1] == "decode" and key in captured):
            p = projected or {}
            q = (p["q"] if "q" in p else weights.linear(prefix + ".indexer.attn_q_b.weight")(qr))
            w = (p["proj"] if "proj" in p else weights.linear(prefix + ".indexer.proj.weight")(x)) * ix.HEADS**-0.5
            keep = min(length, 256)  # the call's last queries (largest visible sets)
            q = q.reshape(1, length, ix.HEADS, ix.DIM)[:, -keep:]
            raw = raw_rows_fn(0, ix.KPOOL * (int(total) // ix.KPOOL))
            ape = weights.array(prefix + ".indexer.kpool_ape")
            mx.eval(q, w, raw, ape)
            captured[key] = {"q": np.array(q[0]), "w": np.array(w[0, -keep:]), "raw": np.array(raw),
                             "ape": np.array(ape), "total": int(total),
                             "positions": np.arange(int(total) - keep, int(total), dtype=np.int32)}
        return original(x, qr, weights, prefix, total=total, raw_rows_fn=raw_rows_fn, projected=projected,
                        **kwargs)

    ix.select_from_rows = capture
    return captured, lambda: setattr(ix, "select_from_rows", original)


def score64(q_row, w_row, pooled_rows):
    """float64 pool scores and their 8 eps32 sum|terms| bounds (design 6.2 (b))."""
    import numpy as np

    q64, w64, k64 = (np.asarray(v, np.float64) for v in (q_row, w_row, pooled_rows))
    dots = q64 @ k64.T * 128**-0.5  # (32, n)
    exact = (np.maximum(dots, 0) * w64[:, None]).sum(axis=0)
    terms = np.abs(q64) @ np.abs(k64).T * 128**-0.5
    return exact, 8 * 2.0**-23 * (np.abs(w64)[:, None] * terms).sum(axis=0)


def emulated_blocked(ix, masked, count, block):
    """The blocked running top-k law applied to the SAME (global) scores:
    a pure selection-law check, independent of block-shaped score GEMMs."""
    import mlx.core as mx

    length, pools = masked.shape[1], masked.shape[2]
    best_s = mx.zeros((1, length, 0), mx.float32)
    best_i = mx.zeros((1, length, 0), mx.int32)
    for start in range(0, pools, block):
        stop = min(pools, start + block)
        sc = mx.concatenate((best_s, masked[..., start:stop]), axis=-1)
        ids = mx.concatenate((best_i, mx.broadcast_to(mx.arange(start, stop, dtype=mx.int32), (1, length, stop - start))),
                             axis=-1)
        keep = ix.stable_topk(sc, min(count, sc.shape[-1]))
        best_s, best_i = mx.take_along_axis(sc, keep, axis=-1), mx.take_along_axis(ids, keep, axis=-1)
    return best_i, best_s > -mx.inf


def blocked_vs_global(mx, ix, item, block):
    """Global topk_v1 vs blocked selection on real data, in 128-query chunks (9.6).

    Two statements: (1) on identical scores the blocked running law selects
    exactly the global ids (selection law); (2) blocked_select, which scores
    each block with its own GEMM shape, is compared word for word with the
    global select, and every difference is classified: ``bounded_score_flip``
    when both arms' scores of the moved pools are within 8 eps32 sum|terms| of
    float64 (a consequence of the scoring shape, recorded with both arms'
    512th/513th gaps and the inter-arm score difference), else ``unexplained``.
    """
    import numpy as np

    pooled = ix.pool_fixed_order(mx.array(item["raw"]), mx.array(item["ape"]))[None]
    pooled_np = np_of(mx, pooled[0])
    pools = pooled.shape[1]
    count = min(ix.TOPK // ix.KPOOL, pools)
    law_words, words, diffs = 0, 0, []
    for start in range(0, item["q"].shape[0], ix.QUERY_CHUNK):
        q = mx.array(item["q"][start : start + ix.QUERY_CHUNK])[None]
        w = mx.array(item["w"][start : start + ix.QUERY_CHUNK])[None]
        pos = mx.array(item["positions"][start : start + ix.QUERY_CHUNK])
        scores = ix.pool_scores(q, w, pooled)
        visible = ix.visible_pools(pos, pools)[None]
        masked = ix._masked(scores, visible)
        glob = ix.expand_selection(*ix.select_pools(scores, visible, count, law="topk_v1"), pos)
        law = ix.expand_selection(*emulated_blocked(ix, masked, count, block), pos)
        # Capture the producer's own block scores (each block's GEMM shape).
        blocks, original = [], ix.pool_scores

        def capture(q_, w_, keys_, original=original, blocks=blocks):
            out = original(q_, w_, keys_)
            blocks.append(out)
            return out

        ix.pool_scores = capture
        try:
            blk = ix.expand_selection(*ix.blocked_select(q, w, pooled, pos, count, block=block), pos)
            mx.eval(blk)
        finally:
            ix.pool_scores = original
        arm_scores = mx.concatenate(blocks, axis=-1)
        # blocked_select's ids must be the chronological top-k of its OWN scores.
        own = ix.expand_selection(*ix.select_pools(arm_scores, visible, count, law="topk_v1"), pos)
        g, e, b = np_of(mx, glob), np_of(mx, law), np_of(mx, blk)
        own_rows = (np_of(mx, own)[0] == b[0]).all(axis=-1)
        law_ok, law_n = bitwise(g, e)
        law_words += 0 if law_ok else max(law_n, 1)
        same, n = bitwise(g, b)
        if same:
            continue
        words += n
        masked_np = np_of(mx, masked)[0]
        arm_np = np_of(mx, ix._masked(arm_scores, visible))[0]
        for row in np.nonzero((g[0] != b[0]).any(axis=-1))[0]:
            arm = arm_np[row]
            gp, bp = set(g[0, row][g[0, row] >= 0] // 4), set(b[0, row][b[0, row] >= 0] // 4)
            moved = sorted(int(x) for x in gp ^ bp)
            exact, bound = score64(item["q"][start + row], item["w"][start + row], pooled_np[moved])
            within = (np.abs(masked_np[row, moved] - exact) <= bound) & (np.abs(arm[moved] - exact) <= bound)
            explained = bool(own_rows[row]) and bool(within.all())

            def gap(v):
                o = np.sort(v)[::-1]
                return float(o[count - 1] - o[count]) if o.shape[0] > count else None

            diffs.append({"query": int(start + row), "pools_moved": moved[:16],
                          "global_scores": [float(masked_np[row, m]) for m in moved[:16]],
                          "blocked_scores": [float(arm[m]) for m in moved[:16]],
                          "inter_arm_max_abs": float(np.max(np.abs(masked_np[row, moved] - arm[moved]))),
                          "gap_512_513": {"global": gap(masked_np[row]), "blocked": gap(arm)},
                          "ids_are_topk_of_own_scores": bool(own_rows[row]),
                          "classification": "bounded_score_flip" if explained else "unexplained"})
    return {"block": block, "pools": pools, "selection_law_words": law_words, "differing_words": words,
            "differences": diffs[:32], "difference_queries": len(diffs),
            "unexplained": sum(d["classification"] == "unexplained" for d in diffs)}


def gate_select(args):
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import glm53_gguf_indexer as ix

    gate = Gate("select", args)
    error = None
    adapter = restore = None
    try:
        comparator_selftest(gate)
        mx.set_default_device(mx.gpu)
        try:
            first_use = ix.select_kernel_self_check()
            gate_error = None
        except Exception as exc:  # noqa: BLE001 -- the receipt carries it
            first_use, gate_error = False, repr(exc)
        gate.check("kernel_first_use_gate", first_use is True and ix.select_kernel_ready(),
                   gate_error=gate_error, route=ix.route_status()["kernels"])
        failures, launches = [], ix._SCHEDULED["select_kernel"]
        for seed in (11, 12):
            for name, s, pos, count in synthetic_select_cases(seed):
                a = np_of(mx, ix.select_expand(mx.array(s), mx.array(pos), count))
                e = np_of(mx, ix.select_expand_ops(mx.array(s), mx.array(pos), count))
                same, words = bitwise(a, e)
                if not same:
                    failures.append({"seed": seed, "case": name, "words": words})
        gate.check("kernel_vs_ops_synthetic", not failures and ix._SCHEDULED["select_kernel"] > launches,
                   failures=failures, launches=ix._SCHEDULED["select_kernel"] - launches)

        adapter = load(gate, args, mx)
        model = adapter.model
        layers = mla_layers(model)
        targets = [layers[0], layers[len(layers) // 2], layers[-1]]
        use(gate, adapter, "capture", {**REFERENCE, "glm53_indexer_pools": "recompute_fixed_order",
                                       "glm53_indexer_select": "topk_v1"})
        captured, restore = capture_selection(model, targets, mx)
        ids = needle_prompt(adapter, args.tokens, minimum=16384)  # P >= 4096 for R3 and the blocked cells
        cache, logits = prefill(model, ids, args.chunk, mx)
        greedy(model, cache, logits, 2, mx)
        restore()
        restore = None
        gate.receipt["captured"] = {f"{k[0]}:{k[1]}": {"total": v["total"], "queries": int(v["q"].shape[0])}
                                    for k, v in captured.items()}
        gate.check("captured_real_operands", len(captured) == 2 * len(targets), prompt_tokens=len(ids),
                   layers=targets)

        kernel_fail, nonfinite, blocked, law_audits = [], {}, [], {}
        for (prefix, phase), item in sorted(captured.items()):
            pooled = ix.pool_fixed_order(mx.array(item["raw"]), mx.array(item["ape"]))[None]
            pools = pooled.shape[1]
            count = min(ix.TOPK // ix.KPOOL, pools)
            q, w = mx.array(item["q"])[None], mx.array(item["w"])[None]
            pos = mx.array(item["positions"])
            scores = ix.pool_scores(q, w, pooled)
            nonfinite[f"{prefix}:{phase}"] = int(np.count_nonzero(~np.isfinite(np_of(mx, scores))))
            for start in range(0, q.shape[1], ix.QUERY_CHUNK):
                s = scores[0, start : start + ix.QUERY_CHUNK]
                p = pos[start : start + ix.QUERY_CHUNK]
                same, words = bitwise(np_of(mx, ix.select_expand(s, p, count)),
                                      np_of(mx, ix.select_expand_ops(s, p, count)))
                if not same:
                    kernel_fail.append({"layer": prefix, "phase": phase, "start": start, "words": words})
            if phase == "prompt":
                for block in (256, 1024, 2048):
                    if pools > block:
                        blocked.append({"layer": prefix, **blocked_vs_global(mx, ix, item, block)})
                rows = slice(0, min(128, item["q"].shape[0]))
                law_audits[prefix] = pool_law_audit(item["raw"], item["ape"], item["q"][rows], item["w"][rows],
                                                    item["positions"][rows], count)
        gate.check("kernel_vs_ops_real", not kernel_fail, failures=kernel_fail)
        gate.check("no_nonfinite_real_scores", not any(nonfinite.values()), nonfinite=nonfinite)
        gate.check("blocked_selection_law_on_identical_scores",
                   bool(blocked) and all(b["selection_law_words"] == 0 for b in blocked),
                   words=[(b["layer"], b["block"], b["selection_law_words"]) for b in blocked])
        # Differences from block-shaped score GEMMs are recorded and classified,
        # never silently accepted: only bounded score flips may pass.
        gate.check("blocked_select_real_classified", bool(blocked) and all(b["unexplained"] == 0 for b in blocked),
                   exact=all(b["differing_words"] == 0 for b in blocked), results=blocked)
        violations = {k: v["violations"] for k, v in law_audits.items() if v["violations"]}
        gate.check("r9_score_and_pool_bounds", not violations, audits=law_audits)
        gate.falsifier("R9", bool(violations), violations=violations)

        timing = {}
        decode_item = captured.get((f"blk.{targets[-1]}", "decode"))
        if decode_item is not None:
            pooled = ix.pool_fixed_order(mx.array(decode_item["raw"]), mx.array(decode_item["ape"]))[None]
            q, w = mx.array(decode_item["q"])[None], mx.array(decode_item["w"])[None]
            pos = mx.array(decode_item["positions"])
            full = ix.pool_scores(q, w, pooled)
            mx.eval(full)
            for pools in (1024, 4096):
                if pools > full.shape[-1]:
                    raise RuntimeError(f"R3 timing needs {pools} pools; the capture has {full.shape[-1]}")
                s = full[..., :pools]
                count = min(512, pools)
                vis = ix.visible_pools(pos, pools)[None]
                def ops(law, s=s, vis=vis, count=count):
                    return ix.expand_selection(*ix.select_pools(s, vis, count, law=law), pos)

                timing[pools] = {
                    "argsort_ms": time_ms(mx, lambda: ops("argsort")),
                    "topk_v1_ops_ms": time_ms(mx, lambda: ops("topk_v1")),
                    "kernel_ms": time_ms(mx, lambda s=s, count=count: ix.select_expand(s[0], pos, count)),
                }
        gate.receipt["select_timing_per_layer_ms"] = timing
        gate.falsifier("R3", any(t["kernel_ms"] >= t["argsort_ms"] for t in timing.values()) if timing else None,
                       timing=timing, rule="kernel per-layer select+expand not below argsort at P in {1024, 4096}")
    except BaseException as exc:  # noqa: BLE001
        error = exc
    finally:
        if restore is not None:
            restore()
        if adapter is not None:
            adapter.close()
    return gate.finish(mx, error)


# ---------------------------------------------------------------------------
# pools

def gate_pools(args):
    import mlx.core as mx
    import numpy as np
    from glm53_policy_ab import pool_bounds

    from mlx2.runtime.models import glm53_gguf_indexer as ix

    gate = Gate("pools", args)
    error = None
    adapter = None
    try:
        comparator_selftest(gate)
        adapter = load(gate, args, mx)
        model = adapter.model
        merged = use(gate, adapter, "audit", {**SELECTION, "glm53_indexer_pools": "incremental_v1_audit"})
        ids = needle_prompt(adapter, args.tokens, minimum=16384)  # T = 16384 must be covered
        before = dict(model.calls)
        try:
            cache, logits = prefill(model, ids, args.chunk, mx)
            greedy(model, cache, logits, args.decode, mx)
            audit_error = None
        except ValueError as exc:
            cache, audit_error = None, repr(exc)
        eng = engagement(merged, before, dict(model.calls), context_tokens=len(ids) + args.decode,
                         prompt_tokens=len(ids), decode_steps=args.decode)
        gate.check("model_path_incremental_audit", audit_error is None and eng["engaged"],
                   audit_error=audit_error, engagement=eng, prompt_tokens=len(ids))
        if cache is None:
            raise RuntimeError("no cache to audit offline")
        lengths = sorted({t for t in (2049, 4096, 16384) if t <= len(ids)} | {len(ids)})
        rng = np.random.default_rng(5303)
        failures, invariance, bounds = [], [], {}
        for layer in mla_layers(model):
            prefix = f"blk.{layer}"
            ape = adapter.weights.array(prefix + ".indexer.kpool_ape")
            keys = cache[layer].caches[1].keys
            raw_all = keys[0, 0, : len(ids)]
            mx.eval(raw_all)
            fn = lambda a, b, raw_all=raw_all: raw_all[a:b]
            longest = ix.pool_fixed_order(raw_all[: 4 * (len(ids) // 4)], ape)
            mx.eval(longest)
            for total in lengths:
                full = np_of(mx, ix.pool_fixed_order(raw_all[: 4 * (total // 4)], ape))
                same, words = bitwise(full, np_of(mx, longest[: total // 4]))
                if not same:
                    invariance.append({"layer": layer, "total": total, "words": words})
                variants = {}
                one = ix.Glm53PoolCache()
                variants["one_shot"] = one.extend(fn, total, ape)
                chunked = ix.Glm53PoolCache()
                for t in list(range(2048, total, 2048)) + [total]:
                    out = chunked.extend(fn, t, ape)
                variants["chunks_2048"] = out
                stepped, t = ix.Glm53PoolCache(), 0
                while t < total:
                    t = min(total, t + int(rng.integers(1, 10 if t < 4096 else 301)))
                    out = stepped.extend(fn, t, ape)
                    if rng.random() < 0.02:
                        mx.eval(stepped.pooled)
                variants["random_steps"] = out
                cut = ix.Glm53PoolCache()
                cut.extend(fn, total, ape)
                cut.cut(max(0, total - 5))
                variants["cut_reextend"] = cut.extend(fn, total, ape)
                for name, value in variants.items():
                    same, words = bitwise(np_of(mx, value[0]), full)
                    if not same:
                        failures.append({"layer": layer, "total": total, "variant": name, "words": words})
            bounds[prefix] = pool_bounds(np.array(raw_all), np.array(ape))["a"]
        gate.check("extend_equals_recompute_bitwise", not failures, failures=failures, lengths=lengths)
        gate.check("pool_shape_invariance_bitwise", not invariance, failures=invariance)
        violations = {k: [n for n, v in b.items() if not math.isfinite(v) or v > 1.0] for k, b in bounds.items()}
        violations = {k: v for k, v in violations.items() if v}
        gate.check("r9a_pool_bounds", not violations, bound_ratios=bounds)
        gate.falsifier("R5", bool(failures) or audit_error is not None, failures=len(failures),
                       audit_error=audit_error)
    except BaseException as exc:  # noqa: BLE001
        error = exc
    finally:
        if adapter is not None:
            adapter.close()
    return gate.finish(mx, error)


# ---------------------------------------------------------------------------
# attention

def causal_ids(rng, total, length, *, pad_every=4, empty_row=None):
    """(1, L, W) ascending ids for the last L positions of a T-token latent."""
    import numpy as np

    ids = np.full((length, WIDTH), -1, np.int32)
    for row in range(length):
        if row == empty_row:
            continue
        visible = total - length + row + 1
        n = min(WIDTH - (row % pad_every), visible)
        ids[row, :n] = np.sort(rng.choice(visible, size=n, replace=False))
    return ids[None]


def oracle64(q, latent, ids):
    """float64 gathered attention: q (1,H,L,D), latent (T,D), ids (1,L,W) -> (1,H,L,D)."""
    import numpy as np

    q64 = q.astype(np.float64)
    out = np.full(q.shape, np.nan)
    for row in range(ids.shape[1]):
        sel = ids[0, row][ids[0, row] >= 0]
        if not sel.size:
            continue
        keys = latent[sel].astype(np.float64)
        s = q64[0, :, row] @ keys.T / 16.0
        p = np.exp(s - s.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        out[0, :, row] = p @ keys
    return out


def errors(x, ref):
    import numpy as np

    rows = np.isfinite(ref).all(axis=(0, 1, 3))
    if not rows.any():
        return {"max_abs": None, "rel_rms": None}
    a, r = x[:, :, rows].astype(np.float64), ref[:, :, rows]
    return {"max_abs": float(np.max(np.abs(a - r))),
            "rel_rms": float(np.sqrt(np.mean((a - r) ** 2)) / np.sqrt(np.mean(r**2)))}


def mask_consumer(q, latent, ids, mx):
    """The model's mask_reference consumer (Glm53GGUFModel._mla_masked), decode phase."""
    from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel

    stub = SimpleNamespace(calls=defaultdict(int), mla_serial_pieces=False,
                           _prompt_mask=Glm53GGUFModel._prompt_mask)
    return Glm53GGUFModel._mla_masked(stub, q, latent, ids, q.shape[2], latent.shape[2])


def block_invariance(mx, sa, q, latent, ids):
    """sparse_prefill at blocks 1/4/32/L: every pair bitwise (cold/warm law).

    One-row decode is recorded but not required to match: Metal SDPA runs a
    one-query batch on another kernel, and decode rows are not prompt rows.
    """
    length = q.shape[2]
    outs = {f"block{b}": np_of(mx, sa.sparse_prefill(q, latent, ids, block=b)) for b in (1, 4, 32, length)}
    rows = [sa.sparse_decode(q[:, :, r : r + 1], latent, ids[:, r : r + 1], policy="gathered_v1")
            for r in range(length)]
    outs["decode_rows"] = np_of(mx, mx.concatenate(rows, axis=2))
    ref = outs["block32"]
    diffs = {k: bitwise(v, ref)[1] for k, v in outs.items()}
    return all(d == 0 for k, d in diffs.items() if k != "decode_rows"), diffs


def peak_above_active(mx, fn, inputs):
    mx.eval(inputs)
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    out = fn()
    mx.eval(out)
    return mx.get_peak_memory() - base


def gate_attention(args):
    import mlx.core as mx
    import numpy as np

    from mlx2.adapters.glm53_gguf_memory import GATHERED_PREFILL_BLOCK_BYTES
    from mlx2.runtime.models import glm53_sparse_attention as sa

    gate = Gate("attention", args)
    error = None
    adapter = restore = None
    try:
        comparator_selftest(gate)
        mx.set_default_device(mx.gpu)
        try:
            fused_gate, fused_error = sa.self_check(), None
        except Exception as exc:  # noqa: BLE001
            fused_gate, fused_error = False, repr(exc)
        gate.check("fused_first_use_gate", fused_gate is True, error=fused_error, route=sa.route_status())
        rng = np.random.default_rng(5304)
        oracle_rows, nan_rows = [], []
        for total in (4096, 16384):
            latent_np = rng.normal(size=(total, LATENT)).astype(np.float32)
            latent = mx.array(latent_np)[None, None]
            for length in (1, 8):
                q_np = rng.normal(size=(1, HEADS, length, LATENT)).astype(np.float32)
                ids_np = causal_ids(rng, total, length, empty_row=length - 1 if length > 1 else None)
                q, ids = mx.array(q_np), mx.array(ids_np)
                ref = oracle64(q_np, latent_np, ids_np)
                got = {"gathered_v1": np_of(mx, sa.sparse_decode(q, latent, ids, policy="gathered_v1")),
                       "mask_reference": np_of(mx, mask_consumer(q, latent, ids, mx))}
                if fused_gate is True:
                    got["fused_decode_v1"] = np_of(mx, sa.fused_decode_attention(q, latent, ids))
                # The prompt law (batch >= 2, one-query blocks padded) is its own consumer
                # since decode rows run one query per launch: oracle-bound it too.
                got["gathered_prefill"] = np_of(mx, sa.sparse_prefill(q, latent, ids))
                row = {"total": total, "length": length,
                       **{k: errors(v, ref) for k, v in got.items()}}
                if "fused_decode_v1" in got:
                    finite = np.isfinite(ref).all(axis=(0, 1, 3))
                    row["fused_vs_gathered_max_abs"] = float(np.max(np.abs(
                        got["fused_decode_v1"][:, :, finite] - got["gathered_v1"][:, :, finite])))
                oracle_rows.append(row)
                if length > 1:
                    empty = length - 1
                    nan_rows.append({"total": total,
                                     "gathered_nan": bool(np.isnan(got["gathered_v1"][:, :, empty]).all())
                                     and bool(np.isnan(got["gathered_prefill"][:, :, empty]).all()),
                                     "fused_nan": bool(np.isnan(got["fused_decode_v1"][:, :, empty]).all())
                                     if "fused_decode_v1" in got else None,
                                     "mask_finite": bool(np.isfinite(got["mask_reference"][:, :, empty]).all())})
        # Every new consumer measured, fused included (9.5 / 9.7); a missing or NaN error is a
        # failure. mask_reference is today's law: recorded, not gated (it measured 1.13e-6
        # rel RMS at L = 8 on 2026-10-31, above the design's 1e-6).
        consumers = ("gathered_v1", "gathered_prefill") + (("fused_decode_v1",) if fused_gate is True else ())
        # Metal fp32 SDPA itself sits near 1e-6 rel RMS of float64 (today's mask path
        # 0.95-1.13e-6, 2026-10-31), so a new consumer passes at 1e-6 or at most 1.25x
        # today's error on the same row; max_abs stays 1e-5. A consumer meaningfully
        # less accurate than today's law still fails.
        within = all(r[k]["max_abs"] is not None and r[k]["max_abs"] <= 1e-5
                     and r[k]["rel_rms"] <= max(1e-6, 1.25 * r["mask_reference"]["rel_rms"])
                     for r in oracle_rows for k in consumers)
        gate.check("oracle_bound_9_5", within, rows=oracle_rows,
                   bound={"max_abs": 1e-5, "rel_rms": "max(1e-6, 1.25 x mask_reference on the row)"})
        if fused_gate is True:
            gate.check("fused_within_1e-5_of_gathered",
                       all(r["fused_vs_gathered_max_abs"] <= 1e-5 for r in oracle_rows), rows=oracle_rows)
        gate.check("empty_selection_contract", all(r["gathered_nan"] and r["fused_nan"] in (True, None)
                                                   for r in nan_rows), rows=nan_rows)

        total, length = 16384, 64
        latent = mx.array(rng.normal(size=(1, 1, total, LATENT)).astype(np.float32))
        q = mx.array(rng.normal(size=(1, HEADS, length, LATENT)).astype(np.float32))
        ids = mx.array(causal_ids(rng, total, length))
        same, diffs = block_invariance(mx, sa, q, latent, ids)
        gate.check("block_invariance_synthetic", same, differing_words=diffs)

        peaks = {}
        for queries in (32, 256):
            qq = mx.array(rng.normal(size=(1, HEADS, queries, LATENT)).astype(np.float32))
            ii = mx.array(causal_ids(rng, total, queries))
            peaks[queries] = peak_above_active(mx, lambda qq=qq, ii=ii: sa.sparse_prefill(qq, latent, ii),
                                               [qq, ii, latent])
        growth = peaks[256] - peaks[32]
        allowance = GATHERED_PREFILL_BLOCK_BYTES // 2 + (256 - 32) * HEADS * LATENT * 4 * 3
        gate.check("one_live_prefill_block", growth <= allowance, peak_bytes=peaks, growth=growth,
                   allowance=allowance, block_workspace=GATHERED_PREFILL_BLOCK_BYTES)

        timing = {}
        for t in (2560, 4096, 8192, 16384):
            lat = latent[:, :, :t]
            q1 = mx.array(rng.normal(size=(1, HEADS, 1, LATENT)).astype(np.float32))
            i1 = mx.array(causal_ids(rng, t, 1))
            mx.eval(lat, q1, i1)
            row = {"mask_reference_ms": time_ms(mx, lambda q1=q1, lat=lat, i1=i1: mask_consumer(q1, lat, i1, mx)),
                   "gathered_v1_ms": time_ms(mx, lambda q1=q1, lat=lat, i1=i1: sa.sparse_decode(
                       q1, lat, i1, policy="gathered_v1"))}
            if fused_gate is True:
                row["fused_decode_v1_ms"] = time_ms(mx, lambda q1=q1, lat=lat, i1=i1: sa.fused_decode_attention(
                    q1, lat, i1))
            timing[t] = row
        gate.receipt["attention_timing_per_layer_ms"] = timing
        hi = timing[16384]
        gate.falsifier("R4", None if "fused_decode_v1_ms" not in hi else hi["fused_decode_v1_ms"] > 0.8 * hi["gathered_v1_ms"],
                       rule="fused per-layer time <= 0.8 x gathered at 16K", at_16k=hi)
        gate.falsifier("R2_op_level", timing[4096]["gathered_v1_ms"] * 0.97 > timing[4096]["mask_reference_ms"],
                       rule="op-level only; the model-level R2 is the policy A/B at 4K", at_4k=timing[4096])

        if args.capture_tokens:
            adapter = load(gate, args, mx)
            model = adapter.model
            use(gate, adapter, "capture", {**GATHERED, "glm53_sparse_prefill": "gathered_v1_all"})
            captured = {}
            original = sa.sparse_prefill

            def capture(q, latent, ids, **kwargs):
                if "q" not in captured and q.shape[2] >= 64 and not isinstance(latent, sa.TreeLatentView):
                    keep = min(q.shape[2], 96)
                    captured.update(q=q[:, :, :keep], latent=latent, ids=ids[:, :keep])
                    mx.eval(captured["q"], captured["latent"], captured["ids"])
                return original(q, latent, ids, **kwargs)

            sa.sparse_prefill = capture
            restore = lambda: setattr(sa, "sparse_prefill", original)
            ids_prompt = needle_prompt(adapter, args.capture_tokens)
            prefill(model, ids_prompt, args.chunk, mx)
            restore()
            restore = None
            gate.check("captured_real_operands", "q" in captured, prompt_tokens=len(ids_prompt))
            if "q" in captured:
                same, diffs = block_invariance(mx, sa, captured["q"], captured["latent"], captured["ids"])
                gate.check("block_invariance_real", same, differing_words=diffs)
                g = np_of(mx, sa.sparse_decode(captured["q"], captured["latent"], captured["ids"], policy="gathered_v1"))
                p = np_of(mx, sa.sparse_prefill(captured["q"], captured["latent"], captured["ids"]))
                real = {"gathered_finite": bool(np.isfinite(g).all()), "prefill_finite": bool(np.isfinite(p).all())}
                q_np = np.array(captured["q"][:, :, :4])
                ref = oracle64(q_np, np.array(captured["latent"][0, 0]), np.array(captured["ids"][:, :4]))
                real["gathered_vs_f64"] = errors(g[:, :, :4], ref)
                real["prefill_vs_f64"] = errors(p[:, :, :4], ref)
                if fused_gate is True:
                    f = np_of(mx, sa.fused_decode_attention(captured["q"], captured["latent"], captured["ids"]))
                    real["fused_vs_gathered"] = errors(f, g.astype(np.float64))
                    real["fused_finite"] = bool(np.isfinite(f).all())
                gate.check("real_operands_finite", real["gathered_finite"] and real["prefill_finite"]
                           and real.get("fused_finite", True), **real)
                pe = real["prefill_vs_f64"]
                gate.check("real_prefill_oracle_bound", pe["max_abs"] is not None and pe["max_abs"] <= 1e-5,
                           **pe)
    except BaseException as exc:  # noqa: BLE001
        error = exc
    finally:
        if restore is not None:
            restore()
        if adapter is not None:
            adapter.close()
    return gate.finish(mx, error)


# ---------------------------------------------------------------------------
# rows

def gate_rows(args):
    import mlx.core as mx
    import numpy as np

    from mlx2.runtime.models import glm53_sparse_attention as sa

    gate = Gate("rows", args)
    error = None
    saved = sa.BATCHED_ROWS_EXACT
    try:
        comparator_selftest(gate)
        mx.set_default_device(mx.gpu)
        rng = np.random.default_rng(5305)
        base_np = rng.normal(size=(20480, LATENT)).astype(np.float32)  # t0 reaches 17761 at R = 16
        base = mx.array(base_np)
        try:
            fused = sa.self_check() is True
        except Exception:  # noqa: BLE001 -- recorded by the attention gate
            fused = False
        results, batched_law = {}, {}
        for rows in (2, 4, 8, 16):
            sources = []
            for r in range(rows):
                if r % 2:
                    t0 = 4096 + 911 * r
                    tail = mx.array(rng.normal(size=(1 + r % 4, LATENT)).astype(np.float32))
                    source = sa.TreeLatentView(base[:t0], tail, t0)
                    total = t0 + tail.shape[0]
                else:
                    total = 2049 + 997 * r
                    source = base[:total][None, None]
                ids = mx.array(causal_ids(rng, total, 1, pad_every=3 + r))
                sources.append((source, ids))
            q = mx.array(rng.normal(size=(rows, HEADS, 1, LATENT)).astype(np.float32))
            # The committed setting (per-row unless BATCHED_ROWS_EXACT) is what V1 gates.
            batched = np_of(mx, sa.rows_attention(q, sources, policy="gathered_v1"))
            sa.BATCHED_ROWS_EXACT = True  # the batched law itself, for R7 only
            batched_law[rows] = bitwise(np_of(mx, sa.rows_attention(q, sources, policy="gathered_v1")),
                                        np_of(mx, mx.concatenate([sa.sparse_decode(q[r][None], src, ids,
                                                                                   policy="gathered_v1")
                                                                  for r, (src, ids) in enumerate(sources)],
                                                                 axis=0)))[1]
            sa.BATCHED_ROWS_EXACT = saved
            one = np_of(mx, mx.concatenate([sa.sparse_decode(q[r][None], src, ids, policy="gathered_v1")
                                            for r, (src, ids) in enumerate(sources)], axis=0))
            order = list(range(rows))[::-1]
            permuted = np_of(mx, sa.rows_attention(q[mx.array(order)], [sources[r] for r in order],
                                                   policy="gathered_v1"))[np.argsort(order)]
            row = {"batched_vs_one_row_words": bitwise(batched, one)[1],
                   "permuted_vs_batched_words": bitwise(permuted, batched)[1]}
            if fused:
                fr = np_of(mx, sa.rows_attention(q, sources, policy="fused_decode_v1"))
                fo = np_of(mx, mx.concatenate([sa.fused_decode_attention(q[r][None], src, ids)
                                               for r, (src, ids) in enumerate(sources)], axis=0))
                row["fused_rows_vs_one_row_words"] = bitwise(fr, fo)[1]
            results[rows] = row
        exact = all(v == 0 for row in results.values() for v in row.values())
        gate.check("rows_bitwise_v1", exact, results=results, fused_checked=fused,
                   batched_rows_exact=saved)
        gate.falsifier("R7", not all(v == 0 for v in batched_law.values()), batched_law_words=batched_law,
                       rule="batched SDPA not bitwise -> set BATCHED_ROWS_EXACT = False (per-row fallback)")
    except BaseException as exc:  # noqa: BLE001
        error = exc
    finally:
        sa.BATCHED_ROWS_EXACT = saved
    return gate.finish(mx, error)


# ---------------------------------------------------------------------------
# decode

def capture_first_step_ids(model):
    """Wrap model._mla_attend: when ``state['on']``, keep each MLA call's ids."""
    import numpy as np

    state = {"on": False, "ids": []}
    original = model._mla_attend

    def wrapped(q, selected, source):
        if state["on"] and selected is not None:
            state["ids"].append(np.array(selected))
        return original(q, selected, source)

    model._mla_attend = wrapped
    return state, lambda: model.__dict__.pop("_mla_attend", None)


def gate_decode(args):
    import mlx.core as mx

    from mlx2.adapters.glm53_ast_target import clone

    gate = Gate("decode", args)
    error = None
    adapter = restore = None
    try:
        comparator_selftest(gate)
        adapter = load(gate, args, mx)
        model = adapter.model
        arms = {"reference": REFERENCE, "mask": SELECTION, "gathered": GATHERED}
        if not args.skip_fused:
            arms["fused"] = {**SELECTION, "glm53_sparse_attention": "fused_decode_v1"}
        capture, restore = capture_first_step_ids(model)
        cells = []
        for tokens in [int(t) for t in args.tokens_list.split(",")]:
            ids = needle_prompt(adapter, tokens)
            runs, shared = {}, None
            for name, overrides in arms.items():
                try:
                    merged = use(gate, adapter, name, overrides)
                    if name in ("reference", "mask"):
                        # The mask arm's prompt phase is the gathered/fused arms' too
                        # (same pools, selection and dense_reference prefill).
                        start, first = prefill(model, ids, args.chunk, mx)
                        if name == "mask":
                            shared = (start, first)
                    else:
                        start, first = shared
                    cache = clone(start)  # ``start`` stays the decode-start snapshot for replays
                    before = dict(model.calls)
                    # First decode step (the token of the prefill logits) with id capture.
                    token = int(mx.argmax(first[0, -1]).item())
                    capture.update(on=True, ids=[])
                    step = model(mx.array([[token]]), cache, logits_to_keep=1)
                    mx.eval(step)
                    capture["on"] = False
                    first_ids = capture["ids"]
                    rest, finite = greedy(model, cache, step, args.decode - 1, mx)
                    finite &= bool(mx.all(mx.isfinite(first[0, -1])).item())
                    out = [token, *rest]
                    served, text = served_ids(adapter, out)
                    runs[name] = {"policy": merged, "start": start, "first": first, "output_ids": out,
                                  "served": served, "finite": finite, "text": text,
                                  "needle_found": needle_found(text), "first_step_ids": first_ids,
                                  "engagement": engagement(merged, before, dict(model.calls),
                                                           context_tokens=len(ids) + args.decode,
                                                           prompt_tokens=0, decode_steps=args.decode)}
                except Exception as exc:  # noqa: BLE001 -- an arm that raises is that arm's failure
                    capture["on"] = False
                    runs[name] = {"error": repr(exc)}
            ok_runs = {k: v for k, v in runs.items() if "error" not in v}
            same_ids = {}
            for name in ("gathered", "fused"):
                if name in ok_runs and "mask" in ok_runs:
                    a, b = ok_runs["mask"]["first_step_ids"], ok_runs[name]["first_step_ids"]
                    same_ids[name] = len(a) == len(b) and len(a) > 0 and all(bitwise(x, y)[0] for x, y in zip(a, b))
            pairs = [("reference", "mask"), ("mask", "gathered")] + ([("gathered", "fused")] if "fused" in arms else [])
            comparisons = []
            for ref, cand in pairs:
                if ref in ok_runs and cand in ok_runs:
                    comparisons.append({"reference": ref, "candidate": cand,
                                        **compare_runs(gate, adapter, ok_runs[ref], ok_runs[cand], mx)})
            keep = ("policy", "output_ids", "served", "finite", "text", "needle_found", "engagement", "error")
            cells.append({"prompt_tokens": len(ids), "same_selection_ids_9_1": same_ids, "comparisons": comparisons,
                          "runs": {k: {kk: vv for kk, vv in v.items() if kk in keep} for k, v in runs.items()}})
            gate.receipt["cells"] = cells
            gate.flush()
            gate.check(f"arms_ran@{tokens}", len(ok_runs) == len(arms),
                       errors={k: v["error"] for k, v in runs.items() if "error" in v})
            gate.check(f"selection_shared_across_consumers@{tokens}",
                       bool(same_ids) and len(same_ids) == len(arms) - 2 and all(same_ids.values()), same=same_ids)
            gate.check(f"cross_policy_classified@{tokens}",
                       len(comparisons) == len(pairs) and all(comparison_passes(c) for c in comparisons),
                       verdicts=[(c["reference"], c["candidate"], c["verdict"], c.get("clause"),
                                  c.get("first_divergence")) for c in comparisons])
            gate.check(f"engaged@{tokens}", all(r["engagement"]["engaged"] for r in ok_runs.values()),
                       missing={k: r["engagement"]["missing"] for k, r in ok_runs.items()})
            gate.check(f"every_arm_finite@{tokens}", all(r["finite"] for r in ok_runs.values()),
                       finite={k: r["finite"] for k, r in ok_runs.items()})
            ref_run = ok_runs.get("reference", {})
            gate.check(f"reference_oracle@{tokens}", bool(ref_run.get("needle_found")),
                       text=ref_run.get("text", "")[:200])
            del runs, ok_runs, shared
    except BaseException as exc:  # noqa: BLE001
        error = exc
    finally:
        if restore is not None:
            restore()
        if adapter is not None:
            adapter.close()
    return gate.finish(mx, error)


# ---------------------------------------------------------------------------
# tree

def gate_tree(args):
    import mlx.core as mx
    from glm53_state_oracle import run as oracle

    gate = Gate("tree", args)
    error = None
    adapter = None
    try:
        comparator_selftest(gate)
        adapter = load(gate, args, mx)
        if adapter.kernel_policy["glm53_ast_target"] != "layered":
            raise SystemExit("the tree gate needs a base policy with glm53_ast_target=layered")
        arms = [("reference", REFERENCE, ("mega_v1",))]
        for rows in ("per_row", "batched_v1"):
            for frontier in ("clone", "tails_v1"):
                arms.append((f"{rows}+{frontier}", {**GATHERED, "glm53_tree_rows_attention": rows,
                                                    "glm53_tree_frontier": frontier}, ("mega_v1", "reference")))
        if not args.skip_fused:
            arms.append(("batched_v1+tails_v1+fused", {**GATHERED, "glm53_sparse_attention": "fused_decode_v1",
                                                         "glm53_tree_rows_attention": "batched_v1",
                                                         "glm53_tree_frontier": "tails_v1"}, ("mega_v1",)))
        for context in [int(c) for c in args.contexts.split(",")]:
            if context <= 2048:
                # The depth-1 row must not select and a depth-3 row must (T0 + 1 <= 2048 < T0 + 3).
                ids = needle_prompt(adapter, 2047, minimum=2046)
                if not len(ids) + 1 <= 2048 < len(ids) + 3:
                    raise RuntimeError(f"straddle cell needs 2046..2047 prompt tokens, got {len(ids)}")
            else:
                ids = needle_prompt(adapter, context)
            for name, overrides, hcs in arms:
                for hc in hcs:
                    policy = {**adapter.kernel_policy, **overrides, "glm53_hc": hc}
                    gate.receipt["policies"][f"{name}/hc={hc}"] = {k: policy[k] for k in (*REFERENCE, "glm53_hc")}
                    label = f"{name}.hc-{hc}.ctx-{len(ids)}"
                    try:
                        result = oracle(adapter, gate.out / f"oracle-{label}.json", CHECKPOINT,
                                        gate.receipt["source_revision"], context_tokens=ids,
                                        kernel_policy=policy, prime_chunk=args.chunk)
                        ok, detail = result["status"] == "passed", [
                            {"capture_layers": v["capture_layers"], "engagement": v["engagement"]}
                            for v in result["variants"]]
                    except Exception as exc:  # noqa: BLE001 -- a failed oracle is this check's verdict
                        ok, detail = False, repr(exc)
                    gate.check(f"tree_oracle[{label}]", ok, detail=detail)
    except BaseException as exc:  # noqa: BLE001
        error = exc
    finally:
        if adapter is not None:
            adapter.close()
    return gate.finish(mx, error)


# ---------------------------------------------------------------------------
# replay

def leaves(value):
    import mlx.core as mx

    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in leaves(v)]
    return []


def serialized_roundtrip(cache, mx):
    """Serialize every cache through host memory and rebuild with from_state (an APCv2-style restore)."""
    import numpy as np

    def host(value):
        if isinstance(value, mx.array):
            if value.dtype == mx.bfloat16:
                return mx.array(np.array(value.astype(mx.float32))).astype(mx.bfloat16)
            return mx.array(np.array(value))
        if isinstance(value, (list, tuple)):
            return type(value)(host(v) for v in value)
        return value

    return [type(c).from_state(host(c.state), copy.deepcopy(c.meta_state)) for c in cache]


def compare_planes(a, b, mx):
    import numpy as np

    words, worst, metas = 0, 0.0, 0
    for x, y in zip(a, b, strict=True):
        metas += int(x.meta_state != y.meta_state)
        for p, q in zip(leaves(x.state), leaves(y.state), strict=True):
            pa, qa = np_of(mx, p), np_of(mx, q)
            same, n = bitwise(pa, qa)
            words += 0 if same else (n if n > 0 else pa.size or 1)
            if not same and pa.shape == qa.shape and pa.size and pa.dtype.kind == "f":
                worst = max(worst, float(np.max(np.abs(pa.astype(np.float64) - qa))))
    return {"differing_words": words, "max_abs": worst, "meta_mismatches": metas}


def compare_pools(a, b, mx):
    words = 0
    for x, y in zip(a, b):
        pa, pb = getattr(x, "pools", None), getattr(y, "pools", None)
        if pa is None and pb is None:
            continue
        if (pa is None) != (pb is None) or pa.complete != pb.complete:
            return -1
        if pa.pooled is not None:
            words += bitwise(np_of(mx, pa.pooled[:, : pa.complete]), np_of(mx, pb.pooled[:, : pb.complete]))[1]
    return words


def gate_replay(args):
    import mlx.core as mx

    from mlx2.adapters.glm53_ast_target import clone

    gate = Gate("replay", args)
    error = None
    adapter = None
    try:
        comparator_selftest(gate)
        adapter = load(gate, args, mx)
        model = adapter.model
        ids = needle_prompt(adapter, args.tokens)
        policies = {"reference": REFERENCE,
                    "candidate": {**GATHERED, "glm53_sparse_prefill": "gathered_v1_all"}}
        cells = []
        for name, overrides in policies.items():
            merged = use(gate, adapter, name, overrides)
            incremental = merged["glm53_indexer_pools"] in ("incremental_v1", "incremental_v1_audit")
            cold, cold_logits = prefill(model, ids, args.chunk, mx)
            cold_planes = clone(cold)  # committed planes at the end of the prompt (decode-start snapshot)
            cold_out, cold_finite = greedy(model, cold, cold_logits, args.decode, mx)
            cold_served, cold_text = served_ids(adapter, cold_out)
            cold_run = {"policy": merged, "start": cold_planes, "first": cold_logits, "served": cold_served,
                        "needle_found": needle_found(cold_text)}
            for split in [int(s) for s in args.splits.split(",")]:
                prefix, _ = prefill(model, ids, args.chunk, mx, stop=split)
                control, control_logits = prefill(model, ids, args.chunk, mx, cache=clone(prefix), start=split)
                restored = serialized_roundtrip(prefix, mx)
                dropped = all(getattr(c, "pools", None) is None for c, layer in zip(restored, model.layers)
                              if not layer.is_linear)
                before = model.calls["indexer_pool_recompute_full"]
                warm, warm_logits = prefill(model, ids, args.chunk, mx, cache=restored, start=split)
                rebuilt = model.calls["indexer_pool_recompute_full"] - before
                pools_words = compare_pools(warm, control, mx) if incremental else 0
                planes = compare_planes(warm, control, mx)
                evidence_planes = compare_planes(warm, cold_planes, mx)
                warm_start = clone(warm)
                control_out, control_finite = greedy(model, control, control_logits, args.decode, mx)
                warm_out, warm_finite = greedy(model, warm, warm_logits, args.decode, mx)
                warm_served, warm_text = served_ids(adapter, warm_out)
                unaligned = compare_runs(gate, adapter, cold_run, {
                    "policy": merged, "start": warm_start, "first": warm_logits, "served": warm_served,
                    "needle_found": needle_found(warm_text)}, mx)
                use(gate, adapter, name, overrides)
                del warm_start
                cell = {"policy": name, "split": split, "prompt_tokens": len(ids),
                        "ids": {"cold": cold_out, "control": control_out, "warm": warm_out},
                        "text": {"cold": cold_text, "warm": warm_text},
                        "finite": {"cold": cold_finite, "control": control_finite, "warm": warm_finite},
                        "pools_dropped_on_restore": dropped,
                        "pool_rebuild_calls": rebuilt, "restore_vs_unrestored_planes": planes,
                        "restore_vs_unrestored_pool_words": pools_words,

                        "evidence_unaligned_vs_uniform_cold": {
                            **unaligned, "planes": evidence_planes}}
                cells.append(cell)
                gate.receipt["cells"] = cells
                gate.flush()
                tokens_exact = warm_served == served_ids(adapter, control_out)[0]
                ok = (dropped and planes["differing_words"] == 0 and planes["meta_mismatches"] == 0
                      and pools_words == 0 and tokens_exact and (rebuilt > 0 or not incremental)
                      and warm_finite and control_finite and cold_finite)
                gate.check(f"restore_exact[{name}@{split}]", ok, planes=planes, pools_words=pools_words,
                           served_tokens_exact=tokens_exact, pools_dropped=dropped, rebuild_calls=rebuilt,
                           finite={"cold": cold_finite, "control": control_finite, "warm": warm_finite})
    except BaseException as exc:  # noqa: BLE001
        error = exc
    finally:
        if adapter is not None:
            adapter.close()
    return gate.finish(mx, error)


# ---------------------------------------------------------------------------
# memory

def gate_memory(args):
    import mlx.core as mx

    from mlx2.adapters.glm53_ast_target import Glm53TargetTreeExecutor, clone
    from mlx2.adapters.glm53_gguf_memory import Glm53CacheBudget
    from mlx2.runtime.models.glm53_sparse_attention import Glm53MLAFrontier

    gate = Gate("memory", args)
    error = None
    adapter = None
    try:
        comparator_selftest(gate)
        adapter = load(gate, args, mx)
        model = adapter.model
        ids = needle_prompt(adapter, args.tokens, minimum=16384)
        prefill_runs = {}
        live = None
        # Dense last: the arm kept resident for the decode/verify checks must not
        # sit under the other arms' absolute peaks (equal baselines for R6).
        for name, overrides in (("gathered_v1_8k", {**GATHERED, "glm53_sparse_prefill": "gathered_v1_8k"}),
                                ("gathered_v1_all", {**GATHERED, "glm53_sparse_prefill": "gathered_v1_all"}),
                                ("dense", SELECTION)):
            merged = use(gate, adapter, name, overrides)
            budget = Glm53CacheBudget.from_config(adapter.config, mtp=False, tree_rows=0, kernel_policy=merged)
            cache, chunks = model.make_cache(), []
            mx.clear_cache()
            calls = dict(model.calls)
            for begin in range(0, len(ids), args.chunk):
                piece = mx.array([ids[begin : begin + args.chunk]])
                base = mx.get_active_memory()
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                with model._prompt_scope(piece.shape[1]):
                    logits = model(piece, cache, logits_to_keep=1)
                mx.eval(logits, [c.state for c in cache])
                chunks.append({"start": begin, "stop": begin + piece.shape[1], "seconds": time.perf_counter() - t0,
                               "peak_bytes": mx.get_peak_memory(), "peak_above_active": mx.get_peak_memory() - base,
                               "booked": budget.prefill_transient_bytes(begin + piece.shape[1], piece.shape[1])})
            delta = {k: model.calls[k] - calls.get(k, 0) for k in model.calls if model.calls[k] != calls.get(k, 0)}
            prefill_runs[name] = {"chunks": chunks, "seconds": sum(c["seconds"] for c in chunks),
                                  "arm_peak_bytes": max(c["peak_bytes"] for c in chunks),
                                  "max_peak_above_active": max(c["peak_above_active"] for c in chunks),
                                  "over_booked": [c["start"] for c in chunks if c["peak_above_active"] > c["booked"]],
                                  "calls": delta}
            if name == "dense":
                live = cache
            del cache
        gate.receipt["prefill"] = prefill_runs
        gate.check("prefill_engaged", prefill_runs["dense"]["calls"].get("sparse_prefill_dense_pieces", 0) > 0
                   and all(prefill_runs[n]["calls"].get("sparse_prefill_gathered_blocks", 0) > 0
                           for n in ("gathered_v1_8k", "gathered_v1_all")),
                   calls={n: r["calls"] for n, r in prefill_runs.items()})
        for name in ("gathered_v1_8k", "gathered_v1_all"):
            gate.check(f"prefill_within_booked[{name}]", not prefill_runs[name]["over_booked"],
                       over_booked=prefill_runs[name]["over_booked"],
                       reference_over_booked=prefill_runs["dense"]["over_booked"])
        dense, g8 = prefill_runs["dense"], prefill_runs["gathered_v1_8k"]
        # Whole-arm absolute peaks, the threshold-crossing chunk included.
        gate.falsifier("R6", (g8["seconds"] > 0.9 * dense["seconds"]) or g8["arm_peak_bytes"] > dense["arm_peak_bytes"],
                       ttft_ratio=g8["seconds"] / dense["seconds"],
                       arm_peak_bytes={n: r["arm_peak_bytes"] for n, r in prefill_runs.items()},
                       rule="TTFT(16K, gathered_v1_8k) <= 0.9 x dense and metal peak not above the dense run")

        decode_peaks, decode_engaged = {}, {}
        token = ids[-1]  # any in-vocabulary token: only the step's memory is measured
        for name, overrides in (("mask_reference", SELECTION), ("gathered_v1", GATHERED)):
            use(gate, adapter, name, overrides)
            cache = clone(live)
            # One unmeasured step absorbs the clone's copy-on-write of the planes.
            mx.eval(model(mx.array([[token]]), cache, logits_to_keep=1), [c.state for c in cache])
            mx.clear_cache()
            base = mx.get_active_memory()
            mx.reset_peak_memory()
            calls = dict(model.calls)
            mx.eval(model(mx.array([[token]]), cache, logits_to_keep=1), [c.state for c in cache])
            decode_peaks[name] = mx.get_peak_memory() - base
            counter = "sparse_attention_mask" if name == "mask_reference" else "sparse_attention_gathered"
            decode_engaged[name] = model.calls[counter] - calls.get(counter, 0)
            del cache
        gate.check("decode_peak_gathered_not_above_mask",
                   all(decode_engaged.values())
                   and decode_peaks["gathered_v1"] <= 1.05 * decode_peaks["mask_reference"],
                   peaks=decode_peaks, engaged_calls=decode_engaged)

        verify = {}
        logits_by = {}
        tokens = [ids[-1], 19, 2518, 6303]
        trees = {"chain4": [-1, 0, 1, 2], "tree8": [-1, 0, 0, 1, 1, 2, 2, 3]}
        for name in ("clone", "tails_v1"):
            use(gate, adapter, name, {**GATHERED, "glm53_tree_rows_attention": "batched_v1",
                                      "glm53_tree_frontier": name})
            provider = Glm53TargetTreeExecutor(model, implementation="layered")
            for shape, parents in trees.items():
                inputs = (tokens * 2)[: len(parents)]
                _, _, txn = provider.forward(model, inputs, parents, live, (len(model.layers) - 1,))
                txn.abort()  # warm
                mx.clear_cache()
                base = mx.get_active_memory()
                mx.reset_peak_memory()
                t0 = time.perf_counter()
                calls = model.calls["tree_rows_attention_batched"]
                out, _, txn = provider.forward(model, inputs, parents, live, (len(model.layers) - 1,))
                mx.eval(out)
                frontiers = sum(isinstance(state, Glm53MLAFrontier) for row in txn.frontiers for state in row)
                verify[f"{name}/{shape}"] = {"peak_above_active": mx.get_peak_memory() - base,
                                             "seconds": time.perf_counter() - t0,
                                             "batched_rows_calls": model.calls["tree_rows_attention_batched"] - calls,
                                             "frontier_states": frontiers}
                logits_by[f"{name}/{shape}"] = np_of(mx, out)
                txn.abort()
        gate.receipt["verify"] = verify
        same = {shape: bitwise(logits_by[f"clone/{shape}"], logits_by[f"tails_v1/{shape}"])[1] for shape in trees}
        engaged = all(v["batched_rows_calls"] > 0 for v in verify.values()) and all(
            (v["frontier_states"] > 0) == key.startswith("tails_v1") for key, v in verify.items())
        gate.check("verify_policies_engaged", engaged, verify=verify)
        gate.check("verify_logits_clone_equals_tails", all(v == 0 for v in same.values()), differing_words=same)
        gate.falsifier("R8_memory", any(verify[f"tails_v1/{s}"]["peak_above_active"]
                                        >= 0.95 * verify[f"clone/{s}"]["peak_above_active"] for s in trees),
                       rule="verify-step peak with tails_v1 not below 0.95 x clone (throughput half: ladder-mtp 16K)",
                       verify=verify)
    except BaseException as exc:  # noqa: BLE001
        error = exc
    finally:
        if adapter is not None:
            adapter.close()
    return gate.finish(mx, error)


# ---------------------------------------------------------------------------

GATES = {
    "select": (gate_select, "P1 select+expand kernel, blocked select, R3, R9b/c"),
    "pools": (gate_pools, "incremental pools bitwise (R5, 9.3), pooling bounds (R9a)"),
    "attention": (gate_attention, "P2 gathered/fused attention vs float64, block invariance, one live block"),
    "rows": (gate_rows, "rows_attention R in {2,4,8,16} bitwise (V1, R7)"),
    "decode": (gate_decode, "gathered vs mask_reference decode, first divergence, near-tie (9.1, 9.10)"),
    "tree": (gate_tree, "tree verify == one-row decode, both tree policies, fused and reference HC"),
    "replay": (gate_replay, "serialized restore at 8191/8193, pool rebuild, exact continuation"),
    "memory": (gate_memory, "peak memory: booked prefill transient, decode, verify (R6, R8)"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="gate", required=True)
    for name, (_, help_text) in GATES.items():
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--out", type=Path, required=True, help="new or empty evidence directory")
        p.add_argument("--source-revision", required=True, help="the mlx2 commit this run executes from (HEAD)")
        p.add_argument("--i-own-the-gpu", action="store_true", help="required: run only under the GPU lease")
        p.add_argument("--model", default=MODEL)
        p.add_argument("--base-policy", type=Path, default=BASE_POLICY,
                       help="weight-level policy the checkpoint loads with (sparse keys are switched in process)")
        p.add_argument("--chunk", type=int, default=512 if name == "tree" else 2048,
                       help="prompt chunk rows (tree: the oracle's priming chunk)")
        p.add_argument("--allow-revision-mismatch", action="store_true")
        p.add_argument("--allow-dirty-src", action="store_true")
        if name in ("select", "pools", "memory"):
            p.add_argument("--tokens", type=int, default=16400)
        if name == "replay":
            p.add_argument("--tokens", type=int, default=8900)
            p.add_argument("--splits", default="8191,8193")
        if name in ("pools", "decode", "replay"):
            p.add_argument("--decode", type=int, default={"pools": 16, "decode": 48, "replay": 32}[name])
        if name == "attention":
            p.add_argument("--capture-tokens", type=int, default=4096, help="0 skips the real-operand capture")
        if name == "decode":
            p.add_argument("--tokens-list", default="4096,16384")
        if name in ("decode", "tree"):
            p.add_argument("--skip-fused", action="store_true", help="leave out the fused_decode_v1 arm")
        if name == "tree":
            p.add_argument("--contexts", default="2046,4100")
    args = parser.parse_args(argv)
    if not args.i_own_the_gpu:
        raise SystemExit("GPU gate: pass --i-own-the-gpu and run under the GPU lease")
    return GATES[args.gate][0](args)


if __name__ == "__main__":
    raise SystemExit(main())
