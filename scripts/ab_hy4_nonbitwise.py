#!/usr/bin/env python3
"""A/B of the Hy4-flash opt-in audit repairs (H1/H6, H3, H4, H5) at long context.

One arm per process (a Hy4 checkpoint is loaded once per arm), so each run
writes timings, logits and indexer selections, and ``compare`` checks them
against the first (reference) arm:

    run      load ``HYV4FlashAdapter`` with the arm's policy, then for each
             --contexts length:
             1. prefill the prompt in --chunk rows (the served prefill step):
                prefill tok/s, the last row's logits, and the sorted recalled
                ids of the last --select-rows queries of every indexed layer
                in the final chunk;
             2. B=1 greedy decode of --decode tokens: tok/s, tokens and every
                step's logits;
             3. MLX active / peak memory after load and per context.
             Receipts: the arm's mechanisms (selected / observed_used),
             checkpoint, mlx2 commit, MLX version, swap delta.
    compare  arm JSONs (the first is the reference): prefill and decode
             speedups, max |logit gap| at the prefill row and over the decode
             steps before the first token divergence, top-1 agreement, the
             first divergent step, and the indexer top-k set agreement (mean
             Jaccard of the recalled id sets).  A non-reference arm whose
             selected mechanism never engaged is reported as refused.

GPU (one model per process, the GPU owned by this run; not thermally gated:
quote the numbers only after a controlled performance run):

    PYTHONPATH=src .venv/bin/python scripts/ab_hy4_nonbitwise.py run --i-own-the-gpu \\
        --model <hy4 artifact> --arm reference --out ab/reference.json
    PYTHONPATH=src .venv/bin/python scripts/ab_hy4_nonbitwise.py run --i-own-the-gpu \\
        --model <hy4 artifact> --arm indexer-simd --out ab/indexer-simd.json
    PYTHONPATH=src .venv/bin/python scripts/ab_hy4_nonbitwise.py compare ab/reference.json ab/indexer-simd.json

fp8 arms compare against ``fp8-legacy`` (the pre-H3 route), not against
``reference``: ``fp8-legacy`` and ``fp8`` must be bitwise.  ``--cpu`` runs on
the CPU device (smoke test on tiny fixtures).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# name -> (execution policy, research switches).  Every arm runs the served
# defaults plus its keys; "vision": false keeps the tower out of memory.
ARMS = {
    "reference": ({}, {}),
    "indexer-v1": ({"indexer_fused_scores": True}, {}),                       # H1
    "indexer-simd": ({"indexer_fused_scores": "simd"}, {}),                   # H6
    "union64": ({"sparse_prefill_union_rows": 64}, {}),                       # H4
    "union16": ({"sparse_prefill_union_rows": 16}, {}),                       # H4
    "resident": ({"resident_bf16": True}, {}),                                # H5, every part
    "resident-router": ({"resident_bf16": ["router"]}, {}),                   # H5, router only
    "resident-router-keys-simd": ({"resident_bf16": ["router", "indexer_keys"],
                                   "indexer_fused_scores": "simd"}, {}),       # H5 + H6
    "fp8-legacy": ({"latent_kv": "fp8"}, {"fp8_bf16_direct": False, "fp8_in_kernel": False}),  # pre-H3
    "fp8": ({"latent_kv": "fp8"}, {}),                                        # H3
    "fp8-union64": ({"latent_kv": "fp8", "sparse_prefill_union_rows": 64}, {}),  # H3 + H4
}


def _swapouts():
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
        return int(next(l for l in out.splitlines() if l.startswith("Swapouts")).split(":")[1].strip(" ."))
    except Exception:  # noqa: BLE001
        return None


def _commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _prompt(adapter, n):
    tok = adapter.tokenizer
    text = "\n\n".join((ROOT / p).read_text() for p in ("docs/SERVING.md", "docs/QUALIFICATION.md",
                                                       "docs/HY4_FLASH.md", "docs/ARCHITECTURE.md"))
    ids = list(tok.encode(text, add_special_tokens=False))
    vocab = adapter.model.args.vocab_size
    ids = [i for i in ids if 3 < i < vocab]
    if not ids:  # a tokenizer that maps the text to specials only (tiny fixtures)
        ids = [4 + (7 * i) % (vocab - 4) for i in range(n)]
    out = []
    k = 0
    while len(out) < n:  # shifted repeats, so no two copies align exactly
        out += ids[k % len(ids):] + ids[: k % len(ids)]
        k += 977
    return out[:n]


def _switches(switches):
    from mlx2.runtime.models import hy_v4_flash_fp8kv as F8
    from mlx2.runtime.models import hy_v4_flash_sparse_prefill as S

    F8.BF16_DIRECT = switches.get("fp8_bf16_direct", True)
    S.FP8_IN_KERNEL = switches.get("fp8_in_kernel", True)


def _receipts(adapter):
    d = adapter.diagnostics()
    fus = d.get("fusions") or {}
    return {"layout": adapter.layout, "checkpoint": d.get("checkpoint"),
            "sparse_prefill_attention": d.get("sparse_prefill_attention"),
            "indexer_fused_scores": fus.get("indexer_fused_scores"),
            "resident_bf16": d.get("resident_bf16"), "latent_kv": d.get("latent_kv")}


def _selected_unused(receipts):
    out = []
    sp = receipts.get("sparse_prefill_attention") or {}
    union = sp.get("union") or {}
    if union.get("selected") and not union.get("observed_used"):
        out.append("sparse_prefill_union_rows selected but not observed_used")
    ix = receipts.get("indexer_fused_scores") or {}
    if ix.get("selected") and not ix.get("observed_used"):
        out.append("indexer_fused_scores selected but not observed_used")
    if ix.get("variant") == "simd" and ix.get("selected") and not ix.get("simd_calls"):
        out.append(f"indexer simd selected but never ran (declines {ix.get('simd_declines')})")
    res = receipts.get("resident_bf16") or {}
    if res.get("selected") and not res.get("observed_used"):
        out.append("resident_bf16 selected but not observed_used")
    lk = receipts.get("latent_kv") or {}
    if lk.get("format") == "fp8" and sp.get("selected") and sp.get("sparse_layer_forwards") \
            and not sp.get("fp8_in_kernel_layer_forwards") and not union.get("observed_used"):
        out.append(f"fp8 in-kernel dequant never ran (declines {sp.get('fp8_in_kernel_declined_layer_forwards')})")
    return out


def run(a):
    import mlx.core as mx
    import numpy as np

    if a.cpu:
        mx.set_default_device(mx.cpu)
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter
    from mlx2.runtime.models import hy_v4_flash as H

    policy, switches = ARMS[a.arm]
    _switches(switches)
    swap0 = _swapouts()
    mx.reset_peak_memory()
    adapter = HYV4FlashAdapter(a.model, execution_policy={"vision": False, **policy})
    arrays = {}
    try:
        model = adapter.model
        mx.eval(model.parameters())
        # Baseline after the load: an ~85 GB load can push other apps to swap in
        # its first seconds, which is not the arm's decode/prefill behaviour.
        swap_loaded = _swapouts()
        out = {"arm": a.arm, "policy": policy, "switches": switches, "mlx": mx.__version__,
               "mlx2_commit": _commit(), "device": str(mx.default_device()),
               "checkpoint": adapter.diagnostics()["checkpoint"], "chunk": a.chunk,
               "memory_after_load": {"active_bytes": mx.get_active_memory(), "peak_bytes": mx.get_peak_memory()},
               "contexts": {}}
        seen = []
        original = H.CompressedIndexer.select

        def spy(self, hidden, q_resid, start, group_keys, where=None):
            picks = original(self, hidden, q_resid, start, group_keys, where)
            if spy.armed:
                seen.append(np.sort(np.array(picks[0, -a.select_rows:]), axis=-1))
            return picks

        spy.armed = False
        H.CompressedIndexer.select = spy
        for n in a.contexts:
            prompt = _prompt(adapter, n)
            mx.reset_peak_memory()
            cache = model.make_cache()
            seen.clear()
            mx.synchronize()
            t0 = time.perf_counter()
            for s in range(0, n, a.chunk):
                spy.armed = s + a.chunk >= n
                h = model.hidden(mx.array([prompt[s:s + a.chunk]]), cache)
                mx.eval(h)
            spy.armed = False
            logits = model.logits(h[:, -1:])[:, -1]
            mx.eval(logits)
            mx.synchronize()
            prefill_s = time.perf_counter() - t0
            rows, tokens = [logits], []
            mx.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.decode):
                t = int(mx.argmax(rows[-1], axis=-1).item())
                tokens.append(t)
                rows.append(model(mx.array([[t]]), cache)[:, -1])
            mx.eval(rows[-1])
            mx.synchronize()
            decode_s = time.perf_counter() - t0
            stacked = np.array(mx.concatenate(rows).astype(mx.float32))
            arrays[f"logits_{n}"] = stacked
            for i, sel in enumerate(seen):
                arrays[f"select_{n}_{i}"] = sel.astype(np.int32)
            out["contexts"][str(n)] = {
                "prefill_s": prefill_s, "prefill_tok_s": n / prefill_s,
                "decode_tok_s": a.decode / decode_s if a.decode else None, "tokens": tokens,
                "logits_sha256": hashlib.sha256(stacked.tobytes()).hexdigest(),
                "indexed_layer_selections": len(seen), "peak_bytes": mx.get_peak_memory(),
                "cache_bytes": sum(c.nbytes for c in cache)}
            del cache, rows
            mx.clear_cache()
        H.CompressedIndexer.select = original
        out["receipts"] = _receipts(adapter)
        swap1 = _swapouts()
        out["swapouts_delta"] = None if swap0 is None or swap1 is None else swap1 - swap0
        out["swapouts_load"] = None if swap0 is None or swap_loaded is None else swap_loaded - swap0
        out["swapouts_after_load"] = None if swap_loaded is None or swap1 is None else swap1 - swap_loaded
    finally:
        adapter.close()
    np.savez_compressed(Path(a.out).with_suffix(".npz"), **arrays)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("receipts",)}, indent=1, default=str))
    return out


def compare(paths):
    import numpy as np

    arms = [json.loads(Path(p).read_text()) for p in paths]
    npz = [np.load(Path(p).with_suffix(".npz")) for p in paths]
    ref, refz = arms[0], npz[0]
    report = {"reference": ref["arm"], "checkpoint": ref["checkpoint"], "mlx2_commit": ref["mlx2_commit"],
              "arms": {}}
    for arm, z in zip(arms[1:], npz[1:]):
        if arm["checkpoint"] != ref["checkpoint"]:
            raise SystemExit("arms ran different checkpoints")
        row = {"mlx2_commit": arm["mlx2_commit"], "contexts": {},
               "memory_after_load_delta_bytes": (arm["memory_after_load"]["active_bytes"]
                                                 - ref["memory_after_load"]["active_bytes"]),
               "refused": _selected_unused(arm.get("receipts") or {})}
        for n, r in ref["contexts"].items():
            c = arm["contexts"].get(n)
            if c is None:
                continue
            la, lb = refz[f"logits_{n}"], z[f"logits_{n}"]
            div = next((i for i, (x, y) in enumerate(zip(r["tokens"], c["tokens"])) if x != y), None)
            upto = 1 + (len(r["tokens"]) if div is None else div)  # rows fed identical tokens
            scale = float(np.abs(la[:upto]).max()) or 1.0
            sel = []
            i = 0
            while f"select_{n}_{i}" in refz.files and f"select_{n}_{i}" in z.files:
                for p, q in zip(refz[f"select_{n}_{i}"], z[f"select_{n}_{i}"]):
                    p, q = set(p[p >= 0].tolist()), set(q[q >= 0].tolist())
                    sel.append(len(p & q) / max(1, len(p | q)))
                i += 1
            row["contexts"][n] = {
                "prefill_speedup": r["prefill_s"] / c["prefill_s"],
                "decode_speedup": (c["decode_tok_s"] / r["decode_tok_s"]) if r.get("decode_tok_s") else None,
                "prefill_tok_s": c["prefill_tok_s"], "decode_tok_s": c["decode_tok_s"],
                "bitwise": c["logits_sha256"] == r["logits_sha256"],
                "prefill_row_max_abs_logit_gap": float(np.abs(la[0] - lb[0]).max()),
                "max_abs_logit_gap_before_divergence": float(np.abs(la[:upto] - lb[:upto]).max()),
                "relative_gap": float(np.abs(la[:upto] - lb[:upto]).max()) / scale,
                "top1_agreement": float(np.mean(la[:upto].argmax(-1) == lb[:upto].argmax(-1))),
                "first_divergent_decode_step": div,
                "indexer_topk_jaccard_mean": float(np.mean(sel)) if sel else None,
                "indexer_topk_exact_rows": float(np.mean([s == 1.0 for s in sel])) if sel else None,
                "peak_bytes_delta": c["peak_bytes"] - r["peak_bytes"],
                "cache_bytes_delta": c["cache_bytes"] - r["cache_bytes"],
            }
        report["arms"][arm["arm"]] = row
    print(json.dumps(report, indent=1))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--contexts", nargs="+", type=int, default=[16384, 65536, 131072])
    r.add_argument("--chunk", type=int, default=2048, help="prefill rows per forward (the served step cap)")
    r.add_argument("--decode", type=int, default=32)
    r.add_argument("--select-rows", type=int, default=8,
                   help="last query rows per indexed layer whose recalled sets are compared")
    r.add_argument("--i-own-the-gpu", action="store_true")
    r.add_argument("--cpu", action="store_true", help="run on the CPU device (smoke test)")
    c = sub.add_parser("compare")
    c.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        return compare(a.paths)
    if not (a.cpu or a.i_own_the_gpu):
        ap.error("refusing Metal execution without --i-own-the-gpu (or pass --cpu)")
    if Path(a.out).exists() and Path(a.out).stat().st_size:
        ap.error(f"{a.out} exists; refusing to overwrite evidence")
    return run(a)


if __name__ == "__main__":
    main()
