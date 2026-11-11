#!/usr/bin/env python3
"""Encode DeepSeek V4.1 Source routed experts to upstream EXL3 tensors and check them.

For each layer: the most-used and the median-used expert (as check_q2_numerics.py
and proto_g32_hadamard.py pick them from trace.npz). Evaluation rows are the
first 64 rows that selected the expert (the same rows those scripts use);
Hessians come from the remaining selected rows, never the evaluation rows. A
median expert has ~100-120 calibration rows, so its Hessian is
0.5 * selected + 0.5 * a 4096-row layer sample (both excluding evaluation rows).
Down's Hessian uses the original expert's fp8-QAT SwiGLU intermediate.

Output per (bits, layer, expert): <out>/w<words>/layers.<L>.ffn.experts.<E>.safetensors
with upstream exllamav3 DeepSeek-V4 names (layers.L.ffn.experts.E.w1|w3|w2 .trellis
I16 [I/16, O/16, words], .suh F16 [I], .svh F16 [O], .mcg I32 marker). The whole-
expert check decodes the SAVED tensors with mlx2's reference EXL3 decoder
(exl3_reference.effective_weight), not the encoder's own reconstruction.

With --backend metal and the v2 kernel, experts stream through
exl3_encode.encode_experts (loading and Hessians on a prep thread, saving and
the reference-decode check on a post thread); encode_s is then the interval
between consecutive finished experts. --no-stream restores one-at-a-time.

usage: convert_experts.py [--layers 5,20,35] [--words 32,48,64] [--backend cpu|metal] [--inflight 2]
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "ds41_expert_q2"))
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)
from convert import decode_mxfp4  # noqa: E402
from mlx2.adapters.deepseek_v41_source_artifact import inspect_artifact, read_tensors  # noqa: E402
from mlx2.adapters.deepseek_v41_source_weights import expert_names, PROJECTIONS  # noqa: E402
from mlx2.runtime.models.deepseek_v4 import limited_swiglu  # noqa: E402
from mlx2.runtime.models.deepseek_v41_quant import quantize as qat  # noqa: E402
from mlx2.runtime.models.exl3_reference import effective_weight  # noqa: E402
from mlx2.runtime.trellis.upstream import CODEBOOKS  # noqa: E402
from safetensors.numpy import load_file, save_file  # noqa: E402

import exl3_encode as E  # noqa: E402
import viterbi_metal as VM  # noqa: E402
from viterbi_metal import MX_LOCK  # noqa: E402

MODEL = str(Path.home() / "mlx-models/DeepSeek-V4.1-Flash-1017-Source")
CAP = Path.home() / "mlx-models/.ds41-routing-capture-20261108"
TRACE = Path.home() / "Desktop/mlx2/output/ds41-source-20261108/cache-design/sched/trace.npz"
EVAL_ROWS, MIN_CALIB, ALL_SAMPLE = 64, 2048, 4096


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def fp8(x):
    return np.array(qat(mx.array(np.asarray(x, np.float32)), "fp8").astype(mx.float32))


def swiglu_mid(xq, wg, wu):
    g, u = mx.array(xq @ wg.T), mx.array(xq @ wu.T)
    return fp8(np.array(limited_swiglu(g, u)))


def gram(x):
    x = np.asarray(x, np.float64)
    return x.T @ x / len(x)


def expert_out(xq, wg, wu, wd):
    """Whole expert, weights as torch [out, in] float32."""
    return swiglu_mid(xq, wg, wu) @ wd.T


def prepare_expert(art, ids, rng, layer, pick):
    """Dequantized weights, Hessians and evaluation inputs for one picked expert."""
    counts = np.bincount(ids[:, layer].reshape(-1), minlength=384)
    picks = {"top": int(np.argmax(counts)), "median": int(np.argsort(counts)[len(counts) // 2])}
    X = np.load(CAP / "x_by_layer" / f"L{layer:02d}.npy", mmap_mode="r")
    expert = picks[pick]
    sel = np.where((ids[:, layer] == expert).any(-1))[0]
    ev, cal = sel[:EVAL_ROWS], sel[EVAL_ROWS:]
    raw = read_tensors(art, expert_names(layer, expert))
    W = {k: decode_mxfp4(raw[f"layers.{layer}.ffn.experts.{expert}.{p}.weight"],
                         raw[f"layers.{layer}.ffn.experts.{expert}.{p}.scale"]) for k, p in PROJECTIONS.items()}
    with MX_LOCK:
        xe = fp8(X[ev].astype(np.float32))
        xc = fp8(X[cal].astype(np.float32))
        mid = swiglu_mid(xc, W["gate"], W["up"])
    h_in, h_mid = gram(xc), gram(mid)
    calib = {"selected_rows": int(len(cal)), "mixed_with_layer_sample": False}
    if len(cal) < MIN_CALIB:
        pool = np.setdiff1d(np.arange(len(X)), ev)
        pool = np.sort(rng.choice(pool, size=min(ALL_SAMPLE, len(pool)), replace=False))
        with MX_LOCK:
            xa = fp8(X[pool].astype(np.float32))
            mida = swiglu_mid(xa, W["gate"], W["up"])
        h_in = 0.5 * h_in + 0.5 * gram(xa)
        h_mid = 0.5 * h_mid + 0.5 * gram(mida)
        calib.update(mixed_with_layer_sample=True, layer_sample_rows=int(len(pool)))
    with MX_LOCK:
        ref = expert_out(xe, W["gate"], W["up"], W["down"])
    return {"layer": layer, "expert": expert, "pick": pick, "uses": int(counts[expert]), "eval_rows": int(len(ev)),
            "calib": calib, "W": W, "h_in": h_in, "h_mid": h_mid, "xe": xe, "ref": ref}


def save_and_check(ctx, words, enc, out, args, dt):
    layer, expert = ctx["layer"], ctx["expert"]
    tensors = {}
    for (kind, part), e in zip(PROJECTIONS.items(), enc):
        base = f"layers.{layer}.ffn.experts.{expert}.{part}"
        tensors[f"{base}.trellis"] = e["trellis"].view(np.int16)
        tensors[f"{base}.suh"], tensors[f"{base}.svh"] = e["suh"], e["svh"]
        tensors[f"{base}.mcg"] = np.array([CODEBOOKS["mcg"]], np.uint32).view(np.int32)
    dst = out / f"w{words}" / f"layers.{layer}.ffn.experts.{expert}.safetensors"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")   # review 2026-11-09: never leave a truncated file under the real name
    save_file(tensors, str(tmp), metadata={"format": "exl3", "encoder": "mlx2 ds41_exl3"})
    os.replace(tmp, dst)
    # Decode the saved file through mlx2's reference decoder.
    f = load_file(str(dst))
    D, exact = {}, True
    for (kind, part), e in zip(PROJECTIONS.items(), enc):
        base = f"layers.{layer}.ffn.experts.{expert}.{part}"
        weff = effective_weight(f[f"{base}.trellis"].view(np.uint16), f[f"{base}.suh"], f[f"{base}.svh"], "mcg")
        exact &= bool(np.allclose(weff, e["weight"], rtol=0, atol=1e-6 * np.abs(e["weight"]).max()))
        D[kind] = weff.T.astype(np.float32)  # back to [out, in]
    with MX_LOCK:
        y = expert_out(ctx["xe"], D["gate"], D["up"], D["down"])
    ref = ctx["ref"]
    row = {"layer": layer, "expert": expert, "pick": ctx["pick"], "uses": ctx["uses"],
           "eval_rows": ctx["eval_rows"], "calibration": ctx["calib"], "words": words,
           "bits_per_weight": words / 16, "backend": args.backend, "probe": args.probe, "ldlq": not args.no_ldlq,
           "encode_s": round(dt, 1), "decode_matches_encoder": exact,
           "expert_rel_err": round(rel(y, ref), 4), "expert_cos": round(cos(y, ref), 4),
           "proj": {k: {"weight_rel_err": round(e["stats"]["weight_rel_err"], 4),
                        "proxy_rel_err": round(e["stats"]["proxy_rel_err"], 4)}
                    for k, e in zip(PROJECTIONS, enc)},
           "file": str(dst.relative_to(out)), "bytes": dst.stat().st_size}
    if "timing" in enc[0]["stats"]:
        row["timing"] = enc[0]["stats"]["timing"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="5,20,35")
    ap.add_argument("--words", default="32,48,64")
    ap.add_argument("--picks", default="top,median")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--probe", type=int, default=32)
    ap.add_argument("--no-ldlq", action="store_true", help="ablation: plain per-tile Viterbi, no Hessian feedback")
    ap.add_argument("--no-stream", action="store_true",
                    help="metal: encode one expert at a time (encode_many) instead of the streaming pipeline")
    ap.add_argument("--inflight", type=int, default=2, help="metal streaming: experts sharing the GPU queue")
    ap.add_argument("--out", default=str(Path.home() / "mlx-models/.ds41-exl3-encoder-20261109"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    art = inspect_artifact(MODEL)
    ids = np.load(TRACE)["ids"]
    rng = np.random.default_rng(0)
    results = []
    tasks = [(layer, pick, words) for layer in map(int, a.layers.split(",")) for pick in a.picks.split(",")
             for words in map(int, a.words.split(","))]

    def emit(row):
        results.append(row); print(json.dumps(row), flush=True)
        tmp = out / "results.json.partial"; tmp.write_text(json.dumps(results, indent=1)); os.replace(tmp, out / "results.json")

    stream = (a.backend == "metal" and not a.no_stream and not a.no_ldlq
              and all(VM.v2_supported(w) for w in map(int, a.words.split(","))))
    if not stream:
        ctx, key = None, None
        for layer, pick, words in tasks:
            if key != (layer, pick):
                ctx, key = prepare_expert(art, ids, rng, layer, pick), (layer, pick)
            t0 = time.time()
            enc = E.encode_many([(ctx["W"]["gate"].T, ctx["h_in"]), (ctx["W"]["up"].T, ctx["h_in"]),
                                 (ctx["W"]["down"].T, ctx["h_mid"])], words, seed=layer * 1000 + ctx["expert"],
                                probe=a.probe, backend=a.backend, ldlq=not a.no_ldlq)
            emit(save_and_check(ctx, words, enc, out, a, time.time() - t0))
        return
    # Streaming: one encode_experts stream over the tasks in the same order as
    # --no-stream (layer, pick, words), each unit carrying its bit rate, so every
    # expert is prepared once (same rng draws as before) on the prep thread;
    # save + reference-decode checks run on a post thread and rows are written
    # as they complete, so the GPU search queue never waits for them.
    from concurrent.futures import ThreadPoolExecutor

    cache = {}
    ctxs = []
    post = ThreadPoolExecutor(1)

    def loader(layer, pick, words):
        def load():
            key = (layer, pick)
            if key not in cache:
                cache.clear()
                cache[key] = prepare_expert(art, ids, rng, layer, pick)
            ctx = cache[key]
            ctxs.append((ctx, words))
            W = ctx["W"]
            return [(W["gate"].T, ctx["h_in"]), (W["up"].T, ctx["h_in"]), (W["down"].T, ctx["h_mid"])], \
                layer * 1000 + ctx["expert"], words
        return load

    futs = []
    t_prev = time.time()
    for n, enc in enumerate(E.encode_experts((loader(*t) for t in tasks), tasks[0][2],
                                             inflight=a.inflight, probe=a.probe)):
        now = time.time()
        ctx, words = ctxs[n]
        ctxs[n] = None  # release this expert's inputs once its check has them
        futs.append(post.submit(save_and_check, ctx, words, enc, out, a, now - t_prev))
        t_prev = now
        while futs and futs[0].done():
            emit(futs.pop(0).result())
    for f in futs:
        emit(f.result())
    post.shutdown()


if __name__ == "__main__":
    main()
