#!/usr/bin/env python3
"""Convert a targeted DS4.1 expert set to EXL3 (Pierre 2026-11-09: "Hot set, 3 bpw").

Selection: the top-N experts per layer by routing count in the train-split usage map
(usage-train-split.json, the same map the reduced tiers use). Calibration and the
per-expert check are those of convert_experts.py (Hessians from rows that selected the
expert, mixed with a 4096-row layer sample below 2048 rows; 64 held-out evaluation rows;
decode of the SAVED file through mlx2's reference EXL3 decoder). Resumable: an expert whose
file and result row already exist is skipped. Writes manifest.json with per-file sha256.

usage: convert_bank.py [--top 50] [--words 48] [--layers 0-39] [--out DIR] [--inflight 2]
"""
import argparse, hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_experts as C  # noqa: E402  (sets paths, CPU default device)
import exl3_encode as E  # noqa: E402
from viterbi_metal import MX_LOCK, v2_supported  # noqa: E402

USAGE = C.CAP / "usage-train-split.json"


def prepare_by_id(art, ids, rng, layer, expert, rank, uses):
    X = np.load(C.CAP / "x_by_layer" / f"L{layer:02d}.npy", mmap_mode="r")
    sel = np.where((ids[:, layer] == expert).any(-1))[0]
    ev, cal = sel[:C.EVAL_ROWS], sel[C.EVAL_ROWS:]
    raw = C.read_tensors(art, C.expert_names(layer, expert))
    W = {k: C.decode_mxfp4(raw[f"layers.{layer}.ffn.experts.{expert}.{p}.weight"],
                           raw[f"layers.{layer}.ffn.experts.{expert}.{p}.scale"]) for k, p in C.PROJECTIONS.items()}
    with MX_LOCK:
        xe, xc = C.fp8(X[ev].astype(np.float32)), C.fp8(X[cal].astype(np.float32))
        mid = C.swiglu_mid(xc, W["gate"], W["up"])
    h_in, h_mid = C.gram(xc), C.gram(mid)
    calib = {"selected_rows": int(len(cal)), "mixed_with_layer_sample": False}
    if len(cal) < C.MIN_CALIB:
        pool = np.setdiff1d(np.arange(len(X)), ev)
        pool = np.sort(rng.choice(pool, size=min(C.ALL_SAMPLE, len(pool)), replace=False))
        with MX_LOCK:
            xa = C.fp8(X[pool].astype(np.float32))
            mida = C.swiglu_mid(xa, W["gate"], W["up"])
        h_in, h_mid = 0.5 * h_in + 0.5 * C.gram(xa), 0.5 * h_mid + 0.5 * C.gram(mida)
        calib.update(mixed_with_layer_sample=True, layer_sample_rows=int(len(pool)))
    with MX_LOCK:
        ref = C.expert_out(xe, W["gate"], W["up"], W["down"])
    return {"layer": layer, "expert": expert, "pick": f"rank{rank}", "uses": uses, "eval_rows": int(len(ev)),
            "calib": calib, "W": W, "h_in": h_in, "h_mid": h_mid, "xe": xe, "ref": ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--words", type=int, default=48)
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--probe", type=int, default=32)
    ap.add_argument("--out", default="~/mlx-models/DeepSeek-V4.1-Flash-1017-Source-Expert-EXL3-3bpw-hot50")
    a = ap.parse_args()
    if not v2_supported(a.words):
        raise SystemExit("streaming v2 kernel does not support this rate")
    lo, hi = map(int, a.layers.split("-"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(json.loads(USAGE.read_text())["counts"])
    plan = [(l, int(e), r, int(counts[l, e])) for l in range(lo, hi + 1)
            for r, e in enumerate(np.argsort(-counts[l], kind="stable")[:a.top])]
    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {(r["layer"], r["expert"]) for r in results if (out / r["file"]).exists()}
    todo = [p for p in plan if (p[0], p[1]) not in done]
    print(f"{len(plan)} planned, {len(done)} done, {len(todo)} to encode", flush=True)
    art = C.inspect_artifact(C.MODEL)
    ids = np.load(C.TRACE)["ids"]
    rng = np.random.default_rng(0)
    args = argparse.Namespace(backend="metal", probe=a.probe, no_ldlq=False)
    ctxs, post, futs = [], ThreadPoolExecutor(1), []

    def emit(row):
        results.append(row); print(json.dumps({k: row[k] for k in ("layer", "expert", "pick", "expert_rel_err", "encode_s")}), flush=True)
        tmp = res_path.with_suffix(".tmp"); tmp.write_text(json.dumps(results, indent=1)); tmp.replace(res_path)

    def loader(layer, expert, rank, uses):
        def load():
            ctx = prepare_by_id(art, ids, rng, layer, expert, rank, uses)
            ctxs.append(ctx); W = ctx["W"]
            return [(W["gate"].T, ctx["h_in"]), (W["up"].T, ctx["h_in"]), (W["down"].T, ctx["h_mid"])], \
                layer * 1000 + expert, a.words
        return load

    t_prev = time.time()
    for n, enc in enumerate(E.encode_experts((loader(*p) for p in todo), a.words, inflight=a.inflight, probe=a.probe)):
        now = time.time(); ctx = ctxs[n]; ctxs[n] = None
        futs.append(post.submit(C.save_and_check, ctx, a.words, enc, out, args, now - t_prev)); t_prev = now
        while futs and futs[0].done():
            emit(futs.pop(0).result())
    for f in futs:
        emit(f.result())
    post.shutdown()
    files = sorted((out / f"w{a.words}").glob("*.safetensors"))
    manifest = {"source": "DeepSeek-V4.1-Flash-1017-Source@c30fd25e3cc79ba7044686fa3f9a7a1c0f514f77ee6293d2f5f62bc409e02ddb",
                "format": "exl3 (upstream exllamav3 tensor layout, mcg codebook)", "bits_per_weight": a.words / 16,
                "selection": f"top {a.top} per layer by usage-train-split counts, layers {lo}-{hi}",
                "files": {str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True).encode()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("manifest", manifest["fingerprint"], len(files), "files", flush=True)


if __name__ == "__main__":
    main()
