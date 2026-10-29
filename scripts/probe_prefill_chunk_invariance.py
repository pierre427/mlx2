#!/usr/bin/env python3
"""Does load-dependent prefill slicing change the state APCv2 would publish?

Loads one artifact through its registry adapter (the adapter installs its own
load-time prefill mechanisms, as serving does), installs the serving lane
policy in ``auto`` mode exactly as ``serving.py`` does, then for each arm
(lane enabled / lane disabled = stock kernels) prefills the same real-text
prompt under several slice schedules on a fresh cache.  The prompt's final
token is held back and run as a one-row step, mirroring the scheduler's
reserved final-token segment, so the compared cache is the prompt-only state
a cold request would publish.

Per schedule versus the arm's reference schedule (the route's normal chunk):
every cache layer's state arrays (attention K/V, GDN conv + recurrent),
bitwise equality, element-differing counts, max abs diff, the first differing
layer, and the next-token logits after the final one-row step.

  PYTHONPATH=src .venv/bin/python scripts/probe_prefill_chunk_invariance.py \
      --i-own-the-gpu --model PATH --rows 2048 --normal 512 \
      --schedules full 512 64 128 256 mixed512 --out out.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "mlx2.prefill-chunk-invariance.v1"

MIXED = {
    # Adaptive-style slices (64/128/256/512 subset), each summing to 2048.
    "mixed512": [256, 64, 128, 512, 64, 256, 128, 64, 512, 64],
    "mixed256": [256, 64, 128, 256, 64, 256, 128, 64, 256, 64, 256, 128, 64, 64],
}


def corpus_text() -> str:
    files = sorted((ROOT / "docs").rglob("*.md"))
    return "\n\n".join(f.read_text(errors="ignore") for f in files)


def swapouts() -> int:
    text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    line = next(l for l in text.splitlines() if l.startswith("Swapouts"))
    return int(line.split(":")[1].strip().rstrip("."))


def schedule_rows(spec: str, rows: int) -> list[int]:
    if spec == "full":
        return [rows]
    if spec.endswith("r"):
        # A bit-for-bit repeat of the base schedule: the determinism control.
        return schedule_rows(spec[:-1], rows)
    if spec in MIXED:
        out = list(MIXED[spec])
        if sum(out) != rows:
            raise SystemExit(f"{spec} sums to {sum(out)}, not {rows}")
        return out
    step = int(spec)
    out, pos = [], 0
    while pos < rows:
        out.append(min(step, rows - pos))
        pos += out[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", type=int, default=2048, help="prefilled rows (prompt = rows + 1)")
    ap.add_argument("--normal", required=True, help="the route's normal chunk schedule (reference)")
    ap.add_argument("--schedules", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["lane_auto", "stock"],
                    choices=["lane_auto", "stock"])
    ap.add_argument("--decode", type=int, default=32, help="greedy tokens after the prompt")
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--cache-limit-gib", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx

    from mlx2.adapters.registry import inspect_model, resolve_adapter

    swap0 = swapouts()
    resolution = inspect_model(a.model)
    checkpoint = f"{Path(a.model).name}@sha256:{resolution.artifact_fingerprint}"
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src"],
                           capture_output=True, text=True).stdout.strip()
    t0 = time.perf_counter()
    adapter = resolve_adapter(a.model, qualification_mode=True)(a.model)
    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(a.cache_limit_gib << 30)
    load_s = time.perf_counter() - t0

    # Serving's automatic lane law (serving.py, lane_matmul == "auto").
    from mlx2.runtime import lane
    from mlx2.runtime.lane.policy import detect, resolve

    lane_receipt = None
    if "lane_auto" in a.arms and lane.available():
        try:
            config = json.loads((Path(a.model) / "config.json").read_text())
        except (OSError, ValueError):
            config = None
        policy = resolve(
            detect(model, config),
            family=resolution.descriptor.family,
            adapter=(adapter.lane_policy_defaults()
                     if callable(getattr(adapter, "lane_policy_defaults", None)) else None),
            mode="auto",
        )
        offered = getattr(adapter, "lane_projection_groups", None)
        lane_receipt = lane.apply_policy(
            model, policy, declared=offered() if callable(offered) else ()) or {
                "law_id": "stock", "covered": {}}
    covered = bool((lane_receipt or {}).get("covered"))

    text = corpus_text()
    tok = adapter.tokenizer
    try:
        ids = tok.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tok.encode(text)
    ids = [int(i) for i in getattr(ids, "ids", ids)]
    prompt = ids[: a.rows + 1]
    assert len(prompt) == a.rows + 1

    def run(rows_plan):
        cache = model.make_cache() if hasattr(model, "make_cache") else None
        if cache is None:
            from mlx2.runtime.models.cache import make_prompt_cache
            cache = make_prompt_cache(model)
        pos = 0
        mx.synchronize()
        started = time.perf_counter()
        for n in rows_plan:
            out = model(mx.array([prompt[pos:pos + n]], mx.uint32), cache=cache)
            out = getattr(out, "logits", out)
            mx.eval(out, [c.state for c in cache])
            pos += n
        prefill_s = time.perf_counter() - started
        # Prompt-only state (what a cold prefill publishes), materialised.
        from mlx.utils import tree_flatten

        state = []
        for i, c in enumerate(cache):
            if isinstance(getattr(c, "cache", None), list):
                # ArraysCache: the live [conv, recurrent] arrays only (not the
                # left_padding/lengths bookkeeping).
                arrays = [x for x in c.cache if x is not None]
            else:
                arrays = [x for _, x in tree_flatten(c.state) if hasattr(x, "shape")]
                offset = getattr(c, "offset", None)
                if isinstance(offset, int):
                    # KVCache.state is the whole step-grown buffer; compare
                    # the rows actually written.
                    arrays = [x[..., :offset, :] if x.ndim == 4 and x.shape[2] > offset else x
                              for x in arrays]
            # Copy so the final step cannot alias/mutate the snapshot.
            arrays = [mx.array(x) for x in arrays]
            state.append({"layer": i, "kind": type(c).__name__, "arrays": arrays,
                          "offset": getattr(c, "offset", None)})
        mx.eval([s["arrays"] for s in state])
        y = model(mx.array([[prompt[-1]]], mx.uint32), cache=cache)
        y = getattr(y, "logits", y)
        logits = y[0, -1].astype(mx.float32)
        mx.eval(logits)
        tokens, nxt = [], int(mx.argmax(logits).item())
        for _ in range(a.decode):
            tokens.append(nxt)
            y = model(mx.array([[nxt]], mx.uint32), cache=cache)
            y = getattr(y, "logits", y)
            nxt = int(mx.argmax(y[0, -1]).item())
        del cache
        return {"state": state, "logits": logits, "prefill_s": prefill_s, "tokens": tokens}

    def compare(r, b):
        layers, first = [], None
        for sr, sb in zip(r["state"], b["state"]):
            entry = {"layer": sr["layer"], "kind": sr["kind"], "arrays": []}
            for j, (x, y) in enumerate(zip(sr["arrays"], sb["arrays"])):
                if x.shape != y.shape:
                    entry["arrays"].append({"index": j, "shape_mismatch": [list(x.shape), list(y.shape)]})
                    continue
                eq = bool(mx.array_equal(x, y).item())
                d = {"index": j, "shape": list(x.shape), "dtype": str(x.dtype), "identical": eq}
                if not eq:
                    xf, yf = x.astype(mx.float32), y.astype(mx.float32)
                    d["elements_differing"] = int(mx.sum(x != y).item())
                    d["elements"] = int(x.size)
                    d["max_abs_diff"] = float(mx.max(mx.abs(xf - yf)).item())
                    d["max_abs_ref"] = float(mx.max(mx.abs(yf)).item())
                entry["arrays"].append(d)
            entry["identical"] = all(d.get("identical") for d in entry["arrays"])
            if not entry["identical"] and first is None:
                first = {"layer": sr["layer"], "kind": sr["kind"]}
            layers.append(entry)
        by_kind = {}
        for e in layers:
            k = by_kind.setdefault(e["kind"], {"layers": 0, "differing_layers": 0,
                                                "max_abs_diff_by_array": {}})
            k["layers"] += 1
            k["differing_layers"] += 0 if e["identical"] else 1
            for d in e["arrays"]:
                key = str(d["index"])
                k["max_abs_diff_by_array"][key] = max(
                    k["max_abs_diff_by_array"].get(key, 0.0), d.get("max_abs_diff", 0.0))
        lr, lb = r["logits"], b["logits"]
        return {
            "state_identical": all(e["identical"] for e in layers),
            "first_differing_layer": first,
            "by_kind": by_kind,
            "logits_identical": bool(mx.array_equal(lr, lb).item()),
            "logits_max_abs_diff": float(mx.max(mx.abs(lr - lb)).item()),
            "argmax_same": int(mx.argmax(lr).item()) == int(mx.argmax(lb).item()),
            "greedy_tokens_identical": r["tokens"] == b["tokens"],
            "greedy_first_divergence": next(
                (i for i, (x, y) in enumerate(zip(r["tokens"], b["tokens"])) if x != y), None),
            "layers": layers,
        }

    results = {
        "schema": SCHEMA, "checkpoint": checkpoint, "mlx2_commit": commit,
        "src_dirty": bool(dirty), "model_path": a.model, "mlx": mx.__version__,
        "family": resolution.descriptor.family, "adapter": type(adapter).__name__,
        "prefill_execution_identity": getattr(adapter, "prefill_execution_identity", None),
        "lane_receipt": {k: v for k, v in (lane_receipt or {}).items() if k in ("law_id", "policy")}
        | {"covered_count": len((lane_receipt or {}).get("covered") or {})},
        "prompt_tokens": len(prompt), "prefilled_rows": a.rows,
        "prompt_source": "docs/**/*.md concatenated in sorted order, first rows+1 tokens",
        "final_token": "held back, run as one-row step after the compared snapshot",
        "normal": a.normal, "load_s": round(load_s, 1), "arms": {},
        "swapouts_start": swap0,
    }
    out_path = Path(a.out)
    for arm in a.arms:
        if arm == "lane_auto" and not covered:
            results["arms"][arm] = {"skipped": "lane law covered no projection on this model"}
            continue
        if lane_receipt and covered:
            lane.set_enabled(arm == "lane_auto")
        order = [a.normal] + [s for s in a.schedules if s != a.normal]
        runs, arm_out = {}, {}
        for spec in order:
            plan = schedule_rows(spec, a.rows)
            r = run(plan)
            runs[spec] = r
            entry = {"slices": plan, "prefill_s": round(r["prefill_s"], 3),
                     "logits_argmax": int(mx.argmax(r["logits"]).item()),
                     "greedy_tokens": r["tokens"],
                     "logits_sha256": hashlib.sha256(bytes(memoryview(r["logits"]))).hexdigest()}
            if spec != a.normal:
                entry["vs_normal"] = compare(r, runs[a.normal])
                if "full" in runs and spec != "full":
                    c = compare(r, runs["full"])
                    c.pop("layers")
                    entry["vs_full"] = c
            entry["swapouts_delta"] = swapouts() - swap0
            arm_out[spec] = entry
            summary = {k: v for k, v in entry.get("vs_normal", {}).items() if k != "layers"}
            print(arm, spec, json.dumps({"prefill_s": entry["prefill_s"], **summary}), flush=True)
            if entry["swapouts_delta"] > a.max_swapout_pages:
                results["arms"][arm] = arm_out
                results["aborted"] = f"swapouts grew by {entry['swapouts_delta']} pages"
                out_path.write_text(json.dumps(results, indent=1))
                raise SystemExit(results["aborted"])
            # Keep only the reference (and full) resident.
            if spec not in (a.normal, "full"):
                runs.pop(spec)
            mx.clear_cache()
        results["arms"][arm] = arm_out
        runs.clear()
        mx.clear_cache()
        out_path.write_text(json.dumps(results, indent=1))
    results["swapouts_end"] = swapouts()
    out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
