"""rm06 diagnostic: where does an external round's time go, and are greedy
divergences near-ties?  Metal; run only under the GPU queue.

1. Raw target forward cost at M = 1, 2, 4, 8, 16 after a real prompt prefill
   (forward_with_taps + eval, cache trimmed back each time).
2. Draft cost: draft_distributions at the configured K, alone.
3. External B1 greedy generation under cProfile (top functions by cumtime).
4. Greedy parity: ordinary vs external tokens per prompt; at the first
   divergence, the ordinary (M=1) top-2 logit margin at that position.
"""
import argparse
import copy
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, "/private/tmp/mlx2-rm06-spec-north-laguna/scripts")
sys.path.insert(0, "/private/tmp/mlx2-rm06-spec-north-laguna/src")

import bench_external_draft_acceptance as bench  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--num-draft", type=int, required=True)
    p.add_argument("--prompts", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import mlx.core as mx

    args = argparse.Namespace(family=a.family, target=Path(a.target), draft=Path(a.draft))
    adapter = bench._adapter(args, a.num_draft)
    adapter.external_policy["num_draft"] = a.num_draft
    prompts = bench._prompt_ids(adapter, a.prompts, False)
    model = adapter.model
    out = {"family": a.family, "num_draft": a.num_draft}

    # 1. forward cost vs M
    layers = tuple(adapter.draft_model.target_layer_ids) if hasattr(adapter.draft_model, "target_layer_ids") else None
    try:
        batch = adapter.create_external_batch(completion_batch_size=1, stop_tokens=bench._stops(adapter))
        layers = batch.layers
        batch.close()
    except Exception as exc:  # noqa: BLE001
        out["layers_error"] = repr(exc)
    cache = model.make_cache()
    prompt = mx.array([prompts[0]])
    logits, feats = model.forward_with_taps(prompt, cache, layers)
    mx.eval(logits, feats, [c.state for c in cache])
    base_offsets = [c.offset for c in cache]
    fwd = {}
    for m in (1, 2, 4, 8, 16):
        times = []
        for rep in range(12):
            trial = copy.deepcopy(cache)
            mx.eval([c.state for c in trial])
            tok = mx.array([[prompts[0][-1]] * m])
            t0 = time.perf_counter()
            lg, ft = model.forward_with_taps(tok, trial, layers)
            mx.eval(lg, ft)
            dt = time.perf_counter() - t0
            if rep >= 2:
                times.append(dt)
        times.sort()
        fwd[m] = {"median_ms": 1000 * times[len(times) // 2], "min_ms": 1000 * times[0]}
    out["target_forward_ms_by_M"] = fwd
    # deepcopy cost of a lane-sized cache (recovery capture does this per round)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        c2 = copy.deepcopy(cache)
        mx.eval([c.state for c in c2])
        times.append(time.perf_counter() - t0)
    out["cache_deepcopy_ms"] = 1000 * sorted(times)[5]
    out["prefill_offsets"] = base_offsets[:2]

    # 3. external B1 greedy under cProfile + 4. parity
    ords, ord_tps, _ = bench._run_ordinary(adapter, prompts, 1, 0.0, a.max_tokens)
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    exts, ext_tps, mech = bench._run_external(adapter, prompts, 1, 0.0, a.max_tokens)
    prof.disable()
    out["external_wall_s"] = time.perf_counter() - t0
    out["ordinary_tok_s"] = ord_tps
    out["external_tok_s"] = ext_tps
    out["mechanism"] = {k: v for k, v in mech.items() if k != "draft_stats"}
    out["draft_stats"] = dict(mech.get("draft_stats") or {})
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(45)
    out["cprofile_cumulative"] = s.getvalue()
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(30)
    out["cprofile_tottime"] = s.getvalue()
    out["round_times_ms"] = mech.get("round_times_ms")

    parity = []
    for i in range(len(prompts)):
        o, e = ords.get(i, []), exts.get(i, [])
        k = 0
        while k < min(len(o), len(e)) and o[k] == e[k]:
            k += 1
        row = {"prompt": i, "ordinary_len": len(o), "external_len": len(e), "first_divergence": k}
        if k < min(len(o), len(e)):
            c = model.make_cache()
            seq = prompts[i] + o[:k]
            lg, _ = model.forward_with_taps(mx.array([seq]), c, layers)
            last = lg[0, -1].astype(mx.float32)
            mx.eval(last)
            top = mx.argsort(-last)[:2].tolist()
            row.update({"ordinary_token": o[k], "external_token": e[k], "full_prefill_argmax": top[0],
                        "top2": top, "top2_margin": float(last[top[0]] - last[top[1]]),
                        "margin_ord_vs_ext": float(last[o[k]] - last[e[k]]),
                        "ordinary_text": adapter.tokenizer.decode(o[max(0, k - 8):k + 4]),
                        "external_text": adapter.tokenizer.decode(e[max(0, k - 8):k + 4])})
        parity.append(row)
    out["parity"] = parity
    adapter.close()
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("cprofile")}, indent=1, default=str))


if __name__ == "__main__":
    main()
