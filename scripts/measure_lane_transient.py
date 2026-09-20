"""Measure the per-lane verify-forward transient on a real model.

Definition being measured (matches SelfMTPLaneAdmissionController):
  transient_gib_per_lane = (peak active GPU memory during ONE lane's
  M=(k+1) verify forward, above the steady resident state that survives the
  forward) -- i.e. excluding model weights and excluding the KV/recurrent
  cache, including its own growth. Reported per lane: total / N.
"""
import argparse, gc, json, time

import mlx.core as mx


def gib(x):
    return x / (1 << 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--contexts", default="1024,4096,16384")
    ap.add_argument("--lanes", default="1,2,4")
    ap.add_argument("--depths", default="0,1,2")
    ap.add_argument("--out", default="/tmp/transient.json")
    ap.add_argument("--no-mtp", action="store_true")
    args = ap.parse_args()

    from mlx2.adapters.registry import resolve_adapter

    use_mtp = not args.no_mtp
    cls = resolve_adapter(args.model, mtp=use_mtp)
    t0 = time.time()
    adapter = cls(args.model, require_mtp=True) if use_mtp else cls(args.model)
    model = adapter.model
    mx.eval(model.parameters())
    mx.clear_cache()
    weights_gib = gib(mx.get_active_memory())
    print(f"loaded in {time.time()-t0:.1f}s  resident weights {weights_gib:.3f} GiB", flush=True)

    targs = getattr(adapter.model, "args", None)
    tc = getattr(targs, "text_config", None)
    vocab = int(tc["vocab_size"]) if isinstance(tc, dict) else int(getattr(targs, "vocab_size"))
    rows = []
    contexts = [int(c) for c in args.contexts.split(",")]
    lanes_list = [int(n) for n in args.lanes.split(",")]
    depths = [int(d) for d in args.depths.split(",")]

    for ctx in contexts:
        for n in lanes_list:
            # --- build an N-lane cache at context ctx by real prefill ---
            mx.clear_cache()
            base = mx.get_active_memory()
            try:
                cache = model.make_cache()
                toks = mx.random.randint(0, vocab, (n, ctx), dtype=mx.uint32)
                step = 512
                for s in range(0, ctx, step):
                    out = model(toks[:, s:s + step], cache=cache)
                    mx.eval(out)
                    del out
                del toks
                mx.clear_cache()
                cache_gib = gib(mx.get_active_memory() - base)
            except Exception as e:  # OOM or unsupported
                print(f"SKIP ctx={ctx} n={n}: {type(e).__name__}: {e}", flush=True)
                try:
                    del cache
                except Exception:
                    pass
                mx.clear_cache()
                continue
            print(f"ctx={ctx} n={n} cache={cache_gib:.3f} GiB", flush=True)

            for k in depths:
              width = k + 1
              best = None
              for rep in range(3):
                try:
                    mx.clear_cache()
                    gc.collect()
                    mx.reset_peak_memory()
                    a_before = mx.get_active_memory()
                    c_before = mx.get_cache_memory()
                    vt = mx.random.randint(0, vocab, (n, width), dtype=mx.uint32)
                    mx.eval(vt)
                    a_in = mx.get_active_memory()
                    logits = model(vt, cache=cache)
                    mx.eval(logits)
                    peak = mx.get_peak_memory()
                    a_after = mx.get_active_memory()
                    c_after = mx.get_cache_memory()
                    del logits, vt
                    mx.clear_cache()
                    a_final = mx.get_active_memory()
                    # roll the cache back: drop the width tokens we just added
                    for c in cache:
                        if hasattr(c, "offset"):
                            try:
                                c.offset -= width
                            except Exception:
                                pass
                    row = dict(
                        context=ctx, lanes=n, k=k, width=width, rep=rep,
                        cache_gib=cache_gib,
                        a_before_gib=gib(a_before), a_in_gib=gib(a_in),
                        peak_gib=gib(peak), a_after_gib=gib(a_after),
                        a_final_gib=gib(a_final),
                        mx_cache_before_gib=gib(c_before), mx_cache_after_gib=gib(c_after),
                        transient_total_gib=gib(peak - a_final),
                        transient_per_lane_gib=gib(peak - a_final) / n,
                        transient_above_before_per_lane_gib=gib(peak - a_before) / n,
                    )
                    if rep > 0 and (best is None or row["transient_per_lane_gib"] > best["transient_per_lane_gib"]):
                        best = row
                    print(
                        f"  k={k} w={width} rep={rep}: peak={gib(peak):.3f} a_before={gib(a_before):.3f} "
                        f"a_after={gib(a_after):.3f} a_final={gib(a_final):.3f} "
                        f"-> transient/lane={row['transient_per_lane_gib']:.4f} GiB",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  k={k} rep={rep} FAILED: {type(e).__name__}: {e}", flush=True)
                    mx.clear_cache()
              if best is not None:
                rows.append(best)

            del cache
            gc.collect()
            mx.clear_cache()

    with open(args.out, "w") as f:
        json.dump({"weights_gib": weights_gib, "rows": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
