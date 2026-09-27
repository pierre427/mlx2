"""Flash-Next native-MTP decode under MoE expert streaming, MTP head streamed vs resident.

One arm per process (the streamed install replaces modules and cannot be
undone in place). Loads the adapter, installs expert streaming with
``--cache-gib`` and ``--mtp-resident`` on or off, prefills ``--context``
tokens and decodes ``--gen`` greedy tokens on the self-MTP route (B1; streaming
is single lane by construction). Records page-ins and hits per emitted token,
per-cache page-ins for the MTP head, decode tok/s (streamed throughput is a
diagnostic, not a benchmark), the plan, and a token hash.

  PYTHONPATH=src .venv/bin/python scripts/bench_fn_stream_mtp_resident.py --i-own-the-gpu \
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --prompt-file p.txt \
      --cache-gib 24 [--mtp-resident] --out arm.json
"""

import argparse
import hashlib
import json
import time

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--cache-gib", type=float, default=24.0)
    ap.add_argument("--mtp-resident", action="store_true")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--prefill-step", type=int, default=8192)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime import generate as G
    from mlx2.runtime.sample_utils import LaneRNG
    from mlx2.runtime.weight_stream import install_expert_streaming, is_mtp_path

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(8 << 30)
    loaded_gib = mx.get_active_memory() / 2**30
    t0 = time.perf_counter()
    manager = install_expert_streaming(
        model, a.model, ceiling_bytes=int(a.cache_gib * (1 << 30)), top_k=10,
        read_workers=16, mtp_resident=a.mtp_resident,
    )
    install_s = time.perf_counter() - t0
    active_after_install = mx.get_active_memory() / 2**30
    print("INSTALLED", json.dumps(manager.plan.as_dict()), f"active={active_after_install:.1f}GiB", flush=True)

    mtp_caches = [c for p, c in manager.caches.items() if is_mtp_path(p)]
    mtp_misses = {"n": 0}
    for cache in mtp_caches:  # count page-ins that land on the MTP head's caches
        original = cache.reader.read_bytes

        def counted(expert, _orig=original):
            mtp_misses["n"] += 1
            return _orig(expert)

        cache.reader.read_bytes = counted

    ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))[: a.context]
    gen = G.BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                           prefill_step_size=a.prefill_step,
                           self_mtp={"num_draft": 2, "persistent": True, "rate_gate": False,
                                     "prefill_step_size": a.prefill_step})
    gen.insert([ids], max_tokens=[a.gen], lane_rngs=[LaneRNG(1)],
               self_mtp_configs=[{"sampling_temp": 0.0}])
    tokens, t_first, t_end, emitted = [], None, None, 0
    base_counters = None
    try:
        done = False
        while not done:
            _p, responses = gen.next()
            now = time.perf_counter()
            for r in responses:
                tokens.append(int(r.token))
                done = done or bool(r.finish_reason)
            if responses and t_first is None:
                t_first = now
                base_counters = dict(manager.counters())
                base_mtp = mtp_misses["n"]
                continue
            if t_first is not None:
                emitted += len(responses)
                t_end = now
    finally:
        gen.close()
    end = manager.counters()
    decode = {k: end[k] - base_counters.get(k, 0) for k in end if k.endswith("_total")}
    rec = {
        "label": a.label, "mtp_resident": a.mtp_resident, "cache_gib": a.cache_gib,
        "context": a.context, "gen": a.gen, "emitted": emitted,
        "decode_tps": emitted / (t_end - t_first),
        "page_ins_per_token": decode["stream_page_ins_total"] / max(1, emitted),
        "page_in_mib_per_token": decode["stream_page_in_bytes_total"] / max(1, emitted) / 2**20,
        "hit_rate": decode["stream_expert_hits_total"] / max(1, decode["stream_expert_hits_total"] + decode["stream_expert_misses_total"]),
        "mtp_head_page_ins_decode": mtp_misses["n"] - base_mtp,
        "decode_counters": decode, "end_counters": end,
        "plan": manager.plan.as_dict(), "install_s": install_s,
        "loaded_active_gib": loaded_gib, "active_after_install_gib": active_after_install,
        "peak_gib": mx.get_peak_memory() / 2**30,
        "tokens_sha": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16],
        "mlx": mx.__version__,
    }
    manager.close()
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("end_counters", "plan")}))


if __name__ == "__main__":
    main()
