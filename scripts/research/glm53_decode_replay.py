"""GLM-5.3 decode record/replay: identical decode traffic, re-run cheaply.

``record``: prefill a realistic prompt once, save the prompt cache
(``save_prompt_cache``: MLA latent/indexer planes and KDA state) and the
greedy token stream of ``--steps`` decode steps.

``replay``: reload the snapshot and feed the recorded tokens as inputs, step
for step, so every replay sends the same traffic through the model. Each
replay checks its greedy outputs against the recording and times every step
as host build (graph construction + async_eval) and exposed GPU wait. Arms
(``--arms``): ``serial``, ``pipelined``, ``pipelined_nogc``,
``pipelined_cprofile`` (cProfile of the build phase only; the top functions
by cumulative time are kept). Research only; no serving path is changed.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_ordinary_profile import CHECKPOINT, MODEL, prompt_ids  # noqa: E402


def load(policy_path):
    import mlx.core as mx

    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    info = mx.device_info()
    if info.get("max_recommended_working_set_size"):
        mx.set_wired_limit(info["max_recommended_working_set_size"])
    policy = json.loads(Path(policy_path).read_text())
    return Glm53GGUFAdapter(MODEL, execution_policy=policy), policy


def record(args):
    import mlx.core as mx

    from mlx2.runtime.models.cache import save_prompt_cache

    adapter, policy = load(args.policy)
    model = adapter.model
    ids = prompt_ids(adapter, args.tokens)
    cache = model.make_cache()
    for begin in range(0, len(ids), 2048):
        piece = mx.array([ids[begin:begin + 2048]])
        with model._prompt_scope(piece.shape[1]):
            logits = model(piece, cache, logits_to_keep=1)
        mx.eval(logits, [c.state for c in cache])
    first = int(mx.argmax(logits[0, -1]).item())
    save_prompt_cache(str(args.out / "cache.safetensors"), cache)
    # Post-prefill state treatments, to find why decode right after a live
    # prefill is slower than decode on a restored snapshot of the same state.
    if args.after_prefill in ("clear_cache", "reload"):
        mx.clear_cache()
    if args.after_prefill == "reload":
        from mlx2.runtime.models.cache import load_prompt_cache

        del cache
        mx.clear_cache()
        cache = load_prompt_cache(str(args.out / "cache.safetensors"))
        for c in cache:
            if hasattr(c, "identity"):
                c.identity = model.identity
        mx.eval([c.state for c in cache])
    tokens = [first]
    gaps = []
    if args.pipelined_record:
        # Decode straight after the live prefill, pipelined like the server, so
        # the post-prefill cache state is timed (a replay restores a fresh snapshot).
        pending = mx.argmax(model(mx.array([[first]]), cache)[0, -1])
        mx.async_eval(pending)
        last = time.perf_counter()
        for _ in range(args.steps - 1):
            nxt = mx.argmax(model(pending.reshape(1, 1), cache)[0, -1])
            mx.async_eval(nxt)
            tokens.append(int(pending.item()))
            now = time.perf_counter()
            gaps.append(now - last)
            last = now
            pending = nxt
        tokens.append(int(pending.item()))
    else:
        for _ in range(args.steps):
            t0 = time.perf_counter()
            logits = model(mx.array([[tokens[-1]]]), cache)
            tokens.append(int(mx.argmax(logits[0, -1]).item()))
            gaps.append(time.perf_counter() - t0)
    (args.out / "trace.json").write_text(json.dumps({
        "schema": "mlx2.glm53-decode-trace.v1", "checkpoint": CHECKPOINT, "source_revision": args.source_revision,
        "policy": policy, "prompt_tokens": len(ids), "tokens": tokens}) + "\n")
    ordered = sorted(gaps)
    print(json.dumps({"recorded": len(tokens) - 1, "prompt_tokens": len(ids),
                      "mode": "pipelined" if args.pipelined_record else "serial",
                      "after_prefill": args.after_prefill,
                      "step_p50_ms": ordered[len(ordered) // 2] * 1e3 if ordered else None,
                      "first_64_p50_ms": sorted(gaps[:64])[len(gaps[:64]) // 2] * 1e3 if gaps else None,
                      "last_256_p50_ms": sorted(gaps[-256:])[len(gaps[-256:]) // 2] * 1e3 if gaps else None}),
          flush=True)


def replay(args):
    import mlx.core as mx

    from mlx2.runtime.models.cache import load_prompt_cache

    adapter, policy = load(args.policy)
    model = adapter.model
    trace = json.loads((args.trace / "trace.json").read_text())
    tokens = trace["tokens"]
    steps = min(args.steps or len(tokens) - 1, len(tokens) - 1)
    result = {"schema": "mlx2.glm53-decode-replay.v1", "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "trace": str(args.trace), "policy": policy,
              "prompt_tokens": trace["prompt_tokens"], "steps": steps, "arms": [],
              "mlx_env": {k: v for k, v in __import__("os").environ.items() if k.startswith("MLX_")}}
    for arm in args.arms:
        cache = load_prompt_cache(str(args.trace / "cache.safetensors"))
        if args.rebind_identity:
            # The KDA state identity binds the numerics revision, a hash of the
            # model sources; a host-only source edit changes it although the
            # stored state is the same. Rebind to the loaded model's identity
            # (research only); tokens_match still checks every output.
            recorded = {c.identity for c in cache if hasattr(c, "identity")}
            for c in cache:
                if hasattr(c, "identity"):
                    c.identity = model.identity
            result["rebound_identity_from"] = sorted(recorded)
        if args.merge:
            # The serving path decodes on merged batch caches (generate._merge_caches),
            # not the plain prompt caches the snapshot restores.
            from mlx2.runtime.generate import _merge_caches

            cache = _merge_caches([cache])
            result["merged_cache_types"] = sorted({type(c).__name__ for c in cache})
        mx.eval([c.state for c in cache])
        mx.clear_cache()
        builds, waits, outs, graphs, submits = [], [], [], [], []
        profiler = cProfile.Profile() if arm == "pipelined_cprofile" else None
        if arm == "pipelined_nogc":
            gc.collect()
            gc.disable()
        peaks = []
        if arm == "serial":
            for i in range(steps):
                mx.reset_peak_memory()
                active = mx.get_active_memory()
                t0 = time.perf_counter()
                y = mx.argmax(model(mx.array([[tokens[i]]]), cache)[0, -1])
                t1 = time.perf_counter()
                outs.append(int(y.item()))
                builds.append(t1 - t0)
                waits.append(time.perf_counter() - t1)
                # A full-buffer KV copy shows as a context-sized transient here.
                peaks.append(mx.get_peak_memory() - active)
        elif arm == "interleaved":
            # Same process, same cache: model policy attributes switch every
            # --block steps (A B C D A B ...). The levers compared here are
            # bitwise-exact, so the token stream is unchanged and thermal or
            # swap drift lands on every variant alike. The first --settle steps
            # after each switch (compile warm-up/retrace) are not counted.
            variants = json.loads(args.variants)
            labels = []
            pending = mx.argmax(model(mx.array([[tokens[0]]]), cache)[0, -1])
            mx.async_eval(pending)
            last = time.perf_counter()
            for i in range(1, steps):
                block = (i - 1) // args.block
                variant = variants[block % len(variants)]
                for attr, value in variant["attrs"].items():
                    setattr(model, attr, value)
                nxt = mx.argmax(model(mx.array([[tokens[i]]]), cache)[0, -1])
                mx.async_eval(nxt)
                outs.append(int(pending.item()))
                now = time.perf_counter()
                builds.append(now - last)
                waits.append(0.0)
                labels.append((variant["name"], (i - 1) % args.block >= args.settle))
                last = now
                pending = nxt
            outs.append(int(pending.item()))
            per, per_phase = {}, {}
            for k, ((name, counted), gap) in enumerate(zip(labels, builds)):
                if counted:
                    per.setdefault(name, []).append(gap)
                    phase = (trace["prompt_tokens"] + k) % 4
                    per_phase.setdefault(name, {}).setdefault(phase, []).append(gap)
            interleaved = {name: {"steps": len(v), "p50_ms": sorted(v)[len(v) // 2] * 1e3,
                                  "phase_p50_ms": {str(ph): sorted(g)[len(g) // 2] * 1e3
                                                   for ph, g in sorted(per_phase[name].items())}}
                           for name, v in per.items()}
        elif arm.startswith("pipelined_depth"):
            # Build up to D steps ahead of the token being read (the graph does
            # not depend on token values, so lookahead needs no prediction).
            depth = int(arm.removeprefix("pipelined_depth"))
            queue = []
            last = time.perf_counter()
            for i in range(steps):
                t0 = time.perf_counter()
                nxt = mx.argmax(model(mx.array([[tokens[i]]]), cache)[0, -1])
                tg = time.perf_counter()
                mx.async_eval(nxt)
                queue.append(nxt)
                graphs.append(tg - t0)
                submits.append(time.perf_counter() - tg)
                if len(queue) > depth:
                    outs.append(int(queue.pop(0).item()))
                    now = time.perf_counter()
                    builds.append(now - last)
                    waits.append(0.0)
                    last = now
            outs.extend(int(y.item()) for y in queue)
        else:
            pending = mx.argmax(model(mx.array([[tokens[0]]]), cache)[0, -1])
            mx.async_eval(pending)
            for i in range(1, steps):
                # Peak over [build step i, wait step i-1]: both steps' live
                # memory; a full-plane KV copy adds ~one plane set (~1 GB at
                # 32K over 11 layers) on top.
                mx.reset_peak_memory()
                active = mx.get_active_memory()
                t0 = time.perf_counter()
                if profiler:
                    profiler.enable()
                nxt = mx.argmax(model(mx.array([[tokens[i]]]), cache)[0, -1])
                tg = time.perf_counter()
                mx.async_eval(nxt)
                if profiler:
                    profiler.disable()
                t1 = time.perf_counter()
                outs.append(int(pending.item()))
                peaks.append(mx.get_peak_memory() - active)
                builds.append(t1 - t0)
                graphs.append(tg - t0)
                submits.append(t1 - tg)
                waits.append(time.perf_counter() - t1)
                pending = nxt
            outs.append(int(pending.item()))
        gc.enable()
        expected = tokens[1:steps + 1]
        if arm != "interleaved":
            interleaved = None

        def p50(values):
            values = sorted(values)
            return values[len(values) // 2] * 1e3 if values else None
        gaps = [b + w for b, w in zip(builds, waits)]
        row = {"arm": arm, "tokens_match": outs == expected, "interleaved": interleaved,
               "first_mismatch": next((i for i, (a, b) in enumerate(zip(outs, expected)) if a != b), None),
               "step_p50_ms": p50(gaps), "build_p50_ms": p50(builds), "wait_p50_ms": p50(waits),
               # build = Python graph construction (graph) + async_eval (submit; blocks
               # when MLX's command-buffer queue is full, i.e. GPU backpressure).
               "graph_p50_ms": p50(graphs), "submit_p50_ms": p50(submits),
               "peak_above_active_mb_p50": (sorted(peaks)[len(peaks) // 2] / 2**20) if peaks else None,
               "peak_above_active_mb_p90": (sorted(peaks)[int(len(peaks) * 0.9)] / 2**20) if peaks else None,
               "peaks_mb": [round(p / 2**20, 1) for p in peaks],
               "first_64_step_p50_ms": p50(gaps[:64]), "last_256_step_p50_ms": p50(gaps[-256:]),
               # Gap k waits on step k's token: input position prompt_tokens + k.
               # Phase 3 (position % 4 == 3) is the step that completes a pool.
               "phase_p50_ms": {str(ph): p50([g for k, g in enumerate(gaps)
                                              if (trace["prompt_tokens"] + k) % 4 == ph]) for ph in range(4)},
               "slow_steps": [{"i": i, "build_ms": round(b * 1e3, 1), "wait_ms": round(w * 1e3, 1)}
                              for i, (b, w) in enumerate(zip(builds, waits)) if b + w > 0.05],
               "builds_ms": [round(b * 1e3, 3) for b in builds], "waits_ms": [round(w * 1e3, 3) for w in waits]}
        if profiler:
            buf = io.StringIO()
            pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(40)
            row["build_profile_top"] = buf.getvalue()[-12000:]
        result["arms"].append(row)
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
                          if k not in ("builds_ms", "waits_ms", "build_profile_top", "slow_steps", "peaks_mb")}),
              "slow:", len(row["slow_steps"]), flush=True)
        del cache
        mx.clear_cache()
    (args.out / "replay.json").write_text(json.dumps(result, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Top-level, so the bench.sh lock recipe can pass them before the subcommand.
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("record", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--policy", type=Path, required=True)
        p.add_argument("--steps", type=int, default=512)
        if name == "record":
            p.add_argument("--tokens", type=int, default=32768)
            p.add_argument("--pipelined-record", action="store_true")
            p.add_argument("--after-prefill", choices=("none", "clear_cache", "reload"), default="none")
        else:
            p.add_argument("--trace", type=Path, required=True)
            p.add_argument("--rebind-identity", action="store_true",
                           help="replay a trace recorded under other model sources (state is checked by tokens_match)")
            p.add_argument("--variants", default="[]",
                           help='interleaved arm: JSON [{"name":..,"attrs":{model attr: value}}]')
            p.add_argument("--merge", action="store_true",
                           help="decode on merged batch caches, as the server does")
            p.add_argument("--block", type=int, default=64)
            p.add_argument("--settle", type=int, default=4)
            p.add_argument("--arms", nargs="+", default=["serial", "pipelined", "pipelined_nogc",
                                                         "pipelined_cprofile"])
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    (record if args.command == "record" else replay)(args)


if __name__ == "__main__":
    main()
