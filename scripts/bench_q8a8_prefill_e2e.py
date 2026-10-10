"""In-process A/B of int8 prefill weight modes on one GS64 affine checkpoint.

One model load. Each arm is an int8_prefill policy (JSON mapping, or "off")
applied with ``int8_prefill.apply`` and removed after the arm, so every arm
runs the same weights in the same process. Arms rotate per rep, with a settle
sleep before every timed prefill to limit thermal carry-over.

Throughput: prompt tokens / wall time to evaluated caches, chunked prefill
from an empty cache. Quality: teacher-forced next-token distributions of each
arm vs the "off" arm on per-category texts (single chunk, rows >= threshold so
the int8 path runs): mean KL(off||arm), top-1 agreement, delta NLL.

A mechanism check refuses an arm whose int8 counters did not move.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def build_texts(tokenizer, repo: Path, n_tokens: int):
    prose = "\n\n".join(
        p.read_text() for p in sorted((repo / "docs").glob("*.md"))[:40]
    )
    code = "\n\n".join(
        p.read_text() for p in sorted((repo / "src/mlx2/runtime").glob("*.py"))[:30]
    )
    tools = [
        {"type": "function", "function": {
            "name": f"tool_{i}", "description": f"Operation number {i} on the record store.",
            "parameters": {"type": "object", "properties": {
                "record_id": {"type": "string"}, "limit": {"type": "integer"},
                "fields": {"type": "array", "items": {"type": "string"}}},
                "required": ["record_id"]}}} for i in range(12)]
    turns = []
    for i in range(60):
        call = {"name": f"tool_{i % 12}", "arguments": {
            "record_id": f"rec-{i * 7919 % 10007}", "limit": i % 9 + 1,
            "fields": ["title", "owner", "status"][: i % 3 + 1]}}
        turns.append({"role": "user", "content": f"Look up record rec-{i * 7919 % 10007}."})
        turns.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function", "function": {
                "name": call["name"], "arguments": json.dumps(call["arguments"])}}]})
        turns.append({"role": "tool", "tool_call_id": f"call_{i}", "content": json.dumps(
            {"record_id": call["arguments"]["record_id"], "title": f"Item {i}",
             "owner": f"user{i % 5}", "status": ["open", "closed", "pending"][i % 3]})})
    try:
        tool_text = tokenizer.apply_chat_template(
            turns, tools=tools, tokenize=False, add_generation_prompt=False)
        tool_source = "chat_template"
    except Exception as error:  # template without tool support
        print("tool template failed, using JSON text:", error, flush=True)
        tool_text = json.dumps({"tools": tools, "messages": turns}, indent=1)
        tool_source = f"json_fallback: {type(error).__name__}: {error}"
    reasoning = "\n".join(
        f"Step {i}: if x_{i} = {i * 3 % 17} and y_{i} = {i * 5 % 13}, then "
        f"x_{i} * y_{i} + {i} = {(i * 3 % 17) * (i * 5 % 13) + i}. Therefore the "
        f"running total is {sum((j * 3 % 17) * (j * 5 % 13) + j for j in range(i + 1))}."
        for i in range(400))
    out = {}
    for name, text in (("prose", prose), ("code", code), ("tool_calls", tool_text),
                       ("reasoning", reasoning)):
        ids = list(tokenizer.encode(text))
        if len(ids) < n_tokens:
            ids = (ids * (n_tokens // max(len(ids), 1) + 1))
        out[name] = ids[:n_tokens]
    return out, tool_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--arms", required=True, help="JSON object name -> policy ('off' or mapping)")
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 8192, 16384])
    ap.add_argument("--step", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--settle", type=float, nargs="+", default=[10, 30, 45, 60],
                    help="settle seconds per context (aligned with --contexts)")
    ap.add_argument("--quality-tokens", type=int, default=2048)
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--host-cpu-limit", type=float, default=50.0,
                    help="a foreign process at or above this %%CPU (or a pytest at >= 5%%) "
                    "marks a timed run contaminated")
    ap.add_argument("--quality-first", action="store_true",
                    help="run the teacher-forced quality phase before the timed arms")
    ap.add_argument("--memory-log-s", type=float, default=0.0,
                    help="log own MLX memory, free pages and Swapouts every N seconds, "
                    "and hard-exit on a Swapouts rise")
    ap.add_argument("--wait-quiet", type=float, default=0.0,
                    help="seconds to wait for a quiet host before each timed run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--adapter-select", action="store_true",
                    help="install through int8_prefill.apply_for_adapter (the adapter's "
                    "declared scopes and int8_prefill_select, as serving binds it) "
                    "instead of the default path classifier")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx2.adapters.registry import resolve_adapter

    arms = json.loads(a.arms)
    if "off" not in arms:
        ap.error("arms must include 'off' (the reference)")
    # The adapter pins import-time env flags; model modules import after it.
    adapter = resolve_adapter(a.model, mtp=False)(a.model)
    from mlx2.runtime import int8_prefill
    from mlx2.runtime.models.cache import make_prompt_cache
    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(6 << 30)
    texts, tool_source = build_texts(adapter.tokenizer, Path(a.repo), max(max(a.contexts), a.quality_tokens))
    base_ids = texts["prose"] + texts["code"]

    def swapouts():
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Swapouts"):
                return int(line.split()[-1].rstrip("."))
        return -1

    def state_arrays(cache):
        return [v for _, v in tree_flatten([getattr(c, "state", None) for c in cache])
                if isinstance(v, mx.array)]

    def foreign_load():
        """Foreign CPU-heavy processes (or any pytest) on the host right now."""
        import os
        import subprocess
        out = subprocess.run(["ps", "-Ao", "pcpu=,pid=,command="],
                             capture_output=True, text=True).stdout
        hits = []
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3 or int(parts[1]) == os.getpid():
                continue
            cpu = float(parts[0])
            # A shell that merely names pytest (an idle wrapper) is not load.
            if cpu >= a.host_cpu_limit or ("pytest" in parts[2] and cpu >= 5.0):
                hits.append(f"{parts[0]}% pid {parts[1]} {parts[2][:100]}")
        return hits

    def quiet_host():
        deadline = time.monotonic() + a.wait_quiet
        hits = foreign_load()
        while hits and time.monotonic() < deadline:
            time.sleep(10)
            hits = foreign_load()
        return hits

    def install(name):
        policy = arms[name]
        if policy == "off":
            return None
        policy = int8_prefill.Int8PrefillPolicy.from_value(policy)
        if a.adapter_select:
            return int8_prefill.apply_for_adapter(adapter, policy)
        return int8_prefill.apply(model, policy)

    def status(handle):
        return {} if handle is None else handle.status()

    def prefill(n):
        ids = mx.array(base_ids[:n], mx.uint32)[None]
        cache = make_prompt_cache(model)
        start = time.perf_counter()
        pos = 0
        while pos < n:
            chunk = ids[:, pos: pos + a.step]
            out = model(chunk, cache=cache)
            mx.eval(out, state_arrays(cache))
            pos += chunk.shape[1]
        elapsed = time.perf_counter() - start
        del cache, out
        mx.clear_cache()
        return elapsed

    def logprobs(ids):
        cache = make_prompt_cache(model)
        logits = model(mx.array(ids, mx.uint32)[None], cache=cache)[0]
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        lp = lp.astype(mx.float16)
        mx.eval(lp)
        del cache, logits
        mx.clear_cache()
        return lp

    def run_quality():
        """Per category: the stock reference, then every arm against it, so one
        full-vocab reference is resident at a time."""
        quality = {name: {} for name in names if name != "off"}
        for cat, ids in texts.items():
            ref = logprobs(ids[: a.quality_tokens])
            for name in quality:
                h = install(name)
                lp = logprobs(ids[: a.quality_tokens])
                int8_prefill.remove(h)
                p = mx.exp(ref.astype(mx.float32))
                kl = (p * (ref.astype(mx.float32) - lp.astype(mx.float32))).sum(-1)
                top1 = (mx.argmax(ref, -1) == mx.argmax(lp, -1)).astype(mx.float32)
                tgt = mx.array(ids[1: a.quality_tokens], mx.int32)
                nll_ref = -mx.take_along_axis(ref[:-1].astype(mx.float32), tgt[:, None], -1)
                nll_arm = -mx.take_along_axis(lp[:-1].astype(mx.float32), tgt[:, None], -1)
                mx.eval(kl, top1, nll_ref, nll_arm)
                quality[name][cat] = {
                    "mean_kl": kl.mean().item(), "p99_kl": float(mx.sort(kl)[int(0.99 * kl.size)].item()),
                    "frac_kl_gt_1": (kl > 1).astype(mx.float32).mean().item(),
                    "top1_agree": top1.mean().item(),
                    "delta_nll": (nll_arm.mean() - nll_ref.mean()).item(),
                }
                del lp, p, kl, top1, nll_ref, nll_arm
                mx.clear_cache()
                print(name, cat, quality[name][cat], flush=True)
            del ref
            mx.clear_cache()
        return quality

    s0 = swapouts()
    print("LOADED; swapouts", s0, flush=True)
    names = list(arms)
    quality = {}
    if a.memory_log_s:
        import os
        import subprocess
        import threading

        def sampler():
            while True:
                out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
                page = 16384
                free = sw = -1
                for line in out.splitlines():
                    if "page size of" in line:
                        page = int(line.split("page size of")[1].split()[0])
                    elif line.startswith("Pages free"):
                        free = int(line.split()[-1].rstrip("."))
                    elif line.startswith("Swapouts"):
                        sw = int(line.split()[-1].rstrip("."))
                print(f"MEM {time.strftime('%H:%M:%S')} active {mx.get_active_memory() / 2**30:.1f} "
                      f"cache {mx.get_cache_memory() / 2**30:.1f} peak {mx.get_peak_memory() / 2**30:.1f} "
                      f"GiB; host free {free * page / 2**30:.1f} GiB; swapouts +{sw - s0}", flush=True)
                if sw > s0 + 1000:
                    print("ABORT: host started swapping (sampler)", flush=True)
                    os._exit(3)
                time.sleep(a.memory_log_s)

        threading.Thread(target=sampler, daemon=True).start()
    if not a.skip_quality and a.quality_first:
        quality = run_quality()
    # Warm-up every arm once (JIT, allocator, weight copies for cached modes).
    for name in names:
        h = install(name)
        prefill(min(a.contexts))
        int8_prefill.remove(h)
    results = {name: {str(c): [] for c in a.contexts} for name in names}
    contamination = {name: {str(c): [] for c in a.contexts} for name in names}
    engaged = {}
    for rep in range(a.reps):
        order = names[rep % len(names):] + names[: rep % len(names)]
        for ci, ctx in enumerate(a.contexts):
            for name in order:
                h = install(name)
                if h is not None and hasattr(h, "warmup"):
                    h.warmup()
                time.sleep(a.settle[min(ci, len(a.settle) - 1)])
                before = quiet_host()
                el = prefill(ctx)
                foreign = before + [h for h in foreign_load() if h not in before]
                contamination[name][str(ctx)].append(foreign)
                st = status(h)
                if h is not None:
                    # Mechanism check: the arm's kernels must have run.
                    counts = st.get("counts", {})
                    need = ["engaged_calls"]
                    need += ["q8_calls"] if h.policy.q8_inplace else []
                    need += ["q45_calls"] if h.policy.q45_inplace else []
                    idle = [key for key in need if not counts.get(key)]
                    if idle:
                        raise SystemExit(f"ABORT: arm {name} did not engage: {idle}")
                engaged.setdefault(name, st)
                int8_prefill.remove(h)
                results[name][str(ctx)].append(el)
                print(f"{time.strftime('%H:%M:%S')} rep{rep} ctx{ctx} {name}: {el:.2f}s "
                      f"{ctx / el:.0f} tok/s" + (f" CONTAMINATED {foreign}" if foreign else ""),
                      flush=True)
                if swapouts() > s0 + 1000:
                    raise SystemExit("ABORT: host started swapping")
    summary = {
        name: {ctx: {"tok_s_median": int(ctx) / statistics.median(v), "seconds": v}
               for ctx, v in per.items()} for name, per in results.items()}
    paired = {}
    for name in names:
        if name == "off":
            continue
        paired[name] = {}
        for ctx in map(str, a.contexts):
            gains = [o / x - 1 for o, x in zip(results["off"][ctx], results[name][ctx])]
            paired[name][ctx] = {"median_gain": statistics.median(gains),
                                 "min": min(gains), "max": max(gains)}
            # Pairs where neither run saw a foreign CPU-heavy process.
            clean = [g for g, co, cx in zip(gains, contamination["off"][ctx],
                                            contamination[name][ctx]) if not co and not cx]
            paired[name][ctx]["clean_pairs"] = len(clean)
            if clean:
                paired[name][ctx].update(clean_median_gain=statistics.median(clean),
                                         clean_min=min(clean), clean_max=max(clean))

    if not a.skip_quality and not a.quality_first:
        quality = run_quality()
    rec = {"model": a.model, "step": a.step, "reps": a.reps, "arms": arms,
           "adapter": type(adapter).__name__, "adapter_select": a.adapter_select,
           "tool_text_source": tool_source,
           "summary": summary, "paired_gain_vs_off": paired, "contamination": contamination, "quality_vs_off": quality,
           "int8_status_first_run": engaged, "peak_gib": mx.get_peak_memory() / 2**30,
           "swapouts_delta": swapouts() - s0, "mlx": mx.__version__}
    Path(a.out).write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"paired_gain_vs_off": paired, "quality": quality}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
