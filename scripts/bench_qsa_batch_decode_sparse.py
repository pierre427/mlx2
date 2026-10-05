"""In-process A/B of the batched one-token sparse QSA arm (oMLX #4070 port).

One model load, the served Flash-Next policy.  Each config ``route:lanes:context``
prefills ``context`` real tokens per lane (one distinct slice of
``--prompt-file`` per lane) through the served BatchGenerator and decodes
``--gen`` greedy tokens.  Route ``mtp`` runs native MTP with the served
MTP->ordinary handoff (max MTP width 3), route ``ordinary`` plain decode.

Per config, after a discarded short warm-up per arm:
  * one full run per arm (``base`` first, then each ``--arms`` arm): greedy
    tokens for identity against base, aggregate decode tok/s;
  * one ALTERNATING run: the arm switches every ``--window`` response polls
    (base, arm, base, arm, ...; the first ``--settle`` polls of each window are
    dropped as possibly dispatched under the previous arm), giving paired
    per-window tok/s on the same rows at the same depth.  Only with exactly
    one non-base arm.
Before a config runs, the serving memory admission
(``SelfMTPLaneAdmissionController`` with the adapter's cache estimator, live
headroom) is asked to seat the cohort atomically; a config it would not seat
is recorded as refused and skipped.  Run ONE config per process: after a
config's runs the process footprint stays high, so live headroom (and with it
admission) reads lower for every later config in the same process.  A run of a non-base arm whose mechanism
counter stayed 0 is marked invalid.

  MLX_ENABLE_TF32=0 PYTHONPATH=src python scripts/bench_qsa_batch_decode_sparse.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --prompt-file prompt.txt --configs ordinary:4:32768 --arms gather --out ab.json
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

IDLE_POLL_LIMIT = 4096


def _drain_poll(gen, idle):
    _p, responses = gen.next()
    take = getattr(gen, "take_lane_failures", None)
    lost = take() if take is not None else []
    if lost:
        raise RuntimeError("generator dropped lane(s): " + "; ".join(str(f) for f in lost))
    idle = 0 if responses else idle + 1
    if idle > IDLE_POLL_LIMIT:
        raise RuntimeError(f"no response in {IDLE_POLL_LIMIT} polls")
    return responses, idle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--configs", nargs="+", default=["ordinary:4:32768"])
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["gather"],
        choices=[
            "gather",
            "indexed",
            "varlen",
            "tensorfold",
            "composed",
            "varlen_tensorfold",
            "indexed_tensorfold",
            "composed_tensorfold",
        ],
    )
    ap.add_argument(
        "--varlen-composition",
        action="store_true",
        help="install sparse-MoE live-row compaction and batch prompt rows",
    )
    ap.add_argument(
        "--tensorfold-composition",
        action="store_true",
        help="install default-off native TensorFold prefill and toggle it per arm",
    )
    ap.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        help="per-lane prompt lengths for a single composition config",
    )
    ap.add_argument("--min-context", type=int, default=16384,
                    help="arm floor for this run (the policy default is 32768; 16384 measured the 16K cell)")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--alt-gen", type=int, default=384)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--settle", type=int, default=2)
    ap.add_argument("--no-full-runs", action="store_true")
    ap.add_argument("--prefill-step", type=int, default=8192)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--handoff-width", type=int, default=3)
    ap.add_argument("--admission-table", action="store_true",
                    help="print what admission seats at 4/8 lanes x 16K..64K first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-source")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    root = Path(__file__).resolve().parents[1]
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if a.expected_source is not None:
        if source_revision != a.expected_source:
            raise RuntimeError("source revision differs from explicit gate binding")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
            raise RuntimeError("source worktree must be clean")

    from mlx2.adapters.flash_next import FlashNextAdapter

    execution_policy = {}
    if a.varlen_composition:
        execution_policy["varlen_sparse_moe"] = True
    if a.tensorfold_composition:
        execution_policy["tensorfold_prefill"] = True
        execution_policy["tensorfold_qmv_rows"] = True
    adapter = FlashNextAdapter(a.model, execution_policy=execution_policy or None)
    from mlx2.memory import execution_headroom, host_memory_gib, metal_advisory_gib
    from mlx2.runtime import generate as G
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
    from mlx2.runtime.models import qwen4_exp as Q
    from mlx2.runtime.models import qwen4_qsa_indexed as I
    from mlx2.runtime.models import flash_tensorfold_qmv as TFQ
    from mlx2.runtime.sample_utils import LaneRNG

    model = adapter.model
    varlen_handle = getattr(adapter, "varlen_sparse_moe", None)
    if a.varlen_composition and varlen_handle is None:
        raise RuntimeError("varlen composition requested but adapter installed no handle")
    tensorfold_handle = getattr(adapter, "tensorfold_prefill", None)
    tensorfold_qmv_handle = getattr(adapter, "tensorfold_qmv", None)
    if a.tensorfold_composition and tensorfold_handle is None:
        raise RuntimeError("TensorFold composition requested but adapter installed no handle")
    if a.tensorfold_composition and tensorfold_qmv_handle is None:
        raise RuntimeError("TensorFold composition requested but adapter installed no QMV handle")
    tensorfold_modules = [
        module
        for _name, module in model.named_modules()
        if getattr(module, "_prefill_counts", None)
        is (None if tensorfold_handle is None else tensorfold_handle["counters"])
    ]
    if a.tensorfold_composition and not tensorfold_modules:
        raise RuntimeError("TensorFold composition installed no toggleable modules")
    tensorfold_qmv_modules = [
        module
        for _name, module in model.named_modules()
        if hasattr(module, "_tensorfold_qmv_enabled")
    ]
    if a.tensorfold_composition and not tensorfold_qmv_modules:
        raise RuntimeError("TensorFold composition installed no toggleable QMV modules")
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    ids = list(adapter.tokenizer.encode(open(a.prompt_file).read()))
    print("LOADED", f"tokens={len(ids)}", f"active={mx.get_active_memory() / 2**30:.1f}GiB",
          "policy", adapter.policy.as_dict(), flush=True)

    budget = adapter.cache_budget(mtp=True)
    controller = SelfMTPLaneAdmissionController(
        host_memory_gib=host_memory_gib(), advisory_gib=metal_advisory_gib(),
        cache_estimator=budget.project,
        transient_gib_per_lane=getattr(budget, "transient_gib_per_lane",
                                       SelfMTPLaneAdmissionController.K2_TRANSIENT_GIB_PER_LANE))

    def admission(route, lanes, context):
        mx.clear_cache()
        free = execution_headroom() / 2**30
        decision = controller.decide([context] * lanes, free, atomic_cohort=True,
                                     max_draft=2 if route == "mtp" else 0)
        return {"free_gib": round(free, 2), "stage": decision.stage,
                "modes": list(decision.modes), "estimated_gib": round(decision.estimated_gib, 2),
                "usable_gib": round(decision.usable_gib, 2),
                "seated": all(m != "queue" for m in decision.modes)}

    def configure(arm):
        qsa_arms = {
            "indexed", "composed", "indexed_tensorfold", "composed_tensorfold"
        }
        varlen_arms = {
            "varlen", "composed", "varlen_tensorfold", "composed_tensorfold"
        }
        tensorfold_arms = {
            "tensorfold", "varlen_tensorfold", "indexed_tensorfold",
            "composed_tensorfold",
        }
        qsa_arm = (
            "indexed"
            if arm in qsa_arms
            else "gather" if arm == "gather" else "off"
        )
        Q.set_qsa_batch_decode_sparse(qsa_arm, min_context=a.min_context)
        if varlen_handle is not None:
            object.__setattr__(
                model,
                "_varlen_sparse_moe",
                varlen_handle if arm in varlen_arms else None,
            )
        for module in tensorfold_modules:
            object.__setattr__(module, "_prefill_enabled", arm in tensorfold_arms)
        for module in tensorfold_qmv_modules:
            object.__setattr__(module, "_tensorfold_qmv_enabled", arm in tensorfold_arms)

    def counter_delta(before, after):
        keys = set(before) | set(after)
        delta = {
            key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in keys
        }
        if any(value < 0 for value in delta.values()):
            raise RuntimeError("varlen counters moved backwards")
        return {key: value for key, value in sorted(delta.items()) if value}

    def indexed_widths():
        status = I.qsa_indexed_status()
        return {"counts": dict(status["counts"]), "widths": json.loads(json.dumps(status["query_width_counts"]))}

    def run(route, lanes, context, gen, arm, alternate=None):
        lengths = (
            list(a.lengths)
            if a.lengths is not None and max(a.lengths) <= context
            else [context] * lanes
        )
        if len(lengths) != lanes or any(length < 2 or length > context for length in lengths):
            raise ValueError("lengths must provide one value in 2..context per lane")
        cursor = a.offset
        prompts = []
        for length in lengths:
            prompts.append(ids[cursor : cursor + length])
            cursor += length
        assert all(len(prompt) == length for prompt, length in zip(prompts, lengths)), (
            "prompt file too short"
        )
        kwargs = dict(
            completion_batch_size=lanes,
            prefill_batch_size=lanes if a.varlen_composition else 1,
            prefill_step_size=a.prefill_step,
        )
        insert = {"max_tokens": [gen] * lanes, "lane_rngs": [LaneRNG(1 + i) for i in range(lanes)]}
        if route == "mtp":
            kwargs["self_mtp"] = adapter.execution_config(max_lanes=lanes, prefill_step=a.prefill_step)
            kwargs["mtp_ordinary_handoff"] = MTPOrdinaryHandoffPolicy(
                enabled=True, max_mtp_width=a.handoff_width)
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}] * lanes
        configure(arm if alternate is None else "base")
        Q.qsa_batch_decode_sparse_status(reset=True)
        varlen_before = (
            dict(varlen_handle["counters"]) if varlen_handle is not None else {}
        )
        tensorfold_before = (
            dict(tensorfold_handle["counters"])
            if tensorfold_handle is not None
            else {}
        )
        tensorfold_qmv_before = TFQ.counters()
        before = indexed_widths()
        g = G.BatchGenerator(model, **kwargs)
        run_started = time.perf_counter()
        uids = g.insert(prompts, **insert)
        tokens, done, started = {}, set(), set()
        t_first = t_end = None
        emitted = steps = 0
        windows = []  # (arm, tokens, seconds)
        polls_in_window = 0
        window_arm = "base"
        window_tokens = 0
        window_start = None
        try:
            idle = 0
            while len(done) < lanes:
                responses, idle = _drain_poll(g, idle)
                now = time.perf_counter()
                if t_first is not None and responses:
                    steps += 1
                started.update(r.uid for r in responses)
                first_now = False
                if responses and t_first is None and started >= set(uids):
                    t_first = now
                    emitted -= len(responses)
                    first_now = True
                for r in responses:
                    tokens.setdefault(r.uid, []).append(int(r.token))
                    if r.finish_reason:
                        done.add(r.uid)
                if t_first is not None:
                    emitted += len(responses)
                    t_end = now
                if alternate is not None and t_first is not None and responses and not first_now:
                    polls_in_window += 1
                    if polls_in_window == a.settle:
                        window_start, window_tokens = now, 0
                    elif polls_in_window > a.settle:
                        window_tokens += len(responses)
                    if polls_in_window == a.window:
                        if window_start is not None and len(done) == 0:
                            windows.append((window_arm, window_tokens, now - window_start))
                        window_arm = alternate if window_arm == "base" else "base"
                        configure(window_arm)
                        polls_in_window, window_start = 0, None
        finally:
            g.close()
            configure("base")
        mx.clear_cache()
        lane_tokens = [tokens.get(u, []) for u in uids]
        sha = hashlib.sha256(json.dumps(lane_tokens).encode()).hexdigest()[:16]
        wall = t_end - t_first
        after = indexed_widths()
        sparse = Q.qsa_batch_decode_sparse_status()
        varlen = counter_delta(
            varlen_before,
            dict(varlen_handle["counters"]) if varlen_handle is not None else {},
        )
        tensorfold = counter_delta(
            tensorfold_before,
            dict(tensorfold_handle["counters"])
            if tensorfold_handle is not None
            else {},
        )
        tensorfold_qmv = counter_delta(tensorfold_qmv_before, TFQ.counters())
        rec = {"arm": arm if alternate is None else f"alt:{alternate}", "tps": emitted / wall,
               "ms_per_step": 1e3 * wall / max(1, steps), "tokens_per_step": emitted / max(1, steps),
               "steps": steps, "sha": sha, "prompt_lengths": lengths,
               "ttft_seconds": t_first - run_started,
               "complete_seconds": t_end - run_started,
               "batch_decode_sparse": sparse, "varlen_sparse_moe": varlen,
               "tensorfold_prefill": tensorfold,
               "tensorfold_qmv_rows": tensorfold_qmv,
               "indexed_counts_delta": {k: v - before["counts"].get(k, 0) for k, v in after["counts"].items()
                                        if v - before["counts"].get(k, 0)},
               "handoff": {k: v for k, v in (getattr(g, "scheduler_stats", {}) or {}).items()
                           if "handoff" in str(k)}}
        if alternate is not None:
            rec["windows"] = windows
        engaged_arm = arm if alternate is None else alternate
        qsa_expected = engaged_arm in {
            "gather", "indexed", "composed", "indexed_tensorfold",
            "composed_tensorfold",
        }
        varlen_expected = engaged_arm in {
            "varlen", "composed", "varlen_tensorfold", "composed_tensorfold"
        }
        tensorfold_expected = engaged_arm in {
            "tensorfold", "varlen_tensorfold", "indexed_tensorfold",
            "composed_tensorfold",
        }
        if qsa_expected and sparse["engagements"] == 0:
            rec["INVALID"] = "mechanism counter stayed 0"
        if not qsa_expected and sparse["engagements"]:
            rec["INVALID"] = "QSA engaged in a control arm"
        compacted = varlen.get("moe_compaction_calls", 0)
        scattered = varlen.get("moe_scatter_calls", 0)
        if varlen_expected and (compacted < 1 or compacted != scattered):
            rec["INVALID"] = "varlen compaction proof stayed 0 or unbalanced"
        if not varlen_expected and varlen:
            rec["INVALID"] = "varlen engaged in a control arm"
        tensorfold_prefill_calls = tensorfold.get(
            "grouped_calls", 0
        ) + tensorfold.get("swiglu_calls", 0)
        tensorfold_qmv_calls = tensorfold_qmv.get("kernel_calls", 0)
        if tensorfold_expected and tensorfold_prefill_calls < 1:
            rec["INVALID"] = "TensorFold prefill proof stayed 0"
        if tensorfold_expected and tensorfold_qmv_calls < 1:
            rec["INVALID"] = "TensorFold decode-row proof stayed 0"
        if not tensorfold_expected and tensorfold:
            rec["INVALID"] = "TensorFold prefill engaged in a control arm"
        if not tensorfold_expected and tensorfold_qmv_calls:
            rec["INVALID"] = "TensorFold decode rows engaged in a control arm"
        return rec, lane_tokens

    results = {}
    if a.admission_table:
        table = {}
        for route in ("ordinary", "mtp"):
            for lanes in (4, 8):
                for context in (16384, 24576, 32768, 49152, 65536):
                    adm = admission(route, lanes, context)
                    table[f"{route}:{lanes}:{context}"] = adm
                    print("ADMISSION", route, lanes, context, adm["stage"], adm["seated"],
                          adm["estimated_gib"], adm["usable_gib"], adm["free_gib"], flush=True)
        results["admission_table"] = table
    for config in a.configs:
        route, lanes, context = config.split(":")
        lanes, context = int(lanes), int(context)
        adm = admission(route, lanes, context)
        print(config, "admission", adm, flush=True)
        if not adm["seated"]:
            results[config] = {"admission": adm, "refused": True}
            json.dump({"partial": True, "results": results}, open(a.out, "w"), indent=1)
            continue
        for arm in ["base", *a.arms]:  # warm-up: build each arm's kernels
            run(route, lanes, 4096, 32, arm)
        runs = {}
        lane_tokens = {}
        if not a.no_full_runs:
            for arm in ["base", *a.arms]:
                rec, toks = run(route, lanes, context, a.gen, arm)
                runs[arm], lane_tokens[arm] = rec, toks
                print(f"{config} {arm} {rec['tps']:.2f} tok/s {rec['ms_per_step']:.2f} ms/step "
                      f"sha={rec['sha']} sparse={rec['batch_decode_sparse']['counts']} "
                      f"indexed={rec['indexed_counts_delta']} handoff={rec['handoff']}"
                      + (f" INVALID={rec['INVALID']}" if "INVALID" in rec else ""), flush=True)
        identity = {}
        for arm in a.arms:
            if arm not in lane_tokens:
                continue
            ref, got = lane_tokens["base"], lane_tokens[arm]
            first = None
            lanes_equal = 0
            for lane_ref, lane in zip(ref, got):
                k = next((i for i, (x, y) in enumerate(zip(lane_ref, lane)) if x != y), None)
                if k is None and len(lane_ref) == len(lane):
                    lanes_equal += 1
                elif k is not None:
                    first = k if first is None else min(first, k)
            identity[arm] = {"identical": lanes_equal == len(ref), "lanes_identical": lanes_equal,
                             "lanes": len(ref), "first_divergence": first}
        if "indexed" in lane_tokens and "composed" in lane_tokens:
            ref, got = lane_tokens["indexed"], lane_tokens["composed"]
            first = None
            lanes_equal = 0
            for lane_ref, lane in zip(ref, got):
                k = next(
                    (i for i, (x, y) in enumerate(zip(lane_ref, lane)) if x != y),
                    None,
                )
                if k is None and len(lane_ref) == len(lane):
                    lanes_equal += 1
                elif k is not None:
                    first = k if first is None else min(first, k)
            identity["composed_vs_indexed"] = {
                "identical": lanes_equal == len(ref),
                "lanes_identical": lanes_equal,
                "lanes": len(ref),
                "first_divergence": first,
            }
        if "composed" in lane_tokens and "composed_tensorfold" in lane_tokens:
            ref, got = lane_tokens["composed"], lane_tokens["composed_tensorfold"]
            first = None
            lanes_equal = 0
            for lane_ref, lane in zip(ref, got):
                k = next(
                    (i for i, (x, y) in enumerate(zip(lane_ref, lane)) if x != y),
                    None,
                )
                if k is None and len(lane_ref) == len(lane):
                    lanes_equal += 1
                elif k is not None:
                    first = k if first is None else min(first, k)
            identity["composed_tensorfold_vs_composed"] = {
                "identical": lanes_equal == len(ref),
                "lanes_identical": lanes_equal,
                "lanes": len(ref),
                "first_divergence": first,
            }
        alt = None
        if len(a.arms) == 1 and a.alt_gen > 0:
            alt, _ = run(route, lanes, context, a.alt_gen, "base", alternate=a.arms[0])
            per = {"base": [], a.arms[0]: []}
            for arm, n, sec in alt["windows"]:
                per[arm].append(n / sec)
            pairs = [100 * (x / y - 1) for x, y in zip(per[a.arms[0]], per["base"])]
            alt["paired_window_delta_pct"] = {
                "n": len(pairs), "median": statistics.median(pairs) if pairs else None,
                "min": min(pairs) if pairs else None, "max": max(pairs) if pairs else None,
                "faster": sum(p > 0 for p in pairs)}
            alt["window_tps_median"] = {k: statistics.median(v) if v else None for k, v in per.items()}
            print(f"{config} ALT {alt['paired_window_delta_pct']} medians={alt['window_tps_median']} "
                  f"sparse={alt['batch_decode_sparse']['counts']}"
                  + (f" INVALID={alt['INVALID']}" if "INVALID" in alt else ""), flush=True)
        results[config] = {"admission": adm, "runs": runs, "identity": identity, "alternating": alt}
        print(config, "identity", identity, flush=True)
        json.dump({"partial": True, "results": results}, open(a.out, "w"), indent=1)
    json.dump({"source_revision": source_revision,
               "prompt_file_sha256": hashlib.sha256(Path(a.prompt_file).read_bytes()).hexdigest(),
               "model": a.model, "gen": a.gen, "alt_gen": a.alt_gen, "window": a.window,
               "settle": a.settle, "arms": a.arms, "min_context": a.min_context,
               "varlen_composition": a.varlen_composition,
               "tensorfold_composition": a.tensorfold_composition,
               "lengths": a.lengths,
               "policy": adapter.policy.as_dict(), "results": results,
               "peak_gib": mx.get_peak_memory() / 2**30, "mlx": mx.__version__},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
