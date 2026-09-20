#!/usr/bin/env python3
"""GPU measurement package for self-MTP depth economics.

Every Metal/model execution refuses to run without ``--i-own-the-gpu``;
``--dry-run`` prints the plan and imports no MLX.

The confidence *scheduler* this package was written to evaluate (rm10) was
measured a no-go and is not on this tree.  What remains is the measurement
apparatus, which is what the verdict was built from and what any later depth
work needs: a measured cycle-cost table, a measured acceptance log, and an
interleaved A/B with per-request route assertions.

``profile``
    In-process cycle-cost table.  For each lane count and each fixed depth
    0..max, time ``BatchGenerator.next()`` cycles after warmup and write a
    ``mlx2.mtp_verify_cost.v2`` JSON whose ``cycle_table`` is normalised to
    the depth-0, 1-lane cycle.  Feed it to ``scripts/best_constant_depth.py``
    together with an acceptance log.
``collect``
    Launch ``mlx2.server --qualification-mode --mtp-acceptance-log`` at a fixed
    depth with greedy lookahead, drive the bundled mixed prompt set at the
    requested widths, and refuse the result if ``mtp_acceptance_log_records``
    is 0.
``ab``
    Interleaved arms (A B C A B C ...), one server per arm and pair, driven by
    ``scripts/benchmark_serving.py``.  Arms:
    - ``ordinary``: ``--ordinary`` (no self-MTP at all);
    - ``fixed``: the native self-MTP route at the adapter's depth;
    - ``ewma``: ``--adaptive-mtp-depth`` (the acceptance-EWMA controller).

    Every request of an MTP arm must carry a self-MTP receipt and the
    ``ordinary`` arm must carry none, and an arm whose mechanism counter is 0
    is refused (``adaptive_mtp_boundaries`` for ``ewma``).
``diag``
    Where a cycle's time goes: interleaved static depths against the EWMA
    controller pinned at a depth.

Run under the lab lock wrapper::

    cpg_job.py run --label ... --out <log> --lock -- <this command>
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PROFILE_PROMPTS = (0, 3, 4, 7)  # long-form: compiler, ISO date, LRU cache, sqrt(2)
PROMPTS = [
    # chat
    "Explain how a compiler works, in numbered sections.",
    "Write a short, friendly email declining a meeting invitation.",
    "Summarize the causes of the French Revolution in five bullet points.",
    # code
    "Write a Python function that parses an ISO-8601 date without libraries, with tests.",
    "Implement a thread-safe LRU cache in Go and explain the locking.",
    "Refactor this loop into a list comprehension: for x in xs: if x % 2: out.append(x * x)",
    # math / reasoning
    "Solve step by step: a train leaves at 3pm at 60 km/h, another at 4pm at 90 km/h; when does the second catch up?",
    "Prove that the square root of 2 is irrational.",
    # structured / repetitive (high acceptance)
    "List the first 40 prime numbers separated by commas.",
    "Produce a JSON array of 12 objects with fields id, name, and email for fictional users.",
]


_NATIVE_MTP_FLAG: list[bool] = []


def _server_has_native_mtp_flag() -> bool:
    """True when this checkout's server has an explicit ``--native-mtp`` opt-in.

    Main 16059c7 made Qwen3.6-35B default to the ordinary route; there a
    self-MTP arm must pass ``--native-mtp`` or it silently runs ordinary.
    """
    if not _NATIVE_MTP_FLAG:
        text = (ROOT / "src" / "mlx2" / "server.py").read_text()
        _NATIVE_MTP_FLAG.append('"--native-mtp"' in text)
    return _NATIVE_MTP_FLAG[0]


_MTP_ROUTES = ("segmented_self_mtp", "continuous_batched_self_mtp")


def _assert_route(arm: str, bench: dict) -> dict:
    """Every request of an MTP arm must carry a self-MTP receipt; ordinary none."""
    routes: dict[str, int] = {}
    for row in [*bench.get("warmup", {}).values(), *(r for x in bench.get("rows", []) for r in x["requests"])]:
        mtp = (row.get("receipt") or {}).get("mtp") or {}
        route = mtp.get("route", "ordinary")
        routes[route] = routes.get(route, 0) + 1
    if arm == "ordinary":
        if set(routes) != {"ordinary"}:
            raise SystemExit(f"REFUSED arm ordinary: saw MTP routes {routes}")
    elif not routes or not set(routes) <= set(_MTP_ROUTES):
        raise SystemExit(f"REFUSED arm {arm}: expected a self-MTP route on every request, saw {routes}")
    return routes


def _server_cmd(args, arm: str, extra: list[str]) -> list[str]:
    cmd = [
        sys.executable, "-m", "mlx2.server",
        "--model", args.model,
        "--port", str(args.port),
        "--max-lanes", str(max(args.widths)),
        "--max-inflight", str(max(args.widths) * 3),
        "--max-context", "32768",
        "--qualification-mode",
    ]
    if arm == "ordinary":
        return cmd + ["--ordinary"] + extra
    if args.execution_policy:
        cmd += ["--execution-policy", str(args.execution_policy)]
    if _server_has_native_mtp_flag():
        cmd += ["--native-mtp"]
    return cmd + extra


def _arm_flags(args, arm: str) -> list[str]:
    if arm in ("fixed", "ordinary"):
        return []
    if arm == "ewma":
        return ["--adaptive-mtp-depth"]
    raise SystemExit(f"unknown arm {arm!r}")


_MECHANISM = {
    "ewma": "adaptive_mtp_boundaries",
}


def _status(port: int) -> dict:
    with urlopen(f"http://127.0.0.1:{port}/v1/status", timeout=30) as response:
        return json.load(response)


def _start(cmd, log_path: Path, port: int, timeout: float):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early; see {log_path}")
        try:
            if _status(port).get("state") == "ready":
                return proc, log
        except OSError:
            pass
        time.sleep(2)
    proc.terminate()
    raise SystemExit(f"server not ready in {timeout}s; see {log_path}")


def _stop(proc, log) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    time.sleep(3)


def _drive(args, out: Path) -> dict:
    """Mixed prompt set at each width (greedy), then the standard benchmark."""
    bench = [
        sys.executable, str(ROOT / "scripts" / "benchmark_serving.py"),
        "--url", f"http://127.0.0.1:{args.port}", "--output", str(out),
        "--rounds", str(args.rounds), "--widths", *map(str, args.widths),
        "--max-tokens", str(args.max_tokens),
    ]
    subprocess.run(bench, check=True, cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    return json.loads(out.read_text())


def _mixed_prompts(args) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from urllib.request import Request

    def one(text):
        body = {
            "messages": [{"role": "user", "content": text}],
            "temperature": 0, "max_tokens": args.max_tokens, "enable_thinking": False,
        }
        request = Request(
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=1800) as response:
            json.load(response)

    for width in args.widths:
        for start in range(0, len(PROMPTS), width):
            with ThreadPoolExecutor(width) as pool:
                list(pool.map(one, PROMPTS[start : start + width]))


# --------------------------------------------------------------------------


def cmd_collect(args) -> None:
    out = Path(args.out)
    cmd = _server_cmd(args, "collect", [
        "--mtp-acceptance-log", str(out),
        "--mtp-acceptance-log-lookahead", str(args.lookahead),
    ])
    if args.dry_run:
        print(shlex.join(cmd))
        print(f"# then: {len(PROMPTS)} mixed prompts at widths {args.widths}, max_tokens {args.max_tokens}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    proc, log = _start(cmd, out.with_suffix(".server.log"), args.port, args.startup_timeout)
    try:
        _mixed_prompts(args)
        records = int(_status(args.port).get("scheduler", {}).get("mtp_acceptance_log_records", 0))
    finally:
        _stop(proc, log)
    if records <= 0:
        raise SystemExit("REFUSED: mtp_acceptance_log_records == 0 (logger never engaged)")
    print(json.dumps({"records": records, "log": str(out)}))


def cmd_ab(args) -> None:
    out_dir = Path(args.out)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    plan = [(pair, arm) for pair in range(args.pair_start, args.pair_start + args.pairs) for arm in arms]
    if args.dry_run:
        for pair, arm in plan:
            print(f"# pair {pair} arm {arm}")
            print(shlex.join(_server_cmd(args, arm, _arm_flags(args, arm))))
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    ledger = out_dir / "ab.jsonl"
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            if row["pair"] < args.pair_start:
                row["bench"] = json.loads((out_dir / row["bench_file"]).read_text())
                results.append(row)
    for pair, arm in plan:
        tag = f"{arm}-p{pair}"
        proc, log = _start(
            _server_cmd(args, arm, _arm_flags(args, arm)),
            out_dir / f"server-{tag}.log", args.port, args.startup_timeout,
        )
        try:
            bench = _drive(args, out_dir / f"bench-{tag}.json")
            scheduler = _status(args.port).get("scheduler", {})
        finally:
            _stop(proc, log)
        counter = _MECHANISM.get(arm)
        if counter and int(scheduler.get(counter, 0)) <= 0:
            raise SystemExit(f"REFUSED arm {arm}: mechanism counter {counter} == 0")
        routes = _assert_route(arm, bench)
        results.append({"pair": pair, "arm": arm, "bench": bench, "scheduler": scheduler, "routes": routes})
        with ledger.open("a") as handle:
            handle.write(json.dumps({"pair": pair, "arm": arm, "scheduler": scheduler, "routes": routes, "bench_file": f"bench-{tag}.json"}) + "\n")
    summary = _summarize(results, arms)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def _aggregate_tps(bench: dict) -> dict:
    """Extract aggregate tok/s per width from a benchmark_serving output."""
    return {
        int(width): float(row["median_aggregate_tokens_per_second"])
        for width, row in bench.get("summary", {}).items()
    }


def _summarize(results, arms) -> dict:
    by_arm: dict[str, dict[int, list[float]]] = {arm: {} for arm in arms}
    for item in results:
        for width, tps in _aggregate_tps(item["bench"]).items():
            by_arm[item["arm"]].setdefault(width, []).append(tps)
    medians = {
        arm: {w: statistics.median(v) for w, v in widths.items()}
        for arm, widths in by_arm.items()
    }
    base = "ewma" if "ewma" in medians else arms[0]
    deltas = {
        arm: {
            w: medians[arm][w] / medians[base][w] - 1.0
            for w in medians[arm] if w in medians.get(base, {})
        }
        for arm in medians if arm != base
    }
    vs_fixed = {}
    if "fixed" in medians:
        vs_fixed = {
            arm: {w: medians[arm][w] / medians["fixed"][w] - 1.0 for w in medians[arm] if w in medians["fixed"]}
            for arm in medians if arm != "fixed"
        }
    # Greedy width-1 outputs must be identical across arms (exact MTP).
    w1 = {}
    for item in results:
        for row in item["bench"].get("rows", []):
            if row["width"] == 1:
                for req in row["requests"]:
                    w1.setdefault(item["arm"], set()).add(req["output_sha256"])
    w1_identical = len({frozenset(v) for k, v in w1.items() if k != "ordinary"}) <= 1
    w1_matches_ordinary = (
        None if "ordinary" not in w1
        else all(v == w1["ordinary"] for v in w1.values())
    )
    counters = {}
    for item in results:
        sched = item.get("scheduler", {})
        counters.setdefault(item["arm"], []).append(
            {k: v for k, v in sched.items() if k.startswith(("adaptive_mtp_", "mtp_confidence", "mtp_acceptance"))}
            | {"routes": item.get("routes")}
        )
    return {
        "median_tps": medians, "delta_vs": base, "deltas": deltas, "deltas_vs_fixed": vs_fixed,
        "pairs": sorted({item["pair"] for item in results}),
        "width1_greedy_identical": w1_identical,
        "width1_matches_ordinary": w1_matches_ordinary,
        "width1_output_sets": {k: sorted(v) for k, v in w1.items()},
        "counters": counters, "samples": by_arm,
    }


def cmd_profile(args) -> None:
    lanes_list = list(args.widths)
    plan = {"model": args.model, "lanes": lanes_list, "depths": list(range(0, args.max_depth + 1)),
            "cycles": args.cycles, "warmup": args.warmup,
            "probe_overhead": bool(args.probe_overhead), "probe_reps": args.probe_reps,
            "cycle_reps": args.cycle_reps}
    if args.dry_run:
        print(json.dumps({"profile_plan": plan}))
        return
    import mlx.core as mx

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    adapter = resolve_adapter(args.model, mtp=True)(args.model)
    model, tokenizer = adapter.model, adapter.tokenizer
    # Long-form prompts only, so no lane finishes inside the timed window.
    prompts = [list(tokenizer.encode(PROMPTS[i])) for i in PROFILE_PROMPTS]
    table: dict[str, list[float]] = {}
    raw = []

    def time_cycles(lanes, depth, probe_log=None):
        gen = BatchGenerator(
            model, completion_batch_size=lanes, prefill_batch_size=lanes,
            prefill_step_size=2048,
            self_mtp={"num_draft": max(1, depth), "persistent": True, "rate_gate": False,
                      "segment_aware_live_tip": True, "segment_aware_cohort_size": lanes},
            adaptive_mtp_depth=({"current_depth": 0, "loss_rounds": 1 << 30, "gain_rounds": 1 << 30} if depth == 0 else None),
            mtp_acceptance_log=probe_log,
        )
        gen.insert(
            prompts[:lanes], max_tokens=[(args.warmup + args.cycles) * 8] * lanes,
            lane_rngs=[LaneRNG(i) for i in range(lanes)],
            self_mtp_configs=[{"sampling_temp": 0.0}] * lanes,
        )
        cycles = 0
        started = None
        while cycles < args.warmup + args.cycles:
            _, responses = gen.next()
            if not responses:
                continue
            if len({r.uid for r in responses}) != lanes and cycles >= args.warmup:
                raise SystemExit(f"lane count changed inside the timed window at lanes={lanes} depth={depth}")
            cycles += 1
            if cycles == args.warmup:
                mx.synchronize()
                started = time.perf_counter()
        mx.synchronize()
        seconds = (time.perf_counter() - started) / args.cycles
        stats = dict(getattr(gen, "scheduler_stats", {}) or {})
        gen.close()
        return seconds, stats

    # Interleaved repetitions, median per cell: a single sample mis-stated the
    # 27B depth-3 cost by 1.2x (see diag-27b-b1), and the controller's choice
    # follows this table directly.
    samples: dict[tuple[int, int], list[float]] = {}
    for rep in range(args.cycle_reps):
        for lanes in lanes_list:
            for depth in range(0, args.max_depth + 1):
                seconds, _ = time_cycles(lanes, depth)
                samples.setdefault((lanes, depth), []).append(seconds)
                raw.append({"rep": rep, "lanes": lanes, "depth": depth, "seconds_per_cycle": seconds})
                print(json.dumps(raw[-1]), flush=True)
    for lanes in lanes_list:
        table[str(lanes)] = [
            statistics.median(samples[(lanes, depth)]) for depth in range(0, args.max_depth + 1)
        ]
    # Probe overhead: depth max, probe off/on interleaved (off on off on),
    # probe = acceptance logger at lookahead 0 (the same per-position device
    # reductions and host pull the confidence controller pays each cycle).
    overhead = []
    if args.probe_overhead:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="rm10-probe-"))
        for lanes in lanes_list:
            off, on, fired = [], [], []
            for rep in range(args.probe_reps):
                off.append(time_cycles(lanes, args.max_depth)[0])
                seconds, stats = time_cycles(
                    lanes, args.max_depth,
                    {"path": str(tmp / f"probe-{lanes}-{rep}.jsonl"), "lookahead": 0},
                )
                on.append(seconds)
                fired.append(int(stats.get("mtp_confidence_feature_cycles", 0)))
            if min(fired) <= 0:
                raise SystemExit(f"REFUSED probe-overhead lanes={lanes}: mtp_confidence_feature_cycles == 0")
            item = {
                "lanes": lanes, "depth": args.max_depth,
                "off_seconds": off, "on_seconds": on, "feature_cycles": fired,
                "overhead": statistics.median(on) / statistics.median(off) - 1.0,
            }
            overhead.append(item)
            print(json.dumps(item), flush=True)
    unit = table[str(lanes_list[0])][0]
    # Schema v2 drops the `m_points`/`costs` pair that v1 carried.  Those were
    # a HARDCODED cold-NAX curve copied into every profile this script wrote,
    # sitting beside a genuinely measured `cycle_table`, so a v1 file looks
    # measured end to end and is not.  Two sessions read betas off it above
    # m = 4 before noticing.  Nothing on this tree consumes them
    # (`best_constant_depth.py` reads `cycle_table` only), so they are gone
    # rather than relabelled, and every array a v2 file does carry is named in
    # `provenance` with how it was obtained.
    profile = {
        "schema": "mlx2.mtp_verify_cost.v2",
        "draft_step_cost": args.draft_step_cost,
        "overhead": 0.0,
        "source": f"profile:{Path(args.model).name}:{time.strftime('%Y%m%d')}",
        "cycle_table": {k: [v / unit for v in row] for k, row in table.items()},
        "raw": raw,
        "probe_overhead": overhead,
        "provenance": {
            "cycle_table": (
                "measured: median BatchGenerator.next() seconds per (lanes, "
                f"depth) on {Path(args.model).name}, normalised to the "
                "depth-0 1-lane cycle"
            ),
            "raw": "measured: per-repeat cycle seconds behind cycle_table",
            "probe_overhead": (
                "measured: draft-feature probe off/on cycle seconds"
                if overhead
                else "not measured in this run"
            ),
            "draft_step_cost": "operator input: --draft-step-cost",
            "overhead": "constant 0.0",
        },
    }
    Path(args.out).write_text(json.dumps(profile, indent=1))


def cmd_diag(args) -> None:
    """Where does a dynamic-depth cycle's time go?  B=lanes, 1 lane default.

    Arms (interleaved, ``--probe-reps`` rounds):
    static2 / static3: fixed num_draft 2 / 3;
    ewma_pin2: num_draft 3 + EWMA controller pinned at depth 2.

    static2 vs ewma_pin2 is the cost of running a controller at all, against
    a static route of the depth it settles on.
    """
    lanes = args.widths[0]
    arms = {
        "static2": ({"num_draft": 2}, None),
        "static3": ({"num_draft": 3}, None),
        "ewma_pin2": ({"num_draft": 3}, {"current_depth": 2, "loss_rounds": 1 << 30, "gain_rounds": 1 << 30}),
    }
    if args.dry_run:
        print(json.dumps({"diag_arms": list(arms), "lanes": lanes, "reps": args.probe_reps}))
        return
    import mlx.core as mx

    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG

    adapter = resolve_adapter(args.model, mtp=True)(args.model)
    model, tokenizer = adapter.model, adapter.tokenizer
    prompts = [list(tokenizer.encode(PROMPTS[i])) for i in PROFILE_PROMPTS]
    out = []
    for rep in range(args.probe_reps):
        for name, (mtp, adaptive) in arms.items():
            gen = BatchGenerator(
                model, completion_batch_size=lanes, prefill_batch_size=lanes, prefill_step_size=2048,
                self_mtp={**mtp, "persistent": True, "rate_gate": False,
                          "segment_aware_live_tip": True, "segment_aware_cohort_size": lanes},
                adaptive_mtp_depth=adaptive,
            )
            gen.insert(prompts[:lanes], max_tokens=[(args.warmup + args.cycles) * 8] * lanes,
                       lane_rngs=[LaneRNG(i) for i in range(lanes)],
                       self_mtp_configs=[{"sampling_temp": 0.0}] * lanes)
            cycles, tokens, started = 0, 0, None
            while cycles < args.warmup + args.cycles:
                _, responses = gen.next()
                if not responses:
                    continue
                cycles += 1
                if cycles > args.warmup:
                    tokens += len(responses)
                if cycles == args.warmup:
                    mx.synchronize()
                    started = time.perf_counter()
            mx.synchronize()
            seconds = time.perf_counter() - started
            stats = {k: v for k, v in gen.scheduler_stats.items() if "mtp" in k}
            gen.close()
            item = {"rep": rep, "arm": name, "lanes": lanes, "ms_per_cycle": 1000 * seconds / args.cycles,
                    "tokens_per_cycle": tokens / args.cycles / lanes, "tok_s": tokens / seconds, "stats": stats}
            out.append(item)
            print(json.dumps(item), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("profile", "collect", "ab", "diag"))
    parser.add_argument("--i-own-the-gpu", action="store_true", help="required for any Metal/model execution")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no MLX import, no server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--execution-policy", type=Path, help="adapter policy JSON, e.g. {\"num_draft\": 3}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=8285)
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--lookahead", type=int, default=2,
                        help="collect: unverified greedy lookahead drafts per cycle, so the "
                             "acceptance log observes positions past the served depth")
    parser.add_argument("--arms", default="ordinary,fixed,ewma",
                        help="ab: comma-separated subset of ordinary,fixed,ewma")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--draft-step-cost", type=float, default=0.06)
    parser.add_argument("--probe-overhead", action="store_true", help="profile: also time depth-max with the probe on vs off")
    parser.add_argument("--probe-reps", type=int, default=2)
    parser.add_argument("--cycle-reps", type=int, default=3, help="profile: interleaved repetitions per (lanes, depth) cell; the table takes the median")
    parser.add_argument("--pair-start", type=int, default=0, help="ab: first pair index; earlier pairs are read back from <out>/ab.jsonl")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and not args.i_own_the_gpu:
        parser.error("refusing Metal/model execution without --i-own-the-gpu (or pass --dry-run)")
    if any(w not in (1, 2, 4) for w in args.widths):
        parser.error("widths must be drawn from 1, 2, 4")
    {"profile": cmd_profile, "collect": cmd_collect, "ab": cmd_ab, "diag": cmd_diag}[args.command](args)


if __name__ == "__main__":
    main()
