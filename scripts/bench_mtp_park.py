"""One-load, in-process A/B of MTP->ordinary handoff policies.

Arms (``--arms``):
  ordinary        no MTP (reference rate);
  static:N        native MTP with the static handoff at ``max_mtp_width`` N;
  adaptive:N      native MTP with the measured park (oMLX #4112 port), cold-
                  start threshold N.

The BatchGenerator is built the way ``scripts/bench_qwen36_decode_wins.py``
builds it (``adapter.execution_config(max_lanes=rows, prefill_step=1024)``,
``completion_batch_size=rows``, ``prefill_batch_size=1``, rate from the first
emitted token).  A fresh generator per wave, as in that script; the adaptive
arm's park memory is keyed by model and route, so it outlives the generator,
as it outlives a cohort in serving.  Each lane count runs one warm-up wave per
arm (recorded, not summarised), then ``--reps`` waves per arm in rotated,
alternated order.  Records per-lane token hashes, handoff events/reasons,
adaptive-park counters and the memory snapshot.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def _parse_arm(text):
    if text == "ordinary":
        return ("ordinary", None)
    kind, _, width = text.partition(":")
    if kind not in ("static", "adaptive") or not width.isdigit() or int(width) < 1:
        raise argparse.ArgumentTypeError(f"bad arm {text!r}")
    return (kind, int(width))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--lanes", nargs="+", type=int, required=True)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--execution-policy", default=None, help="JSON object")
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    arms = [_parse_arm(x) for x in a.arms]
    labels = list(a.arms)
    if min(a.lanes) < 1 or min(a.gen, a.reps, a.context) < 1:
        ap.error("lanes/gen/reps/context must be positive")
    prompts = json.loads(Path(a.prompt_file).read_text())
    if not isinstance(prompts, list) or not prompts or not all(
        isinstance(p, str) and p.strip() for p in prompts
    ):
        ap.error("prompt file must be a nonempty JSON list of strings")
    policy = None if a.execution_policy is None else json.loads(a.execution_policy)

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime import generate as G
    from mlx2.runtime.adaptive_policy import MTPOrdinaryHandoffPolicy
    from mlx2.runtime.sample_utils import LaneRNG

    factory = resolve_adapter(a.model, mtp=True, qualification_mode=True)
    kwargs = {} if policy is None else {"execution_policy": policy}
    adapter = factory(a.model, **kwargs)
    tok = adapter.tokenizer

    def encode(text):
        if a.chat_template:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
            )
            if isinstance(ids, str):
                ids = tok.encode(ids, add_special_tokens=False)
            elif not isinstance(ids, list):
                ids = ids["input_ids"]
        else:
            ids = tok.encode(text)
        return list(ids)[: a.context]

    encoded = [encode(p) for p in prompts]
    root = Path(__file__).resolve().parents[1]
    record = {
        "schema": "mlx2.bench-mtp-park.v1",
        "artifact": (
            {k: v for k, v in adapter.identity.items() if k != "files"}
            if isinstance(getattr(adapter, "identity", None), dict)
            else a.model
        ),
        "adapter": factory.__name__,
        "mlx": mx.__version__,
        "prompt_sha256": hashlib.sha256(
            Path(a.prompt_file).read_bytes()
        ).hexdigest(),
        "arms": labels,
        "lanes": a.lanes,
        "gen": a.gen,
        "reps": a.reps,
        "execution_policy": policy,
        "chat_template": a.chat_template,
        "qualification": "unqualified",
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                root / "src/mlx2/runtime/generate.py",
                root / "src/mlx2/runtime/mtp_park_memory.py",
                root / "src/mlx2/runtime/adaptive_policy.py",
                Path(__file__).resolve(),
            )
        },
        "runs": [],
        "park_memory": {},
    }

    def run(label, kind, width, rows, rep, warm):
        gen_kwargs = {}
        if kind != "ordinary":
            handoff = {"enabled": True, "max_mtp_width": width}
            if kind == "adaptive":
                handoff["adaptive_park"] = {"enabled": True}
            gen_kwargs["self_mtp"] = adapter.execution_config(
                max_lanes=rows, prefill_step=1024
            )
            gen_kwargs["mtp_ordinary_handoff"] = MTPOrdinaryHandoffPolicy.from_value(
                handoff
            )
        gen = G.BatchGenerator(
            adapter.model,
            completion_batch_size=rows,
            prefill_batch_size=1,
            prefill_step_size=1024,
            **gen_kwargs,
        )
        pp = [encoded[(rep * rows + i) % len(encoded)] for i in range(rows)]
        insert = {
            "max_tokens": [a.gen] * rows,
            "lane_rngs": [LaneRNG(i + 1) for i in range(rows)],
        }
        if kind != "ordinary":
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}] * rows
        uids = gen.insert(pp, **insert)
        tokens = {uid: [] for uid in uids}
        done, failures = set(), []
        emitted = steps = 0
        t0 = t1 = None
        full_start = time.perf_counter()
        try:
            while len(done) < rows:
                _, responses = gen.next()
                for failure in gen.take_lane_failures():
                    failures.append(failure)
                    done.add(failure["uid"])
                now = time.perf_counter()
                if t0 is not None:
                    steps += 1
                    emitted += len(responses)
                    t1 = now
                for r in responses:
                    tokens[r.uid].append(int(r.token))
                    if r.finish_reason:
                        done.add(r.uid)
                if t0 is None and responses:
                    t0 = t1 = now
            stats = dict(gen.scheduler_stats)
            snapshot = gen.mtp_park_memory_snapshot()
        finally:
            gen.close()
        if failures:
            raise ValueError(f"{label}@{rows}: lane failures {failures}")
        duration = (t1 - t0) if t0 is not None else 0.0
        if duration <= 0 or emitted < 1:
            raise ValueError("no steady decode interval")
        item = {
            "arm": label,
            "rows": rows,
            "rep": rep,
            "warmup": warm,
            "decode_tps": emitted / duration,
            "ms_per_step": duration * 1000 / max(steps, 1),
            "full_seconds": time.perf_counter() - full_start,
            "lane_tokens": [len(tokens[u]) for u in uids],
            "lane_sha256": [
                hashlib.sha256(json.dumps(tokens[u]).encode()).hexdigest()[:16]
                for u in uids
            ],
            "handoff": {
                k: v for k, v in stats.items() if k.startswith("mtp_ordinary_handoff")
            },
            "adaptive_park": {
                k: v for k, v in stats.items() if k.startswith("mtp_adaptive_park")
            },
        }
        if snapshot is not None:
            record["park_memory"][f"{label}@{rows}"] = snapshot
        mx.clear_cache()
        return item

    try:
        for rows in a.lanes:
            for label, (kind, width) in zip(labels, arms):
                item = run(label, kind, width, rows, 0, True)
                record["runs"].append(item)
                print("warm", rows, label, round(item["decode_tps"], 1), flush=True)
            for rep in range(a.reps):
                order = list(range(len(labels)))
                order = order[rep % len(order):] + order[: rep % len(order)]
                if rep % 2:
                    order.reverse()
                for i in order:
                    kind, width = arms[i]
                    item = run(labels[i], kind, width, rows, rep, False)
                    record["runs"].append(item)
                    Path(a.out).write_text(json.dumps(record, indent=2) + "\n")
                    print(
                        rows, rep, labels[i], round(item["decode_tps"], 1),
                        item["handoff"].get("mtp_ordinary_handoff_events", 0),
                        flush=True,
                    )
        summary = {}
        base = labels[0]
        for rows in a.lanes:
            cells = [r for r in record["runs"] if r["rows"] == rows and not r["warmup"]]
            ref = {r["rep"]: r for r in cells if r["arm"] == base}
            for label in labels:
                rr = [r for r in cells if r["arm"] == label]
                same = sum(
                    x == y
                    for r in rr
                    for x, y in zip(r["lane_sha256"], ref[r["rep"]]["lane_sha256"])
                )
                summary[f"{label}@{rows}"] = {
                    "median_tps": statistics.median(r["decode_tps"] for r in rr),
                    "tps": [round(r["decode_tps"], 2) for r in rr],
                    f"paired_delta_pct_vs_{base}": [
                        round(
                            100 * (r["decode_tps"] / ref[r["rep"]]["decode_tps"] - 1), 2
                        )
                        for r in rr
                    ],
                    f"lanes_identical_to_{base}": f"{same}/{rows * len(rr)}",
                    "handoff_events": [
                        r["handoff"].get("mtp_ordinary_handoff_events", 0) for r in rr
                    ],
                    "handoff_reasons": sorted(
                        {
                            k.removeprefix("mtp_ordinary_handoff_")
                            for r in rr
                            for k in r["handoff"]
                            if k not in (
                                "mtp_ordinary_handoff_events",
                                "mtp_ordinary_handoff_lanes",
                            )
                        }
                    ),
                }
        record["summary"] = summary
        record["verdict"] = "measured"
    except Exception as exc:  # recorded, never a silent win
        record["verdict"] = "fail"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()
        Path(a.out).write_text(json.dumps(record, indent=2) + "\n")
    return 0 if record["verdict"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
