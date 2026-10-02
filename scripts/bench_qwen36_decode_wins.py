"""One-load full-model interleaved decode A/B on real prompts.

Default matrix: B1/4/16, native MTP off/on. Records token hashes, paired
rates, per-step time and engagement, never turns a non-engaging arm into a win.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

ARMS = (
    "off",
    "gate_up",
    "routed",
    "shared",
    "topk",
    "gdn_batch",
    "gdn_verify",
    "window",
    "all",
    "old_candidate",
    "candidate_views",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS[:9]))
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["ordinary:1", "mtp:1", "ordinary:4", "mtp:4", "ordinary:16", "mtp:16"],
    )
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--num-draft", type=int, default=2)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")
    if "off" not in a.arms:
        ap.error("an off reference arm is required")
    if min(a.context, a.gen, a.reps, a.num_draft) < 1:
        ap.error("context/gen/reps/num-draft must be positive")
    configs = []
    for item in a.configs:
        route, rows = item.split(":")
        rows = int(rows)
        if route not in ("ordinary", "mtp") or rows not in (1, 4, 16):
            ap.error("configs require ordinary/mtp:1/4/16")
        configs.append((route, rows))
    contents = Path(a.prompt_file).read_text()
    prompts = json.loads(contents) if contents.lstrip().startswith("[") else [contents]
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(p, str) or not p.strip() for p in prompts)
    ):
        ap.error(
            "prompt file must contain real text or a nonempty JSON list of strings"
        )
    import mlx.core as mx

    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter

    adapter = Qwen3635BA3BAdapter(
        a.model,
        require_mtp=any(r == "mtp" for r, _ in configs),
        execution_policy={"eager_dispatch_stride": 0, "num_draft": a.num_draft},
    )
    from mlx2.runtime import generate as G
    from mlx2.runtime.models import qwen4_routed_decode as RD
    from mlx2.runtime.models import qwen36_35b as Q
    from mlx2.runtime.models import qwen36_moe_decode as M
    from mlx2.runtime.sample_utils import LaneRNG

    blocks = [
        m
        for _, m in adapter.model.named_modules()
        if isinstance(m, M.Qwen36SparseMoeBlock)
    ]
    gdns = [
        m for _, m in adapter.model.named_modules() if isinstance(m, Q.GatedDeltaNet)
    ]
    encoded = [list(adapter.tokenizer.encode(p))[: a.context] for p in prompts]
    if any(not p for p in encoded):
        raise ValueError("empty tokenized prompt")
    original_gate, original_down = (
        RD.candidate_gate_up_swiglu,
        RD.candidate_down_combine,
    )
    record = {
        "artifact": adapter.identity,
        "mlx": mx.__version__,
        "prompt_sha256": hashlib.sha256(contents.encode()).hexdigest(),
        "arms": a.arms,
        "configs": a.configs,
        "eager_dispatch_stride": 0,
        "num_draft": a.num_draft,
        "qualification": "unqualified",
        "runs": [],
    }
    root = Path(__file__).resolve().parents[1]
    record["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *root.glob("src/mlx2/runtime/models/qwen36*.py"),
            root / "src/mlx2/runtime/models/qwen4_fused_gdn_verify.py",
            root / "src/mlx2/runtime/models/qwen4_routed_decode.py",
            root / "src/mlx2/runtime/models/qwen4_moe_window.py",
        ]
    }

    def select(arm):
        RD.candidate_gate_up_swiglu = original_gate
        RD.candidate_down_combine = original_down
        if arm == "candidate_views":
            RD.candidate_gate_up_swiglu = lambda *args: original_gate(*args, views=True)
            RD.candidate_down_combine = lambda *args: original_down(*args, views=True)
        for block in blocks:
            block.set_moe_routed_candidate_mode("off")
            block.set_fused_expert_kernel_mode("stock")
            mode = {
                "gate_up": "gate_up",
                "routed": "gate_up_down",
                "shared": "gate_up_down_shared",
                "all": "gate_up_down_shared",
            }.get(arm, "off")
            block.set_moe_routed_decode_mode(mode)
            block.set_moe_topk_mode("launch" if arm in ("topk", "all") else "off")
            block.set_moe_window_consumers(
                ("batch_decode", "verify", "row_exact")
                if arm in ("window", "all")
                else ()
            )
            if arm in ("old_candidate", "candidate_views"):
                block.set_moe_routed_candidate_mode("two_launch")
        for layer in gdns:
            layer.set_fused_gdn_decode_mode(
                "fused" if arm in ("gdn_batch", "all") else "stock"
            )
            layer.set_fused_gdn_batch_decode_mode(
                "row_exact" if arm in ("gdn_batch", "all") else "off"
            )
            layer.set_fused_gdn_verify_mode(
                "row_exact" if arm in ("gdn_verify", "all") else "off"
            )
            layer.set_fused_gdn_batch_verify_mode(
                "row_exact" if arm in ("gdn_verify", "all") else "off"
            )

    def counters():
        from mlx2.runtime.models.qwen3_next import routed_candidate_stats

        out = Q.qwen36_decode_wins_stats(adapter.model, reset=True)
        out["b1_gdn"] = Q.qwen36_fused_gdn_stats(adapter.model, reset=True)
        out["old_candidate"] = routed_candidate_stats(adapter.model, reset=True)
        return out

    def run(route, rows, rep, arm, warm=False):
        select(arm)
        counters()
        kwargs = (
            {}
            if route == "ordinary"
            else {
                "self_mtp": {
                    "num_draft": a.num_draft,
                    "persistent": True,
                    "rate_gate": False,
                    "prefill_step_size": 1024,
                }
            }
        )
        gen = G.BatchGenerator(
            adapter.model,
            completion_batch_size=rows,
            prefill_batch_size=1,
            prefill_step_size=1024,
            **kwargs,
        )
        pp = [encoded[(rep + i) % len(encoded)] for i in range(rows)]
        insert = {
            "max_tokens": [a.gen] * rows,
            "lane_rngs": [LaneRNG(i + 1) for i in range(rows)],
        }
        if route == "mtp":
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}] * rows
        uids = gen.insert(pp, **insert)
        done = set()
        started = set()
        tokens = {uid: [] for uid in uids}
        emitted = 0
        steps = 0
        t0 = None
        t1 = None
        full_start = time.perf_counter()
        try:
            while len(done) < rows:
                _, responses = gen.next()
                now = time.perf_counter()
                if t0 is not None:
                    steps += 1
                    emitted += len(responses)
                    t1 = now
                for response in responses:
                    started.add(response.uid)
                    tokens[response.uid].append(int(response.token))
                    if response.finish_reason:
                        done.add(response.uid)
                if t0 is None and started == set(uids):
                    t0 = now
                    t1 = now
        finally:
            gen.close()
        duration = (t1 - t0) if t0 is not None else 0
        if duration <= 0 or emitted < 1:
            raise ValueError("no steady decode interval (EOS or short generation)")
        stats = counters()
        result = {
            "route": route,
            "rows": rows,
            "rep": rep,
            "arm": arm,
            "warmup": warm,
            "decode_tps": emitted / duration,
            "ms_per_step": duration * 1000 / max(steps, 1),
            "decode_steps": steps,
            "full_seconds": time.perf_counter() - full_start,
            "counters": stats,
            "tokens_sha256": hashlib.sha256(
                json.dumps([tokens[u] for u in uids]).encode()
            ).hexdigest(),
            "prompt_tokens": [len(p) for p in pp],
        }
        result["observed_used"] = bool(
            stats["moe"]["calls"]
            or stats["moe"]["routed_gate_up_calls"]
            or stats["moe"]["topk_launch_calls"]
            or stats["b1_gdn"]["fused_calls"]
            or stats["old_candidate"]["calls"]
            or any(v["calls"] for v in stats["gdn"].values())
        )
        mx.clear_cache()
        return result

    try:
        for route, rows in configs:
            for arm in a.arms:
                run(route, rows, 0, arm, warm=True)
            for rep in range(a.reps):
                order = a.arms[rep % len(a.arms) :] + a.arms[: rep % len(a.arms)]
                if rep % 2:
                    order = list(reversed(order))
                for arm in order:
                    item = run(route, rows, rep, arm)
                    record["runs"].append(item)
                    Path(a.out).write_text(json.dumps(record, indent=2) + "\n")
                    print(
                        route,
                        rows,
                        rep,
                        arm,
                        round(item["decode_tps"], 2),
                        item["observed_used"],
                        flush=True,
                    )
        summary = {}
        for route, rows in configs:
            cells = [
                r for r in record["runs"] if r["route"] == route and r["rows"] == rows
            ]
            baseline = {r["rep"]: r for r in cells if r["arm"] == "off"}
            for arm in a.arms:
                rr = [r for r in cells if r["arm"] == arm]
                summary[f"{route}:{rows}/{arm}"] = {
                    "median_tps": statistics.median(r["decode_tps"] for r in rr),
                    "paired_delta_pct": [
                        100 * (r["decode_tps"] / baseline[r["rep"]]["decode_tps"] - 1)
                        for r in rr
                    ],
                    "tokens_identical": all(
                        r["tokens_sha256"] == baseline[r["rep"]]["tokens_sha256"]
                        for r in rr
                    ),
                    "observed_used": all(r["observed_used"] for r in rr),
                }
        record["summary"] = summary
        record["verdict"] = "measured"
    except Exception as exc:
        record["verdict"] = "fail"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        select("off")
        adapter.close()
        Path(a.out).write_text(json.dumps(record, indent=2) + "\n")
    return 0 if record["verdict"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
