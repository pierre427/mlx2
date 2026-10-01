#!/usr/bin/env python3
"""Cost of the Flash-Next row-exact verify route: in-process, interleaved A/B.

One process, one loaded model, greedy B=1 decode.  Arms:

* ``off``       ordinary one-token decode (MTP off)
* ``mtp``       native self-MTP, route copy drafts, stock verify
* ``rowexact``  the same with the row-exact verify route enabled, with the
                wave-2 window kernels (policy ``row_exact_window_kernels``:
                attention window + HC row-exact mode) switched on in-process
* ``rowexact_base``  (opt-in, ``--arms``) the route without those window
                kernels (the wave-1 route), to isolate their increment

Every rep visits all arms in a rotated order (ABC, BCA, CAB, ...); rep 0 is a
discarded warm-up.  Decode tok/s is measured from the first generated token to
the last, so prefill is excluded.  Reports median and min..max per arm and
prompt, and the rowexact/mtp and rowexact/off ratios of medians.  Output
tokens are hashed per arm so a run also shows whether rowexact reproduced
``off`` (it should, byte for byte).

  scratchpad/gpuq.sh l5-bench env PYTHONPATH=src MLX_ENABLE_TF32=0 \\
      .venv/bin/python scripts/bench_row_exact_verify.py --i-own-the-gpu --out bench.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ARMS = ("off", "mtp", "rowexact")


def _set_window_stages(on: bool):
    from mlx2.runtime.models import qwen4_attn_window as AW
    from mlx2.runtime.models import qwen4_hc_decode as HCD

    AW.set_enabled(on)
    HCD.set_hc_row_exact_enabled(on)


def _decode(adapter, prompt_ids, *, arm, max_tokens, prefill_step, copy_policy):
    import mlx.core as mx
    from mlx2.runtime import generate as G
    from mlx2.runtime.sample_utils import LaneRNG

    kwargs = dict(completion_batch_size=1, prefill_batch_size=1, prefill_step_size=prefill_step)
    if arm != "off":
        kwargs["self_mtp"] = adapter.policy.batch_config(max_lanes=1, prefill_step=prefill_step)
        if copy_policy is not None:
            kwargs["copy_draft"] = copy_policy
    adapter.row_exact_verify.enable(arm.startswith("rowexact"))
    _set_window_stages(arm == "rowexact")
    gen = G.BatchGenerator(adapter.model, **kwargs)
    tokens, first, last = [], None, None
    try:
        insert = dict(max_tokens=[max_tokens], lane_rngs=[LaneRNG(1)])
        if arm != "off":
            insert["self_mtp_configs"] = [{"sampling_temp": 0.0}]
        gen.insert([list(prompt_ids)], **insert)
        done = False
        while not done:
            _p, responses = gen.next()
            now = time.perf_counter()
            for response in responses:
                tokens.append(int(response.token))
                done = done or bool(response.finish_reason)
            if responses:
                if first is None:
                    first = now
                    first_count = len(tokens)
                last = now
    finally:
        gen.close()
        adapter.row_exact_verify.enable(False)
        _set_window_stages(False)
    mx.clear_cache()
    decoded = len(tokens) - first_count
    return {
        "tokens": len(tokens),
        "decode_tps": decoded / (last - first) if last > first else 0.0,
        "sha": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(Path("~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP").expanduser()))
    parser.add_argument("--prompts", default="prose,copy")
    parser.add_argument("--reps", type=int, default=4, help="measured reps (plus one warm-up)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--no-copy", action="store_true")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--execution-policy", default=None, help="JSON FlashNextPolicy mapping")
    parser.add_argument("--out", required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("Metal run: pass --i-own-the-gpu under the GPU lock")

    import mlx.core as mx
    from check_mtp_row_exact import PROMPTS
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.runtime.models.qwen4_row_exact import install

    mx.set_cache_limit(4 << 30)
    policy = json.loads(args.execution_policy) if args.execution_policy else None
    adapter = FlashNextAdapter(args.model, execution_policy=policy)
    arms = tuple(a for a in args.arms.split(",") if a)
    assert {"off", "mtp", "rowexact"} <= set(arms), arms
    adapter.row_exact_verify = install(adapter.model)
    copy_policy = None
    if not args.no_copy:
        copy_policy = adapter.default_route_execution_policy["native_mtp"]["self_mtp_copy_draft"]
    names = [p for p in args.prompts.split(",") if p]
    prompt_ids = {
        name: list(adapter.prompt_tokens({"messages": [{"role": "user", "content": PROMPTS[name]}]}))
        for name in names
    }
    samples = {name: {arm: [] for arm in arms} for name in names}
    shas = {name: {arm: set() for arm in arms} for name in names}
    order_log = []
    for rep in range(args.reps + 1):
        for name in names:
            order = arms[rep % len(arms) :] + arms[: rep % len(arms)]
            order_log.append([rep, name, list(order)])
            for arm in order:
                result = _decode(
                    adapter, prompt_ids[name], arm=arm, max_tokens=args.max_tokens,
                    prefill_step=args.prefill_step, copy_policy=copy_policy,
                )
                shas[name][arm].add(result["sha"])
                if rep:
                    samples[name][arm].append(result["decode_tps"])
                print(rep, name, arm, f"{result['decode_tps']:.1f} tok/s", result["sha"], flush=True)
    summary = {}
    for name in names:
        cell = {}
        for arm in arms:
            values = samples[name][arm]
            cell[arm] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "samples": values,
                "token_sha": sorted(shas[name][arm]),
            }
        cell["rowexact_over_mtp"] = cell["rowexact"]["median"] / cell["mtp"]["median"]
        cell["rowexact_over_off"] = cell["rowexact"]["median"] / cell["off"]["median"]
        cell["mtp_over_off"] = cell["mtp"]["median"] / cell["off"]["median"]
        cell["rowexact_output_equals_off"] = shas[name]["rowexact"] == shas[name]["off"]
        if "rowexact_base" in arms:
            cell["rowexact_over_base"] = cell["rowexact"]["median"] / cell["rowexact_base"]["median"]
            cell["rowexact_base_over_mtp"] = cell["rowexact_base"]["median"] / cell["mtp"]["median"]
            cell["rowexact_base_output_equals_off"] = shas[name]["rowexact_base"] == shas[name]["off"]
        summary[name] = cell
    report = {
        "schema": "mlx2.bench-row-exact-verify.v1",
        "mlx": mx.__version__,
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "order": order_log,
        "policy": adapter.policy.as_dict(),
        "route_status": adapter.row_exact_verify.status(),
        "summary": summary,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({n: {k: (v if not isinstance(v, dict) else v["median"]) for k, v in c.items()} for n, c in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
