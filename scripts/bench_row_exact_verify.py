#!/usr/bin/env python3
"""Cost of the Flash-Next row-exact verify route: in-process, interleaved A/B.

One process, one loaded model, greedy B=1 decode.  Arms:

* ``off``       ordinary one-token decode (MTP off)
* ``mtp``       native self-MTP, route copy drafts, stock verify
* ``rowexact``  the same with the row-exact verify route enabled, with the
                W2-B window kernels (attention window + HC row-exact mode)
                switched on in-process
* ``rowexact_base``  (opt-in) the route without the W2-B window kernels

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
# Optional arms (--arms): "mtp_window" (stock MTP with the plain-verify MoE
# row window), "rowexact_window" (row-exact verify with its MoE row window);
# any arm may carry ":launch" or ":fold" (router top-k mode in the window and
# at one token).  "rowexact" and "rowexact_window" also switch on the W2-B
# attention window and HC row-exact mode; "rowexact_base" leaves them off.


def _set_window_stages(on: bool):
    from mlx2.runtime.models import qwen4_attn_window as AW
    from mlx2.runtime.models import qwen4_hc_decode as HCD

    AW.set_enabled(on)
    HCD.set_hc_row_exact_enabled(on)


def _configure_moe(adapter, arm):
    from mlx2.runtime.models.qwen3_next import Qwen3NextSparseMoeBlock

    name, *opts = arm.split(":")
    consumers = {"mtp_window": {"verify"}, "rowexact_window": {"row_exact"}}.get(name, set())
    topk = "fold" if "fold" in opts else ("launch" if "launch" in opts else "off")
    for _, module in adapter.model.named_modules():
        if isinstance(module, Qwen3NextSparseMoeBlock):
            module.set_moe_window_consumers(consumers)
            module.set_moe_topk_mode(topk)
    return name


def _moe_window_calls(adapter):
    from mlx2.runtime.models.qwen3_next import Qwen3NextSparseMoeBlock

    total = {}
    for _, module in adapter.model.named_modules():
        if isinstance(module, Qwen3NextSparseMoeBlock):
            for key, value in module.moe_window_calls.items():
                total[key] = total.get(key, 0) + value
            total["fallbacks"] = total.get("fallbacks", 0) + sum(module.moe_window_fallbacks.values())
    return total


def _decode(adapter, prompt_ids, *, arm, max_tokens, prefill_step, copy_policy):
    import mlx.core as mx
    from mlx2.runtime import generate as G
    from mlx2.runtime.sample_utils import LaneRNG

    name = _configure_moe(adapter, arm)
    kwargs = dict(completion_batch_size=1, prefill_batch_size=1, prefill_step_size=prefill_step)
    if name != "off":
        kwargs["self_mtp"] = adapter.policy.batch_config(max_lanes=1, prefill_step=prefill_step)
        if copy_policy is not None:
            kwargs["copy_draft"] = copy_policy
    adapter.row_exact_verify.enable(name.startswith("rowexact"))
    _set_window_stages(name in ("rowexact", "rowexact_window"))
    gen = G.BatchGenerator(adapter.model, **kwargs)
    tokens, first, last = [], None, None
    try:
        insert = dict(max_tokens=[max_tokens], lane_rngs=[LaneRNG(1)])
        if name != "off":
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
        _configure_moe(adapter, "off")
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
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="comma list; also mtp_window, rowexact_window, with :launch/:fold")
    parser.add_argument("--policy", "--execution-policy", dest="policy", default=None,
                        help="JSON execution policy for the adapter")
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
    arms = tuple(a for a in args.arms.split(",") if a)
    if "off" not in arms or "mtp" not in arms:
        parser.error("--arms must include off and mtp (the ratios are taken against them)")
    adapter = FlashNextAdapter(args.model, execution_policy=json.loads(args.policy) if args.policy else None)
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
    window_calls = {name: {arm: [] for arm in arms} for name in names}
    order_log = []
    n = len(arms)
    for rep in range(args.reps + 1):
        for name in names:
            order = arms[rep % n :] + arms[: rep % n]
            if rep % 2 and n > 2:
                order = order[::-1]
            order_log.append([rep, name, list(order)])
            for arm in order:
                before = _moe_window_calls(adapter)
                result = _decode(
                    adapter, prompt_ids[name], arm=arm, max_tokens=args.max_tokens,
                    prefill_step=args.prefill_step, copy_policy=copy_policy,
                )
                after = _moe_window_calls(adapter)
                window_calls[name][arm].append({k: after[k] - before.get(k, 0) for k in after})
                shas[name][arm].add(result["sha"])
                if rep:
                    samples[name][arm].append(result["decode_tps"])
                print(rep, name, arm, f"{result['decode_tps']:.1f} tok/s", result["sha"],
                      json.dumps(window_calls[name][arm][-1]), flush=True)
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
                "output_equals_off": shas[name][arm] == shas[name]["off"],
                "output_equals_mtp": shas[name][arm] == shas[name]["mtp"],
                "moe_window_calls": window_calls[name][arm][-1],
            }
        for arm in arms:
            if arm != "off":
                cell[f"{arm}_over_off"] = cell[arm]["median"] / cell["off"]["median"]
            if arm not in ("off", "mtp"):
                cell[f"{arm}_over_mtp"] = cell[arm]["median"] / cell["mtp"]["median"]
        if "rowexact" in arms:
            cell["rowexact_output_equals_off"] = shas[name]["rowexact"] == shas[name]["off"]
        summary[name] = cell
    report = {
        "schema": "mlx2.bench-row-exact-verify.v1",
        "mlx": mx.__version__,
        "reps": args.reps,
        "max_tokens": args.max_tokens,
        "arms": list(arms),
        "policy": adapter.policy.as_dict(),
        "order": order_log,
        "route_status": adapter.row_exact_verify.status(),
        "summary": summary,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({n: {k: (v if not isinstance(v, dict) else v["median"]) for k, v in c.items()} for n, c in summary.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
