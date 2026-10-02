#!/usr/bin/env python3
"""Flash-Next options sweep: functional smoke per option, then in-process A/B.

Phase ``smoke`` (one process per arm, ``--arm NAME``): construct the adapter
the way the server does for the served policy (``num_draft`` 2, adaptive depth
and handoff off, the adapter's route defaults: copy drafts on) plus the arm's
overrides, run a fixed set of real workloads, and record per workload the
greedy tokens (sha + full lists), decoded text, the non-zero deltas of every
numeric leaf of ``adapter.diagnostics()`` and of the generator's
``scheduler_stats`` (engagement), wall time, and any exception.  The arm's
receipt label (``policy.as_dict()``) and environment delta are recorded too.
``--phase compare`` diffs every smoke JSON against the default arm's.

Phase ``perf`` (one process, one model load): arms toggled in-process
(``PERF_TOGGLES``), arms rotated per rep (odd reps reversed), one warm-up rep
discarded, content-varied prompts per rep.  Decode tok/s is measured from the
first token to the last (prefill excluded) and paired per rep against the
default arm.

Workloads (smoke):
  b1_ord    B=1 ordinary decode, 4 chat prompts x 96 tokens
  b1_mtp    B=1 native MTP (served config), same prompts
  b8_mtp    8 lanes native MTP, 8 chat prompts x 48 tokens
  b4_ord    4 lanes ordinary decode, 4 prompts x 48 tokens
  long_mtp  ~20K-token document prompt (docs/*.md), 32 tokens, native MTP

  gpuq.sh smoke-default env PYTHONPATH=src MLX_ENABLE_TF32=0 \\
      .venv/bin/python scripts/flash_next_options_sweep.py --phase smoke \\
      --arm default --out smoke/default.json --i-own-the-gpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = str(Path("~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP").expanduser())
# The served policy file (policies/flash-next-uncensored-policy.json).  The
# engine strips adaptive_mtp_depth / mtp_ordinary_handoff before the adapter.
SERVED_ADAPTER_POLICY = {"num_draft": 2}
HANDOFF = {"enabled": True, "max_mtp_width": 4}
FLY = {"enabled": True, "entropy_threshold": 2.0, "window": 2, "min_prob": 0.01}
DEPTH_BUDGET = 8192 * 16384

# name -> {"policy": adapter policy overrides, "engine": generator options,
#          "post": in-process lever applied after load (env-only switches)}
ARMS = {
    "default": {},
    "default_rerun": {},  # cross-process determinism control
    # --- FlashNextPolicy fields (non-default values) ---
    "row_exact_verify": {"policy": {"row_exact_verify": True}},
    "moe_window_batch_decode": {"policy": {"moe_window_batch_decode": True}},
    "moe_window_verify": {"policy": {"moe_window_verify": True}},
    "moe_topk_fold": {"policy": {"moe_topk_fold": "fold"}},
    "moe_topk_off": {"policy": {"moe_topk_fold": "off"}},
    "routed_gate_up": {"policy": {"moe_routed_decode": "gate_up"}},
    "routed_gate_up_down": {"policy": {"moe_routed_decode": "gate_up_down"}},
    "routed_two_launch": {"policy": {"moe_routed_decode": "two_launch"}},
    "routed_off": {"policy": {"moe_routed_decode": "off"}},
    "hc_multi_row_on": {"policy": {"hc_decode_multi_row": "on"}},
    "hc_multi_row_off": {"policy": {"hc_decode_multi_row": "off"}},
    "fused_gdn_dynamic_accept": {"policy": {"fused_gdn_dynamic_accept": True}},
    "moe_router_kernel": {"policy": {"moe_router_kernel": True}},
    "qsa_nax_decode": {"policy": {"qsa_nax_decode": True}},
    "gdn_core": {"policy": {"gdn_core": True}},
    "indexed_fused_merge": {"policy": {"indexed_fused_merge": True}},
    "indexed_output_gate": {"policy": {"indexed_output_gate": True}},
    "fp32_head_logits": {"policy": {"fp32_head_logits": True}},
    "mtp_draft_vocab": {"policy": {"mtp_draft_vocab": True}},
    "tensorfold_qmv_rows": {"policy": {"tensorfold_qmv_rows": True}},
    "tensorfold_prefill": {"policy": {"tensorfold_prefill": True}},
    "tensorfold_prefill_metal": {
        "policy": {"tensorfold_prefill": True, "tensorfold_prefill_backend": "metal"}},
    "gdn_prefill_chunk_8": {"policy": {"gdn_prefill_chunk": 8}},
    "gdn_prefill_chunk_16": {"policy": {"gdn_prefill_chunk": 16}},
    "gdn_prefill_chunk_16_seg1024": {
        "policy": {"gdn_prefill_chunk": 16, "gdn_prefill_segment_rows": 1024}},
    "prefill_step_2048": {"policy": {"prefill_step": 2048}},
    "prefill_depth_budget": {"policy": {"prefill_depth_budget": DEPTH_BUDGET}},
    "gdn_state_fp16": {"policy": {"gdn_state_dtype": "float16"}},
    "num_draft_1": {"policy": {"num_draft": 1}},
    "num_draft_3": {"policy": {"num_draft": 3}},
    "num_draft_4": {"policy": {"num_draft": 4}},
    "verify_max_steps_8": {"policy": {"fused_gdn_verify_max_steps": 8}},
    # --- server-owned execution-policy options (generator level) ---
    "handoff": {"engine": {"mtp_ordinary_handoff": HANDOFF}},
    "adaptive_mtp_depth": {"engine": {"adaptive_mtp_depth": True}},
    "fly_verification": {"engine": {"fly_verification": FLY}},
    "copy_draft_off": {"engine": {"copy_draft": False}},
    "sp_qmm": {"post": "sp_qmm"},
    # --- env-only switches (not policy fields; adapter strips MLX_QWEN*) ---
    "env_fused_gate_inject": {"post": "fused_gate_inject"},
}

EXTRA_PROMPTS = {
    "history": "Summarize the causes of the 1929 stock market crash in three short paragraphs.",
    "sql": "Write a SQL query that returns the top five customers by total order value in "
    "2025 from tables customers(id, name) and orders(id, customer_id, total, created_at).",
    "email": "Draft a polite two-paragraph email asking a colleague to review a pull request "
    "by Friday.",
    "math": "Prove that the sum of the first n odd numbers is n squared.",
}


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("Swapouts"))
    return int(line.split(":")[1].strip().rstrip("."))


def flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            out.update(flatten(item, f"{prefix}{i}."))
    elif isinstance(value, bool):
        pass
    elif isinstance(value, (int, float)):
        out[prefix[:-1]] = value
    return out


def strings(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(strings(item, f"{prefix}{key}."))
    elif isinstance(value, str) or isinstance(value, bool) or value is None:
        out[prefix[:-1]] = value
    return out


def delta(after, before):
    return {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}


def repetition(tokens, n=4):
    """Largest count of one n-gram (a degenerate-loop signal)."""
    seen = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i:i + n])
        seen[key] = seen.get(key, 0) + 1
    return max(seen.values()) if seen else 0


class Harness:
    def __init__(self, arm_name, *, model=MODEL, cache_limit_gib=4):
        import mlx.core as mx

        self.arm_name = arm_name
        arm = ARMS[arm_name]
        policy = dict(SERVED_ADAPTER_POLICY)
        policy.update(arm.get("policy", {}))
        self.policy_mapping = policy
        self.engine = dict(arm.get("engine", {}))
        from mlx2.adapters.flash_next import FlashNextAdapter

        t0 = time.perf_counter()
        # Construct the adapter before importing runtime model modules: it
        # pins the environment those modules read at import.
        self.adapter = FlashNextAdapter(model, execution_policy=policy)
        self.load_s = time.perf_counter() - t0
        mx.set_cache_limit(cache_limit_gib << 30)
        self.mx = mx
        self.post = None
        if arm.get("post") == "fused_gate_inject":
            from mlx2.runtime.models import qwen4_gate_inject as GI

            GI.set_fused_gate_inject_enabled(True)
            self.post = {"fused_gate_inject": GI.fused_gate_inject_enabled()}
        elif arm.get("post") == "sp_qmm":
            from mlx2.runtime.models import sp_qmm

            handle = sp_qmm.apply(self.adapter.model)
            self.post = {"sp_qmm_modules": len(handle)}
        from mlx2.runtime.copy_draft import CopyDraftPolicy

        copy = self.engine.get("copy_draft", None)
        if copy is False:
            self.copy_policy = None
        else:
            self.copy_policy = CopyDraftPolicy.from_value(
                type(self.adapter).default_route_execution_policy["native_mtp"]["self_mtp_copy_draft"]
            )

    def prompts(self):
        from check_mtp_row_exact import PROMPTS

        texts = dict(PROMPTS)
        texts.update(EXTRA_PROMPTS)
        return {
            name: list(self.adapter.prompt_tokens({"messages": [{"role": "user", "content": text}]}))
            for name, text in texts.items()
        }

    def long_prompt(self, tokens=20000, offset=0):
        corpus = "\n\n".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")))
        tok = self.adapter.tokenizer
        ids = list(tok.encode(corpus))[offset: offset + tokens]
        body = tok.decode(ids)
        request = {"messages": [{"role": "user", "content": "Here is a project document:\n\n" + body
                                 + "\n\nIn five bullet points, what are the main components it describes?"}]}
        return list(self.adapter.prompt_tokens(request))

    def generator(self, *, lanes, mtp, prefill_step=None, engine=None, num_draft=None,
                  depth_budget=None):
        from mlx2.runtime import generate as G
        from mlx2.runtime.adaptive_policy import AdaptiveMTPDepthPolicy, MTPOrdinaryHandoffPolicy
        from mlx2.runtime.speculative_sampling import FLyVerificationPolicy

        engine = self.engine if engine is None else engine
        policy = self.adapter.policy
        step = prefill_step or policy.prefill_step
        stats = {}
        kwargs = dict(completion_batch_size=lanes, prefill_batch_size=min(2, lanes),
                      prefill_step_size=step, scheduler_stats=stats,
                      prefill_depth_budget=(depth_budget if depth_budget is not None
                                            else self.adapter.prefill_depth_budget_default()))
        if mtp:
            config = policy.batch_config(max_lanes=lanes, prefill_step=step)
            if num_draft is not None:
                config["num_draft"] = num_draft
            kwargs["self_mtp"] = config
            if self.copy_policy is not None and engine.get("copy_draft", True) is not False:
                kwargs["copy_draft"] = self.copy_policy
            if engine.get("mtp_ordinary_handoff"):
                kwargs["mtp_ordinary_handoff"] = MTPOrdinaryHandoffPolicy.from_value(
                    engine["mtp_ordinary_handoff"])
            if engine.get("adaptive_mtp_depth"):
                kwargs["adaptive_mtp_depth"] = AdaptiveMTPDepthPolicy.from_value(
                    engine["adaptive_mtp_depth"]).controller_kwargs()
            if engine.get("fly_verification"):
                kwargs["fly_verification"] = FLyVerificationPolicy.from_value(engine["fly_verification"])
        return G.BatchGenerator(self.adapter.model, **kwargs), stats

    def run(self, prompts, *, max_tokens, mtp, temp=0.0, **gen_kwargs):
        """Decode ``prompts`` as one cohort; returns tokens, timing, stats."""
        from mlx2.runtime.sample_utils import LaneRNG

        mx = self.mx
        gen, stats = self.generator(lanes=len(prompts), mtp=mtp, **gen_kwargs)
        insert = {"max_tokens": [max_tokens] * len(prompts),
                  "lane_rngs": [LaneRNG(1 + i) for i in range(len(prompts))]}
        if mtp:
            insert["self_mtp_configs"] = [{"sampling_temp": temp}] * len(prompts)
        t0 = time.perf_counter()
        uids = gen.insert([list(p) for p in prompts], **insert)
        tokens, done, started = {}, set(), set()
        t_first = t_end = t_any = None
        emitted = steps = 0
        try:
            while len(done) < len(prompts):
                _p, responses = gen.next()
                now = time.perf_counter()
                if responses and t_any is None:
                    t_any = now
                if t_first is not None and responses:
                    steps += 1
                started.update(r.uid for r in responses)
                if responses and t_first is None and started >= set(uids):
                    t_first = now
                    emitted -= len(responses)
                for r in responses:
                    tokens.setdefault(r.uid, []).append(int(r.token))
                    if r.finish_reason:
                        done.add(r.uid)
                if t_first is not None:
                    emitted += len(responses)
                    t_end = now
        finally:
            gen.close()
        mx.clear_cache()
        lanes = [tokens.get(u, []) for u in uids]
        span = (t_end - t_first) if (t_end and t_first and t_end > t_first) else None
        return {
            "lanes": lanes,
            "sha": hashlib.sha256(json.dumps(lanes).encode()).hexdigest()[:16],
            "decode_tps": emitted / span if span else None,
            "ttft_s": (t_any - t0) if t_any else None,
            "wall_s": time.perf_counter() - t0,
            "steps": steps,
            "tokens_per_step": emitted / steps if steps else None,
            "scheduler_stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))},
        }


def smoke(args):
    import mlx.core as mx

    swap0 = swapouts()
    report = {"arm": args.arm, "spec": ARMS[args.arm], "mlx": mx.__version__,
              "model": args.model, "swapouts_start": swap0}
    try:
        h = Harness(args.arm, model=args.model)
    except Exception as error:  # noqa: BLE001 - a refusal is a smoke result
        report["load_error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        Path(args.out).write_text(json.dumps(report, indent=1))
        print("LOAD ERROR", report["load_error"], flush=True)
        return 0
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    default_env = FlashNextPolicy.from_mapping(SERVED_ADAPTER_POLICY).environment()
    env = h.adapter.policy.environment()
    report.update({
        "load_s": h.load_s,
        "policy_mapping": h.policy_mapping,
        "receipt_policy": h.adapter.policy.as_dict(),
        "receipt_policy_default": FlashNextPolicy.from_mapping(SERVED_ADAPTER_POLICY).as_dict(),
        "environment_delta": {k: [default_env.get(k), v] for k, v in env.items() if default_env.get(k) != v}
        | {k: [v, None] for k, v in default_env.items() if k not in env},
        "engine": h.engine, "post": h.post,
        "active_gib": mx.get_active_memory() / 2**30,
    })
    print("LOADED", args.arm, f"{h.load_s:.1f}s", f"active={report['active_gib']:.1f}GiB", flush=True)
    prompts = h.prompts()
    four = [prompts[n] for n in ("prose", "code", "copy", "reason")]
    eight = four + [prompts[n] for n in ("history", "sql", "email", "math")]
    workloads = [
        ("b1_ord", [[p] for p in four], dict(max_tokens=96, mtp=False)),
        ("b1_mtp", [[p] for p in four], dict(max_tokens=96, mtp=True)),
        ("b8_mtp", [eight], dict(max_tokens=48, mtp=True)),
        ("b4_ord", [four], dict(max_tokens=48, mtp=False)),
        ("long_mtp", [[h.long_prompt()]], dict(max_tokens=32, mtp=True)),
    ]
    if args.workloads:
        keep = set(args.workloads.split(","))
        workloads = [w for w in workloads if w[0] in keep]
    results = {}
    diag = lambda: flatten(h.adapter.diagnostics())  # noqa: E731
    for name, cohorts, kw in workloads:
        entry = {"cohorts": []}
        before = diag()
        mx.reset_peak_memory()
        try:
            for cohort in cohorts:
                out = h.run(cohort, **kw)
                out["text"] = [h.adapter.tokenizer.decode(lane)[:400] for lane in out["lanes"]]
                out["max_4gram_repeat"] = [repetition(lane) for lane in out["lanes"]]
                entry["cohorts"].append(out)
                print(name, out["sha"], f"tps={out['decode_tps']}", f"tok/step={out['tokens_per_step']}",
                      json.dumps(out["scheduler_stats"])[:300], flush=True)
        except Exception as error:  # noqa: BLE001 - recorded as a smoke failure
            entry["error"] = f"{type(error).__name__}: {error}"
            entry["traceback"] = traceback.format_exc()
            print("ERROR", name, entry["error"], flush=True)
        entry["engagement"] = delta(diag(), before)
        entry["peak_gib"] = mx.get_peak_memory() / 2**30
        results[name] = entry
        swap = swapouts() - swap0
        if swap > args.max_swapout_pages:
            report["aborted"] = f"swapouts +{swap} pages after {name}"
            break
    final = h.adapter.diagnostics()
    report["final_strings"] = {k: v for k, v in strings(final).items()
                               if v not in (None, "") and any(s in k for s in (
                                   "last", "reason", "mode", "error", "decline", "fallback", "refus",
                                   "enabled", "selected", "engaged"))}
    report["workloads"] = results
    report["swapouts_delta"] = swapouts() - swap0
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("DONE", args.arm, "swapouts_delta", report["swapouts_delta"], flush=True)
    return 0


# ---------------------------------------------------------------- compare --

def compare(args):
    root = Path(args.smoke_dir)
    ref = json.loads((root / "default.json").read_text())
    rows = {}
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text())
        arm = data["arm"]
        row = {"load_error": data.get("load_error"), "aborted": data.get("aborted"),
               "receipt_label": {k: v for k, v in (data.get("receipt_policy") or {}).items()
                                 if (data.get("receipt_policy_default") or {}).get(k, object()) != v},
               "env_delta": data.get("environment_delta"), "workloads": {}}
        for name, entry in (data.get("workloads") or {}).items():
            refw = ref["workloads"].get(name)
            w = {"error": entry.get("error"), "peak_gib": round(entry.get("peak_gib", 0), 1)}
            if refw and not entry.get("error"):
                same, first_div = [], []
                for c, rc in zip(entry["cohorts"], refw["cohorts"]):
                    for lane, rlane in zip(c["lanes"], rc["lanes"]):
                        same.append(lane == rlane)
                        k = next((i for i, (x, y) in enumerate(zip(lane, rlane)) if x != y), None)
                        if k is None and len(lane) != len(rlane):
                            k = min(len(lane), len(rlane))
                        first_div.append(k)
                w["identical_lanes"] = f"{sum(same)}/{len(same)}"
                w["first_divergence"] = first_div
                if name == "b1_mtp" and "b1_ord" in ref["workloads"]:
                    # MTP-on vs the default arm's MTP-off output (row-exact target)
                    ordw = (data.get("workloads") or {}).get("b1_ord") or ref["workloads"]["b1_ord"]
                    w["equals_mtp_off"] = f"{sum(l == o for c, oc in zip(entry['cohorts'], ordw['cohorts']) for l, o in zip(c['lanes'], oc['lanes']))}/{len(same)}"
                w["max_4gram_repeat"] = max(max(c["max_4gram_repeat"]) for c in entry["cohorts"])
                tps = [c["decode_tps"] for c in entry["cohorts"] if c["decode_tps"]]
                rtps = [c["decode_tps"] for c in refw["cohorts"] if c["decode_tps"]]
                if tps and rtps:
                    w["tps_ratio_xprocess"] = round(statistics.mean(tps) / statistics.mean(rtps), 3)
                tps_step = [c["tokens_per_step"] for c in entry["cohorts"] if c["tokens_per_step"]]
                if tps_step:
                    w["tokens_per_step"] = round(statistics.mean(tps_step), 3)
                eng = entry.get("engagement", {})
                reng = refw.get("engagement", {})
                keys = set(eng) | set(reng)
                w["engagement_diff"] = {k: [reng.get(k, 0), eng.get(k, 0)] for k in sorted(keys)
                                        if reng.get(k, 0) != eng.get(k, 0)
                                        and not any(s in k for s in ("time", "_ms", "_ns", "seconds", "bytes",
                                                                       "ple_tables", "last_", "lru",
                                                                       "hits", "misses", "evict",
                                                                       "candidate", "prefetch_rows"))}
                sched = {}
                for c in entry["cohorts"]:
                    for k, v in c["scheduler_stats"].items():
                        sched[k] = sched.get(k, 0) + v
                rsched = {}
                for c in refw["cohorts"]:
                    for k, v in c["scheduler_stats"].items():
                        rsched[k] = rsched.get(k, 0) + v
                w["scheduler_diff"] = {k: [rsched.get(k, 0), sched.get(k, 0)]
                                       for k in sorted(set(sched) | set(rsched))
                                       if rsched.get(k, 0) != sched.get(k, 0)}
            row["workloads"][name] = w
        rows[arm] = row
    Path(args.out).write_text(json.dumps(rows, indent=1))
    for arm, row in rows.items():
        cells = []
        for name, w in row["workloads"].items():
            cells.append(f"{name}:{w.get('identical_lanes', 'ERR' if w.get('error') else '?')}")
        print(f"{arm:30s}", row["load_error"] or "", " ".join(cells))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("smoke", "compare", "list"), required=True)
    ap.add_argument("--arm", default="default")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workloads", default=None, help="comma subset of smoke workloads")
    ap.add_argument("--smoke-dir", default=None)
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    args = ap.parse_args()
    if args.phase == "list":
        print("\n".join(ARMS))
        return 0
    if args.phase == "compare":
        return compare(args)
    if args.arm not in ARMS:
        ap.error(f"unknown arm {args.arm}")
    if not args.i_own_the_gpu:
        ap.error("Metal run: pass --i-own-the-gpu under the GPU lock")
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    return smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
