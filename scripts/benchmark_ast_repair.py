#!/usr/bin/env python3
"""One artifact, paired-source B1 AST repair benchmark; never qualification.

Run under cpg_job.py and run_with_gpu_locks.py. Reuses the historical smoke's
prompts, serving shape, token trace, and process-group cleanup. Each source arm
has a private server and cache. Raw outputs remain in the external evidence dir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = Path("~/mlx-models")
POLICIES = ROOT / "qualification/runs/ast-smokes-20261018/policies"
CONFIGS = {
    "qwen27": ("Qwen3.8-27B-oQ4e-mtp", POLICIES / "qwen38-27b-ast.json", "off", None),
    "muse": ("Muse-Glimmer-30B-mlx-8bit", POLICIES / "muse-ast.json", "exact", '{"max_rows":64}'),
    "qwen36": ("Qwen3.6-35B-A3B-UD-Q8_K_XL-mlx", POLICIES / "qwen36-35b-ast.json", "off", None),
    "flash_native": ("Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP", POLICIES / "flash-next-native-mtp-ast.json", None, None),
    "flash_dflash": ("Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP", ROOT / "qualification/runs/flash-next-dflash2-a22a1744/policies/k3-tile2-ast-b2-no-rootless.json", None, None),
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def command(*argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def arm_arguments(arm, policy):
    return ["--ordinary"] if arm == "ordinary" else ["--external-draft", "--execution-policy", str(policy)]


def unstable_reference(report, labels, repeats):
    unstable = {}
    for label in labels:
        values = [report["phases"][f"b1_r{r + 1}_{label}"]["decode_tokens_per_second"]
                  for r in range(repeats)]
        if min(values) / max(values) < 0.85:
            unstable[label] = values
    return unstable


def host_load():
    rows = []
    for line in command("ps", "-axo", "pid=,pcpu=,comm=").splitlines():
        pid, cpu, executable = line.strip().split(None, 2)
        if float(cpu) > 50:
            rows.append({"pid": int(pid), "cpu_percent": float(cpu), "executable": executable})
    return rows


def source(root):
    dirty = command("git", "status", "--porcelain", "--untracked-files=no", cwd=root)
    if dirty:
        raise RuntimeError(f"dirty tracked checkout: {root}\n{dirty}")
    return command("git", "rev-parse", "HEAD", cwd=root)


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def ownership(cpg_owner):
    paired = [json.loads((Path(p) / "owner.json").read_text())
              for p in ("/Users/Shared/mlxuag/gpu.lock", "/tmp/gpu.lock")]
    if paired[0] != paired[1] or paired[0]["pid"] != os.getppid():
        raise RuntimeError("paired GPU owners do not match this wrapper")
    cpg = json.loads((cpg_owner / "owner.json").read_text())
    radio = json.loads(Path(cpg["log"] + ".radio.json").read_text())
    claim = radio["claim_task"]
    if not claim.get("claimed") or claim["owner_agent_id"] != cpg["agent_id"]:
        raise RuntimeError("CPG lease owner mismatch")
    os.kill(cpg["pid"], 0)
    return {"paired": paired, "cpg": cpg, "radio": radio}


def inventory():
    from mlx2.adapters.registry import inspect_model
    from mlx2.adapters.dflash2 import inspect_drafter
    targets, auxiliaries = {}, {}
    for name, policy_path, _lane, _lane_policy in CONFIGS.values():
        path = MODELS / name
        if name not in targets:
            resolution = inspect_model(path)
            info = resolution.artifact.get("identity", resolution.artifact)
            targets[name] = {"checkpoint": name + "@" + info["fingerprint"],
                             "artifact": info,
                             "weight_bytes": sum(p.stat().st_size for p in path.glob("*.safetensors"))}
        draft = Path(json.loads(policy_path.read_text())["draft_model"])
        if draft != path and str(draft) not in auxiliaries:
            record = inspect_drafter(draft, path)
            info = {key: record[key] for key in ("path", "fingerprint", "files", "weight_sha256", "config_sha256")}
            auxiliaries[str(draft)] = {"checkpoint": draft.name + "@" + info["fingerprint"],
                                      "artifact": info, "role": "auxiliary_not_target"}
    return {"targets": targets, "auxiliaries": auxiliaries,
            "smallest_first": sorted(targets, key=lambda k: targets[k]["weight_bytes"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", action="store_true")
    ap.add_argument("--config", choices=CONFIGS)
    ap.add_argument("--old-root", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--cpg-owner", type=Path)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--start-max-c", type=float, default=60)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--ast-planner", action="store_true", help="select the candidate planner in the fixed arm only")
    ap.add_argument("--auto-tile", action="store_true", help="use calculated tiles in the candidate arm")
    args = ap.parse_args()
    identities = inventory()
    if args.inventory:
        print(json.dumps(identities, indent=2, default=str))
        return
    if not all((args.config, args.old_root, args.out_dir, args.cpg_owner)) or args.repeats < 2:
        ap.error("config, old-root, out-dir, cpg-owner and at least two repeats are required")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    source_heads = {"fixed": source(ROOT), "before": source(args.old_root)}
    owned = ownership(args.cpg_owner)
    smoke = load(ROOT / "qualification/runs/ast-smokes-20261018/ast_smoke.py", "ast_smoke_repair")
    guards = load(ROOT / "scripts/run_with_gpu_locks.py", "ast_bench_locks")
    foreign = guards.foreign_model_processes()
    if foreign:
        raise RuntimeError(f"foreign model processes: {foreign}")
    # Freeze long-prompt bytes from the control tree; never let source changes
    # alter prompt contents between arms.
    smoke.ROOT = args.old_root
    prompts = {**smoke.PROMPTS, "long_4k": smoke.long_prompt()}
    atomic(args.out_dir / "prompts.json", prompts)
    name, policy_path, lane, lane_policy = CONFIGS[args.config]
    policy = json.loads(policy_path.read_text())
    atomic(args.out_dir / "policy.json", policy)
    fixed_policy = json.loads(json.dumps(policy))
    if args.ast_planner:
        fixed_policy["continuation_pool"]["ast_planner"] = True
    if args.auto_tile:
        if not args.ast_planner:
            ap.error("auto-tile benchmark requires ast-planner")
        fixed_policy["progressive_verification_tile"] = "auto"
    atomic(args.out_dir / "fixed-policy.json", fixed_policy)
    meta = {"schema": "mlx2.ast-repair-performance.v1", "qualification": False,
            "config": args.config, "source_heads": source_heads,
            "checkpoint": identities["targets"][name]["checkpoint"],
            "inventory": identities, "ownership": owned,
            "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "policy": policy, "prompt_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in prompts.items()},
            "hardware": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "thermal_start_ceiling_c": args.start_max_c,
            "shape": {"max_context": 32768, "cache_gib": 8, "max_lanes": 4, "max_inflight": 8,
                      "request_width": 1, "max_tokens": 192, "prefix_writes": True, "repeats": args.repeats},
            "arms": {}, "status": "running"}
    atomic(args.out_dir / "meta.json", meta)
    from mlx2.apple_telemetry import TemperatureSampler
    temps = TemperatureSampler()
    raw_stream = smoke.stream

    def thermal():
        return {"at": time.time(), **temps.die_summary(), "pmset": command("pmset", "-g", "therm"),
                "power_source": command("pmset", "-g", "batt"), "host_load": host_load()}

    def checked_stream(*positional, **keywords):
        deadline = time.monotonic() + 180
        before = thermal()
        while before["die_max_c"] is not None and before["die_max_c"] > args.start_max_c:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"thermal start gate not reached: {before}")
            time.sleep(5)
            before = thermal()
        if before["die_max_c"] is None:
            raise RuntimeError("die temperature unavailable")
        if any("VirtualMachine" in p["executable"] and p["cpu_percent"] > 100 for p in before["host_load"]):
            raise RuntimeError("busy virtual machine contaminates the performance window")
        swap_before = smoke.swap_state()
        row = raw_stream(*positional, **keywords)
        row["thermal_before"], row["thermal_after"] = before, thermal()
        row["swap_before"], row["swap_after"] = swap_before, smoke.swap_state()
        row["wall_tokens_per_second"] = row.get("completion_tokens", 0) / max(row.get("elapsed_seconds", 0), 1e-9)
        if (row.get("error") or not row.get("done") or not row.get("prompt_tokens")
                or not row.get("completion_tokens") or not row.get("route") or not row.get("token_ids")):
            row["error"] = row.get("error") or "incomplete stream or missing trace/route"
        return row

    smoke.stream = checked_stream
    serving = argparse.Namespace(python=sys.executable, model=str(MODELS / name), max_context=32768,
                                cache_gib=8, lane_matmul=lane, lane_policy=lane_policy,
                                server_arg=["--int8-prefill", "off"], load_timeout=1200,
                                skip_writing_prefix_cache=False, max_tokens=192)
    reports = {}
    order = ["ordinary", "before", "fixed"]
    if args.reverse:
        order.reverse()
    meta["order"] = order
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    try:
        for arm in order:
            ownership(args.cpg_owner)
            smoke.ROOT = args.old_root if arm == "before" else ROOT
            expected = source_heads["before" if arm == "before" else "fixed"]
            if source(smoke.ROOT) != expected:
                raise RuntimeError("source revision moved during campaign")
            arm_policy = "fixed-policy.json" if arm == "fixed" else "policy.json"
            extra = arm_arguments(arm, args.out_dir / arm_policy)
            cells = {"warmup": smoke.SHORT_2}
            for repeat in range(args.repeats):
                labels = list(prompts)
                labels = labels[repeat % len(labels):] + labels[:repeat % len(labels)]
                cells.update({f"r{repeat + 1}_{label}": prompts[label] for label in labels})
            meta["arms"][arm] = {"source_revision": source(smoke.ROOT),
                                 "policy": fixed_policy if arm == "fixed" else policy if arm == "before" else None}
            atomic(args.out_dir / "meta.json", meta)
            reports[arm] = smoke.run_arm(serving, arm, extra, args.out_dir, {"b1": cells})
            foreign = guards.foreign_model_processes()
            if foreign:
                raise RuntimeError(f"model process survived arm cleanup: {foreign}")
            if arm == "ordinary" and (unstable := unstable_reference(reports[arm], prompts, args.repeats)):
                raise RuntimeError(f"ordinary reference varies more than 15 percent: {unstable}")
        summary = {}
        for label in prompts:
            entry = {}
            for arm, report in reports.items():
                rows = [report["phases"][f"b1_r{r + 1}_{label}"] for r in range(args.repeats)]
                reference = [reports["ordinary"]["phases"][f"b1_r{r + 1}_{label}"] for r in range(args.repeats)]
                entry[arm] = {
                    "decode_tps_median": statistics.median(r["decode_tokens_per_second"] for r in rows),
                    "wall_tps_median": statistics.median(r["wall_tokens_per_second"] for r in rows),
                    "decode_tps": [r["decode_tokens_per_second"] for r in rows],
                    "exact_token_parity": [r["token_ids"] == ref["token_ids"] for r, ref in zip(rows, reference)],
                    "first_divergence": [smoke._first_diff(r["token_ids"], ref["token_ids"]) for r, ref in zip(rows, reference)],
                    "swapouts_delta": [r["swap_after"]["swapouts"] - r["swap_before"]["swapouts"] for r in rows],
                    "thermal_start_c": [r["thermal_before"]["die_max_c"] for r in rows],
                    "thermal_end_c": [r["thermal_after"]["die_max_c"] for r in rows],
                }
            entry["fixed_vs_before_pct"] = 100 * (entry["fixed"]["decode_tps_median"] / entry["before"]["decode_tps_median"] - 1)
            entry["fixed_vs_ordinary_ratio"] = entry["fixed"]["decode_tps_median"] / entry["ordinary"]["decode_tps_median"]
            summary[label] = entry
        atomic(args.out_dir / "summary.json", summary)
        meta["status"] = "completed"
    except BaseException as exc:
        meta["status"], meta["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        temps.close()
        meta["finished_at"] = time.time()
        atomic(args.out_dir / "meta.json", meta)


if __name__ == "__main__":
    main()
