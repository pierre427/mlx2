#!/usr/bin/env python3
"""Qwen3.8-27B compact GDN replay vs per-step snapshots: memory and parity probe.

Two modes:

``--analytic`` (CPU only, no model load, no Metal): read the artifact's
``config.json`` and print the per-sequence recurrent / rollback state bytes
for both rollback modes at the given verify widths and lane widths.

GPU mode (``--out`` and ``--i-own-the-gpu``; run only inside a GPU ownership
window): one model load on the served native-MTP route.  The adapter loads
with its default (compact replay off); the probe then flips every live Qwen3.8
GDN layer between

* ``snapshots`` -- the selected default: fused verify returns per-step
  state/conv snapshots (``speculative_rollback: exact-snapshots``);
* ``compact``   -- ``fused_gdn_compact_replay``: verify records the
  correction tape (keys, corrections, decay) and reconstructs a state only
  on a partial rejection.

and keeps ``adapter.fused_gdn_compact_replay`` equal to the live layers so the
numerics receipt cannot disagree.  For each width it runs greedy B lanes,
one realistic chat prompt per lane, warm-up discarded, then each mode and a
replicate of the first.  Per run: peak memory above the pre-run active
baseline, per-lane tokens, and the GDN counter deltas that show what engaged
(``compact_verify_calls``, ``compact_rollback_calls``, ``compact_tape_bytes``,
``compact_snapshot_bytes_avoided``, ``verify_calls``, fallback reasons).
Before the MTP runs every lane prompt is decoded MTP-off at B=1 (the ordinary
reference).  Timing fields are exploratory only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODES = ("snapshots", "compact")
SCHEMA = "mlx2.qwen38-27b-compact-replay-memory.v1"
GPU_LOCK = Path("/Users/Shared/mlxuag/gpu.lock")
RESULT_NAME = "result.json"


def _ancestor_pids(start_pid: int) -> set[int]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from qualification_gpu_ownership import _ancestor_pids as ancestors

    return ancestors(start_pid)


def verify_gpu_lock(lock: Path = GPU_LOCK, *, pid: int | None = None, ancestors=_ancestor_pids) -> dict:
    """Refuse unless ``lock`` is a held lock directory owned by an ancestor.

    ``scripts/run_with_gpu_locks.py`` replaces the idle empty sentinel file
    with a directory holding ``owner.json`` whose ``pid`` is the wrapper's
    own; a run under it therefore has that pid in its ancestor chain.
    """
    lock = Path(lock)
    if not lock.is_dir():
        raise SystemExit(f"refusing: {lock} is not a held lock directory (run under scripts/run_with_gpu_locks.py)")
    try:
        owner = json.loads((lock / "owner.json").read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"refusing: unreadable {lock / 'owner.json'}: {type(exc).__name__}")
    owner_pid = owner.get("pid") if isinstance(owner, dict) else None
    if type(owner_pid) is not int:
        raise SystemExit(f"refusing: {lock / 'owner.json'} has no integer pid")
    chain = ancestors(os.getpid() if pid is None else pid)
    if owner_pid not in chain:
        raise SystemExit(f"refusing: GPU lock owner pid {owner_pid} is not an ancestor of this process")
    return owner


def refuse_nonempty_evidence_dir(path: Path) -> Path:
    """The evidence directory must be absent or empty, so retries stay distinct."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit(f"refusing to write into nonempty evidence path {path}")
    return path


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    os.replace(tmp, path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model", required=True, help="artifact directory (config.json read-only)")
    ap.add_argument("--analytic", action="store_true",
                    help="CPU only: print the analytic byte estimate and exit (no model load)")
    ap.add_argument("--verify-widths", nargs="+", type=int, default=[4, 8, 16],
                    help="analytic: verify block widths S (anchor + drafts); default route native MTP d3 = 4")
    ap.add_argument("--lanes", nargs="+", type=int, default=[1, 4, 8, 16],
                    help="analytic: lane widths")
    ap.add_argument("--state-dtype", choices=("float32", "float16"), default="float32",
                    help="analytic: GDN recurrent state dtype (adapter default float32)")
    ap.add_argument("--widths", nargs="+", type=int, default=[1, 4],
                    help="GPU: live lane widths to run")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--cache-limit-gib", type=int, default=4)
    ap.add_argument("--max-swapout-pages", type=int, default=2000)
    ap.add_argument("--out", type=Path,
                    help="GPU: evidence directory, absent or empty; result.json is rewritten atomically per cell")
    ap.add_argument("--i-own-the-gpu", action="store_true",
                    help="GPU: assert the caller holds the shared GPU lock; verified against "
                         f"{GPU_LOCK}/owner.json (pid must be an ancestor, i.e. run_with_gpu_locks.py)")
    return ap


def parse_args(argv=None) -> argparse.Namespace:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.analytic:
        if any(not 2 <= s <= 17 for s in a.verify_widths):
            ap.error("--verify-widths must be 2..17 (the proven fused verify bound)")
        if any(n < 1 for n in a.lanes):
            ap.error("--lanes must be positive")
        return a
    if a.out is None or not a.i_own_the_gpu:
        ap.error("GPU mode needs --out and --i-own-the-gpu (or pass --analytic)")
    if any(n < 1 for n in a.widths):
        ap.error("--widths must be positive")
    if len(set(a.modes)) != len(a.modes) or len(a.modes) < 2:
        ap.error("--modes needs both modes once each")
    return a


def gdn_geometry(config: dict) -> dict:
    text = config.get("text_config", config)
    layer_types = text.get("layer_types") or []
    return {
        "gdn_layers": sum(1 for kind in layer_types if kind == "linear_attention"),
        "num_key_heads": int(text["linear_num_key_heads"]),
        "num_value_heads": int(text["linear_num_value_heads"]),
        "key_head_dim": int(text["linear_key_head_dim"]),
        "value_head_dim": int(text["linear_value_head_dim"]),
        "conv_kernel": int(text["linear_conv_kernel_dim"]),
    }


def estimate(geometry: dict, verify_width: int, *, state_bytes: int = 4,
             activation_bytes: int = 2) -> dict:
    """Per-sequence bytes for one verify transaction of ``verify_width`` rows.

    Formulas match the code: snapshot slabs are the fused verify outputs
    ``(S-1, HV, DV, DK)`` state + ``(S-1, K-1, conv_dim)`` conv
    (qwen4_fused_gdn_verify.qwen4_fused_gdn_verify); the tape is
    ``Qwen38StatePlaneOperations.authorize_compact_gdn_replay``'s
    ``tape_bytes`` (keys in the activation dtype, fp32 corrections and decay).
    """
    g = geometry
    hk, hv = g["num_key_heads"], g["num_value_heads"]
    dk, dv = g["key_head_dim"], g["value_head_dim"]
    conv_dim = 2 * hk * dk + hv * dv
    state = hv * dv * dk * state_bytes
    conv = (g["conv_kernel"] - 1) * conv_dim * activation_bytes
    tape_steps = verify_width - 1
    per_layer = {
        "state": state,
        "conv": conv,
        "snapshot_record": tape_steps * (state + conv),
        "compact_tape": tape_steps * (hk * dk * activation_bytes + hv * dv * 4 + hv * 4),
        # Approximate: the stock multi-row path (qwen3_5.GatedDeltaNet) keeps
        # q, k, v, a, b and the conv input alive in its replay closure.
        "reference_replay_inputs": activation_bytes * (
            verify_width * (2 * hk * dk + hv * dv + 2 * hv)
            + (g["conv_kernel"] - 1 + verify_width) * conv_dim
        ),
    }
    n = g["gdn_layers"]
    live = n * (state + conv)
    snap = n * per_layer["snapshot_record"]
    tape = n * per_layer["compact_tape"]
    return {
        "verify_width": verify_width,
        "per_layer": per_layer,
        "live_recurrent": live,
        "snapshot_record": snap,
        "compact_tape": tape,
        # checkpoint pinned by the rollback record + the new final state + store
        "verify_peak_snapshots": 2 * live + snap,
        "verify_peak_compact": 2 * live + tape,
        "reference_replay_inputs": n * per_layer["reference_replay_inputs"],
        "rollback_store_reduction": 1 - tape / snap,
        "verify_peak_reduction": 1 - (2 * live + tape) / (2 * live + snap),
    }


def analytic_report(config: dict, verify_widths, lanes, state_dtype="float32") -> dict:
    geometry = gdn_geometry(config)
    state_bytes = 4 if state_dtype == "float32" else 2
    rows = []
    for s in verify_widths:
        e = estimate(geometry, s, state_bytes=state_bytes)
        e["lanes"] = {
            str(b): {
                # What the fused snapshot verify stores for b lanes (B1, or
                # the opt-in row-exact batched verify, per lane).
                "snapshot_record": b * e["snapshot_record"],
                "compact_tape": b * e["compact_tape"],
                # If a batched compact verify existed (it does not today).
                "potential_saving": b * (e["snapshot_record"] - e["compact_tape"]),
                # The served default B>1 path (stock multi-row, not fused).
                "default_b_gt_1_reference_inputs": b * e["reference_replay_inputs"] if b > 1 else 0,
                # Compact replay is B1-only: batched verify declines it
                # ('batched compact replay unavailable') and the row-exact
                # executor drops to its serial path, so B>1 saves nothing today.
                "current_saving": (e["snapshot_record"] - e["compact_tape"]) if b == 1 else 0,
            }
            for b in lanes
        }
        rows.append(e)
    return {"geometry": geometry, "state_dtype": state_dtype, "rows": rows}


def _print_analytic(report: dict) -> None:
    mib = 2 ** 20
    g = report["geometry"]
    print(f"GDN layers {g['gdn_layers']}  HK {g['num_key_heads']} HV {g['num_value_heads']} "
          f"DK {g['key_head_dim']} DV {g['value_head_dim']} K {g['conv_kernel']}  state {report['state_dtype']}")
    for e in report["rows"]:
        print(f"S={e['verify_width']}: live {e['live_recurrent'] / mib:.1f} MiB/seq; "
              f"snapshot record {e['snapshot_record'] / mib:.1f} MiB; compact tape {e['compact_tape'] / mib:.2f} MiB "
              f"(-{100 * e['rollback_store_reduction']:.1f}% store, -{100 * e['verify_peak_reduction']:.1f}% "
              f"verify-peak recurrent)")
        for b, row in e["lanes"].items():
            print(f"   B={b}: snapshots {row['snapshot_record'] / mib:.1f} MiB vs tape "
                  f"{row['compact_tape'] / mib:.2f} MiB; potential -{row['potential_saving'] / mib:.1f} MiB, "
                  f"current -{row['current_saving'] / mib:.1f} MiB; served-default B>1 stock path holds "
                  f"~{row['default_b_gt_1_reference_inputs'] / mib:.1f} MiB of replay inputs")


def main(argv=None) -> int:
    a = parse_args(argv)
    config = json.loads((Path(a.model).expanduser() / "config.json").read_text())
    if a.analytic:
        report = analytic_report(config, a.verify_widths, a.lanes, a.state_dtype)
        _print_analytic(report)
        print(json.dumps(report, indent=1))
        return 0
    return gpu_probe(a)


def gpu_probe(a) -> int:  # pragma: no cover - GPU only
    owner = verify_gpu_lock()
    out_dir = refuse_nonempty_evidence_dir(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / RESULT_NAME
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from bench_fn_batch_verify import TASKS, _drain_poll

    from mlx2.adapters.registry import resolve_adapter

    adapter = resolve_adapter(a.model, mtp=True)(a.model)
    import mlx.core as mx

    from mlx2.runtime import generate as G
    from mlx2.runtime.models import qwen38_fused_gdn as fused
    from mlx2.runtime.sample_utils import LaneRNG

    model = adapter.model
    mx.eval(model.parameters())
    mx.set_cache_limit(a.cache_limit_gib << 30)
    operations = model.state_plane_operations

    def swapouts():
        text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        line = next(l for l in text.splitlines() if l.startswith("Swapouts"))
        return int(line.split(":")[1].strip().rstrip("."))

    swap0 = swapouts()

    def check_swap():
        grew = swapouts() - swap0
        if grew > a.max_swapout_pages:
            raise SystemExit(f"aborting: Swapouts grew by {grew} pages since load")

    gdn = [m for _, m in model.named_modules() if isinstance(m, fused.GatedDeltaNet)]
    original_modes = [m.fused_gdn_compact_replay_enabled for m in gdn]
    original_adapter_mode = getattr(adapter, "fused_gdn_compact_replay", False)
    default_compact = set(original_modes)
    loaded_gib = mx.get_active_memory() / 2**30
    print("LOADED", len(gdn), "GDN layers", f"active={loaded_gib:.2f}GiB",
          "default compact", default_compact, flush=True)
    copy_policy = adapter.default_route_execution_policy["native_mtp"]["self_mtp_copy_draft"]
    tok = adapter.tokenizer

    def prompt(i):
        task = TASKS[i % len(TASKS)]
        return list(adapter.prompt_tokens({"messages": [{"role": "user", "content": task}]}))

    def configure(mode):
        compact = mode == "compact"
        for layer in gdn:
            layer.set_fused_gdn_compact_replay_enabled(compact, operations)
        adapter.fused_gdn_compact_replay = compact

    def restore_modes():
        for layer, mode in zip(gdn, original_modes):
            layer.set_fused_gdn_compact_replay_enabled(mode, operations)
        adapter.fused_gdn_compact_replay = original_adapter_mode

    def counters():
        report = fused.stats(model)
        return {k: v for k, v in report.items() if isinstance(v, int) and not isinstance(v, bool)}, \
            dict(report["reasons"])

    def drain(gen, uids):
        tokens, done, idle = {}, set(), 0
        t0 = time.perf_counter()
        while len(done) < len(uids):
            responses, idle = _drain_poll(gen, idle)
            for r in responses:
                tokens.setdefault(r.uid, []).append(int(r.token))
                if r.finish_reason:
                    done.add(r.uid)
        return [tokens.get(u, []) for u in uids], time.perf_counter() - t0

    # The 27B adapter returns None (serving then autoscales by prompt length);
    # use the serving fallback, the schedule's last step, as the server would.
    step = adapter.prefill_step_default()
    if step is None:
        from mlx2.runtime.prefill_plan import PROMPT_LENGTH_PREFILL_SCHEDULE
        step = int(PROMPT_LENGTH_PREFILL_SCHEDULE[-1][1])

    def solo(i):
        gen = G.BatchGenerator(model, completion_batch_size=1, prefill_batch_size=1,
                               prefill_step_size=step)
        try:
            uids = gen.insert([prompt(i)], max_tokens=[a.gen], lane_rngs=[LaneRNG(1 + i)])
            return drain(gen, uids)[0][0]
        finally:
            gen.close()
            mx.clear_cache()

    def run(batch):
        prompts = [prompt(i) for i in range(batch)]
        gen = G.BatchGenerator(
            model, completion_batch_size=batch, prefill_batch_size=1,
            prefill_step_size=step, scheduler_stats={},
            self_mtp=adapter.execution_config(max_lanes=batch, prefill_step=step),
            copy_draft=copy_policy)
        before, reasons0 = counters()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        try:
            uids = gen.insert(prompts, max_tokens=[a.gen] * batch,
                              lane_rngs=[LaneRNG(1 + i) for i in range(batch)],
                              self_mtp_configs=[{"sampling_temp": 0.0}] * batch)
            lanes, secs = drain(gen, uids)
        finally:
            gen.close()
        peak = mx.get_peak_memory()
        after, reasons1 = counters()
        mx.clear_cache()
        return {
            "base_active_gib": base / 2**30, "peak_gib": peak / 2**30,
            "peak_above_base_gib": (peak - base) / 2**30,
            "lanes": lanes, "sha": hashlib.sha256(json.dumps(lanes).encode()).hexdigest()[:16],
            "elapsed_s_exploratory": round(secs, 2),
            "gdn_delta": {k: after[k] - before.get(k, 0) for k in after},
            "fallback_reasons_delta": {k: v - reasons0.get(k, 0) for k, v in reasons1.items()
                                       if v != reasons0.get(k, 0)},
        }

    def first_div(x, y):
        return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q),
                    None if len(x) == len(y) else min(len(x), len(y)))

    report = {
        "schema": SCHEMA, "model": a.model, "identity": getattr(adapter, "identity", None),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "src_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=ROOT,
                                                  text=True).strip()),
        "status": "running", "gpu_lock_owner": owner,
        "mlx": mx.__version__, "gen": a.gen, "temperature": 0.0, "copy_draft": copy_policy,
        "loaded_active_gib": loaded_gib, "default_compact": sorted(default_compact), "gdn_layers": len(gdn),
        "analytic": analytic_report(json.loads((Path(a.model).expanduser() / "config.json").read_text()),
                                    [int(getattr(adapter, "default_num_draft", 3)) + 1],
                                    a.widths),
    }
    cells = []
    report["cells"] = cells
    write_atomic(out_path, report)
    try:
        t0 = time.perf_counter()
        reference = [solo(i) for i in range(max(a.widths))]
        check_swap()
        report["ordinary_reference"] = {"lanes": reference,
                                        "lane0_text": tok.decode(reference[0]) if hasattr(tok, "decode") else None,
                                        "elapsed_s_exploratory": round(time.perf_counter() - t0, 1)}
        write_atomic(out_path, report)
        for batch in a.widths:
            configure(a.modes[0])
            run(batch)  # warm-up, discarded
            per_mode = {}
            for mode in a.modes:
                configure(mode)
                per_mode[mode] = run(batch)
                check_swap()
                r = per_mode[mode]
                print(f"B={batch} mode={mode} peak={r['peak_gib']:.2f}GiB (+{r['peak_above_base_gib']:.3f}) "
                      f"sha={r['sha']} gdn={json.dumps(r['gdn_delta'])}", flush=True)
            configure(a.modes[0])
            again = run(batch)
            check_swap()
            m0, m1 = a.modes[0], a.modes[1]
            cell = {
                "batch": batch, "runs": per_mode, "replicate_" + m0: again,
                "lanes_equal_between_modes": [x == y for x, y in zip(per_mode[m0]["lanes"], per_mode[m1]["lanes"])],
                "first_divergence_between_modes": [first_div(x, y) for x, y in
                                                   zip(per_mode[m0]["lanes"], per_mode[m1]["lanes"])],
                "replicate_equal": again["lanes"] == per_mode[m0]["lanes"],
                "vs_ordinary_reference": {
                    mode: {"equal": [x == y for x, y in zip(r["lanes"], reference)],
                           "first_divergence": [first_div(y, x) for x, y in zip(r["lanes"], reference)]}
                    for mode, r in per_mode.items()},
                "peak_above_base_gib": {mode: r["peak_above_base_gib"] for mode, r in per_mode.items()},
                "compact_engaged": per_mode.get("compact", {}).get("gdn_delta", {}).get("compact_verify_calls", 0) > 0,
            }
            cell["peak_above_base_gib"]["replicate_" + m0] = again["peak_above_base_gib"]
            cells.append(cell)
            report["swapouts_delta_pages"] = swapouts() - swap0
            write_atomic(out_path, report)
            print("CELL", json.dumps({k: v for k, v in cell.items() if k not in ("runs", "replicate_" + m0)}),
                  flush=True)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "aborted"
        report["abort"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        restore_modes()
        report["swapouts_delta_pages"] = swapouts() - swap0
        report["process_peak_gib"] = mx.get_peak_memory() / 2**30
        write_atomic(out_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
