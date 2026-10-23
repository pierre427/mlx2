#!/usr/bin/env python3
"""Plan or run the controlled Nemotron AST pack-mode A/B campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXACT_WRAPPER = ROOT / "scripts/thermal_ladder_exact_contexts.py"
THERMAL_LADDER = ROOT / "benchmark_results/2026-09-29/scripts/series/thermal_ladder.py"
CONTEXTS = (1024, 16384)
ARM_ORDER = (
    "ordered_live_frontier", "eager", "eager",
    "ordered_live_frontier", "ordered_live_frontier", "eager",
)
CONTENT_SCOPES = ("src", "scripts", "benchmark_results/2026-09-29/scripts/series")


class UnitInterrupted(KeyboardInterrupt):
    pass


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def worktree_content_manifest() -> dict:
    listing = subprocess.check_output(
        [
            "git", "-C", str(ROOT), "ls-files", "--cached", "--others",
            "--exclude-standard", "--", *CONTENT_SCOPES,
        ],
        text=True,
    ).splitlines()
    files = []
    for relative in sorted(set(listing)):
        path = ROOT / relative
        if path.is_file():
            files.append({"path": relative, "sha256": sha256(path)})
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "mlx2.worktree-content-binding.v1",
        "scopes": list(CONTENT_SCOPES),
        "files": files,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def source_binding(
    expected_head: str,
    expected_branch: str,
    expected_diff: str,
    expected_content: str | None = None,
) -> dict:
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    patch = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary", "HEAD"])
    diff = hashlib.sha256(patch).hexdigest()
    content = worktree_content_manifest()
    if (head, branch, diff) != (expected_head, expected_branch, expected_diff):
        raise RuntimeError(
            f"source binding changed: head={head} branch={branch} diff_sha256={diff}"
        )
    if expected_content is not None and content["sha256"] != expected_content:
        raise RuntimeError(
            "worktree content binding changed: "
            f"{content['sha256']} != {expected_content}"
        )
    return {
        "head": head, "branch": branch, "diff_sha256": diff,
        "worktree_content": content,
    }


def _near_tie_pending(cell: dict) -> bool:
    """Recognize an unclassified width>1 replay mismatch, never call it passing."""
    if cell.get("width") != 4 or cell.get("error") is not None:
        return False
    runs = cell.get("runs") or []
    if len(runs) != 1:
        return False
    row = runs[0]
    summary = row.get("summary") or {}
    return (
        row.get("contaminated") is False
        and summary.get("streams_done") is True
        and summary.get("needle_correct") == summary.get("needle_total") == 8
        and summary.get("warm_apc_hits") == 4
        and type(summary.get("warm_equals_cold")) is int
        and summary["warm_equals_cold"] < 4
    )


def _validate_owner(value: object, session: str, label: str, where: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{where}: paired-lock ownership receipt absent"]
    before, after = value.get("before"), value.get("after")
    if not isinstance(before, dict) or before != after:
        return [f"{where}: paired-lock owner changed during the row"]
    errors = []
    if before.get("session") != session or before.get("label") != label:
        errors.append(f"{where}: paired-lock session/label differs")
    if not isinstance(before.get("lease_id"), str) or not before["lease_id"]:
        errors.append(f"{where}: paired-lock lease_id is absent")
    return errors


def _validate_request_shape(request: object, where: str) -> tuple[list[str], list[dict]]:
    if not isinstance(request, dict):
        return [f"{where}: request batch is absent"], []
    rows = request.get("rows")
    errors = []
    if not isinstance(rows, list) or len(rows) != 4:
        return [f"{where}: expected exactly four response rows"], []
    cohort = request.get("batch_cohort") or {}
    if cohort.get("size") != 4 or not isinstance(cohort.get("id"), str):
        errors.append(f"{where}: declared batch cohort is not size four")
    for index, row in enumerate(rows):
        if (
            row.get("done") is not True
            or type(row.get("prompt_tokens")) is not int
            or row["prompt_tokens"] <= 0
            or type(row.get("completion_tokens")) is not int
            or row["completion_tokens"] <= 0
        ):
            errors.append(f"{where}/{index}: prompt/completion geometry is incomplete")
    return errors, rows


def validate_ladder(
    ladder: dict,
    mode: str,
    *,
    session: str,
    label: str,
    main_rc: int,
    route_validation: dict,
) -> dict:
    errors: list[str] = []
    pending: list[str] = []
    cells = ladder.get("cells") or []
    if tuple(cell.get("requested_tokens") for cell in cells) != CONTEXTS:
        errors.append("context schedule differs from exact 1K/16K design")
    if ladder.get("runs_per_cell") != 1:
        errors.append("unit must contain one measured repetition per context")
    if route_validation.get("passed") is not True:
        errors.append("main harness route-receipt validation did not pass")

    receipts = exact_launches = retirement_receipts = 0
    max_service_gap = None
    primary_metric = {}
    measured_pack_contexts: set[int] = set()
    for cell in cells:
        context = cell.get("requested_tokens")
        prefix = f"{context}"
        near_tie = _near_tie_pending(cell)
        if cell.get("width") != 4:
            errors.append(f"{prefix}: width is not four")
        if cell.get("error") is not None:
            errors.append(f"{prefix}: cell error is non-null")
        if len(cell.get("runs") or []) != 1:
            errors.append(f"{prefix}: expected exactly one measured run")
        if cell.get("passed") is not True:
            if near_tie:
                pending.append(
                    f"{prefix}: cold/warm mismatch awaits bounded first-divergence classification"
                )
            else:
                errors.append(f"{prefix}: cell did not pass")
        for phase, row in [
            ("warmup", cell.get("warmup")),
            *[(f"run{index}", value) for index, value in enumerate(cell.get("runs") or ())],
        ]:
            where = f"{prefix} {phase}"
            if not isinstance(row, dict):
                errors.append(f"{where}: row is absent")
                continue
            errors.extend(_validate_owner(row.get("gpu_lock_owner"), session, label, where))
            if row.get("width") != 4 or len(row.get("calibrated_tokens") or []) != 4:
                errors.append(f"{where}: calibrated B4 geometry differs")
            if row.get("contaminated") is not False or row.get("contamination") != []:
                errors.append(f"{where}: contaminated")
            if phase.startswith("run"):
                if (row.get("thermal_pre") or {}).get("stable") is not True:
                    errors.append(f"{where}: thermal admission was not stable")
                if (row.get("thermal_post") or {}).get("breached") is not False:
                    errors.append(f"{where}: post-run thermal gate breached")
                primary_metric[str(context)] = (
                    ((row.get("requests") or {}).get("cold") or {}).get(
                        "aggregate_completion_tokens_per_second"
                    )
                )
            row_pack_receipts = 0
            for arm in ("cold", "warm"):
                shape_errors, response_rows = _validate_request_shape(
                    (row.get("requests") or {}).get(arm), f"{where}/{arm}"
                )
                errors.extend(shape_errors)
                for index, response in enumerate(response_rows):
                    location = f"{where}/{arm}/{index}"
                    receipt = response.get("receipt") or {}
                    if receipt.get("route") != "external_draft":
                        errors.append(f"{location}: route is not external_draft")
                        continue
                    receipts += 1
                    progressive = (
                        (receipt.get("speculation") or {}).get(
                            "progressive_verification"
                        )
                        or {}
                    )
                    executor = progressive.get("target_row_executor") or {}
                    launches = executor.get("launches")
                    if (
                        type(launches) is not int
                        or launches < 1
                        or executor.get("adapter_owned_launches") != launches
                        or executor.get("row_exact_launches") != launches
                        or executor.get("legacy_launches") != 0
                        or executor.get("nonexact_launches") != 0
                    ):
                        errors.append(
                            f"{location}: target executor is not adapter-owned row-exact"
                        )
                    else:
                        exact_launches += launches
                    packing = (
                        (progressive.get("ordered_alternatives") or {}).get(
                            "cross_session_ast_packing"
                        )
                        or {}
                    )
                    if packing.get("configured_pack_rows") != 8:
                        errors.append(f"{location}: AST pack cap differs")
                    observed_pack = packing.get("observed_used") is True
                    expected_mode = (
                        "ordered_live_frontier"
                        if mode == "ordered_live_frontier"
                        else "eager_all_branches"
                    )
                    if observed_pack and packing.get("mode") != expected_mode:
                        errors.append(f"{location}: AST pack mode receipt differs")
                    frontier = packing.get("live_frontier")
                    if mode == "ordered_live_frontier":
                        frontier_required = observed_pack or isinstance(frontier, dict)
                        frontier_invalid = not isinstance(frontier, dict) or (
                            frontier.get("finished") is not True
                            or type(frontier.get("sessions_retired")) is not int
                            or frontier["sessions_retired"] < 1
                            or type(frontier.get("max_service_gap")) is not int
                        )
                        if frontier_required and frontier_invalid:
                            errors.append(
                                f"{location}: adaptive queue-retirement receipt missing"
                            )
                        elif isinstance(frontier, dict):
                            retirement_receipts += 1
                            gap = frontier["max_service_gap"]
                            max_service_gap = (
                                gap if max_service_gap is None else max(max_service_gap, gap)
                            )
                    elif frontier is not None:
                        errors.append(
                            f"{location}: eager control unexpectedly used a live frontier"
                        )
                    if observed_pack:
                        row_pack_receipts += 1
            if phase.startswith("run") and row_pack_receipts:
                measured_pack_contexts.add(context)

    if not measured_pack_contexts:
        errors.append("cross-session AST packing was not observed in any measured cell")

    scheduler = (ladder.get("final") or {}).get("scheduler") or {}
    retired = scheduler.get("external_progressive_ast_frontier_sessions_retired", 0)
    scheduler_gap = scheduler.get("external_progressive_ast_frontier_max_service_gap", 0)
    if mode == "ordered_live_frontier" and (
        type(retired) is not int or retired < 1 or type(scheduler_gap) is not int
    ):
        errors.append("final scheduler adaptive queue-retirement counters are absent")
    if mode == "eager" and (retired != 0 or scheduler_gap != 0):
        errors.append("eager control published live-frontier scheduler counters")
    if not pending:
        if ladder.get("passed") is not True:
            errors.append("ladder did not pass")
        if main_rc != 0:
            errors.append(f"main harness returned {main_rc}")
    elif main_rc not in (0, 1):
        errors.append(f"pending near-tie run had unexpected main harness rc={main_rc}")
    status = "failed" if errors else (
        "pending_bounded_first_divergence" if pending else "passed"
    )
    return {
        "schema": "mlx2.nemotron-ast-pack-ab-unit-validation.v2",
        "status": status,
        "passed": status == "passed",
        "correctness_failed": bool(errors),
        "performance_evidence_usable": status == "passed",
        "mode": mode,
        "contexts": list(CONTEXTS),
        "primary_metric": {
            "name": "cold aggregate completion tokens per second",
            "values_by_context": primary_metric,
        },
        "response_receipts": receipts,
        "row_exact_target_launches": exact_launches,
        "queue_retirement_receipts": retirement_receipts,
        "measured_pack_contexts": sorted(measured_pack_contexts),
        "max_service_gap": max_service_gap,
        "scheduler_sessions_retired": retired,
        "scheduler_max_service_gap": scheduler_gap,
        "main_harness_returncode": main_rc,
        "route_receipt_validation_passed": route_validation.get("passed"),
        "pending_classification": pending,
        "errors": sorted(set(errors)),
    }


def unit_command(args, unit: dict) -> list[str]:
    unit_dir = Path(unit["output"])
    python = str(args.venv / "bin/python")
    child = [
        "/usr/bin/env", f"PYTHONPATH={ROOT / 'src'}", python,
        str(Path(__file__).resolve()), "run-unit",
        "--main-harness", str(args.main_harness),
        "--b4-wrapper", str(args.b4_wrapper),
        "--model", str(args.model), "--head", str(args.head),
        "--venv", str(args.venv), "--out", str(unit_dir / "run"),
        "--cpg-owner-lock", str(unit_dir / "cpg-owner"),
        "--session", unit["session"], "--label", unit["label"],
        "--mode", unit["mode"], "--expected-head", args.expected_head,
        "--expected-branch", args.expected_branch,
        "--expected-diff-sha256", args.expected_diff_sha256,
        "--expected-worktree-content-sha256", args.expected_worktree_content_sha256,
        "--expected-coordinator-sha256", sha256(Path(__file__)),
        "--expected-exact-wrapper-sha256", sha256(EXACT_WRAPPER),
        "--expected-main-harness-sha256", sha256(args.main_harness),
        "--expected-b4-wrapper-sha256", sha256(args.b4_wrapper),
        "--expected-thermal-ladder-sha256", sha256(THERMAL_LADDER),
        "--port", str(unit["port"]),
    ]
    locked = [
        python, str(ROOT / "scripts/run_with_gpu_locks.py"),
        "--session", unit["session"], "--label", unit["label"],
        "--cpg-owner-lock", str(unit_dir / "cpg-owner"), "--cpg-task", args.cpg_task,
        "--receipt", str(unit_dir / "gpu-lock.json"), "--", *child,
    ]
    owned = [
        # The ownership wrapper talks to the CPG radio and therefore must run
        # in the CPG environment; the model and lock wrappers remain in the
        # fresh campaign environment.
        str(args.cpg_python), str(ROOT / "scripts/run_qualification_owned.py"),
        "--cpg-owner-lock", str(unit_dir / "cpg-owner"),
        "--cpg-task", args.cpg_task, "--timeout", str(args.timeout),
        "--receipt", str(unit_dir / "owned.json"), "--", *locked,
    ]
    return [
        str(args.cpg_python), str(args.cpg_job), "run", "--label", unit["label"],
        "--out", str(unit_dir / "cpg.log"), "--require-radio", "--lease",
        "--lock", "--lock-dir", str(unit_dir / "cpg-owner"), "--", *owned,
    ]


def make_plan(args) -> dict:
    binding = source_binding(
        args.expected_head, args.expected_branch, args.expected_diff_sha256
    )
    args.expected_worktree_content_sha256 = binding["worktree_content"]["sha256"]
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty evidence directory: {out}")
    for path in (
        args.venv / "bin/python", args.model, args.head, args.main_harness,
        args.b4_wrapper, args.cpg_python, args.cpg_job, EXACT_WRAPPER, THERMAL_LADDER,
    ):
        path.resolve(strict=True)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"eager": 0, "ordered_live_frontier": 0}
    units = []
    for index, mode in enumerate(ARM_ORDER):
        counts[mode] += 1
        short = "live" if mode == "ordered_live_frontier" else "eager"
        unit = {
            "index": index, "mode": mode, "arm_repetition": counts[mode],
            "contexts": list(CONTEXTS), "pack_rows": 8,
            "output": str(out / f"{index + 1:02d}-{short}-r{counts[mode]}"),
            "session": f"nemotron-ast-pack-ab-{index + 1:02d}-{short}",
            "label": f"nemotron-ast-pack8-{short}-r{counts[mode]}",
            "port": args.port_base + index,
        }
        unit["command"] = unit_command(args, unit)
        Path(unit["output"]).mkdir(parents=True, exist_ok=False)
        units.append(unit)
    return {
        "schema": "mlx2.nemotron-ast-pack-mode-ab-plan.v2",
        "status": "planned", "created_at": time.time(), "source": binding,
        "artifacts": {"model": str(args.model), "head": str(args.head)},
        "design": {
            "arms": ["eager", "ordered_live_frontier"],
            "arm_order": list(ARM_ORDER), "repetitions_per_arm": 3,
            "contexts": list(CONTEXTS), "batch_width": 4, "pack_rows": 8,
            "primary_metric": "cold aggregate completion tokens per second",
            "unit_scope": "one independent CPG/radio and paired-lock window",
            "context_override": {
                "mechanism": "frozen thermal module explicit LENGTHS and EXPLICIT_WIDTHS",
                "loaded_module_assertion_required": True,
                "legacy_dict_monkeypatch_restored_before main": True,
            },
        },
        "harnesses": {
            "coordinator": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            "exact_context_wrapper": {"path": str(EXACT_WRAPPER), "sha256": sha256(EXACT_WRAPPER)},
            "main": {"path": str(args.main_harness), "sha256": sha256(args.main_harness)},
            "b4": {"path": str(args.b4_wrapper), "sha256": sha256(args.b4_wrapper)},
            "thermal_ladder": {"path": str(THERMAL_LADDER), "sha256": sha256(THERMAL_LADDER)},
        },
        "units": units,
    }


def stage_unit_harness(main_harness: Path, b4_wrapper: Path, unit_out: Path) -> dict:
    shadow = unit_out.parent / "harness-shadow"
    shadow.mkdir(parents=False, exist_ok=False)
    paths = {
        "main": shadow / main_harness.name,
        "exact_wrapper": shadow / "thermal_ladder_ast_b4.py",
        "base_b4": shadow / "base_thermal_ladder_ast_b4.py",
        "thermal_source": shadow / "thermal_ladder_canonical.py",
        "thermal_module": shadow / "thermal_ladder_b4_exact.py",
    }
    shutil.copy2(main_harness, paths["main"])
    shutil.copy2(EXACT_WRAPPER, paths["exact_wrapper"])
    shutil.copy2(b4_wrapper, paths["base_b4"])
    shutil.copy2(THERMAL_LADDER, paths["thermal_source"])
    main_source = paths["main"].read_text()
    policy_needle = '        "layerwise_short_exact", "auto", "auto",\n'
    policy_replacement = (
        '        "layerwise_short_exact_routed_batch", "auto", "auto",\n'
    )
    if main_source.count(policy_needle) != 1:
        raise RuntimeError("Nemotron Mamba policy seam changed")
    paths["main"].write_text(main_source.replace(policy_needle, policy_replacement))
    source = paths["thermal_source"].read_text()
    needle = "for width in dict.fromkeys((1, args.wide)):"
    if source.count(needle) != 1:
        raise RuntimeError("canonical thermal ladder width seam changed")
    paths["thermal_module"].write_text(
        source.replace(needle, "for width in EXPLICIT_WIDTHS:")
    )
    manifest = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }
    manifest["transformation"] = [
        {
            "from": policy_needle.strip(),
            "to": policy_replacement.strip(),
            "purpose": "adapter-owned exact routed-expert batching",
        },
        {
            "from": needle,
            "to": "for width in EXPLICIT_WIDTHS:",
            "purpose": "explicit B4 selection without relying on a fake module-level dict",
        },
    ]
    atomic_json(shadow / "manifest.json", manifest)
    return {"root": shadow, "paths": paths, "manifest": manifest}


class SubprocessProxy:
    def __init__(self, lifecycle: dict, server_processes: list):
        self.lifecycle = lifecycle
        self.server_processes = server_processes

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, command, *args, **kwargs):
        process = subprocess.Popen(command, *args, **kwargs)
        words = [str(item) for item in command]
        if "mlx2.server" in words:
            record = {"pid": process.pid, "pgid": os.getpgid(process.pid), "command": words}
            self.lifecycle["servers"].append(record)
            self.server_processes.append(process)
        return process


def process_group_members(pgid: int) -> list[dict]:
    """Return non-zombie members without signalling a possibly reused PGID."""
    output = subprocess.check_output(
        ["ps", "-axo", "pid=,pgid=,state=,command="], text=True
    )
    members = []
    for line in output.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) < 3:
            continue
        try:
            pid, observed_pgid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        state = fields[2]
        if observed_pgid == pgid and not state.startswith("Z"):
            members.append(
                {
                    "pid": pid,
                    "state": state,
                    "command": fields[3] if len(fields) > 3 else "",
                }
            )
    return members


def process_group_alive(pgid: int) -> bool:
    return bool(process_group_members(pgid))


def stop_process_group(pgid: int, lifecycle: dict, *, term_wait: float = 6) -> None:
    action = {"pgid": pgid, "term_sent": False, "kill_sent": False}
    lifecycle["cleanup_actions"].append(action)
    if not process_group_alive(pgid):
        action["already_stopped"] = True
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
        action["term_sent"] = True
    except ProcessLookupError:
        return
    except PermissionError as error:
        action["signal_error"] = f"PermissionError: {error}"
        action["members_after_signal_error"] = process_group_members(pgid)
        return
    deadline = time.monotonic() + term_wait
    while process_group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_group_alive(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
            action["kill_sent"] = True
        except ProcessLookupError:
            action["exited_before_kill"] = True
        deadline = time.monotonic() + 3
        while process_group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.1)
    action["alive_after"] = process_group_alive(pgid)


def install_interrupt_handlers(lifecycle: dict) -> dict[int, object]:
    old_handlers = {}

    def interrupted(signum, _frame):
        lifecycle["signals"].append({"signal": signum, "at": time.time()})
        raise UnitInterrupted(f"signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    return old_handlers


def restore_interrupt_handlers(old_handlers: dict[int, object]) -> None:
    for signum, handler in old_handlers.items():
        signal.signal(signum, handler)


def run_unit(args) -> int:
    initial_binding = source_binding(
        args.expected_head, args.expected_branch, args.expected_diff_sha256,
        args.expected_worktree_content_sha256,
    )
    originals = {
        Path(__file__).resolve(): args.expected_coordinator_sha256,
        EXACT_WRAPPER: args.expected_exact_wrapper_sha256,
        args.main_harness: args.expected_main_harness_sha256,
        args.b4_wrapper: args.expected_b4_wrapper_sha256,
        THERMAL_LADDER: args.expected_thermal_ladder_sha256,
    }
    for path, expected in originals.items():
        if sha256(path) != expected:
            raise RuntimeError(f"harness source binding changed: {path}")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty evidence directory: {out}")
    staged = stage_unit_harness(args.main_harness, args.b4_wrapper, out)
    os.environ.update(
        MLX2_LADDER_CONTEXTS=",".join(map(str, CONTEXTS)),
        MLX2_THERMAL_LADDER_WRAPPER=str(staged["paths"]["base_b4"]),
        MLX2_THERMAL_LADDER_MODULE=str(staged["paths"]["thermal_module"]),
        MLX2_CONTEXT_OVERRIDE_RECEIPT=str(out / "context-override.json"),
    )
    lifecycle = {
        "schema": "mlx2.nemotron-ast-pack-ab-lifecycle.v1",
        "started_at": time.time(), "servers": [], "signals": [],
        "cleanup_actions": [], "source_before": initial_binding,
    }
    old_handlers = install_interrupt_handlers(lifecycle)

    spec = importlib.util.spec_from_file_location(
        "nemotron_main_ladder", staged["paths"]["main"]
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    server_processes = []
    module.subprocess = SubprocessProxy(lifecycle, server_processes)

    def bounded_stop(process):
        if process is not None:
            stop_process_group(process.pid, lifecycle)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                lifecycle.setdefault("reap_errors", []).append(process.pid)

    module.stop_group = bounded_stop
    sys.argv = [
        str(staged["paths"]["main"]), "--root", str(ROOT),
        "--venv", str(args.venv), "--model", str(args.model),
        "--head", str(args.head), "--out", str(out),
        "--cpg-owner-lock", str(args.cpg_owner_lock),
        "--session", args.session, "--label", args.label,
        "--topology", "adaptive", "--active-sessions", "4",
        "--cross-session-pack-rows", "8", "--cross-session-pack-mode", args.mode,
        "--expected-head", args.expected_head, "--expected-branch", args.expected_branch,
        "--expected-diff-sha256", args.expected_diff_sha256,
        "--min-length", "1024", "--max-length", "16384", "--port", str(args.port),
    ]
    rc = 1
    error = None
    try:
        rc = module.main()
    except BaseException as caught:  # noqa: BLE001 - cleanup before re-raise
        error = caught
        lifecycle["error"] = f"{type(caught).__name__}: {caught}"
    finally:
        for server in lifecycle["servers"]:
            try:
                stop_process_group(server["pgid"], lifecycle)
            except Exception as cleanup_error:  # noqa: BLE001 - keep receipt writable
                lifecycle.setdefault("cleanup_errors", []).append(
                    f"pgid {server['pgid']}: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        for process in server_processes:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                lifecycle.setdefault("reap_errors", []).append(process.pid)
        lifecycle["remaining_server_groups"] = [
            {"pgid": row["pgid"], "members": process_group_members(row["pgid"])}
            for row in lifecycle["servers"]
            if process_group_alive(row["pgid"])
        ]
        restore_interrupt_handlers(old_handlers)
        try:
            lifecycle["source_after"] = source_binding(
                args.expected_head, args.expected_branch, args.expected_diff_sha256,
                args.expected_worktree_content_sha256,
            )
            lifecycle["original_hashes_unchanged"] = all(
                sha256(path) == expected for path, expected in originals.items()
            )
            lifecycle["staged_hashes_unchanged"] = all(
                sha256(Path(record["path"])) == record["sha256"]
                for key, record in staged["manifest"].items()
                if key != "transformation"
            )
        except Exception as binding_error:  # noqa: BLE001 - preserve cleanup evidence
            lifecycle["binding_error"] = f"{type(binding_error).__name__}: {binding_error}"
        lifecycle["finished_at"] = time.time()
        lifecycle["passed"] = (
            not lifecycle["remaining_server_groups"]
            and not lifecycle.get("reap_errors")
            and not lifecycle.get("cleanup_errors")
            and lifecycle.get("original_hashes_unchanged") is True
            and lifecycle.get("staged_hashes_unchanged") is True
            and "binding_error" not in lifecycle
        )
        atomic_json(out.parent / "unit-lifecycle.json", lifecycle)
    if error is not None:
        raise error

    ladder_path = out / "ladder.json"
    route_path = out / "route-receipt-validation.json"
    ladder = json.loads(ladder_path.read_text()) if ladder_path.is_file() else {}
    route = json.loads(route_path.read_text()) if route_path.is_file() else {}
    validation = validate_ladder(
        ladder, args.mode, session=args.session, label=args.label,
        main_rc=rc, route_validation=route,
    )
    if lifecycle["passed"] is not True:
        validation["errors"].append("unit lifecycle or immutable binding did not pass")
        validation.update(status="failed", passed=False, correctness_failed=True,
                          performance_evidence_usable=False)
    override = out / "context-override.json"
    if not override.is_file() or json.loads(override.read_text()).get("passed") is not True:
        validation["errors"].append("context override loaded-module assertion is absent")
        validation.update(status="failed", passed=False, correctness_failed=True,
                          performance_evidence_usable=False)
    atomic_json(out / "ab-validation.json", validation)
    return 0 if validation["status"] in {"passed", "pending_bounded_first_divergence"} else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="operation", required=True)
    common = argparse.ArgumentParser(add_help=False)
    for name in ("main-harness", "model", "head", "venv", "out"):
        common.add_argument(f"--{name}", type=Path, required=True)
    common.add_argument("--expected-head", required=True)
    common.add_argument("--expected-branch", required=True)
    common.add_argument("--expected-diff-sha256", required=True)
    plan = sub.add_parser("plan", parents=[common])
    plan.add_argument("--b4-wrapper", type=Path, required=True)
    plan.add_argument("--cpg-python", type=Path, required=True)
    plan.add_argument("--cpg-job", type=Path, required=True)
    plan.add_argument("--cpg-task", required=True)
    plan.add_argument("--timeout", type=int, default=7200)
    plan.add_argument("--port-base", type=int, default=18990)
    unit = sub.add_parser("run-unit", parents=[common])
    unit.add_argument("--b4-wrapper", type=Path, required=True)
    unit.add_argument("--cpg-owner-lock", type=Path, required=True)
    unit.add_argument("--session", required=True)
    unit.add_argument("--label", required=True)
    unit.add_argument("--mode", choices=("eager", "ordered_live_frontier"), required=True)
    for name in (
        "worktree-content", "coordinator", "exact-wrapper", "main-harness",
        "b4-wrapper", "thermal-ladder",
    ):
        unit.add_argument(f"--expected-{name}-sha256", required=True)
    unit.add_argument("--port", type=int, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for name in ("main_harness", "model", "head", "venv", "out"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.b4_wrapper = args.b4_wrapper.expanduser().resolve()
    if args.operation == "run-unit":
        return run_unit(args)
    # Preserve the virtualenv entrypoint itself.  Resolving this symlink selects
    # the underlying uv-managed interpreter and drops the venv site-packages,
    # including the MCP client required by the CPG radio wrapper.
    args.cpg_python = args.cpg_python.expanduser().absolute()
    args.cpg_job = args.cpg_job.expanduser().resolve()
    plan = make_plan(args)
    atomic_json(args.out / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
