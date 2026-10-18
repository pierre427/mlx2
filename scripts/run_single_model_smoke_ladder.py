#!/usr/bin/env python3
"""Run one ordinary reference smoke and one thermally controlled ladder.

The caller must hold both shared GPU locks and the CPG lease.  This runner
loads only one model at a time, always terminates its loopback server, and
refuses to overwrite evidence. ``--runs 1`` is intentionally exploratory.
The explicit ``--qualification-ladder`` mode instead requires three measured
runs per cell; it provides ladder evidence but does not by itself issue a
qualification receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
LADDER = ROOT / "benchmark_results/2026-09-29/scripts/series/thermal_ladder.py"
THERMAL_PROBE = ROOT / "scripts/thermal_probe.swift"


def read_owner(path: Path) -> dict:
    owner = path / "owner.json"
    if not path.is_dir() or not owner.is_file():
        raise RuntimeError(f"GPU lock is not owned: {path}")
    return json.loads(owner.read_text())


def validate_ownership(session: str, label: str) -> dict:
    owners = [read_owner(path) for path in LOCKS]
    if owners[0] != owners[1]:
        raise RuntimeError("shared GPU lock owner receipts differ")
    owner = owners[0]
    if owner.get("session") != session or owner.get("label") != label:
        raise RuntimeError(
            f"GPU owner mismatch: expected {session}/{label}, got "
            f"{owner.get('session')}/{owner.get('label')}"
        )
    return owner


def validate_campaign_ownership(
    session: str,
    label: str,
    *,
    cpg_task: str,
    cpg_owner_lock: Path,
    cpg_generation: int,
) -> dict:
    """Bind matching paired locks to a live ancestor-owned CPG radio lease."""
    paired = validate_ownership(session, label)
    from qualification_gpu_ownership import require_qualification_lease

    lease = require_qualification_lease(
        task_id=cpg_task,
        cpg_owner_lock=cpg_owner_lock.expanduser().resolve(strict=True),
        generation=cpg_generation,
    )
    return {"paired_lock_owner": paired, **lease}


def campaign_locks_still_owned(session: str, label: str, owner: dict) -> bool:
    """Revalidate the paired-lock portion of the combined ownership receipt."""
    paired = owner.get("paired_lock_owner") if isinstance(owner, dict) else None
    return isinstance(paired, dict) and validate_ownership(session, label) == paired


def thermal_sample() -> dict:
    result = subprocess.run(
        ["/usr/bin/swift", str(THERMAL_PROBE)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    sample = json.loads(result.stdout)
    if not isinstance(sample, dict) or type(sample.get("thermal_state")) is not int:
        raise RuntimeError("thermal probe returned a malformed record")
    return sample


def wait_health(base: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during load with rc={process.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=3) as response:
                health = json.load(response)
            if health.get("status") == "ok":
                with urllib.request.urlopen(base + "/v1/status", timeout=10) as response:
                    return json.load(response)
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy within {timeout:g}s")


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_startup_status(
    status: dict,
    *,
    expected_route: str = "ordinary",
    expected_artifact: str | None,
    expected_profile: str | None,
    expected_cache_layout: str | None,
) -> None:
    """Bind a campaign to the requested artifact/profile/layout and route."""
    execution = status.get("execution") or {}
    expected = {
        "artifact": (status.get("artifact"), expected_artifact),
        "profile": (status.get("profile"), expected_profile),
        "cache layout": (
            execution.get("cache_layout", execution.get("layout")),
            expected_cache_layout,
        ),
    }
    for label, (observed, wanted) in expected.items():
        if wanted is not None and observed != wanted:
            raise RuntimeError(f"startup {label} mismatch: expected {wanted!r}, got {observed!r}")
    settings = status.get("settings") or {}
    if settings.get("route") != expected_route or settings.get("mtp") is not False:
        raise RuntimeError(
            f"startup did not select {expected_route} non-MTP route: {settings!r}"
        )
    if status.get("qualification") != "candidate":
        raise RuntimeError(
            "qualification-mode startup must publish qualification='candidate'"
        )


def ladder_semantics(*, qualification_ladder: bool, runs: int) -> str:
    if qualification_ladder:
        if runs != 3:
            raise ValueError("qualification ladder requires exactly three runs per cell")
        return (
            "three thermally admitted runs per cell; qualification-ladder evidence "
            "requiring a separate qualification verdict"
        )
    if runs != 1:
        raise ValueError("exploratory ladder requires exactly one run per cell")
    return "one exploratory run per cell; not thermally replicated qualification"


def server_capacity(*, wide: int, max_inflight: int | None) -> tuple[int, int]:
    lanes = max(1, wide)
    inflight = lanes if max_inflight is None else max_inflight
    if inflight < lanes:
        raise ValueError("--max-inflight must be at least --wide")
    return lanes, inflight


def load_execution_policy(
    path: Path | None,
    *,
    route: str,
    expected_fidelity: str | None,
    require_ast: bool = False,
    require_apt: bool | None = None,
) -> dict | None:
    """Validate campaign-level external policy requirements before model load."""
    if require_apt is not None:
        require_ast = bool(require_apt)
    if route == "ordinary":
        if path is not None or expected_fidelity is not None or require_ast:
            raise ValueError(
                "external policy, expected fidelity, and AST assertions require "
                "--route external_draft"
            )
        return None
    if route != "external_draft":
        raise ValueError("unsupported campaign route")
    if path is None:
        raise ValueError("--route external_draft requires --execution-policy")
    policy = json.loads(path.expanduser().resolve(strict=True).read_text())
    if not isinstance(policy, dict):
        raise TypeError("--execution-policy must contain a JSON object")
    fidelity = policy.get("progressive_verification_fidelity", "exact")
    if expected_fidelity is None:
        raise ValueError("external_draft requires --expected-fidelity")
    if fidelity != expected_fidelity:
        raise ValueError(
            f"execution policy fidelity is {fidelity!r}, expected {expected_fidelity!r}"
        )
    if require_ast:
        tile = policy.get("progressive_verification_tile")
        if type(tile) is not int or tile < 1:
            raise ValueError("AST requires a positive progressive verification tile")
        required = {
            "progressive_alternative_strategy": "confidence_tree",
            "progressive_cross_session_pack_mode": "ordered_live_frontier",
        }
        for key, wanted in required.items():
            if policy.get(key) != wanted:
                raise ValueError(f"AST requires {key}={wanted!r}")
        branches = policy.get("progressive_alternative_branches")
        if type(branches) is not int or not 2 <= branches <= 4:
            raise ValueError("AST requires two through four alternative branches")
        multilane_cap = policy.get("progressive_multilane_draft_cap")
        if multilane_cap is not None and (
            type(multilane_cap) is not int or multilane_cap <= tile
        ):
            raise ValueError(
                "AST requires progressive_multilane_draft_cap to exceed "
                "progressive_verification_tile"
            )
        if policy.get("progressive_cross_session_pack_rows") not in {4, 8}:
            raise ValueError("AST requires a four- or eight-row cross-session pack")
    return policy


def _response_speculation_receipts(report: dict):
    for cell in report.get("cells") or ():
        rows = [cell.get("warmup"), *(cell.get("runs") or ())]
        for row in rows:
            if not isinstance(row, dict):
                continue
            requests = row.get("requests") or {}
            for arm in ("cold", "warm"):
                for response in (requests.get(arm) or {}).get("rows") or ():
                    speculation = (response.get("receipt") or {}).get("speculation")
                    if isinstance(speculation, dict):
                        yield speculation


def validate_ast_report(
    report: dict,
    *,
    expected_fidelity: str,
    expected_width: int,
    expected_branches: int,
    expected_pack_rows: int,
) -> dict:
    """Require physical cross-session AST use and its source-bound receipts."""
    scheduler = (report.get("final") or {}).get("scheduler") or {}
    minimums = {
        "external_progressive_verify_max_cohort_width": expected_width,
        "external_progressive_alternative_rounds": 1,
        "external_progressive_alternative_branches": 1,
        "external_progressive_cross_session_ast_rounds": 1,
        "external_progressive_cross_session_ast_sessions": expected_width,
        "external_progressive_cross_session_ast_packs": 1,
        "external_progressive_ast_frontier_rows_admitted": 1,
    }
    failures = []
    for key, floor in minimums.items():
        value = scheduler.get(key)
        if type(value) is not int or value < floor:
            failures.append(f"{key} must be at least {floor}, got {value!r}")
    observation_failures = scheduler.get(
        "external_progressive_topology_observation_failures"
    )
    if observation_failures != 0:
        failures.append(
            "external_progressive_topology_observation_failures must be zero, "
            f"got {observation_failures!r}"
        )

    matched_receipts = 0
    receipt_failures = []
    for speculation in _response_speculation_receipts(report):
        progressive = speculation.get("progressive_verification") or {}
        alternatives = progressive.get("ordered_alternatives") or {}
        packing = alternatives.get("cross_session_ast_packing") or {}
        if not packing.get("observed_used"):
            continue
        matched_receipts += 1
        if progressive.get("fidelity") != expected_fidelity:
            receipt_failures.append("progressive fidelity mismatch")
        if progressive.get("observed_used") is not True:
            receipt_failures.append("progressive verification was not observed")
        if alternatives.get("mechanism_id") != "adaptive_speculation_topology":
            receipt_failures.append("AST mechanism identity is absent")
        if alternatives.get("configured_branches") != expected_branches:
            receipt_failures.append("AST configured branch count mismatch")
        if not isinstance(alternatives.get("rounds"), int) or alternatives["rounds"] < 1:
            receipt_failures.append("AST alternative rounds were not observed")
        if packing.get("configured_pack_rows") != expected_pack_rows:
            receipt_failures.append("AST configured pack width mismatch")
        if packing.get("configured_mode") != "ordered_live_frontier":
            receipt_failures.append("AST receipt has the wrong configured packing mode")
        if packing.get("mode") != "ordered_live_frontier":
            receipt_failures.append("AST receipt has the wrong observed packing mode")
    if matched_receipts == 0:
        failures.append("no response receipt observed cross-session AST packing")
    failures.extend(sorted(set(receipt_failures)))
    return {
        "schema": "mlx2.single-model-ast-validation.v1",
        "passed": not failures,
        "expected_fidelity": expected_fidelity,
        "expected_width": expected_width,
        "expected_branches": expected_branches,
        "expected_pack_rows": expected_pack_rows,
        "matched_response_receipts": matched_receipts,
        "scheduler": {
            key: scheduler.get(key)
            for key in (
                *minimums,
                "external_progressive_topology_observation_failures",
            )
        },
        "failures": failures,
    }


# Deprecated callable retained for scripts importing the interim name.
validate_apt_report = validate_ast_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--route", choices=("ordinary", "external_draft"), default="ordinary"
    )
    parser.add_argument("--execution-policy", type=Path)
    parser.add_argument(
        "--expected-fidelity", choices=("exact", "near_tie_bounded")
    )
    parser.add_argument(
        "--require-ast",
        dest="require_ast",
        action="store_true",
        help="fail unless cross-session Adaptive Speculation Topology is observed",
    )
    parser.add_argument(
        "--require-apt",
        dest="require_ast",
        action="store_true",
        help="deprecated alias for --require-ast",
    )
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cpg-task", required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument("--cpg-generation", type=int, required=True)
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--smoke-max-tokens", type=int, default=16)
    parser.add_argument("--wide", type=int, default=4)
    parser.add_argument("--wide-max-context", type=int, default=32768)
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--cache-gib", type=int, default=4)
    parser.add_argument(
        "--max-inflight",
        type=int,
        help="server request capacity; defaults to --wide",
    )
    parser.add_argument(
        "--memory-only-cache",
        action="store_true",
        help="omit --cache-dir so the isolated APCv2 cache remains memory-only",
    )
    parser.add_argument("--expected-artifact")
    parser.add_argument("--expected-profile")
    parser.add_argument("--expected-cache-layout")
    parser.add_argument("--top-logprobs", type=int, default=0)
    parser.add_argument("--logprobs-token", type=int, default=0)
    parser.add_argument("--verify-bitexact", action="store_true")
    parser.add_argument(
        "--qualification-ladder",
        action="store_true",
        help="require three measured runs per cell and label the evidence accordingly",
    )
    args = parser.parse_args(argv)
    try:
        execution_policy = load_execution_policy(
            args.execution_policy,
            route=args.route,
            expected_fidelity=args.expected_fidelity,
            require_ast=args.require_ast,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        parser.error(str(error))
    try:
        semantics = ladder_semantics(
            qualification_ladder=args.qualification_ladder,
            runs=args.runs,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.require_ast and args.wide < 2:
        parser.error("--require-ast requires --wide of at least two")
    if not 1 <= args.max_tokens <= 128:
        parser.error("--max-tokens must be 1..128")
    if not 1 <= args.smoke_max_tokens <= 16:
        parser.error("--smoke-max-tokens must be 1..16")
    if args.max_context < 1024 or not 1024 <= args.max_length <= args.max_context:
        parser.error("context bounds must admit at least the 1K rung")
    if args.cache_gib <= 0:
        parser.error("--cache-gib must be positive")
    if not 0 <= args.top_logprobs <= 20:
        parser.error("--top-logprobs must be 0..20")
    if args.logprobs_token < 0:
        parser.error("--logprobs-token must be nonnegative")
    if args.logprobs_token and not args.top_logprobs:
        parser.error("--logprobs-token requires --top-logprobs")
    try:
        max_lanes, max_inflight = server_capacity(
            wide=args.wide,
            max_inflight=args.max_inflight,
        )
    except ValueError as error:
        parser.error(str(error))

    model = args.model.expanduser().resolve(strict=True)
    out = args.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"refusing to overwrite nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.cpg_generation < 1:
        parser.error("--cpg-generation must be positive")
    owner = validate_campaign_ownership(
        args.session,
        args.label,
        cpg_task=args.cpg_task,
        cpg_owner_lock=args.cpg_owner_lock,
        cpg_generation=args.cpg_generation,
    )
    initial_thermal = thermal_sample()
    if initial_thermal["thermal_state"] not in {0, 1}:
        parser.error(
            "campaign reference smoke requires nominal or fair initial thermal state"
        )
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            parser.error(f"port {args.port} is already occupied")

    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_dirty = bool(subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip())
    campaign = {
        "schema": "mlx2.single-model-smoke-ladder.v1",
        "status": "running",
        "semantics": semantics,
        "source_revision": revision,
        "source_tracked_dirty": tracked_dirty,
        "model": str(model),
        "model_name": args.model_name,
        "route": args.route,
        "execution_policy_path": (
            None
            if args.execution_policy is None
            else str(args.execution_policy.expanduser().resolve())
        ),
        "execution_policy": execution_policy,
        "expected_fidelity": args.expected_fidelity,
        "require_ast": args.require_ast,
        "max_context": args.max_context,
        "max_length": args.max_length,
        "runs_per_cell": args.runs,
        "qualification_ladder": args.qualification_ladder,
        "smoke_max_tokens": args.smoke_max_tokens,
        "cache_bytes": args.cache_gib << 30,
        "max_lanes": max_lanes,
        "max_inflight": max_inflight,
        "memory_only_cache": args.memory_only_cache,
        "top_logprobs": args.top_logprobs,
        "logprobs_token": args.logprobs_token,
        "verify_bitexact": args.verify_bitexact,
        "owner": owner,
        "initial_thermal": initial_thermal,
        "started_at": time.time(),
    }
    atomic_json(out / "campaign.json", campaign)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX2_CAMPAIGN_ROOT": str(ROOT),
    }
    smoke_command = [
        sys.executable, str(ROOT / "scripts/smoke_local_model.py"),
        "--model", str(model), "--out", str(out / "ordinary-smoke.json"),
        "--max-tokens", str(args.smoke_max_tokens), "--i-own-gpu",
        "--cpg-lease", f"cpg:{args.session}:{args.label}",
    ]
    server_command = [
        sys.executable, "-u", "-m", "mlx2.server", "--model", str(model),
        "--host", "127.0.0.1", "--port", str(args.port),
        f"--{args.route.replace('_', '-')}", "--qualification-mode",
        "--max-context", str(args.max_context),
        "--max-lanes", str(max_lanes), "--max-inflight", str(max_inflight),
        "--cache-bytes", str(args.cache_gib << 30),
    ]
    if args.execution_policy is not None:
        server_command += [
            "--execution-policy", str(args.execution_policy.expanduser().resolve())
        ]
    if not args.memory_only_cache:
        server_command += ["--cache-dir", str(out / "apcv2-cache")]
    if args.verify_bitexact:
        server_command.append("--verify-bitexact")
    base = f"http://127.0.0.1:{args.port}"
    server = None
    rc = 1
    try:
        with (out / "ordinary-smoke.log").open("w") as log:
            subprocess.run(smoke_command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=args.load_timeout)
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(server_command, cwd=ROOT, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            status = wait_health(base, server, args.load_timeout)
            validate_startup_status(
                status,
                expected_route=args.route,
                expected_artifact=args.expected_artifact,
                expected_profile=args.expected_profile,
                expected_cache_layout=args.expected_cache_layout,
            )
            atomic_json(out / "server-startup.json", {
                "command": server_command,
                "pid": server.pid,
                "status": status,
            })
            ladder_command = [
                sys.executable, str(LADDER), "--url", base,
                "--output", str(out / "ladder.json"), "--model", args.model_name,
                "--model-id", model.name, "--route", args.route,
                "--max-context", str(args.max_context), "--max-length", str(args.max_length),
                "--server-pid", str(server.pid), "--runs", str(args.runs),
                "--max-tokens", str(args.max_tokens), "--wide", str(args.wide),
                "--wide-max-context", str(args.wide_max_context),
            ]
            if args.top_logprobs:
                ladder_command += ["--top-logprobs", str(args.top_logprobs)]
            if args.logprobs_token:
                ladder_command += ["--logprobs-token", str(args.logprobs_token)]
            if args.verify_bitexact:
                ladder_command.append("--verify-bitexact")
            with (out / "ladder.log").open("w") as ladder_log:
                completed = subprocess.run(
                    ladder_command,
                    cwd=ROOT,
                    env=env,
                    stdout=ladder_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            rc = completed.returncode
            if args.require_ast and (out / "ladder.json").is_file():
                ast_validation = validate_ast_report(
                    json.loads((out / "ladder.json").read_text()),
                    expected_fidelity=args.expected_fidelity,
                    expected_width=args.wide,
                    expected_branches=execution_policy[
                        "progressive_alternative_branches"
                    ],
                    expected_pack_rows=execution_policy[
                        "progressive_cross_session_pack_rows"
                    ],
                )
                atomic_json(out / "ast-validation.json", ast_validation)
                if not ast_validation["passed"]:
                    rc = 1
            campaign["server_command"] = server_command
            campaign["ladder_command"] = ladder_command
            campaign["ladder_returncode"] = rc
            campaign["status"] = "passed" if rc == 0 else "failed"
    except BaseException as error:  # preserve evidence and cleanup on every exit
        campaign["status"] = "failed"
        campaign["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop_server(server)
        campaign["finished_at"] = time.time()
        campaign["locks_still_owned_before_return"] = campaign_locks_still_owned(
            args.session, args.label, owner
        )
        atomic_json(out / "campaign.json", campaign)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
