"""Stock8 baseline-window replay with the unchanged safe ownership/restoration lifecycle."""

import argparse
import hashlib
import json
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = Path(__file__).parent
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "scripts/research"),
]
import kimi_expert_budget_probe as budget
import run_with_gpu_locks as paired
import service_restore_protocol

from mlx2.desktop_campaign_lease import DesktopCPGLease
from mlx2.desktop_campaign_process import spawn
from mlx2.desktop_campaign_worker import atomic
from mlx2.desktop_runner import digest


def load(p):
    return json.loads(Path(p).read_text())


def bridge(out, reference, arm):
    rows = []
    for p in out.glob("*.*.json"):
        if not p.stem.rpartition(".")[2].isdigit():
            continue
        d = load(p)
        cid = d["case"]["case_id"]
        found = list(Path(reference).glob(arm + "-*/" + cid + ".json"))
        if len(found) != 1:
            raise RuntimeError("bridge reference identity not unique " + cid)
        expected = load(found[0])["attempts"][-1]["response"]["mlx2"][
            "probe_output_tokens"
        ]
        actual = d["attempts"][-1]["response"]["mlx2"]["probe_output_tokens"]
        pos = next(
            (i for i, (x, y) in enumerate(zip(expected, actual)) if x != y), None
        )
        rows.append(
            {
                "case_id": cid,
                "reference": str(found[0]),
                "exact_tokens": expected == actual,
                "first_divergence": pos,
            }
        )
    verdict = {
        "rows": rows,
        "passed": bool(rows) and all(x["exact_tokens"] for x in rows),
        "scope": "Bounded source/profile bridge only. Exact parity on these cases does not assert all400new-source outputs were observed.",
    }
    atomic(out / "bridge.json", verdict)
    return verdict["passed"]


def replay_validation(out, cases, baseline_root):
    rows = []
    for case in cases:
        cid = case["case_id"]
        found = list(Path(baseline_root).glob("default8-*/" + cid + ".json"))
        if len(found) != 1:
            raise RuntimeError("Baseline replay identity not unique: " + cid)
        expected = load(found[0])["attempts"][-1]["response"]["mlx2"][
            "probe_output_tokens"
        ]
        data = load(out / (cid + ".json"))
        windows = data["windows"]
        if [v["position_zero_based"] for v in windows] != case["replay_positions"]:
            raise RuntimeError("Selected replay positions changed: " + cid)
        for window in windows:
            pos = window["position_zero_based"]
            if pos >= len(expected):
                raise RuntimeError("Replay position outside reference: " + cid)
            actual = window["response"]["mlx2"].get("probe_output_tokens", [])
            prefix_verified = actual == expected[: pos + 1]
            if prefix_verified != window["replayed_prefix_and_selection"]:
                raise RuntimeError("Replay verification field inconsistent: " + cid)
            choices = window["response"].get("choices", [])
            logrows = (
                ((choices[0].get("logprobs") or {}).get("content") or [])
                if choices
                else []
            )
            bounded_logit = (
                len(logrows) == 1
                and logrows[0].get("id") == expected[pos]
                and isinstance(logrows[0].get("logprob"), (int, float))
                and math.isfinite(logrows[0]["logprob"])
                and 0 < len(logrows[0].get("top_logprobs", [])) <= 20
            )
            verified = prefix_verified and bounded_logit
            mismatch = next(
                (i for i, (x, y) in enumerate(zip(expected, actual)) if x != y), None
            )
            rows.append(
                {
                    "case_id": cid,
                    "position_zero_based": pos,
                    "verified": verified,
                    "prefix_verified": prefix_verified,
                    "bounded_selected_logit": bounded_logit,
                    "first_actual_token_divergence": mismatch,
                    "length_mismatch_without_token_divergence": mismatch is None
                    and not prefix_verified,
                }
            )
    return {
        "rows": rows,
        "verified": sum(r["verified"] for r in rows),
        "unclassified": sum(not r["verified"] for r in rows),
        "passed": bool(rows) and all(r["verified"] for r in rows),
        "scope": "Only previously selected positions. A source/profile prefix drift invalidates the requested logit window for root-cause interpretation; diagnose its earlier first divergence separately. No numerical-equivalence claim.",
    }


def main():
    from mlx2.adapters.kimi_linear import inspect_artifact

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--cases", type=Path)
    p.add_argument("--replay-only", action="store_true")
    p.add_argument("--baseline", type=Path)
    p.add_argument("--bridge-reference", type=Path)
    p.add_argument("--bridge-arm")
    p.add_argument("--long", action="store_true")
    p.add_argument("--frozen", type=Path)
    p.add_argument("--reference", type=Path)
    a = p.parse_args()
    if not a.replay_only or a.long or a.bridge_reference:
        p.error("This separate driver only accepts --replay-only corpus windows")
    if json.loads(a.candidate) != {"mode": "fixed", "k": 8}:
        p.error("Baseline replay must use the stock fixed8 policy")
    if a.baseline is None or a.cases is None:
        p.error("Baseline and preselected case positions are required")
    selected_cases = load(a.cases)
    if not selected_cases or len({c["case_id"] for c in selected_cases}) != len(
        selected_cases
    ):
        p.error("Replay cases must be nonempty and unique")
    for case in selected_cases:
        positions = case.get("replay_positions", [])
        if (
            not positions
            or positions != sorted(set(positions))
            or any(type(i) is not int or i < 0 for i in positions)
        ):
            p.error("Replay positions must be sorted unique nonnegative integers")
        references = list(a.baseline.glob("default8-*/" + case["case_id"] + ".json"))
        if len(references) != 1:
            p.error("Baseline case identity must be unique")
        reference = load(references[0])
        if any(
            case[k] != reference["case"][k] for k in ("case_id", "domain", "question")
        ):
            p.error("Replay case identity differs from the baseline")
        expected_ids = reference["attempts"][-1]["response"]["mlx2"][
            "probe_output_tokens"
        ]
        if positions[-1] >= len(expected_ids):
            p.error("Replay position lies outside the reference output")

    def interrupted(n, f):
        raise KeyboardInterrupt(str(n))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    # No claim or service mutation while another GPU owner/waiter exists.
    service_restore_protocol.admit()
    if any(path.is_dir() for path in paired.LOCKS) or any(paired.WAITERS.glob("*")):
        raise RuntimeError("GPU busy; yield for20-minute heartbeat")
    before = budget.http(8282, "/v1/status", timeout=5)
    if (
        before.get("state") != "ready"
        or before.get("inflight") != 0
        or before.get("queue_depth", 0) != 0
        or before.get("model") != budget.MODEL.name
    ):
        raise RuntimeError("original Kimi service not idle/ready")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidate = json.loads(a.candidate)
    artifact = inspect_artifact(budget.MODEL)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    checkpoint = budget.MODEL.name + "@" + artifact["identity"]["fingerprint"]
    source = budget.source_manifest(ROOT)
    spec = {
        "out": str(out),
        "job_id": out.name,
        "model": str(budget.MODEL),
        "checkpoint": checkpoint,
        "source_revision": revision,
        "candidate": candidate,
        "source_snapshot_sha256": digest(source),
        "benchmark_default_routes": not a.long,
        "port": 18324 if not a.long else 18326,
        "mcp_url": "http://127.0.0.1:8766/mcp",
        "resume_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if a.long:
        import loop_probe

        spec.update(
            frozen=str(a.frozen.resolve()),
            reference=str(a.reference.resolve()) if a.reference else None,
            max_context=65536,
            harness_sha256=loop_probe.harness(),
        )
        script = WORK / "loop_probe.py"
    else:
        spec.update(
            cases=load(a.cases),
            baseline=str(a.baseline.resolve()) if a.baseline else None,
            offset=0,
            replay_only=True,
        )
        script = ROOT / "scripts/research/kimi_domain_adaptive_sweep.py"
    spec["protected_restore_helper_sha256"] = {
        name: hashlib.sha256((WORK / name).read_bytes()).hexdigest()
        for name in ["service_restore_protocol.py"]
    }
    atomic(out / "spec.json", spec)
    atomic(out / "source.json", source)
    atomic(out / "service-before.json", before)
    snapshot = service_restore_protocol.capture(out / "private-service", checkpoint=checkpoint, source_revision=revision)
    plist_sha = hashlib.sha256(budget.PLIST.read_bytes()).hexdigest()
    suspended = False
    child = None
    rc = None
    try:
        # Check again immediately before stopping the authorized idle service.
        current = budget.http(8282, "/v1/status", timeout=5)
        if current.get("inflight") != 0 or current.get("queue_depth", 0) != 0:
            raise RuntimeError("Kimi became busy; yield")
        suspended = True  # Restore even if interruption lands during bootout.
        service_restore_protocol.suspend(snapshot)
        with DesktopCPGLease(
            out, spec["job_id"], checkpoint=checkpoint, source_revision=revision
        ) as lease:
            cmd = [
                sys.executable,
                str(ROOT / "scripts/run_with_gpu_locks.py"),
                "--session",
                lease.owner["cpg_session"],
                "--label",
                spec["job_id"],
                "--desktop-cpg-lease",
                str(lease.path),
                "--receipt",
                str(out / "paired-locks.json"),
                "--",
                sys.executable,
                "-u",
                str(script),
                "unit",
                "--output",
                str(out),
            ]
            with (out / "run.log").open("w") as log:
                child = spawn(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=ROOT,
                )
                deadline = None
                try:
                    while child.poll() is None:
                        lease.check()
                        try:
                            locked = (
                                load(out / "paired-locks.json").get("status")
                                == "running"
                            )
                        except (FileNotFoundError, json.JSONDecodeError):
                            locked = False
                        if locked and deadline is None:
                            deadline = time.monotonic() + 1800
                        if deadline is not None and time.monotonic() > deadline:
                            raise TimeoutError("bounded GPU ownership window")
                        time.sleep(1)
                    rc = child.returncode
                    lease.success = rc == 0
                finally:
                    child.stop(timeout=25)
                    child = None
        if rc:
            raise RuntimeError("unit failed; preserve receipt and diagnose")
        validation = replay_validation(out, selected_cases, a.baseline)
        atomic(out / "replay-validation.json", validation)
        if not validation["passed"]:
            raise RuntimeError(
                "Baseline replay evidence incomplete or prefix drift; preserve as unclassified and localize any earlier first divergence"
            )
        if a.bridge_reference and not bridge(out, a.bridge_reference, a.bridge_arm):
            raise RuntimeError(
                "source bridge differs; require matched controls before reuse"
            )
        atomic(
            out / "unit-driver.json",
            {"completed": True, "source_revision": revision, "checkpoint": checkpoint},
        )
    finally:
        if child is not None:
            child.stop(timeout=25)
        if suspended:
            if hashlib.sha256(budget.PLIST.read_bytes()).hexdigest() != plist_sha:
                raise RuntimeError("Original service definition changed")
            for name, expected in spec["protected_restore_helper_sha256"].items():
                if hashlib.sha256((WORK / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError("Protected restoration helper changed during unit")
            restored = service_restore_protocol.restore(out / "protected-service-restore", snapshot=snapshot)
            atomic(out / "service-restored-ready.json", restored)


if __name__ == "__main__":
    main()
