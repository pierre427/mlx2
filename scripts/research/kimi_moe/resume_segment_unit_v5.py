"""One source-bound resumed unit with idle-service suspension and safe restoration."""

import argparse
import hashlib
import json
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


def main():
    from mlx2.adapters.kimi_linear import inspect_artifact

    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--legacy-control", action="store_true")
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--bridge-reference", type=Path)
    p.add_argument("--bridge-arm")
    p.add_argument("--long", action="store_true")
    p.add_argument("--frozen", type=Path)
    p.add_argument("--reference", type=Path)
    a = p.parse_args()

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
    if a.frozen or a.long or a.legacy_control:
        raise ValueError("Routing recapture is serial corpus only")
    if candidate.get("loops") != []:
        raise ValueError("Routing recapture requires empty loops")
    if candidate.get("mode") == "fixed":
        if candidate.get("k") != 8 or set(candidate) != {"mode", "k", "loops"}:
            raise ValueError("Only stock8 control supported")
    elif candidate.get("mode") == "segmented":
        import segment_math_v1
        if set(candidate) != {"mode", "segments", "loops"}:
            raise ValueError("Invalid segment root fields")
        segment_math_v1.validate(candidate["segments"])
    else:
        raise ValueError("Unknown policy")
    cases = load(a.cases)
    if not cases or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Empty or duplicate cases")
    import kimi_domain_adaptive_sweep as domain
    canonical = {c["case_id"]: c for c in load(domain.CORPUS)["cases"]}
    if any(c != canonical.get(c["case_id"]) for c in cases):
        raise ValueError("Case does not match corpus")
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
        "benchmark_default_routes": False,
        "port": 18327,
        "mcp_url": "http://127.0.0.1:8766/mcp",
        "resume_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    import segment_probe_v5 as loop_worker
    script = WORK / "segment_probe_v5.py"
    spec["cases"] = cases
    spec["cases_file_sha256"] = hashlib.sha256(a.cases.read_bytes()).hexdigest()
    spec.update(
        frozen=str(a.frozen.resolve()) if a.frozen else None,
        reference=str(a.reference.resolve()) if a.reference else None,
        max_context=65536 if a.frozen else 8192,
        harness_sha256=loop_worker.harness(),
        request_counter_capture=True,
    )
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
