#!/usr/bin/env python3
"""Run exact native-MTP correctness smoke for Nemotron 3.5 Lightning q8.

The caller must hold the CPG lease and both filesystem locks. The fresh MTP
server replays saved ordinary B1 prompts and requires exact terminal token,
text, finish, and usage parity plus nonzero segmented-MTP mechanism counters.
This is correctness-only component smoke, not qualification or performance
evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import run_north_feature_smoke as common
from run_nemotron35_lightning_feature_smoke import (
    EXPECTED_ARTIFACT,
    EXPECTED_LAYOUT,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROFILE = "nemotron35-lightning-q8-apcv2-mtp2"
REFERENCE_SCHEMA = "mlx2.nemotron35-lightning-feature-smoke.v1"


def load_ordinary_references(path: Path) -> list[dict]:
    value = json.loads(path.read_text())
    if value.get("schema") != REFERENCE_SCHEMA or value.get("passed") is not True:
        raise ValueError("ordinary reference is not a passed Lightning feature receipt")
    parity = (value.get("checks") or {}).get("b1_bn_token_parity") or {}
    if parity.get("passed") is not True:
        raise ValueError("ordinary reference lacks passed B1/B4 parity")
    responses = (parity.get("evidence") or {}).get("b1")
    if not isinstance(responses, list) or len(responses) != 4:
        raise ValueError("ordinary reference must contain four B1 responses")
    return responses


def response_identity(response: dict) -> dict:
    choice = response["choices"][0]
    usage = response.get("usage") or {}
    return {
        "token_ids": common.token_ids(response),
        "text": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def startup_binding_errors(status: dict) -> list[str]:
    errors = []
    if status.get("artifact") != EXPECTED_ARTIFACT:
        errors.append("artifact")
    if status.get("profile") != EXPECTED_PROFILE:
        errors.append("profile")
    if (status.get("apcv2") or {}).get("layout_name") != EXPECTED_LAYOUT:
        errors.append("cache_layout")
    if status.get("qualification") != "candidate":
        errors.append("qualification")
    if "mtp" not in (status.get("capabilities") or []):
        errors.append("mtp_capability")
    execution = status.get("execution") or {}
    if execution.get("mtp_head_present") is not True:
        errors.append("mtp_head")
    if execution.get("mtp_candidate") is not True:
        errors.append("mtp_candidate")
    return errors


def mtp_receipt_errors(response: dict) -> list[str]:
    route = common.receipt(response)
    mtp = route.get("mtp") or {}
    stats = mtp.get("stats") or {}
    errors = []
    if route.get("route") != "native_mtp":
        errors.append("route")
    if route.get("route_selection_source") != "explicit_flag":
        errors.append("route_selection_source")
    if route.get("ordinary_compute_width") is not None:
        errors.append("ordinary_compute_width")
    if mtp.get("route") != "segmented_self_mtp":
        errors.append("mtp.route")
    if mtp.get("verification") != "exact":
        errors.append("mtp.verification")
    if int(mtp.get("num_draft", 0) or 0) != 2:
        errors.append("mtp.num_draft")
    if mtp.get("observed_compute_widths") != [1]:
        errors.append("mtp.observed_compute_widths")
    if int(stats.get("draft_cycles", 0) or 0) <= 0:
        errors.append("mtp.stats.draft_cycles")
    if int(stats.get("draft_proposed", 0) or 0) <= 0:
        errors.append("mtp.stats.draft_proposed")
    return errors


def mechanism_errors(before: dict, after: dict) -> list[str]:
    errors = []
    for name in (
        "requests",
        "engaged",
        "b1_target_forwards",
        "b1_draft_forwards",
        "transaction_branches",
        "committed_cycles",
    ):
        if int(after.get(name, 0) or 0) <= int(before.get(name, 0) or 0):
            errors.append(f"no_positive_delta:{name}")
    for name in (
        "failures",
        "full_prefix_materializations",
        "physical_b2_formations",
        "true_batched_engaged",
    ):
        if int(after.get(name, 0) or 0) != int(before.get(name, 0) or 0):
            errors.append(f"unexpected_delta:{name}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ordinary-reference", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--port", type=int, default=18955)
    parser.add_argument("--load-timeout", type=float, default=900)
    args = parser.parse_args(argv)

    model = args.model.expanduser().resolve(strict=True)
    reference_path = args.ordinary_reference.expanduser().resolve(strict=True)
    ordinary = load_ordinary_references(reference_path)
    ordinary_identities = [response_identity(response) for response in ordinary]
    out = args.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"refusing to overwrite nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    owner = common.validate_ownership(args.session, args.label)
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
        "schema": "mlx2.nemotron35-lightning-mtp-campaign.v1",
        "status": "running",
        "semantics": "native-MTP correctness-only component smoke; not qualification or performance evidence",
        "source_revision": revision,
        "source_tracked_dirty": tracked_dirty,
        "model": str(model),
        "ordinary_reference": str(reference_path),
        "owner": owner,
        "route": "native_mtp",
        "num_draft": 2,
        "started_at": time.time(),
    }
    common.atomic_json(out / "campaign.json", campaign)
    policy_path = out / "execution-policy.json"
    common.atomic_json(policy_path, {"num_draft": 2})
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX2_CAMPAIGN_ROOT": str(ROOT),
    }
    server_command = [
        sys.executable, "-u", "-m", "mlx2.server",
        "--model", str(model),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--native-mtp",
        "--qualification-mode",
        "--max-context", "16384",
        "--max-lanes", "1",
        "--max-inflight", "1",
        "--cache-bytes", str(4 * 1024**3),
        "--cache-dir", str(out / "apcv2-cache"),
        "--execution-policy", str(policy_path),
    ]
    server = None
    report = None
    try:
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(
                server_command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            base = f"http://127.0.0.1:{args.port}"
            initial = common.wait_health(base, server, args.load_timeout)
            binding_errors = startup_binding_errors(initial)
            common.atomic_json(out / "server-startup.json", {
                "command": server_command,
                "pid": server.pid,
                "status": initial,
                "binding_errors": binding_errors,
            })
            if binding_errors:
                raise RuntimeError(
                    "Nemotron Lightning MTP startup binding failed: "
                    + ", ".join(binding_errors)
                )
            before = initial.get("segmented_self_mtp") or {}
            requests = common.build_parity_prompts("nemotron-lightning-parity")
            candidate = [common.chat(base, body, timeout=600) for body in requests]
            candidate_identities = [response_identity(response) for response in candidate]
            receipt_errors = [mtp_receipt_errors(response) for response in candidate]
            final = common.request_json(base, "/v1/status")
            after = final.get("segmented_self_mtp") or {}
            mechanism = mechanism_errors(before, after)
            quiescent = (
                final.get("inflight") == 0
                and final.get("queue_depth") == 0
                and (final.get("apcv2") or {}).get("cow", {}).get("active_leases") == 0
                and final.get("healthy") is True
                and final.get("error") is None
            )
            report = {
                "schema": "mlx2.nemotron35-lightning-mtp-smoke.v1",
                "semantics": "native-MTP correctness-only component smoke; not qualification or performance evidence",
                "ordinary_reference": str(reference_path),
                "ordinary_identities": ordinary_identities,
                "candidate_responses": candidate,
                "candidate_identities": candidate_identities,
                "exact_parity": candidate_identities == ordinary_identities,
                "receipt_errors": receipt_errors,
                "mechanism_before": before,
                "mechanism_after": after,
                "mechanism_errors": mechanism,
                "quiescent": quiescent,
                "final_status": final,
            }
            report["passed"] = (
                report["exact_parity"]
                and not any(receipt_errors)
                and not mechanism
                and quiescent
            )
            common.atomic_json(out / "mtp-smoke.json", report)
            campaign["status"] = "passed" if report["passed"] else "failed"
    except BaseException as error:
        campaign["status"] = "failed"
        campaign["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        common.stop_server(server)
        campaign["finished_at"] = time.time()
        campaign["locks_still_owned_before_return"] = (
            common.validate_ownership(args.session, args.label) == owner
        )
        common.atomic_json(out / "campaign.json", campaign)
    return 0 if report is not None and report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
