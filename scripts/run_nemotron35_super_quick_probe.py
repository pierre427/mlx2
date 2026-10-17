#!/usr/bin/env python3
"""One artifact/route, ordinary reference, and one bounded loopback server.

Exploratory component evidence only. The caller owns a live CPG/radio lease
and matching paired GPU locks. Every server admits one conversation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import nemotron35_probe_common as common
import run_qualification_matrix as matrix
from run_single_model_smoke_ladder import wait_health

ROOT = common.ROOT
CASES = (
    {
        "id": "arithmetic",
        "prompt": "Reply with only the integer: 17 multiplied by 19.",
        "max_tokens": 32,
    },
    {
        "id": "retrieval",
        "prompt": "The records are ALPHA=17, NEMOTRON_K7=9, OMEGA=42. Return only JSON with keys code and count for NEMOTRON_K7.",
        "max_tokens": 48,
    },
    {
        "id": "explanation",
        "prompt": "Explain how binary search finds a value in a sorted array. Give numbered steps and a concrete example. Use about 120 words.",
        "max_tokens": 128,
    },
)


def image_smoke_case():
    """Independent synthetic vision check, separate from training calibration."""
    import base64
    import io

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(image).rectangle((96, 96, 416, 416), fill="red")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    return {
        "id": "vision_red",
        "max_tokens": 8,
        "prompt": [
            {"type": "image_url", "image_url": {"url": url}},
            {
                "type": "text",
                "text": "What color is the large rectangle? Reply with only one color word.",
            },
        ],
    }


def policy_for(
    route,
    head,
    depth,
    verification_tile=None,
    alternative_branches=1,
    alternative_strategy="distinct_root",
    alternative_depth_policy="full",
    expert_gather_sort="auto",
    mamba_conv="auto",
    mamba_norm="auto",
    mamba_ssm="auto",
    mamba_projection="auto",
    mamba_segment="auto",
    moe_relu2_down="auto",
    draft_weight_quantization="auto",
):
    if expert_gather_sort not in {"auto", "unsorted_decode"}:
        raise ValueError("Super expert_gather_sort must be auto or unsorted_decode")
    if mamba_conv not in {"auto", "fused_b1", "fused_b1_compact", "fused_short"}:
        raise ValueError(
            "Super mamba_conv must be auto, fused_b1, fused_b1_compact, or fused_short"
        )
    if mamba_norm not in {"auto", "fused_b1", "fused_short"}:
        raise ValueError("Super mamba_norm must be auto, fused_b1, or fused_short")
    if mamba_ssm not in {"auto", "fused_short", "fused_short_gate"}:
        raise ValueError(
            "Super mamba_ssm must be auto, fused_short, or fused_short_gate"
        )
    if mamba_projection not in {
        "auto",
        "compact_padded",
        "compact_row_exact",
        "adaptive_row_exact",
        "short_row_exact",
        "native_qmm_residual_b1",
    }:
        raise ValueError(
            "Super mamba_projection must be auto, compact_padded, "
            "compact_row_exact, adaptive_row_exact, short_row_exact, or "
            "native_qmm_residual_b1"
        )
    if moe_relu2_down not in {"auto", "fused_b1"}:
        raise ValueError("Super moe_relu2_down must be auto or fused_b1")
    if mamba_segment not in {
        "auto",
        "layerwise_short_exact",
        "layerwise_short_exact_routed_batch",
    }:
        raise ValueError(
            "Super mamba_segment must be auto, layerwise_short_exact, or "
            "layerwise_short_exact_routed_batch"
        )
    if draft_weight_quantization not in {"auto", "affine_q8_g64"}:
        raise ValueError(
            "Nemotron draft_weight_quantization must be auto or affine_q8_g64"
        )

    def candidates(policy):
        if expert_gather_sort != "auto":
            policy["expert_gather_sort"] = expert_gather_sort
        if mamba_conv != "auto":
            policy["mamba_conv"] = mamba_conv
        if mamba_norm != "auto":
            policy["mamba_norm"] = mamba_norm
        if mamba_ssm != "auto":
            policy["mamba_ssm"] = mamba_ssm
        if mamba_projection != "auto":
            policy["mamba_projection"] = mamba_projection
        if mamba_segment != "auto":
            policy["mamba_segment"] = mamba_segment
        if moe_relu2_down != "auto":
            policy["moe_relu2_down"] = moe_relu2_down
        return policy

    if verification_tile is not None and route not in {"lilicorr", "dspark"}:
        raise ValueError("Progressive verification requires an external head")
    if (
        type(alternative_branches) is not int
        or not 1 <= alternative_branches <= 4
        or alternative_branches > 1
        and (route not in {"lilicorr", "dspark"} or verification_tile is None)
    ):
        raise ValueError(
            "Alternative branches require a capable progressive external head "
            "and must be 1 through 4"
        )
    if alternative_strategy not in {
        "distinct_root",
        "shared_prefix",
        "paired_continuations",
        "confidence_tree",
    }:
        raise ValueError("Unsupported alternative branch strategy")
    if alternative_depth_policy not in {"full", "confidence_tail"}:
        raise ValueError("Unsupported alternative depth policy")
    if alternative_depth_policy == "confidence_tail" and (
        alternative_strategy != "confidence_tree" or alternative_branches < 3
    ):
        raise ValueError(
            "confidence_tail requires confidence_tree with at least three branches"
        )
    if (
        alternative_strategy
        in {"shared_prefix", "paired_continuations", "confidence_tree"}
        and alternative_branches <= 1
    ):
        raise ValueError("Shared-prefix alternatives require multiple branches")
    if route == "ordinary":
        if (
            head is not None
            or depth is not None
            or draft_weight_quantization != "auto"
        ):
            raise ValueError("Ordinary reference has no head or draft depth")
        policy = candidates({})
        return policy or None
    if route == "native_mtp":
        if (
            head is not None
            or type(depth) is not int
            or not 1 <= depth <= 16
            or draft_weight_quantization != "auto"
        ):
            raise ValueError("Native MTP requires a bounded depth and no external head")
        policy = {"num_draft": depth}
        return candidates(policy)
    maximum = {"lilicorr": 7, "dspark": 8}[route]
    depth = maximum if depth is None else depth
    if head is None or type(depth) is not int or not 1 <= depth <= maximum:
        raise ValueError("External head/depth differs from its trained block")
    policy = {
        "draft_model": str(Path(head).resolve()),
        "num_draft": depth,
        "proposal_composition": False,
    }
    candidates(policy)
    if draft_weight_quantization != "auto":
        policy["draft_weight_quantization"] = draft_weight_quantization
    if verification_tile is not None:
        if type(verification_tile) is not int or not 1 <= verification_tile < depth:
            raise ValueError(
                "Verification tile must be a positive integer below proposal depth"
            )
        policy.update(
            progressive_verification_tile=verification_tile,
            progressive_verification_fidelity="exact",
        )
        if alternative_branches > 1:
            policy["progressive_alternative_branches"] = alternative_branches
            policy["progressive_alternative_strategy"] = alternative_strategy
            policy["progressive_alternative_depth_policy"] = alternative_depth_policy
    return policy


def server_command(args, policy_path):
    route = {"ordinary": "--ordinary", "native_mtp": "--native-mtp"}.get(
        args.route, "--external-draft"
    )
    active_sessions = int(getattr(args, "active_sessions", 1))
    command = [
        sys.executable,
        "-u",
        "-m",
        "mlx2.server",
        "--model",
        str(args.model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        route,
        "--qualification-mode",
        "--max-context",
        "8192",
        "--prefill-step",
        str(int(getattr(args, "prefill_step", 512))),
        "--max-lanes",
        str(active_sessions),
        "--max-inflight",
        str(active_sessions),
        "--cache-bytes",
        str(common.GIB),
        "--gpu-keep-warm-seconds",
        "0",
    ]
    if policy_path is not None:
        command += ["--execution-policy", str(policy_path)]
    return command


def expected_artifact(artifact, head_record, policy=None):
    if head_record is None:
        return artifact["identity"]["fingerprint"]
    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter

    contract = Nemotron35SuperAdapter.execution_numerics_contract(
        SimpleNamespace(conversion=artifact["conversion"])
    )
    numerical = json.dumps(
        {"schema": "mlx2.adapter-execution-numerics.v1", "contract": contract},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    composition = ""
    if (policy or {}).get("progressive_verification_tile") is not None:
        from mlx2.adapters.external_draft_policy import progressive_external_identity
        from mlx2.runtime.progressive_external_verify import (
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            progressive_target_protocol_binding,
        )

        declaration = SimpleNamespace(
            progressive_external_verify_protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION
        )
        binding = progressive_target_protocol_binding(declaration, "exact")
        composition = progressive_external_identity(policy, binding)
    if (policy or {}).get("draft_weight_quantization") is not None:
        composition += json.dumps(
            {
                "draft_weight_quantization": policy[
                    "draft_weight_quantization"
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    # These arms explicitly disable composition/continuation policies.
    # Fail closed if the adapter's combined identity ever changes.
    direct = hashlib.sha256(
        (
            artifact["identity"]["fingerprint"]
            + head_record["fingerprint"]
            + Nemotron35SuperAdapter.EXTERNAL_ROUTE_TAG
            + composition
            + numerical
        ).encode()
    ).hexdigest()
    # Serving also binds APCv2's adapter numerical namespace. The other
    # numerical wrappers remain stock/off for this explicitly bounded probe;
    # any changed law produces a different identity and fails this check.
    namespace = (
        "external-learning-v1",
        "adapter-execution-numerics-v1",
        hashlib.sha256(numerical.encode()).hexdigest(),
    )
    payload = json.dumps(
        namespace, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256((direct + payload).encode()).hexdigest()


def binding_errors(status, *, artifact, route, active_sessions=1):
    settings = status.get("settings") or {}
    expected_route = route if route in {"ordinary", "native_mtp"} else "external_draft"
    errors = []
    if status.get("artifact") != artifact:
        errors.append("artifact identity differs")
    if settings.get("route") != expected_route:
        errors.append("selected route differs")
    if (
        settings.get("max_lanes") != active_sessions
        or settings.get("max_inflight") != active_sessions
    ):
        errors.append("server session limits differ from the bounded probe")
    if status.get("qualification") not in {"candidate", "unqualified"}:
        errors.append("unexpected qualification state")
    return errors


def response_binding_errors(response, status, initial):
    """Chat receipts expose route/profile; identity comes from live status."""
    errors = []
    for key in ("runtime", "artifact", "profile"):
        if status.get(key) != initial.get(key):
            errors.append("server identity changed during request: " + key)
    receipt = response.get("mlx2") or {}
    if receipt.get("profile") != initial.get("profile"):
        errors.append("response profile differs")
    if receipt.get("route") != initial["settings"].get("route"):
        errors.append("response route differs")
    if status.get("settings") != initial.get("settings"):
        errors.append("server settings changed during request")
    return errors


def calibration_command(args, unit_receipt):
    return [
        sys.executable,
        "-u",
        str(ROOT / "scripts/collect_nemotron35_calibration.py"),
        "--model",
        str(args.model),
        "--corpus",
        str(args.calibration_corpus),
        "--out-dir",
        str(args.out_dir / "calibration"),
        "--unit-receipt",
        str(unit_receipt),
        "--cpg-owner-lock",
        str(args.cpg_owner_lock),
        "--start",
        str(args.calibration_start),
        "--count",
        str(args.calibration_count),
        "--max-seconds",
        str(args.calibration_seconds),
        "--mtp-packets",
        str(args.calibration_mtp_packets),
    ]


def functional_check(case_id, response):
    text = (response["choices"][0]["message"].get("content") or "").strip()
    if case_id == "arithmetic":
        return {"passed": text == "323", "expected": "323", "actual": text}
    if case_id == "retrieval":
        try:
            actual = json.loads(text)
        except ValueError:
            actual = None
        return {"passed": actual == {"code": "NEMOTRON_K7", "count": 9}, "actual": text}
    if case_id == "vision_red":
        return {
            "passed": text.lower().strip(" .\n") == "red",
            "expected": "red",
            "actual": text,
            "scope": "synthetic image-path check, not calibration data or broad vision quality evidence",
        }
    return {
        "passed": bool(text),
        "scope": "visible bounded output only; no quality score",
    }


def contamination_with_swap_policy(before, after, *, ignore_swap_growth):
    raw = common.contamination(before, after)
    ignored = []
    if ignore_swap_growth:
        ignored = [
            reason
            for reason in raw
            if reason in {"swap-outs increased", "swap usage increased"}
        ]
    return [reason for reason in raw if reason not in ignored], ignored


def batched_methodology_assessment(
    results, before_scheduler, after_scheduler, *, active_sessions
):
    """Classify whether a batched cell exercised its intended physical shape."""
    findings = []
    proposal_hashes = []
    observed_widths = []
    for result in results:
        progressive = (
            (result.get("mlx2") or {})
            .get("speculation", {})
            .get("progressive_verification", {})
        )
        observed_widths.append(progressive.get("max_cohort_width"))
        proposal_hashes.append(progressive.get("last_proposal_sha256"))
    if any(width != active_sessions for width in observed_widths):
        findings.append("progressive requests did not all execute at requested width")
    if None in proposal_hashes or len(set(proposal_hashes)) != active_sessions:
        findings.append("batched lanes did not expose distinct proposal hashes")
    required_counters = ["external_progressive_verify_batched_rounds"]
    prefix = f"external_progressive_verify_b{active_sessions}"
    required_counters.append(prefix + "_aligned_tile_launches")
    observed_counters = list(required_counters)
    if active_sessions == 4:
        observed_counters.extend(
            f"external_progressive_verify_b{width}_{suffix}"
            for width in range(1, 5)
            for suffix in (
                "aligned_tile_launches",
                "aligned_continued_launches",
            )
        )
        observed_counters.append(prefix + "_disjoint_first_tiles")
    observed_counters = list(dict.fromkeys(observed_counters))
    counter_deltas = {
        counter: int(after_scheduler.get(counter, 0))
        - int(before_scheduler.get(counter, 0))
        for counter in observed_counters
    }
    for counter in required_counters:
        if counter_deltas[counter] <= 0:
            findings.append(f"batched cell did not increase {counter}")
    surviving_continuations = sum(
        counter_deltas.get(
            f"external_progressive_verify_b{width}_aligned_continued_launches",
            0,
        )
        for width in range(1, active_sessions + 1)
    )
    return findings, {
        "requested_width": active_sessions,
        "observed_widths": observed_widths,
        "proposal_hashes_distinct": (
            None not in proposal_hashes and len(set(proposal_hashes)) == active_sessions
        ),
        "counter_deltas": counter_deltas,
        "surviving_continuation_launches": surviving_continuations,
        "any_branch_continued": surviving_continuations > 0,
        "first_tiles_pairwise_disjoint": (
            counter_deltas.get(prefix + "_disjoint_first_tiles", 0) > 0
        ),
    }


def reference_child(args, artifact):
    common.require_owner(args.cpg_owner_lock)
    before = common.host_snapshot()
    errors = common.admission_errors(
        before,
        minimum_available_bytes=artifact["conversion"]["tensor_bytes"] + 8 * common.GIB,
    )
    if errors:
        raise RuntimeError("Reference admission failed: " + "; ".join(errors))
    import mlx.core as mx
    from smoke_local_model import _ordinary_generate

    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter

    adapter = None
    receipt = {
        "status": "running",
        "qualification": "unqualified",
        "owner": common.require_owner(args.cpg_owner_lock),
        "before": before,
        "artifact": artifact["identity"]["fingerprint"],
    }
    try:
        loaded = time.monotonic()
        adapter = Nemotron35SuperAdapter(str(args.model))
        receipt["load_seconds"] = time.monotonic() - loaded
        receipt["output"] = _ordinary_generate(adapter, CASES[0]["prompt"], 16)
        receipt["peak_active_bytes"] = mx.get_peak_memory()
        receipt["after"] = common.host_snapshot()
        receipt["contamination"], receipt["ignored_contamination"] = (
            contamination_with_swap_policy(
                before,
                receipt["after"],
                ignore_swap_growth=args.ignore_swap_growth,
            )
        )
        if receipt["contamination"]:
            raise RuntimeError("Ordinary reference exceeded host guards")
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        if adapter is not None:
            adapter.close()
        common.atomic_json(args.out_dir / "ordinary-reference.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cpg-owner-lock", type=Path, required=True)
    parser.add_argument(
        "--route",
        choices=("ordinary", "native_mtp", "lilicorr", "dspark"),
        default="ordinary",
    )
    parser.add_argument("--head", type=Path)
    parser.add_argument("--num-draft", type=int)
    parser.add_argument(
        "--verification-tile",
        type=int,
        help="External progressive tile; 4 gives 4+4 target input rows for seven proposals",
    )
    parser.add_argument(
        "--alternative-branches",
        type=int,
        choices=(1, 2, 3, 4),
        default=1,
        help="Ordered LiLiCorr paths for one request; default keeps the stock B1 path",
    )
    parser.add_argument(
        "--alternative-strategy",
        choices=(
            "distinct_root",
            "shared_prefix",
            "paired_continuations",
            "confidence_tree",
        ),
        default="distinct_root",
        help="Keep the retained distinct-root baseline or reuse one shared first tile",
    )
    parser.add_argument(
        "--alternative-depth-policy",
        choices=("full", "confidence_tail"),
        default="full",
        help="Keep every branch full depth or shorten lower confidence-tree branches",
    )
    parser.add_argument(
        "--expert-gather-sort",
        choices=("auto", "unsorted_decode"),
        default="auto",
        help="Adapter-owned SwitchMLP gather policy; candidate remains unqualified",
    )
    parser.add_argument(
        "--mamba-conv",
        choices=("auto", "fused_b1", "fused_b1_compact", "fused_short"),
        default="auto",
        help="Adapter-owned Super conv4/SiLU/window candidate; remains unqualified",
    )
    parser.add_argument(
        "--mamba-norm",
        choices=("auto", "fused_b1", "fused_short"),
        default="auto",
        help="Adapter-owned Super gate-before-group-RMS candidate; remains unqualified",
    )
    parser.add_argument(
        "--mamba-ssm",
        choices=("auto", "fused_short", "fused_short_gate"),
        default="auto",
        help="Adapter-owned Super S2-8 SSM+dt candidate; remains unqualified",
    )
    parser.add_argument(
        "--mamba-projection",
        choices=(
            "auto",
            "compact_padded",
            "compact_row_exact",
            "adaptive_row_exact",
            "short_row_exact",
            "native_qmm_residual_b1",
        ),
        default="auto",
        help="Adapter-owned Mamba varlen projection candidate; remains unqualified",
    )
    parser.add_argument(
        "--mamba-segment",
        choices=(
            "auto",
            "layerwise_short_exact",
            "layerwise_short_exact_routed_batch",
        ),
        default="auto",
        help="Adapter-owned exact layerwise short-segment candidate; unqualified",
    )
    parser.add_argument(
        "--moe-relu2-down",
        choices=("auto", "fused_b1"),
        default="auto",
        help="Adapter-owned Super ReLU2/down/FP32 top22 candidate; remains unqualified",
    )
    parser.add_argument(
        "--draft-weight-quantization",
        choices=("auto", "affine_q8_g64"),
        default="auto",
        help="Nemotron external-draft weight profile; remains unqualified",
    )
    parser.add_argument("--port", type=int, default=18959)
    parser.add_argument(
        "--prefill-step",
        type=int,
        choices=(256, 512, 1024, 2048, 4096, 8192),
        default=512,
        help=(
            "Bounded server prefill chunk. The historical quick-test shape is 512; "
            "2048 matches the adapter default."
        ),
    )
    parser.add_argument("--load-timeout", type=float, default=1200)
    parser.add_argument(
        "--active-sessions",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="Bounded active request sessions; B2/B4 are progressive-smoke only",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0,
        help="Bounded post-load idle wait before warmup; recorded in startup evidence",
    )
    parser.add_argument(
        "--ignore-swap-growth",
        action="store_true",
        help="Record but do not fail on swap growth; all other guards remain enforced",
    )
    parser.add_argument("--calibration-corpus", type=Path)
    parser.add_argument("--calibration-start", type=int, default=0)
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument("--calibration-seconds", type=int, default=300)
    parser.add_argument("--calibration-mtp-packets", type=int, default=4)
    parser.add_argument(
        "--kernel-inputs",
        type=Path,
        help="Prepared real-weight NVFP4 component inputs; ordinary route only",
    )
    parser.add_argument(
        "--reference-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    from mlx2.adapters.nemotron35_super import inspect_artifact

    args.model = args.model.expanduser().resolve(strict=True)
    args.out_dir = args.out_dir.expanduser().absolute()
    artifact = inspect_artifact(args.model)
    if args.reference_child:
        reference_child(args, artifact)
        return
    policy = policy_for(
        args.route,
        args.head,
        args.num_draft,
        args.verification_tile,
        args.alternative_branches,
        args.alternative_strategy,
        args.alternative_depth_policy,
        args.expert_gather_sort,
        args.mamba_conv,
        args.mamba_norm,
        args.mamba_ssm,
        args.mamba_projection,
        args.mamba_segment,
        args.moe_relu2_down,
        args.draft_weight_quantization,
    )
    calibrating = args.calibration_corpus is not None
    if args.active_sessions in {2, 4} and (
        calibrating
        or args.route not in {"lilicorr", "dspark"}
        or args.verification_tile is None
    ):
        parser.error("B2/B4 smoke requires a progressive external-draft route")
    if args.alternative_branches > 1 and args.active_sessions != 1:
        parser.error("Ordered alternative branches require one request session")
    if args.kernel_inputs is not None and (calibrating or args.route != "ordinary"):
        parser.error(
            "Kernel component probes require an ordinary unit separate from calibration"
        )
    kernel_manifest = None
    if args.kernel_inputs is not None:
        from mlx2.conversion.kernel_probe import verify

        args.kernel_inputs = args.kernel_inputs.resolve(strict=True)
        kernel_manifest = verify(args.kernel_inputs)
    if calibrating and (
        args.route != "ordinary"
        or not 1 <= args.calibration_count <= 64
        or not 30 <= args.calibration_seconds <= 480
        or not 0 <= args.calibration_mtp_packets <= 8
    ):
        parser.error(
            "Offline image calibration requires ordinary route and bounded sample/time/packet settings"
        )
    head_record = None
    if args.route in {"lilicorr", "dspark"}:
        if args.route == "dspark":
            from mlx2.adapters.nemotron35_dspark import inspect_drafter

            head_record = inspect_drafter(args.head, args.model)
        else:
            from mlx2.adapters.lilicorr_nemotron import inspect_drafter

            head_record = inspect_drafter(args.head, target_artifact=artifact)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error("Refusing to overwrite nonempty probe evidence")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    owner = common.require_owner(args.cpg_owner_lock)
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Probe loopback port is occupied")
    source = common.source_identity()
    dependencies = common.runtime_dependencies()
    campaign = {
        "schema": "mlx2.nemotron35-super-quick-probe.v1",
        "status": "running",
        "owner": owner,
        "source": source,
        "dependencies": dependencies,
        "target": artifact["identity"],
        "head": None if head_record is None else head_record["fingerprint"],
        "route": args.route,
        "policy": policy,
        "active_sessions": args.active_sessions,
        "alternative_branches": args.alternative_branches,
        "alternative_strategy": args.alternative_strategy,
        "alternative_depth_policy": args.alternative_depth_policy,
        "ignore_swap_growth": args.ignore_swap_growth,
        "qualification": "unqualified",
        "semantics": "bounded exploratory quick comparison; no full preflight or formal qualification",
        "cases": [],
    }
    if calibrating:
        campaign["semantics"] = (
            "bounded offline calibration unit with ordinary/server references; no performance evidence"
        )
    if kernel_manifest is not None:
        campaign["kernel_inputs"] = kernel_manifest
    common.atomic_json(args.out_dir / "campaign.json", campaign)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
    }
    child = server = None
    previous = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(
                sig, lambda signum, frame: sys.exit(128 + signum)
            )
        before = common.host_snapshot()
        extra = 4 * common.GIB if head_record is not None else 0
        errors = common.admission_errors(
            before,
            minimum_available_bytes=artifact["conversion"]["tensor_bytes"]
            + extra
            + 8 * common.GIB,
            performance=not calibrating,
        )
        common.atomic_json(
            args.out_dir / "admission.json", {"host": before, "errors": errors}
        )
        if errors:
            raise RuntimeError("Quick probe admission failed: " + "; ".join(errors))
        reference = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--model",
            str(args.model),
            "--out-dir",
            str(args.out_dir),
            "--cpg-owner-lock",
            str(args.cpg_owner_lock),
            "--reference-child",
        ]
        if args.ignore_swap_growth:
            reference.append("--ignore-swap-growth")
        with (args.out_dir / "ordinary-reference.log").open("x") as log:
            child = subprocess.Popen(
                reference,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            child.wait(timeout=args.load_timeout)
            common.stop_group(child)
            if child.returncode != 0:
                raise RuntimeError(
                    "Ordinary reference failed; inspect its retained log/receipt"
                )
            child = None
        policy_path = None
        if policy:
            policy_path = args.out_dir / "execution-policy.json"
            common.atomic_json(policy_path, policy)
        command = server_command(args, policy_path)
        common.require_owner(args.cpg_owner_lock)
        with (args.out_dir / "server.log").open("x") as log:
            server = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            base = f"http://127.0.0.1:{args.port}"
            initial = wait_health(base, server, args.load_timeout)
            errors = binding_errors(
                initial,
                artifact=expected_artifact(artifact, head_record, policy),
                route=args.route,
                active_sessions=args.active_sessions,
            )
            common.atomic_json(
                args.out_dir / "startup.json",
                {
                    "command": command,
                    "status": initial,
                    "binding_errors": errors,
                    "settle_seconds": args.settle_seconds,
                },
            )
            if errors:
                raise RuntimeError("Server binding failed: " + "; ".join(errors))
            if not 0 <= args.settle_seconds <= 300:
                raise RuntimeError(
                    "Post-load settle must be between zero and 300 seconds"
                )
            if args.settle_seconds:
                time.sleep(args.settle_seconds)
                common.require_owner(args.cpg_owner_lock)
            client = matrix.Client(base, timeout=300)
            common.require_owner(args.cpg_owner_lock)
            warmup_host = common.host_snapshot()
            errors = common.admission_errors(
                warmup_host, minimum_available_bytes=4 * common.GIB
            )
            common.atomic_json(
                args.out_dir / "warmup-admission.json",
                {"host": warmup_host, "errors": errors},
            )
            if errors:
                raise RuntimeError("Warmup admission failed: " + "; ".join(errors))
            # Exercise the longer row shape once; retain but exclude warmup.
            warmup_case = CASES[0] if calibrating else CASES[-1]
            warmup = client.post(
                matrix.prompt_body(
                    warmup_case["prompt"]
                    if calibrating
                    else "Warmup only. " + warmup_case["prompt"],
                    16 if calibrating else 128,
                )
            )
            common.atomic_json(args.out_dir / "warmup.json", warmup)
            measured_cases = (*CASES[:1], image_smoke_case()) if calibrating else CASES
            if args.active_sessions in {2, 4}:
                if args.active_sessions == 2:
                    measured_cases = (
                        CASES[-1],
                        {
                            "id": "merge_sort_explanation",
                            "prompt": (
                                "Explain how merge sort orders an array. Give numbered "
                                "steps and a concrete example. Use about 120 words."
                            ),
                            "max_tokens": 128,
                        },
                    )
                else:
                    measured_cases = tuple(
                        {
                            "id": f"repeat_{word}",
                            "prompt": (
                                f"Reply with exactly 120 copies of the word {word} "
                                "separated by single spaces. Do not add anything else."
                            ),
                            "max_tokens": 128,
                        }
                        for word in ("apple", "banana", "ocean", "music")
                    )
                for cache_state in ("first", "repeat"):
                    common.require_owner(args.cpg_owner_lock)
                    pre = common.host_snapshot()
                    errors = common.admission_errors(
                        pre,
                        minimum_available_bytes=4 * common.GIB,
                        performance=True,
                    )
                    if errors:
                        raise RuntimeError(
                            "Cell admission failed: " + "; ".join(errors)
                        )
                    requests = [
                        {
                            **matrix.prompt_body(case["prompt"], case["max_tokens"]),
                            "seed": 7 + 4 * lane_index,
                            "repetition_penalty": 1.0,
                            "batch_cohort": {
                                "id": (
                                    f"nemotron35-progressive-b{args.active_sessions}-"
                                    f"{cache_state}-{time.time_ns()}"
                                ),
                                "size": args.active_sessions,
                            },
                        }
                        for lane_index, case in enumerate(measured_cases)
                    ]
                    cohort = dict(requests[0]["batch_cohort"])
                    for request in requests[1:]:
                        request["batch_cohort"] = dict(cohort)
                    pre_status = client.get("/v1/status")
                    barrier = threading.Barrier(args.active_sessions + 1)

                    def post_batched(request, *, start_barrier=barrier):
                        start_barrier.wait(timeout=30)
                        return matrix.Client(base, timeout=300).post(request)

                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=args.active_sessions
                    ) as pool:
                        futures = [
                            pool.submit(post_batched, request) for request in requests
                        ]
                        barrier.wait(timeout=30)
                        results = [future.result(timeout=300) for future in futures]
                    cell_status = client.get("/v1/status")
                    post = common.host_snapshot()
                    environment_failures, ignored_contamination = (
                        contamination_with_swap_policy(
                            pre,
                            post,
                            ignore_swap_growth=args.ignore_swap_growth,
                        )
                    )
                    environment_failures.extend(
                        common.admission_errors(
                            post,
                            minimum_available_bytes=4 * common.GIB,
                            performance=True,
                        )
                    )
                    for result in results:
                        environment_failures.extend(
                            response_binding_errors(result, cell_status, initial)
                        )
                    before_scheduler = pre_status.get("scheduler", {})
                    after_scheduler = cell_status.get("scheduler", {})
                    methodology_failures, methodology_evidence = (
                        batched_methodology_assessment(
                            results,
                            before_scheduler,
                            after_scheduler,
                            active_sessions=args.active_sessions,
                        )
                    )
                    for lane_index, (case, request, result) in enumerate(
                        zip(measured_cases, requests, results, strict=True)
                    ):
                        functional = functional_check(case["id"], result)
                        outcome_failures = (
                            []
                            if functional["passed"]
                            else ["functional output oracle did not pass"]
                        )
                        campaign["cases"].append(
                            {
                                "id": case["id"],
                                "pass": cache_state,
                                "batched_lane": lane_index,
                                "request": request,
                                "response": result,
                                "server_status_after": cell_status,
                                "functional": functional,
                                "thermal_host_before": pre,
                                "thermal_host_after": post,
                                "contamination": list(environment_failures),
                                "environment_failures": list(environment_failures),
                                "ignored_contamination": list(ignored_contamination),
                                "methodology_failures": list(methodology_failures),
                                "methodology_evidence": methodology_evidence,
                                "outcome_failures": outcome_failures,
                                "observation_valid": (
                                    not environment_failures and not outcome_failures
                                ),
                                "requested_shape_observed": (not methodology_failures),
                                "performance_reportable": (
                                    not environment_failures
                                    and not methodology_failures
                                    and not outcome_failures
                                ),
                            }
                        )
                    common.atomic_json(args.out_dir / "campaign.json", campaign)
                    if environment_failures:
                        raise RuntimeError(
                            "Measured cell environment invalid; release GPU before retry"
                        )
            else:
                for case in measured_cases:
                    for cache_state in (
                        ("reference",) if calibrating else ("first", "repeat")
                    ):
                        common.require_owner(args.cpg_owner_lock)
                        pre = common.host_snapshot()
                        errors = common.admission_errors(
                            pre,
                            minimum_available_bytes=4 * common.GIB,
                            performance=not calibrating,
                        )
                        if errors:
                            raise RuntimeError(
                                "Cell admission failed: " + "; ".join(errors)
                            )
                        request = {
                            **matrix.prompt_body(case["prompt"], case["max_tokens"]),
                            "seed": 7,
                            "repetition_penalty": 1.0,
                        }
                        result = client.post(request)
                        cell_status = client.get("/v1/status")
                        post = common.host_snapshot()
                        environment_failures, ignored_contamination = (
                            contamination_with_swap_policy(
                                pre,
                                post,
                                ignore_swap_growth=args.ignore_swap_growth,
                            )
                        )
                        if not calibrating:
                            environment_failures.extend(
                                common.admission_errors(
                                    post,
                                    minimum_available_bytes=4 * common.GIB,
                                    performance=True,
                                )
                            )
                        environment_failures.extend(
                            response_binding_errors(result, cell_status, initial)
                        )
                        functional = functional_check(case["id"], result)
                        outcome_failures = (
                            []
                            if functional["passed"]
                            else ["functional output oracle did not pass"]
                        )
                        cell = {
                            "id": case["id"],
                            "pass": cache_state,
                            "request": request,
                            "response": result,
                            "server_status_after": cell_status,
                            "functional": functional,
                            "thermal_host_before": pre,
                            "thermal_host_after": post,
                            "contamination": environment_failures,
                            "environment_failures": environment_failures,
                            "ignored_contamination": ignored_contamination,
                            "methodology_failures": [],
                            "methodology_evidence": {},
                            "outcome_failures": outcome_failures,
                            "observation_valid": (
                                not environment_failures and not outcome_failures
                            ),
                            "requested_shape_observed": True,
                            "performance_reportable": (
                                not calibrating
                                and not environment_failures
                                and not outcome_failures
                            ),
                        }
                        campaign["cases"].append(cell)
                        common.atomic_json(args.out_dir / "campaign.json", campaign)
                        if environment_failures:
                            raise RuntimeError(
                                "Measured cell environment invalid; release GPU before retry"
                            )
            campaign["functional_passed"] = all(
                row["functional"]["passed"] for row in campaign["cases"]
            )
            campaign["methodology_passed"] = not any(
                row.get("methodology_failures") for row in campaign["cases"]
            )
            campaign["environment_passed"] = not any(
                row.get("environment_failures") for row in campaign["cases"]
            )
            if not campaign["functional_passed"]:
                campaign["status"] = "completed_with_outcome_failures"
            elif not campaign["methodology_passed"]:
                campaign["status"] = "completed_with_methodological_limitations"
            else:
                campaign["status"] = "completed"
            campaign["final_server_status"] = client.get("/v1/status")
        if calibrating:
            if not campaign["functional_passed"]:
                raise RuntimeError(
                    "Ordinary server functional check failed before calibration"
                )
            server_pgid = server.pid
            common.stop_group(server)
            server = None
            owner = common.require_owner(args.cpg_owner_lock)
            unit_path = args.out_dir / "calibration-server-unit.json"
            common.atomic_json(
                unit_path,
                {
                    "schema": "mlx2.nemotron35-calibration-server-unit.v1",
                    "artifact": artifact["identity"]["fingerprint"],
                    "source_sha256": source["sha256"],
                    "ordinary_reference_passed": json.loads(
                        (args.out_dir / "ordinary-reference.json").read_text()
                    )["status"]
                    == "passed",
                    "ordinary_server_passed": True,
                    "server_group_stopped": not common.group_running(server_pgid),
                    "server_pgid": server_pgid,
                    "max_lanes": initial["settings"]["max_lanes"],
                    "max_inflight": initial["settings"]["max_inflight"],
                    "cpg_owner": owner["cpg_owner"],
                    "paired_owner": owner["paired_gpu_owners"]["shared"],
                    "dependencies": dependencies,
                },
            )
            with (args.out_dir / "calibration.log").open("x") as log:
                child = subprocess.Popen(
                    calibration_command(args, unit_path),
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                child.wait(timeout=args.load_timeout + args.calibration_seconds + 120)
                common.stop_group(child)
                if child.returncode != 0:
                    raise RuntimeError(
                        "Offline calibration failed; retain its unit receipt and partial statistics"
                    )
                child = None
            campaign["calibration_unit"] = json.loads(
                (args.out_dir / "calibration/unit.json").read_text()
            )
            if not campaign["calibration_unit"].get("eligible_for_merge"):
                if campaign["calibration_unit"]["state"] == "yielded":
                    campaign["status"] = "yielded"
                else:
                    raise RuntimeError(
                        "Calibration unit failed its final source/dependency/ownership publication gates"
                    )
        if kernel_manifest is not None:
            if not campaign["functional_passed"]:
                raise RuntimeError(
                    "Ordinary server functional check failed before component probes"
                )
            server_pgid = server.pid
            common.stop_group(server)
            server = None
            owner = common.require_owner(args.cpg_owner_lock)
            unit_path = args.out_dir / "kernel-server-unit.json"
            common.atomic_json(
                unit_path,
                {
                    "schema": "mlx2.nemotron35-kernel-server-unit.v1",
                    "artifact": artifact["identity"]["fingerprint"],
                    "source_sha256": source["sha256"],
                    "kernel_inputs": kernel_manifest["fingerprint"],
                    "ordinary_reference_passed": json.loads(
                        (args.out_dir / "ordinary-reference.json").read_text()
                    )["status"]
                    == "passed",
                    "ordinary_server_passed": True,
                    "server_group_stopped": not common.group_running(server_pgid),
                    "server_pgid": server_pgid,
                    "max_lanes": initial["settings"]["max_lanes"],
                    "max_inflight": initial["settings"]["max_inflight"],
                    "cpg_owner": owner["cpg_owner"],
                    "paired_owner": owner["paired_gpu_owners"]["shared"],
                    "dependencies": dependencies,
                },
            )
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/probe_nemotron35_nvfp4_kernels.py"),
                "--inputs",
                str(args.kernel_inputs),
                "--out-dir",
                str(args.out_dir / "kernels"),
                "--unit-receipt",
                str(unit_path),
                "--cpg-owner-lock",
                str(args.cpg_owner_lock),
            ]
            with (args.out_dir / "kernels.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                child.wait(timeout=240)
                common.stop_group(child)
                if child.returncode != 0:
                    raise RuntimeError(
                        "Kernel component probe failed; inspect retained evidence"
                    )
                child = None
            campaign["kernel_probe"] = json.loads(
                (args.out_dir / "kernels/kernels.json").read_text()
            )
            if (
                campaign["kernel_probe"]["state"] != "completed"
                or campaign["kernel_probe"]["finalization_errors"]
            ):
                raise RuntimeError(
                    "Kernel component probe failed its final publication gates"
                )
    except BaseException as error:
        campaign.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        cleanup_errors = []
        for process in (child, server):
            try:
                common.stop_group(process)
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                cleanup_errors.append(str(error))
        campaign["cleanup_errors"] = cleanup_errors
        try:
            campaign["dependencies_unchanged"] = (
                common.runtime_dependencies() == dependencies
            )
        except (OSError, RuntimeError, ValueError) as error:
            campaign.update(
                dependencies_unchanged=False, dependency_after_error=str(error)
            )
        campaign["finished_at"] = time.time()
        try:
            campaign["source_after"] = common.source_identity()["sha256"]
            campaign["source_unchanged"] = campaign["source_after"] == source["sha256"]
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            campaign.update(source_unchanged=False, source_after_error=str(error))
        try:
            campaign["owner_after"] = common.require_owner(args.cpg_owner_lock)
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as error:
            campaign["owner_after_error"] = str(error)
        if (
            cleanup_errors
            or not campaign["source_unchanged"]
            or not campaign["dependencies_unchanged"]
            or campaign.get("owner_after_error")
        ):
            for row in campaign["cases"]:
                row["performance_reportable"] = False
        common.atomic_json(args.out_dir / "campaign.json", campaign)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
