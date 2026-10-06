#!/usr/bin/env python3
"""Run the bounded LLaDA direct-denoising campaign unit.

This is a one-artifact, batch-width-one exploratory smoke.  It deliberately
does not start the autoregressive server and makes no KV-cache or APCv2 claim.
The caller must already hold matching paired GPU locks and a live CPG radio
lease.  One unmeasured warm-up precedes the single thermally admitted measured
repetition at each exact 1K and 4K total-canvas cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCKS = (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock"))
TOTAL_CANVASES = (1024, 4096)
GENERATION = {"gen_length": 32, "block_length": 32, "steps": 32}
CAMPAIGN_SEMANTICS = {
    "execution_kind": "bidirectional-denoising",
    "batch_width": 1,
    "generation": GENERATION,
    "total_canvas_tokens": list(TOTAL_CANVASES),
    "autoregressive_kv": "not_applicable",
    "apcv2": "not_applicable",
    "streaming": "not_applicable",
    "runs_per_cell": 1,
    "interpretation": "exploratory smoke; not thermally replicated qualification",
}
UPSTREAM_MODEL_SOURCE = {
    "repository": "git@<private-git-host>:user/mlx-lm-unified.git",
    "ref": "forgejo/unified",
    "revision": "4de6f2a0a7d4cef4683477b66bd5de4067677011",
    "path": "mlx_lm/models/llada.py",
    "sha256": "f23ec7e7c0f797a2ee8a198b20c6401360d9d975342063c74351d8e353e88935",
}
FOREIGN = re.compile(
    r"(mlx2\.server|mlx_lm|mlx_vlm|mlx-lm|rapid-mlx|ltx-2-mlx|llama-server|"
    r"ollama|--i-own-the-gpu|scripts/(probe|bench|qualify|gpu_|run_))",
    re.IGNORECASE,
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_owner(path: Path) -> dict[str, Any]:
    receipt = path / "owner.json"
    if not path.is_dir() or not receipt.is_file():
        raise RuntimeError(f"GPU lock is not owned: {path}")
    value = json.loads(receipt.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"GPU lock owner is not an object: {path}")
    return value


def validate_ownership(
    session: str,
    label: str,
    cpg_radio: Path,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Bind this child to matching locks and an open, live CPG lease."""
    now = time.time() if now is None else now
    owners = [_read_owner(path) for path in LOCKS]
    if owners[0] != owners[1]:
        raise RuntimeError("paired GPU lock owner receipts disagree")
    owner = owners[0]
    if owner.get("session") != session or owner.get("label") != label:
        raise RuntimeError(
            f"GPU owner mismatch: expected {session}/{label}, got "
            f"{owner.get('session')}/{owner.get('label')}"
        )
    expected_lease = f"{session}-{label}"
    if owner.get("lease_id") != expected_lease:
        raise RuntimeError("GPU lock lease_id does not match session and label")
    if type(owner.get("pid")) is not int or owner["pid"] != os.getppid():
        raise RuntimeError("GPU lock is not owned by this runner's direct parent")
    try:
        os.kill(owner["pid"], 0)
    except ProcessLookupError as exc:
        raise RuntimeError("GPU lock owner process is no longer live") from exc

    radio = json.loads(cpg_radio.read_text())
    if not isinstance(radio, dict):
        raise TypeError("CPG radio receipt is not an object")
    if "release_task" in radio or "complete_worker" in radio:
        raise RuntimeError("CPG radio receipt is already closed")
    agent_id = radio.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id.startswith(f"job-{label}-"):
        raise RuntimeError("CPG radio agent is not bound to the campaign label")
    claim = radio.get("claim_task")
    register = radio.get("register_worker")
    if not isinstance(claim, dict) or claim.get("claimed") is not True:
        raise RuntimeError("CPG GPU lease was not claimed")
    if claim.get("owner_agent_id") != agent_id:
        raise RuntimeError("CPG claim owner does not match the registered agent")
    if not isinstance(register, dict) or register.get("agent_id") != agent_id:
        raise RuntimeError("CPG worker registration does not match the agent")
    generation = claim.get("lease_generation")
    if type(generation) is not int or generation < 1:
        raise RuntimeError("CPG claim lacks a positive lease generation")
    live = radio.get("renew_lease", claim)
    if (
        not isinstance(live, dict)
        or live.get("task_id") != claim.get("task_id")
        or live.get("lease_generation") != generation
        or (live is not claim and live.get("renewed") is not True)
        or type(live.get("lease_expires_at")) not in (int, float)
        or live["lease_expires_at"] <= now
    ):
        raise RuntimeError("CPG GPU lease is mismatched or expired")
    return {
        "paired_locks": owner,
        "cpg_agent_id": agent_id,
        "cpg_claim": claim,
        "cpg_renewal": radio.get("renew_lease"),
    }


def source_identity() -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        ).strip()
    )
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "mlx2").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    runner = Path(__file__).resolve()
    runtime_model = ROOT / "src/mlx2/runtime/models/llada.py"
    runtime_model_sha256 = hashlib.sha256(runtime_model.read_bytes()).hexdigest()
    return {
        "git_head": revision,
        "tracked_tree_dirty": dirty,
        "src_sha256": digest.hexdigest(),
        "runner": str(runner),
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "runtime_model": str(runtime_model),
        "runtime_model_sha256": runtime_model_sha256,
        "upstream_model_source": UPSTREAM_MODEL_SOURCE,
        "runtime_model_byte_identical_to_upstream_revalidation": (
            runtime_model_sha256 == UPSTREAM_MODEL_SOURCE["sha256"]
        ),
    }


def rendered_tokens(tokenizer: Any, prompt: str) -> list[int]:
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    if isinstance(tokens, Mapping):
        tokens = tokens["input_ids"]
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise TypeError("chat template did not return a flat integer token list")
    return tokens


def _prompt_text(units: int, nonce: str, needle: str) -> str:
    return (
        f"Archive session {nonce}. Read every record, then answer the question.\n"
        + " record" * units
        + f"\nThe unique launch code is {needle}.\n"
        + f"Reply with {needle} first."
    )


def calibrate_prompt(tokenizer: Any, target_prompt_tokens: int, nonce: str, needle: str) -> dict[str, Any]:
    """Find an exact rendered length; refuse an inexact total canvas."""
    low, high = 0, target_prompt_tokens * 2
    if len(rendered_tokens(tokenizer, _prompt_text(low, nonce, needle))) > target_prompt_tokens:
        raise ValueError("fixed semantic prompt exceeds the requested prompt canvas")
    while len(rendered_tokens(tokenizer, _prompt_text(high, nonce, needle))) < target_prompt_tokens:
        high *= 2
    while low < high:
        middle = (low + high + 1) // 2
        count = len(rendered_tokens(tokenizer, _prompt_text(middle, nonce, needle)))
        if count <= target_prompt_tokens:
            low = middle
        else:
            high = middle - 1
    prompt = _prompt_text(low, nonce, needle)
    count = len(rendered_tokens(tokenizer, prompt))
    if count != target_prompt_tokens:
        raise ValueError(
            f"could not calibrate exact prompt canvas: wanted {target_prompt_tokens}, got {count}"
        )
    return {
        "prompt": prompt,
        "prompt_tokens": count,
        "filler_units": low,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "needle": needle,
    }


def validate_result(
    result: dict[str, Any],
    *,
    expected_fingerprint: str,
    mask_token_id: int,
    needle: str | None,
    active_block_postprocess: bool = False,
    active_block_head: bool = False,
    verify_active_block_head: bool = False,
    require_semantic_needle: bool = True,
    require_visible_output: bool = True,
) -> dict[str, bool]:
    stats = result.get("stats")
    text = result.get("text")
    canvas = result.get("canvas_token_ids")
    checks = {
        "route_exact": result.get("route")
        == (
            "denoising-exact-active-head"
            if active_block_head
            else "denoising-exact-active-block"
            if active_block_postprocess
            else "denoising-exact"
        ),
        "fingerprint_exact": result.get("artifact_fingerprint") == expected_fingerprint,
        "forwards_exact": isinstance(stats, dict) and stats.get("forwards") == 32,
        "steps_exact": isinstance(stats, dict) and stats.get("steps") == 32,
        "prefix_snapshot_absent": isinstance(stats, dict)
        and "prefix_snapshot" in stats
        and stats.get("prefix_snapshot") is None,
        "prefix_snapshot_unused": isinstance(stats, dict)
        and stats.get("prefix_snapshot_used") is False,
        "visible_output": isinstance(text, str) and bool(text.strip()),
        "canvas_is_integer_list": isinstance(canvas, list)
        and all(type(token) is int for token in canvas),
        "no_mask_tokens": isinstance(canvas, list) and mask_token_id not in canvas,
        "semantic_needle": needle is None
        or (isinstance(text, str) and needle.casefold() in text.casefold()),
        "active_block_receipt": isinstance(stats, dict)
        and stats.get("active_block_postprocess") is active_block_postprocess
        and (
            stats.get("postprocess_rows_per_forward") == 32
            if active_block_postprocess
            else type(stats.get("postprocess_rows_per_forward")) is int
            and stats["postprocess_rows_per_forward"] >= 32
        ),
        "active_head_receipt": isinstance(stats, dict)
        and stats.get("active_block_head") is active_block_head
        and (
            stats.get("lm_head_rows_per_forward") == 32
            if active_block_head
            else type(stats.get("lm_head_rows_per_forward")) is int
            and stats["lm_head_rows_per_forward"] >= 32
        )
        and (
            stats.get("active_block_head_parity_checks") == 32
            if verify_active_block_head
            else type(stats.get("active_block_head_parity_checks")) is int
        ),
    }
    required = dict(checks)
    if not require_semantic_needle:
        required.pop("semantic_needle")
    if not require_visible_output:
        required.pop("visible_output")
    if not all(required.values()):
        failed = [name for name, passed in required.items() if not passed]
        raise AssertionError(f"LLaDA result gates failed: {failed}")
    return checks


def ancestors(pid: int) -> set[int]:
    found: set[int] = set()
    while pid > 1 and pid not in found:
        found.add(pid)
        row = subprocess.run(
            ["/bin/ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        pid = int(row) if row else 0
    return found


def process_tree(root_pids: set[int]) -> set[int]:
    rows = subprocess.run(
        ["/bin/ps", "-Ao", "pid=,ppid="], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    children: dict[int, list[int]] = {}
    for row in rows:
        fields = row.split()
        if len(fields) == 2:
            children.setdefault(int(fields[1]), []).append(int(fields[0]))
    seen: set[int] = set()
    stack = list(root_pids)
    while stack:
        pid = stack.pop()
        if pid not in seen:
            seen.add(pid)
            stack.extend(children.get(pid, []))
    return seen


def _cpu_seconds(value: str) -> float:
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    parts = [float(part) for part in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def foreign_snapshot(own: set[int]) -> dict[int, dict[str, Any]]:
    rows = subprocess.run(
        ["/bin/ps", "-Ao", "pid=,time=,command="],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found: dict[int, dict[str, Any]] = {}
    for row in rows.splitlines():
        fields = row.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid = int(fields[0])
        if pid not in own and FOREIGN.search(fields[2]):
            found[pid] = {
                "cpu_seconds": _cpu_seconds(fields[1]),
                "command": fields[2][:240],
            }
    return found


def foreign_activity(
    before: dict[int, dict[str, Any]],
    after: dict[int, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    active = []
    for pid, row in after.items():
        start = before.get(pid, {"cpu_seconds": 0.0})["cpu_seconds"]
        delta = row["cpu_seconds"] - start
        if delta > threshold:
            active.append(
                {
                    "pid": pid,
                    "cpu_seconds_delta": round(delta, 2),
                    "new": pid not in before,
                    "command": row["command"],
                }
            )
    return active


def swapouts() -> int:
    text = subprocess.run(
        ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True
    ).stdout
    match = re.search(r"Swapouts:\s+(\d+)", text)
    if not match:
        raise RuntimeError("vm_stat did not expose Swapouts")
    return int(match.group(1))


def post_run_thermal(matrix: Any, policy: dict[str, Any]) -> dict[str, Any]:
    command = policy.get("command")
    samples = [matrix.sample_thermal(command)]
    if not matrix.thermally_stable(samples[0], policy):
        time.sleep(
            float(
                policy.get(
                    "post_sample_interval_seconds",
                    policy.get("sample_interval_seconds", 15),
                )
            )
        )
        samples.append(matrix.sample_thermal(command))
    stable = [matrix.thermally_stable(row, policy) for row in samples]
    for row in samples:
        row.pop("raw_evidence", None)
    return {
        "samples": samples,
        "stable": stable,
        "breached": len(samples) == 2 and not any(stable),
        "rule": "invalid only after two consecutive policy-unstable samples",
    }


def _one_generation(
    adapter: Any,
    prompt: str,
    *,
    active_block_postprocess: bool = False,
    active_block_head: bool = False,
    verify_active_block_head: bool = False,
    require_visible_output: bool = True,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    result = adapter.generate(
        messages=[{"role": "user", "content": prompt}],
        active_block_postprocess=active_block_postprocess,
        active_block_head=active_block_head,
        verify_active_block_head=verify_active_block_head,
        require_visible_output=require_visible_output,
        **GENERATION,
    )
    return result, time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-fingerprint", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cpg-radio", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--foreign-cpu-threshold", type=float, default=0.5)
    parser.add_argument("--swapout-tolerance-pages", type=int, default=0)
    parser.add_argument("--active-block-postprocess", action="store_true")
    parser.add_argument("--active-block-head", action="store_true")
    parser.add_argument("--verify-active-block-head", action="store_true")
    parser.add_argument("--candidate-parity-only", action="store_true")
    args = parser.parse_args(argv)
    if args.runs != 1:
        parser.error("this exploratory LLaDA unit requires exactly --runs 1")
    if args.candidate_parity_only and not args.active_block_postprocess:
        parser.error("--candidate-parity-only requires --active-block-postprocess")
    if args.active_block_head and not args.active_block_postprocess:
        parser.error("--active-block-head requires --active-block-postprocess")
    if args.verify_active_block_head and not args.active_block_head:
        parser.error("--verify-active-block-head requires --active-block-head")
    if args.foreign_cpu_threshold < 0 or args.swapout_tolerance_pages < 0:
        parser.error("contamination thresholds must be nonnegative")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_fingerprint):
        parser.error("--expected-fingerprint must be a lowercase SHA-256 hex digest")

    model = args.model.expanduser().resolve(strict=True)
    out = args.out_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"refusing to overwrite nonempty evidence directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    # These gates precede adapter import and therefore precede all Metal/model work.
    ownership = validate_ownership(args.session, args.label, args.cpg_radio.resolve())
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_qualification_matrix as matrix

    from mlx2.adapters.llada import LLaDADenoisingAdapter, inspect_artifact

    artifact = inspect_artifact(model)
    actual_fingerprint = artifact["identity"]["fingerprint"]
    if actual_fingerprint != args.expected_fingerprint:
        raise RuntimeError(
            f"artifact fingerprint mismatch: expected {args.expected_fingerprint}, got {actual_fingerprint}"
        )
    if artifact.get("execution_kind") != "bidirectional-denoising":
        raise RuntimeError("artifact is not the direct bidirectional-denoising route")
    if artifact.get("has_autoregressive_cache") is not False:
        raise RuntimeError("artifact unexpectedly claims autoregressive cache state")

    thermal_policy = json.loads(
        (ROOT / "qualification/four-model-experiments.json").read_text()
    )["thermal"]
    campaign: dict[str, Any] = {
        "schema": "mlx2.llada-denoising-smoke-ladder.v1",
        "status": "running",
        "source": source_identity(),
        "artifact": artifact["identity"],
        "ownership": ownership,
        "semantics": CAMPAIGN_SEMANTICS,
        "candidate": {
            "active_block_postprocess": args.active_block_postprocess,
            "active_block_head": args.active_block_head,
            "verify_active_block_head": args.verify_active_block_head,
            "parity_only": args.candidate_parity_only,
            "selected": False,
            "qualified": False,
        },
        "thermal_policy": thermal_policy,
        "reference": None,
        "cells": [],
        "started_at": time.time(),
    }
    atomic_json(out / "campaign.json", campaign)
    os.environ.update(
        MLX_ENABLE_TF32="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )
    adapter = None
    rc = 1
    try:
        ownership = validate_ownership(args.session, args.label, args.cpg_radio.resolve())
        load_started = time.monotonic()
        adapter = LLaDADenoisingAdapter(str(model))
        campaign["load_seconds"] = time.monotonic() - load_started
        if adapter.identity.get("fingerprint") != args.expected_fingerprint:
            raise AssertionError("loaded adapter identity differs from inspected artifact")
        mask_token_id = int(adapter.config["mask_token_id"])

        validate_ownership(args.session, args.label, args.cpg_radio.resolve())
        reference_result, reference_seconds = _one_generation(
            adapter, "Say hello in one short sentence."
        )
        reference_checks = validate_result(
            reference_result,
            expected_fingerprint=args.expected_fingerprint,
            mask_token_id=mask_token_id,
            needle=None,
            active_block_postprocess=False,
        )
        campaign["reference"] = {
            "status": "passed",
            "prompt": "Say hello in one short sentence.",
            "elapsed_seconds": reference_seconds,
            "checks": reference_checks,
            "result": reference_result,
        }
        atomic_json(out / "reference.json", campaign["reference"])
        atomic_json(out / "campaign.json", campaign)

        for total_canvas in TOTAL_CANVASES:
            target_prompt = total_canvas - GENERATION["gen_length"]
            cell: dict[str, Any] = {
                "schema": "mlx2.llada-denoising-cell.v1",
                "status": "running",
                "total_canvas_tokens": total_canvas,
                "target_prompt_tokens": target_prompt,
                "batch_width": 1,
                "generation": GENERATION,
                "warmup": None,
                "measured": None,
            }
            campaign["cells"].append(cell)
            cell_path = out / f"canvas-{total_canvas}.json"
            atomic_json(cell_path, cell)

            warm_nonce = hashlib.sha256(
                f"warmup-{total_canvas}-{time.time_ns()}".encode()
            ).hexdigest()[:12]
            warm_needle = f"LLADA-{warm_nonce[:6].upper()}"
            warm_prompt = calibrate_prompt(
                adapter.tokenizer, target_prompt, warm_nonce, warm_needle
            )
            validate_ownership(args.session, args.label, args.cpg_radio.resolve())
            cell_reference = None
            if args.active_block_postprocess:
                cell_reference, cell_reference_seconds = _one_generation(
                    adapter,
                    warm_prompt["prompt"],
                    require_visible_output=not args.candidate_parity_only,
                )
                validate_result(
                    cell_reference,
                    expected_fingerprint=args.expected_fingerprint,
                    mask_token_id=mask_token_id,
                    needle=warm_needle,
                    active_block_postprocess=False,
                    active_block_head=False,
                    verify_active_block_head=False,
                    require_semantic_needle=not args.candidate_parity_only,
                    require_visible_output=not args.candidate_parity_only,
                )
                cell["ordinary_reference"] = {
                    "measured": False,
                    "elapsed_seconds": cell_reference_seconds,
                    "result": cell_reference,
                }
            warm_result, warm_seconds = _one_generation(
                adapter,
                warm_prompt["prompt"],
                active_block_postprocess=args.active_block_postprocess,
                active_block_head=args.active_block_head,
                verify_active_block_head=args.verify_active_block_head,
                require_visible_output=not args.candidate_parity_only,
            )
            warm_checks = validate_result(
                warm_result,
                expected_fingerprint=args.expected_fingerprint,
                mask_token_id=mask_token_id,
                needle=warm_needle,
                active_block_postprocess=args.active_block_postprocess,
                active_block_head=args.active_block_head,
                verify_active_block_head=args.verify_active_block_head,
                require_semantic_needle=not args.candidate_parity_only,
                require_visible_output=not args.candidate_parity_only,
            )
            if cell_reference is not None and (
                warm_result["canvas_token_ids"] != cell_reference["canvas_token_ids"]
                or warm_result["token_ids"] != cell_reference["token_ids"]
                or warm_result["text"] != cell_reference["text"]
            ):
                raise AssertionError("active-block candidate diverged from ordinary reference")
            cell["warmup"] = {
                "measured": False,
                **{key: value for key, value in warm_prompt.items() if key != "prompt"},
                "elapsed_seconds": warm_seconds,
                "checks": warm_checks,
                "result": warm_result,
            }
            atomic_json(cell_path, cell)

            try:
                thermal_samples = matrix.stabilize_thermal(thermal_policy)
                thermal_pre = {"stable": True, "samples": thermal_samples}
            except TimeoutError as exc:
                thermal_pre = {
                    "stable": False,
                    "samples": [matrix.sample_thermal()],
                    "error": str(exc),
                }
                cell["measured"] = {"status": "not_admitted", "thermal_pre": thermal_pre}
                cell["status"] = "failed"
                atomic_json(cell_path, cell)
                raise RuntimeError(f"thermal admission timed out for {total_canvas}") from exc

            # Thermal admission can wait for up to max_wait_seconds. Bind the
            # measured operation to a still-live lease and paired locks after
            # that wait, immediately before collecting the measured evidence.
            ownership_before = validate_ownership(
                args.session, args.label, args.cpg_radio.resolve()
            )

            nonce = hashlib.sha256(
                f"measured-{total_canvas}-{time.time_ns()}".encode()
            ).hexdigest()[:12]
            needle = f"LLADA-{nonce[:6].upper()}"
            calibrated = calibrate_prompt(adapter.tokenizer, target_prompt, nonce, needle)
            own = ancestors(os.getpid()) | process_tree({os.getpid()})
            foreign_before = foreign_snapshot(own)
            swap_pages_before = swapouts()
            swap_used_before = matrix.sample_swap()
            measured_started = time.time()
            result, elapsed = _one_generation(
                adapter,
                calibrated["prompt"],
                active_block_postprocess=args.active_block_postprocess,
                active_block_head=args.active_block_head,
                verify_active_block_head=args.verify_active_block_head,
                require_visible_output=not args.candidate_parity_only,
            )
            measured_finished = time.time()
            result_checks = validate_result(
                result,
                expected_fingerprint=args.expected_fingerprint,
                mask_token_id=mask_token_id,
                needle=needle,
                active_block_postprocess=args.active_block_postprocess,
                active_block_head=args.active_block_head,
                verify_active_block_head=args.verify_active_block_head,
                require_semantic_needle=not args.candidate_parity_only,
                require_visible_output=not args.candidate_parity_only,
            )
            swap_pages_after = swapouts()
            swap_used_after = matrix.sample_swap()
            own = ancestors(os.getpid()) | process_tree({os.getpid()})
            foreign = foreign_activity(
                foreign_before,
                foreign_snapshot(own),
                args.foreign_cpu_threshold,
            )
            thermal_post = post_run_thermal(matrix, thermal_policy)
            ownership_after = validate_ownership(
                args.session, args.label, args.cpg_radio.resolve()
            )
            contamination = []
            swap_delta = swap_pages_after - swap_pages_before
            if swap_delta > args.swapout_tolerance_pages:
                contamination.append(f"swapouts rose by {swap_delta} pages")
            if foreign:
                contamination.append(f"foreign GPU-capable process active: {foreign}")
            if thermal_post["breached"]:
                contamination.append("two consecutive post-run samples showed throttling")
            cell["measured"] = {
                "status": "passed" if not contamination else "contaminated",
                "measured": True,
                **{key: value for key, value in calibrated.items() if key != "prompt"},
                "started_at": measured_started,
                "finished_at": measured_finished,
                "elapsed_seconds": elapsed,
                "checks": result_checks,
                "result": result,
                "thermal_pre": thermal_pre,
                "thermal_post": thermal_post,
                "swapouts": {
                    "before": swap_pages_before,
                    "after": swap_pages_after,
                    "delta": swap_delta,
                    "tolerance_pages": args.swapout_tolerance_pages,
                },
                "swap_used_bytes": {
                    "before": swap_used_before["used_bytes"],
                    "after": swap_used_after["used_bytes"],
                },
                "foreign_activity": foreign,
                "foreign_cpu_threshold": args.foreign_cpu_threshold,
                "ownership_before": ownership_before,
                "ownership_after": ownership_after,
                "contamination": contamination,
            }
            cell["status"] = "passed" if not contamination else "failed"
            atomic_json(cell_path, cell)
            atomic_json(out / "campaign.json", campaign)
            if contamination:
                raise RuntimeError(f"contaminated {total_canvas} cell: {contamination}")

        campaign["status"] = (
            "component_parity_passed"
            if args.candidate_parity_only
            else "smoke_passed"
        )
        campaign["states"] = {
            "implemented": True,
            "component_parity_passed": args.candidate_parity_only,
            "smoke_passed": not args.candidate_parity_only,
            "qualified": False,
            "selected": False,
            "observed_used_in_this_unit": True,
        }
        rc = 0
    except BaseException as exc:  # noqa: BLE001 - persist every bounded failure
        campaign["status"] = "failed"
        campaign["error_type"] = type(exc).__name__
        campaign["error"] = str(exc)
        campaign["traceback"] = traceback.format_exc(limit=12)
        campaign["states"] = {
            "implemented": True,
            "smoke_passed": False,
            "qualified": False,
            "selected": False,
            "observed_used_in_this_unit": adapter is not None,
        }
    finally:
        if adapter is not None:
            adapter.close()
        campaign["finished_at"] = time.time()
        campaign["elapsed_seconds"] = campaign["finished_at"] - campaign["started_at"]
        atomic_json(out / "campaign.json", campaign)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
