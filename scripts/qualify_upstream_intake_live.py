#!/usr/bin/env python3
"""Route-specific live gates for the 2026-10-15 upstream intake.

This is deliberately not a formal qualification producer: it never runs or
accepts a preflight receipt.  It records bounded candidate evidence for the
LFM2.5 APCv2 warm-hit and Qwen generation-template thinking-release fixes.
Run one artifact per process under ``scripts/run_with_gpu_locks.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_identity(path: Path) -> dict:
    names = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
    }
    files = sorted(
        {
            *path.glob("*.safetensors"),
            *(path / name for name in names if (path / name).is_file()),
        }
    )
    return {
        "name": path.name,
        "files": {
            file.name: {"size": file.stat().st_size, "sha256": _sha256(file)}
            for file in files
        },
    }


def collect(engine, request: dict, timeout: float) -> dict:
    job = engine.submit(dict(request))
    deltas = []
    terminal = None
    deadline = time.monotonic() + timeout
    while terminal is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request did not finish")
        event = job.events.get(timeout=remaining)
        if "error" in event:
            raise RuntimeError(event["error"])
        delta = event.get("delta") or {}
        if delta:
            deltas.append(
                {
                    key: delta[key]
                    for key in ("reasoning_content", "content")
                    if delta.get(key)
                }
            )
        if "finish_reason" in event:
            terminal = event
    receipt = terminal.get("receipt") or {}
    return {
        "finish_reason": terminal["finish_reason"],
        "cached_tokens": int(job.cached_tokens),
        "deltas": deltas,
        "token_trace": (receipt.get("token_trace") or {}).get("token_ids") or [],
        "request_controls": receipt.get("request_controls") or {},
        "route": receipt.get("route"),
        "route_receipt": receipt.get("route_receipt"),
    }


def run_lfm(engine, timeout: float) -> dict:
    initial_messages = [
        {"role": "user", "content": "Reply with exactly: BLUE READY"},
    ]
    base = {
        "messages": initial_messages,
        "enable_thinking": False,
        "max_tokens": 16,
        "temperature": 0,
        "return_token_trace": True,
    }
    cold = collect(engine, base, timeout)
    assistant = "".join(delta.get("content", "") for delta in cold["deltas"]).strip()
    if not assistant:
        raise AssertionError("initial LFM2.5 turn produced no assistant content")
    followup = {
        **base,
        "messages": [
            *initial_messages,
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": "Now reply with exactly: GREEN READY"},
        ],
    }
    followup_cold = collect(engine, followup, timeout)
    followup_warm = collect(engine, followup, timeout)
    passed = (
        followup_warm["cached_tokens"] > 0
        and followup_cold["token_trace"] == followup_warm["token_trace"]
        and followup_cold["deltas"] == followup_warm["deltas"]
    )
    return {
        "gate": "lfm25_apcv2_multiturn_warm_hit",
        "passed": passed,
        "cold": cold,
        "followup_cold": followup_cold,
        "followup_warm": followup_warm,
    }


def run_qwen(engine, timeout: float) -> dict:
    adapter = engine.adapter
    release_ids = list(adapter.thinking_release_token_ids() or ())
    release_text = adapter.tokenizer.decode(release_ids) if release_ids else None
    request = {
        "messages": [
            {
                "role": "user",
                "content": "Think briefly, then answer with exactly the word READY.",
            }
        ],
        "enable_thinking": True,
        "thinking_budget": 2,
        "thinking_budget_mode": "state_aware",
        "thinking_steer_alpha": 0,
        "max_tokens": 12,
        "temperature": 0,
        "return_token_trace": True,
    }
    result = collect(engine, request, timeout)
    trace = result["token_trace"]
    offsets = [
        index
        for index in range(len(trace) - len(release_ids) + 1)
        if trace[index : index + len(release_ids)] == release_ids
    ]
    guard = result["request_controls"].get("thinking_guard") or {}
    passed = bool(
        release_ids
        and release_text == "\n</think>\n\n"
        and offsets
        and guard.get("forced_close") is True
        and result["request_controls"]
        .get("thinking_mechanisms", {})
        .get("state_aware_guard")
    )
    return {
        "gate": "qwen_full_template_thinking_release",
        "passed": passed,
        "release_token_ids": release_ids,
        "release_text": release_text,
        "release_offsets_in_committed_trace": offsets,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=("lfm25", "qwen"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_own_the_gpu:
        parser.error("explicit GPU ownership is required")

    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    from mlx2.serving import ServingEngine

    model = args.model.expanduser().resolve()
    engine = ServingEngine(
        str(model),
        max_lanes=1,
        max_inflight=2,
        max_context=4096,
        default_max_tokens=32,
        cache_bytes=512 << 20,
        mtp=False,
        prefill_step=128,
        qualification_mode=True,
    )
    try:
        if not engine.ready.wait(args.timeout) or engine.error:
            raise RuntimeError(engine.error or "startup timeout")
        status = engine.status()
        gate = (
            run_lfm(engine, args.timeout)
            if args.kind == "lfm25"
            else run_qwen(engine, args.timeout)
        )
        payload = {
            "schema": "mlx2.upstream-intake-live-candidate.v1",
            "timestamp": datetime.now(UTC).isoformat(),
            "qualification_state": "candidate_evidence",
            "formal_qualification": False,
            "preflight_run": False,
            "performance_claim": False,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
            "device": mx.device_info().get("device_name"),
            "artifact": artifact_identity(model),
            "runtime": status.get("runtime"),
            "settings": status.get("settings"),
            "route_qualification": status.get("qualification"),
            "gate": gate,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "artifact": model.name,
                    "gate": gate["gate"],
                    "passed": gate["passed"],
                    "qualification_state": payload["qualification_state"],
                },
                sort_keys=True,
            )
        )
        if not gate["passed"]:
            raise SystemExit(1)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
