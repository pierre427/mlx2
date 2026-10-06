#!/usr/bin/env python3
"""Bounded real-model smoke for Nemotron-H exact-prefix cache reuse.

The caller must hold both lab GPU locks and a live CPG lease.  This is a
direct-model functional smoke, not serving qualification or a performance
run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKS = (
    Path("/Users/Shared/mlxuag/gpu.lock/owner.json"),
    Path("/tmp/gpu.lock/owner.json"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity() -> dict:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    )
    if status:
        raise RuntimeError("Nemotron smoke requires a clean tracked source tree")
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src" / "mlx2").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return {"git_head": head, "tracked_tree_clean": True, "src_sha256": digest.hexdigest()}


def _ownership(label: str, cpg_radio: Path) -> dict:
    locks = [json.loads(path.read_text()) for path in LOCKS]
    if locks[0] != locks[1]:
        raise RuntimeError("GPU lock owner receipts disagree")
    owner = locks[0]
    if owner.get("label") != label or not owner.get("lease_id"):
        raise RuntimeError("GPU locks do not belong to this smoke")
    try:
        os.kill(int(owner["pid"]), 0)
    except (KeyError, TypeError, ValueError, ProcessLookupError) as exc:
        raise RuntimeError("GPU lock owner is not live") from exc
    radio = json.loads(cpg_radio.read_text())
    claim = radio.get("claim_task")
    if not isinstance(claim, dict) or claim.get("claimed") is not True:
        raise RuntimeError("CPG GPU lease was not claimed")
    expiry = claim.get("lease_expires_at")
    if not isinstance(expiry, (int, float)) or expiry <= time.time():
        raise RuntimeError("CPG GPU lease is expired or unbounded")
    if radio.get("agent_id") != claim.get("owner_agent_id"):
        raise RuntimeError("CPG lease owner does not match the registered job")
    return {"locks": owner, "cpg": claim, "cpg_agent_id": radio["agent_id"]}


def _cache_diff(left, right, mx) -> dict:
    from mlx2.runtime.models.cache import ArraysCache

    if len(left) != len(right) or [type(row) for row in left] != [type(row) for row in right]:
        raise RuntimeError("cache topology differs")
    checks = []
    offsets_match = True
    for index, (actual, expected) in enumerate(zip(left, right)):
        if isinstance(actual, ArraysCache):
            pairs = list(zip(actual.cache, expected.cache))
            kind = "recurrent"
        else:
            offsets_match &= actual.offset == expected.offset
            pairs = list(zip(actual.keys_and_values(), expected.keys_and_values()))
            kind = "attention"
        for plane, (a, b) in enumerate(pairs):
            if a is None or b is None:
                checks.append(
                    {"layer": index, "kind": kind, "plane": plane, "exact": a is b, "max_abs": None}
                )
                continue
            delta = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
            exact = mx.all(a == b)
            mx.eval(delta, exact)
            checks.append(
                {
                    "layer": index,
                    "kind": kind,
                    "plane": plane,
                    "exact": bool(exact.item()),
                    "max_abs": float(delta.item()),
                }
            )
    return {
        "exact": offsets_match and all(row["exact"] for row in checks),
        "offsets_match": offsets_match,
        "max_abs": max((row["max_abs"] or 0.0 for row in checks), default=0.0),
        "planes": checks,
    }


def _forward_token(model, cache, token: int, mx):
    logits = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
    mx.eval(logits)
    return logits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock-label", required=True)
    parser.add_argument("--cpg-radio", type=Path, required=True)
    args = parser.parse_args()

    started = time.time()
    source = _source_identity()
    ownership = _ownership(args.lock_label, args.cpg_radio)
    os.environ.update(MLX_ENABLE_TF32="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import mlx.core as mx

    from mlx2.adapters.nemotron35_lightning import Nemotron35LightningAdapter

    load_started = time.monotonic()
    adapter = Nemotron35LightningAdapter(str(args.model))
    load_seconds = time.monotonic() - load_started
    model = adapter.model
    prompt_ids = adapter.prompt_tokens(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with one short sentence about exact cache state.",
                }
            ],
            "enable_thinking": False,
        }
    )
    prompt = mx.array([prompt_ids], dtype=mx.uint32)
    base = model.make_cache()
    prompt_logits = model(prompt, cache=base)
    mx.eval(prompt_logits)

    generation = copy.deepcopy(base)
    current = int(mx.argmax(prompt_logits[0, -1]).item())
    candidates = []
    for _ in range(3):
        candidates.append(current)
        logits = _forward_token(model, generation, current, mx)
        current = int(mx.argmax(logits[0, -1]).item())

    accepted = 2
    reference_prefix = copy.deepcopy(base)
    for token in candidates[:accepted]:
        _forward_token(model, reference_prefix, token, mx)

    verification = adapter.verify_exact_prefix_path(
        copy.deepcopy(base), candidates, capture_layers=(len(model.layers) - 1,)
    )
    branches, mechanism = verification.commit_and_fork(accepted, sibling_count=2)
    committed_a = _cache_diff(branches[0], reference_prefix, mx)
    committed_b = _cache_diff(branches[1], reference_prefix, mx)

    reference_extended = copy.deepcopy(reference_prefix)
    expected_logits = _forward_token(model, reference_extended, candidates[accepted], mx)
    actual_logits = _forward_token(model, branches[0], candidates[accepted], mx)
    logit_delta = mx.max(
        mx.abs(actual_logits.astype(mx.float32) - expected_logits.astype(mx.float32))
    )
    logits_exact = mx.all(actual_logits == expected_logits)
    mx.eval(logit_delta, logits_exact)
    extended = _cache_diff(branches[0], reference_extended, mx)
    sibling_unchanged = _cache_diff(branches[1], reference_prefix, mx)

    passed = all(
        (
            committed_a["exact"],
            committed_b["exact"],
            extended["exact"],
            sibling_unchanged["exact"],
            bool(logits_exact.item()),
            mechanism["common_tokens_recomputed"] == 0,
            mechanism["recurrent_layers"] == 23,
            mechanism["attention_layers"] == 6,
        )
    )
    result = {
        "schema": "mlx2.nemotron-exact-prefix-real-model-smoke.v1",
        "status": "passed" if passed else "failed",
        "source": source,
        "harness": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "ownership": ownership,
        "artifact": adapter.identity,
        "model_type": model.model_type,
        "mlx_version": getattr(mx, "__version__", None),
        "prompt_tokens": len(prompt_ids),
        "candidate_tokens": candidates,
        "accepted_tokens": accepted,
        "load_seconds": load_seconds,
        "mechanism_receipt": mechanism,
        "checks": {
            "committed_branch_0": committed_a,
            "committed_branch_1": committed_b,
            "extended_branch": extended,
            "sibling_unchanged": sibling_unchanged,
            "continuation_logits_exact": bool(logits_exact.item()),
            "continuation_logits_max_abs": float(logit_delta.item()),
        },
        "states": {
            "implemented": True,
            "smoke_passed": passed,
            "qualified": False,
            "selected": False,
            "observed_used_in_this_smoke": True,
        },
        "started_at": started,
        "finished_at": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output), "checks": result["checks"]}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
