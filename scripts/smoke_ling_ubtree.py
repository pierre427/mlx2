#!/usr/bin/env python3
"""Owned, bounded Ling ordinary/UBTree AST smoke; never qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thermal(repo: Path) -> dict:
    result = subprocess.run(
        ["/usr/bin/swift", str(repo / "scripts/thermal_probe.swift")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def ownership() -> dict:
    rows = []
    for path in (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock")):
        owner = path / "owner.json"
        if not path.is_dir() or not owner.is_file():
            raise RuntimeError(f"missing owned GPU lock: {path}")
        rows.append({"path": str(path), "owner": json.loads(owner.read_text())})
    if rows[0]["owner"] != rows[1]["owner"]:
        raise RuntimeError("paired GPU lock receipts do not match")
    return {"paired_locks": rows, "matching": True}


def drain(batch, prompt: list[int], max_tokens: int) -> tuple[list[int], dict | None]:
    (uid,) = batch.insert([prompt], max_tokens=[max_tokens])
    tokens: list[int] = []
    last_receipt = None
    for _ in range(max_tokens * 4 + 32):
        _, responses = batch.next()
        failures = batch.take_lane_failures()
        if failures:
            raise RuntimeError(f"Ling smoke lane failure: {failures[0]}")
        for response in responses:
            if response.uid != uid:
                continue
            tokens.append(int(response.token))
            speculative_receipt = getattr(response, "speculative_receipt", None)
            if speculative_receipt is not None:
                last_receipt = dict(speculative_receipt)
            if response.finish_reason is not None:
                return tokens, last_receipt
    raise RuntimeError("bounded Ling smoke stalled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--thermal-wait-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.max_tokens < 2 or args.max_tokens > 64:
        parser.error("--max-tokens must be in 2..64")
    if not 0 <= args.thermal_wait_seconds <= 1800:
        parser.error("--thermal-wait-seconds must be in 0..1800")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("refusing to overwrite a nonempty evidence directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    for command in (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
    ):
        if subprocess.run(command, cwd=repo, check=False).returncode:
            raise RuntimeError("Ling smoke requires committed source changes")
    initial_thermal = thermal(repo)

    receipt = {
        "schema": "mlx2.ling-ubtree-smoke.v1",
        "mlx2_revision": revision,
        "implemented": True,
        "qualified": False,
        "performance_claim": False,
        "scope": "one artifact; ordinary reference then explicit UBTree AST",
        "model": str(args.model.resolve()),
        "drafter": str(args.drafter.resolve()),
        "ownership": ownership(),
        "thermal": [initial_thermal],
        "swap_before": subprocess.check_output(
            ["/usr/sbin/sysctl", "vm.swapusage"], text=True
        ).strip(),
        "source_sha256": {
            path: sha256(repo / path)
            for path in (
                "src/mlx2/adapters/ling3_flash.py",
                "src/mlx2/adapters/ubtree.py",
                "src/mlx2/runtime/drafters/ubtree.py",
                "src/mlx2/runtime/drafters/ubtree_tree.py",
                "src/mlx2/runtime/hybrid_tree_executor.py",
                "src/mlx2/runtime/models/bailing_hybrid.py",
                "scripts/smoke_ling_ubtree.py",
            )
        },
        "passed": False,
        "phase": "thermal_gate",
    }
    output = args.output_dir / "receipt.json"
    atomic_json(output, receipt)
    deadline = time.monotonic() + args.thermal_wait_seconds
    while int(receipt["thermal"][-1]["thermal_state"]) != 0:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Ling smoke timed out waiting for nominal thermals")
        time.sleep(min(15.0, remaining))
        sample = thermal(repo)
        sample["waited_seconds"] = args.thermal_wait_seconds - max(
            0.0, deadline - time.monotonic()
        )
        receipt["thermal"].append(sample)
        atomic_json(output, receipt)
    receipt["phase"] = "starting"
    atomic_json(output, receipt)

    import mlx.core as mx

    from mlx2.adapters.ling3_flash import Ling3FlashAdapter
    from mlx2.runtime.generate import BatchGenerator

    mx.set_default_device(mx.gpu)
    policy = {
        "draft_model": str(args.drafter.resolve()),
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
    }
    adapter = Ling3FlashAdapter(args.model, execution_policy=policy)
    prompt = adapter.prompt_tokens(
        {"prompt": "State the two prime numbers immediately after 40."}
    )
    stops = [(int(token),) for token in adapter.tokenizer.eos_token_ids]
    receipt.update(
        phase="loaded",
        artifact_identity=dict(adapter.identity),
        target_fingerprint=adapter._external_target_revision,
        drafter_fingerprint=adapter._external_draft_revision,
        execution_policy=policy,
        prompt_tokens=len(prompt),
        adapter_diagnostics=adapter.diagnostics(),
    )
    atomic_json(output, receipt)

    ordinary = BatchGenerator(
        adapter.model,
        completion_batch_size=1,
        prefill_batch_size=1,
        stop_tokens=stops,
    )
    try:
        started = time.monotonic()
        reference_tokens, _ = drain(ordinary, prompt, args.max_tokens)
        receipt["ordinary"] = {
            "tokens": reference_tokens,
            "text": adapter.tokenizer.decode(reference_tokens),
            "seconds": time.monotonic() - started,
            "reference": True,
        }
    finally:
        ordinary.close()
    mx.clear_cache()
    receipt["phase"] = "ordinary_complete"
    atomic_json(output, receipt)

    ast = adapter.create_external_batch(
        completion_batch_size=1,
        stop_tokens=stops,
    )
    try:
        started = time.monotonic()
        ast_tokens, route_receipt = drain(ast, prompt, args.max_tokens)
        stats = {
            key: int(value)
            for key, value in ast.scheduler_stats.items()
            if key.startswith("external_")
            and isinstance(value, (bool, int, float))
            and int(value)
        }
        receipt["ast"] = {
            "tokens": ast_tokens,
            "text": adapter.tokenizer.decode(ast_tokens),
            "seconds": time.monotonic() - started,
            "route_receipt": route_receipt,
            "scheduler_stats": stats,
        }
    finally:
        ast.close()
    receipt["output_token_equal"] = reference_tokens == ast_tokens
    target_tree = (route_receipt or {}).get("target_tree", {})
    receipt["ast_observed_used"] = (
        target_tree.get("provider") == "hybrid_tree_serial_exact_v1"
        and int(target_tree.get("dispatches", 0)) > 0
    )
    if not receipt["output_token_equal"]:
        raise RuntimeError("UBTree AST changed greedy target output")
    if not receipt["ast_observed_used"]:
        raise RuntimeError("UBTree AST target-tree provider was not observed used")
    receipt["thermal"].append(thermal(repo))
    receipt["swap_after"] = subprocess.check_output(
        ["/usr/sbin/sysctl", "vm.swapusage"], text=True
    ).strip()
    receipt["phase"] = "complete"
    receipt["passed"] = True
    atomic_json(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
