#!/usr/bin/env python3
"""Nemotron-H exact tree verify: real-artifact per-node and commit parity.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them.  The unit, in one process, on the Super VL NVFP4
export with the external route's exact kernel policy:

for each context (default 1K / 8K / 32K tokens of a non-repeating prompt):

1. prefill once; build a 16-row tree whose trunk is the ordinary greedy
   continuation and whose branches take the second-ranked target token at
   depths 1..4 (plus one grandchild per branch), i.e. realistic drafts;
2. ``NemotronTreeExecutor`` vs the serial oracle (``HybridTreeExecutor``:
   ordinary one-token decode of every node from a snapshot of its parent):
   logits and taps bit for bit, per node;
3. the trunk's rows vs the exact chain verify (``forward_with_taps`` on a
   speculating copy), bit for bit;
4. for several accepted paths (trunk, branch, grandchild): committed
   conv / SSM / KV planes vs the oracle's commit, and the next token's
   logits, bit for bit;
5. receipts (fused or serial backend, parity gate verdicts) and rough
   wall times (no thermal hold: attribution only, never statistics).

Every result names the checkpoint fingerprint and the mlx2 commit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
TARGET = Path.home() / "mlx-models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
EXACT_POLICY = {
    "mamba_conv": "fused_short",
    "mamba_projection": "short_row_exact",
    "mamba_segment": "layerwise_short_exact_routed_batch",
    "mamba_ssm": "fused_short_gate",
}
CAPTURES = (1, 30, 60)


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def source():
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", "src"]).returncode != 0
    return {"commit": commit, "src_dirty": dirty}


def checkpoint():
    from mlx2.adapters.registry import inspect_model

    return f"{TARGET.name}@sha256:{inspect_model(TARGET).artifact_fingerprint}"


def _prompt(tokenizer, length):
    text = " ".join(
        f"Ledger {i}: crate {i * 37 % 1009} moved from bay {i * 11 % 97} to dock "
        f"{i * 53 % 211} at {i % 24:02d}:{i * 7 % 60:02d} carrying {i * 13 % 89} units."
        for i in range(length // 8 + 64)
    )
    ids = tokenizer.encode(text)
    if len(ids) < length:
        raise ValueError("prompt generator too short")
    return ids[:length]


def _bits(value):
    import mlx.core as mx
    import numpy as np

    mx.eval(value)
    view = mx.uint16 if value.dtype in (mx.bfloat16, mx.float16) else mx.uint32
    return np.asarray(value.view(view))


def _equal(left, right):
    import numpy as np

    a, b = _bits(left), _bits(right)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _digest(value):
    return hashlib.sha256(_bits(value).tobytes()).hexdigest()[:16]


def _planes_equal(live, oracle):
    from mlx2.runtime.models.cache import ArraysCache

    mismatches = []
    for index, (left, right) in enumerate(zip(live, oracle, strict=True)):
        if isinstance(left, ArraysCache):
            for plane, (a, e) in enumerate(zip(left.cache, right.cache, strict=True)):
                if not _equal(a, e):
                    mismatches.append(f"layer-cache {index} plane {plane}")
        else:
            if left.offset != right.offset:
                mismatches.append(f"layer-cache {index} offset {left.offset}!={right.offset}")
                continue
            for plane, (a, e) in enumerate(zip(left.state, right.state, strict=True)):
                if not _equal(a[..., : left.offset, :], e[..., : right.offset, :]):
                    mismatches.append(f"layer-cache {index} kv {plane}")
    return mismatches


def _tree(model, cache, anchor):
    """Trunk = ordinary greedy; branches = second-ranked target tokens."""
    import mlx.core as mx

    probe = copy.deepcopy(cache)
    trunk, runner_up, token = [], [], anchor
    for _ in range(8):
        logits = model(mx.array([[token]]), probe)[0, -1].astype(mx.float32)
        ranked = mx.argsort(-logits)[:2]
        mx.eval(ranked)
        token = int(ranked[0].item())
        trunk.append(token)
        runner_up.append(int(ranked[1].item()))
    tokens = [anchor] + trunk[:7]
    parents = [-1, 0, 1, 2, 3, 4, 5, 6]
    for depth in range(4):  # a sibling of trunk node depth+1
        tokens.append(runner_up[depth])
        parents.append(depth)
        tokens.append(trunk[(depth + 3) % 8])  # one grandchild under it
        parents.append(len(tokens) - 2)
    return tokens, parents


def _paths(parents):
    paths = []
    for row, parent in enumerate(parents):
        paths.append([row] if parent < 0 else paths[parent] + [row])
    return paths


def unit(out, args):
    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
    from mlx2.runtime.models.nemotron_mamba_tree import parity_status
    from mlx2.runtime.nemotron_tree_executor import NemotronTreeExecutor

    report = {
        "schema": "mlx2.nemotron-tree-check.v1", "checkpoint": checkpoint(),
        "source": source(), "mlx": mx.__version__, "policy": EXACT_POLICY,
        "captures": list(CAPTURES), "qualified": False,
        "evidence_kind": "functional exactness; wall times are attribution only, no thermal hold",
        "contexts": {},
    }
    adapter = resolve_adapter(TARGET)(str(TARGET), execution_policy=dict(EXACT_POLICY))
    model = adapter.model
    report["layout"] = adapter.layout
    for context in args.contexts:
        row = {}
        ids = _prompt(adapter.tokenizer, context)
        cache = model.make_cache()
        model._checkpoint_logits(mx.array([ids[:-1]]), cache)
        mx.eval([entry.state for entry in cache])
        tokens, parents = _tree(model, cache, ids[-1])
        row["tree"] = {"tokens": tokens, "parents": parents}

        started = time.perf_counter()
        oracle_logits, oracle_taps, oracle_tx = HybridTreeExecutor().forward(
            model, tokens, parents, cache, CAPTURES, cached=False
        )
        mx.eval(oracle_logits, oracle_taps)
        row["oracle_seconds"] = time.perf_counter() - started
        oracle_tx.abort()

        executor = NemotronTreeExecutor()
        timings = []
        for repeat in range(3):
            live = copy.deepcopy(cache)
            mx.eval([entry.state for entry in live])
            started = time.perf_counter()
            logits, taps, transaction = executor.forward(
                model, tokens, parents, live, CAPTURES, cached=False
            )
            mx.eval(logits, taps)
            timings.append(time.perf_counter() - started)
            transaction.abort()
        row["tree_seconds"] = timings
        per_node = [
            bool(_equal(logits[:, node], oracle_logits[:, node])
                 and _equal(taps[:, node], oracle_taps[:, node]))
            for node in range(len(tokens))
        ]
        row["per_node_bitwise"] = per_node
        row["all_nodes_bitwise"] = all(per_node)
        row["logits_digest"] = _digest(logits)

        # Chain verify of the trunk (rows 0..7) on a speculating copy.
        owner = HybridVerifyRows([copy.deepcopy(cache)])
        chain_tx = owner.begin([8])
        started = time.perf_counter()
        chain_logits, chain_taps = model.forward_with_taps(
            mx.array([tokens[:8]]), chain_tx.caches, CAPTURES
        )
        mx.eval(chain_logits, chain_taps)
        row["chain8_seconds"] = time.perf_counter() - started
        chain_tx.abort()
        row["trunk_vs_chain_bitwise"] = [
            bool(_equal(logits[:, node], chain_logits[:, node])) for node in range(8)
        ]

        paths = _paths(parents)
        commits = {}
        for terminal in (0, 7, 8, 9, 12, 15):
            path = paths[terminal]
            live = copy.deepcopy(cache)
            _l, _t, transaction = executor.forward(
                model, tokens, parents, live, CAPTURES, cached=False
            )
            mx.eval(_l, _t)
            transaction.commit_paths([path])
            oracle = copy.deepcopy(cache)
            _ol, _ot, otx = HybridTreeExecutor().forward(
                model, tokens, parents, oracle, CAPTURES, cached=False
            )
            mx.eval(_ol, _ot)
            otx.commit_paths([path])
            mismatches = _planes_equal(live, oracle)
            step = mx.array([[tokens[terminal]]])
            commits[str(terminal)] = {
                "path": path,
                "plane_mismatches": mismatches,
                "next_logits_bitwise": bool(_equal(model(step, live), model(step, oracle))),
            }
            del live, oracle
            mx.clear_cache()
        row["commits"] = commits
        row["receipt"] = executor.receipt()
        row["receipt"].pop("source_identity", None)
        report["contexts"][str(context)] = row
        print(json.dumps({"context": context, "all_nodes_bitwise": row["all_nodes_bitwise"],
                          "trunk_vs_chain": all(row["trunk_vs_chain_bitwise"]),
                          "commits_exact": all(not c["plane_mismatches"] and c["next_logits_bitwise"]
                                               for c in commits.values()),
                          "backend": row["receipt"]["mamba_backend"],
                          "tree_s": min(timings), "chain8_s": row["chain8_seconds"]}), flush=True)
        atomic(out / "report.json", report)
        del cache
        mx.clear_cache()
    report["kernel_parity"] = parity_status()
    report["diagnostics_tree_verify"] = adapter.diagnostics().get("tree_verify")
    atomic(out / "report.json", report)
    adapter.close()
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    job_id = "nemotron-tree-check-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=checkpoint(),
                         source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--contexts", *map(str, args.contexts)]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 8192, 32768])
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    raise SystemExit(main())
