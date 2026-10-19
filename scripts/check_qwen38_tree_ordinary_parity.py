#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""One-arm Qwen3.8 27B greedy token capture: TensorFold tree route or ordinary.

Starts one loopback ``mlx2.server`` from ``--src`` (one model per process),
sends a fixed set of varied, realistic prompts one at a time with greedy
decoding and per-token logprobs, records every completion's token IDs, route
receipt and wall time, snapshots ``/v1/status`` before and after, and stops
the server's whole process group in ``finally``.  ``--arm tree`` selects the
adapter's artifact-bound external-draft default (tree15 batch-size route for
the pinned target/DFlash2 pair); ``--arm ordinary`` selects ``--ordinary``.
``--compare`` reports the first differing token per prompt between receipts.

Never changes the product default or qualifies a route.  It records whether
the requested arm was selected and observed for these requests.  Refuses a
nonempty ``--out`` so retries stay distinguishable; run under
``scripts/run_with_gpu_locks.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / "mlx-models" / "Qwen3.8-27B-oQ4e-mtp"
DRAFT = Path.home() / "mlx-models" / "Qwen3.8-27B-DFlash2"
POLICY = {
    "draft_model": str(DRAFT),
    "draft_revision": "34ec93d71399f3dd6db9646194f1ad4db345715c61258d72318325e0d8095c90",
    "target_revision": "e59471c5c6fa8c6819b81cb5957bcab10736020db8bacb47fbb9089813fa93f8",
}
PROMPTS = {
    "prose_history": "Explain in three paragraphs how the printing press changed the spread of "
                     "scientific ideas in sixteenth-century Europe, with two concrete examples.",
    "code_python": "Write a Python function that parses an ISO 8601 duration such as "
                   "'P3DT4H12M' into a datetime.timedelta, with docstring, input validation and "
                   "three pytest test cases.",
    "explain_systems": "A web service's p99 latency doubled after a deploy while p50 stayed flat. "
                       "Walk through how you would investigate it, which metrics you would pull, "
                       "and the three most likely causes.",
    "code_sql": "Given tables orders(id, customer_id, placed_at, total_cents) and "
                "customers(id, name, country), write a PostgreSQL query returning each country's "
                "top three customers by 2025 revenue, then explain the window function you used.",
    "story": "Write a short scene, about 250 words, in which a lighthouse keeper on a remote "
             "island receives an unexpected letter during a storm. Use vivid sensory detail.",
}


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _get(base: str, path: str, timeout: float = 5.0) -> dict:
    with urlopen(base + path, timeout=timeout) as response:
        return json.load(response)


def _completion(base: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "enable_thinking": False, "logprobs": True, "top_logprobs": 0,
            "skip_writing_prefix_cache": True, "seed": 1017}
    request = Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    wall = time.perf_counter() - start
    choice = result["choices"][0]
    events = (choice.get("logprobs") or {}).get("content") or []
    ids = [event.get("id") for event in events]
    if not ids or len(ids) != result["usage"]["completion_tokens"] or any(type(i) is not int for i in ids):
        raise RuntimeError("complete integer logprob token IDs are required")
    text = choice["message"].get("content") or ""
    return {"token_ids": ids, "completion_tokens": len(ids), "wall_s": wall,
            "tok_per_s": len(ids) / wall if wall > 0 else None,
            "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text,
            "finish_reason": choice.get("finish_reason"), "route_receipt": result.get("mlx2")}


def _stop(process) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def _scheduler_numbers(status: dict) -> dict:
    out = {}

    def walk(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}" if prefix else key, item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if (
                prefix == "scheduler.external_rounds"
                or any(word in prefix for word in ("tree", "tensorfold", "fallback", "restore", "chain"))
            ):
                out[prefix] = value
    walk("", status)
    return out


def _source_identity(src: Path, model: Path) -> dict:
    root = src.resolve().parent
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src"], cwd=root, text=True
    ).strip()
    env = dict(os.environ, PYTHONPATH=str(src.resolve()))
    code = (
        "import json,sys; from mlx2.adapters.registry import inspect_model; "
        "r=inspect_model(sys.argv[1]); i=r.artifact['identity']; "
        "print(json.dumps({"
        "'artifact_name':i['path'].rstrip('/').rsplit('/',1)[-1],"
        "'artifact_fingerprint':i['fingerprint']}))"
    )
    artifact = json.loads(subprocess.check_output(
        [sys.executable, "-c", code, str(model.resolve())], env=env, text=True
    ))
    return {
        "mlx2_commit": head,
        "src": str(src.resolve()),
        "tracked_source_dirty": bool(tracked),
        "tracked_source_status": tracked,
        **artifact,
        "checkpoint": f"{artifact['artifact_name']}@{artifact['artifact_fingerprint']}",
    }


def _route_observation(arm: str, prompts: dict, status_deltas: dict) -> dict:
    rows = [row.get("route_receipt") or {} for row in prompts.values()]
    external = [row for row in rows if row.get("route") == "external_draft"]
    providers = []
    request_rounds = []
    request_dispatches = 0
    ordinary_fallbacks = 0
    for row in external:
        speculation = row.get("speculation") or {}
        target = speculation.get("target_tree") or {}
        providers.append(target.get("provider"))
        rounds = speculation.get("external_rounds")
        if type(rounds) is int:
            request_rounds.append(rounds)
        request_dispatches += sum(
            int(value)
            for value in (target.get("lane_width_histogram") or {}).values()
        )
        ordinary_fallbacks += int(bool(speculation.get("ordinary_fallback")))
    selected = bool(rows) and all(
        row.get("route") == ("external_draft" if arm == "tree" else "ordinary")
        for row in rows
    )
    observed = selected and (
        arm == "ordinary"
        or (
            len(request_rounds) == len(rows)
            and all(rounds > 0 for rounds in request_rounds)
            and all(provider == "qwen38_tensorfold_v1" for provider in providers)
            and ordinary_fallbacks == 0
        )
    )
    return {
        "selected_for_requests": selected,
        "observed_used": observed,
        "request_count": len(rows),
        "external_draft_request_count": len(external),
        "tensorfold_provider_request_count": sum(
            provider == "qwen38_tensorfold_v1" for provider in providers
        ),
        "ordinary_fallback_request_count": ordinary_fallbacks,
        # Request receipts and scheduler counters have different ownership and
        # must not be presented as the same round count.
        "request_receipt_external_rounds_sum": sum(request_rounds),
        "request_receipt_target_dispatches_sum": request_dispatches,
        "scheduler_external_rounds": status_deltas.get("scheduler.external_rounds"),
        "scheduler_external_auto_tree_rounds": status_deltas.get(
            "scheduler.external_auto_tree_rounds"
        ),
        "scheduler_external_tree_rounds": status_deltas.get(
            "scheduler.external_tree_rounds"
        ),
        "scheduler_target_dispatches": status_deltas.get(
            "scheduler.external_target_tree_target_rounds"
        ),
        "scheduler_snapshot_state": (
            "caught_up"
            if int(status_deltas.get("scheduler.external_target_tree_target_rounds", 0))
            >= request_dispatches
            else "periodic_snapshot_lag"
        ),
    }


def run_arm(args) -> int:
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing nonempty evidence directory {out}")
    out.mkdir(parents=True, exist_ok=True)
    src = args.src.expanduser().resolve()
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(args.model), "--host",
               "127.0.0.1", "--port", str(args.port), "--max-lanes", "4", "--max-context", "32768",
               "--cache-bytes", str(8 << 30), "--cache-dir", str(out / "cache"),
               "--default-max-tokens", "4096", *args.server_arg]
    if args.arm == "tree":
        _atomic_json(out / "policy.json", POLICY)
        command += ["--external-draft", "--execution-policy", str(out / "policy.json")]
    else:
        command += ["--ordinary"]
    env = dict(os.environ, PYTHONPATH=str(src), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    base = f"http://127.0.0.1:{args.port}"
    source_identity = _source_identity(src, args.model.expanduser().resolve())
    receipt = {"schema": "mlx2.qwen38-tree-ordinary-parity.v2", "arm": args.arm, "src": str(src),
               "command": command, "model": str(args.model), "max_tokens": args.max_tokens,
               "source_identity": source_identity,
               "checkpoint": source_identity["checkpoint"],
               "artifact_fingerprint": source_identity["artifact_fingerprint"],
               "qualified": False,
               "route_selected": None,
               "product_default_changed": False,
               "route_observed_used": False,
               "prompts": {}, "started": time.time()}
    for name in ("lane_glue.py", "lane_fuse.py"):
        path = src / "mlx2/runtime/tensorfold_qwen38" / name
        receipt.setdefault("vendored_sha256", {})[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    process = None
    try:
        with (out / "server.log").open("w") as log:
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("server did not become ready")
                try:
                    status = _get(base, "/v1/status")
                    if status.get("healthy") and status.get("ready", True):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1.0)
            receipt["load_s"] = args.startup_timeout - (deadline - time.monotonic())
            receipt["status_before"] = status
            for name in args.prompts.split(","):
                row = _completion(base, PROMPTS[name], args.max_tokens, args.request_timeout)
                receipt["prompts"][name] = row
                print(json.dumps({"prompt": name, "tokens": row["completion_tokens"],
                                  "tok_per_s": round(row["tok_per_s"], 2)}), flush=True)
            receipt["status_after"] = _get(base, "/v1/status")
            before = _scheduler_numbers(receipt["status_before"])
            after = _scheduler_numbers(receipt["status_after"])
            receipt["status_deltas"] = {key: after[key] - before.get(key, 0) for key in after
                                        if after[key] != before.get(key, 0)}
            receipt["route_observation"] = _route_observation(
                args.arm, receipt["prompts"], receipt["status_deltas"]
            )
            receipt["route_selected"] = receipt["route_observation"][
                "selected_for_requests"
            ]
            receipt["route_observed_used"] = receipt["route_observation"]["observed_used"]
            if not receipt["route_selected"] or not receipt["route_observed_used"]:
                raise RuntimeError("requested route was not selected and observed in every request")
        receipt["passed"] = True
    except Exception as exc:  # noqa: BLE001 - recorded
        receipt["passed"] = False
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _stop(process)
        receipt["finished"] = time.time()
        _atomic_json(out / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "error": receipt.get("error")}))
    return 0 if receipt["passed"] else 1


def compare(paths) -> int:
    """Token-by-token comparison of arm receipts against the first; 1 unless every arm is identical.

    A prompt missing from either receipt, an empty comparison, or a receipt
    that did not pass is a difference, never identical.
    """
    receipts = [json.loads(Path(p).read_text()) for p in paths]
    reference = receipts[0]
    report = {"reference": str(paths[0]), "arms": []}
    for path, other in zip(paths[1:], receipts[1:]):
        rows = {}
        names = list(reference.get("prompts", {})) + [
            name for name in other.get("prompts", {}) if name not in reference.get("prompts", {})
        ]
        for name in names:
            ref = reference.get("prompts", {}).get(name)
            got = other.get("prompts", {}).get(name)
            if ref is None or got is None:
                rows[name] = {"identical": False, "first_difference": None,
                              "missing_from": "reference" if ref is None else "arm"}
                continue
            first = next((i for i, (a, b) in enumerate(zip(ref["token_ids"], got["token_ids"])) if a != b),
                         None)
            if first is None and len(ref["token_ids"]) != len(got["token_ids"]):
                first = min(len(ref["token_ids"]), len(got["token_ids"]))
            rows[name] = {"identical": first is None, "first_difference": first,
                          "tokens": [len(ref["token_ids"]), len(got["token_ids"])]}
        passed = reference.get("passed") is True and other.get("passed") is True
        report["arms"].append({"receipt": str(path), "prompts": rows, "receipts_passed": passed,
                               "identical": passed and bool(rows) and all(r["identical"] for r in rows.values())})
    report["identical"] = bool(report["arms"]) and all(arm["identical"] for arm in report["arms"])
    print(json.dumps(report, indent=2))
    return 0 if report["identical"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("tree", "ordinary"))
    ap.add_argument("--src", type=Path, default=ROOT / "src")
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--port", type=int, default=8913)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--startup-timeout", type=float, default=900)
    ap.add_argument("--request-timeout", type=float, default=600)
    ap.add_argument("--server-arg", action="append", default=[],
                    help="extra mlx2.server argument for both arms, e.g. --server-arg=--lane-matmul=exact")
    ap.add_argument("--compare", nargs="+", type=Path)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare)
    if not (args.arm and args.out):
        ap.error("--arm and --out are required")
    if not args.i_own_the_gpu:
        ap.error("run under scripts/run_with_gpu_locks.py and pass --i-own-the-gpu")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
