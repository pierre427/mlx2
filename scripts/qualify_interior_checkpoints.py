#!/usr/bin/env python3
"""GPU qualification for exact APCv2 interior hybrid checkpoints (rm04).

Launches ``mlx2.server`` once per (round, arm), interleaving arms across rounds
(A B C / C B A ...), drives multi-request workloads over streaming
``/v1/chat/completions``, and records per request: client TTFT, prompt and
cached tokens, interior-hit counter deltas, and resident interior bytes.

Arms: ``off`` (count 0), ``pow2`` (legacy lattice, count 2, stride 64, the
quality-campaign opt-in) and ``auto`` (turn/tail placement preset).

Workloads (all greedy, fixed max_tokens):
  shared_system  long agent system prompt + tool schemas, N new sessions with
                 different first user messages (interior must hit).
  rag            shared long document, different questions (interior must hit).
  branch         a multi-turn conversation, then regenerations from an earlier
                 turn with different user text.
  linear         growing multi-turn conversation: no-harm control (interior
                 hits may be 0; TTFT/memory must not regress).

Correctness: each measured request is replayed once per arm under a unique
``X-Tenant-ID`` (no cache sharing: a cold prefill) and the greedy token text
must match exactly.

Mechanism gate: an arm other than ``off`` whose shared_system/rag workload
records zero interior hits is REFUSED (exit 3): a null arm is not evidence.

Refuses to run without ``--i-own-the-gpu``; ``--dry-run`` prints the plan.
Run under the lab GPU lock wrapper (cpg_job.py ... --lock).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ARMS = {
    "off": {"count": 0, "min_stride": 1},
    "pow2": {"count": 2, "min_stride": 64},
    "auto": "auto",
}
AUTO_POLICY = {
    "count": 4,
    "min_stride": 256,
    "placement": "auto",
    "headroom_fraction": 0.25,
}
WORKLOADS = ("shared_system", "rag", "branch", "linear")
MUST_HIT = ("shared_system", "rag")


def _words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i % 97}-{i}" for i in range(n))


def build_workloads(scale: float) -> dict:
    """Deterministic message lists; token counts scale roughly with ``scale``."""
    w = lambda n: max(8, int(n * scale))  # noqa: E731
    system = {
        "role": "system",
        "content": "You are a careful coding agent. Rules: " + _words(w(6000), "rule")
        + "\nTools: " + _words(w(2000), "tool"),
    }
    out = {}
    out["shared_system"] = [
        [system, {"role": "user", "content": f"Task {k}: " + _words(w(120), f"t{k}_")}]
        for k in range(8)
    ]
    doc = "Document:\n" + _words(w(20000), "doc")
    out["rag"] = [
        [
            {"role": "system", "content": "Answer from the document only."},
            {"role": "user", "content": doc + f"\n\nQuestion {k}: " + _words(w(24), f"q{k}_")},
        ]
        for k in range(6)
    ]
    convo = [system]
    linear = []
    for turn in range(8):
        convo = convo + [
            {"role": "user", "content": f"Step {turn}: " + _words(w(2500), f"s{turn}_")}
        ]
        linear.append(list(convo))
        convo = convo + [
            {"role": "assistant", "content": f"Done with step {turn}. " + _words(w(200), f"a{turn}_")}
        ]
    out["linear"] = linear
    branch = linear[:6]
    base = linear[3]
    for k in range(3):
        branch.append(
            base[:-1] + [{"role": "user", "content": f"Alternative {k}: " + _words(w(300), f"b{k}_")}]
        )
    out["branch"] = branch
    return out


def chat_body(messages, *, model: str, max_tokens: int) -> dict:
    """Greedy streaming request.  ``model`` must be the served model id from
    ``/v1/status``: the server answers 404 "unknown model" to any other name."""
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "enable_thinking": False,
    }


class Client:
    def __init__(self, base: str, timeout: float):
        self.base, self.timeout = base.rstrip("/"), timeout
        self.throttled = 0

    def status(self) -> dict:
        with urllib.request.urlopen(self.base + "/v1/status", timeout=self.timeout) as r:
            return json.loads(r.read())

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def chat(self, messages, *, tenant: str, max_tokens: int, model: str) -> dict:
        body = chat_body(messages, model=model, max_tokens=max_tokens)
        request = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-Tenant-ID": tenant},
            method="POST",
        )
        for attempt in range(4):
            try:
                return self._stream(request)
            except urllib.error.HTTPError as error:
                # Admission backpressure (429) is the server working as
                # designed under a large resident cache, not a failed
                # measurement: back off and retry the same request.
                if error.code != 429 or attempt == 3:
                    raise
                delay = float(error.headers.get("Retry-After", 1) or 1)
                self.throttled += 1
                time.sleep(min(30.0, delay * (attempt + 1)))
        raise RuntimeError("unreachable")

    def _stream(self, request) -> dict:
        start = time.perf_counter()
        ttft = None
        text = []
        usage = None
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or ():
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + (
                        delta.get("reasoning_content") or ""
                    )
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        text.append(piece)
        return {
            "ttft_s": ttft,
            "total_s": time.perf_counter() - start,
            "text": "".join(text),
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": ((usage or {}).get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
        }


def _apcv2(status: dict) -> dict:
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "apcv2" and isinstance(value, dict):
                    found.update(value)
                walk(value)

    walk(status)
    return found


def _mechanism(status: dict) -> dict:
    counts = status.get("counts") or {}
    apc = _apcv2(status)
    lifetime = apc.get("lifetime") or {}
    interior = apc.get("interior") or {}
    return {
        # Gate on the engine's live counter: the ``apcv2`` status block is a
        # periodic snapshot and lagged by a request in the first GPU smoke.
        # (``lifetime`` already includes the live generation; never add both.)
        "interior_hits": int(counts.get("apc_interior_hits", 0)),
        "apc_snapshot_interior_hits": int(
            lifetime.get("interior_hits", apc.get("interior_hits", 0))
        ),
        "turn_boundary_hits": int(counts.get("apc_interior_hits_turn_boundary", 0)),
        "interior_hit_tokens": int(counts.get("apc_interior_hit_tokens", 0)),
        "captured": int(counts.get("apc_interior_checkpoints_captured", 0)),
        "published": int(counts.get("apc_interior_checkpoints_published", 0)),
        "degraded": int(counts.get("apc_interior_checkpoints_degraded", 0)),
        "skipped_publish_failed": int(
            counts.get("apc_interior_checkpoints_skipped_publish_failed", 0)
        ),
        "skipped_approximate": int(
            counts.get("apc_interior_checkpoints_skipped_approximate", 0)
        ),
        "skipped_write_suppressed": int(
            counts.get("apc_interior_checkpoints_skipped_write_suppressed", 0)
        ),
        "headroom_capped": int(counts.get("apc_interior_positions_headroom_capped", 0)),
        "skipped_continuation": int(
            counts.get("apc_interior_requests_skipped_continuation", 0)
        ),
        "budget_mib_last": int(counts.get("apc_interior_budget_mib_last", 0)),
        "deepest_checkpoint_mib_last": int(
            counts.get("apc_interior_deepest_mib_last", 0)
        ),
        "planned_turn": int(counts.get("apc_interior_positions_planned_turn", 0)),
        "planned_tail": int(counts.get("apc_interior_positions_planned_tail", 0)),
        "planned_lattice": int(counts.get("apc_interior_positions_planned_lattice", 0)),
        "memory_admission_deferred": int(counts.get("memory_admission_deferred", 0)),
        "interior_resident_bytes": int(interior.get("resident_bytes", 0)),
        "interior_entries": int(interior.get("entries", 0)),
        "interior_max_entries": int(interior.get("max_entries", 0)),
        "resident_bytes": int((apc.get("idle_disk") or {}).get("resident_bytes", 0)),
    }


def _delta(after: dict, before: dict) -> dict:
    return {
        key: after[key] - before.get(key, 0)
        if key not in {
            "interior_resident_bytes", "interior_entries", "interior_max_entries",
            "resident_bytes", "budget_mib_last", "deepest_checkpoint_mib_last",
        }
        else after[key]
        for key in after
    }


def server_command(args, policy_file: Path) -> list[str]:
    command = [
        sys.executable, "-u", "-m", "mlx2.server",
        "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port),
        "--max-context", str(args.max_context), "--max-lanes", "1",
        "--max-inflight", "4", "--cache-bytes", str(args.cache_gib << 30),
        "--qualification-mode", "--execution-policy", str(policy_file),
    ]
    if args.route == "ordinary":
        command.append("--ordinary")
    return command


def server_env() -> dict:
    """Pin the server to THIS checkout's ``src``.

    ``python -m mlx2.server`` with an inherited environment can import mlx2
    from whichever checkout happens to be importable, so a queued run from a
    worktree could silently measure another branch's engine.  The tree holding
    this script wins, ahead of anything already on PYTHONPATH.
    """
    source = str(Path(__file__).resolve().parents[1] / "src")
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    parts = [source] + [part for part in existing.split(os.pathsep) if part and part != source]
    environment["PYTHONPATH"] = os.pathsep.join(parts)
    return environment


def arm_policy(args, arm: str) -> dict:
    base = json.loads(Path(args.base_policy).read_text()) if args.base_policy else {}
    setting = ARMS[arm]
    fraction = getattr(args, "auto_headroom_fraction", None)
    uncached = getattr(args, "auto_min_uncached_fraction", None)
    if arm == "auto" and (fraction is not None or uncached is not None):
        setting = dict(AUTO_POLICY)
        if fraction is not None:
            setting["headroom_fraction"] = float(fraction)
        if uncached is not None:
            setting["min_uncached_fraction"] = float(uncached)
    return {**base, "apc_interior_checkpoints": setting}


def run_arm(args, arm: str, round_index: int, workloads: dict, reference: dict, log_dir: Path):
    policy = arm_policy(args, arm)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(policy, handle)
        policy_file = Path(handle.name)
    log_path = log_dir / f"server-{arm}-r{round_index}.log"
    command = server_command(args, policy_file)
    with open(log_path, "w") as log:
        server = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=server_env(),
            start_new_session=True,
        )
    client = Client(f"http://127.0.0.1:{args.port}", args.timeout)
    try:
        deadline = time.monotonic() + args.startup_timeout
        while not client.healthy():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"server for arm {arm} failed to start; see {log_path}")
            time.sleep(2)
        initial = client.status()
        served_model = initial.get("model")
        if not served_model:
            raise RuntimeError(f"/v1/status reported no model for arm {arm}")
        settings = (initial.get("settings") or {}).get("apc_interior_checkpoints")
        results = {"arm": arm, "round": round_index, "settings": settings, "workloads": {}}
        for name, requests in workloads.items():
            tenant = f"rm04-{name}-{arm}-r{round_index}-{uuid.uuid4().hex[:6]}"
            rows = []
            for index, messages in enumerate(requests):
                before = _mechanism(client.status())
                warm = client.chat(
                    messages, tenant=tenant, max_tokens=args.max_tokens, model=served_model
                )
                after = _mechanism(client.status())
                row = {"index": index, **warm, "mechanism": _delta(after, before)}
                # Per (arm, round): the reference must come from the same
                # server process.  Round 1 warm answers differed from round 0
                # cold answers even for requests with no cache hit at all, so
                # this model's greedy output is reproducible within a process
                # but not across restarts.
                key = (arm, round_index, name, index)
                if key not in reference:
                    cold = client.chat(
                        messages, tenant=f"cold-{uuid.uuid4().hex}",
                        max_tokens=args.max_tokens, model=served_model,
                    )
                    reference[key] = cold["text"]
                    row["cold_ttft_s"] = cold["ttft_s"]
                    if index == 0:
                        # Determinism control: a second cold-tenant replay of
                        # the same prompt.  A warm-vs-cold diff is evidence
                        # against checkpoint exactness only where the model
                        # itself reproduces its own greedy output.
                        repeat = client.chat(
                            messages, tenant=f"cold-{uuid.uuid4().hex}",
                            max_tokens=args.max_tokens, model=served_model,
                        )
                        row["cold_matches_cold"] = repeat["text"] == cold["text"]
                row["matches_cold"] = warm["text"] == reference[key]
                rows.append(row)
                print(
                    f"[{arm} r{round_index}] {name}#{index} P={warm['prompt_tokens']} "
                    f"cached={warm['cached_tokens']} ttft={warm['ttft_s']:.3f}s "
                    f"interior_hits+={row['mechanism']['interior_hits']} "
                    f"match={row['matches_cold']}",
                    flush=True,
                )
            results["workloads"][name] = rows
        results["final_mechanism"] = _mechanism(client.status())
        results["throttled_requests"] = client.throttled
        return results
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        except Exception:  # noqa: BLE001
            os.killpg(server.pid, signal.SIGKILL)
        policy_file.unlink(missing_ok=True)


def summarize(runs: list, workloads) -> dict:
    summary = {"arms": {}, "gates": {}}
    by_arm = {}
    for run in runs:
        by_arm.setdefault(run["arm"], []).append(run)
    for arm, arm_runs in by_arm.items():
        entry = {}
        for name in workloads:
            rows = [row for run in arm_runs for row in run["workloads"].get(name, ())]
            later = [row for row in rows if row["index"] > 0]
            entry[name] = {
                "median_ttft_s_after_first": statistics.median(
                    [row["ttft_s"] for row in later if row["ttft_s"] is not None]
                ) if later else None,
                "cached_fraction_after_first": (
                    sum(row["cached_tokens"] or 0 for row in later)
                    / max(1, sum(row["prompt_tokens"] or 0 for row in later))
                ),
                "interior_hits": sum(row["mechanism"]["interior_hits"] for row in rows),
                "turn_boundary_hits": sum(row["mechanism"]["turn_boundary_hits"] for row in rows),
                "max_interior_resident_bytes": max(
                    (row["mechanism"]["interior_resident_bytes"] for row in rows), default=0
                ),
                "memory_admission_deferred": sum(
                    row["mechanism"]["memory_admission_deferred"] for row in rows
                ),
                "correctness_diffs": sum(1 for row in rows if not row["matches_cold"]),
                "cold_replay_diffs": sum(
                    1 for row in rows if row.get("cold_matches_cold") is False
                ),
            }
        summary["arms"][arm] = entry
    refused = []
    for arm, entry in summary["arms"].items():
        if arm == "off":
            continue
        for name in MUST_HIT:
            if name in entry and entry[name]["interior_hits"] == 0:
                refused.append(f"{arm}/{name}: interior_hits == 0 (mechanism did not run)")
    diffs = sum(
        wl["correctness_diffs"] for entry in summary["arms"].values() for wl in entry.values()
    )
    cold_diffs = sum(
        wl.get("cold_replay_diffs", 0)
        for entry in summary["arms"].values()
        for wl in entry.values()
    )
    ttft_cut = {}
    off = summary["arms"].get("off", {})
    for arm, entry in summary["arms"].items():
        if arm == "off":
            continue
        for name in MUST_HIT:
            base = (off.get(name) or {}).get("median_ttft_s_after_first")
            value = (entry.get(name) or {}).get("median_ttft_s_after_first")
            if base and value:
                ttft_cut[f"{arm}/{name}"] = 1.0 - value / base
    linear_regression = None
    if "linear" in off and "auto" in summary["arms"]:
        base = off["linear"]["median_ttft_s_after_first"]
        value = summary["arms"]["auto"]["linear"]["median_ttft_s_after_first"]
        if base and value:
            linear_regression = value / base - 1.0
    # A model that does not reproduce its own greedy output cold-to-cold makes
    # the exactness gate unmeasurable: report it rather than crediting or
    # blaming the checkpoint.
    go = (
        not refused
        and diffs == 0
        and all(ttft_cut.get(f"auto/{name}", 0.0) >= 0.30 for name in MUST_HIT if name in off)
        and (linear_regression is None or linear_regression <= 0.03)
    )
    summary["gates"] = {
        "refused_arms": refused,
        "correctness_diffs": diffs,
        "cold_replay_diffs": cold_diffs,
        "ttft_cut": ttft_cut,
        "linear_ttft_regression": linear_regression,
        "go": go,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--route", choices=("mtp", "ordinary"), default="mtp")
    parser.add_argument("--base-policy", help="route execution policy JSON to merge")
    parser.add_argument("--arms", default="off,pow2,auto")
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--scale", type=float, default=0.2,
                        help="token-size multiplier for workload prompts (0.2 gives the "
                        "plan's sizes on Qwen3.8 tokenizers: ~12K system prompt, ~33K "
                        "document, linear up to ~52K; 1.0 is ~5x that)")
    parser.add_argument("--auto-headroom-fraction", type=float, default=None,
                        help="override the auto arm's headroom_fraction (diagnostic)")
    parser.add_argument("--auto-min-uncached-fraction", type=float, default=None,
                        help="override the auto arm's min_uncached_fraction "
                        "(skip capture on deep-hit continuation turns)")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-context", type=int, default=65536)
    parser.add_argument("--cache-gib", type=int, default=24)
    parser.add_argument("--port", type=int, default=8297)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args()

    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        parser.error(f"unknown arms: {sorted(unknown)}")
    names = [name.strip() for name in args.workloads.split(",") if name.strip()]
    if set(names) - set(WORKLOADS):
        parser.error(f"unknown workloads: {sorted(set(names) - set(WORKLOADS))}")
    all_workloads = build_workloads(args.scale)
    workloads = {name: all_workloads[name] for name in names}
    # Every prompt must fit the server context, or the arm measures 400s instead
    # of prefill.  Character count bounds token count from above for these
    # ASCII word lists (measured ~1.2 chars/token on Qwen3.8), so this refuses
    # only a certainly-oversized plan without needing a tokenizer here.
    max_chars = {
        name: max(len("".join(m["content"] for m in messages)) for messages in requests)
        for name, requests in workloads.items()
    }
    oversized = {
        name: chars for name, chars in max_chars.items()
        if chars / 1.2 + args.max_tokens > args.max_context
    }
    if oversized:
        parser.error(
            f"workload prompts exceed --max-context {args.max_context} at --scale "
            f"{args.scale} (max chars {oversized}); lower --scale"
        )
    schedule = [
        (round_index, arm)
        for round_index in range(args.rounds)
        for arm in (arms if round_index % 2 == 0 else list(reversed(arms)))
    ]
    plan = {
        "model": args.model,
        "route": args.route,
        "schedule": schedule,
        "policies": {arm: arm_policy(args, arm) for arm in arms},
        "scale": args.scale,
        "workloads": {name: len(requests) for name, requests in workloads.items()},
        "max_prompt_chars": max_chars,
        "server_command": server_command(args, Path("<policy.json>")),
        "server_pythonpath": server_env()["PYTHONPATH"].split(os.pathsep)[0],
        "gates": {
            "must_hit": list(MUST_HIT),
            "ttft_cut_min": 0.30,
            "linear_regression_max": 0.03,
            "correctness_diffs": 0,
        },
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        print("refusing: this loads a real model on the GPU; pass --i-own-the-gpu "
              "under the lab lock wrapper (or --dry-run)", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out.parent / (out.stem + "-logs")
    log_dir.mkdir(exist_ok=True)
    reference: dict = {}
    runs = []
    for round_index, arm in schedule:
        runs.append(run_arm(args, arm, round_index, workloads, reference, log_dir))
        out.write_text(json.dumps({"plan": plan, "runs": runs}, indent=1))
    summary = summarize(runs, names)
    out.write_text(json.dumps({"plan": plan, "runs": runs, "summary": summary}, indent=1))
    print(json.dumps(summary["gates"], indent=2))
    if summary["gates"]["refused_arms"]:
        return 3
    return 0 if summary["gates"]["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
