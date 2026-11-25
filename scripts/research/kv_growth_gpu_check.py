"""Planned K/V growth: a GPU A/B of ``MLX2_KV_PLANNED_GROWTH`` on the ordinary route.

Prompt K/V caches grow to each lane's planned prompt span (default, ``1``)
instead of one 256-row step per prefill chunk (``0``).  The claims are that
outputs are bitwise identical and that memory does not rise above what
admission charges.  This check runs one model on the ordinary serving route
once per arm (``0`` then ``1``; ``--both-orders`` adds ``1`` then ``0``) and
compares:

* greedy token ids of a fixed request set: a long prompt of about
  ``--context`` tokens (``scripts/realistic_haystack``) with a short question,
  a second request sharing that prefix with another question (APCv2 warm
  reuse), and one short prompt;
* ``prompt_tokens``, ``cached_tokens``, TTFT and decode tok/s per request;
* peak memory per arm: the server's own ``/v1/status`` counters
  (``metal_peak_bytes`` is MLX's process peak since start, the run's final
  value; ``metal_active_bytes`` and ``process_physical_footprint_bytes`` are
  sampled), plus the resident set of the server's process group from
  ``ps`` (sampled every ``--sample-s`` seconds);
* ``scheduler.kv_planned_growth`` as each server reports it, so the receipt
  names the arm that actually ran.

Each server is a loopback ``mlx2.server --ordinary --max-lanes 1`` with APCv2
on (its default) and a private ``--cache-dir`` per run; it is stopped as a
process group in ``finally``.  ``kv_growth_check.json`` is written
atomically into ``--out-dir``, which must be empty.  This is a sanity A/B
with no thermal hold, not publishable statistics; run it under the GPU lock.

Usage: python scripts/research/kv_growth_gpu_check.py --model <dir> --out-dir <empty dir>
       [--context 16384] [--port 8399] [--both-orders]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

ENV = "MLX2_KV_PLANNED_GROWTH"
RESERVED_PORT = 8282
REPORT_NAME = "kv_growth_check.json"
SYSTEM = "You are a careful assistant. Answer from the reference material when it is relevant."
QUESTION_A = ("Using the reference material above, name two mechanisms it describes and say in one "
              "sentence what each is for.")
QUESTION_B = "Which single sentence of the reference material above is the most surprising, and why?"
SHORT = "In two sentences, explain what a key-value cache does in transformer inference."
MEMORY_KEYS = ("metal_peak_bytes", "metal_active_bytes", "process_physical_footprint_bytes")


# -- guards ------------------------------------------------------------------------


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def check_port(port: int, *, free=port_free) -> None:
    if port == RESERVED_PORT:
        raise SystemExit(f"port {RESERVED_PORT} is reserved for production release trees")
    if not 1024 <= port <= 65535:
        raise SystemExit(f"port {port} is outside 1024-65535")
    if not free(port):
        raise SystemExit(f"port {port} is already in use; refusing to measure another server")


def prepare_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise SystemExit(f"{path} is not empty; refusing to overwrite evidence")
    return path


def write_atomic(path: Path, report: dict) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(report, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def arms_for(both_orders: bool) -> list[str]:
    return ["0", "1", "1", "0"] if both_orders else ["0", "1"]


# -- the request set ----------------------------------------------------------------


def calibrated_document(model: Path, context: int):
    """Haystack text of about ``context`` tokens (tokenizer-calibrated when possible)."""
    from realistic_haystack import haystack

    chars = int(context * 4.0)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                                  trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001 - a missing tokenizer only loses calibration
        return haystack(chars, root=ROOT), {"calibrated": False, "error": repr(exc)[:300]}
    tokens = None
    for _ in range(5):
        text = haystack(chars, root=ROOT)
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if abs(tokens - context) <= max(32, context // 100):
            break
        chars = max(256, int(chars * context / tokens))
    return text, {"calibrated": True, "document_tokens": tokens, "chars": len(text)}


def request_set(document: str) -> list[dict]:
    long_a = f"Reference material:\n\n{document}\n\n{QUESTION_A}"
    long_b = f"Reference material:\n\n{document}\n\n{QUESTION_B}"
    return [
        {"name": "long", "messages": [{"role": "system", "content": SYSTEM},
                                      {"role": "user", "content": long_a}]},
        {"name": "long_shared_prefix", "messages": [{"role": "system", "content": SYSTEM},
                                                    {"role": "user", "content": long_b}]},
        {"name": "short", "messages": [{"role": "user", "content": SHORT}]},
    ]


def request_body(messages, *, max_tokens: int, reasoning_effort: str | None) -> dict:
    # No "model" key: the server accepts only its own directory name there.
    # logprobs carries the token ids (parse_stream builds token_ids from it).
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 7,
            "stream": True, "logprobs": True}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return body


def ask(base: str, spec: dict, a, *, send=None) -> dict:
    """One greedy request; drops ``reasoning_effort`` if the route refuses it (400)."""
    if send is None:
        from inkling2_context_ladder import stream_one

        def send(body):
            return stream_one(base, body, timeout=a.request_timeout)
    effort = a.reasoning_effort if a.reasoning_effort not in (None, "", "none") else None
    body = request_body(spec["messages"], max_tokens=a.max_tokens, reasoning_effort=effort)
    try:
        r = send(body)
    except urllib.error.HTTPError as exc:
        if exc.code != 400 or effort is None:
            raise
        effort = None
        r = send(request_body(spec["messages"], max_tokens=a.max_tokens, reasoning_effort=None))
    receipt = r.get("receipt") or {}
    return {"name": spec["name"], "reasoning_effort_sent": effort,
            "prompt_tokens": receipt.get("prompt_tokens"),
            "cached_tokens": receipt.get("cached_tokens"),
            "ttft_s": r.get("ttft_s"), "decode_tok_s": r.get("decode_tok_s"),
            "finish_reason": r.get("finish_reason"), "route": receipt.get("route"),
            "token_ids": list(r.get("token_ids") or [])}


# -- the server ---------------------------------------------------------------------


def server_command(a, cache_dir: Path) -> list[str]:
    return [sys.executable, "-u", "-m", "mlx2.server", "--model", str(a.model),
            "--host", "127.0.0.1", "--port", str(a.port), "--ordinary",
            "--max-lanes", "1", "--max-inflight", "1",
            "--max-context", str(a.max_context or a.context + 4096),
            "--cache-dir", str(cache_dir)]


def server_env(arm: str) -> dict:
    return {**os.environ, ENV: arm, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1"}


def start(a, arm: str, cmd, log_path: Path):
    import hy4_context_ladder as H

    check_port(a.port)
    log = open(log_path, "w")  # noqa: SIM115 - closed by the caller
    server = subprocess.Popen(cmd, cwd=ROOT, env=server_env(arm), stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True)
    base = f"http://127.0.0.1:{a.port}"
    t0 = time.monotonic()
    while server.poll() is None and time.monotonic() - t0 < a.load_timeout:
        try:
            if H.get_json(base, "/health").get("status") == "ok":
                return server, log, round(time.monotonic() - t0, 1)
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            pass
        time.sleep(1)
    H.stop_group(server)
    log.close()
    raise SystemExit(f"server did not become healthy (rc {server.poll()}); see {log_path}")


def group_rss_bytes(pgid: int) -> int | None:
    """Resident set of every process in group ``pgid`` (``ps``; no vmmap)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,pgid=,rss="], capture_output=True,
                             text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total, seen = 0, False
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] == str(pgid):
            total, seen = total + int(parts[2]) * 1024, True
    return total if seen else None


class MemorySampler:
    """Max of ``/v1/status`` memory counters and the group's RSS, every ``period`` s."""

    def __init__(self, base: str, pgid: int, period: float, *, status=None, rss=group_rss_bytes):
        import hy4_context_ladder as H

        self.status = status or (lambda: H.get_json(base, "/v1/status", timeout=5))
        self.rss, self.pgid, self.period = rss, pgid, period
        self.max = {key: None for key in (*MEMORY_KEYS, "group_rss_bytes")}
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def sample(self):
        values = {}
        try:
            status = self.status()
            values.update({key: status.get(key) for key in MEMORY_KEYS})
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            pass
        values["group_rss_bytes"] = self.rss(self.pgid)
        for key, value in values.items():
            if isinstance(value, (int, float)):
                self.max[key] = value if self.max[key] is None else max(self.max[key], value)
        self.samples += 1

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.period)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(10)
        self.sample()


def checkpoint_name(status: dict) -> str | None:
    model, artifact = status.get("model"), status.get("artifact")
    return f"{model}@{artifact}" if model and artifact else None


def run_arm(a, arm: str, index: int, specs, out: Path, runs: list, save) -> dict:
    import hy4_context_ladder as H

    run_dir = out / f"run{index}-arm{arm}"
    run_dir.mkdir()
    cache_dir = run_dir / "apc-cache"
    cache_dir.mkdir(mode=0o700)
    cmd = server_command(a, cache_dir)
    base = f"http://127.0.0.1:{a.port}"
    run = {"arm": arm, "index": index, "env": {ENV: arm}, "server_command": cmd, "requests": []}
    runs.append(run)  # partial evidence survives a failed request
    server, log, run["load_s"] = start(a, arm, cmd, run_dir / "server.log")
    try:
        ready = H.get_json(base, "/v1/status", timeout=30)
        run["checkpoint"] = checkpoint_name(ready)
        run["runtime"] = ready.get("runtime")
        run["ready_memory"] = {key: ready.get(key) for key in MEMORY_KEYS}
        run["kv_planned_growth"] = (ready.get("scheduler") or {}).get("kv_planned_growth")
        with MemorySampler(base, server.pid, a.sample_s) as sampler:
            for spec in specs:
                result = ask(base, spec, a)
                run["requests"].append(result)
                save()
                print(f"arm {arm} {spec['name']}: prompt {result['prompt_tokens']} cached "
                      f"{result['cached_tokens']} ttft {result['ttft_s']} "
                      f"decode {result['decode_tok_s']}", flush=True)
            time.sleep(max(1.0, 2 * a.sample_s))  # let the worker publish its last snapshot
        run["memory_max"] = dict(sampler.max)
        run["memory_samples"] = sampler.samples
        final = H.get_json(base, "/v1/status", timeout=30)
        run["final_memory"] = {key: final.get(key) for key in MEMORY_KEYS}
        scheduler = final.get("scheduler") or {}
        run["scheduler"] = scheduler
        run["kv_planned_growth_final"] = scheduler.get("kv_planned_growth")
        run["apcv2"] = final.get("apcv2")
    finally:
        run["stop"] = H.stop_group(server)
        log.close()
        save()
    return run


# -- the verdict --------------------------------------------------------------------


def first_divergence(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _peak(run: dict, key: str):
    values = [(run.get("memory_max") or {}).get(key), (run.get("final_memory") or {}).get(key)]
    values = [v for v in values if isinstance(v, (int, float))]
    return max(values) if values else None


def _delta(new, old):
    return None if new is None or old is None else round(new - old, 4)


def _mean(values):
    return round(sum(values) / len(values), 4) if values else None


def _pct(new, old):
    return None if new is None or not old else round(100.0 * (new - old) / old, 2)


def build_verdict(runs: list[dict]) -> dict:
    if not runs:
        raise ValueError("no runs to compare")
    names = [r["name"] for r in runs[0]["requests"]]
    reference = {r["name"]: r["token_ids"] for r in runs[0]["requests"]}
    differences, empty = [], []
    for run in runs:
        got = {r["name"]: r["token_ids"] for r in run["requests"]}
        if list(got) != names:
            differences.append({"run": run["index"], "arm": run["arm"], "request": None,
                                "reason": "request set differs"})
            continue
        for name in names:
            if not got[name]:
                empty.append({"run": run["index"], "request": name})
            at = first_divergence(reference[name], got[name])
            if at is not None:
                differences.append({"run": run["index"], "arm": run["arm"], "request": name,
                                    "first_divergence": at,
                                    "reference_token": (reference[name][at:at + 1] or [None])[0],
                                    "token": (got[name][at:at + 1] or [None])[0]})
    arms = sorted({run["arm"] for run in runs})
    observed = {arm: [run.get("kv_planned_growth") for run in runs if run["arm"] == arm]
                for arm in arms}
    receipts_match = all(
        value is (arm == "1") for arm, values in observed.items() for value in values
    ) and all(run.get("kv_planned_growth_final", run.get("kv_planned_growth"))
              is run.get("kv_planned_growth") for run in runs)
    peak = {}
    for arm in arms:
        mine = [run for run in runs if run["arm"] == arm]
        peak[arm] = {}
        for key in (*MEMORY_KEYS, "group_rss_bytes"):
            values = [_peak(run, key) for run in mine]
            values = [v for v in values if v is not None]
            peak[arm][key] = max(values) if values else None
    timing = {}
    for name in names:
        per_arm = {}
        for arm in arms:
            rows = [next(r for r in run["requests"] if r["name"] == name)
                    for run in runs if run["arm"] == arm]
            ttft = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
            decode = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s") is not None]
            # Headline = mean over every run of the arm. Arms run in ABBA order
            # and a cell drifts slower as it goes, so best-of favours arm 0
            # (its first run is the earliest of all); the mean cancels a linear
            # drift. Best-of and the per-run values stay for inspection.
            per_arm[arm] = {"ttft_s": _mean(ttft), "decode_tok_s": _mean(decode),
                            "ttft_s_best": min(ttft) if ttft else None,
                            "decode_tok_s_best": max(decode) if decode else None,
                            "ttft_s_runs": ttft, "decode_tok_s_runs": decode,
                            "cached_tokens": [r.get("cached_tokens") for r in rows],
                            "prompt_tokens": [r.get("prompt_tokens") for r in rows]}
        timing[name] = per_arm
    verdict = {
        "tokens_identical": not differences and not empty,
        "token_differences": differences,
        "empty_token_ids": empty,
        "kv_planned_growth_observed": observed,
        "arm_receipts_match": receipts_match,
        "checkpoints": sorted({str(run.get("checkpoint")) for run in runs}),
        "same_checkpoint": len({run.get("checkpoint") for run in runs}) == 1
        and runs[0].get("checkpoint") is not None,
        "peak_memory": peak,
        "timing": timing,
        "server_stops": [run.get("stop") for run in runs],
        "apc_warm_reuse": {
            str(run["index"]): next((r.get("cached_tokens") or 0) > 0 for r in run["requests"]
                                    if r["name"] == "long_shared_prefix")
            if "long_shared_prefix" in names else None
            for run in runs
        },
    }
    if {"0", "1"} <= set(arms):
        verdict["peak_memory_delta_bytes"] = {
            key: _delta(peak["1"][key], peak["0"][key]) for key in peak["0"]
        }
        verdict["timing_delta"] = {
            name: {"ttft_s": _delta(t["1"]["ttft_s"], t["0"]["ttft_s"]),
                   "decode_tok_s": _delta(t["1"]["decode_tok_s"], t["0"]["decode_tok_s"]),
                   "ttft_pct": _pct(t["1"]["ttft_s"], t["0"]["ttft_s"]),
                   "decode_pct": _pct(t["1"]["decode_tok_s"], t["0"]["decode_tok_s"]),
                   "basis": "mean over all runs of each arm (ABBA)"}
            for name, t in timing.items()
        }
    return verdict


# -- main ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-context", type=int, default=None,
                        help="server --max-context (default: --context + 4096)")
    parser.add_argument("--port", type=int, default=8399)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--reasoning-effort", default="low",
                        help="sent when the route accepts it; 'none' omits it")
    parser.add_argument("--both-orders", action="store_true",
                        help="run arms 0,1,1,0 instead of 0,1")
    parser.add_argument("--sample-s", type=float, default=0.5)
    parser.add_argument("--load-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=1800)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    check_port(a.port)
    if not a.model.is_dir():
        raise SystemExit(f"{a.model} is not a model directory")
    out = prepare_out_dir(a.out_dir)
    path = out / REPORT_NAME
    report = {"script": "scripts/research/kv_growth_gpu_check.py",
              "kind": "sanity A/B (no thermal hold; not publishable statistics)",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model_dir": str(a.model),
              "context": a.context, "max_tokens": a.max_tokens,
              "reasoning_effort": a.reasoning_effort, "arms": arms_for(a.both_orders),
              "runs": []}
    git = ["git", "-C", str(ROOT)]
    report["mlx2_commit"] = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True,
                                           text=True, check=False).stdout.strip()
    report["mlx2_dirty_src"] = bool(subprocess.run(
        [*git, "status", "--porcelain", "--", "src"], capture_output=True, text=True,
        check=False).stdout.strip())
    document, report["document"] = calibrated_document(a.model, a.context)
    specs = request_set(document)
    write_atomic(path, report)
    for index, arm in enumerate(report["arms"]):
        run = run_arm(a, arm, index, specs, out, report["runs"],
                      lambda: write_atomic(path, report))
        if run["stop"] != "stopped":
            raise SystemExit(f"server for arm {arm} did not stop; refusing to start another")
    report["checkpoint"] = report["runs"][0].get("checkpoint")
    report["verdict"] = build_verdict(report["runs"])
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_atomic(path, report)
    v = report["verdict"]
    print(json.dumps({"checkpoint": report["checkpoint"], "mlx2_commit": report["mlx2_commit"],
                      "tokens_identical": v["tokens_identical"],
                      "kv_planned_growth_observed": v["kv_planned_growth_observed"],
                      "peak_memory_delta_bytes": v.get("peak_memory_delta_bytes"),
                      "timing_delta": v.get("timing_delta")}, indent=1), flush=True)
    return 0 if v["tokens_identical"] and v["arm_receipts_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
