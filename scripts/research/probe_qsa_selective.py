#!/usr/bin/env python3
"""F6 correctness probe: selective QSA stage-one scoring vs the default selector.

Feature 6 (7af38665c) adds ``qsa_stage1_direct_selector: "selective"`` on
Flash-Next: a scalar-F32 indexer score producer with bounded candidates and
exact Top-K recovery. Its score arithmetic differs from the default radix
selector's GEMM scores, so token differences are possible at near ties.

Stage one engages only at prefill query width >= 64 and physical KV >= 65024
tokens, and selective refuses calls above 4096 rows, so both arms serve with
``--prefill-step 4096`` on the ordinary-reference route and receive the same
~72K-token needle prompts (distinct, no shared prefix). Per arm, one server:

* engagement: Flash-Next ``qsa_stage1`` status before/after (engagements,
  selector counts, ``selective`` native execution);
* functional oracle: the needle code in the answer;
* ordinary-reference comparison: greedy token ids against the default arm,
  with the first divergence's top-5 logprob windows for near-tie
  classification.

Prefill time is recorded but this is not a performance A/B (the first native
selective kernel is serial per row). Original mlx2 code.

Usage: ``probe_qsa_selective.py owned OUT --model PATH``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
PORT = 8394
SHAPE = ["--ordinary", "--max-lanes", "1", "--max-inflight", "1", "--max-context", "98304",
         "--prefill-step", "4096", "--cache-bytes", str(4 << 30)]

_A = ["the archivist", "a ferry pilot", "the night baker", "a glass blower", "the surveyor",
      "a beekeeper", "the clock repairer", "a cartographer", "the lighthouse keeper", "a potter",
      "the river guide", "a violin maker", "the orchard owner", "a tailor", "the stonemason",
      "a lamplighter", "the harbour master", "a weaver", "the bookbinder", "a falconer"]
_V = ["catalogued", "repaired", "measured", "painted", "traded", "sketched", "weighed", "sorted",
      "polished", "wrapped", "labelled", "inspected", "carried", "restored", "counted", "mended"]
_O = ["copper kettles", "maps of the delta", "oak barrels", "spare sails", "brass hinges",
      "seed packets", "old ledgers", "lantern wicks", "wool blankets", "clay tiles",
      "pocket compasses", "silver spoons", "tide tables", "rope coils", "garden shears",
      "glass floats", "tin whistles", "linen sacks", "iron nails", "cedar boxes"]
_P = ["near the north pier", "behind the market hall", "on the hill road", "by the mill race",
      "in the east quarter", "under the bridge arches", "at the old customs house",
      "beside the chapel", "along the canal", "past the salt flats", "at the ferry steps",
      "inside the grain store"]

CASES = [  # (seed, needle fraction, code)
    (11, 0.45, "LM-7392"),
    (23, 0.12, "RT-5186"),
]


def document(seed: int, fraction: float, code: str, lines: int = 2300) -> str:
    rng = random.Random(seed)
    body = []
    needle_at = int(lines * fraction)
    for index in range(lines):
        body.append(f"Entry {seed}-{index}: {rng.choice(_A)} {rng.choice(_V)} {rng.randint(2, 997)} "
                    f"{rng.choice(_O)} {rng.choice(_P)} in week {rng.randint(1, 52)}.")
        if index == needle_at:
            body.append(f"Entry {seed}-{index} note: the strongroom access code is {code}.")
    return "\n".join(body)


def oracle(text: str, code: str) -> bool:
    import re

    text = re.sub(r"[‐-―]", "-", text or "")
    return re.search(rf"(?<!\w){re.escape(code)}(?!\w)", text) is not None


def checkout_env() -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT / "src"), environment.get("PYTHONPATH", "")) if p)
    return environment


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def source() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"),
            "dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no"))}


def host_sample() -> dict:
    return {"thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True),
            "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()}


def _find(tree, key):
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(tree, list):
        for value in tree:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def call(path, payload=None, timeout=3600):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def ask(model_name: str, seed: int, fraction: float, code: str) -> dict:
    messages = [{"role": "system", "content": "Answer from the log only, briefly."},
                {"role": "user", "content": document(seed, fraction, code)
                 + "\n\nWhat is the strongroom access code? Reply with the code only."}]
    payload = {"model": model_name, "messages": messages, "temperature": 0, "max_tokens": 48,
               "enable_thinking": False, "return_token_trace": True, "logprobs": True,
               "top_logprobs": 5, "stream": False}
    start = time.monotonic()
    try:
        result = call("/v1/chat/completions", payload)
    except Exception as exc:  # noqa: BLE001 - a fail-closed refusal is evidence
        body = getattr(exc, "read", lambda: b"")()
        return {"seed": seed, "error": repr(exc), "error_body": body.decode(errors="replace")[:2000],
                "elapsed_seconds": time.monotonic() - start}
    choice = result["choices"][0]
    content = choice["message"].get("content") or ""
    return {"seed": seed, "code": code, "content": content, "oracle_passed": oracle(content, code),
            "elapsed_seconds": time.monotonic() - start, "ttft_seconds": _find(result, "ttft_seconds"),
            "usage": result.get("usage"),
            "token_ids": ((result.get("mlx2") or {}).get("token_trace") or {}).get("token_ids"),
            "logprobs": (choice.get("logprobs") or {}).get("content"),
            "route": (result.get("mlx2") or {}).get("route")}


def run_arm(out: Path, model: Path, name: str, extra: list[str]) -> dict:
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
               "--port", str(PORT), "--cache-dir", str(out / f"{name}-cache"), *SHAPE, *extra]
    row = {"name": name, "command": command, "host_before": host_sample()}
    log = (out / f"{name}.server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env=checkout_env(), cwd=ROOT)
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"{name}: server exited before readiness")
            try:
                status = call("/v1/status", timeout=10)
                if status.get("healthy"):
                    break
            except (URLError, TimeoutError, ConnectionError, OSError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"{name}: readiness timeout")
            time.sleep(0.5)
        row["qsa_stage1_before"] = _find(status, "qsa_stage1")
        if row["qsa_stage1_before"] is None:
            raise RuntimeError(f"{name}: /v1/status carries no qsa_stage1 receipt")
        row["requests"] = [ask(model.name, *case) for case in CASES]
        row["qsa_stage1_after"] = _find(call("/v1/status"), "qsa_stage1")
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=30)
        log.close()
        row["server_stopped"] = server.poll() is not None
        row["host_after"] = host_sample()
    return row


def _window(row: dict, index: int):
    rows = row.get("logprobs") or []
    return rows[index] if index < len(rows) else None


def divergence(a: dict, b: dict) -> dict | None:
    left, right = a.get("token_ids") or [], b.get("token_ids") or []
    for index, (x, y) in enumerate(zip(left, right)):
        if x != y:
            return {"position": index, "tokens": [x, y], "window_default": _window(a, index),
                    "window_selective": _window(b, index)}
    if len(left) != len(right):
        return {"position": min(len(left), len(right)), "length_only": True}
    return None


def unit(out: Path, model: Path) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.adapters.registry import inspect_model

    policy = out / "policy-selective.json"
    policy.write_text(json.dumps({"qsa_stage1_direct_selector": "selective"}) + "\n")
    receipt = {"schema": "mlx2.f6-qsa-selective-probe.v1", "source": source(),
               "checkpoint": f"{model.name}@sha256:{inspect_model(model).artifact_fingerprint}",
               "shape": SHAPE, "cases": CASES, "qualified": False, "arms": []}
    try:
        for name, extra in (("default", []), ("selective", ["--execution-policy", str(policy)])):
            try:
                receipt["arms"].append(run_arm(out, model, name, extra))
            finally:
                atomic_json(out / "receipt.json", receipt)
        default, selective = receipt["arms"]
        receipt["comparison"] = [divergence(a, b) for a, b in
                                 zip(default["requests"], selective["requests"])]
        sel = (selective.get("qsa_stage1_after") or {}).get("candidates") or {}
        dflt = default.get("qsa_stage1_after") or {}
        receipt["checks"] = {
            "default_stage1_engaged": int(dflt.get("engagements") or 0) > 0,
            "selective_selected": bool(sel.get("selective_selected")),
            "selective_dispatched": int((sel.get("runtime_counts") or {})
                                        .get("selective_scheduled_dispatches", 0)) > 0,
            "selective_native_observed": bool((sel.get("selective") or {})
                                              .get("native_execution_observed")),
            "no_request_errors": not any("error" in r for a in receipt["arms"] for r in a["requests"]),
            "oracle_all": all(r.get("oracle_passed") for a in receipt["arms"] for r in a["requests"]),
            "tokens_exact": all(d is None for d in receipt["comparison"]),
        }
        receipt["passed_engagement_and_oracle"] = all(
            v for k, v in receipt["checks"].items() if k != "tokens_exact")
    except BaseException as exc:
        receipt.update(error=repr(exc))
        raise
    finally:
        atomic_json(out / "receipt.json", receipt)
    print(json.dumps({"checks": receipt.get("checks"), "comparison": receipt.get("comparison")},
                     default=str)[:3000], flush=True)
    return 0


def owned(out: Path, model: Path, job_id: str) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out, job_id, checkpoint=model.name, source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out), "--model", str(model)]
        child = subprocess.Popen(command, cwd=ROOT, env=checkout_env(), start_new_session=True)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--job-id", default="f6-qsa-selective-20261105")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "unit":
        return unit(out, args.model)
    return owned(out, args.model, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
