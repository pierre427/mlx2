#!/usr/bin/env python3
"""One pinned Tiel artifact/route per bounded CPG and paired-lock window.

Functional and directional A/B evidence only. No thermal admission holds,
default promotion, or publication-quality speed claims. Keep failed attempts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def environment():
    return {key: subprocess.check_output(command, text=True) for key, command in {
        "swap": ["sysctl", "vm.swapusage"], "thermals": ["pmset", "-g", "therm"],
        "processes": ["ps", "-axo", "pid,ppid,command"],
        "services": ["launchctl", "list"],
    }.items()}


def source():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/mlx2").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return {"source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": digest.hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def request_body(repetitions, lane=0):
    identifier = f"TIEL-{73 + lane}"
    instruction = (f"Return the identifier {identifier} exactly, with no explanation."
        if lane == 0 else f"Start with {identifier}, then explain in detail why cached state must match the model revision.")
    return {"messages": [{"role": "user", "content":
        f"Context cell {repetitions}, lane {lane}. The identifier is {identifier}.\n" +
        "Repository note: every cached state belongs to its exact model revision.\n" * repetitions +
        f"\n{instruction}"}],
        "enable_thinking": False, "temperature": 0, "max_tokens": 32,
        "return_token_trace": True}


def text(response):
    return response["choices"][0]["message"].get("content") or ""


def has_identifier(value, code):
    """``code`` as a whole token: TIEL-730 or XTIEL-73 do not answer TIEL-73."""
    # Only Unicode dash punctuation is normalized; identifier case stays exact.
    import re
    import unicodedata
    value = "".join("-" if unicodedata.category(c) == "Pd" else c for c in value)
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(code)}(?![A-Za-z0-9])", value) is not None


def functional(response, lane=0):
    return has_identifier(text(response), f"TIEL-{73 + lane}")


def ordinary_reference(adapter, body, *, prefill_step=2048):
    """Bound the rendered prompt, including this checkpoint's long template."""
    import mlx.core as mx

    prompt = list(adapter.prompt_tokens(body))
    if not 1 <= len(prompt) <= 8192:
        raise ValueError("rendered probe prompt must contain 1..8192 tokens")
    cache = adapter.model.make_cache()
    started = time.perf_counter()
    for start in range(0, len(prompt), prefill_step):
        logits = adapter.model(mx.array([prompt[start:start + prefill_step]]), cache=cache)
        mx.eval(logits)
    tokens, finite = [], True
    for _ in range(body["max_tokens"]):
        row = logits[0, -1]
        finite &= bool(mx.all(mx.isfinite(row)).item())
        token = int(mx.argmax(row).item())
        tokens.append(token)
        if token in adapter.tokenizer.eos_token_ids:
            break
        logits = adapter.model(mx.array([[token]]), cache=cache)
    return {"prompt_tokens": len(prompt), "tokens": tokens,
        "text": adapter.tokenizer.decode(tokens, skip_special_tokens=True),
        "finite": finite, "elapsed_seconds": time.perf_counter() - started}


def cpu_unit(spec):
    """Real checkpoint loading/ordinary decode without touching the GPU."""
    out = Path(spec["out"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    report = {**source(), "checkpoint": spec["checkpoint"], "device": "cpu",
        "scope": "strict load and short ordinary CPU decode only",
        "qualification": "unqualified", "gpu_execution": False, "passed": False,
        "before": environment()}
    adapter = None
    try:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        # Availability-only guards must never opt a CPU probe into a Metal
        # candidate. The production adapter and its CPU fallback stay intact.
        mx.metal.is_available = lambda: False
        from mlx2.adapters.registry import inspect_model

        resolution = inspect_model(spec["model"])
        if (resolution.artifact_fingerprint != spec["fingerprint"]
                or report["source_revision"] != spec["source_revision"]):
            raise RuntimeError("source or checkpoint identity changed")
        started = time.perf_counter()
        adapter = resolution.adapter_type(spec["model"])
        report["load_seconds"] = time.perf_counter() - started
        report["load_passed"] = True
        report["diagnostics"] = adapter.diagnostics()
        atomic(out / "progress.json", report)
        report["output"] = ordinary_reference(adapter, {
            "messages": [{"role": "user", "content": "Return exactly TIEL-73 with no explanation."}],
            "enable_thinking": False, "max_tokens": 16,
        })
        report["passed"] = report["output"]["finite"] and has_identifier(report["output"]["text"], "TIEL-73")
    except BaseException as error:
        report["passed"] = False
        report.update(error=str(error), traceback=traceback.format_exc())
    finally:
        if adapter is not None:
            adapter.close()
        report["after"] = environment()
        atomic(out / "result.json", report)
    print(json.dumps({"passed": report["passed"], "error": report.get("error"), "out": str(out)}), flush=True)
    return 0 if report["passed"] else 1


def unit(spec):
    from mlx2.desktop_campaign_worker import check_ownership, http_json
    from mlx2.generation_trace import read_token_trace
    from run_single_model_smoke_ladder import stop_server, wait_health

    out = Path(spec["out"])
    report = {**source(), "checkpoint": spec["checkpoint"], "route": spec["route"],
        "qualification": "unqualified", "scope": "bounded functional and directional A/B",
        "thermal_policy": "record only; no holds", "passed": False, "cells": []}
    server = None
    try:
        report["ownership"] = check_ownership(spec)
        report["before"] = environment()
        if report["source_revision"] != spec["source_revision"]:
            raise RuntimeError("source commit changed")
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(1 << 30)
        report["device"] = mx.device_info()
        from mlx2.adapters.registry import inspect_model
        resolution = inspect_model(spec["model"])
        if resolution.artifact_fingerprint != spec["fingerprint"]:
            raise RuntimeError("artifact identity changed")
        # Ordinary target reference, before a server or speculative installer.
        adapter = resolution.adapter_type(spec["model"])
        report["reference"] = []
        prefill_step = spec.get("prefill_step", 2048)
        for repetitions in spec.get("contexts", [24, 120]):
            body = request_body(repetitions)
            result = ordinary_reference(adapter, body, prefill_step=prefill_step)
            report["reference"].append(result)
            if not result["finite"] or not has_identifier(result["text"], "TIEL-73"):
                raise AssertionError("ordinary reference failed functional oracle")
        report["reference_diagnostics"] = adapter.diagnostics()
        adapter.close()
        del adapter
        gc.collect()
        mx.clear_cache()
        atomic(out / "progress.json", report)
        check_ownership(spec)
        policy = spec.get("policy", {})
        atomic(out / "policy.json", policy)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command = [sys.executable, "-m", "mlx2.server", "--model", spec["model"],
            "--host", "127.0.0.1", "--port", str(port), "--fabric", "off",
            "--max-lanes", "4", "--max-inflight", "8", "--max-context", "8192",
            "--prefill-step", str(prefill_step), "--cache-bytes", str(2 << 30),
            "--cache-dir", str(out / "apcv2"), "--execution-policy", str(out / "policy.json"),
            {"ordinary": "--ordinary", "native_mtp": "--native-mtp", "external_draft": "--external-draft"}[spec["route"]]]
        report["server_command"] = command
        with (out / "server.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        base = f"http://127.0.0.1:{port}"
        report["status_before"] = wait_health(base, server, 600)
        server_artifact = report["status_before"].get("artifact")
        if isinstance(server_artifact, dict):
            server_artifact = server_artifact.get("fingerprint")
        binding = report["status_before"].get("execution", {}).get("artifact_binding", {})
        report["route_artifact"] = server_artifact
        if spec["route"] == "external_draft":
            from mlx2.runtime.drafters.native_mtp import inspect_native_mtp_policy

            expected_head = inspect_native_mtp_policy(policy, spec["model"], spec["fingerprint"])
            if (binding.get("target_fingerprint") != spec["fingerprint"]
                    or binding.get("draft_fingerprint") != expected_head["fingerprint"]):
                raise RuntimeError("server target/native-head binding differs")
        elif server_artifact != spec["fingerprint"]:
            raise RuntimeError("server loaded a different artifact")
        for index, repetitions in enumerate(spec.get("contexts", [24, 120])):
            body = request_body(repetitions)
            cell = {"repetitions": repetitions, "arms": []}
            for temperature in ("cold", "warm"):
                started = time.perf_counter()
                response = http_json(base, "/v1/chat/completions", body)
                trace = read_token_trace(response["mlx2"])
                cell["arms"].append({"phase": temperature, "response": response,
                    "tokens": trace["token_ids"], "functional": functional(response),
                    "elapsed_seconds": time.perf_counter() - started})
                atomic(out / "progress.json", report)
            cold, warm = cell["arms"]
            cell["cold_warm_exact"] = cold["tokens"] == warm["tokens"]
            cell["reference_exact"] = cold["tokens"] == report["reference"][index]["tokens"]
            cell["warm_cached_tokens"] = warm["response"]["mlx2"]["cached_tokens"]
            # Preserve first differing token before any diagnostic logprob run.
            for name, left, right in (("cold_warm", cold["tokens"], warm["tokens"]),
                    ("ordinary_reference", report["reference"][index]["tokens"], cold["tokens"])):
                if left != right:
                    cell[name + "_first_divergence"] = next(
                        (i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
            cell["passed"] = (cell["cold_warm_exact"] and cell["reference_exact"]
                and cell["warm_cached_tokens"] > 0 and all(arm["functional"] for arm in cell["arms"]))
            report["cells"].append(cell)
            atomic(out / "progress.json", report)
        # Equal prompts arrive together, but differ in identifiers. One server
        # owns both physical lanes; receipts, rather than client count, prove width.
        barrier = Barrier(2)
        def concurrent(lane):
            barrier.wait(timeout=10)
            body = request_body(64, lane)
            if spec.get("declared_cohort"):
                body["batch_cohort"] = {"id": spec["job_id"] + "-batch", "size": 2}
            position = spec.get("logprob_positions", {}).get(str(lane))
            if position is not None:
                body.update(logprobs=True, top_logprobs=8,
                    logprobs_start=position, logprobs_end=position)
            response = http_json(base, "/v1/chat/completions", body)
            return {"request": body, "response": response, "functional": functional(response, lane),
                "trace": read_token_trace(response["mlx2"])}
        with ThreadPoolExecutor(max_workers=2) as pool:
            report["batch"] = list(pool.map(concurrent, (1, 2)))
        report["status_after"] = http_json(base, "/v1/status")
        report["batch_observed_widths"] = sorted({width for arm in report["batch"]
            for width in arm["trace"]["execution_widths"]})
        coding = http_json(base, "/v1/chat/completions", {
            "messages": [{"role": "user", "content":
                "Return only the Python expression a + b, with no explanation or code fences."}],
            "enable_thinking": False, "temperature": 0, "max_tokens": 32,
            "return_token_trace": True,
        })
        report["coding"] = {"response": coding,
            "passed": text(coding).strip().strip("`").replace(" ", "") == "a+b"}
        report["passed"] = all(cell["passed"] for cell in report["cells"]) and all(
            arm["functional"] for arm in report["batch"]) and any(
                width > 1 for width in report["batch_observed_widths"]) and report["coding"]["passed"]
        report["status_after"] = http_json(base, "/v1/status")
    except BaseException as error:
        report["passed"] = False
        report.update(error=str(error), traceback=traceback.format_exc())
    finally:
        stop_server(server)
        report["after"] = environment()
        report["source_after"] = source()
        report["server_stopped"] = server is None or server.poll() is not None
        atomic(out / "result.json", report)
    print(json.dumps({"passed": report["passed"], "error": report.get("error"), "out": str(out)}), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("owned", "unit", "cpu"))
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    spec = json.loads(args.spec.read_text())
    if args.mode == "cpu":
        return cpu_unit(spec)
    if args.mode == "unit":
        return unit(spec)
    from mlx2.desktop_campaign_lease import DesktopCPGLease
    out = Path(spec["out"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    with DesktopCPGLease(out, spec["job_id"], checkpoint=spec["checkpoint"], source_revision=spec["source_revision"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"), "--session", lease.owner["cpg_session"],
            "--label", spec["job_id"], "--desktop-cpg-lease", str(lease.path), "--receipt", str(out / "paired-locks.json"),
            "--", sys.executable, str(Path(__file__)), "unit", str(args.spec.resolve())]
        child = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + 1200
        try:
            while child.poll() is None:
                lease.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("bounded artifact window expired")
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
