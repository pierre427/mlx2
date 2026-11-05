#!/usr/bin/env python3
"""One owned, real-model resident-kernel reload diagnostic; no performance claim.

Run under an existing CPG/radio lease and paired GPU locks. This process owns
one engine and one loopback server, retaining the checkpoint once throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def interrupt_for_cleanup(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


@contextmanager
def cleanup_signals():
    previous = {signum: signal.signal(signum, interrupt_for_cleanup)
                for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_text(version):
    """A real lazy Metal factory: edit its version, retain bit-identical logits."""
    return f'''from functools import lru_cache
import mlx.core as mx

VERSION = {version}
MODE = "a"
CALLS = {{}}
BUILDS = []

@lru_cache(maxsize=4)
def _kernel(mode):
    BUILDS.append([VERSION, mode])
    return mx.fast.metal_kernel(
        name=f"mlx2_reload_identity_v{{VERSION}}_{{mode}}",
        input_names=["x"], output_names=["y"],
        source="const uint i = thread_position_in_grid.x; y[i] = x[i];",
        ensure_row_contiguous=True,
    )

def identity(x):
    key = str(VERSION) + ":" + MODE
    CALLS[key] = CALLS.get(key, 0) + 1
    return _kernel(MODE)(
        inputs=[x], grid=(x.size, 1, 1),
        threadgroup=(min(256, x.size), 1, 1),
        output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]

def set_mode(value):
    global MODE
    if value not in ("a", "b"):
        raise ValueError("unknown probe mode")
    MODE = value
    return MODE
'''


def install_probe(adapter, alias):
    """Attach identity math after an existing output projection, without copies."""
    model = adapter.model
    if callable(getattr(model, "logits", None)):
        target, method, label = model, "logits", "model.logits"
    else:
        candidates = [(name, module) for name, module in model.named_modules()
                      if name.split(".")[-1] == "lm_head" and callable(module)]
        if len(candidates) != 1:
            raise ValueError("probe needs an unambiguous resident logits/lm_head seam")
        label, target = candidates[0]
        method = "__call__"
    original = type(target)
    original_call = getattr(original, method)

    def traced(self, *args, **kwargs):
        # This captured from-import-style alias must survive the source reload.
        return alias(original_call(self, *args, **kwargs))

    probe_type = type("ResidentKernelProbe_" + original.__name__, (original,), {method: traced})
    target.__class__ = probe_type
    return target, original, {"seam": label, "method": method, "object_id": id(target)}


def ownership(args):
    if args.ownership_spec is not None:
        from mlx2.desktop_campaign_worker import check_ownership

        return check_ownership(json.loads(args.ownership_spec.read_text()))
    from qualification_gpu_ownership import require_qualification_lease

    if args.cpg_task is None or args.cpg_generation is None or args.cpg_owner_lock is None:
        raise ValueError("an externally owned worker needs --cpg-task, --cpg-generation and --cpg-owner-lock")
    return require_qualification_lease(
        task_id=args.cpg_task, generation=args.cpg_generation,
        cpg_owner_lock=args.cpg_owner_lock,
    )


def request(base, path, *, body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    req = Request(base + path, headers=headers,
                  data=None if body is None else json.dumps(body).encode())
    with urlopen(req, timeout=180) as response:
        return json.load(response)


def metrics_checks(payload):
    samples = {}
    for line in payload.splitlines():
        if line and not line.startswith("#"):
            name, value = line.rsplit(" ", 1)
            samples[name] = float(value)
    expected = {
        "mlx2_kernel_reload_requests_total": 3,
        'mlx2_kernel_reload_outcomes_total{outcome="completed"}': 3,
        'mlx2_kernel_reload_outcomes_total{outcome="failed"}': 0,
        'mlx2_kernel_reload_outcomes_total{outcome="rejected"}': 0,
        "mlx2_kernel_reload_epoch": 3,
        "mlx2_kernel_reload_in_progress": 0,
        "mlx2_kernel_reload_enabled": 1,
        "mlx2_kernel_reload_weights_preserved_total": 3,
        "mlx2_kernel_reload_weight_checks_total": 3,
        "mlx2_kernel_reload_weight_check_failures_total": 0,
    }
    return {
        "metrics_reload_counts_exact": all(samples.get(name) == value for name, value in expected.items()),
        "metrics_apcv2_purge_observed": samples.get('mlx2_kernel_reload_apcv2_purged_total{kind="entries"}', 0) > 0,
        "metrics_apcv2_bytes_purged_observed": samples.get("mlx2_kernel_reload_apcv2_purged_bytes_total", 0) > 0,
        "metrics_stage_timings_observed": all(
            samples.get(f'mlx2_kernel_reload_duration_seconds_count{{stage="{stage}"}}') == 3
            for stage in ("prepare", "drain", "purge", "apply", "total")
        ),
    }


def artifact_name(model, override=None):
    """Name cached HF artifacts independently of their snapshot revision."""
    if override is not None:
        if not override.strip() or any(character in override for character in "@\r\n"):
            raise ValueError("artifact name must be nonempty and cannot contain @ or line breaks")
        return override.strip()
    path = Path(model).resolve()
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        pieces = path.parent.parent.name.split("--", 2)
        if len(pieces) == 3 and pieces[2]:
            return pieces[2]
    return Path(model).name


def host_snapshot(common):
    """Recognize only the verified, unloaded on-demand Music3 service pair."""
    snapshot = common.host_snapshot()
    labels = {"com.example.music3-api", "com.example.music3-mcp"}
    candidates = [row for row in snapshot["foreign_live_services"] if row.get("label") in labels]
    if not candidates:
        return snapshot
    endpoint = "http://127.0.0.1:8600/health"
    evidence = {"endpoint": endpoint, "observed_at": time.time(), "health": None, "processes": [], "errors": []}
    snapshot["music3_idle_check"] = evidence
    expected_executable = str(Path.home() / "Library/Application Support/mlxuag-music3/runtime/.venv/bin/python")
    try:
        with urlopen(endpoint, timeout=2) as response:
            if response.status != 200 or response.geturl() != endpoint:
                raise ValueError("Music3 health did not return its expected local endpoint")
            health = json.load(response)
        evidence["health"] = health
    except (OSError, ValueError) as error:
        evidence["errors"].append("health: " + str(error))
        return snapshot
    healthy_idle = (
        isinstance(health, dict) and health.get("status") == "ok"
        and health.get("model") == "MiniMaxAI/MiniMax-Music3"
        and health.get("loaded") is False and health.get("busy") is False
        and health.get("lifecycle") == "on_demand"
    )
    evidence["health_idle"] = healthy_idle
    if not healthy_idle:
        return snapshot
    idle = []
    for row in candidates:
        try:
            pid = row.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
                raise ValueError("Music3 service has no live PID")
            fields = subprocess.check_output(
                ["/bin/ps", "-p", str(pid), "-o", "pid=,rss=,comm="], text=True, timeout=2,
            ).strip().split(None, 2)
            if len(fields) != 3 or int(fields[0]) != pid:
                raise ValueError("Music3 service PID changed or disappeared")
            process = {**row, "rss_kib": int(fields[1]), "executable": fields[2]}
            evidence["processes"].append(process)
            if 0 <= process["rss_kib"] < 256 * 1024 and process["executable"] == expected_executable:
                idle.append(process)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            evidence["errors"].append(f"PID {row.get('pid')}: {error}")
    idle_keys = {(row["label"], row["pid"]) for row in idle}
    snapshot["foreign_live_services"] = [
        row for row in snapshot["foreign_live_services"] if (row.get("label"), row.get("pid")) not in idle_keys
    ]
    snapshot["observed_idle_music3_services"] = idle
    return snapshot


def run(args):
    if not args.i_own_the_gpu:
        raise ValueError("pass --i-own-the-gpu under the CPG/radio and paired GPU lease")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty kernel-reload evidence")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("MLX_ENABLE_TF32=0 is required")
    lease = ownership(args)
    name = artifact_name(args.model, args.artifact_name)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = args.out / "sources"
    sources.mkdir()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    import nemotron35_probe_common as common

    before = host_snapshot(common)
    report = {
        "schema": "mlx2.resident-kernel-reload-probe.v1", "status": "running",
        "source_revision": revision, "model": str(args.model.resolve()), "artifact_name": name,
        "source_files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (Path(__file__), ROOT / "src/mlx2/kernel_reload.py",
                                      ROOT / "src/mlx2/kernel_reload_serving.py",
                                      ROOT / "src/mlx2/serving.py", ROOT / "src/mlx2/server.py",
                                      ROOT / "src/mlx2/prometheus.py")},
        "ownership": lease, "host_before": before,
        "profile": {"route": "ordinary", "max_lanes": 1, "max_inflight": 1,
                    "cache_bytes": args.cache_bytes, "max_context": args.max_context,
                    "persistence": "private disk tier", "thermal_holds": False},
        "qualified": False, "performance_reportable": False, "cells": [],
        "reloads": [], "checks": {}, "model_loads": 0,
    }
    atomic(args.out / "receipt.json", report)
    engine = server = server_thread = None
    target = original_type = None
    signals = {}
    try:
        # Correctness testing records thermals/swap without thermal holds.
        if before["foreign_model_processes"] or before["foreign_live_services"]:
            raise RuntimeError("foreign model process/service present before model load")
        for signum in (signal.SIGINT, signal.SIGTERM):
            signals[signum] = signal.signal(signum, interrupt_for_cleanup)
        # Import accelerator modules only after checking live ownership.
        from http.server import ThreadingHTTPServer

        import mlx.core as mx

        from mlx2.adapters.registry import inspect_model
        from mlx2.generation_trace import read_token_trace
        from mlx2.kernel_reload import resident_weight_identity
        from mlx2.server import handler_for
        from mlx2.serving import ServingEngine

        if not mx.metal.is_available():
            raise RuntimeError("the resident-kernel probe requires Metal")
        mx.set_default_device(mx.gpu)
        report["mlx_version"] = mx.__version__
        report["device"] = mx.device_info()
        resolution = inspect_model(args.model)
        report["checkpoint"] = name + "@" + resolution.artifact_fingerprint
        module_name = "mlx2.runtime.kernel_reload_probe_" + str(os.getpid())
        module_path = sources / "probe.py"
        module_path.write_text(source_text(1))
        (sources / "version1.txt").write_text(source_text(1))
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        alias = module.identity
        load_started = time.monotonic()
        engine = ServingEngine(
            args.model, adapter_factory=resolution.adapter_type,
            kernel_dev_root=sources, mtp=False, prompt_lookup=False,
            max_lanes=1, max_inflight=1, max_context=args.max_context,
            cache_bytes=args.cache_bytes, cache_dir=args.out / "apcv2",
            qualification_mode=False, residency_keep_seconds=0,
        )
        deadline = time.monotonic() + args.load_timeout_seconds
        while not engine.ready.wait(.1):
            if engine.error or not engine.thread.is_alive():
                raise RuntimeError("model loading failed: " + str(engine.error))
            if time.monotonic() >= deadline:
                raise TimeoutError("model loading timed out")
        if engine.error:
            raise RuntimeError(engine.error)
        report["model_loads"] = 1
        report["load_seconds"] = time.monotonic() - load_started
        report["artifact"] = dict(engine.adapter.identity)
        report["checkpoint"] = name + "@" + engine.adapter.identity["fingerprint"]
        initial_weights = resident_weight_identity(engine.adapter)
        token = os.urandom(32).hex()
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine, admin_token=token))
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        report["server"] = {"pid": os.getpid(), "port": server.server_port}
        body = {
            "model": args.model.name,
            "messages": [{"role": "user", "content": args.prompt}],
            "temperature": 0, "max_tokens": args.max_tokens,
            "enable_thinking": False, "return_token_trace": True,
        }

        def cell(name):
            ownership(args)
            start = time.monotonic()
            response = request(base, "/v1/chat/completions", body=body)
            trace = read_token_trace(response["mlx2"])
            text = response["choices"][0]["message"].get("content", "")
            if not trace["token_ids"] or not text.strip():
                raise AssertionError("ordinary functional oracle requires nonempty completion")
            result = {
                "name": name, "seconds": time.monotonic() - start,
                "checkpoint": report["checkpoint"], "source_revision": revision,
                "token_ids": trace["token_ids"], "text": text,
                "cached_tokens": response["usage"]["prompt_tokens_details"]["cached_tokens"],
                "kernel_calls": dict(module.CALLS), "kernel_builds": list(module.BUILDS),
                "response": response,
            }
            report["cells"].append(result)
            atomic(args.out / (name + ".json"), result)
            atomic(args.out / "receipt.json", report)
            return result

        reference = cell("ordinary-reference-cold")
        reference_warm = cell("ordinary-reference-warm")

        def attach(adapter):
            nonlocal target, original_type
            target, original_type, hook = install_probe(adapter, alias)
            return hook

        report["probe_hook"] = engine._exclusive_adapter_operation("install_kernel_probe", attach)
        # The attach is diagnostic identity math. Purge explicitly so arm A
        # still proves its own cold/warm behavior.
        request(base, "/v1/admin/apc/clear", body={"reason": "kernel-probe-installed"}, token=token)
        first = cell("version1-cold")
        first_warm = cell("version1-warm")
        module_path.write_text(source_text(2))
        (sources / "version2.txt").write_text(source_text(2))
        report["reloads"].append(request(base, "/v1/admin/kernels/reload", token=token,
                                         body={"modules": [module_name], "drain_timeout_seconds": 30}))
        second = cell("version2-cold")
        second_warm = cell("version2-warm")
        report["reloads"].append(request(base, "/v1/admin/kernels/reload", token=token,
                                         body={"calls": [{"target": module_name + ":set_mode", "args": ["b"]}]}))
        variant = cell("variant-b-cold")
        variant_warm = cell("variant-b-warm")
        report["reloads"].append(request(base, "/v1/admin/kernels/reload", token=token,
                                         body={"settings": [{"target": module_name + ":MODE", "value": "a"}]}))
        restored = cell("variant-a-restored-cold")
        restored_warm = cell("variant-a-restored-warm")
        final_weights = resident_weight_identity(engine.adapter)
        report["checks"] = {
            "exact_tokens_all_arms": all(row["token_ids"] == reference["token_ids"] for row in report["cells"]),
            "resident_weights_unchanged": final_weights == initial_weights,
            "captured_alias_identity_preserved": module.identity is alias,
            "version1_kernel_used": first["kernel_calls"].get("1:a", 0) > 0,
            "version2_kernel_used": second["kernel_calls"].get("2:a", 0) > 0,
            "new_metal_factory_built": [2, "a"] in second["kernel_builds"],
            "live_setter_engaged": variant["kernel_calls"].get("2:b", 0) > 0,
            "setter_kernel_built": [2, "b"] in variant["kernel_builds"],
            "setting_restored": module.MODE == "a" and restored["kernel_calls"].get("2:a", 0) > second["kernel_calls"].get("2:a", 0),
            "all_reload_receipts_complete": all(row["status"] == "completed" for row in report["reloads"]),
            "cold_after_every_change": all(row["cached_tokens"] == 0 for row in (reference, first, second, variant, restored)),
            "warm_after_every_arm": all(row["cached_tokens"] > 0 for row in (reference_warm, first_warm, second_warm, variant_warm, restored_warm)),
            "one_model_load": report["model_loads"] == 1,
        }
        report["weights_identity_sha256"] = hashlib.sha256(repr(initial_weights).encode()).hexdigest()
        report["kernel_status"] = request(base, "/v1/admin/kernels", token=token)
        report["status_after"] = request(base, "/v1/status")
        with urlopen(base + "/metrics", timeout=30) as response:
            metrics = response.read().decode()
        metrics_path = args.out / "metrics.prom"
        metrics_path.write_text(metrics)
        report["metrics"] = {"path": str(metrics_path), "sha256": hashlib.sha256(metrics.encode()).hexdigest()}
        report["checks"].update(metrics_checks(metrics))
        report["observed_used"] = True
        report["passed"] = all(report["checks"].values())
        report["status"] = "completed" if report["passed"] else "failed"
        if not report["passed"]:
            raise AssertionError("kernel reload checks failed: " + repr(report["checks"]))
    except BaseException as error:
        report.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        cleanup_errors = []
        try:
            if server is not None:
                server.shutdown()
                server.server_close()
            if server_thread is not None:
                server_thread.join(timeout=5)
        except BaseException as error:  # noqa: BLE001 -- preserve cleanup failures in the atomic receipt
            cleanup_errors.append("HTTP cleanup: " + repr(error))
        try:
            if target is not None and original_type is not None:
                target.__class__ = original_type
            if engine is not None:
                engine.close()
                report["engine_stopped"] = not engine.thread.is_alive()
                if not report["engine_stopped"]:
                    cleanup_errors.append("model worker did not stop")
        except BaseException as error:  # noqa: BLE001 -- attempt every independent cleanup
            cleanup_errors.append("model cleanup: " + repr(error))
        try:
            report["host_after"] = host_snapshot(common)
            report["contamination"] = common.contamination(before, report["host_after"])
        except BaseException as error:  # noqa: BLE001 -- receipt failure must not skip ownership/finalization
            cleanup_errors.append("host receipt: " + repr(error))
        try:
            report["ownership_after"] = ownership(args)
        except BaseException as error:  # noqa: BLE001 -- seal lost ownership rather than abandoning evidence
            cleanup_errors.append("ownership receipt: " + repr(error))
        report["cleanup_errors"] = cleanup_errors
        if cleanup_errors:
            report.update(status="failed", passed=False)
        report["finished_at"] = time.time()
        try:
            atomic(args.out / "receipt.json", report)
        finally:
            for signum, previous in signals.items():
                signal.signal(signum, previous)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact-name", help="Receipt checkpoint name; defaults to HF repository name or model basename")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ownership-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cpg-task")
    parser.add_argument("--cpg-owner-lock", type=Path)
    parser.add_argument("--cpg-generation", type=int)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8766/mcp")
    parser.add_argument("--window-seconds", type=float, default=600)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--cache-bytes", type=int, default=256 << 20)
    parser.add_argument("--load-timeout-seconds", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    return parser


def owned(args):
    """Acquire one bounded ownership window and retire its entire child group."""
    from nemotron35_probe_common import stop_group

    from mlx2.adapters.registry import inspect_model
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty kernel-reload evidence")
    resolution = inspect_model(args.model)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    name = artifact_name(args.model, args.artifact_name)
    checkpoint = name + "@" + resolution.artifact_fingerprint
    args.out.mkdir(parents=True, exist_ok=True)
    spec = {
        "job_id": "kernel-reload-" + args.out.name,
        "out": str(args.out.resolve()), "mcp_url": args.mcp_url,
        "checkpoint": checkpoint, "source_revision": revision, "artifact_name": name,
    }
    spec_path = args.out / "spec.json"
    atomic(spec_path, spec)
    child = None
    receipt = {**spec, "status": "running", "worker_output": str(args.out / "run")}
    with cleanup_signals(), DesktopCPGLease(
        args.out, spec["job_id"], mcp_url=args.mcp_url,
        checkpoint=checkpoint, source_revision=revision,
    ) as lease:
        command = [
            sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
            "--session", lease.owner["cpg_session"], "--label", spec["job_id"],
            "--desktop-cpg-lease", str(lease.path), "--mcp-url", args.mcp_url,
            "--receipt", str(args.out / "paired-locks.json"), "--",
            sys.executable, str(Path(__file__).resolve()), "--worker", "--i-own-the-gpu",
            "--ownership-spec", str(spec_path.resolve()),
            "--model", str(args.model.resolve()), "--out", str((args.out / "run").resolve()),
            "--artifact-name", name,
            "--max-context", str(args.max_context), "--cache-bytes", str(args.cache_bytes),
            "--load-timeout-seconds", str(args.load_timeout_seconds),
            "--max-tokens", str(args.max_tokens), "--prompt", args.prompt,
        ]
        receipt["command"] = command
        try:
            environment = dict(os.environ, MLX_ENABLE_TF32="0", PYTHONPATH=str(ROOT / "src"))
            with (args.out / "worker.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                receipt["process_group"] = child.pid
                atomic(args.out / "unit.json", receipt)
                deadline = time.monotonic() + args.window_seconds
                while child.poll() is None:
                    lease.check()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("bounded kernel-reload artifact window expired")
                    time.sleep(.5)
            receipt["returncode"] = child.returncode
            receipt["status"] = "completed" if child.returncode == 0 else "failed"
            lease.success = child.returncode == 0
        except BaseException as error:
            receipt.update(status="failed", error=repr(error))
            raise
        finally:
            try:
                stop_group(child)
                receipt["process_group_stopped"] = True
            except BaseException as error:
                receipt.update(status="failed", cleanup_error=repr(error))
                lease.success = False
                raise
            finally:
                receipt["finished_at"] = time.time()
                atomic(args.out / "unit.json", receipt)
    print(json.dumps({"status": receipt["status"], "checkpoint": checkpoint,
                      "evidence": str(args.out / "run/receipt.json")}))
    return receipt.get("returncode", 1)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.max_tokens <= 128 or not 128 <= args.max_context <= 16384:
        parser.error("probe requires max-tokens 1..128 and max-context 128..16384")
    if not 30 <= args.window_seconds <= 3600:
        parser.error("window-seconds must be between 30 and 3600")
    if not args.worker and not args.i_own_the_gpu:
        return owned(args)
    report = run(args)
    print(json.dumps({"passed": report["passed"], "checkpoint": report["checkpoint"],
                      "evidence": str(args.out / "receipt.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
