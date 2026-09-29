#!/usr/bin/env python3
"""Shared plumbing for the series' extra rounds (kvquant, experimental).

Import-safe on CPU: nothing here imports MLX in this process.  Adapter facts
(declared approximate-KV operations, int8 scopes, env survival through the
adapter's ``configure_environment``) come from ``probe`` run in a child
process pinned to the MLX CPU device, which never loads a weight shard.

A job script owns one model server at a time on ``campaign_config.PORT``,
started from ``campaign_config.server_args`` plus extra flags, and writes its
result to ``series/jobs/<model>/<round>.json`` (the file ``series.py
extract()`` reads back and the wiki report renders).
"""

from __future__ import annotations

import http.client
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MLX2_CAMPAIGN_ROOT", str(HERE.parents[2]))
import campaign_config as cc  # noqa: E402

ROOT, RUN, PYTHON, PORT, BASE_URL = cc.ROOT, cc.RUN, cc.PYTHON, cc.PORT, cc.BASE_URL
JOBS = RUN / "series" / "jobs"
RUNTIME = RUN / ".runtime"
SHIM_DIR = RUN / "kernel_shim"
SERIES_LOCK = Path("/tmp/gpu.lock.series-20260924")
SERVICE = "com.localuser.fn-uncensored-mlx-serve"
PLIST = Path.home() / "Library/LaunchAgents" / f"{SERVICE}.plist"
BASE_ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

# The same small oracle the smoke round uses (feature_smoke.FIXED_PROMPTS).
PROMPT_SUM = "What is 37 plus 58? Reply with the number only."
PROMPT_CAPITAL = "Name the capital of Canada in one word."
PROMPT_OPEN = "In about fifty words, explain why a prefix cache speeds up a chat server."


def oracle(prompt: str, text: str) -> bool | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    if prompt == PROMPT_SUM:
        return normalized == "95"
    if prompt == PROMPT_CAPITAL:
        return normalized == "ottawa"
    return None


def model_by_name(name: str):
    for model in cc.MODELS:
        if model.name == name:
            return model
    known = ", ".join(model.name for model in cc.MODELS)
    raise SystemExit(f"unknown series model {name!r}; known: {known}")


def route_by_name(model, name: str):
    for route in model.routes:
        if route.name == name:
            return route
    raise KeyError(f"{model.name} has no route {name!r}")


def stage_name(round_name: str, model, route_name: str) -> str:
    # Job ids follow build_jobs(): "<round>-<model>-<route>", which is what
    # wiki_report.job_index() parses.
    return f"{round_name}-{model.name}-{route_name}"


# --- CPU probe (child process) -----------------------------------------------

_PROBE = r"""
import json, os, sys
from pathlib import Path
import mlx.core as mx
mx.set_default_device(mx.cpu)
sys.path.insert(0, sys.argv[1])
import campaign_config as cc
from mlx2.adapters.registry import inspect_model
from mlx2.adapters.base import approximate_kv_operations
from mlx2.runtime.int8_prefill import adapter_scopes
request = json.loads(sys.argv[2])
out = {"bitexact_api": hasattr(mx.metal, "set_qmv_bitexact"), "mlx": mx.__version__, "models": {}}
baseline_env = dict(os.environ)
for name in request["models"]:
    model = next(m for m in cc.MODELS if m.name == name)
    resolved = inspect_model(model.path)
    kind = resolved.adapter_type
    instance = object.__new__(kind)
    try:
        operations = sorted(approximate_kv_operations(instance))
    except Exception as error:
        operations = []
    config = {}
    try:
        config = json.loads((Path(model.path) / "config.json").read_text())
    except Exception:
        pass
    text = config.get("text_config", config) if isinstance(config, dict) else {}
    quantized = bool(config.get("quantization") or config.get("quantization_config")
                     or text.get("quantization") or text.get("quantization_config"))
    module = sys.modules[kind.__module__]
    configure = getattr(module, "configure_environment", None)
    survival = {}
    for label, env in (request.get("kernel_env", {}).get(name) or {}).items():
        os.environ.clear(); os.environ.update(baseline_env); os.environ.update(env)
        if configure is None:
            effective = {key: os.environ.get(key) for key in env}
        else:
            try:
                configure()
            except TypeError:
                configure(Path(model.path), None)
            effective = {key: os.environ.get(key) for key in env}
        survival[label] = {"requested": env, "effective": effective,
                           "survives": all(effective[k] == v for k, v in env.items()),
                           "adapter_configures_environment": configure is not None}
    os.environ.clear(); os.environ.update(baseline_env)
    out["models"][name] = {
        "adapter": kind.__name__, "module": kind.__module__,
        "descriptor_family": resolved.descriptor.family,
        "capabilities": sorted(getattr(c, "value", str(c)) for c in resolved.descriptor.capabilities),
        "approximate_kv_operations": operations,
        "int8_prefill_scopes": sorted(adapter_scopes(instance)),
        "spomin_backend": callable(getattr(kind, "spomin_backend", None)),
        "quantized": quantized,
        "kernel_env_survival": survival,
    }
print("PROBE-JSON " + json.dumps(out))
"""


def probe(models, kernel_env=None, timeout=900):
    """Adapter facts for ``models`` from a CPU-pinned child process."""
    env = {**BASE_ENV, "PYTHONPATH": str(ROOT / "src"), "MLX2_CAMPAIGN_ROOT": str(ROOT)}
    request = {"models": [m.name for m in models], "kernel_env": kernel_env or {}}
    result = subprocess.run(
        [str(PYTHON), "-c", _PROBE, str(RUN), json.dumps(request)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )
    for line in result.stdout.splitlines():
        if line.startswith("PROBE-JSON "):
            return json.loads(line[len("PROBE-JSON "):])
    raise RuntimeError(f"adapter probe failed: {(result.stderr or result.stdout)[-2000:]}")


# --- series job file ---------------------------------------------------------

def write_job(round_name, model, stage, status, started, summary):
    finished = time.time()
    data = {
        "model": model.name, "stage": stage, "round": round_name, "host": "local",
        "status": status, "started": started, "finished": finished,
        "seconds": round(finished - started), "summary": summary,
    }
    path = JOBS / model.name / f"{round_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    tmp.replace(path)
    print(f"WROTE {path} status={status}", flush=True)
    return path


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    tmp.replace(path)


# --- server commands ---------------------------------------------------------

def strip_policy(args):
    out, skip = [], False
    for item in args:
        if skip:
            skip = False
            continue
        if item == "--execution-policy":
            skip = True
            continue
        out.append(item)
    return out


def route_policy(route, *, opt_in=False) -> dict:
    """The route's own policy (adapter keys and route knobs), as a dict."""
    name = route.opt_in_policy if opt_in else route.policy
    path = cc.policy_path(name)
    return json.loads(path.read_text()) if path else {}


def server_command(model, route, phase, cache_dir: Path, *, policy_file=None, extra=(), keep_route_policy=False):
    args = cc.server_args(model, route, phase)
    if not keep_route_policy:
        args = strip_policy(args)
        if policy_file is not None:
            args += ["--execution-policy", str(policy_file)]
    args += [
        "--cache-dir", str(cache_dir),
        "--admin-token-file", str(RUNTIME / "admin-token"),
        "--reasoning-signing-key-file", str(RUNTIME / "reasoning-key"),
        *extra,
    ]
    return [str(PYTHON), "-u", "-m", "mlx2.server", *args]


def cpu_only():
    """Dry runs pin MLX to the CPU before anything imports mlx2.server."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)


def validate_command(command, parser=None):
    """CPU validation of one generated server command (no model resolution)."""
    return cc.validate_server_arguments(command[4:], parser=parser)


def stage_env(model, *, extra_env=None, shim=False):
    env = {**BASE_ENV, "PYTHONPATH": cc.stage_pythonpath(model),
           "MLX2_CAMPAIGN_ROOT": str(ROOT)}
    if shim:
        env["PYTHONPATH"] = os.pathsep.join((str(SHIM_DIR), env["PYTHONPATH"]))
    env.update(extra_env or {})
    return env


def ensure_runtime_files():
    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    for name in ("admin-token", "reasoning-key"):
        path = RUNTIME / name
        if not path.exists():
            path.write_text(secrets.token_urlsafe(48) + "\n")
            path.chmod(0o600)


# --- server lifecycle --------------------------------------------------------

def health_state():
    from run_campaign import Campaign
    return Campaign.health_state()


class Server:
    """One owned model server on PORT.  ``start()`` returns the readiness record."""

    def __init__(self, name, command, env, log_path: Path, *, startup_timeout=2400):
        self.name, self.command, self.env, self.log_path = name, command, env, log_path
        self.startup_timeout = startup_timeout
        self.process = None
        self.handle = None

    def start(self):
        from run_campaign import Campaign
        if health_state()["reachable"]:
            return {"ready": False, "reason": f"port {PORT} already serves another process"}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("w")
        self.handle.write("$ " + " ".join(self.command) + "\n")
        self.handle.flush()
        self.process = subprocess.Popen(self.command, cwd=ROOT, env=self.env, stdout=self.handle,
                                        stderr=subprocess.STDOUT, start_new_session=True)
        waiter = Campaign.__new__(Campaign)
        started = time.monotonic()
        result = waiter.wait_for_server(self.process, timeout=self.startup_timeout)
        result["load_seconds"] = round(time.monotonic() - started, 1)
        if not result["ready"]:
            result["log_tail"] = self.log_tail()
        return result

    def log_tail(self, size=4000):
        try:
            return self.log_path.read_text(errors="replace")[-size:]
        except OSError:
            return ""

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self.process.wait(120)
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                pass
        if self.handle is not None:
            self.handle.close()
        # Let the port and unified memory drain before the next start.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and health_state()["reachable"]:
            time.sleep(1)
        time.sleep(3)


class Ownership:
    """The private series lock plus the production-service bootout/restore
    that ``run_campaign.py`` performs around every job."""

    def __enter__(self):
        from run_campaign import launchd_loaded
        if SERIES_LOCK.exists():
            raise RuntimeError(f"series GPU lock exists: {SERIES_LOCK.read_text().strip()}")
        SERIES_LOCK.write_text(f"mlx2 series-20260924 extra-round pid {os.getpid()}\n")
        self.was_loaded = launchd_loaded()
        if self.was_loaded:
            subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/{SERVICE}"],
                           capture_output=True, check=False)
            time.sleep(8)
        return self

    def __exit__(self, *exc):
        from run_campaign import restore_production
        try:
            if SERIES_LOCK.exists() and f"pid {os.getpid()}" in SERIES_LOCK.read_text():
                SERIES_LOCK.unlink()
        finally:
            self.restore = restore_production(self.was_loaded)
        return False


# --- HTTP --------------------------------------------------------------------

class HTTP:
    def __init__(self, base=BASE_URL, timeout=900, model_id="campaign"):
        self.base, self.timeout, self.model_id = base.rstrip("/"), timeout, model_id

    def request(self, method, path, body=None, *, timeout=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, method=method, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                raw = response.read().decode(errors="replace")
                return {"status": response.status, "body": json.loads(raw) if raw else None}
        except urllib.error.HTTPError as error:
            raw = error.read().decode(errors="replace")
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = raw
            return {"status": error.code, "body": parsed}
        except Exception as error:  # noqa: BLE001 - every request becomes a record
            return {"status": None, "body": None, "error": f"{type(error).__name__}: {error}"}

    def get(self, path):
        return self.request("GET", path)

    def post(self, path, body, *, timeout=None):
        return self.request("POST", path, body, timeout=timeout)

    def status(self):
        reply = self.get("/v1/status")
        return reply["body"] if reply["status"] == 200 and isinstance(reply["body"], dict) else {}

    def settled(self, *, quiet_s=2.5, timeout=60):
        """Status after the worker is idle and its >=1 s telemetry snapshot
        (apcv2, scheduler, execution, cache_capsules, spomin) has refreshed."""
        return settled_status(self.status, quiet_s=quiet_s, timeout=timeout)

    def chat_body(self, prompt, *, messages=None, **extra):
        body = {"model": self.model_id,
                "messages": messages or [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": 64, "enable_thinking": False}
        body.update(extra)
        return body

    def chat(self, prompt, *, messages=None, timeout=None, **extra):
        return self.post("/v1/chat/completions", self.chat_body(prompt, messages=messages, **extra),
                         timeout=timeout)

    def stream(self, prompt, *, timeout=None, **extra):
        body = self.chat_body(prompt, stream=True, stream_options={"include_usage": True}, **extra)
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        pieces, receipt, usage, done, events = [], {}, {}, False, 0
        started, first = time.monotonic(), None
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                for raw in response:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    event = json.loads(payload)
                    events += 1
                    if event.get("error"):
                        return {"status": 200, "error": event["error"], "text": "".join(pieces), "done": False}
                    usage = event.get("usage") or usage
                    receipt = event.get("mlx2") or receipt
                    for choice in event.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece and first is None:
                            first = time.monotonic()
                        pieces.append(piece)
                return {"status": response.status, "text": "".join(pieces), "done": done, "events": events,
                        "usage": usage, "receipt": receipt,
                        "client_ttft_s": (first - started) if first else None,
                        "wall_s": time.monotonic() - started}
        except urllib.error.HTTPError as error:
            return {"status": error.code, "error": error.read().decode(errors="replace")[:1000], "done": False}
        except Exception as error:  # noqa: BLE001
            return {"status": None, "error": f"{type(error).__name__}: {error}", "done": False}

    def abandon(self, prompt, after_seconds, **extra):
        """Send a streaming request and drop the connection after ``after_seconds``
        (the server cancels on disconnect).  Returns bytes seen before the drop."""
        body = json.dumps(self.chat_body(prompt, stream=True, **extra)).encode()
        host, port = self.base.split("//", 1)[1].split(":")
        connection = http.client.HTTPConnection(host, int(port), timeout=after_seconds + 30)
        connection.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        time.sleep(after_seconds)
        try:
            connection.sock.shutdown(2)
        except OSError:
            pass
        connection.close()
        return {"abandoned_after_s": after_seconds}

    def abandon_after_progress(self, prompt, minimum_processed, **extra):
        """Close a streaming request after an observed prefill boundary.

        Wall-clock cancellation races the model's first prefill slice. A
        progress event proves that a reusable partial cache actually exists.
        """
        body = json.dumps(self.chat_body(prompt, stream=True,
                                         return_progress=True, **extra)).encode()
        host, port = self.base.split("//", 1)[1].split(":")
        connection = http.client.HTTPConnection(host, int(port), timeout=60)
        try:
            connection.request("POST", "/v1/chat/completions", body,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"progress stream returned HTTP {response.status}")
            while raw := response.readline():
                if not raw.startswith(b"data: "):
                    continue
                try:
                    event = json.loads(raw[6:])
                except (ValueError, UnicodeDecodeError):
                    continue
                progress = event.get("prompt_progress") or {}
                processed = int(progress.get("processed") or 0)
                if processed >= minimum_processed:
                    return {"processed": processed, "total": int(progress.get("total") or 0)}
            raise RuntimeError("stream ended before requested prefill progress")
        finally:
            if connection.sock is not None:
                try:
                    connection.sock.shutdown(2)
                except OSError:
                    pass
            connection.close()

    def count(self, text):
        reply = self.post("/v1/messages/count_tokens", {
            "model": self.model_id, "messages": [{"role": "user", "content": text}]})
        body = reply.get("body") or {}
        return int(body.get("input_tokens") or 0) if reply["status"] == 200 else 0


def settled_status(get, *, quiet_s=2.5, timeout=60):
    deadline = time.monotonic() + timeout
    status = get() or {}
    while time.monotonic() < deadline and (status.get("inflight") or status.get("queue_depth")):
        time.sleep(0.5)
        status = get() or {}
    time.sleep(quiet_s)
    return get() or status


def text_of(reply):
    body = reply.get("body") or {}
    choices = body.get("choices") or [{}]
    return (choices[0].get("message") or {}).get("content") or ""


def receipt_of(reply):
    body = reply.get("body") or {}
    return body.get("mlx2") or {}


def cached_tokens(reply):
    body = reply.get("body") or {}
    cached = ((body.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = receipt_of(reply).get("cached_tokens", 0)
    return int(cached or 0)


def dig(mapping, *path, default=0):
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def delta(before, after, *path):
    a, b = dig(after, *path), dig(before, *path)
    return (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def parallel(calls):
    """Run zero-argument callables concurrently, results in order."""
    with ThreadPoolExecutor(max_workers=max(1, len(calls))) as pool:
        futures = [pool.submit(call) for call in calls]
        return [future.result() for future in futures]


def filler(units: int, tag: str) -> str:
    return (f"Record {tag}. " + " ".join(
        f"Entry {i}: the archive notes that shipment {i} arrived on schedule and was logged." for i in range(units)))
