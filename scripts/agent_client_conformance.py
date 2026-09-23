#!/usr/bin/env python3
"""Drive the real Codex CLI and Claude Code binaries against mlx2.

Two modes:

``--scripted`` (CPU, no weights)
    Serves mlx2's real HTTP handler over a scripted engine with the default
    ``--agent-compat auto`` policy (clients are recognized by their identity
    headers; nothing opts in explicitly), then runs the locally installed ``codex exec``
    (``wire_api = "responses"``) and ``claude -p`` in isolated homes and
    scratch workspaces.  The scripted model plays one tool turn (Codex: a
    freeform ``apply_patch`` custom call; Claude Code: ``Read``) and then a
    final answer.  Passes only when every logged request returned 200, the
    client executed the tool (the patch landed / the tool result reached the
    second turn) and every required agent-compat counter is nonzero.

``--model PATH --i-own-the-gpu`` (GPU, real sessions)
    Launches ``python -m mlx2.server --agent-compat auto`` on the model and runs
    small repository tasks through both clients.  Refuses to run without
    ``--i-own-the-gpu``; ``--dry-run`` prints the plan.  Intended to run under
    the lab lock wrapper (see the rm08 plan).

Every request/response pair is appended to ``<out>/requests.jsonl`` (auth
headers dropped) so a failing client field is visible verbatim.  Codex only
declares its freeform ``apply_patch`` tool for catalogued model families, so
the harness copies the installed binary's own ``gpt-5.5`` catalog entry,
renames its slug to the served model id and points ``model_catalog_json`` at
it; nothing from the binary is written outside ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_HEADERS = {"authorization", "x-api-key", "cookie", "proxy-authorization"}
INHERITED_CLIENT_ENV = ("CLAUDE", "ANTHROPIC", "CODEX", "OPENAI")


def clean_env(**extra):
    """The parent's environment minus any inherited agent-client settings.

    A harness launched from inside Claude Code or Codex would otherwise leak
    that host's session variables (and change the client's identity headers).
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(INHERITED_CLIENT_ENV) and key != "CLAUDECODE"
    }
    env.update(extra)
    return env
PATCH = "*** Begin Patch\n*** Add File: hello.txt\n+hello from mlx2\n*** End Patch\n"
CODEX_REQUIRED = (
    "agent_compat_requests",
    "agent_compat_detected_codex",
    "agent_compat_custom_tools_declared",
    "agent_compat_custom_tool_calls",
    "agent_compat_custom_tool_replays",
    "agent_compat_custom_tool_grammar_validated",
    "agent_compat_phase_outputs",
)
CLAUDE_REQUIRED = (
    "agent_compat_detected_claude_code",
    "agent_compat_adaptive_thinking",
    "agent_compat_output_effort",
    "agent_compat_system_folded",
)
GPU_TASKS = {
    "add_file": {
        "prompt": "Create a file named hello.txt whose only content is the line: hello from mlx2",
        "files": {},
        "check": lambda ws: (ws / "hello.txt").is_file()
        and (ws / "hello.txt").read_text().strip() == "hello from mlx2",
    },
    "fix_test": {
        "prompt": "calc.py has a bug. Fix calc.py so `python3 test_calc.py` exits 0. Do not edit test_calc.py.",
        "files": {
            "calc.py": "def add(a, b):\n    return a - b\n",
            "test_calc.py": "from calc import add\nassert add(2, 3) == 5\nprint('ok')\n",
        },
        "check": lambda ws: subprocess.run(
            [sys.executable, "test_calc.py"], cwd=ws, capture_output=True
        ).returncode
        == 0,
    },
}


# ---------------------------------------------------------------------------
# Recording proxy


def start_proxy(upstream: str, log_path: Path, port: int = 0):
    lock = threading.Lock()

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _go(self, method):
            size = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(size) if size else None
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower()
                not in {"host", "content-length", "accept-encoding", "connection"}
            }
            request = urllib.request.Request(
                upstream + self.path, data=body, method=method, headers=headers
            )
            try:
                with urllib.request.urlopen(request, timeout=3600) as response:
                    status, reply_headers, reply = (
                        response.status, dict(response.headers), response.read()
                    )
            except urllib.error.HTTPError as error:
                status, reply_headers, reply = error.code, dict(error.headers), error.read()
            try:
                parsed = json.loads(body) if body else None
            except ValueError:
                parsed = None
            record = {
                "time": time.time(),
                "method": method,
                "path": self.path,
                "user_agent": self.headers.get("user-agent"),
                "headers": {
                    key.lower(): value
                    for key, value in self.headers.items()
                    if key.lower() not in SECRET_HEADERS
                },
                "body": parsed,
                "status": status,
                "response": reply.decode("utf-8", "replace")[-20000:],
            }
            with lock, log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            self.send_response(status)
            for key, value in reply_headers.items():
                if key.lower() not in {"content-length", "transfer-encoding", "connection"}:
                    self.send_header(key, value)
            self.send_header("content-length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def do_GET(self):
            self._go("GET")

        def do_POST(self):
            self._go("POST")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Proxy)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------------------------------------------------------------------------
# Scripted engine (CPU)


def start_scripted_server(model: str, read_path: Path, mode: str = "auto"):
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.agent_compat import AgentCompatPolicy
    from mlx2.reasoning_signatures import ReasoningSigner
    from mlx2.server import handler_for
    from mlx2.serving import Job

    class ScriptedEngine:
        model_path = model
        max_context = 262144

        def __init__(self):
            self.counts = Counter()
            self.reasoning_signer = ReasoningSigner(b"agent-client-conformance")

        def status(self):
            return {
                "healthy": True,
                "error": None,
                "model": model,
                "structured_output": {"thinking_deferral": True},
                "settings": {"constrained_tool_grammar": False},
                "counts": dict(self.counts),
            }

        def batching_status(self):
            return {"schema": "mlx2.batch-runtime.v1", "gauges": {"queue_depth": 0}}

        def count_tokens(self, request):
            return 7

        def submit(self, request, *, tenant_id="default"):
            job = Job(request)
            job.tenant_id = tenant_id
            job.prompt_tokens, job.cached_tokens = 6, 1
            if request.get("enable_thinking"):
                job.events.put({"delta": {"reasoning_content": "careful thought"}})
            names = {tool["function"]["name"] for tool in request.get("tools", ())}
            history = any(m.get("role") == "tool" for m in request["messages"])
            call = None
            if not history and "apply_patch" in names:
                call = ("apply_patch", json.dumps({"input": PATCH}))
            elif not history and "Read" in names:
                call = ("Read", json.dumps({"file_path": str(read_path)}))
            if call:
                job.events.put({"delta": {"content": "Let me do that."}})
                job.events.put({"delta": {"tool_calls": [{
                    "index": 0, "id": "call_scripted_1", "type": "function",
                    "function": {"name": call[0], "arguments": call[1]},
                }]}})
                job.completion_tokens = 5
                job.events.put({"finish_reason": "tool_calls", "receipt": {"cache": "apcv2"}})
            else:
                suffix = " (tool result seen)" if history else ""
                job.events.put({"delta": {"content": "hello from mlx2" + suffix}})
                job.completion_tokens = 2
                job.events.put({"finish_reason": "stop", "receipt": {"cache": "apcv2"}})
            return job

    engine = ScriptedEngine()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_for(
            engine,
            agent_compat=AgentCompatPolicy(mode, "validate"),
        ),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return engine, server


# ---------------------------------------------------------------------------
# Clients


def codex_binary_catalog(served_model: str, out: Path) -> Path | None:
    """Copy the installed Codex's own gpt-5.5 catalog entry for ``served_model``."""
    codex = shutil.which("codex")
    if codex is None:
        return None
    root = Path(os.path.realpath(codex)).parents[1]
    candidates = list(root.glob("node_modules/@openai/codex-*/vendor/*/bin/codex"))
    if not candidates:
        return None
    blob = candidates[0].read_bytes()
    anchor = blob.find(b'"slug": "gpt-5.5"')
    start = blob.rfind(b'{\n  "models"', 0, anchor)
    if anchor < 0 or start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for end in range(start, len(blob)):
        char = blob[end : end + 1]
        if in_string:
            if escaped:
                escaped = False
            elif char == b"\\":
                escaped = True
            elif char == b'"':
                in_string = False
            continue
        if char == b'"':
            in_string = True
        elif char == b"{":
            depth += 1
        elif char == b"}":
            depth -= 1
            if depth == 0:
                break
    catalog = json.loads(blob[start : end + 1])
    entry = next(m for m in catalog["models"] if m["slug"] == "gpt-5.5")
    entry = {**entry, "slug": served_model, "display_name": served_model}
    path = out / "codex_model_catalog.json"
    path.write_text(json.dumps({"models": [entry]}))
    return path


def run_codex(base_url, model, workspace, home, prompt, timeout, catalog):
    home.mkdir(parents=True, exist_ok=True)
    lines = [
        f'model = "{model}"',
        'model_provider = "mlx2"',
        'approval_policy = "never"',
        'sandbox_mode = "workspace-write"',
    ]
    if catalog is not None:
        lines.append(f'model_catalog_json = "{catalog}"')
    lines += [
        "[model_providers.mlx2]",
        'name = "mlx2 local"',
        f'base_url = "{base_url}/v1"',
        'wire_api = "responses"',
        'env_key = "MLX2_API_KEY"',
        "request_max_retries = 0",
        "stream_max_retries = 0",
        "[analytics]",
        "enabled = false",
    ]
    (home / "config.toml").write_text("\n".join(lines) + "\n")
    env = clean_env(CODEX_HOME=str(home), MLX2_API_KEY="local")
    return subprocess.run(
        ["codex", "exec", "--skip-git-repo-check", prompt],
        cwd=workspace, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def run_claude(base_url, model, workspace, config_dir, prompt, timeout, tools):
    config_dir.mkdir(parents=True, exist_ok=True)
    env = clean_env(
        CLAUDE_CONFIG_DIR=str(config_dir),
        ANTHROPIC_BASE_URL=base_url,
        ANTHROPIC_API_KEY="local",
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        DISABLE_TELEMETRY="1",
    )
    return subprocess.run(
        ["claude", "-p", prompt, "--model", model, "--allowedTools", tools,
         "--permission-mode", "acceptEdits"],
        cwd=workspace, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def summarize(log_path: Path, agent: str):
    records = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    mine = [r for r in records if r["method"] == "POST" and agent in (r.get("user_agent") or "").lower()]
    return {
        "requests": len(mine),
        "non_200": [
            {"path": r["path"], "status": r["status"], "response": r["response"][:300]}
            for r in mine if r["status"] != 200
        ],
    }


def scripted(args) -> int:
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "requests.jsonl"
    log.unlink(missing_ok=True)
    work = Path(tempfile.mkdtemp(prefix="agent-conf-", dir=out))
    notes = work / "claude_ws" / "notes.txt"
    notes.parent.mkdir(parents=True)
    notes.write_text("probe file contents\n")
    engine, server = start_scripted_server(args.served_model, notes, args.compat_mode)
    proxy = start_proxy(f"http://127.0.0.1:{server.server_port}", log)
    base = f"http://127.0.0.1:{proxy.server_port}"
    results, failures = {}, []
    try:
        if "codex" in args.clients:
            if shutil.which("codex") is None:
                failures.append("codex: binary not installed")
            else:
                before = Counter(engine.counts)
                workspace = work / "codex_ws"
                workspace.mkdir()
                catalog = codex_binary_catalog(args.served_model, out)
                run = run_codex(base, args.served_model, workspace, work / "codex_home",
                                "Create hello.txt", args.timeout, catalog)
                delta = Counter(engine.counts) - before
                summary = summarize(log, "codex")
                patched = (workspace / "hello.txt").is_file()
                summary.update(returncode=run.returncode, patch_applied=patched,
                               counters=dict(delta), catalog=catalog is not None)
                results["codex"] = summary
                required = CODEX_REQUIRED if args.compat_mode == "auto" else CODEX_REQUIRED[:1] + CODEX_REQUIRED[2:]
                missing = [key for key in required if not delta[key]]
                if run.returncode or summary["non_200"] or not patched or missing or summary["requests"] < 2:
                    failures.append(f"codex: rc={run.returncode} non_200={len(summary['non_200'])} "
                                    f"patched={patched} zero_counters={missing}")
                (out / "codex.log").write_text(run.stdout + run.stderr)
        if "claude" in args.clients:
            if shutil.which("claude") is None:
                failures.append("claude: binary not installed")
            else:
                before = Counter(engine.counts)
                run = run_claude(base, args.served_model, notes.parent, work / "claude_cfg",
                                 "Read notes.txt", args.timeout, "Read")
                delta = Counter(engine.counts) - before
                summary = summarize(log, "claude")
                seen = "tool result seen" in run.stdout
                summary.update(returncode=run.returncode, tool_round_trip=seen, counters=dict(delta))
                results["claude"] = summary
                required = CLAUDE_REQUIRED if args.compat_mode == "auto" else CLAUDE_REQUIRED[1:]
                missing = [key for key in required if not delta[key]]
                if run.returncode or summary["non_200"] or not seen or missing:
                    failures.append(f"claude: rc={run.returncode} non_200={len(summary['non_200'])} "
                                    f"round_trip={seen} zero_counters={missing}")
                (out / "claude.log").write_text(run.stdout + run.stderr)
    finally:
        proxy.shutdown()
        server.shutdown()
        shutil.rmtree(work, ignore_errors=True)
    verdict = {"mode": "scripted", "results": results, "failures": failures,
               "passed": not failures}
    (out / "summary.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return int(bool(failures))


# ---------------------------------------------------------------------------
# GPU verdict (pure; unit-tested on CPU)

_CACHED_RE = re.compile(r'"(?:cached_tokens|cache_read_input_tokens)"\s*:\s*(\d+)')


def status_counts(upstream: str) -> dict:
    with urllib.request.urlopen(upstream + "/v1/status", timeout=30) as response:
        return json.load(response).get("counts", {})


def request_cached_tokens(record) -> int | None:
    """Largest cached-token count a logged response reports (None if absent).

    Covers Responses ``input_tokens_details.cached_tokens`` and Messages
    ``usage.cache_read_input_tokens`` in both SSE and JSON bodies."""
    text = record.get("response") or ""
    if not isinstance(text, str):
        text = json.dumps(text)
    found = [int(value) for value in _CACHED_RE.findall(text)]
    return max(found) if found else None


def counter_delta(before: dict, after: dict) -> dict:
    return {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}


def gpu_verdict(rows, agent_counts, clients, compat_mode, min_pass_rate=0.75):
    """Pre-registered rm08 go criterion (plan "GPU test plan").

    go = every request 200; agent_compat_custom_tool_calls > 0 when codex ran;
    both clients auto-detected (auto mode); no grammar rejection on a passing
    task; pass rate >= 3/4 per client; every turn after the first in a session
    reports cached tokens > 0."""
    failures = []
    if any(row["non_200"] for row in rows):
        failures.append("non-200 responses")
    if "codex" in clients and not agent_counts.get("agent_compat_custom_tool_calls"):
        failures.append("codex never produced a custom apply_patch call (counter 0)")
    if compat_mode == "auto":
        for client in clients:
            key = "agent_compat_detected_" + ("codex" if client == "codex" else "claude_code")
            if not agent_counts.get(key):
                failures.append(f"{client} was not auto-detected ({key} is 0)")
    rejected_passing = [
        f"{row['client']}/{row['task']}#{row['repeat']}" for row in rows
        if row["passed"] and row.get("counter_delta", {}).get("agent_compat_custom_tool_grammar_rejected")
    ]
    if rejected_passing:
        failures.append(f"grammar rejections on passing tasks: {rejected_passing}")
    per_client = {}
    for client in clients:
        mine = [row for row in rows if row["client"] == client]
        passed = sum(row["passed"] for row in mine)
        per_client[client] = f"{passed}/{len(mine)}"
        if not mine or passed < min_pass_rate * len(mine):
            failures.append(f"{client} pass rate {passed}/{len(mine)} below {min_pass_rate:.0%}")
    later = [(row, value) for row in rows for value in row.get("cached_tokens", [])[1:]]
    uncached = [f"{row['client']}/{row['task']}#{row['repeat']}" for row, value in later if not value]
    if not later:
        failures.append("no multi-turn session to check cached_tokens")
    elif uncached:
        failures.append(f"later turns without cached tokens: {sorted(set(uncached))}")
    cache_hits = f"{sum(bool(v) for _, v in later)}/{len(later)}"
    return {"pass_rate": per_client, "later_turn_cache_hits": cache_hits,
            "grammar_rejected_total": agent_counts.get("agent_compat_custom_tool_grammar_rejected", 0),
            "failures": failures, "go": not failures}


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex((host, int(port))) == 0


def gpu(args) -> int:
    out = Path(args.out).resolve()
    port = args.port
    server_cmd = [
        sys.executable, "-m", "mlx2.server", "--model", args.model,
        "--port", str(port), "--agent-compat", args.compat_mode,
        "--custom-tool-grammar", args.custom_tool_grammar, *args.server_arg,
    ]
    # mlx2.server refuses to start without a qualification record or explicit
    # candidate mode; this harness is candidate evidence, not a qualified route.
    if not any(arg == "--qualification" or arg.startswith("--qualification=")
               or arg == "--qualification-mode" for arg in args.server_arg):
        server_cmd.append("--qualification-mode")
    plan = {"server": server_cmd, "clients": args.clients, "tasks": list(GPU_TASKS),
            "repeats": args.repeats, "out": str(out)}
    if args.dry_run or not args.i_own_the_gpu:
        print(json.dumps(plan, indent=2))
        if not args.i_own_the_gpu and not args.dry_run:
            print("refusing to launch a model server without --i-own-the-gpu", file=sys.stderr)
            return 2
        return 0
    # mlx2.server binds its port before loading the model, so on a collision
    # the launched server exits at once while another session's server keeps
    # answering on that port.  Refuse rather than measure a neighbour.
    if port_in_use(port):
        print(f"refusing launch: port {port} is already in use", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    log = out / "requests.jsonl"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    server = subprocess.Popen(server_cmd, env=env, stdout=(out / "server.log").open("w"),
                              stderr=subprocess.STDOUT)
    upstream = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + args.startup_timeout
        status = None
        while time.time() < deadline:
            if server.poll() is not None:
                print(f"server exited with {server.returncode} before readiness; "
                      f"see {out / 'server.log'}", file=sys.stderr)
                return 2
            try:
                with urllib.request.urlopen(upstream + "/v1/status", timeout=5) as response:
                    status = json.load(response)
                if status.get("healthy"):
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(5)
        if not status or not status.get("healthy"):
            print("server did not become healthy", file=sys.stderr)
            return 2
        expected_model = Path(args.model).name
        if server.poll() is not None or status.get("model") != expected_model:
            print(f"port {port} serves {status.get('model')!r}, not the launched "
                  f"{expected_model!r} (launched server returncode "
                  f"{server.poll()})", file=sys.stderr)
            return 2
        served = status["model"]
        catalog = codex_binary_catalog(served, out)
        proxy = start_proxy(upstream, log)
        base = f"http://127.0.0.1:{proxy.server_port}"
        rows = []
        for repeat in range(args.repeats):
            for client in args.clients:
                for name, task in GPU_TASKS.items():
                    workspace = Path(tempfile.mkdtemp(prefix=f"{client}-{name}-", dir=out))
                    for filename, content in task["files"].items():
                        (workspace / filename).write_text(content)
                    before = log.read_text().count("\n") if log.exists() else 0
                    counts_before = status_counts(upstream)
                    started = time.time()
                    run = None
                    try:
                        if client == "codex":
                            run = run_codex(base, served, workspace, workspace.parent / f".home-{workspace.name}",
                                            task["prompt"], args.timeout, catalog)
                        else:
                            run = run_claude(base, served, workspace, workspace.parent / f".cfg-{workspace.name}",
                                             task["prompt"], args.timeout,
                                             "Read,Edit,Write,Bash(python3 test_calc.py)")
                        returncode = run.returncode
                    except subprocess.TimeoutExpired:
                        returncode = "timeout"
                    seconds = round(time.time() - started, 1)
                    if run is not None:
                        (workspace.parent / f"{workspace.name}.client.log").write_text(
                            (run.stdout or "") + "\n--- stderr ---\n" + (run.stderr or ""))
                    records = [json.loads(line) for line in log.read_text().splitlines()[before:]]
                    posts = [r for r in records if r["method"] == "POST"]
                    delta = counter_delta(counts_before, status_counts(upstream))
                    rows.append({
                        "repeat": repeat, "client": client, "task": name,
                        "returncode": returncode, "passed": bool(task["check"](workspace)),
                        "requests": len(posts),
                        "non_200": sum(r["status"] != 200 for r in posts),
                        "cached_tokens": [request_cached_tokens(r) for r in posts],
                        "counter_delta": {k: v for k, v in delta.items() if k.startswith("agent_compat")},
                        "seconds": seconds,
                    })
                    print(json.dumps(rows[-1]), flush=True)
        counts = status_counts(upstream)
        agent_counts = {k: v for k, v in counts.items() if k.startswith("agent_compat")}
        verdict = {"mode": "gpu", "compat_mode": args.compat_mode, "model": served,
                   "server": server_cmd, "rows": rows, "agent_compat_counts": agent_counts,
                   **gpu_verdict(rows, agent_counts, args.clients, args.compat_mode)}
        if server.poll() is not None:
            verdict["failures"].append(
                f"launched server exited with {server.returncode} during measurement"
            )
            verdict["go"] = False
        failures = verdict["failures"]
        (out / "summary.json").write_text(json.dumps(verdict, indent=2))
        print(json.dumps(verdict, indent=2))
        proxy.shutdown()
        return int(bool(failures))
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripted", action="store_true", help="CPU scripted-engine mode")
    parser.add_argument("--model", help="GPU mode: model path for mlx2.server")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clients", default="codex,claude",
                        type=lambda value: [item for item in value.split(",") if item])
    parser.add_argument("--served-model", default="agent-fixture")
    parser.add_argument("--custom-tool-grammar", choices=("off", "validate"), default="validate")
    parser.add_argument(
        "--compat-mode", choices=("auto", "on"), default="auto",
        help="server agent-compat mode; auto (default) exercises client detection",
    )
    parser.add_argument("--server-arg", action="append", default=[],
                        help="extra argument forwarded to mlx2.server (repeatable)")
    parser.add_argument("--port", type=int, default=8297)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=None,
                        help="per client run seconds (default 120 scripted, 1800 GPU)")
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--out", default="agent-client-conformance")
    args = parser.parse_args()
    if args.scripted == bool(args.model):
        parser.error("choose exactly one of --scripted or --model")
    if args.timeout is None:
        args.timeout = 120.0 if args.scripted else 1800.0
    return scripted(args) if args.scripted else gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())
