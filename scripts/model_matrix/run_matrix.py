#!/usr/bin/env python3
"""Smoke one artifact through one or more serving routes over loopback HTTP.

For each route this starts ``python -m mlx2.server`` in its own process group,
waits for ``GET /health == 200`` (``/v1/models`` answers during load), sends a
fixed set of varied realistic prompts (code, explanation, multi-turn
follow-up, one ~4K-token frozen-Spomin transcript, default sampling, and a
tool call when the adapter declares tools), checks output sanity, and writes
an atomic per-cell receipt.  The server group is always stopped in
``finally``.  A nonempty evidence directory is never overwritten.

This is smoke evidence (implemented / observed-used), never qualification.
The caller owns the GPU lock: run it under ``scripts/run_with_gpu_locks.py``
(M5) or the M3 queue wrapper, one artifact per lock window.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SCHEMA = "mlx2.model-matrix-cell.v1"
SPOMIN_CORPUS = ROOT / "qualification/corpora/spomin-20x20-long-multiturn-corpus-20260915.json"
PAGE = 16384

ROUTE_KIND = {
    "ordinary": "ordinary",
    "pld_default": "prompt_lookup",
    "pld_advanced": "prompt_lookup",
    "native_mtp": "native_mtp",
}

CODE_PROMPT = (
    "Write a Python function `merge_intervals(intervals)` that takes a list of "
    "[start, end] pairs, merges every overlapping or touching pair, and returns "
    "the merged list sorted by start. Include a short docstring and two example "
    "calls with their expected output."
)
EXPLAIN_PROMPT = (
    "Explain to a curious high-school student why the daytime sky looks blue "
    "but sunsets look red or orange. Keep it to one short paragraph."
)
FOLLOWUP_HISTORY = [
    {"role": "user", "content": "I'm planning a three-day trip to Lisbon in late October. What should I pack?"},
    {"role": "assistant", "content": (
        "Late October in Lisbon is mild, usually 15-22 C, with a real chance of rain. "
        "Pack layers: a light sweater, a waterproof jacket, comfortable walking shoes "
        "with grip for the steep cobblestone streets, and a compact umbrella."
    )},
    {"role": "user", "content": (
        "Good to know about the hills. Which two neighbourhoods would you suggest for "
        "a first visit, and how would I get between them without a car?"
    )},
]
TOOL_PROMPT = "What is the weather like in Paris right now? Use the available tool."
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}


# ------------------------------------------------------------------ host --

def vm_counters() -> dict:
    text = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    values = {}
    for line in text.splitlines():
        match = re.match(r'^"?([^":]+)"?:\s+(\d+)\.?$', line.strip())
        if match:
            values[match.group(1).strip()] = int(match.group(2))
    return values


def available_bytes(counters: dict) -> int:
    pages = sum(counters.get(key, 0) for key in (
        "Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"))
    return pages * PAGE


def swap_used_bytes() -> int:
    text = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    match = re.search(r"used = ([\d.]+)M", text)
    return int(float(match.group(1)) * (1 << 20)) if match else -1


def model_processes() -> list[str]:
    rows = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    pattern = re.compile(r"-m\s+mlx2\.server|mlx_lm\.server|mlx_vlm\.server|llama-server|tensorfold-native")
    own = {os.getpid(), os.getppid()}
    found = []
    for row in rows.splitlines():
        fields = row.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) in own or not pattern.search(fields[1]):
            continue
        # Shell drivers and ssh clients merely mention a server in their
        # arguments, and agent CLIs carry prompts that quote server commands;
        # none of them own this host's GPU (as run_with_gpu_locks).
        executable = Path(fields[1].split(None, 1)[0]).name
        if executable in {"bash", "zsh", "sh", "ssh", "git", "node", "codex", "claude"}:
            continue
        arguments = fields[1].split()
        script = Path(arguments[1]).name if len(arguments) > 1 else ""
        # A queued GPU-lock wrapper, or an agent CLI whose prompt quotes a
        # server command line, owns no GPU (as run_with_gpu_locks).
        if script == "run_with_gpu_locks.py" or (
            executable == "node" and script in {"codex", "claude"}
        ):
            continue
        found.append(row.strip()[:300])
    return found


def process_tree_rss(pid: int) -> int:
    rows = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for row in rows.splitlines():
        fields = row.split()
        if len(fields) == 3:
            p, pp, r = map(int, fields)
            children.setdefault(pp, []).append(p)
            rss[p] = r
    total, stack = 0, [pid]
    while stack:
        current = stack.pop()
        total += rss.get(current, 0) * 1024
        stack.extend(children.get(current, ()))
    return total


def phys_footprint(pid: int) -> int | None:
    """macOS physical footprint (includes Metal allocations) via footprint(1)."""
    try:
        text = subprocess.run(["footprint", "-p", str(pid)], capture_output=True,
                              text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"phys_footprint:\s+([\d.]+)\s*([KMG])B", text)
    if not match:
        match = re.search(r"Footprint:\s+([\d.]+)\s*([KMG])B", text)
    if not match:
        return None
    scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[match.group(2)]
    return int(float(match.group(1)) * scale)


class Sampler(threading.Thread):
    def __init__(self, pid: int, interval: float = 2.0):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.peak_rss = 0
        self.peak_footprint = 0
        self.stop_event = threading.Event()

    def run(self) -> None:
        tick = 0
        while not self.stop_event.is_set():
            try:
                self.peak_rss = max(self.peak_rss, process_tree_rss(self.pid))
                if tick % 5 == 0:
                    value = phys_footprint(self.pid)
                    if value:
                        self.peak_footprint = max(self.peak_footprint, value)
            except Exception:  # noqa: BLE001 - sampling is best effort
                pass
            tick += 1
            self.stop_event.wait(self.interval)


class SwapRise(RuntimeError):
    pass


# ------------------------------------------------------------------ http --

def http_json(url: str, body: dict | None = None, timeout: float = 600) -> tuple[int, dict | str, float]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            code = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        code = error.code
    elapsed = time.monotonic() - started
    try:
        return code, json.loads(raw), elapsed
    except json.JSONDecodeError:
        return code, raw.decode(errors="replace")[:4000], elapsed


def wait_health(base: str, process: subprocess.Popen, timeout: float, swap_guard) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during load with rc={process.returncode}")
        swap_guard()
        try:
            with urllib.request.urlopen(base + "/health", timeout=3) as response:
                if response.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy within {timeout:g}s")


def stop_server(process: subprocess.Popen | None) -> dict:
    if process is None:
        return {"stopped": "never_started"}
    if process.poll() is not None:
        return {"stopped": "already_exited", "returncode": process.returncode}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
        result = {"stopped": "sigterm", "returncode": process.returncode}
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)
        result = {"stopped": "sigkill", "returncode": process.returncode}
    # Any stragglers in the group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return result


# ---------------------------------------------------------------- sanity --

def degenerate_loop(text: str) -> str | None:
    stripped = text.rstrip()
    if len(stripped) < 40:
        return None
    for period in range(1, 121):
        if len(stripped) < period * 4:
            break
        unit = stripped[-period:]
        if stripped.endswith(unit * 4) and (period > 1 or unit not in " \n"):
            covered = period * 4
            while stripped.endswith(unit * (covered // period + 1)):
                covered += period
            if covered >= max(48, len(stripped) * 0.15):
                return f"tail repeats a {period}-char unit {covered // period}x"
    if len(stripped) > 300:
        ratio = len(zlib.compress(stripped.encode())) / len(stripped.encode())
        if ratio < 0.12:
            return f"compression ratio {ratio:.3f} indicates repetition"
    words = re.findall(r"\w+", stripped.lower())
    if len(words) >= 40:
        grams = [" ".join(words[i:i + 6]) for i in range(len(words) - 5)]
        counts: dict[str, int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        top = max(counts.values())
        if top >= 5 and top * 6 > len(words) * 0.5:
            return f"a 6-word phrase repeats {top}x"
    return None


def coherence(text: str) -> dict:
    words = text.split()
    alpha_words = [w for w in words if re.search(r"[A-Za-z]{2,}", w)]
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    return {
        "words": len(words),
        "alpha_word_ratio": round(len(alpha_words) / len(words), 3) if words else 0.0,
        "replacement_chars": text.count("�"),
        "printable_ratio": round(printable / len(text), 3) if text else 0.0,
    }


def evaluate(kind: str, response: dict) -> dict:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    tool_calls = message.get("tool_calls") or []
    finish = choice.get("finish_reason")
    text = content or reasoning
    issues, warnings = [], []
    if not text.strip() and not tool_calls:
        issues.append("empty output")
    loop = degenerate_loop(content) or degenerate_loop(reasoning)
    if loop:
        issues.append("degenerate repetition: " + loop)
    stats = coherence(text) if text else {}
    if text.strip():
        if stats["replacement_chars"] > 3:
            issues.append(f"{stats['replacement_chars']} U+FFFD replacement characters")
        if stats["printable_ratio"] < 0.97:
            issues.append(f"printable ratio {stats['printable_ratio']}")
        if stats["words"] >= 8 and stats["alpha_word_ratio"] < 0.45 and kind != "code":
            issues.append(f"alpha word ratio {stats['alpha_word_ratio']}")
    if finish not in {"stop", "length", "tool_calls"}:
        issues.append(f"finish_reason {finish!r}")
    checks = {}
    lowered = (content + "\n" + reasoning).lower()
    if kind == "code":
        checks["has_def"] = "def merge_intervals" in content or "def merge_intervals" in reasoning
    elif kind in {"explain", "explain_default_sampling"}:
        checks["topic"] = any(t in lowered for t in ("scatter", "rayleigh", "wavelength", "blue light"))
    elif kind == "followup":
        checks["topic"] = any(t in lowered for t in (
            "alfama", "baixa", "chiado", "bairro", "belém", "belem", "príncipe", "tram", "metro"))
    elif kind == "tool":
        names = [((call.get("function") or {}).get("name")) for call in tool_calls]
        checks["tool_called"] = "get_current_weather" in names
        arguments_ok = False
        for call in tool_calls:
            try:
                arguments = json.loads((call.get("function") or {}).get("arguments") or "{}")
                arguments_ok = arguments_ok or "paris" in json.dumps(arguments).lower()
            except (TypeError, json.JSONDecodeError):
                pass
        checks["arguments_parse_with_city"] = arguments_ok
        if not checks["tool_called"]:
            # Tool use is a declared capability: missing it is a failure.
            issues.append("declared tool support produced no get_current_weather call")
    for key, value in checks.items():
        if value is False and kind != "tool":
            warnings.append(f"content check {key} failed")
    return {
        "sane": not issues,
        "issues": issues,
        "warnings": warnings,
        "finish_reason": finish,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "tool_calls": tool_calls,
        "checks": checks,
        "coherence": stats,
        "content_head": content[:600],
        "content_tail": content[-300:] if len(content) > 600 else "",
        "reasoning_head": reasoning[:300],
    }


def spomin_messages() -> tuple[list[dict], dict]:
    sys.path.insert(0, str(SCRIPTS))
    from run_spomin_20x20 import prepare_case  # frozen-corpus transcript builder

    corpus = json.loads(SPOMIN_CORPUS.read_text())

    class CharTokenizer:
        """Model-independent sizing (~4 chars/token) so every model sees identical bytes."""

        @staticmethod
        def encode(text, add_special_tokens=False):
            return range(max(1, len(text) // 4))

    case = corpus["cases"][0]
    prepared = prepare_case(case, CharTokenizer(), 5250)
    messages = [{"role": "system", "content": corpus["system"]}, *prepared["full_messages"]]
    oracle = {"case_id": case["case_id"], "needles": prepared["needles"], "sentinel": prepared["sentinel"],
              "approx_tokens": prepared["receipt"]["full_ledger_tokens"]}
    return messages, oracle


def spomin_oracle(response: dict, oracle: dict) -> dict:
    message = ((response.get("choices") or [{}])[0]).get("message") or {}
    text = (message.get("content") or "") + "\n" + (message.get("reasoning_content") or "")
    normalized = re.sub(r"[‐-―−]", "-", text)
    found = {label: value in normalized for label, value in oracle["needles"].items()}
    return {"needles_found": found, "sentinel_found": oracle["sentinel"] in normalized,
            "all_found": all(found.values()) and oracle["sentinel"] in normalized}


def summarize_mlx2(block: dict | None) -> dict:
    if not isinstance(block, dict):
        return {}
    speculation = block.get("speculation")
    summary = {
        key: block.get(key)
        for key in ("route", "route_selection_source", "profile", "qualification", "route_receipt",
                    "prompt_tokens", "completion_tokens", "cached_tokens", "ttft_seconds",
                    "elapsed_seconds")
        if key in block
    }
    if isinstance(speculation, dict):
        summary["speculation"] = {
            key: speculation.get(key)
            for key in ("kind", "proposed", "accepted", "cycles", "round_proposed", "round_accepted",
                        "draft_width", "target_width", "external_rounds", "acceptance_rate",
                        "admission_state", "qualification_authority", "route", "mode")
            if key in speculation
        }
        summary["speculation_keys"] = sorted(speculation)[:60]
    elif speculation is not None:
        summary["speculation"] = speculation
    mtp = block.get("mtp")
    if isinstance(mtp, dict):
        summary["mtp"] = {k: mtp.get(k) for k in ("proposed", "accepted", "depth", "rounds", "acceptance_rate",
                                                  "draft_tokens", "accepted_tokens", "mode", "num_draft")
                          if k in mtp}
        stats = mtp.get("stats")
        if isinstance(stats, dict):
            # Native-MTP lanes report draft counters under stats.
            summary["mtp"].update(
                proposed=stats.get("draft_proposed"), accepted=stats.get("draft_accepted"),
                total_emitted=stats.get("total_emitted"), draft_acceptance=stats.get("draft_acceptance"))
        summary["mtp_keys"] = sorted(mtp)[:60]
    elif mtp is not None:
        summary["mtp"] = mtp
    return summary


def find_keys(value, needle: str, prefix: str = "", out: dict | None = None, depth: int = 0) -> dict:
    out = {} if out is None else out
    if depth > 6:
        return out
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if needle in str(key).lower() and isinstance(child, (int, float, str, bool)):
                out[path] = child
            find_keys(child, needle, path, out, depth + 1)
    return out


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


# ------------------------------------------------------------------ cell --

def run_cell(args, target: dict, route: dict, out: Path, revision: str, dirty: bool) -> dict:
    route_name = route["route"]
    # --ast resolves its own route (tree tiles -> external_draft, else the
    # family's best chain route), so its receipt is recorded, not asserted.
    expected_kind = route.get("expected_route_kind") or ROUTE_KIND.get(
        route_name, "external_draft" if route_name.startswith("external") else route_name)
    receipt: dict = {
        "schema": SCHEMA,
        "status": "running",
        "verdict": None,
        "checkpoint": target["checkpoint"],
        "model_path": target["path"],
        "adapter": target.get("adapter"),
        "route": route_name,
        "expected_route_kind": expected_kind,
        "draft": route.get("draft"),
        "host": args.host_label,
        "source_revision": revision,
        "source_tracked_dirty": dirty,
        "evidence_kind": "smoke (implemented/observed-used only; never qualification)",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out.mkdir(parents=True, exist_ok=True)
    counters = vm_counters()
    receipt["host_before"] = {
        "available_bytes": available_bytes(counters),
        "swapouts": counters.get("Swapouts"),
        "swap_used_bytes": swap_used_bytes(),
        "foreign_model_processes": model_processes(),
    }
    weight = int(target.get("weight_bytes") or 0) + int(route.get("draft_weight_bytes") or 0)
    ple_rows = Path(target["path"]) / "ple_rows.bin"
    if ple_rows.is_file():
        # Flash-Next serves per-layer embeddings from ple_rows.bin (host
        # offload); shards that also carry the PLE tensors overstate residency.
        try:
            names = json.loads((Path(target["path"]) / "model.safetensors.index.json").read_text())["weight_map"]
        except (OSError, ValueError, KeyError):
            names = {}
        if any("ngram_embedding.shard_" in name for name in names):
            weight = max(0, weight - ple_rows.stat().st_size)
    need = int(weight * args.memory_factor) + (4 << 30)
    if receipt["host_before"]["foreign_model_processes"]:
        receipt.update(status="skipped", verdict="skipped",
                       reason="foreign model process present; refusing to contend")
        atomic_json(out / "cell.json", receipt)
        return receipt
    if args.max_model_bytes and weight > args.max_model_bytes:
        receipt.update(status="skipped", verdict="skipped",
                       reason=f"weights {weight / 1e9:.1f} GB exceed host budget {args.max_model_bytes / 1e9:.1f} GB")
        atomic_json(out / "cell.json", receipt)
        return receipt
    settle_deadline = time.monotonic() + max(0.0, getattr(args, "memory_settle_seconds", 0.0))
    while (receipt["host_before"]["available_bytes"] < need and not args.ignore_free_memory
           and time.monotonic() < settle_deadline):
        # The previous cell's server releases tens of GiB of Metal buffers
        # for a while after it exits; wait for them instead of skipping.
        time.sleep(5)
        counters = vm_counters()
        receipt["host_before"]["available_bytes"] = available_bytes(counters)
        receipt["host_before"]["memory_settle_waited_seconds"] = round(
            receipt["host_before"].get("memory_settle_waited_seconds", 0) + 5, 1)
    if receipt["host_before"]["available_bytes"] < need and not args.ignore_free_memory:
        receipt.update(status="skipped", verdict="skipped",
                       reason=f"available {receipt['host_before']['available_bytes'] / 1e9:.1f} GB < needed {need / 1e9:.1f} GB")
        atomic_json(out / "cell.json", receipt)
        return receipt

    swapouts_start = counters.get("Swapouts", 0)

    def swap_guard() -> None:
        now = vm_counters().get("Swapouts", 0)
        if now - swapouts_start > args.swap_pages_limit:
            raise SwapRise(f"Swapouts rose by {now - swapouts_start} pages")

    command = [
        sys.executable, "-u", "-m", "mlx2.server", "--model", target["path"],
        "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
        "--max-context", str(args.max_context), "--cache-bytes", str(args.cache_gib << 30),
    ]
    for flag in route["server_flags"]:
        if flag == "<policy.json>":
            policy_path = out / "policy.json"
            atomic_json(policy_path, route["execution_policy"])
            command.append(str(policy_path))
        else:
            command.append(flag)
    receipt["server_command"] = command
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "PYTHONUNBUFFERED": "1",
    }
    base = f"http://127.0.0.1:{args.port}"
    process = None
    sampler = None
    requests_log = []
    try:
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                raise RuntimeError(f"port {args.port} already in use")
        log = (out / "server.log").open("w")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        receipt["server_pid"] = process.pid
        sampler = Sampler(process.pid)
        sampler.start()
        try:
            receipt["load_seconds"] = round(wait_health(base, process, args.load_timeout, swap_guard), 2)
        except RuntimeError as error:
            receipt["load_ok"] = False
            receipt["load_error"] = str(error)
            raise
        receipt["load_ok"] = True
        code, status, _ = http_json(base + "/v1/status", timeout=30)
        receipt["startup_status"] = {
            key: status.get(key) for key in (
                "artifact", "profile", "qualification", "route_receipt", "settings", "execution",
                "route", "route_selection", "model", "unqualified", "capabilities")
            if isinstance(status, dict) and key in status
        }
        code, models, _ = http_json(base + "/v1/models", timeout=30)
        model_id = ((models.get("data") or [{}])[0].get("id")) if isinstance(models, dict) else None
        model_id = model_id or Path(target["path"]).name
        caps = set(target.get("capabilities") or ())
        spomin, oracle = spomin_messages()
        plan = [
            ("code", {"messages": [{"role": "user", "content": CODE_PROMPT}], "temperature": 0}),
            ("explain", {"messages": [{"role": "user", "content": EXPLAIN_PROMPT}], "temperature": 0}),
            ("followup", {"messages": FOLLOWUP_HISTORY, "temperature": 0}),
            ("spomin_4k", {"messages": spomin, "temperature": 0, "max_tokens": 256, "enable_thinking": False}),
            ("explain_default_sampling", {"messages": [{"role": "user", "content": EXPLAIN_PROMPT}]}),
        ]
        if "tools" in caps:
            plan.append(("tool", {"messages": [{"role": "user", "content": TOOL_PROMPT}],
                                  "tools": [WEATHER_TOOL], "tool_choice": "auto", "temperature": 0,
                                  "max_tokens": 256, "enable_thinking": False}))
        for kind, body in plan:
            swap_guard()
            payload = {"model": model_id, "max_tokens": args.max_tokens, "stream": False, **body}
            code, response, wall = http_json(base + "/v1/chat/completions", payload, timeout=args.request_timeout)
            entry = {"kind": kind, "http_status": code, "wall_seconds": round(wall, 3),
                     "request": {k: v for k, v in payload.items() if k != "messages"},
                     "message_count": len(payload["messages"])}
            if code == 400 and "enable_thinking" in payload and "thinking" in json.dumps(response).lower():
                entry["thinking_toggle_rejected"] = response
                payload.pop("enable_thinking")
                code, response, wall = http_json(base + "/v1/chat/completions", payload, timeout=args.request_timeout)
                entry.update(http_status=code, wall_seconds=round(wall, 3), retried_without_thinking_toggle=True)
            if code != 200 or not isinstance(response, dict):
                entry["error"] = response if isinstance(response, (dict, str)) else str(response)
                entry["sanity"] = {"sane": False, "issues": [f"HTTP {code}"]}
                requests_log.append(entry)
                if process.poll() is not None:
                    raise RuntimeError(f"server died during request {kind} rc={process.returncode}")
                continue
            entry["sanity"] = evaluate(kind, response)
            usage = response.get("usage") or {}
            block = response.get("mlx2")
            entry["usage"] = usage
            entry["mlx2"] = summarize_mlx2(block)
            completion = usage.get("completion_tokens") or (block or {}).get("completion_tokens") or 0
            ttft = (block or {}).get("ttft_seconds")
            elapsed = (block or {}).get("elapsed_seconds") or wall
            entry["ttft_seconds"] = ttft
            decode_window = (elapsed - ttft) if isinstance(ttft, (int, float)) else elapsed
            entry["decode_tokens_per_second"] = (
                round((completion - 1) / decode_window, 2) if completion > 1 and decode_window > 0 else None)
            if kind == "spomin_4k":
                entry["functional_oracle"] = spomin_oracle(response, oracle)
                entry["oracle_meta"] = oracle
            requests_log.append(entry)
        code, final_status, _ = http_json(base + "/v1/status", timeout=30)
        receipt["final_status_memory"] = find_keys(final_status, "peak") | find_keys(final_status, "memory")
        receipt["final_status_speculation"] = {
            k: v for k, v in (find_keys(final_status, "accept") | find_keys(final_status, "propos")).items()
        }
        receipt["status"] = "completed"
    except SwapRise as error:
        receipt["status"] = "aborted_swap"
        receipt["error"] = str(error)
    except Exception as error:  # noqa: BLE001 - every exit writes evidence
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        if sampler is not None:
            sampler.stop_event.set()
            receipt["peak_rss_bytes"] = sampler.peak_rss
            receipt["peak_phys_footprint_bytes"] = sampler.peak_footprint or None
        receipt["server_stop"] = stop_server(process)
        counters = vm_counters()
        receipt["host_after"] = {
            "available_bytes": available_bytes(counters),
            "swapouts": counters.get("Swapouts"),
            "swapouts_delta_pages": counters.get("Swapouts", 0) - swapouts_start,
            "swap_used_bytes": swap_used_bytes(),
        }
        receipt["requests"] = requests_log
        log_path = out / "server.log"
        if log_path.is_file():
            tail = log_path.read_text(errors="replace").splitlines()[-60:]
            receipt["server_log_tail"] = tail
        receipt["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt.update(verdict_for(receipt, route, expected_kind))
        atomic_json(out / "cell.json", receipt)
    return receipt


def artifact_offline(receipt: dict) -> bool:
    """The model directory vanished (e.g. an external volume unmounted)."""
    tail = "\n".join(receipt.get("server_log_tail") or ())
    model = receipt.get("model_path") or "\0"
    return "cannot inspect --model" in tail and f"No such file or directory: '{model}/config.json'" in tail


def effective_verdict(receipt: dict) -> str | None:
    """Verdict for readers of older receipts: an offline artifact is not a route failure."""
    if receipt.get("verdict") == "fail" and not receipt.get("load_ok") and artifact_offline(receipt):
        return "artifact_offline"
    return receipt.get("verdict")


def verdict_for(receipt: dict, route: dict, expected_kind: str) -> dict:
    failures, warnings = [], []
    expected_fail_closed = bool(route.get("expected", "").startswith("fail_closed"))
    if receipt.get("status") == "skipped":
        return {"verdict": "skipped"}
    if receipt.get("status") == "aborted_swap":
        return {"verdict": "aborted_swap", "failures": [receipt.get("error")]}
    if not receipt.get("load_ok"):
        tail = "\n".join(receipt.get("server_log_tail") or ())
        if artifact_offline(receipt):
            return {"verdict": "artifact_offline", "failures": [],
                    "reason": "artifact path missing at run time (volume not mounted)"}
        if expected_fail_closed and "does not declare prompt-lookup" in tail:
            return {"verdict": "fail_closed_expected", "failures": []}
        return {"verdict": "fail", "failures": ["load: " + str(receipt.get("load_error") or receipt.get("error"))]}
    if receipt.get("status") != "completed":
        failures.append(str(receipt.get("error")))
    routes_seen = set()
    speculation_seen = False
    unqualified_labels = set()
    for entry in receipt.get("requests", ()):
        sanity = entry.get("sanity") or {}
        if not sanity.get("sane"):
            failures.append(f"{entry['kind']}: {'; '.join(sanity.get('issues') or ['insane'])}")
        warnings.extend(f"{entry['kind']}: {w}" for w in sanity.get("warnings") or ())
        block = entry.get("mlx2") or {}
        if block.get("route"):
            routes_seen.add(block["route"])
        if block.get("route_receipt") is not None:
            unqualified_labels.add(json.dumps(block.get("route_receipt"))[:80])
        spec = block.get("speculation") or block.get("mtp")
        if isinstance(spec, dict) and any(isinstance(v, (int, float)) and v > 0 for k, v in spec.items()
                                          if k in {"proposed", "accepted", "cycles", "external_rounds",
                                                   "draft_tokens", "rounds"}):
            speculation_seen = True
        oracle = entry.get("functional_oracle")
        if oracle and not oracle.get("all_found"):
            warnings.append("spomin_4k: functional oracle incomplete " + json.dumps(oracle["needles_found"]))
    if expected_kind != "any" and routes_seen and expected_kind not in routes_seen:
        failures.append(f"route receipt {sorted(routes_seen)} != expected {expected_kind}")
    if expected_kind != "ordinary" and not speculation_seen:
        warnings.append("no speculation/MTP proposal observed in any response receipt")
    if expected_fail_closed:
        warnings.append("route expected to fail closed but loaded")
    verdict = "fail" if failures else ("pass_with_warnings" if warnings else "pass")
    return {"verdict": verdict, "failures": failures, "warnings": warnings,
            "routes_seen": sorted(routes_seen), "speculation_observed": speculation_seen,
            "route_receipt_labels": sorted(unqualified_labels)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--target", required=True, help="checkpoint, path, or artifact directory name")
    parser.add_argument("--routes", default="all", help="comma-separated route names or 'all'")
    parser.add_argument("--out-root", type=Path, default=ROOT / "qualification/runs/model-matrix-20261021")
    parser.add_argument("--host-label", default="m5")
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--cache-gib", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--load-timeout", type=float, default=1200)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--memory-factor", type=float, default=1.08)
    parser.add_argument("--max-model-bytes", type=float, default=0)
    parser.add_argument("--ignore-free-memory", action="store_true")
    parser.add_argument("--memory-settle-seconds", type=float, default=180.0,
                        help="wait up to this long for memory freed by the previous cell before skipping")
    parser.add_argument("--swap-pages-limit", type=int, default=2048)
    parser.add_argument("--suffix", default="", help="evidence directory suffix for a distinguishable retry")
    args = parser.parse_args(argv)
    inventory = json.loads(args.inventory.read_text())
    matches = [t for t in inventory["targets"] if args.target in {t.get("checkpoint"), t["path"], Path(t["path"]).name, t.get("name")}]
    if len(matches) != 1:
        parser.error(f"target {args.target!r} matched {len(matches)} inventory rows")
    target = matches[0]
    routes = target.get("routes") or []
    if args.routes != "all":
        wanted = args.routes.split(",")
        exact = {w for w in wanted if not w.endswith("*")}
        prefixes = tuple(w[:-1] for w in wanted if w.endswith("*"))
        routes = [r for r in routes if r["route"] in exact or (prefixes and r["route"].startswith(prefixes))]
        missing = exact - {r["route"] for r in routes}
        if missing and not routes:
            parser.error(f"routes not in inventory for this target: {sorted(missing)}")
        if missing:
            print(f"routes not in inventory for this target (skipped): {sorted(missing)}", file=sys.stderr)
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip() \
        if (ROOT / ".git").exists() else (ROOT / "SYNCED_HEAD").read_text().strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True).strip()) \
        if (ROOT / ".git").exists() else None
    artifact_dir = args.out_root / re.sub(r"[^A-Za-z0-9._-]+", "_", target["name"])
    results = []
    for route in routes:
        out = artifact_dir / (route["route"] + args.suffix)
        if out.exists() and any(out.iterdir()):
            print(f"refusing to overwrite nonempty evidence directory: {out}", file=sys.stderr)
            results.append({"route": route["route"], "verdict": "refused_existing_evidence"})
            continue
        receipt = run_cell(args, target, route, out, revision, dirty)
        print(json.dumps({"route": route["route"], "verdict": receipt.get("verdict"),
                          "failures": receipt.get("failures"), "warnings": receipt.get("warnings"),
                          "reason": receipt.get("reason")})[:2000], flush=True)
        results.append({"route": route["route"], "verdict": receipt.get("verdict")})
        if receipt.get("verdict") == "aborted_swap":
            break
    return 0 if all(r["verdict"] in {"pass", "pass_with_warnings", "fail_closed_expected", "skipped"} for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
