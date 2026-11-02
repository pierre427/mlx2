"""GLM-5.3 served decode: where the server's per-token host time goes.

Starts one loopback ``mlx2.server`` (ordinary route, the given execution
policy) through an in-process wrapper (``--serve``) that, only while a window
is open, samples every thread's Python stack every ``--sample-ms`` and times
``GenerationBatch.next`` (inside vs. between calls) plus the generation step's
own ``STEP_TRACE`` split. The client streams one request per arm on a
realistic ~``--tokens`` prompt and opens the window between delta 64 and delta
``max_tokens - 64``, so prefill and the first/last tokens stay outside it.

Per window the wrapper writes ``<arm>.profile.json``: gen-thread inclusive and
leaf attribution (ms/token), per-thread CPU (``ps -M``) deltas, ``next()``
in/out split, step-trace split, and the model's ``calls`` counter deltas
(fast-path engagement). Research only; nothing in serving changes. Run under
the GPU lock recipe (bench.sh).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

MODEL = "~/mlx-models/GLM-5.3-Flash-REAP50-Q3_K_M"
CHECKPOINT = ("GLM-5.3-Flash-REAP50-Q3_K_M.gguf@"
              "5a865918aea87a362ff8ae419581a0447929bfd68534a44d1125b394c272762d")
ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- wrapper (server side)

def _thread_cpu():
    """{thread index: cpu seconds} from ``ps -M`` for this process."""
    try:
        out = subprocess.run(["ps", "-M", "-p", str(os.getpid())], capture_output=True, text=True,
                             timeout=5).stdout.splitlines()[1:]
    except Exception:  # noqa: BLE001
        return {}
    rows = {}
    for i, line in enumerate(out):
        parts = line.split()
        # USER PID TT %CPU STAT PRI STIME UTIME COMMAND (first row), later rows omit PID/TT/COMMAND
        times = [p for p in parts if ":" in p and p.replace(":", "").replace(".", "").isdigit()]
        if len(times) >= 2:
            tot = 0.0
            for t in times[-2:]:
                m, s = t.split(":")
                tot += int(m) * 60 + float(s)
            rows[i] = tot
    return rows


class Window:
    def __init__(self, name, sample_s):
        self.name, self.sample_s = name, sample_s
        self.samples = collections.Counter()      # (thread name, folded stack) -> count
        self.thread_samples = collections.Counter()
        self.ticks = 0
        self.next_spans = []                       # (enter, exit)
        self.t0 = time.perf_counter()
        self.cpu0 = _thread_cpu()


def serve(argv):
    import mlx2.runtime.generate as gen

    control = Path(os.environ["GLM53_PROFILE_CONTROL"])
    sample_s = float(os.environ.get("GLM53_PROFILE_SAMPLE_MS", "2")) / 1e3
    state = {"window": None}
    gen.STEP_TRACE = []
    original_next = gen.GenerationBatch.next

    def timed_next(self):
        w = state["window"]
        if w is None:
            return original_next(self)
        t0 = time.perf_counter()
        try:
            return original_next(self)
        finally:
            w.next_spans.append((t0, time.perf_counter()))

    gen.GenerationBatch.next = timed_next

    def adapter_calls():
        import gc

        for obj in gc.get_objects():
            if type(obj).__name__ == "Glm53GGUFAdapter":
                try:
                    calls = dict(obj.model.calls)
                    # Module-level counters of the compiled decode regions.
                    from mlx2.runtime.models import glm53_compile_regions

                    calls.update({"compile_regions." + k: v for k, v in glm53_compile_regions._COUNTERS.items()})
                    return calls, {"lean": getattr(obj.model, "_mlx2_lean_greedy", None),
                                                   "lean_steps": getattr(obj.model, "_mlx2_lean_greedy_scheduled_steps", None)}
                except Exception as error:  # noqa: BLE001
                    return {"error": repr(error)}, {}
        return {}, {}

    def finish(w):
        elapsed = time.perf_counter() - w.t0
        cpu1 = _thread_cpu()
        names = {t.ident: t.name for t in threading.enumerate()}
        gen_samples = {k: v for k, v in w.samples.items() if k[0] == "mlx2-generation"}
        total_gen = sum(gen_samples.values()) or 1
        incl, leaf = collections.Counter(), collections.Counter()
        for (_, stack), count in gen_samples.items():
            frames = stack.split(";")
            for f in set(frames):
                incl[f] += count
            leaf[frames[-1]] += count
        spans = w.next_spans
        tokens = len(spans)
        inside = [b - a for a, b in spans]
        between = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        trace = gen.STEP_TRACE[w.trace0:]
        calls1, lean1 = adapter_calls()
        per_tok = lambda count: (count / total_gen) * elapsed * 1e3 / max(tokens, 1)  # noqa: E731

        def p50(v):
            v = sorted(v)
            return v[len(v) // 2] * 1e3 if v else None

        def mean(v):
            return sum(v) / len(v) * 1e3 if v else None
        record = {
            "window": w.name, "elapsed_s": elapsed, "tokens": tokens, "ticks": w.ticks,
            "ms_per_token_wall": elapsed * 1e3 / max(tokens, 1),
            "next_inside_p50_ms": p50(inside), "next_inside_mean_ms": mean(inside),
            "between_next_p50_ms": p50(between), "between_next_mean_ms": mean(between),
            "step_trace": {"n": len(trace),
                           "check_p50_ms": p50([t[1] / 1e3 for t in trace]),
                           "build_p50_ms": p50([t[2] / 1e3 for t in trace]),
                           "wait_p50_ms": p50([t[3] / 1e3 for t in trace]),
                           "forward_py_p50_ms": p50([t[4] / 1e3 for t in trace]),
                           "modes": dict(collections.Counter(t[0] for t in trace))},
            "thread_samples": dict(w.thread_samples),
            "thread_cpu_delta_s": {str(k): round(cpu1.get(k, 0) - v, 3) for k, v in w.cpu0.items()},
            "thread_names_now": sorted(set(names.values())),
            "gen_inclusive_ms_per_token": {f: round(per_tok(c), 3) for f, c in incl.most_common(80)},
            "gen_leaf_ms_per_token": {f: round(per_tok(c), 3) for f, c in leaf.most_common(60)},
            "gen_folded_top": [[s, c] for (_, s), c in sorted(gen_samples.items(), key=lambda kv: -kv[1])[:150]],
            "other_threads_leaf": [[t, s.split(";")[-1], c] for (t, s), c in w.samples.most_common(200)
                                   if t != "mlx2-generation"][:40],
            "model_calls_delta": {k: v - w.calls0.get(k, 0) for k, v in calls1.items()
                                  if isinstance(v, (int, float)) and v != w.calls0.get(k, 0)},
            "lean_greedy": {"start": w.lean0, "end": lean1},
            "switchinterval": sys.getswitchinterval(),
            "memory": {"start": w.mem0, "end": {"active_gib": mx.get_active_memory() / 2**30,
                                                 "cache_gib": mx.get_cache_memory() / 2**30,
                                                 "peak_gib": mx.get_peak_memory() / 2**30}},
        }
        (control / f"{w.name}.profile.json").write_text(json.dumps(record, indent=1) + "\n")

    import mlx.core as mx

    limits = {}

    def apply_setting(name):
        """Research ablations, applied between requests (never mid-window)."""
        if name == "memlimit_off":
            # MLX's own default limit (1.5x the recommended working set), as the
            # bare-model replay runs; the server lowers it to advisory - reserve.
            limits.setdefault("server", mx.set_memory_limit(int(1.5 * mx.device_info()["max_recommended_working_set_size"])))
        elif name == "memlimit_on":
            if "server" in limits:
                mx.set_memory_limit(limits.pop("server"))
        elif name in ("lean_on", "lean_off"):
            import gc

            for obj in gc.get_objects():
                if type(obj).__name__ == "Glm53GGUFAdapter":
                    obj.model._mlx2_lean_greedy = name == "lean_on"
        elif name == "clear_cache":
            mx.clear_cache()
        else:
            raise ValueError(name)
        return {"setting": name, "saved_server_limit_gib": (limits["server"] / 2**30) if "server" in limits else None,
                "active_gib": mx.get_active_memory() / 2**30, "cache_gib": mx.get_cache_memory() / 2**30}

    def sampler():
        me = threading.get_ident()
        while True:
            w = state["window"]
            if w is None:
                time.sleep(0.01)
                for req in control.glob("set-*"):
                    name = req.name.removeprefix("set-")
                    req.unlink()
                    try:
                        ack = apply_setting(name)
                    except Exception as error:  # noqa: BLE001
                        ack = {"setting": name, "error": repr(error)}
                    (control / f"ack-{name}.json").write_text(json.dumps(ack))
                starts = list(control.glob("start-*"))
                if starts:
                    name = starts[0].name.removeprefix("start-")
                    starts[0].unlink()
                    w = Window(name, sample_s)
                    w.trace0 = len(gen.STEP_TRACE)
                    w.calls0, w.lean0 = adapter_calls()
                    w.mem0 = {"active_gib": mx.get_active_memory() / 2**30, "cache_gib": mx.get_cache_memory() / 2**30}
                    state["window"] = w
                continue
            stop = control / f"stop-{w.name}"
            if stop.exists():
                stop.unlink()
                state["window"] = None
                try:
                    finish(w)
                except Exception as error:  # noqa: BLE001
                    (control / f"{w.name}.profile-error.txt").write_text(repr(error))
                continue
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                if ident == me:
                    continue
                stack = []
                f = frame
                while f is not None and len(stack) < 60:
                    code = f.f_code
                    stack.append(f"{code.co_name}@{Path(code.co_filename).name}:{f.f_lineno}")
                    f = f.f_back
                tname = names.get(ident, str(ident))
                w.samples[(tname, ";".join(reversed(stack)))] += 1
                w.thread_samples[tname] += 1
            w.ticks += 1
            time.sleep(w.sample_s)

    threading.Thread(target=sampler, name="glm53-profiler", daemon=True).start()
    import runpy

    sys.argv = ["mlx2.server", *argv]
    runpy.run_module("mlx2.server", run_name="__main__", alter_sys=True)


# --------------------------------------------------------------------------- client

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(url, path, body=None, headers=None, timeout=1800):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_file(path, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def stream(url, body, control, name, lo, hi):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps, final, text, opened = [], None, [], None
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            choices = event.get("choices") or []
            content = choices and (choices[0].get("delta") or {}).get("content")
            if content:
                stamps.append(time.monotonic() - t0)
                text.append(content)
                if len(stamps) == lo:
                    (control / f"start-{name}").touch()
                    opened = len(stamps)
                elif len(stamps) == hi and opened:
                    (control / f"stop-{name}").touch()
                    opened = None
            if event.get("mlx2"):
                final = event["mlx2"]
    if opened:
        (control / f"stop-{name}").touch()
    return stamps, final, "".join(text)


def nonstream(url, body, control, name):
    # The window spans the whole request (prefill included): use the profile's
    # next() spans, not ms_per_token_wall, for this arm.
    (control / f"start-{name}").touch()
    t0 = time.monotonic()
    out = call(url, "/v1/chat/completions", body)
    elapsed = time.monotonic() - t0
    (control / f"stop-{name}").touch()
    return elapsed, out


def gaps_summary(stamps, lo, hi):
    gaps = [b - a for a, b in zip(stamps, stamps[1:])][lo:hi]
    gaps.sort()
    return {"n": len(gaps), "p50_ms": gaps[len(gaps) // 2] * 1e3 if gaps else None,
            "mean_ms": sum(gaps) / len(gaps) * 1e3 if gaps else None}


QUESTION = "\n\nSummarize the material above in detail, section by section."


def serve_text(tokens):
    """The user message every profiled arm sends (also rendered by the replay record)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import realistic_haystack

    chars = int(tokens * 4.2)
    text = realistic_haystack.haystack(chars * 8)
    return text, chars


def serve_messages(tokens):
    text, chars = serve_text(tokens)
    return [{"role": "user", "content": text[:chars] + QUESTION}]


def run_replay(args, label, record, trace=None):
    """Bare-model record (first call) and pipelined replay of the served prompt, same policy and env.
    ``trace``: replay that existing recording instead (identity rebound; tokens still checked)."""
    out = args.out / label
    extra = ["--rebind-identity"] if trace is not None else []
    trace = trace or args.out / "replay-trace"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **json.loads(args.server_env)}
    here = Path(__file__).resolve().parent
    if record:
        cmd = [sys.executable, str(here / "glm53_decode_replay.py"), "--out", str(trace),
               "--source-revision", args.source_revision, "record", "--policy", str(args.policy),
               "--tokens", str(args.tokens), "--steps", str(args.max_tokens), "--serve-prompt"]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        (args.out / "replay-record.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            return {"label": "record", "rc": proc.returncode}
    cmd = [sys.executable, str(here / "glm53_decode_replay.py"), "--out", str(out),
           "--source-revision", args.source_revision, "replay", "--policy", str(args.policy),
           "--trace", str(trace), "--steps", str(args.max_tokens), *extra, "--arms", "pipelined"]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    (args.out / f"{label}.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        return {"label": label, "rc": proc.returncode}
    rep = json.loads((out / "replay.json").read_text())
    arm = rep["arms"][0]
    gaps = [b + w for b, w in zip(arm["builds_ms"], arm["waits_ms"])][64:-64]
    gaps.sort()
    trace_json = json.loads((trace / "trace.json").read_text())
    return {"label": label, "tokens_match": arm["tokens_match"], "step_p50_ms": arm["step_p50_ms"],
            "build_p50_ms": arm["build_p50_ms"], "wait_p50_ms": arm["wait_p50_ms"],
            "graph_p50_ms": arm["graph_p50_ms"], "submit_p50_ms": arm["submit_p50_ms"],
            "window_p50_ms": gaps[len(gaps) // 2] if gaps else None,
            "text_sha256": trace_json.get("text_sha256"), "prompt_tokens": trace_json["prompt_tokens"],
            "trace": str(trace)}


ARMS = {
    # name: (request overrides, streamed)
    "greedy_stream": ({}, True),
    "greedy_nostream": ({}, False),
    "greedy_logprobs_stream": ({"logprobs": True, "top_logprobs": 1}, True),
    "temp07_stream": ({"temperature": 0.7, "seed": 7}, True),
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        return serve(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sample-ms", type=float, default=2.0)
    parser.add_argument("--arms", nargs="+", default=["greedy_stream", "greedy_nostream"])
    parser.add_argument("--no-profile-arm", action="store_true",
                        help="first run greedy_stream with no sampling window, as the unperturbed control")
    parser.add_argument("--server-env", default="{}", help="JSON env overrides for the server process")
    parser.add_argument("--replay-around", action="store_true",
                        help="record the served prompt bare-model and replay it before and after the server")
    parser.add_argument("--extra-trace", type=Path, default=None,
                        help="with --replay-around: also replay this recording before and after the server")
    parser.add_argument("--source-revision", default="unrecorded")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    control = args.out / "control"
    control.mkdir()
    for key in ("admin-token", "reasoning-key"):
        (args.out / key).write_text(secrets.token_hex(32))
        (args.out / key).chmod(0o600)
    replays = []
    if args.replay_around:
        replays.append(run_replay(args, "replay-before", record=True))
        print(json.dumps(replays[-1]), flush=True)
        if args.extra_trace:
            replays.append(run_replay(args, "extra-replay-before", record=False, trace=args.extra_trace))
            print(json.dumps(replays[-1]), flush=True)
    port = free_port()
    server_args = ["--model", MODEL, "--host", "127.0.0.1", "--port", str(port),
                   "--fabric", "off", "--max-context", str(args.tokens + args.max_tokens + 4096), "--max-lanes", "1",
                   "--max-inflight", "1", "--prefill-step", "2048", "--cache-bytes", str(4 << 30),
                   "--cache-dir", str(args.out / "cache"), "--execution-policy", str(args.policy),
                   "--admin-token-file", str(args.out / "admin-token"),
                   "--reasoning-signing-key-file", str(args.out / "reasoning-key"), "--ordinary"]
    argv = [sys.executable, "-u", str(Path(__file__).resolve()), "--serve", *server_args]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "GLM53_PROFILE_CONTROL": str(control),
           "GLM53_PROFILE_SAMPLE_MS": str(args.sample_ms), **json.loads(args.server_env)}
    log = (args.out / "server.log").open("w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    url = f"http://127.0.0.1:{port}"
    admin = {"Authorization": "Bearer " + (args.out / "admin-token").read_text()}
    record = {"schema": "mlx2.glm53-serve-profile.v1", "replays": replays, "checkpoint": CHECKPOINT,
              "source_revision": args.source_revision, "tokens": args.tokens, "max_tokens": args.max_tokens,
              "policy": json.loads(args.policy.read_text()), "server_env": json.loads(args.server_env),
              "sample_ms": args.sample_ms, "argv": argv, "arms": []}
    try:
        deadline = time.time() + 900
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"server exited {proc.returncode}")
            try:
                model = call(url, "/v1/status", timeout=5)["model"]
                break
            except Exception:  # noqa: BLE001 -- not ready yet
                if time.time() > deadline:
                    raise
                time.sleep(3)
        text, chars = serve_text(args.tokens)
        question = QUESTION
        warm = {"model": model, "messages": [{"role": "user", "content": text[-4000:] + question}],
                "max_tokens": 16, "temperature": 0, "enable_thinking": False}
        call(url, "/v1/chat/completions", warm)
        lo, hi = 64, args.max_tokens - 64
        arms = (["control_greedy_stream"] if args.no_profile_arm else []) + list(args.arms)
        for i, arm in enumerate(arms):
            try:
                call(url, "/v1/admin/apc/clear", {}, headers=admin)
            except Exception as error:  # noqa: BLE001
                record.setdefault("apc_clear_errors", []).append(repr(error))
            base, _, setting = arm.removeprefix("control_").partition("@")
            overrides, streamed = ARMS[base]
            settings = setting.split("+") if setting else []
            acks = []
            for name in settings:
                (control / f"set-{name}").touch()
                ack = control / f"ack-{name}.json"
                if not wait_file(ack):
                    raise RuntimeError(f"setting {name} not acknowledged")
                acks.append(json.loads(ack.read_text()))
                ack.unlink()
                if "error" in acks[-1]:
                    raise RuntimeError(f"setting {name} failed: {acks[-1]['error']}")
            # The same prompt for every arm: each arm decodes the same tokens (greedy arms).
            body = {"model": model, "messages": [{"role": "user", "content": text[:chars] + question}],
                    "max_tokens": args.max_tokens, "temperature": 0, "enable_thinking": False,
                    **overrides}
            name = f"{i}-{arm}"
            row = {"arm": arm, "name": name, "settings": acks}
            if streamed:
                body.update({"stream": True, "stream_options": {"include_usage": True}})
                if arm.startswith("control_"):
                    stamps, final, out_text = stream(url, body, control, name, 10**9, 10**9)
                else:
                    stamps, final, out_text = stream(url, body, control, name, lo, hi)
                # decode_tok_s counts SSE content deltas (about one per token, not
                # exactly); the profile's next() spans are the per-token figures.
                row.update({"deltas": len(stamps), "ttft_s": stamps[0] if stamps else None,
                            "decode_tok_s": (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None,
                            "window_gaps": gaps_summary(stamps, lo, hi), "text_sha256": hashlib.sha256(out_text.encode()).hexdigest(),
                            "text_head": out_text[:200], "receipt_usage": (final or {}).get("usage")})
            else:
                elapsed, out = nonstream(url, body, control, name)
                row.update({"elapsed_s": elapsed, "usage": out.get("usage"),
                            "text_sha256": hashlib.sha256((out["choices"][0]["message"].get("content") or "").encode()).hexdigest(),
                            "text_head": (out["choices"][0]["message"].get("content") or "")[:200]})
            if not arm.startswith("control_"):
                prof = control / f"{name}.profile.json"
                if wait_file(prof):
                    row["profile"] = json.loads(prof.read_text())
            record["arms"].append(row)
            for name_ in settings:
                undo = {"memlimit_off": "memlimit_on", "lean_on": "lean_off"}.get(name_)
                if undo:
                    (control / f"set-{undo}").touch()
                    ack = control / f"ack-{undo}.json"
                    if not wait_file(ack) or "error" in json.loads(ack.read_text()):
                        raise RuntimeError(f"restoring {undo} failed")
                    ack.unlink()
            brief = {k: row.get(k) for k in ("arm", "deltas", "decode_tok_s", "window_gaps", "elapsed_s")}
            p = row.get("profile") or {}
            brief.update({k: p.get(k) for k in ("ms_per_token_wall", "next_inside_p50_ms", "between_next_p50_ms")})
            brief["step_trace"] = p.get("step_trace")
            print(json.dumps(brief), flush=True)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        log.close()
        if args.replay_around:
            if args.extra_trace:
                replays.append(run_replay(args, "extra-replay-after", record=False, trace=args.extra_trace))
                print(json.dumps(replays[-1]), flush=True)
            replays.append(run_replay(args, "replay-after", record=False))
            print(json.dumps(replays[-1]), flush=True)
        record["replays"] = replays
        (args.out / "serve-profile.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
