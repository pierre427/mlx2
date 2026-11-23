"""Inkling 2: APCv2 across chat turns and across a server restart.

kapitsa found that Inkling's chat template renders the same text with and
without the generation prompt, so their engine found no turn boundary and
re-read the whole conversation on every follow-up turn (5.4K tokens: 79 s
against 7.5 s). This probe checks mlx2 does not do that, including when the
sliding-window rings and the conv state must be restored from disk.

One server (``--max-lanes 1``, a private ``--apc-persist-dir``,
``--apc-persist-on-shutdown``), greedy, then:

1. **Turns 1-4.** Each request carries the whole history (a ~4K-token
   document in turn 1, the model's earlier replies, a new question). It
   records ``prompt_tokens``, ``cached_tokens``, TTFT and the reply ids.
   ``cached_tokens`` of turn k should cover the turn k-1 prompt plus most of
   its reply.
2. **Turn 4 again.** The same request warm. Its reply ids must equal the
   first turn-4 reply (width-one cold/warm exactness).
3. **Restart.** The server process group is stopped (SIGTERM, persisting on
   shutdown), the persist directory is listed, and a new server on the same
   directory gets turn 4 a third time. ``cached_tokens`` is the restore from
   disk, and the reply ids must equal the first turn-4 reply.

``--route dspark`` runs the same probe on the DSpark external draft, which
also checks that the drafter's state survives the restore.

This is a sanity probe with no thermal hold, not publishable statistics.

Usage: python scripts/research/inkling2_multiturn_apc_probe.py --model <dir> --out-dir <empty dir>
       [--route ordinary|dspark] [--port 8397]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import hy4_context_ladder as H
from inkling2_context_ladder import DSPARK_POLICY, stream_one
from realistic_haystack import haystack

QUESTIONS = (
    ("Using the reference material above, name three things mlx2's prefix cache keeps per "
     "request, and say in one sentence why each matters."),
    "Which of those would break first if the server restarted mid-conversation? Answer in two sentences.",
    "Suggest one test that would catch that failure, as a short numbered procedure.",
    "Summarize our conversation so far in at most four bullet points.",
)
SYSTEM = "You are a careful assistant. Answer from the reference material when it is relevant."


def server_command(a, token_file, persist, policy_file):
    cmd = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(a.model),
           "--host", "127.0.0.1", "--port", str(a.port), "--max-lanes", "1", "--max-inflight", "1",
           "--max-context", "16384", "--admin-token-file", str(token_file),
           "--apc-persist-dir", str(persist), "--apc-persist-on-shutdown"]
    if a.route == "ordinary":
        cmd.append("--ordinary")
    else:
        cmd += ["--external-draft", "--execution-policy", str(policy_file)]
    return cmd


def port_free(port):
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def start(a, cmd, log_path):
    if not port_free(a.port):
        raise SystemExit(f"port {a.port} is already in use; refusing to measure another server")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1"}
    log = open(log_path, "w")  # noqa: SIM115 - closed by the caller
    server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True)
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


def ask(base, messages, a):
    # No "model" key: the server accepts only its own directory name there.
    # logprobs carries the token ids (parse_stream builds token_ids from it).
    body = {"messages": messages, "max_tokens": a.max_tokens, "temperature": 0,
            "seed": 7, "reasoning_effort": a.reasoning_effort, "stream": True, "logprobs": True}
    r = stream_one(base, body, timeout=900)
    receipt = r.get("receipt") or {}
    spec = H.find_key(receipt, "speculation") if a.route == "dspark" else None
    return {"prompt_tokens": receipt.get("prompt_tokens"), "cached_tokens": receipt.get("cached_tokens"),
            "ttft_s": r.get("ttft_s"), "decode_tok_s": r.get("decode_tok_s"),
            "finish_reason": r.get("finish_reason"), "route": receipt.get("route"),
            "token_ids": r.get("token_ids"), "text": r.get("text"),
            "reasoning_text_chars": len(r.get("reasoning_text") or ""), "speculation": spec}


def listing(path):
    files = [p for p in path.rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--route", choices=("ordinary", "dspark"), default="ordinary")
    parser.add_argument("--port", type=int, default=8397)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--doc-chars", type=int, default=16000)
    parser.add_argument("--load-timeout", type=float, default=600)
    a = parser.parse_args()
    if a.port == 8282:
        raise SystemExit("port 8282 is reserved for production release trees")
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    persist = out / "apc"
    persist.mkdir(mode=0o700)
    token_file = out / "admin-token"
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(fd, os.urandom(24).hex().encode() + b"\n")
    os.close(fd)
    policy_file = out / "policy.json"
    policy_file.write_text(json.dumps(DSPARK_POLICY) + "\n")
    report = {"script": "scripts/research/inkling2_multiturn_apc_probe.py", "route": a.route,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "kind": "sanity probe (no thermal hold; not publishable statistics)",
              "max_tokens": a.max_tokens, "reasoning_effort": a.reasoning_effort, "turns": []}
    try:
        report["mlx2"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        pass
    path = out / "multiturn_apc.json"

    def save():
        tmp = path.with_suffix(".partial")
        tmp.write_text(json.dumps(report, indent=1, default=str))
        os.replace(tmp, path)

    doc = haystack(a.doc_chars, root=ROOT)
    base = f"http://127.0.0.1:{a.port}"
    cmd = server_command(a, token_file, persist, policy_file)
    report["server_command"] = cmd
    server, log, load_s = start(a, cmd, out / "server-1.log")
    report["load_seconds"] = [load_s]
    messages = [{"role": "system", "content": SYSTEM}]
    try:
        report["health_start"] = H.get_json(base, "/health")
        for i, question in enumerate(QUESTIONS):
            content = f"Reference material:\n\n{doc}\n\n{question}" if i == 0 else question
            messages.append({"role": "user", "content": content})
            turn = ask(base, messages, a)
            turn["turn"] = i + 1
            report["turns"].append(turn)
            print(f"turn {i + 1}: prompt {turn['prompt_tokens']} cached {turn['cached_tokens']} "
                  f"ttft {turn['ttft_s']}", flush=True)
            save()
            if i < len(QUESTIONS) - 1:
                messages.append({"role": "assistant", "content": turn["text"]})
        last = messages
        warm = ask(base, last, a)
        warm["turn"] = "4-warm"
        warm["same_as_turn4"] = warm["token_ids"] == report["turns"][-1]["token_ids"]
        report["turns"].append(warm)
        print(f"turn 4 warm: cached {warm['cached_tokens']} same {warm['same_as_turn4']}", flush=True)
        save()
    finally:
        report["stop_1"] = H.stop_group(server)
        log.close()
        if report["stop_1"] != "stopped":
            token_file.unlink(missing_ok=True)
            save()
            raise SystemExit(f"server process group {server.pid} did not stop")
    report["persist_after_stop"] = listing(persist)
    save()
    if not report["persist_after_stop"]["files"]:
        print("warning: the persist directory is empty after shutdown", flush=True)
    try:
        server, log, load_s = start(a, cmd, out / "server-2.log")
    except SystemExit:
        token_file.unlink(missing_ok=True)
        raise
    report["load_seconds"].append(load_s)
    try:
        restored = ask(base, last, a)
        restored["turn"] = "4-after-restart"
        restored["same_as_turn4"] = restored["token_ids"] == report["turns"][3]["token_ids"]
        report["turns"].append(restored)
        print(f"turn 4 after restart: cached {restored['cached_tokens']} ttft {restored['ttft_s']} "
              f"same {restored['same_as_turn4']}", flush=True)
    finally:
        report["stop_2"] = H.stop_group(server)
        log.close()
        token_file.unlink(missing_ok=True)
    turns = report["turns"]
    follow = turns[1:4]
    if any(not t["token_ids"] for t in turns):
        raise SystemExit("a turn returned no token ids; the exactness checks would be vacuous")
    report["verdict"] = {
        "server_stops": [report["stop_1"], report["stop_2"]],
        "followups_hit_cache": all((t["cached_tokens"] or 0) >= 0.5 * (t["prompt_tokens"] or 1)
                                   for t in follow),
        "followup_cached_fraction": [round((t["cached_tokens"] or 0) / (t["prompt_tokens"] or 1), 3)
                                     for t in follow],
        "warm_exact": turns[4]["same_as_turn4"],
        "restart_hit_cache": (turns[5]["cached_tokens"] or 0) >= 0.5 * (turns[5]["prompt_tokens"] or 1),
        "restart_exact": turns[5]["same_as_turn4"],
        "ttft_s": {"turn4_cold": turns[3]["ttft_s"], "turn4_warm": turns[4]["ttft_s"],
                   "turn4_restart": turns[5]["ttft_s"]},
    }
    for t in turns:
        t.pop("text", None)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    print(json.dumps(report["verdict"]), flush=True)


if __name__ == "__main__":
    main()
