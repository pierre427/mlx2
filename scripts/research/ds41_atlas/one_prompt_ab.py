#!/usr/bin/env python3
"""One-prompt A/B for DS4.1 Source policies: one server per arm, greedy, receipts kept.

usage: one_prompt_ab.py MODEL OUT_DIR PROMPT_ID ARM=POLICY.json [ARM=POLICY.json ...] [--max-tokens 64]
PROMPT_ID names a prompt of the fixed 72-prompt set (ed_prompts.CATS, e.g. translate-6).
Refuses a nonempty OUT_DIR; stops each server's process group in finally.
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
sys.path.insert(0, "/Volumes/T7/kimi-k3.1-pro/expert-draft-linear-20261107")
from ed_prompts import CATS  # noqa: E402


def get(url, timeout=30):
    return json.loads(urllib.request.urlopen(url, timeout=timeout).read())


def write(path, data):
    tmp = Path(str(path) + ".tmp"); tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str)); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out", type=Path); ap.add_argument("prompt_id"); ap.add_argument("arms", nargs="+")
    ap.add_argument("--max-tokens", type=int, default=64); ap.add_argument("--port", type=int, default=8295)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f"refusing nonempty {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    cat, idx = a.prompt_id.rsplit("-", 1); text = CATS[cat][int(idx)]
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
    results = []
    for spec in a.arms:
        name, policy = spec.split("=", 1)
        d = a.out / name; d.mkdir()
        cmd = [sys.executable, "-m", "mlx2.server", "--model", a.model, "--port", str(a.port), "--ordinary",
               "--execution-policy", policy, "--max-context", "16384"]
        write(d / "source.json", {"mlx2_commit": rev, "tracked_dirty": dirty, "command": cmd, "prompt_id": a.prompt_id,
                                  "prompt": text, "policy": json.loads(Path(policy).read_text())})
        log = open(d / "server.log", "w")
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")})
        base = f"http://127.0.0.1:{a.port}"
        try:
            t0 = time.time()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"{name}: server exited {proc.returncode}")
                try:
                    get(base + "/health", 5); break
                except Exception:
                    time.sleep(5)
            model = get(base + "/v1/models")["data"][0]["id"]
            body = {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": a.max_tokens,
                    "temperature": 0, "logprobs": True, "top_logprobs": 2}
            req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            t = time.time(); resp = json.loads(urllib.request.urlopen(req, timeout=3600).read()); el = time.time() - t
            m = resp.get("mlx2") or {}
            row = {"arm": name, "load_s": round(t - t0, 1), "elapsed_s": round(el, 2),
                   "reply": resp["choices"][0]["message"].get("content"), "finish": resp["choices"][0].get("finish_reason"),
                   "usage": resp.get("usage"), "ttft_s": m.get("ttft_seconds"),
                   "decode_tok_s": round((m.get("completion_tokens", 1) - 1) / max(1e-9, m.get("elapsed_seconds", 0) - m.get("ttft_seconds", 0)), 3),
                   "logprobs": resp["choices"][0].get("logprobs")}
            write(d / "response.json", resp); write(d / "status-after.json", get(base + "/v1/status"))
            results.append(row); write(a.out / "results.json", results)
            print(json.dumps({k: v for k, v in row.items() if k != "logprobs"}, ensure_ascii=False), flush=True)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM); proc.wait(90)
            except Exception:
                try: os.killpg(proc.pid, signal.SIGKILL)
                except Exception: pass


if __name__ == "__main__":
    main()
