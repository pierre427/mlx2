"""End-to-end A/B: ordinary batched decode at 6 and 7 lanes on a dense affine-q8 model,
lane matmul q8 min_rows 8 (default crossover) vs 6. ABAB server arms, thermal settle
before each width, lane counters read around each width to assert engagement.

Usage: .venv/bin/python scripts/ab_lane_q8_min_rows_serving.py <mlx2 tree> <empty out dir> <dense q8 model dir>
(under the GPU lock; uses port 8399 and the primary checkout venv path below)
"""
import json, os, signal, subprocess, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

WT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); MODEL = sys.argv[3]
PY = str(Path.home() / "Desktop/mlx2/.venv/bin/python")
PORT = 8399
sys.path.insert(0, str(WT / "src"))
from mlx2.apple_telemetry import TemperatureSampler  # noqa: E402

ARMS = {"min8_default": None, "min6": '{"min_rows":{"q8":6}}'}
ORDER = ["min8_default", "min6", "min8_default", "min6"]
TOPICS = ["the history of the printing press", "how bridges handle wind loads",
          "the chemistry of sourdough bread", "why the sky changes colour at dusk",
          "the design of early mechanical clocks", "how coral reefs form over centuries",
          "the economics of container shipping"]
MAX_TOKENS = 192
temps = TemperatureSampler()


def die():
    vals = [v for k, v in temps.read().items() if "tdie" in k]
    return max(vals)


def settle(limit_c=58.0, cap_s=420):
    t0 = time.time(); start = die()
    while die() > limit_c and time.time() - t0 < cap_s:
        time.sleep(5)
    return dict(start_c=round(start, 1), end_c=round(die(), 1), waited_s=round(time.time() - t0, 1))


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=30) as r:
        return json.load(r)


def chat(prompt, record):
    body = json.dumps({"model": Path(MODEL).name, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAX_TOKENS, "min_tokens": MAX_TOKENS, "temperature": 0,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body,
                                 {"content-type": "application/json"})
    t0 = time.perf_counter(); first = None; text = []; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = ch.get("delta", {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text.append(piece)
    record.update(t0=t0, first=first, done=time.perf_counter(), text="".join(text), usage=usage)


def width_run(width):
    prompts = [f"Write a detailed, multi-paragraph essay about {TOPICS[i]}." for i in range(width)]
    recs = [{} for _ in prompts]
    th = [threading.Thread(target=chat, args=(p, r)) for p, r in zip(prompts, recs)]
    for t in th: t.start()
    for t in th: t.join()
    toks = sum(r["usage"]["completion_tokens"] for r in recs)
    last_first = max(r["first"] for r in recs); first_done = min(r["done"] for r in recs)
    # Window in which every lane is decoding: after the last first token, before the first finish.
    # Approximate: all lanes emit at the same step rate in ordinary batched decode.
    window = first_done - last_first
    per_lane = [r["usage"]["completion_tokens"] for r in recs]
    agg = toks / (max(r["done"] for r in recs) - min(r["t0"] for r in recs))
    steady = width * (min(per_lane) - 1) / (first_done - last_first) if window > 0 else None
    return dict(width=width, completion_tokens=per_lane, aggregate_tok_s=round(agg, 2),
                steady_all_lanes_tok_s=None if steady is None else round(steady, 2),
                texts=[r["text"] for r in recs])


def swapouts():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int([l for l in out.splitlines() if l.startswith("Swapouts")][0].split()[-1].rstrip("."))


def lane_counts():
    return get("/v1/status").get("lane_matmul", {}).get("counts", {})


results = []
for rep, arm in enumerate(ORDER):
    cmd = [PY, "-u", "-m", "mlx2.server", "--model", MODEL, "--host", "127.0.0.1", "--port", str(PORT),
           "--ordinary", "--max-lanes", "8", "--max-inflight", "8", "--default-max-tokens", "512",
           "--fabric", "off", "--cache-bytes", str(8 << 30)]
    if ARMS[arm]:
        cmd += ["--lane-policy", ARMS[arm]]
    log = open(OUT / f"server-{rep}-{arm}.log", "w")
    proc = subprocess.Popen(cmd, cwd=WT, env={**os.environ, "PYTHONPATH": str(WT / "src")},
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while True:
            try:
                get("/v1/models"); break
            except Exception:
                if proc.poll() is not None or time.time() - t0 > 900:
                    raise RuntimeError(f"server {arm} failed to start")
                time.sleep(3)
        status = get("/v1/status")
        lane = status.get("lane_matmul", {})
        rec = dict(rep=rep, arm=arm, lane_min_rows=lane.get("min_rows"), lane_law=lane.get("law_id"),
                   lane_installed=lane.get("installed"), model=status.get("model") or status.get("identity"),
                   widths=[])
        while True:  # /v1/models answers while the worker is still loading
            try:
                chat("Say hello in five words.", {}); break
            except urllib.error.HTTPError as exc:
                if exc.code != 503 or time.time() - t0 > 900 or proc.poll() is not None:
                    raise
                time.sleep(5)
        status = get("/v1/status")
        lane = status.get("lane_matmul", {})
        rec.update(lane_min_rows=lane.get("min_rows"), lane_law=lane.get("law_id"),
                   lane_installed=lane.get("installed"))
        for width in (6, 7):
            th = settle()
            before = lane_counts(); s0 = swapouts()
            res = width_run(width)
            after = lane_counts(); res["swapouts_delta_pages"] = swapouts() - s0
            res["lane_delta"] = {k: after.get(k, 0) - before.get(k, 0) for k in after
                                 if after.get(k, 0) != before.get(k, 0)}
            res["thermal"] = th
            rec["widths"].append(res)
            print(json.dumps({k: v for k, v in res.items() if k != "texts"} | {"arm": arm, "rep": rep}), flush=True)
        results.append(rec)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        log.close()
(OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
print("done")
