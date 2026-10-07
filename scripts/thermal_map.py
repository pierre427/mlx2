#!/usr/bin/env python3
"""Map thermal throttling across serving work types and routes, at equal heat.

All arms (server configurations) are loaded at once, each in its own
process, and driven in strict alternation, so every route sees the same
thermal history.  Phases:

1. cold: wait until the die cools to ``cold_c`` (or ``cold_wait_s`` passes);
2. hot: back-to-back cycles for ``hot_s``; every cycle runs, per arm, a
   fresh-prefix prefill (``prefill_tokens``, one output token), a B1 decode
   and a B4 decode;
3. cooldown: idle for ``cooldown_s`` (recovery time constants);
4. duty: the same cycles with ``gap_s`` idle after each segment.

Every segment is bracketed by IOReport energy reads (exact joules, GPU
DVFS residency) and a die-temperature read; a 2 Hz background series of the
same counters is written alongside for plotting.  Output: ``segments.jsonl``,
``series.jsonl`` and ``summary.json`` (cold vs hot rate, J/token, power and
P-state per arm and segment, plus throttle onset).

  scripts/run_with_gpu_locks.py --session S --label thermal --receipt r.json -- \\
    .venv/bin/python scripts/thermal_map.py --spec spec.json --out DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mlx2.apple_telemetry import EnergySampler, TemperatureSampler  # noqa: E402
from profile_features import SHORT_PROMPTS, Server, concurrent, document, user  # noqa: E402


class Series(threading.Thread):
    def __init__(self, path, hz=2.0):
        super().__init__(daemon=True)
        self.path, self.dt = path, 1.0 / hz
        self.energy, self.temps = EnergySampler(), TemperatureSampler()
        self.stop = threading.Event()
        self.label = "init"

    def run(self):
        with open(self.path, "w") as f:
            while not self.stop.wait(self.dt):
                e = self.energy.read()
                row = {"t": time.time(), "label": self.label, **e.as_dict(),
                       **self.temps.die_summary()}
                row.pop("gpu_states", None)
                f.write(json.dumps(row) + "\n")
                f.flush()


def segment(name, arm, fn, energy, temps, series, out, phase, cycle):
    series.label = f"{phase}:{arm}:{name}"
    energy.read()
    t0 = time.time()
    tokens, extra = fn()
    e = energy.read()
    joules = ((e.gpu_watts or 0) + (e.dram_watts or 0)) * e.seconds
    row = {"t": t0, "phase": phase, "cycle": cycle, "arm": arm, "segment": name,
           "wall_s": e.seconds, "tokens": tokens, "rate": tokens / e.seconds if e.seconds else None,
           "joules": joules, "j_per_token": joules / tokens if tokens else None,
           **{k: v for k, v in e.as_dict().items() if k != "gpu_states"},
           **temps.die_summary(), **extra}
    out.write(json.dumps(row) + "\n")
    out.flush()
    return row


def make_segments(client, spec, cycle_index):
    pt = int(spec.get("prefill_tokens", 8000))
    n1 = int(spec.get("decode_tokens", 256))
    n4 = int(spec.get("batch_tokens", 128))
    width = int(spec.get("batch_width", 4))

    def prefill():
        doc = document(pt, 1000 + cycle_index * 17)
        r = client.chat(user(f"[{cycle_index}] Document:\n\n" + doc + "\n\nTitle?"), max_tokens=1)
        return (r["prompt_tokens"] or 0) - (r["cached_tokens"] or 0), {
            "ttft_s": r["ttft_s"], "cached": r["cached_tokens"], "error": r["error"]}

    def decode_b1():
        r = client.chat(user(SHORT_PROMPTS[cycle_index % len(SHORT_PROMPTS)]
                             + " Write about 300 words."), max_tokens=n1)
        return r["completion_tokens"] or 0, {"decode_tps": r["decode_tps"], "error": r["error"]}

    def decode_batch():
        jobs = [dict(messages=user(SHORT_PROMPTS[(cycle_index + i) % len(SHORT_PROMPTS)]
                                   + f" (variant {i}) Write about 200 words."), max_tokens=n4)
                for i in range(width)]
        res = concurrent(client, jobs)
        return sum(r["completion_tokens"] or 0 for r in res), {
            "errors": [r["error"] for r in res if r["error"]]}

    return [("prefill", prefill), ("decode_b1", decode_b1), (f"decode_b{width}", decode_batch)]


def run_cycles(phase, until, servers, spec, energy, temps, series, out, start_cycle, gap):
    cycle = start_cycle
    names = list(servers)
    while time.time() < until:
        shift = cycle % len(names)
        for arm in names[shift:] + names[:shift]:
            for name, fn in make_segments(servers[arm].client, spec, cycle):
                segment(name, arm, fn, energy, temps, series, out, phase, cycle)
                if gap:
                    series.label = f"{phase}:gap"
                    time.sleep(gap)
        cycle += 1
    return cycle


def summarize(rows, spec):
    out = {}
    n = int(spec.get("summary_cycles", 3))
    for arm in sorted({r["arm"] for r in rows}):
        for seg in sorted({r["segment"] for r in rows}):
            hot = [r for r in rows if r["arm"] == arm and r["segment"] == seg and r["phase"] == "hot"]
            duty = [r for r in rows if r["arm"] == arm and r["segment"] == seg and r["phase"] == "duty"]
            if not hot:
                continue
            cold, steady = hot[:n], hot[-n:]

            def med(rs, k):
                v = [r[k] for r in rs if r.get(k) is not None]
                return statistics.median(v) if v else None

            entry = {}
            for label, rs in (("cold", cold), ("hot", steady), ("duty", duty[-n:] if duty else [])):
                entry[label] = {k: med(rs, k) for k in
                                ("rate", "j_per_token", "gpu_w", "dram_w", "gpu_mean_pstate",
                                 "gpu_active", "die_max_c")}
            c, h = entry["cold"]["rate"], entry["hot"]["rate"]
            entry["hot_vs_cold_rate_pct"] = round(100 * (h / c - 1), 1) if c and h else None
            onset = None
            if c:
                for r in hot:
                    if r["rate"] and r["rate"] < 0.95 * c:
                        onset = r["t"] - hot[0]["t"]
                        break
            entry["throttle_onset_s"] = onset
            out[f"{arm}/{seg}"] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8951)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "spec.json").write_text(json.dumps(spec, indent=1))
    servers = {}
    try:
        for i, (name, arm) in enumerate(spec["arms"].items()):
            servers[name] = Server(spec, name, arm, args.port + i, args.out)
        for srv in servers.values():
            srv.wait_ready()
            srv.client.chat(user("Say hello."), max_tokens=8)
        energy, temps = EnergySampler(), TemperatureSampler()
        series = Series(args.out / "series.jsonl")
        series.start()
        # cold precondition
        series.label = "cold-wait"
        deadline = time.time() + float(spec.get("cold_wait_s", 600))
        target = float(spec.get("cold_c", 50.0))
        while time.time() < deadline:
            d = temps.die_summary()["die_max_c"]
            if d is not None and d <= target:
                break
            time.sleep(5)
        with open(args.out / "segments.jsonl", "w") as out:
            cyc = run_cycles("hot", time.time() + float(spec.get("hot_s", 1200)), servers,
                             spec, energy, temps, series, out, 0, 0)
            series.label = "cooldown"
            time.sleep(float(spec.get("cooldown_s", 360)))
            run_cycles("duty", time.time() + float(spec.get("duty_s", 480)), servers, spec,
                       energy, temps, series, out, cyc, float(spec.get("gap_s", 3)))
        series.stop.set()
        rows = [json.loads(x) for x in open(args.out / "segments.jsonl")]
        summary = summarize(rows, spec)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))
    finally:
        for srv in servers.values():
            srv.stop()


if __name__ == "__main__":
    main()
