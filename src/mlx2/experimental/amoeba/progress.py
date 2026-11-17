"""Progress: a terminal line, a JSONL event log and a build-time Prometheus exporter.

Events (``<out>/events.jsonl``, one JSON object per line, ``ts`` + ``event`` + fields) are the single
source: the terminal line and the exporter both read the in-memory state the events update. Child
processes report through their stdout: a line ``@@amoeba-event {json}`` is an event, and the carve's
own step lines (``<tag> lam=.. step N main=.. kept=.. (S s/step, peak G GB)``) are parsed into
``step`` events. The parent samples each child's physical footprint (not RSS) every 15 s.

Metrics (loopback only, ``--metrics-port``; 0 = off), all prefixed ``mlx2_amoeba_``:
  build_info{checkpoint,commit,profile}, stage_status{stage,status}, stage_seconds{stage},
  boxes{state=total|done|pass|miss|failed|queued}, current_box_info{box}, carve_step, carve_steps,
  step_seconds, mlx_peak_memory_gb, footprint_gb, footprint_peak_gb, gen_tokens_per_second,
  gen_tokens_total, eta_seconds, held_dce_nats (histogram over finished boxes, dCE < 0 counted as 0),
  events_total.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVENT_PREFIX = "@@amoeba-event "
STEP_RE = re.compile(r"(?P<tag>\S+) lam=\S+ step (?P<step>\d+) main=(?P<main>[-\d.]+) kept=(?P<kept>[-\d.]+) "
                     r"\((?P<sps>[\d.]+) s/step, peak (?P<peak>[\d.]+|None) GB\)")
HELD_BUCKETS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0)


def child_event(event: str, **fields) -> None:
    """Called inside a child process: report one event to the parent through stdout."""
    print(EVENT_PREFIX + json.dumps({"event": event, **fields}, default=str), flush=True)


class Progress:
    def __init__(self, events_path: Path, *, tty: bool | None = None, quiet: bool = False, interval_s: float = 1.0):
        self.path = Path(events_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.tty = sys.stderr.isatty() if tty is None else tty
        self.quiet = quiet
        self.interval_s = interval_s if self.tty else 30.0
        self._last_line = 0.0
        self._line_len = 0
        self.state = {"stage": None, "stages": {}, "stage_seconds": {}, "boxes": {}, "box_order": [],
                      "current_box": None, "step": None, "steps": None, "s_per_step": None, "mlx_peak_gb": None,
                      "footprint_gb": None, "footprint_peak_gb": 0.0, "gen_tok_s": None, "gen_tokens": 0,
                      "eta_s": None, "last_s_per_step": None, "events": 0, "info": {}, "box_walls": [], "stage_t0": None}
        from mlx2.prometheus import CumulativeHistogram
        self.held_hist = CumulativeHistogram(HELD_BUCKETS)

    # ------------------------------------------------------------ events
    def emit(self, event: str, **fields) -> None:
        rec = {"ts": round(time.time(), 3), "event": event, **fields}
        with self.lock:
            with self.path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            self._apply(rec)
        self.render(force=event not in ("step", "footprint", "gen"))

    def _apply(self, e: dict) -> None:
        s = self.state
        s["events"] += 1
        ev = e["event"]
        if ev == "build_start":
            s["info"] = {k: e.get(k) for k in ("checkpoint", "commit", "profile", "out")}
            for st in e.get("stages", []):
                s["stages"].setdefault(st, "pending")
        elif ev == "stage_start":
            s["stage"], s["stage_t0"] = e["stage"], e["ts"]
            s["stages"][e["stage"]] = "running"
            s.update(step=None, steps=None, s_per_step=None, eta_s=None, current_box=None)
        elif ev == "stage_skip":
            s["stages"][e["stage"]] = "done"
        elif ev == "stage_end":
            s["stages"][e["stage"]] = e.get("status", "done")
            s["stage_seconds"][e["stage"]] = e.get("wall_s")
            if s["stage"] == e["stage"]:
                s.update(stage=None, step=None, steps=None, eta_s=None, current_box=None)
        elif ev == "steps":
            s["steps"] = e.get("steps")
        elif ev == "step":
            s["step"], s["s_per_step"] = e.get("step"), e.get("s_per_step")
            if e.get("s_per_step"):
                s["last_s_per_step"] = e["s_per_step"]
            if e.get("peak_gb") is not None:
                s["mlx_peak_gb"] = e["peak_gb"]
        elif ev == "footprint":
            s["footprint_gb"] = e.get("gb")
            if e.get("gb") is not None:
                s["footprint_peak_gb"] = max(s["footprint_peak_gb"] or 0.0, e["gb"])
        elif ev == "gen":
            s["gen_tok_s"] = e.get("tok_s")
            s["gen_tokens"] = e.get("tokens_total", s["gen_tokens"])
        elif ev == "layer":                       # rolling capture, per layer and chunk
            s["rolling"] = {k: e.get(k) for k in ("layer", "chunk", "chunks", "load_s", "compute_s", "stall_s",
                                                  "state_bytes", "prefetched_bytes")}
            s["rolling_stall_total"] = (s.get("rolling_stall_total") or 0.0) + float(e.get("stall_s") or 0.0)
            s["rolling_load_total"] = (s.get("rolling_load_total") or 0.0) + float(e.get("load_s") or 0.0)
        elif ev == "boxes_queued":
            for b in e["boxes"]:
                s["boxes"].setdefault(b, "queued")
                if b not in s["box_order"]:
                    s["box_order"].append(b)
        elif ev == "box_start":
            s["current_box"] = e["box"]
            s["boxes"][e["box"]] = "running"
            s.update(step=None, s_per_step=None)
            if e.get("steps"):
                s["steps"] = e["steps"]
        elif ev == "box_end":
            v = e.get("verdict")      # pass | miss | failed | covered
            s["boxes"][e["box"]] = v
            if e.get("wall_s"):
                s["box_walls"].append(e["wall_s"])
            if e.get("held_dce") is not None:
                self.held_hist.observe(max(0.0, float(e["held_dce"])))
            if s["current_box"] == e["box"]:
                s.update(current_box=None, step=None, s_per_step=None)
        s["eta_s"] = self._eta()

    def _eta(self):
        s = self.state
        rem = None
        if s["step"] is not None and s["steps"] and s["s_per_step"]:
            rem = max(0, s["steps"] - s["step"] - 1) * s["s_per_step"]
        queued = sum(v == "queued" for v in s["boxes"].values())
        if s["stage"] == "boxes" and s["box_walls"]:
            per = sum(s["box_walls"]) / len(s["box_walls"])
            return round((rem or 0) + queued * per, 1)
        return None if rem is None else round(rem, 1)

    def box_counts(self) -> dict:
        v = list(self.state["boxes"].values())
        return {"total": len(v), "done": sum(x in ("pass", "miss", "failed", "covered") for x in v),
                "pass": v.count("pass"), "miss": v.count("miss"), "failed": v.count("failed"),
                "covered": v.count("covered"), "queued": v.count("queued"), "running": v.count("running")}

    # ------------------------------------------------------------ terminal line
    def line(self) -> str:
        s = self.state
        parts = [f"[amoeba] {s['stage'] or 'idle'}"]
        bc = self.box_counts()
        if bc["total"]:
            parts.append(f"boxes {bc['done']}/{bc['total']} (pass {bc['pass']}, miss {bc['miss']}"
                         f"{', failed ' + str(bc['failed']) if bc['failed'] else ''})")
        if s["current_box"]:
            parts.append(f"box {s['current_box']}")
        if s["step"] is not None:
            parts.append(f"step {s['step'] + 1}/{s['steps'] or '?'}")
        if s["s_per_step"]:
            parts.append(f"{s['s_per_step']:.1f} s/step")
        if s["mlx_peak_gb"] is not None:
            parts.append(f"peak {s['mlx_peak_gb']} GB")
        if s["footprint_gb"] is not None:
            parts.append(f"fp {s['footprint_gb']:.1f} GB")
        if s["gen_tok_s"]:
            parts.append(f"gen {s['gen_tok_s']:.0f} tok/s")
        if s["eta_s"] is not None:
            e = int(s["eta_s"])
            parts.append(f"ETA {e // 3600}:{e % 3600 // 60:02d}:{e % 60:02d}")
        return " | ".join(parts)

    def render(self, force: bool = False) -> None:
        if self.quiet:
            return
        t = time.time()
        if not force and t - self._last_line < self.interval_s:
            return
        self._last_line = t
        ln = self.line()
        if self.tty:
            pad = max(0, self._line_len - len(ln))
            sys.stderr.write("\r" + ln + " " * pad)
            self._line_len = len(ln)
        else:
            sys.stderr.write(ln + "\n")
        sys.stderr.flush()

    def newline(self) -> None:
        if self.tty and not self.quiet and self._line_len:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._line_len = 0

    # ------------------------------------------------------------ child output
    def feed_line(self, line: str, stage: str) -> None:
        if line.startswith(EVENT_PREFIX):
            try:
                e = json.loads(line[len(EVENT_PREFIX):])
            except json.JSONDecodeError:
                return
            ev = e.pop("event", "child")
            self.emit(ev, stage=stage, **e)
            return
        m = STEP_RE.search(line)
        if m:
            peak = m["peak"]
            self.emit("step", stage=stage, tag=m["tag"], step=int(m["step"]), loss=float(m["main"]),
                      kept=float(m["kept"]), s_per_step=float(m["sps"]),
                      peak_gb=None if peak == "None" else float(peak))

    # ------------------------------------------------------------ Prometheus
    def metrics(self) -> str:
        from mlx2.prometheus import PrometheusBuilder
        b = PrometheusBuilder()
        with self.lock:
            s = json.loads(json.dumps(self.state, default=str))
            hist = self.held_hist.snapshot()
        p = "mlx2_amoeba_"
        info = s["info"]
        b.gauge(p + "build_info", "aMoEba build identity", 1,
                {"checkpoint": info.get("checkpoint") or "", "commit": info.get("commit") or "",
                 "profile": info.get("profile") or ""})
        for st, status in s["stages"].items():
            for k in ("pending", "running", "done", "failed", "refused"):
                b.gauge(p + "stage_status", "1 for the stage's current status", int(status == k),
                        {"stage": st, "status": k})
        for st, w in s["stage_seconds"].items():
            if w is not None:
                b.gauge(p + "stage_seconds", "wall seconds of a finished stage", w, {"stage": st})
        for k, v in self.box_counts().items():
            b.gauge(p + "boxes", "boxes by state", v, {"state": k})
        if s["current_box"]:
            b.gauge(p + "current_box_info", "the box being carved", 1, {"box": s["current_box"]})
        for name, key, help_ in (("carve_step", "step", "current carve step (0-based)"),
                                 ("carve_steps", "steps", "carve steps per mask"),
                                 ("step_seconds", "last_s_per_step", "seconds per step of the latest carve / capture (running mean)"),
                                 ("mlx_peak_memory_gb", "mlx_peak_gb", "MLX peak memory of the child, GB"),
                                 ("footprint_gb", "footprint_gb", "child physical footprint, GB"),
                                 ("footprint_peak_gb", "footprint_peak_gb", "peak child physical footprint, GB"),
                                 ("gen_tokens_per_second", "gen_tok_s", "on-policy generation tokens per second"),
                                 ("eta_seconds", "eta_s", "estimated seconds left in the stage")):
            if s[key] is not None:
                b.gauge(p + name, help_, s[key])
        b.gauge(p + "gen_tokens", "on-policy tokens generated so far", s["gen_tokens"] or 0)
        r = s.get("rolling")
        if r:
            for name, key, help_ in (("rolling_layer", "layer", "rolling capture: layer just finished"),
                                     ("rolling_chunk", "chunk", "rolling capture: chunk index (0-based)"),
                                     ("rolling_chunks", "chunks", "rolling capture: corpus chunks (weight reads)"),
                                     ("rolling_layer_load_seconds", "load_s", "rolling: last layer's weight load"),
                                     ("rolling_layer_compute_seconds", "compute_s", "rolling: last layer's compute"),
                                     ("rolling_layer_stall_seconds", "stall_s", "rolling: last layer's prefetch stall"),
                                     ("rolling_state_bytes", "state_bytes", "rolling: resident corpus state"),
                                     ("rolling_prefetched_bytes", "prefetched_bytes", "rolling: bytes read ahead")):
                if r.get(key) is not None:
                    b.gauge(p + name, help_, r[key])
            b.counter(p + "rolling_prefetch_stall_seconds_total", "rolling: total prefetch stall",
                      s.get("rolling_stall_total") or 0.0)
            b.counter(p + "rolling_load_seconds_total", "rolling: total weight load time",
                      s.get("rolling_load_total") or 0.0)
        b.counter(p + "events_total", "progress events written", s["events"])
        b.histogram(p + "held_dce_nats", "held dCE (core + box vs full, pooled) of finished boxes", hist)
        return b.render()


class MetricsServer:
    def __init__(self, progress: Progress, port: int, host: str = "127.0.0.1"):
        prog = progress

        class H(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path.split("?")[0] != "/metrics":
                    self.send_error(404)
                    return
                from mlx2.prometheus import CONTENT_TYPE
                body = prog.metrics().encode()
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer((host, port), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
