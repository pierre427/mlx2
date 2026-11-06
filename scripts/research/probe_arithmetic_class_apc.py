#!/usr/bin/env python3
"""F7 engagement probe: lane arithmetic-class identity in the APCv2 namespace.

Feature 7 (7af38665c) put an arithmetic-class fingerprint into every lane-law
APCv2 namespace by default. This probe checks, on one dense lane-covered
checkpoint, that the identity engages and that it does what it is for:

  S0  --lane-matmul off, fresh cache                 ordinary (stock) reference
  S1  --lane-matmul auto, persist dir D              receipt + cold + warm
  S2  same as S1 after a graceful restart            rescan keeps lane entries; warm from disk
  S3  --lane-matmul off on the same D                lane entries are not served to stock

Pass: S1 status carries the arithmetic class and fingerprint; S1 warm reuses
cache and matches S1 cold; S2 rescan registers entries with no
identity_mismatch discard and its request reuses cache and matches S1 cold;
S3 reuses nothing from the lane namespace; every arm passes the functional
oracle. Lane versus stock tokens (S1 vs S0) are recorded with the first
divergence and its top-logprob windows for near-tie classification.

Not performance and not a thermal-gated run.

Usage: ``probe_arithmetic_class_apc.py owned OUT_DIR --model PATH`` (CPG
lease + paired GPU locks), or ``unit OUT_DIR --model PATH`` inside a lock.
Original mlx2 code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
PORT = 8391

# Varied filler (no repeated strings: reasoning models loop on them).
_SUBJECTS = ["the archivist", "a ferry pilot", "the night baker", "a glass blower", "the surveyor",
             "a beekeeper", "the clock repairer", "a cartographer", "the lighthouse keeper", "a potter",
             "the river guide", "a violin maker", "the orchard owner", "a tailor", "the stonemason"]
_VERBS = ["catalogued", "repaired", "measured", "painted", "traded", "sketched", "weighed", "sorted",
          "polished", "wrapped", "labelled", "inspected", "carried", "restored", "counted"]
_OBJECTS = ["copper kettles", "maps of the delta", "oak barrels", "spare sails", "brass hinges",
            "seed packets", "old ledgers", "lantern wicks", "wool blankets", "clay tiles",
            "pocket compasses", "silver spoons", "tide tables", "rope coils", "garden shears"]
_PLACES = ["near the north pier", "behind the market hall", "on the hill road", "by the mill race",
           "in the east quarter", "under the bridge arches", "at the old customs house",
           "beside the chapel", "along the canal", "past the salt flats"]


def _context() -> str:
    rng = random.Random(20261105)
    lines = []
    for day in range(1, 61):
        lines.append(f"Day {day}: {rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.randint(3, 97)} "
                     f"{rng.choice(_OBJECTS)} {rng.choice(_PLACES)}.")
        if day == 37:
            lines.append("Day 37 note: the vault seven access code is QF-6043.")
    return "\n".join(lines)


MESSAGES = [
    {"role": "system", "content": "You answer from the log only, briefly."},
    {"role": "user", "content": _context() + "\n\nWhat is the vault seven access code? "
                                             "Reply with the code only."},
]


def oracle(text: str) -> bool:
    import re

    text = re.sub(r"[‐-―]", "-", text or "")
    return re.search(r"(?<!\w)QF-6043(?!\w)", text) is not None


def checkout_env() -> dict:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT / "src"), existing) if p)
    return environment


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def source() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def host_sample() -> dict:
    return {"thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True),
            "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip()}


def _find(tree, key):
    """First value under ``key`` anywhere in a nested status document."""
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(tree, list):
        for value in tree:
            found = _find(value, key)
            if found is not None:
                return found
    return None


class Server:
    def __init__(self, out: Path, name: str, model: Path, extra: list[str]):
        self.out, self.name = out, name
        self.root = f"http://127.0.0.1:{PORT}"
        self.command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model),
                        "--host", "127.0.0.1", "--port", str(PORT), "--max-lanes", "1",
                        "--max-inflight", "1", "--max-context", "8192",
                        "--cache-bytes", str(2 << 30), *extra]
        self.log = (out / f"{name}.server.log").open("w")
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True, env=checkout_env(), cwd=ROOT)

    def get(self, path):
        with urlopen(self.root + path, timeout=30) as response:
            return json.load(response)

    def ready(self) -> dict:
        deadline = time.monotonic() + 600
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name}: server exited before readiness")
            try:
                status = self.get("/v1/status")
                if status.get("healthy"):
                    return status
            except (URLError, TimeoutError, ConnectionError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.name}: readiness timeout")
            time.sleep(0.5)

    def chat(self, model_name: str) -> dict:
        payload = {"model": model_name, "messages": MESSAGES, "temperature": 0, "max_tokens": 48,
                   "enable_thinking": False, "return_token_trace": True, "logprobs": True,
                   "top_logprobs": 5, "stream": False}
        request = Request(self.root + "/v1/chat/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=900) as response:
            result = json.load(response)
        content = result["choices"][0]["message"].get("content", "") or ""
        return {"content": content, "oracle_passed": oracle(content),
                "token_ids": result.get("mlx2", {}).get("token_trace", {}).get("token_ids"),
                "cached_tokens": ((result.get("usage") or {}).get("prompt_tokens_details") or {})
                .get("cached_tokens", 0),
                "prompt_tokens": (result.get("usage") or {}).get("prompt_tokens"),
                "logprobs": (result["choices"][0].get("logprobs") or {}).get("content"),
                "route": {k: v for k, v in (result.get("mlx2") or {}).items() if k != "token_trace"}}

    def stop(self) -> dict:
        """Graceful SIGTERM (persist-on-shutdown runs here), then the whole group."""
        stopped = {"graceful": False}
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=90)
                stopped["graceful"] = True
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        stopped["returncode"] = self.process.returncode
        self.log.close()
        return stopped


def _window(row: dict, index: int):
    rows = row.get("logprobs") or []
    return rows[index] if index < len(rows) else None


def first_divergence(a: dict, b: dict) -> dict | None:
    left, right = a.get("token_ids") or [], b.get("token_ids") or []
    for index, (x, y) in enumerate(zip(left, right)):
        if x != y:
            return {"position": index, "tokens": [x, y],
                    "window_a": _window(a, index), "window_b": _window(b, index)}
    if len(left) != len(right):
        return {"position": min(len(left), len(right)), "length_only": True}
    return None


def unit(out: Path, model: Path) -> int:
    receipt = {"schema": "mlx2.f7-arithmetic-class-apc.v1", "source": source(), "model": str(model),
               "host_before": host_sample(), "stages": {}}
    name = model.name
    persist = out / "apc-persist"
    stages = [
        ("S0_stock_reference", ["--lane-matmul", "off", "--cache-dir", str(out / "s0-scratch")], 1),
        ("S1_lane_cold_warm", ["--lane-matmul", "auto", "--apc-persist-dir", str(persist),
                               "--apc-persist-on-shutdown"], 2),
        ("S2_lane_after_restart", ["--lane-matmul", "auto", "--apc-persist-dir", str(persist),
                                   "--apc-persist-on-shutdown"], 1),
        ("S3_stock_on_lane_dir", ["--lane-matmul", "off", "--apc-persist-dir", str(persist)], 1),
    ]
    try:
        for stage, extra, requests in stages:
            server = Server(out, stage, model, extra)
            row = {"command": server.command}
            receipt["stages"][stage] = row
            try:
                status = server.ready()
                row["lane_matmul"] = status.get("lane_matmul") or _find(status, "lane_matmul")
                row["artifact_fingerprint"] = _find(status, "artifact_fingerprint")
                row["persistence_at_start"] = _find(status, "persistence")
                row["requests"] = [server.chat(name) for _ in range(requests)]
                row["persistence_at_end"] = _find(server.get("/v1/status"), "persistence")
            finally:
                row["stop"] = server.stop()
                atomic_json(out / "receipt.json", receipt)
        s = receipt["stages"]
        s0, s1, s2, s3 = (s[k] for k in ("S0_stock_reference", "S1_lane_cold_warm",
                                          "S2_lane_after_restart", "S3_stock_on_lane_dir"))
        cold, warm = s1["requests"]
        rescan = (s2.get("persistence_at_start") or {}).get("rescan") or {}
        rescan3 = (s3.get("persistence_at_start") or {}).get("rescan") or {}
        lane = s1.get("lane_matmul") or {}
        checks = {
            "s0_lane_not_installed": not (s0.get("lane_matmul") or {}).get("installed"),
            "s1_lane_installed": bool(lane.get("installed")),
            "s1_arithmetic_class_reported": bool(lane.get("arithmetic_class"))
            and bool(lane.get("arithmetic_class_fingerprint")),
            "s1_warm_reused": warm["cached_tokens"] > 0,
            "s1_cold_warm_exact": cold["token_ids"] == warm["token_ids"],
            "s1_persist_graceful": s1["stop"]["graceful"],
            "s2_same_fingerprint": (s2.get("lane_matmul") or {}).get("arithmetic_class_fingerprint")
            == lane.get("arithmetic_class_fingerprint"),
            "s2_rescan_registered": int(rescan.get("registered", 0)) > 0,
            "s2_no_identity_mismatch": not (rescan.get("discarded") or {}).get("identity_mismatch"),
            "s2_reused_from_disk": s2["requests"][0]["cached_tokens"] > 0,
            "s2_matches_s1_cold": s2["requests"][0]["token_ids"] == cold["token_ids"],
            "s3_no_lane_reuse": s3["requests"][0]["cached_tokens"] == 0,
            "s3_matches_s0": s3["requests"][0]["token_ids"] == s0["requests"][0]["token_ids"],
            "oracle_all": all(r["oracle_passed"] for st in s.values() for r in st["requests"]),
        }
        receipt["checks"] = checks
        receipt["s3_rescan"] = rescan3
        receipt["lane_vs_stock_exact"] = cold["token_ids"] == s0["requests"][0]["token_ids"]
        receipt["lane_vs_stock_first_divergence"] = first_divergence(s0["requests"][0], cold)
        receipt["passed"] = all(checks.values())
    except BaseException as exc:
        receipt.update(passed=False, error=repr(exc))
        raise
    finally:
        receipt["host_after"] = host_sample()
        atomic_json(out / "receipt.json", receipt)
    print(json.dumps({"passed": receipt["passed"], "checks": receipt.get("checks")}), flush=True)
    return 0 if receipt["passed"] else 1


def owned(out: Path, model: Path, job_id: str) -> int:
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out, job_id, checkpoint=model.name, source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out), "--model", str(model)]
        child = subprocess.Popen(command, cwd=ROOT, env=checkout_env(), start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--job-id", default="f7-arithmetic-class-apc-20261105")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.mode == "unit":
        return unit(out, args.model)
    return owned(out, args.model, args.job_id)


if __name__ == "__main__":
    raise SystemExit(main())
