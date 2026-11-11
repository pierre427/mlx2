#!/usr/bin/env python3
"""Super VL NVFP4: served image and video against the ordinary reference.

``owned OUT`` takes a desktop CPG lease and the paired GPU locks and runs
``unit OUT`` under them.  Arms: ``ordinary`` (``--ordinary``, the reference)
and ``default`` (no route flag: the best-route LiLiCorr external route).
Each arm, at temperature 0 with the token trace, thinking off, 96 tokens:
the shapes image cold then warm (APCv2), then a synthetic 4 s clip (a full
frame fading red to blue, 320x240 at 10 fps, sampled at 2 fps) cold then
warm.  Recorded: text, token ids, route, media/speculation receipts, an
oracle word check (image: red circle blue square; video: red, blue) and,
against the ordinary arm, token identity or the first differing position.
Smoke evidence only, no thermal hold.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, str(ROOT / "src"))
import super_vl_nvfp4_unit as unit_mod  # noqa: E402

rm, mpp = unit_mod.rm, unit_mod.mpp
ARMS = {"ordinary": ["--ordinary"], "default": []}
VIDEO_QUESTION = ("This is a short video. In one sentence, what colour is the frame at "
                  "the start and what colour is it at the end?")


def synthetic_video() -> bytes:
    import cv2
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fade.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
        for index in range(40):
            t = index / 39
            frame = np.zeros((240, 320, 3), np.uint8)
            frame[..., 2] = int(255 * (1 - t))  # BGR: red channel
            frame[..., 0] = int(255 * t)        # blue channel
            writer.write(frame)
        writer.release()
        return path.read_bytes()


def video_messages(payload):
    data = base64.b64encode(payload).decode()
    return [{"role": "user", "content": [
        {"type": "input_video", "video_url": {"url": f"data:video/mp4;base64,{data}"}},
        {"type": "text", "text": VIDEO_QUESTION},
    ]}]


def first_difference(left, right):
    if left is None or right is None:
        return "missing"
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def summarize(kind, status, response, wall):
    if status != 200 or not isinstance(response, dict):
        return {"kind": kind, "http_status": status, "error": response}
    choice = (response.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    block = response.get("mlx2") or {}
    words = mpp.IMAGE_ORACLE if kind.startswith("image") else ("red", "blue")
    return {
        "kind": kind, "http_status": status, "wall_seconds": round(wall, 3),
        "content": content, "finish_reason": choice.get("finish_reason"),
        "token_ids": (block.get("token_trace") or {}).get("token_ids"),
        "prompt_tokens": (response.get("usage") or {}).get("prompt_tokens"),
        "cached_tokens": ((response.get("usage") or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens"),
        "route": block.get("route"), "media_decode": block.get("media_decode"),
        "speculation": block.get("speculation"),
        "oracle": all(word in content.lower() for word in words),
    }


def run_arm(name, flags, out, args, model_id, video, reference):
    out.mkdir(parents=True)
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(unit_mod.TARGET),
               "--host", "127.0.0.1", "--port", str(args.port), "--fabric", "off",
               "--max-context", "32768", "--cache-bytes", str(2 << 30), *flags]
    row = {"arm": name, "command": command, "requests": []}
    base = f"http://127.0.0.1:{args.port}"
    log = (out / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                              start_new_session=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    try:
        row["load_seconds"] = rm.wait_health(base, server, 1800, lambda: None)
        for kind, messages in (("image", mpp.image_messages()), ("image_warm", mpp.image_messages()),
                               ("video", video_messages(video)), ("video_warm", video_messages(video))):
            body = {"model": model_id, "messages": messages, "temperature": 0, "max_tokens": 96,
                    "stream": False, "return_token_trace": True, "enable_thinking": False}
            status, response, wall = unit_mod.pp.post(base, body, 900)
            summary = summarize(kind, status, response, wall)
            if reference is not None:
                ref = next((r for r in reference if r["kind"] == kind), None)
                summary["first_difference_vs_ordinary"] = first_difference(
                    summary.get("token_ids"), None if ref is None else ref.get("token_ids"))
            row["requests"].append(summary)
            print(json.dumps({"arm": name, **{k: summary.get(k) for k in (
                "kind", "http_status", "oracle", "route", "prompt_tokens", "cached_tokens",
                "first_difference_vs_ordinary")}, "content": (summary.get("content") or "")[:160],
                "error": summary.get("error")}), flush=True)
        row["status"] = rm.http_json(base + "/v1/status", None, timeout=30)[1]
        row["passed"] = True
    except BaseException as error:  # noqa: BLE001
        row["passed"] = False
        row["error"] = repr(error)
        print(json.dumps({"arm": name, "error": repr(error)}), flush=True)
    finally:
        row["server_stop"] = rm.stop_server(server)
        log.close()
        unit_mod.atomic(out / "arm.json", row)
    return row


def unit(out, args):
    ident, adapter = unit_mod.checkpoint()
    video = synthetic_video()
    (out / "fade.mp4").write_bytes(video)
    report = {"schema": "mlx2.super-vl-media-check.v1", "checkpoint": ident, "adapter": adapter,
              "source": unit_mod.source(), "qualified": False,
              "evidence_kind": "served media smoke vs ordinary, no thermal hold", "arms": []}
    reference = None
    for name in args.arms:
        row = run_arm(name, ARMS[name], out / name, args, unit_mod.TARGET.name, video, reference)
        if name == "ordinary" and row.get("passed"):
            reference = row["requests"]
        report["arms"].append({k: v for k, v in row.items() if k != "status"})
        unit_mod.atomic(out / "report.json", report)
    return 0


def owned(out, args):
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    ident, _adapter = unit_mod.checkpoint()
    job_id = "super-vl-media-check-" + out.name
    with DesktopCPGLease(out, job_id, checkpoint=ident,
                         source_revision=unit_mod.source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", job_id,
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(out),
                   "--arms", *args.arms, "--port", str(args.port)]
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                 env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--arms", nargs="+", default=["ordinary", "default"], choices=list(ARMS))
    parser.add_argument("--port", type=int, default=8797)
    args = parser.parse_args()
    if args.mode == "unit":
        args.out.mkdir(parents=True, exist_ok=True)
    return {"owned": owned, "unit": unit}[args.mode](args.out, args)


if __name__ == "__main__":
    sys.exit(main())
