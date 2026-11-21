#!/usr/bin/env python3
"""Image + text HTTP smoke of a Hy4-flash artifact on its no-flag default route.

Original mlx2 script.  The caller owns the GPU (CPG lease + paired locks).
Starts one loopback ``mlx2.server`` (never on 8282) with ``--server-args``
(none = the artifact's no-flag default route), then sends

* an image request: a generated PNG with large rendered text, asking for the
  text (oracle: the rendered word and number appear, case-insensitive);
* a text request (oracle: ``Paris``);

each with the vendor's sampling defaults (the request carries no temperature
or top_p) and a fixed seed, and records the raw responses, the terminal mlx2
receipts (route, speculation) and the server's ``/v1/status`` route receipt
and vision stats.  Writes ``vision.json`` atomically; refuses a nonempty
output directory; stops the server process group in ``finally``.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORD, NUMBER = "LIGHTHOUSE", "4721"


def image_uri() -> str:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 320), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 88)
    except OSError:
        font = ImageFont.load_default(size=88)
    draw.text((40, 50), WORD, fill="black", font=font)
    draw.text((200, 170), NUMBER, fill="navy", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def get_json(base, path, timeout=10):
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.load(response)


def checkpoint_label(name, status):
    """``<artifact name>@<fingerprint>`` from ``/v1/status``.

    The adapter's ``execution.checkpoint`` names the artifact; the top-level
    ``artifact`` field is a route digest once a drafter is bound (the no-flag
    default here), so it is only the fallback.
    """
    label = (status.get("execution") or {}).get("checkpoint")
    if isinstance(label, str) and "@" in label:
        return f"{name}@{label.split('@', 1)[1]}"
    artifact = status.get("artifact")
    return f"{name}@{artifact}" if isinstance(artifact, str) else None


def chat(base, messages, *, seed, max_tokens, timeout=600):
    body = {"messages": messages, "max_tokens": max_tokens, "seed": seed, "stream": False}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.load(response)
    except urllib.error.HTTPError as error:
        return {"error": f"HTTP {error.code}", "body": error.read().decode("utf-8", "replace")[:4000]}
    reply["_elapsed_s"] = round(time.perf_counter() - started, 3)
    return reply


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8285)
    parser.add_argument("--server-args", default="",
                        help="extra mlx2.server arguments (shell-quoted); empty = the no-flag default route")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--max-tokens", type=int, default=32)
    a = parser.parse_args(argv)
    if a.port == 8282:
        parser.error("port 8282 is reserved for release trees")
    out = a.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error(f"{out} is not empty")
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", a.port)) == 0:
            parser.error(f"port {a.port} is occupied")
    model = a.model.expanduser().resolve(strict=True)
    base = f"http://127.0.0.1:{a.port}"
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    command = [sys.executable, "-u", "-m", "mlx2.server", "--model", str(model), "--host", "127.0.0.1",
               "--port", str(a.port), "--max-lanes", "1", "--max-inflight", "1", "--no-cache-dir",
               *shlex.split(a.server_args)]
    report = {"schema": "mlx2.hy4-vision-http-smoke.v1", "commit": commit, "command": command,
              "kind": "smoke; unqualified; no thermal hold", "started_at": datetime.now(UTC).isoformat(),
              "status": "failed"}
    server = None
    try:
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        log = open(out / "server.log", "w")
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        t0 = time.monotonic()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode} during load")
            try:
                if get_json(base, "/health").get("status") == "ok":
                    break
            except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                pass
            if time.monotonic() - t0 > 900:
                raise TimeoutError("server not healthy within 900 s")
            time.sleep(1)
        report["load_seconds"] = round(time.monotonic() - t0, 1)
        status = get_json(base, "/v1/status")
        report["checkpoint"] = checkpoint_label(model.name, status)
        report["status_start"] = {k: status.get(k) for k in ("route_receipt", "settings", "capabilities",
                                                             "qualification", "media_route", "profile")}
        image = chat(base, [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_uri()}},
            {"type": "text", "text": "What text is written in this image? Reply with the text only."}]}],
            seed=a.seed, max_tokens=a.max_tokens)
        text = chat(base, [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
                    seed=a.seed, max_tokens=a.max_tokens)
        for name, reply, check in (("image", image, lambda s: WORD.lower() in s.lower() and NUMBER in s),
                                   ("text", text, lambda s: "paris" in s.lower())):
            content = ((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            reply["oracle_pass"] = bool(content) and check(content)
            print(f"[{name}] {'PASS' if reply['oracle_pass'] else 'FAIL'} {content!r} "
                  f"{reply.get('error', '')}", flush=True)
        report.update(image=image, text=text)
        status = get_json(base, "/v1/status")
        report["status_end"] = {k: status.get(k) for k in ("route_receipt", "media_route", "settings",
                                                           "route_default_selection")}
        try:
            with urllib.request.urlopen(base + "/metrics", timeout=10) as response:
                metrics = response.read().decode()
            (out / "metrics.prom").write_text(metrics)
            report["metrics_engram_families"] = sorted({line.split("{")[0].split(" ")[0]
                                                        for line in metrics.splitlines()
                                                        if line.startswith("mlx2_engram")})
        except (urllib.error.URLError, OSError) as error:
            report["metrics_error"] = repr(error)
        report["status"] = "passed" if image["oracle_pass"] and text["oracle_pass"] else "failed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if server is not None and server.poll() is None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(60)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        report["finished_at"] = datetime.now(UTC).isoformat()
        tmp = out / "vision.json.partial"
        tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        os.replace(tmp, out / "vision.json")
    print(json.dumps({"status": report["status"], "checkpoint": report.get("checkpoint"),
                      "commit": commit, "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
