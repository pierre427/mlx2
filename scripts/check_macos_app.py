#!/usr/bin/env python3
"""CPU-only desktop integration check. Never starts or stops a model.

The native Trash round trip uses a new disposable fixture and restores it
before the temporary directory is removed. No library artifact is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=Path("build/macos/MLX2.app"))
    parser.add_argument("--models", type=Path, default=Path.home() / "mlx-models")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    binary = args.app.resolve() / "Contents/MacOS/MLX2"
    runtime = json.loads((args.app.resolve() / "Contents/Resources/runtime.json").read_text())
    runtime_source = Path(runtime["source"])
    env = {**os.environ, "PYTHONPATH": str(runtime_source / "src")}
    revision = runtime["build_revision"]
    for relative, expected in runtime["source_hashes"].items():
        source = runtime_source / relative if relative.startswith("src/") else root / relative
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Installed source identity differs: {relative}")
    with tempfile.TemporaryDirectory(prefix="mlx2-desktop-check-") as temporary:
        temporary = Path(temporary)
        fixture = temporary / f"disposable-desktop-check-{uuid.uuid4().hex}"
        fixture.mkdir()
        (fixture / "fixture.txt").write_text(
            "Disposable native Trash integration fixture.\n"
        )
        info = fixture.stat()
        identity = f"{info.st_dev}:{info.st_ino}:{info.st_ctime_ns}"
        refused = subprocess.run(
            [str(binary), "--trash", str(fixture), "stale"],
            capture_output=True,
            text=True,
            check=False,
        )
        if refused.returncode == 0 or not fixture.is_dir():
            raise RuntimeError("Native helper accepted a stale identity")
        result = subprocess.run(
            [str(binary), "--trash", str(fixture), identity],
            capture_output=True,
            text=True,
            check=True,
        )
        destination = Path(json.loads(result.stdout)["destination"])
        if fixture.exists() or not destination.is_dir():
            raise RuntimeError(
                "Native Trash operation did not produce a recoverable folder"
            )
        shutil.move(str(destination), str(fixture))
        if (
            fixture / "fixture.txt"
        ).read_text() != "Disposable native Trash integration fixture.\n":
            raise RuntimeError("Trash restore changed the fixture")
        home = temporary / "app-home"
        home.mkdir()
        (home / "desktop-settings.json").write_text(
            json.dumps(
                {
                    "root": str(args.models.expanduser().resolve()),
                    "port": 18285,
                    "context": 8192,
                    "lanes": 1,
                    "route": "auto",
                }
            )
        )
        child = subprocess.Popen(
            [
                runtime["python"],
                "-u",
                "-m",
                "mlx2.desktop",
                "--home",
                str(home),
                "--trash-helper",
                str(binary),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        try:

            def request(command):
                identifier = uuid.uuid4().hex
                child.stdin.write(
                    json.dumps({"id": identifier, "command": command}) + "\n"
                )
                child.stdin.flush()
                if not select.select([child.stdout], [], [], 45)[0]:
                    raise RuntimeError("Desktop protocol response timed out")
                line = child.stdout.readline()
                if not line:
                    raise RuntimeError("Desktop exited: " + child.stderr.read())
                reply = json.loads(line)
                if reply["id"] != identifier:
                    raise RuntimeError("Desktop protocol response identity differs")
                return reply

            if request("unknown")["ok"]:
                raise RuntimeError("Unknown command was accepted")
            deadline = time.monotonic() + 90
            while True:
                response = request("snapshot")
                if not response["ok"]:
                    raise RuntimeError(response["error"])
                snapshot = response["result"]
                if not snapshot["library"]["scanning"]:
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Inventory scan did not finish")
                time.sleep(0.1)
            if snapshot["worker"] is not None:
                raise RuntimeError(
                    "Read-only desktop check unexpectedly created a worker"
                )
            if snapshot["library"]["error"]:
                raise RuntimeError(snapshot["library"]["error"])
            receipt = {
                "source_revision": revision,
                "source_files_sha256": runtime["source_hashes"],
                "runtime_python": runtime["python"],
                "runtime_source": runtime["source"],
                "kind": "desktop-tooling-integration-not-model-qualification",
                "native_trash_round_trip": "passed",
                "stale_identity_refusal": "passed",
                "json_protocol": "passed",
                "library_folders": len(snapshot["library"]["rows"]),
                "host_thermal": snapshot["host"]["thermal_state"],
                "read_only_servers_observed": len(snapshot["servers"]),
                "model_work_started": False,
            }
        finally:
            child.stdin.close()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=15)
        if child.returncode:
            raise RuntimeError(
                "Desktop did not shut down cleanly: " + child.stderr.read()
            )
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
