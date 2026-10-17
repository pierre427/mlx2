#!/usr/bin/env python3
"""Download pinned training shards after BF16 acquisition, then prepare CPU corpus."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlx2.adapters.nemotron35_super_config import SOURCE_REVISION
from mlx2.conversion.calibration_corpus import (
    DATASET,
    REVISION,
    SHARDS,
    prepare_corpus,
    verify_corpus,
)
from mlx2.conversion.nvfp4 import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--source-marker", type=Path, required=True)
    parser.add_argument("--reserve-bytes", type=int, default=50_000_000_000)
    parser.add_argument("--code-plan", type=Path, help="Expected worktree branch and preparation code hashes")
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    lock = (args.state / "calibration-prepare.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def status(state, **fields):
        record = {"state": state, "updated_at": datetime.datetime.now(datetime.UTC).isoformat(), **fields}
        temporary = args.state / "calibration-status.json.tmp"
        with temporary.open("w") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(args.state / "calibration-status.json")

    try:
        while not args.source_marker.is_file():
            status("waiting_for_verified_download", destination=str(args.destination))
            time.sleep(30)
        marker = json.loads(args.source_marker.read_text())
        if (marker.get("revision") != SOURCE_REVISION or marker.get("files") != 88
                or marker.get("total_bytes") != 248937279780):
            raise ValueError("BF16 acquisition completion marker differs")
        if args.code_plan:
            plan = json.loads(args.code_plan.read_text())
            branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=plan["worktree"], text=True).strip()
            if branch != plan["branch"]:
                raise ValueError("Calibration worktree is no longer on the requested branch")
            for filename, expected in plan["code_sha256"].items():
                if sha256(Path(plan["worktree"]) / filename) != expected:
                    raise ValueError("Queued preparation source changed; review and refresh its plan")
        if args.destination.exists():
            manifest, _ = verify_corpus(args.destination)
            status("prepared", corpus_sha256=manifest["corpus_sha256"], samples=manifest["samples"], reused=True)
            return
        # Explicitly read the user's requested credential in this process.
        # Never put it in the environment, command line or status/log output.
        token = (Path.home() / ".hf_pr").read_text().strip()
        if not token:
            raise ValueError("Requested HF credential file is empty")
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ.pop(name, None)
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from huggingface_hub import hf_hub_download

        args.cache.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(args.cache).free < args.reserve_bytes + sum(item[2] for item in SHARDS) + (5 << 30):
            raise OSError("Calibration acquisition would consume the free-space reserve")
        status("downloading_training_shards", files=len(SHARDS), bytes=sum(item[2] for item in SHARDS))

        def download(item):
            _, filename, size, expected = item
            path = Path(hf_hub_download(DATASET, filename, repo_type="dataset", revision=REVISION,
                                       token=token, local_dir=args.cache))
            if path.is_symlink() or path.stat().st_size != size or sha256(path) != expected:
                raise ValueError("Calibration download bytes differ from pinned training shard")

        with ThreadPoolExecutor(max_workers=4) as workers:
            list(workers.map(download, SHARDS))
        manifest = prepare_corpus(args.cache, args.destination, reserve_bytes=args.reserve_bytes,
                                  progress=lambda item: status(**item))
        verified, _ = verify_corpus(args.destination, expected_digest=manifest["corpus_sha256"])
        status("prepared", samples=verified["samples"], corpus_sha256=verified["corpus_sha256"],
               destination=str(args.destination), gpu_used=False,
               preparer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    except Exception as error:
        status("failed", error=type(error).__name__ + ": " + str(error))
        raise


if __name__ == "__main__":
    main()
