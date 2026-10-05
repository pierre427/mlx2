"""Default-off process boundary for a TensorFold-owned Qwen3.8 profile.

This adapter serves only through explicit experimental startup and request
selection. APCv2 publication of its opaque worker cache remains disabled until
a revision-bound bridge is implemented and qualified. The existing mlx2
ordinary route stays the production reference.
"""

from __future__ import annotations

import json
import select
import subprocess
import sys
from pathlib import Path

from ..runtime.tensorfold_owned_worker import (
    artifact_identity,
    mlx_lm_identity,
    profile_identity,
    source_identity,
)


class TensorfoldOwnedB1Profile:
    """One worker-owned model with serial and tree cache operations."""

    def __init__(self, source, target, drafter, *, mlx_lm_source=None,
                 enabled=False, fixture=False):
        if enabled is not True:
            raise ValueError("TensorFold-owned B1 profile requires explicit opt-in")
        self.fixture = bool(fixture)
        self.source = str(Path(source).expanduser().resolve()) if source else None
        self.target = str(Path(target).expanduser().resolve()) if target else None
        self.drafter = str(Path(drafter).expanduser().resolve()) if drafter else None
        self.mlx_lm_source = (str(Path(mlx_lm_source).expanduser().resolve())
                              if mlx_lm_source else None)
        if fixture:
            self.identity = "fixture"
        else:
            if not all((self.source, self.target, self.drafter, self.mlx_lm_source)):
                raise ValueError("source, target, drafter and mlx_lm source are required")
            config = json.loads((Path(self.target) / "config.json").read_text())
            topology = config.get("text_config", config)
            if (config.get("model_type") != "qwen3_5"
                    or (topology.get("num_hidden_layers"), topology.get("hidden_size"))
                    != (64, 5120)):
                raise ValueError("TensorFold-owned profile requires the Qwen3.8 27B target topology")
            self.identity = profile_identity(
                source_identity(self.source), artifact_identity(self.target),
                artifact_identity(self.drafter), mlx_lm_identity(self.mlx_lm_source),
            )
        self.cache_layout = f"qwen38-tensorfold-owned:{self.identity}"
        self.apcv2_namespace = f"tensorfold-owned-qwen38-b1:{self.identity}"
        self.apcv2_state_bridge_qualified = False
        self.process = None

    def start(self, *, timeout=180):
        if self.process is not None:
            raise RuntimeError("TensorFold worker already started")
        command = [sys.executable, "-m", "mlx2.runtime.tensorfold_owned_worker"]
        command += (["--fixture"] if self.fixture else [
            "--source", self.source, "--target", self.target, "--drafter", self.drafter,
            "--mlx-lm-source", self.mlx_lm_source,
        ])
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1,
        )
        try:
            receipt = self.call("ping", timeout=timeout)
            if receipt["identity"] != self.identity or receipt["cache_layout"] != self.cache_layout:
                raise RuntimeError("TensorFold worker identity disagrees with adapter pin")
            return receipt
        except BaseException:
            self.close()
            raise

    def call(self, method, *, timeout=180, **params):
        process = self.process
        if process is None or process.poll() is not None:
            raise RuntimeError("TensorFold worker is not running")
        if method not in {"ping", "open", "serial", "tree", "compare_round", "close",
                          "live_open", "live_step", "live_close", "live_compare_serial"}:
            raise ValueError("unsupported TensorFold worker method")
        process.stdin.write(json.dumps({"method": method, "params": params}) + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise TimeoutError("TensorFold worker response timed out")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"TensorFold worker exited with {process.poll()}")
        response = json.loads(line)
        if response.get("ok") is not True:
            raise RuntimeError(f"TensorFold worker {response.get('error')}: {response.get('message')}")
        return response["result"]

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["TensorfoldOwnedB1Profile"]
