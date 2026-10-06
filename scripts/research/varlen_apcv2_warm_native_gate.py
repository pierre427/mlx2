"""One unselected Qwen3 APCv2-warm native request gate; GPU opt-in only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import signal
import subprocess
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/tmp/mlx2-varlen-price-eaed9ea1/model")
ARTIFACT = Path("/tmp/mlx2-varlen-price-eaed9ea1/artifact.json")
WHEEL = (
    Path.home()
    / ".cache/uv/sdists-v9/path/6b22317da775785a/d5Co9cAdGp8_XULv/mlx-0.32.2.dev20260919+39400a0d4-cp312-cp312-macosx_26_0_arm64.whl"
)
KERNEL = Path("/tmp/mlx2-paged-host-wait-6696c546-build/_paged_kv_native.cpython-312-darwin.so")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identity() -> dict:
    from mlx2.runtime.paged_price_identity import compute_live_price_identity

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise RuntimeError("warm native gate requires a clean checkout")
    live = compute_live_price_identity(
        ARTIFACT, WHEEL, KERNEL, adapter_artifact_root=MODEL)
    return {"source_commit": commit, "live": live,
            "script_sha256": sha256(Path(__file__)),
            "artifact_manifest_sha256": sha256(ARTIFACT),
            "native_binary_sha256": sha256(KERNEL),
            "wheel_sha256": sha256(WHEEL)}


def preflight(frozen: dict) -> dict:
    if frozen != identity():
        raise RuntimeError("frozen source/model/wheel/kernel identity drifted")
    session = os.environ.get("GPUQ_SESSION")
    lease = os.environ.get("GPUQ_LEASE")
    if not session or not lease:
        raise RuntimeError("shared gpuq session and lease are required")
    for path in (Path("/Users/Shared/mlxuag/gpu.lock"), Path("/tmp/gpu.lock")):
        owner = json.loads((path / "owner.json").read_text())
        if owner.get("session") != session or owner.get("lease_id") != lease:
            raise RuntimeError(f"GPU lock ownership differs: {path}")
    with urllib.request.urlopen("http://127.0.0.1:8600/health", timeout=2) as response:
        music = json.load(response)
    if music.get("busy") is not False or music.get("loaded") is not False:
        raise RuntimeError("Music3 service owns the GPU")
    return {"session": session, "lease_id": lease,
            "host": platform.node(), "locks": "matched", "music3": music}


def run(frozen: dict) -> dict:
    import _paged_kv_native
    import mlx.core as mx
    import numpy as np
    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.paged_native_batch_lifecycle import (
        prepare_queued_native_first_response, run_research_native_queued_qwen3,
    )
    from mlx2.runtime.paged_request_transaction import CandidateRequest
    from varlen_live_qwen3_request_driver import Qwen3RequestDriver, retired_native_state

    lease = preflight(frozen)
    if Path(_paged_kv_native.__file__).resolve() != KERNEL.resolve():
        raise RuntimeError("loaded native binding differs from frozen path")
    if not mx.metal.is_available():
        raise RuntimeError("Metal is unavailable")
    mx.set_default_device(mx.gpu)
    signal.alarm(120)
    driver = apc = batch = owner = candidate = hit = None
    uid = None
    installed = False
    lock = threading.RLock()
    try:
        driver = Qwen3RequestDriver({"paths": {"artifact": str(ARTIFACT)}})
        model = driver.model
        prefix = tuple([1000] * 63)
        prompt = prefix + (1001,)
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([prefix], dtype=mx.int32), cache=cache))
        if len(cache) != 28 or any(leaf.offset != 63 for leaf in cache):
            raise RuntimeError("ordinary prefix did not fill 28 KV leaves")
        apc = APCv2(max_size=4, max_bytes=2 << 30, layout_name="qwen3-fp16-kv")
        key = APCKey(model="Qwen3-0.6B", revision=driver.revision,
                     cache_layout_fingerprint="qwen3-fp16-kv")
        stored = apc.store(key, prefix, cache)
        if not stored.stored:
            raise RuntimeError("APCv2 refused exact ordinary prefix")
        hit = apc.lookup(key, prompt, allow_disk_restore=False)
        if hit.cached_tokens != 63 or tuple(hit.remaining_tokens) != (1001,):
            raise RuntimeError("APCv2 did not return exact 63-token prefix")
        store_before = apc.apc_stats["stores"]
        owner, candidate = driver.adapter.create_native_paged_qwen3_request(
            revision=driver.revision, prompt_tokens=64, max_tokens=2,
            apc_cache=hit.cache, cached_tokens=63, permit_candidate=True)
        with owner.snapshot() as view:
            if view.offset != 63 or view.generation != 0:
                raise RuntimeError("native restore did not publish exact prefix")
        batch = BatchGenerator(model, max_tokens=2, prefill_batch_size=1,
                               completion_batch_size=1, prefill_step_size=1,
                               stop_tokens=[])
        with lock:
            uid = batch.insert([[1001]], max_tokens=[2], caches=[hit.cache],
                               all_tokens=[list(prefix)],
                               samplers=[lambda lp: mx.argmax(lp, axis=-1)])[0]
        probe = run_research_native_queued_qwen3(
            batch, lock, CandidateRequest(uid, driver.revision, 1, ("kv",)),
            owner, candidate, research_permit=True)
        if not probe.research_executed or probe.reason != "research_executed":
            raise RuntimeError(f"warm research suffix refused: {probe.reason}")
        prepared = prepare_queued_native_first_response(batch, lock, owner, probe)
        handoff = batch.install_native_queued(
            prepared, owner, candidate, lock, permit_native=True,
            research_only=True)
        if handoff["selected"] or handoff["reason"] != "native_installed":
            raise RuntimeError("warm research handoff was not private")
        installed = True
        reference = model(mx.array([prompt], dtype=mx.int32))[0, -1]
        mx.eval(reference)
        responses = []
        for _ in range(4):
            with lock:
                _prompts, produced = batch.next()
            if batch.take_lane_failures():
                raise RuntimeError("warm native lane failed")
            responses.extend(produced)
            if len(responses) >= 2:
                break
        if len(responses) != 2:
            raise RuntimeError("warm native request did not emit two responses")
        first = int(responses[0].token)
        expected_first = int(mx.argmax(reference).item())
        expected_second = int(mx.argmax(model(mx.array(
            [prompt + (first,)], dtype=mx.int32))[0, -1]).item())
        if [int(response.token) for response in responses] != [expected_first, expected_second]:
            raise RuntimeError("warm native tokens differ from ordinary reference")
        native_lp = np.asarray(responses[0].logprobs, dtype=np.float32)
        ordinary_lp = np.asarray(reference - mx.logsumexp(reference), dtype=np.float32)
        nrms = float(np.sqrt(np.mean((native_lp - ordinary_lp) ** 2)) /
                     max(np.sqrt(np.mean(ordinary_lp ** 2)), 1e-12))
        if nrms > 0.02:
            raise RuntimeError(f"warm native first-token logprobs differ: {nrms}")
        receipt = dict(responses[-1].mtp_receipt or {})
        if (receipt.get("selected") is not False or
                receipt.get("observed_used") is not False or
                receipt.get("research_executed") is not True or
                receipt.get("apcv2_restored_tokens") != 63 or
                receipt.get("native_read_calls") != 56 or
                receipt.get("terminal_successes") != 56):
            raise RuntimeError("warm native response receipt lacks real terminal proof")
        with lock:
            batch.remove([uid])
        uid = None
        mx.synchronize()
        pending, pages, released = retired_native_state(candidate.backend.writer)
        if not released or not owner.fully_retired or apc.apc_stats["stores"] != store_before:
            raise RuntimeError("warm native request leaked state or wrote another APCv2 entry")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak > 24 * (1 << 30):
            raise RuntimeError("warm native gate exceeded 24 GiB resident cap")
        return {"schema": "mlx2.varlen-apcv2-warm-native-gate.v1",
                "status": "passed", "identity": frozen, "lease": lease,
                "apcv2_cached_tokens": 63, "apcv2_stores": store_before,
                "native_output_tokens": [int(r.token) for r in responses],
                "ordinary_output_tokens": [expected_first, expected_second],
                "first_logprob_nrms": nrms, "route_receipt": receipt,
                "pending_native_epochs": pending, "retained_pages": pages,
                "owner_fully_retired": owner.fully_retired,
                "peak_resident_bytes": peak, "qualified": False}
    finally:
        signal.alarm(0)
        if batch is not None:
            if uid is not None:
                with lock:
                    batch.remove([uid])
            batch.close()
        if owner is not None and not installed:
            from mlx2.runtime.paged_apcv2_native_restore import retire_failed_native_restore

            retire_failed_native_restore(owner, candidate.backend.writer)
        if hit is not None and hit.cache is not None and hasattr(hit.cache, "close"):
            hit.cache.close()
        if apc is not None:
            apc.close()
        if driver is not None:
            driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.freeze:
        args.freeze.write_text(json.dumps(identity(), indent=2) + "\n")
        return
    if not args.preflight:
        parser.error("--preflight is required for GPU execution")
    frozen = json.loads(args.preflight.read_text())
    if not args.execute_gpu:
        print(json.dumps({"status": "planned", "identity": frozen,
                          "gpu_executed": False}, indent=2))
        return
    if not args.receipt:
        parser.error("--receipt is required for GPU execution")
    try:
        result = run(frozen)
    except BaseException as exc:
        result = {"schema": "mlx2.varlen-apcv2-warm-native-gate.v1",
                  "status": "failed", "identity": frozen,
                  "error_type": type(exc).__name__, "error": str(exc),
                  "qualified": False}
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
