"""One explicit Qwen3 B1 serving-route spot after exact research calibration.

Run only from a clean, frozen source revision under a short shared gpuq lease.
This tests the selected BatchGenerator seam, not HTTP or full qualification.
"""

from __future__ import annotations

import argparse
import json
import resource
import signal
import threading
from pathlib import Path
from types import SimpleNamespace

from varlen_live_request_price import preflight_cpu
from varlen_live_qwen3_request_driver import Qwen3RequestDriver, retired_native_state
from varlen_pack_price_bench import MAX_RESIDENT_BYTES, _gpuq_owner


def _timeout(_signum, _frame):
    raise TimeoutError("selected native Qwen3 spot exceeded 120 seconds")


def execute(manifest: dict, calibration_path: Path) -> dict:
    """Require real native response and cleanup evidence for one selected lane."""
    preflight_cpu(manifest)
    identity, gpuq_owner = manifest["identity"], _gpuq_owner()
    import _paged_kv_native
    import mlx.core as mx
    from mlx2 import serving
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.paged_pack_price import load_research_calibration
    from mlx2.runtime.paged_price_identity import compute_live_price_identity

    kernel = Path(_paged_kv_native.__file__).resolve()
    if kernel != Path(manifest["paths"]["kernel"]).resolve():
        raise RuntimeError("loaded native binary differs from pinned manifest")
    calibration = load_research_calibration(
        calibration_path, live_identity=identity, context_tokens=(63,))
    if calibration.profile_id != manifest["profile_id"]:
        raise RuntimeError("research calibration profile differs")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("GPU execution required")
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(120)
    driver = None
    batch = None
    uid = None
    captured = {}
    try:
        driver = Qwen3RequestDriver(manifest)
        adapter = driver.adapter
        live = compute_live_price_identity(
            manifest["paths"]["artifact"], manifest["paths"]["mlx_wheel"],
            kernel, adapter_artifact_root=adapter.identity["path"])
        if live != identity:
            raise RuntimeError("loaded model/source/wheel identity differs")
        factory = adapter.create_native_paged_qwen3_request

        def create(**kwargs):
            owner, candidate = factory(**kwargs)
            captured.update(owner=owner, candidate=candidate)
            return owner, candidate

        facade = SimpleNamespace(identity=adapter.identity,
                                 create_native_paged_qwen3_request=create)
        lock = threading.RLock()
        batch = BatchGenerator(driver.model, max_tokens=2,
                               prefill_batch_size=1, completion_batch_size=1,
                               prefill_step_size=63, stop_tokens=[])
        with lock:
            uid = batch.insert([list(driver.prompt)], max_tokens=[2])[0]
        job = SimpleNamespace(uid=uid, preempted=False, cached_tokens=0,
                              request={"paged_native_qwen3": True,
                                       "skip_writing_prefix_cache": True,
                                       "temperature": 0, "max_tokens": 2},
                              cancelled=threading.Event())
        attached = serving.install_explicit_native_qwen3_request(
            batch, facade, job, prompt_tokens=list(driver.prompt), maximum=2,
            lifecycle_lock=lock, price_path=calibration_path,
            manifest_path=manifest["paths"]["artifact"],
            mlx_wheel_path=manifest["paths"]["mlx_wheel"])
        if (attached.get("selected") is not True or
                attached.get("qualified") is not False or
                attached.get("observed_used") is not False or
                attached.get("price_provenance") != "research_calibrated"):
            raise RuntimeError("explicit native route did not select honestly")
        responses = []
        for _ in range(6):
            with lock:
                _, produced = batch.next()
            if batch.take_lane_failures():
                raise RuntimeError("selected native lane failed")
            for response in produced:
                if response.uid != uid:
                    raise RuntimeError("selected response UID drifted")
                mx.eval(response.logprobs)
                responses.append(response)
            if len(responses) == 2:
                break
        mx.synchronize()
        if len(responses) != 2 or [r.token for r in responses] != [14582, 198]:
            raise RuntimeError("selected route output differs from paired reference")
        first, second = (dict(r.mtp_receipt or {}) for r in responses)
        for receipt, reads in ((first, 28), (second, 56)):
            if (receipt.get("route") != "native_qwen3_paged" or
                    receipt.get("qualified") is not False or
                    receipt.get("selected") is not True or
                    receipt.get("observed_used") is not True or
                    receipt.get("research_executed") is not False or
                    receipt.get("price_provenance") != "research_calibrated" or
                    receipt.get("price_evidence_sha256") != calibration.evidence_sha256 or
                    receipt.get("native_read_calls") != reads or
                    receipt.get("terminal_successes") != reads or
                    receipt.get("native_reader_lease") != "held_through_token_step" or
                    receipt.get("apcv2") != "native_checkpoint_unavailable" or
                    receipt.get("ordinary_forward_calls") != 0):
                raise RuntimeError("selected response/reader/terminal proof missing")
        with lock:
            batch.remove([uid])
        uid = None
        mx.synchronize()
        owner, writer = captured["owner"], captured["candidate"].backend.writer
        pending, retained, released = retired_native_state(writer)
        if not released or not owner.fully_retired:
            raise RuntimeError("selected native request retained pages or epochs")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak > MAX_RESIDENT_BYTES:
            raise RuntimeError("selected native request exceeded memory cap")
        return {
            "schema": "mlx2.varlen-selected-qwen3-spot.v1",
            "status": "passed", "gpu_executed": True,
            "identity": identity, "gpuq_owner": gpuq_owner,
            "calibration_evidence_sha256": calibration.evidence_sha256,
            "calibration_scope": "research_calibrated",
            "route": "native_qwen3_paged", "implemented": True,
            "qualified": False, "selected": True, "observed_used": True,
            "output_token_ids": [int(r.token) for r in responses],
            "native_read_calls": second["native_read_calls"],
            "terminal_successes": second["terminal_successes"],
            "reader_lease": second["native_reader_lease"],
            "pending_native_epochs": pending, "retained_pages": retained,
            "owner_fully_retired": owner.fully_retired,
            "apcv2": "cold_no_write", "peak_resident_bytes": peak,
            "qualification_note": "one exact explicit route spot; major B1 q1 regression remains",
        }
    finally:
        signal.alarm(0)
        if batch is not None:
            if uid is not None:
                with lock:
                    batch.remove([uid])
            batch.close()
        if driver is not None:
            driver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    try:
        result = execute(manifest, args.calibration)
    except Exception as exc:
        result = {"schema": "mlx2.varlen-selected-qwen3-spot.v1",
                  "status": "failed", "gpu_executed": None,
                  "qualified": False, "selected": None,
                  "error_type": type(exc).__name__, "error": str(exc)}
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        raise
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
