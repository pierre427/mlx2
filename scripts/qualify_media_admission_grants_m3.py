"""Bounded M3 integration check for cold media feature admission grants.

The caller owns the CPG gpu:m3 lease. This process acquires the host lock.
It substitutes a small headroom reading after model load; it does not create
physical memory pressure or qualify concurrent multimodal execution.
"""

import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path


ROOT = Path("/tmp/mlx2-candidate-m3.oOgHrM")
LOCK = "/Users/Shared/mlxuag/gpu.lock"
GIB = 1 << 30


def image_uri(color):
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def request(color, *, cohort=None):
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe this image briefly."},
            {"type": "input_image", "image_url": image_uri(color)},
        ]}],
        "temperature": 0,
        "max_tokens": 4,
    }
    if cohort is not None:
        body["batch_cohort"] = dict(cohort)
    return body


def collect(job):
    while True:
        event = job.events.get(timeout=180)
        if "error" in event:
            return {"error": event["error"], "status": event.get("status")}
        if "finish_reason" in event:
            receipt = event.get("receipt") or {}
            return {"finish_reason": event["finish_reason"],
                    "route": receipt.get("route"),
                    "qualification": receipt.get("qualification"),
                    "cached_tokens": receipt.get("cached_tokens"),
                    "prompt_tokens": receipt.get("prompt_tokens"),
                    "completion_tokens": receipt.get("completion_tokens")}


def run(report):
    import mlx.core as mx
    import mlx2

    # The receipt hashes files under ROOT/src; refuse to run any other mlx2.
    package = Path(mlx2.__file__).resolve().parent
    if package != (ROOT / "src" / "mlx2").resolve():
        raise AssertionError(f"executing mlx2 from {package}, not {ROOT / 'src' / 'mlx2'}")
    from mlx2 import memory, serving

    mx.set_default_device(mx.gpu)
    actual_headroom = memory.execution_headroom
    override = {"bytes": None}

    def bounded_headroom(**kwargs):
        if override["bytes"] is not None:
            return override["bytes"]
        return actual_headroom(**kwargs)

    memory.execution_headroom = bounded_headroom
    original_required = serving.lane_admission_required_gib
    requirements = []

    def record_required(controller, **kwargs):
        value = original_required(controller, **kwargs)
        requirements.append({"gib": value, "reserve_gib": controller.hard_reserve_gib,
                             "prefill_gib": kwargs.get("prefill_gib"),
                             "context_tokens": kwargs.get("context_tokens")})
        return value

    serving.lane_admission_required_gib = record_required
    engine = None
    try:
        engine = serving.ServingEngine(
            str(ROOT / "models" / "smol"), qualification_mode=True, mtp=False,
            max_lanes=2, max_inflight=2, max_context=4096, prefill_step=256,
            cache_bytes=1 << 30,
            execution_policy={"vision_feature_reuse": "tower_only_candidate_v1"},
        )
        if not engine.ready.wait(150) or engine.error:
            raise RuntimeError(f"engine load failed: {engine.error}")
        calibration = collect(engine.submit(request((30, 70, 140))))
        if "error" in calibration or not requirements:
            raise AssertionError(f"cold calibration failed: {calibration}")
        reserve = requirements[-1]["reserve_gib"]
        required = requirements[-1]["gib"]
        lane = required - reserve
        if not lane > 0 or not requirements[-1]["prefill_gib"] > 0:
            raise AssertionError(f"cold media feature cost missing: {requirements}")
        # One lane fits, two attached lanes cannot spend the same measurement.
        override["bytes"] = int((reserve + 1.5 * lane) * GIB)
        before = dict(engine.counts)
        feature_before = vars(engine.adapter._vision_feature_reuse_counters).copy()
        cohort = {"id": "two-cold-smol-images", "size": 2}
        jobs = [engine.submit(request(color, cohort=cohort)) for color in
                ((70, 30, 120), (130, 90, 35))]
        results = [collect(job) for job in jobs]
        deferred = (engine.counts["memory_admission_deferred_behind_grants"]
                    - before.get("memory_admission_deferred_behind_grants", 0))
        feature_after = vars(engine.adapter._vision_feature_reuse_counters).copy()
        if not all(row.get("status") == 429 for row in results) or deferred < 1:
            raise AssertionError(f"cohort did not fail behind grant: {results}, {deferred}")
        if feature_after != feature_before:
            raise AssertionError("refused cohort executed or stored media features")
        override["bytes"] = None
        recovery = collect(engine.submit(request((60, 130, 50))))
        if "error" in recovery:
            raise AssertionError(f"serial recovery failed: {recovery}")
        report.update({
            "ok": True, "model": str(ROOT / "models" / "smol"),
            "model_config_sha256": hashlib.sha256(
                (ROOT / "models" / "smol" / "config.json").read_bytes()
            ).hexdigest(),
            "request_media_sha256": [hashlib.sha256(image_uri(color).encode()).hexdigest()
                                     for color in ((30, 70, 140), (70, 30, 120),
                                                   (130, 90, 35), (60, 130, 50))],
            "calibration": calibration, "required_gib": required,
            "hard_reserve_gib": reserve, "cold_prefill_gib": requirements[-1]["prefill_gib"],
            "synthetic_headroom_bytes": int((reserve + 1.5 * lane) * GIB),
            "cohort": results, "deferred_behind_grants": deferred,
            "feature_counters_before_cohort": feature_before,
            "feature_counters_after_cohort": feature_after,
            "serial_recovery": recovery, "counts": dict(engine.counts),
            "source_sha256": {name: hashlib.sha256((ROOT / "src" / "mlx2" / name).read_bytes()).hexdigest()
                              for name in ("serving.py", "adapters/pinned_vlm_candidate.py")},
        })
    finally:
        if engine is not None:
            engine.close()
        serving.lane_admission_required_gib = original_required
        memory.execution_headroom = actual_headroom


if __name__ == "__main__":
    report = {"schema": "mlx2.m3-media-admission-grants.v1", "ok": False}
    descriptor = None
    owned = False
    try:
        descriptor = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        owned = True
        os.write(descriptor, (f"mlx2 M3 media grant gate pid={os.getpid()} "
                              f"CPG generation={os.environ.get('MLX2_CPG_GENERATION', 'unbound')}\n").encode())
        os.close(descriptor)
        descriptor = None
        run(report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        print(json.dumps(report, indent=2, default=str), flush=True)
        if descriptor is not None:
            os.close(descriptor)
        if owned:
            os.unlink(LOCK)
    sys.exit(0 if report["ok"] else 1)
