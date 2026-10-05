"""Source-bound, unselected APCv2-63 warm Qwen3 complete-request calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import subprocess
import time
from pathlib import Path
from threading import RLock

from varlen_pack_price_bench import _gpuq_owner, verify_artifact_manifest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "mlx2.paged-pack-price.v1"
SCOPE = "research_live_warm_request_calibration"
MAX_SECONDS = 180
MAX_BYTES = 24 * (1 << 30)
REPEATS = 3
STAGE = "not_started"
COMPLETED_PAIRS: list[dict] = []
CURRENT_PAIR: dict = {}


def preflight(manifest: dict) -> dict:
    from mlx2.runtime.paged_price_identity import compute_live_price_identity

    if (manifest.get("context_tokens") != [64] or
            manifest.get("request_shape") != {"prompt_tokens": 64,
                                              "cached_tokens": 63,
                                              "suffix_rows": 1,
                                              "output_tokens": 2,
                                              "decode_rows_after_prefill": 1} or
            manifest.get("hard_seconds") != MAX_SECONDS or
            manifest.get("max_resident_bytes") != MAX_BYTES or
            not isinstance(manifest.get("profile_id"), str) or
            not manifest["profile_id"]):
        raise RuntimeError("exact warm request calibration profile required")
    paths = manifest.get("paths")
    if not isinstance(paths, dict) or set(paths) != {"artifact", "mlx_wheel", "kernel"}:
        raise RuntimeError("exact warm source paths required")
    artifact = verify_artifact_manifest(Path(paths["artifact"]))
    live = compute_live_price_identity(
        paths["artifact"], paths["mlx_wheel"], paths["kernel"],
        adapter_artifact_root=artifact["root"])
    if manifest.get("identity") != live:
        raise RuntimeError("warm source, model, wheel, or kernel identity differs")
    config = json.loads((Path(artifact["root"]) / "config.json").read_text())
    if (config.get("model_type") != "qwen3" or
            config.get("num_hidden_layers") != 28 or
            config.get("num_attention_heads") != 16 or
            config.get("num_key_value_heads") != 8 or
            config.get("head_dim") != 128 or config.get("rope_scaling") is not None):
        raise RuntimeError("warm Qwen3 artifact geometry differs")
    return {"status": "preflight_passed", "gpu_executed": False,
            "identity": live, "context_tokens": [64]}


class WarmRequestDriver:
    def __init__(self, manifest: dict):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx2.adapters.standard_decoder import StandardDecoderAdapter
        from mlx2.runtime.apc_v2 import APCKey, APCv2
        from mlx2.runtime.models.cache import make_prompt_cache

        artifact = verify_artifact_manifest(Path(manifest["paths"]["artifact"]))
        self.adapter = StandardDecoderAdapter(artifact["root"])
        self.model = self.adapter.model
        self.model.apply(lambda value: value.astype(mx.float16))
        mx.eval(self.model.parameters())
        if (len(self.model.layers) != 28 or
                any(value.dtype != mx.float16
                    for _, value in tree_flatten(self.model.parameters()))):
            raise RuntimeError("warm paired requests require pinned fp16 Qwen3")
        self.mx = mx
        self.revision = self.adapter.identity["fingerprint"]
        self.prefix = tuple([1000] * 63)
        self.prompt = self.prefix + (1001,)
        self.apc = APCv2(max_size=4, max_bytes=2 << 30,
                         layout_name="qwen3-fp16-kv")
        self.key = APCKey(model="Qwen3-0.6B", revision=self.revision,
                          cache_layout_fingerprint="qwen3-fp16-kv")
        cache = make_prompt_cache(self.model)
        mx.eval(self.model(mx.array([self.prefix], dtype=mx.int32), cache=cache))
        if not self.apc.store(self.key, self.prefix, cache).stored:
            raise RuntimeError("ordinary APCv2 prefix priming failed")

    def close(self):
        self.apc.close()
        self.adapter.close()

    def run_request(self, arm: str) -> dict:
        import mlx.core as mx
        from mlx2.runtime.generate import BatchGenerator
        from mlx2.runtime.paged_native_batch_lifecycle import (
            prepare_queued_native_first_response, run_research_native_queued_qwen3,
        )
        from mlx2.runtime.paged_request_transaction import CandidateRequest
        from varlen_live_qwen3_request_driver import retired_native_state

        if arm not in ("ordinary", "paged"):
            raise ValueError("exact ordinary or paged warm arm required")
        stores_before = self.apc.apc_stats["stores"]
        hit = self.apc.lookup(self.key, self.prompt, allow_disk_restore=False)
        if hit.cached_tokens != 63 or tuple(hit.remaining_tokens) != (1001,):
            raise RuntimeError("exact APCv2-63 warm hit missing")
        lock = RLock()
        batch = BatchGenerator(self.model, max_tokens=2, prefill_batch_size=1,
                               completion_batch_size=1, prefill_step_size=1,
                               stop_tokens=[])
        owner = candidate = uid = None
        installed = False
        model_type = type(self.model)
        original_call = model_type.__call__
        forward_calls = 0

        def count_forward(instance, *args, **kwargs):
            nonlocal forward_calls
            if instance is self.model:
                forward_calls += 1
            return original_call(instance, *args, **kwargs)

        model_type.__call__ = count_forward
        responses = []
        q1_start = q1_ms = None
        try:
            with lock:
                uid = batch.insert([[1001]], max_tokens=[2], caches=[hit.cache],
                                   all_tokens=[list(self.prefix)],
                                   samplers=[lambda lp: mx.argmax(lp, axis=-1)])[0]
            if arm == "paged":
                owner, candidate = self.adapter.create_native_paged_qwen3_request(
                    revision=self.revision, prompt_tokens=64, max_tokens=2,
                    apc_cache=hit.cache, cached_tokens=63, permit_candidate=True)
                with owner.snapshot() as view:
                    if view.offset != 63:
                        raise RuntimeError("APCv2 prefix not restored to native owner")
                probe = run_research_native_queued_qwen3(
                    batch, lock, CandidateRequest(uid, self.revision, 1, ("kv",)),
                    owner, candidate, research_permit=True)
                if not probe.research_executed or probe.reason != "research_executed":
                    raise RuntimeError(f"warm research suffix refused: {probe.reason}")
                prepared = prepare_queued_native_first_response(batch, lock, owner, probe)
                handoff = batch.install_native_queued(
                    prepared, owner, candidate, lock, permit_native=True,
                    research_only=True)
                if (handoff.get("reason") != "native_installed" or
                        handoff.get("selected") is not False or
                        handoff.get("research_only") is not True):
                    raise RuntimeError("warm research handoff did not install privately")
                installed = True
            for _ in range(6):
                if len(responses) == 1 and q1_start is None:
                    q1_start = time.perf_counter_ns()
                with lock:
                    _, produced = batch.next()
                if batch.take_lane_failures():
                    raise RuntimeError("warm BatchGenerator lane failed")
                for response in produced:
                    if response.uid != uid:
                        raise RuntimeError("warm response UID changed")
                    mx.eval(response.logprobs)
                    mx.synchronize()
                    responses.append(response)
                    if len(responses) == 2:
                        q1_ms = (time.perf_counter_ns() - q1_start) / 1e6
                if len(responses) >= 2:
                    break
            if len(responses) != 2 or q1_ms is None or q1_ms <= 0:
                raise RuntimeError("two warm responses and q1 timing required")
            first = dict(responses[0].mtp_receipt or {})
            second = dict(responses[1].mtp_receipt or {})
            with lock:
                batch.remove([uid])
            uid = None
            mx.synchronize()
            if arm == "paged":
                pending, retained, released = retired_native_state(candidate.backend.writer)
                released = (released and owner.fully_retired and
                            not candidate.backend.writer.poisoned)
                route = {**second, "serving_selected": False}
                reads = candidate.backend.read_submissions
                terminals = candidate.backend.terminal_successes
            else:
                pending = retained = reads = terminals = 0
                released = True
                route = {"route": "ordinary", "selected": True,
                         "observed_used": True}
            stores_delta = self.apc.apc_stats["stores"] - stores_before
            if stores_delta:
                raise RuntimeError("warm calibration changed APCv2 store")
            return {"arm": arm, "request_inserted": True, "admitted": True,
                    "cache_transaction_published": True,
                    "response_emitted": True, "sampled_output": True,
                    "synchronized": True, "request_removed": True,
                    "request_state_released": released,
                    "output_token_ids": [int(response.token) for response in responses],
                    "model_layers": 28, "peak_resident_bytes": resource.getrusage(
                        resource.RUSAGE_SELF).ru_maxrss,
                    "route_receipt": route, "first_response_receipt": first,
                    "second_response_receipt": second,
                    "apcv2_cached_tokens": 63, "apcv2_stores_delta": stores_delta,
                    "ordinary_model_forward_calls": forward_calls,
                    "paged_read_calls": reads, "terminal_successes": terminals,
                    "pending_native_epochs": pending, "retained_pages": retained,
                    "owner_fully_retired": bool(owner.fully_retired) if owner else True,
                    "writer_poisoned": bool(candidate.backend.writer.poisoned)
                    if candidate else False,
                    "q1_step_ms": q1_ms}
        finally:
            model_type.__call__ = original_call
            if uid is not None:
                with lock:
                    batch.remove([uid])
            batch.close()
            if owner is not None and not installed:
                from mlx2.runtime.paged_apcv2_native_restore import retire_failed_native_restore

                retire_failed_native_restore(owner, candidate.backend.writer)
            if hit.cache is not None and hasattr(hit.cache, "close"):
                hit.cache.close()


def execute(manifest: dict) -> dict:
    global STAGE, CURRENT_PAIR
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("warm calibration exceeded 180 seconds")))
    signal.alarm(MAX_SECONDS)
    driver = None
    try:
        STAGE = "source_preflight"
        preflight(manifest)
        STAGE = "gpu_owner"
        owner = _gpuq_owner()
        import mlx.core as mx
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise RuntimeError("explicit MLX GPU required")
        STAGE = "model_load"
        driver = WarmRequestDriver(manifest)
        native_ms, ordinary_ms, native_q1, ordinary_q1, pairs = [], [], [], [], []
        for index in range(REPEATS):
            pair = {}
            CURRENT_PAIR = pair
            for arm in ("ordinary", "paged"):
                STAGE = f"pair_{index}_{arm}"
                start = time.perf_counter_ns()
                proof = driver.run_request(arm)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if elapsed <= 0 or not math.isfinite(elapsed):
                    raise RuntimeError("invalid warm complete-request duration")
                pair[arm] = proof
                (ordinary_ms if arm == "ordinary" else native_ms).append(elapsed)
                (ordinary_q1 if arm == "ordinary" else native_q1).append(
                    proof["q1_step_ms"])
            if pair["ordinary"]["output_token_ids"] != pair["paged"]["output_token_ids"]:
                raise RuntimeError("warm ordinary/native output tokens differ")
            pairs.append(pair)
            COMPLETED_PAIRS.append(pair)
            CURRENT_PAIR = {}
        STAGE = "receipt_validation"
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak > MAX_BYTES:
            raise MemoryError("warm calibration exceeded 24 GiB resident cap")
        receipt = {"schema": SCHEMA, "status": "calibrated", "gpu_executed": True,
                   "measurement_scope": SCOPE, "profile_id": manifest["profile_id"],
                   "measured_route": "native_qwen3_paged", "identity": manifest["identity"],
                   "context_tokens": [64], "gpuq_owner": owner,
                   "request_shape": manifest["request_shape"],
                   "hard_seconds": MAX_SECONDS, "max_resident_bytes": MAX_BYTES,
                   "peak_resident_bytes": peak,
                   "cases": [{"warm_native_request_ms": native_ms,
                              "warm_ordinary_request_ms": ordinary_ms,
                              "native_q1_step_ms": native_q1,
                              "ordinary_q1_step_ms": ordinary_q1,
                              "request_proofs": pairs,
                              "kernel_engagement": {
                                  "paged_read_calls": sum(pair["paged"]["paged_read_calls"]
                                                          for pair in pairs),
                                  "terminal_successes": sum(
                                      pair["paged"]["terminal_successes"]
                                      for pair in pairs)}}],
                   "qualified": False, "selected": False, "serving_selected": False,
                   "research_executed": True, "price_usable": False}
        encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        receipt["research_warm_evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
        # Validation uses the exact loader but the receipt stays unselected.
        STAGE = "complete"
        return receipt
    finally:
        if driver is not None:
            driver.close()
        signal.alarm(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not args.execute_gpu:
        result = preflight(manifest)
    else:
        if args.receipt is None:
            parser.error("GPU execution requires --receipt")
        try:
            result = execute(manifest)
        except BaseException as error:
            result = {"schema": SCHEMA, "status": "failed",
                      "gpu_executed": STAGE not in (
                          "not_started", "source_preflight", "gpu_owner"),
                      "identity": manifest.get("identity"),
                      "failure_stage": STAGE,
                      "completed_pairs": COMPLETED_PAIRS,
                      "current_pair": CURRENT_PAIR,
                      "error_type": type(error).__name__, "error": str(error),
                      "qualified": False, "selected": False, "price_usable": False}
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            raise
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        if args.execute_gpu:
            from mlx2.runtime.paged_pack_price import load_research_warm_calibration

            load_research_warm_calibration(
                args.receipt, live_identity=manifest["identity"],
                context_tokens=(64,))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
