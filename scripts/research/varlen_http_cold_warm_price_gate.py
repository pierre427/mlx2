"""One default-off loopback HTTP cold/warm parity gate after real calibrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from varlen_live_request_price import preflight_cpu as cold_preflight
from varlen_native_http_gate import _http_post, _prompt_for_63_tokens
from varlen_pack_price_bench import _gpuq_owner, verify_artifact_manifest
from varlen_warm_request_price import preflight as warm_preflight

SCHEMA = "mlx2.varlen-http-cold-warm-price-gate.v1"
MAX_SECONDS = 180
MAX_BYTES = 24 * (1 << 30)
STAGE = "not_started"
OBSERVATIONS: dict = {}


def preflight(cold_manifest: dict, warm_manifest: dict) -> dict:
    cold = cold_preflight(cold_manifest)
    warm = warm_preflight(warm_manifest)
    if cold["identity"] != warm["identity"]:
        raise RuntimeError("cold and warm calibrations have different live sources")
    if cold_manifest["paths"] != warm_manifest["paths"]:
        raise RuntimeError("cold and warm model/wheel/native paths differ")
    return {"schema": SCHEMA, "status": "preflight_passed",
            "gpu_executed": False, "identity": cold["identity"],
            "cold_context_tokens": [63], "warm_context_tokens": [64],
            "max_resident_bytes": MAX_BYTES, "hard_seconds": MAX_SECONDS}


def _warm_prompt(adapter, cold_text: str, cold_tokens: list[int]) -> tuple[str, list[int]]:
    tokenizer = adapter.tokenizer
    candidates = [tokenizer.decode(cold_tokens + [token])
                  for token in range(1000, 1100)]
    candidates.extend(cold_text + suffix for suffix in
                      (" x", " z", "!", ".", " 1", " hello"))
    for text in candidates:
        tokens = list(adapter.prompt_tokens({"prompt": text}))
        if len(tokens) == 64 and tokens[:63] == cold_tokens:
            return text, tokens
    raise RuntimeError("no round-trip 64-token prompt extends exact APCv2-63 prefix")


def _route(body: dict) -> dict:
    receipt = body.get("mlx2", {}).get("route_receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("HTTP route receipt missing")
    return receipt


def _native_response(body: dict, *, cached_tokens: int, evidence: str,
                     samples: list) -> dict:
    receipt = _route(body)
    if (body.get("mlx2", {}).get("route") != "native_qwen3_paged" or
            body["mlx2"].get("cached_tokens") != cached_tokens or
            receipt.get("route") != "native_qwen3_paged" or
            receipt.get("implemented") is not True or
            receipt.get("qualified") is not False or
            receipt.get("selected") is not True or
            receipt.get("observed_used") is not True or
            receipt.get("price_provenance") != "research_calibrated" or
            receipt.get("price_evidence_sha256") != evidence or
            receipt.get("ordinary_forward_calls") != 0 or
            receipt.get("native_read_calls") != 56 or
            receipt.get("terminal_successes") != 56 or
            len(samples) != 2 or
            samples[0][1].get("native_read_calls") != 28 or
            samples[1][1].get("native_read_calls") != 56 or
            samples[1][1].get("terminal_successes") != 56 or
            (cached_tokens == 63 and
             receipt.get("apcv2_restored_tokens") != 63)):
        raise RuntimeError("HTTP native selected/used/read/restore proof missing")
    return receipt


def _ordinary_response(body: dict, label: str) -> None:
    expected = {"cold_ordinary": 0, "warm_ordinary": 63}.get(label)
    if (expected is None or body.get("mlx2", {}).get("route") != "ordinary" or
            body.get("mlx2", {}).get("cached_tokens") != expected):
        raise RuntimeError(f"{label} ordinary route or APCv2 hit differs")


def execute(cold_manifest: dict, warm_manifest: dict,
            cold_price_path: Path, warm_price_path: Path) -> dict:
    global STAGE
    import _paged_kv_native
    import mlx.core as mx
    from mlx2 import serving
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.paged_pack_price import (
        load_research_calibration, load_research_warm_calibration,
    )
    from mlx2.runtime.paged_price_identity import compute_live_price_identity
    from mlx2.server import handler_for
    from varlen_live_qwen3_request_driver import retired_native_state

    STAGE = "source_preflight"
    frozen = preflight(cold_manifest, warm_manifest)
    STAGE = "gpu_owner"
    lease = _gpuq_owner()
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("explicit MLX GPU required")
    kernel = Path(_paged_kv_native.__file__).resolve()
    if kernel != Path(cold_manifest["paths"]["kernel"]).resolve():
        raise RuntimeError("loaded native extension differs from frozen source")
    cold_price = load_research_calibration(
        cold_price_path, live_identity=frozen["identity"], context_tokens=(63,))
    warm_price = load_research_warm_calibration(
        warm_price_path, live_identity=frozen["identity"], context_tokens=(64,))
    if cold_price.profile_id != cold_manifest["profile_id"] or \
            warm_price.profile_id != warm_manifest["profile_id"]:
        raise RuntimeError("HTTP calibration profile differs")
    artifact = verify_artifact_manifest(Path(cold_manifest["paths"]["artifact"]))
    model_root = Path(artifact["root"])
    captured = []
    sampled = []
    original_next = BatchGenerator.next

    def observed_next(batch, *args, **kwargs):
        pending, responses = original_next(batch, *args, **kwargs)
        sampled.extend((int(response.token), dict(response.mtp_receipt or {}))
                       for response in responses)
        return pending, responses

    class GateAdapter(StandardDecoderAdapter):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.model.apply(lambda value: value.astype(mx.float16))
            mx.eval(self.model.parameters())
            if len(self.model.layers) != 28:
                raise RuntimeError("HTTP gate requires pinned Qwen3-0.6B")

        def create_native_paged_qwen3_request(self, **kwargs):
            result = super().create_native_paged_qwen3_request(**kwargs)
            captured.append(result)
            return result

    old_env = {key: os.environ.get(key) for key in (
        "MLX2_NATIVE_PAGED_PRICE", "MLX2_NATIVE_PAGED_MANIFEST",
        "MLX2_NATIVE_PAGED_MLX_WHEEL")}
    os.environ.update({"MLX2_NATIVE_PAGED_PRICE": str(cold_price_path),
                       "MLX2_NATIVE_PAGED_MANIFEST": cold_manifest["paths"]["artifact"],
                       "MLX2_NATIVE_PAGED_MLX_WHEEL": cold_manifest["paths"]["mlx_wheel"]})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("HTTP cold/warm gate exceeded 180 seconds")))
    signal.alarm(MAX_SECONDS)
    engine = server = thread = None
    BatchGenerator.next = observed_next
    try:
        STAGE = "http_model_load"
        engine = serving.ServingEngine(
            str(model_root), adapter_factory=GateAdapter, max_lanes=1,
            max_inflight=1, mtp=False, qualification_mode=True,
            prefill_step=63, cache_bytes=1 << 29)
        if not engine.ready.wait(30) or engine.error:
            raise RuntimeError(f"HTTP model did not load: {engine.error}")
        live = compute_live_price_identity(
            cold_manifest["paths"]["artifact"],
            cold_manifest["paths"]["mlx_wheel"], kernel,
            adapter_artifact_root=engine.adapter.identity["path"])
        if live != frozen["identity"]:
            raise RuntimeError("live HTTP source/artifact/wheel/native identity differs")
        cold_text, cold_tokens = _prompt_for_63_tokens(engine.adapter)
        warm_text, warm_tokens = _warm_prompt(engine.adapter, cold_text, cold_tokens)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        requests = {
            "cold_native": {"model": model_root.name, "prompt": cold_text,
                            "max_tokens": 2, "temperature": 0,
                            "paged_native_qwen3": True,
                            "skip_writing_prefix_cache": True},
            "cold_ordinary": {"model": model_root.name, "prompt": cold_text,
                              "max_tokens": 2, "temperature": 0},
            "warm_native": {"model": model_root.name, "prompt": warm_text,
                            "max_tokens": 2, "temperature": 0,
                            "paged_native_qwen3": True,
                            "skip_writing_prefix_cache": True},
            "warm_ordinary": {"model": model_root.name, "prompt": warm_text,
                              "max_tokens": 2, "temperature": 0,
                              "skip_writing_prefix_cache": True},
        }
        observations = OBSERVATIONS
        for label in ("cold_native", "cold_ordinary", "warm_native", "warm_ordinary"):
            STAGE = label
            if label == "warm_native":
                if engine.apc.lifetime_stats().get("stores", 0) < 1:
                    raise RuntimeError("cold ordinary HTTP did not seed APCv2")
                os.environ["MLX2_NATIVE_PAGED_PRICE"] = str(warm_price_path)
            start_sample = len(sampled)
            start = time.perf_counter_ns()
            status, body = _http_post(base, requests[label])
            duration_ms = (time.perf_counter_ns() - start) / 1e6
            records = sampled[start_sample:]
            observations[label] = {"status": status, "body": body,
                                   "response_receipts": records,
                                   "complete_http_request_ms": duration_ms,
                                   "apcv2_stats": dict(engine.apc.lifetime_stats())}
            if status != 200 or duration_ms <= 0:
                raise RuntimeError(f"{label} HTTP request failed: {status}: {body}")
            if len(records) != 2:
                raise RuntimeError(f"{label} did not emit two responses")
            if label.endswith("native"):
                expected = (cold_price.evidence_sha256 if label == "cold_native"
                            else warm_price.evidence_sha256)
                observations[label]["route_receipt"] = _native_response(
                    body, cached_tokens=0 if label == "cold_native" else 63,
                    evidence=expected, samples=records)
                owner, candidate = captured[-1]
                deadline = time.monotonic() + 3
                while not owner.fully_retired and time.monotonic() < deadline:
                    time.sleep(.005)
                pending, pages, released = retired_native_state(candidate.backend.writer)
                if not released or not owner.fully_retired:
                    raise RuntimeError(f"{label} retained native owner/pages/epochs")
                observations[label]["retirement"] = {"pending_epochs": pending,
                                                     "retained_pages": pages,
                                                     "owner_fully_retired": True}
            else:
                _ordinary_response(body, label)
        if ([token for token, _ in observations["cold_native"]["response_receipts"]] !=
                [token for token, _ in observations["cold_ordinary"]["response_receipts"]] or
                [token for token, _ in observations["warm_native"]["response_receipts"]] !=
                [token for token, _ in observations["warm_ordinary"]["response_receipts"]] or
                observations["cold_native"]["body"]["choices"][0]["text"] !=
                observations["cold_ordinary"]["body"]["choices"][0]["text"] or
                observations["warm_native"]["body"]["choices"][0]["text"] !=
                observations["warm_ordinary"]["body"]["choices"][0]["text"]):
            raise RuntimeError("HTTP cold/warm ordinary/native response parity differs")
        if (observations["cold_native"]["apcv2_stats"].get("stores") != 0 or
                observations["warm_native"]["apcv2_stats"].get("stores") !=
                observations["cold_ordinary"]["apcv2_stats"].get("stores")):
            raise RuntimeError("native HTTP route changed APCv2 store count")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak > MAX_BYTES:
            raise RuntimeError("HTTP gate exceeded 24 GiB resident cap")
        STAGE = "complete"
        return {"schema": SCHEMA, "status": "passed", "gpu_executed": True,
                "identity": frozen["identity"], "gpuq_owner": lease,
                "cold_calibration_evidence_sha256": cold_price.evidence_sha256,
                "warm_calibration_evidence_sha256": warm_price.evidence_sha256,
                "cold_prompt_sha256": hashlib.sha256(cold_text.encode()).hexdigest(),
                "warm_prompt_sha256": hashlib.sha256(warm_text.encode()).hexdigest(),
                "cold_prompt_tokens": len(cold_tokens),
                "warm_prompt_tokens": len(warm_tokens),
                "observations": observations,
                "native_owner_allocations": len(captured),
                "peak_resident_bytes": peak,
                "price_usable": False, "qualified": False,
                "default_selected": False}
    finally:
        signal.alarm(0)
        BatchGenerator.next = original_next
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        if engine is not None:
            engine.close()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-manifest", type=Path, required=True)
    parser.add_argument("--warm-manifest", type=Path, required=True)
    parser.add_argument("--cold-price", type=Path)
    parser.add_argument("--warm-price", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--execute-gpu", action="store_true")
    args = parser.parse_args()
    cold_manifest = json.loads(args.cold_manifest.read_text())
    warm_manifest = json.loads(args.warm_manifest.read_text())
    if args.execute_gpu:
        if not all((args.cold_price, args.warm_price, args.receipt)):
            parser.error("GPU gate requires both real calibrations and failure receipt")
        try:
            result = execute(cold_manifest, warm_manifest,
                             args.cold_price, args.warm_price)
        except BaseException as error:
            result = {"schema": SCHEMA, "status": "failed",
                      "gpu_executed": STAGE not in (
                          "not_started", "source_preflight", "gpu_owner"),
                      "identity": cold_manifest.get("identity"),
                      "failure_stage": STAGE, "observations": OBSERVATIONS,
                      "error_type": type(error).__name__, "error": str(error),
                      "price_usable": False, "qualified": False,
                      "default_selected": False}
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            raise
    else:
        result = preflight(cold_manifest, warm_manifest)
    if args.receipt is not None:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
