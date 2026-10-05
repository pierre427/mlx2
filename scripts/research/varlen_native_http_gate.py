"""Bounded actual-native Qwen3 loopback HTTP functional gate.

The explicit native route stays default-off and unqualified. Run only under a
shared gpuq lease after a clean exact-source calibration has been recorded.
"""

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
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from varlen_live_request_price import preflight_cpu
from varlen_pack_price_bench import MAX_RESIDENT_BYTES, _gpuq_owner

SCHEMA = "mlx2.varlen-native-http-gate.v1"


def _timeout(_signum, _frame):
    raise TimeoutError("native HTTP gate exceeded 120 seconds")


def _http_post(base: str, body: dict) -> tuple[int, dict]:
    request = Request(base + "/v1/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=25) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        with error:
            return error.code, json.load(error)


def _prompt_for_63_tokens(adapter) -> tuple[str, list[int]]:
    """Use real HTTP tokenization; never inject an out-of-band token vector."""
    tokenizer = adapter.tokenizer
    candidates = [tokenizer.decode([1000] * 63)]
    candidates.extend("hello " * count for count in range(45, 85))
    for prompt in candidates:
        tokens = adapter.prompt_tokens({"prompt": prompt})
        if len(tokens) == 63:
            return prompt, list(tokens)
    raise RuntimeError("cannot find a round-trip 63-token HTTP prompt")


def _route(body: dict) -> dict:
    mlx2 = body.get("mlx2")
    if type(mlx2) is not dict or type(mlx2.get("route_receipt")) is not dict:
        raise RuntimeError("HTTP response has no route receipt")
    return mlx2["route_receipt"]


def _assert_native(body: dict, calibration) -> dict:
    receipt = _route(body)
    if (body["mlx2"].get("route") != "native_qwen3_paged" or
            body["mlx2"].get("cached_tokens") != 0 or
            receipt.get("route") != "native_qwen3_paged" or
            receipt.get("implemented") is not True or
            receipt.get("qualified") is not False or
            receipt.get("selected") is not True or
            receipt.get("observed_used") is not True or
            receipt.get("price_provenance") != "research_calibrated" or
            receipt.get("price_evidence_sha256") != calibration.evidence_sha256 or
            receipt.get("apcv2") != "native_checkpoint_unavailable" or
            receipt.get("ordinary_forward_calls") != 0 or
            receipt.get("native_reader_lease") != "held_through_token_step" or
            type(receipt.get("native_read_calls")) is not int or
            receipt["native_read_calls"] < 56 or
            receipt.get("terminal_successes") != receipt["native_read_calls"]):
        raise RuntimeError("native HTTP route/read/terminal receipt is incomplete")
    return receipt


def _assert_refusal(status: int, body: dict, needle: str):
    if status != 400 or needle not in body.get("error", {}).get("message", ""):
        raise RuntimeError(f"expected fail-closed HTTP 400 containing {needle!r}")


def execute(manifest: dict, calibration_path: Path) -> dict:
    preflight_cpu(manifest)
    owner_receipt = _gpuq_owner()
    import _paged_kv_native
    import mlx.core as mx
    from varlen_live_qwen3_request_driver import retired_native_state

    from mlx2 import serving
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.paged_pack_price import load_research_calibration
    from mlx2.runtime.paged_price_identity import compute_live_price_identity
    from mlx2.server import handler_for

    kernel = Path(_paged_kv_native.__file__).resolve()
    if kernel != Path(manifest["paths"]["kernel"]).resolve():
        raise RuntimeError("loaded native binary differs from pinned manifest")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("GPU execution required")
    calibration = load_research_calibration(
        calibration_path, live_identity=manifest["identity"], context_tokens=(63,))
    if calibration.profile_id != manifest["profile_id"]:
        raise RuntimeError("calibration profile differs")

    captured: list[tuple[object, object]] = []
    sampled: list[tuple[int, dict]] = []
    original_next = BatchGenerator.next

    def observed_next(batch, *args, **kwargs):
        pending, responses = original_next(batch, *args, **kwargs)
        sampled.extend((int(response.token), dict(response.mtp_receipt or {}))
                       for response in responses)
        return pending, responses

    class GateAdapter(StandardDecoderAdapter):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.model.apply(lambda parameter: parameter.astype(mx.float16))
            mx.eval(self.model.parameters())
            if (len(self.model.layers) != 28 or
                    self.model.model.embed_tokens.weight.dtype != mx.float16):
                raise RuntimeError("HTTP gate needs pinned fp16 Qwen3-0.6B")

        def create_native_paged_qwen3_request(self, **kwargs):
            result = super().create_native_paged_qwen3_request(**kwargs)
            captured.append(result)
            return result

    adapter_root = Path(manifest["paths"]["artifact"])
    from varlen_pack_price_bench import verify_artifact_manifest
    model_root = Path(verify_artifact_manifest(adapter_root)["root"])
    engine = server = thread = None
    BatchGenerator.next = observed_next
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(120)
    old_environment = {key: os.environ.get(key) for key in (
        "MLX2_NATIVE_PAGED_PRICE", "MLX2_NATIVE_PAGED_MANIFEST",
        "MLX2_NATIVE_PAGED_MLX_WHEEL")}
    os.environ.update({"MLX2_NATIVE_PAGED_PRICE": str(calibration_path),
                       "MLX2_NATIVE_PAGED_MANIFEST": str(adapter_root),
                       "MLX2_NATIVE_PAGED_MLX_WHEEL": manifest["paths"]["mlx_wheel"]})
    try:
        engine = serving.ServingEngine(
            str(model_root), adapter_factory=GateAdapter, max_lanes=1,
            max_inflight=1, mtp=False, qualification_mode=True,
            prefill_step=63, cache_bytes=1 << 29)
        if not engine.ready.wait(30) or engine.error:
            raise RuntimeError(f"HTTP serving engine did not load: {engine.error}")
        adapter = engine.adapter
        live = compute_live_price_identity(
            adapter_root, Path(manifest["paths"]["mlx_wheel"]), kernel,
            adapter_artifact_root=adapter.identity["path"])
        if live != manifest["identity"]:
            raise RuntimeError("loaded HTTP model/source/wheel identity differs")
        prompt, tokens = _prompt_for_63_tokens(adapter)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        body = {"model": model_root.name, "prompt": prompt, "max_tokens": 2,
                "temperature": 0}
        # The native request is first, so the same prompt is cold for parity.
        native_request = {**body, "paged_native_qwen3": True,
                          "skip_writing_prefix_cache": True}
        status_native, native = _http_post(base, native_request)
        if status_native != 200:
            raise RuntimeError(f"native HTTP request failed: {status_native}: {native}")
        native_receipt = _assert_native(native, calibration)
        native_samples = sampled[:]
        if len(native_samples) != 2 or len(captured) != 1:
            raise RuntimeError("native HTTP request did not emit exactly two tokens")
        writer = captured[0][1].backend.writer
        deadline = time.monotonic() + 3
        while not captured[0][0].fully_retired and time.monotonic() < deadline:
            time.sleep(.005)
        pending, retained, released = retired_native_state(writer)
        if not released or not captured[0][0].fully_retired:
            raise RuntimeError("native HTTP request retained pages or epochs")
        if engine.apc is None:
            raise RuntimeError("APCv2 missing from HTTP service")
        native_cache = dict(engine.apc.lifetime_stats())
        if native_cache.get("stores") != 0:
            raise RuntimeError("native HTTP request wrote an APCv2 checkpoint")

        ordinary_start = len(sampled)
        status_ordinary, ordinary = _http_post(base, body)
        ordinary_samples = sampled[ordinary_start:]
        if (status_ordinary != 200 or len(ordinary_samples) != 2 or
                ordinary["mlx2"].get("route") == "native_qwen3_paged" or
                any(receipt.get("route") == "native_qwen3_paged"
                    for _, receipt in ordinary_samples) or
                [token for token, _ in ordinary_samples] !=
                [token for token, _ in native_samples] or
                ordinary["choices"][0]["text"] != native["choices"][0]["text"]):
            raise RuntimeError("ordinary/default HTTP request and native parity differ")
        if len(captured) != 1:
            raise RuntimeError("ordinary HTTP default allocated a native owner")
        ordinary_cache = dict(engine.apc.lifetime_stats())
        if ordinary_cache.get("stores", 0) < 1:
            raise RuntimeError("ordinary HTTP request did not seed APCv2")

        status_write, write_refusal = _http_post(
            base, {**body, "paged_native_qwen3": True})
        _assert_refusal(status_write, write_refusal, "skip_writing_prefix_cache")
        status_warm, warm_refusal = _http_post(base, native_request)
        _assert_refusal(status_warm, warm_refusal, "cold no-APCv2")
        if len(captured) != 1:
            raise RuntimeError("refused HTTP requests allocated native owners")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak > MAX_RESIDENT_BYTES:
            raise RuntimeError("native HTTP request exceeded memory cap")
        return {"schema": SCHEMA, "status": "passed", "gpu_executed": True,
                "identity": manifest["identity"], "gpuq_owner": owner_receipt,
                "calibration_evidence_sha256": calibration.evidence_sha256,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "prompt_tokens": len(tokens), "http": {
                    "native_status": status_native, "native_body": native,
                    "ordinary_status": status_ordinary, "ordinary_body": ordinary,
                    "write_refusal_status": status_write,
                    "write_refusal_body": write_refusal,
                    "warm_refusal_status": status_warm,
                    "warm_refusal_body": warm_refusal},
                "native_token_ids": [token for token, _ in native_samples],
                "ordinary_token_ids": [token for token, _ in ordinary_samples],
                "native_route_receipt": native_receipt,
                "native_cache_stats_after_request": native_cache,
                "ordinary_cache_stats_after_request": ordinary_cache,
                "pending_native_epochs": pending, "retained_pages": retained,
                "owner_fully_retired": True, "native_owner_allocations": len(captured),
                "peak_resident_bytes": peak, "qualified": False,
                "selected": True, "observed_used": True}
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
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.preflight:
        result = preflight_cpu(manifest)
    else:
        if args.calibration is None or args.receipt is None:
            parser.error("GPU execution requires --calibration and --receipt")
        try:
            result = execute(manifest, args.calibration)
        except Exception as error:
            result = {"schema": SCHEMA, "status": "failed", "gpu_executed": None,
                      "identity": manifest.get("identity"),
                      "error_type": type(error).__name__, "error": str(error),
                      "qualified": False}
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            raise
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
