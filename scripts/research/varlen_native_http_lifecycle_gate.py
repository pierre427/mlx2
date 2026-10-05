"""Actual-native Qwen3 HTTP lane failure and client cancellation gate.

The failure is injected on the host *after* one real native read has terminal
proof. This does not simulate a failed GPU command buffer. Both arms remain
explicit, cold-only, default-off, and unqualified.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import resource
import signal
import socket
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from varlen_live_request_price import preflight_cpu
from varlen_native_http_gate import _http_post, _prompt_for_63_tokens, _timeout
from varlen_pack_price_bench import (
    MAX_RESIDENT_BYTES,
    _gpuq_owner,
    verify_artifact_manifest,
)

SCHEMA = "mlx2.varlen-native-http-lifecycle.v1"
FAULT_TEXT = "injected post-terminal native read failure"


def _retirement(owner, candidate) -> dict:
    from varlen_live_qwen3_request_driver import retired_native_state

    writer = candidate.backend.writer
    deadline = time.monotonic() + 3
    while not owner.fully_retired and time.monotonic() < deadline:
        time.sleep(.005)
    pending, retained, released = retired_native_state(writer)
    result = {"pending_native_epochs": pending, "retained_pages": retained,
              "owner_fully_retired": bool(owner.fully_retired),
              "quarantined_branches": len(owner._quarantine),
              "open_branches": len(owner._open_branches),
              "reader_leases": sum(owner._readers.values()),
              "writer_poisoned": bool(writer.poisoned),
              "failed_arena_torn_down": bool(writer.failed_arena_torn_down),
              "native_read_submissions": candidate.backend.read_submissions,
              "native_terminal_successes": candidate.backend.terminal_successes}
    if (not released or not result["owner_fully_retired"] or
            result["quarantined_branches"] or result["open_branches"] or
            result["reader_leases"] or result["writer_poisoned"]):
        raise RuntimeError(f"native lifecycle retained state: {result}")
    return result


def _stream_then_disconnect(base: str, body: dict) -> dict:
    """Close a real loopback HTTP stream after its first visible token."""
    host, port = base.removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=10)
    connection.request("POST", "/v1/completions", json.dumps(body),
                       {"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"native stream returned HTTP {response.status}")
    first = None
    try:
        for _ in range(32):
            line = response.fp.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                raise RuntimeError("native stream finished before cancellation")
            chunk = json.loads(data)
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                first = chunk
                break
        if first is None:
            raise RuntimeError("native stream produced no first visible token")
        if connection.sock is not None:
            connection.sock.shutdown(socket.SHUT_RDWR)
        return {"status": response.status, "first_chunk": first,
                "client_closed_after_first_token": True}
    finally:
        response.close()
        connection.close()


def execute(manifest: dict, calibration_path: Path) -> dict:
    preflight_cpu(manifest)
    gpuq_owner = _gpuq_owner()
    import _paged_kv_native
    import mlx.core as mx

    from mlx2 import serving
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime.paged_native_atomic_owner import NativeAtomicRequestOwner
    from mlx2.runtime.paged_pack_price import load_research_calibration
    from mlx2.runtime.paged_price_identity import compute_live_price_identity
    from mlx2.runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend
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

    captured = []
    close_transitions: dict[int, int] = {}
    first_delta_queued = threading.Event()
    disconnect_observed = threading.Event()
    phase = {"value": "failure", "fault_fired": False,
             "cancel_job": None, "cancel_timeout": False}
    original_read = NativeQwen3PagedBackend.read_completed
    original_close = NativeAtomicRequestOwner.close
    original_emit = serving.ServingEngine._emit

    def observed_read(backend, *args, **kwargs):
        result = original_read(backend, *args, **kwargs)
        if (phase["value"] == "failure" and captured and
                backend is captured[0][1].backend and
                backend.read_submissions == 29 and not phase["fault_fired"]):
            phase["fault_fired"] = True
            raise RuntimeError(FAULT_TEXT)
        return result

    def observed_close(owner):
        if not owner._closed:
            key = id(owner)
            close_transitions[key] = close_transitions.get(key, 0) + 1
        return original_close(owner)

    def observed_emit(engine, job, event):
        result = original_emit(engine, job, event)
        if (phase["value"] == "cancel" and
                getattr(job, "native_paged_receipt", None) is not None and
                "delta" in event and event["delta"].get("content") and
                not first_delta_queued.is_set()):
            phase["cancel_job"] = job
            first_delta_queued.set()
            if job.cancelled.wait(5):
                disconnect_observed.set()
            else:
                phase["cancel_timeout"] = True
        return result

    class GateAdapter(StandardDecoderAdapter):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.model.apply(lambda parameter: parameter.astype(mx.float16))
            mx.eval(self.model.parameters())
            if (len(self.model.layers) != 28 or
                    self.model.model.embed_tokens.weight.dtype != mx.float16):
                raise RuntimeError("HTTP lifecycle gate needs pinned fp16 Qwen3-0.6B")

        def create_native_paged_qwen3_request(self, **kwargs):
            result = super().create_native_paged_qwen3_request(**kwargs)
            captured.append(result)
            return result

    model_root = Path(verify_artifact_manifest(
        Path(manifest["paths"]["artifact"]))["root"])
    old_environment = {key: os.environ.get(key) for key in (
        "MLX2_NATIVE_PAGED_PRICE", "MLX2_NATIVE_PAGED_MANIFEST",
        "MLX2_NATIVE_PAGED_MLX_WHEEL")}
    os.environ.update({"MLX2_NATIVE_PAGED_PRICE": str(calibration_path),
                       "MLX2_NATIVE_PAGED_MANIFEST": manifest["paths"]["artifact"],
                       "MLX2_NATIVE_PAGED_MLX_WHEEL": manifest["paths"]["mlx_wheel"]})
    NativeQwen3PagedBackend.read_completed = observed_read
    NativeAtomicRequestOwner.close = observed_close
    serving.ServingEngine._emit = observed_emit
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(120)
    engine = server = thread = None
    try:
        engine = serving.ServingEngine(
            str(model_root), adapter_factory=GateAdapter, max_lanes=1,
            max_inflight=1, mtp=False, qualification_mode=True,
            prefill_step=63, cache_bytes=1 << 29)
        if not engine.ready.wait(30) or engine.error:
            raise RuntimeError(f"HTTP serving engine did not load: {engine.error}")
        live = compute_live_price_identity(
            Path(manifest["paths"]["artifact"]),
            Path(manifest["paths"]["mlx_wheel"]), kernel,
            adapter_artifact_root=engine.adapter.identity["path"])
        if live != manifest["identity"]:
            raise RuntimeError("loaded HTTP model/source/wheel identity differs")
        prompt, tokens = _prompt_for_63_tokens(engine.adapter)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        ordinary = {"model": model_root.name, "prompt": prompt,
                    "max_tokens": 2, "temperature": 0,
                    "skip_writing_prefix_cache": True}
        native = {**ordinary, "paged_native_qwen3": True}
        before_cache = engine.apc.lifetime_stats()
        if before_cache["stores"]:
            raise RuntimeError("isolated HTTP service has preexisting APCv2 store")

        failure_status, failure_body = _http_post(base, native)
        if (failure_status != 500 or
                FAULT_TEXT not in failure_body.get("error", {}).get("message", "") or
                not phase["fault_fired"] or len(captured) != 1):
            raise RuntimeError(f"post-terminal native HTTP failure not contained: {failure_status}")
        failed_owner, failed_candidate = captured[0]
        failure_retirement = _retirement(failed_owner, failed_candidate)
        if (failure_retirement["native_read_submissions"] != 29 or
                failure_retirement["native_terminal_successes"] != 29 or
                close_transitions.get(id(failed_owner)) != 1 or
                engine.counts["executor_lane_failures"] != 1 or
                engine.apc.lifetime_stats()["stores"] != 0):
            raise RuntimeError("failed native lane retained state or polluted APCv2")

        survival_status, survival_body = _http_post(base, ordinary)
        if (survival_status != 200 or
                survival_body["mlx2"].get("cached_tokens") != 0 or
                survival_body["mlx2"].get("route") == "native_qwen3_paged" or
                len(captured) != 1 or engine.apc.lifetime_stats()["stores"] != 0):
            raise RuntimeError("ordinary HTTP route did not survive native failure")

        phase["value"] = "cancel"
        stream = _stream_then_disconnect(base, {**native, "stream": True})
        if (not first_delta_queued.wait(3) or not disconnect_observed.wait(6) or
                phase["cancel_timeout"]):
            raise RuntimeError("native stream disconnect did not reach serving worker")
        deadline = time.monotonic() + 4
        while (len(captured) < 2 or not captured[1][0].fully_retired) and time.monotonic() < deadline:
            time.sleep(.005)
        if len(captured) != 2:
            raise RuntimeError("cancelled native request did not allocate one owner")
        cancelled_owner, cancelled_candidate = captured[1]
        cancel_retirement = _retirement(cancelled_owner, cancelled_candidate)
        if (close_transitions.get(id(cancelled_owner)) != 1 or
                engine.counts["cancelled"] != 1 or
                engine.counts["client_disconnects"] < 1 or
                cancel_retirement["native_read_submissions"] != 28 or
                cancel_retirement["native_terminal_successes"] != 28 or
                engine.apc.lifetime_stats()["stores"] != 0):
            raise RuntimeError("disconnected native lane was not cancelled once and retired")

        final_status, final_body = _http_post(base, ordinary)
        if (final_status != 200 or final_body["mlx2"].get("cached_tokens") != 0 or
                engine.apc.lifetime_stats()["stores"] != 0 or len(captured) != 2):
            raise RuntimeError("ordinary HTTP route did not survive cancellation")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak > MAX_RESIDENT_BYTES:
            raise RuntimeError("native HTTP lifecycle gate exceeded memory cap")
        return {"schema": SCHEMA, "status": "passed", "gpu_executed": True,
                "identity": manifest["identity"], "gpuq_owner": gpuq_owner,
                "calibration_evidence_sha256": calibration.evidence_sha256,
                "prompt_tokens": len(tokens), "failure": {
                    "status": failure_status, "body": failure_body,
                    "fault_kind": "host_post_terminal_exception",
                    "device_fault_claimed": False,
                    "post_terminal_fault_fired": phase["fault_fired"],
                    "retirement": failure_retirement,
                    "close_transitions": close_transitions[id(failed_owner)]},
                "cancel": {"stream": stream, "disconnect_observed": True,
                           "retirement": cancel_retirement,
                           "close_transitions": close_transitions[id(cancelled_owner)]},
                "ordinary_after_failure": {"status": survival_status,
                                           "body": survival_body},
                "ordinary_after_cancel": {"status": final_status,
                                          "body": final_body},
                "apcv2": engine.apc.lifetime_stats(),
                "engine_counts": {"executor_lane_failures": engine.counts["executor_lane_failures"],
                                  "cancelled": engine.counts["cancelled"],
                                  "client_disconnects": engine.counts["client_disconnects"]},
                "peak_resident_bytes": peak, "qualified": False}
    finally:
        signal.alarm(0)
        NativeQwen3PagedBackend.read_completed = original_read
        NativeAtomicRequestOwner.close = original_close
        serving.ServingEngine._emit = original_emit
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
