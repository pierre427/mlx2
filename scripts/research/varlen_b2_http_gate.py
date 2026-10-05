"""One bounded, source-bound, two-request Qwen3 B2 loopback HTTP gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from varlen_staged_graph_price import preflight_cpu
from varlen_pack_price_bench import MAX_RESIDENT_BYTES, _gpuq_owner, verify_artifact_manifest

SCHEMA = "mlx2.varlen-b2-http-gate.v1"


def _contexts(manifest):
    contexts = manifest.get("context_tokens")
    if (type(contexts) is not list or len(contexts) != 2 or
            any(type(length) is not int or not 32 <= length <= 127
                for length in contexts) or contexts[0] == contexts[1]):
        raise RuntimeError("HTTP B2 needs distinct 32-127 token contexts")
    return tuple(contexts)


def _timeout(_signum, _frame):
    raise TimeoutError("B2 HTTP gate exceeded 120 seconds")


def _post(base, body):
    request = Request(base + "/v1/completions", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        with error:
            return error.code, json.load(error)


def _pair(base, bodies):
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(_post, base, body) for body in bodies]
        return tuple(future.result(timeout=35) for future in futures)


def _ordered_cohort_pair(base, bodies, engine):
    """Publish lane 63 first while keeping both HTTP requests concurrent."""
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(_post, base, bodies[0])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with engine.submission_lock:
                staged = engine.pending_cohorts.get(("default", "gate-b2"))
                if staged is not None and len(staged["jobs"]) == 1:
                    break
            time.sleep(.002)
        else:
            raise RuntimeError("first B2 HTTP job did not stage its cohort")
        second = workers.submit(_post, base, bodies[1])
        return first.result(timeout=35), second.result(timeout=35)


def _prompt(adapter, token, count):
    tokenizer = adapter.tokenizer
    candidates = [tokenizer.decode([token] * count)]
    candidates.extend(("hello " if token == 1000 else "world ") * n
                      for n in range(1, 150))
    for value in candidates:
        tokens = adapter.prompt_tokens({"prompt": value})
        if len(tokens) == count:
            return value, tuple(tokens)
    raise RuntimeError(f"no exact {count}-token HTTP prompt")


def _route(body):
    receipt = body.get("mlx2", {}).get("route_receipt")
    if type(receipt) is not dict:
        raise RuntimeError("HTTP route receipt missing")
    return receipt


def execute(manifest, profile_path):
    contexts = _contexts(manifest)
    preflight_cpu(manifest, context_validator=_contexts)
    gpuq_owner = _gpuq_owner()
    import _paged_kv_native
    import mlx.core as mx
    from mlx2 import serving
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    from mlx2.runtime import qwen3_paged_graph_factory
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.paged_b2_research_profile import (
        COMBINED_FLAGS, STOCK_SDPA_FLAG, STRIPES_FLAG,
        SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5,
        SCHEMA_V6, SCHEMA_V7,
        load_b2_research_profile)
    from mlx2.runtime.paged_price_identity import compute_live_price_identity
    from mlx2.server import handler_for
    from varlen_live_qwen3_request_driver import retired_native_state

    kernel = Path(_paged_kv_native.__file__).resolve()
    if kernel != Path(manifest["paths"]["kernel"]).resolve():
        raise RuntimeError("loaded native binary differs from manifest")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("GPU execution required")
    profile = load_b2_research_profile(
        profile_path, live_identity=manifest["identity"], context_lengths=contexts)
    options = profile["combined_optimizations"] if profile["schema"] == SCHEMA_V7 else {
        "deferred_eval": False, "deferred_write_eval": False,
        "inline_metadata": False,
        "grouped_sampler": False, "stock_sdpa": False}
    stripes = profile["q1_simd_stripes"] if profile["schema"] == SCHEMA_V7 else 4
    model_root = Path(verify_artifact_manifest(
        Path(manifest["paths"]["artifact"]))["root"])
    captures, sampled, effective_sampling = [], {}, {}
    original_next = BatchGenerator.next
    original_factory = qwen3_paged_graph_factory.create_shared_qwen3_graph_pack
    engine = server = server_thread = None

    def observed_next(batch, *args, **kwargs):
        prompt, responses = original_next(batch, *args, **kwargs)
        for response in responses:
            job = next((job for job in engine.jobs.values()
                        if job.uid == response.uid), None)
            if job is None:
                raise RuntimeError("generator response has no live HTTP job")
            effective_sampling[job.id] = dict(job.effective_sampling or {})
            sampled.setdefault(job.id, []).append(
                (int(response.token), dict(response.mtp_receipt or {})))
        return prompt, responses

    def captured_factory(*args, **kwargs):
        owners, candidate = original_factory(*args, **kwargs)
        captures.append((owners, candidate))
        return owners, candidate

    class GateAdapter(StandardDecoderAdapter):
        def __init__(self, path, **kwargs):
            super().__init__(path, **kwargs)
            self.model.apply(lambda parameter: parameter.astype(mx.float16))
            mx.eval(self.model.parameters())
            if (len(self.model.layers) != 28 or
                    self.model.model.embed_tokens.weight.dtype != mx.float16):
                raise RuntimeError("gate needs pinned fp16 Qwen3-0.6B")

    old_env = {key: os.environ.get(key) for key in (
        "MLX2_NATIVE_PAGED_B2_PROFILE", "MLX2_NATIVE_PAGED_MANIFEST",
        "MLX2_NATIVE_PAGED_MLX_WHEEL", "MLX2_PAGED_Q1_SIMD_TILE",
        "MLX2_PAGED_GROUPED_Q1_WRITE", "MLX2_PAGED_PRIVATE_TAIL_REUSE",
        "MLX2_PAGED_B2_PACKED_PREFILL", "MLX2_PAGED_B2_SUSTAINED_DECODE",
        "MLX2_PAGED_Q1_VECTOR_ROPE", "MLX2_PAGED_GROUPED_DIRECT_FENCE",
        STRIPES_FLAG, STOCK_SDPA_FLAG, *COMBINED_FLAGS)}
    os.environ.update({
        "MLX2_NATIVE_PAGED_B2_PROFILE": str(profile_path),
        "MLX2_NATIVE_PAGED_MANIFEST": manifest["paths"]["artifact"],
        "MLX2_NATIVE_PAGED_MLX_WHEEL": manifest["paths"]["mlx_wheel"],
        "MLX2_PAGED_Q1_SIMD_TILE": "1", "MLX2_PAGED_GROUPED_Q1_WRITE": "1",
        "MLX2_PAGED_PRIVATE_TAIL_REUSE": "1",
        STRIPES_FLAG: str(stripes),
        STOCK_SDPA_FLAG: "1" if options["stock_sdpa"] else "0",
        **{key: "1" if enabled else "0" for key, enabled in zip(
            COMBINED_FLAGS, (options["deferred_eval"],
                             options["deferred_write_eval"],
                             options["inline_metadata"],
                             options["grouped_sampler"]))},
    })
    if profile["schema"] in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7):
        os.environ["MLX2_PAGED_B2_PACKED_PREFILL"] = "1"
    if profile["schema"] in (SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7):
        os.environ["MLX2_PAGED_B2_SUSTAINED_DECODE"] = "1"
    if profile["schema"] == SCHEMA_V5:
        os.environ["MLX2_PAGED_Q1_VECTOR_ROPE"] = "1"
    if profile["schema"] in (SCHEMA_V6, SCHEMA_V7):
        os.environ["MLX2_PAGED_GROUPED_DIRECT_FENCE"] = "1"
    BatchGenerator.next = observed_next
    qwen3_paged_graph_factory.create_shared_qwen3_graph_pack = captured_factory
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(120)
    try:
        engine = serving.ServingEngine(
            str(model_root), adapter_factory=GateAdapter, max_lanes=2,
            max_inflight=2, mtp=False, qualification_mode=False,
            prefill_step=max(65, max(contexts)), cache_bytes=1 << 29)
        if not engine.ready.wait(30) or engine.error:
            raise RuntimeError(f"HTTP serving engine did not load: {engine.error}")
        live = compute_live_price_identity(
            manifest["paths"]["artifact"], manifest["paths"]["mlx_wheel"],
            kernel, adapter_artifact_root=engine.adapter.identity["path"])
        if live != manifest["identity"]:
            raise RuntimeError("loaded HTTP model/source/wheel differs")
        prompts = (_prompt(engine.adapter, 1000, contexts[0]),
                   _prompt(engine.adapter, 1001, contexts[1]))
        if prompts[0][1] == prompts[1][1]:
            raise RuntimeError("B2 prompts are not independent")
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine))
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        common = [{"model": model_root.name, "prompt": prompt,
                   "max_tokens": profile["max_tokens"], "temperature": 0,
                   "repetition_penalty": 1,
                   "presence_penalty": 0, "frequency_penalty": 0,
                   "skip_writing_prefix_cache": True}
                  for prompt, _ in prompts]
        if profile["schema"] not in (SCHEMA_V2, SCHEMA_V3, SCHEMA_V4, SCHEMA_V5,
                                      SCHEMA_V6, SCHEMA_V7):
            for body in common:
                body.update(top_p=1, top_k=0, min_p=0)
        native_bodies = tuple({**body, "paged_native_qwen3_b2": True,
                               "batch_cohort": {"id": "gate-b2", "size": 2}}
                              for body in common)
        native = _ordered_cohort_pair(base, native_bodies, engine)
        if any(status != 200 for status, _ in native):
            raise RuntimeError(f"native B2 HTTP cohort failed: {native}")
        if any(body.get("mlx2", {}).get("qualification") != "unqualified"
               for _, body in native):
            raise RuntimeError("native B2 HTTP service status is not unqualified")
        native_receipts = tuple(_route(body) for _, body in native)
        native_effective = [effective_sampling.get(body.get("id")) for _, body in native]
        if any(type(values) is not dict or values.get("temperature") != 0 or
               values.get("repetition_penalty") != 1 or
               values.get("presence_penalty") != 0 or
               values.get("frequency_penalty") != 0
               for values in native_effective):
            raise RuntimeError("native effective greedy controls differ")
        for receipt in native_receipts:
            if (receipt.get("route") != "native_qwen3_paged_b2" or
                    receipt.get("selected") is not True or
                    receipt.get("observed_used") is not True or
                    receipt.get("qualified") is not False or
                    receipt.get("price_usable") is not False or
                    receipt.get("admission_profile") != profile["profile_id"] or
                    receipt.get("prefill_mode") != (
                        "packed_staged" if profile["schema"] in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7)
                        else "serial_native") or
                    receipt.get("native_read_delta") != 28 or
                    receipt.get("terminal_success_delta") != 28 or
                    receipt.get("native_span_counts") != [2] * 28 or
                    receipt.get("physical_dispatches") != {
                        "grouped_q1_writes": 28, "native_write_dispatches": 0,
                        "q1_tile_dispatches": 0 if options["stock_sdpa"] else 28} or
                    receipt.get("native_read_calls") != (
                        28 * profile["max_tokens"] if profile["schema"] in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7)
                        else 28 * (profile["max_tokens"] + 1)) or
                    receipt.get("terminal_successes") != (
                        28 * profile["max_tokens"] if profile["schema"] in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7)
                        else 28 * (profile["max_tokens"] + 1)) or
                    len(receipt.get("output_token_ids", ())) != profile["max_tokens"] or
                    receipt.get("vector_q1_rope_calls") != (
                        28 * (profile["max_tokens"] - 1)
                        if profile["schema"] == SCHEMA_V5 else 0) or
                    receipt.get("direct_grouped_fence_reads") != (
                        28 * (profile["max_tokens"] - 1)
                        if profile["schema"] in (SCHEMA_V6, SCHEMA_V7) else 0) or
                    receipt.get("combined_optimizations") != options or
                    receipt.get("q1_simd_stripes") != stripes or
                    receipt.get("combined_submission_delta") != {
                        "grouped_write_async_evals": 0 if options["deferred_eval"] or options["deferred_write_eval"] else 28,
                        "staged_read_async_evals": 0 if options["deferred_eval"] else 28,
                        "deferred_q1_write_roots": 28 if options["deferred_eval"] or options["deferred_write_eval"] else 0,
                        "deferred_q1_read_roots": 28 if options["deferred_eval"] or options["deferred_write_eval"] else 0,
                        "deferred_q1_final_evals": 1 if options["deferred_eval"] or options["deferred_write_eval"] else 0,
                        "deferred_q1_failure_flushes": 0,
                        "q1_metadata_dispatches": 28 if options["inline_metadata"] and not options["stock_sdpa"] else 0,
                        "q1_gather_dispatches": 28 if options["stock_sdpa"] else 0,
                        "stock_sdpa_graph_calls": 28 if options["stock_sdpa"] else 0,
                        **{f"q1_stripe_dispatches_{value}": 28 if not options["stock_sdpa"] and stripes == value else 0
                           for value in (8, 16, 32)}} or
                    receipt.get("grouped_sampler_eval") is not
                        options["grouped_sampler"]):
                raise RuntimeError(f"native B2 HTTP physical route proof differs: {receipt}")
            if (profile["schema"] in (SCHEMA_V3, SCHEMA_V4, SCHEMA_V5, SCHEMA_V6, SCHEMA_V7) and
                    receipt.get("native_prefill_proof") != {
                        "read_calls": 28, "terminal_successes": 28,
                        "span_counts": [2] * 28}):
                raise RuntimeError("packed B2 HTTP prefill proof differs")
        native_tokens = [receipt["output_token_ids"] for receipt in native_receipts]
        if len(captures) != 1:
            raise RuntimeError("B2 did not allocate exactly one shared arena")
        owners, candidate = captures[0]
        deadline = time.monotonic() + 3
        while not all(owner.fully_retired for owner in owners) and time.monotonic() < deadline:
            time.sleep(.005)
        pending, retained, released = retired_native_state(candidate.backend.writer)
        if not released or not all(owner.fully_retired for owner in owners):
            raise RuntimeError("B2 native owners retained pages or epochs")
        ordinary = _pair(base, common)
        if any(status != 200 for status, _ in ordinary):
            raise RuntimeError(f"ordinary HTTP reference failed: {ordinary}")
        if any(body.get("mlx2", {}).get("qualification") != "unqualified"
               for _, body in ordinary):
            raise RuntimeError("ordinary HTTP service status is not unqualified")
        ordinary_tokens = []
        ordinary_effective = [effective_sampling.get(body.get("id"))
                              for _, body in ordinary]
        if ordinary_effective != native_effective:
            raise RuntimeError("ordinary and native effective sampler fields differ")
        for _, body in ordinary:
            if body.get("mlx2", {}).get("route") == "native_qwen3_paged_b2":
                raise RuntimeError("ordinary HTTP selected B2")
            samples = sampled.get(body.get("id"))
            if samples is None or len(samples) != profile["max_tokens"]:
                raise RuntimeError("ordinary HTTP sampled token IDs missing")
            ordinary_tokens.append([token for token, _ in samples])
        if len(captures) != 1:
            raise RuntimeError("ordinary default allocated native B2 arena")
        if [body["choices"][0]["text"] for _, body in ordinary] != [
                body["choices"][0]["text"] for _, body in native]:
            raise RuntimeError("ordinary and native HTTP text differ")
        if ordinary_tokens != native_tokens:
            raise RuntimeError("ordinary and native HTTP token IDs differ")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RESIDENT_BYTES:
            raise RuntimeError("B2 HTTP gate exceeded memory cap")
        return {"schema": SCHEMA, "status": "passed", "gpu_executed": True,
                "identity": live, "gpuq_owner": gpuq_owner,
                "profile_id": profile["profile_id"], "qualified": False,
                "price_usable": False,
                "prompt_sha256": [hashlib.sha256(value.encode()).hexdigest()
                                  for value, _ in prompts],
                "context_tokens": [len(tokens) for _, tokens in prompts],
                "native_token_ids": native_tokens,
                "ordinary_token_ids": ordinary_tokens,
                "native_effective_sampling": native_effective,
                "ordinary_effective_sampling": ordinary_effective,
                "native_http": [body for _, body in native],
                "ordinary_http": [body for _, body in ordinary],
                "native_route_receipts": native_receipts,
                "pending_native_epochs": pending, "retained_pages": retained,
                "owners_fully_retired": True,
                "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    finally:
        signal.alarm(0)
        BatchGenerator.next = original_next
        qwen3_paged_graph_factory.create_shared_qwen3_graph_pack = original_factory
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=5)
        if engine is not None:
            engine.close()
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.preflight:
        contexts = _contexts(manifest)
        result = preflight_cpu(manifest, context_validator=_contexts)
        from mlx2.runtime.paged_b2_research_profile import load_b2_research_profile
        load_b2_research_profile(
            args.profile, live_identity=manifest["identity"],
            context_lengths=contexts)
    else:
        if args.receipt is None:
            parser.error("GPU gate requires --receipt")
        try:
            result = execute(manifest, args.profile)
        except Exception as error:
            result = {"schema": SCHEMA, "status": "failed",
                      "identity": manifest.get("identity"),
                      "error_type": type(error).__name__, "error": str(error),
                      "qualified": False, "price_usable": False}
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            raise
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
