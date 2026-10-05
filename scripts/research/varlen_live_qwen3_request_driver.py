"""Research-only real BatchGenerator Qwen3 cold request and q=1 step driver.

The factory is called only by varlen_live_request_price after source, artifact,
wheel, kernel, host, and GPU lease preflight. No synthetic price is installed.
"""

from __future__ import annotations

import resource
import time
from pathlib import Path
from threading import RLock


def retired_native_state(writer) -> tuple[int, int, bool]:
    """Read the same native cleanup proof used by a completed live request."""
    pending = len(writer.pending_epochs) + writer.ledger.pending_count
    retained = writer.pool.allocated_count
    if type(pending) is not int or type(retained) is not int:
        raise RuntimeError("native retirement counts are invalid")
    return pending, retained, pending == 0 and retained == 0


class Qwen3RequestDriver:
    def __init__(self, manifest: dict):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx2.adapters.standard_decoder import StandardDecoderAdapter
        from varlen_pack_price_bench import verify_artifact_manifest

        artifact = verify_artifact_manifest(Path(manifest["paths"]["artifact"]))
        self.adapter = StandardDecoderAdapter(artifact["root"])
        self.model = self.adapter.model
        # The source artifact is BF16; pin both paired arms to the same fp16
        # weights outside the measured request span for the native backend.
        self.model.apply(lambda parameter: parameter.astype(mx.float16))
        mx.eval(self.model.parameters())
        if any(parameter.dtype != mx.float16
               for _, parameter in tree_flatten(self.model.parameters())):
            raise RuntimeError("research request has non-fp16 weights")
        if len(self.model.layers) != 28 or self.model.model.embed_tokens.weight.dtype != mx.float16:
            raise RuntimeError("research request needs the pinned fp16 Qwen3-0.6B")
        self.revision = self.adapter.identity["fingerprint"]
        self.prompt = tuple([1000] * 63)
        self.max_tokens = 2
        self.mx = mx

    def close(self):
        self.adapter.close()

    def run_request(self, arm: str, context_tokens: tuple[int, ...]) -> dict:
        if arm not in ("ordinary", "paged") or context_tokens != (63,):
            raise ValueError("only the pinned q=1 research cell is supported")
        from mlx2.runtime.generate import BatchGenerator

        mx = self.mx
        lifecycle_lock = RLock()
        batch = BatchGenerator(self.model, max_tokens=self.max_tokens,
                               prefill_batch_size=1, completion_batch_size=1,
                               prefill_step_size=63, stop_tokens=[])
        owner = candidate = uid = None
        installed = published = False
        forward_calls = 0
        model_type = type(self.model)
        original_call = model_type.__call__

        def count_forward(instance, *args, **kwargs):
            nonlocal forward_calls
            if instance is self.model:
                forward_calls += 1
            return original_call(instance, *args, **kwargs)

        model_type.__call__ = count_forward
        response_tokens = []
        first_receipt = last_receipt = None
        q1_start = q1_elapsed = None
        try:
            with lifecycle_lock:
                uid = batch.insert([list(self.prompt)], max_tokens=[self.max_tokens])[0]
            if arm == "paged":
                from mlx2.runtime.paged_native_batch_lifecycle import (
                    prepare_queued_native_first_response,
                    run_research_native_queued_qwen3,
                )
                from mlx2.runtime.paged_request_transaction import CandidateRequest

                owner, candidate = self.adapter.create_native_paged_qwen3_request(
                    revision=self.revision, prompt_tokens=len(self.prompt),
                    max_tokens=self.max_tokens, permit_candidate=True)
                probe = run_research_native_queued_qwen3(
                    batch, lifecycle_lock,
                    CandidateRequest(uid, self.revision, len(self.prompt), ("kv",)),
                    owner, candidate, research_permit=True)
                if probe.reason != "research_executed" or not probe.research_executed:
                    raise RuntimeError(f"research prefill refused: {probe.reason}")
                prepared = prepare_queued_native_first_response(
                    batch, lifecycle_lock, owner, probe)
                published = prepared.generation is not None and prepared.reason == "native_continuation_not_installed"
                with lifecycle_lock:
                    install = batch.install_native_queued(
                        prepared, owner, candidate, lifecycle_lock,
                        permit_native=True, research_only=True)
                if (install.get("reason") != "native_installed" or
                        install.get("selected") is not False or
                        install.get("research_only") is not True):
                    raise RuntimeError(f"research handoff refused: {install}")
                installed = True
            for _ in range(6):
                if len(response_tokens) == 1 and q1_start is None:
                    q1_start = time.perf_counter_ns()
                with lifecycle_lock:
                    _, responses = batch.next()
                failures = batch.take_lane_failures()
                if failures:
                    raise RuntimeError(f"generation failure: {failures}")
                for response in responses:
                    if response.uid != uid:
                        raise RuntimeError("response UID drifted")
                    mx.eval(response.logprobs)
                    mx.synchronize()
                    if len(response_tokens) == 0:
                        first_receipt = dict(response.mtp_receipt or {})
                    response_tokens.append(int(response.token))
                    last_receipt = dict(response.mtp_receipt or {})
                    if len(response_tokens) == 2:
                        q1_elapsed = (time.perf_counter_ns() - q1_start) / 1e6
                    if len(response_tokens) > 2:
                        raise RuntimeError("request produced unexpected output count")
                if len(response_tokens) == 2:
                    break
            if len(response_tokens) != 2 or q1_elapsed is None:
                raise RuntimeError("full prefill plus q=1 response missing")
            if arm == "paged":
                prefill_reads = first_receipt.get("native_read_calls")
                reads = last_receipt.get("native_read_calls")
                terminals = last_receipt.get("terminal_successes")
                if (type(prefill_reads) is not int or type(reads) is not int or
                        type(terminals) is not int or reads - prefill_reads < 28 or
                        terminals != reads):
                    raise RuntimeError("physical native q=1 reads missing")
                route = {**last_receipt, "serving_selected": False}
            else:
                prefill_reads = reads = terminals = 0
                route = {"route": "ordinary", "selected": True,
                         "observed_used": True}
                published = True
            with lifecycle_lock:
                batch.remove([uid])
            uid = None
            mx.synchronize()
            if arm == "paged":
                writer = candidate.backend.writer
                pending, retained, released = retired_native_state(writer)
                q1_tile_dispatches = writer.backend.q1_tile_dispatch_count()
            else:
                pending = retained = 0
                released = True
                q1_tile_dispatches = 0
            return {
                "arm": arm, "request_inserted": True, "admitted": True,
                "cache_transaction_published": published,
                "response_emitted": True, "sampled_output": True,
                "synchronized": True, "request_removed": True,
                "request_state_released": released,
                "output_token_id": response_tokens[-1],
                "output_token_ids": response_tokens,
                "model_layers": len(self.model.layers),
                "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "route_receipt": route,
                "first_response_receipt": first_receipt,
                "second_response_receipt": last_receipt,
                "prefill_read_calls": prefill_reads,
                "ordinary_model_forward_calls": forward_calls,
                "paged_read_calls": reads,
                "decode_read_calls": reads - prefill_reads,
                "terminal_successes": terminals,
                "pending_native_epochs": pending, "retained_pages": retained,
                "q1_tile_dispatches": q1_tile_dispatches,
                "q1_step_ms": q1_elapsed,
            }
        finally:
            model_type.__call__ = original_call
            if uid is not None:
                with lifecycle_lock:
                    batch.remove([uid])
            if owner is not None and not installed:
                owner.close()
                owner.reap_retired()
                owner.reap_quarantine()


def create(manifest: dict) -> Qwen3RequestDriver:
    return Qwen3RequestDriver(manifest)
