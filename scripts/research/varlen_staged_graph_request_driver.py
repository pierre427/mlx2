"""Research-only paired live B2 Qwen3 ready-step driver."""

from __future__ import annotations

import resource
import time
from threading import RLock

from varlen_live_qwen3_request_driver import Qwen3RequestDriver, retired_native_state


def retained_native_state_diagnostics(receipts: dict, uids: tuple[int, ...]) -> list[dict]:
    """Read publication proof from emitted responses after owners may close."""
    return [{
        "pre_step_reader_offset": receipts[uid][-1].get("native_reader_offset"),
        "published_layer_offsets": receipts[uid][-1].get("published_layer_offsets"),
    } for uid in uids]


def _failure_physical_counters(candidate, before_host):
    if before_host is None:
        return None
    try:
        after_host = candidate.backend.profile_counters_snapshot()
        return {
            "grouped_q1_writes": (after_host["grouped_q1_writes"] -
                                  before_host["grouped_q1_writes"]),
            "native_write_dispatches": (after_host["native_write_dispatches"] -
                                        before_host["native_write_dispatches"]),
            "pending_native_epochs": len(candidate.backend.writer.pending_epochs),
        }
    except Exception:  # noqa: BLE001 - diagnostic snapshot must not mask the request failure.
        return None


class StagedGraphRequestDriver(Qwen3RequestDriver):
    def __init__(self, manifest: dict):
        super().__init__(manifest)
        contexts = tuple(manifest["context_tokens"])
        if contexts not in ((63, 65), (63, 63)):
            raise ValueError("unsupported B2 context control")
        self.contexts = contexts
        self.prompts = (tuple([1000] * contexts[0]), tuple([1001] * contexts[1]))
        self.max_tokens = 2
        self.capture_diagnostics = manifest.get("diagnostic_mode") is True

    def run_request(self, arm: str, context_tokens: tuple[int, ...]) -> dict:
        if arm not in ("ordinary", "paged") or context_tokens != self.contexts:
            raise ValueError("B2 contexts differ from manifest")
        from mlx2.runtime.generate import BatchGenerator

        mx = self.mx
        lock = RLock()
        batch = BatchGenerator(self.model, max_tokens=2,
                               prefill_batch_size=2, completion_batch_size=2,
                               prefill_step_size=max(self.contexts), stop_tokens=[])
        owners = candidate = None
        installed = set()
        uids = ()
        model_type = type(self.model)
        original_call = model_type.__call__
        forward_calls = 0

        def count_forward(instance, *args, **kwargs):
            nonlocal forward_calls
            if instance is self.model:
                forward_calls += 1
            return original_call(instance, *args, **kwargs)

        model_type.__call__ = count_forward
        outputs: dict[int, list[int]] = {}
        receipts: dict[int, list[dict]] = {}
        widths: dict[int, list[int]] = {}
        logprob_diagnostics: dict[int, list[dict]] = {}
        ready_step_ms = None
        ready_native_host_profile = None
        published = arm == "ordinary"
        try:
            with lock:
                uids = tuple(batch.insert([list(prompt) for prompt in self.prompts],
                                          max_tokens=[2, 2]))
            if len(uids) != 2:
                raise RuntimeError("B2 request insertion did not produce two UIDs")
            outputs = {uid: [] for uid in uids}
            receipts = {uid: [] for uid in uids}
            widths = {uid: [] for uid in uids}
            logprob_diagnostics = {uid: [] for uid in uids}
            if arm == "paged":
                from mlx2.runtime.paged_native_batch_lifecycle import (
                    prepare_queued_native_first_response,
                    run_research_native_queued_qwen3,
                )
                from mlx2.runtime.paged_request_transaction import CandidateRequest
                from mlx2.runtime.qwen3_paged_graph_factory import (
                    create_shared_qwen3_graph_pack,
                )

                owners, candidate = create_shared_qwen3_graph_pack(
                    self.adapter,
                    tuple((self.revision, len(prompt), 2) for prompt in self.prompts),
                    permit_candidate=True, profile_host=self.capture_diagnostics)
                for uid, prompt, owner in zip(uids, self.prompts, owners):
                    probe = run_research_native_queued_qwen3(
                        batch, lock, CandidateRequest(
                            uid, self.revision, len(prompt), ("kv",)),
                        owner, candidate, research_permit=True)
                    if probe.reason != "research_executed" or not probe.research_executed:
                        raise RuntimeError(f"research packed prefill refused: {probe.reason}")
                    preparation = prepare_queued_native_first_response(
                        batch, lock, owner, probe)
                    if (preparation.reason != "native_continuation_not_installed" or
                            preparation.generation is None):
                        raise RuntimeError(f"native packed preparation refused: {preparation.reason}")
                    with lock:
                        result = batch.install_native_queued(
                            preparation, owner, candidate, lock,
                            permit_native=True, research_only=True)
                    if (result.get("reason") != "native_installed" or
                            result.get("selected") is not False or
                            result.get("research_only") is not True):
                        raise RuntimeError(f"research packed handoff refused: {result}")
                    installed.add(uid)
                published = True
            for _ in range(8):
                ready = all(len(outputs[uid]) == 1 for uid in uids)
                before_host = (candidate.backend.profile_counters_snapshot()
                               if ready and arm == "paged" and self.capture_diagnostics
                               else None)
                begin = time.perf_counter_ns() if ready else None
                try:
                    with lock:
                        _, responses = batch.next()
                except Exception as exc:
                    exc.physical_counters = _failure_physical_counters(candidate, before_host)
                    raise
                failures = batch.take_lane_failures()
                if failures:
                    # Failure objects can embed one full native read-plan dict
                    # per layer. Keep the receipt small while preserving the
                    # physical counters needed to diagnose missed grouping.
                    failure = RuntimeError(
                        f"paired B2 generator failed in {len(failures)} lane(s)")
                    failure.physical_counters = _failure_physical_counters(
                        candidate, before_host)
                    raise failure
                if ready and len(responses) != 2:
                    raise RuntimeError("ready B2 did not emit two responses in one generator step")
                for response in responses:
                    if response.uid not in outputs or len(outputs[response.uid]) >= 2:
                        raise RuntimeError("paired B2 response UID or count drifted")
                    mx.eval(response.logprobs)
                    outputs[response.uid].append(int(response.token))
                    if self.capture_diagnostics:
                        top = mx.argsort(response.logprobs)[-2:][::-1]
                        mx.eval(top)
                        logprob_diagnostics[response.uid].append({
                            "top_token_ids": [int(top[0].item()), int(top[1].item())],
                            "top_logprobs": [float(response.logprobs[int(top[0].item())].item()),
                                             float(response.logprobs[int(top[1].item())].item())],
                            "selected_logprob": float(response.logprobs[int(response.token)].item()),
                        })
                    receipts[response.uid].append(dict(response.mtp_receipt or {}))
                    widths[response.uid].append(response.execution_width)
                    if ready and arm == "paged" and response.execution_width != 2:
                        raise RuntimeError("native B2 response was not physically packed")
                mx.synchronize()
                if ready:
                    ready_step_ms = (time.perf_counter_ns() - begin) / 1e6
                    if before_host is not None:
                        # The timed step is over; copy detailed read plans once.
                        after_host = candidate.backend.profile_snapshot()
                        ready_native_host_profile = {
                            "host_ns": {name: after_host["host_ns"][name] - value
                                        for name, value in before_host["host_ns"].items()},
                            "reads": list(after_host["reads"][before_host["read_count"]:]),
                            "q1_tile_dispatches": (after_host["q1_tile_dispatches"] -
                                                   before_host["q1_tile_dispatches"]),
                            "grouped_q1_writes": (after_host["grouped_q1_writes"] -
                                                  before_host["grouped_q1_writes"]),
                            "native_write_dispatches": (
                                after_host["native_write_dispatches"] -
                                before_host["native_write_dispatches"]),
                            "write_epoch_delta": (after_host["write_epoch"] -
                                                  before_host["write_epoch"]),
                        }
                    break
            if (ready_step_ms is None or
                    any(len(outputs[uid]) != 2 for uid in uids)):
                raise RuntimeError("paired ready B2 step or outputs missing")
            # The final response may have already retired its native owner.
            # Capture publication evidence from the response retained at the
            # live step, never by reopening a closed owner after completion.
            state_diagnostics = (retained_native_state_diagnostics(receipts, uids)
                                 if arm == "paged" and self.capture_diagnostics else [])
            with lock:
                batch.remove(list(uids))
            mx.synchronize()
            if arm == "paged":
                writer = candidate.backend.writer
                pending, retained, released = retired_native_state(writer)
                first_reads = max(receipts[uid][0].get("native_read_calls", 0) for uid in uids)
                last = receipts[uids[-1]][-1]
                reads = last.get("native_read_calls")
                terminals = last.get("terminal_successes")
                if (type(reads) is not int or type(terminals) is not int or
                        reads != terminals or reads - first_reads != len(self.model.layers) or
                        any(receipts[uid][-1].get("route") !=
                            "native_qwen3_paged_graph_research" or
                            receipts[uid][-1].get("packed_lanes") != 2 or
                            receipts[uid][-1].get("native_span_counts") !=
                            [2] * len(self.model.layers) or
                            receipts[uid][-1].get("native_read_delta") != len(self.model.layers)
                            for uid in uids)):
                    raise RuntimeError("one physical two-span native read per layer missing")
                route = dict(last)
                route["serving_selected"] = False
            else:
                pending = retained = reads = terminals = first_reads = 0
                released = True
                route = {"route": "ordinary", "selected": True,
                         "observed_used": True}
            return {
                "arm": arm, "request_inserted": True, "admitted": True,
                "cache_transaction_published": published,
                "response_emitted": True, "sampled_output": True,
                "synchronized": True, "request_removed": True,
                "request_state_released": released,
                "ordered_context_tokens": list(self.contexts),
                "output_token_ids": [outputs[uid] for uid in uids],
                "logprob_diagnostics": [logprob_diagnostics[uid] for uid in uids],
                "native_state_diagnostics": state_diagnostics,
                "sampler_config": {"mode": "default_argmax", "custom_samplers": False,
                                   "logits_processors": False},
                "output_token_id": outputs[uids[-1]][-1],
                "model_layers": len(self.model.layers),
                "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "route_receipt": route,
                "response_receipts": [receipts[uid] for uid in uids],
                "response_execution_widths": [widths[uid] for uid in uids],
                "ordinary_model_forward_calls": forward_calls,
                "paged_read_calls": reads,
                "prefill_read_calls": first_reads,
                "decode_read_calls": reads - first_reads,
                "terminal_successes": terminals,
                "pending_native_epochs": pending, "retained_pages": retained,
                "ready_pack_step_ms": ready_step_ms,
                "ready_native_host_profile": ready_native_host_profile,
                "scheduler_step": {"scope": "live_scheduler_pack_step",
                                   "reserved": [["decode", 1], ["decode", 1]],
                                   "prefill_rows": 0, "context_tokens": list(self.contexts),
                                   "active_lane_count": 2,
                                   "generator_step": True, "sampled_response": True,
                                   "synchronized": True},
            }
        finally:
            model_type.__call__ = original_call
            if uids:
                with lock:
                    batch.remove(list(uids))
            batch.close()
            if owners is not None:
                for uid, owner in zip(uids, owners):
                    if uid not in installed:
                        owner.close()
                        owner.reap_retired()
                        owner.reap_quarantine()


def create(manifest: dict) -> StagedGraphRequestDriver:
    return StagedGraphRequestDriver(manifest)
