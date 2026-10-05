"""Default-off dense Qwen3 packed-row serving seam.

The native backend is intentionally absent. A backend must publish every KV
suffix after terminal write success and retain each read pin to terminal proof.
This module never substitutes a paged cache for the ordinary APCv2 cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from ..runtime.paged_attention_pack import (
    PackedTokenRead,
    prepare_packed_token_read,
    prepare_staged_token_read,
)
from ..runtime.paged_kv_token import PagedKVTokenOwner


@dataclass(frozen=True)
class PackedLane:
    """One request's new token suffix and one private owner per model layer."""

    token_ids: tuple[int, ...]
    layers: tuple[PagedKVTokenOwner, ...]


class PagedQwen3Backend(Protocol):
    """Native-read/COW publication boundary; CPU fakes may implement it."""

    def append_completed(self, owners: tuple[PagedKVTokenOwner, ...],
                         keys: object, values: object,
                         row_counts: tuple[int, ...]) -> None: ...

    def read_completed(self, use: PackedTokenRead, queries: object,
                       *, scale: float) -> object: ...


class Qwen3PackedCandidate:
    """One bounded packed forward; selection must be explicit per invocation.

    ``append_completed`` must return only after accepted KV publication. The
    reader backend must submit and close ``use`` with a real terminal proof.
    Failure after submission must keep its pin until that proof arrives.
    """

    def __init__(self, model, backend: PagedQwen3Backend):
        self.model = model
        self.backend = backend
        self.state_planes = ("kv",)

    @property
    def native_layer_count(self):
        return len(self.model.layers)

    def forward_staged(self, lanes: tuple[PackedLane, ...], branches: tuple,
                       *, permit_candidate: bool = False):
        """Explicit B1/B2 private graph; all native terminals precede return.

        The caller still owns branch companion staging, atomic publication,
        sampler, response, and rollback. This path is not a serving selection.
        """
        import mlx.core as mx

        from ..runtime.paged_native_atomic_owner import NativeBranch
        from ..runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend

        if not permit_candidate:
            raise RuntimeError("staged Qwen3 graph requires explicit enablement")
        args = self.model.args
        if (args.model_type != "qwen3" or args.num_experts or
                args.rope_scaling is not None or args.head_dim not in (128, 256) or
                args.num_attention_heads % args.num_key_value_heads):
            raise ValueError("staged graph requires dense full-attention Qwen3")
        if (type(self.backend) is not NativeQwen3PagedBackend or
                type(lanes) is not tuple or type(branches) is not tuple or
                not 1 <= len(lanes) == len(branches) <= 2):
            raise ValueError("staged graph requires one or two native branch lanes")
        depth = len(self.model.layers)
        writer = self.backend.writer
        counts = tuple(len(lane.token_ids) for lane in lanes)
        offsets = tuple(branch._origin.layers[0].offset for branch in branches)
        for lane, branch, count, offset in zip(lanes, branches, counts, offsets):
            if (type(lane) is not PackedLane or type(branch) is not NativeBranch or
                    branch._closed or type(lane.token_ids) is not tuple or not count or
                    any(type(token) is not int or token < 0 for token in lane.token_ids) or
                    count != branch._request.proposed_rows or
                    type(lane.layers) is not tuple or len(lane.layers) != depth or
                    any(actual is not expected or actual.writer is not writer or
                        actual.offset != offset for actual, expected in
                        zip(lane.layers, branch.layers))):
                raise ValueError("staged lane does not match its private native branch")
        if len({id(branch) for branch in branches}) != len(branches):
            raise ValueError("staged graph branches must be distinct")
        profile = lanes[0].layers[0].profile
        if (profile.dtype != "float16" or profile.kv_heads != args.num_key_value_heads or
                profile.head_dim != args.head_dim or
                any(layer.profile != profile for lane in lanes for layer in lane.layers)):
            raise ValueError("staged lanes disagree with native Qwen3 geometry")
        if writer.pending_epochs or writer.ledger.pending_count:
            raise RuntimeError("staged graph requires an idle native writer")
        tokens = tuple(token for lane in lanes for token in lane.token_ids)
        hidden = self.model.paged_embed(tokens)
        uses = []
        defer_all = bool(getattr(self, "_defer_staged_q1_eval", False))
        defer_writes = bool(getattr(self, "_defer_staged_q1_writes", False))
        if defer_all and defer_writes:
            raise ValueError("all-Q1 and write-only deferral are mutually exclusive")
        deferred = bool((defer_all or defer_writes) and counts == (1, 1))
        all_layers = tuple(layer for lane in lanes for layer in lane.layers)
        arena = writer.backend
        if deferred:
            arena.begin_deferred_q1(writes_only=defer_writes)
        try:
            for index in range(depth):
                owners = tuple(lane.layers[index] for lane in lanes)
                if getattr(self, "_vector_q1_rope", False) and counts == (1, 1):
                    queries, keys, values = self.model.paged_project(
                        index, hidden, counts, offsets, vector_q1_rope=True)
                    self._vector_q1_rope_calls = getattr(
                        self, "_vector_q1_rope_calls", 0) + 1
                else:
                    queries, keys, values = self.model.paged_project(
                        index, hidden, counts, offsets)
                expected_q = (sum(counts), args.num_attention_heads, args.head_dim)
                expected_kv = (sum(counts), args.num_key_value_heads, args.head_dim)
                if (tuple(queries.shape) != expected_q or tuple(keys.shape) != expected_kv or
                        tuple(values.shape) != expected_kv or
                        any(value.dtype != mx.float16 for value in (queries, keys, values))):
                    raise ValueError("staged projected rows do not match the native profile")
                tickets = self.backend.append_staged(owners, keys, values, counts)
                use = prepare_staged_token_read(owners, counts,
                                                query_heads=args.num_attention_heads,
                                                permit_candidate=True,
                                                profile_host=self.backend.profiling_enabled)
                uses.append(use)
                attended = self.backend.read_staged(
                    use, queries, tickets, scale=args.head_dim ** -0.5)
                hidden = self.model.paged_finish_layer(index, hidden, attended)
            logits = self.model.paged_logits(hidden)
            graph_eval_started = time.perf_counter_ns() if self.backend.profiling_enabled else 0
            graph_eval_cpu_started = time.process_time_ns() if self.backend.profiling_enabled else 0
            if deferred:
                arena.deferred_q1_final_evals += 1
            mx.eval(logits)
            if self.backend.profiling_enabled:
                self.backend.host_profile_ns["graph_eval"] += (
                    time.perf_counter_ns() - graph_eval_started)
                self.backend.host_profile_ns["graph_eval_process_cpu"] = (
                    self.backend.host_profile_ns.get("graph_eval_process_cpu", 0) +
                    time.process_time_ns() - graph_eval_cpu_started)
            proofs = self.backend.drain_staged(all_layers, tuple(uses))
        except BaseException:
            if deferred:
                try:
                    self.backend.abort_deferred_q1(all_layers, tuple(uses))
                except BaseException:
                    # Ambiguous evaluation keeps the arena and leases pinned.
                    pass
            raise
        finally:
            if deferred:
                arena.end_deferred_q1(retain_roots=
                                      self.backend.retain_deferred_q1_roots(tuple(uses)))
        for index, proof in enumerate(proofs):
            for lane_index, branch in enumerate(branches):
                branch.prove_staged_layer_read(index, lane_index, proof)
        receipt = {
            "route": "qwen3-paged-staged-research",
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "native_read_calls": self.backend.read_submissions,
            "terminal_successes": self.backend.terminal_successes,
            "packed_lanes": len(lanes),
            "rows": sum(counts),
            "apcv2": "ordinary_route_only",
        }
        if self.backend.profiling_enabled:
            receipt["native_host_profile"] = self.backend.profile_counters_snapshot()
        if deferred:
            receipt["deferred_eval_submissions"] = arena.eval_submission_snapshot()
            receipt["deferred_write_eval"] = defer_writes
        return logits, receipt

    def forward(self, lanes: tuple[PackedLane, ...], *, permit_candidate: bool = False,
                requested_capabilities: frozenset[str] = frozenset({"text"}),
                atomic_branch=None):
        if not permit_candidate:
            raise RuntimeError("Qwen3 paged candidate requires explicit enablement")
        if (type(requested_capabilities) is not frozenset or
                requested_capabilities != frozenset({"text"})):
            raise ValueError("paged Qwen3 candidate supports text only")
        args = self.model.args
        if (args.model_type != "qwen3" or args.num_experts or
                args.rope_scaling is not None):
            raise ValueError("paged candidate requires dense full-attention Qwen3")
        if (args.head_dim not in (128, 256) or
                args.num_attention_heads % args.num_key_value_heads):
            raise ValueError("unsupported paged Qwen3 attention geometry")
        if type(lanes) is not tuple or not lanes or any(type(lane) is not PackedLane for lane in lanes):
            raise ValueError("nonempty packed lanes are required")
        depth = len(self.model.layers)
        if atomic_branch is not None:
            # A native atomic branch belongs to one request. Its exact private
            # layer owners must be the ones read by this forward.
            from ..runtime.paged_native_atomic_owner import NativeBranch
            from ..runtime.qwen3_paged_native_backend import NativeQwen3PagedBackend

            if (type(atomic_branch) is not NativeBranch or atomic_branch._closed or
                    len(lanes) != 1 or type(lanes[0].layers) is not tuple or
                    type(lanes[0].token_ids) is not tuple or
                    len(lanes[0].token_ids) != atomic_branch._request.proposed_rows or
                    type(self.backend) is not NativeQwen3PagedBackend or
                    len(atomic_branch.layers) != depth or
                    any(actual is not expected for actual, expected in
                        zip(lanes[0].layers, atomic_branch.layers))):
                raise ValueError("atomic Qwen3 read requires one matching native branch lane")
        counts = tuple(len(lane.token_ids) for lane in lanes)
        if (any(not count or type(lane.token_ids) is not tuple or
                any(type(token) is not int or token < 0 for token in lane.token_ids) or
                type(lane.layers) is not tuple or len(lane.layers) != depth
                for lane, count in zip(lanes, counts))):
            raise ValueError("each lane needs token rows and one owner per layer")
        offsets = tuple(lane.layers[0].offset for lane in lanes)
        for layer in range(depth):
            owners = tuple(lane.layers[layer] for lane in lanes)
            if any(type(owner) is not PagedKVTokenOwner for owner in owners):
                raise ValueError("paged candidate requires token-page owners")
            if len({id(owner) for owner in owners}) != len(owners):
                raise ValueError("each lane needs a distinct layer owner")
            first = owners[0]
            if (first.profile.kv_heads != args.num_key_value_heads or
                    first.profile.head_dim != args.head_dim or
                    first.profile.dtype not in ("float16", "bfloat16") or
                    any(owner.writer is not first.writer or owner.profile != first.profile
                        for owner in owners)):
                raise ValueError("layer owners must share the Qwen3 page geometry and arena")
            for owner, count, offset in zip(owners, counts, offsets):
                if owner.offset != offset:
                    raise ValueError("all Qwen3 layer offsets must agree")
                owner.accepted_handles()  # Refuses pending, failed, or closed state.
                owner.planned_spans(count)  # Refuses shared tails until native COW exists.
        tokens = tuple(token for lane in lanes for token in lane.token_ids)
        hidden = self.model.paged_embed(tokens)
        simulated_reads = 0
        for layer in range(depth):
            owners = tuple(lane.layers[layer] for lane in lanes)
            if tuple(owner.offset for owner in owners) != offsets:
                raise ValueError("all Qwen3 layer offsets must agree")
            queries, keys, values = self.model.paged_project(layer, hidden, counts, offsets)
            if (getattr(queries, "shape", None) !=
                    (sum(counts), args.num_attention_heads, args.head_dim) or
                    getattr(keys, "shape", None) !=
                    (sum(counts), args.num_key_value_heads, args.head_dim) or
                    getattr(values, "shape", None) !=
                    (sum(counts), args.num_key_value_heads, args.head_dim) or
                    any(str(value.dtype).split(".")[-1] != owners[0].profile.dtype
                        for value in (queries, keys, values))):
                raise ValueError("Qwen3 projected rows do not match the paged profile")
            self.backend.append_completed(owners, keys, values, counts)
            if any(owner.offset != old + count for owner, old, count in zip(owners, offsets, counts)):
                raise RuntimeError("paged backend returned before KV publication")
            use = prepare_packed_token_read(owners, counts,
                                            query_heads=args.num_attention_heads,
                                            permit_candidate=True)
            try:
                if atomic_branch is None:
                    attended = self.backend.read_completed(
                        use, queries, scale=args.head_dim ** -0.5)
                else:
                    attended = self.backend.read_completed(
                        use, queries, scale=args.head_dim ** -0.5,
                        terminal_proof=lambda read, event, index=layer:
                            atomic_branch.prove_layer_read(index, read, event))
            except Exception:
                if use.state == "prepared":
                    use.abort_before_submit()
                raise
            if use.state != "closed":
                if use.state == "prepared":
                    use.abort_before_submit()
                raise RuntimeError("paged read backend returned without terminal proof")
            hidden = self.model.paged_finish_layer(layer, hidden, attended)
            simulated_reads += 1
        return self.model.paged_logits(hidden), {
            "route": "qwen3-paged-candidate",
            "implemented": True,
            "qualified": False,
            "selected": True,
            "observed_used": False,
            "simulated_or_backend_reads": simulated_reads,
            "rows": sum(counts),
            "apcv2": "ordinary_route_only",
        }
