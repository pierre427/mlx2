# SPDX-License-Identifier: MIT
"""Explicit V4.1 tiled IQ1 route with block-end expert miss resolution.

Attention writes correct cache/shared state before experts execute. A missing
expert invalidates only the routed branch: retain its exact inputs and repair
that branch before another block or APCv2 publication can consume the output.
See provenance/deepseek-v41-iq1-tiled-adapter.json.
"""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..runtime.io_ownership import ReadPool, storage_owner
from .deepseek_v41_iq1_weights import CompactV41Weights, _inventory
from .deepseek_v41_mlx_weights import _refuse_quarantined_uploads

NUMERIC_PROFILE = "native_iq1_s_tiled"
REVISION = "mlx2.v41-iq1-tiled-block-end-repair.v1"


def numerical_profile(tiled_decode):
    if tiled_decode == "row_exact":
        return NUMERIC_PROFILE, REVISION
    if tiled_decode == "column_f32":
        from ..runtime.gguf.v41_iq1_column import NUMERIC_PROFILE as COLUMN_PROFILE
        from ..runtime.gguf.v41_iq1_column import REVISION as COLUMN_REVISION

        return COLUMN_PROFILE, REVISION + ":" + COLUMN_REVISION
    raise ValueError("unsupported tiled IQ1 decode arithmetic")


def _host_route_ids(ids):
    """The sole route-ID readback, reachable only on miss or explicit slow use."""
    return np.asarray(ids)


def _arrays(value):
    import mlx.core as mx

    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)


class TiledV41Weights(CompactV41Weights):
    compute_policy = NUMERIC_PROFILE + ":" + REVISION

    def __init__(
        self,
        artifact,
        *,
        expert_cache_bytes,
        expert_read_workers=2,
        expert_prefill_policy="reference",
        arena_completion="sync",
        tiled_decode="row_exact",
    ):
        from ..runtime.gguf.v41_iq1_tiled import REVISION as KERNEL_REVISION
        from .deepseek_v41_iq1_arena import TiledExpertStore, tiled_weight_plan

        _refuse_quarantined_uploads()
        if expert_prefill_policy not in ("reference", "segmented_nax_fp16"):
            raise ValueError("unsupported tiled IQ1 expert prefill profile")
        if arena_completion not in ("sync", "async"):
            raise ValueError("unsupported tiled arena completion policy")
        self.arena_completion = arena_completion
        self.tiled_decode = tiled_decode
        self.numeric_profile, numerics = numerical_profile(tiled_decode)
        self.compute_policy = self.numeric_profile + ":" + numerics
        self._publication_values = ()
        self.expert_prefill_policy = expert_prefill_policy
        self.artifact = {**artifact, "manifest": deepcopy(artifact["manifest"])}
        self.root = Path(artifact["root"]).resolve()
        self.metadata = deepcopy(artifact["metadata"])
        self.bounds = tiled_weight_plan(
            self.artifact, expert_cache_bytes, read_workers=expert_read_workers
        )
        self.tensors = {
            name: SimpleNamespace(name=name, **row)
            for name, row in self.artifact["manifest"]["tensors"].items()
        }
        self._dense, self._quant, self._packed = {}, {}, {}
        self._pending_dense, self._pending_dense_host = {}, None
        self._loaded = self._parked = self._closed = False
        self._error = None
        (
            self.scheduled_expert_calls,
            self.scheduled_dense_calls,
            self.scheduled_grouped_calls,
        ) = {}, {}, {}
        self.scheduled_expert_buffer_operations = self.expert_buffer_counters = {}
        self.block_transactions = {
            "blocks": 0,
            "all_hit": 0,
            "misses": 0,
            "repaired": 0,
            "route_id_readbacks": 0,
            "flag_readbacks": 0,
            "direct_ensured_calls": 0,
        }
        self._block_active = False
        self._dense_files, _, _ = _inventory(self.artifact)
        self._read_pool = ReadPool(
            1,
            "v41-tiled-dense",
            timeout=60.0,
            max_pending=1,
            owner=storage_owner("v41-tiled-dense", self.root),
        )
        self.expert_stream = None
        try:
            self.expert_stream = TiledExpertStore(
                self.artifact,
                expert_cache_bytes,
                read_workers=expert_read_workers,
                _owned_manifest=self.artifact["manifest"],
                _bounds=self.bounds,
            )
        except BaseException:
            self._read_pool.close()
            raise
        self.compute_policy += (
            ":arena="
            + self.expert_stream.apc_revision()
            + ":kernel="
            + KERNEL_REVISION
            + ":completion="
            + arena_completion
        )

    def _routed(self, x, ids, scores, prefix):
        import mlx.core as mx

        from ..runtime.gguf import v41_iq1_tiled
        from ..runtime.gguf.segmented_prefill import segmented_prefill_active

        self._ready()
        args = self.artifact["args"]
        if (
            x.ndim != 2
            or x.shape[1] != args.hidden_size
            or not 1 <= x.shape[0] <= 1024
            or ids.dtype != mx.uint32
            or ids.ndim != 2
            or ids.shape[0] != x.shape[0]
            or not 1 <= ids.shape[1] <= args.num_experts_per_tok
            or scores.shape != ids.shape
        ):
            raise ValueError("tiled IQ1 activation/routing shape mismatch")
        bank = self.expert_stream.bank(prefix)
        prefill = (
            self.expert_prefill_policy == "segmented_nax_fp16"
            and segmented_prefill_active(x.shape[0])
        )
        missing = mx.any(
            v41_iq1_tiled.missing_routes(ids, bank.slot_map, bank.capacity)
        )
        kernel = v41_iq1_tiled.moe
        route = "v41_iq1_tiled_nax_fp16" if prefill else "v41_iq1_tiled_row_exact"
        if not prefill and self.tiled_decode == "column_f32":
            from ..runtime.gguf.v41_iq1_column import moe as kernel

            route = "v41_iq1_tiled_column_f32"
        output = kernel(
            x,
            ids,
            scores,
            bank.planes,
            bank.slot_map,
            capacity=bank.capacity,
            hidden_size=args.hidden_size,
            intermediate_size=args.moe_intermediate_size,
            prefill=prefill,
            counters=self.expert_buffer_counters,
        )
        self.scheduled_expert_calls[route] = (
            self.scheduled_expert_calls.get(route, 0) + 1
        )
        return output, missing

    def _missing(self, flag):
        self.block_transactions["flag_readbacks"] += 1
        return bool(flag.item())

    def _ensure_routes(self, prefix, ids):
        self.block_transactions["route_id_readbacks"] += 1
        self.expert_stream.ensure(prefix, _host_route_ids(ids))

    def evaluate_block(self, model, block, h, pre, state, cache, shared, *, taps):
        """Resolve expert misses without replaying valid attention/cache work."""
        import mlx.core as mx

        if self._block_active or not self.expert_stream._active:
            raise ValueError("tiled block requires one owned expert execution window")
        self._block_active = True
        self.block_transactions["blocks"] += 1
        try:
            # No state from this forward may be published until all blocks and
            # the final output evaluation complete in the owning model.
            if not all(item._in_forward for item in cache):
                raise ValueError("tiled block requires unpublished forward state")
            attended, x, ffn_pre, post, comb = block.attention_half(
                h, pre, state, cache, shared
            )
            ids, scores = block.router(x)
            width = model.args.num_experts_per_tok
            ids, scores = ids.reshape(-1, width), scores.reshape(-1, width)
            flat = x.reshape(-1, model.args.hidden_size)
            prefix = block.prefix[:-1]
            routed, missing = self._routed(flat, ids, scores, prefix)
            shared_out = block.shared_expert(x)
            output = block.finish(
                attended, routed.reshape(x.shape), shared_out, post, comb
            )
            # The miss scalar is evaluated with the block, never before its
            # expert dispatch. Retained exact operands make a retry independent
            # of changes to cache arrays, including in-place CSA2 pool appends.
            values = (
                output,
                ffn_pre,
                [v.state for v in cache],
                shared,
                taps,
                missing,
                attended,
                flat,
                ids,
                scores,
                post,
                comb,
                shared_out,
            )
            if self.arena_completion == "async":
                # Submit the complete branch before waiting for the small router
                # flag, allowing queued expert work to overlap host graph setup.
                mx.async_eval(list(_arrays(values)))
            else:
                model._weight_eval(*values)
            if not self._missing(missing):
                self.block_transactions["all_hit"] += 1
            else:
                self.block_transactions["misses"] += 1
                if self.arena_completion == "async":
                    # The provisional branch may reference the old bank. Finish
                    # it before rebuilding; only the valid retry is deferred.
                    mx.eval(list(_arrays(values)))
                self._ensure_routes(prefix, ids)
                routed, missing = self._routed(flat, ids, scores, prefix)
                output = block.finish(
                    attended, routed.reshape(x.shape), shared_out, post, comb
                )
                if self.arena_completion == "async":
                    mx.async_eval(output, ffn_pre, missing)
                else:
                    model._weight_eval(
                        output, ffn_pre, [v.state for v in cache], shared, taps, missing
                    )
                if self._missing(missing):
                    raise RuntimeError(
                        "tiled bank still misses routes after an exact fill"
                    )
                self.block_transactions["repaired"] += 1
            if self.arena_completion == "async":
                # A real copy primitive depends on the complete output producer;
                # it is not an unrelated scalar or a reference-count heuristic.
                marker = mx.asarray(output.reshape(-1)[:1], copy=True)
                publication = tuple(_arrays(([v.state for v in cache], shared, taps)))
                mx.async_eval(output, ffn_pre, marker, list(publication))
                self._publication_values = publication
                self.expert_stream.defer_epoch(marker)
            return output, ffn_pre
        finally:
            self._block_active = False

    def moe(self, x, ids, scores, prefix, *, ticket=None):
        """Explicit eager slow path for standalone calls, never unchecked output.

        Ordinary serving uses evaluate_block. Direct/layerwise callers must
        ensure routes first, so a bypass cannot publish missing-expert zeros.
        """
        from contextlib import nullcontext

        if ticket is not None or self._block_active:
            raise ValueError(
                "tiled direct MoE cannot bypass an active block transaction"
            )
        owned = not self.expert_stream._active
        with self.expert_stream.window() if owned else nullcontext():
            self._ensure_routes(prefix, ids)
            output, missing = self._routed(x, ids, scores, prefix)
            self.evaluate_dense_window(output, missing)
            if self._missing(missing):
                raise RuntimeError("tiled direct MoE bank misses ensured routes")
            self.block_transactions["direct_ensured_calls"] += 1
            return output

    def dense_receipt(self):
        result = super().dense_receipt()
        result.update(
            numeric_profile=self.numeric_profile,
            tiled_decode=self.tiled_decode,
            block_transactions=dict(self.block_transactions),
            routing="GPU slot map; scalar miss readback after block submission",
            completion=(
                "at most two marker-owned layer epochs"
                if self.arena_completion == "async"
                else "one layer per fenced epoch"
            ),
            asynchronous_epochs=self.arena_completion == "async",
        )
        return result

    def evaluate_dense_window(self, *values):
        """Embedding/head/direct-call boundary; drain before state publication."""
        import mlx.core as mx

        mx.eval(list(_arrays(values)), list(self._publication_values))
        self.expert_stream.drain_epochs()
        self._publication_values = ()

    def close(self):
        super().close()
        self._publication_values = ()
