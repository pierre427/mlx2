"""Adaptive prompt-copy proposals with V4's exact snapshot/replay owner.

Proposal generation is model-free. Target verification is always ordinary B1
math; no packed tree, learned DSpark or speedup claim is made.
"""

from .deepseek_v4_external import DeepSeekV4VerifyRows, DeepSeekV4VerifyTransaction


class DeepSeekV41VerifyRows(DeepSeekV4VerifyRows):
    """Exclusive width-one verifier with bounded local frontier checkpoints."""

    def __init__(self, model, rows, layers):
        from ..runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache

        super().__init__(model, rows, layers)
        if any(type(c) is not DeepSeekV41LayerCache for c in self.rows[0]):
            raise ValueError("V4.1 local frontiers require source-bound V4.1 caches")

    def begin(self, lengths, *, plain_single_token=False):
        if (self._poisoned or self._active is not None and not self._active.closed
                or len(lengths) != 1 or type(lengths[0]) is not int or not 1 <= lengths[0] <= 6):
            raise ValueError("invalid or active V4.1 local-frontier transaction")
        self._active = DeepSeekV41VerifyTransaction(self, lengths[0], plain_single_token)
        return self._active


class DeepSeekV41VerifyTransaction(DeepSeekV4VerifyTransaction):
    def __init__(self, *args):
        self._locals = []
        super().__init__(*args)

    @property
    def retained_nbytes(self):
        return super().retained_nbytes+sum(x.nbytes for row in self._locals for _, values, _, _ in row
                                           for x in values if x is not None)

    def _close(self):
        self._locals = []
        super()._close()

    def _forward(self, inputs, taps):
        import mlx.core as mx
        from ..runtime.models.deepseek_v4_cache import _copy

        if not taps and getattr(self.owner.model, "verify_rows_policy", "serial") == "layerwise":
            logits, self._locals = self.owner.model.verify_layerwise(inputs, self.caches, retain_frontiers=True)
            return logits, None

        logits, features = [], []
        for i in range(inputs.shape[1]):
            out, hidden = super()._forward(inputs[:, i:i+1], taps)
            logits.append(out)
            if taps:
                features.append(hidden)
            if i+1 < self.width:
                row = []
                for c in self.caches:
                    c.validate_state()
                    c.cache_operations_contract.validate_publication(c)
                    local = _copy(c.xp, (c.keys, c.buf_kv, c.buf_gate, c.hash_tail))
                    row.append((c.meta_state, local, c._pool_count, c._pooled is not None))
                self._locals.append(row)
        return mx.concatenate(logits, 1), mx.concatenate(features, 1) if taps else None

    def commit(self, accepted_lengths):
        if (self.closed or self._inputs is None or self.owner._active is not self
                or len(accepted_lengths) != 1 or type(accepted_lengths[0]) is not int
                or not 0 <= accepted_lengths[0] <= self.width):
            raise ValueError("invalid V4.1 verified prefix commit")
        accepted = accepted_lengths[0]
        if accepted in (0, self.width):
            return super().commit(accepted_lengths)
        from ..runtime.models.deepseek_v4_cache import _copy

        try:
            row = self._locals[accepted-1]
            end = self._base_offset+self.width
            for c, (meta, _, count, present) in zip(self.caches, row):
                if (c.pending or c.offset != end or meta[:9] != c.meta_state[:9]
                        or int(meta[-1]) != self._base_offset+accepted
                        or c._pool_count < count or present and c._pooled is None):
                    raise ValueError("V4.1 verified prefix identity/frontier changed")
            # Appends never modify a source pool's visible prefix. Restore all
            # mutable local planes and expose only the accepted pool counts.
            for c, (meta, local, count, present) in zip(self.caches, row):
                c.keys, c.buf_kv, c.buf_gate, c.hash_tail = _copy(c.xp, local)
                c.offset, c._pool_count = int(meta[-1]), count
                if not present:
                    c._pooled = c._index_keys = None
                    c._capacity = 0
                c._in_forward = False
                c.validate_state()
                c.cache_operations_contract.validate_publication(c)
            stats = getattr(self.owner.model, "verified_commit_stats", None)
            if stats is not None:
                stats["partial_commits"] = stats.get("partial_commits", 0)+1
                stats["replay_rows_avoided"] = stats.get("replay_rows_avoided", 0)+accepted
            result = [list(self.caches)]
            self._close()
            return result
        except BaseException:
            self.abort()
            raise


def local_frontier_drafter(model, *, num_draft):
    from ..runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel

    class LocalFrontierDraft(PromptLookupDraftModel):
        @property
        def receipt_settings(self):
            return {**super().receipt_settings, "verified_prefix_commit": "v41_local_frontier",
                    "arithmetic_rows": 1, "maximum_checkpoint_rows": num_draft}

        def extra_memory_reservation(self, cache, *, append, pending_tokens, prefill):
            if prefill:
                return 0
            c = model.args
            local = c.num_hidden_layers*c.sliding_window*c.head_dim*4
            local += len(c.kv_source_layer_ids)*c.head_dim*8+16
            # Saved boundaries plus one restore copy, with metadata slack.
            return (num_draft+1)*local+(1 << 20)

    return LocalFrontierDraft(model, num_draft=num_draft)


def create_batch(model, draft_model, *, binding, completion_batch_size=1, **kwargs):
    from ..runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
    from ..runtime.external_speculative import ExternalDraftBatchGenerator

    if (
        completion_batch_size != 1
        or not isinstance(draft_model, PromptLookupDraftModel)
        or draft_model.base is not None
    ):
        raise ValueError("V4.1 requires its model-free B1 copy-chain drafter")
    blocked = (
        "ast_tree_tile_rows",
        "target_tree_executor_provider",
        "ast_target_tree_executor_provider",
        "progressive_verification_tile",
        "dynamic_singleton_tree",
        "ast_b1_tile_policy",
    )
    if any(kwargs.get(k) for k in blocked):
        raise ValueError(
            "V4.1 packed-tree AST is unavailable; select the ordinary copy chain"
        )

    class OwnedGenerator(ExternalDraftBatchGenerator):
        def _target_owner(self, rows):
            owner = DeepSeekV41VerifyRows if getattr(self.model, "verification_commit_policy", "replay") == "local_frontier" else DeepSeekV4VerifyRows
            return owner(self.model, rows, self.layers)

        def _cohort_target_forward(
            self, inputs, transaction, cohort, *, site, taps=True
        ):
            if not isinstance(transaction, DeepSeekV4VerifyTransaction):
                raise TypeError("foreign V4.1 verification transaction")
            return transaction.forward(inputs, taps=taps)

    return OwnedGenerator(
        model,
        draft_model=draft_model,
        binding=binding,
        completion_batch_size=1,
        **kwargs,
    )
