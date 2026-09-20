# SPDX-License-Identifier: Apache-2.0
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
from __future__ import annotations
from typing import Any, Sequence
import mlx.core as mx
from .models.cache import ArraysCache, KVCache, QuantizedKVCache
from .models.qwen4_exp import (
    BatchQSAKVCache,
    QSACompactBlocks,
    QSAKVCache,
    Qwen4ArraysCache,
    qsa_dense_attention_from_selection,
)
from .models.qwen4_qsa_indexed import (
    QSAIndexedProbeDeclined,
    qwen4_qsa_indexed_private_delta_attention,
    qwen4_qsa_indexed_private_delta_exact_set_attention,
    qwen4_qsa_indexed_private_delta_exact_set_preflight,
    qwen4_qsa_indexed_private_delta_preflight,
    qwen4_qsa_private_delta_min_context,
)


class SegmentedBatchUnsupported(TypeError):
    """The row caches cannot be represented by the first batched consumer."""


def _host_offset(cache: Any) -> int:
    offset = getattr(cache, "offset", None)
    if isinstance(offset, int):
        return offset
    size = getattr(cache, "size", None)
    if callable(size):
        value = size()
        if isinstance(value, int):
            return value
    raise SegmentedBatchUnsupported(
        f"{type(cache).__name__} has no host-readable scalar offset"
    )


def _pad_sequence(value: mx.array, left: int, right: int, axis: int) -> mx.array:
    padding = [(0, 0)] * value.ndim
    padding[axis] = (int(left), int(right))
    return value if not left and (not right) else mx.pad(value, padding)


def _slice_row_tree(values, row: int):
    return [
        None if value is None else mx.contiguous(value[row : row + 1])
        for value in values
    ]


class SegmentedBatchQSAKVCache(BatchQSAKVCache):
    """QSA batch ABI backed by independent, unquantized B1 QSA caches."""

    def __init__(
        self, rows: Sequence[QSAKVCache], note=None, *, shared_qsa_prefix=False
    ):
        from .qsa_shared_suffix import SharedSuffixQSAKVCache

        plain = bool(rows) and all((type(row) is QSAKVCache for row in rows))
        shared = bool(rows) and all(
            (isinstance(row, SharedSuffixQSAKVCache) for row in rows)
        )
        if not (plain or shared):
            names = ", ".join((type(row).__name__ for row in rows))
            raise SegmentedBatchUnsupported(
                f"true batched QSA requires uniformly plain or shared-suffix rows; got {names or 'none'}"
            )
        self.rows = list(rows)
        self._note = note
        offsets = [_host_offset(row) for row in rows]
        aligned = min(offsets, default=0) // 4 * 4
        if shared:
            bases = [row.base for row in rows]
            if any((base is not bases[0] for base in bases[1:])):
                raise SegmentedBatchUnsupported(
                    "shared-suffix QSA rows do not reference one immutable base"
                )
            self._private_delta_base_tokens = int(bases[0].length)
        else:
            self._private_delta_base_tokens = (
                aligned
                if shared_qsa_prefix and aligned > 0 and (len(set(offsets)) == 1)
                else None
            )
        self._step_lengths = None
        self._right_padding = None
        self._mtp_share_topk = False
        self._mtp_shared_topk = None
        self._mtp_shared_topk_n_blocks = None
        self._qsa_pooled_keys = None
        self._qsa_pooled_ratio = None
        self._qsa_summary_identity = None
        self._qsa_summary_restored = False
        self._qsa_pending_pooled = None
        self.index_keys = None
        self.keys = None
        self.values = None
        self._configure_attention_backend("sdpa")
        self._refresh_geometry()

    def _bump(self, key, amount=1):
        if self._note is not None:
            self._note(key, amount)

    def _refresh_geometry(self):
        lengths = [_host_offset(row) for row in self.rows]
        width = max(lengths, default=0)
        self._base_lengths = lengths
        self._base_width = width
        self._idx = width
        self.offset = mx.array(lengths)
        self.left_padding = mx.array([width - value for value in lengths])
        self.keys = None
        self.values = None
        self.index_keys = None

    @property
    def batch_size(self):
        return len(self.rows)

    def prepare(self, *, lengths=None, right_padding=None, **_kwargs):
        if lengths is None:
            raise ValueError("segmented QSA prepare requires per-row lengths")
        lengths = [int(value) for value in lengths]
        if len(lengths) != len(self.rows) or any((value < 0 for value in lengths)):
            raise ValueError("segmented QSA lengths must cover every row")
        self._refresh_geometry()
        self._step_lengths = lengths
        right_padding = (
            [0] * len(lengths)
            if right_padding is None
            else [int(value) for value in right_padding]
        )
        if len(right_padding) != len(lengths):
            raise ValueError("segmented QSA right padding must cover every row")
        width = max(lengths, default=0)
        if any((width - value != pad for (value, pad) in zip(lengths, right_padding))):
            raise ValueError("segmented QSA lengths/right padding disagree")
        self._right_padding = mx.array(right_padding) if any(right_padding) else None

    def prepare_self_mtp_step(self, **kwargs):
        share = self._mtp_share_topk
        shared = self._mtp_shared_topk
        shared_n_blocks = self._mtp_shared_topk_n_blocks
        self.prepare(**kwargs)
        self._mtp_share_topk = share
        self._mtp_shared_topk = shared
        self._mtp_shared_topk_n_blocks = shared_n_blocks

    def segmented_attention(self, attention, hidden: mx.array, _mask):
        """Run QSA against each B1 history without joining historical K/V.

        The surrounding decoder layer remains batch-shaped, so its projection-
        independent trunk, recurrent layer and MoE stream weights once.  This
        first exact consumer deliberately invokes the QSA sublayer per row;
        a future segment-aware attention kernel can fuse those reductions
        without changing the cache contract.
        """
        if self._step_lengths is None:
            raise RuntimeError("segmented attention outside prepare/finalize")
        from .segmented_self_mtp import qsa_private_delta_enabled

        length = int(hidden.shape[1])
        self._arm_row_qsa_share()
        shared_rows = bool(self.rows) and getattr(
            self.rows[0], "supports_shared_qsa_suffix", False
        )
        private_candidate = (
            qsa_private_delta_enabled()
            and self._private_delta_base_tokens is not None
            and (len(set(self._step_lengths)) == 1)
            and (self._step_lengths[0] > 0)
        )
        if private_candidate:
            from .segmented_self_mtp import (
                note_qsa_exact_set_fold_event,
                note_qsa_private_delta_event,
                qsa_private_delta_exact_set_fold_enabled,
            )

            note_qsa_private_delta_event("request", width=length)
            if not shared_rows and int(
                self._private_delta_base_tokens
            ) < qwen4_qsa_private_delta_min_context(length):
                (admitted, reason) = (False, "context_out_of_range")
            else:
                (admitted, reason) = qwen4_qsa_indexed_private_delta_preflight(
                    length=length,
                    base_tokens=int(self._private_delta_base_tokens),
                    head_dim=int(attention.head_dim),
                    num_query_heads=int(attention.num_heads),
                    num_kv_heads=int(attention.num_kv_heads),
                    block_size=int(attention.indexer.compress_ratio),
                    selected_blocks=int(attention.indexer.block_topk),
                    training=bool(attention.training),
                )
            if admitted:
                exact_set_fold = qsa_private_delta_exact_set_fold_enabled() and (
                    not shared_rows
                )
                if exact_set_fold:
                    note_qsa_exact_set_fold_event("request")
                    (exact_admitted, exact_reason) = (
                        qwen4_qsa_indexed_private_delta_exact_set_preflight(
                            batch=len(self.rows),
                            length=length,
                            base_tokens=int(self._private_delta_base_tokens),
                            head_dim=int(attention.head_dim),
                            num_query_heads=int(attention.num_heads),
                            num_kv_heads=int(attention.num_kv_heads),
                            block_size=int(attention.indexer.compress_ratio),
                            selected_blocks=int(attention.indexer.block_topk),
                            training=bool(attention.training),
                        )
                    )
                    if not exact_admitted:
                        note_qsa_exact_set_fold_event("declined", reason=exact_reason)
                        exact_set_fold = False
                output = self._private_delta_attention(
                    attention, hidden, exact_set_fold=exact_set_fold
                )
                self._capture_row_qsa_share()
                return output
            note_qsa_private_delta_event("declined", width=length, reason=reason)
            self._bump("private_delta_preflight_declines")
        width = length
        projected = attention._project_segmented_qsa(hidden)
        outputs = []
        gates = []
        for index, (row, valid) in enumerate(zip(self.rows, self._step_lengths)):
            if valid == 0:
                pre_o_shape = (
                    1,
                    width,
                    int(attention.num_heads) * int(attention.head_dim),
                )
                outputs.append(mx.zeros(pre_o_shape, dtype=hidden.dtype))
                gates.append(mx.zeros(pre_o_shape, dtype=hidden.dtype))
                continue
            row_hidden = hidden[index : index + 1, :valid]
            row_mask = row.make_mask(valid, return_array=True, window_size=None)
            if row_mask is not None and row_mask.ndim == 2:
                row_mask = row_mask[None, None]
            row_projected = tuple(
                (value[index : index + 1, :valid] for value in projected)
            )
            if getattr(row, "supports_shared_qsa_suffix", False):
                (output, gate) = self._shared_suffix_row_attention(
                    attention, row_hidden, row_mask, row, row_projected
                )
            else:
                (output, gate) = attention(
                    row_hidden,
                    row_mask,
                    row,
                    _projected=row_projected,
                    _return_pre_o=True,
                )
            outputs.append(_pad_sequence(output, 0, width - valid, 1))
            gates.append(_pad_sequence(gate, 0, width - valid, 1))
        self._bump("segmented_attention_calls")
        self._bump("independent_lineages_consumed", len(self.rows))
        self._bump("row_state_splits", len(self.rows))
        self._capture_row_qsa_share()
        self._refresh_geometry()
        output = mx.concatenate(outputs, axis=0)
        gate = mx.concatenate(gates, axis=0)
        return attention.o_proj(output * mx.sigmoid(gate))

    def _arm_row_qsa_share(self):
        """Mirror a batched MTP share cycle onto its independent QSA rows."""
        if not self._mtp_share_topk:
            return
        shared = self._mtp_shared_topk
        if shared is not None and int(shared.shape[0]) != len(self.rows):
            raise RuntimeError("segmented QSA shared top-k batch size changed")
        if shared is not None and self._mtp_shared_topk_n_blocks is None:
            raise RuntimeError(
                "segmented QSA shared top-k lacks one common captured block count"
            )
        for index, row in enumerate(self.rows):
            row._mtp_share_topk = True
            if shared is not None:
                row._mtp_shared_topk = mx.contiguous(shared[index : index + 1])
                row._mtp_shared_topk_n_blocks = self._mtp_shared_topk_n_blocks

    def _capture_row_qsa_share(self):
        """Expose per-lineage selections on the batched cycle receipt/state."""
        if not self._mtp_share_topk:
            return
        selected = [row._mtp_shared_topk for row in self.rows]
        shapes = {tuple(value.shape[1:]) for value in selected if value is not None}
        captured = {
            row._mtp_shared_topk_n_blocks
            for row in self.rows
            if row._mtp_shared_topk is not None
        }
        common_grid = len(captured) == 1 and None not in captured
        self._mtp_shared_topk = (
            mx.concatenate(selected, axis=0)
            if selected
            and all((value is not None for value in selected))
            and (len(shapes) == 1)
            and common_grid
            else None
        )
        self._mtp_shared_topk_n_blocks = (
            next(iter(captured)) if common_grid else None
        )

    @staticmethod
    def _shared_suffix_row_attention(
        attention, hidden: mx.array, row_mask, row, projected
    ):
        """Fail closed through the split-aware B1 selector.

        Ragged rows and a private-delta path that declines before mutation
        cannot use stock indexer/cache append APIs on shared-suffix storage.
        Append the row-private
        ledgers once, materialize only the temporary stock consumer view, and
        leave the authoritative row as immutable-base plus suffix.
        """
        (_qg, k_flat, v_flat, projected_qk) = projected
        (batch, length, _) = hidden.shape
        if batch != 1:
            raise RuntimeError("shared-suffix fallback requires one request row")
        offset = _host_offset(row)
        if getattr(row, "_mtp_shared_topk", None) is not None:
            qk_width = int(attention.indexer.n_heads) * int(attention.indexer.head_dim)
            (_q, raw) = mx.split(projected_qk, [qk_width], axis=-1)
            row.append_index_keys(raw.reshape(batch, length, -1))
        selection = attention.indexer.select_shared_suffix(
            hidden, row_mask, row, projected_qk=projected_qk
        )
        k = k_flat.reshape(batch, length, attention.num_kv_heads, attention.head_dim)
        v = v_flat.reshape(batch, length, attention.num_kv_heads, attention.head_dim)
        k = attention.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        k = attention.rope(k, offset=offset)
        row.append_kv(k, v)
        (dense_cache, _receipt) = row.materialize_to_qsa()
        (keys, values) = dense_cache.keys_and_values()
        return attention(
            hidden,
            row_mask,
            dense_cache,
            _projected=projected,
            _return_pre_o=True,
            _selection=selection,
            _fetched_kv=(keys, values),
        )

    def _private_delta_attention(
        self, attention, hidden: mx.array, *, exact_set_fold: bool = False
    ):
        """Consume one attested common prefix plus request-private suffixes."""
        (batch, length, _) = hidden.shape
        base_tokens = int(self._private_delta_base_tokens)
        if length != int(self._step_lengths[0]):
            raise RuntimeError("private-delta QSA requires an unpadded query slab")
        projected = attention._project_segmented_qsa(hidden)
        (qg, k_flat, v_flat, projected_qk) = projected
        row_compacts = []
        row_selections = []
        offsets = []
        row_masks = []
        for row in self.rows:
            offset = _host_offset(row)
            if offset < base_tokens:
                raise RuntimeError("QSA private delta trimmed through its base")
            offsets.append(offset)
            row_mask = row.make_mask(length, return_array=True, window_size=None)
            if row_mask is not None and row_mask.ndim == 2:
                row_mask = row_mask[None, None]
            row_masks.append(row_mask)
        shared_rows = getattr(self.rows[0], "supports_shared_qsa_suffix", False)
        if shared_rows and batch >= 2 and (len(set(offsets)) == 1):
            selections = attention.indexer.select_shared_suffix_batch(
                hidden, row_masks, self.rows, projected_qk=projected_qk
            )
            self._bump("shared_qsa_batched_selections")
        else:
            selections = []
            for index, (row, row_mask) in enumerate(zip(self.rows, row_masks)):
                if getattr(row, "supports_shared_qsa_suffix", False):
                    selection = attention.indexer.select_shared_suffix(
                        hidden[index : index + 1],
                        row_mask,
                        row,
                        projected_qk=projected_qk[index : index + 1],
                    )
                else:
                    selection = attention.indexer(
                        hidden[index : index + 1],
                        row_mask,
                        row,
                        projected_qk=projected_qk[index : index + 1],
                    )
                selections.append(selection)
        for selection in selections:
            compact = selection.compact_blocks()
            if selection.kind != "explicit" or compact is None:
                raise RuntimeError(
                    "QSA private delta requires an explicit compact selection"
                )
            if compact.left_padding is not None:
                raise RuntimeError("QSA private delta rows must be unpadded")
            row_compacts.append(compact)
            row_selections.append(selection)
        (q, gate) = mx.split(
            qg.reshape(batch, length, attention.num_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(batch, length, -1)
        k = k_flat.reshape(batch, length, attention.num_kv_heads, attention.head_dim)
        v = v_flat.reshape(batch, length, attention.num_kv_heads, attention.head_dim)
        q_rows = []
        k_rows = []
        for index, offset in enumerate(offsets):
            row_q = attention.q_norm(q[index : index + 1]).transpose(0, 2, 1, 3)
            row_k = attention.k_norm(k[index : index + 1]).transpose(0, 2, 1, 3)
            q_rows.append(attention.rope(row_q, offset=offset))
            k_rows.append(attention.rope(row_k, offset=offset))
        q = mx.concatenate(q_rows, axis=0)
        k = mx.concatenate(k_rows, axis=0)
        v = v.transpose(0, 2, 1, 3)
        row_keys = []
        row_values = []
        row_delta_keys = []
        row_delta_values = []
        delta_lengths = []
        for index, row in enumerate(self.rows):
            if getattr(row, "supports_shared_qsa_suffix", False):
                (suffix_k, suffix_v) = row.append_kv(
                    k[index : index + 1], v[index : index + 1]
                )
                keys = values = None
                delta_length = int(row.suffix_length)
                row_delta_keys.append(suffix_k)
                row_delta_values.append(suffix_v)
            else:
                (keys, values) = row.update_and_fetch(
                    k[index : index + 1], v[index : index + 1]
                )
                delta_length = int(keys.shape[2]) - base_tokens
                row_delta_keys.append(
                    keys[:, :, base_tokens : base_tokens + delta_length]
                )
                row_delta_values.append(
                    values[:, :, base_tokens : base_tokens + delta_length]
                )
            if delta_length < 0:
                raise RuntimeError("QSA private delta has a negative suffix")
            row_keys.append(keys)
            row_values.append(values)
            delta_lengths.append(delta_length)
        delta_width = max(delta_lengths)

        def suffix_batch(values):
            pieces = []
            for value, delta_length in zip(values, delta_lengths):
                pieces.append(
                    _pad_sequence(
                        value[:, :, :delta_length],
                        0,
                        delta_width - delta_length,
                        axis=2,
                    )
                )
            return mx.concatenate(pieces, axis=0)

        if getattr(self.rows[0], "supports_shared_qsa_suffix", False):
            base_k = self.rows[0].base.keys
            base_v = self.rows[0].base.values
        else:
            base_k = row_keys[0][:, :, :base_tokens]
            base_v = row_values[0][:, :, :base_tokens]
        delta_k = suffix_batch(row_delta_keys)
        delta_v = suffix_batch(row_delta_values)
        total = base_tokens + delta_width
        masks = [compact.causal_mask for compact in row_compacts]
        if all((mask is None for mask in masks)):
            causal_mask = None
        elif any((mask is None for mask in masks)):
            raise RuntimeError("QSA private delta rows disagree on mask mode")
        else:
            causal_mask = mx.concatenate(
                [
                    _pad_sequence(
                        mask, 0, total - int(mask.shape[-1]), axis=mask.ndim - 1
                    )
                    for mask in masks
                ],
                axis=0,
            )
        compact = QSACompactBlocks(
            block_ids=mx.concatenate([item.block_ids for item in row_compacts], axis=0),
            block_counts=mx.concatenate(
                [item.block_counts for item in row_compacts], axis=0
            ),
            tail_start=mx.concatenate(
                [item.tail_start for item in row_compacts], axis=0
            ),
            tail_stop=mx.concatenate([item.tail_stop for item in row_compacts], axis=0),
            left_padding=None,
            block_size=row_compacts[0].block_size,
            physical_width=total,
            causal_mask=causal_mask,
        )
        exact_set_engaged = False
        try:
            if exact_set_fold:
                try:
                    from .segmented_self_mtp import note_qsa_exact_set_fold_event

                    note_qsa_exact_set_fold_event("proof")
                    output = qwen4_qsa_indexed_private_delta_exact_set_attention(
                        q,
                        base_k,
                        base_v,
                        delta_k,
                        delta_v,
                        mx.array(delta_lengths, dtype=mx.uint32),
                        compact,
                        scale=attention.scale,
                    )
                except (QSAIndexedProbeDeclined, RuntimeError) as error:
                    from .segmented_self_mtp import note_qsa_exact_set_fold_event

                    note_qsa_exact_set_fold_event(
                        "private_fallback",
                        reason=error.reason
                        if isinstance(error, QSAIndexedProbeDeclined)
                        else "dispatch_raised",
                    )
                    output = qwen4_qsa_indexed_private_delta_attention(
                        q,
                        base_k,
                        base_v,
                        delta_k,
                        delta_v,
                        mx.array(delta_lengths, dtype=mx.uint32),
                        compact,
                        scale=attention.scale,
                    )
                else:
                    exact_set_engaged = True
            else:
                output = qwen4_qsa_indexed_private_delta_attention(
                    q,
                    base_k,
                    base_v,
                    delta_k,
                    delta_v,
                    mx.array(delta_lengths, dtype=mx.uint32),
                    compact,
                    scale=attention.scale,
                )
        except (QSAIndexedProbeDeclined, RuntimeError) as error:
            dense_rows = []
            for index in range(batch):
                (keys, values) = (row_keys[index], row_values[index])
                dense_cache = self.rows[index]
                if keys is None:
                    (dense_cache, _receipt) = dense_cache.materialize_to_qsa()
                    (keys, values) = dense_cache.keys_and_values()
                dense_rows.append(
                    qsa_dense_attention_from_selection(
                        q[index : index + 1],
                        keys,
                        values,
                        row_selections[index],
                        dense_cache,
                        scale=attention.scale,
                    )
                )
            output = mx.concatenate(dense_rows, axis=0)
            from .segmented_self_mtp import note_qsa_private_delta_event

            note_qsa_private_delta_event(
                "declined",
                width=length,
                reason=error.reason
                if isinstance(error, QSAIndexedProbeDeclined)
                else "dispatch_raised",
            )
            self._bump("private_delta_late_gather_fallbacks")
        else:
            from .segmented_self_mtp import (
                note_qsa_exact_set_fold_event,
                note_qsa_private_delta_event,
            )

            note_qsa_private_delta_event(
                "engaged",
                width=length,
                base_tokens=base_tokens,
                rows=len(self.rows),
                duplicate_base_storage_bytes_not_formed=(len(self.rows) - 1)
                * int(base_k.nbytes + base_v.nbytes),
            )
            if exact_set_engaged:
                note_qsa_exact_set_fold_event("engaged", rows=len(self.rows))
        self._bump("segmented_attention_calls")
        self._bump("independent_lineages_consumed", len(self.rows))
        self._refresh_geometry()
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return attention.o_proj(output * mx.sigmoid(gate))

    def update_index_keys(self, keys: mx.array):
        del keys
        raise RuntimeError(
            "segmented QSA forbids dense index-ledger materialization; Attention.segmented_attention must consume request-private rows"
        )

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        del keys, values
        raise RuntimeError(
            "segmented QSA forbids dense K/V materialization; Attention.segmented_attention must consume request-private rows"
        )

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        from .models.base import create_causal_mask

        del return_array
        return create_causal_mask(
            N, offset=self._base_width, left_padding=self.left_padding, **kwargs
        )

    def last_valid_query(self, values: mx.array) -> mx.array:
        if self._right_padding is None:
            return values[:, -1]
        rows = mx.arange(values.shape[0], dtype=mx.int32)
        positions = values.shape[1] - self._right_padding.astype(mx.int32) - 1
        return values[rows, positions]

    def max_left_padding(self) -> int:
        return max(
            (self._base_width - value for value in self._base_lengths), default=0
        )

    def release_qsa_cycle(self, _who: str, *, keep_shared=False, **_kwargs):
        shared = self._mtp_shared_topk if keep_shared else None
        shared_n_blocks = self._mtp_shared_topk_n_blocks if keep_shared else None
        share = self._mtp_share_topk if keep_shared else False
        if not keep_shared:
            for row in self.rows:
                row.release_qsa_cycle(_who)
        self._mtp_share_topk = share
        self._mtp_shared_topk = shared
        self._mtp_shared_topk_n_blocks = shared_n_blocks
        self._qsa_pooled_keys = None
        self._qsa_pooled_ratio = None

    def finalize(self):
        self._step_lengths = None
        self._right_padding = None
        self.release_qsa_cycle("SegmentedBatchQSAKVCache.finalize")
        self._refresh_geometry()

    def finalize_self_mtp_step(self):
        share = self._mtp_share_topk
        shared = self._mtp_shared_topk
        shared_n_blocks = self._mtp_shared_topk_n_blocks
        self._step_lengths = None
        self._right_padding = None
        self._refresh_geometry()
        self._mtp_share_topk = share
        self._mtp_shared_topk = shared
        self._mtp_shared_topk_n_blocks = shared_n_blocks

    def supports_ragged_trim(self):
        return True

    def preflight_ragged_trim(self, counts, *, validate=True):
        counts = [int(value) for value in counts]
        if len(counts) != len(self.rows):
            raise ValueError("segmented QSA trim must cover every row")
        for row, count in zip(self.rows, counts):
            if validate and count > _host_offset(row):
                raise ValueError("segmented QSA trim exceeds a row offset")
        return counts

    def trim_ragged(self, counts, *, validate=True):
        counts = self.preflight_ragged_trim(counts, validate=validate)
        for row, count in zip(self.rows, counts):
            if count:
                row.trim(count)
        self.release_qsa_cycle("SegmentedBatchQSAKVCache.trim_ragged")
        self._refresh_geometry()
        return counts

    def trim(self, count):
        count = int(count)
        self.trim_ragged([count] * len(self.rows))
        return count

    def is_trimmable(self):
        return all((row.is_trimmable() for row in self.rows))

    def empty(self):
        return all((row.empty() for row in self.rows))

    @property
    def nbytes(self):
        return 0


class SegmentedBatchArraysCache(Qwen4ArraysCache):
    """Coherent recurrent-state compute view whose writes split into B1 rows."""

    def __init__(self, rows: Sequence[ArraysCache], note=None):
        if not rows or not all((isinstance(row, ArraysCache) for row in rows)):
            names = ", ".join((type(row).__name__ for row in rows))
            raise SegmentedBatchUnsupported(
                f"true batched recurrent state requires ArraysCache rows; got {names}"
            )
        sizes = {len(row.cache) for row in rows}
        if len(sizes) != 1:
            raise SegmentedBatchUnsupported("segmented recurrent slot counts differ")
        super().__init__(sizes.pop())
        self.rows = list(rows)
        self._note = note
        self.speculating = True
        self._refresh_state()
        self._checkpoints = [
            list(row._checkpoints[0]) if len(row._checkpoints) == 1 else []
            for row in self.rows
        ]

    def _bump(self, key, amount=1):
        if self._note is not None:
            self._note(key, amount)

    def _refresh_state(self):
        joined = []
        for slot in range(len(self.cache)):
            values = [row[slot] for row in self.rows]
            present = [value for value in values if value is not None]
            if not present:
                joined.append(None)
                continue
            if len(present) != len(values):
                raise SegmentedBatchUnsupported(
                    f"recurrent slot {slot} is initialized for only some rows"
                )
            shape = present[0].shape[1:]
            if any((value.shape[1:] != shape for value in present)):
                raise SegmentedBatchUnsupported(
                    f"recurrent slot {slot} row geometries differ"
                )
            joined_value = mx.concatenate(values, axis=0)
            joined.append(joined_value)
            self._bump("recurrent_state_materializations")
            self._bump("recurrent_state_materialized_bytes", joined_value.nbytes)
        self.cache = joined

    def __setitem__(self, idx, value):
        self.cache[idx] = value
        if value is None:
            for row in self.rows:
                row[idx] = None
        else:
            if value.shape[0] != len(self.rows):
                raise ValueError("segmented recurrent write has wrong batch size")
            for index, row in enumerate(self.rows):
                row[idx] = mx.contiguous(value[index : index + 1])
        self._bump("row_state_splits", len(self.rows))

    @property
    def batch_size(self):
        return len(self.rows)

    def prepare(self, lengths=None, **kwargs):
        del kwargs
        if lengths is None or len(lengths) != len(self.rows):
            raise ValueError("segmented recurrent prepare must cover every row")
        super().prepare(lengths=lengths)
        for row, length in zip(self.rows, lengths):
            row.prepare(lengths=[int(length)])

    def finalize(self):
        first_error = None
        for row in self.rows:
            try:
                row.finalize()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        super().finalize()
        if first_error is not None:
            raise first_error

    def advance(self, N):
        return ArraysCache.advance(self, N)

    def _row_closure(self, fn, row):
        if fn is None:
            return None

        def sliced(value, _fn=fn, _row=row):
            return _slice_row_tree(_fn(value), _row)

        return sliced

    def stage_ple_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None):
        del per_row_fn
        for index, row in enumerate(self.rows):
            stage = getattr(row, "stage_ple_rollback", None)
            if stage is not None:
                stage(
                    num_tokens,
                    self._row_closure(fn, index),
                    _slice_row_tree(snapshot, index),
                )

    def record_rollback(self, num_tokens, fn, snapshot, *, per_row_fn=None, **_kwargs):
        del per_row_fn
        for index, row in enumerate(self.rows):
            row.record_rollback(
                num_tokens,
                self._row_closure(fn, index),
                _slice_row_tree(snapshot, index),
            )

    def supports_ragged_trim(self):
        return True

    def preflight_ragged_trim(self, counts, *, validate=True):
        counts = [int(value) for value in counts]
        if len(counts) != len(self.rows):
            raise ValueError("segmented recurrent trim must cover every row")
        for row, count in zip(self.rows, counts):
            row.preflight_ragged_trim([count], validate=validate)
        return counts

    def trim_ragged(self, counts, *, validate=True):
        counts = self.preflight_ragged_trim(counts, validate=validate)
        for row, count in zip(self.rows, counts):
            row.trim_ragged([count], validate=False)
        self._refresh_state()
        return counts

    def trim(self, count):
        count = int(count)
        self.trim_ragged([count] * len(self.rows))
        return count

    def is_trimmable(self):
        return all((row.is_trimmable() for row in self.rows))

    def empty(self):
        return all((row.empty() for row in self.rows))

    @property
    def nbytes(self):
        return sum(
            (
                int(getattr(value, "nbytes", 0))
                for value in self.cache
                if value is not None
            )
        )


def build_segmented_batch_cache_group(groups, *, note=None, shared_qsa_prefix=False):
    """Transpose B1 cache groups into per-layer batched compute adapters."""
    groups = [list(group) for group in groups]
    if not groups:
        return []
    widths = {len(group) for group in groups}
    if len(widths) != 1:
        raise SegmentedBatchUnsupported("segmented cache groups have different layers")
    result = []
    for layer_index, layer_rows in enumerate(zip(*groups)):
        first = layer_rows[0]
        if isinstance(first, ArraysCache):
            result.append(SegmentedBatchArraysCache(layer_rows, note=note))
        elif isinstance(first, QSAKVCache) or getattr(
            first, "supports_shared_qsa_suffix", False
        ):
            result.append(
                SegmentedBatchQSAKVCache(
                    layer_rows, note=note, shared_qsa_prefix=shared_qsa_prefix
                )
            )
        elif type(first) is QuantizedKVCache:
            # Approximate KV composed with self-MTP: target rows quantized by
            # the adapter-declared operation.  Exact rows never reach here.
            from .segmented_plain_kv import SegmentedBatchQuantizedKVCache

            result.append(SegmentedBatchQuantizedKVCache(layer_rows, note=note))
        elif isinstance(first, KVCache):
            from .segmented_plain_kv import SegmentedBatchKVCache
            result.append(SegmentedBatchKVCache(layer_rows, note=note))
        else:
            raise SegmentedBatchUnsupported(
                f"unsupported segmented cache layer {type(first).__name__}"
            )
    return result


def build_segmented_batch_cache_pair(row_pairs, *, note=None, shared_qsa_prefix=False):
    from .hybrid_speculative import SelfMTPCachePair

    return SelfMTPCachePair(
        target=build_segmented_batch_cache_group(
            [pair.target for pair in row_pairs],
            note=note,
            shared_qsa_prefix=shared_qsa_prefix,
        ),
        draft=build_segmented_batch_cache_group(
            [pair.draft for pair in row_pairs],
            note=note,
            shared_qsa_prefix=shared_qsa_prefix,
        ),
    )
