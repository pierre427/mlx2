"""Revision-bound, persistable Inkling four-shortconv cache identity."""

from collections import deque

import mlx.core as mx

from ...cache_operations import (
    ROLLBACK_PUBLICATION_CONDITIONS,
    CacheOperationsContract,
    CachePlaneDeclaration,
)
from ..cache import ArraysCache, _committed_descriptor_fields

OPERATIONS = CacheOperationsContract(
    family="recurrent",
    topology="checkpointed_hybrid",
    state_revision="mlx2.inkling-four-shortconv.v1",
    cache_layout="four-fp32-causal-histories",
    numerics="stored_dtype",
    planes=(
        CachePlaneDeclaration(
            "inkling_shortconv", ("cache", "left_padding", "lengths")
        ),
    ),
    exact_prefix=True,
    restores_at_checkpoints=True,
    rollback="recurrent_record_trim",
    offset_field=None,
    publication_conditions=ROLLBACK_PUBLICATION_CONDITIONS,
)


class InklingConvolutionCache(ArraysCache):
    def __init__(self, size=4, left_padding=None):
        if size != 4:
            raise ValueError("Inkling requires exactly four convolution histories")
        super().__init__(size, left_padding)
        self.apcv2_recurrent_kind = "inkling_shortconv"

    def cache_operations_contract(self):
        return OPERATIONS

    def committed_descriptor_clone(self):
        """Cache-owned committed capture, mined from ArraysCache's contract."""
        if (
            type(self) is not InklingConvolutionCache
            or len(self.cache) != 4
            or self.apcv2_recurrent_kind != "inkling_shortconv"
            or self.speculating
            or self._rollbacks
            or self._rollback_position
            or self._rollback_positions is not None
        ):
            return None
        allowed = {
            "cache",
            "left_padding",
            "lengths",
            "speculating",
            "_rollbacks",
            "_rollback_window",
            "_rollback_epoch",
            "_rollback_position",
            "_rollback_positions",
            "_rollback_invalid_reason",
            "_host_lengths",
            "_host_left_padding",
            "_checkpoints",
            "apcv2_recurrent_kind",
            "apcv2_layer_index",
            "_recurrent_codec_restored_leaves",
        }
        fields = _committed_descriptor_fields(self, allowed)
        if fields is None or any(
            value is not None and not isinstance(value, mx.array)
            for value in self.cache
        ):
            return None
        if (
            "_recurrent_codec_restored_leaves" in fields
            and type(fields["_recurrent_codec_restored_leaves"]) is not int
        ):
            return None
        clone = object.__new__(InklingConvolutionCache)
        clone.__dict__.update(fields)
        clone.cache = [None if x is None else mx.stop_gradient(x) for x in self.cache]
        clone._rollbacks = deque()
        clone._checkpoints = [
            [(position, list(snapshot)) for position, snapshot in lane]
            for lane in self._checkpoints
        ]
        for name in ("left_padding", "lengths"):
            value = fields[name]
            setattr(clone, name, None if value is None else mx.stop_gradient(value))
        clone._host_lengths = clone._host_left_padding = None
        return clone

    @property
    def meta_state(self):
        return (
            "inkling-shortconv-v1",
            str(getattr(self, "apcv2_layer_index", -1)),
            super().meta_state,
        )

    @meta_state.setter
    def meta_state(self, value):
        if len(value) != 3 or value[0] != "inkling-shortconv-v1":
            raise ValueError("Inkling convolution metadata revision differs")
        self.apcv2_recurrent_kind = "inkling_shortconv"
        self.apcv2_layer_index = int(value[1])
        ArraysCache.meta_state.fset(self, value[2])

    def exact_decode_checkpoint_contract(self):
        if (
            self.speculating
            or self._rollbacks
            or self._rollback_position
            or self._rollback_positions is not None
        ):
            return None
        return ("inkling-shortconv-v1",)

    def restore_exact_decode_checkpoint(self, contract):
        if contract != self.exact_decode_checkpoint_contract():
            raise ValueError("Inkling exact checkpoint contains incompatible state")
        self._host_lengths = self._host_left_padding = None
