"""Revision-bound, persistable Inkling four-shortconv cache identity."""

from ...cache_operations import (
    ROLLBACK_PUBLICATION_CONDITIONS,
    CacheOperationsContract,
    CachePlaneDeclaration,
)
from ..cache import ArraysCache

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
