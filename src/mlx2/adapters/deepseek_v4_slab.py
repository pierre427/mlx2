"""Opt-in row-exact causal K+1 chain; no packed tree-AST claim.

The ordinary compressed-cache transaction supplies complete snapshots. A full
layer-major target forward evaluates every row with ordinary width-one math,
including its local/pool/indexer state and post-FFN HC features. Partial accept
restores all state planes and replays its accepted inputs at ordinary width one;
generic KV trimming cannot repair compressor carry or completed pooled windows.

The previous vectorized slab was not row-invariant. It must not choose accepted
tokens or publish features from a different arithmetic path than replay.
Full route qualification is separate from this width-one execution contract.
"""
from __future__ import annotations

from ..runtime.drafters.deepseek_v4_dspark import DSparkContractError
from .deepseek_v4_external import DeepSeekV4VerifyRows, DeepSeekV4VerifyTransaction


class DeepSeekV4CausalSlabVerifyRows(DeepSeekV4VerifyRows):
    """Exclusive B1 causal-slab owner, default-off and unqualified."""

    def __init__(self, model, rows, layers, *, maximum_width=6):
        if type(maximum_width) is not int or not 1 <= maximum_width <= 6:
            raise DSparkContractError('causal slab width must be bounded1..6')
        super().__init__(model,rows,layers)
        self.maximum_width=maximum_width
        self.scheduled_slab_forwards=0
        self.scheduled_slab_rows=0
        self.scheduled_max_causal_target_rows=0

    @property
    def receipt_settings(self):
        return {'verification':'causal_kplus1_layer_major_width_one_v2',
                'maximum_width':self.maximum_width,'qualified':False,
                'logical_batch_width':1,'physical_batch_width':1,
                'target_width_semantics':'logical batch width, not causal verification rows',
                'maximum_causal_target_rows':self.maximum_width,
                'scheduled_max_causal_target_rows':self.scheduled_max_causal_target_rows,
                'packed_tree_ast':False,'target_row_invariance_established':False,
                'arithmetic_rows':1,'execution_order':'layer_major',
                'scheduled_slab_forwards':self.scheduled_slab_forwards,
                'scheduled_slab_rows':self.scheduled_slab_rows,
                'counter_scope':'scheduled Python forwards; not GPU completion or qualification'}

    def begin(self,lengths,*,plain_single_token=False):
        if self._poisoned:
            raise DSparkContractError('verification owner poisoned by failed restore')
        if self._active is not None and not self._active.closed:
            raise DSparkContractError('verification owner already active')
        if (len(lengths)!=1 or type(lengths[0]) is not int
                or not 1<=lengths[0]<=self.maximum_width):
            raise DSparkContractError('invalid bounded B1 causal slab length')
        self._active=DeepSeekV4CausalSlabVerifyTransaction(self,lengths[0],plain_single_token)
        return self._active


class DeepSeekV4CausalSlabVerifyTransaction(DeepSeekV4VerifyTransaction):
    """One layer-major chain; inherited partial-commit replay stays width one."""

    def forward(self,inputs,*,taps=True):
        if self.closed or self.owner._active is not self or self._inputs is not None:
            raise DSparkContractError('stale or repeated causal slab forward')
        if tuple(inputs.shape)!=(1,self.width):
            raise DSparkContractError('causal slab input geometry mismatch')
        import mlx.core as mx
        self._inputs=mx.asarray(inputs,copy=True)
        try:
            logits,features=self.owner.model.verify_rows(
                inputs,self.caches,self.owner.layers if taps else ())
            if (tuple(logits.shape[:2])!=(1,self.width)
                    or taps and (features is None or tuple(features.shape[:2])!=(1,self.width))):
                raise DSparkContractError('causal slab must preserve every verification row')
            self.owner.scheduled_slab_forwards+=1
            self.owner.scheduled_slab_rows+=self.width
            self.owner.scheduled_max_causal_target_rows=max(self.owner.scheduled_max_causal_target_rows,self.width)
            return logits,features
        except BaseException:
            self.abort()
            raise
