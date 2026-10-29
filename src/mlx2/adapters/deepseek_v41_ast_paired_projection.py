# SPDX-License-Identifier: MIT
"""Unselected research wrapper joining original MXFP8 dense output rows.

Preparation owns duplicate coefficients; use keeps the existing M1 batch-axis
QMM/K reduction and rounded stores. Changed output width still requires actual
GPU raw parity. No policy, model identity or production selection is changed.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from threading import Lock

_INSTALL_LOCK = Lock()
PROFILE = 'source_ast_joined_original_mxfp8_output_rows_candidate_v1'


def _stamp(array):
    return id(array), tuple(array.shape), str(array.dtype), array.ndim, array.nbytes


def _geometry(model):
    c=model.args
    values=tuple(getattr(c,name,None)for name in('num_hidden_layers','hidden_size','q_lora_rank','head_dim','moe_intermediate_size'))
    if (any(type(value)is not int or value<=0 for value in values)
            or values[0]!=40 or values[1]%32):
        raise ValueError('Paired projection requires40-layer original group32 geometry')
    return values


def _idle(model):
    if(getattr(model,'_weight_executions',0)!=0
            or getattr(model,'_mlx2_ds41_ast_active',None)is not None
            or getattr(model,'_mlx2_ds41_ast_poisoned',False)):
        raise ValueError('Paired projection installation requires an idle unpoisoned target')


@dataclass(frozen=True)
class Pair:
    names: tuple
    owners: tuple
    stamps: tuple
    q: object
    scales: object
    widths: tuple


class PreparedPairs:
    def __init__(self,model,pairs,geometry):
        self.model=model;self.weights=model.weights;self.identity=model.identity
        self.quant=self.weights._quant;self.geometry=geometry
        self.pairs={pair.names:pair for pair in pairs}
        self.policy=self.weights.source_dense_qat_policy
        self.logical_preparation_bytes=sum(pair.q.nbytes+pair.scales.nbytes for pair in pairs)

    def _bound(self,model):
        if(model is not self.model or model.weights is not self.weights
                or model.identity!=self.identity or model.weights._quant is not self.quant
                or _geometry(model)!=self.geometry
                or model.weights.source_dense_qat_policy!=self.policy):
            raise ValueError('Prepared paired projection model/weight binding changed')
        model.weights._ready()

    def _pair(self,model,names):
        self._bound(model)
        pair=self.pairs[names]
        for name,owner,stamp in zip(pair.names,pair.owners,pair.stamps):
            current=self.quant.get(name)
            if(current is not owner or tuple(_stamp(value)for value in current)!=stamp):
                raise ValueError('Prepared paired projection coefficient owner/layout changed: '+name)
        return pair

    def evaluation_roots(self):
        return tuple(value for pair in self.pairs.values()
                     for value in(*pair.owners[0],*pair.owners[1],pair.q,pair.scales))

    def receipt(self):
        owner_digest=hashlib.sha256(repr(tuple(pair.stamps for pair in self.pairs.values())).encode()).hexdigest()
        return dict(profile=PROFILE,pairs=len(self.pairs),original_coefficient_pairs=len(self.pairs)*2,
            logical_preparation_bytes=self.logical_preparation_bytes,
            duplicate_coefficient_bytes=self.logical_preparation_bytes,
            original_codes_and_scales_unchanged=True,group_size=32,format='mxfp8',
            model_identity=self.identity,owner_binding_digest=owner_digest,
            owner_binding='Exact model/weights/dictionary/array identities and metadata rechecked per eligible use; owners retained',
            entries=[dict(names=list(pair.names),output_widths=list(pair.widths),
                          joined_shape=list(pair.q.shape),scale_shape=list(pair.scales.shape),
                          bytes=pair.q.nbytes+pair.scales.nbytes)for pair in self.pairs.values()],
            preparation_inside_timer=False,qualified=False,selected=False,
            dispatch_limit='Joined output width can change stock kernel dispatch; actual raw GPU gate mandatory',
            physical_peak_memory_claim=False)


def prepare(model):
    """Build lazy row joins outside timing; caller owns evaluation/completion."""
    import mlx.core as mx
    _idle(model);geometry=_geometry(model);weights=model.weights
    weights._ready();weights._ensure_dense()
    if getattr(weights,'source_dense_qat_policy','reference')!='fused_fp32':
        raise ValueError('Paired projection requires existing fused_fp32 dense input policy')
    quant=getattr(weights,'_quant',None)
    if not isinstance(quant,dict):raise ValueError('Paired projection requires original quantized coefficients')
    _,hidden,qrank,head,intermediate=geometry
    pairs=[]
    for layer in range(40):
        prefix=f'blk.{layer}.'
        for names,widths in(((prefix+'attn_q_a.weight',prefix+'attn_kv.weight'),(qrank,head)),
                             ((prefix+'ffn_gate_shexp.weight',prefix+'ffn_up_shexp.weight'),(intermediate,intermediate))):
            owners=[]
            for name,width in zip(names,widths):
                owner=quant.get(name)
                if(not isinstance(owner,tuple)or len(owner)!=2):
                    raise ValueError('Paired projection missing original coefficient pair: '+name)
                q,scales=owner
                if(q.ndim!=2 or q.dtype!=mx.uint32 or q.shape!=(width,hidden//4)
                        or scales.ndim!=2 or scales.dtype!=mx.uint8 or scales.shape!=(width,hidden//32)):
                    raise ValueError('Paired projection original MXFP8 group32 geometry mismatch: '+name)
                owners.append(owner)
            stamps=tuple(tuple(_stamp(value)for value in owner)for owner in owners)
            joined=mx.concatenate((owners[0][0],owners[1][0]),axis=0)
            scales=mx.concatenate((owners[0][1],owners[1][1]),axis=0)
            pairs.append(Pair(names,tuple(owners),stamps,joined,scales,widths))
    return PreparedPairs(model,pairs,geometry)


@contextmanager
def installed(prepared):
    """Patch one research execution window; no evaluation/fence in any use."""
    from . import deepseek_v41_ast_batched_dense_exact as exact
    if not isinstance(prepared,PreparedPairs):raise ValueError('Paired projection needs prepared owners')
    _idle(prepared.model);prepared._bound(prepared.model)
    if not _INSTALL_LOCK.acquire(blocking=False):
        raise ValueError('Paired projection research entry is already installed')
    original=exact.paired_linear_tile
    metrics=dict(joined_calls=0,baseline_calls=0,joined_output_rows=0)
    def paired(model,x,a,b,rounded=True):
        if model is not prepared.model:
            raise ValueError('Paired projection research window received a foreign model')
        names=(a,b)
        if names not in prepared.pairs:
            metrics['baseline_calls']+=1
            return original(model,x,a,b,rounded=rounded)
        if type(rounded)is not bool:raise ValueError('Paired projection rounded flag must be boolean')
        rows=exact._rows(x,width=prepared.geometry[1])
        pair=prepared._pair(model,names)
        source=model.weights._dense_input(x)
        raw=exact._quant_batch_matmul(source,pair.q,pair.scales)
        first=raw[...,:pair.widths[0]];second=raw[...,pair.widths[0]:]
        if rounded:first,second=model.round(first),model.round(second)
        counts=model.weights.scheduled_dense_calls
        key='source_ast_m1_batch_axis_mxfp8';counts[key]=counts.get(key,0)+1
        exact._record(model,'paired',rows)
        metrics['joined_calls']+=1;metrics['joined_output_rows']+=sum(pair.widths)
        return first,second
    exact.paired_linear_tile=paired
    try:yield metrics
    finally:
        conflict=exact.paired_linear_tile is not paired
        exact.paired_linear_tile=original
        _INSTALL_LOCK.release()
        if conflict:raise RuntimeError('Paired projection research entry changed concurrently')


def implementation_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
