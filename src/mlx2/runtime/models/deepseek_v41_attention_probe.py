# SPDX-License-Identifier: MIT
"""Unwired all-retained Source attention dataflow candidates, not a policy.

Only owning query tiling and F32 equations are reused. Broadcast/einsum and
contiguous-pool matmul lowerings still require their own actual GPU bit gates.
See provenance/deepseek-v41-attention-dataflow-probe.json. No work on import.
"""

VARIANTS=('reference','broadcast','matmul','flat_pool','matmul_tail_safe','flat_pool_tail_safe')
DEFAULT_WORKSPACE=16<<20


def tile_plan(heads,length,dim,local_count,selected_count,*,workspace_bytes=DEFAULT_WORKSPACE):
    if (any(type(v)is not int for v in (heads,length,dim,local_count,selected_count,workspace_bytes))
            or not 1<=heads<=64 or not 1<=length<=128 or not 1<=dim<=512
            or not 1<=local_count<=255 or not 1<=selected_count<=512 or workspace_bytes<1):
        raise ValueError('Source attention probe geometry outside bounded component')
    per_row=4*(heads*(local_count+selected_count)+selected_count*dim+2*heads*dim)
    tile=max(1,min(128,workspace_bytes//max(1,per_row)))
    return dict(heads=heads,length=length,dim=dim,local_count=local_count,
        selected_count=selected_count,tile=tile,per_row_workspace_formula_bytes=per_row,
        tiles=tuple((i,min(length,i+tile))for i in range(0,length,tile)),
        hypothetical_full_gather_bytes=length*selected_count*dim*4,
        numerical_profile='unwired-finite-all-retained-dataflow-candidate')


def tile_variant(variant,rows,tile):
    """Keep original partial-query tiles on the reference floating program."""
    if variant not in VARIANTS or type(rows)is not int or type(tile)is not int or not 1<=rows<=tile:
        raise ValueError('Source attention probe tile selection')
    if variant.endswith('_tail_safe'):
        return 'reference' if rows<tile else variant.removesuffix('_tail_safe')
    return variant


def geometry(q,local,local_mask,pooled,indices,valid,sinks,*,variant,
             all_retained=False,gathered=None,workspace_bytes=DEFAULT_WORKSPACE):
    import mlx.core as mx
    if variant not in VARIANTS:
        raise ValueError('Source attention probe unknown variant')
    if type(all_retained)is not bool or (variant!='reference'and not all_retained):
        raise ValueError('Source attention candidate requires trusted all-retained contract')
    if (q.ndim!=4 or q.shape[0]!=1 or q.dtype!=mx.float32
            or local.ndim!=4 or local.shape[:2]!=(1,1) or local.shape[-1]!=q.shape[-1]
            or local.dtype!=mx.float32 or pooled is None or pooled.ndim!=3
            or pooled.shape[0]!=1 or pooled.shape[-1]!=q.shape[-1]or pooled.dtype!=mx.float32):
        raise ValueError('Source attention probe requires F32 B1 query/local/pool')
    h,l,d=q.shape[1:];count=indices.shape[-1]if indices.ndim==3 else 0
    p=tile_plan(h,l,d,local.shape[2],count,workspace_bytes=workspace_bytes)
    if (indices.shape!=(1,l,count)or indices.dtype not in (mx.uint32,mx.int32)
            or valid.shape!=indices.shape or valid.dtype!=mx.bool_
            or local_mask.shape!=(1,1,l,local.shape[2])or local_mask.dtype!=mx.bool_
            or sinks.shape!=(h,)or sinks.dtype!=mx.float32 or count>pooled.shape[1]):
        raise ValueError('Source attention probe mask/index/sink geometry mismatch')
    if variant!='reference'and count!=pooled.shape[1]:
        raise ValueError('Source attention candidate must retain every pool row')
    if gathered is not None and (l!=1 or gathered.shape!=(1,1,count,d)or gathered.dtype!=pooled.dtype):
        raise ValueError('Source attention probe gathered input requires matching width1')
    return p


def attention(q,local,local_mask,pooled,indices,valid,sinks,*,variant='reference',
              all_retained=False,gathered=None,workspace_bytes=DEFAULT_WORKSPACE,
              return_stages=False):
    """Pure graph construction; caller owns arrays and verifies finite arange IDs.

    Normal reference uses the owning function. Diagnostic stages mirror that
    function's exact tree and are checked against it before candidate claims.
    No CPU array readbacks, persistent cache, state changes or GPU ownership.
    """
    import mlx.core as mx
    if type(return_stages)is not bool:
        raise ValueError('Source attention probe stage flag must be boolean')
    p=geometry(q,local,local_mask,pooled,indices,valid,sinks,variant=variant,
        all_retained=all_retained,gathered=gathered,workspace_bytes=workspace_bytes)
    from .deepseek_v41 import selected_attention,_selected_rows
    if variant=='reference'and not return_stages:
        return selected_attention(q,local,local_mask,pooled,indices,valid,sinks,
            gathered=gathered,workspace_bytes=workspace_bytes)
    out=[];stages={key:[]for key in ('scaled_query','local_scores_raw','pool_scores_raw',
        'local_scores','pool_scores','normalizer','local_pv','pool_pv')}
    for first,last in p['tiles']:
        method=tile_variant(variant,last-first,p['tile'])
        query=q[:,:,first:last].astype(mx.float32)*p['dim']**-.5
        if method in ('matmul','flat_pool'):rows=None
        elif gathered is not None:rows=gathered
        elif method=='broadcast':rows=mx.broadcast_to(pooled[:,None],(1,last-first,p['selected_count'],p['dim']))
        else:rows=_selected_rows(pooled,indices[:,first:last])
        ls_raw=query@mx.swapaxes(local.astype(mx.float32),-1,-2)
        ls=mx.where(local_mask[...,first:last,:],ls_raw,-mx.inf)
        if method=='flat_pool':
            flat_q=mx.contiguous(query).reshape(p['heads']*(last-first),p['dim'])
            ps_raw=(flat_q@pooled[0].astype(mx.float32).T).reshape(1,p['heads'],last-first,p['selected_count'])
        elif method=='matmul':ps_raw=query@mx.swapaxes(pooled[:,None].astype(mx.float32),-1,-2)
        else:ps_raw=mx.einsum('bhld,blkd->bhlk',query,rows)
        ps=mx.where(valid[:,None,first:last],ps_raw,-mx.inf)
        norm=mx.logaddexp(mx.logsumexp(ls,-1,keepdims=True),mx.logsumexp(ps,-1,keepdims=True))
        norm=mx.logaddexp(norm,sinks[None,:,None,None].astype(mx.float32))
        lpv=mx.exp(ls-norm)@local.astype(mx.float32)
        if method=='flat_pool':
            flat_prob=mx.contiguous(mx.exp(ps-norm)).reshape(p['heads']*(last-first),p['selected_count'])
            ppv=(flat_prob@pooled[0].astype(mx.float32)).reshape(1,p['heads'],last-first,p['dim'])
        elif method=='matmul':ppv=mx.exp(ps-norm)@pooled[:,None].astype(mx.float32)
        else:ppv=mx.einsum('bhlk,blkd->bhld',mx.exp(ps-norm),rows)
        result=lpv+ppv
        out.append(result.astype(q.dtype))
        if return_stages:
            for key,value in zip(stages,(query,ls_raw,ps_raw,ls,ps,norm,lpv,ppv)):stages[key].append(value)
    result=mx.concatenate(out,2)
    return (result,{key:mx.concatenate(values,2)for key,values in stages.items()})if return_stages else result
