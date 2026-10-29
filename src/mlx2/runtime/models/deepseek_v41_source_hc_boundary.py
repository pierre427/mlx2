"""Unwired Source HC boundary composite; caller owns the bounded GPU window.

The exact-first candidate retains stock matrix, four-term sum and RMSNorm
frontiers. Only pointwise producers/consumers are composed. Inline collapse
variants are unproved alternatives, never automatic fallbacks or defaults.
"""
from functools import lru_cache
import math

COLLAPSE_POLICIES = ('stock_frontier', 'inline_sequential', 'inline_pairwise')
MAX_ROWS = 128
DIM = 5120
HC = 4


def can_expand_collapse(branch,residual,post,comb,pre):
    """Narrow production StageA geometry; unsupported cases stay ordinary."""
    import mlx.core as mx
    values=(branch,residual,post,comb,pre)
    if not all(isinstance(v,mx.array)and v.dtype==mx.float32 for v in values):return False
    if branch.ndim!=3 or branch.shape[0]!=1 or branch.shape[1]not in(1,99)or branch.shape[2]!=DIM:return False
    rows=branch.shape[1]
    return (residual.shape==(1,rows,HC,DIM)and post.shape==pre.shape==(1,rows,HC)
            and comb.shape==(1,rows,HC,HC))


def expand_collapse(branch,residual,post,comb,pre,*,return_frontiers=False):
    """StageA only; leave coefficient prediction, RMSNorm and QAT to caller."""
    if type(return_frontiers)is not bool or not can_expand_collapse(branch,residual,post,comb,pre):
        raise ValueError('Source HC expand/collapse requires original B1 M1/M99 F32 geometry')
    mx=_device();rows=branch.shape[1]
    mixed=mx.matmul(mx.swapaxes(comb,-1,-2),residual.astype(mx.float32))
    code=compilation_source('residual',collapse_policy='inline_sequential',return_frontiers=return_frontiers)
    shapes={'updated_h':residual.shape,'products':residual.shape,'expanded_raw':residual.shape,'collapsed':branch.shape}
    values=_kernel('residual','inline_sequential',return_frontiers)(inputs=[branch,mixed,post,pre],
        template=[('ELEMENTS',rows*DIM),('DIM',DIM)],grid=(rows*DIM,1,1),threadgroup=(128,1,1),
        output_shapes=[shapes[n]for n in code['outputs']],output_dtypes=[mx.float32]*len(code['outputs']))
    generated=dict(zip(code['outputs'],values))
    collapsed=generated['collapsed'].astype(mx.bfloat16).astype(mx.float32)
    outputs=(generated['updated_h'],collapsed)
    if not return_frontiers:return outputs
    return outputs,dict(mixed_residual=mixed,expanded_raw=generated['expanded_raw'],
        collapse_products=generated['products'],collapsed_raw=generated['collapsed'])


def plan(branch, residual, post, comb, pre, norm_weight, *, eps=1e-6,
         collapse_policy='stock_frontier', return_frontiers=False):
    """CPU metadata only; no device/weight access or numerical claim."""
    import mlx.core as mx
    if collapse_policy not in COLLAPSE_POLICIES or type(return_frontiers) is not bool:
        raise ValueError('Source HC boundary unknown collapse/diagnostic option')
    if type(eps) not in (int,float) or not math.isfinite(eps) or eps <= 0:
        raise ValueError('Source HC boundary epsilon must be finite positive')
    arrays=(branch,residual,post,comb,pre,norm_weight)
    if not all(isinstance(v,mx.array) for v in arrays):
        raise ValueError('Source HC boundary requires MLX arrays')
    if branch.ndim != 3 or branch.shape[0] != 1 or branch.shape[2] != DIM:
        raise ValueError('Source HC boundary requires B1 rows by5120')
    rows=branch.shape[1]
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError('Source HC boundary rows must be1..128')
    if (residual.shape != (1,rows,HC,DIM) or post.shape != (1,rows,HC)
            or pre.shape != post.shape or comb.shape != (1,rows,HC,HC)
            or norm_weight.shape != (DIM,)):
        raise ValueError('Source HC boundary coefficient/residual shape')
    if any(v.dtype != mx.float32 for v in arrays[:-1]) or norm_weight.dtype not in (mx.float32,mx.bfloat16):
        raise ValueError('Source HC boundary requires original F32 planes and BF16/F32 norm weight')
    return dict(rows=rows,dim=DIM,hc=HC,eps=float(eps),collapse_policy=collapse_policy,
                stock_sum_frontier=collapse_policy=='stock_frontier',return_frontiers=return_frontiers,
                mixed_residual_bytes=rows*HC*DIM*4,updated_residual_bytes=rows*HC*DIM*4,
                collapse_products_bytes=rows*HC*DIM*4 if collapse_policy=='stock_frontier' or return_frontiers else 0,
                normalized_router_bytes=rows*DIM*4,prepared_fp8_represented_bytes=rows*DIM*4,
                scope='Array payload geometry only, not allocator/DRAM/performance evidence')


def compilation_source(stage, *, collapse_policy='stock_frontier', return_frontiers=False):
    """Return Metal source and binding names for CPU/offline inspection."""
    if collapse_policy not in COLLAPSE_POLICIES or type(return_frontiers) is not bool:
        raise ValueError('Source HC boundary unknown source option')
    from .deepseek_v41_fused import HEADER as ROUND_HEADER
    if stage == 'residual':
        outputs=['updated_h']
        if collapse_policy=='stock_frontier' or return_frontiers: outputs.append('products')
        if collapse_policy!='stock_frontier': outputs.append('collapsed')
        if return_frontiers: outputs.append('expanded_raw')
        source='''
const uint i=thread_position_in_grid.x;
if(i>=ELEMENTS) return;
const uint row=i/DIM, d=i%DIM;
const float y=branch[i];
float weighted[4];
for(uint h=0;h<4;h++) {
    const uint offset=(row*4+h)*DIM+d;
    const float contribution=post[row*4+h]*y;
    const float raw=contribution+mixed[offset];
    const float stored=bf16_value(raw);
    updated_h[offset]=stored;
    weighted[h]=stored*pre[row*4+h];
    PRODUCT_STORE
    RAW_STORE
}
COLLAPSE_STORE
'''
        source=source.replace('PRODUCT_STORE','products[offset]=weighted[h];' if 'products' in outputs else '')
        source=source.replace('RAW_STORE','expanded_raw[offset]=raw;' if return_frontiers else '')
        if collapse_policy=='inline_sequential':
            collapse='float sum=0.0f; for(uint h=0;h<4;h++) sum=sum+weighted[h]; collapsed[i]=sum;'
        elif collapse_policy=='inline_pairwise':
            collapse='const float a=weighted[0]+weighted[1]; const float b=weighted[2]+weighted[3]; collapsed[i]=a+b;'
        else: collapse=''
        return dict(header=ROUND_HEADER,source=source.replace('COLLAPSE_STORE',collapse),
                    inputs=['branch','mixed','post','pre'],outputs=outputs)
    if stage == 'norm_qat':
        from .deepseek_v41_quant import FUSED_HEADER,FUSED_SOURCE
        anchor='const float v=value[i];'
        if FUSED_SOURCE.count(anchor) != 1:
            raise ValueError('Source HC boundary QAT body changed')
        source=FUSED_SOURCE.replace(anchor,'const float v=bf16_value(value[i]);\nrouter[i]=v;')
        return dict(header=ROUND_HEADER+FUSED_HEADER,source=source,
                    inputs=['value','inverse'],outputs=['router','out'])
    raise ValueError('Source HC boundary unknown stage')


@lru_cache(maxsize=12)
def _kernel(stage,collapse_policy,return_frontiers):
    import mlx.core as mx
    code=compilation_source(stage,collapse_policy=collapse_policy,return_frontiers=return_frontiers)
    return mx.fast.metal_kernel(name=f'mlx2_source_hc_boundary_{stage}_{collapse_policy}_{int(return_frontiers)}_v1',
        input_names=code['inputs'],output_names=code['outputs'],header=code['header'],source=code['source'],
        ensure_row_contiguous=True)


def _device():
    from .deepseek_v41_source_hc import _device as owned_device,_verify_installed
    mx=owned_device();_verify_installed();return mx


def reference(branch,residual,post,comb,pre,norm_weight,*,eps=1e-6,return_frontiers=False):
    """Ordinary Source tree, original coefficients; CPU needs a CPU QAT oracle."""
    plan(branch,residual,post,comb,pre,norm_weight,eps=eps,return_frontiers=return_frontiers)
    import mlx.core as mx
    from .deepseek_v41_quant import fused_quantize
    mixed=mx.matmul(mx.swapaxes(comb,-1,-2),residual.astype(mx.float32))
    raw=post[...,None]*branch[...,None,:].astype(mx.float32)
    raw+=mixed
    updated=raw.astype(mx.bfloat16).astype(mx.float32)
    products=updated.astype(mx.float32)*pre[...,None]
    collapsed_raw=mx.sum(products,-2)
    collapsed=collapsed_raw.astype(mx.bfloat16).astype(mx.float32)
    norm_raw=mx.fast.rms_norm(collapsed.astype(mx.float32),norm_weight.astype(mx.float32),eps)
    router=norm_raw.astype(mx.bfloat16).astype(mx.float32)
    prepared=fused_quantize(router.astype(mx.float32),'fp8').astype(mx.float32)
    outputs=(updated,router,prepared)
    frontiers=dict(mixed_residual=mixed,expanded_raw=raw,collapse_products=products,
                   collapsed_raw=collapsed_raw,collapsed_bf16=collapsed,norm_raw=norm_raw)
    return (outputs,frontiers) if return_frontiers else outputs


def boundary(branch,residual,post,comb,pre,norm_weight,*,eps=1e-6,
             collapse_policy='stock_frontier',return_frontiers=False):
    """Partial composite, unselected; all returned arrays retain their inputs.

    Default keeps stock F32 reductions. Optional inline reductions must pass
    their own raw/BF16 GPU oracles and never replace default after a failure.
    Coefficient prediction/Sinkhorn continue consuming returned updated_h.
    """
    p=plan(branch,residual,post,comb,pre,norm_weight,eps=eps,
           collapse_policy=collapse_policy,return_frontiers=return_frontiers)
    mx=_device()
    # Identical stock operand shapes, transpose and precision to hc_expand.
    mixed=mx.matmul(mx.swapaxes(comb,-1,-2),residual.astype(mx.float32))
    code=compilation_source('residual',collapse_policy=collapse_policy,return_frontiers=return_frontiers)
    shapes={'updated_h':residual.shape,'products':residual.shape,
            'expanded_raw':residual.shape,'collapsed':branch.shape}
    values=_kernel('residual',collapse_policy,return_frontiers)(inputs=[branch,mixed,post,pre],
        template=[('ELEMENTS',p['rows']*DIM),('DIM',DIM)],grid=(p['rows']*DIM,1,1),threadgroup=(128,1,1),
        output_shapes=[shapes[n]for n in code['outputs']],output_dtypes=[mx.float32]*len(code['outputs']))
    generated=dict(zip(code['outputs'],values))
    collapsed_raw=(mx.sum(generated['products'],-2)if collapse_policy=='stock_frontier'else generated['collapsed'])
    collapsed=collapsed_raw.astype(mx.bfloat16).astype(mx.float32)
    norm_raw=mx.fast.rms_norm(collapsed.astype(mx.float32),norm_weight.astype(mx.float32),eps)
    router,prepared=_kernel('norm_qat',collapse_policy,return_frontiers)(
        inputs=[norm_raw,mx.array([1/448],mx.float32)],
        template=[('ELEMENTS',p['rows']*DIM),('BLOCK',32),('KIND',0)],grid=(p['rows']*DIM,1,1),threadgroup=(128,1,1),
        output_shapes=[branch.shape,branch.shape],output_dtypes=[mx.float32,mx.float32])
    outputs=(generated['updated_h'],router,prepared)
    if not return_frontiers:return outputs
    return outputs,dict(mixed_residual=mixed,expanded_raw=generated['expanded_raw'],
        collapse_products=generated['products'],collapsed_raw=collapsed_raw,collapsed_bf16=collapsed,norm_raw=norm_raw)
