# SPDX-License-Identifier: MIT
"""Occurrence-parallel Source DOWN with each row's exact B1 dot law.

Rows are already in the caller's per-expert occurrence order. One grid.y
coordinate owns one entire2304-input dot; local M never changes its SIMD
reduction or selects a wide/split-K path. The output is BF16-rounded F32,
ready for the adapter's stable ordered expert sum. No compile on import.
Provenance: deepseek-v41-source-ast-down-rows.json.
"""
from functools import lru_cache
from types import SimpleNamespace

MAX_ROWS = 16
HIDDEN = 5120
INTERMEDIATE = 2304


def plan(rows, kind):
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError('Source AST DOWN requires1..16 occurrence rows')
    if kind not in ('original', 'q1'):
        raise ValueError('Source AST DOWN unknown expert format')
    return dict(rows=rows, kind=kind, grid=(1280*32,rows,1),
                threadgroup=(32,1,1), output_shape=(rows,HIDDEN))


def kernel_spec(kind):
    """Source generation only: reuse pinned B1 helpers without new dot math."""
    from .source_down_fusion import kernel_spec as single_spec
    plan(1,kind)
    spec=single_spec((kind,),tile_rows=4)
    source=spec['source']
    call='mlx2_source_q1_down_shared(x0,' if kind=='q1' else 'nullptr,x0,'
    replacement='mlx2_source_q1_down_shared(x0+occurrence*2304,' if kind=='q1' else 'nullptr,x0+occurrence*2304,'
    anchor='if(sg==0 && lane<4) {'
    if source.count(call)!=1 or source.count(anchor)!=1:
        raise RuntimeError('Source AST DOWN pinned source anchors changed')
    # Keep the helper call, scratch placement and barrier exactly. Replace
    # only the one-expert zero+sum sink with its rounded contribution.
    source='const uint occurrence=threadgroup_position_in_grid.y;\n'+source.replace(call,replacement)
    prefix=source[:source.index(anchor)]
    source=prefix+'''if(sg==0 && lane<4) {
    output[occurrence*5120+tile*4+lane]=float(bfloat(partials[lane]));
}
'''
    return dict(header=spec['header'],source=source,
                input_names=spec['input_names'],output_names=['output'])


@lru_cache(maxsize=2)
def _kernel(kind):
    import mlx.core as mx
    return mx.fast.metal_kernel(name=f'mlx2_source_ast_m1_down_{kind}_v1',
                               ensure_row_contiguous=True,**kernel_spec(kind))


def prepare_inputs(x,expert):
    """Validate all packed fields with no slicing, evaluation or kernel work."""
    import mlx.core as mx
    from .source_down_fusion import prepare_inputs as single_inputs
    if (getattr(x,'ndim',len(x.shape))!=2 or x.shape[1]!=INTERMEDIATE
            or x.dtype!=mx.float32):
        raise ValueError('Source AST DOWN requires F32[M1..16,2304]')
    kind='q1' if 'down.biases' in expert else 'original'
    descriptor=plan(x.shape[0],kind)
    # The reused validator is shape-only; a structural one-row descriptor
    # avoids constructing a GPU slice merely to validate a packed tensor.
    first=SimpleNamespace(shape=(1,INTERMEDIATE),dtype=mx.float32)
    _,args=single_inputs((first,),(expert,),(0,))
    args[0]=x
    return descriptor,args


def down(x,expert):
    """F32[M,2304] -> BF16-rounded F32[M,5120], exact B1 dot per row."""
    import mlx.core as mx
    descriptor,args=prepare_inputs(x,expert)
    if mx.default_device()!=mx.gpu or not mx.metal.is_available() or mx.device_info().get('device_name')!='Apple M5 Max':
        raise ValueError('Source AST DOWN requires an owned M5 Max GPU window')
    return _kernel(descriptor['kind'])(inputs=args,grid=descriptor['grid'],threadgroup=descriptor['threadgroup'],
        output_shapes=[descriptor['output_shape']],output_dtypes=[mx.float32])[0]
