#!/usr/bin/env python3
"""Caller-owned low-bit TensorOps availability/correctness probe, no model law."""
import hashlib
import time
import numpy as np

HEADER = '#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n'


def shader(kind):
    typ,storage,destination=("int8_t","int8_t","int") if kind=="w8a8" else ("metal_fp4_e2m1_format","uchar","float")
    # MPP's inline dispatch compares unqualified element types. Read-only
    # inputs are viewed through its nonconst tensor API; the op never writes
    # either input. Explicit int8_t is required: plain Metal char is distinct.
    return f'''
tensor<device {typ},dextents<int,2>,tensor_inline> a((device {storage}*)left,dextents<int,2>(256,16));
tensor<device {typ},dextents<int,2>,tensor_inline> b((device {storage}*)right,dextents<int,2>(256,32));
tensor<device {destination},dextents<int,2>,tensor_inline> c(out,dextents<int,2>(32,16));
constexpr auto desc=matmul2d_descriptor(16,32,256,false,true,false,matmul2d_descriptor::mode::multiply);
matmul2d<desc,execution_simdgroup> op;op.run(a,b,c);
'''


def run_cases(guard):
    import mlx.core as mx
    rows=[];rng=np.random.default_rng(20261024)
    for kind in ["w8a8","w4a4"]:
        guard();source=shader(kind)
        row=dict(kind=kind,shape=dict(M=16,N=32,K=256),source_sha256=hashlib.sha256((HEADER+source).encode()).hexdigest(),scales="unity; no model quantization",available=False)
        try:
            if kind=="w8a8":
                a=rng.integers(-127,128,(16,256),dtype=np.int8);b=rng.integers(-127,128,(32,256),dtype=np.int8)
                reference=a.astype(np.int32)@b.astype(np.int32).T;dtype=mx.int32
            else:
                levels=np.array([0,.5,1,1.5,2,3,4,6],np.float32)
                def pack(shape):
                    codes=rng.integers(0,16,shape,dtype=np.uint8)
                    decoded=levels[codes&7]*np.where(codes&8,-1,1)
                    packed=(codes[:,::2]|(codes[:,1::2]<<4)).astype(np.uint8)
                    return packed,decoded.astype(np.float32)
                a,da=pack((16,256));b,db=pack((32,256))
                reference=da@db.T;dtype=mx.float32
            x,y=mx.array(a),mx.array(b);mx.eval(x,y)
            kernel=mx.fast.metal_kernel(name="mlx2_research_"+kind+"_v1",input_names=["left","right"],output_names=["out"],header=HEADER,source=source,ensure_row_contiguous=True)
            def run():return kernel(inputs=[x,y],grid=(32,1,1),threadgroup=(32,1,1),output_shapes=[(16,32)],output_dtypes=[dtype])[0]
            result=run();mx.eval(result);actual=np.array(result)
            row.update(available=True,exact=bool(np.array_equal(actual,reference)),max_abs_error=float(np.abs(actual.astype(np.float64)-reference).max()),finite=bool(np.isfinite(actual).all()))
            samples=[]
            for _ in range(3):
                guard();t=time.perf_counter();result=run();mx.eval(result);samples.append((time.perf_counter()-t)*1000)
            row['dispatch_ms']=samples
        except Exception as error:
            row.update(error_type=type(error).__name__,error=str(error)[:3000])
        rows.append(row)
    return dict(scope="synthetic unscaled native TensorOps availability/exactness only; neither model quality nor hardware throughput qualification",seed=20261024,cases=rows)
