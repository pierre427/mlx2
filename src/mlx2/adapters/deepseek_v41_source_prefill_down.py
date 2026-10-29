# SPDX-License-Identifier: MIT
"""Explicit Source PP-only Q1 DOWN policies; no new dot arithmetic.

The caller retains its ordinary provider lease and ordered assembly. This
helper owns the BF16 output boundary, so its returned F32 must not be cast
again at that call site. See source-prefill-down-policy provenance.
"""
from functools import lru_cache
import hashlib
import importlib.metadata
from pathlib import Path

POLICY = 'q1_m1_owner_store'
TINY_POLICY = 'q1_tiny_batch_v1'
COUNTER = 'official_source_prefill_q1_m1_owner_store'
TINY_COUNTER = 'official_source_prefill_q1_tiny_down_'
TINY_CONTRACT = 'PP-only Q1 localM1..3 contiguous GU-prefix old DOWN batch; unchanged dot/BF16 owner; per-expert output views and ordinary ordered concat; original/large-Q1/decode unchanged; explicit/unqualified'
LIBRARY_SHA256 = 'd8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6'
SOURCE_PINS = {
    'source_q1.py': '151be733a1edda7feb12fa617860d10008bdb315cee8250d0bf263d16e9aeb05',
    'source_q1_down_owner.py': 'f36787d0a5fd2bfa156377bb058b4279be7a8eb3fb4f5d597cbb43160364d7a8',
}
TINY_SOURCE_PINS = {
    'source_q1.py': '151be733a1edda7feb12fa617860d10008bdb315cee8250d0bf263d16e9aeb05',
    'source_down_prefill.py': '77c7c8603a6a04209968533efe725f452ff95ebc09135503d466d97eb2301337',
    'source_down_fusion.py': '0a5d5e1f7d6571079b58e6a511f531fd0aa02c829076030112fd73bf9847c780',
}


def configuration_supported(args):
    return all(getattr(args, k, None) == v for k, v in (
        ('hidden_size', 5120), ('moe_intermediate_size', 2304),
        ('n_routed_experts', 384), ('num_hidden_layers', 40),
        ('num_experts_per_tok', 6), ('activation_dtype', 'bfloat16'),
        ('simulate_quantization', True)))


def validate_policy(policy, *, expert_math, prefill_batch):
    if type(policy) is not str or policy not in ('reference', POLICY, TINY_POLICY):
        raise ValueError('Invalid Source prefill DOWN policy')
    if policy != 'reference' and (expert_math != 'routed_fused_fp32' or prefill_batch != 'gate_up'):
        raise ValueError('Source Q1 prefill DOWN requires gate_up and routed_fused_fp32')


@lru_cache(maxsize=1)
def implementation_identity():
    import mlx.core as mx
    if importlib.metadata.version('mlx') != '0.32.4':
        raise RuntimeError('Source Q1 M1 DOWN MLX version changed; re-audit required')
    library = Path(mx.__file__).resolve().parent/'lib/libmlx.dylib'
    if hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_SHA256:
        raise RuntimeError('Source Q1 M1 DOWN MLX build changed; re-audit required')
    own = Path(__file__)
    root = own.parent.parent/'runtime/gguf'
    records = ['mlx0.32.4:'+LIBRARY_SHA256]
    for name, expected in SOURCE_PINS.items():
        actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError('Source Q1 M1 DOWN implementation changed; re-audit required')
        records.append(name+':'+actual)
    records.append('policy:'+hashlib.sha256(own.read_bytes()).hexdigest())
    records.append('buffers:'+hashlib.sha256((own.parent/'deepseek_v41_source_buffers.py').read_bytes()).hexdigest())
    return hashlib.sha256((POLICY+':'+':'.join(records)).encode()).hexdigest()


@lru_cache(maxsize=1)
def tiny_implementation_identity():
    """Bind the selected old-Q1-batch mechanism and ordinary assembly path."""
    import mlx.core as mx
    if importlib.metadata.version('mlx') != '0.32.4':
        raise RuntimeError('Source Q1 tiny DOWN MLX version changed; re-audit required')
    library=Path(mx.__file__).resolve().parent/'lib/libmlx.dylib'
    if hashlib.sha256(library.read_bytes()).hexdigest()!=LIBRARY_SHA256:
        raise RuntimeError('Source Q1 tiny DOWN MLX build changed; re-audit required')
    own=Path(__file__);root=own.parent.parent/'runtime/gguf';records=['mlx0.32.4:'+LIBRARY_SHA256]
    for name,expected in TINY_SOURCE_PINS.items():
        actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
        if actual!=expected:raise RuntimeError('Source Q1 tiny DOWN implementation changed; re-audit required')
        records.append(name+':'+actual)
    for path in (own,own.parent/'deepseek_v41_source_weights.py',own.parent/'deepseek_v41_source_prefill.py',
                 own.parent/'deepseek_v41_source_math.py',own.parent/'deepseek_v41_source_buffers.py'):
        records.append(path.name+':'+hashlib.sha256(path.read_bytes()).hexdigest())
    return hashlib.sha256((TINY_POLICY+':'+':'.join(records)).encode()).hexdigest()


def tiny_prefix(batch,*,prefill_rows):
    """Return eligible job/row prefix without gathering or splitting any job."""
    import operator
    if type(prefill_rows)is not int or not 2<=prefill_rows<=1024 or batch.kind!='q1':return 0,0
    try:
        if any(isinstance(v,bool)for v in (*batch.row_counts,*batch.offsets,*batch.job_indices)):raise TypeError
        counts=tuple(operator.index(v)for v in batch.row_counts)
        offsets=tuple(operator.index(v)for v in batch.offsets)
        jobs=tuple(operator.index(v)for v in batch.job_indices)
    except TypeError as error:raise ValueError('Source Q1 tiny prefix requires integer batch metadata')from error
    expected=[0]
    for count in counts:expected.append(expected[-1]+count)
    if (not 1<=len(counts)<=15 or len(jobs)!=len(counts)or len(set(jobs))!=len(jobs)
            or any(j<0 for j in jobs)or any(not 1<=n<=1024 for n in counts)
            or any(a>b for a,b in zip(counts,counts[1:]))or tuple(expected)!=offsets):
        raise ValueError('Source Q1 tiny prefix batch ownership/ordering mismatch')
    count=0
    while count<len(counts)and counts[count]<=3:count+=1
    return count,offsets[count]


def maybe_tiny_project(x,experts,row_counts,*,policy,expert_math,prefill_batch,prefill_rows,counters):
    """Borrow one contiguous GU prefix; return its already-owned F32 DOWN rows."""
    if policy!=TINY_POLICY:return None
    validate_policy(policy,expert_math=expert_math,prefill_batch=prefill_batch)
    import mlx.core as mx
    from .deepseek_v41_source_buffers import PackedExpert
    experts=tuple(experts);counts=tuple(row_counts)
    if (type(prefill_rows)is not int or not 2<=prefill_rows<=1024
            or not isinstance(x,mx.array)or x.dtype!=mx.float32 or x.ndim!=2
            or not 1<=len(counts)==len(experts)<=15
            or any(type(n)is not int or not 1<=n<=3 for n in counts)
            or x.shape!=(sum(counts),2304)
            or any(not isinstance(e,PackedExpert)or e.layout.kind!='q1'or not e.has_compact_backing for e in experts)):
        return None
    if mx.default_device()!=mx.gpu:return None
    from ..runtime.gguf.source_mxfp4_pair import _device_supported
    if not _device_supported():return None
    tiny_implementation_identity()
    from ..runtime.gguf.source_down_prefill import pp_down
    output=pp_down(x,experts,counts)
    for suffix,amount in (('batches',1),('jobs',len(experts)),('rows',sum(counts)),('output_views',len(experts))):
        key=TINY_COUNTER+suffix;counters[key]=counters.get(key,0)+amount
    return output


def eligible(x, data, *, prefill_rows, check_device=True):
    # ExpertLayout validates the complete contiguous Q1 schema. Native sealed
    # files may arrange those tensors differently from canonical_layout; the
    # kernel consumes separate named DOWN views, never whole-slab offsets.
    import mlx.core as mx
    from .deepseek_v41_source_buffers import ExpertLayout, PackedExpert
    if (type(prefill_rows) is not int or not 2 <= prefill_rows <= 1024
            or not isinstance(x, mx.array) or x.dtype != mx.float32 or x.shape != (1, 2304)
            or not isinstance(data, PackedExpert) or not isinstance(data.layout, ExpertLayout)
            or data.layout.kind != 'q1'
            or not isinstance(data.backing, mx.array) or data.backing.dtype != mx.uint8
            or data.backing.shape != (data.layout.nbytes,)):
        return False
    for field, shape, dtype in (('qweight', (5120, 72), mx.uint32), ('scales', (5120, 36), mx.float16), ('biases', (5120, 36), mx.float16)):
        value = data.get('down.'+field)
        if not isinstance(value, mx.array) or value.shape != shape or value.dtype != dtype:
            return False
    if not check_device:
        return True
    if mx.default_device() != mx.gpu:
        return False
    from ..runtime.gguf.source_mxfp4_pair import _device_supported
    return bool(_device_supported())


def maybe_project(x, data, *, policy, expert_math, prefill_batch, prefill_rows, counters):
    """Return None for unsupported geometry; selected faults propagate.

    No acquire, readback, coefficient copy, cast or persistent object cache is
    introduced. The existing provider keeps all typed views/backing leased.
    """
    if policy == 'reference':
        return None
    validate_policy(policy, expert_math=expert_math, prefill_batch=prefill_batch)
    if policy!=POLICY:
        return None
    if not eligible(x, data, prefill_rows=prefill_rows):
        return None
    implementation_identity()
    from ..runtime.gguf.source_q1_down_owner import project
    result = project(x, data['down.qweight'], data['down.scales'], data['down.biases'])
    counters[COUNTER] = counters.get(COUNTER, 0)+1
    return result
