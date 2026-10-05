"""Explicit whole-prompt long packed research contract; no short-domain widening."""
from .paged_pack_price import _identity
from .hybrid_packed_prefill import packed_environment
SCHEMA='mlx2.hybrid-packed-prefill-long-research.v1'
COUNTS=(6950,6929)
BUDGET=40<<30


def long_environment():
    env=packed_environment(False,1)
    env.update(MLX2_PAGED_PREFILL_NAX_LONG_FUSED='1',MLX2_PAGED_Q1_STOCK_LONG='1',
        MLX2_PAGED_Q1_STOCK_REDUCTION='0',MLX2_PAGED_Q1_STOCK_SINGLETON='0',
        MLX2_PAGED_Q1_STOCK_LONG_INLINE_METADATA='0',MLX_SDPA_BLOCKS='0')
    return env


def make_long_profile(identity, *, research_output_cap20=False):
    if type(research_output_cap20) is not bool:raise ValueError('long cap20 selector must be boolean')
    result= {'schema':SCHEMA,'profile_id':'packed-long-6950-6929-nax-fused-research-v1',
        'identity':identity,'storage_dtype':'bfloat16','memory_budget_bytes':BUDGET,
        'required_environment':long_environment(),'qualified':False,'price_usable':False,
        'serving_default':False,'context_lengths':list(COUNTS),'q1_simd_stripes':16,
        'stock_reduction':False,'stock_singleton':False,'prefill_nax_exact':False,
        'prefill_long_nax':True,'prefill_eval_block_size':1}
    if research_output_cap20:
        result.update(research_output_cap20=True,profile_id='packed-long-6950-6929-nax-cap20-research-v1')
    return result


def validate_long_profile(profile,*,live_identity,counts,environment):
    extended=profile.get('research_output_cap20',False)
    if type(extended) is not bool:raise ValueError('long cap20 selector must be boolean')
    expected=make_long_profile(live_identity,research_output_cap20=extended)
    if (type(profile) is not dict or set(profile)!=set(expected) or
        _identity(profile.get('identity'))!=_identity(live_identity) or
        type(counts) is not tuple or sorted(counts)!=sorted(COUNTS) or
        profile.get('context_lengths')!=list(counts)):
        raise ValueError('long packed profile identity/whole-prompt domain differs')
    for key,value in expected.items():
        if key in ('identity','context_lengths'):continue
        if type(profile[key]) is not type(value) or profile[key]!=value:
            raise ValueError('long packed profile selector/budget differs: '+key)
    if any(environment.get(k,'0')!=v for k,v in expected['required_environment'].items()):
        raise ValueError('long packed source environment differs')
    return profile


def require_long_capabilities(native,backend_type,counts):
    for name in ('grouped_multirow_write','grouped_multirow_write_count','grouped_multirow_row_count',
                 'prefill_matrix_dispatch_count','prefill_long_nax_capability','prefill_long_nax_dispatch_count',
                 'q1_stock_long_partial_dispatch_count','q1_stock_long_reduce_dispatch_count'):
        if not callable(getattr(native,name,None)):raise ValueError('long native capability absent: '+name)
    if not callable(getattr(backend_type,'append_packed_multirow',None)):
        raise ValueError('long packed writer host capability absent')
    cap=native.prefill_long_nax_capability()
    expected={'version':1,'storage_dtype':'bfloat16','head_dim':256,'query_heads':24,'kv_heads':4,
        'max_spans':2,'min_query_count':256,'max_query_count':8192,'max_causal_end':8192,
        'origin_zero':True,'window_zero':True,'architecture':'s','scratch_bytes':0,'physical_dispatches':1,'qualified':False}
    if (type(cap) is not dict or any(type(cap.get(k)) is not type(v) or cap.get(k)!=v for k,v in expected.items()) or
            len(counts)!=2 or any(type(n) is not int or not 256<=n<=8192 for n in counts)):
        raise ValueError('long fused native geometry/zero-scratch capability differs')
    return cap


def require_host_budget(profile,charge,device_info):
    memory=device_info.get('memory_size')
    if (type(memory) is not int or memory<64<<30 or type(charge) is not int or
            not 0<charge<=profile['memory_budget_bytes']<=min(BUDGET,memory*3//4)):
        raise MemoryError('long research charge requires explicit40GiB cap and >=64GiB host')
