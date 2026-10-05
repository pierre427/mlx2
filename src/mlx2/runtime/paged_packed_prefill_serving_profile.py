"""Explicit source-bound short packed-prefill serving permit, default off.

The serial hybrid admission profile remains separate. No measured price or
qualification is inferred from a direct model or primitive parity receipt.
"""
from __future__ import annotations
import json
from pathlib import Path
from .hybrid_packed_prefill import SCHEMA as FACTORY_SCHEMA,packed_environment,validate_profile,evaluation_block_size
SCHEMA='mlx2.native-packed-prefill-b2-serving-research.v1'


def validate_request(body):
    selected=body.get('paged_native_hybrid_packed_prefill',False)
    if type(selected) is not bool:raise ValueError('paged_native_hybrid_packed_prefill must be boolean')
    extended=body.get('paged_native_long_cap20_research',False)
    if type(extended) is not bool or (extended and not selected):raise ValueError('long20 research requires explicit packed route')
    if selected and (body.get('paged_native_hybrid_b2') is not True or
            body.get('skip_writing_prefix_cache') is not True or
            body.get('paged_native_qwen3') is True or body.get('paged_native_qwen3_b2') is True or
            type(body.get('max_tokens')) is not int or not 1<=body['max_tokens']<=(20 if extended else 4) or
            body.get('temperature')!=0):
        raise ValueError('packed prefill requires explicit cold hybrid B2 greedy caps1..4 and no APCv2 write')
    return selected


def make_profile(identity, *, prefill_eval_block_size=1, long_fused=False, research_output_cap20=False):
    if type(research_output_cap20) is not bool or (research_output_cap20 and not long_fused):raise ValueError('cap20 requires explicit long research profile')
    if type(long_fused) is not bool:raise ValueError('long prefill selector must be boolean')
    if long_fused:
        from .hybrid_packed_prefill_long import make_long_profile,COUNTS
        if type(prefill_eval_block_size) is not int or prefill_eval_block_size!=1:raise ValueError('long prefill requires eager block1')
        p=make_long_profile(identity,research_output_cap20=research_output_cap20);p['schema']='mlx2.native-packed-prefill-long-serving-research.v1'
        p.update(warm_apcv2=False,max_tokens=20 if research_output_cap20 else 4,sampling={'mode':'greedy','processors':False},numerical_reference='same_geometry_ordinary_mixed')
        return p
    block=evaluation_block_size(prefill_eval_block_size)
    profile= {'schema':SCHEMA,'profile_id':'native-packed-prefill-b2-32-96-nax-serving-v1',
        'identity':identity,'storage_dtype':'bfloat16','memory_budget_bytes':12<<30,
        'required_environment':packed_environment(True,block),'qualified':False,'price_usable':False,
        'serving_default':False,'warm_apcv2':False,'context_lengths':[32,96],'max_tokens':4,
        'sampling':{'mode':'greedy','processors':False},'numerical_reference':'same_geometry_ordinary_mixed','q1_simd_stripes':16,
        'stock_reduction':True,'stock_singleton':True,'prefill_nax_exact':True}
    if block!=1:profile['prefill_eval_block_size']=block
    return profile


def factory_profile(profile,counts):
    result={k:v for k,v in profile.items() if k not in ('warm_apcv2','max_tokens','sampling','numerical_reference')}
    result['schema']=('mlx2.hybrid-packed-prefill-long-research.v1' if profile.get('prefill_long_nax') is True else FACTORY_SCHEMA);result['context_lengths']=list(counts)
    return result


def load_profile(path,*,live_identity,context_lengths,environment):
    data=json.loads(Path(path).read_text())
    if data.get('schema')=='mlx2.native-packed-prefill-long-serving-research.v1':
        from .hybrid_packed_prefill_long import validate_long_profile,COUNTS
        extended=data.get('research_output_cap20',False)
        if type(extended) is not bool:raise ValueError('long cap20 selector must be boolean')
        expected=make_profile(live_identity,long_fused=True,research_output_cap20=extended)
        if (set(data)!=set(expected) or data.get('context_lengths')!=list(COUNTS) or
                any(type(data.get(k)) is not type(v) or data.get(k)!=v for k,v in expected.items() if k not in ('identity',))):
            raise ValueError('long serving profile whole-prompt scope differs')
        validate_long_profile(factory_profile(data,context_lengths),live_identity=live_identity,counts=context_lengths,environment=environment)
        return data
    expected=set(make_profile(live_identity))
    if (type(data) is not dict or not expected<=set(data)<=expected|{'prefill_eval_block_size'} or data.get('schema')!=SCHEMA or
            data.get('context_lengths')!=[32,96] or data.get('max_tokens')!=4 or type(data.get('max_tokens')) is not int or
            data.get('numerical_reference')!='same_geometry_ordinary_mixed' or data.get('warm_apcv2') is not False or data.get('sampling')!={'mode':'greedy','processors':False} or
            data.get('prefill_nax_exact') is not True or type(context_lengths) is not tuple or
            len(context_lengths)!=2 or any(type(n) is not int for n in context_lengths) or
            sorted(context_lengths)!=[32,96]):
        raise ValueError('packed prefill serving requires exact source-bound cold32/96 NAX scope')
    validate_profile(factory_profile(data,context_lengths),live_identity=live_identity,counts=context_lengths,environment=environment)
    return data


def startup_environment(path):
    """Validate declared selector scope before applying an explicit CLI profile.

    Live artifact/source/native identity is verified at admission before any
    native allocation. This only binds the declared process selector values.
    """
    raw=json.loads(Path(path).read_text())
    if type(raw) is not dict:raise ValueError('packed-prefill profile object required')
    profile=load_profile(path,live_identity=raw.get('identity'),context_lengths=(6950,6929) if raw.get('prefill_long_nax') is True else (32,96),
        environment=raw.get('required_environment',{}))
    return dict(profile['required_environment'])
