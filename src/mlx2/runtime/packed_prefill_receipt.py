"""Generic candidate-bound attribution for a terminal-proved packed bootstrap.

No model-name decisions, no tensor imports; invalid evidence fails closed.
Missing packed evidence retains the existing imported-bootstrap description.
"""
from .paged_pack_price import _identity


def bootstrap_prefill_attribution(candidate):
    if getattr(candidate,'_serving_n20',False) is True:
        from .hybrid_packed_prefill_n import bootstrap_attribution
        return bootstrap_attribution(candidate)
    proof=getattr(candidate,'_packed_prefill_receipt',None)
    if proof is None:
        return {'prefill_mode':'ordinary_completed_import','bootstrap_generation':0,
                'native_prefill_observed_used':False,'native_prefill_attention_calls':0}
    depth=getattr(candidate,'native_layer_count',None)
    lengths=proof.get('segment_lengths',()) if isinstance(proof,dict) else ()
    long_fused=proof.get('prefill_long_nax',False) if isinstance(proof,dict) else False
    exact_nax=proof.get('prefill_nax_exact',False) if isinstance(proof,dict) else False
    expected={'grouped_multirow_write_count':depth,'grouped_multirow_row_count':depth*sum(lengths) if type(depth) is int else 0,'prefill_matrix_dispatch_count':depth}
    if long_fused:expected['prefill_long_nax_dispatch_count']=depth
    if exact_nax:expected.update({'prefill_nax_'+stage+'_dispatch_count':depth for stage in ('score','softmax','value')})
    if (type(proof) is not dict or type(long_fused) is not bool or (long_fused and exact_nax) or type(exact_nax) is not bool or type(depth) is not int or depth<1 or
        proof.get('prefill_mode')!='packed_hybrid_real_rows' or
        proof.get('bootstrap_generation')!=0 or getattr(candidate,'bootstrap_generation',1)!=0 or
        type(lengths) not in (list,tuple) or len(lengths)!=2 or
        any(type(n) is not int or n<2 for n in lengths) or
        proof.get('real_projection_rows')!=sum(lengths) or
        proof.get('full_attention_layers')!=depth or proof.get('terminal_read_count')!=depth or
        proof.get('qualified') is not False or proof.get('price_usable') is not False or
        proof.get('selected') is not True or proof.get('observed_used') is not True or
        proof.get('physical_counters')!=expected):
        raise ValueError('packed bootstrap attribution lacks terminal/row/dispatch proof')
    if exact_nax and (any(not 9<=n<=129 for n in lengths) or
        proof.get('native_reader_scratch_bytes')!=sum(lengths)*24*129*4 or
        proof.get('native_reader_scratch_bound_bytes')!=3195072 or
        proof.get('prefill_attention_arithmetic')!='nax_three_stage_stock_short'):
        raise ValueError('packed NAX attribution lacks declared charge/geometry proof')
    if long_fused and (sorted(lengths)!=[6929,6950] or proof.get('native_reader_scratch_bytes')!=0 or
            proof.get('native_reader_simultaneous_scratch_bytes')!=0 or
            proof.get('bootstrap_charge_components',{}).get('native_reader_bytes')!=0 or
            proof.get('prefill_attention_arithmetic')!='nax_fused_stock_long' or proof.get('prefill_eval_block_size')!=1):
        raise ValueError('long packed attribution lacks whole-prompt/zero-scratch proof')
    identity=_identity(proof.get('source_identity'))
    backend=candidate.backend;arena=backend.writer.backend
    if backend.read_submissions<depth or backend.terminal_successes<depth:
        raise ValueError('packed bootstrap terminal counters are incomplete')
    for name,required in proof['physical_counters'].items():
        getter=getattr(arena._native,name,None)
        if not callable(getter) or int(getter(arena._arena))<required:
            raise ValueError('packed bootstrap live native counter proof differs')
    return {'prefill_mode':'native_packed_prefill','prefill_layout':'real_rows',
        'bootstrap_generation':0,'native_prefill_observed_used':True,
        'native_prefill_attention_calls':depth,'native_prefill_source_identity':identity,
        'native_prefill_proof':dict(proof)}
