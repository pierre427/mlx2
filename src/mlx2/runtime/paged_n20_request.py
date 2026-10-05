"""Pure HTTP contract for the explicit, unqualified packed N20 research route."""
def validate_request(body):
    selected=body.get('paged_native_packed_n20_research',False)
    if type(selected) is not bool:raise ValueError('paged_native_packed_n20_research must be boolean')
    identifier=body.get('native_research_input_id');digest=body.get('native_research_inputs_sha256')
    if identifier is not None or digest is not None or selected:
        if (not isinstance(identifier,str) or not 1<=len(identifier)<=256 or any(ord(c)<32 for c in identifier) or
                not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('native research metadata requires a concrete case ID and lowercase SHA256')
    if not selected:return body
    cohort=body.get('batch_cohort');cap=body.get('max_tokens')
    if (body.get('skip_writing_prefix_cache') is not True or type(cohort) is not dict or type(cohort.get('size')) is not int or
            not 1<=cohort['size']<=20 or type(cap) is not int or not 1<=cap<=192 or body.get('temperature')!=0 or
            body.get('enable_thinking') is not False or body.get('min_tokens',0)!=0 or
            any(body.get(k) is True for k in ('paged_native_qwen3','paged_native_qwen3_b2','paged_native_hybrid_b2','paged_native_hybrid_packed_prefill'))):
        raise ValueError('native packed N20 requires a separate cold1..20-member greedy cohort, max192 and thinking off')
    return body


def validate_ready_job(job,width):
    """Named source-bound guards for actual resolved serving state."""
    import math
    request=job.request;sampling=job.effective_sampling or {}
    checks={
        'explicit_selection':request.get('paged_native_packed_n20_research') is True,
        'no_apc_write':request.get('skip_writing_prefix_cache') is True,
        'exclusive_capability':not any(request.get(k) is True for k in ('paged_native_qwen3','paged_native_qwen3_b2','paged_native_hybrid_b2','paged_native_hybrid_packed_prefill')),
        'complete_cohort':request.get('batch_cohort',{}).get('size')==width,
        'not_preempted':not job.preempted,'not_cancelled':not job.cancelled.is_set(),
        'attached_uid':type(job.uid) is int,'zero_cached_tokens':job.native_b2_cached_tokens==0,
        'output_cap':type(job.effective_max_tokens) is int and 1<=job.effective_max_tokens<=192,
        'effective_greedy':sampling.get('temperature')==0,
        'neutral_repetition':sampling.get('repetition_penalty')==1,
        'neutral_frequency':sampling.get('frequency_penalty')==0,
        'bounded_presence':type(sampling.get('presence_penalty')) in (int,float) and math.isfinite(sampling['presence_penalty']) and -2<=sampling['presence_penalty']<=2,
        'no_thinking_guard':job.thinking_guard is None,'no_structured_processor':job.structured is None,
        'no_sampling_profile':request.get('sampling_profile') is None,'no_minimum':request.get('min_tokens',0)==0,
        'no_extra_processors':all(request.get(k) in (None,False,0,(),[]) for k in ('grammar','response_format','logit_bias','stop','tools','tool_choice','thinking_budget','thinking_steer_alpha')),
    }
    failed=[name for name,ok in checks.items() if not ok]
    if failed:raise ValueError('nativeN20 ready-job refusal: '+','.join(failed))
    return sampling


def validate_ordinary_processors(processors,sampling,prompt_length):
    """Preserve the existing ordinary generated-only presence processor."""
    presence=sampling['presence_penalty']
    if not processors and presence==0:return ()
    if type(processors) is not list or len(processors)!=1 or presence==0:
        raise ValueError('nativeN20 queued-processor refusal: unsupported_processor_count')
    processor=processors[0]
    if (not callable(processor) or getattr(processor,'presence_window',None)!=(presence,0,prompt_length) or
            getattr(processor,'history_pure',False) is not True or getattr(processor,'probe',None) is not processor or
            getattr(processor,'__module__',None)!='mlx2.runtime.sample_utils' or
            getattr(processor,'__qualname__',None)!='make_presence_penalty.<locals>.presence_penalty_processor'):
        raise ValueError('nativeN20 queued-processor refusal: ordinary_presence_contract')
    return (processor,)
