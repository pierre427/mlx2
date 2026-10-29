from types import SimpleNamespace

import mlx.core as mx

from mlx2.adapters.nanbeige43_cache import candidate_q8_cache
from mlx2.runtime.generate import _corrupt_rows
from mlx2.runtime.greedy_scores import greedy_scores, receipt
from mlx2.runtime.models.base import scaled_dot_product_attention
from mlx2.runtime.models.cache import KVCache


def run():
    mx.random.seed(4276)
    rows=[]
    logits = mx.array([[-2e-8, 0.] + [-.1] * 14], dtype=mx.float32)
    normalized = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    ids, scores = greedy_scores(logits)
    mx.eval(ids, normalized)
    assert mx.argmax(logits, axis=-1).tolist() == [1]
    assert mx.argmax(normalized, axis=-1).tolist() == [0]
    assert ids.tolist() == [0]
    assert mx.array_equal(scores[0].materialize_logprobs(), normalized[0]).item()
    rows.append({"kind": "normalization_created_top_tie", "batch": 1,
                 "vocab": 16, "raw_argmax": 1, "normalized_argmax": 0,
                 "candidate_argmax": 0, "full_score_row_exact": True})
    for b,v in [(1,3),(4,4096),(1,166912),(2,166912)]:
        x=mx.random.normal((b,v)).astype(mx.float32)
        # Force exact ties at nonzero positions and extreme common offsets.
        if v>10: x[:,5]=4.;x[:,7]=4.
        for offset in [0.,1e8,-1e8]:
            logits=x+offset
            actual,scores=greedy_scores(logits)
            expected=mx.argmax(logits-mx.logsumexp(logits,axis=-1,keepdims=True),axis=-1)
            mx.eval(actual,expected)
            assert mx.array_equal(actual,expected).item(),(b,v,offset)
            assert all(mx.array_equal(s.materialize_logprobs(),(logits-mx.logsumexp(logits,axis=-1,keepdims=True))[i]).item() for i,s in enumerate(scores))
            rows.append({'batch':b,'vocab':v,'offset':offset,'token_parity':True})
    for values in [[-float('inf')]*3,[float('nan'),0.,1.],[float('inf'),0.,1.]]:
        ids,scores=greedy_scores(mx.array([values]));_,valid=_corrupt_rows(ids,scores);mx.eval(valid)
        assert valid.tolist()==[False]
    reserved=KVCache().reserve(1024)
    k=mx.random.normal((1,8,513,128)).astype(mx.bfloat16)
    reserved.update_and_fetch(k,k);mx.eval(reserved.state)
    assert reserved.offset==513 and reserved.keys.shape[2]==1024
    assert mx.array_equal(reserved.keys_and_values()[0],k).item()
    qrows=[]
    for context in [17,512,4096]:
        cache=candidate_q8_cache(SimpleNamespace(layers=[1]),allow_approximate=True)[0]
        k=mx.random.normal((1,8,context,128)).astype(mx.bfloat16)
        v=mx.random.normal((1,8,context,128)).astype(mx.bfloat16)
        qk,qv=cache.update_and_fetch(k,v)
        q=mx.random.normal((1,56,1,128)).astype(mx.bfloat16)
        y=scaled_dot_product_attention(q,qk,qv,cache,128**-.5,None)
        reference=mx.fast.scaled_dot_product_attention(q,mx.dequantize(*qk,group_size=64,bits=8),mx.dequantize(*qv,group_size=64,bits=8),scale=128**-.5)
        mx.eval(y,reference);assert mx.isfinite(y).all().item()
        qrows.append({'context':context,'max_abs_error_same_representation':mx.max(mx.abs(y-reference)).item(),'packed_cache_bytes':sum(t.nbytes for t in (*cache.keys,*cache.values))})
    return {'greedy':rows,'nonfinite_rejected':True,'reservation_exact':True,'q8':qrows,'greedy_receipt':receipt()}
