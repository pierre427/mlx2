"""Small CPU equation tests; no artifact/GPU/qualification claims."""
import dataclasses
import numpy as np
import pytest
import mlx.core as mx
mx.set_default_device(mx.cpu)
from mlx2.runtime.models.deepseek_v4 import (
    DeepSeekV4Config, DeepSeekV4Model, Rotary, attend, limited_swiglu,
    route_experts, sinkhorn_mix,
)


def config(ratios=(0, 4, 128)):
    return DeepSeekV4Config(vocab_size=19, hidden_size=8, num_hidden_layers=len(ratios),
        num_attention_heads=2, head_dim=8, q_lora_rank=4, qk_rope_head_dim=4,
        o_groups=2, o_lora_rank=4, n_routed_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=6, num_hash_layers=1, rms_norm_eps=1e-6,
        index_n_heads=2, index_head_dim=4, index_topk=2, compress_ratios=ratios, simulate_quantization=False)


class SmallWeights:
    """Independent dense NumPy oracle storage with real GGUF semantic names."""
    def __init__(self, c):
        self.c, self.data = c, {}
        rng = np.random.default_rng(27)
        def put(name, shape, ones=False):
            self.data[name] = mx.array(np.ones(shape, np.float32) if ones else rng.normal(0, .08, shape).astype(np.float32))
        put('token_embd.weight', (c.vocab_size,c.hidden_size))
        put('output.weight',(c.vocab_size,c.hidden_size))
        put('output_norm.weight',(c.hidden_size,),True)
        for prefix, head in [('output_hc', True)]+[(f'blk.{i}.hc_{kind}',False) for i in range(c.num_hidden_layers) for kind in ('attn','ffn')]:
            put(prefix+'_fn.weight',((c.hc_mult if head else (2+c.hc_mult)*c.hc_mult),c.hc_mult*c.hidden_size))
            put(prefix+'_base.weight',((c.hc_mult if head else (2+c.hc_mult)*c.hc_mult),))
            put(prefix+'_scale.weight',((1 if head else 3),),True)
        for i,ratio in enumerate(c.compress_ratios):
            p=f'blk.{i}.'
            for name,n,k in [('attn_q_a',c.q_lora_rank,c.hidden_size),('attn_q_b',c.num_attention_heads*c.head_dim,c.q_lora_rank),('attn_kv',c.head_dim,c.hidden_size),('attn_output_a',c.o_groups*c.o_lora_rank,c.num_attention_heads*c.head_dim//c.o_groups),('attn_output_b',c.hidden_size,c.o_groups*c.o_lora_rank),('ffn_gate_inp',c.n_routed_experts,c.hidden_size),('ffn_gate_shexp',c.moe_intermediate_size,c.hidden_size),('ffn_up_shexp',c.moe_intermediate_size,c.hidden_size),('ffn_down_shexp',c.hidden_size,c.moe_intermediate_size)]: put(p+name+'.weight',(n,k))
            for name,n in [('attn_q_a_norm',c.q_lora_rank),('attn_kv_a_norm',c.head_dim),('attn_norm',c.hidden_size),('ffn_norm',c.hidden_size)]: put(p+name+'.weight',(n,),True)
            put(p+'attn_sinks.weight',(c.num_attention_heads,))
            if i<c.num_hash_layers:
                self.data[p+'ffn_gate_tid2eid.weight']=mx.array(np.tile(np.arange(c.num_experts_per_tok,dtype=np.int32),(c.vocab_size,1)))
            else: put(p+'exp_probs_b.bias',(c.n_routed_experts,))
            for kind,shape in [('gate',(c.n_routed_experts,c.moe_intermediate_size,c.hidden_size)),('up',(c.n_routed_experts,c.moe_intermediate_size,c.hidden_size)),('down',(c.n_routed_experts,c.hidden_size,c.moe_intermediate_size))]: put(p+'ffn_'+kind+'_exps.weight',shape)
            if ratio:
                for comp,dim in [('attn_compressor',c.head_dim)]+([('indexer_compressor',c.index_head_dim)] if ratio==4 else []):
                    width=dim*(2 if ratio==4 else 1)
                    put(p+comp+'_ape.weight',(ratio,width))
                    put(p+comp+'_norm.weight',(dim,),True)
                    put(p+comp+'_kv.weight',(width,c.hidden_size))
                    put(p+comp+'_gate.weight',(width,c.hidden_size))
            if ratio==4:
                put(p+'indexer.attn_q_b.weight',(c.index_n_heads*c.index_head_dim,c.q_lora_rank))
                put(p+'indexer.proj.weight',(c.index_n_heads,c.hidden_size))

    def array(self,name): return self.data[name]
    def linear(self,name): return lambda x: x @ self.data[name].T
    def embedding(self,ids): return self.data['token_embd.weight'][ids]
    def grouped(self,x,name,groups):
        w=self.data[name].reshape(groups,-1,x.shape[-1])
        return mx.einsum('...gi,goi->...go',x,w)
    def moe(self,x,ids,scores,prefix):
        gate=self.data[prefix+'.ffn_gate_exps.weight'][ids]
        up=self.data[prefix+'.ffn_up_exps.weight'][ids]
        down=self.data[prefix+'.ffn_down_exps.weight'][ids]
        g=mx.sum(gate*x[:,None,None,:],axis=-1)
        u=mx.sum(up*x[:,None,None,:],axis=-1)
        z=limited_swiglu(g,u)
        return mx.sum(mx.sum(down*z[:,:,None,:],axis=-1)*scores[...,None],axis=1)


def test_prefill_chunked_and_decode_same_prefix():
    c=config(); w=SmallWeights(c); model=DeepSeekV4Model(c,w,identity='synthetic@cpu')
    ids=mx.array(np.arange(137)[None]%c.vocab_size,dtype=mx.int32)
    whole=model(ids,model.make_cache())
    state=model.make_cache(); pieces=[]
    for a,b in [(0,3),(3,7),(7,129),(129,133),(133,134),(134,137)]:
        pieces.append(model(ids[:,a:b],state))
    chunk=mx.concatenate(pieces,axis=1)
    mx.eval(whole,chunk)
    np.testing.assert_allclose(np.array(whole),np.array(chunk),atol=2e-6,rtol=2e-5)
    assert all(s.offset==137 for s in state)
    assert state[1].compressor.pooled.shape[1]==34
    assert state[2].compressor.pooled.shape[1]==1


def test_router_correction_selects_but_does_not_reweight():
    logits=mx.array([[[1.,2.,3.,4.]]]); correction=mx.array([100.,0,0,0])
    ids,weights=route_experts(logits,top_k=2,scale=1.5,correction=correction)
    assert 0 in np.array(ids)
    s=np.sqrt(np.logaddexp(0,np.array(logits)))
    expected=np.take_along_axis(s,np.array(ids),axis=-1); expected=expected/expected.sum(-1,keepdims=True)*1.5
    np.testing.assert_allclose(np.array(weights),expected,rtol=2e-6)


def test_rotary_last_coordinates_inverse():
    x=mx.array(np.arange(48,dtype=np.float32).reshape(1,2,3,8)/50)
    rope=Rotary(4,10000.)
    y=rope(x,7); z=rope(y,7,inverse=True)
    np.testing.assert_array_equal(np.array(y)[...,:4],np.array(x)[...,:4])
    np.testing.assert_allclose(np.array(z),np.array(x),atol=1e-7)


def test_sink_mass_has_no_value_contribution():
    q=mx.ones((1,1,1,4)); kv=mx.ones((1,1,1,4)); mask=mx.ones((1,1,1,1),dtype=mx.bool_)
    y=attend(q,kv,mask,sinks=mx.array([2.]))
    np.testing.assert_allclose(np.array(y),np.full((1,1,1,4),.5),atol=1e-6)


def test_sinkhorn_fp16_loaded_base_is_numeric_not_bitcast():
    z=mx.zeros((1,1,24)); scale=mx.ones((3,),dtype=mx.float16); base=mx.arange(24,dtype=mx.float16)/10
    pre,post,comb=sinkhorn_mix(z,scale,base,4,20,1e-6)
    np.testing.assert_allclose(np.array(comb).sum(-1),1,atol=1e-5)
    assert np.isfinite(np.array(pre)).all() and np.isfinite(np.array(post)).all()


def test_batch_and_cache_mismatch_fail_closed():
    c=config((0,)); model=DeepSeekV4Model(c,SmallWeights(c),identity='synthetic')
    with pytest.raises(ValueError,match='B1'): model(mx.ones((2,1),dtype=mx.int32))
    with pytest.raises(ValueError,match='layer count'): model(mx.ones((1,1),dtype=mx.int32),[])


def _fp8_levels():
    values=[0.]
    for code in range(1,127):
        exp,frac=code>>3,code&7
        values.append(frac*2.**-9 if exp==0 else (1+frac/8)*2.**(exp-7))
    return np.array(values,np.float32)


def _round_levels(value,levels):
    result=[]
    for x in np.asarray(value).flat:
        dist=np.abs(levels-abs(x)); choices=np.flatnonzero(dist==dist.min())
        even=choices[choices%2==0]; code=(even[0] if len(even) else choices[0])
        result.append(np.copysign(levels[code],x))
    return np.array(result,np.float32).reshape(value.shape)


def test_fp8_qat_matches_independent_codebook_including_even_midpoints():
    from mlx2.runtime.models.deepseek_v4 import simulate_fp8_kv
    levels=_fp8_levels(); middle=(levels[:-1]+levels[1:])/2
    values=np.concatenate((levels,middle,-levels,-middle))
    padded=np.pad(values,(0,(-len(values))%64)).reshape(-1,64)
    # Force scale=1 in every block, including subnormal test points.
    padded[:,-1]=448.
    tails=np.full((len(padded),64),1.234567,np.float32)
    x=np.concatenate((padded,tails),axis=1)
    expected=_round_levels(padded,levels)
    actual=np.array(simulate_fp8_kv(mx.array(x),64))
    np.testing.assert_array_equal(actual[:,:64],expected)
    np.testing.assert_array_equal(actual[:,64:],tails)


def test_indexer_qat_numpy_butterfly_and_even_code_oracle():
    from mlx2.runtime.models.deepseek_v4 import simulate_index_fp4
    rng=np.random.default_rng(44); original=rng.normal(size=(5,128)).astype(np.float32)
    rotated=original.copy()
    for stride in (1,2,4,8,16,32,64):
        for base in range(0,128,2*stride):
            a=rotated[:,base:base+stride].copy(); b=rotated[:,base+stride:base+2*stride].copy()
            rotated[:,base:base+stride]=a+b; rotated[:,base+stride:base+2*stride]=a-b
    rotated*=np.float32(1/np.sqrt(128))
    groups=rotated.reshape(5,4,32)
    amax=np.maximum(np.max(np.abs(groups),axis=-1,keepdims=True),np.float32(7.052966104933725e-38))
    scale=2.**np.ceil(np.log2(amax/6))
    expected=_round_levels(groups/scale,np.array([0,.5,1,1.5,2,3,4,6],np.float32))*scale
    actual=np.array(simulate_index_fp4(mx.array(original)))
    np.testing.assert_allclose(actual,expected.reshape(original.shape),rtol=0,atol=0)


def test_quantized_activation_forward_chunk_consistency():
    c=dataclasses.replace(config((4,)), head_dim=128, qk_rope_head_dim=64,
                          index_head_dim=128, simulate_quantization=True)
    model=DeepSeekV4Model(c,SmallWeights(c),identity='qat-synthetic')
    ids=mx.array(np.arange(17)[None]%c.vocab_size,dtype=mx.int32)
    full=model(ids,model.make_cache())
    cache=model.make_cache()
    chunk=mx.concatenate([model(ids[:,:3],cache),model(ids[:,3:11],cache),model(ids[:,11:],cache)],axis=1)
    np.testing.assert_allclose(np.array(full),np.array(chunk),atol=1e-6,rtol=1e-5)


def test_sparse_attention_matches_explicit_dense_selected_oracle():
    rng=np.random.default_rng(8)
    q=rng.normal(size=(1,2,5,4)).astype(np.float32)
    local=rng.normal(size=(1,1,7,4)).astype(np.float32)
    pool=rng.normal(size=(1,9,4)).astype(np.float32)
    indices=np.tile(np.array([0,4,7]),(1,5,1)).astype(np.int32)
    lm=np.ones((1,1,5,7),bool); pm=np.ones((1,1,5,9),bool); pm[:,:,:,7]=False
    sink=np.array([.1,-.3],np.float32)
    actual=np.array(attend(mx.array(q),mx.array(local),mx.array(lm),pooled=mx.array(pool),pool_mask=mx.array(pm),indices=mx.array(indices),sinks=mx.array(sink)))
    expected=np.zeros_like(q)
    for h in range(2):
        for row in range(5):
            values=np.concatenate([local[0,0],pool[0,[0,4]]],axis=0)
            scores=q[0,h,row]@values.T/2
            denom=np.exp(scores).sum()+np.exp(sink[h])
            expected[0,h,row]=np.exp(scores)@values/denom
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)


def test_exact_indexer_tiling_preserves_selected_set():
    from mlx2.runtime.models.deepseek_v4 import index_select
    rng=np.random.default_rng(19)
    q=rng.normal(size=(1,3,7,4)).astype(np.float32)
    pool=rng.normal(size=(1,11,4)).astype(np.float32)
    weights=rng.normal(size=(1,7,3)).astype(np.float32)
    mask=np.ones((1,1,7,11),bool); mask[:,:,:,:2]=False
    actual=np.array(index_select(mx.array(q),mx.array(pool),mx.array(weights),mx.array(mask),3,workspace_bytes=1))
    scores=np.maximum(q@pool[:,None].swapaxes(-1,-2),0)*weights.transpose(0,2,1)[...,None]/2
    scores=scores.sum(1); scores=np.where(mask[:,0],scores,-np.inf)
    expected=np.argsort(-scores,axis=-1)[...,:3]
    np.testing.assert_array_equal(np.sort(actual,axis=-1),np.sort(expected,axis=-1))


def test_prefill_head_checkpoint_context_and_explicit_full_mode():
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
    c=config((0,)); w=SmallWeights(c); model=DeepSeekV4Model(c,w,identity='head-test')
    ids=mx.array([[1,2,3,4]],dtype=mx.int32)
    full=model(ids,model.make_cache())
    assert full.shape==(1,4,c.vocab_size)
    with prefill_checkpoint_rows([4],width=4) as request:
        narrow=model.prefill_forward(ids,model.make_cache())
    assert request.applied==1 and narrow.shape==(1,1,c.vocab_size)
    np.testing.assert_allclose(np.array(narrow),np.array(full[:,-1:]),atol=1e-6)
    with prefill_checkpoint_rows([4],width=4) as request:
        explicit=model(ids,model.make_cache(),logits_to_keep=0)
    assert request.applied==0 and explicit.shape==full.shape
    two=model(ids,model.make_cache(),logits_to_keep=2)
    np.testing.assert_allclose(np.array(two),np.array(full[:,-2:]),atol=1e-6)
    with pytest.raises(ValueError): model(ids,model.make_cache(),logits_to_keep=True)


def test_invalid_logits_request_and_foreign_cache_do_not_advance_state():
    c=config((0,)); w=SmallWeights(c); model=DeepSeekV4Model(c,w,identity='owned')
    ids=mx.array([[1,2]],dtype=mx.int32); state=model.make_cache()
    with pytest.raises(ValueError,match='logits_to_keep'):
        model(ids,state,logits_to_keep=-1)
    assert state[0].offset==0
    other=DeepSeekV4Model(c,w,identity='foreign').make_cache()
    with pytest.raises(ValueError,match='identity'):
        model(ids,other)
    assert other[0].offset==0


def test_dspark_taps_are_uniform_post_ffn_hc_mean_and_keep_all_positions():
    c=config(); w=SmallWeights(c); model=DeepSeekV4Model(c,w,identity='tap-checkpoint')
    ids=mx.array([[1,2,3,4,5]],dtype=mx.int32)
    x=w.embedding(ids).astype(mx.float32)
    h=mx.broadcast_to(x[:,:,None],(*x.shape[:2],c.hc_mult,c.hidden_size))
    manual=[]; state=model.make_cache()
    for i,(layer,cache) in enumerate(zip(model.layers,state)):
        h=layer(h,ids,cache)
        if i in (0,2): manual.append(mx.mean(h,axis=-2))
    expected=mx.concatenate(manual,axis=-1)
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
    with prefill_checkpoint_rows([5],width=5):
        logits,taps,receipt=model.forward_with_taps(ids,model.make_cache(),(0,2),return_metadata=True)
    assert logits.shape==(1,1,c.vocab_size)
    assert taps.shape==(1,5,2*c.hidden_size)
    np.testing.assert_array_equal(np.array(taps),np.array(expected))
    assert receipt['target_identity']=='tap-checkpoint' and receipt['end_position']==5
    none,body=model.forward_with_taps(ids,model.make_cache(),(0,2),body_only=True)
    assert none is None
    np.testing.assert_array_equal(np.array(body),np.array(expected))
    invalid=model.make_cache()
    with pytest.raises(ValueError): model.forward_with_taps(ids,invalid,(2,0))
    assert all(c.offset==0 for c in invalid)


def test_sinkhorn_independent_numpy_equations_include_every_epsilon():
    rng=np.random.default_rng(219)
    z=rng.normal(0,2,(2,3,24)).astype(np.float32)
    scale=np.array([.7,.3,1.2],np.float32); base=rng.normal(0,.2,24).astype(np.float32)
    eps=np.float32(.03)
    pre,post,comb=sinkhorn_mix(mx.array(z),mx.array(scale),mx.array(base),4,5,float(eps))
    p=1/(1+np.exp(-(z[...,:4]*scale[0]+base[:4])))+eps
    po=2/(1+np.exp(-(z[...,4:8]*scale[1]+base[4:8])))
    logits=(z[...,8:]*scale[2]+base[8:]).reshape(2,3,4,4)
    e=np.exp(logits-logits.max(-1,keepdims=True))
    expected=e/e.sum(-1,keepdims=True)+eps
    expected/=expected.sum(-2,keepdims=True)+eps
    for _ in range(4):
        expected/=expected.sum(-1,keepdims=True)+eps
        expected/=expected.sum(-2,keepdims=True)+eps
    np.testing.assert_allclose(np.array(pre),p,atol=1e-6)
    np.testing.assert_allclose(np.array(post),po,atol=1e-6)
    np.testing.assert_allclose(np.array(comb),expected,atol=1e-6)


def test_opt_in_packed_cache_full_model_matches_fp32_qat_cache_cpu():
    c=dataclasses.replace(config((4,128)),head_dim=512,qk_rope_head_dim=64,
                          index_head_dim=128,simulate_quantization=True)
    weights=SmallWeights(c)
    reference=DeepSeekV4Model(c,weights,identity='packed-integration')
    candidate=DeepSeekV4Model(c,weights,identity='packed-integration',cache_policy='fp8_mixed')
    assert candidate.identity!=reference.identity
    ids=mx.array(np.arange(137)[None]%c.vocab_size,dtype=mx.int32)
    expected=reference(ids,reference.make_cache())
    actual=candidate(ids,candidate.make_cache())
    np.testing.assert_array_equal(np.array(actual),np.array(expected))
    # Compare identical serving shapes. Baseline QAT full-vs-chunk sensitivity
    # is a separate numerical diagnostic, not a physical-storage difference.
    state=candidate.make_cache(); base_state=reference.make_cache()
    chunks=[(0,3),(3,129),(129,137)]
    packed_chunks=mx.concatenate([candidate(ids[:,a:b],state) for a,b in chunks],axis=1)
    base_chunks=mx.concatenate([reference(ids[:,a:b],base_state) for a,b in chunks],axis=1)
    np.testing.assert_array_equal(np.array(packed_chunks),np.array(base_chunks))
    assert all(x.offset==137 for x in state)
    assert state[0].compressor.pooled.shape[1]==34
    assert state[1].compressor.pooled.shape[1]==1


def test_expert_prefill_policy_only_scopes_explicit_prompt_entries():
    from mlx2.runtime.gguf.segmented_prefill import segmented_prefill_active
    class ObservedWeights(SmallWeights):
        def __init__(self,c):
            super().__init__(c)
            self.scopes=[]
        def moe(self,x,ids,scores,prefix):
            self.scopes.append(segmented_prefill_active(x.shape[0]))
            return super().moe(x,ids,scores,prefix)
    c=config((0,)); weights=ObservedWeights(c)
    model=DeepSeekV4Model(c,weights,identity='prefill-scope',expert_prefill_policy='segmented_nax_fp16')
    reference=DeepSeekV4Model(c,weights,identity='prefill-scope')
    assert model.identity!=reference.identity
    ids=mx.array(np.arange(16)[None]%c.vocab_size,dtype=mx.int32)
    model(ids,model.make_cache())
    model.prefill_forward(ids,model.make_cache())
    model.prefill_body(ids,model.make_cache(),(0,))
    model.forward_with_taps(ids,model.make_cache(),(0,))
    model.prefill_forward(ids[:,:8],model.make_cache())
    assert weights.scopes==[False,True,True,False,False]
    assert not segmented_prefill_active(16)
    with pytest.raises(ValueError,match='cache identity'):
        model(ids,reference.make_cache())
    with pytest.raises(ValueError,match='expert prefill'):
        DeepSeekV4Model(c,weights,expert_prefill_policy='implicit')
