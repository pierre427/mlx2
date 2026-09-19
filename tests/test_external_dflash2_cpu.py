"""Tiny tensor tests force CPU before model construction; no artifact loads."""
import copy
import json
import struct
from pathlib import Path
from typing import ClassVar

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.drafters.dflash2_config import DFlash2Config
from mlx2.runtime.external_speculative import (
    ExternalDraftBatchGenerator,
    ExternalDraftState,
)
from mlx2.runtime.models.muse_glimmer import Model
from mlx2.runtime.speculative_sampling import RequestRNG, verify_proposals


def _write_safetensors_headers(path, tensors, *, dtype="BF16"):
    item_size = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
    offset = 0
    header = {}
    for name, shape in tensors.items():
        count = 1
        for value in shape:
            count *= value
        end = offset + count * item_size
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, end]}
        offset = end
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return header


def _write_safetensors_header_records(path, header):
    raw = json.dumps(header, separators=(",", ":")).encode()
    size = max((record["data_offsets"][1] for record in header.values()), default=0)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(size))


def _tiny_dflash_args():
    return DFlash2Config(hidden_size=8,intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=4,vocab_size=32,num_target_layers=4,target_layer_ids=[0,3],conv_kernel_size=2,conv_group_size=2,selector_rank=4,selector_top_k=4,block_size=4,mask_token_id=31,max_position_embeddings=128,sliding_window=3,layer_types=['sliding_attention']*2)


def tiny():
    mx.random.seed(8)
    m=Model(ModelArgs(hidden_size=8,intermediate_size=16,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=4,vocab_size=32,sliding_window=3,max_position_embeddings=128))
    d=DFlash2DraftModel(_tiny_dflash_args()).bind(m)
    return m,d


def generator(m,d,**kwargs):
    return ExternalDraftBatchGenerator(m,draft_model=d,binding='test',num_draft=2,prefill_step_size=3,**kwargs)


def test_external_insert_accepts_shared_serving_seam_and_rejects_unsupported_inputs():
    m, d = tiny()
    compatible = generator(m, d)
    try:
        uids = compatible.insert(
            [[1, 2]],
            max_tokens=[1],
            apc_interior_positions=[()],
            prefill_inputs=[None],
        )
        assert len(uids) == 1
    finally:
        compatible.close()

    with pytest.raises(ValueError, match="interior checkpoints"):
        generator(m, d).insert(
            [[1, 2]], max_tokens=[1], apc_interior_positions=[(1,)]
        )
    with pytest.raises(ValueError, match="multimodal prefill"):
        generator(m, d).insert(
            [[1, 2]], max_tokens=[1], prefill_inputs=[{"pixel_values": []}]
        )


def test_real_dflash2_header_schema_reconciles_without_payload_load():
    from mlx2.adapters.dflash2 import inspect_drafter

    if not (Path.home() / "mlx-models/Muse-Glimmer-30B-DFlash2").exists():
        pytest.skip("local qualification artifact is not installed")
    record = inspect_drafter(
        str(Path.home() / "mlx-models/Muse-Glimmer-30B-DFlash2"),
        str(Path.home() / "mlx-models/Muse-Glimmer-30B-mlx-4bit"),
    )
    assert len(record["header_sha256"]) == 1
    assert record["header_sha256"][0] == "9e37d992653b60ea5c75714e75b66115c4443c626ce1fd5152b6bad522807292"
    assert len(record["files"]) == 1 and record["files"][0][1] == 5_544_328_424


def test_dflash2_header_schema_fails_before_payload_load(tmp_path):
    from mlx2.adapters.dflash2 import _expected_weight_shapes, inspect_drafter

    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir(); draft.mkdir()
    target_config = {
        "model_type": "muse_glimmer",
        "hidden_size": 8,
        "vocab_size": 32,
        "num_hidden_layers": 4,
    }
    draft_config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "dtype": "bfloat16",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "max_position_embeddings": 128,
        "sliding_window": 3,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "num_target_layers": 4,
        "dflash_config": {
            "block_size": 4,
            "mask_token_id": 31,
            "target_layer_ids": [0, 3],
            "conv_kernel_size": 2,
            "conv_group_size": 2,
            "selector_rank": 4,
            "selector_top_k": 4,
        },
    }
    (target / "config.json").write_text(json.dumps(target_config))
    (draft / "config.json").write_text(json.dumps(draft_config))
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config
    shapes = _expected_weight_shapes(DFlash2Config.from_dict(draft_config))
    shapes["fc.weight"] = [8, 8]
    _write_safetensors_headers(draft / "model.safetensors", shapes)
    with pytest.raises(ValueError, match="wrong_shapes"):
        inspect_drafter(draft, target)


def test_safetensors_header_and_index_duplicate_json_keys_fail_closed(tmp_path):
    from mlx2.adapters.dflash2 import _decode_unique_json, _read_safetensors_header

    record='{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}'
    raw=(f'{{"same":{record},"same":{record}}}').encode()
    path=tmp_path/'model.safetensors'
    path.write_bytes(struct.pack('<Q',len(raw))+raw+b'\0\0')
    with pytest.raises(ValueError,match='Duplicate JSON key'):
        _read_safetensors_header(path)
    with pytest.raises(ValueError,match='Duplicate JSON key'):
        _decode_unique_json(b'{"weight_map":{"x":"a","x":"b"}}','draft weight index')


@pytest.mark.parametrize(('mutation','error'),[
    ('wrong_dtype','dtype/shape mismatch'),
    ('overlap','Overlapping'),
    ('index_mismatch','index/shard mismatch'),
    ('missing','schema mismatch'),
    ('extra','schema mismatch'),
])
def test_safetensors_header_schema_negative_cases(tmp_path, mutation, error):
    from mlx2.adapters.dflash2 import _expected_weight_shapes, _validate_weight_headers

    args=_tiny_dflash_args();shapes=_expected_weight_shapes(args)
    if mutation=='missing':
        shapes.pop(next(iter(shapes)))
    elif mutation=='extra':
        shapes['unexpected.weight']=[1]
    path=tmp_path/'model.safetensors'
    header=_write_safetensors_headers(path,shapes,dtype='F16' if mutation=='wrong_dtype' else 'BF16')
    if mutation=='overlap':
        names=list(header)
        first,second=header[names[0]],header[names[1]]
        length=second['data_offsets'][1]-second['data_offsets'][0]
        second['data_offsets']=[first['data_offsets'][0],first['data_offsets'][0]+length]
        _write_safetensors_header_records(path,header)
    mapping={name:path.name for name in shapes}
    if mutation=='index_mismatch':
        mapping[next(iter(mapping))]='wrong.safetensors'
    with pytest.raises(ValueError,match=error):
        _validate_weight_headers([path],mapping,args,'bfloat16')


def drain(b):
    output={}; final={}
    for _ in range(100):
        _,responses=b.next()
        for r in responses:
            output.setdefault(r.uid,[]).append(r.token)
            if r.finish_reason:final[r.uid]=r
        if not b.lanes:return output,final
    raise AssertionError('scheduler stalled')


def test_body_taps_match_normal_forward():
    m,_d=tiny();x=mx.array([[1,2,3]])
    logits,taps=m.forward_with_taps(x,m.make_cache(),[0,3])
    ordinary=m(x,cache=m.make_cache())
    np.testing.assert_allclose(np.asarray(logits),np.asarray(ordinary),atol=1e-6)
    assert taps.shape==(1,3,16)
    np.testing.assert_allclose(np.asarray(taps),np.asarray(m.prefill_body(x,m.make_cache(),[0,3])),atol=1e-6)


def test_batched_variable_length_greedy_matches_reference_and_pairs():
    m,d=tiny();b=generator(m,d)
    prompts=[[1,2,3,4,5],[1,2]]
    ids=b.insert(prompts,max_tokens=[7,7],sampling_configs=[{'sampling_temp':0}]*2)
    got,final=drain(b)
    for uid,prompt in zip(ids,prompts):
        cache=m.make_cache();tokens=list(prompt);reference=[]
        for i in range(7):
            logits=m(mx.array([tokens if i==0 else [tokens[-1]]]),cache=cache)
            token=int(mx.argmax(logits[0,-1]).item());tokens.append(token);reference.append(token)
        assert got[uid]==reference
        state=final[uid].cache_sidecar;state.validate('test',len(final[uid].all_tokens))
    assert b.scheduler_stats['target_max_width']==2


def test_exact_residual_distribution_and_rng_restore():
    p=np.array([.7,.2,.1]);q=np.array([.1,.2,.7]);rng=RequestRNG(123)
    counts=np.zeros(3)
    for _ in range(12000):
        token=rng.sample(q)
        result=verify_proposals([token],[q],[p,p],rng)
        counts[result.emitted[0]]+=1
    np.testing.assert_allclose(counts/counts.sum(),p,atol=.013)
    state=rng.snapshot();clone=RequestRNG(state=state)
    assert [rng.uniform() for _ in range(20)]==[clone.uniform() for _ in range(20)]


def test_selector_exports_actual_nonzero_proposal_law():
    m,d=tiny();cache=m.make_cache();h=m.prefill_body(mx.array([[1,2,3]]),cache,[0,3]);rng=RequestRNG(4)
    tokens,q=d.draft_distributions([4],h,d.make_cache(),2,[rng],[.8])
    for t,law in zip(tokens[0],q[0]):
        assert law[t]>0 and abs(law.sum()-1)<1e-9
        assert np.count_nonzero(law)==4


def test_external_draft_processors_sample_and_verify_with_masked_q(monkeypatch):
    import mlx2.runtime.external_speculative as module

    m,d=tiny();b=generator(m,d)
    captured={}
    original=module.verify_proposals

    def ban_selector_argmax(tokens, value):
        hosted=np.asarray(value[0],dtype=np.float32)
        banned=int(np.argmax(hosted))
        mask=np.zeros(hosted.shape[-1],dtype=np.float32);mask[banned]=-np.inf
        return value+mx.array(mask)[None]

    def inspect(tokens, proposals, targets, rng, **kwargs):
        captured['tokens']=list(tokens);captured['proposals']=proposals
        return original(tokens,proposals,targets,rng,**kwargs)

    monkeypatch.setattr(module,'verify_proposals',inspect)
    b.insert([[1,2,3]],max_tokens=[4],logits_processors=[[ban_selector_argmax]],sampling_configs=[{'sampling_temp':.8}])
    b.next()
    assert len(captured['tokens'])==2
    for token,law in zip(captured['tokens'],captured['proposals']):
        assert law[token]>0
        assert np.count_nonzero(law)==d.config.selector_top_k-1
    assert b.scheduler_stats['external_draft_masked_positions']==2
    b.close()


def test_external_masking_preserves_greedy_target_output():
    m,d=tiny();prompt=[1,2,3];maximum=8
    plain=generator(m,d);plain.insert([prompt],max_tokens=[maximum])
    expected,_=drain(plain)
    banned=next(token for token in range(32) if token not in expected[0])

    def processor(_tokens,value):
        mask=np.zeros(value.shape[-1],dtype=np.float32);mask[banned]=-np.inf
        return value+mx.array(mask)

    constrained=generator(m,d)
    constrained.insert([prompt],max_tokens=[maximum],logits_processors=[[processor]])
    actual,_=drain(constrained)
    assert actual[0]==expected[0]
    assert constrained.scheduler_stats['external_draft_masked_positions']>0


def test_external_draft_processor_probe_never_mutates_persistent_state():
    from mlx2.structured_output import StructuredOutputProcessor

    class Tokenizer:
        eos_token_ids = (0,)
        vocab_size = 32

        def decode(self, tokens, **_kwargs):
            return "".join(chr(65 + token) for token in tokens if token != 0)

    class OnlyB:
        pattern = None

        @staticmethod
        def canonicalize(value):
            return value

        @staticmethod
        def fullmatch(value, *, partial=False, timeout=None):
            del timeout
            valid = all(char == "B" for char in value)
            return object() if valid and (partial or value) else None

    m,d=tiny();processor=StructuredOutputProcessor(Tokenizer(),3,OnlyB())
    b=generator(m,d);uid=b.insert([[1,2,3]],max_tokens=[4],logits_processors=[[processor]])[0]
    while b.lanes[uid].anchor is None:
        b._prefill(b.lanes[uid])
    lane=b.lanes[uid]
    before=(processor.failure,processor.constrained_steps,processor.tail_mass_bound,
            dict(processor._allowed_cache),dict(processor._partial_cache))
    d.draft_distributions(
        [lane.anchor],lane.tail,d.batch_caches([lane.draft_cache]),2,
        [lane.rng],[0.0],logits_processors=[[processor]],
        processor_histories=[list(lane.history)],
    )
    after=(processor.failure,processor.constrained_steps,processor.tail_mass_bound,
           dict(processor._allowed_cache),dict(processor._partial_cache))
    assert after==before
    b.close()


def test_external_draft_min_tokens_and_grammar_complete_on_cpu():
    from mlx2.serving import minimum_tokens_processor
    from mlx2.structured_output import StructuredOutputProcessor

    class Tokenizer:
        eos_token_ids = (0,)
        vocab_size = 32

        def decode(self, tokens, **_kwargs):
            return "".join(chr(65 + int(token)) for token in tokens if int(token) != 0)

    class Bs:
        pattern = None

        @staticmethod
        def canonicalize(value):
            return value

        @staticmethod
        def fullmatch(value, *, partial=False, timeout=None):
            del timeout
            valid = all(char == "B" for char in value)
            return object() if valid and (partial or len(value) >= 2) else None

    prompt = [1, 2, 3]
    processors = [
        minimum_tokens_processor(mx, [0], len(prompt), 2),
        StructuredOutputProcessor(Tokenizer(), len(prompt), Bs()),
    ]
    model, draft = tiny()
    batch = generator(model, draft, stop_tokens=[[0]])
    batch.insert(
        [prompt],
        max_tokens=[6],
        logits_processors=[processors],
        sampling_configs=[{"sampling_temp": 0.0}],
    )
    output, final = drain(batch)
    assert final[0].finish_reason == "stop"
    assert output[0][:2] == [1, 1]
    assert output[0][-1] == 0


def test_no_legal_dflash_candidate_truncates_round_without_lane_demotion():
    from mlx2.structured_output import StructuredOutputProcessor

    class Tokenizer:
        eos_token_ids = (0,)
        vocab_size = 32

        def decode(self, tokens, **_kwargs):
            return "".join(chr(65 + token) for token in tokens if token != 0)

    class OnlyB:
        pattern = None

        @staticmethod
        def canonicalize(value):
            return value

        @staticmethod
        def fullmatch(value, *, partial=False, timeout=None):
            del timeout
            valid = all(char == "B" for char in value)
            return object() if valid and (partial or value) else None

    m,d=tiny();processor=StructuredOutputProcessor(Tokenizer(),3,OnlyB())
    b=generator(m,d)
    b.insert([[1,2,3]],max_tokens=[6],logits_processors=[[processor]],
             sampling_configs=[{"sampling_temp":0}])
    got,final=drain(b)
    # EOS decodes to an empty string and remains legal after the first B.
    assert got[0][0]==1 and set(got[0])<={0,1}
    assert processor.failure is None
    assert b.scheduler_stats['draft_fallbacks']==0
    assert final[0].speculative_receipt['ordinary_fallback'] is False


def test_external_fly_receipt_counter_and_structured_disable(monkeypatch):
    import mlx2.runtime.external_speculative as module
    from mlx2.runtime.speculative_sampling import VerifiedBlock

    m,d=tiny()
    policy={"enabled":True,"entropy_threshold":0.5,"window":1,"min_prob":0.01}
    original=module.verify_proposals

    def relaxed(tokens,proposals,targets,rng,**kwargs):
        assert kwargs['fly_verification'].enabled
        if tokens:
            return VerifiedBlock(
                len(tokens),
                tuple(tokens)+(0,),
                tuple(targets),
                False,
                1,
            )
        return original(tokens,proposals,targets,rng,**kwargs)

    monkeypatch.setattr(module,'verify_proposals',relaxed)
    enabled=generator(m,d,fly_verification=policy)
    enabled.insert([[1,2,3]],max_tokens=[3])
    _,final=drain(enabled);receipt=final[0].speculative_receipt
    assert receipt['verification']=='fly'
    assert receipt['parameters']==policy
    assert receipt['relaxed_accepts']==1
    assert enabled.scheduler_stats['fly_relaxed_accepts']==1

    seen=[]
    def exact_for_structured(tokens,proposals,targets,rng,**kwargs):
        seen.append(dict(kwargs))
        return original(tokens,proposals,targets,rng,**kwargs)
    monkeypatch.setattr(module,'verify_proposals',exact_for_structured)
    structured=generator(m,d,fly_verification=policy)
    structured.insert([[1,2,3]],max_tokens=[1],logits_processors=[[lambda _y,x:x]])
    _,final=drain(structured);receipt=final[0].speculative_receipt
    assert seen==[{}]
    assert receipt['verification']=='exact'
    assert receipt['fly_disabled']=='logits_processors'
    assert structured.scheduler_stats['fly_relaxed_accepts']==0


def test_apcv2_disk_pairs_external_revision_and_tail(tmp_path, monkeypatch):
    from mlx2.runtime.apc_v2 import APCKey, APCv2
    monkeypatch.setattr(mx,'clear_cache',lambda:None)
    m,d=tiny();b=generator(m,d);b.insert([[1,2,3,4,5]],max_tokens=[4])
    _,final=drain(b);end=final[0]
    now=[10.]
    apc=APCv2(layout_name='test-external',idle_disk_seconds=1,idle_disk_dir=str(tmp_path),now_fn=lambda:now[0])
    key=APCKey('target+draft',revision='revision-A',cache_layout_fingerprint='test-external')
    apc.store(key,end.all_tokens,end.prompt_cache,sidecar=end.cache_sidecar)
    now[0]+=3
    assert apc.spill_idle_entries()==1
    hit=apc.lookup(key,end.all_tokens+[end.token])
    assert hit.hit and isinstance(hit.sidecar,ExternalDraftState)
    hit.sidecar.validate('test',len(end.all_tokens))
    assert hit.sidecar.kind=='external_draft_v1'
    np.testing.assert_array_equal(np.asarray(hit.sidecar.rng_key),np.asarray(end.cache_sidecar.rng_key))
    with pytest.raises(ValueError,match='revision'):
        hit.sidecar.validate('another-target',len(end.all_tokens))
    assert not apc.lookup(APCKey('target+draft',revision='revision-B',cache_layout_fingerprint='test-external'),end.all_tokens+[end.token]).hit
    if hasattr(hit.cache,'close'):hit.cache.close()
    apc.clear(release_memory=False)


def test_apcv2_cow_external_sidecar_exact_reuse_is_descriptor_safe(monkeypatch):
    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.cow_cache import COWPromptCacheBranch

    m, d = tiny()
    initial = generator(m, d)
    initial.insert([[1, 2, 3, 4, 5]], max_tokens=[4])
    _, final = drain(initial)
    end = final[0]

    apc = APCv2(max_size=2, layout_name="test-external-cow")
    key = APCKey(
        "target+draft",
        revision="revision-A",
        cache_layout_fingerprint="test-external-cow",
    )
    apc.store(
        key,
        end.all_tokens,
        end.prompt_cache,
        sidecar=end.cache_sidecar,
    )

    hit = apc.lookup(key, end.all_tokens + [end.token])
    assert hit.hit and hit.hit_kind == "external_draft_sidecar"
    assert isinstance(hit.cache, COWPromptCacheBranch)
    assert apc.apc_stats["cow"]["active_leases"] == 1

    resumed = generator(m, d)
    uid = resumed.insert(
        [[end.token]],
        max_tokens=[2],
        caches=[hit.cache],
        all_tokens=[end.all_tokens],
        cache_states=[hit.sidecar],
    )[0]
    lane = resumed.lanes[uid]
    for expected, actual in zip(hit.sidecar.state[0], lane.draft_cache):
        assert actual.offset == expected.offset
        for expected_array, actual_array in zip(expected.state, actual.state):
            np.testing.assert_array_equal(
                np.asarray(actual_array), np.asarray(expected_array)
            )
    np.testing.assert_array_equal(
        np.asarray(lane.tail), np.asarray(hit.sidecar.state[1])
    )
    outgoing = resumed._sidecar(lane)
    outgoing.validate("test", len(end.all_tokens))
    assert resumed.scheduler_stats["paired_cache_resumes"] == 1
    assert all(
        not hasattr(cache, "_cow_segment_tokens")
        for cache in outgoing.state[0]
    )

    prompt = resumed._prefill(lane)
    assert prompt.end_of_prompt
    boundary = resumed.pop_prompt_boundary(uid)
    assert boundary["covered_tokens"] == len(end.all_tokens)
    assert all(
        not hasattr(cache, "_cow_segment_tokens")
        for cache in boundary["target_cache"]
    )

    before = copy.deepcopy(lane.__dict__)
    real_forward = resumed.model.forward_with_taps

    def fail_forward(*args, **kwargs):
        raise RuntimeError("injected target failure")

    monkeypatch.setattr(resumed.model, "forward_with_taps", fail_forward)
    with pytest.raises(RuntimeError, match="injected target failure"):
        resumed._round([lane])
    assert lane.history == before["history"]
    assert lane.anchor == before["anchor"]
    assert lane.generated == before["generated"]
    assert lane.rng.snapshot() == before["rng"].snapshot()
    monkeypatch.setattr(resumed.model, "forward_with_taps", real_forward)

    output, completed = drain(resumed)
    assert len(output[uid]) == 2
    finish = completed[uid]
    finish.cache_sidecar.validate("test", len(finish.all_tokens))
    assert all(
        not hasattr(cache, "_cow_segment_tokens")
        for cache in finish.prompt_cache
    )

    removable = generator(m, d)
    remove_uid = removable.insert(
        [[end.token]],
        max_tokens=[2],
        caches=[hit.cache],
        all_tokens=[end.all_tokens],
        cache_states=[hit.sidecar],
    )[0]
    returned = removable.remove([remove_uid], return_prompt_caches=True)
    assert remove_uid in returned
    assert all(
        not hasattr(cache, "_cow_segment_tokens")
        for cache in returned[remove_uid]
    )

    hit.cache.close()
    assert apc.apc_stats["cow"]["active_leases"] == 0
    apc.clear(release_memory=False)


def test_fresh_seed_not_overridden_by_apc_and_explicit_resume_restores():
    from mlx2.runtime.sample_utils import LaneRNG
    m,d=tiny();b=generator(m,d);b.insert([[1,2,3]],max_tokens=[4],sampling_configs=[{'sampling_temp':.8}],lane_rngs=[LaneRNG(7)])
    _,final=drain(b);end=final[0]
    def restored(resume):
        g=generator(m,d);seed=LaneRNG(99)
        uid=g.insert([[end.token]],max_tokens=[2],caches=[copy.deepcopy(end.prompt_cache)],all_tokens=[end.all_tokens],cache_states=[end.cache_sidecar],lane_rngs=[seed],resume_rng=resume)[0]
        assert g.scheduler_stats['paired_cache_resumes']==1
        return g.lanes[uid].rng.snapshot(),RequestRNG(np.asarray(seed.key).tolist()).snapshot()
    fresh,expected=restored(False);resume,_=restored(True)
    assert fresh==expected
    assert resume==json.loads(bytes(np.asarray(end.cache_sidecar.rng_key,dtype=np.uint8)).decode())
    assert fresh!=resume


def test_atomic_failure_after_one_row_publication_restores_every_lane(monkeypatch):
    m,d=tiny();b=generator(m,d);b.insert([[1,2],[1,3]],max_tokens=[1,1])
    for lane in b.lanes.values():b._prefill(lane)
    cohort=list(b.lanes.values());before=copy.deepcopy([l.__dict__ for l in cohort])
    real=b._sidecar;calls=[0]
    def fail_second(lane):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('injected publication failure')
        return real(lane)
    monkeypatch.setattr(b,'_sidecar',fail_second)
    with pytest.raises(RuntimeError,match='injected'):b._round(cohort)
    assert b.scheduler_stats['recovery_checkpoint_captures']==2
    assert b.scheduler_stats['recovery_checkpoint_restores']==2
    for lane,old in zip(cohort,before):
        assert lane.history==old['history'] and lane.anchor==old['anchor'] and lane.generated==old['generated']
        assert not lane.ready and lane.rng.snapshot()==old['rng'].snapshot()
        for c,previous in zip(lane.cache,old['cache']):
            assert c.offset==previous.offset
            for a,v in zip(c.state,previous.state):np.testing.assert_array_equal(np.asarray(a),np.asarray(v))
    monkeypatch.setattr(b,'_sidecar',real)
    output,_=drain(b);assert all(len(v)==1 for v in output.values())


def test_drafter_failure_falls_back_ordinary_and_rng_membership_independent(monkeypatch):
    from mlx2.runtime.external_speculative import DraftUnavailable
    from mlx2.runtime.sample_utils import LaneRNG
    m,d=tiny();b=generator(m,d)
    assert b.scheduler_stats['draft_fallbacks']==0
    def unavailable(*args,**kwargs):raise DraftUnavailable('candidate unavailable')
    monkeypatch.setattr(d,'draft_distributions',unavailable)
    b.insert([[1,2,3],[3,4]],max_tokens=[5,5],lane_rngs=[LaneRNG(4),LaneRNG(5)])
    got,_=drain(b)
    assert len(got[0])==5 and b.scheduler_stats['draft_fallbacks']>=1
    assert b.scheduler_stats['ordinary_rounds']>0


def test_permanent_ordinary_fast_path_skips_draft_taps_transactions_and_sidecar(monkeypatch):
    from mlx2.runtime.external_speculative import DraftUnavailable

    m,d=tiny();prompt=[1,2,3]
    cache=m.make_cache();tokens=list(prompt);reference=[]
    for index in range(4):
        logits=m(mx.array([tokens if index==0 else [tokens[-1]]]),cache=cache)
        token=int(mx.argmax(logits[0,-1]).item());tokens.append(token);reference.append(token)

    b=generator(m,d);uid=b.insert([prompt],max_tokens=[4])[0]
    while b.lanes[uid].anchor is None:b._prefill(b.lanes[uid])
    monkeypatch.setattr(d,'draft_distributions',lambda *a,**k: (_ for _ in ()).throw(DraftUnavailable('offline')))
    monkeypatch.setattr(d,'append_context',lambda *a,**k: (_ for _ in ()).throw(AssertionError('draft context must be skipped')))
    monkeypatch.setattr(m,'forward_with_taps',lambda *a,**k: (_ for _ in ()).throw(AssertionError('target taps must be skipped')))
    got,final=drain(b)
    assert got[uid]==reference
    assert final[uid].cache_sidecar is None
    assert b.scheduler_stats['external_ordinary_fast_path_rounds']==4
    assert b.scheduler_stats['external_ordinary_fast_path_lanes']==4
    assert b.scheduler_stats['external_draft_context_skipped']==4
    assert b.scheduler_stats['external_taps_skipped']==4
    assert b.scheduler_stats['external_transactions_skipped']==4
    assert b.scheduler_stats['segmented_transactions']==0


def test_permanent_ordinary_lanes_share_one_batched_target_forward(monkeypatch):
    m,d=tiny();b=generator(m,d)
    uids=b.insert([[1,2,3],[3,4,5]],max_tokens=[4,4])
    for uid in uids:
        while b.lanes[uid].anchor is None:
            b._prefill(b.lanes[uid])
        b.disable_speculation(uid)
    widths=[];real=type(m).__call__

    def counted(model, inputs, *args, **kwargs):
        widths.append(int(inputs.shape[0]))
        return real(model, inputs, *args, **kwargs)

    monkeypatch.setattr(type(m),'__call__',counted)
    _,final=drain(b)
    assert widths==[2,2,2,2]
    assert all(response.execution_width==2 for response in final.values())
    assert all(response.all_tokens is not None for response in final.values())
    assert b.scheduler_stats['target_max_width']==2


def test_final_ordinary_budget_round_retains_request_level_dflash_evidence(monkeypatch):
    import mlx2.runtime.external_speculative as module
    from mlx2.runtime.speculative_sampling import VerifiedBlock

    m,d=tiny();b=generator(m,d)
    original=module.verify_proposals
    def reject_one(tokens, proposals, targets, rng):
        if tokens:
            return VerifiedBlock(0,(5,),(targets[0],),True)
        return original(tokens,proposals,targets,rng)
    monkeypatch.setattr(module,'verify_proposals',reject_one)
    b.insert([[1,2,3]],max_tokens=[3])
    _,final=drain(b);receipt=final[0].speculative_receipt
    assert receipt['current_execution']=='ordinary_target'
    assert receipt['execution']=='external_draft_verify'
    assert receipt['external_rounds']==2 and receipt['proposed']==3
    assert receipt['round_proposed']==0 and receipt['accepted']==0
    assert receipt['target_width']==1 and receipt['draft_width']==1
    assert receipt['ordinary_fallback'] is False
    assert final[0].cache_sidecar is not None
    assert b.scheduler_stats['external_ordinary_fast_path_rounds']==0
    assert b.scheduler_stats['segmented_transactions']==3
    assert b.scheduler_stats['segmented_rollbacks']==2


def test_cancel_discards_future_block_and_membership_rejoins():
    m,d=tiny();b=generator(m,d)
    first=b.insert([[1,2,3],[3,4]],max_tokens=[20,5])
    b.next();b.remove([first[0]])
    third=b.insert([[2,3]],max_tokens=[4])[0]
    got,_=drain(b)
    assert first[0] not in got and len(got[third])==4
    assert not b.boundaries


def test_memory_defers_without_mutating_or_drawing():
    m,d=tiny();b=generator(m,d,memory_headroom=lambda:0)
    uid=b.insert([[1,2,3]],max_tokens=[4])[0];state=b.lanes[uid].rng.snapshot()
    assert b.next()==([],[])
    assert b.lanes[uid].rng.snapshot()==state and b.lanes[uid].history==[]
    assert b.scheduler_stats['memory_deferred']==1


def test_selector_batched_projection_and_single_lane_agree():
    m,d=tiny();taps=[m.prefill_body(mx.array([[1,2]]),m.make_cache(),[0,3]),m.prefill_body(mx.array([[4,3]]),m.make_cache(),[0,3])]
    batched=d.draft_distributions([3,2],mx.concatenate(taps),d.batch_caches([d.make_cache(),d.make_cache()]),2,[RequestRNG(4),RequestRNG(8)],[.8,.8])
    for row,seed in enumerate([4,8]):
        single=d.draft_distributions([[3,2][row]],taps[row],d.make_cache(),2,[RequestRNG(seed)],[.8])
        assert batched[0][row]==single[0][0]
        np.testing.assert_allclose(batched[1][row],single[1][0],atol=1e-6)


def test_sampled_lane_seed_is_independent_of_short_neighbor_and_cancellation():
    from mlx2.runtime.sample_utils import LaneRNG
    m,d=tiny()
    def run(neighbor):
        b=generator(m,d)
        b.insert([[1,2,3,4]],max_tokens=[11],sampling_configs=[{'sampling_temp':.8,'top_k':12,'top_p':.9}],lane_rngs=[LaneRNG(123)])
        if neighbor:b.insert([[4,3]],max_tokens=[1],sampling_configs=[{'sampling_temp':.6}],lane_rngs=[LaneRNG(321)])
        return drain(b)[0][0]
    assert run(False)==run(True)


def test_already_closed_commit_failure_restores_lane_and_original_error(monkeypatch):
    from mlx2.runtime.segmented_rotating_kv import SegmentedKVTransaction
    m,d=tiny();b=generator(m,d);b.insert([[1,2,3]],max_tokens=[5])
    lane=b.lanes[0];b._prefill(lane);before=copy.deepcopy(lane.__dict__)
    real=SegmentedKVTransaction.commit
    def fail_after_commit(self,*args,**kwargs):
        real(self,*args,**kwargs)
        raise RuntimeError('injected closed transaction')
    monkeypatch.setattr(SegmentedKVTransaction,'commit',fail_after_commit)
    with pytest.raises(RuntimeError,match='injected closed transaction'):
        b._round([lane])
    assert lane.rng.snapshot()==before['rng'].snapshot()
    assert lane.history==before['history'] and lane.generated==0 and not lane.ready
    for now,old in zip(lane.cache,before['cache']):
        assert now.offset==old.offset
        for a,v in zip(now.state,old.state):np.testing.assert_array_equal(np.asarray(a),np.asarray(v))


def test_snapshot_failure_does_not_leak_transaction_lock(monkeypatch):
    import mlx2.runtime.external_speculative as module
    m,d=tiny();b=generator(m,d);b.insert([[1,2]],max_tokens=[3]);lane=b.lanes[0];b._prefill(lane)
    def fail(value):raise MemoryError('injected snapshot failure')
    monkeypatch.setattr(module.copy,'deepcopy',fail)
    with pytest.raises(MemoryError):b._round([lane])
    assert not b._open
    b.close();assert not b.lanes


def test_stop_inside_accepted_block_commits_only_prefix(monkeypatch):
    m,d=tiny();prompt=[1,2,3];cache=m.make_cache();tokens=list(prompt);expected=[]
    for i in range(3):
        logits=m(mx.array([tokens if i==0 else [tokens[-1]]]),cache=cache)
        token=int(mx.argmax(logits[0,-1]).item());expected.append(token);tokens.append(token)
    real=d.draft_distributions
    def exact(anchors,hidden,caches,count,rngs,temps):
        # Preserve the real context update, replacing only the proposal law.
        real(anchors,hidden,caches,count,rngs,temps)
        laws=[]
        for token in expected[:count]:
            law=np.zeros(32);law[token]=1; laws.append(law)
        return [expected[:count]],[laws]
    monkeypatch.setattr(d,'draft_distributions',exact)
    b=generator(m,d,stop_tokens=[[expected[0]]]);b.insert([prompt],max_tokens=[10])
    got,final=drain(b)
    assert got[0]==[expected[0]] and final[0].finish_reason=='stop'
    assert final[0].all_tokens==prompt
    assert all(c.offset==len(prompt) for c in final[0].prompt_cache)
    final[0].cache_sidecar.validate('test',len(prompt))
    assert b.scheduler_stats['cancelled']==0


def test_cli_keeps_ordinary_reference_distinct_before_allocation():
    from mlx2.server import build_parser, native_mtp_mode
    p=build_parser();base=['--model','not-loaded']
    assert native_mtp_mode(p.parse_args(base),None)
    assert not native_mtp_mode(p.parse_args(base+['--ordinary']),None)
    external={'draft_model':'not-loaded-drafter'}
    assert not native_mtp_mode(p.parse_args(base+['--external-draft']),external)
    with pytest.raises(ValueError,match='ordinary'):native_mtp_mode(p.parse_args(base+['--ordinary']),external)
    with pytest.raises(ValueError,match='requires'):native_mtp_mode(p.parse_args(base+['--external-draft']),None)
    with pytest.raises(ValueError,match='requires'):native_mtp_mode(p.parse_args(base),external)
    with pytest.raises(SystemExit):p.parse_args(base+['--ordinary','--external-draft'])


def test_one_token_prompt_final_sidecar_is_disk_serializable(tmp_path):
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache
    m,d=tiny();b=generator(m,d);b.insert([[1]],max_tokens=[1])
    _,final=drain(b);state=final[0].cache_sidecar
    state.validate('test',1)
    path=str(tmp_path/'empty-draft.safetensors')
    save_prompt_cache(path,state.state[0]);restored=load_prompt_cache(path)
    assert all(c.offset==0 for c in restored)


def test_empty_serialized_draft_cache_never_promotes_bfloat16_first_append():
    from mlx2.runtime.drafters.dflash_base import append_context_kv
    _m,d=tiny()
    for length in (1,3):
        cache=d.make_cache()[0]
        empty=mx.zeros((1,1,0,4),dtype=mx.bfloat16)
        keys,values=append_context_kv(cache,empty,empty)
        assert keys.dtype==mx.bfloat16 and values.dtype==mx.bfloat16
        real=mx.ones((1,1,length,4),dtype=mx.bfloat16)
        keys,values=append_context_kv(cache,real,real)
        assert keys.dtype==mx.bfloat16 and cache.keys.dtype==mx.bfloat16
        assert values.dtype==mx.bfloat16 and cache.values.dtype==mx.bfloat16


@pytest.mark.parametrize('prompt',[[1],[1,2,3]])
def test_bfloat16_tiny_execution_keeps_real_draft_kv_bfloat16(prompt):
    m,d=tiny();m.set_dtype(mx.bfloat16);d.set_dtype(mx.bfloat16)
    b=generator(m,d);b.insert([prompt],max_tokens=[4])
    _,final=drain(b)
    assert all(c.keys.dtype==mx.bfloat16 and c.values.dtype==mx.bfloat16 for c in final[0].cache_sidecar.state[0])


@pytest.mark.parametrize("ordinary", [False, True])
def test_partial_fit_rotates_lanes_without_ready_queue_masking(monkeypatch, ordinary):
    m,d=tiny();budget=[10**12];b=generator(m,d,memory_headroom=lambda:budget[0])
    b.insert([[1,2],[3,4]],max_tokens=[100,100])
    for lane in b.lanes.values():
        while lane.anchor is None:b._prefill(lane)
        lane.ordinary=ordinary
    real=d.draft_distributions;q=np.eye(32)[0];p=np.eye(32)[1]
    def proposals(anchors,hidden,cache,count,rngs,temperatures):
        real(anchors,hidden,cache,count,rngs,temperatures)
        return [[0]*count for _ in anchors],[[q]*count for _ in anchors]
    monkeypatch.setattr(d,'draft_distributions',proposals)
    monkeypatch.setattr(b,'_target_law',lambda *args:p)
    append=1 if ordinary else b.num_draft+1
    b._admit([b.lanes[0]],append);budget[0]=int(b.scheduler_stats['reservation_bytes']*1.8)
    assert not b._admit(list(b.lanes.values()),append)
    seen=[]
    for _ in range(8):
        _,responses=b.next();seen.extend(r.uid for r in responses)
    assert seen==[0,1]*4
    assert [l.generated for l in b.lanes.values()]==[4,4]
    b.close()


def test_bounded_reclaim_remeasures_and_preserves_pending_rng():
    m,d=tiny();budget=[0];events=[]
    def reclaim():events.append('allocator')
    def evict():
        events.append('unleased_checkpoint');budget[0]=10**12
        return True
    b=generator(m,d,memory_headroom=lambda:budget[0],reclaim_memory=reclaim,evict_checkpoint=evict)
    b.insert([[1,2]],max_tokens=[4]);b.next()
    assert events==['allocator','unleased_checkpoint','allocator']
    assert b.lanes[0].generated>0
    b.close()
    events.clear();budget[0]=0
    b=generator(m,d,memory_headroom=lambda:0,reclaim_memory=reclaim,evict_checkpoint=lambda:events.append('unleased_checkpoint') or True)
    b.insert([[1,2],[3,4],[4,5]],max_tokens=[4]*3)
    before=[l.rng.snapshot() for l in b.lanes.values()]
    assert b.next()==([],[])
    assert events.count('unleased_checkpoint')==2 and events.count('allocator')==3
    assert before==[l.rng.snapshot() for l in b.lanes.values()]
    b.close()


def test_lane_capacity_does_not_reclaim_checkpoints():
    m,d=tiny();events=[]
    b=generator(m,d,completion_batch_size=1,memory_headroom=lambda:10**12,reclaim_memory=lambda:events.append('allocator'),evict_checkpoint=lambda:events.append('checkpoint') or True)
    b.insert([[1,2],[3,4]],max_tokens=[8,8]);b.next();b.next()
    assert not events
    b.close()


@pytest.mark.parametrize('gross,expected', [(30,10),(10,0)])
def test_actual_serving_factory_preserves_reserve_and_reclaims(monkeypatch,gross,expected):
    from types import SimpleNamespace as NS

    from mlx2 import memory, serving
    from mlx2.runtime import apc_v2
    captured={};events=[]
    class APC:
        apc_stats: ClassVar[dict] = {}
        def __init__(self,**kwargs):pass
        def key(self,*args,**kwargs):return 'key'
        def clear(self):pass
        def evict_oldest_unleased(self):events.append('unleased');return True
    class Adapter:
        max_context=128;environment: ClassVar[dict] = {};identity: ClassVar[dict] = {'fingerprint':'test'};layout='test';tokenizer=NS(eos_token_ids=[])
        def __init__(self,*a,**kw):pass
        def execution_config(self,**kw):return {'backend':'external_draft','num_draft':2}
        def create_external_batch(self,**kw):
            captured.update(kw);captured['net']=kw['memory_headroom']()
            kw['reclaim_memory']();assert kw['evict_checkpoint']()
            raise RuntimeError('stop at factory seam without model allocation')
        def close(self):pass
    monkeypatch.setattr(memory,'execution_headroom',lambda:gross*(1<<30))
    monkeypatch.setattr(apc_v2,'APCv2',APC)
    monkeypatch.setattr(serving,'runtime_identity',lambda:{'source_sha256':'test'})
    monkeypatch.setattr(mx,'synchronize',lambda:events.append('sync'))
    monkeypatch.setattr(mx,'clear_cache',lambda:events.append('clear'))
    engine=serving.ServingEngine('no-model',adapter_factory=Adapter,qualification_mode=True,mtp=False)
    engine.thread.join(timeout=5)
    assert not engine.thread.is_alive() and captured['net']==expected*(1<<30)
    assert events[:3]==['sync','clear','unleased']
    assert engine.counts['memory_pressure_evictions']==1


def test_external_receipt_ordinary_width_is_only_ordinary_execution():
    from types import SimpleNamespace as NS

    from mlx2.serving import ordinary_compute_width
    r=NS(mtp_receipt=None,speculative_receipt={'execution':'external_draft_verify'})
    assert ordinary_compute_width(r,3,mtp=False,external_draft=True) is None
    r.speculative_receipt={'execution':'ordinary_target','ordinary_fallback':True}
    assert ordinary_compute_width(r,3,mtp=False,external_draft=True)==3
    assert ordinary_compute_width(r,3,mtp=False,external_draft=False)==3


def test_real_apcv2_pressure_reclaims_only_unleased_checkpoint():
    from mlx2.runtime.apc_v2 import APCKey, APCv2
    from mlx2.runtime.models.cache import KVCache
    apc=APCv2(max_size=4,layout_name='test-external');key=APCKey('pressure')
    def cache():
        c=KVCache();x=mx.ones((1,1,3,1));c.update_and_fetch(x,x);return [c]
    apc.store(key,[1,2,3],cache())
    hit=apc.lookup(key,[1,2,3,4]);owner=hit.cache.cow_owner;generation=owner.generation
    apc.store(key,[8,9,10],cache())
    m,d=tiny();b=generator(m,d,memory_headroom=lambda:10**12 if len(apc)==1 else 0,
                          reclaim_memory=lambda:None,evict_checkpoint=apc.evict_oldest_unleased)
    b.insert([[1,2]],max_tokens=[3]);b.next()
    assert b.lanes[0].generated>0 and len(apc)==1
    assert owner.generation==generation and not apc.evict_oldest_unleased()
    assert not apc.lookup(key,[8,9,10,11]).hit
    hit.cache.close();assert apc.evict_oldest_unleased()
    b.close();apc.clear(release_memory=False)


def test_unreachable_verify_rows_never_reach_logits_processors(monkeypatch):
    """A row after a zero-probability draft token is unused; a strict processor
    (structured output) must not be asked about its out-of-language history."""
    m,d=tiny();prompt=[1,2,3]
    cache=m.make_cache();logits=m(mx.array([prompt]),cache=cache)
    forbidden=int(mx.argmax(logits[0,-1]).item())  # the model's own first choice
    seen=[]
    def strict(tokens,value):
        generated=[int(t) for t in tokens.tolist()][len(prompt):]
        seen.append(generated)
        assert forbidden not in generated, "asked about a history the grammar already rejected"
        mask=np.zeros(value.shape[-1],dtype=np.float32);mask[forbidden]=-np.inf
        return value+mx.array(mask)
    real=d.draft_distributions
    def propose_forbidden(anchors,hidden,caches,count,rngs,temps,**_kwargs):
        real(anchors,hidden,caches,count,rngs,temps)
        drafts=[forbidden]+[(forbidden+1+i)%32 for i in range(count-1)]
        laws=[]
        for token in drafts:
            law=np.zeros(32);law[token]=1;laws.append(law)
        return [drafts],[laws]
    monkeypatch.setattr(d,'draft_distributions',propose_forbidden)
    b=generator(m,d);b.insert([prompt],max_tokens=[6],logits_processors=[[strict]],sampling_configs=[{'sampling_temp':0}])
    got,_final=drain(b)
    assert len(got[0])==6 and forbidden not in got[0]
    assert b.scheduler_stats['external_rounds']>0 and seen
