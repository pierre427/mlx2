"""CPU transaction oracles: real DFlash executor, substitute device calculator law.

These do not claim Metal arithmetic validation; the parent owns that separate
check. They exercise draft isolation, accepted-prefix state, recovery and batching.
"""
import copy
import threading
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.amoeba_calc_resident import ResidentCalculatorProcessor
from mlx2.runtime.amoeba_calc_speculative import SpeculativeResidentCalculatorProcessor as Processor
from mlx2.runtime.processor_probe import VerifyWindow,probe_logits_processors
from mlx2.runtime.speculative_topology import processors_branch_safe
from mlx2.runtime import amoeba_det_tiles as D


def processor(monkeypatch,plan=(7,8,9,0),prompt=3):
    def device_law(self,tokens,logits):
        # Opaque array replacement has the same snapshot semantics as the
        # resident kernel. Only the actual arithmetic kernel is substituted.
        index=int(self.state[6].item())
        target=self._test_plan[min(index,len(self._test_plan)-1)]
        state=np.array(self.state,copy=True)
        if self.first:state[5]+=1
        state[6]+=1
        self.state=mx.array(state);self.first=False;self.entry['state']=self.state
        mask=mx.where(mx.arange(logits.shape[-1])==target,0.,-1e9)
        return logits+mask
    monkeypatch.setattr(ResidentCalculatorProcessor,'__call__',device_law)
    p=object.__new__(Processor)
    p.tiles=SimpleNamespace(route='external_draft',lock=threading.Lock())
    p.state=mx.zeros((288,),dtype=mx.int32);p.first=True;p.prompt_length=prompt
    p._position=prompt-1;p._speculative_lane=None;p._verify_pending=False;p._detached=False
    p.entry={'state':p.state,'seen':[0,0,0,0],'finished':False}
    p._test_plan=plan
    return p


def selected(p,tokens):
    return int(mx.argmax(p(mx.array(tokens),mx.zeros((1,32),dtype=mx.float32))).item())


def test_provisional_probe_replays_suffix_without_changing_live_state(monkeypatch):
    p=processor(monkeypatch);p.bind_speculative_lane(0)
    before=p.verify_ledger();published=p.entry['state']
    for prefix,want in [([1,2,3],7),([1,2,3,7],8),([1,2,3,7,8],9)]:
        value=probe_logits_processors([p],mx.array(prefix),mx.zeros((1,32)))
        assert int(mx.argmax(value).item())==want
    assert p.state is before[0] and p.first and p._position==2
    assert p.entry['state'] is published


@pytest.mark.parametrize('used',[0,1,2,3])
def test_accepted_rows_commit_device_state_and_only_committed_counters(monkeypatch,used):
    p=processor(monkeypatch);p.bind_speculative_lane(0);published=p.entry['state']
    window=VerifyWindow([p]);prefix=[1,2,3]
    for token in [7,8,9]:
        assert selected(p,prefix)==token;window.mark();prefix.append(token)
        assert p.entry['state'] is published
    window.settle(used)
    assert p._position==2+used
    assert int(p.state[6].item())==used
    assert int(p.entry['state'][6].item())==used
    assert p.first is (used==0)
    assert not p._verify_pending
    assert selected(p,[1,2,3]+[7,8,9][:used])==[7,8,9,0][used]


def test_recovery_preserves_published_entry_and_shared_tile_ownership(monkeypatch):
    p=processor(monkeypatch);p.bind_speculative_lane(4)
    saved=copy.deepcopy(p);entry=p.entry
    assert saved.tiles is p.tiles and saved.entry is not entry
    window=VerifyWindow([p]);selected(p,[1,2,3]);window.mark()
    p.restore_from_recovery(saved)
    assert p.entry is entry and p.first and p._position==2
    assert int(entry['state'][6].item())==0
    assert not p._verify_pending


def test_missing_wrong_or_nonconsecutive_bindings_fail_closed(monkeypatch):
    p=processor(monkeypatch)
    with pytest.raises(ValueError,match='lane binding'):selected(p,[1,2,3])
    p.bind_speculative_lane(2)
    with pytest.raises(ValueError,match='shared between'):p.bind_speculative_lane(3)
    with pytest.raises(ValueError,match='consecutive'):selected(p,[1,2,3,4])
    assert not processors_branch_safe([p])
    p.tiles.route='ordinary'
    with pytest.raises(ValueError,match='lane binding'):selected(p,[1,2,3])


@pytest.mark.parametrize('temperature',[0.0,0.7])
@pytest.mark.parametrize('width',[1,2])
def test_real_dflash_executor_forces_exact_plan_and_keeps_lane_counters(monkeypatch,temperature,width):
    from test_external_dflash2_cpu import tiny,generator,drain
    model,draft=tiny();ps=[processor(monkeypatch,plan=(7+i,8+i,9+i,0)) for i in range(width)]
    b=generator(model,draft,stop_tokens=[[0]],completion_batch_size=width)
    try:
        uids=b.insert([[1,2,3]]*width,max_tokens=[8]*width,
            sampling_configs=[{'sampling_temp':temperature}]*width,
            logits_processors=[[p] for p in ps])
        output,final=drain(b)
        for i,(uid,p) in enumerate(zip(uids,ps)):
            assert output[uid]==[7+i,8+i,9+i,0]
            assert final[uid].finish_reason=='stop'
            assert int(p.entry['state'][5].item())==1
            assert int(p.entry['state'][6].item())==4
            assert not p._verify_pending
        assert b.scheduler_stats['proposed_tokens']>0
        assert b.scheduler_stats['segmented_transactions']>0
    finally:b.close()


def manifest(tmp_path,parser=D.PARSER_V5):
    spec={'id':'calc','kind':'calculator','read_layer':0,'write_layer':0,'write_mode':'force',
          'parser':parser,'arithmetic_backend':'gpu','gate':{'kind':'validated_syntax'}}
    return D.write_manifest(tmp_path,source={'fingerprint':'toy'},layers=1,tiles=[spec],
                            tensors={'placeholder':mx.zeros((1,))})


def test_external_route_only_admits_resident_calculator_manifest(tmp_path):
    path=manifest(tmp_path/'resident')
    assert D.refuse_policy(str(path),{'draft_model':'local'},allow_external=True)=='external_draft'
    with pytest.raises(D.DetTileRefused):D.refuse_policy(str(path),{'draft_model':'local'})
    legacy=manifest(tmp_path/'legacy',D.PARSER_V4)
    with pytest.raises(D.DetTileRefused,match='resident'):D.refuse_policy(str(legacy),{'draft_model':'local'},allow_external=True)
    for policy in [{'draft_models':['local']},{'draft_model':'local','prompt_lookup':True}]:
        with pytest.raises(D.DetTileRefused):D.refuse_policy(str(path),policy,allow_external=True)
    ts=D.build(path,lambda x:x,source_fingerprint='toy',layers_count=1,route='external_draft')
    ts.check_route('external_draft')
    with pytest.raises(D.DetTileRefused):ts.check_route('ordinary')
    assert ts.receipt()['speculative_protocol']=='resident-calculator-exact-verify-v1'


def test_tree_walk_tracks_only_selected_path_and_commits_used_rows(monkeypatch):
    from test_external_history_work_cpu import executor,lane
    from mlx2.runtime.external_speculative import TreeDraftRow
    p=processor(monkeypatch);p.bind_speculative_lane(1)
    b=executor();b.stops={0};current=lane([1,2],[p]);current.anchor=3
    # Two unused sibling paths are scored by the target tensor but never
    # passed to the stateful calculator or to the lane RNG.
    block=TreeDraftRow([17,7,18,8,9],[-1,-1,0,1,3])
    result=b._verify_tree(current,block,mx.zeros((1,6,32)))
    assert result.emitted==[7,8,9,0]
    assert result.commit_rows==[0,2,4,5]
    assert int(p.entry['state'][6].item())==0
    result.verify_window.settle(3) # request budget cut before the fourth row
    assert int(p.entry['state'][6].item())==3
    assert p._position==5


def test_real_executor_recovery_restores_original_processor_and_counters(monkeypatch):
    from test_external_dflash2_cpu import tiny,generator,drain
    m,d=tiny();p=processor(monkeypatch);b=generator(m,d,stop_tokens=[[0]])
    try:
        uid=b.insert([[1,2,3]],max_tokens=[8],logits_processors=[[p]])[0]
        while b.lanes[uid].anchor is None:b._prefill(b.lanes[uid])
        current=b.lanes[uid];saved=b._snapshot_round([current]);entry=p.entry
        window=VerifyWindow([p]);selected(p,[1,2,3]);window.mark()
        b._restore_round([current],saved)
        assert current.processors[0] is p and p.entry is entry
        assert p.first and p._position==2 and not p._verify_pending
        output,_=drain(b)
        assert output[uid]==[7,8,9,0]
        assert int(entry['state'][6].item())==4
    finally:b.close()


def test_qwen_policy_fingerprint_sees_wrapped_dense_projections(tmp_path):
    from test_qwen38_hybrid_native_flat_cpu import _tiny_quantized_mtp
    from mlx2.adapters.qwen38_hybrid_native_flat import _target_execution_policy
    model=_tiny_quantized_mtp();before=_target_execution_policy(model)
    layers=model.language_model.model.layers
    path=D.write_manifest(tmp_path/'native',source={'fingerprint':'toy'},layers=len(layers),
        tiles=[{'id':'calc','kind':'calculator','read_layer':0,'write_layer':0,'write_mode':'force',
                'parser':D.PARSER_V5,'arithmetic_backend':'gpu','gate':{'kind':'validated_syntax'}}],
        tensors={'placeholder':mx.zeros((1,))})
    D.install_on_adapter(layers,path,{'mlp':'mlp','activation_fn':lambda x:x},
                         source_fingerprint='toy',route='external_draft')
    after=_target_execution_policy(model)
    assert after[-1]==before[-1] # exact underlying layer arithmetic geometry
    assert after[0]!=before[0] # wrapper ownership remains visible in module identity
