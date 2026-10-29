"""Synthetic proposal-context flow; no target checkpoint qualification."""
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from test_sweep_20261006_spec_tree_prelaunch import lane, make_gen


@pytest.fixture(autouse=True)
def synthetic_receipt(record_property):
    previous=mx.default_device(); mx.set_default_device(mx.cpu)
    record_property('source_revision',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    root=Path(__file__).resolve().parents[1]
    sources=['src/mlx2/runtime/external_speculative.py','src/mlx2/adapters/nanbeige43.py','src/mlx2/adapters/nanbeige43_quantized_tree.py']
    record_property('working_source_sha256',hashlib.sha256(b''.join((root/p).read_bytes() for p in sources)).hexdigest())
    record_property('checkpoint','synthetic-tree-context-flow@sha256:'+hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    yield
    mx.set_default_device(previous)


def observed_generator(width,monkeypatch,*,declared=True):
    gen=make_gen(width); gen.draft.requires_tree_proposal_contexts=declared
    calls=[]
    original=gen.draft.start_tree
    def start(anchors,tail,caches,count,**kwargs):
        calls.append({'anchors':list(anchors),'tail':tail,'kwargs':kwargs})
        return original(anchors,tail,caches,count,**kwargs)
    monkeypatch.setattr(gen.draft,'start_tree',start)
    return gen,calls


@pytest.mark.parametrize('width',[1,2,4])
def test_current_proposal_contexts_are_lane_ordered_owned_copies(width,monkeypatch):
    gen,calls=observed_generator(width,monkeypatch)
    lanes=[lane(i) for i in range(width)]
    for i,item in enumerate(lanes):
        item.history=[31+i,41+i]; item.processors=[object()] if i%2 else []
    options=gen._tree_options(count_hits=False,lanes=lanes)
    assert options['proposal_histories']==[[31+i,41+i] for i in range(width)]
    assert options['proposal_has_processors']==[bool(i%2) for i in range(width)]
    options['proposal_histories'][0].append(999)
    assert lanes[0].history==[31,41]
    gen._prelaunch_tree_cohort(lanes,[SimpleNamespace(emitted=[99]) for _ in lanes])
    assert calls[0]['kwargs']['proposal_histories']==[item.history for item in lanes]
    assert all(99 not in history for history in calls[0]['kwargs']['proposal_histories'])


@pytest.mark.parametrize("explicit_inputs",[None,[51,52]])
def test_single_early_context_matches_committed_inputs_before_publication(monkeypatch,explicit_inputs):
    gen,calls=observed_generator(1,monkeypatch); item=lane(0)
    item.history=[10,20,30]; before=list(item.history)
    decision=SimpleNamespace(commit_rows=(0,2),emitted=[81,82],committed_inputs=explicit_inputs)
    features=mx.arange(16).reshape(1,4,4)
    record=gen._prelaunch_tree_early(item,decision,features)
    assert record is not None and len(calls)==1
    assert calls[0]['anchors']==[82]
    expected=[*before,*(explicit_inputs if explicit_inputs is not None else [item.anchor,81])]
    assert calls[0]['kwargs']['proposal_histories']==[expected]
    assert 82 not in expected  # New bonus is the next anchor, not committed KV.
    if explicit_inputs is None:
        assert item.anchor in expected
    assert item.history==before
    assert record.key[:3]==(5,82,item.generated+2)
    assert bool(mx.array_equal(calls[0]['tail'],features[:,[0,2]]).item())


@pytest.mark.parametrize('width',[2,4])
@pytest.mark.parametrize('explicit_inputs',[False,True])
def test_cohort_early_context_matches_each_committed_input_boundary(width,monkeypatch,explicit_inputs):
    gen,calls=observed_generator(width,monkeypatch)
    lanes=[lane(i) for i in range(width)]
    for i,item in enumerate(lanes): item.history=[10+i,20+i]
    before=[list(item.history) for item in lanes]
    decisions=[SimpleNamespace(commit_rows=(0,2),emitted=[80+i,90+i],
        committed_inputs=[41+i,51+i] if explicit_inputs else None) for i in range(width)]
    features=mx.arange(width*4*4).reshape(width,4,4)
    records=gen._prelaunch_tree_cohort_early(lanes,decisions,features)
    assert len(records)==1 and len(calls)==1
    assert calls[0]['anchors']==[90+i for i in range(width)]
    expected=[[ *before[i], *([41+i,51+i] if explicit_inputs else [lanes[i].anchor,80+i])] for i in range(width)]
    assert calls[0]['kwargs']['proposal_histories']==expected
    assert all(90+i not in expected[i] for i in range(width))
    assert [item.history for item in lanes]==before
    assert calls[0]['kwargs']['proposal_has_processors']==[False]*width
    assert bool(mx.array_equal(calls[0]['tail'],features[:,[0,2]]).item())


@pytest.mark.parametrize('declared',[False,'true',1])
def test_unrelated_drafters_receive_no_new_options_even_for_future_context(monkeypatch,declared):
    gen,calls=observed_generator(1,monkeypatch,declared=declared)
    item=lane(0)
    assert gen._tree_options(count_hits=False,lanes=[item])=={}
    decision=SimpleNamespace(commit_rows=(0,),emitted=[85],committed_inputs=None)
    record=gen._prelaunch_tree_early(item,decision,mx.zeros((1,2,4)))
    assert record is not None
    assert 'proposal_histories' not in calls[0]['kwargs']
    assert 'proposal_has_processors' not in calls[0]['kwargs']


def test_early_cohort_contexts_follow_tail_groups_and_lane_order(monkeypatch):
    gen,calls=observed_generator(4,monkeypatch); lanes=[lane(i) for i in range(4)]
    decisions=[SimpleNamespace(emitted=[70+i],committed_inputs=None) for i in range(4)]
    # Equal-tail groups intentionally interleave lane IDs.
    views=[SimpleNamespace(anchor=70+i,tail=mx.zeros((1,1+i%2,4)),history=4,generated=11) for i in range(4)]
    records=gen._prelaunch_tree_cohort(lanes,decisions,views)
    assert len(records)==2 and len(calls)==2
    by_anchor={tuple(call['anchors']):call['kwargs']['proposal_histories'] for call in calls}
    assert by_anchor[(70,72)]==[[1,2,3,5],[1,2,3,5]]
    assert by_anchor[(71,73)]==[[1,2,3,5],[1,2,3,5]]


@pytest.mark.parametrize('declared',[True,False])
def test_current_fresh_propose_passes_context_only_to_declared_drafter(monkeypatch,declared):
    gen,_calls=observed_generator(2,monkeypatch,declared=declared)
    lanes=[lane(0),lane(1)]; lanes[0].history=[17,19]; lanes[1].history=[23,29,31]
    lanes[1].processors=[object()]
    seen=[]; original=gen.draft.propose_tree
    def propose(*args,**kwargs): seen.append(kwargs); return original(*args,**kwargs)
    monkeypatch.setattr(gen.draft,'propose_tree',propose)
    blocks=gen._propose(lanes)
    assert len(blocks)==2 and len(seen)==1
    if declared:
        assert seen[0]['proposal_histories']==[[17,19],[23,29,31]]
        assert seen[0]['proposal_has_processors']==[False,True]
    else:
        assert 'proposal_histories' not in seen[0] and 'proposal_has_processors' not in seen[0]


def test_zero_accepted_early_context_commits_old_anchor_and_excludes_new_bonus(monkeypatch):
    gen,calls=observed_generator(1,monkeypatch); item=lane(0)
    before=list(item.history); old_anchor=item.anchor
    decision=SimpleNamespace(commit_rows=(0,),emitted=[85],committed_inputs=None)
    record=gen._prelaunch_tree_early(item,decision,mx.zeros((1,2,4)))
    assert record is not None
    assert calls[0]['anchors']==[85]
    assert calls[0]['kwargs']['proposal_histories']==[[*before,old_anchor]]
    assert item.history==before and item.anchor==old_anchor


def test_early_proposal_context_equals_actual_commit_publication(monkeypatch):
    """Cross-check the predicted history against real target commit semantics."""
    from test_dflash_tree_bridge import _started, _tree_env, generator, tiny
    _tree_env(monkeypatch,pipeline_draft=True)
    target,draft=tiny(vocab=64,top_k=16,block_size=8)
    batch=generator(target,draft)
    batch.insert([[1,2,3]],max_tokens=[12],sampling_configs=[{'sampling_temp':0.0}])
    item=_started(batch)
    batch.draft.requires_tree_proposal_contexts=True
    original_start=batch.draft.start_tree
    def consume_contexts(*args,**kwargs):
        # This fixture's legacy drafter does not implement the new heuristic;
        # the wrapper observes context while retaining its real tree arithmetic.
        kwargs.pop('proposal_histories',None)
        kwargs.pop('proposal_has_processors',None)
        return original_start(*args,**kwargs)
    monkeypatch.setattr(batch.draft,'start_tree',consume_contexts)
    original_early=batch._prelaunch_tree_early
    predictions=[]
    def early(lane,decision,features):
        record=original_early(lane,decision,features)
        if record is not None:
            predictions.append((lane,record,list(lane.history),lane.anchor,list(decision.emitted)))
        return record
    monkeypatch.setattr(batch,'_prelaunch_tree_early',early)
    original_commit=batch._commit; comparisons=[]
    def commit(*args,**kwargs):
        result=original_commit(*args,**kwargs)
        for committed_lane,record,before,anchor,emitted in predictions[len(comparisons):]:
            comparisons.append((record.options['proposal_histories'][0],list(committed_lane.history)))
            assert committed_lane.history==[*before,anchor,*emitted[:-1]]
        return result
    monkeypatch.setattr(batch,'_commit',commit)
    try:
        while item.ready: batch.next()
        batch.next()
        assert comparisons
        assert all(predicted==published for predicted,published in comparisons)
    finally:
        batch.close()
