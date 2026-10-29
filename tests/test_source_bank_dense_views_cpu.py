"""Central Source view-cache integration; no model/GPU execution."""
from types import SimpleNamespace
import mlx.core as mx
import pytest
from mlx2.adapters.deepseek_v41_source_weights import SourceV41Weights
from mlx2.adapters.deepseek_v41_mlx_weights import NativeV41Weights
from mlx2.runtime.gguf import source_dense

@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    old=mx.default_device();mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**kw:pytest.fail('GPU forbidden'))
    yield
    mx.set_default_device(old)

def weights():
    w=object.__new__(SourceV41Weights);w.source_bf16_loads_policy='woa_f32'
    w.source_resident_bank_session=SimpleNamespace(active=False,replaying=False,poisoned=False,operand_policy='coefficient_views_v1')
    w.expert_stream=SimpleNamespace(_active=False);w._source_completion=None
    w._dense={'blk.0.attn_output_a.weight':object()};w._ensure_dense=lambda:None
    w._resident_dense_coefficient_views={};w.scheduled_grouped_calls={}
    w.array=lambda name:w._dense[name]
    return w

def test_preparation_cached_once_and_hot_bank_grouped_passes_cap(monkeypatch):
    w=weights();calls=[]
    class Cap:
        def __init__(self,full):self.full=full
        def operand(self,full,kind):assert full is self.full and kind=='wo_a';return 'view'
    def prepare(full,kind,**kwargs):calls.append((full,kind));return Cap(full)
    monkeypatch.setattr(source_dense,'prepare_coefficient_view',prepare)
    w.prepare_resident_coefficient_views(ownercheck=lambda:True)
    w.prepare_resident_coefficient_views(ownercheck=lambda:True)
    assert len(calls)==1
    seen=[]
    monkeypatch.setattr(source_dense,'maybe_project',lambda *a,**kw:seen.append(kw)or'mapped-output')
    w.source_resident_bank_session.active=True
    assert w.grouped(mx.zeros((1,1,8,4096)), 'blk.0.attn_output_a.weight')=='mapped-output'
    assert seen[0]['operand_policy']=='coefficient_views_v1'
    assert seen[0]['coefficient_view']is w._resident_dense_coefficient_views['blk.0.attn_output_a.weight']
    w.source_resident_bank_session.active=False
    w.grouped(mx.zeros((1,1,8,4096)), 'blk.0.attn_output_a.weight')
    assert seen[-1]=={}

def test_active_preparation_and_unprepared_hot_path_refuse(monkeypatch):
    w=weights();w.source_resident_bank_session.active=True
    with pytest.raises(ValueError,match='idle prepared'):w.prepare_resident_coefficient_views(ownercheck=lambda:True)
    monkeypatch.setattr(source_dense,'maybe_project',lambda *a,**kw:pytest.fail('unprepared launched'))
    with pytest.raises(ValueError,match='not prepared'):w.grouped(mx.zeros((1,1,8,4096)), 'blk.0.attn_output_a.weight')

@pytest.mark.parametrize('operation',['park','close'])
def test_cap_roots_retired_only_after_successful_provider_fence(monkeypatch,operation):
    w=weights();w.source_resident_bank_session=None;cap=object();w._resident_dense_coefficient_views={'name':cap}
    def fail(self):assert self._resident_dense_coefficient_views['name']is cap;raise RuntimeError('fence failed')
    monkeypatch.setattr(NativeV41Weights,operation,fail)
    with pytest.raises(RuntimeError,match='fence failed'):getattr(w,operation)()
    assert w._resident_dense_coefficient_views['name']is cap
    def success(self):assert self._resident_dense_coefficient_views['name']is cap;return 'retired'
    monkeypatch.setattr(NativeV41Weights,operation,success)
    assert getattr(w,operation)()=='retired'and w._resident_dense_coefficient_views=={}
