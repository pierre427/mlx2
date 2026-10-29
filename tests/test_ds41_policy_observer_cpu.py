"""synthetic@cpu-observer; no MLX import/model/GPU qualification."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('ds41_policy_observer',ROOT/'scripts/research/ds41_policy_observer.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)


def setup(tmp_path,*,fail=False):
    writes=[];o=m.Observer(tmp_path,writer=lambda path,value:writes.append(value))
    f=SimpleNamespace()
    f.rounded=lambda value:value
    f.rotate=lambda value:value
    f.router=lambda value:value
    f.rotate_qat=lambda value:value
    f.expand=lambda value:value
    f.self_check=lambda:f.rounded('synthetic')
    f.epilogue_check=lambda:f.expand('synthetic')
    hc=SimpleNamespace(**{name:(lambda value:value)for name in ('split_sinkhorn','weighted_sum','expand','head_reduce')})
    q=SimpleNamespace(fused_quantize=lambda value:value)
    class Model:
        identity='synthetic@cpu-observer';args=SimpleNamespace(simulate_quantization=True)
        qat_policy='fused_fp32'
        def qat(self,value):return q.fused_quantize(value)
        def _forward_ordinary(self,value):
            f.self_check();f.epilogue_check();f.rounded(value);f.rotate(value);f.router(value)
            f.rotate_qat(value);f.expand(value);hc.expand(value);self.qat(value)
            if fail:raise ValueError('later failure')
            return value
    m.patch_modules(o,SimpleNamespace(DeepSeekV41Model=Model),hc,f,q)
    return o,Model,f,writes


def test_counts_actual_helpers_excludes_check_fixtures_and_flushes_once(tmp_path):
    o,Model,f,writes=setup(tmp_path)
    assert Model()._forward_ordinary('real')=='real'
    s=o.snapshot();c=s['calls_after_successful_return']
    assert c['fused.rounded']==1 and c['fused.expand']==1
    assert c['model.qat.fused_fp32']==c['qat.fused_quantize']==1
    assert s['synthetic_check_calls_excluded']==2
    assert s['completed_outer_forwards']==1 and len(writes)==1
    f.rounded('outside-forward');assert o.snapshot()['calls_after_successful_return']==c
    o.uninstall();Model()._forward_ordinary('no observer');assert len(writes)==1


def test_helper_exception_is_not_counted_and_failed_forward_remains_visible(tmp_path):
    o,Model,f,writes=setup(tmp_path,fail=True)
    with pytest.raises(ValueError,match='later failure'):Model()._forward_ordinary('real')
    s=o.snapshot();assert s['failed_outer_forwards']==1 and s['completed_outer_forwards']==0
    assert s['calls_after_successful_return']['fused.rounded']==1
    assert len(writes)==1
    o.uninstall()


def test_successful_return_required_for_specific_helper_count(tmp_path):
    o=m.Observer(tmp_path,writer=lambda path,value:None)
    def bad():raise RuntimeError('helper failed')
    ns=SimpleNamespace(f=bad);o.patch(ns,'f','actual')
    o.local.forward_depth=1
    with pytest.raises(RuntimeError):ns.f()
    assert o.snapshot()['calls_after_successful_return']=={}
    o.uninstall()


def test_raw_chain_rejection_distinguishes_delivery_truncation(tmp_path):
    o=m.Observer(tmp_path,writer=lambda path,value:None)
    decisions=[SimpleNamespace(accepted=3,emitted=[7]),SimpleNamespace(accepted=1,emitted=[8,9])]
    class Batch:
        receipt_kind='external_deepseek_v41_embedded_dspark_chain'
        def _verify(self,cohort,blocks,logits):
            assert logits is sentinel
            return decisions
    sentinel=object();o.patch_verify(Batch)
    blocks=[SimpleNamespace(lengths=[3]),SimpleNamespace(lengths=[3])]
    assert Batch()._verify([1,2],blocks,sentinel)is decisions
    s=o.snapshot();records=s['chain_verification_decisions']
    assert records[0]['raw_accepted']==3 and not records[0]['rejection_observed']
    assert records[0]['emitted_after_stop_count']==1 and records[0]['full_acceptance_observed']
    assert records[1]['rejection_observed']and records[1]['raw_accepted']==1
    assert s['calls_after_successful_return']['external.chain_rejection_rounds']==1
    o.uninstall()


def test_verification_observer_declines_device_metadata_without_reading_it(tmp_path):
    o=m.Observer(tmp_path,writer=lambda path,value:None)
    class Device:
        __module__='mlx.core'
        def __getitem__(self,index):raise AssertionError('must not read device length')
    class Batch:
        def _verify(self,*args):return [SimpleNamespace(accepted=0,emitted=[])]
    o.patch_verify(Batch)
    Batch()._verify([], [SimpleNamespace(lengths=Device())], object())
    assert o.snapshot()['chain_verification_metadata_declined']==1
    assert not o.snapshot()['chain_verification_decisions']
    o.uninstall()


def test_shutdown_marker_distinguishes_final_from_live_partial_snapshot(tmp_path):
    writes=[];o=m.Observer(tmp_path,writer=lambda path,value:writes.append(value))
    assert o.snapshot()['shutdown_snapshot'] is False
    assert o.finalize()['shutdown_snapshot'] is True
    assert writes[-1]['shutdown_snapshot'] is True


def test_native_vision_observer_requires_encoder_and_target_prefill_success(tmp_path):
    writes=[];o=m.Observer(tmp_path,writer=lambda path,value:writes.append(value))
    class Vision:
        artifact={'identity':{'fingerprint':'source-binding'}}
        target=SimpleNamespace(identity='target-binding')
        def __call__(self,tokens,cache,images):return ('embedding','mask')
    owner=Vision()
    class Model:
        identity='target-binding'
        def prefill_forward(self,tokens,cache=None,*,image_inputs=None):
            if image_inputs is not None:owner(tokens,cache,image_inputs)
            return 'target-logits'
    o.patch_native_vision(Vision,Model)
    assert Model().prefill_forward(SimpleNamespace(shape=(1,205)),image_inputs=[object()])=='target-logits'
    s=o.snapshot();c=s['calls_after_successful_return']
    assert c['vision.encoder_calls_completed']==c['vision.images_encoded']==c['vision.native_image_prefill_returns']==1
    assert s['native_vision']['source_fingerprint']=='source-binding'
    assert s['native_vision']['prefill_returned']is True
    assert len(writes)==1
    o.uninstall()
