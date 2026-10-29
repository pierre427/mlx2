"""HTTP harness process ownership and receipt laws; no model or GPU."""
import importlib.util
import io
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/research'))
spec=importlib.util.spec_from_file_location('http_probe',ROOT/'scripts/research/probe_deepseek_v4_http.py')
probe=importlib.util.module_from_spec(spec);sys.modules[spec.name]=probe;spec.loader.exec_module(probe)


def test_command_is_loopback_b1_explicit_policy_and_optional_disk(tmp_path):
    ordinary=probe.command(tmp_path/'model',18871,tmp_path/'policy.json',tmp_path/'cache',external=False,prefill_step=128,cache_bytes=512<<20,persist=False)
    assert ordinary[:3]==[sys.executable,'-m','mlx2.server']
    assert ordinary[ordinary.index('--host')+1]=='127.0.0.1'
    assert ordinary[ordinary.index('--max-lanes')+1]=='1'
    assert '--ordinary' in ordinary and '--external-draft' not in ordinary
    assert '--qualification-mode' not in ordinary and '--apc-persist-dir' not in ordinary
    from mlx2.server import build_parser
    parsed=build_parser().parse_args(ordinary[3:])
    assert parsed.fabric=='off' and parsed.ordinary and parsed.cache_bytes==512<<20
    external=probe.command(tmp_path/'model',18871,tmp_path/'policy.json',tmp_path/'cache',external=True,prefill_step=128,cache_bytes=512<<20,persist=True)
    assert '--external-draft' in external and '--ordinary' not in external
    assert '--apc-persist-on-shutdown' in external
    # Regression: the real server rejects distinct cache/persistence paths
    # before model load; exercise the production parser's same-directory law.
    parsed_external=build_parser().parse_args(external[3:])
    assert Path(parsed_external.cache_dir).resolve()==Path(parsed_external.apc_persist_dir).resolve()
    assert Path(parsed_external.cache_dir)==tmp_path/'cache'/'persist'


def test_http_probe_requires_existing_owned_gpu_before_any_source(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'source_receipt',lambda:pytest.fail('source without ownership'))
    with pytest.raises(SystemExit):probe.main(['--model-path',str(tmp_path/'model'),'--output',str(tmp_path/'new')])
    assert not (tmp_path/'new').exists()
    monkeypatch.setattr(probe,'verify_ownership',lambda *a:(_ for _ in ()).throw(RuntimeError('no live owner')))
    with pytest.raises(RuntimeError,match='live owner'):
        probe.main(['--model-path',str(tmp_path/'model'),'--output',str(tmp_path/'new'),'--run-gpu','--ownership-window',str(tmp_path/'lease')])
    assert not (tmp_path/'new').exists()


def test_owned_stop_escalates_only_recorded_group_and_closes_log(monkeypatch):
    calls=[];alive=[True]
    def killpg(pid,sig):
        calls.append((pid,sig))
        if not alive[0]:raise ProcessLookupError()
        if sig==signal.SIGKILL:alive[0]=False
    class Process:
        pid=987654;returncode=None
        def wait(self,timeout):
            if alive[0]:raise subprocess.TimeoutExpired('owned',timeout)
            self.returncode=-9;return -9
    monkeypatch.setattr(probe.os,'killpg',killpg)
    monkeypatch.setattr(probe.os,'getpgrp',lambda:123)
    log=io.StringIO();result=probe.stop_owned(probe.OwnedServer(Process(),log,987654,['python','-m','mlx2.server']),grace=0)
    assert result['process_group_absent'] and result['returncode']==-9 and log.closed
    assert {pid for pid,_ in calls}=={987654}
    assert signal.SIGTERM in [s for _,s in calls] and signal.SIGKILL in [s for _,s in calls]
    with pytest.raises(RuntimeError,match='foreign'):
        probe.stop_owned(probe.OwnedServer(Process(),io.StringIO(),123,['foreign']))


def test_start_server_uses_new_session_offline_source_and_no_shell(tmp_path,monkeypatch):
    captured={}
    def popen(cmd,**kwargs):
        captured.update(command=cmd,**kwargs)
        return SimpleNamespace(pid=777777)
    monkeypatch.setattr(probe.subprocess,'Popen',popen)
    monkeypatch.setattr(probe.os,'getpgid',lambda pid:pid)
    guards=[];server=probe.start_server(['python','-m','mlx2.server'],tmp_path/'server.log',guard=lambda:guards.append(True),startup_seconds=30)
    assert guards==[True] and captured['start_new_session'] is True
    assert captured['env']['PYTHONPATH']==str(ROOT/'src') and captured['env']['HF_HUB_OFFLINE']=='1'
    assert 'shell' not in captured and server.pgid==777777
    server.log.close()


def row(cached,route='ordinary'):
    return {'text':'Hello','reasoning_content':'','token_ids':[42],
        'receipt':{'cache':'apcv2','cached_tokens':cached,'route':route,'route_receipt':'candidate_validation','prompt_tokens':256}}


def test_pair_checks_exact_output_and_observed_apc_hit_without_equivalence_promotion():
    assert probe.pair_check(row(0),row(128),external=False,law='exact',expected='Hello')['passed']
    assert not probe.pair_check(row(0),row(0),external=False,law='exact',expected='Hello')['passed']
    bad=row(128);bad['token_ids']=[43]
    result=probe.pair_check(row(0),bad,external=False,law='exact',expected='Hello')
    assert not result['passed'] and 'cold/warm token IDs differ' in result['failures']
    assert 'no near-tie' in result['equivalence']
    assert probe.pair_check(row(0,'external_draft'),row(128,'external_draft'),external=True,law='exact',expected='Hello')['passed']


def test_status_identity_checks_exact_build_artifact_and_route():
    status={'artifact':'abc','build_runtime':{'source_sha256':'def'},'settings':{'max_lanes':1,'max_inflight':1,'route':'ordinary'}}
    probe.identity_check(status,fingerprint='abc',build_source='def',external=False)
    with pytest.raises(RuntimeError):probe.identity_check(status,fingerprint='abc',build_source='wrong',external=False)
    with pytest.raises(RuntimeError):probe.identity_check(status,fingerprint='abc',build_source='def',external=True)


def test_sse_records_reasoning_ids_receipt_and_done_or_preserves_partial(tmp_path,monkeypatch):
    events=[{'choices':[{'delta':{'content':'Hello','reasoning_content':'considered'},'logprobs':{'content':[{'id':42}]}}]},
            {'choices':[{'finish_reason':'stop','delta':{}}],'mlx2':row(0)['receipt']}]
    lines=[('data: '+json.dumps(e)+'\n\n').encode() for e in events]+[b'data: [DONE]\n\n']
    class Response:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def __iter__(self):return iter(lines)
    monkeypatch.setattr(probe.urllib.request,'urlopen',lambda *a,**k:Response())
    result=probe.stream_chat('http://127.0.0.1:18871',{},tmp_path/'full.sse',timeout=30)
    assert result['token_ids']==[42] and result['text']=='Hello' and result['reasoning_content']=='considered'
    lines.pop()
    with pytest.raises(RuntimeError,match='incomplete'):
        probe.stream_chat('http://127.0.0.1:18871',{},tmp_path/'partial.sse',timeout=30)
    assert (tmp_path/'partial.sse').exists()


def test_explicit_draft_confidence_is_removed_from_ordinary_reference_policy():
    from mlx2.adapters.deepseek_v4_gguf import validate_execution_policy
    selected={'draft_model':'/sealed/draft.gguf','num_draft':5,
              'deepseek_verifier':'causal_slab','deepseek_draft_confidence':0.6,
              'deepseek_packed':'cooperative'}
    ordinary={k:v for k,v in selected.items() if k not in probe.EXTERNAL_KEYS}
    assert validate_execution_policy(ordinary)['deepseek_packed']=='cooperative'
    assert not (set(ordinary)&probe.EXTERNAL_KEYS)


def test_failed_identity_persists_ready_status_and_per_field_diagnostics(tmp_path):
    entry={};result={'server_runs':[entry]};path=tmp_path/'result.json'
    status={'artifact':'external-composite','build_runtime':{'source_sha256':'source'},
            'settings':{'max_lanes':1,'max_inflight':1,'route':'external_draft'}}
    with pytest.raises(RuntimeError,match='artifact'):
        probe.record_ready_status(entry,status,result,path,fingerprint='wrong-target-only',build_source='source',external=True)
    saved=json.loads(path.read_text())['server_runs'][0]
    assert saved['status_before']==status
    assert saved['identity_assessment']['mismatched_fields']==['artifact']
    assert saved['identity_assessment']['checks']['artifact']=={
        'expected':'wrong-target-only','observed':'external-composite','passed':False}


@pytest.mark.parametrize('confidence',[0.,0.6])
def test_expected_external_binding_matches_real_adapter_wrapper_without_gpu(monkeypatch,confidence):
    from dataclasses import asdict

    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx2.adapters import deepseek_v4_dspark_artifact as seal
    from mlx2.adapters.deepseek_v4_gguf import (
        CACHE_LAYOUT,
        DESCRIPTOR,
        DeepSeekV4GGUFAdapter,
        split_execution_policy,
    )
    from mlx2.runtime.drafters import deepseek_v4_dspark as draft
    from mlx2.runtime.gguf import weights as weights_module
    from mlx2.runtime.models.deepseek_v4 import DeepSeekV4Config
    target_config=DeepSeekV4Config(compress_ratios=(0,)*43)
    side_config=draft.DeepSeekV4DSparkConfig(3,5,256,0,(40,41,42))
    metadata={'general.architecture':'deepseek4','deepseek4.checkpoint_variant':'vision-exp',
              'general.source.revision':draft.VISION_REVISION,'deepseek4.block_count':43}
    target_identity={'fingerprint':'target-fingerprint','weight_sha256':draft.TARGET_SHA256,'path':'/sealed/target'}
    record={'path':'/sealed/sidecar','stat':{'size':42},'sha256':draft.SIDECAR_SHA256,'config':asdict(side_config)}
    policy={'draft_model':record['path'],'num_draft':3,'deepseek_verifier':'causal_slab',
            'deepseek_draft_confidence':confidence,'deepseek_hc':'fused_fp32',
            'deepseek_packed':'cooperative','deepseek_embedding':'file_rows'}
    monkeypatch.setattr(seal,'inspect_sidecar',lambda _:record)
    monkeypatch.setattr(seal,'file_stat',lambda _:{'size':42})
    monkeypatch.setattr(DeepSeekV4Config,'from_gguf',lambda _:target_config)
    monkeypatch.setattr(draft.DeepSeekV4DSparkConfig,'from_index',lambda *_:side_config)
    class EmptySideWeights:
        def __init__(self,*a,**kw):self.metadata={};self.tensors={};self._dense={}
    monkeypatch.setattr(weights_module,'GGUFWeights',EmptySideWeights)
    OriginalBackbone=draft.DeepSeekV4DSparkBackbone
    def make_backbone(c,target,w,target_weights,**kw):
        value=object.__new__(OriginalBackbone)
        value.identity=draft.verify_runtime_binding(c,**kw)
        value.config=c;value.target_config=target;value.target_weights=target_weights
        value.preload=lambda:value # upload is explicitly mocked in this identity-only seam
        return value
    monkeypatch.setattr(draft,'DeepSeekV4DSparkBackbone',make_backbone)
    kernel,external=split_execution_policy(policy)
    adapter=DeepSeekV4GGUFAdapter.__new__(DeepSeekV4GGUFAdapter)
    adapter.external_record=record;adapter.external_source={'source_snapshot_sha256':'f'*64}
    adapter.external_policy=external;adapter.kernel_policy=kernel
    adapter.identity=dict(target_identity);adapter.layout=CACHE_LAYOUT;adapter.descriptor=DESCRIPTOR;adapter.metadata=metadata
    adapter.weights=SimpleNamespace(compute_policy='cooperative',embedding_policy='file_rows')
    # Independent explicit old model identity equation, not server status.
    adapter.model=SimpleNamespace(args=target_config,weights=adapter.weights,
        identity='target-fingerprint:native-packed-fp32-qat-stable-index-v2:compute=cooperative:hc=fused_fp32')
    adapter._load_external() # production adapter + real wrapper.bind(), mocked weight upload only
    expected=probe.expected_external_identity({'identity':target_identity,'metadata':metadata},policy,{'source_snapshot_sha256':'f'*64})
    assert expected['fingerprint']==adapter.identity['fingerprint']
    assert expected['execution_binding']==adapter.draft_model.execution_binding
    assert expected['target_fingerprint']==adapter.identity['target_fingerprint']
    assert expected['fingerprint']!='target-fingerprint'
    wrong=probe.expected_external_identity({'identity':target_identity,'metadata':metadata},
        {**policy,'num_draft':4},{'source_snapshot_sha256':'f'*64})
    assert wrong['fingerprint']!=expected['fingerprint']
