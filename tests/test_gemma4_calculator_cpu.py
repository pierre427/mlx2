"""CPU-only Gemma adapter admission, identity, and unchanged tensor computation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.adapters import gemma4 as G, mlx_vlm
from mlx2.runtime import amoeba_det_tiles as D
from mlx2.runtime.models import gemma4 as native

mx.set_default_device(mx.cpu)


def manifest(tmp_path, *, fingerprint="fixture-fingerprint", layers=30, parser=D.PARSER_V5, kind="calculator"):
    root=tmp_path/'tile';root.mkdir()
    payload=b'{"kind":"resident_fixture"}\n';(root/'resident.json').write_bytes(payload)
    spec={
        "schema":D.SCHEMA,"source":{"fingerprint":fingerprint},"layers":layers,
        "host_tiles_manifest_sha256":None,
        "files":{"resident.json":hashlib.sha256(payload).hexdigest()},
        "tiles":[{"id":"calculator","kind":kind,"write_mode":"force",
                  "read_layer":layers-1,"write_layer":layers-1,"file":"resident.json",
                  "parser":parser,"arithmetic_backend":"gpu","gate":{"kind":"validated_syntax"}}],
    }
    path=root/'manifest.json';path.write_text(json.dumps(spec));return path


@pytest.fixture
def inspected(monkeypatch):
    def inspect(cls,path):return {"fingerprint":"fixture-fingerprint", "variant":cls.descriptor.variant}
    monkeypatch.setattr(G._Gemma4Adapter,'_inspect_adapter_artifact',classmethod(inspect))


@pytest.mark.parametrize('cls,layers',[(G.Gemma4A4BAdapter,30),(G.Gemma431BAdapter,60)])
def test_resident_manifest_admission_before_model_load(tmp_path,inspected,cls,layers):
    path=manifest(tmp_path,layers=layers)
    result=cls._admit_calculator_manifest('/fixture',str(path))
    assert result['source']['fingerprint']=='fixture-fingerprint'
    assert result['tiles'][0]['read_layer']==layers-1


@pytest.mark.parametrize('change,match',[
    ({'fingerprint':'other-model'},'another checkpoint'),
    ({'layers':29},'different layer count'),
    ({'parser':D.PARSER_V4},'one resident calculator'),
])
def test_unsupported_manifest_is_refused_before_loading(tmp_path,inspected,monkeypatch,change,match):
    path=manifest(tmp_path,**change)
    def forbidden(*args,**kwargs):raise AssertionError('weights must not load')
    monkeypatch.setattr(mlx_vlm._MLXVLMAdapter,'__init__',forbidden)
    with pytest.raises(D.DetTileRefused,match=match):
        G.Gemma4A4BAdapter('/fixture',execution_policy={'amoeba_det_tiles':str(path)})


@pytest.mark.parametrize('policy',[True,1,'x',[],[['amoeba_det_tiles','some-path']]])
def test_policy_must_be_object(policy):
    with pytest.raises(ValueError,match='must be an object'):
        G.Gemma4A4BAdapter('/missing',execution_policy=policy)


def test_cache_identity_and_factory_forward_actual_request(tmp_path,inspected,monkeypatch):
    path=manifest(tmp_path)
    layers=[SimpleNamespace(mlp=nn.Identity()) for _ in range(30)]
    tokenizer=object()
    def base(self,model_path,*,execution_policy=None):
        self.identity={'fingerprint':'fixture-fingerprint'};self.layout='gemma-base'
        self.model=SimpleNamespace(_model=SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))))
        self.tokenizer=tokenizer;self.tensor_backend='native'
    monkeypatch.setattr(mlx_vlm._MLXVLMAdapter,'__init__',base)
    adapter=G.Gemma4A4BAdapter('/fixture',execution_policy={'amoeba_det_tiles':str(path)})
    digest=D.file_sha256(path)
    assert adapter.layout=='gemma-base+det-'+digest
    assert adapter.identity['amoeba_det_tiles_manifest_sha256']==digest
    assert adapter.execution_numerics_contract()['amoeba_det_tiles']==adapter.amoeba_det_tiles.identity
    assert hasattr(layers[-1].mlp,'amoeba_det_wrapped')
    assert not hasattr(layers[0].mlp,'amoeba_det_wrapped')
    captured={}
    def factory(tiles,tok,**kwargs):captured.update(tiles=tiles,tok=tok,**kwargs);return ('processor',)
    monkeypatch.setattr(D,'processors_for',factory)
    request={'messages':[{'role':'user','content':'Calculate 3+4.'}]}
    assert adapter.request_logits_processors(request,prompt_length=17)==('processor',)
    assert captured=={'tiles':adapter.amoeba_det_tiles,'tok':tokenizer,'request':request,'prompt_length':17}
    adapter.close()


def test_base_numerical_identity_remains_unchanged():
    adapter=G.Gemma4A4BAdapter.__new__(G.Gemma4A4BAdapter)
    assert adapter.execution_numerics_contract() is None


def test_manifest_change_during_load_fails_closed(tmp_path,inspected,monkeypatch):
    path=manifest(tmp_path)
    layers=[SimpleNamespace(mlp=nn.Identity()) for _ in range(30)]
    def base(self,*args,**kwargs):
        spec=json.loads(path.read_text());spec['changed_during_load']=True;path.write_text(json.dumps(spec))
        self.identity={'fingerprint':'fixture-fingerprint'};self.layout='gemma-base'
        self.model=SimpleNamespace(_model=SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers))))
    monkeypatch.setattr(mlx_vlm._MLXVLMAdapter,'__init__',base)
    with pytest.raises(ValueError,match='changed during model loading'):
        G.Gemma4A4BAdapter('/fixture',execution_policy={'amoeba_det_tiles':str(path)})


@pytest.mark.parametrize('moe',[False,True])
def test_dense_binding_hook_preserves_complete_feed_forward(tmp_path,moe):
    config=native.TextConfig.from_dict({
        'hidden_size':16,'num_hidden_layers':1,'intermediate_size':32,
        'num_attention_heads':2,'head_dim':8,'global_head_dim':8,'num_key_value_heads':1,
        'num_global_key_value_heads':1,'vocab_size':32,'layer_types':['sliding_attention'],
        'sliding_window':8,'hidden_size_per_layer_input':0,'num_kv_shared_layers':0,
        'enable_moe_block':moe,'num_experts':4,'top_k_experts':2,'moe_intermediate_size':16,
    })
    mx.random.seed(17);layer=native.DecoderLayer(config,0)
    x=mx.random.normal((2,3,16));before=layer._feed_forward(x);mx.eval(before)
    sparse=layer.experts if moe else None
    path=manifest(tmp_path,layers=1)
    tiles=D.install_on_adapter([layer],str(path),{'mlp':'mlp','activation_fn':nn.gelu_approx},source_fingerprint='fixture-fingerprint')
    after=layer._feed_forward(x);mx.eval(after)
    assert mx.array_equal(before,after).item()
    assert tiles.stash_shape['calculator']==(2,3)
    assert tiles.counters['calc']['gate_open']==0
    if moe:assert layer.experts is sparse


def test_cpu_manifest_preparation_is_artifact_bound(tmp_path,monkeypatch):
    artifact={'fingerprint':'f'*64,'variant':'26b-a4b'}
    monkeypatch.setattr(G,'inspect_gemma4_artifact',lambda path:artifact)
    prepare=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/research/amoeba_calc_gemma4.py'))['prepare']
    path=prepare(tmp_path/'model',tmp_path/'prepared')
    loaded=D.load_manifest(path)
    assert loaded['source']['fingerprint']=='f'*64
    assert loaded['layers']==30 and loaded['tiles'][0]['read_layer']==29
    setup=json.loads(path.with_name('setup.json').read_text())
    assert setup['model_loaded'] is False and setup['gpu_used'] is False
    assert setup['qualification']=='unqualified'
    assert json.loads(path.with_name('execution-policy.json').read_text())=={'amoeba_det_tiles':str(path)}
    with pytest.raises(FileExistsError):prepare(tmp_path/'model',tmp_path/'prepared')
