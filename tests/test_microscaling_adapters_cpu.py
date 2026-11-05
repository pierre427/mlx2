"""CPU adapter declarations and actual standard MXFP4 projection census.

These synthetic module trees establish compatibility, not model qualification.
"""
from types import SimpleNamespace
import pytest
import mlx.core as mx
from mlx import nn

from mlx2.adapters.microscaling_support import decode_row_multiplier, projection_law_compatible
from mlx2.runtime.microscaling.installer import adapter_scopes, default_select, module_spec


def _standard(family='qwen3', **attributes):
    from mlx2.adapters.standard_decoder import StandardDecoderAdapter
    adapter=object.__new__(StandardDecoderAdapter)
    adapter.config={'model_type':family,'num_experts':4 if family=='qwen3_moe' else 0,
                    'num_experts_per_tok':2}
    adapter.external_policy={}
    for name,value in attributes.items(): setattr(adapter,name,value)
    return adapter


def test_standard_families_declare_only_supported_seams():
    assert adapter_scopes(_standard()) == {'mlp','all'}
    assert adapter_scopes(_standard('qwen3_moe')) == {'experts','mlp','all'}
    for family in ('qwen2','llama'):
        assert adapter_scopes(_standard(family)) == set()
    assert _standard().microscaling_decode_row_multiplier() == 1
    assert _standard('qwen3_moe').microscaling_decode_row_multiplier() == 2


@pytest.mark.parametrize('module_name,class_name,scopes',[
    ('flash_next','FlashNextAdapter',{'experts','mlp','all'}),
    ('qwen38_27b','Qwen3827BAdapter',{'mlp','all'}),
    ('qwen35_9b','Qwen359BAdapter',{'mlp','all'}),
    ('qwen35_4b','Qwen354BAdapter',{'mlp','all'}),
    ('qwen36_35b','Qwen3635BA3BAdapter',{'experts','mlp','all'}),
])
def test_hybrid_adapter_declarations_and_inheritance(module_name,class_name,scopes):
    import importlib
    cls=getattr(importlib.import_module('mlx2.adapters.'+module_name),class_name)
    adapter=object.__new__(cls)
    adapter.config={'text_config':{'num_experts':512,'num_experts_per_tok':10}}
    assert adapter_scopes(adapter) == scopes
    assert adapter.microscaling_decode_row_multiplier() == 10
    adapter.policy=SimpleNamespace(lane_dense=True)
    assert adapter_scopes(adapter) == set()


@pytest.mark.parametrize('attributes',[
    {'external_policy':{'target_verify_row_exact':True}},
    {'policy':{'lane_dense':True}},
    {'policy':SimpleNamespace(tensorfold_prefill=True)},
    {'invariant_prefill':object()},
    {'tensorfold_qmv':{'selected':True}},
    {'weight_stream':object()},
    {'varlen_dense_mlp':object()},
    {'policy':{'compact_moe_prefill':True}},
])
def test_custom_projection_owners_decline(attributes):
    adapter=_standard(**attributes)
    assert not projection_law_compatible(adapter)
    assert adapter_scopes(adapter) == set()


@pytest.mark.parametrize('config',[
    {'num_experts':True,'num_experts_per_tok':2},
    {'num_experts':4},
    {'num_experts':4,'num_experts_per_tok':True},
    {'num_experts':4,'num_experts_per_tok':5},
    {'text_config':[]},
])
def test_invalid_expansion_metadata_refuses(config):
    with pytest.raises(ValueError): decode_row_multiplier(config)


def test_expansion_uses_conservative_config_and_text_top_k():
    assert decode_row_multiplier({'num_experts':8,'num_experts_per_tok':2,
        'text_config':{'num_experts':8,'num_experts_per_token':4}})==4


@pytest.mark.parametrize('family', ['qwen3','qwen3_moe'])
def test_loaded_standard_mxfp4_shapes_select_only_intended_projections(family):
    from mlx2.runtime.models.standard_decoder import Model,ModelArgs
    config=dict(model_type=family,hidden_size=256,num_hidden_layers=1,
        intermediate_size=512,num_attention_heads=2,num_key_value_heads=2,
        head_dim=128,rms_norm_eps=1e-6,vocab_size=1024,
        max_position_embeddings=4096,rope_theta=1e6,tie_word_embeddings=False,
        num_experts=4 if family=='qwen3_moe' else 0,num_experts_per_tok=2,
        moe_intermediate_size=128,decoder_sparse_step=1,mlp_only_layers=[],norm_topk_prob=True)
    model=Model(ModelArgs.from_dict(config)); model.set_dtype(mx.bfloat16)
    nn.quantize(model,group_size=32,bits=4,mode='mxfp4')
    selected={}
    for scope in ('experts','mlp','all'):
        choose=default_select(scope)
        selected[scope]={path:spec for path,module in model.named_modules()
            if choose(path,module) and (spec:=module_spec(path,module)[0]) is not None}
    assert all('lm_head' not in p and 'embed_tokens' not in p and '.gate' != p[-5:]
               for p in selected['all'])
    assert len(selected['all']) == 7
    assert len(selected['mlp']) == 3
    if family=='qwen3_moe':
        assert len(selected['experts']) == 3
        assert all(spec.kind=='grouped' and spec.experts==4 for spec in selected['experts'].values())
    else:
        assert selected['experts'] == {}


def test_kimi_retains_its_pinned_bf16_only_contract():
    from mlx2.adapters.kimi_linear import KimiLinearAdapter
    adapter=object.__new__(KimiLinearAdapter)
    assert adapter_scopes(adapter)==set()
