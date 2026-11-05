"""Synthetic CPU loss/VJP tests; no real Kimi checkpoint or memory qualification."""
import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten
import numpy as np
import pytest

from mlx2.experimental.kimi_linear_lora.training import (
    LoRATrainingSession, chunked_head_cross_entropy, masked_cross_entropy, select_targets,
)
from mlx2.adapters.kimi_linear import parse_policy
from mlx2.runtime.models.kimi_linear import Model, ModelArgs
from mlx2.runtime.models import moe_nax_gather


@pytest.fixture(autouse=True)
def cpu_only():
    device, mode = mx.default_device(), moe_nax_gather.set_mode('off')
    mx.set_default_device(mx.cpu)
    mx.random.seed(531)
    yield
    mx.set_default_device(device)
    moe_nax_gather.set_mode(mode)


def assert_close(a, b, atol=2e-6, rtol=2e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=rtol)


@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('chunk', [1, 4, 128])
@pytest.mark.parametrize('mask_kind', ['full', 'sparse_fractional', 'zero'])
def test_frozen_head_loss_and_hidden_gradient_match_dense(quantized, chunk, mask_kind):
    head = nn.Linear(64, 128, bias=False)
    if quantized:
        head = head.to_quantized(group_size=64, bits=8)
    head.freeze()
    hidden = mx.random.normal((2, 11, 64))
    labels = mx.array([[i % 128 for i in range(11)], [(i+7) % 128 for i in range(11)]])
    mask = (mx.ones((2, 11)) if mask_kind == 'full' else mx.zeros((2, 11))
            if mask_kind == 'zero' else mx.array([[0, 0, 0, 0, .5, 0, 1, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1, 0, 0, .25, 0, 1]]))
    dense, dg = mx.value_and_grad(lambda x: masked_cross_entropy(head(x), labels, mask))(hidden)
    chunked, cg = mx.value_and_grad(lambda x: chunked_head_cross_entropy(
        head, x, labels, mask, chunk_size=chunk))(hidden)
    mx.eval(dense, dg, chunked, cg)
    assert_close(dense, chunked)
    assert_close(dg, cg)
    if mask_kind == 'zero':
        assert dense.item() == chunked.item() == 0
        assert np.count_nonzero(np.array(cg)) == 0
    assert np.count_nonzero(np.array(cg)[np.array(mask) == 0]) == 0


def test_projection_visits_bounded_chunks_and_complete_remainder():
    seen = []
    class Head(nn.Linear):
        def __call__(self, value):
            seen.append(value.shape[1])
            return super().__call__(value)
    head = Head(64, 128, bias=False)
    head.freeze()
    value = chunked_head_cross_entropy(head, mx.ones((1, 11, 64)), mx.zeros((1, 11), mx.int32),
                                       mx.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]), chunk_size=4)
    mx.eval(value)
    assert seen == [4, 4, 3]
    # This checks traced projection shapes only, not device allocation lifetime.


@pytest.mark.parametrize('chunk', [0, -1, True, 1.5, '128', None])
def test_invalid_chunk_size_refused(chunk):
    head = nn.Linear(64, 128)
    head.freeze()
    with pytest.raises(ValueError, match='positive integer'):
        chunked_head_cross_entropy(head, mx.ones((1, 2, 64)), mx.zeros((1, 2), mx.int32),
                                   mx.ones((1, 2)), chunk_size=chunk)


def test_trainable_head_and_invalid_shapes_refused():
    head = nn.Linear(64, 128)
    with pytest.raises(ValueError, match='frozen'):
        chunked_head_cross_entropy(head, mx.ones((1, 2, 64)), mx.zeros((1, 2), mx.int32),
                                   mx.ones((1, 2)), chunk_size=1)
    head.freeze()
    with pytest.raises(ValueError, match='Expected hidden'):
        chunked_head_cross_entropy(head, mx.ones((1, 2, 64)), mx.zeros((1, 1), mx.int32),
                                   mx.ones((1, 2)), chunk_size=1)


def tiny_model(layers):
    global_layers = list(range(4, layers+1, 4)) + ([layers] if layers % 4 else [])
    args = ModelArgs(model_type='kimi_linear', vocab_size=128, hidden_size=64,
        intermediate_size=64, moe_intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=0, layer_group_size=4, max_position_embeddings=1024,
        rms_norm_eps=1e-5, rope_theta=10000., routed_scaling_factor=2.446,
        head_dim=64, kv_lora_rank=64, qk_rope_head_dim=32, qk_nope_head_dim=32,
        v_head_dim=32, linear_attn_config={'full_attn_layers': global_layers,
            'kda_layers': [i for i in range(1,layers+1) if i not in global_layers],
            'num_heads': 2, 'head_dim': 32, 'short_conv_kernel_size': 4})
    model = Model(args)
    model.eval()
    nn.quantize(model, group_size=64, bits=8,
                class_predicate=lambda _, m: hasattr(m, 'to_quantized') and m.weight.shape[-1] % 64 == 0)
    model.configure(parse_policy({'mla': 'expanded', 'mla_flash': 'off', 'kda': 'reference'}))
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize('layers', [2, 27])
@pytest.mark.parametrize('chunk_size', [None, 7])
def test_complete_kimi_factor_vjps_and_normalized_hidden_contract(layers, chunk_size, tmp_path):
    model = tiny_model(layers)
    modules = ('attention.o_proj', 'attention.q_proj', 'mlp.shared_experts.gate_proj',
               'mlp.shared_experts.up_proj', 'mlp.shared_experts.down_proj')
    keys = select_targets(model, last_n_layers=min(layers,25), modules=modules)
    tokens = mx.array([[i+1 for i in range(19)]])
    mask = mx.array([[0, 0] + [1]*16])
    with LoRATrainingSession(model, keys, rank=8, scale=.5, kda_policy='chunk_wy',
            projection_chunk_size=chunk_size,
            base_checkpoint='synthetic-kimi-chunked-loss@fixture-v1', source_revision='cpu-fixture',
            base_paths=[tmp_path/'base']) as session:
        session.restore_weights({k:v+(.003 if k.endswith('lora_b') else 0) for k,v in session.weights().items()})
        inputs, targets = tokens[:,:-1], tokens[:,1:]
        hidden = session.forward_hidden(inputs)
        assert hidden.shape == (1,18,64)
        assert_close(model.lm_head(hidden), session.forward(inputs), atol=0, rtol=0)
        old_loss, old_grads = nn.value_and_grad(model, lambda _: masked_cross_entropy(
            session.forward(inputs), targets, mask))(model)
        value, grads = nn.value_and_grad(model, lambda _: session.loss(inputs, targets, mask))(model)
        mx.eval(old_loss, old_grads, value, grads)
        assert_close(old_loss, value, atol=0 if chunk_size is None else 2e-5,
                     rtol=0 if chunk_size is None else 2e-5)
        dense, chunked = map(lambda g:dict(tree_flatten(g)), [old_grads,grads])
        assert set(dense)==set(chunked)==set(session.weights())
        assert len(chunked) == (250 if layers == 27 else 20)
        for key in dense:
            assert_close(dense[key],chunked[key],atol=0 if chunk_size is None else 3e-5,
                         rtol=0 if chunk_size is None else 3e-4)
            assert np.isfinite(np.array(chunked[key])).all()
        assert all(np.any(np.array(v)) for v in chunked.values())
    assert not model.training


def test_explicit_training_policies_are_restored_without_running_unsupported_geometry(tmp_path):
    model = tiny_model(2)
    attn = model.layers[1].attention
    assert attn.training_mla_policy == 'expanded'
    with LoRATrainingSession(model, select_targets(model), rank=8,
            mla_policy='padded_sdpa', projection_chunk_size=128,
            base_checkpoint='synthetic-kimi-policy@fixture-v1', source_revision='cpu-fixture',
            base_paths=[tmp_path/'base']) as session:
        assert session.mla_policy == attn.training_mla_policy == 'padded_sdpa'
        assert session.projection_chunk_size == 128
        assert attn.policy == 'expanded' and attn.flash == 'off'
    assert attn.training_mla_policy == 'expanded'
    assert not model.training


@pytest.mark.parametrize('options', [{'mla_policy':'auto'}, {'mla_policy':None},
                                     {'projection_chunk_size':True}, {'projection_chunk_size':0}])
def test_bad_session_options_do_not_mutate_model(tmp_path, options):
    model = tiny_model(2)
    before = [(module, set(module._no_grad), module.training) for _,module in model.named_modules()]
    with pytest.raises(ValueError):
        LoRATrainingSession(model, select_targets(model),
            base_checkpoint='synthetic-kimi-policy@fixture-v1', source_revision='cpu-fixture',
            base_paths=[tmp_path/'base'], **options)
    assert all(module._no_grad==frozen and module.training==flag for module,frozen,flag in before)


@pytest.mark.parametrize('quantized', [False, True])
def test_bfloat16_head_chunking_preserves_fp32_loss_and_finite_hidden_vjp(quantized):
    head = nn.Linear(64,128,bias=False)
    if quantized:
        head = head.to_quantized(group_size=64,bits=8)
    head.set_dtype(mx.bfloat16)
    head.freeze()
    hidden = mx.random.normal((1,13,64)).astype(mx.bfloat16)
    target = mx.array([[i for i in range(13)]])
    mask = mx.array([[0,0,1,1,0,1,1,1,0,1,0,1,1]])
    dense,dg = mx.value_and_grad(lambda x:masked_cross_entropy(head(x),target,mask))(hidden)
    chunked,cg = mx.value_and_grad(lambda x:chunked_head_cross_entropy(
        head,x,target,mask,chunk_size=5))(hidden)
    mx.eval(dense,dg,chunked,cg)
    assert dense.dtype==chunked.dtype==mx.float32
    assert_close(dense,chunked,atol=2e-6)
    assert_close(dg.astype(mx.float32),cg.astype(mx.float32),atol=2e-4,rtol=.01)
    assert np.isfinite(np.array(cg.astype(mx.float32))).all()
