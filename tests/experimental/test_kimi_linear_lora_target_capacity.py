"""CPU fixtures for pre-allocation weight-program target capacity checks."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.experimental.kimi_linear_lora import training
from mlx2.experimental.weight_space.programs import MAX_TARGETS
from mlx2.runtime.models import moe_nax_gather
from mlx2.runtime.models.kimi_linear import Model, ModelArgs


ALL_MODULES = ('attention.q_proj', 'attention.o_proj',
               'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
               'mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj',
               'mlp.shared_experts.down_proj')


@pytest.fixture(autouse=True)
def cpu_only():
    device = mx.default_device()
    mode = moe_nax_gather.set_mode('off')
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(device)
    moe_nax_gather.set_mode(mode)


def inventory_model(count=135):
    # Header-equivalent names only: no checkpoint tensors or fake tensor shapes.
    keys = [f'model.layers.{i}.{suffix}' for i in range(27)
            for suffix in (('attention.q_proj', 'attention.o_proj') +
                (('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj') if i == 0 else
                 ('mlp.shared_experts.gate_proj', 'mlp.shared_experts.up_proj',
                  'mlp.shared_experts.down_proj')))]
    selected = keys[:count]
    return SimpleNamespace(layers=[None] * 27,
                           named_modules=lambda: [(key, None) for key in selected])


def tiny_model():
    args = ModelArgs(model_type='kimi_linear', vocab_size=128, hidden_size=64,
        intermediate_size=64, moe_intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=0, layer_group_size=2, max_position_embeddings=1024,
        rms_norm_eps=1e-5, rope_theta=10000., routed_scaling_factor=2.446,
        head_dim=64, kv_lora_rank=64, qk_rope_head_dim=32, qk_nope_head_dim=32,
        v_head_dim=32, linear_attn_config={'full_attn_layers': [2], 'kda_layers': [1],
            'num_heads': 2, 'head_dim': 32, 'short_conv_kernel_size': 4})
    model = Model(args)
    model.eval()
    return model


@pytest.mark.parametrize('count', [129, 135])
def test_selection_refuses_export_overflow(count):
    with pytest.raises(ValueError, match=f'at most 128 LoRA targets; got {count}'):
        training.select_targets(inventory_model(count), last_n_layers=27,
                                modules=ALL_MODULES)


def test_selection_accepts_export_boundary():
    assert MAX_TARGETS == training.MAX_TARGETS == 128
    keys = training.select_targets(inventory_model(128), last_n_layers=27,
                                   modules=ALL_MODULES)
    assert len(keys) == MAX_TARGETS


@pytest.mark.parametrize('last_n, count, first', [(4, 20, 23), (25, 125, 2)])
def test_legal_suffix_counts_and_order(last_n, count, first):
    keys = training.select_targets(inventory_model(), last_n_layers=last_n,
                                   modules=ALL_MODULES)
    assert len(keys) == len(set(keys)) == count
    assert keys[0] == f'model.layers.{first}.attention.q_proj'
    assert keys[-1] == 'model.layers.26.mlp.shared_experts.down_proj'


@pytest.mark.parametrize('count', [129, 135])
def test_explicit_keys_refused_before_resolution_allocation_or_mutation(count, monkeypatch, tmp_path):
    model = tiny_model()
    before = [(module, module.training, set(module._no_grad))
              for _, module in model.named_modules()]
    original_attentions = [layer.attention.q_proj for layer in model.layers]
    keys = (f'model.layers.{i}.attention.q_proj' for i in range(count))
    def forbidden(*args, **kwargs):
        raise AssertionError('Capacity must be checked before resolution/allocation/mutation')
    with monkeypatch.context() as patch:
        patch.setattr(Model, 'named_modules', forbidden)
        patch.setattr(Model, 'update_modules', forbidden)
        patch.setattr(Model, 'freeze', forbidden)
        patch.setattr(Model, 'train', forbidden)
        patch.setattr(training, 'TrainingLoRALinear', forbidden)
        with pytest.raises(ValueError, match=f'at most 128 LoRA targets; got {count}'):
            training.LoRATrainingSession(model, keys, rank=8,
                base_checkpoint='synthetic-kimi-capacity@fixture-v1',
                source_revision='cpu-fixture-v1', base_paths=[tmp_path / 'base'])
    assert getattr(model, '_kimi_lora_training_session', None) is None
    assert [layer.attention.q_proj for layer in model.layers] == original_attentions
    assert all(module.training == flag and module._no_grad == frozen
               for module, flag, frozen in before)
    assert not (tmp_path / 'base').exists()


def test_explicit_boundary_advances_to_projection_validation(tmp_path):
    model = tiny_model()
    keys = [f'model.layers.{i}.attention.q_proj' for i in range(MAX_TARGETS)]
    with pytest.raises(ValueError, match='Unsupported training/serving projection'):
        training.LoRATrainingSession(model, keys,
            base_checkpoint='synthetic-kimi-capacity@fixture-v1',
            source_revision='cpu-fixture-v1', base_paths=[tmp_path / 'base'])
    assert getattr(model, '_kimi_lora_training_session', None) is None


@pytest.mark.parametrize('keys', [[], ['model.layers.1.attention.q_proj'] * 129])
def test_empty_or_duplicate_keys_keep_explicit_key_validation(keys, tmp_path):
    with pytest.raises(ValueError, match='Explicit unique projection keys'):
        training.LoRATrainingSession(tiny_model(), keys,
            base_checkpoint='synthetic-kimi-capacity@fixture-v1',
            source_revision='cpu-fixture-v1', base_paths=[tmp_path / 'base'])


def test_legal_session_still_wraps_and_restores(tmp_path):
    model = tiny_model()
    before = [(module, module.training, set(module._no_grad))
              for _, module in model.named_modules()]
    keys = training.select_targets(model)
    with training.LoRATrainingSession(model, keys, rank=8,
            base_checkpoint='synthetic-kimi-capacity@fixture-v1',
            source_revision='cpu-fixture-v1', base_paths=[tmp_path / 'base']) as session:
        assert len(session.keys) == 2
        assert len(session.weights()) == 4
        assert all(isinstance(dict(model.named_modules())[key], training.TrainingLoRALinear)
                   for key in keys)
    assert getattr(model, '_kimi_lora_training_session', None) is None
    assert all(module.training == flag and module._no_grad == frozen
               for module, flag, frozen in before)
