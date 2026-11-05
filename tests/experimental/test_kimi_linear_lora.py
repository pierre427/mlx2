"""Synthetic CPU oracles, not a Kimi checkpoint qualification result."""
import json

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten
import numpy as np
import pytest

from mlx2.adapters.kimi_linear import parse_policy
from mlx2.experimental.kimi_linear_lora import (
    LoRATrainingSession, masked_cross_entropy, select_targets,
)
from mlx2.runtime.lora import install_lora
from mlx2.runtime.models.kimi_linear import Model, ModelArgs
from mlx2.runtime.models import moe_nax_gather


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    previous_mode = moe_nax_gather.set_mode('off')
    mx.set_default_device(mx.cpu)
    mx.random.seed(118)
    yield
    mx.set_default_device(previous)
    moe_nax_gather.set_mode(previous_mode)


def model_fixture(quantized=False):
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
    if quantized:
        nn.quantize(model, group_size=64, bits=8,
                    class_predicate=lambda _, module: hasattr(module, 'to_quantized')
                    and module.weight.shape[-1] % 64 == 0)
    model.configure(parse_policy({'mla': 'expanded', 'mla_flash': 'off', 'kda': 'reference'}))
    mx.eval(model.parameters())
    return model


def session_for(model, tmp_path, **kwargs):
    return LoRATrainingSession(model, select_targets(model, last_n_layers=kwargs.pop('last_n_layers', 1)),
        base_checkpoint='synthetic-kimi-cpu@fixture-v1', source_revision='91379787e07099213fb27b48ec5bd7bdc6a17d62',
        base_paths=(tmp_path / 'protected-base',), **kwargs)


def snapshot(model):
    return {key: np.array(value) for key, value in tree_flatten(model.parameters())}


def close(a, b, atol=2e-6):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=2e-5)


@pytest.mark.parametrize('quantized', [False, True])
def test_zero_lora_gradients_and_frozen_base(quantized, tmp_path):
    model = model_fixture(quantized)
    original = snapshot(model)
    tokens = mx.array([[1, 2, 3, 4, 5]])
    baseline = model(tokens)
    mx.eval(baseline)
    with session_for(model, tmp_path, rank=4) as session:
        close(session.forward(tokens), baseline, atol=0)
        weights = session.weights()
        assert all(np.any(np.array(value)) == key.endswith('lora_a') for key, value in weights.items())
        loss_fn = nn.value_and_grad(model, lambda _: masked_cross_entropy(
            session.forward(tokens[:, :-1]), tokens[:, 1:], mx.ones((1, 4))))
        optimizer = optimizers.SGD(learning_rate=.05)
        loss, grads = loss_fn(model)
        mx.eval(loss, grads)
        assert np.isfinite(loss.item())
        flat_grads = dict(tree_flatten(grads))
        assert all(np.all(np.isfinite(np.array(g))) for g in flat_grads.values())
        assert all(np.any(np.array(g)) for k, g in flat_grads.items() if k.endswith('lora_b'))
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        _, grads = loss_fn(model)
        mx.eval(grads)
        assert all(np.any(np.array(g)) for k, g in tree_flatten(grads) if k.endswith('lora_a'))
        after = dict(tree_flatten(model.parameters()))
        for key, before in original.items():
            wrapped_key = next((key.replace(target + '.', target + '.linear.', 1)
                                for target in session.keys if key.startswith(target + '.')), key)
            np.testing.assert_array_equal(np.array(after[wrapped_key]), before)
    assert not model.training
    for key, value in snapshot(model).items():
        np.testing.assert_array_equal(value, original[key])
    close(model(tokens), baseline, atol=0)


@pytest.mark.parametrize('last_n_layers', [1, 2])
def test_checkpoint_forward_and_gradients_agree(last_n_layers, tmp_path):
    model = model_fixture(True)
    # Cross the sorted-expert threshold as a real prompt does.
    tokens = mx.array([[1, 3, 2, 7, 4, 5, 8, 9, 11, 12, 4, 2, 7]])
    with session_for(model, tmp_path, last_n_layers=last_n_layers) as session:
        session.restore_weights({key: value + (.01 if key.endswith('lora_b') else 0)
                                 for key, value in session.weights().items()})
        def loss(checkpoint):
            return nn.value_and_grad(model, lambda _: masked_cross_entropy(
                session.forward(tokens[:, :-1], checkpoint_layers=checkpoint),
                tokens[:, 1:], mx.array([[0] + [1] * 11])))(model)
        a, ga = loss(False)
        b, gb = loss(True)
        close(a, b)
        for (_, x), (_, y) in zip(tree_flatten(ga), tree_flatten(gb), strict=True):
            close(x, y)
        # Every call starts recurrence from zero; no training cache persists.
        close(session.forward(tokens), session.forward(tokens))


def test_policy_and_trainability_restoration(tmp_path):
    model = model_fixture()
    model.freeze()
    model.layers[0].attention.fused_projections = True
    model.layers[0].attention.prepare_projections()
    model.layers[0].attention.fused_norm = True
    model.layers[0].attention.reference = False
    model.layers[1].attention.policy = 'auto'
    model.layers[1].attention.flash = 'all'
    with session_for(model, tmp_path) as session:
        assert session.first_layer == 1
        assert model.layers[0].attention.fused_projections
        assert model.layers[0].attention.fused_norm
        assert not model.layers[0].attention.reference
        assert not model.layers[0].training
        assert model.layers[1].training
        assert model.layers[1].attention.policy == 'expanded'
        assert model.layers[1].attention.flash == 'off'
    assert not model.training
    assert not tree_flatten(model.trainable_parameters())
    assert model.layers[1].attention.policy == 'auto'
    assert model.layers[1].attention.flash == 'all'
    with session_for(model, tmp_path, last_n_layers=2):
        assert model.layers[0].attention.reference
        assert not model.layers[0].attention.fused_projections
        assert not model.layers[0].attention.fused_norm
    assert model.layers[0].attention.fused_projections
    assert model.layers[0].attention.fused_norm
    assert not model.layers[0].attention.reference
    assert not model.layers[1].mlp.gate.training_indices


def test_program_serving_roundtrip_and_protected_base(tmp_path):
    model = model_fixture(True)
    tokens = mx.array([[1, 2, 4, 5]])
    baseline = model(tokens)
    mx.eval(baseline)
    output = tmp_path / 'adapter'
    with session_for(model, tmp_path, scale=2.0) as session:
        session.restore_weights({key: value + (.02 if key.endswith('lora_b') else 0)
                                 for key, value in session.weights().items()})
        trained = session.forward(tokens)
        mx.eval(trained)
        with pytest.raises(ValueError, match='protected base'):
            session.export_program(tmp_path / 'protected-base' / 'adapter', {})
        program = session.export_program(output, {'fixture': True})
        assert program.checkpoint == 'synthetic-kimi-cpu@fixture-v1'
    assert not (tmp_path / 'protected-base').exists()
    config = json.loads((output / 'adapter_config.json').read_text())
    assert config['lora_parameters']['scale'] == 1.0
    serving = install_lora(model, name='fixture', path=output,
                           base_checkpoint='synthetic-kimi-cpu@fixture-v1')
    try:
        close(model(tokens), trained, atol=0)
        assert serving.program_application_count == 2
    finally:
        serving.restore(model)
    close(model(tokens), baseline, atol=0)


def test_reject_unsafe_keys_nested_session_and_bad_snapshot(tmp_path):
    model = model_fixture()
    for key in ('model.layers.1.attention.kv_b_proj', 'model.layers.1.mlp.gate.gate_proj',
                'model.layers.1.mlp.switch_mlp.gate_proj'):
        with pytest.raises(ValueError, match='Unsupported'):
            LoRATrainingSession(model, [key], base_checkpoint='synthetic@fixture',
                source_revision='fixture', base_paths=[tmp_path / 'base'])
    with session_for(model, tmp_path) as session:
        with pytest.raises(ValueError, match='already active'):
            session_for(model, tmp_path)
        with pytest.raises(ValueError, match='snapshot'):
            session.restore_weights({})
    with pytest.raises(ValueError, match='not active'):
        session.forward(mx.array([[1]]))


def test_loss_mask_ignores_unselected_targets():
    logits = mx.array([[[1., 2., 3.], [5., -1., 2.]]])
    mask = mx.array([[0, 1]])
    a = masked_cross_entropy(logits, mx.array([[0, 2]]), mask)
    b = masked_cross_entropy(logits, mx.array([[1, 2]]), mask)
    close(a, b, atol=0)
    with pytest.raises(ValueError, match='Expected logits'):
        masked_cross_entropy(logits, mx.array([1]), mx.array([1]))


def test_bfloat16_quantized_backward_and_all_token_logits(tmp_path):
    from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows

    model = model_fixture(True)
    model.set_dtype(mx.bfloat16)
    tokens = mx.array([[1, 4, 7, 2]])
    baseline = model(tokens)
    mx.eval(baseline)
    with session_for(model, tmp_path, last_n_layers=2) as session:
        with prefill_checkpoint_rows([4], width=4) as request:
            got = session.forward(tokens)
            assert got.shape == (1, 4, 128)
            assert request.applied == 0
        np.testing.assert_array_equal(np.array(got.astype(mx.float32)),
                                      np.array(baseline.astype(mx.float32)))
        loss, grads = nn.value_and_grad(model, lambda _: masked_cross_entropy(
            session.forward(tokens[:, :-1]), tokens[:, 1:], mx.ones((1, 3))))(model)
        mx.eval(loss, grads)
        assert np.isfinite(loss.item())
        assert all(np.all(np.isfinite(np.array(g))) for _, g in tree_flatten(grads))
        assert all(np.any(np.array(g)) for k, g in tree_flatten(grads) if k.endswith('lora_b'))


def test_refuse_unverified_quantized_kernel_before_mutating(tmp_path):
    model = model_fixture(True)
    model.layers[1].attention.q_proj._lane_tiled = True
    with pytest.raises(ValueError, match='stock, untiled'):
        session_for(model, tmp_path)
    assert not model.training
    assert type(model.layers[1].attention.q_proj) is nn.QuantizedLinear


def test_repeated_adam_updates_ignore_dynamic_observability(tmp_path):
    model = model_fixture(True)
    original = snapshot(model)
    optimizer = optimizers.Adam(learning_rate=3e-5)
    with session_for(model, tmp_path, last_n_layers=2) as session:
        expected_keys = set(session.weights())
        value_grad = nn.value_and_grad(model, lambda _, tokens: masked_cross_entropy(
            session.forward(tokens[:, :-1]), tokens[:, 1:],
            mx.ones((1, tokens.shape[1] - 1))))
        # Start without a warmup, introduce new route and shape counters, and
        # cross the unsorted/sorted expert boundary in a later update.
        for step, length in enumerate((4, 13, 2, 8)):
            model.layers[1].attention.calls[f'diagnostic_step_{step}'] += 1
            tokens = mx.array([[1 + i % 16 for i in range(length)]])
            loss, grads = value_grad(model, tokens)
            assert set(dict(tree_flatten(grads))) == expected_keys
            optimizer.update(model, grads)
            mx.eval(loss, model.trainable_parameters(), optimizer.state)
            assert np.isfinite(loss.item())
            assert optimizer.step.item() == step + 1
        assert model.layers[1].attention.calls['expanded_sdpa'] > 0
        assert all(model.layers[1].attention.calls[f'diagnostic_step_{step}'] == 1 for step in range(4))
        assert not any('calls' in key for key, _ in tree_flatten(optimizer.state))
    for key, value in snapshot(model).items():
        np.testing.assert_array_equal(value, original[key])


def test_real_session_checkpoint_covers_only_factors_and_moments(tmp_path):
    from mlx2.experimental.kimi_linear_lora.checkpoint import save_checkpoint, restore_checkpoint

    model = model_fixture(True)
    tokens = mx.array([[1, 2, 3, 4]])
    optimizer = optimizers.Adam(3e-5)
    with session_for(model, tmp_path) as session:
        _, grads = nn.value_and_grad(model, lambda _: masked_cross_entropy(
            session.forward(tokens[:, :-1]), tokens[:, 1:], mx.ones((1, 3))))(model)
        optimizer.update(model, grads)
        mx.eval(model.trainable_parameters(), optimizer.state)
        provenance = {'checkpoint': session.base_checkpoint, 'source_revision': session.source_revision,
            'code_sha256': {'checkpoint.py': 'a' * 64}, 'data_sha256': 'b' * 64, 'spec_sha256': 'c' * 64}
        manifest = save_checkpoint(tmp_path / 'state-1', session, optimizer,
                                   next_data_cursor=1, provenance=provenance)
        assert manifest['state_bytes'] < 32_768
        expected = {key: np.array(value) for key, value in session.weights().items()}
        session.restore_weights({key: mx.zeros_like(value) for key, value in session.weights().items()})
        fresh_optimizer = optimizers.Adam(3e-5)
        restored = restore_checkpoint(tmp_path / 'state-1', session, fresh_optimizer,
                                      expected_provenance=provenance)
        assert restored['next_data_cursor'] == 1
        assert fresh_optimizer.step.item() == 1
        for key, value in session.weights().items():
            np.testing.assert_array_equal(np.array(value), expected[key])


@pytest.mark.parametrize('length', [8, 33])
def test_wy_training_gradients_match_reference_and_restore(length, tmp_path):
    model = model_fixture(True)
    tokens = mx.array([[i % 100 + 1 for i in range(length)]])
    snapshots = []
    for policy in ('reference', 'chunk_wy'):
        mx.random.seed(913)
        with session_for(model, tmp_path, last_n_layers=2, kda_policy=policy) as session:
            session.restore_weights({key: value + (.01 if key.endswith('lora_b') else 0)
                                     for key, value in session.weights().items()})
            loss, grads = nn.value_and_grad(model, lambda _: masked_cross_entropy(
                session.forward(tokens[:, :-1]), tokens[:, 1:], mx.ones((1, length-1))))(model)
            mx.eval(loss, grads)
            snapshots.append((float(loss.item()), {k:np.array(v) for k,v in tree_flatten(grads)}))
            if policy == 'chunk_wy' and length >= 18:
                assert model.layers[0].attention.calls['chunk_wy'] > 0
        assert not model.layers[0].attention.chunk_wy
        assert model.layers[0].attention.reference
    assert snapshots[0][0] == pytest.approx(snapshots[1][0], abs=2e-5)
    for key in snapshots[0][1]:
        np.testing.assert_allclose(snapshots[0][1][key], snapshots[1][1][key], atol=3e-5, rtol=3e-4)


def test_invalid_training_kda_policy_refused(tmp_path):
    with pytest.raises(ValueError, match='Training KDA policy'):
        session_for(model_fixture(), tmp_path, kda_policy='vector_kernel')
