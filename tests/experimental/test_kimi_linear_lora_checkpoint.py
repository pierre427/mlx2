"""CPU checkpoint fidelity on synthetic adapter factors, no model qualification."""
import copy
import os
from pathlib import Path

import mlx.core as mx
from mlx import nn, optimizers
from mlx.utils import tree_flatten, tree_unflatten
import numpy as np
import pytest

from mlx2.experimental.kimi_linear_lora.checkpoint import save_checkpoint, restore_checkpoint


@pytest.fixture(autouse=True)
def cpu():
    prior = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(51)
    yield
    mx.set_default_device(prior)


class Session:
    def __init__(self, tmp_path):
        self.model = nn.Module()
        self.model.adapter = nn.Module()
        self.model.adapter.lora_a = mx.random.normal((8, 2))
        self.model.adapter.lora_b = mx.zeros((2, 4))
        self.keys, self.rank, self.scale = ('adapter',), 2, 1.0
        self.base_checkpoint = 'synthetic-kimi-cpu@checkpoint-fixture-v1'
        self.source_revision = '91379787e07099213fb27b48ec5bd7bdc6a17d62'
        self.base_paths = (tmp_path / 'base',)

    def weights(self):
        return dict(tree_flatten(self.model.trainable_parameters()))

    def restore_weights(self, values):
        if set(values) != set(self.weights()):
            raise ValueError('coverage')
        self.model.update(tree_unflatten(list(values.items())))


def binding(session):
    return {'checkpoint': session.base_checkpoint, 'source_revision': session.source_revision,
            'code_sha256': {'training.py': '1' * 64, 'checkpoint.py': '2' * 64},
            'data_sha256': '3' * 64, 'spec_sha256': '4' * 64}


def update(session, optimizer, cursor):
    inputs = mx.random.normal((3 + cursor % 2, 8))
    targets = mx.random.normal((3 + cursor % 2, 4))
    def loss(model):
        out = (inputs @ model.adapter.lora_a) @ model.adapter.lora_b
        return mx.mean((out - targets) ** 2)
    value, grads = nn.value_and_grad(session.model, loss)(session.model)
    optimizer.update(session.model, grads)
    mx.eval(value, session.model.trainable_parameters(), optimizer.state)
    return value.item()


def values(tree):
    return {key: np.array(value) for key, value in tree_flatten(tree)}


def assert_values(a, b):
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_resume_matches_uninterrupted_adam_and_rng(tmp_path):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    for cursor in range(4):
        update(session, optimizer, cursor)
    provenance = binding(session)
    output = tmp_path / 'checkpoint-4'
    manifest = save_checkpoint(output, session, optimizer, next_data_cursor=4, provenance=provenance)
    assert manifest['optimizer_step'] == 4
    assert manifest['state_bytes'] < 16_384
    assert set(p.name for p in output.iterdir()) == {'checkpoint.json', 'state.safetensors'}
    expected_losses = [update(session, optimizer, cursor) for cursor in range(4, 7)]
    expected_factors = values(session.weights())
    expected_optimizer = values(optimizer.state)
    expected_random = np.array(mx.random.normal((5,)))
    resumed, resumed_optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    loaded = restore_checkpoint(output, resumed, resumed_optimizer, expected_provenance=provenance)
    assert loaded == manifest
    got_losses = [update(resumed, resumed_optimizer, cursor) for cursor in range(loaded['next_data_cursor'], 7)]
    np.testing.assert_array_equal(got_losses, expected_losses)
    assert_values(values(resumed.weights()), expected_factors)
    assert_values(values(resumed_optimizer.state), expected_optimizer)
    np.testing.assert_array_equal(np.array(mx.random.normal((5,))), expected_random)


@pytest.mark.parametrize('field', ['checkpoint', 'source_revision', 'code_sha256', 'data_sha256', 'spec_sha256'])
def test_binding_mismatch_does_not_mutate_state(tmp_path, field):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    update(session, optimizer, 0)
    provenance = binding(session)
    output = tmp_path / 'checkpoint-1'
    save_checkpoint(output, session, optimizer, next_data_cursor=1, provenance=provenance)
    expected = copy.deepcopy(provenance)
    if field == 'code_sha256':
        expected[field]['training.py'] = 'f' * 64
    elif field.endswith('sha256'):
        expected[field] = 'f' * 64
    else:
        expected[field] = 'mismatch'
    before, before_optimizer = values(session.weights()), values(optimizer.state)
    before_rng = values(list(mx.random.state))
    with pytest.raises(ValueError, match='provenance|binding'):
        restore_checkpoint(output, session, optimizer, expected_provenance=expected)
    assert_values(values(session.weights()), before)
    assert_values(values(optimizer.state), before_optimizer)
    assert_values(values(list(mx.random.state)), before_rng)


def test_corruption_and_optimizer_mismatch_fail_before_restore(tmp_path):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    update(session, optimizer, 0)
    provenance = binding(session)
    output = tmp_path / 'checkpoint-1'
    save_checkpoint(output, session, optimizer, next_data_cursor=1, provenance=provenance)
    with pytest.raises(ValueError, match='binding'):
        restore_checkpoint(output, session, optimizers.Adam(1e-4), expected_provenance=provenance)
    payload = output / 'state.safetensors'
    with payload.open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'!')
    with pytest.raises(ValueError, match='payload hash'):
        restore_checkpoint(output, session, optimizer, expected_provenance=provenance)


def test_no_overwrite_or_base_writes(tmp_path):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    update(session, optimizer, 0)
    kwargs = {'next_data_cursor': 1, 'provenance': binding(session)}
    with pytest.raises(ValueError, match='protected base'):
        save_checkpoint(tmp_path / 'base' / 'checkpoint', session, optimizer, **kwargs)
    assert not (tmp_path / 'base').exists()
    output = tmp_path / 'checkpoint'
    output.mkdir()
    with pytest.raises(ValueError, match='already exists'):
        save_checkpoint(output, session, optimizer, **kwargs)
    with pytest.raises(ValueError, match='constant learning rate'):
        save_checkpoint(tmp_path / 'scheduled', session, optimizers.Adam(lambda step: step + 1.), **kwargs)


def test_exclusive_reservation_refuses_racing_empty_directory(tmp_path, monkeypatch):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    update(session, optimizer, 0)
    output = tmp_path / 'racing'
    mkdir = os.mkdir
    def race(path, *args, **kwargs):
        if Path(path) == output:
            mkdir(path)
        return mkdir(path, *args, **kwargs)
    monkeypatch.setattr(os, 'mkdir', race)
    with pytest.raises(FileExistsError):
        save_checkpoint(output, session, optimizer, next_data_cursor=1, provenance=binding(session))
    assert output.is_dir() and not list(output.iterdir())


def test_interrupted_publication_remains_incomplete_and_refused(tmp_path, monkeypatch):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    update(session, optimizer, 0)
    output = tmp_path / 'interrupted'
    replace = os.replace
    def interrupt(source, target):
        if Path(target) == output / 'checkpoint.json':
            raise OSError('injected publish interruption')
        return replace(source, target)
    monkeypatch.setattr(os, 'replace', interrupt)
    with pytest.raises(OSError, match='injected publish'):
        save_checkpoint(output, session, optimizer, next_data_cursor=1, provenance=binding(session))
    assert (output / 'state.safetensors').is_file()
    assert not (output / 'checkpoint.json').exists()
    with pytest.raises(ValueError, match='missing'):
        restore_checkpoint(output, session, optimizer, expected_provenance=binding(session))
    with pytest.raises(ValueError, match='already exists'):
        save_checkpoint(output, session, optimizer, next_data_cursor=1, provenance=binding(session))
