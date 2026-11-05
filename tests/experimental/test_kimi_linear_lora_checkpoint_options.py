"""CPU metadata compatibility tests; no historical checkpoint migration claim."""
import hashlib
import json

import mlx.core as mx
from mlx import optimizers
import numpy as np
import pytest

from mlx2.experimental.kimi_linear_lora.checkpoint import save_checkpoint, restore_checkpoint
from test_kimi_linear_lora_checkpoint import Session, binding, update, values, assert_values


@pytest.fixture(autouse=True)
def cpu():
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(71)
    yield
    mx.set_default_device(device)


@pytest.mark.parametrize('field,value', [('mla_policy','padded_sdpa'), ('projection_chunk_size',128)])
def test_restore_rejects_actual_session_policy_change_before_mutation(tmp_path, field, value):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    update(session, optimizer, 0)
    p = tmp_path/'checkpoint'
    manifest = save_checkpoint(p,session,optimizer,next_data_cursor=1,provenance=binding(session))
    assert manifest['adapter']['training_mla_policy'] == 'expanded'
    assert manifest['adapter']['projection_chunk_size'] is None
    before = values(session.weights())
    rng = np.array(mx.random.state[0])
    setattr(session,field,value)
    with pytest.raises(ValueError,match='binding mismatch'):
        restore_checkpoint(p,session,optimizer,expected_provenance=binding(session))
    assert_values(before, values(session.weights()))
    np.testing.assert_array_equal(rng,np.array(mx.random.state[0]))


@pytest.mark.parametrize('padded', [False, True])
def test_missing_historical_fields_mean_only_expanded_dense(tmp_path, padded):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    update(session, optimizer, 0)
    p = tmp_path/'checkpoint'
    manifest = save_checkpoint(p,session,optimizer,next_data_cursor=1,provenance=binding(session))
    # Synthetic old-shape metadata with a recomputed legitimate identity;
    # code/source/provenance are still checked exactly by the restore API.
    manifest['adapter'].pop('training_mla_policy')
    manifest['adapter'].pop('projection_chunk_size')
    manifest.pop('checkpoint_id')
    manifest['checkpoint_id'] = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    (p/'checkpoint.json').write_text(json.dumps(manifest))
    if padded:
        session.mla_policy = 'padded_sdpa'
        with pytest.raises(ValueError,match='binding mismatch'):
            restore_checkpoint(p,session,optimizer,expected_provenance=binding(session))
    else:
        restored = restore_checkpoint(p,session,optimizer,expected_provenance=binding(session))
        assert restored == manifest
        assert 'training_mla_policy' not in restored['adapter']


def test_explicit_policy_and_loss_settings_roundtrip(tmp_path):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    session.mla_policy, session.projection_chunk_size = 'padded_sdpa', 128
    update(session,optimizer,0)
    p = tmp_path/'checkpoint'
    provenance = {**binding(session), 'training_loss': {'projection_chunk_size':128}}
    manifest = save_checkpoint(p,session,optimizer,next_data_cursor=1,provenance=provenance)
    restored = restore_checkpoint(p,session,optimizer,expected_provenance=provenance)
    assert restored == manifest
    assert restored['adapter']['training_mla_policy']=='padded_sdpa'
    assert restored['adapter']['projection_chunk_size']==128


@pytest.mark.parametrize('field,value', [
    ('projection_chunk_size', True), ('projection_chunk_size', 1.0),
    ('projection_chunk_size', 0), ('projection_chunk_size', '1'),
    ('training_mla_policy', None), ('training_mla_policy', 'auto')])
def test_invalid_checkpoint_option_schema_before_mutation(tmp_path, field, value):
    session, optimizer = Session(tmp_path), optimizers.Adam(3e-5)
    session.projection_chunk_size = 1
    update(session, optimizer, 0)
    path = tmp_path/'checkpoint'
    manifest = save_checkpoint(path, session, optimizer, next_data_cursor=1,
                               provenance=binding(session))
    manifest['adapter'][field] = value
    manifest.pop('checkpoint_id')
    manifest['checkpoint_id'] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(',',':'), allow_nan=False).encode()).hexdigest()
    (path/'checkpoint.json').write_text(json.dumps(manifest))
    before = values(session.weights())
    rng = np.array(mx.random.state[0])
    with pytest.raises(ValueError, match=field):
        restore_checkpoint(path, session, optimizer, expected_provenance=binding(session))
    assert_values(before, values(session.weights()))
    np.testing.assert_array_equal(rng, np.array(mx.random.state[0]))
