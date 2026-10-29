"""Private numerical states may execute, never silently donate exact APC."""
import mlx.core as mx
import numpy as np
import pytest

from test_deepseek_v41_ast_target_cpu import fixture
from mlx2.runtime.models.deepseek_v41_cache import DeepSeekV41LayerCache, STATE_REVISION
from mlx2.runtime.cow_cache import freeze_prompt_cache, snapshot_prompt_cache_descriptors
from mlx2.runtime.checkpoint_publication import require_checkpoint_publishable

PRIVATE='source_ast_native_batched_segments_v1'


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast,'metal_kernel',lambda *a,**k: (_ for _ in ()).throw(AssertionError('GPU forbidden')))


def test_private_profile_refuses_snapshot_ordinary_restore_apcv2_and_descriptor_publication():
    model,_,cache=fixture()
    row=cache[1];row.ast_private_profile=PRIVATE
    assert row.meta_state[0]==STATE_REVISION+':private:'+PRIVATE
    row.validate_state()
    assert 'private'in row.checkpoint_publication_refusal()
    with pytest.raises(ValueError,match='metadata revision'):
        DeepSeekV41LayerCache.from_state(row.state,row.meta_state,xp=mx)
    with pytest.raises(Exception,match='private'):row.snapshot()
    with pytest.raises(Exception,match='private'):row.cache_operations_contract.validate_segmented_publication(row)
    with pytest.raises(Exception,match='private'):require_checkpoint_publishable(cache)
    with pytest.raises(Exception,match='private'):
        freeze_prompt_cache(cache,key='synthetic-private@seed567',tokens=(1,2),cache_type='deepseek_v41',layer_segments=True)
    with pytest.raises(Exception,match='private'):snapshot_prompt_cache_descriptors(cache)


def test_private_ordinary_continuation_retains_flag_and_cannot_publish_afterwards():
    model,_,cache=fixture()
    for row in cache:row.ast_private_profile=PRIVATE
    logits,_=model._forward_impl(mx.array([[3]],mx.int32),cache)
    assert np.isfinite(np.asarray(logits)).all()
    assert all(row.offset==3 and row.ast_private_profile==PRIVATE for row in cache)
    assert all(row.meta_state[0].endswith(':private:'+PRIVATE)for row in cache)
    with pytest.raises(Exception,match='private'):require_checkpoint_publishable(cache)
    # Explicit recovery cloning may retain private scratch; it cannot clear
    # the marker or convert it into a publishable ordinary checkpoint.
    recovery,_,_=snapshot_prompt_cache_descriptors(cache,require_committed=False)
    assert all(row.ast_private_profile==PRIVATE for row in recovery)
    with pytest.raises(Exception,match='private'):require_checkpoint_publishable(recovery)


def test_ordinary_cache_identity_snapshot_and_apcv2_are_unchanged():
    _,_,cache=fixture()
    assert all(row.ast_private_profile is None and row.meta_state[0]==STATE_REVISION for row in cache)
    assert all(row.checkpoint_publication_refusal()is None for row in cache)
    require_checkpoint_publishable(cache)
    for row in cache:row.cache_operations_contract.validate_segmented_publication(row)
    row=cache[1];saved=row.snapshot()
    restored=DeepSeekV41LayerCache.from_state(saved.arrays,saved.metadata,xp=mx)
    assert restored.ast_private_profile is None and restored.meta_state==row.meta_state
    frozen,_=freeze_prompt_cache(cache,key='synthetic-ordinary@seed567',tokens=(1,2),cache_type='deepseek_v41')
    branch=frozen.branch();branch.close();frozen.close()


def test_empty_or_invalid_private_profile_is_not_a_valid_state():
    _,_,cache=fixture()
    for value in('',True,7):
        cache[1].ast_private_profile=value
        with pytest.raises(ValueError,match='private AST'):cache[1].validate_state()
