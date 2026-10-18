import dataclasses

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import inspect_apc_capabilities
from mlx2.runtime.cache_operations import (
    CacheOperationsUnsupported,
    KV_OPERATIONS,
    QSA_KV_OPERATIONS,
    ROTATING_KV_OPERATIONS,
    cache_operations,
)
from mlx2.runtime.cow_cache import COWCacheUnsupported, freeze_prompt_cache
from mlx2.runtime.hybrid_speculative import _reject_unsupported_self_mtp_caches
from mlx2.runtime.models.cache import ArraysCache, KVCache, RotatingKVCache
from mlx2.runtime.models.nemotron_h import NemotronMambaCache
from mlx2.runtime.models.qwen4_exp import QSAKVCache, Qwen4ArraysCache
from mlx2.runtime.models.qwen4_exp import QSAQuantizedKVCache
from mlx2.runtime.pld import _rewind, _snapshot, _validate_cache
from mlx2.runtime.qsa_shared_suffix import SharedSuffixQSAKVCache


def _append(cache, count):
    keys = mx.zeros((1, 1, count, 2), dtype=mx.float32)
    values = mx.ones((1, 1, count, 2), dtype=mx.float32)
    cache.update_and_fetch(keys, values)


def test_contract_is_immutable_and_binds_state_layout_and_numerics():
    with pytest.raises(dataclasses.FrozenInstanceError):
        KV_OPERATIONS.cache_layout = "drift"
    assert KV_OPERATIONS.state_revision
    assert KV_OPERATIONS.cache_layout
    assert KV_OPERATIONS.numerics
    assert len(KV_OPERATIONS.fingerprint) == 64


def test_unknown_subclass_fails_closed_in_apc_cow_and_pld():
    class UnknownKV(KVCache):
        pass

    cache = UnknownKV()
    assert not inspect_apc_capabilities([cache]).exact_prefix
    with pytest.raises(
        COWCacheUnsupported, match="checkpoint-publication operations declaration"
    ):
        freeze_prompt_cache([cache], key="k", tokens=(), cache_type="unknown")
    with pytest.raises(NotImplementedError, match="declared exact rollback"):
        _validate_cache([cache])


def test_own_class_declaration_admits_custom_family_without_name_inference():
    class DeclaredKV(KVCache):
        cache_operations_contract = KV_OPERATIONS

    cache = DeclaredKV()
    assert cache_operations(cache, capability="test") is KV_OPERATIONS
    assert inspect_apc_capabilities([cache]).exact_prefix
    frozen, _ = freeze_prompt_cache(
        [cache], key="k", tokens=(), cache_type="declared"
    )
    frozen.close()


def test_self_mtp_cache_admission_ignores_hostile_windowed_looking_name():
    class RotatingSinkWindowLookalike(KVCache):
        cache_operations_contract = KV_OPERATIONS

    _reject_unsupported_self_mtp_caches([RotatingSinkWindowLookalike()])


def test_self_mtp_cache_admission_rejects_innocuous_windowed_name_by_topology():
    class OrdinaryLookingCache(RotatingKVCache):
        cache_operations_contract = ROTATING_KV_OPERATIONS

    with pytest.raises(ValueError, match=r"family=rotating_kv"):
        _reject_unsupported_self_mtp_caches([OrdinaryLookingCache(max_size=8)])


def test_self_mtp_cache_admission_fails_closed_without_contract():
    class PlainLookingCache(KVCache):
        pass

    with pytest.raises(ValueError, match="explicit cache operations contract"):
        _reject_unsupported_self_mtp_caches([PlainLookingCache()])


def test_pld_rollback_uses_declared_provider_operation_and_is_exact():
    cache = KVCache()
    _append(cache, 2)
    snapshot = _snapshot([cache])
    _append(cache, 3)
    assert cache.offset == 5
    _rewind([cache], snapshot)
    assert cache.offset == 2


def test_pld_rollback_preflights_every_provider_before_mutating_any_cache():
    class UnknownKV(KVCache):
        pass

    first = KVCache()
    _append(first, 2)
    # Snapshot below the live offset, so a restore that ran before the
    # refusal would be visible as offset 2.
    first_snapshot = _snapshot([first])[0]
    _append(first, 3)
    assert first.offset == 5
    unknown = UnknownKV()
    with pytest.raises(CacheOperationsUnsupported, match="explicit"):
        _rewind([first, unknown], [first_snapshot, ("invalid",)])
    assert first.offset == 5


def test_pld_rollback_refuses_cross_family_cache_replacement():
    original = KVCache()
    _append(original, 2)
    snapshot = _snapshot([original])
    replacement = QSAKVCache()
    _append(replacement, 5)
    replacement.update_index_keys(mx.ones((1, 5, 2), dtype=mx.float32))

    with pytest.raises(CacheOperationsUnsupported, match="provider revision|owner"):
        _rewind([replacement], snapshot)
    assert replacement.offset == 5
    assert replacement.index_keys.shape[1] == 5


def test_pld_rollback_refuses_same_family_cache_replacement():
    original = KVCache()
    _append(original, 2)
    snapshot = _snapshot([original])
    replacement = KVCache()
    _append(replacement, 5)

    with pytest.raises(CacheOperationsUnsupported, match="owner changed"):
        _rewind([replacement], snapshot)
    assert replacement.offset == 5


def test_pld_rollback_refuses_undeclared_cache_list_subclass():
    from mlx2.runtime.models.cache import CacheList

    class StatefulCacheList(CacheList):
        pass

    with pytest.raises(CacheOperationsUnsupported, match="explicit"):
        _snapshot([StatefulCacheList(KVCache())])


@pytest.mark.parametrize("snapshot", [(), ("list",), ("list", ())])
def test_pld_rollback_rejects_malformed_container_snapshot(snapshot):
    from mlx2.runtime.models.cache import CacheList

    cache = CacheList(KVCache())
    with pytest.raises(CacheOperationsUnsupported, match="malformed|container changed"):
        _rewind([cache], [snapshot])


def test_current_qwen4_cache_families_declare_own_contracts():
    assert (
        cache_operations(Qwen4ArraysCache(4), capability="test").family
        == "recurrent"
    )
    assert cache_operations(QSAKVCache(), capability="test") is QSA_KV_OPERATIONS
    assert (
        cache_operations(QSAQuantizedKVCache(), capability="test").family
        == "indexed_quantized_kv"
    )
    assert "cache_operations_contract" in SharedSuffixQSAKVCache.__dict__


def test_arrays_cache_row_transforms_preserve_provider_identity():
    rows = []
    for _ in range(2):
        cache = ArraysCache(2)
        cache.apcv2_recurrent_kind = "mamba"
        cache.apcv2_layer_index = 7
        rows.append(cache)

    merged = ArraysCache.merge(rows)
    extracted = merged.extract(0)

    for cache in (merged, extracted):
        assert cache.apcv2_recurrent_kind == "mamba"
        assert cache.apcv2_layer_index == 7
        assert (
            cache_operations(cache, capability="test").fingerprint
            == cache_operations(rows[0], capability="test").fingerprint
        )


def test_adapter_owned_mamba_cache_keeps_contract_through_row_transforms():
    rows = []
    for _ in range(2):
        cache = NemotronMambaCache(2)
        cache.apcv2_recurrent_kind = "mamba"
        cache.apcv2_layer_index = 7
        rows.append(cache)

    merged = NemotronMambaCache.merge(rows)
    extracted = merged.extract(0)

    assert type(merged) is NemotronMambaCache
    assert type(extracted) is NemotronMambaCache
    assert cache_operations(merged, capability="test").family == "recurrent"
    assert (
        cache_operations(extracted, capability="test").fingerprint
        == cache_operations(rows[0], capability="test").fingerprint
    )


@pytest.mark.parametrize(
    ("attribute", "left", "right"),
    [
        ("apcv2_recurrent_kind", "mamba", None),
        ("apcv2_layer_index", 7, 8),
    ],
)
def test_arrays_cache_merge_refuses_mixed_provider_identity(
    attribute, left, right
):
    rows = [ArraysCache(2), ArraysCache(2)]
    for cache in rows:
        cache.apcv2_recurrent_kind = "mamba"
        cache.apcv2_layer_index = 7
    if right is None:
        delattr(rows[1], attribute)
    else:
        setattr(rows[0], attribute, left)
        setattr(rows[1], attribute, right)

    with pytest.raises(ValueError, match=rf"provider identity {attribute} differs"):
        ArraysCache.merge(rows)


def test_checkpoint_and_segmented_publication_have_distinct_guards():
    recurrent = ArraysCache(1)
    recurrent.start_speculation()
    operations = cache_operations(recurrent, capability="test")
    with pytest.raises(CacheOperationsUnsupported, match="speculation transaction"):
        operations.validate_publication(recurrent)
    # Segmented verify requires its private rollback epoch to remain active.
    operations.validate_segmented_publication(recurrent)

    qsa = QSAKVCache()
    qsa._mtp_share_topk = True
    operations = cache_operations(qsa, capability="test")
    with pytest.raises(CacheOperationsUnsupported, match="QSA/MTP transaction"):
        operations.validate_publication(qsa)
    with pytest.raises(CacheOperationsUnsupported, match="QSA/MTP transaction"):
        operations.validate_segmented_publication(qsa)


def test_inherited_qsa_declaration_is_not_accepted():
    class UnknownQSA(QSAKVCache):
        pass

    with pytest.raises(CacheOperationsUnsupported, match="explicit"):
        cache_operations(UnknownQSA(), capability="test")


def test_declared_projection_fails_closed_when_a_required_field_is_absent():
    class Incomplete:
        keys = None

    with pytest.raises(CacheOperationsUnsupported, match="declared cache field"):
        KV_OPERATIONS.projection(Incomplete(), KV_OPERATIONS.planes[0])


@pytest.mark.parametrize(
    "payload",
    [
        ("kv_trim",),
        ("kv_trim", None),
        ("kv_trim", 1, "extra"),
    ],
)
def test_kv_rollback_snapshot_shape_and_types_are_validated(payload):
    cache = KVCache()
    snapshot = ("mlx2.rollback.v1", KV_OPERATIONS.fingerprint, cache, payload)
    with pytest.raises(CacheOperationsUnsupported, match="integer position"):
        KV_OPERATIONS.validate_rollback_snapshot(snapshot, cache=cache)
