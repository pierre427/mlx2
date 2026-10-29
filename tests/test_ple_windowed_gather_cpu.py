"""Synthetic shard fixtures; no model or GPU qualification."""
import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.apc_numerics import CANDIDATE_ENV
from mlx2.runtime.models import qwen4_ple_window as window
from mlx2.runtime.models.qwen4_exp import ShardedEmbedding


@pytest.mark.parametrize("bits", [None, 4, 8])
def test_exact_rows_across_head_and_shard_boundaries_without_host_id_reads(bits, monkeypatch):
    table = ShardedEmbedding(128, 32, 8)
    for shard in range(8):
        getattr(table, f"shard_{shard}").weight = (
            mx.arange(16 * 32).reshape(16, 32) / 100 + shard
        )
    if bits:
        nn.quantize(table, bits=bits, group_size=32)
    ranges = ((0, 33), (33, 80), (80, 128))
    ids = mx.array([[[0, 33, 80], [15, 47, 95], [16, 64, 96], [32, 79, 127]]])
    # Build the independent ordinary gather before refusing host evaluations.
    expected = table(ids)
    with monkeypatch.context() as guard:
        guard.setattr(mx, "eval", lambda *args: (_ for _ in ()).throw(AssertionError("host sync")))
        got = window.lookup(table, ids, ranges, metal=False)
    assert mx.array_equal(got, expected).item()
    assert window.plan_windows(128, 8, ranges) == ((0, 3), (2, 5), (5, 8))


def test_geometry_is_bounded_and_policy_is_off_by_default():
    table = ShardedEmbedding(128, 32, 8)
    assert window.lookup(table, mx.zeros((1, 65, 1), mx.int64),
                         ((0, 128),), metal=False) is None
    with pytest.raises(ValueError, match="bounded"):
        window.plan_windows(160, 10, ((0, 160),))
    for ranges in ((), ((-1, 8),), ((0, 129),), ((8, 8),)):
        with pytest.raises(ValueError):
            window.plan_windows(128, 8, ranges)
    assert "ple_windowed_gather" not in FlashNextPolicy().as_dict()
    assert CANDIDATE_ENV["MLX_QWEN4_PLE_WINDOWED_GATHER"] == "flag"
    # The current FlashNext adapter installs file-backed PLE unconditionally.
    # Resident-only experiments must not advertise an unusable serving policy.
    with pytest.raises(ValueError):
        FlashNextPolicy.from_mapping({"ple_windowed_gather": True})


def test_hash_metadata_reload_requires_explicit_preparation_and_bad_bounds_refuse(monkeypatch):
    from qsa_oracle import tiny_args

    from mlx2.runtime.models.qwen4_exp import NGramEmbedding

    module = NGramEmbedding(tiny_args(), 128, 0, 0)
    module.windowed_gather = True
    module.hash_backend = "metal"
    monkeypatch.setattr(window, "eligible", lambda *a, **k: True)
    ids = mx.array([[1, 2]])
    assert not module._window_eligible(ids)
    with pytest.raises(ValueError, match="resident shard preparation"):
        module.prepare_windowed_gather()
    module.prepare_windowed_gather(resident_shards_prepared=True)
    assert module._window_eligible(ids)

    module.ngram_heads_offsets = module.ngram_heads_offsets + 0
    assert not module._window_eligible(ids)
    module.prepare_windowed_gather(resident_shards_prepared=True)
    assert module._window_eligible(ids)
    module.ngram_size = 4
    assert not module._window_eligible(ids)
    module.ngram_heads_offsets = mx.full((module.ngram_heads,), -1, mx.int64)
    with pytest.raises(ValueError, match="outside"):
        module.prepare_windowed_gather(resident_shards_prepared=True)


@pytest.mark.parametrize("parameter", ["scales", "biases"])
def test_quantized_resident_binding_cannot_survive_parameter_replacement(parameter):
    table = ShardedEmbedding(128, 32, 8)
    nn.quantize(table, bits=4, group_size=32)
    owner = window.ResidentBinding(table)
    assert owner.matches(table)
    shard = table.shard_0
    setattr(shard, parameter, getattr(shard, parameter) + 0)
    assert not owner.matches(table)


def test_resident_binding_does_not_pin_replaced_weights():
    from weakref import ref

    table = ShardedEmbedding(128, 32, 8)
    owner = window.ResidentBinding(table)
    previous = ref(table.shard_0.weight)
    table.shard_0.weight = mx.zeros((16, 32))
    assert previous() is None
    assert not owner.matches(table)


def test_live_shard_weight_replacement_never_reuses_old_values():
    table = ShardedEmbedding(128, 32, 8)
    ids, ranges = mx.array([[[3]]]), ((0, 16),)
    old = window.lookup(table, ids, ranges, metal=False)
    mx.eval(old)
    table.shard_0.weight = mx.ones((16, 32)) * 7
    new = window.lookup(table, ids, ranges, metal=False)
    assert mx.array_equal(new, table(ids)).item()
    assert not mx.array_equal(new, old).item()


@pytest.mark.parametrize("replacement", ["table", "shard", "weight", "quantize"])
def test_producer_resident_preparation_is_invalidated_by_table_replacement(monkeypatch, replacement):
    from qsa_oracle import tiny_args

    from mlx2.runtime.models.qwen4_exp import NGramEmbedding

    module = NGramEmbedding(tiny_args(), 128, 0, 0)
    module.windowed_gather, module.hash_backend = True, "metal"
    monkeypatch.setattr(window, "eligible", lambda *a, **k: True)
    module.prepare_windowed_gather(resident_shards_prepared=True)
    ids = mx.array([[1, 2]])
    table = module.ngram_embedding
    assert module._window_eligible(ids)
    if replacement == "table":
        module.ngram_embedding = ShardedEmbedding(table.vocab_size, table.dims, table.num_shards)
    elif replacement == "shard":
        table.shard_0 = nn.Embedding(table.rows_per_shard, table.dims)
    elif replacement == "weight":
        table.shard_0.weight = table.shard_0.weight + 0
    else:
        nn.quantize(table, bits=4, group_size=32)
    assert not module._window_eligible(ids)
    module.prepare_windowed_gather(resident_shards_prepared=True)
    assert module._window_eligible(ids)
