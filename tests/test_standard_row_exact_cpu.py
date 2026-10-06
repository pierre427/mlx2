"""Opt-in target geometry contracts; numerical execution is strictly CPU."""

import copy
import json

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_map
from test_standard_parallel_draft_adapter import pack  # noqa: F401
from test_standard_xpress_serving_cpu import drain, tiny
from test_xpress_artifact_metadata import artifact  # noqa: F401

from mlx2.adapters.standard_decoder import StandardDecoderAdapter
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.segmented_plain_kv import SegmentedBatchKVCache
from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows


@pytest.fixture(autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def equal(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual.astype(mx.float32)), np.asarray(expected.astype(mx.float32))
    )


def equal_cache(actual, expected):
    for a, b in zip(actual, expected):
        assert a.offset == b.offset
        if a.offset:
            for x, y in zip(a.keys_and_values(), b.keys_and_values()):
                equal(x, y)


@pytest.mark.parametrize("tied", [False, True])
def test_selection_preserves_modules_parameters_ordinary_and_prefill(tied):
    model, _ = tiny(tied)
    tokens = mx.array([[1, 2, 3], [4, 5, 6]])
    original = model(tokens, cache=model.make_cache())
    body = model.prefill_body(tokens, model.make_cache(), [0, 2])
    modules = [(name, id(module)) for name, module in model.named_modules()]
    parameters = [(name, id(value)) for name, value in tree_flatten(model.parameters())]
    assert model.external_execution_receipt is None
    model.configure_target_verify_row_exact(True)
    equal(model(tokens, cache=model.make_cache()), original)
    equal(model.prefill_body(tokens, model.make_cache(), [0, 2]), body)
    assert modules == [(name, id(module)) for name, module in model.named_modules()]
    assert parameters == [
        (name, id(value)) for name, value in tree_flatten(model.parameters())
    ]
    receipt = model.external_execution_receipt["target_verify_row_exact"]
    assert not receipt["observed_used"] and receipt["executed_forwards"] == 0
    model.forward_with_taps(tokens, model.make_cache(), [0, 2])
    receipt = model.external_execution_receipt["target_verify_row_exact"]
    assert receipt["observed_used"] and receipt["physical_query_rows"] == 6
    assert receipt["observation_scope"] == "model_lifetime_not_per_request"
    assert not receipt["qualified"] and not receipt["performance_claim"]


@pytest.mark.parametrize("tied", [False, True])
def test_b15_ragged_prefix_matches_original_s1_and_commits_only_prefix(
    tied, monkeypatch
):
    model, _ = tiny(tied)
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    rows = []
    for index in range(15):
        caches = model.make_cache()
        prefix = [1, 2, 3][: index % 4]
        if prefix:
            mx.eval(model(mx.array([prefix]), cache=caches))
        rows.append(caches)
    frozen = copy.deepcopy(rows)
    lengths = [1 + index % 4 for index in range(15)]
    tokens = mx.array(
        [
            [index % 7, (index + 1) % 7, (index + 2) % 7, (index + 3) % 7]
            for index in range(15)
        ]
    )
    geometry = []
    old_linear, old_head = nn.Linear.__call__, nn.Embedding.as_linear

    def linear(module, value):
        geometry.append(value.shape)
        return old_linear(module, value)

    def head(module, value):
        geometry.append(value.shape)
        return old_head(module, value)

    monkeypatch.setattr(nn.Linear, "__call__", linear)
    monkeypatch.setattr(nn.Embedding, "as_linear", head)
    owner = SegmentedKVRows(rows)
    tx = owner.begin(lengths)
    model.configure_target_verify_row_exact(True)
    logits, taps = model.forward_with_taps(tokens, tx.caches, [0, 2])
    assert geometry and all(
        shape[:2] == (1, 1) and len(shape) == 3 for shape in geometry
    )
    model.configure_target_verify_row_exact(False)
    accepted = [min(2, length) for length in lengths]
    committed = tx.commit(accepted_lengths=accepted)
    for index, length in enumerate(lengths):
        reference_cache = copy.deepcopy(frozen[index])
        for position in range(length):
            expected, features = model.forward_with_taps(
                tokens[index : index + 1, position : position + 1],
                reference_cache,
                [0, 2],
            )
            equal(logits[index : index + 1, position : position + 1], expected)
            equal(taps[index : index + 1, position : position + 1], features)
        fresh = copy.deepcopy(frozen[index])
        for position in range(accepted[index]):
            mx.eval(
                model(tokens[index : index + 1, position : position + 1], cache=fresh)
            )
        equal_cache(committed[index], fresh)
        equal(
            model(mx.array([[7]]), cache=copy.deepcopy(committed[index])),
            model(mx.array([[7]]), cache=copy.deepcopy(fresh)),
        )


def test_future_tokens_do_not_change_reached_prefix_plain_or_segmented():
    model, _ = tiny()
    model.configure_target_verify_row_exact(True)
    a = mx.array([[1, 2, 3, 4], [1, 2, 5, 6]])
    logits, taps = model.forward_with_taps(a, model.make_cache(), [0, 2])
    equal(logits[0, :2], logits[1, :2])
    equal(taps[0, :2], taps[1, :2])
    rows = [model.make_cache(), model.make_cache()]
    views = [SegmentedBatchKVCache(list(layer)) for layer in zip(*rows)]
    for view in views:
        view.prepare(lengths=[4, 2])
    ragged, _ = model.forward_with_taps(a, views, [0, 2])
    equal(ragged[0, :2], ragged[1, :2])
    assert [cache.offset for cache in rows[1]] == [2, 2, 2]


def test_abort_after_later_layer_append_restores_all_rows_and_retry(monkeypatch):
    import mlx2.runtime.models.standard_decoder as runtime

    model, _ = tiny()
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    model.configure_target_verify_row_exact(True)
    rows = [model.make_cache(), model.make_cache()]
    for index, row in enumerate(rows):
        mx.eval(model(mx.array([[1, 2][: index + 1]]), cache=row))
    frozen = copy.deepcopy(rows)
    owner = SegmentedKVRows(rows)
    tx = owner.begin([3, 2])
    old = runtime._prefix_attention
    calls = []

    def fail_late(*args, **kwargs):
        value = old(*args, **kwargs)
        mx.eval(value)
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected after later layer append")
        return value

    monkeypatch.setattr(runtime, "_prefix_attention", fail_late)
    tokens = mx.array([[3, 4, 5], [4, 5, 6]])
    with pytest.raises(RuntimeError, match="later layer append"):
        model.forward_with_taps(tokens, tx.caches, [0, 2])
    tx.abort()
    for actual, expected in zip(rows, frozen, strict=True):
        equal_cache(actual, expected)
        equal(
            model(mx.array([[7]]), cache=copy.deepcopy(actual)),
            model(mx.array([[7]]), cache=copy.deepcopy(expected)),
        )
    assert (
        model.external_execution_receipt["target_verify_row_exact"]["executed_forwards"]
        == 0
    )
    monkeypatch.setattr(runtime, "_prefix_attention", old)
    retry = owner.begin([3, 2])
    logits, _ = model.forward_with_taps(tokens, retry.caches, [0, 2])
    kept = retry.commit(accepted_lengths=[2, 1])
    model.configure_target_verify_row_exact(False)
    for index, count in enumerate([2, 1]):
        reference_cache = copy.deepcopy(frozen[index])
        for position in range(count):
            expected = model(
                tokens[index : index + 1, position : position + 1],
                cache=reference_cache,
            )
            equal(logits[index : index + 1, position : position + 1], expected)
        equal_cache(kept[index], reference_cache)
        equal(
            model(mx.array([[7]]), cache=copy.deepcopy(kept[index])),
            model(mx.array([[7]]), cache=copy.deepcopy(reference_cache)),
        )


@pytest.mark.parametrize("permute", [False, True])
def test_mixed_or_permuted_transactions_rejected_before_write(permute):
    model, _ = tiny()
    model.configure_target_verify_row_exact(True)
    rows = [[model.make_cache()], [model.make_cache()]]
    transactions = [SegmentedKVRows(group).begin([2]) for group in rows]
    caches = list(transactions[0].caches)
    if permute:
        caches[0], caches[1] = caches[1], caches[0]
    else:
        caches[-1] = transactions[1].caches[-1]
    with pytest.raises(ValueError, match="canonical layer transaction"):
        model.forward_with_taps(mx.array([[1, 2]]), caches, [0, 2])
    assert all(
        cache.offset == 0 and cache.keys is None
        for group in rows
        for row in group
        for cache in row
    )
    for transaction in transactions:
        transaction.abort()


@pytest.mark.parametrize(
    "kind", ["rotating", "subclass", "late_layer", "pld_mask", "offset", "kernel"]
)
def test_unsupported_owner_fails_before_any_cache_append(kind, monkeypatch):
    model, _ = tiny()
    model.configure_target_verify_row_exact(True)
    caches = model.make_cache()
    if kind == "rotating":
        caches[-1] = RotatingKVCache(max_size=8)
    elif kind == "subclass":

        class CustomKV(KVCache):
            pass

        caches[-1] = CustomKV()
    elif kind == "late_layer":
        caches[-1] = object()
    elif kind == "pld_mask":
        caches[-1]._pld_ordinary_mask_padding = 0
    elif kind == "offset":
        caches[-1].offset = 1
    else:
        monkeypatch.setenv("MLX2_FP_DECODE_KERNEL", "1")
    with pytest.raises(ValueError, match="row-exact"):
        model.forward_with_taps(mx.array([[1, 2]]), caches, [0, 2])
    assert caches[0].offset == 0 and caches[0].keys is None
    assert not model.external_execution_receipt["target_verify_row_exact"][
        "observed_used"
    ]


@pytest.mark.parametrize("facade", ["plain", "transaction", "prepared"])
@pytest.mark.parametrize(
    "poison", ["capacity", "dtype", "integer", "nonarray", "wrong_target_dtype"]
)
def test_poisoned_late_kv_storage_rejected_before_first_write(facade, poison):
    model, _ = tiny()
    caches = model.make_cache()
    mx.eval(model(mx.array([[1]]), cache=caches))
    model.configure_target_verify_row_exact(True)
    bad = caches[-1]
    if poison == "capacity":
        bad.values = bad.values[:, :, :1]
    elif poison == "dtype":
        bad.values = bad.values.astype(mx.float16)
    elif poison == "integer":
        bad.keys = bad.keys.astype(mx.int32)
        bad.values = bad.values.astype(mx.int32)
    elif poison == "wrong_target_dtype":
        bad.keys = bad.keys.astype(mx.float16)
        bad.values = bad.values.astype(mx.float16)
    else:
        bad.values = [1]
    prefix = [copy.deepcopy(cache) for cache in caches[:-1]]
    stored = [(id(cache.keys), id(cache.values), cache.offset) for cache in caches]
    transaction = None
    if facade == "transaction":
        transaction = SegmentedKVRows([caches]).begin([2])
        views = transaction.caches
    elif facade == "prepared":
        views = [SegmentedBatchKVCache([cache]) for cache in caches]
        for view in views:
            view.prepare(lengths=[2])
    else:
        views = caches
    with pytest.raises(ValueError, match="invalid plain KV storage"):
        model.forward_with_taps(mx.array([[2, 3]]), views, [0, 2])
    assert stored == [
        (id(cache.keys), id(cache.values), cache.offset) for cache in caches
    ]
    equal_cache(caches[:-1], prefix)
    if transaction is not None:
        transaction.abort()


def test_backbone_cast_after_configuration_fails_closed_and_fresh_reconfigure():
    model, _ = tiny()
    model.configure_target_verify_row_exact(True)
    previous_cache = model.make_cache()
    mx.eval(model(mx.array([[1]]), cache=previous_cache))
    model.model.update(
        tree_map(lambda value: value.astype(mx.bfloat16), model.model.parameters())
    )
    fresh = model.make_cache()
    with pytest.raises(ValueError, match="dtype changed after configuration"):
        model.forward_with_taps(mx.array([[2, 3]]), fresh, [0, 2])
    assert all(cache.offset == 0 and cache.keys is None for cache in fresh)
    model.configure_target_verify_row_exact(True)
    with pytest.raises(ValueError, match="invalid plain KV storage"):
        model.forward_with_taps(mx.array([[2, 3]]), previous_cache, [0, 2])
    assert all(
        cache.offset == 1 and cache.keys.dtype == mx.float32 for cache in previous_cache
    )
    logits, _ = model.forward_with_taps(mx.array([[2, 3]]), model.make_cache(), [0, 2])
    assert logits.shape == (1, 2, 9)


def test_mixed_backbone_dtype_refused_but_independent_output_head_allowed():
    model, _ = tiny()
    model.layers[-1].mlp.down_proj.weight = model.layers[
        -1
    ].mlp.down_proj.weight.astype(mx.bfloat16)
    with pytest.raises(ValueError, match="homogeneous floating backbone dtype"):
        model.configure_target_verify_row_exact(True)
    assert model.external_execution_receipt is None
    model, _ = tiny()
    model.lm_head.weight = model.lm_head.weight.astype(mx.bfloat16)
    model.configure_target_verify_row_exact(True)
    model.forward_with_taps(mx.array([[1, 2]]), model.make_cache(), [0, 2])
    assert model.external_execution_receipt["target_verify_row_exact"]["observed_used"]


@pytest.mark.parametrize("value", [0, 1, None, "true", {}])
def test_strict_policy_rejected_before_allocation(pack, monkeypatch, value):  # noqa: F811
    import mlx2.runtime.models.standard_decoder as runtime

    draft, target = pack
    monkeypatch.setattr(runtime, "Model", lambda *a: pytest.fail("allocated target"))
    with pytest.raises(ValueError, match="boolean"):
        StandardDecoderAdapter(
            str(target),
            execution_policy={
                "draft_model": str(draft),
                "target_verify_row_exact": value,
            },
        )


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ({"quantization": {"bits": 4, "group_size": 64}}, "unquantized dense"),
        ({"rope_scaling": {}}, "unquantized dense"),
        ({"sliding_window": 32}, "requires full attention"),
        ({"num_experts": 2}, "does not support expert layers"),
    ],
)
def test_unsupported_target_rejected_before_allocation(pack, monkeypatch, config, error):  # noqa: F811
    import mlx2.runtime.models.standard_decoder as runtime

    draft, target = pack
    path = target / "config.json"
    original = json.loads(path.read_text())
    path.write_text(json.dumps({**original, **config}))
    monkeypatch.setattr(runtime, "Model", lambda *a: pytest.fail("allocated target"))
    with pytest.raises(ValueError, match=error):
        StandardDecoderAdapter(
            str(target),
            execution_policy={
                "draft_model": str(draft),
                "target_verify_row_exact": True,
            },
        )


def test_adapter_metadata_route_binding_receipt_and_paired_resume_refusal(pack):  # noqa: F811
    draft, target = pack
    base = {"draft_model": str(draft), "num_draft": 2}
    ordinary_math = StandardDecoderAdapter(str(target), execution_policy=base)
    explicit_off = StandardDecoderAdapter(
        str(target), execution_policy={**base, "target_verify_row_exact": False}
    )
    exact = StandardDecoderAdapter(
        str(target), execution_policy={**base, "target_verify_row_exact": True}
    )
    assert ordinary_math.identity == explicit_off.identity
    assert ordinary_math.layout == explicit_off.layout
    assert exact.identity["fingerprint"] != ordinary_math.identity["fingerprint"]
    assert exact._external_target_revision != ordinary_math._external_target_revision
    assert (
        exact.identity["artifact_fingerprint"]
        == ordinary_math.identity["target_fingerprint"]
    )
    assert exact.descriptor.metadata["target_verify_row_exact"]["qualified"] is False
    engine = exact.create_external_batch()
    uid = engine.insert([[1, 2, 3]], max_tokens=[4])[0]
    _, finished = drain(engine)
    response = finished[uid]
    receipt = response.speculative_receipt["target_protocol"]["target_verify_row_exact"]
    assert receipt["observed_used"] and receipt["executed_forwards"] > 0
    with pytest.raises(ValueError, match="revision/boundary mismatch"):
        response.cache_sidecar.validate(
            ordinary_math.identity["fingerprint"], len(response.all_tokens)
        )
    engine.close()
