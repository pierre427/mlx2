"""CPU falsifiers for the paired harness's cache-state digest oracle.

``scripts/paired_direct_ab.state_digest`` binds cache class identity,
``state`` and ``meta_state`` with a tagged encoding over raw storage bits.
Each test below would pass vacuously under the previous oracle, which hashed
``state`` only and repr'd unknown objects.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.external_speculative import ExternalDraftState
from mlx2.runtime.models.cache import KVCache, _BaseCache
from scripts import paired_direct_ab as H


def _kv(offset=4, seed=0):
    cache = KVCache()
    keys = mx.random.normal((1, 2, 8, 4), key=mx.random.key(seed)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, 8, 4), key=mx.random.key(seed + 1)).astype(mx.bfloat16)
    cache.state = (keys, values)
    cache.offset = offset
    return cache


class _Recurrent(_BaseCache):
    """Recurrent-style cache: fixed arrays plus checkpoint metadata."""

    def __init__(self, arrays, meta):
        self.arrays, self.meta = arrays, meta

    @property
    def state(self):
        return self.arrays

    @property
    def meta_state(self):
        return self.meta


class _KVSubclass(KVCache):
    pass


class _StateOnly:
    def __init__(self, arrays):
        self.state = arrays


def _complete(obj):
    digest = H.state_digest(obj)
    assert digest["status"] == "complete", digest
    return digest["sha256"]


def test_offset_changes_the_digest_even_when_arrays_match():
    a, b = _kv(offset=4), _kv(offset=5)
    assert a.state[0] is not b.state[0]
    assert mx.array_equal(a.state[0], b.state[0]).item()
    assert _complete([a]) != _complete([b])
    # The old state-only oracle could not see this difference.
    assert H.state_digest([_StateOnly(a.state)])["state_only_sha256"] == \
        H.state_digest([_StateOnly(b.state)])["state_only_sha256"]


def test_recurrent_metadata_and_lengths_change_the_digest():
    conv = mx.zeros((1, 3, 8))
    lengths = mx.array([5], dtype=mx.int32)
    base = _complete([_Recurrent([conv, lengths], ("ckptv1", "5"))])
    assert base != _complete([_Recurrent([conv, lengths], ("ckptv1", "6"))])
    assert base != _complete([_Recurrent([conv, mx.array([6], dtype=mx.int32)], ("ckptv1", "5"))])
    assert base != _complete([_Recurrent([conv, lengths], "")])


def test_identical_independent_trees_match():
    assert _complete([_kv(), _kv()]) == _complete([_kv(), _kv()])

    def sidecar():
        draft = _kv(offset=3, seed=9)
        tail = mx.array([[1, 2]], dtype=mx.int32)
        rng = mx.array(list(b'{"k":1}'), dtype=mx.uint8)
        return ExternalDraftState(([draft], tail), 5, rng, 2, binding="b")

    assert _complete(sidecar()) == _complete(sidecar())
    other = sidecar()
    other.covered_tokens = 6
    assert _complete(other) != _complete(sidecar())


def test_class_identity_is_bound():
    a, b = _kv(), _KVSubclass()
    b.state, b.offset = _kv().state, 4
    assert _complete([a]) != _complete([b])


@pytest.mark.parametrize("left,right", [
    (("1",), (1,)), ((1,), (1.0,)), ((1,), (True,)), ((0.0,), (-0.0,)),
    (("",), ()), (["a"], ("a",)), ({"a": 1}, {"a": "1"}),
])
def test_scalar_and_container_metadata_distinctions(left, right):
    arrays = [mx.zeros((2,))]
    assert _complete([_Recurrent(arrays, left)]) != _complete([_Recurrent(arrays, right)])


def test_storage_bits_are_preserved():
    pos = mx.array([0.0, 1.0], dtype=mx.bfloat16)
    neg = mx.array([-0.0, 1.0], dtype=mx.bfloat16)
    assert mx.array_equal(pos, neg).item()
    assert _complete([pos]) != _complete([neg])
    quiet = np.array([np.float32("nan")]).view(np.uint32)
    payload = (quiet | np.uint32(1)).view(np.float32)
    a, b = mx.array(quiet.view(np.float32)), mx.array(payload)
    assert math.isnan(a.item()) and math.isnan(b.item())
    assert _complete([a]) != _complete([b])
    assert _complete([mx.zeros((2, 3))]) != _complete([mx.zeros((3, 2))])
    assert _complete([mx.zeros((2,), dtype=mx.float16)]) != _complete([mx.zeros((2,), dtype=mx.bfloat16)])


def test_mutation_after_capture_does_not_change_the_captured_digest():
    cache = _kv(offset=4)
    captured = H.state_digest([cache])
    frozen = dict(captured)
    cache.offset = 7
    after = H.state_digest([cache])
    assert captured == frozen
    assert after["sha256"] != captured["sha256"]


def test_shared_references_are_not_cycles():
    shared = mx.ones((2,))
    assert H.state_digest([shared, shared, [shared]])["status"] == "complete"


class _Raising(_BaseCache):
    @property
    def state(self):
        raise RuntimeError("boom")


@pytest.mark.parametrize("build,reason", [
    (lambda: (lambda cyc: (cyc.append(cyc), cyc)[1])([]), "cycle"),
    (lambda: [object()], "unsupported builtins.object"),
    (lambda: [_Raising()], "state raised RuntimeError"),
    (lambda: {1: mx.zeros(1)}, "non-string keys"),
    (lambda: None, "no state returned"),
])
def test_unsupported_or_cyclic_trees_are_unavailable(build, reason):
    digest = H.state_digest(build())
    assert digest["status"] == "unavailable" and digest["sha256"] is None
    assert reason in digest["reason"]
    assert H.state_hash(build()) is None


def test_missing_metadata_is_labelled_not_exact():
    digest = H.state_digest([_StateOnly([mx.zeros(2)])])
    assert digest["status"] == "metadata_unavailable" and digest["sha256"] is None
    assert digest["state_only_sha256"] and "no meta_state" in digest["reason"]


def test_harness_labels_and_compares_digest_statuses():
    complete = {"status": "complete", "sha256": "a", "reason": None}
    state_only = {"status": "metadata_unavailable", "sha256": None, "state_only_sha256": "a"}
    missing = {"status": "unavailable", "sha256": None, "reason": "x"}
    runs = [{"k": complete}, {"k": complete}]
    assert H._comparison_label(runs, "k") == "compared"
    assert H._comparison_label([{"k": state_only}] * 2, "k").startswith("state_only")
    assert H._comparison_label([{"k": complete}, {"k": missing}], "k") == "unavailable"
