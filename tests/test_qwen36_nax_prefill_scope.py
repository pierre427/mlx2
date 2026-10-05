"""The Qwen3.6 trunk opens the NAX MoE prefill scope for prefill forwards only.

Options sweep 2026-10-02 bug 3: ``MLX2_MOE_NAX_GATHER`` was a silent no-op on
Qwen3.6 because only Flash-Next's trunk opened ``moe_nax_gather.forward_scope``;
every sorted gather was counted ``not_prefill``.  The Qwen3.6 trunk now decides
the phase once per forward, the same way: prefill yes; decode, verify windows
(verify scope or a speculating cache) and the MTP head no.  Off stays off.
"""

import mlx.core as mx
import pytest

from mlx2.runtime import verify_scope
from mlx2.runtime.models import moe_nax_gather as nax
from mlx2.runtime.models import qwen36_moe_decode
from tests.test_qwen36_35b_port import _tiny_mtp_model


@pytest.fixture(autouse=True)
def _reset():
    old = nax.MODE
    nax.status(reset=True)
    yield
    nax.set_mode(old)
    nax.status(reset=True)


@pytest.fixture
def phases(monkeypatch):
    """Phase seen by every Qwen3.6 MoE block call, one entry per forward."""
    seen = []
    original = qwen36_moe_decode.Qwen36SparseMoeBlock.__call__

    def record(self, x):
        seen.append(nax.prefill_active())
        return original(self, x)

    monkeypatch.setattr(qwen36_moe_decode.Qwen36SparseMoeBlock, "__call__", record)
    return seen


def _forward(model, ids, cache):
    mx.eval(model(mx.array(ids, mx.uint32), cache=cache))


@pytest.mark.parametrize("mode", ["fused", "gather"])
def test_scope_opens_on_prefill_and_not_on_decode(monkeypatch, phases, mode):
    model = _tiny_mtp_model(monkeypatch, fuse_gate_up=True)
    nax.set_mode(mode)
    cache = model.make_cache()
    layers = len(model.layers)
    _forward(model, [[1, 2, 3, 4, 5, 6, 7, 8]], cache)
    assert phases == [True] * layers
    phases.clear()
    _forward(model, [[9]], cache)  # ordinary decode
    assert phases == [False] * layers
    phases.clear()
    _forward(model, [[9], [10]], None)  # batched decode, one row per lane
    assert phases == [False] * layers
    assert nax.prefill_active() is False  # the scope never leaks out


def test_verify_windows_are_not_prefill(monkeypatch, phases):
    model = _tiny_mtp_model(monkeypatch, fuse_gate_up=False)
    nax.set_mode("fused")
    cache = model.make_cache()
    _forward(model, [[1, 2, 3, 4]], cache)
    phases.clear()
    with verify_scope.verify_forward():
        _forward(model, [[5, 6, 7]], cache)
    assert phases and not any(phases)
    phases.clear()
    for entry in cache:
        entry.speculating = True
    try:
        _forward(model, [[5, 6, 7]], cache)
    finally:
        for entry in cache:
            entry.speculating = False
    assert phases and not any(phases)


def test_input_embeddings_prefill_opens_the_scope(monkeypatch, phases):
    model = _tiny_mtp_model(monkeypatch, fuse_gate_up=True)
    nax.set_mode("gather")
    trunk = model.language_model.model
    embeddings = trunk.embed_tokens(mx.array([[1, 2, 3, 4, 5]], mx.uint32))
    mx.eval(trunk(None, model.make_cache(), input_embeddings=embeddings))
    assert phases and all(phases)


def test_off_mode_opens_no_scope(monkeypatch, phases):
    model = _tiny_mtp_model(monkeypatch, fuse_gate_up=True)
    nax.set_mode("off")
    _forward(model, [[1, 2, 3, 4, 5, 6, 7, 8]], model.make_cache())
    assert phases and not any(phases)
