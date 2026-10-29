"""gdn_core must not pass chunk_size to an MLX primitive that lacks it."""

import pytest

from mlx2.runtime.models import gated_delta as gd


def _primitive(signature):
    def fn(*args, **kwargs):
        fn.calls.append(kwargs)
        return "out"

    fn.calls = []
    fn.__doc__ = signature + "\n\n    Chunked gated delta network forward pass."
    return fn


STOCK = ("gated_delta_update(q, k, v, gamma, beta, initial_state=None, mask=None, "
         "stream=None) -> list[mlx.core.array]")
CUSTOM = ("gated_delta_update(q, k, v, gamma, beta, initial_state=None, mask=None, "
          "chunk_size=8, stream=None) -> list[mlx.core.array]")


def test_signature_detection():
    assert not gd._accepts_chunk_size(_primitive(STOCK))
    assert gd._accepts_chunk_size(_primitive(CUSTOM))
    assert not gd._accepts_chunk_size(object())


def test_installed_mlx_detection_matches_its_docstring():
    if gd._core_gated_delta_update is None:
        pytest.skip("primitive absent")
    first = gd._core_gated_delta_update.__doc__.split("\n", 1)[0]
    assert gd._CORE_ACCEPTS_CHUNK_SIZE == ("chunk_size" in first)
    assert gd.gdn_core_status()["chunk_size"] in (gd._CORE_GDN_CHUNK_SIZE, "primitive_default")


def test_prefill_scan_chunk8_refused_without_chunk_size(monkeypatch):
    monkeypatch.setattr(gd, "_core_gated_delta_update", _primitive(STOCK))
    monkeypatch.setattr(gd, "_CORE_ACCEPTS_CHUNK_SIZE", False)
    with pytest.raises(ValueError, match="accepts chunk_size"):
        gd.install_prefill_scan(object(), 8, 256)
