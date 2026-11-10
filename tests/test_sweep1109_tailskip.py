"""Bug sweep 2026-11-09, lane tailskip (TS-1..TS-4). CPU, tiny random weights.

No checkpoint is loaded (``synthetic@n/a``); the factories are the ones the
tail-skip family tests use.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime import prefill_tail_skip as pts  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _claimed_chunk(model, cache, vocab, width, salt):
    rng = np.random.default_rng(salt)
    tokens = mx.array(rng.integers(1, vocab - 1, size=(1, width)), dtype=mx.int32)
    with prefill_checkpoint_rows([width], width=width) as request:
        out = model(tokens, cache)
        mx.eval(out)
    return out, request.applied


# -- TS-1 ---------------------------------------------------------------------


@pytest.mark.parametrize("wrapper", ["qwen36_35b", "qwen35_122b"])
def test_ts1_sparse_qwen_families_stay_opt_in(monkeypatch, wrapper):
    from test_prefill_tail_skip_qwen38_cpu import VOCAB, _sparse

    model = _sparse()
    if wrapper == "qwen35_122b":
        from mlx2.runtime.models.qwen35_122b import Model as Qwen35Model

        model.__class__ = Qwen35Model
    cache = model.language_model.make_cache
    monkeypatch.delenv(pts.ENV, raising=False)
    before = pts.STATS["qwen36_tail_split"]
    out, applied = _claimed_chunk(model, cache(), VOCAB, 9, 31)
    assert pts.STATS["qwen36_tail_split"] == before  # unbenched: not on by default
    assert out.shape == (1, 1, VOCAB) and applied == 1  # ordinary head trim
    assert pts.tail_skip_status()["effective_modes"]["qwen36_35b"] == "0"
    monkeypatch.setenv(pts.ENV, "1")
    _claimed_chunk(model, cache(), VOCAB, 9, 32)
    assert pts.STATS["qwen36_tail_split"] == before + 1  # explicit opt-in works


def test_ts1_qwen38_dense_stays_default_on_and_zero_disables(monkeypatch):
    from test_prefill_tail_skip_qwen38_cpu import VOCAB, _dense

    model = _dense()
    monkeypatch.delenv(pts.ENV, raising=False)
    before = pts.STATS["qwen38_tail_split"]
    _claimed_chunk(model, model.make_cache(), VOCAB, 7, 33)
    assert pts.STATS["qwen38_tail_split"] == before + 1
    monkeypatch.setenv(pts.ENV, "0")
    _claimed_chunk(model, model.make_cache(), VOCAB, 7, 34)
    assert pts.STATS["qwen38_tail_split"] == before + 1


# -- TS-2 ---------------------------------------------------------------------


def test_ts2_status_reports_family_defaults_and_effective_mode(monkeypatch):
    from test_prefill_tail_skip_qwen38_cpu import VOCAB, _dense

    monkeypatch.delenv(pts.ENV, raising=False)
    model = _dense()
    _claimed_chunk(model, model.make_cache(), VOCAB, 7, 35)
    status = pts.tail_skip_status()
    assert status["configured"] == "unset"
    assert status["effective_modes"]["qwen38_27b"] == "1"
    assert status["family_defaults"] == pts.FAMILY_DEFAULTS
    assert status["unlisted_family_default"] == "0"
    assert "invalid_value" not in status
    text = repr(status)
    assert "gemma4_opt_in_elsewhere" not in text and "'default': '0'" not in text


# -- TS-3 ---------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["", "2", "enabled", "1x"])
def test_ts3_unrecognised_value_fails_to_reference_and_is_reported(monkeypatch, raw):
    monkeypatch.setenv(pts.ENV, raw)
    assert pts.tail_skip_mode("gpt_oss") == "0"  # default-on family -> reference path
    assert pts.tail_skip_mode("kimi_linear") == "0"
    assert pts.tail_skip_mode() == "0"
    status = pts.tail_skip_status()
    assert status["invalid_value"] == raw
    assert status["invalid_value_action"] == "resolved_to_0_reference_path"


@pytest.mark.parametrize("raw, want", [
    ("0", "0"), (" 1 ", "1"), ("HEAD", "head"),
    ("off", "0"), (" False ", "0"), ("NO", "0"),
    ("on", "1"), ("True", "1"), (" yes", "1"),
])
def test_ts3_documented_values_still_win(monkeypatch, raw, want):
    monkeypatch.setenv(pts.ENV, raw)
    assert pts.tail_skip_mode("gpt_oss") == want
    assert pts.tail_skip_mode("kimi_linear") == want
    assert "invalid_value" not in pts.tail_skip_status()


# -- TS-4 ---------------------------------------------------------------------

_DECODE_LAW = ("auto_latent", "latent_single", "latent_flash", "latent_single_tiled",
               "latent_tiled", "fused_mla_decode", "single_row_calls")


def test_ts4_kimi_tail_keeps_multirow_bookkeeping_and_no_decode_law(monkeypatch):
    from test_prefill_tail_skip_kimi_linear_cpu import VOCAB, _model

    monkeypatch.setenv(pts.ENV, "1")
    model = _model()
    attention = model.model.layers[-1].attention
    cache = model.make_cache()
    before = pts.STATS["kimi_linear_tail_split"]
    _claimed_chunk(model, cache, VOCAB, 24, 41)
    _claimed_chunk(model, cache, VOCAB, 24, 42)
    assert pts.STATS["kimi_linear_tail_split"] == before + 2
    calls = dict(attention.calls)
    assert calls["multirow_calls"] == 2 and calls["multirow_tokens"] == 48
    for name in _DECODE_LAW:
        assert calls.get(name, 0) == 0, (name, calls)
    # The one-row law still has its own receipt.
    assert any(name.startswith("tail_") for name in calls), calls
    # A real decode step still books the decode law.
    step = mx.array([[5]], dtype=mx.int32)
    mx.eval(model(step, cache=cache))
    assert attention.calls["single_row_calls"] == 1


def test_ts4_qwen4_exp_tail_books_no_decode_receipts(monkeypatch):
    from mlx2.runtime.models import qwen4_exp as q4
    from test_prefill_tail_skip_qwen4_exp_cpu import COUNTER, _ids, _model

    monkeypatch.setenv(pts.ENV, "1")
    monkeypatch.setattr(q4, "_QSA_NAX_DECODE", True)
    previous = q4._qsa_b1.set_enabled(True)
    try:
        with mx.stream(mx.cpu):
            model = _model()
            cache = model.make_cache()
            nax0 = q4.qsa_nax_decode_status()["attempts"]
            b10 = sum(q4._qsa_b1.status()["declines"].values())
            before = pts.STATS[COUNTER]
            for salt, width in ((1, 24), (2, 20)):
                tokens = mx.array([_ids(width, salt)], dtype=mx.int32)
                with prefill_checkpoint_rows([width], width=width):
                    mx.eval(model(tokens, cache=cache))
            assert pts.STATS[COUNTER] == before + 2
            assert q4.qsa_nax_decode_status()["attempts"] == nax0
            assert sum(q4._qsa_b1.status()["declines"].values()) == b10
            # A real decode step still books its receipts.
            mx.eval(model(mx.array([[5]], dtype=mx.int32), cache=cache))
            assert q4.qsa_nax_decode_status()["attempts"] == nax0 + 1
    finally:
        q4._qsa_b1.set_enabled(previous)
