"""2026-10-06 sweep (kernels lane), Qwen4 kernel guards.

KR-08: the research-only fused_outproj GDN decode kernel joins its 12
threadgroups through a grid-wide spin barrier on relaxed atomics (no forward
progress guarantee, no device-scope ordering); it is refused, not run.

KR-09: the NAX QSA prefill kernel derives validity from query positions and
left padding alone, so a mask carrying more structure must keep the indexed
path, which ANDs the mask in.  Plainness is proven by the causal-mask
registry (the cache that built the mask), never inferred from shape, and the
opt-in NAX decode rows take the same proof (rfix-kad 2026-10-07).
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.models import qsdpa_verify_metal as QV
from mlx2.runtime.models import qwen4_exp as QE
from mlx2.runtime.models import qwen4_fused_gdn as FG


def test_fused_outproj_mode_is_refused():
    module = SimpleNamespace(fused_gdn_decode_mode="stock")
    with pytest.raises(ValueError, match="fused_outproj is disabled"):
        QE.GatedDeltaNet.set_fused_gdn_decode_mode(module, "fused_outproj")
    assert module.fused_gdn_decode_mode == "stock"
    QE.GatedDeltaNet.set_fused_gdn_decode_mode(module, "fused")
    assert module.fused_gdn_decode_mode == "fused"


def test_fused_outproj_kernel_cannot_be_built():
    FG._kernel_outproj.cache_clear()
    with pytest.raises(RuntimeError, match="fused_outproj is disabled"):
        FG._kernel_outproj()


def _sel(mask=None, *, batch=1, length=512, width=16384, passthrough=None):
    return SimpleNamespace(
        kind="explicit", batch=batch, length=length, physical_width=width,
        causal_mask=mask, passthrough_mask=passthrough,
    )


def _decide(selection):
    return QE.decide_qsa_nax_admission(
        selection, training=False, layout_ok=True, device_supported=True,
        kernel_available=True, min_physical_kv=16384, batched=True,
    )


def _qsa_masks():
    """Masks as the Qwen4 caches hand them to attention (registered)."""
    single = QE.QSAKVCache()
    single.offset = 64
    batch = QE.BatchQSAKVCache([0, 2, 1])
    return (
        QE.lift_causal_mask(single.make_mask(8, return_array=True, window_size=None)),
        batch.make_mask(8, return_array=True),
    )


def test_plain_causal_masks_are_admitted():
    (single, batched) = _qsa_masks()
    for (mask, batch) in ((None, 1), ("causal", 1), (single, 1), (batched, 3)):
        assert _decide(_sel(mask, batch=batch)).engage


def _segmented():
    mask = mx.ones((1, 1, 512, 16384), dtype=mx.bool_)
    QV.register_causal_mask(mask, None, kind="segmented")
    return mask


@pytest.mark.parametrize(
    "selection",
    [
        _sel(mx.zeros((512, 16384), dtype=mx.float16)),          # additive bias
        _sel(mx.zeros((1, 8, 512, 16384), dtype=mx.bool_)),      # per-head
        _sel(mx.zeros((2, 1, 512, 16384), dtype=mx.bool_)),      # batch mismatch
        _sel(passthrough=mx.zeros((1, 1, 512, 16384), dtype=mx.bool_)),
        _sel("sliding"),
        # Same-shape boolean masks are not plain by shape: an all-false or
        # segment-diagonal mask has no provenance, and a segmented one is
        # registered as such.
        _sel(mx.zeros((512, 16384), dtype=mx.bool_)),
        _sel(mx.zeros((1, 1, 512, 16384), dtype=mx.bool_)),
        _sel(mx.zeros((3, 1, 512, 16384), dtype=mx.bool_), batch=3),
        _sel(_segmented()),
    ],
)
def test_structured_masks_fall_back_with_a_reason(selection):
    decision = _decide(selection)
    assert not decision.engage and decision.reason == "nonstandard_mask"


def test_registration_follows_the_lifted_mask_only():
    (single, _batched) = _qsa_masks()
    assert QV.registered_mask_kind(single) == "left_padded"
    structured = single & mx.ones_like(single)
    assert QV.registered_mask_kind(structured) is None
    assert not _decide(_sel(structured)).engage


# --- opt-in NAX decode (MLX_QWEN4_QSA_NAX_DECODE) ---------------------------


def _attention():
    from qsa_oracle import tiny_args

    mx.random.seed(7)
    attention = QE.Attention(tiny_args(indexer_budget=8))  # top-2 blocks of 4
    attention.eval()
    mx.eval(attention.parameters())
    return attention


def _batched_cache(attention, lengths):
    rows = []
    for length in lengths:
        mx.random.seed(1000 + length)
        row = QE.QSAKVCache(attention.indexer.summary_identity)
        hidden = mx.random.normal((1, length, 16))
        mask = row.make_mask(length, return_array=True, window_size=None)
        mx.eval(attention(hidden, mask, row), row.state)
        rows.append(row)
    cache = QE.BatchQSAKVCache.merge(rows)
    mx.eval(cache.state)
    return cache


@pytest.fixture
def nax_decode(monkeypatch):
    calls = []

    def fake_nax(q, *args, **kwargs):
        calls.append(q.shape)
        return mx.zeros(q.shape, dtype=q.dtype)

    monkeypatch.setattr(QE, "_QSA_NAX_DECODE", True)
    monkeypatch.setattr(QE, "nax_kernel_available", lambda: True)
    monkeypatch.setattr(QE, "nax_qsa_attention", fake_nax)
    QE.qsa_nax_decode_status(reset=True)
    yield calls
    QE.qsa_nax_decode_status(reset=True)


def _decode_once(attention, cache, mask):
    mx.random.seed(23)
    hidden = mx.random.normal((cache.left_padding.shape[0], 1, 16))
    out = attention(hidden, mask, cache)
    mx.eval(out)
    return out


def test_nax_decode_engages_on_the_cache_mask(nax_decode):
    attention = _attention()
    attention._nax_layout_ok = True
    cache = _batched_cache(attention, [24, 37])
    _decode_once(attention, cache, cache.make_mask(1, return_array=True))
    assert len(nax_decode) == 1
    assert QE.qsa_nax_decode_status()["last_receipt"]["reason"] == "engaged"


def test_nax_decode_refuses_a_structured_mask(nax_decode):
    attention = _attention()
    attention._nax_layout_ok = True
    cache = _batched_cache(attention, [24, 37])
    plain = cache.make_mask(1, return_array=True)
    # Hide one valid key of row 1: same shape and dtype, more structure.
    hide = mx.arange(plain.shape[-1]) != plain.shape[-1] - 5
    row = mx.arange(plain.shape[0])[:, None, None, None] == 1
    structured = plain & (hide | ~row)
    _decode_once(attention, cache, structured)
    assert nax_decode == []
    status = QE.qsa_nax_decode_status()
    assert status["engagements"] == 0
    assert status["last_receipt"]["reason"] == "nonstandard_mask"


def test_model_forward_masks_carry_the_plain_proof(monkeypatch):
    """The forward's own masks (prefill array, lifted to 4-D) stay admissible."""
    from mlx2.runtime.models.cache import make_prompt_cache
    from test_batched_mtp import _tiny_qwen4_model

    seen = []
    decide = QE.decide_qsa_nax_admission

    def spy(selection, **kwargs):
        if getattr(selection, "causal_mask", None) is not None:
            seen.append(QE._qsa_nax_plain_mask(selection))
        return decide(selection, **kwargs)

    monkeypatch.setattr(QE, "decide_qsa_nax_admission", spy)
    model = _tiny_qwen4_model()
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], mx.uint32), cache=cache))
    assert seen and all(seen)
