"""A one-row lineage in a row-exact window reads the ordinary B=1 mask
(sweep 2026-10-02 S1).

In a ragged segmented window a lane on its last token (or at K=0) reaches
``_RowExactAttention`` with one row and its B1 row cache's mask, which is
None; the ordinary B=1 route reads an explicit all-valid mask, and MLX's SDPA
gives different bits for the two.  The one-row shortcut passed None through
and left the window labelled exact.  CPU only: the stock attention records
the mask it receives.
"""

import mlx.core as mx

from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.models import qwen4_row_exact as RE
from mlx2.runtime.models.cache import KVCache


class _StockAttention:
    def __init__(self):
        self.seen = []

    def __call__(self, x, mask, cache, *, _projected=None, _return_pre_o=False,
                 _selection=None, _fetched_kv=None, _defer_sdpa=None):
        self.seen.append(mask)
        return x


def _row(offset=5):
    row = KVCache()
    row.update_and_fetch(mx.zeros((1, 1, offset, 4)), mx.zeros((1, 1, offset, 4)))
    return row


def _attention():
    cls = RE._subclass(RE._RowExactAttention, _StockAttention)
    attn = cls.__new__(cls)
    _StockAttention.__init__(attn)
    return attn


def test_one_row_lineage_in_window_gets_the_ordinary_b1_mask():
    attn = _attention()
    row = _row()
    row_mask = row.make_mask(1, return_array=True, window_size=None)
    assert row_mask is None  # what the segmented consumer passes
    want = RE._ordinary_b1_mask(row)
    record = REV.Window(rows=3)
    with REV.window(record):
        attn(mx.zeros((1, 1, 8)), row_mask, row, _projected=(mx.zeros((1, 1, 8)),))
    got = attn.seen[0]
    assert got is not None
    assert got.shape == want.shape and mx.array_equal(got, want).item()
    assert record.exact
    assert record.stages["attention"]["per_row"] == 1


def test_one_row_lineage_without_host_offset_fails_the_window():
    attn = _attention()
    row = _row()
    row.offset = mx.array([5, 6])  # not one host offset
    record = REV.Window(rows=3)
    with REV.window(record):
        attn(mx.zeros((1, 1, 8)), None, row, _projected=(mx.zeros((1, 1, 8)),))
    assert not record.exact
    assert "attention_offset_not_host" in record.failures


def test_outside_a_window_the_mask_is_untouched():
    attn = _attention()
    attn(mx.zeros((1, 1, 8)), None, _row(), _projected=(mx.zeros((1, 1, 8)),))
    assert attn.seen == [None]


def test_single_lane_segmented_passthrough_is_recorded_not_exact(monkeypatch):
    from test_batched_mtp import _tiny_qwen4_model

    model = _tiny_qwen4_model()
    handle = RE.install(model)
    handle.enable(True)

    class Segmented:
        def segmented_attention(self, attention, hidden, mask):
            raise AssertionError("not reached: backbone is stubbed")

    monkeypatch.setattr(
        model, "mtp_backbone",
        lambda tokens, cache=None: (mx.zeros((*tokens.shape, 32)),) * 2,
    )
    start = handle.snapshot()
    handle.verify_backbone(mx.ones((1, 1), mx.uint32), [Segmented()])
    status = handle.status()
    assert status["windows_not_exact"] == 1
    assert status["failures"]["segmented_one_token_target_not_row_exact"] == 1
    assert handle.receipt(start)["row_exact"] is False
    # A plain (non-segmented) single-lane passthrough is still not a window.
    handle.verify_backbone(mx.ones((1, 1), mx.uint32), [KVCache()])
    assert handle.status()["windows"] == 1
