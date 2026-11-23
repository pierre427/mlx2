"""Page-out working memory against the stock routed bank (CPU, scaled-down Hy4-like geometry).

The paged bank runs one stock-shaped pass per resident block slot, so its transient memory is about one stock call
plus a float32 [tokens, k, D] accumulator, independent of Kmax. The pre-review k x Kmax fan-out (kept here as the
reference ``_fanout``) copied the input and the float32 partials Kmax times.
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime import amoeba_pageout as P  # noqa: E402
from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort  # noqa: E402

D, E, K, T, BLOCK = 128, 32, 8, 512, 64


def _bank(F, seed=0):
    mx.random.seed(seed)
    sw = SwitchGLU(D, F, E, activation=H.ClampedSwiGLU(10.0))
    nn.quantize(sw, group_size=64, bits=4)
    mx.eval(sw.parameters())
    return sw


def _inputs(seed=1):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, T, D)).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(np.argsort(rng.random((1, T, E)), axis=-1)[..., :K].astype(np.int32))
    return x, idx


def _fanout(self, x, indices):
    """The pre-review paged call: every (slot, block) pair as one flat gather."""
    ids = self._table[indices]
    shp = ids.shape
    flat = ids.reshape(*shp[:-2], shp[-2] * shp[-1])
    xe, idx, inv = _gather_sort(mx.expand_dims(x, (-2, -3)), flat)
    h = self.activation(self.up_proj(xe, idx, sorted_indices=True), self.gate_proj(xe, idx, sorted_indices=True))
    y = _scatter_unsort(self.down_proj(h.astype(mx.float32), idx, sorted_indices=True), inv, flat.shape).squeeze(-2)
    return y.reshape(*shp[:-1], shp[-1], y.shape[-1]).sum(axis=-2).astype(x.dtype)


def _peak(fn):
    mx.eval(fn())                       # warm-up
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    out = fn()
    mx.eval(out)
    return mx.get_peak_memory() - base, out


@pytest.mark.parametrize("F", [256, 512])
def test_paged_transient_is_one_stock_pass_not_a_kmax_fanout(F):
    sw = _bank(F)
    x, idx = _inputs()
    paged, rec, _ = P.page_out_bank(sw, [], E=E, F=F, block=BLOCK)       # nothing paged: Kmax = F / 64
    assert rec["kmax"] == F // BLOCK
    p_stock, y_stock = _peak(lambda: sw(x, idx))
    p_paged, y_paged = _peak(lambda: paged(x, idx))
    p_fan, y_fan = _peak(lambda: _fanout(paged, x, idx))
    a, b = np.array(y_stock.astype(mx.float32)), np.array(y_paged.astype(mx.float32))
    assert np.allclose(a, b, rtol=2e-2, atol=2e-2)                       # bf16 rows; reassociated fp32 sums
    acc = T * K * D * 4                                                    # the float32 accumulator
    assert p_paged <= 1.5 * p_stock + 2 * acc, (p_stock, p_paged)
    assert p_fan > 2 * p_paged, (p_fan, p_paged)                         # the fan-out grew with Kmax
    print(f"F={F} Kmax={F // BLOCK}: stock {p_stock / 2**20:.1f} MiB, paged {p_paged / 2**20:.1f} MiB, "
          f"fan-out {p_fan / 2**20:.1f} MiB")


def test_paged_bank_with_closed_blocks_matches_the_zero_mask_bank():
    F = 256
    sw = _bank(F)
    x, idx = _inputs()
    rng = np.random.default_rng(3)
    blocks = rng.choice(E * (F // BLOCK), size=40, replace=False)
    closed = np.concatenate([np.arange(b * BLOCK, (b + 1) * BLOCK) for b in blocks])
    keep = np.ones(E * F, bool)
    keep[closed] = False
    zero = _bank(F)
    for proj in ("gate_proj", "up_proj"):                                 # the zero law on the stock bank
        lin = zero[proj]
        k = mx.array(keep.reshape(E, F, 1))
        lin["scales"] = mx.where(k, lin["scales"], 0)
        lin["biases"] = mx.where(k, lin["biases"], 0)
    paged, rec, paged_b = P.page_out_bank(zero, closed, E=E, F=F, block=BLOCK)
    assert rec["paged_blocks"] == 40 and rec["bytes_freed"] > 0
    a = np.array(zero(x, idx).astype(mx.float32))
    b = np.array(paged(x, idx).astype(mx.float32))
    assert np.allclose(a, b, rtol=2e-2, atol=2e-2)
    # the CPU bf16 stock kernel rounds its own dot product more than the paged float32 partials do: against the zero
    # bank run at float32 input (rounded to bf16 at the end), the paged rows are identical but for reassociation
    a32 = np.array(zero(x.astype(mx.float32), idx).astype(mx.bfloat16).astype(mx.float32))
    assert np.mean(a32 == b) > 0.999 and np.abs(a32 - b).max() <= 2 ** -9
