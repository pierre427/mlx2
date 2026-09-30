import mlx.core as mx

from mlx2.runtime.spomin_standard_surgery import _tail_rows


def test_tail_rows_zero_is_empty_not_everything():
    ring = mx.arange(5, dtype=mx.float32).reshape(1, 1, 5, 1)
    assert _tail_rows(ring, 0).shape[2] == 0
    assert _tail_rows(ring, 2).tolist() == [[[[3.0], [4.0]]]]
    assert _tail_rows(ring, 9).shape[2] == 5
