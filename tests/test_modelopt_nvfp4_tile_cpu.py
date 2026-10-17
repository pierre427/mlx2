import mlx.core as mx

from mlx2.runtime.models.modelopt_nvfp4 import ModelOptNVFP4Linear
from mlx2.runtime.models.modelopt_nvfp4_tile import MAX_ROWS, admit


def _linear():
    return ModelOptNVFP4Linear(64, 32)


def test_row_tile_admits_one_through_sixteen_bf16_rows():
    linear = _linear()
    for rows in (1, 2, 4, 8, MAX_ROWS):
        result = admit(
            linear,
            mx.zeros((rows, 64), dtype=mx.bfloat16),
            require_gpu=False,
        )
        assert result.accepted, result.reason


def test_row_tile_refuses_changed_precision_and_oversized_rows():
    linear = _linear()
    assert (
        admit(
            linear,
            mx.zeros((17, 64), dtype=mx.bfloat16),
            require_gpu=False,
        ).reason
        == "rows"
    )
    assert (
        admit(
            linear,
            mx.zeros((4, 64), dtype=mx.float32),
            require_gpu=False,
        ).reason
        == "activation_dtype"
    )
