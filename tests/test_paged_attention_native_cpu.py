"""No Metal construction: native read's Python admission is default-off."""

import pytest

from mlx2.runtime.paged_attention_native import (
    complete_packed_read_after_event,
    native_paged_attention_read_fp16,
    poll_native_paged_read_events,
    wait_native_paged_read_events,
)


def test_native_read_refuses_before_extension_or_mlx_import():
    with pytest.raises(RuntimeError, match="explicit candidate"):
        native_paged_attention_read_fp16(None, None, None, None)
    with pytest.raises(TypeError, match="exact native backend"):
        native_paged_attention_read_fp16(None, None, None, None,
                                         permit_candidate=True)
    with pytest.raises(TypeError, match="exact arena backend"):
        poll_native_paged_read_events(None)
    with pytest.raises(TypeError, match="exact arena backend"):
        wait_native_paged_read_events(None, 0.01)
    with pytest.raises(ValueError, match="submitted native use"):
        complete_packed_read_after_event(None, (1, True))
