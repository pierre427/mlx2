"""Conservative workspace bounds across short and long prefill geometry."""

import pytest

from mlx2.adapters.inkling_memory import InklingCacheBudget


def test_wide_short_context_accounts_for_the_full_mask_tile():
    budget = InklingCacheBudget()
    # 512 x 1024 x 32 heads fits below the 64 MiB BF16 mask cap.
    mask_workspace = 3 * 32 * 512 * 1024 * 4
    assert budget.prefill_transient_bytes(1024, 512) >= mask_workspace
    assert budget.prefill_transient_bytes(1024, 512) > budget.prefill_transient_bytes(
        1024, 128
    )


def test_long_context_workspace_is_tiled_but_activations_grow():
    budget = InklingCacheBudget()
    assert budget.prefill_transient_bytes(32768, 1024) < 1 << 30
    assert budget.prefill_transient_bytes(32768, 1024) > budget.prefill_transient_bytes(
        32768, 128
    )
    assert budget.prefill_transient_bytes(32768, 0) == 0


@pytest.mark.parametrize("value", [-1, True, 0.5])
def test_invalid_geometry_is_refused(value):
    budget = InklingCacheBudget()
    with pytest.raises(ValueError):
        budget.prefill_transient_bytes(1024, value)
