"""Host-only alias reachability checks; no model or device execution."""

import gc

import pytest

from mlx2.kernel_reload import KernelReloader, KernelReloadError


@pytest.mark.parametrize("live", [False, True])
def test_alias_scan_distinguishes_unreachable_cycles_from_live_tuple_captures(live):
    def original(value):
        return value + 1

    def replacement(value):
        return value + 2

    # Disable automatic collection so this precisely reproduces the stale
    # get_referrers inventory seen in a larger imported-module test process.
    enabled = gc.isenabled()
    gc.disable()
    try:
        captured = (original, "captured")
        cycle = [captured]
        cycle.append(cycle)
        if not live:
            del captured, cycle
            KernelReloader._rebind_aliases({id(original): (original, replacement)})
        else:
            with pytest.raises(KernelReloadError, match="immutable tuple"):
                KernelReloader._rebind_aliases({id(original): (original, replacement)})
            assert captured[0] is original
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def test_external_tuple_equal_to_internal_replacement_pair_still_refuses():
    def original(value):
        return value + 1

    def replacement(value):
        return value + 2

    captured = (original, replacement)
    aliases = {id(original): (original, replacement)}
    assert captured is not aliases[id(original)]
    with pytest.raises(KernelReloadError, match="immutable tuple"):
        KernelReloader._rebind_aliases(aliases)
    assert captured[0] is original and captured[1] is replacement
