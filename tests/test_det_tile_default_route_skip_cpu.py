"""A checkpoint-default calculator never 409s a request on an unadmitted route (2026-11-22)."""
from types import SimpleNamespace

from mlx2.serving import det_tile_default_skips_route

TILES = SimpleNamespace(served_routes=("ordinary", "native_mtp"))


def test_default_tiles_step_aside_on_an_unadmitted_route():
    assert det_tile_default_skips_route(TILES, "prompt_lookup", {}, True)


def test_admitted_route_keeps_the_tiles():
    assert not det_tile_default_skips_route(TILES, "ordinary", {}, True)


def test_explicit_request_or_operator_tile_still_fails_closed():
    assert not det_tile_default_skips_route(TILES, "prompt_lookup", {"det_tiles": {"calculator": True}}, True)
    assert not det_tile_default_skips_route(TILES, "prompt_lookup", {}, False)
