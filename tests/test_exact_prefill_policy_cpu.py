"""Exact projection law must not switch at a cold/warm prefill row boundary."""
from mlx2.runtime.lane.policy import resolve


def test_exact_defaults_to_one_chunked_projection_law():
    policy = resolve({"backend": "mpp", "formats": {"q4": 1}}, mode="exact")
    assert policy["chunk_above_max"] is True
    assert policy["max_rows"] == 32
    assert policy["sources"]["chunk_above_max"] == "exact_mode_default"


def test_explicit_stock_crossover_control_is_preserved():
    for keyword in ("adapter", "overrides"):
        policy = resolve({"backend": "mpp", "formats": {"q4": 1}}, mode="exact", **{keyword: {"chunk_above_max": False}})
        assert policy["chunk_above_max"] is False


def test_crossover_and_off_modes_do_not_acquire_exact_mode_default():
    for mode in ("crossover", "off"):
        policy = resolve({"backend": "mpp", "formats": {"q4": 1}}, mode=mode)
        assert policy["chunk_above_max"] is False
