"""APCv2 disk tier byte cap (--cache-dir-max-bytes, 2026-11-25).

The tier used a fixed 64 GiB cap; one loopback server reached 63 GiB of spill
on a volume with ~270 GiB free.  The default is now a fifth of the free space
at startup, clamped to 4..64 GiB, and the cap reaches the APCv2 store.
"""

from types import SimpleNamespace

import pytest

from mlx2 import server
from mlx2.server import build_parser, resolve_cache_disk_cap, serving_engine_kwargs

GiB = 1 << 30


def _args(tmp_path, *argv):
    return build_parser().parse_args(["--model", "m", "--cache-dir", str(tmp_path), *argv])


def _free(gib):
    return lambda _path: SimpleNamespace(free=int(gib * GiB))


@pytest.mark.parametrize("free, cap", [(270, 54), (1000, 64), (10, 4), (40, 8)])
def test_default_is_a_fifth_of_free_space_clamped(tmp_path, free, cap):
    args = _args(tmp_path)
    got, source = resolve_cache_disk_cap(args, disk_usage=_free(free))
    assert source == "default_free_fraction"
    assert got == int(min(64, max(4, free * 0.2)) * GiB) and round(got / GiB) == cap
    assert args.apc_session_pinned_disk_bytes_global == got  # unset pin cap follows the tier


def test_explicit_cap_wins_and_bounds_the_pin_cap(tmp_path):
    args = _args(tmp_path, "--cache-dir-max-bytes", str(2 * GiB))
    assert resolve_cache_disk_cap(args, disk_usage=_free(500)) == (2 * GiB, "explicit")
    too_big = _args(tmp_path, "--cache-dir-max-bytes", str(2 * GiB),
                    "--apc-session-pinned-disk-bytes-global", str(3 * GiB))
    with pytest.raises(ValueError, match="exceeds the disk tier cap"):
        resolve_cache_disk_cap(too_big, disk_usage=_free(500))
    with pytest.raises(ValueError, match="positive"):
        resolve_cache_disk_cap(_args(tmp_path, "--cache-dir-max-bytes", "0"))


def test_no_disk_tier_keeps_the_historical_cap():
    args = build_parser().parse_args(["--model", "m", "--no-cache-dir", "--port", "0"])
    assert resolve_cache_disk_cap(args) == (64 * GiB, "default_max")


def test_unreadable_volume_keeps_the_historical_cap(tmp_path):
    def broken(_path):
        raise OSError("no statfs")
    assert resolve_cache_disk_cap(_args(tmp_path), disk_usage=broken) == (64 * GiB, "default_max")


def test_cap_reaches_the_engine_and_the_store(tmp_path, monkeypatch):
    from mlx2 import serving
    from mlx2.runtime import apc_v2

    args = _args(tmp_path, "--cache-dir-max-bytes", str(3 * GiB))
    resolve_cache_disk_cap(args)
    kwargs = serving_engine_kwargs(args, None, native_mtp=False, approximate_kv=None, max_request_bytes=1 << 20)
    assert kwargs["apc_disk_max_bytes"] == 3 * GiB
    assert kwargs["apc_session_pinned_disk_bytes_global"] == 3 * GiB
    # Without the resolver (embedders, older tests) the historical cap applies.
    bare = build_parser().parse_args(["--model", "m"])
    kwargs = serving_engine_kwargs(bare, None, native_mtp=False, approximate_kv=None, max_request_bytes=1 << 20)
    assert kwargs["apc_disk_max_bytes"] == server.CACHE_DISK_DEFAULT_MAX
    assert kwargs["apc_session_pinned_disk_bytes_global"] == server.CACHE_DISK_DEFAULT_MAX
    # The store takes the engine's cap, not a constant.
    src = (serving.__file__)
    assert "idle_disk_max_bytes=self.apc_disk_max_bytes" in open(src).read()
    store = apc_v2.APCv2(layout_name="t", max_size=4, max_bytes=1 << 20, max_tokens=64,
                         idle_disk_seconds=0, idle_disk_dir=None, idle_disk_max_bytes=3 * GiB)
    assert store._idle_disk_max_bytes == 3 * GiB
