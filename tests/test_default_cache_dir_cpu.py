"""The APCv2 disk tier is on by default, in a private per-port directory.

Without it, pressure evictions destroy entries and long-context warm reuse
depends on free memory alone (Flash-Next 16K B4 warm hits 1/4 vs 4/4,
qualification/runs/fn-fixes-20261028/cachedir-l4).
"""

import pytest

from mlx2.server import DEFAULT_CACHE_DIR_ROOT, build_parser, resolve_cache_dir


def _args(*argv):
    return build_parser().parse_args(["--model", "m", *argv])


def test_default_is_a_per_port_directory():
    path, source = resolve_cache_dir(_args("--port", "8282"))
    assert source == "default_per_port"
    assert path == str(DEFAULT_CACHE_DIR_ROOT / "127.0.0.1-8282")
    other, _ = resolve_cache_dir(_args("--port", "8283"))
    assert other != path  # live servers never share a swept scratch directory


def test_explicit_disabled_and_persist_dir_win():
    assert resolve_cache_dir(_args("--cache-dir", "/x")) == ("/x", "explicit")
    assert resolve_cache_dir(_args("--no-cache-dir")) == (None, "disabled")
    assert resolve_cache_dir(_args("--apc-persist-dir", "/p")) == (None, "explicit")
    with pytest.raises(ValueError):
        resolve_cache_dir(_args("--no-cache-dir", "--cache-dir", "/x"))


def test_ephemeral_port_gets_no_default():
    assert resolve_cache_dir(_args("--port", "0")) == (None, "none_ephemeral_port")
