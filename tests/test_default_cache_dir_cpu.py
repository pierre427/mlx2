"""The APCv2 disk tier is on by default, in a private per-port directory.

Without it, pressure evictions destroy entries and long-context warm reuse
depends on free memory alone (Flash-Next 16K B4 warm hits 1/4 vs 4/4,
qualification/runs/fn-fixes-20261028/cachedir-l4).
"""

from pathlib import Path

import pytest

from mlx2.server import DEFAULT_CACHE_DIR_ROOT, build_parser, resolve_cache_dir


def _args(*argv):
    return build_parser().parse_args(["--model", "m", *argv])


def test_default_is_a_per_port_directory(tmp_path, monkeypatch):
    import mlx2.server as server

    root = tmp_path / "apcv2-idle"
    monkeypatch.setattr(server, "DEFAULT_CACHE_DIR_ROOT", root)
    path, source = resolve_cache_dir(_args("--port", "8282"))
    assert source == "default_per_port"
    assert path == str(root / "127.0.0.1-8282")
    assert Path(path).is_dir()  # created and checked up front, not at model load
    other, _ = resolve_cache_dir(_args("--port", "8283"))
    assert other != path  # live servers never share a swept scratch directory


def test_default_root_is_under_the_user_cache_directory():
    assert DEFAULT_CACHE_DIR_ROOT == Path.home() / "Library/Caches/mlx2/apcv2-idle"


def test_unusable_default_root_serves_in_memory_labelled(tmp_path, monkeypatch):
    """The implicit disk tier is an optimisation, not a request: a daemon HOME,
    a sandboxed launch or a root-owned leaf from a sudo run must not keep the
    model from loading (APCv2 fails closed on a directory it cannot own)."""
    import os

    import mlx2.server as server

    if os.getuid() == 0:
        pytest.skip("root can write anywhere")
    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    monkeypatch.setattr(server, "DEFAULT_CACHE_DIR_ROOT", unwritable / "apcv2-idle")
    try:
        path, source = resolve_cache_dir(_args("--port", "8282"))
    finally:
        unwritable.chmod(0o700)
    assert path is None
    assert source.startswith("default_unavailable:PermissionError")

    root = tmp_path / "linked"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "127.0.0.1-8282").symlink_to(elsewhere)
    monkeypatch.setattr(server, "DEFAULT_CACHE_DIR_ROOT", root)
    assert resolve_cache_dir(_args("--port", "8282")) == (None, "default_unavailable:symlink")

    # An explicit directory is never silently dropped: it stays the operator's.
    assert resolve_cache_dir(_args("--cache-dir", str(unwritable / "x"))) == (
        str(unwritable / "x"), "explicit",
    )


def test_no_cache_dir_conflicts_with_a_persist_dir():
    # The persistent tier spills too; "disk tier: off" would be a false label.
    with pytest.raises(ValueError, match="apc-persist-dir"):
        resolve_cache_dir(_args("--no-cache-dir", "--apc-persist-dir", "/p"))


def test_explicit_disabled_and_persist_dir_win():
    assert resolve_cache_dir(_args("--cache-dir", "/x")) == ("/x", "explicit")
    assert resolve_cache_dir(_args("--no-cache-dir")) == (None, "disabled")
    assert resolve_cache_dir(_args("--apc-persist-dir", "/p")) == (None, "explicit")
    with pytest.raises(ValueError):
        resolve_cache_dir(_args("--no-cache-dir", "--cache-dir", "/x"))


def test_ephemeral_port_gets_no_default():
    assert resolve_cache_dir(_args("--port", "0")) == (None, "none_ephemeral_port")
