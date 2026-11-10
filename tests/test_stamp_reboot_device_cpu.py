# SPDX-License-Identifier: MIT
"""Persisted file seals must survive a reboot that renumbers the volume (st_dev), 2026-11-09."""
import os

import pytest

from mlx2.runtime.immutable_weight_pages import file_stamp, same_stamp
from mlx2.adapters.deepseek_v41_source_artifact import checked_file


def test_same_stamp_ignores_only_the_device(tmp_path):
    f = tmp_path / "w.bin"; f.write_bytes(b"abc")
    live = file_stamp(f.stat())
    assert same_stamp(live, dict(live, device=live["device"] + 4))
    assert same_stamp(dict(live, dev=1), dict(live, dev=2))
    for key in ("size", "mtime_ns", "ctime_ns", "inode"):
        assert not same_stamp(live, dict(live, **{key: live[key] + 1}))
    missing = dict(live); missing.pop("inode")
    assert not same_stamp(live, missing)
    assert not same_stamp(live, None)


def test_source_checked_file_accepts_renumbered_volume_but_not_edits(tmp_path):
    f = tmp_path / "config.json"; f.write_text("{}")
    seal = file_stamp(f.stat())
    assert checked_file(tmp_path, "config.json", dict(seal, device=seal["device"] + 4)) == f
    os.utime(f, ns=(seal["mtime_ns"] + 10, seal["mtime_ns"] + 10))
    with pytest.raises(ValueError, match="changed"):
        checked_file(tmp_path, "config.json", seal)
