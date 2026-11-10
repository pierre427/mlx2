"""Tier-declared shard symlinks for the DS4.1 Source artifact (storage tiering across volumes)."""
import json
import os

import pytest

from mlx2.adapters.deepseek_v41_source_artifact import TIERS, checked_file
from mlx2.runtime.immutable_weight_pages import file_stamp


def _layout(tmp_path):
    root, tier, other = tmp_path / "root", tmp_path / "tier", tmp_path / "other"
    for d in (root, tier, other):
        d.mkdir()
    (root / "a.safetensors").write_bytes(b"local")
    (tier / "b.safetensors").write_bytes(b"tiered")
    (other / "c.safetensors").write_bytes(b"stray")
    (root / "b.safetensors").symlink_to(tier / "b.safetensors")
    (root / "c.safetensors").symlink_to(other / "c.safetensors")
    return root, tier, other


def test_symlink_refused_without_declaration(tmp_path):
    root, _, _ = _layout(tmp_path)
    assert checked_file(root, "a.safetensors") == root / "a.safetensors"
    with pytest.raises(ValueError, match="outside declared tiers"):
        checked_file(root, "b.safetensors")


def test_declared_tier_returns_target_and_stamps_it(tmp_path):
    root, tier, _ = _layout(tmp_path)
    (root / TIERS).write_text(json.dumps({"schema": "mlx2.v41.source-tiers.v1", "directories": [str(tier)]}))
    path = checked_file(root, "b.safetensors")
    assert path == (tier / "b.safetensors").resolve() and not path.is_symlink()
    stamp = file_stamp(path.lstat())
    assert checked_file(root, "b.safetensors", stamp) == path
    with pytest.raises(ValueError, match="outside declared tiers"):
        checked_file(root, "c.safetensors")
    # opening the returned path with O_NOFOLLOW works (it is the regular target)
    os.close(os.open(path, os.O_RDONLY | os.O_NOFOLLOW))


def test_renamed_target_and_changed_target_refused(tmp_path):
    root, tier, _ = _layout(tmp_path)
    (root / TIERS).write_text(json.dumps({"schema": "mlx2.v41.source-tiers.v1", "directories": [str(tier)]}))
    (tier / "x.safetensors").write_bytes(b"x")
    (root / "d.safetensors").symlink_to(tier / "x.safetensors")
    with pytest.raises(ValueError, match="outside declared tiers"):
        checked_file(root, "d.safetensors")
    stamp = file_stamp(checked_file(root, "b.safetensors").lstat())
    (tier / "b.safetensors").write_bytes(b"replaced bytes")
    with pytest.raises(ValueError, match="changed"):
        checked_file(root, "b.safetensors", stamp)


def test_malformed_or_symlinked_declaration_refused(tmp_path):
    root, tier, _ = _layout(tmp_path)
    (root / TIERS).write_text(json.dumps({"schema": "mlx2.v41.source-tiers.v1", "directories": ["relative"]}))
    with pytest.raises(ValueError, match="malformed"):
        checked_file(root, "b.safetensors")
    (root / TIERS).unlink()
    decl = tmp_path / "decl.json"
    decl.write_text(json.dumps({"schema": "mlx2.v41.source-tiers.v1", "directories": [str(tier)]}))
    (root / TIERS).symlink_to(decl)
    with pytest.raises(ValueError, match="regular file"):
        checked_file(root, "b.safetensors")
