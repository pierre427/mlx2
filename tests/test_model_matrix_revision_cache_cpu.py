"""The inventory must never emit a stale target/drafter content pin."""

import importlib.util
import json
import os
from pathlib import Path

import pytest


def _cache_type():
    path = Path(__file__).resolve().parents[1] / "scripts/model_matrix/inventory.py"
    spec = importlib.util.spec_from_file_location("matrix_inventory_revision_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._RevisionCache


@pytest.fixture
def target(tmp_path):
    (tmp_path / "config.json").write_text('{"hidden_size": 8}')
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.weight": "model.safetensors"},
    }))
    (tmp_path / "model.safetensors").write_bytes(b"test payload")
    return tmp_path


def test_unchanged_inventory_pin_survives_cache_reload(target):
    cls = _cache_type()
    cache = cls({})
    key = ("qwen38_target", str(target))
    cache[key] = "old-pin"
    restored = cls(json.loads(json.dumps(cache.rows)))
    assert key in restored
    assert restored[key] == "old-pin"


@pytest.mark.parametrize("name,new_bytes", [
    ("config.json", b'{"hidden_size": 9}'),
    ("model.safetensors.index.json", b'{"weight_map":{"other.weight":"model.safetensors"}}'),
    ("model.safetensors", b"new payload!"),
])
def test_inventory_pin_invalidates_on_changed_revision_inputs(target, name, new_bytes):
    cache = _cache_type()({})
    key = ("qwen38_target", str(target))
    cache[key] = "old-pin"
    path = target / name
    stat = path.stat()
    path.write_bytes(new_bytes)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert key not in cache
