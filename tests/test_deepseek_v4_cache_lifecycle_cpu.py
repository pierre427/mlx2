"""Exercise the same guarded probe lifecycle on CPU before a GPU lease."""
import importlib
from pathlib import Path
import sys


def test_generic_disk_and_apc_lifecycle_preserves_packed_uint8(tmp_path,monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/'scripts/research'
    monkeypatch.syspath_prepend(str(scripts))
    probe=importlib.import_module('probe_deepseek_v4_cache_lifecycle')
    cases=probe.run_lifecycle(tmp_path,gpu=False)
    assert len(cases)==5 and all(c['passed'] for c in cases)
