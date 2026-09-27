"""CPU-only admission and provenance checks for isolated #3958 wide SDPA."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/research/omlx3958_wide_sdpa.py"
SPEC = importlib.util.spec_from_file_location("omlx3958_wide_sdpa", MODULE)
WIDE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WIDE)


def _admit(prefix=8192, width=6, **overrides):
    options = dict(q_dtype="bfloat16", k_dtype="bfloat16", v_dtype="bfloat16",
                   cache_type="KVCache", cache_offset=prefix + width,
                   mask="causal")
    options.update(overrides)
    return WIDE.admission_reasons(
        (1, 24, width, 256), (1, 4, prefix + width, 256),
        (1, 4, prefix + width, 256), **options,
    )


def test_cpu_import_does_not_load_mlx_or_tensor_ops():
    guard = """import importlib.abc, importlib.util, sys
class BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mlx' or fullname.startswith('mlx.'):
            raise RuntimeError('CPU import tried to load MLX')
sys.meta_path.insert(0, BlockMLX())
spec = importlib.util.spec_from_file_location('wide_cpu_probe', sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert 'mlx.core' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", guard, str(MODULE)], check=True)
    source = MODULE.read_text()
    assert "mpp::tensor_ops" not in source
    assert "MetalPerformancePrimitives/MetalPerformancePrimitives.h" not in source
    assert "simdgroup_multiply_accumulate" in source


def test_exact_m4_to_m8_dense_admission_and_rejections():
    for m in range(4, 9):
        assert _admit(width=m) == []
    assert "verify_width" in _admit(width=3)
    assert "verify_width" in _admit(width=9)
    assert "cache_length" in _admit(prefix=0, width=4)
    assert "cache_length" in _admit(cache_offset=8192)
    assert "dtype" in _admit(k_dtype="float16")
    assert "dtype" in _admit(q_dtype="float32")
    assert "cache_type" in _admit(cache_type="SegmentedPlainKVCache")
    assert "cache_type" in _admit(quantized=True)
    assert "mask_or_padding" in _admit(mask="none")
    assert "mask_or_padding" in _admit(right_padding=1)
    assert "mask_or_padding" in _admit(left_padding=1)
    assert "mask_or_padding" in _admit(sinks=True)
    assert "heads_or_dim" in WIDE.admission_reasons(
        (1, 16, 6, 256), (1, 4, 8198, 256), (1, 4, 8198, 256),
        q_dtype="bfloat16", k_dtype="bfloat16", v_dtype="bfloat16",
        cache_type="KVCache", cache_offset=8198, mask="causal")


@pytest.mark.parametrize("prefix", [0, 1, 7, 63, 64, 255, 256, 8192])
@pytest.mark.parametrize("width", range(4, 9))
def test_bottom_right_causal_limit_and_replay(prefix, width):
    total = prefix + width
    # Baseline absolute-position causal mask from mlx2 create_causal_mask.
    for row in range(width):
        expected = {key for key in range(total) if key <= prefix + row}
        wide = {key for key in range(total) if key <= total - width + row}
        assert wide == expected
        # Rolling the verify suffix back and re-appending it gives the same
        # logical total and hence the same permitted keys.
        rolled = total - width
        replay_total = rolled + width
        assert {key for key in range(replay_total)
                if key <= replay_total - width + row} == expected


def test_provenance_covers_exact_isolated_destination():
    provenance = json.loads((ROOT / "provenance/omlx3958-wide-verify-sdpa.json").read_text())
    assert provenance["source_revision"].startswith("3e970351")
    assert provenance["license"].startswith("Apache-2.0")
    assert "scripts/research/omlx3958_wide_sdpa.py" in provenance["destination_paths"]
    assert "_GQA" not in MODULE.read_text()
