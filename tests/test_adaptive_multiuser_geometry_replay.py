import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "geometry_replay",
    ROOT / "scripts/research/adaptive_multiuser_geometry_replay.py",
)
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


def test_trace_is_multiuser_multiturn_and_includes_speculation():
    receipt = REPLAY.build_receipt()
    assert receipt["traffic"]["users"] == 4
    assert receipt["traffic"]["turns_per_user"] == 2
    assert receipt["traffic"]["speculative_requests"] == 6


def test_adaptive_uses_mixed_and_packed_geometry_without_speed_claim():
    receipt = REPLAY.build_receipt()
    adaptive = receipt["arms"]["adaptive"]
    assert adaptive["geometry_rounds"]["mixed_decode_first"] > 0
    assert adaptive["geometry_rounds"]["packed"] > 0
    assert adaptive["verification_rows"] > 0
    assert receipt["performance_claim"] is False
    assert receipt["adaptive_vs_padded"]["charged_rows_avoided"] > 0
