"""CPU checks for the research-only Q4 call-shape observer."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1] / "scripts/research/q4_projection_m_census.py"
spec = spec_from_file_location("q4_projection_m_census", SOURCE)
census_module = module_from_spec(spec)
spec.loader.exec_module(census_module)


def test_single_row_rollback_representation_is_logically_identical():
    entry = SimpleNamespace(_rollback_position=0, _rollback_positions=None)
    before = census_module._cache_positions([entry])
    entry._rollback_positions = [0]
    assert census_module._cache_positions([entry]) == before == [0]


def test_phase_tagged_batch_decode_records_flattened_m():
    observer = census_module.Census()
    module = SimpleNamespace(weight=SimpleNamespace(shape=(17408, 640)),
                             bits=4, group_size=64)
    x = SimpleNamespace(shape=(8, 1, 5120))
    with observer.phase("b8_equal_prefix_decode"):
        observer.record("model.layers.0.mlp.gate_proj", module, x)
    assert observer.rows() == [{
        "phase": "b8_equal_prefix_decode", "kind": "gate_proj",
        "batch": 8, "sequence": 1, "M": 8, "K": 5120, "N": 17408,
        "bits": 4, "group_size": 64, "calls": 1,
        "example_module": "model.layers.0.mlp.gate_proj",
    }]
