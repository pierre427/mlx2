"""Receipt serialization must preserve all telemetry labels and outcomes."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('kimi_lora_round_metrics', Path(__file__).parents[1] / 'scripts/research/kimi_lora_round.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_receipt_preserves_tuple_label_metrics(tmp_path):
    from mlx2.runtime.weight_space_metrics import WeightSpaceMetrics
    metrics = WeightSpaceMetrics()
    metrics.record('program_export', 'ok', bytes_count=123)
    metrics.record('program_export', 'error')
    metrics.record('program_load', 'ok')
    original = metrics.snapshot()
    target = tmp_path / 'receipt.json'
    runner.atomic(target, {'status': 'completed', 'weight_space_metrics': runner.json_metrics(original)})
    restored = json.loads(target.read_text())['weight_space_metrics']
    assert restored['events'] == [
        {'operation': 'program_export', 'outcome': 'error', 'count': 1},
        {'operation': 'program_export', 'outcome': 'ok', 'count': 1},
        {'operation': 'program_load', 'outcome': 'ok', 'count': 1},
    ]
    assert restored['bytes']['program_export'] == 123
    assert ('program_export', 'ok') in original['events']
