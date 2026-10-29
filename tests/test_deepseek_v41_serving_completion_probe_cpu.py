"""CPU plan/command/bounds only: never execute the smoke or obtain a lease."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).parents[1] / 'scripts/research/probe_deepseek_v41_serving_completion.py'
    spec = importlib.util.spec_from_file_location('v41_completion_probe_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_baseline_is_original_ordinary_and_server_selects_explicit_profile(probe, tmp_path):
    assert probe.baseline_policy({'draft_model': 'embedded_dspark', 'deepseek_ced': 'dependency_exact'}) == {}
    spec = {'policy': {'draft_model': 'embedded_dspark'}, 'model': '/exact/source', 'port': 8291,
            'max_context': 4096, 'prefill_step': 128, 'cache_bytes': 512 << 20}
    command = probe.server_command(spec, tmp_path)
    assert '--external-draft' in command and '--ordinary' not in command
    assert command[command.index('--host') + 1] == '127.0.0.1'
    assert command[command.index('--execution-policy') + 1] == str(tmp_path / 'policy.json')
    spec['policy'] = {'deepseek_physical_batch': 'b4'}
    command = probe.server_command(spec, tmp_path)
    assert '--ordinary' in command
    assert command[command.index('--max-lanes') + 1] == '4'
    assert command[command.index('--max-inflight') + 1] == '4'


def test_prepare_freezes_bound_policy_and_refuses_nonempty_or_changed_plan(probe, monkeypatch, tmp_path):
    from mlx2.adapters import deepseek_v41_source_artifact
    monkeypatch.setattr(deepseek_v41_source_artifact, 'inspect_artifact', lambda path: {'identity': {'fingerprint': 'c' * 64}})
    monkeypatch.setattr(probe, 'source_manifest', lambda root: {'src/example.py': 'a' * 64})
    monkeypatch.setattr(probe.subprocess, 'check_output', lambda *a, **kw: 'source-revision\n')
    policy = tmp_path / 'input-policy.json'
    policy.write_text(json.dumps({'deepseek_kv_storage': 'native_packed', 'deepseek_candidate_score': 'candidate_only', 'deepseek_ced': 'dependency_exact', 'constrained_tool_grammar': True}))
    args = NS(output=tmp_path / 'evidence', model=tmp_path / 'Source', policy=policy, request_json=None,
              mcp_url='http://127.0.0.1:8766/mcp', port=8291, max_context=4096, prefill_step=128,
              cache_bytes=512 << 20, deadline_seconds=1800, thermal_max_c=60)
    spec = probe.prepare(args)
    assert spec['checkpoint'] == 'Source@' + 'c' * 64
    assert spec['qualification'] is False and spec['performance_claim'] is False
    assert spec['server_policy']['moe_expert_streaming'] == {'enabled': True, 'cache_gib': 24}
    assert 'constrained_tool_grammar' not in spec['policy']
    assert probe.verify_plan(args.output) == spec
    with pytest.raises(ValueError, match='nonempty'):
        probe.prepare(args)
    monkeypatch.setattr(probe, 'source_manifest', lambda root: {'src/example.py': 'b' * 64})
    with pytest.raises(ValueError, match='source changed'):
        probe.verify_plan(args.output)
    (args.output / 'results.json').write_text('{}')
    with pytest.raises(ValueError, match='prior evidence'):
        probe.verify_plan(args.output)


def test_sse_tool_and_logprob_receipts_preserve_real_events(probe, tmp_path):
    path = tmp_path / 'calls.sse'
    events = [{'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call_1', 'function': {'name': 'ns::f', 'arguments': '{"x":'}}]}, 'logprobs': {'content': [{'id': 5, 'logprob': -.2}]}}]},
              {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '1}'}}]}}]}]
    path.write_text(''.join('data: ' + json.dumps(event) + '\n\n' for event in events) + 'data: [DONE]\n\n')
    assert probe._calls_from_sse(path) == [{'id': 'call_1', 'name': 'ns::f', 'arguments': '{"x":1}'}]
    assert list(probe._logprobs(path)) == [-.2]


def test_replay_requires_explicit_adaptive_refusal_or_real_canonical_hit(probe):
    def result(cached):
        return {'receipt': {'cache': 'apcv2', 'route': 'ordinary',
                            'route_receipt': 'unqualified', 'cached_tokens': cached}}
    status = {'settings': {'apcv2_reuse': {'enabled': False,
               'reason': 'adaptive_prefill_partitions_not_qualified'}}}
    assert probe.validate_replay_receipts({}, status, result(0), result(0)) == 'independent_repeated_execution_apcv2_disabled'
    with pytest.raises(RuntimeError, match='explicitly disable'):
        probe.validate_replay_receipts({}, {}, result(0), result(0))
    with pytest.raises(RuntimeError, match='receipt'):
        probe.validate_replay_receipts({}, status, result(0), result(4))
    policy = {'deepseek_source_prefill_partition': 'canonical_m1_v1'}
    assert probe.validate_replay_receipts(policy, {}, result(0), result(4)) == 'apcv2_cold_warm'
    with pytest.raises(RuntimeError, match='receipt'):
        probe.validate_replay_receipts(policy, {}, result(0), result(0))
