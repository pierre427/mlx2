"""Host-only consumed-spec and exact-resume guards for opt-in training paths."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT/'scripts/research'))
SPEC = importlib.util.spec_from_file_location('kimi_lora_training_options_runner', ROOT/'scripts/research/kimi_lora_round.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def spec(tmp_path, **kwargs):
    return {'model': str(tmp_path/'copy'), 'original_model': str(tmp_path/'original'),
            'out': str(tmp_path/'out'), 'mode': 'train', 'data': 'unused.jsonl', **kwargs}


@pytest.mark.parametrize('value', [0, -1, True, 1.25, '128', []])
def test_invalid_projection_size_refused_before_runtime_import(tmp_path, monkeypatch, value):
    p = tmp_path/'spec.json'
    p.write_text(json.dumps(spec(tmp_path, projection_chunk_size=value)))
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'mlx' or name.startswith(('mlx.', 'mlx2.adapters', 'mlx2.experimental')):
            raise AssertionError('Runtime import before spec rejection')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ValueError, match='projection_chunk_size'):
        runner.run(p)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('value', [None, 'auto', False, []])
def test_unknown_training_mla_refused(tmp_path, value):
    with pytest.raises(ValueError, match='training_mla'):
        runner.validate_spec(spec(tmp_path, training_mla=value))


@pytest.mark.parametrize('option', [{'projection_chunk_size':128}, {'training_mla':'padded_sdpa'}])
def test_evaluation_cannot_silently_ignore_training_option(tmp_path, option):
    with pytest.raises(ValueError, match='only'):
        runner.validate_spec(spec(tmp_path, mode='evaluate', **option))


def parent_spec(tmp_path, parent_options=None):
    parent = spec(tmp_path, **(parent_options or {}))
    p = tmp_path/'parent.json'
    payload = json.dumps(parent).encode()
    p.write_bytes(payload)
    return p, hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize('parent_options,child_options', [
    ({}, {'projection_chunk_size':128}), ({'projection_chunk_size':128}, {}),
    ({'projection_chunk_size':128}, {'projection_chunk_size':64}),
    ({}, {'training_mla':'padded_sdpa'}),
    ({'training_mla':'padded_sdpa'}, {}),
])
def test_resume_cannot_change_loss_or_attention_profile(tmp_path, parent_options, child_options):
    p, digest = parent_spec(tmp_path, parent_options)
    child = spec(tmp_path, resume_checkpoint='checkpoint', resume_from_spec=str(p),
                 resume_from_spec_sha256=digest, **child_options)
    with pytest.raises(ValueError, match='profile changed'):
        runner.resume_training_binding(child)


@pytest.mark.parametrize('options', [{}, {'projection_chunk_size':128},
                                     {'training_mla':'padded_sdpa', 'projection_chunk_size':128}])
def test_consumed_parent_spec_snapshot_is_bound(tmp_path, monkeypatch, options):
    p, digest = parent_spec(tmp_path, options)
    child = spec(tmp_path, resume_checkpoint='checkpoint', resume_from_spec=str(p),
                 resume_from_spec_sha256=digest, **options)
    real = runner.read_bound_bytes
    def replace_after_read(path, **kwargs):
        result = real(path, **kwargs)
        p.write_text('{"changed_after_snapshot":true}')
        return result
    monkeypatch.setattr(runner, 'read_bound_bytes', replace_after_read)
    result = runner.resume_training_binding(child)
    assert result['sha256'] == digest
    assert result['training_loss'] == runner.training_loss_profile(child)
    assert result['training_attention'] == runner.training_attention_profile(child)


def test_default_profile_and_resume_hash_refusal(tmp_path):
    assert runner.training_loss_profile(spec(tmp_path)) == runner.training_loss_profile(
        spec(tmp_path, projection_chunk_size=None))
    assert runner.training_attention_profile(spec(tmp_path))['mla'] == 'expanded'
    p, _ = parent_spec(tmp_path)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        runner.resume_training_binding(spec(tmp_path, resume_checkpoint='checkpoint',
            resume_from_spec=str(p), resume_from_spec_sha256='0'*64))


@pytest.mark.parametrize('name,value', [
    ('MLX_SDPA_VJP_BLOCK', '512'), ('MLX_SDPA_VJP_BQ', '32'),
    ('MLX_SDPA_VJP_BK', '64'), ('MLX_SDPA_VJP_FALLBACK', '1'),
    ('MLX_SDPA_VJP_BLOCK', ''), ('MLX_SDPA_VJP_BLOCK', 'nonsense')])
def test_padded_sdpa_refuses_environment_overrides(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        runner.training_attention_profile(spec(tmp_path, training_mla='padded_sdpa'))
    assert runner.training_attention_profile(spec(tmp_path)) == {'kda':'reference','mla':'expanded'}


def test_padded_sdpa_binds_effective_default_environment(tmp_path, monkeypatch):
    defaults = {'MLX_SDPA_VJP_BLOCK':1024, 'MLX_SDPA_VJP_BQ':0,
                'MLX_SDPA_VJP_BK':0, 'MLX_SDPA_VJP_FALLBACK':0}
    for name in defaults: monkeypatch.delenv(name, raising=False)
    absent = runner.training_attention_profile(spec(tmp_path, training_mla='padded_sdpa'))
    for name, value in defaults.items(): monkeypatch.setenv(name, str(value))
    explicit = runner.training_attention_profile(spec(tmp_path, training_mla='padded_sdpa'))
    assert absent == explicit
    assert absent['mlx_sdpa_vjp_effective_environment'] == defaults
