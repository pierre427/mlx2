"""Failure-path regressions; synthetic files only, no checkpoint load."""
import json
from pathlib import Path

import numpy as np
import pytest
from test_amoeba_cli_cpu import _build, _ctx

from mlx2.experimental.amoeba import cli
from mlx2.experimental.amoeba import extract as X
from mlx2.experimental.amoeba import stages as S
from mlx2.experimental.amoeba.state import file_outputs


@pytest.mark.parametrize('change', ['removed', 'modified'])
def test_resume_checks_artifact_bytes(tmp_path, change):
    b = _build(tmp_path)
    p = b.stage_dir('core') / 'core.npz'
    p.write_bytes(b'original')
    b.end(b.begin('core', {'commit': 'test', 'dirty': False}), 'done', file_outputs({'core': p}))
    if change == 'removed':
        p.unlink()
    else:
        p.write_bytes(b'changed!')
    with pytest.raises(SystemExit, match='redo core'):
        b.is_done('core')
    with pytest.raises(SystemExit, match='redo core'):
        b.begin('map', {'commit': 'test', 'dirty': False})


def test_partial_stage_refuses_changed_inputs(tmp_path):
    b = _build(tmp_path)
    b.end(b.begin('core', {'commit': 'test', 'dirty': False}), 'failed', error='interrupted')
    b.rec['config']['compute']['core_steps'] += 1
    with pytest.raises(SystemExit, match='redo core'):
        b.is_done('core')


@pytest.mark.parametrize('stage,section,key,value', [
    ('core', 'options', 'gen_backend', 'server'),
    ('core', 'options', 'gen_route', '--ordinary'),
    ('core', 'config', 'grad_gate', ['--different-gate']),
    ('map', 'options', 'cpu', True),
    ('map', 'inputs', 'taxonomy', {'sha256': 'different'}),
])
def test_cache_keys_cover_execution_inputs(tmp_path, stage, section, key, value):
    b = _build(tmp_path)
    b.rec['plan'] = {'moe_units': 'expert'}
    b.rec['inputs'] = {'probe_corpus': {'sha256': 'p'}, 'core_corpora': [{'sha256': 'c'}],
                       'taxonomy': {'sha256': 't'}}
    b.rec['options'] = {'cpu': False, 'gen_backend': 'local', 'gen_route': ''}
    np.savez(b.stage_dir('core') / 'core.npz', core=np.ones((1, 4)))
    before = X._cache_key(b, stage)
    b.rec[section][key] = value
    assert X._cache_key(b, stage) != before


def test_map_retries_partial_label_arrays_and_passes_taxonomy(tmp_path, monkeypatch):
    b = _build(tmp_path)
    b.rec['inputs']['probe_corpus']['path'] = str(tmp_path / 'probes')
    b.rec['inputs']['taxonomy'] = {'path': str(tmp_path / 'custom-taxonomy.json')}
    b.rec['config']['compute']['map_probes_per_node'] = 0
    cap = b.stage_dir('map') / 'capture'
    cap.mkdir()
    (cap / 'meta.json').write_text(json.dumps({'fingerprint': b.info['fingerprint'], 'widths': [4], 'sequences': 2}))
    cl = b.out / 'map/clusters'
    cl.mkdir()
    for n in ['clusters.json', 'clusters.npz', 'virtual_experts.npz']:
        (cl / n).write_text('{}')
    arrays = b.out / 'map/label-arrays'
    arrays.mkdir()
    (arrays / 'partial.npy').write_bytes(b'incomplete')
    monkeypatch.setattr(S, 'map_core_mask', lambda *a: np.zeros((1, 4), bool))
    def child(cmd, **kw):
        assert not arrays.exists(), 'stale arrays prevent the labeller fresh_dir from restarting'
        assert cmd[cmd.index('--taxonomy') + 1] == b.rec['inputs']['taxonomy']['path']
        lab = b.out / 'map/labels'
        lab.mkdir()
        (lab / 'labels.json').write_text(json.dumps({'clusters': [], 'summary': {}}))
    monkeypatch.setattr(S, 'run_child', child)
    S.run_map(b, _ctx(events=tmp_path / 'events'))


def test_inputs_resolve_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pc, tax, core, man = [Path(n) for n in ('p.jsonl', 't.json', 'c.jsonl', 'm.json')]
    pc.write_text('{}\n')
    tax.write_text('{"version":"test","nodes":[]}')
    core.write_text('{}\n')
    man.write_text(json.dumps({'corpus_sha256': S.paths.sha256_file(pc)}))
    args = cli.parse(['build', '--model', 'toy', '--out', 'out', '--probe-corpus', str(pc),
                      '--probe-manifest', str(man), '--taxonomy', str(tax), '--core-corpus', str(core)])
    inp = cli.resolve_inputs(args, tmp_path / 'tmp')
    for f in [inp['probe_corpus'], inp['probe_manifest'], inp['taxonomy'], *inp['core_corpora']]:
        assert Path(f['path']).is_absolute()


def _failed_box_fixture(tmp_path, monkeypatch):
    b = _build(tmp_path)
    b.rec['plan'] = {}
    bd = b.stage_dir('boxes') / 'boxset' / 'topic'
    bd.mkdir(parents=True)
    for name in ['docs', 'val', 'held', 'prompts_carve', 'prompts_val', 'prompts_held']:
        (bd / f'{name}.jsonl').write_text('')
    (bd.parent / 'boxes.json').write_text(json.dumps({'boxes': [
        {'box': 'topic', 'node': 'sci.physics', 'kind': 'topic'}]}))
    gen = b.out / 'boxes/gen'
    gen.mkdir()
    (gen / 'gen.jsonl').write_text('')
    carves = b.out / 'boxes/carves'
    carves.mkdir()
    (carves / 'verdicts.jsonl').write_text(json.dumps(
        {'box': 'topic', 'node': 'sci.physics', 'verdict': 'failed', 'error': 'gradient gate failed'}) + '\n')
    monkeypatch.setattr(S, 'generate', lambda *a, **kw: {})
    monkeypatch.setattr(S, 'run_child', lambda *a, **kw: None)
    return b


def test_failed_box_prevents_stage_completion(tmp_path, monkeypatch):
    b = _failed_box_fixture(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match='failed.*topic'):
        S.run_boxes(b, _ctx(events=tmp_path / 'events'))


def test_dirty_resume_cannot_publish_under_original_clean_commit(tmp_path):
    b = _build(tmp_path)
    b.rec['mlx2_commit'] = {'commit': 'original', 'dirty': False}
    b.begin('core', {'commit': 'current', 'dirty': True})
    assert X.publish(b, 'core') is None


@pytest.mark.parametrize('mismatch', [False, True])
def test_sweep_resume_preserves_completed_outputs(tmp_path, monkeypatch, mismatch):
    from types import SimpleNamespace

    from research.amoeba.sweep import capture as C
    from research.amoeba.sweep import family as F
    out = tmp_path / 'capture'
    out.mkdir()
    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text(json.dumps({'id': 'a', 'text': 'a' * 60, 'topics': ['sci.physics'], 'split': 'fit'}) + '\n')
    fam = SimpleNamespace(L=1, widths=[4], kinds=['dense'], checkpoint='toy@abc', fingerprint='abc',
                          tok=SimpleNamespace(encode=lambda s: [1] * len(s)))
    monkeypatch.setattr(F, 'load_family', lambda *a, **kw: fam)
    meta = {'complete': True, 'corpus_sha256': C.T.sha_file(corpus), 'checkpoint': fam.checkpoint,
            'fingerprint': 'other' if mismatch else 'abc', 'sequences': 1,
            'args': {'seq_len': 64, 'min_len': 48, 'limit': 0, 'family': 'toy'}}
    (out / 'meta.json').write_text(json.dumps(meta))
    (out / 'items.json').write_text('original items')
    monkeypatch.setattr(C, 'sweep', lambda *a, **kw: pytest.fail('completed run must not need deleted scratch'))
    args = ['--family', 'toy', '--model', 'toy', '--corpus', str(corpus), '--out', str(out),
            '--state-dir', str(tmp_path / 'state'), '--resume']
    if mismatch:
        with pytest.raises(SystemExit, match='different run'):
            C.main(args)
    else:
        C.main(args)
    assert (out / 'items.json').read_text() == 'original items'
    assert json.loads((out / 'meta.json').read_text()) == meta


def test_generation_interrupted_before_manifest_remains_resumable(tmp_path, monkeypatch):
    import sys

    S.paths.research_imports()
    import amoeba_onpolicy_gen as G

    corpus = tmp_path / 'corpus.jsonl'
    corpus.write_text(json.dumps({'text': ' '.join(f'word{i}' for i in range(80)), 'domain': 'text'}) + '\n')
    out = tmp_path / 'gen'
    monkeypatch.setattr(sys, 'argv', ['gen', '--base-url', 'http://stub', '--checkpoint', 'synthetic@fixture',
                                    '--corpus', str(corpus), '--out', str(out), '--samples', '0', '--workers', '1'])
    calls = []
    monkeypatch.setattr(G, 'post', lambda *a, **kw: calls.append(1) or {
        'choices': [{'text': ' continuation'}], 'usage': {'completion_tokens': 1}})
    write_text = Path.write_text
    def interrupted(path, data, *a, **kw):
        if 'mlx2.amoeba-onpolicy-gen.v1' in data:
            raise OSError('simulated interruption at manifest write')
        return write_text(path, data, *a, **kw)
    monkeypatch.setattr(Path, 'write_text', interrupted)
    with pytest.raises(OSError, match='simulated interruption'):
        G.main()
    assert not (out / 'gen.jsonl').exists(), 'completion marker must not precede its manifest'
    monkeypatch.setattr(Path, 'write_text', write_text)
    prior_calls = len(calls)
    G.main()
    assert len(calls) == prior_calls
    man = json.loads((out / 'manifest.json').read_text())
    assert man['rows'] == 1 and man['sha256'] == S.paths.sha256_file(out / 'gen.jsonl')


def test_failed_extraction_prevents_stage_completion(tmp_path, monkeypatch):
    b = _failed_box_fixture(tmp_path, monkeypatch)
    (b.out / 'boxes').rename(b.out / 'box')
    b.rec['extract'] = {'topic': 'sci.physics'}
    core = b.stage_dir('core') / 'carve'
    core.mkdir()
    (core / 'carve.json').write_text(json.dumps({'shape': {'widths': [4]}}))
    monkeypatch.setattr(X, 'generate', lambda *a, **kw: {})
    monkeypatch.setattr(X, 'run_child', lambda *a, **kw: None)
    monkeypatch.setattr(X, '_topic_seed', lambda *a: (None, {'clusters': []}))
    with pytest.raises(SystemExit, match='failed.*topic'):
        X.run_box(b, _ctx(events=tmp_path / 'events'))
