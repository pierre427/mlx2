"""CLI warm-start interrupted versus uninterrupted training on all 49 layers."""
import argparse
import hashlib
import json
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-existing', action='store_true', help='Inspect existing run artifacts without repeating training')
    args = parser.parse_args()
    if not args.compare_existing:
        args.output.mkdir(parents=True, exist_ok=False)
    elif not args.output.is_dir():
        parser.error("existing run directory required")
    metadata = json.loads((args.checkpoint / 'state.json').read_text())
    config = args.output / 'config.json'
    config.write_text(json.dumps(metadata['config'], indent=2) + '\n')
    # This mechanical fixture is not a certified tokenizer/training corpus.
    tokenizer_label = hashlib.sha256(b'mechanical-hysparse2-training-resume-fixture').hexdigest()
    from mlx2.experimental.hysparse2 import train as training
    common = ['--config', str(config), '--tokens', str(args.tokens),
              '--tokenizer-sha256', tokenizer_label, '--device', 'gpu',
              '--memory-limit-gib', '10', '--sequence', '144', '--batch', '1',
              '--accumulation', '2', '--save-every', '100', '--seed', '42']
    if not args.compare_existing:
        training.main(common + ['--steps', '2', '--initialize-from', str(args.checkpoint), '--output', str(args.output / 'continuous')])
        training.main(common + ['--steps', '1', '--initialize-from', str(args.checkpoint), '--output', str(args.output / 'first')])
        first = args.output / 'first' / 'step-00000001'
        training.main(common + ['--steps', '1', '--resume', str(first), '--output', str(args.output / 'resumed')])
    continuous = args.output / 'continuous' / 'step-00000002'
    resumed = args.output / 'resumed' / 'step-00000002'
    report = {'schema': 'mlx2.hysparse2-training-resume.v1', 'completed': False,
              'training_performed': True, 'training_run_performed_this_invocation': not args.compare_existing,
              'objective_steps_per_lineage': 2,
              'batch': 1, 'sequence': 144, 'accumulation': 2,
              'tokenizer_label_is_mechanical_fixture': True,
              'learned_quality_evaluated': False, 'serving_route_qualified': False,
              'checkpoint_sha256': file_hash(args.checkpoint / 'model.safetensors'),
              'tokens_sha256': file_hash(args.tokens), 'comparisons': {}}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(10 << 30)
        for name in ('model.safetensors', 'optimizer.safetensors'):
            a, b = mx.load(str(continuous / name)), mx.load(str(resumed / name))
            mx.eval(a, b)
            assert a.keys() == b.keys()
            error = max(float(mx.max(mx.abs(value - b[k])).item()) for k, value in a.items())
            report['comparisons'][name] = {'tensors': len(a), 'max_error': error, 'bitwise_equal': error == 0}
            del a, b
            mx.clear_cache()
        x, y = (json.loads((path / 'state.json').read_text()) for path in (continuous, resumed))
        assert x['run'] == y['run'] and x['step'] == y['step'] == 2
        report['run_equal'] = True
        report['ple_binding_equal'] = x['permanent_sidecar'] == y['permanent_sidecar']
        report['bitwise_training_trajectory_equal'] = all(v['bitwise_equal'] for v in report['comparisons'].values())
        report['training_trajectory_qualified'] = report['bitwise_training_trajectory_equal']
        report['source_hashes'] = {str(path): file_hash(path) for path in (Path(__file__), Path(training.__file__))}
        report['peak_memory_bytes_last_run'] = mx.get_peak_memory()
        report['completed'] = True
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
