"""Bounded AdamW settings binding and exact next-update continuation on M3."""
import argparse
import json
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.checkpoint / 'state.json').read_text())
    report = {'schema': 'mlx2.hysparse2-optimizer-settings.v1', 'completed': False,
              'checkpoint_sha256': file_hash(args.checkpoint / 'model.safetensors'),
              'training_performed': False, 'constant_gradient_update_fixture': True,
              'serving_route_qualified': False, 'mismatch_rejections': []}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        from mlx import optimizers
        from mlx.utils import tree_map, tree_flatten
        from mlx2.experimental.hysparse2 import train as training
        from mlx2.experimental.hysparse2.config import Config
        from mlx2.experimental.hysparse2.model import Model

        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(10 << 30)
        mx.set_cache_limit(256 << 20)
        model = Model(Config(**metadata['config']))
        training._load_model_state(args.checkpoint, model)
        settings = dict(betas=[0.8, 0.95], eps=1e-5, bias_correction=True, weight_decay=0.2)
        optimizer = optimizers.AdamW(1e-4, **settings)
        gradient = tree_map(lambda p: mx.full(p.shape, 0.01, dtype=p.dtype), model.trainable_parameters())
        optimizer.update(model, gradient)
        mx.eval(model.parameters(), optimizer.state)
        saved = training.save_checkpoint(args.output / 'checkpoints', model, optimizer, 1, {'fixture': 'constant gradient optimizer continuation'})
        report['saved_settings'] = json.loads((saved / 'state.json').read_text())['optimizer_settings']
        restored = Model(model.config)
        owner = restored._cache_owner
        before = dict(tree_flatten(restored.parameters()))
        for name, value in [('betas', [0.9, 0.99]), ('eps', 1e-4), ('bias_correction', False), ('weight_decay', 0.1)]:
            changed = dict(settings, **{name: value})
            wrong = optimizers.AdamW(1e-4, **changed)
            state = wrong.state
            try:
                training.load_checkpoint(saved, restored, wrong, {'fixture': 'constant gradient optimizer continuation'})
                raise AssertionError('mismatched optimizer settings accepted')
            except ValueError as exc:
                assert 'optimizer settings' in str(exc)
            assert restored._cache_owner is owner and wrong.state is state
            assert all(dict(tree_flatten(restored.parameters()))[k] is v for k, v in before.items())
            report['mismatch_rejections'].append(name)
        matching = optimizers.AdamW(1e-4, **settings)
        assert training.load_checkpoint(saved, restored, matching, {'fixture': 'constant gradient optimizer continuation'}) == 1
        optimizer.update(model, gradient)
        matching.update(restored, gradient)
        mx.eval(model.parameters(), restored.parameters(), optimizer.state, matching.state)
        def compare(a, b):
            left, right = dict(tree_flatten(a)), dict(tree_flatten(b))
            assert left.keys() == right.keys()
            error = max(float(mx.max(mx.abs(value - right[k])).item()) for k, value in left.items())
            assert error == 0
            return {'tensors': len(left), 'max_error': error}
        report['next_update_parameters'] = compare(model.parameters(), restored.parameters())
        report['next_update_optimizer'] = compare(optimizer.state, matching.state)
        report['source_hashes'] = {str(path): file_hash(path) for path in (Path(__file__), Path(training.__file__))}
        report['peak_memory_bytes'] = mx.get_peak_memory()
        report['completed'] = True
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
