"""Isolate unchanged-weight gradient repeatability from checkpoint loading."""
import argparse
import json
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((args.checkpoint / 'state.json').read_text())
    report = {'schema': 'mlx2.hysparse2-gradient-repeat.v1', 'completed': False,
              'checkpoint_sha256': file_hash(args.checkpoint / 'model.safetensors'),
              'tokens_sha256': file_hash(args.tokens), 'training_update_performed': False,
              'serving_route_qualified': False, 'comparisons': {}}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        import numpy as np
        from mlx import nn
        from mlx.utils import tree_flatten
        from mlx2.experimental.hysparse2 import train as training
        from mlx2.experimental.hysparse2 import model as model_module
        from mlx2.experimental.hysparse2.config import Config
        model = model_module.Model(Config(**metadata['config']))
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(10 << 30)
        mx.set_cache_limit(256 << 20)
        training._load_model_state(args.checkpoint, model)
        model.train()
        values = np.load(args.tokens, allow_pickle=False)
        run = metadata['run']
        rng = np.random.default_rng(run['seed'] + metadata['step'] * run['accumulation'])
        starts = rng.integers(0, len(values) - run['sequence'] - 1, size=run['batch'])
        tokens = mx.array(np.stack([values[s:s + run['sequence'] + 2] for s in starts]))
        mx.eval(tokens)
        def compute(m):
            fn = nn.value_and_grad(m, lambda m, b: training.loss(m, b, run['mtp_weight'], run['router_weight'], run['diffusion_weight']))
            value, gradient = fn(m, tokens)
            mx.eval(value, gradient)
            return float(value.item()), dict(tree_flatten(gradient))
        def compare(a, b):
            assert a[1].keys() == b[1].keys()
            errors = [(float(mx.max(mx.abs(v - b[1][k])).item()), k) for k, v in a[1].items()]
            assert all(bool(mx.all(mx.isfinite(v)).item()) for v in b[1].values())
            maximum, worst = max(errors)
            return {'loss_error': abs(a[0] - b[0]), 'gradient_tensors': len(errors),
                    'gradient_max_error': maximum, 'worst_tensor': worst,
                    'different_gradient_tensors': sum(e > 0 for e, _ in errors)}
        model.checkpoint_layers = True
        a, b = compute(model), compute(model)
        report['comparisons']['same_weights_checkpointed_repeat'] = compare(a, b)
        del b
        clone = model_module.Model(model.config)
        training._load_model_state(args.checkpoint, clone)
        clone.train()
        clone.checkpoint_layers = True
        original, restored = dict(tree_flatten(model.parameters())), dict(tree_flatten(clone.parameters()))
        report['restored_parameters_max_error'] = max(float(mx.max(mx.abs(v - restored[k])).item()) for k, v in original.items())
        assert report['restored_parameters_max_error'] == 0
        b = compute(clone)
        report['comparisons']['same_weights_reloaded_model'] = compare(a, b)
        del a, b, clone, original, restored
        mx.clear_cache()
        model.checkpoint_layers = False
        a, b = compute(model), compute(model)
        report['comparisons']['same_weights_uncheckpointed_repeat'] = compare(a, b)
        report['source_hashes'] = {str(path): file_hash(path) for path in (Path(__file__), Path(training.__file__), Path(model_module.__file__))}
        report['peak_memory_bytes'] = mx.get_peak_memory()
        report['completed'] = True
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
