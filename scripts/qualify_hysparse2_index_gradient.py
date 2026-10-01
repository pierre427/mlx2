"""Bounded repeated-index gradient probes on Apple Silicon; no model updates."""
import argparse
import json
from pathlib import Path

from mlx2.experimental.hysparse2.resources import gpu_guard
from mlx2.experimental.hysparse2.train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'mlx2.hysparse2-index-gradient.v1', 'completed': False,
              'training_update_performed': False, 'cases': [], 'repetitions': 8}
    with gpu_guard(wait_seconds=0):
        import mlx.core as mx
        mx.set_default_device(mx.gpu)
        mx.set_memory_limit(2 << 30)
        mx.set_cache_limit(64 << 20)
        mx.random.seed(42)
        def probe(name, fn, parameter, geometry):
            gradient = mx.grad(fn)
            baseline = gradient(parameter)
            mx.eval(baseline)
            maximum, different = 0.0, 0
            for _ in range(7):
                result = gradient(parameter)
                mx.eval(result)
                assert bool(mx.all(mx.isfinite(result)).item())
                error = float(mx.max(mx.abs(result - baseline)).item())
                maximum = max(maximum, error)
                different += error > 0
            report['cases'].append({'name': name, 'geometry': geometry,
                                    'gradient_max_repeat_error': maximum,
                                    'different_repetitions': different})
        for name, rows, width, unique in [('embedding_unique', 32768, 256, True),
                                         ('embedding_repeated', 32768, 256, False),
                                         ('ple_repeated', 16384, 64, False)]:
            ids = mx.arange(288).reshape(2, 144)
            if not unique:
                ids = ids % 16
            upstream = mx.random.normal((2, 144, width))
            weight = mx.zeros((rows, width))
            mx.eval(ids, upstream, weight)
            probe(name, lambda w: mx.sum(w[ids] * upstream), weight,
                  {'rows': rows, 'width': width, 'tokens': 288, 'unique_ids': 288 if unique else 16})
            del ids, upstream, weight
            mx.clear_cache()
        ids = (mx.arange(144) % 16).reshape(1, 144)
        x = mx.random.normal((1, 144, 1, 256))
        upstream = mx.random.normal((1, 144, 1, 64))
        weight = mx.random.normal((32, 256, 64))
        mx.eval(ids, x, upstream, weight)
        probe('moe_repeated_expert_gather_mm',
              lambda w: mx.sum(mx.gather_mm(x, w, rhs_indices=ids) * upstream), weight,
              {'experts': 32, 'input_width': 256, 'output_width': 64, 'tokens': 144, 'selected_unique': 16})
        report['mlx_version'] = mx.__version__
        report['script_sha256'] = file_hash(Path(__file__))
        report['peak_memory_bytes'] = mx.get_peak_memory()
        report['completed'] = True
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
