"""Small CPU QKV gradient parity for an isolated materialization candidate."""

import argparse
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('fresh output required')
    import mlx.core as mx
    from mlx2.experimental.hysparse2 import attention as reference
    from mlx2.experimental.hysparse2.train import file_hash

    mx.set_default_device(mx.cpu)
    mx.random.seed(91)
    spec = importlib.util.spec_from_file_location('isolated_attention_candidate', args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    q = mx.random.normal((1, 2, 2, 8))
    k = mx.random.normal((1, 1, 128, 8))
    v = mx.random.normal((1, 1, 128, 8))

    def loss(function, q, k, v):
        output, _ = function(q, [(k, v, 0)], offset=126, query_tile=1, key_tile=4)
        return mx.sum(output ** 2)

    candidate._MATERIALIZATION_STATS['evaluations'] = 0
    a = mx.grad(lambda q, k, v: loss(reference.attention, q, k, v), argnums=(0, 1, 2))(q, k, v)
    b = mx.grad(lambda q, k, v: loss(candidate.attention, q, k, v), argnums=(0, 1, 2))(q, k, v)
    mx.eval(a, b)
    errors = [float(mx.max(mx.abs(x - y)).item()) for x, y in zip(a, b, strict=True)]
    assert max(errors) == 0 and candidate._MATERIALIZATION_STATS['evaluations'] == 2
    report = {'schema': 'mlx2.hysparse2-attention-materialize-gradient.cpu.v1',
              'completed': True, 'device': 'cpu', 'dtype': 'float32',
              'shape': {'q': [1, 2, 2, 8], 'kv': [1, 1, 128, 8]},
              'offset': 126, 'query_tile': 1, 'key_tile': 4,
              'completed_materializations': candidate._MATERIALIZATION_STATS['evaluations'],
              'qkv_gradient_max_errors': errors,
              'candidate_sha256': file_hash(args.candidate),
              'reference_sha256': file_hash(Path(reference.__file__)),
              'probe_sha256': file_hash(Path(__file__)),
              'full_model_training_qualified': False, 'gpu_qualified': False,
              'serving_route_qualified': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
