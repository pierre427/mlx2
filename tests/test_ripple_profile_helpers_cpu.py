"""CPU check of the profiler's module walk: every projection gets a valid input."""
import importlib.util
from pathlib import Path

import mlx.core as mx
from mlx import nn

mx.set_default_device(mx.cpu)

from test_ripple_memcopy_cpu import build

SPEC = importlib.util.spec_from_file_location(
    "ripple_profile", Path(__file__).parents[1] / "scripts/research/ripple_profile.py")
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def test_every_linear_module_accepts_its_inferred_width():
    for nvfp4 in (False, True):
        model = build(mx.bfloat16, nvfp4)
        if not nvfp4:
            groups = iter([(32, 4), (32, 5), (32, 6), (32, 8)] * 20)
            def pick(_path, module):
                return isinstance(module, nn.Linear) and module.weight.shape[-1] % 32 == 0 and dict(
                    zip(("group_size", "bits"), next(groups)))
            nn.quantize(model, class_predicate=pick)
        modules = profile.linear_modules(model)
        assert len(modules) == 5 * 8 + 1
        for _, module in modules:
            out = module(mx.zeros((1, 1, profile.input_width(module)), dtype=mx.bfloat16))
            mx.eval(out)
