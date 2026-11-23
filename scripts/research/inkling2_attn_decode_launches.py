"""Kernels per one-token Inkling 2 ``Attention`` call, option off vs on.

Original mlx2 script (2026-11-22). Builds the tiny CPU model
(``tests/test_inkling2_cpu.py::_tiny_model``, BF16 non-expert weights),
configures the shipped non-MoE kernel defaults (``KERNEL_DEFAULTS``),
prefills 64 tokens on the CPU, then *builds* (never evaluates) one decode
visit's graph on the GPU stream for a sliding visit (0) and a global visit
(5) and counts non-view primitives with ``inkling2_launch_count.count_graph``
(the attention output plus the cache arrays it writes). On the GPU stream
``mx.fast`` ops and Metal kernels are single primitives, so the count is the
launch count. Nothing runs on the GPU. The tiny model wraps its window-8 ring
and has log floor 4, so the ring copy and the log-position scale are in.

Usage: PYTHONPATH=src python scripts/research/inkling2_attn_decode_launches.py
       [name=json ...]   (extra configure_kernels switches)

Result (2026-11-22, mlx 0.32.4): sliding 38 -> 22, global 55 -> 20 with
``attn_decode_kernel=true``. The composed chain's L2 clips are already
``mx.compile``d and still cost three launches each (Square, Sum, the fused
tail): compile does not merge reductions, which is why the front end is a
Metal kernel and not a compiled graph.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from inkling2_launch_count import VIEWS, cache_arrays, count_graph
    from mlx2.adapters.inkling2 import KERNEL_DEFAULTS
    from mlx2.runtime.models.cache import CacheList
    from test_inkling2_cpu import _tiny_model

    mx.set_default_device(mx.cpu)
    options = {k: (True if v is None else v) for k, v in KERNEL_DEFAULTS.items()
               if not k.startswith("moe_")}
    for item in sys.argv[1:]:
        key, value = item.split("=", 1)
        options[key] = json.loads(value)
    model = _tiny_model()
    model.load_weights([(k, v if ".switch_mlp." in k else v.astype(mx.bfloat16))
                        for k, v in tree_flatten(model.parameters())])
    mx.eval(model.parameters())
    model.configure_kernels(**options)
    mx.random.seed(1)
    tokens = mx.random.randint(0, 500, (1, 64))
    cache = model.make_cache()
    mx.eval(model(tokens, cache=cache), cache_arrays(cache))
    x = mx.random.normal((1, 1, model.config.hidden_size)).astype(mx.bfloat16)
    mx.eval(x)
    out = {"options": options}
    for visit in (0, 5):
        attn = model.model.layers[visit].self_attn
        planes = CacheList(cache[2 * visit], cache[2 * visit + 1])
        mx.set_default_device(mx.gpu)
        try:
            with mx.stream(mx.gpu):
                y = attn(x, cache=planes)
        finally:
            mx.set_default_device(mx.cpu)
        result = count_graph(y, *cache_arrays(planes.caches))
        out["sliding" if attn.is_sliding else "global"] = {
            "kernels": result["kernels"],
            "by_primitive": {k: v for k, v in result["by_primitive"].items() if k not in VIEWS},
            "attn_decode_counts": dict(attn.__dict__.get("attn_decode_counts", {}))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
