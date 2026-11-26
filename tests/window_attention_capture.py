"""Record every vector-SDPA Metal kernel construction and launch, on CPU.

``mx.fast.metal_kernel`` is replaced by a recorder, so nothing compiles or
runs on a GPU: the record is the kernel name, its input/output names, header,
source and flags, and for each launch the template, grid, threadgroup, output
shapes/dtypes and the shapes/dtypes (and small integer/float values) of the
inputs.  ``tests/fixtures/window_attention_legacy_kernels.json`` is this
record taken from the pre-consolidation modules (mlx2 d46e7a3ea);
``tests/test_technologies_window_attention_sources.py`` compares the current
modules' record with it, string for string.

Run as a script to regenerate the fixture (only against the legacy tree):
``PYTHONPATH=src python tests/window_attention_capture.py OUT.json``.
"""

from __future__ import annotations

import json
import sys

import mlx.core as mx


def _dtype(d) -> str:
    return str(d)


class _Recorder:
    def __init__(self):
        self.kernels: list[dict] = []
        self.launches: list[dict] = []

    def metal_kernel(self, **kwargs):
        record = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in kwargs.items()}
        self.kernels.append(record)
        name = kwargs["name"]

        def run(*, inputs, template, grid, threadgroup, output_shapes, output_dtypes, **extra):
            ins = []
            for a in inputs:
                entry = {"shape": list(a.shape), "dtype": _dtype(a.dtype)}
                if a.size <= 2 and a.dtype in (mx.int32, mx.float32):
                    entry["value"] = a.tolist()
                ins.append(entry)
            self.launches.append({
                "kernel": name,
                "inputs": ins,
                "template": [[k, _dtype(v) if isinstance(v, mx.Dtype) else v] for k, v in template],
                "grid": list(grid),
                "threadgroup": list(threadgroup),
                "output_shapes": [list(s) for s in output_shapes],
                "output_dtypes": [_dtype(d) for d in output_dtypes],
                "extra": sorted(extra),
            })
            return [mx.zeros(tuple(s), dtype=d) for s, d in zip(output_shapes, output_dtypes)]

        return run


def capture(monkeypatch) -> dict:
    """The record of the current modules (``monkeypatch``: pytest's)."""
    from mlx2.runtime.models import muse_attn_window as MW
    from mlx2.runtime.models import qwen4_attn_rows as AR
    from mlx2.runtime.models import qwen4_attn_window as AW
    from mlx2.runtime.models import sdpa_vector_d512 as D5

    rec = _Recorder()
    monkeypatch.setattr(mx.fast, "metal_kernel", rec.metal_kernel)
    for module in (MW, AR, AW):
        monkeypatch.setattr(module, "_KERNELS", {})
    D5._pass1.cache_clear()
    D5._pass2.cache_clear()

    # Every kernel name each module can build.
    for name in ("one_chains", "one_chains_mask", "two_chains", "two_chains_mask",
                 "one_combine", "two_combine"):
        MW._kernel(name)
    for name in ("one_chains", "one_chains_mask", "two_chains", "two_chains_mask",
                 "one_combine", "one_combine_gate", "two_combine", "two_combine_gate"):
        AR._kernel(name)
    for name in ("one", "one_mask", "two", "two_mask"):
        AW._kernel(name)
    D5._pass1()
    D5._pass2()
    kernels = list(rec.kernels)
    rec.kernels.clear()

    launches: dict[str, list] = {}

    def record(label, fn):
        rec.launches.clear()
        out = fn()
        mx.eval(out)
        launches[label] = {"out_shape": list(out.shape), "launches": list(rec.launches)}

    bf, f16 = mx.bfloat16, mx.float16
    for devc in ("s", "d"):
        monkeypatch.setattr(MW, "_device_class", lambda devc=devc: devc)
        monkeypatch.setattr(AR, "_device_class", lambda devc=devc: devc)
        # Muse D128 windows: one-pass, a plan switch at 1024 keys, two-pass.
        for n_first, rows, heads, kvh, dtype in (
            (60, 3, 32, 2, bf), (1022, 4, 32, 2, bf), (9000, 2, 16, 4, f16),
        ):
            q = mx.zeros((1, heads, rows, 128), dtype=dtype)
            kv = mx.zeros((1, kvh, n_first + rows - 1, 128), dtype=dtype)
            total = sum(n_first + r for r in range(rows))
            for masked in (False, True):
                masks = mx.ones((total,), dtype=mx.bool_) if masked else None
                record(f"muse/{devc}/{n_first}/{rows}/{heads}/{kvh}/{dtype}/{masked}",
                       lambda: MW.window_sdpa(q, kv, kv, 0.125, n_first=n_first, masks=masks))
        # Qwen4 one-row / few-row SDPA (+gate, +mask).
        for n, rows, heads, kvh, dtype in (
            (100, 1, 24, 2, bf), (100, 2, 24, 2, f16), (5000, 1, 24, 2, bf),
            (20000, 2, 16, 2, bf),
        ):
            q = mx.zeros((1, heads, rows, 256), dtype=dtype)
            kv = mx.zeros((1, kvh, n, 256), dtype=dtype)
            for masked in (False, True):
                for gated in (False, True):
                    mask = mx.ones((1, 1, rows, n), dtype=mx.bool_) if masked else None
                    gate = mx.ones((1, rows, heads * 256), dtype=dtype) if gated else None
                    record(f"rows/{devc}/{n}/{rows}/{heads}/{kvh}/{dtype}/{masked}/{gated}",
                           lambda: AR.sdpa_gate(q, kv, kv, 0.0625, mask=mask, gate=gate))
        # Qwen4 row-exact windows.
        for n_first, rows, heads, kvh, dtype in (
            (60, 3, 24, 2, bf), (1020, 8, 24, 2, bf), (8190, 5, 24, 2, f16),
        ):
            q = mx.zeros((1, heads, rows, 256), dtype=dtype)
            kv = mx.zeros((1, kvh, n_first + rows - 1, 256), dtype=dtype)
            gate = mx.ones((1, rows, heads, 256), dtype=dtype)
            total = sum(n_first + r for r in range(rows))
            for masked in (False, True):
                masks = mx.ones((total,), dtype=mx.bool_) if masked else None
                record(f"window/{devc}/{n_first}/{rows}/{heads}/{kvh}/{dtype}/{masked}",
                       lambda: AW.window_sdpa_gate(q, kv, kv, 0.0625, n_first=n_first,
                                                   gate=gate, masks=masks))
        # D512 TK-2 decode.
        monkeypatch.setattr(D5, "device_class", lambda devc=devc: devc)
        for n, b, heads, kvh, dtype in ((4096, 1, 32, 4, bf), (70000, 2, 16, 2, f16)):
            q = mx.zeros((b, heads, 1, 512), dtype=dtype)
            kv = mx.zeros((b, kvh, n, 512), dtype=dtype)
            record(f"d512/{devc}/{n}/{b}/{heads}/{kvh}/{dtype}",
                   lambda: D5.attention(q, kv, kv, scale=0.04))
    return {"kernels": kernels, "launches": launches}


if __name__ == "__main__":  # pragma: no cover - fixture regeneration
    import pytest

    mx.set_default_device(mx.cpu)
    mp = pytest.MonkeyPatch()
    try:
        data = capture(mp)
    finally:
        mp.undo()
    with open(sys.argv[1], "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
