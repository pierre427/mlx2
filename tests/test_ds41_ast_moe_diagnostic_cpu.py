"""Synthetic diagnostic plumbing only; all GPU/math factories are replaced."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np


def test_diagnostic_returns_original_output_and_restores_hooks(monkeypatch):
    mx.set_default_device(mx.cpu)
    monkeypatch.setattr(mx.fast, "metal_kernel", lambda *a, **k: (_ for _ in ()).throw(AssertionError("GPU forbidden")))
    from mlx2.runtime.gguf import source_gate_up_prefill as pp, source_middle_fusion as mf
    from mlx2.runtime.gguf import source_gate_up_fusion as gu
    from mlx2.runtime.models import deepseek_v41_quant as quant
    monkeypatch.setattr(mf, "exp_spelling", lambda: "metal::exp")
    monkeypatch.setattr(quant, "fused_quantize", lambda x, kind: x)
    monkeypatch.setattr(mx, "quantized_matmul", lambda x, *a, **k: mx.zeros((x.shape[0], 5120)))
    def packed(x, experts, rows, scores, counts, **kwargs):
        middle = mx.zeros((sum(counts), 2304))
        return (middle, middle, middle) if kwargs.get("return_projections") else middle
    monkeypatch.setattr(pp, "pp_middle", packed)
    monkeypatch.setattr(gu, "middle", lambda *a, **k: (mx.zeros((1, 2304)),)*3)
    class Expert(dict):
        layout = SimpleNamespace(kind="original")
    experts = [Expert({"down.qweight": mx.zeros((1,), mx.uint32), "down.scales": mx.zeros((1,), mx.uint8)}) for _ in range(6)]
    stream = SimpleNamespace(_active=True, _leased={(14, i): e for i, e in enumerate(experts)})
    returned = []
    def original(x, ids, scores, prefix, **kwargs):
        if x.shape[0] == 16:
            pp.pp_middle(x, experts, mx.array(np.tile(np.arange(16), 6), mx.uint32),
                         mx.ones((96,)), (16,)*6)
        result = mx.zeros((x.shape[0], 5120))
        returned.append(result)
        return result
    weights = SimpleNamespace(moe=original, expert_stream=stream, source_prefill_batch_policy="gate_up")
    path = Path(__file__).resolve().parents[1]/"scripts/research/ds41_ast_moe_diagnostic.py"
    spec = importlib.util.spec_from_file_location("ast_moe_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.instrument(SimpleNamespace(weights=weights), lambda: True) as report:
        result = weights.moe(mx.zeros((16, 5120)), mx.array(np.tile(np.arange(6), (16, 1)), mx.uint32),
                             mx.ones((16, 6)), "blk.14")
        assert result is returned[0]
        assert len(report["experts"]) == 6
        assert report["same_input_moe"]["raw_bits_equal"]
        assert report["extra_moe_oracle_calls"] == 1
        assert all(e["local_M"] == 16 and e["local_row"] == 12 for e in report["experts"])
    assert weights.moe is original and pp.pp_middle is packed
