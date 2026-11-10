# SPDX-License-Identifier: MIT
"""Source shared-expert middle as one fused epilogue (opt-in, unqualified).

Ordinary shared expert (``DeepSeekV41Block.shared_expert``) after the MXFP8
gate/up products: ``round(gate)``, ``round(up)``, clamped SwiGLU,
``round(.)``, then the down projection re-runs the FP8 block-32 QAT of that
middle before its MXFP8 product -- about eight elementwise launches per layer.

``fused_middle_v1`` feeds the raw F32 gate/up products to the routed
``source_middle_fusion`` kernel with a single score of exactly 1.0, and gives
its output straight to the down MXFP8 product:

* the kernel's BF16 rounds, ``min(g,10)``/``clamp(u,+-10)``, served sigmoid
  spelling and ``silu*u`` order are the routed law that already replaces the
  same ordinary ops (``limited_swiglu`` with BF16 rounds) bit-exactly;
* ``activation*1.0f`` is the identity for every F32 value;
* its FP8 block-32 epilogue is text-identical to ``fused_quantize(.,'fp8')``
  (``FUSED_HEADER`` with ``fp contract(off)``), which is what the ordinary
  down projection applies under ``deepseek_source_dense_qat=fused_fp32``.

Bit equality with the ordinary path is therefore expected by construction;
it is a GPU-checked claim, not a CPU result. Only the shared expert changes;
the gate/up products, the down product and its BF16 store are the ordinary
ones. Provenance: provenance/deepseek-v41-source-shared-middle.json.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

POLICY = 'fused_middle_v1'
POLICIES = ('reference', POLICY)


def validate_policy(selected, *, dense_qat):
    if selected not in POLICIES:
        raise ValueError('invalid Source shared middle policy')
    if selected != 'reference' and dense_qat != 'fused_fp32':
        # The fused epilogue reproduces fused_quantize, not the reference QAT graph.
        raise ValueError('Source shared fused middle requires deepseek_source_dense_qat=fused_fp32')


def supports(weights, x, gate_name, up_name, down_name):
    """Structural gate; False keeps the ordinary shared expert unchanged."""
    import mlx.core as mx
    if (getattr(weights, 'source_shared_middle_policy', 'reference') != POLICY
            or getattr(weights, 'source_dense_qat_policy', 'reference') != 'fused_fp32'):
        return False
    quant = getattr(weights, '_quant', None)
    if not isinstance(quant, dict) or any(name not in quant for name in (gate_name, up_name, down_name)):
        return False
    # The kernel's rounds are BF16; the ordinary m.round follows activation_dtype.
    args = (getattr(weights, 'artifact', None) or {}).get('args')
    if getattr(args, 'activation_dtype', None) != 'bfloat16':
        return False
    # Keep the down product's M equal to the ordinary one: 2-D rows or [1,T,K].
    if (not isinstance(x, mx.array) or x.dtype != mx.float32 or not x.size
            or not (x.ndim == 2 or (x.ndim == 3 and x.shape[0] == 1))):
        return False
    return True


def shared_expert(weights, x, gate_name, up_name, down_name, *, ffn_input_token=None, middle=None):
    """Raw F32 down product; the caller applies its ordinary BF16 store.

    Returns ``None`` before any dispatch when the products are not the
    expected F32 [.., I] pair (I a multiple of 32), so the caller keeps its
    ordinary path.
    """
    import mlx.core as mx
    if not supports(weights, x, gate_name, up_name, down_name):
        return None
    if middle is None and (mx.default_device() != mx.gpu or not mx.metal.is_available()):
        # The routed epilogue needs an owned Metal window; keep the ordinary path.
        counters = weights.scheduled_dense_calls
        counters['source_shared_middle_fallback'] = counters.get('source_shared_middle_fallback', 0)+1
        return None
    kwargs = {'ffn_input_token': ffn_input_token} if ffn_input_token is not None else {}
    gate, up = weights.gate_up(x, gate_name, up_name, **kwargs)
    width = gate.shape[-1]
    if (gate.dtype != mx.float32 or up.dtype != mx.float32 or up.shape != gate.shape
            or width % 32 or gate.shape[:-1] != x.shape[:-1]):
        counters = weights.scheduled_dense_calls
        counters['source_shared_middle_fallback'] = counters.get('source_shared_middle_fallback', 0)+1
        return None
    if middle is None:
        from ..runtime.gguf.source_middle_fusion import middle
    rows = gate.size // width
    prepared = middle(gate.reshape(rows, width), up.reshape(rows, width),
                      mx.array([1.0], mx.float32), mx.zeros((rows,), mx.uint32))
    q, scales = weights._quant[down_name]
    out = mx.quantized_matmul(prepared, q, scales=scales, mode='mxfp8')
    counters = weights.scheduled_dense_calls
    for key in ('source_shared_middle_fused', 'official_mxfp8_act32'):
        counters[key] = counters.get(key, 0)+1
    # The down QAT that the ordinary linear() would run is folded into the epilogue.
    counters['source_dense_qat_calls_avoided'] = counters.get('source_dense_qat_calls_avoided', 0)+1
    return out.reshape(*x.shape[:-1], out.shape[-1])


@lru_cache(maxsize=1)
def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    paths = ('adapters/deepseek_v41_source_shared_middle.py',
             'runtime/gguf/source_middle_fusion.py',
             'runtime/models/deepseek_v41_quant.py')
    records = '\n'.join(p+':'+hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths)
    return hashlib.sha256((POLICY+'\n'+records).encode()).hexdigest()
