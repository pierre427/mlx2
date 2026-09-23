"""Fused GDN admission receipts and the mask geometry GDN layers receive.

The fused Qwen4/Flash-Next GDN decode kernel and the packed Dk=128 GDN kernel
both require a mask-free slab. These tests run on the CPU, where an admitted
fused decode call records "Metal runtime unavailable" instead of launching:
that reason means the GPU would have run the fused kernel, while a geometry
reason means it would not.
"""

from collections import Counter

import mlx.core as mx
import pytest

from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models import qwen4_exp
from mlx2.runtime.models.qwen4_fused_gdn import admit_rollback_span

ADMITTED_ON_CPU = "Metal runtime unavailable"


@pytest.fixture
def production_gdn_qwen4(monkeypatch):
    """Tiny Qwen4 whose GDN layer has the production fused-kernel geometry.

    Every structural admission check passes, so the only refusals left are
    the mask/span ones under test.
    """
    text_config = dict(
        model_type="qwen4_exp_text", hidden_size=64, intermediate_size=0,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=64,
        linear_num_value_heads=48, linear_num_key_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, hc_count=2, hc_lowrank=8,
        ple_layer_ids=[1], ple_embed_dim=32, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=128,
        make_ngram_vocab_size_divisible_by=128, split_ngram_parts=1,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=8, indexer_compress_ratio=2, mtp_num_hidden_layers=0,
        rope_parameters={
            "type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.25,
        },
    )
    mx.random.seed(7)
    model = qwen4_exp.Model(
        qwen4_exp.ModelArgs(model_type="qwen4_exp", text_config=text_config)
    )
    model.set_dtype(mx.bfloat16)
    # The CPU gather matmul is float32-only; the MoE blocks are orthogonal to
    # the GDN admission, so run them in float32.
    moe = qwen4_exp.SparseMoeBlock
    for _, module in model.named_modules():
        if isinstance(module, moe):
            module.set_dtype(mx.float32)
    stock_call = moe.__call__
    monkeypatch.setattr(
        moe,
        "__call__",
        lambda self, x, *a, **k: stock_call(self, x.astype(mx.float32), *a, **k).astype(
            x.dtype
        ),
    )
    model.eval()
    mx.eval(model.parameters())
    for _, module in model.named_modules():
        if isinstance(module, qwen4_exp.GatedDeltaNet):
            module.set_fused_gdn_decode_mode("fused")
    return model


def _decode_receipts(model, prompts, max_tokens, prefill_step_size=64):
    gen = BatchGenerator(
        model, prefill_step_size=prefill_step_size, completion_batch_size=4,
        prefill_batch_size=4,
    )
    tokens = Counter()
    try:
        gen.insert(prompts, max_tokens=[max_tokens] * len(prompts))
        qwen4_exp.qwen4_fused_gdn_stats(model, reset=True)
        finished = 0
        for _ in range(200):
            for response in gen.next()[1]:
                tokens[response.uid] += 1
                finished += bool(response.finish_reason)
            if finished == len(prompts):
                break
    finally:
        gen.close()
    return (qwen4_exp.qwen4_fused_gdn_stats(model), tokens)


def test_an_unpadded_multi_row_slab_is_named_as_a_batch():
    full = mx.ones((2, 3), mx.bool_)
    refusal = admit_rollback_span([3, 3], full, 3, masked_reason="masked verify")
    assert refusal.reason == "batch of 2 rows"
    padded = admit_rollback_span([2], mx.ones((1, 3), mx.bool_), 3, masked_reason="x")
    assert padded.reason == "padded rollback geometry"
    assert admit_rollback_span([3], None, 3, masked_reason="x") is None


def test_prefill_chunks_are_not_booked_as_decode_fallbacks(
    production_gdn_qwen4, monkeypatch
):
    """Only single-token decode forwards are decode-admission candidates.

    A three-chunk prefill used to add one "uninitialized cache" and two span
    declines to the decode histogram, inflating the very refusal count the
    receipt exists to show.
    """
    widths = Counter()
    stock_call = qwen4_exp.GatedDeltaNet.__call__

    def counted(self, inputs, *args, **kwargs):
        widths[int(inputs.shape[1])] += 1
        return stock_call(self, inputs, *args, **kwargs)

    monkeypatch.setattr(qwen4_exp.GatedDeltaNet, "__call__", counted)
    prompt = [(5 * i + 3) % 60 + 2 for i in range(24)]
    (stats, _tokens) = _decode_receipts(
        production_gdn_qwen4, [prompt], max_tokens=6, prefill_step_size=8
    )
    reasons = stats["decode_fallback_reasons"]
    assert "uninitialized cache" not in reasons, reasons
    assert len(widths) > 1  # the prefill really was chunked
    assert stats["fallbacks"] + stats["fused_calls"] == widths[1], (reasons, widths)
