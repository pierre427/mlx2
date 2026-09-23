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


# ---------------------------------------------------------------------------
# ArraysCache.make_mask: None exactly when every row is valid
# ---------------------------------------------------------------------------


def test_cold_single_row_decode_is_admitted(production_gdn_qwen4):
    """A cold B=1 request's decode must reach the fused kernel's gate.

    ``merge`` of empty caches sets ``left_padding=[0]``; an unpadded prompt
    never finalizes, so ``advance`` drove it to ``[-N]`` and ``make_mask``
    returned an all-True array. ``rollback_spans`` could not explain that
    mask and every decode token was refused as "rollback geometry not
    describable" (~70% of Flash-Next's ordinary B=1 decode layer-calls).
    """
    prompt = [(5 * i + 3) % 60 + 2 for i in range(24)]
    (stats, _tokens) = _decode_receipts(production_gdn_qwen4, [prompt], max_tokens=8)
    reasons = stats["decode_fallback_reasons"]
    assert set(reasons) == {ADMITTED_ON_CPU}, reasons
    assert reasons[ADMITTED_ON_CPU] >= 8


def _joined_and_filtered_warm_row():
    from mlx2.runtime.models.cache import ArraysCache

    warm = ArraysCache(2)
    (warm[0], warm[1]) = (mx.ones((1, 3, 4)), mx.ones((1, 2, 2, 2)))
    cold = ArraysCache.merge([ArraysCache(2)])
    (cold[0], cold[1]) = (mx.ones((1, 3, 4)), mx.ones((1, 2, 2, 2)))
    cold.advance(20)
    warm.extend(cold)
    warm.filter([0])
    return warm


def test_warm_row_that_shared_a_batch_with_a_cold_row_stays_unmasked():
    """``extend`` zero-fills a warm row's ``left_padding``; that is not padding."""
    warm = _joined_and_filtered_warm_row()
    assert warm.left_padding.tolist() == [0]
    mask = warm.make_mask(1)
    assert mask is None
    assert warm.rollback_spans(1, mask) == ()


def test_padded_and_ragged_rows_still_get_their_exact_mask():
    from mlx2.runtime.models.cache import ArraysCache

    padded = ArraysCache(2, left_padding=[2, 0])
    assert padded.make_mask(3).tolist() == [[False, False, True], [True, True, True]]
    padded.advance(3)  # the pads are consumed: [-1, -3]
    assert padded.make_mask(1) is None

    ragged = ArraysCache(2)
    ragged.prepare(lengths=[3, 2])  # a verify block where row 1 drafted less
    assert ragged.make_mask(3).tolist() == [[True, True, True], [True, True, False]]
    assert ragged.rollback_spans(3, ragged.make_mask(3)) == [3, 2]
    full = ArraysCache(2)
    full.prepare(lengths=[3, 3])  # every row verifies the whole block
    assert full.make_mask(3) is None
    assert full.rollback_spans(3, None) == [3, 3]

    # Metadata without a host mirror keeps the explicit (exact) mask rather
    # than reading the device to decide.
    unmirrored = ArraysCache(1)
    unmirrored.left_padding = mx.array([0])
    assert unmirrored.make_mask(2).tolist() == [[True, True]]


def _tiny_qwen38(mtp):
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4,
        full_attention_interval=4, mtp_num_hidden_layers=1 if mtp else 0,
        partial_rotary_factor=0.5, rope_parameters=None,
        max_position_embeddings=512,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def _tiny_qwen4(mtp):
    from test_batched_mtp import _tiny_qwen4_model

    mx.random.seed(41)
    return _tiny_qwen4_model()


ROUTES = {
    "ordinary": (dict(), [24]),
    "equal_rows": (dict(), [24, 24]),
    "padded_rows": (dict(), [24, 15, 9]),
    "self_mtp": ({"self_mtp": {"num_draft": 2, "persistent": True}}, [24, 15]),
}


def _greedy_run(monkeypatch, build, route, legacy):
    """Greedy tokens per row plus the mask kind every GDN update received."""
    from mlx2.runtime.models import qwen3_5
    from mlx2.runtime.models.cache import ArraysCache

    (options, lengths) = ROUTES[route]
    masks = Counter()
    with monkeypatch.context() as patch:
        if legacy:
            # The pre-fix behaviour: always build the explicit mask.
            patch.setattr(
                ArraysCache, "_host_all_valid", lambda self, N: False, raising=False
            )
        for module in (qwen3_5, qwen4_exp):
            stock = module.gated_delta_update

            def recorded(q, k, v, a, b, A_log, dt_bias, state=None, mask=None,
                         _stock=stock, **kwargs):
                if mask is None:
                    masks["none"] += 1
                elif bool(mx.all(mask).item()):
                    masks["all_true"] += 1
                else:
                    masks["padding"] += 1
                return _stock(q, k, v, a, b, A_log, dt_bias, state, mask, **kwargs)

            patch.setattr(module, "gated_delta_update", recorded)
        model = build("self_mtp" in options)
        gen = BatchGenerator(
            model, prefill_step_size=64, completion_batch_size=4,
            prefill_batch_size=4, **options,
        )
        prompts = [
            [(5 * i + 3 + row) % 60 + 2 for i in range(n)]
            for (row, n) in enumerate(lengths)
        ]
        extra = {}
        if "self_mtp" in options:
            extra["self_mtp_configs"] = [{"sampling_temp": 0.0}] * len(prompts)
        tokens = {}
        try:
            uids = gen.insert(prompts, max_tokens=[10] * len(prompts), **extra)
            tokens = {uid: [] for uid in uids}
            finished = 0
            for _ in range(200):
                for response in gen.next()[1]:
                    tokens[response.uid].append(int(response.token))
                    finished += bool(response.finish_reason)
                if finished == len(prompts):
                    break
        finally:
            gen.close()
    return ([tokens[uid] for uid in sorted(tokens)], masks)


@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("family", ["qwen38", "qwen4"])
def test_dropping_all_true_masks_leaves_greedy_output_unchanged(
    monkeypatch, family, route
):
    """None and an all-True mask must mean the same thing to every GDN path.

    The legacy arm always builds the explicit mask; the new arm drops it when
    every row is valid. Output must match token for token, and the new arm
    must really have run unmasked updates where the legacy arm masked.
    """
    build = _tiny_qwen38 if family == "qwen38" else _tiny_qwen4
    (legacy, legacy_masks) = _greedy_run(monkeypatch, build, route, legacy=True)
    (fixed, masks) = _greedy_run(monkeypatch, build, route, legacy=False)
    assert fixed == legacy
    assert all(len(row) == 10 for row in fixed)
    assert masks["all_true"] == 0, masks
    # Genuine padding is still masked, exactly as often as before.
    assert masks["padding"] == legacy_masks["padding"]
    if route == "padded_rows":
        # A padded prefill finalizes, which already dropped the metadata.
        assert masks["padding"] > 0
    else:
        assert legacy_masks["all_true"] > 0, legacy_masks
