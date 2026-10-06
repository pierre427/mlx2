from pathlib import Path

import pytest

from scripts.bench_tokenizer_viability import HFMarkedTokenizer, MarkedPrefixEncoder

LOCAL_QWEN38 = Path(
    "~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP"
)


@pytest.mark.skipif(not (LOCAL_QWEN38 / "tokenizer.json").is_file(), reason="local Qwen3.8 tokenizer absent")
def test_real_qwen38_plain_span_change_and_growth_are_exact():
    tokenizer = HFMarkedTokenizer(LOCAL_QWEN38)
    encoder = MarkedPrefixEncoder(tokenizer)
    base = "<|im_start|>user\nQuote <think>literal</think> text.<|im_end|>\n"
    marker = "<think>literal</think>"
    start = base.index(marker)
    spans = ((start, start + len(marker)),)

    normal = encoder.encode(base)
    plain = encoder.encode(base, plain_spans=spans)
    assert normal == tokenizer.encode(base)
    assert plain == tokenizer.encode(base, plain_spans=spans)
    assert plain != normal
    assert encoder.last_reused_chars <= start

    grown = base + "<|im_start|>assistant\n"
    assert encoder.encode(grown, plain_spans=spans) == tokenizer.encode(
        grown, plain_spans=spans
    )
    assert encoder.last_reused_chars > 0


@pytest.mark.skipif(not (LOCAL_QWEN38 / "tokenizer.json").is_file(), reason="local Qwen3.8 tokenizer absent")
def test_real_qwen38_growing_edited_and_interleaved_suite():
    from scripts.bench_real_qwen38_tokenizer_viability import run_suite

    report = run_suite(LOCAL_QWEN38, base_chars=2000, turns=3, repeats=1)
    assert report["all_ids_exact"] is True
    assert report["plain_policy_change_cut_before_span"] is True
    assert report["plain_policy_growth_reused_prefix"] is True
    assert report["artifact"]["model_weights_opened"] is False
    assert report["artifact"]["network_used"] is False
    assert {row["name"] for row in report["cases"]} >= {
        "edited-earlier-turn",
        "interleaved-a2",
        "interleaved-b2",
        "quoted-special-plain-policy-change",
    }
