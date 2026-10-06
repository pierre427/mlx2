from scripts.bench_tokenizer_viability import ByteSpecialTokenizer, MarkedPrefixEncoder


def test_incremental_tokenizer_matches_full_for_growth_edits_and_interleaving():
    tokenizer = ByteSpecialTokenizer()
    encoder = MarkedPrefixEncoder(tokenizer, keep=2)
    prompts = [
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nworld<|im_end|>\n",
        "<|im_start|>user\nother<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nhello edited<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nhello<|im_end|>\n",
    ]
    for prompt in prompts:
        assert encoder.encode(prompt) == tokenizer.encode(prompt)


def test_special_token_lookahead_margin_is_load_bearing():
    tokenizer = ByteSpecialTokenizer(("<x>", "<x>z"))
    safe = MarkedPrefixEncoder(tokenizer)
    old, new = "a<x>TAIL", "a<x>zNEW"
    assert safe.encode(old) == tokenizer.encode(old)
    assert safe.encode(new) == tokenizer.encode(new)
    assert safe.last_reused_chars == 0

    unsafe = MarkedPrefixEncoder(tokenizer, margin_override=0)
    assert unsafe.encode(old) == tokenizer.encode(old)
    assert unsafe.encode(new) != tokenizer.encode(new)


def test_plain_span_policy_change_limits_reuse_even_when_text_is_identical():
    tokenizer = ByteSpecialTokenizer(("<think>", "</think>"))
    encoder = MarkedPrefixEncoder(tokenizer)
    text = "A<think>quoted</think>B"
    assert encoder.encode(text) == tokenizer.encode(text)
    spans = ((1, len(text) - 1),)
    assert encoder.encode(text, plain_spans=spans) == tokenizer.encode(text, plain_spans=spans)
    assert encoder.last_reused_chars == 0


def test_tokenizer_viability_report_is_exact_and_cpu_only():
    from scripts.bench_tokenizer_viability import run_benchmark

    report = run_benchmark(ByteSpecialTokenizer(), base_chars=1000, turns=4, repeats=1)
    assert report["ids_exact"] is True
    assert report["gpu_used"] is False
    assert report["mechanism_selected"] is False
    assert report["last_reused_chars"] > 900
