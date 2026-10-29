import copy

import pytest

from mlx2.adapters.deepseek_v4_gguf_tokenizer import align_tokenizer


def fixture():
    reference = {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}},
                 "added_tokens": [{"id": 2, "content": "old", "special": True}]}
    metadata = {"tokenizer.ggml.tokens": ("a", "b", "new"),
                "tokenizer.ggml.merges": (), "tokenizer.chat_template": "{{ messages }}",
                "tokenizer.ggml.bos_token_id": 0,
                "tokenizer.ggml.eos_token_id": 1,
                "tokenizer.ggml.padding_token_id": 1}
    return reference, metadata


def test_reserved_spelling_retains_id_and_original_input():
    reference, metadata = fixture()
    original = copy.deepcopy(reference)
    result, receipt = align_tokenizer(reference, metadata)
    assert reference == original
    assert result["added_tokens"] == [{"id": 2, "content": "new", "special": True}]
    assert receipt["renamed_reserved_tokens"] == [
        {"id": 2, "reference": "old", "gguf": "new"}]


@pytest.mark.parametrize("change", ["merges", "coverage", "duplicates", "special_id"])
def test_incompatible_tokenizer_is_refused(change):
    reference, metadata = fixture()
    if change == "merges":
        metadata["tokenizer.ggml.merges"] = ("a b",)
    elif change == "coverage":
        reference["added_tokens"][0]["id"] = 3
    elif change == "duplicates":
        metadata["tokenizer.ggml.tokens"] = ("a", "b", "b")
    else:
        metadata["tokenizer.ggml.eos_token_id"] = -1
    with pytest.raises(ValueError):
        align_tokenizer(reference, metadata)


def test_merging_a_renamed_symbol_is_refused():
    reference, metadata = fixture()
    reference["model"]["merges"] = ["old b"]
    metadata["tokenizer.ggml.merges"] = ("old b",)
    with pytest.raises(ValueError, match="participating"):
        align_tokenizer(reference, metadata)
