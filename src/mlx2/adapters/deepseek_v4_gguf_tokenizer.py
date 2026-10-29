"""Bind the checkpoint's exact GGUF vocabulary/template to its BPE algorithm.

The official tokenizer supplies the pre-tokenizer and byte-level decoder, not
permission to rename GGUF token IDs. Vision-Exp GGUF has nine reserved-token
spellings different from the pinned HF JSON. See the accompanying provenance.
This module is CPU-only and never imports a tensor runtime.
"""
from __future__ import annotations

import copy
import hashlib
import json


def align_tokenizer(reference: dict, metadata: dict) -> tuple[dict, dict]:
    """Preserve BPE rules while binding every token spelling to its exact ID."""
    tokens = metadata.get("tokenizer.ggml.tokens")
    merges = metadata.get("tokenizer.ggml.merges")
    template = metadata.get("tokenizer.chat_template")
    if (not isinstance(tokens, (list, tuple)) or not tokens
            or not all(isinstance(x, str) for x in tokens)
            or len(set(tokens)) != len(tokens)):
        raise ValueError("GGUF tokenizer requires a unique ordered vocabulary")
    if not isinstance(template, str) or not template:
        raise ValueError("GGUF checkpoint has no chat template")
    data = copy.deepcopy(reference)
    model = data.get("model", {})
    if model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict):
        raise ValueError("DeepSeek GGUF requires the pinned BPE tokenizer")
    reference_merges = model.get("merges", [])
    reference_merges = [" ".join(x) if isinstance(x, list) else x
                        for x in reference_merges]
    if not isinstance(merges, (list, tuple)) or list(merges) != reference_merges:
        raise ValueError("GGUF and reference BPE merges differ")
    vocabulary = model["vocab"]
    added = data.get("added_tokens", [])
    old_ids = {index: token for token, index in vocabulary.items()}
    for item in added:
        if not isinstance(item, dict) or type(item.get("id")) is not int:
            raise ValueError("invalid reference added-token record")
        old_ids[item["id"]] = item["content"]
    if set(old_ids) != set(range(len(tokens))):
        raise ValueError("GGUF/reference vocabulary ID coverage differs")
    # Match by ID, rather than appending renamed special tokens at new IDs.
    renamed = {old_ids[i]: tokens[i] for i in old_ids if old_ids[i] != tokens[i]}
    for merge in reference_merges:
        if any(part in renamed for part in merge.split(" ")):
            raise ValueError("cannot rename a token participating in BPE merges")
    model["vocab"] = {tokens[index]: index for _, index in vocabulary.items()}
    for item in added:
        item["content"] = tokens[item["id"]]
    if len(model["vocab"]) != len(vocabulary):
        raise ValueError("aligned vocabulary has colliding IDs")
    for key in ("bos_token_id", "eos_token_id", "padding_token_id"):
        value = metadata.get("tokenizer.ggml." + key)
        if type(value) is not int or not 0 <= value < len(tokens):
            raise ValueError("invalid GGUF tokenizer special ID: " + key)
    receipt = {
        "vocab_size": len(tokens), "merges": len(merges),
        "renamed_reserved_tokens": [
            {"id": i, "reference": old_ids[i], "gguf": tokens[i]}
            for i in old_ids if old_ids[i] != tokens[i]
        ],
        "vocabulary_sha256": hashlib.sha256(
            json.dumps(list(tokens), ensure_ascii=False,
                       separators=(",", ":")).encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "tensor_execution": False,
    }
    return data, receipt


def load_tokenizer(reference: dict, metadata: dict):
    """Instantiate only after the complete GGUF/reference identity is checked."""
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    aligned, receipt = align_tokenizer(reference, metadata)
    backend = Tokenizer.from_str(json.dumps(aligned, ensure_ascii=False))
    tokens = metadata["tokenizer.ggml.tokens"]
    if backend.get_vocab() != {token: i for i, token in enumerate(tokens)}:
        raise ValueError("tokenizer backend did not retain GGUF token IDs")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=tokens[metadata["tokenizer.ggml.bos_token_id"]],
        eos_token=tokens[metadata["tokenizer.ggml.eos_token_id"]],
        pad_token=tokens[metadata["tokenizer.ggml.padding_token_id"]],
        chat_template=metadata["tokenizer.chat_template"],
        clean_up_tokenization_spaces=False,
        model_max_length=metadata["deepseek4.context_length"],
    )
    return tokenizer, receipt
