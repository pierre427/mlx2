"""Strict local tokenizer loading shared by decision-only engines."""

from __future__ import annotations

from pathlib import Path


def load_local_tokenizer(model_path: str | Path):
    """Load an artifact tokenizer and restore its declared pre-tokenization."""
    from transformers import AutoTokenizer

    from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

    path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )
    receipt = repair_loaded_tokenizer(tokenizer, path)
    for stage in ("file", "declared"):
        if receipt[stage]["status"] in {"missing", "unsupported"}:
            raise ValueError(
                f"decision tokenizer {stage} repair failed: {receipt[stage]}"
            )
    if receipt["declared"]["status"] not in {"consistent", "repaired"}:
        raise ValueError(
            "decision Qwen tokenizer must declare a repairable pretokenize_regex"
        )
    receipt["reserved_tokens"] = sorted(reserved_token_strings(tokenizer))
    return tokenizer, receipt


def reserved_token_strings(tokenizer) -> frozenset[str]:
    """Every string the tokenizer folds into one added or special token.

    ``encode(..., add_special_tokens=False)`` still matches added tokens inside
    user text, so ``</think>`` or ``<tool_call>`` in a request string becomes
    prompt structure.  The request contract refuses these strings; a tokenizer
    that exposes no added-token list cannot be guarded and is refused.
    """
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if not isinstance(decoder, dict):
        raise TypeError("decision tokenizer exposes no added token list")
    reserved = {str(getattr(token, "content", token)) for token in decoder.values()}
    reserved.update(
        str(token) for token in getattr(tokenizer, "all_special_tokens", ())
    )
    reserved.discard("")
    if not reserved:
        raise ValueError("decision tokenizer declares no added token to reserve")
    return frozenset(reserved)
