"""End-of-sequence ids an artifact declares across its metadata files."""

from __future__ import annotations

import json
from pathlib import Path


def _ints(value) -> list[int]:
    values = value if isinstance(value, (list, tuple)) else [value]
    return [
        int(item) for item in values
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0
    ]


def artifact_eos_token_ids(path, config: dict, tokenizer=None) -> list[int]:
    """Union of the EOS ids in config.json, generation_config.json and the tokenizer.

    Many checkpoints list extra end tokens only in ``generation_config.json``
    (Qwen3: config 151645, generation config ``[151645, 151643]``; Llama 3
    Instruct: 128001 vs ``[128001, 128009]``), which is what mlx-lm and
    transformers stop on.  Reading config.json alone kept generating past an
    ``<|endoftext|>``.  Order: config first, then generation config, then the
    tokenizer, without duplicates.
    """
    text = config.get("text_config") or {}
    ordered = _ints(config.get("eos_token_id")) + _ints(text.get("eos_token_id"))
    generation = Path(path) / "generation_config.json"
    if generation.is_file():
        try:
            ordered += _ints(json.loads(generation.read_text()).get("eos_token_id"))
        except (ValueError, AttributeError):
            pass
    if tokenizer is not None:
        ordered += _ints(getattr(tokenizer, "eos_token_id", None))
    result = []
    for value in ordered:
        if value not in result:
            result.append(value)
    return result
