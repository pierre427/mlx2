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
    return tokenizer, receipt
