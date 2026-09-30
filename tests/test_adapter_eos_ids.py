"""Adapters stop on every end token the artifact declares."""

import json
from types import SimpleNamespace

from mlx2.adapters.eos import artifact_eos_token_ids
from mlx2.adapters.qwen38_27b import resolve_eos_token_ids


def test_generation_config_eos_ids_are_included(tmp_path):
    """Qwen3 lists <|endoftext|> (151643) only in generation_config.json; the
    standard-decoder and pinned-VLM adapters read config.json alone and kept
    generating past it."""
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [151645, 151643]}))
    ids = artifact_eos_token_ids(tmp_path, {"eos_token_id": 151645}, SimpleNamespace(eos_token_id=151645))
    assert ids == [151645, 151643]


def test_text_config_and_tokenizer_ids_join_without_duplicates(tmp_path):
    ids = artifact_eos_token_ids(
        tmp_path, {"text_config": {"eos_token_id": [2, 106]}}, SimpleNamespace(eos_token_id=1)
    )
    assert ids == [2, 106, 1]
    assert artifact_eos_token_ids(tmp_path, {"eos_token_id": True}) == []


def test_resolving_eos_ids_does_not_mutate_the_artifact_config():
    config = {"eos_token_id": [151645]}
    assert resolve_eos_token_ids(config, SimpleNamespace(eos_token_id=151643)) == [151645, 151643]
    assert config["eos_token_id"] == [151645]
