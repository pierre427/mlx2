import json
import math

import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx2.decisions.candidates.base import (
    add_request_tokens,
    format_answers,
    inspect_index,
)
from mlx2.decisions.candidates.decision2 import (
    HEAD_KEYS,
    Decision2Head,
    _inside,
)
from mlx2.decisions.candidates.decision2 import (
    inspect_artifact as inspect_decision2,
)
from mlx2.decisions.candidates.decision2 import (
    render_prompt as render_decision2,
)
from mlx2.decisions.candidates.jev import (
    inspect_artifact as inspect_jev,
)
from mlx2.decisions.candidates.jev import (
    render_prompt as render_jev,
)
from mlx2.decisions.candidates.pplx import (
    SYSTEM,
)
from mlx2.decisions.candidates.pplx import (
    _question as pplx_question,
)
from mlx2.decisions.candidates.pplx import (
    inspect_artifact as inspect_pplx,
)
from mlx2.decisions.candidates.pplx import (
    render_prompt as render_pplx,
)
from mlx2.decisions.schema import (
    DecisionInputTooLong,
    DecisionRequestError,
    normalize_request,
)
from mlx2.decisions.tokenizer import load_local_tokenizer


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    @staticmethod
    def decode(tokens, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        assert enable_thinking is False
        text = "".join(
            f"<|im_start|>{row['role']}\n{row['content']}<|im_end|>\n"
            for row in messages
        )
        text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.encode(text)


def test_decision_tokenizer_uses_mlx2_integrity_repair(monkeypatch, tmp_path):
    import transformers

    from mlx2.runtime import tokenizer_integrity

    sentinel = object()
    calls = {}

    def fake_load(path, **kwargs):
        calls["load"] = (path, kwargs)
        return sentinel

    def fake_repair(tokenizer, path):
        calls["repair"] = (tokenizer, path)
        return {
            "file": {"status": "restored", "components": ["pre_tokenizer"]},
            "declared": {"status": "repaired"},
        }

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_load)
    monkeypatch.setattr(tokenizer_integrity, "repair_loaded_tokenizer", fake_repair)
    tokenizer, receipt = load_local_tokenizer(tmp_path)
    assert tokenizer is sentinel
    assert calls["load"][1] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert calls["repair"] == (sentinel, tmp_path)
    assert receipt["declared"]["status"] == "repaired"


def test_decision_tokenizer_refuses_an_undeclared_integrity_contract(
    monkeypatch, tmp_path
):
    import transformers

    from mlx2.runtime import tokenizer_integrity

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        tokenizer_integrity,
        "repair_loaded_tokenizer",
        lambda *args: {
            "file": {"status": "consistent"},
            "declared": {"status": "absent"},
        },
    )
    with pytest.raises(ValueError, match="declare a repairable"):
        load_local_tokenizer(tmp_path)


def test_candidate_aggregate_prompt_budget_is_bounded():
    assert add_request_tokens(0, [1] * 8, max_context=1) == 8
    with pytest.raises(DecisionInputTooLong, match="aggregate tokens"):
        add_request_tokens(8, [1], max_context=1)


def _request():
    return normalize_request(
        {
            "model": "candidate",
            "state": {"ticket": "refund"},
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "Route?",
                    "criteria": {"sales": None, "billing": "Payments"},
                },
                "mood": {
                    "type": "score",
                    "instructions": "Mood?",
                    "criteria": ["calm", "angry"],
                },
                "urgent": {"type": "noul", "instructions": "Urgent?"},
            },
        },
        default_model="candidate",
    )


def test_decision2_segmented_prompt_and_candidate_rows_are_exact():
    request = _request()
    tokens, candidates, query, labels = render_decision2(
        CharacterTokenizer(),
        request.state,
        "team",
        request.questions["team"],
        max_length=4096,
        truncate=True,
    )
    text = CharacterTokenizer.decode(tokens)
    assert text.startswith(
        'Context:\n{"ticket":"refund"}\n\nTask type: choice\nQuestion:\nRoute?'
    )
    assert labels == ["sales", "billing"]
    assert [text[index] for index in candidates] == [">", ">"]
    assert text[query] == ":"
    assert text.index('"key":"sales"') < text.index('"key":"billing"')
    assert text.endswith("instructions.\nDecision:")


def test_decision2_preserves_explicit_noul_order_and_requires_instructions():
    normalized = normalize_request(
        {
            "model": "candidate",
            "state": "x",
            "questions": {
                "truth": {
                    "type": "noul",
                    "instructions": "True?",
                    "criteria": {"true": "Yes", "false": "No"},
                }
            },
        },
        default_model="candidate",
    )
    _, _, _, labels = render_decision2(
        CharacterTokenizer(),
        normalized.state,
        "truth",
        normalized.questions["truth"],
        max_length=4096,
        truncate=True,
    )
    assert labels == ["true", "false"]
    missing = normalize_request(
        {
            "model": "candidate",
            "state": "x",
            "questions": {
                "truth": {"type": "noul", "criteria": {"false": "No"}}
            },
        },
        default_model="candidate",
    )
    with pytest.raises(DecisionRequestError, match="instructions must be provided"):
        render_decision2(
            CharacterTokenizer(),
            missing.state,
            "truth",
            missing.questions["truth"],
            max_length=4096,
            truncate=True,
        )


def test_decision2_truncates_only_state_and_can_refuse():
    question = _request().questions["team"]
    tokens, _, _, _ = render_decision2(
        CharacterTokenizer(),
        "x" * 1000,
        "team",
        question,
        max_length=400,
        truncate=True,
    )
    assert len(tokens) == 400
    with pytest.raises(DecisionInputTooLong):
        render_decision2(
            CharacterTokenizer(),
            "x" * 1000,
            "team",
            question,
            max_length=400,
            truncate=False,
        )
    empty_tokens, *_ = render_decision2(
        CharacterTokenizer(),
        "",
        "team",
        question,
        max_length=4096,
        truncate=True,
    )
    with pytest.raises(DecisionInputTooLong, match="entire nonempty state"):
        render_decision2(
            CharacterTokenizer(),
            "x",
            "team",
            question,
            max_length=len(empty_tokens),
            truncate=True,
        )


def test_decision2_head_is_separate_f32_candidate_math():
    import mlx.core as mx

    mx.random.seed(4)
    head = Decision2Head(hidden_size=8, width=4)
    scores = head(mx.random.normal((3, 8)), mx.random.normal((8,)))
    mx.eval(scores)
    assert scores.shape == (3,)
    assert mx.all(mx.isfinite(scores)).item()


def test_pplx_prompt_uses_checkpoint_codes_and_calibration_layout():
    request = _request()
    codes = [chr(ord("A") + index) for index in range(26)]
    labels, selected, suffix = pplx_question(
        "urgent", request.questions["urgent"], codes
    )
    assert labels == ["true", "false"]
    assert selected == [1, 0]
    tokens = render_pplx(
        CharacterTokenizer(),
        request.state,
        suffix,
        max_length=4096,
        truncate=True,
    )
    text = CharacterTokenizer.decode(tokens)
    assert SYSTEM in text
    assert "Options:\nA: No / false\nB: Yes / true" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    empty = render_pplx(
        CharacterTokenizer(),
        "",
        suffix,
        max_length=4096,
        truncate=True,
    )
    with pytest.raises(DecisionInputTooLong, match="entire nonempty state"):
        render_pplx(
            CharacterTokenizer(),
            "x",
            suffix,
            max_length=len(empty),
            truncate=True,
        )


def test_jev_prompt_and_family_constraints():
    request = _request()
    codes = tuple(chr(ord("A") + index) for index in range(26))
    tokens, labels = render_jev(
        CharacterTokenizer(),
        request.state,
        "team",
        request.questions["team"],
        codes,
        max_length=4096,
        truncate=True,
    )
    text = CharacterTokenizer.decode(tokens)
    assert labels == ["sales", "billing"]
    assert text.startswith('[kind] choice\n[state] {"ticket": "refund"}')
    assert "A) sales\nB) billing: Payments" in text
    assert text.endswith("[decision]:")
    empty, _ = render_jev(
        CharacterTokenizer(),
        "",
        "team",
        request.questions["team"],
        codes,
        max_length=4096,
        truncate=True,
    )
    with pytest.raises(DecisionInputTooLong, match="entire nonempty state"):
        render_jev(
            CharacterTokenizer(),
            "x",
            "team",
            request.questions["team"],
            codes,
            max_length=len(empty),
            truncate=True,
        )
    with pytest.raises(Exception, match="six levels"):
        render_jev(
            CharacterTokenizer(),
            request.state,
            "mood",
            request.questions["mood"],
            codes,
            max_length=4096,
            truncate=True,
        )


def test_candidate_answer_shape_is_shared_without_sharing_tensor_math():
    request = _request()
    answers = format_answers(
        [
            ("urgent", request.questions["urgent"], ["false", "true"], [0.2, 0.8]),
            (
                "team",
                request.questions["team"],
                ["sales", "billing"],
                [0.1, 0.9],
            ),
        ]
    )
    assert answers["urgent"]["value"] is True
    assert answers["team"]["value"] == "billing"
    with pytest.raises(RuntimeError, match="finite"):
        format_answers(
            [("urgent", request.questions["urgent"], ["false", "true"], [math.nan, 0.0])]
        )


def _write_index(root, keys, shard="model.safetensors"):
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard for key in keys}})
    )
    (root / shard).parent.mkdir(parents=True, exist_ok=True)
    (root / shard).write_bytes(b"12345678")
    (root / "tokenizer.json").write_text("{}")


def test_jEV_artifact_inspection_is_strict_and_cpu_safe(tmp_path):
    decision = {
        "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
        "verbalizer_ids": list(range(24)),
        "bias": [0.0] * 24,
        "temperature_by_type": {"noul": 1.0, "choice": 1.0, "score": 1.0},
    }
    config = {
        "model_type": "jev_text",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "vocab_size": 151936,
        "max_position_embeddings": 262144,
        "decision_config": decision,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    _write_index(
        tmp_path,
        [
            "language_model.model.embed_tokens.weight",
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.norm.weight",
            "language_model.lm_head.weight",
        ],
    )
    artifact = inspect_jev(tmp_path)
    assert artifact["variant"] == "jev-9b"
    decision["verbalizer_ids"][-1] = config["vocab_size"]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="exceed the vocabulary"):
        inspect_jev(tmp_path)
    decision["verbalizer_ids"][-1] = 23
    config["model_type"] = "qwen3_5_text"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="jev_text"):
        inspect_jev(tmp_path)


def test_pplx_artifact_requires_exact_readout(tmp_path):
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5Model"],
        "text_config": {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "max_position_embeddings": 262144,
        },
    }
    decision = {
        "format_version": 1,
        "codes": [f"C{index}" for index in range(255)],
        "token_ids": list(range(255)),
        "temperature": 2.0,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "decision_config.json").write_text(json.dumps(decision))
    save_file(
        {"weight": np.zeros((255, 5120), dtype=np.float16)},
        tmp_path / "readout.safetensors",
    )
    _write_index(
        tmp_path,
        [
            "language_model.embed_tokens.weight",
            "language_model.layers.0.input_layernorm.weight",
            "language_model.norm.weight",
            "visual.patch_embed.weight",
        ],
    )
    assert inspect_pplx(tmp_path)["variant"] == "pplx-decider-v1-27b"
    decision["temperature"] = 0
    (tmp_path / "decision_config.json").write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="temperature"):
        inspect_pplx(tmp_path)


def test_decision2_artifact_requires_exact_lux_head(tmp_path):
    (tmp_path / "backbone").mkdir()
    config = {
        "model_type": "decision2",
        "package_schema": "dev2-package/1",
        "calibration": None,
        "max_input_tokens": 16384,
        "backbone": {
            "config": "backbone/config.json",
            "index": "backbone/model.safetensors.index.json",
        },
        "model_config": "decision_config.json",
        "decision_weights": {"decision_head": "decision_head.safetensors"},
    }
    text = {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "max_position_embeddings": 262144,
    }
    decision = {
        "prompt_version": "decision2-segmented-options-global-query-v1",
        "head_variant": "shared",
        "head_dim": 256,
        "max_options": 255,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "backbone/config.json").write_text(json.dumps(text))
    (tmp_path / "decision_config.json").write_text(json.dumps(decision))
    arrays = {
        "candidate_norm.weight": np.zeros((4096,), dtype=np.float32),
        "candidate_norm.bias": np.zeros((4096,), dtype=np.float32),
        "query_norm.weight": np.zeros((4096,), dtype=np.float32),
        "query_norm.bias": np.zeros((4096,), dtype=np.float32),
        "key.weight": np.zeros((256, 4096), dtype=np.float32),
        "query.weight": np.zeros((256, 4096), dtype=np.float32),
        "candidate_mlp.weight": np.zeros((256, 4096), dtype=np.float32),
        "candidate_mlp.bias": np.zeros((256,), dtype=np.float32),
        "query_mlp.weight": np.zeros((256, 4096), dtype=np.float32),
        "scalar.weight": np.zeros((1, 256), dtype=np.float32),
    }
    assert set(arrays) == HEAD_KEYS
    save_file(arrays, tmp_path / "decision_head.safetensors")
    _write_index(
        tmp_path / "backbone",
        ["embed_tokens.weight", "layers.0.input_layernorm.weight", "norm.weight"],
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    assert inspect_decision2(tmp_path)["variant"] == "decision2-lux-9b"


def test_decision2_accepts_hugging_face_snapshot_symlinks(tmp_path):
    repository = tmp_path / "models--example--lux"
    snapshot = repository / "snapshots" / "revision"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    target = blobs / "config-blob"
    target.write_text("{}")
    (snapshot / "config.json").symlink_to(target)

    lexical = _inside(snapshot, "config.json")
    assert lexical == snapshot / "config.json"
    assert lexical.read_text() == "{}"

    foreign = tmp_path / "foreign.json"
    foreign.write_text("{}")
    (snapshot / "foreign.json").symlink_to(foreign)
    with pytest.raises(ValueError, match="missing or escaped"):
        _inside(snapshot, "foreign.json")


def test_candidate_index_preserves_snapshot_safetensors_name(tmp_path):
    repository = tmp_path / "models--example--candidate"
    snapshot = repository / "snapshots" / "revision"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    index_path = snapshot / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00001.safetensors"
                }
            }
        )
    )
    target = blobs / "extensionless-content-hash"
    target.write_bytes(b"12345678")
    shard = snapshot / "model-00001-of-00001.safetensors"
    shard.symlink_to(target)

    artifact = inspect_index(
        snapshot,
        index_path,
        metadata_paths=(index_path,),
        allowed_prefixes=("model.",),
        required_prefixes=("model.embed_tokens.",),
    )
    assert artifact["shard_files"] == [shard]
