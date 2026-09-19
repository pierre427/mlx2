"""Probability serialization and MTP target-row wiring without MLX imports."""

import ast
from collections import namedtuple
import gc
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from mlx2.logprobs import token_logprob, wants_logprobs
from mlx2.server import validate_request


TOKENIZER = SimpleNamespace(
    convert_ids_to_tokens=lambda ids: [f"token-{i}" for i in ids],
    decode=lambda ids, **_kwargs: "".join(f"token-{i}" for i in ids),
    all_special_ids=(),
    get_added_vocab=lambda: {},
)


def test_emitted_token_is_not_replaced_by_highest_probability_alternative():
    row = np.log(np.array([0.05, 0.7, 0.25]))
    reference = weakref.ref(row)
    result = token_logprob(row, 0, TOKENIZER, top_n=2, array_module=np)
    assert result["id"] == 0
    assert result["logprob"] == pytest.approx(np.log(0.05))
    assert [v["id"] for v in result["top_logprobs"]] == [1, 2]
    assert result["bytes"] == list(b"token-0")
    assert all("bytes" in item for item in result["top_logprobs"])
    del row
    gc.collect()
    assert reference() is None
    json.dumps(result, allow_nan=False)


def test_masked_probabilities_are_json_finite_and_top_count_is_bounded():
    result = token_logprob(np.array([-np.inf, 0.0]), 1, TOKENIZER, top_n=11, array_module=np)
    assert len(result["top_logprobs"]) == 2
    assert result["top_logprobs"][1]["logprob"] == -9999.0
    json.dumps(result, allow_nan=False)
    for value in (np.nan, np.inf):
        with pytest.raises(ValueError, match="invalid"):
            token_logprob(np.array([value]), 0, TOKENIZER, array_module=np)


def test_byte_fallback_and_partial_utf8_tokens_keep_their_raw_bytes():
    pieces = ["a", "\ufffd", "\ufffd"]
    tokenizer = SimpleNamespace(
        convert_ids_to_tokens=lambda ids: [["a", "<0xE2>", "<0x82>"][i] for i in ids],
        decode=lambda ids, **_kwargs: "".join(pieces[i] for i in ids),
        all_special_ids=(),
        get_added_vocab=lambda: {},
    )
    result = token_logprob(np.array([-2.0, -0.1, -1.0]), 1, tokenizer, top_n=2, array_module=np)
    assert result["bytes"] == [0xE2]
    by_id = {entry["id"]: entry["bytes"] for entry in result["top_logprobs"]}
    assert by_id[1] == [0xE2]
    assert by_id[2] == [0x82]


def test_sentencepiece_token_bytes_preserve_the_leading_space_marker():
    tokenizer = SimpleNamespace(
        convert_ids_to_tokens=lambda ids: ["▁hello" for _ in ids],
        # Isolated SentencePiece decode can omit the boundary space.
        decode=lambda ids, **_kwargs: "hello",
        all_special_ids=(),
        get_added_vocab=lambda: {},
    )
    result = token_logprob(np.array([0.0]), 0, tokenizer, array_module=np)
    assert result["bytes"] == list(b" hello")


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_actual_mtp_output_construction_uses_target_rows_for_accept_and_replacement(accepted):
    # Execute the production output-row construction only, with CPU arrays.
    # Draft and target probabilities deliberately disagree: reporting proposal
    # or residual probabilities would fail the target-row assertions.
    path = Path(__file__).parents[1] / "src/mlx2/runtime/hybrid_speculative.py"
    tree = ast.parse(path.read_text())
    append = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Attribute)
                  and isinstance(node.value.func.value, ast.Name)
                  and node.value.func.value.id == "output_rows"
                  and node.value.func.attr == "append")
    rows = np.log(np.array([[.1, .2, .7], [.6, .3, .1], [.2, .7, .1]]))
    output = []
    scope = dict(output_rows=output, MTPToken=namedtuple("MTPToken", "token logprobs from_draft"),
                 drafts=[[2, 0]], row=0, n_accept=accepted, bonus=1, logprobs=rows)
    exec(compile(ast.Module(body=[append], type_ignores=[]), str(path), "exec"), scope)
    emitted = output[0]
    assert len(emitted) == accepted + 1
    for position, item in enumerate(emitted):
        np.testing.assert_array_equal(item.logprobs, rows[position])
        serialized = token_logprob(item.logprobs, item.token, TOKENIZER, top_n=2, array_module=np)
        assert serialized["logprob"] == pytest.approx(rows[position, item.token])
    assert [item.from_draft for item in emitted] == [True] * accepted + [False]
    assert emitted[-1].token == 1


@pytest.mark.parametrize("extra", [
    {"logprobs": 1}, {"logprobs": "yes"}, {"top_logprobs": True},
    {"top_logprobs": -1}, {"top_logprobs": 12}, {"top_logprobs": 1.5},
    {"response_format": {"type": "text", "grammar": "x"}},
    {"response_format": {"type": "json_schema"}}, {"grammar": "("},
    {"tools": [{"type": "function", "function": {"name": "x", "strict": True}}]},
])
def test_invalid_or_unimplemented_controls_remain_rejected(extra):
    with pytest.raises(ValueError):
        validate_request({"prompt": "hi", **extra}, chat=False)


def test_text_format_is_normalized_and_top_alone_requests_probabilities():
    request = validate_request({"prompt": "hi", "response_format": {"type": "text"},
                                "top_logprobs": 3}, chat=False)
    assert "response_format" not in request
    assert wants_logprobs(request)
    assert not wants_logprobs({"logprobs": False, "top_logprobs": 0})
