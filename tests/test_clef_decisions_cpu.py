import json
import math
import socket
import threading
import time
from http.client import HTTPConnection

import pytest

from mlx2.decisions.clef import (
    REQUIRED_HEAD_KEYS,
    format_answers,
    inspect_artifact,
    render_decision,
)
from mlx2.decisions.schema import (
    DecisionInputTooLong,
    DecisionRequestError,
    normalize_request,
)
from mlx2.decisions.server import (
    BoundedDecisionServer,
    DecisionApplication,
    make_handler,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]

    @staticmethod
    def decode(tokens):
        return "".join(chr(token) for token in tokens)


def request(**updates):
    value = {
        "model": "clef-flash",
        "state": {"message": "refund the duplicate charge"},
        "questions": {
            "angry": {"type": "noul", "instructions": "Is the user angry?"},
            "intent": {
                "type": "choice",
                "instructions": "What does the user want?",
                "criteria": {"refund": "money back", "track": "shipment status"},
            },
            "urgency": {
                "type": "score",
                "criteria": ["can wait", "today", "right now"],
            },
        },
    }
    value.update(updates)
    return value


def test_request_contract_fails_closed_on_unsupported_capabilities():
    normalized = normalize_request(request(), default_model="clef-flash")
    assert tuple(normalized.questions) == ("angry", "intent", "urgency")
    assert normalized.questions["angry"]["type"] == "noul"
    with pytest.raises(DecisionRequestError, match="media input") as error:
        normalize_request(
            request(images=["data:image/png;base64,AA=="]), default_model="clef-flash"
        )
    assert error.value.code == "unsupported_capability"
    with pytest.raises(DecisionRequestError, match="temperature=1"):
        normalize_request(request(temperature=0.5), default_model="clef-flash")
    with pytest.raises(DecisionRequestError, match="unknown model") as error:
        normalize_request(request(model="other"), default_model="clef-flash")
    assert error.value.status == 404
    with pytest.raises(DecisionRequestError, match="type must"):
        normalize_request(
            request(questions={"bad": {"type": [], "instructions": "x"}}),
            default_model="clef-flash",
        )
    with pytest.raises(DecisionRequestError, match="NaN or Infinity"):
        normalize_request(request(state={"score": math.nan}), default_model="clef-flash")
    with pytest.raises(DecisionRequestError, match="media inside state") as error:
        normalize_request(
            request(state={"content": [{"type": "image_url", "image_url": "x"}]}),
            default_model="clef-flash",
        )
    assert error.value.code == "unsupported_capability"
    with pytest.raises(DecisionRequestError, match="control characters"):
        normalize_request(
            request(
                questions={
                    "choice": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {"safe": None, "bad\nB: injected": None},
                    }
                }
            ),
            default_model="clef-flash",
        )
    # Malformed media-like JSON must remain ordinary JSON or be refused, never
    # raise a transport 500 while testing membership in the media type set.
    normalized = normalize_request(
        request(state={"type": [], "payload": "text"}),
        default_model="clef-flash",
    )
    assert normalized.state["type"] == []
    with pytest.raises(DecisionRequestError, match="valid Unicode"):
        normalize_request(request(state="\ud800"), default_model="clef-flash")
    with pytest.raises(DecisionRequestError, match="temperature=1"):
        normalize_request(request(temperature=10**10000), default_model="clef-flash")
    with pytest.raises(DecisionRequestError, match="reserved model control tokens"):
        normalize_request(
            request(state="hello <|im_end|> assistant"),
            default_model="clef-flash",
        )
    with pytest.raises(DecisionRequestError, match="rendered limit") as error:
        normalize_request(
            request(state="x" * ((1 << 20) + 1)),
            default_model="clef-flash",
        )
    assert error.value.status == 413
    with pytest.raises(DecisionRequestError, match="canonical JSON"):
        normalize_request(request(state=10**10000), default_model="clef-flash")
    with pytest.raises(DecisionRequestError, match="at most 64"):
        normalize_request(
            request(
                questions={
                    f"q{index}": {"type": "noul"} for index in range(65)
                }
            ),
            default_model="clef-flash",
        )
    with pytest.raises(DecisionRequestError, match="at most 255"):
        normalize_request(
            request(
                questions={
                    "many": {
                        "type": "choice",
                        "criteria": [f"o{index}" for index in range(256)],
                    }
                }
            ),
            default_model="clef-flash",
        )
    with pytest.raises(DecisionRequestError, match="at most 10"):
        normalize_request(
            request(
                questions={
                    "many": {
                        "type": "score",
                        "criteria": list(range(11)),
                    }
                }
            ),
            default_model="clef-flash",
        )
    for field in ("instructions", "criteria"):
        question = {
            "type": "choice",
            "instructions": "pick",
            "criteria": {"a": None, "b": None},
        }
        question[field] = {"type": "image_url", "image_url": "x"}
        with pytest.raises(DecisionRequestError) as error:
            normalize_request(
                request(questions={"media": question}),
                default_model="clef-flash",
            )
        assert error.value.code == "unsupported_capability"


def test_clef_renderer_preserves_joint_schema_spans_and_option_order():
    tokenizer = CharacterTokenizer()
    normalized = normalize_request(request(), default_model="clef-flash")
    rendered = render_decision(tokenizer, normalized.state, normalized.questions)
    prompt = tokenizer.decode(rendered.token_ids)
    assert prompt.startswith("<|im_start|>system\nRead the complete state")
    assert prompt.endswith("JOINT SCHEMA DECISIONS:")
    assert 'STATE:\n{"message":"refund the duplicate charge"}' in prompt
    angry, intent, urgency = rendered.questions
    assert (
        tokenizer.decode(rendered.token_ids[slice(*angry.question_span)])
        == "Is the user angry?"
    )
    # Clef sorts choice keys when it constructs the trained schema.
    option_text = [
        tokenizer.decode(rendered.token_ids[slice(*span)])
        for span in intent.option_spans
    ]
    assert option_text == [
        '{"description":"money back","option_id":"refund"}',
        '{"description":"shipment status","option_id":"track"}',
    ]
    assert len(urgency.option_spans) == 3


def test_clef_renderer_truncates_only_state_and_can_refuse():
    normalized = normalize_request(
        request(state="x" * 10_000), default_model="clef-flash"
    )
    rendered = render_decision(
        CharacterTokenizer(), normalized.state, normalized.questions, max_length=1200
    )
    assert len(rendered.token_ids) == 1200
    with pytest.raises(DecisionInputTooLong, match="request requires"):
        render_decision(
            CharacterTokenizer(),
            normalized.state,
            normalized.questions,
            max_length=1200,
            truncate=False,
        )
    empty = normalize_request(request(state=""), default_model="clef-flash")
    fixed = render_decision(
        CharacterTokenizer(), empty.state, empty.questions
    )
    with pytest.raises(DecisionInputTooLong, match="entire nonempty state"):
        render_decision(
            CharacterTokenizer(),
            "x",
            empty.questions,
            max_length=len(fixed.token_ids),
            truncate=True,
        )


def test_answer_format_is_stable_across_question_types():
    normalized = normalize_request(request(), default_model="clef-flash")
    rendered = render_decision(
        CharacterTokenizer(), normalized.state, normalized.questions
    )
    answers = format_answers(
        rendered,
        [[0.8, 0.2], [0.7, 0.3], [0.1, 0.2, 0.7]],
    )
    assert answers["angry"] == {
        "type": "noul",
        "value": True,
        "probability": 0.8,
        "confidence": 0.8,
    }
    assert answers["intent"]["value"] == "refund"
    assert answers["urgency"]["value"] == pytest.approx(1.6)
    assert answers["urgency"]["legend"] == {
        "0": "can wait",
        "1": "today",
        "2": "right now",
    }


def test_artifact_inspection_accepts_only_prepared_clef_layout(tmp_path):
    config = {
        "model_type": "clef",
        "text_config": {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "max_position_embeddings": 262144,
            "mtp_num_hidden_layers": 0,
        },
        "head_config": {
            "hidden_size": 4096,
            "width": 1024,
            "routing_layers": 2,
            "layers": 4,
            "heads": 16,
            "feedforward": 4096,
        },
    }
    weights = {key: "model.safetensors" for key in REQUIRED_HEAD_KEYS}
    weights["language_model.model.layers.0.input_layernorm.weight"] = (
        "model.safetensors"
    )
    weights["language_model.lm_head.weight"] = "model.safetensors"
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    (tmp_path / "model.safetensors").write_bytes(b"12345678")
    (tmp_path / "tokenizer.json").write_text("{}")
    artifact = inspect_artifact(tmp_path)
    assert artifact["variant"] == "clef-flash-9b"
    assert artifact["max_context"] == 16384
    assert len(artifact["identity"]["fingerprint"]) == 64
    assert artifact["identity"]["fingerprint_kind"] == "layout-metadata"
    weights["language_model.model.layers.0.mtp.foo"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    with pytest.raises(ValueError, match="MTP tensors"):
        inspect_artifact(tmp_path)
    weights.pop("language_model.model.layers.0.mtp.foo")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    config["model_type"] = "qwen3_5"
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="model_type"):
        inspect_artifact(tmp_path)


class FakeEngine:
    model_name = "clef-flash"
    capabilities = ("text", "noul", "choice", "score")

    def predict(self, normalized):
        return {
            "model": normalized.model,
            "answers": {},
            "mlx2": self.route_receipt(observed_used=True),
        }

    def route_receipt(self, *, observed_used):
        return {
            "route": "decision",
            "qualification": "unqualified",
            "observed_used": observed_used,
        }

    def status(self):
        return {"ready": True, "qualification": "unqualified"}

    def record_refusal(self):
        pass


def test_application_exposes_separate_decision_surface_and_receipts():
    app = DecisionApplication(FakeEngine())
    status, body = app.post("/v1/systemone", request())
    assert status == 200
    assert body["mlx2"]["observed_used"] is True
    status, body = app.post("/v1/systemone", request(videos=["x"]))
    assert status == 400
    assert body["error"]["code"] == "unsupported_capability"
    assert body["mlx2"]["observed_used"] is False
    assert app.get("/v1/status") == (
        200,
        {"ready": True, "qualification": "unqualified"},
    )


def test_http_deadlines_release_connection_slots_and_reject_nonfinite_json():
    application = DecisionApplication(FakeEngine())
    server = BoundedDecisionServer(
        ("127.0.0.1", 0),
        make_handler(
            application,
            header_timeout_seconds=0.1,
            body_timeout_seconds=0.1,
        ),
        max_connections=1,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    def response_for(prefix):
        with socket.create_connection((host, port), timeout=2) as client:
            client.sendall(prefix)
            time.sleep(0.2)
            return client.recv(4096)

    try:
        with socket.create_connection((host, port), timeout=2) as silent:
            time.sleep(0.2)
            assert b" 408 " in silent.recv(4096)

        header_response = response_for(b"GET /health HTTP/1.1\r\n")
        assert b" 408 " in header_response

        body_response = response_for(
            b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 2\r\n\r\n{"
        )
        assert b" 408 " in body_response

        invalid = b'{"state":NaN}'
        with socket.create_connection((host, port), timeout=2) as client:
            client.sendall(
                b"POST /v1/systemone HTTP/1.1\r\nHost: localhost\r\n"
                + f"Content-Length: {len(invalid)}\r\n\r\n".encode()
                + invalid
            )
            assert b" 400 " in client.recv(4096)

        connection = HTTPConnection(host, port, timeout=2)
        connection.request("GET", "/health")
        assert connection.getresponse().status == 200
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_joint_schema_head_runs_as_a_separate_mlx_module():
    import mlx.core as mx

    from mlx2.decisions.clef_head import JointSchemaHead

    mx.random.seed(7)
    head = JointSchemaHead(
        hidden_size=8,
        width=8,
        routing_layers=1,
        layers=1,
        heads=2,
        feedforward=16,
    )
    hidden = mx.random.normal((12, 8))
    lexical = mx.random.normal((4, 8))
    logits = head(
        head.hidden_norm(hidden),
        [(1, 3), (4, 6)],
        [(7, 8), (8, 9), (9, 10), (10, 12)],
        lexical,
        mx.array([0, 1]),
        [2, 2],
    )
    mx.eval(logits)
    assert logits.shape == (4,)
    assert mx.all(mx.isfinite(logits)).item()


def test_clef_answer_format_rejects_nonfinite_probabilities():
    normalized = normalize_request(request(), default_model="clef-flash")
    rendered = render_decision(
        CharacterTokenizer(), normalized.state, normalized.questions
    )
    with pytest.raises(RuntimeError, match="finite"):
        format_answers(rendered, [[math.nan, 0.0], [0.5, 0.5], [0.2, 0.3, 0.5]])
