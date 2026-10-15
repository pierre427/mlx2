"""Independent, text-only D3 decisions; provenance/d3-text-decisions.json."""

from __future__ import annotations

import math
from pathlib import Path

from ...adapters.artifact_paths import shard_within_artifact
from ..schema import DecisionInputTooLong, DecisionRequestError
from .base import (
    CandidateEngine,
    add_request_tokens,
    format_answers,
    inspect_index,
    read_object,
    usage,
)
from .d3_format import answer_codes, chat_text, describe, messages, options

SOURCE_REVISION = "cab71e6ad3461b8086d51ddb572c656babd63b0b"
INITIAL_MAX_CONTEXT = 32768


def inspect_artifact(model_path):
    """Inspect exact named contracts without importing MLX or executing model code."""
    from safetensors import safe_open

    root = Path(model_path).expanduser().resolve()
    config = read_object(root / "config.json")
    decision = read_object(root / "decision_config.json")
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != [
        "Qwen3_5Model"
    ]:
        raise ValueError("D3 requires a bare Qwen3_5Model backbone")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise TypeError("D3 requires text_config")
    expected = {
        "num_hidden_layers": 12,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    if any(
        type(text.get(key)) is not int or text[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("D3 initial text adapter requires the published edge topology")
    kinds = text.get("layer_types")
    if (
        not isinstance(kinds, list)
        or len(kinds) != text["num_hidden_layers"]
        or any(kind not in {"linear_attention", "full_attention"} for kind in kinds)
    ):
        raise ValueError("D3 requires explicit valid layer_types")
    if (
        text.get("hidden_act") != "silu"
        or text.get("attn_output_gate") is not True
        or text.get("attention_bias") is not False
        or text.get("num_experts", 0) != 0
        or text.get("dtype") != "bfloat16"
        or config.get("quantization")
        or config.get("quantization_config")
    ):
        raise ValueError("unsupported D3 compute or quantization contract")
    rope = text.get("rope_parameters")
    eps = text.get("rms_norm_eps")
    if (
        not isinstance(rope, dict)
        or rope.get("rope_type") != "default"
        or rope.get("partial_rotary_factor") != 0.25
        or type(rope.get("rope_theta")) not in (int, float)
        or not math.isfinite(rope["rope_theta"])
        or rope["rope_theta"] <= 0
        or type(eps) not in (int, float)
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("unsupported D3 RoPE or normalization contract")
    contract = {
        "format_version": 1,
        "format_id": "d3-code-readout-v1",
        "prompt": "d3",
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "readout_dtype": "float32",
    }
    if type(decision.get("format_version")) is not int or any(
        decision.get(key) != value for key, value in contract.items()
    ):
        raise ValueError("unsupported D3 decision contract")
    codes, ids = decision.get("codes"), decision.get("token_ids")
    if (
        not isinstance(codes, list)
        or not isinstance(ids, list)
        or len(codes) != 255
        or len(ids) != 255
        or any(
            not isinstance(code, str)
            or not code.isascii()
            or not code.isalpha()
            or not code.isupper()
            for code in codes
        )
        or any(
            type(token) is not int or not 0 <= token < text["vocab_size"]
            for token in ids
        )
        or len(set(codes)) != 255
        or len(set(ids)) != 255
    ):
        raise ValueError("D3 requires 255 distinct answer codes and token ids")
    temperature = decision.get("temperature")
    if (
        type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("D3 temperature must be finite and positive")
    maximum = text.get("max_position_embeddings")
    limit = decision.get("max_length")
    if (
        type(maximum) is not int
        or maximum < 1
        or (limit is not None and (type(limit) is not int or limit < 1))
    ):
        raise ValueError("D3 context limits must be positive integers")
    if not (root / "tokenizer.json").is_file():
        raise ValueError("D3 requires a local tokenizer")
    readout_path = root / "readout.safetensors"
    if not shard_within_artifact(root, readout_path.resolve()):
        raise ValueError("D3 readout escaped the artifact")
    with safe_open(readout_path, framework="numpy") as readout:
        if list(readout.keys()) != ["weight"]:
            raise ValueError("D3 readout must contain only weight")
        tensor = readout.get_slice("weight")
        if (
            tuple(tensor.get_shape()) != (255, text["hidden_size"])
            or tensor.get_dtype() != "F32"
        ):
            raise ValueError("D3 requires its trained FP32 255-row readout")
    index_path = root / "model.safetensors.index.json"
    weight_map = None
    if not index_path.is_file():
        # A backbone glob would also load the separate readout as model weights.
        backbone = root / "model.safetensors"
        if not shard_within_artifact(root, backbone.resolve()):
            raise ValueError("D3 backbone escaped the artifact")
        with safe_open(backbone, framework="numpy") as handle:
            weight_map = dict.fromkeys(handle.keys(), "model.safetensors")
    indexed = inspect_index(
        root,
        index_path,
        metadata_paths=[
            root / name
            for name in (
                "config.json",
                "decision_config.json",
                "model.safetensors.index.json",
                "readout.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            )
        ],
        allowed_prefixes=("language_model.", "visual."),
        required_prefixes=(
            "language_model.embed_tokens.",
            "language_model.layers.",
            "language_model.norm.",
        ),
        weight_map=weight_map,
    )
    if any("mtp." in name or "lm_head." in name for name in indexed["weight_map"]):
        raise ValueError("D3 expects a bare backbone without MTP or LM-head tensors")
    return {
        **indexed,
        "path": root,
        "config": config,
        "text_config": text,
        "decision_config": decision,
        "readout_path": readout_path,
        "variant": "d3-edge-text-v1",
        "max_context": min(INITIAL_MAX_CONTEXT, maximum, limit or maximum),
    }


def validate_backbone_headers(artifact):
    """Bind indexed keys to real shards and require the stored BF16 recipe."""
    from safetensors import safe_open

    declared = artifact["weight_map"]
    root = artifact["path"]
    actual = {}
    shapes = expected_backbone_shapes(artifact["text_config"])
    seen_text = set()
    for path in artifact["shard_files"]:
        with safe_open(path, framework="numpy") as handle:
            for name in handle.keys():  # noqa: SIM118 - safetensors is not a mapping
                if name in actual:
                    raise ValueError(f"duplicate D3 backbone tensor: {name}")
                actual[name] = str(path.relative_to(root))
                if name.startswith("language_model."):
                    tensor = handle.get_slice(name)
                    if tensor.get_dtype() != "BF16":
                        raise ValueError(f"D3 backbone tensor must retain BF16: {name}")
                    if name not in shapes or tuple(tensor.get_shape()) != shapes[name]:
                        raise ValueError(
                            f"D3 backbone tensor has an unsupported shape: {name}"
                        )
                    seen_text.add(name)
    if actual != declared:
        raise ValueError("D3 weight index does not match the actual shard keys")
    if seen_text != set(shapes):
        raise ValueError("D3 backbone is missing required text tensors")


def expected_backbone_shapes(text):
    """The unconverted HF text schema; the vision backbone is never executed."""
    hidden, intermediate = text["hidden_size"], text["intermediate_size"]
    shapes = {
        "embed_tokens.weight": (text["vocab_size"], hidden),
        "norm.weight": (hidden,),
    }
    key = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    heads, kv_heads, dim = (
        text["num_attention_heads"],
        text["num_key_value_heads"],
        text["head_dim"],
    )
    for index, kind in enumerate(text["layer_types"]):
        layer = {
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
            "mlp.gate_proj.weight": (intermediate, hidden),
            "mlp.up_proj.weight": (intermediate, hidden),
            "mlp.down_proj.weight": (hidden, intermediate),
        }
        if kind == "linear_attention":
            mixer = {
                "in_proj_qkv.weight": (2 * key + value, hidden),
                "in_proj_z.weight": (value, hidden),
                "in_proj_a.weight": (text["linear_num_value_heads"], hidden),
                "in_proj_b.weight": (text["linear_num_value_heads"], hidden),
                "A_log": (text["linear_num_value_heads"],),
                "dt_bias": (text["linear_num_value_heads"],),
                "conv1d.weight": (2 * key + value, 1, text["linear_conv_kernel_dim"]),
                "norm.weight": (text["linear_value_head_dim"],),
                "out_proj.weight": (hidden, value),
            }
            prefix = "linear_attn."
        else:
            mixer = {
                "q_proj.weight": (heads * dim * 2, hidden),
                "k_proj.weight": (kv_heads * dim, hidden),
                "v_proj.weight": (kv_heads * dim, hidden),
                "o_proj.weight": (hidden, heads * dim),
                "q_norm.weight": (dim,),
                "k_norm.weight": (dim,),
            }
            prefix = "self_attn."
        layer.update({prefix + name: shape for name, shape in mixer.items()})
        shapes.update(
            {f"layers.{index}." + name: shape for name, shape in layer.items()}
        )
    return {"language_model." + name: shape for name, shape in shapes.items()}


def prepare_question(question):
    if question["type"] != "score":
        return question, options(question)[0]
    levels = [describe(level) for level in question["criteria"]]
    if (
        not 2 <= len(levels) <= 10
        or any(not level for level in levels)
        or len(set(levels)) != len(levels)
    ):
        raise DecisionRequestError("D3 score needs two to ten distinct nonempty levels")
    return {
        "type": "choice",
        "instructions": question.get("instructions"),
        "criteria": dict.fromkeys(levels),
    }, [str(i) for i in range(len(levels))]


def render_prompt(tokenizer, state, question, codes, *, max_length):
    text = chat_text(tokenizer, messages(state, question, codes))
    tokens = list(tokenizer.encode(text, add_special_tokens=False))
    if not tokens or len(tokens) > max_length:
        raise DecisionInputTooLong(
            f"D3 prompt requires {len(tokens)} tokens; maximum is {max_length}; no truncation"
        )
    return tokens


class D3Engine(CandidateEngine):
    family = "d3"
    source_revision = SOURCE_REVISION
    inspect_artifact = staticmethod(inspect_artifact)

    def _load(self):
        import mlx.core as mx

        from ...runtime.models.import_env import assert_profile_applied
        from ...runtime.ubc_evict import load_shards_evicting
        from ..tokenizer import load_local_tokenizer
        from .d3_model import D3TextModel

        assert_profile_applied("the D3 decision adapter")
        validate_backbone_headers(self.artifact)
        self.tokenizer, self.pretokenizer_receipt = load_local_tokenizer(
            self.artifact["path"]
        )
        decision = self.artifact["decision_config"]
        self.codes, ids = answer_codes(self.tokenizer)
        if self.codes != decision["codes"] or ids != decision["token_ids"]:
            raise ValueError(
                "D3 tokenizer answer codes do not match the trained readout"
            )
        self.temperature = float(decision["temperature"])
        self.model = D3TextModel(self.artifact["text_config"])
        weights = load_shards_evicting(
            self.artifact["shard_files"], sanitize=self.model.sanitize
        )
        if any(value.dtype != mx.bfloat16 for value in weights.values()):
            raise ValueError("D3 loaded backbone tensors must retain BF16")
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        self.head = mx.load(str(self.artifact["readout_path"]))["weight"]
        mx.eval(self.model.parameters(), self.head)
        if not mx.all(mx.isfinite(self.head)).item():
            raise ValueError("D3 readout contains nonfinite weights")
        weights.clear()
        mx.clear_cache()

    def _predict_locked(self, request):
        import mlx.core as mx

        prepared, total = [], 0
        for name, question in request.questions.items():
            rendered, labels = prepare_question(question)
            tokens = render_prompt(
                self.tokenizer,
                request.state,
                rendered,
                self.codes,
                max_length=self.artifact["max_context"],
            )
            total = add_request_tokens(
                total, tokens, max_context=self.artifact["max_context"]
            )
            prepared.append((name, question, labels, tokens))
        rows = []
        self._begin_execution()
        for name, question, labels, tokens in prepared:
            hidden = self.model(mx.array([tokens]))[0, -1].astype(mx.float32)
            logits = (hidden @ self.head[: len(labels)].T) / self.temperature
            probabilities = mx.softmax(logits)
            mx.eval(probabilities)
            rows.append((name, question, labels, probabilities.tolist()))
        return {
            "model": self.model_name,
            "answers": format_answers(rows),
            "usage": usage(total, []),
        }

    def route_receipt(self, *, observed_used):
        return {
            **super().route_receipt(observed_used=observed_used),
            "attention": "noncausal_full_attention_causal_gdn",
            "question_batch_width": 1,
            "context_limit": self.artifact["max_context"],
            "truncation": "refused",
            "readout_dtype": "float32",
            "backbone_dtype": "bfloat16",
            "numerical_law": "d3-hf-raw-norm-fp32-recurrence-v1",
            "permutation_average": False,
        }
