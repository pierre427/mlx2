"""CPU-safe North Mini Code artifact inspection and serving adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import SamplingDefaults, VendorSampling
from .external_draft_policy import ExternalDraftAdapterMixin

CACHE_LAYOUT = "north-mini-code-layer-segments-v1"
_SAFETENSORS_HEADER_LIMIT = 64 << 20
_DTYPE_BYTES = {"BF16": 2, "U32": 4}

NORTH_MINI_CODE = ModelDescriptor(
    model_type="cohere2_moe",
    family="north-mini-code",
    variant="1.0-ordinary",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            # Target-verified prompt lookup is model-neutral; it rolls stock
            # KVCache and RotatingKVCache planes back exactly.
            Capability.PROMPT_LOOKUP,
            # Constrained decoding works on token text and the tokenizer's EOS
            # ids only; nothing in it is specific to a model family.
            Capability.GRAMMAR,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.north_mini_code.NorthMiniCodeAdapter",
        "qualification": "pending",
        "scope": "text-only",
        "speculation": "none-qualified",
    },
)


def _load_json(path: Path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            value[key] = item
        return value

    return json.loads(path.read_text(), object_pairs_hook=unique)


def _safe_index(path: Path) -> dict:
    value = _load_json(path)
    index = value.get("weight_map") if isinstance(value, dict) else None
    if not isinstance(index, dict) or not index:
        raise ValueError("North artifact must have a nonempty weight index")
    return index


def _quantized_shapes(shape, *, group_size, bits):
    if shape[-1] % group_size or (shape[-1] * bits) % 32:
        raise ValueError("North quantization dimensions are not integral")
    return {
        "weight": [*shape[:-1], shape[-1] * bits // 32],
        "scales": [*shape[:-1], shape[-1] // group_size],
        "biases": [*shape[:-1], shape[-1] // group_size],
    }


def _expected_weight_headers(config):
    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"]) * int(config["head_dim"])
    kv_heads = int(config["num_key_value_heads"]) * int(config["head_dim"])
    experts = int(config["num_experts"])
    intermediate = int(config["intermediate_size"])
    dense_intermediate = int(config["prefix_dense_intermediate_size"])
    quant = config.get("quantization", config.get("quantization_config"))
    if not isinstance(quant, dict):
        raise ValueError("North artifact must declare quantization metadata")
    expected = {}

    def add_quantized(name, shape):
        override = quant.get(name, quant)
        if not isinstance(override, dict):
            raise ValueError(f"Invalid North quantization override: {name}")
        group_size = override.get("group_size")
        bits = override.get("bits")
        if type(group_size) is not int or type(bits) is not int or bits not in {4, 8}:
            raise ValueError(f"Invalid North quantization parameters: {name}")
        for suffix, final_shape in _quantized_shapes(
            shape, group_size=group_size, bits=bits
        ).items():
            expected[f"{name}.{suffix}"] = (
                "U32" if suffix == "weight" else "BF16",
                final_shape,
            )

    add_quantized("model.embed_tokens", [int(config["vocab_size"]), hidden])
    for index in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{index}."
        expected[prefix + "input_layernorm.weight"] = ("BF16", [hidden])
        for projection, shape in (
            ("q_proj", [heads, hidden]),
            ("k_proj", [kv_heads, hidden]),
            ("v_proj", [kv_heads, hidden]),
            ("o_proj", [hidden, heads]),
        ):
            add_quantized(prefix + "self_attn." + projection, shape)
        if index < int(config["first_k_dense_replace"]):
            for projection, shape in (
                ("gate_proj", [dense_intermediate, hidden]),
                ("up_proj", [dense_intermediate, hidden]),
                ("down_proj", [hidden, dense_intermediate]),
            ):
                add_quantized(prefix + "mlp." + projection, shape)
        else:
            add_quantized(prefix + "mlp.gate", [experts, hidden])
            for projection, shape in (
                ("gate_proj", [experts, intermediate, hidden]),
                ("up_proj", [experts, intermediate, hidden]),
                ("down_proj", [experts, hidden, intermediate]),
            ):
                add_quantized(prefix + "mlp.switch_mlp." + projection, shape)
    expected["model.norm.weight"] = ("BF16", [hidden])
    return expected


def _validate_weight_headers(path, index):
    expected = _expected_weight_headers(_load_json(path / "config.json"))
    observed = {}
    header_digests = []
    for name in sorted(set(index.values())):
        shard = (path / name).resolve()
        size = shard.stat().st_size
        with shard.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"Truncated safetensors file: {name}")
            length = struct.unpack("<Q", raw_length)[0]
            if not 0 < length <= min(_SAFETENSORS_HEADER_LIMIT, size - 8):
                raise ValueError(f"Invalid safetensors header length: {name}")
            raw_header = stream.read(length)
        header = json.loads(
            raw_header,
            object_pairs_hook=lambda pairs: _unique_pairs(pairs, name),
        )
        if not isinstance(header, dict):
            raise ValueError(f"Invalid safetensors header object: {name}")
        header_digests.append(hashlib.sha256(raw_header).hexdigest())
        payload_size = size - 8 - length
        ranges = []
        for tensor, record in header.items():
            if tensor == "__metadata__":
                continue
            if tensor in observed:
                raise ValueError(f"Duplicate North tensor across shards: {tensor}")
            if not isinstance(record, dict) or set(record) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValueError(f"Invalid North tensor metadata: {tensor}")
            dtype = record["dtype"]
            shape = record["shape"]
            offsets = record["data_offsets"]
            if dtype not in _DTYPE_BYTES or not isinstance(shape, list) or not all(
                type(value) is int and value >= 0 for value in shape
            ):
                raise ValueError(f"Invalid North tensor dtype/shape: {tensor}")
            if not (
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(type(value) is int for value in offsets)
                and 0 <= offsets[0] <= offsets[1] <= payload_size
            ):
                raise ValueError(f"Invalid North tensor offsets: {tensor}")
            elements = math.prod(shape)
            if offsets[1] - offsets[0] != elements * _DTYPE_BYTES[dtype]:
                raise ValueError(f"Invalid North tensor byte size: {tensor}")
            if index.get(tensor) != name:
                raise ValueError(f"North index/shard mismatch: {tensor}")
            observed[tensor] = (dtype, shape)
            ranges.append((offsets[0], offsets[1], tensor))
        for previous, current in zip(sorted(ranges), sorted(ranges)[1:]):
            if previous[1] > current[0]:
                raise ValueError(
                    f"Overlapping North tensor payloads: {previous[2]}, {current[2]}"
                )
    if set(index) != set(observed):
        raise ValueError("North weight index does not match shard headers")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        wrong = sorted(
            name
            for name in set(expected) & set(observed)
            if expected[name] != observed[name]
        )
        raise ValueError(
            "North checkpoint schema mismatch "
            f"(missing={missing}, extra={extra}, wrong={wrong})"
        )
    return header_digests


def _unique_pairs(pairs, label):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r} in {label}")
        value[key] = item
    return value


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate metadata and shard presence without importing MLX."""
    path = Path(model_path).expanduser().resolve()
    config = _load_json(path / "config.json")
    expected = {
        "model_type": "cohere2_moe",
        "hidden_size": 2048,
        "head_dim": 128,
        "num_hidden_layers": 49,
        "intermediate_size": 768,
        "prefix_dense_intermediate_size": 3072,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 262144,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_shared_experts": 0,
        "first_k_dense_replace": 1,
        "expert_selection_fn": "sigmoid",
        "sliding_window": 4096,
        "rope_theta": 50000,
        "max_position_embeddings": 500000,
        "use_parallel_block": True,
        "use_qk_norm": False,
        "norm_topk_prob": False,
        "tie_word_embeddings": None,
        # Pinned, not decorative: upstream Cohere2Moe selects the norm *class*
        # from rms_norm_eps (present -> RMSNorm, absent -> mean-centred
        # LayerNorm).  Both have the same parameter shapes, so an artifact that
        # dropped this key would load clean and silently mis-normalize.  Fail
        # closed instead.  See runtime/models/cohere2_moe._norm_layer.
        "rms_norm_eps": 1e-06,
        "layer_norm_eps": 1e-05,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match North Mini Code 1.0")
    layer_types = config.get("layer_types")
    required_layers = [
        "full_attention" if i % 4 == 0 else "sliding_attention" for i in range(49)
    ]
    if layer_types != required_layers:
        raise ValueError("artifact layer order does not match North Mini Code 1.0")
    if config.get("architectures") != ["Cohere2MoeForCausalLM"]:
        raise ValueError("artifact architecture does not match North Mini Code 1.0")
    index = _safe_index(path / "model.safetensors.index.json")
    names = sorted(set(index.values()))
    records = []
    for name in names:
        if not isinstance(name, str):
            raise TypeError("weight shard path must be a string")
        item = (path / name).resolve()
        if (
            not item.is_relative_to(path)
            or item.suffix != ".safetensors"
            or not item.is_file()
        ):
            raise ValueError("weight index must reference local safetensors files")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    if any(
        marker in key.lower() for key in index for marker in ("mtp.", "eagle", "draft")
    ):
        raise ValueError("embedded speculative tensors are not a North target artifact")
    if "model.embed_tokens.weight" not in index or "lm_head.weight" in index:
        raise ValueError("North artifact must use its tied embedding output head")
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    header_digests = _validate_weight_headers(path, index)
    for record in records:
        digest.update(json.dumps(record).encode())
    digest.update(json.dumps(header_digests).encode())
    return {
        "config": config,
        "weight_map": index,
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
            "header_sha256": header_digests,
        },
        "layers": 49,
        "sliding_layers": 36,
        "global_layers": 13,
        "sliding_window": 4096,
        "supports_native_mtp": False,
        "qualification": "pending",
    }


def configure_environment() -> dict[str, str]:
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "0",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "0",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


def normalize_messages(messages: list[dict]) -> list[dict]:
    messages = copy.deepcopy(messages)
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("North mlx2 port accepts text content only")
            message["content"] = "".join(part["text"] for part in content)
        for call in message.get("tool_calls", []):
            arguments = call["function"].get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("North tool arguments must be a JSON object")
            call["function"]["arguments"] = arguments
    return messages


def reasoning_policy(request: dict) -> tuple[str, bool]:
    """Resolve North reasoning controls once for prompting, parsing, and receipts."""
    effort = request.get("reasoning_effort")
    if effort is None:
        # Thinking is on unless the client turns it off, as in the vendor chat
        # template.  North is trained to reason first: with the thinking block
        # forced empty it either reasons inside the answer or answers without
        # working (GPU, 2026-09-18: 62+130 -> 292 and "Paris is coastal" off,
        # both correct on, for ~50 extra tokens).  Structured output follows
        # the same default: the grammar is deferred past <|END_THINKING|>.
        effort = "none" if request.get("enable_thinking") is False else "high"
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}
    if effort not in allowed:
        raise ValueError("Unsupported North reasoning_effort")
    thinking = bool(request.get("enable_thinking", effort != "none"))
    return effort, thinking


def chat_template(tokenizer, request: dict, *, tokenize: bool):
    """Render a chat request with North's reasoning controls (ids or text)."""
    effort, thinking = reasoning_policy(request)
    return tokenizer.apply_chat_template(
        normalize_messages(request["messages"]),
        add_generation_prompt=True,
        tokenize=tokenize,
        reasoning=thinking,
        reasoning_effort=effort,
        skip_thinking=not thinking,
        tools=request.get("tools") if request.get("tool_choice") != "none" else None,
    )


class NorthActionProcessor:
    """Force North's action branch once its optional thinking block closes.

    The mask is derived only from the committed token history.  Replaying a
    prompt-lookup or speculative probe row therefore produces the same result
    without advancing request-local state.
    """

    history_pure = True  # P5

    def __init__(
        self,
        *,
        prompt_length: int,
        action_open,
        thinking_close=None,
    ):
        self.prompt_length = int(prompt_length)
        self.action_open = tuple(int(token) for token in action_open)
        self.thinking_close = (
            tuple(int(token) for token in thinking_close)
            if thinking_close is not None
            else None
        )
        if (
            self.prompt_length < 0
            or not self.action_open
            or self.thinking_close == ()
        ):
            raise ValueError("North action constraints require nonempty markers")

    def __call__(self, tokens, logits):
        import mlx.core as mx

        generated = tokens[self.prompt_length :]
        length = generated.shape[0]
        vocabulary = mx.arange(logits.shape[-1])
        if self.thinking_close is None:
            compared = min(length, len(self.action_open))
            matches = (
                mx.array(True)
                if compared == 0
                else mx.all(
                    generated[:compared]
                    == mx.array(self.action_open[:compared], dtype=tokens.dtype)
                )
            )
            if length >= len(self.action_open):
                allowed = matches
            else:
                allowed = mx.logical_and(
                    matches, vocabulary == self.action_open[length]
                )
        else:
            # The decision boundary is a suffix until ACTION_OPEN completes.
            # Looking only at these bounded suffixes avoids rescanning an
            # arbitrarily long reasoning history on every generated token.
            active = mx.array(False)
            constrained = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
            for offset in range(len(self.action_open)):
                suffix = (*self.thinking_close, *self.action_open[:offset])
                if length < len(suffix):
                    continue
                matches = mx.all(
                    generated[-len(suffix) :]
                    == mx.array(suffix, dtype=tokens.dtype)
                )
                active = mx.logical_or(active, matches)
                constrained = mx.logical_or(
                    constrained,
                    mx.logical_and(
                        matches, vocabulary == self.action_open[offset]
                    ),
                )
            allowed = mx.logical_or(constrained, ~active)
        return mx.where(
            allowed, logits, mx.array(float("-inf"), dtype=logits.dtype)
        )

    def probe(self, tokens, logits):
        """Speculative probes use the same pure history-derived mask."""
        return self(tokens, logits)


# Vendor sampling defaults.  CohereLabs/North-Mini-Code-1.0 model card:
# temperature 1.0, top_p 0.95 for all generation, code and agentic tool use
# alike, with no thinking-specific or task-specific profile (read 2026-09-18).
# The artifact's generation_config.json carries no sampling fields.
NORTH_MINI_CODE_SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0, top_p=0.95,
        source="model card (CohereLabs/North-Mini-Code-1.0)",
    ),
    model="CohereLabs/North-Mini-Code-1.0",
)


class NorthMiniCodeAdapter(ExternalDraftAdapterMixin):
    default_route = "ordinary"
    descriptor = NORTH_MINI_CODE
    sampling_defaults = NORTH_MINI_CODE_SAMPLING
    reasoning_effort_semantics = "thinking_toggle"

    @staticmethod
    def spomin_backend(model, prompt_cache):
        """Adapter-owned approximate KV surgery for full + sliding attention."""
        from ..runtime.spomin_standard_surgery import StandardAttentionSpominBackend

        return StandardAttentionSpominBackend(model, prompt_cache)

    @staticmethod
    def thinking_enabled(request: dict) -> bool:
        return reasoning_policy(request)[1]

    def _single_token(self, marker):
        try:
            ids = list(self.tokenizer.encode(marker, add_special_tokens=False))
        except Exception:  # noqa: BLE001 - undeclared marker, not a load failure
            return None
        return int(ids[0]) if len(ids) == 1 else None

    def thinking_close_token_ids(self):
        """<|END_THINKING|> ends the reasoning channel; grammars defer to it."""
        from .north_output import THINK_CLOSE

        token = self._single_token(THINK_CLOSE)
        return (token,) if token is not None else None

    # ------------------------------------------------------------------
    # Run-on reasoning: what to use, and why
    #
    # North is trained to reason before it answers, and it answers best with
    # thinking on (the default here, as in the vendor template).  Its failure
    # mode is *run-on*: on a vague or second-guessable prompt it circles a
    # formed answer ("Budapest? Actually ... Budapest?") and never emits
    # <|END_THINKING|>, burning the whole token budget and returning no
    # content.  The lab's termination work (wiki: reasoning-termination-margin,
    # termination-attractor-steering, puzzle-juice-budget) shows this is a
    # stop-decision failure, not a capability one, so the fix is to help the
    # model *close*, not to cut it off.  Three levers, all default-off, all
    # served through `mlx2.thinking_guard.ThinkingGuard`:
    #
    #   1. JUICE budget      --thinking-budget N   (or request `thinking_budget`)
    #      N is the medium-effort reasoning budget; `reasoning_effort` scales it
    #      (low 0.375x, high 4x, capped at 8192).  Release starts at 80% of it;
    #      the channel is force-closed only at the budget itself.
    #   2. tau run-on alarm  (always armed when a guard exists)
    #      A content-blind CUSUM over recurring 6-grams of the reasoning tokens.
    #      It catches a loop within ~100-300 tokens, long before any budget.
    #      On alarm (or soft budget) a bias on <|END_THINKING|> ramps +2 nats a
    #      step; North then closes by itself within ~11 tokens.
    #   3. alpha steering    --thinking-steer-alpha A  (or request
    #      `thinking_steer_alpha`), optionally --thinking-steer-hammer H
    #      While a lane is reasoning, A * (rms_L * v_hat) is added to the
    #      residual stream after decoder layer L, where v_hat is the calibrated
    #      *commit direction* below.  It is always-on during reasoning, so it
    #      shortens every trace, not just the run-on ones.
    #
    # What the 2026-09-18 campaign measured on this 4-bit artifact (greedy, 16
    # held-out prompts; qualification/runs/north-alpha-calibration-20260918):
    #   off                      13/16 answered, 9,187 reasoning tokens
    #   alpha 0.2 at layer 28    16/16 correct,  1,531 tokens (-83%)
    #   alpha 0.4 at layer 32    16/16 correct,  1,362 tokens
    #   same-norm RANDOM vector  11-12/16  (worse than off: the effect is the
    #                            direction, not the perturbation)
    #   guard only (levers 1+2)  13/13 correct, 63-74% fewer tokens, no hard close
    # Through the server at 20 lanes (20x20 sanity, thinking on): unguarded
    # 394/400 with 6 empty answers and 81.7K tokens; guard only 399/400, 66.7K;
    # guard + alpha 0.2 400/400, 45.9K, on both the ordinary and the
    # prompt-lookup route.  A 16-problem multi-step set stayed 16/16 at alpha
    # 0.2 and 0.4 - but North needed only ~240 reasoning tokens on those, so
    # that says alpha does not break ordinary multi-step work, no more.
    #
    # DEFAULTS.  All three levers are ON by default for this model
    # (`thinking_guard_defaults` below: budget anchor 512, alpha 0.2, no
    # hammer) - Pierre's call on 2026-09-18 after the campaign: 400/400 on the
    # 20x20 run with 44% fewer tokens and no measured accuracy cost.  An
    # operator flag overrides a default, and an explicit 0 turns that lever off
    # (`--thinking-steer-alpha 0`, `--thinking-budget 0`); a request can do the
    # same for itself with `thinking_steer_alpha: 0` / `thinking_budget: 0`.
    # `/v1/status.settings.thinking_defaults_source` says whether the adapter
    # or the operator chose the values in force.
    #
    # Guidance.  Start with `--thinking-budget 512`: levers 1+2 act only on a
    # detected run-on and leave healthy reasoning untouched, and they work on
    # every route.  Add `--thinking-steer-alpha 0.2` when reasoning tokens are
    # the cost that matters; it applies to the ordinary decode step and to the
    # verify forwards of the prompt-lookup and external-draft routes (the
    # external EAGLE route since rm06), pulls *all* reasoning shorter, and is unproven on problems
    # that need long deliberate work - the lab saw an over-strong hammer (0.8
    # from token 140) drop a material guard from a Qwen3.8 code fix.  Keep
    # alpha <= 0.4, and re-run the calibration script if the weights,
    # quantization or chat template change - or if the MODEL BODY changes, which
    # the artifact hash cannot see.  On 2026-09-20 the norm correction changed
    # the residual geometry without touching a single weight byte; the direction
    # had to be recalibrated and `thinking_calibration.SCHEMA` is what forced it.
    # The direction is a property of this
    # exact artifact - which is why the server binds every direction to an
    # artifact identity and recalibrates by itself rather than reuse this one on
    # another build (see `commit_direction_assets`).  Steered lanes are not
    # stored as exact APCv2 prefixes
    # (their K/V was written under steering); the prompt boundary still is.
    #
    # Recalibrate with:
    #   scripts/calibrate_thinking_direction.py trace   --model <path> --out traces.jsonl
    #   scripts/calibrate_thinking_direction.py extract --model <path> --traces traces.jsonl --out dir.npz
    #   scripts/calibrate_thinking_direction.py grid    --model <path> --vectors dir.npz --arms '[...]' --out grid.json
    # and accept a layer only if its cross-trace consistency is clearly positive
    # AND the same-schedule random control does not reproduce the effect.
    # ------------------------------------------------------------------
    THINKING_GUARD_DEFAULTS = {
        "thinking_budget": 512,          # medium-effort anchor; unspecified effort is "high" = 2048
        "thinking_steer_alpha": 0.2,     # the campaign's operating point at layer 28
        "thinking_steer_hammer": 0.0,    # the two-mode hammer never fired in the grid; leave off
    }

    def thinking_guard_defaults(self):
        """Run-on reasoning defaults the server applies when the operator sets none.

        The steering default is a *wish*, not a guarantee: the server only
        steers with a direction bound to the exact artifact it loaded (see
        `commit_direction_assets`).  For any other North artifact it calibrates
        one at startup and, if that does not pass its gates, serves with the
        budget/alarm guard only.
        """
        return dict(self.THINKING_GUARD_DEFAULTS)

    COMMIT_DIRECTION_ASSET = "north_mini_code_commit_direction.npz"
    # L32 on the corrected RMSNorm body (2026-09-20 recalibration): cross-trace
    # consistency peaks there (+0.518, against +0.511 at L28), and L32 is what
    # `choose_layer` selects.  The pre-fix asset's L28 was measured in the
    # LayerNorm residual geometry and is void.  See
    # `qualification/runs/north-requal-20260920/`.
    COMMIT_DIRECTION_LAYER = 32

    def commit_direction_assets(self):
        """Shipped calibrations the server may use IF their identity matches.

        Each `.npz` has a sibling `.json` naming the `artifact_identity` it was
        measured on (`mlx2.thinking_calibration.artifact_identity`: config,
        tokenizer, chat template and sampled weight bytes).  The shipped file is
        bound to `North-Mini-Code-1.0-mlx-4bit`; the 8-bit build, a re-quant or
        a fine-tune hashes differently and will NOT pick it up - a wrong
        direction is a same-norm random vector, which the 2026-09-20 held-out
        grid measured as no better than not steering at all (7,169 reasoning
        tokens against 7,683 unsteered, versus 2,404 calibrated).  For those
        the server runs the calibration
        itself at startup (about two minutes: 24 natural traces, a positional
        direction per layer, then a held-out check against no steering and a
        random control) and stores the result under
        `<cache-dir>/commit-directions/<identity>.npz`, so it happens once.
        If a gate fails, steering stays off; an operator who passed
        `--thinking-steer-alpha` explicitly gets a startup error instead of a
        server that silently does not steer.  `--no-thinking-auto-calibration`
        skips the attempt, and `/v1/status.settings.thinking_steer.calibration`
        reports what happened.
        """
        from pathlib import Path

        return {"paths": [Path(__file__).with_name("assets") / self.COMMIT_DIRECTION_ASSET],
                "layer": self.COMMIT_DIRECTION_LAYER}

    def structured_envelope_token_ids(self):
        """North frames an answer as <|START_TEXT|> ... <|END_TEXT|>.

        The markers are ordinary vocabulary entries, not special tokens, so a
        grammar would otherwise forbid the framing the model was trained on.
        """
        from .north_output import TEXT_CLOSE, TEXT_OPEN

        opener, closer = self._single_token(TEXT_OPEN), self._single_token(TEXT_CLOSE)
        if opener is None or closer is None:
            return None
        return (opener,), (closer,)

    @staticmethod
    def profile_name(mtp):
        if mtp:
            raise ValueError("North Mini Code has no qualified native MTP route")
        return "north-mini-code-apcv2-ordinary"

    # The vendor card serves the EAGLE head with 3 speculative tokens; a chain
    # has no trained block, so EXTERNAL_MAX_BLOCK caps proposals per round.
    EXTERNAL_DEFAULT_NUM_DRAFT = 3
    EXTERNAL_MAX_BLOCK = 8
    EXTERNAL_ROUTE_TAG = "external-cohere-eagle-v1"
    EXTERNAL_PROFILE = "north-mini-code-apcv2-cohere-eagle"

    def execution_config(self, *, max_lanes, prefill_step):
        if getattr(self, "draft_model", None) is not None:
            return self._external_execution_config(max_lanes=max_lanes, prefill_step=prefill_step)
        return {
            "persistent": True,
            "num_draft": 0,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    def cache_budget(self, *, mtp):
        from .north_memory import NorthCacheBudget

        return NorthCacheBudget.from_config(self.config, mtp=mtp)

    def __init__(self, model_path: str, *, execution_policy=None):
        external = self._parse_external_policy(execution_policy, family="North")
        artifact = inspect_artifact(model_path)
        draft_record = None
        if external:
            # Header-only drafter inspection before any target tensor loads.
            from .cohere_eagle import inspect_drafter

            draft_record = inspect_drafter(
                self.external_policy["draft_model"],
                target_config=artifact["config"],
                block_size=self.EXTERNAL_MAX_BLOCK,
            )
            self._check_num_draft(draft_record)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.environment = configure_environment()
        self.layout = CACHE_LAYOUT
        path = Path(self.identity["path"])
        config = artifact["config"]
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.cohere2_moe import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = config.get("quantization", config.get("quantization_config"))
        if quant:
            def predicate(name, module):
                override = quant.get(name)
                if isinstance(override, dict):
                    return override
                return hasattr(module, "to_quantized") and f"{name}.scales" in weights

            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant.get("mode", "affine"),
                class_predicate=predicate,
            )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        eos = config.get("eos_token_id")
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[eos] if isinstance(eos, int) else list(eos or []),
        )
        self.max_context = int(config["max_position_embeddings"])
        if draft_record is not None:
            from .cohere_eagle import load_drafter

            self._bind_external_drafter(draft_record, load_drafter, NORTH_MINI_CODE)

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        return chat_template(self.tokenizer, request, tokenize=True)

    def render_prompt(self, request: dict) -> str:
        """Prompt text whose special-token-free encoding is ``prompt_tokens``."""
        if "messages" not in request:
            return request["prompt"]
        return chat_template(self.tokenizer, request, tokenize=False)

    def output_parser(self, request):
        from .north_output import NorthOutputParser

        _, thinking = reasoning_policy(request)
        return NorthOutputParser(
            chat="messages" in request,
            thinking=thinking,
            tools=request.get("tools") if request.get("tool_choice") != "none" else None,
            stops=request.get("stop", ()),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    def request_logits_processors(self, request, *, prompt_length):
        """Constrain required/named tool requests to North's action branch."""
        if "messages" not in request:
            return ()
        choice = request.get("tool_choice", "auto")
        if choice != "required" and not isinstance(choice, dict):
            return ()
        if not request.get("tools"):
            return ()
        from .north_output import ACTION_OPEN, THINK_CLOSE

        _, thinking = reasoning_policy(request)
        action_open = self.tokenizer.encode(
            ACTION_OPEN, add_special_tokens=False
        )
        thinking_close = (
            self.tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
            if thinking
            else None
        )
        return (
            NorthActionProcessor(
                prompt_length=prompt_length,
                action_open=action_open,
                thinking_close=thinking_close,
            ),
        )

    # Item 12: opener free text must avoid for the ``auto`` tool grammar.
    tool_call_open_marker = "<|START_ACTION|>"

    def tool_constraint(self, request):
        from ..output import constrained_tool_choice
        from .north_output import constrained_tool_grammar

        if not constrained_tool_choice(request):
            return None
        return constrained_tool_grammar(
            request["tools"],
            request["tool_choice"],
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    def diagnostics(self):
        return {
            "architecture": "cohere2_moe",
            "cache_layout": self.layout,
            "sliding_layers": 36,
            "global_layers": 13,
            "sliding_window": 4096,
            "speculation": (
                "external-cohere-eagle-implemented-unqualified"
                if getattr(self, "draft_model", None) is not None
                else "none-qualified"
            ),
        }

    def close(self):
        """Release model ownership before the shared worker shuts down."""

        had_resources = any(
            getattr(self, name, None) is not None for name in ("model", "tokenizer")
        )
        self.model = None
        self.draft_model = None
        self.tokenizer = None
        if had_resources:
            import mlx.core as mx

            mx.clear_cache()
