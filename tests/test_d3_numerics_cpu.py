"""Independent hybrid-backbone checks; local artifacts are never downloaded."""

import json
import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.decisions.candidates.d3 import (
    SOURCE_REVISION,
    D3Engine,
    prepare_question,
    render_prompt,
)


@pytest.fixture
def torch_cpu():
    torch = pytest.importorskip("torch")
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


def reference_forward(torch, model, ids):
    """D3 text law: full visibility in attention, causal scan in GDN."""
    assert all(value.device.type == "cpu" for value in model.parameters())
    assert all(value.device.type == "cpu" for value in model.buffers())
    width = ids.shape[1]
    hidden = model.embed_tokens(ids)
    positions = torch.arange(width).view(1, 1, -1).expand(3, 1, -1)
    rope = model.rotary_emb(hidden, positions)
    visible = torch.ones((1, 1, 1, width), dtype=torch.bool)
    with torch.inference_mode():
        for layer, kind in zip(model.layers, model.config.layer_types, strict=True):
            hidden = layer(
                hidden,
                position_embeddings=rope,
                attention_mask=visible if kind == "full_attention" else None,
                position_ids=positions[0],
                use_cache=False,
            )
        return model.norm(hidden)


@pytest.mark.parametrize("head_dim,partial", [(8, 1.0), (256, 0.25)])
def test_nonperiodic_hybrid_backbone_matches_independent_reference(
    torch_cpu, head_dim, partial
):
    torch = torch_cpu
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextConfig,
        Qwen3_5TextModel,
    )

    from mlx2.decisions.candidates.d3_model import D3TextModel

    config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": head_dim,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "layer_types": [
            "linear_attention",
            "full_attention",
            "full_attention",
            "linear_attention",
        ],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": partial,
            "mrope_section": [1, 1, 2],
        },
    }
    torch.manual_seed(13)
    cfg = Qwen3_5TextConfig(**config)
    cfg._attn_implementation = "sdpa"
    reference = Qwen3_5TextModel(cfg).eval()
    model = D3TextModel(config)
    weights = {
        "language_model." + key: mx.array(value.detach().numpy())
        for key, value in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    ids = torch.tensor([[2, 4, 7, 9, 1, 3, 6, 2, 7, 1, 12, 30, 11]])
    expected = reference_forward(torch, reference, ids).numpy()
    actual = np.array(model(mx.array(ids.numpy())))
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
    with pytest.raises(ValueError, match="one independent"):
        model(mx.array([[1, 2], [3, 4]]))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_production_partial_rope_preserves_reference_rounding_and_tail(
    torch_cpu, dtype
):
    torch = torch_cpu
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextConfig,
        Qwen3_5TextRotaryEmbedding,
        apply_rotary_pos_emb,
    )

    cfg = Qwen3_5TextConfig(
        head_dim=256,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
    )
    from mlx2.decisions.candidates.d3_model import D3TextRoPE

    torch.manual_seed(7)
    value = torch.randn(1, 2, 107, 256).to(getattr(torch, dtype))
    positions = torch.arange(107).view(1, 1, -1).expand(3, 1, -1)
    cosine, sine = Qwen3_5TextRotaryEmbedding(cfg)(value, positions)
    expected, _ = apply_rotary_pos_emb(value, value, cosine, sine)
    own = mx.array(value.float().numpy()).astype(getattr(mx, dtype))
    actual = D3TextRoPE(64, 10000000.0)(own)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        expected.float().numpy(),
        rtol=2e-5 if dtype == "float32" else 0,
        atol=2e-5 if dtype == "float32" else 0,
    )
    np.testing.assert_array_equal(
        np.array(actual[..., 64:].astype(mx.float32)), value[..., 64:].float().numpy()
    )


def test_pinned_artifact_probabilities_match_independent_cpu_reference(torch_cpu):
    # The adapter pins import-time numerical flags. Other CPU tensor tests
    # deliberately import those modules under different profiles.
    if os.environ.get("MLX2_D3_ISOLATED_REFERENCE") != "1":
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", __file__, "-k", "pinned_artifact"],
            env={**os.environ, "MLX2_D3_ISOLATED_REFERENCE": "1"},
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    root = Path(
        os.environ.get(
            "MLX2_D3_REFERENCE_PATH",
            str(
                Path.home()
                / ".cache/huggingface/hub/models--vllm-sr--d3-edge/snapshots"
                / SOURCE_REVISION
            ),
        )
    )
    if not (root / "model-00001-of-00001.safetensors").is_file():
        pytest.skip("pinned local D3 artifact absent; no test download")
    from safetensors.torch import load_file
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextConfig,
        Qwen3_5TextModel,
        Qwen3_5TextRotaryEmbedding,
    )

    torch = torch_cpu
    engine = D3Engine(root)
    try:
        cfg = Qwen3_5TextConfig(
            **json.loads((root / "config.json").read_text())["text_config"]
        )
        cfg._attn_implementation = "sdpa"
        with torch.device("meta"):
            reference = Qwen3_5TextModel(cfg)
        weights = load_file(str(root / "model-00001-of-00001.safetensors"))
        reference.load_state_dict(
            {
                key.removeprefix("language_model."): value
                for key, value in weights.items()
                if key.startswith("language_model.")
            },
            strict=True,
            assign=True,
        )
        del weights
        # assign=True loads parameters, not nonpersistent buffers. A meta RoPE
        # buffer otherwise silently invalidates this oracle on some torch builds.
        reference.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg)
        reference.eval()
        head = load_file(str(root / "readout.safetensors"))["weight"]
        state = "The parcel arrived damaged. The customer has a receipt."
        questions = [
            {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "returns": "Damaged deliveries",
                    "billing": "Payment problems",
                },
            },
            {"type": "noul", "instructions": "Does the customer have a receipt?"},
            {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["Routine", "Soon", "Today"],
            },
        ]
        for question in questions:
            prepared, labels = prepare_question(question)
            tokens = render_prompt(
                engine.tokenizer,
                state,
                prepared,
                engine.codes,
                max_length=engine.artifact["max_context"],
            )
            hidden = engine.model(mx.array([tokens]))[0, -1].astype(mx.float32)
            actual = np.array(
                mx.softmax(hidden @ engine.head[: len(labels)].T / engine.temperature)
            )
            hidden_ref = reference_forward(torch, reference, torch.tensor([tokens]))[
                0, -1
            ].float()
            expected = torch.softmax(
                hidden_ref @ head[: len(labels)].T / engine.temperature, dim=-1
            ).numpy()
            # Independent BF16 backends round at different reductions. This is
            # bounded regression evidence, not bitwise parity or qualification.
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0.005)
            assert actual.argmax() == expected.argmax()
        assert engine.route_receipt(observed_used=False)["qualified"] is False
    finally:
        engine.close()
