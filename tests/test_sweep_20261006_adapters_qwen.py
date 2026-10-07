"""Regression tests from the 2026-10-06 sweep: Qwen-family adapter defaults."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODELS = Path.home() / "mlx-models"


def _qwen35_text(**extra):
    return {
        "model_type": "qwen3_5_text", "hidden_size": 64, "num_hidden_layers": 4,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
        **extra,
    }


def test_qwen35_flat_rope_config_keeps_its_theta():
    """A config without rope_parameters keeps its flat rotary geometry
    instead of a hard-coded theta 100000 / factor 0.25."""
    from mlx2.runtime.models.qwen3_5 import TextModelArgs

    args = TextModelArgs.from_dict(
        _qwen35_text(rope_theta=10_000_000, partial_rotary_factor=0.5)
    )
    assert (args.rope_theta, args.partial_rotary_factor) == (10_000_000, 0.5)
    assert args.rope_scaling["type"] == "default"


def test_qwen35_rope_parameters_still_win_over_flat_fields():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs

    args = TextModelArgs.from_dict(_qwen35_text(
        rope_theta=1.0,
        rope_parameters={
            "rope_type": "default", "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
        },
    ))
    assert (args.rope_theta, args.partial_rotary_factor) == (10_000_000, 0.25)
    assert args.rope_scaling["type"] == "default"


QWEN3_4B = MODELS / "Qwen3-4B-bf16"


def _qwen3_artifact(tmp_path, template):
    from mlx2.adapters.standard_decoder import artifact_vendor_sampling

    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}))
    from mlx2.adapters.standard_decoder import _hybrid_thinking_template

    return artifact_vendor_sampling({
        "config": {"model_type": "qwen3"},
        "sampling_defaults": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "hybrid_thinking": _hybrid_thinking_template(tmp_path),
    })


def test_qwen3_hybrid_chat_samples_the_card_non_thinking_profile(tmp_path):
    """The standard decoder renders every chat non-thinking, so a hybrid
    Qwen3 must not sample at generation_config's thinking-mode values
    (Qwen3 card: non-thinking T=0.7, top_p 0.8, top_k 20, min_p 0)."""
    from mlx2.sampling_defaults import resolve_sampling

    vendor = _qwen3_artifact(
        tmp_path, "{% if enable_thinking is defined and enable_thinking is false %}x{% endif %}"
    )
    chat, record = resolve_sampling({}, vendor, thinking=False)
    assert (chat["temperature"], chat["top_p"], chat["top_k"], chat["min_p"]) == (0.7, 0.8, 20, 0.0)
    assert record["profile"] == "non_thinking"
    raw, _ = resolve_sampling({}, vendor, thinking=None)
    assert (raw["temperature"], raw["top_p"]) == (0.6, 0.95)


def test_qwen3_single_mode_template_keeps_generation_config(tmp_path):
    from mlx2.sampling_defaults import resolve_sampling

    vendor = _qwen3_artifact(tmp_path, "{{ messages }}")
    chat, _ = resolve_sampling({}, vendor, thinking=False)
    assert (chat["temperature"], chat["top_p"]) == (0.6, 0.95)


@pytest.mark.skipif(not QWEN3_4B.is_dir(), reason="Qwen3-4B artifact absent")
def test_local_qwen3_4b_is_detected_as_hybrid_thinking():
    from mlx2.adapters.standard_decoder import inspect_artifact

    assert inspect_artifact(QWEN3_4B)["hybrid_thinking"] is True


def test_flash_next_does_not_declare_undeliverable_compaction():
    """The only live compaction path declines hybrid (recurrent) models, and
    every Flash-Next artifact has GDN layers."""
    from mlx2.adapters.qwen import QWEN4_FLASH_NEXT
    from mlx2.contracts import Capability
    from mlx2.runtime.spomin_live_surgery import SpominLiveSurgeryManager

    manager = SpominLiveSurgeryManager(enabled=True, backend_factory=lambda m, c: None)
    declined = manager.prepare(
        request_id="r", prompt_token_ids=(1, 2, 3),
        transcript=SimpleNamespace(token_ids=(1, 2, 3)), capacity_tokens=2,
        strategy="oldest_first", has_mtp_state=False, has_recurrent_state=True,
        cache_is_request_private=True,
    )
    assert declined is None
    assert Capability.COMPACTION not in QWEN4_FLASH_NEXT.capabilities


def test_qwen35_122b_does_not_inherit_35b_identity_or_tuning():
    from mlx2.adapters.qwen35_122b import CACHE_LAYOUT, Qwen35122BA10BAdapter as C

    adapter = object.__new__(C)
    adapter._kernels, adapter.moe_nax_gather = {}, "off"
    contract = adapter.execution_numerics_contract()
    assert "qwen36_target" not in contract
    assert contract["qwen35_122b_target"]["cache_layout"] == CACHE_LAYOUT
    assert adapter.prefill_step_default() is None
    assert C.default_eager_dispatch_stride == 0


def _dense_text(hidden, intermediate):
    return {
        "hidden_size": hidden, "intermediate_size": intermediate,
        "num_hidden_layers": 32, "full_attention_interval": 4, "num_key_value_heads": 4,
        "head_dim": 256, "linear_num_value_heads": 32, "linear_num_key_heads": 16,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "mtp_num_hidden_layers": 1,
    }


@pytest.mark.parametrize(
    "module,cls,hidden,intermediate,transient,chunk",
    [("qwen35_9b", "Qwen359BAdapter", 4096, 12288, 2.48, 1.6),
     ("qwen35_4b", "Qwen354BAdapter", 2560, 9216, 1.64, 1.06)],
)
def test_qwen35_dense_cache_budget_uses_its_own_geometry(
    module, cls, hidden, intermediate, transient, chunk
):
    """The 9B/4B budgets dropped the selected GDN state dtype (fp16 charged
    and receipted at fp32) and copied the 27B's measured workspace."""
    import importlib

    adapter_type = getattr(importlib.import_module(f"mlx2.adapters.{module}"), cls)
    adapter = object.__new__(adapter_type)
    adapter.model = SimpleNamespace(
        args=SimpleNamespace(text_config=_dense_text(hidden, intermediate))
    )
    adapter.gdn_state = {"state_dtype": "float16"}
    budget = adapter.cache_budget(mtp=False)
    assert budget.recurrent_state_bytes == 2
    assert (budget.transient_gib_per_lane, budget.prefill_chunk_transient_gib) == (
        transient, chunk
    )
    receipt = budget.as_dict()
    assert receipt["recurrent_state_bytes"] == 2
    assert receipt["workspace_basis"].endswith("unmeasured")
    adapter.gdn_state = None
    assert adapter.cache_budget(mtp=False).recurrent_state_bytes == 4
