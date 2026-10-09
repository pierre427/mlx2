"""Qwen3.5 9B/4B status must carry the receipts of inherited 27B mechanisms.

Both adapters inherit Qwen3827BAdapter's constructor, which accepts and
installs ``fused_gdn`` and ``gdn_state_dtype=float16`` (among others).  The
9B ``diagnostics()`` override returned a fixed dict, so status["execution"]
had no ``fused_gdn`` / ``gdn_state`` section: the fp16 state's "unqualified"
label vanished, and the qualifier (which requires feature_qwen38_fused_gdn /
feature_gdn_state_fp16 for these selections) could never observe them.
"""
import importlib

import mlx.core as mx
import pytest

TEXT = {
    "model_type": "qwen3_5_text", "hidden_size": 64, "intermediate_size": 64,
    "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 32, "vocab_size": 64, "linear_num_key_heads": 2,
    "linear_num_value_heads": 4, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4, "full_attention_interval": 4, "rms_norm_eps": 1e-6,
}
ADAPTERS = [("qwen35_9b", "Qwen359BAdapter"), ("qwen35_4b", "Qwen354BAdapter")]
# The family's receipt with nothing selected (unchanged by the fix).
DEFAULT_KEYS = {
    "architecture", "dtype_normalized", "layout", "load_dtype_check",
    "mtp_head_present", "scope",
}


def _adapter(module, name, *, selected):
    from mlx2.runtime.models import qwen38_fused_gdn as route
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

    cls = getattr(importlib.import_module(f"mlx2.adapters.{module}"), name)
    adapter = object.__new__(cls)
    adapter.descriptor = cls.descriptor
    adapter.layout = cls.descriptor.cache_layout
    adapter.model = Model(
        ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": TEXT})
    )
    adapter.external_policy = {}
    adapter.draft_model = None
    adapter.fused_gdn = selected
    adapter.fused_gdn_prefill = False
    adapter.environment = {"MLX2_QWEN38_FUSED_GDN": "1"} if selected else {}
    if selected:
        # What _init_qwen38/_finish_load leave for
        # {"fused_gdn": true, "gdn_state_dtype": "float16"}.
        route.configure(
            adapter.model, True, architecture=cls.fused_gdn_architecture, prefill=False
        )
        adapter._select_gdn_state("float16")
    return adapter


@pytest.mark.parametrize("module,name", ADAPTERS)
def test_dense_qwen35_status_reports_selected_gdn_mechanisms(module, name):
    from mlx2.qualification import required_feature_checks
    from mlx2.runtime.models import gdn_state
    from mlx2.serving import _execution_diagnostics

    adapter = _adapter(module, name, selected=True)
    before = dict(gdn_state.STATS)
    cache = adapter.model.language_model.make_cache()
    mx.eval(adapter.model(mx.array([[1, 2, 3, 4, 5]]), cache=cache))
    assert sum(gdn_state.STATS.values()) > sum(before.values())  # fp16 path ran

    execution = _execution_diagnostics(adapter)
    settings = {
        "environment": adapter.environment,
        "cache_budget": {
            "recurrent_state_bytes": adapter.cache_budget(mtp=False).recurrent_state_bytes
        },
    }
    required = required_feature_checks(settings)
    assert {"feature_gdn_state_fp16", "feature_qwen38_fused_gdn"} <= required

    # The qualifier reads these sections; they must exist and be truthful.
    assert execution["gdn_state"]["state_dtype"] == "float16"
    assert execution["gdn_state"]["qualification"] == "unqualified"
    assert execution["gdn_state"]["layers"] == 3
    assert sum(execution["gdn_state"]["counters"].values()) > 0
    assert execution["fused_gdn"]["architecture"] == "qwen35"
    assert execution["fused_gdn"]["enabled"] is True
    # The family's own fields stay as they were.
    assert execution["scope"] == "text-only"
    assert execution["mtp_head_present"] is False
    assert execution["layout"].endswith(":gdn-state-fp16-v1")
    assert not {"speculation", "segmented_mtp"} & set(execution)


@pytest.mark.parametrize("module,name", ADAPTERS)
def test_dense_qwen35_default_status_is_unchanged(module, name):
    adapter = _adapter(module, name, selected=False)
    assert set(adapter.diagnostics()) == DEFAULT_KEYS
