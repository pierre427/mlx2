"""A DFlash drafter's sliding window must be declared where mlx2 reads it.

TensorRT-LLM #21367: some DFlash2 drafters carry their window only in
``dflash_config.use_swa`` (plus a window size) and type every layer
``full_attention``.  mlx2 decides sliding from ``layer_types`` alone, so such
a drafter would be served unwindowed.  No drafter on disk has that shape
(2026-10-31 inventory), so the parsers refuse it instead of guessing which
layers the window applies to, and refuse a ``use_swa`` or window size that
contradicts the layer types.  The base DFlash parser also refuses a nested
logit transform it does not apply.
"""

import copy
import glob
import json
from pathlib import Path

import pytest

from mlx2.runtime.drafters.base_config import DFlashConfig
from mlx2.runtime.drafters.dflash2_config import DFlash2Config

ON_DISK = sorted(
    glob.glob(str(Path.home() / "mlx-models/*DFlash*/config.json"))
    + glob.glob(str(Path.home() / ".cache/huggingface/hub/models--*DFlash*/snapshots/*/config.json"))
)


def _base(layer_types, sliding_window=None, **dflash):
    return {
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": len(layer_types),
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "vocab_size": 64, "num_target_layers": 4, "block_size": 4,
        "layer_types": list(layer_types), "sliding_window": sliding_window,
        "dflash_config": {"mask_token_id": 63, "target_layer_ids": [0, 3], **dflash},
    }


def _dflash2(layer_types, sliding_window=None, **dflash):
    config = _base(layer_types, sliding_window, **dflash)
    config["dflash_config"].update(conv_kernel_size=2, conv_group_size=2, selector_rank=4, selector_top_k=4)
    config["max_position_embeddings"] = 256
    return config


PARSERS = [(DFlashConfig, _base), (DFlash2Config, _dflash2)]


@pytest.mark.parametrize("parser,make", PARSERS)
def test_window_only_in_dflash_config_is_refused(parser, make):
    config = make(["full_attention"] * 2, None, use_swa=True, swa_window_size=1024)
    with pytest.raises(ValueError, match="use_swa"):
        parser.from_dict(config)


@pytest.mark.parametrize("parser,make", PARSERS)
def test_use_swa_false_with_sliding_layers_is_refused(parser, make):
    config = make(["sliding_attention"] * 2, 8, use_swa=False)
    with pytest.raises(ValueError, match="use_swa"):
        parser.from_dict(config)


@pytest.mark.parametrize("parser,make", PARSERS)
def test_conflicting_window_sizes_are_refused(parser, make):
    config = make(["sliding_attention"] * 2, 8, use_swa=True, swa_window_size=16)
    with pytest.raises(ValueError, match="window"):
        parser.from_dict(config)


@pytest.mark.parametrize("parser,make", PARSERS)
def test_consistent_declarations_still_parse(parser, make):
    parsed = parser.from_dict(make(["sliding_attention"] * 2, 8, use_swa=True, swa_window_size=8))
    assert parsed.layer_types == ["sliding_attention"] * 2 and parsed.sliding_window == 8
    parsed = parser.from_dict(make(["full_attention"] * 2, None, use_swa=False))
    assert parsed.layer_types == ["full_attention"] * 2
    parsed = parser.from_dict(make(["full_attention", "sliding_attention"], 8))
    assert parsed.sliding_window == 8


@pytest.mark.parametrize("key,value", [("output_multiplier", 5.7), ("final_logit_softcapping", 30.0)])
def test_base_parser_refuses_a_nested_logit_transform_it_does_not_apply(key, value):
    with pytest.raises(ValueError, match=key):
        DFlashConfig.from_dict(_base(["full_attention"] * 2, None, **{key: value}))
    # The identity transform (GLM-5.3-Flash-DFlash declares output_multiplier 1.0) is fine.
    identity = 1.0 if key == "output_multiplier" else None
    DFlashConfig.from_dict(_base(["full_attention"] * 2, None, **{key: identity}))


@pytest.mark.parametrize("path", ON_DISK or [None])
def test_on_disk_drafter_windows_are_consistent(path):
    if path is None:
        pytest.skip("no DFlash drafter on this host")
    config = json.loads(Path(path).read_text())
    nested = config.get("dflash_config") or {}
    original = copy.deepcopy(config)
    from mlx2.runtime.drafters.base_config import check_dflash_window_contract
    check_dflash_window_contract(config, nested)
    assert config == original
