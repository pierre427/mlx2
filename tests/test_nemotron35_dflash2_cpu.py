"""Community DFlash2 draft on the Nemotron 3.5 Super VL target (opt-in).

The binder pins the artifact and fails closed on architecture, tap or target
mismatches; the generic DFlash2 drafter then proposes for a Nemotron-H hybrid
(Mamba + attention + MoE) whose exact verify must leave ordinary greedy output.
"""

import copy
import json
import os
import shutil
from contextlib import closing
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.adapters import nemotron35_dflash2 as binder
from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.drafters.dflash2_config import DFlash2Config
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.nemotron_tree_executor import NemotronTreeExecutor
from tests.test_nemotron_tree_cpu import CAPTURES, _model, _prefilled

MODELS = Path.home() / "mlx-models"
TARGET = MODELS / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
DRAFTS = {
    "main-step-75000": MODELS / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-DFlash2",
    "step-51000": MODELS / "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-DFlash2-step51000",
}
KINDS = ["mamba", "moe"] * 44


def _config(**changes):
    config = {
        "architectures": ["DFlash2DraftModel"],
        "hidden_size": 4096,
        "vocab_size": 131072,
        "num_target_layers": 88,
        "dflash_config": {"target_layer_ids": [5, 25, 45, 65, 85]},
    }
    config.update(changes)
    return config


def _text(**changes):
    return {"hidden_size": 4096, "vocab_size": 131072, "layers_block_type": KINDS,
            "num_hidden_layers": 88, **changes}


def test_declared_contract_names_the_tapped_layer_kinds():
    assert binder._check_declared_contract(_config(), _text()) == ["moe"] * 5
    assert tuple(_config()["dflash_config"]["target_layer_ids"]) == binder.TARGET_LAYER_IDS


@pytest.mark.parametrize(
    "config, text",
    [
        (_config(architectures=["LiLiCorrDraftModel"]), _text()),
        (_config(architectures=["DFlash2DraftModel", "X"]), _text()),
        (_config(dflash_config={"target_layer_ids": [4, 24, 44, 64, 84]}), _text()),
        (_config(dflash_config={"target_layer_ids": [5, 25, 45, 65]}), _text()),
        (_config(dflash_config={}), _text()),
        (_config(num_target_layers=52), _text()),
        (_config(), _text(layers_block_type=KINDS[:52])),
        (_config(hidden_size=2048), _text()),
        (_config(), _text(vocab_size=1000)),
    ],
)
def test_architecture_tap_or_target_mismatch_fails_closed(config, text):
    with pytest.raises(ValueError):
        binder._check_declared_contract(config, text)


def _real_text():
    from mlx2.adapters.nemotron35_super_config import language_config

    return language_config(json.loads((TARGET / "config.json").read_text()))


@pytest.mark.parametrize("label", sorted(DRAFTS))
def test_real_artifact_is_pinned_and_schema_exact(label):
    path = DRAFTS[label]
    if not (path / "model.safetensors").is_file() or not (TARGET / "config.json").is_file():
        pytest.skip("Super VL DFlash2 artifacts are not on this host")
    record = binder.inspect_drafter(
        path, target_path=TARGET, target_text=_real_text(),
        target_fingerprint="t" * 64, eos_token_ids=(11,),
    )
    assert record["source_revision_label"] == label
    assert binder.REVISIONS[record["weight_sha256"][0]]["label"] == label
    assert record["config_sha256"] == binder.CONFIG_SHA256
    assert record["args"].block_size == 16 and record["args"].sliding_window == 2048
    assert record["tap_layer_kinds"] == ["moe", "attention", "moe", "moe", "moe"]
    receipt = binder.drafter_receipt(record, target_is_trained_export=True)
    assert receipt["implemented"] and receipt["selected"] and not receipt["qualified"]
    assert receipt["observed_used"] == "see route receipt speculation counters"
    # The target binding and tap convention are part of the route identity.
    other = binder.inspect_drafter(
        path, target_path=TARGET, target_text=_real_text(),
        target_fingerprint="u" * 64, eos_token_ids=(11,),
    )
    assert other["fingerprint"] != record["fingerprint"]
    assert other["content_revision"] == record["content_revision"]
    with pytest.raises(ValueError, match="EOS"):
        binder.inspect_drafter(
            path, target_path=TARGET, target_text=_real_text(),
            target_fingerprint="t" * 64, eos_token_ids=(2,),
        )


def test_edited_config_or_unpinned_weights_are_refused(tmp_path):
    source = DRAFTS["step-51000"]
    if not (source / "model.safetensors").is_file() or not (TARGET / "config.json").is_file():
        pytest.skip("Super VL DFlash2 artifacts are not on this host")
    edited = tmp_path / "edited"
    edited.mkdir()
    config = json.loads((source / "config.json").read_text())
    config["dflash_config"]["mask_token_id"] = 991
    (edited / "config.json").write_text(json.dumps(config))
    os.symlink(source / "model.safetensors", edited / "model.safetensors")
    with pytest.raises(ValueError, match="pinned"):
        binder.inspect_drafter(
            edited, target_path=TARGET, target_text=_real_text(), target_fingerprint="t" * 64
        )
    # The pinned config over other weights of the same schema: header-only
    # copy with a zeroed payload, refused by the payload hash.
    from mlx2.adapters.dflash2 import _read_safetensors_header

    raw_header, _header, payload = _read_safetensors_header(source / "model.safetensors")
    forged = tmp_path / "forged"
    forged.mkdir()
    shutil.copy(source / "config.json", forged / "config.json")
    with (forged / "model.safetensors").open("wb") as stream:
        stream.write(len(raw_header).to_bytes(8, "little") + raw_header)
        stream.truncate(8 + len(raw_header) + payload)  # sparse zeros
    with pytest.raises(ValueError, match="pinned revision"):
        binder.inspect_drafter(
            forged, target_path=TARGET, target_text=_real_text(), target_fingerprint="t" * 64
        )


def test_load_refuses_a_changed_record_or_a_target_without_taps():
    config = {
        "architectures": ["DFlash2DraftModel"], "hidden_size": 4096, "vocab_size": 131072,
        "num_target_layers": 88, "num_hidden_layers": 1, "intermediate_size": 16,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "dflash_config": {"target_layer_ids": [5, 25, 45, 65, 85], "block_size": 16,
                          "mask_token_id": 990, "conv_kernel_size": 2,
                          "conv_group_size": 16, "selector_rank": 4, "selector_top_k": 4},
    }
    args = DFlash2Config.from_dict(config)
    record = {"config_sha256": "0" * 64, "tap_convention": binder.TAP_CONVENTION,
              "config": config, "args": args}
    with pytest.raises(ValueError, match="changed since inspection"):
        binder.load_drafter(record, object())
    record["config_sha256"] = binder.CONFIG_SHA256
    with pytest.raises(ValueError, match="tap contract"):
        binder.load_drafter(record, object())
    with pytest.raises(ValueError):
        binder.load_drafter(record, object(), runtime_quantization="affine_q2_g64")


# -- the drafter on a Nemotron-H hybrid target (tiny, random weights) ----------

def _tiny_draft(model):
    mx.random.seed(5)
    return DFlash2DraftModel(DFlash2Config(
        hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, vocab_size=16,
        num_target_layers=5, target_layer_ids=list(CAPTURES), conv_kernel_size=2,
        conv_group_size=2, selector_rank=4, selector_top_k=4, block_size=16,
        mask_token_id=15, max_position_embeddings=128, sliding_window=6,
        layer_types=["sliding_attention"] * 2,
    )).bind(model)


def _ordinary(model, prompt, count):
    cache = _prefilled(model, prompt[:-1])
    tokens, token = [], prompt[-1]
    for _ in range(count):
        token = int(mx.argmax(model(mx.array([[token]]), cache)[0, -1]).item())
        tokens.append(token)
    return tokens


def _served(batch, prompt, count):
    batch.insert([prompt], max_tokens=[count],
                 sampling_configs=[{"sampling_temp": 0.0, "emit_logprobs": False}])
    tokens = []
    for _ in range(400):
        _, responses = batch.next()
        tokens.extend(response.token for response in responses)
        if not batch.lanes:
            break
    return tokens


@pytest.mark.parametrize("num_draft", [3, 7, 15])
def test_chain_route_serves_ordinary_greedy_tokens(num_draft):
    model = _model()
    prompt = [1, 2, 3, 4, 5, 6]
    expected = _ordinary(model, prompt, 24)
    draft = _tiny_draft(model)
    with closing(ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="nemotron-dflash2-cpu", num_draft=num_draft,
        completion_batch_size=1, prefill_step_size=8, cost_per_commit_handoff=False,
    )) as batch:
        assert batch.receipt_kind == "external_dflash2"
        assert _served(batch, prompt, 24) == expected
        stats = batch.scheduler_stats
        # Would otherwise pass with speculation silently disengaged.
        assert stats["external_rounds"] > 0 and stats["proposed_tokens"] > 0
        assert stats["draft_fallbacks"] == 0


def test_tree_route_serves_ordinary_greedy_tokens():
    model = _model()
    prompt = [1, 2, 3, 4, 5, 6]
    expected = _ordinary(model, prompt, 24)
    draft = _tiny_draft(model)
    executor = NemotronTreeExecutor()
    with closing(ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="nemotron-dflash2-tree-cpu", num_draft=7,
        completion_batch_size=1, prefill_step_size=8,
        dynamic_singleton_tree=True, dynamic_tree_max_width=1,
        dynamic_tree_overflow="chain", tree_node_budget_by_lanes={1: 7},
        target_tree_executor_provider=executor, cost_per_commit_handoff=False,
        ast_adaptive_rows=False,
    )) as batch:
        assert _served(batch, prompt, 24) == expected
        assert batch.scheduler_stats["draft_fallbacks"] == 0
    assert executor.receipt()["observed_used"] and executor.receipt()["dispatches"] > 0


def test_dflash2_is_never_a_default_tree_route(tmp_path):
    from tests.test_nemotron_tree_cpu import _tree_default_adapter

    adapter = _tree_default_adapter(tmp_path, {}, architectures=("DFlash2DraftModel",))
    assert "batch_size_route" not in adapter.external_policy
    assert adapter.tree_route_authority is None


def test_super_adapter_dispatches_dflash2_to_the_pinned_binder(tmp_path, monkeypatch):
    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter

    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["DFlash2DraftModel"]}))
    adapter = object.__new__(Nemotron35SuperAdapter)
    adapter.external_policy = {"draft_model": str(tmp_path)}
    seen = {}

    def inspect(path, **kwargs):
        seen.update(kwargs, path=path)
        return {"args": DFlash2Config.from_dict(_tiny_real_config()),
                "weight_sha256": ["w" * 64], "config_sha256": binder.CONFIG_SHA256,
                "content_revision": "c" * 64, "target_fingerprint": kwargs["target_fingerprint"],
                "tap_layer_kinds": ["moe"] * 5, "tap_convention": binder.TAP_CONVENTION,
                "source_repository": binder.SOURCE_REPOSITORY, "source_revision": "r",
                "source_revision_label": "step-51000"}

    monkeypatch.setattr(binder, "inspect_drafter", inspect)
    monkeypatch.setattr(binder, "load_drafter", lambda record, target, **kw: ("loaded", kw))
    monkeypatch.setattr(
        Nemotron35SuperAdapter, "_runtime_language_config", staticmethod(lambda config: _text())
    )
    artifact = {"config": {}, "identity": {"fingerprint": "f" * 64}, "nvfp4_modules": ["x"]}
    record, loader = adapter._external_drafter("/target", artifact, "affine_q8_g64")
    assert seen["target_fingerprint"] == "f" * 64 and seen["eos_token_ids"] == (11,)
    assert loader(record, object()) == ("loaded", {"runtime_quantization": "affine_q8_g64"})
    receipt = adapter.external_drafter_receipt
    assert receipt["architecture"] == "DFlash2DraftModel"
    assert receipt["target_is_trained_export"] and not receipt["qualified"]
    assert receipt["runtime_quantization"] == "affine_q8_g64"
    assert Nemotron35SuperAdapter._external_default_depth(record) == 15


def _tiny_real_config():
    config = copy.deepcopy(_config())
    config.update(num_hidden_layers=1, intermediate_size=16, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=4)
    config["dflash_config"].update(block_size=16, mask_token_id=990, conv_kernel_size=2,
                                   conv_group_size=16, selector_rank=4, selector_top_k=4)
    return config
