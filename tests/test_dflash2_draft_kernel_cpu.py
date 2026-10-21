"""DFlash2 drafter arithmetic options: lane kernel, block top-k, reduced head (CPU)."""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.drafters import dflash2 as runtime
from mlx2.runtime.drafters.dflash2 import (
    DFlash2DraftModel,
    _fused_context_projections,
    _lattice_candidates,
    block_topk,
    configure_draft_kernel,
    configure_draft_vocab,
    draft_kernel_law,
)
from mlx2.runtime.drafters.dflash2_config import DFlash2Config

from test_qwen38_dflash2_adapter import _draft_config, _pins, _write_artifacts


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _model(**dflash):
    model = DFlash2DraftModel(DFlash2Config.from_dict(_draft_config(**dflash)))
    mx.eval(model.parameters())
    return model


def _values(logits, ids):
    rows = np.asarray(logits.astype(mx.float32))
    return np.sort(np.take_along_axis(rows, np.asarray(ids).astype(np.int64), -1), -1)


@pytest.mark.parametrize("vocab", [248320, 5000, 1000, 300, 17])
def test_block_topk_returns_a_top_k_set(vocab):
    logits = mx.random.normal((2, 15, vocab), key=mx.random.key(vocab)).astype(mx.bfloat16)
    count = min(16, vocab)
    reference = mx.argpartition(logits, -count, axis=-1)[..., -count:]
    picked = block_topk(logits, count)
    assert picked.dtype == mx.uint32 and picked.shape == reference.shape
    # Same values (a tie at the k-th value may pick another member).
    np.testing.assert_array_equal(_values(logits, reference), _values(logits, picked))
    assert all(len(set(row)) == count for row in np.asarray(picked).reshape(-1, count))


def test_stock_kernel_installs_nothing_and_unknown_kernels_fail():
    model = _model()
    assert configure_draft_kernel(model, None) is None
    assert configure_draft_kernel(model, "stock") is None
    assert draft_kernel_law("stock") is None
    with pytest.raises(ValueError, match="draft_kernel must be"):
        configure_draft_kernel(model, "fast")


def test_lane_kernel_fails_closed_without_a_backend():
    # The CPU default device has no lane backend.
    with pytest.raises(ValueError, match="lane matmul backend"):
        draft_kernel_law("lane")
    with pytest.raises(ValueError, match="lane matmul backend"):
        configure_draft_kernel(_model(), "lane")


def test_lane_kernel_refuses_a_bound_head(monkeypatch):
    from mlx2.runtime import lane

    monkeypatch.setattr(lane, "backend", lambda: "mpp")
    model = _model()
    model.lm_head = nn.Linear(16, 64, bias=False)
    with pytest.raises(ValueError, match="before binding"):
        configure_draft_kernel(model, "lane")


def test_lane_law_names_backend_rows_and_lattice(monkeypatch):
    from mlx2.runtime import lane

    monkeypatch.setattr(lane, "backend", lambda: "mpp")
    assert draft_kernel_law("lane") == (
        "lane-matmul-v1+rows-le-128+grouped+fused-context-v1+blocktopk-v1"
    )
    monkeypatch.setattr(lane, "backend", lambda: "simd")
    assert draft_kernel_law("lane").startswith("lane-simd-v2+")


def test_fused_context_projections_match_separate_projections():
    model = _model()
    attn = model.layers[0].self_attn
    x = mx.random.normal((2, 5, 16), key=mx.random.key(1))
    ctx = mx.random.normal((2, 3, 16), key=mx.random.key(2))
    fused = _fused_context_projections(attn, x, ctx)
    plain = (attn.q_proj(x), attn.k_proj(ctx), attn.v_proj(ctx), attn.k_proj(x), attn.v_proj(x))
    for got, want in zip(fused, plain):
        assert got.shape == want.shape
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)
    # Empty context and long context keep the separate projections.
    empty = _fused_context_projections(attn, x, ctx[:, :0])
    assert empty[1].shape == (2, 0, attn.n_kv_heads * attn.head_dim)
    long_ctx = mx.zeros((2, 64, 16))
    assert _fused_context_projections(attn, x, long_ctx)[1].shape[1] == 64


def test_attention_uses_an_installed_projection_override():
    model = _model()
    attn = model.layers[0].self_attn
    calls = []

    def spy(module, x, x_ctx):
        calls.append((x.shape, x_ctx.shape))
        return _fused_context_projections(module, x, x_ctx)

    object.__setattr__(attn, "_fused_projections", spy)
    assert attn._projections(mx.zeros((1, 4, 16)), mx.zeros((1, 2, 16)))[0].shape == (1, 4, 16)
    assert calls == [((1, 4, 16), (1, 2, 16))]


@pytest.mark.parametrize("quantized", [False, True])
def test_reduced_head_drafts_inside_its_subset_and_falls_back(quantized):
    model = _model(selector_top_k=4)
    head = nn.Linear(64, 64, bias=False) if quantized else nn.Linear(16, 64, bias=False)
    if quantized:
        head = nn.Linear(64, 64, bias=False)
        head = nn.QuantizedLinear.from_linear(head, group_size=64, bits=8)
        model.config.hidden_size = 64
        features = mx.random.normal((1, 3, 64), key=mx.random.key(3))
    else:
        features = mx.random.normal((1, 3, 16), key=mx.random.key(3))
    model.lm_head = head
    receipt = configure_draft_vocab(model, 20, special_from=60)
    assert receipt == {
        "order": "id", "rows": 20, "special_from": 60, "head_rows": 24,
        "fallback": "full_head_when_an_anchor_is_outside",
    }
    state = model.__dict__["_draft_vocab"]
    allowed = set(range(20)) | set(range(60, 64))
    candidates, unary = _lattice_candidates(model, features, [5], 4)
    assert set(np.asarray(candidates).reshape(-1)) <= allowed
    full = model._logits(features)
    np.testing.assert_array_equal(
        np.sort(np.asarray(unary), -1),
        np.sort(np.take_along_axis(
            np.asarray(full.astype(mx.float32)), np.asarray(candidates).astype(np.int64), -1
        ), -1),
    )
    assert state["reduced_rounds"] == 1 and state["full_rounds"] == 0
    # An anchor outside the subset: this round drafts from the full head.
    candidates, _ = _lattice_candidates(model, features, [5, 40], 4)
    reference = mx.argpartition(full, -4, axis=-1)[..., -4:]
    np.testing.assert_array_equal(_values(full, candidates), _values(full, reference))
    assert state["full_rounds"] == 1


def test_reduced_head_validates_its_rows():
    model = _model()
    model.lm_head = nn.Linear(16, 64, bias=False)
    assert configure_draft_vocab(model, None, special_from=60) is None
    with pytest.raises(ValueError, match="rows must be"):
        configure_draft_vocab(model, 60, special_from=60)
    with pytest.raises(ValueError, match="special_from"):
        configure_draft_vocab(model, 10, special_from=65)


def test_receipt_settings_name_the_drafter_arithmetic_only_when_set():
    model = _model()
    plain = dict(model.receipt_settings)
    assert not {"runtime_quantization", "projection_kernel", "draft_vocab"} & set(plain)
    object.__setattr__(model, "runtime_quantization", {"bits": 4, "group_size": 64})
    object.__setattr__(model, "draft_kernel_receipt", {"kernel": "lane", "law_id": "L"})
    object.__setattr__(model, "draft_vocab_receipt", {"order": "id", "rows": 20})
    settings = model.receipt_settings
    assert settings["runtime_quantization"] == {"bits": 4, "group_size": 64}
    assert settings["projection_kernel"] == {"kernel": "lane", "law_id": "L"}
    assert settings["draft_vocab"] == {"order": "id", "rows": 20}


def _vocab_artifacts(tmp_path):
    return _write_artifacts(tmp_path, draft={**_draft_config(), "eos_token_id": 60})


def test_draft_kernel_and_vocab_are_part_of_the_draft_identity(tmp_path, monkeypatch):
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    from mlx2.runtime import lane

    monkeypatch.setattr(lane, "backend", lambda: "mpp")
    target, draft = _vocab_artifacts(tmp_path)
    policy = _pins(target, draft)
    plain = inspect_external_policy(policy, target)
    stock = inspect_external_policy({**policy, "draft_kernel": "stock"}, target)
    assert stock["fingerprint"] == plain["fingerprint"] and stock["draft_kernel"] is None
    lane_record = inspect_external_policy({**policy, "draft_kernel": "lane"}, target)
    assert lane_record["fingerprint"] != plain["fingerprint"]
    assert lane_record["draft_kernel"] == "lane"
    q4 = {"bits": 4, "group_size": 64}
    q4_record = inspect_external_policy({**policy, "draft_quantization": q4}, target)
    both = inspect_external_policy(
        {**policy, "draft_quantization": q4, "draft_kernel": "lane"}, target
    )
    assert len({plain["fingerprint"], q4_record["fingerprint"], both["fingerprint"],
                lane_record["fingerprint"]}) == 4
    vocab = {"order": "id", "rows": 20}
    reduced = inspect_external_policy({**policy, "draft_vocab": vocab}, target)
    assert reduced["draft_vocab"] == vocab
    assert reduced["fingerprint"] != plain["fingerprint"]
    # The backend is part of the lane law.
    monkeypatch.setattr(lane, "backend", lambda: "simd")
    simd = inspect_external_policy({**policy, "draft_kernel": "lane"}, target)
    assert simd["fingerprint"] != lane_record["fingerprint"]


@pytest.mark.parametrize(
    "extra, error",
    [
        ({"draft_kernel": "fast"}, "draft_kernel must be"),
        ({"draft_kernel": "lane"}, "lane matmul backend"),
        ({"draft_vocab": {"rows": 20}}, "draft_vocab must be"),
        ({"draft_vocab": {"order": "freq", "rows": 20}}, "draft_vocab must be"),
        ({"draft_vocab": {"order": "id", "rows": 60}}, "below the special"),
    ],
)
def test_drafter_option_policy_negative_cases(tmp_path, extra, error):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _vocab_artifacts(tmp_path)
    with pytest.raises(ValueError, match=error):
        inspect_external_policy({**_pins(target, draft), **extra}, target)


def test_draft_vocab_needs_the_draft_eos(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="eos_token_id"):
        inspect_external_policy(
            {**_pins(target, draft), "draft_vocab": {"order": "id", "rows": 20}}, target
        )


def test_policy_inspection_imports_no_model_module(tmp_path):
    """The server inspects the policy before the adapter pins its import
    environment; a model module imported there fails the load (ImportOrderError)."""
    import subprocess
    import sys
    from pathlib import Path

    target, draft = _vocab_artifacts(tmp_path)
    tests = Path(__file__).resolve().parent
    code = f"""
import sys
sys.path.insert(0, {str(tests)!r})
from mlx2.runtime import lane
lane.backend = lambda: "mpp"
from mlx2.adapters.qwen38_27b import inspect_external_policy
from test_qwen38_dflash2_adapter import _pins
policy = {{**_pins({str(target)!r}, {str(draft)!r}), "draft_kernel": "lane",
          "draft_quantization": {{"bits": 4, "group_size": 64}},
          "draft_vocab": {{"order": "id", "rows": 20}}}}
before = "mlx2.runtime.models.base" in sys.modules
inspect_external_policy(policy, {str(target)!r})
print(before, "mlx2.runtime.models.base" in sys.modules, "mlx2.runtime.drafters.dflash_base" in sys.modules)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split()[-3:] == ["False", "False", "False"], out.stdout + out.stderr


def test_module_constants_are_stable():
    assert runtime.DRAFT_KERNELS == ("stock", "lane")
    assert runtime.DRAFT_LANE_MAX_ROWS == 128
    assert json.dumps(runtime.DRAFT_KERNELS)


def test_ast_drafter_defaults_are_measured_pair_and_backend_only(tmp_path, monkeypatch):
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime import lane

    _target, draft = _vocab_artifacts(tmp_path)
    monkeypatch.setattr(lane, "backend", lambda: "mpp")
    assert Qwen3827BAdapter.ast_drafter_policy_defaults(str(draft)) == {
        "draft_kernel": "lane", "draft_quantization": {"bits": 4, "group_size": 64},
    }
    assert Qwen3827BAdapter.ast_drafter_policy_defaults("mtp") == {}
    assert Qwen3627BAdapter.ast_drafter_policy_defaults(str(draft)) == {}
    config = json.loads((draft / "config.json").read_text())
    (draft / "config.json").write_text(
        json.dumps({**config, "quantization": {"bits": 4, "group_size": 64}})
    )
    assert Qwen3827BAdapter.ast_drafter_policy_defaults(str(draft)) == {"draft_kernel": "lane"}
    monkeypatch.setattr(lane, "backend", lambda: "simd")
    assert Qwen3827BAdapter.ast_drafter_policy_defaults(str(draft)) == {}
