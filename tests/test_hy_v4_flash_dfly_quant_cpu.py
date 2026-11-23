"""Hy4-flash DFly2 ``draft_quantization`` (load-time affine drafter weights, CPU).

Tiny random checkpoint with the real tensor names (``test_hy_v4_flash_cpu``).
The quantized head must hold exactly the expected quantized modules, keep
norms / absorbed MLA maps / sinks in the checkpoint dtype, leave the target
untouched, and serve committed tokens equal to ordinary greedy (the target
verifies every draft).  The policy is refused where no drafter exists.
"""

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_ast_multilane_cpu import PROMPTS  # noqa: E402
from test_hy_v4_flash_cpu import (  # noqa: E402
    _drain,
    _greedy,
    _tiny_artifact,
    build,
    checkpoint,
    tiny_config,
)

Q8 = {"bits": 8, "group_size": 32}


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, drafter = build(config, W)
    return config, W, model, drafter


def _adapter(tmp_path, tiny, policy):
    from mlx2.adapters import hy_v4_flash as A

    config, W, _, _ = tiny
    return A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)), execution_policy=policy)


def _expected_split(head, group):
    """(quantizable Linear paths, Linear paths whose input width misses the group)."""
    ok, skipped = [], []
    for path, module in head.named_modules():
        if isinstance(module, nn.Linear):
            (skipped if module.weight.shape[-1] % group else ok).append(path)
    return sorted(ok), sorted(skipped)


def test_policy_accepted_on_dfly_and_refused_without_a_drafter():
    from mlx2.adapters import hy_v4_flash as A

    _, external, kind = A._parse_policy({"draft_model": "embedded_dfly", "draft_quantization": Q8})
    assert kind == "dfly" and external["draft_quantization"] == Q8
    # Default q8 (Pierre 2026-11-21): no key gives the 8-bit head.
    assert A._parse_policy({"draft_model": "embedded_dfly"})[1]["draft_quantization"] == \
        A.DFLY_DRAFT_QUANTIZATION_DEFAULT
    # null restores bf16 explicitly.
    assert "draft_quantization" not in A._parse_policy(
        {"draft_model": "embedded_dfly", "draft_quantization": None})[1]
    with pytest.raises(ValueError, match="ordinary route has none"):
        A._parse_policy({"draft_quantization": Q8})
    with pytest.raises(ValueError, match="ordinary route has none"):
        A._parse_policy({"engram_table_format": "bf16", "draft_quantization": Q8})
    with pytest.raises(ValueError, match="DFly2 drafter only"):
        A._parse_policy({"draft_model": "prompt_lookup", "draft_quantization": Q8})
    for bad in ({"bits": 3, "group_size": 64}, {"bits": 8}, "q8", {"bits": 8, "group_size": 48}):
        with pytest.raises(ValueError, match="draft_quantization must be"):
            A._parse_policy({"draft_model": "embedded_dfly", "draft_quantization": bad})
    # A route key, never route-neutral (it must not ride onto the ordinary route).
    assert "draft_quantization" in A.AST_KEYS
    assert "draft_quantization" not in A.HYV4FlashAdapter.route_neutral_policy_keys


def test_ordinary_adapter_refuses_draft_quantization(tmp_path, tiny):
    with pytest.raises(ValueError, match="ordinary route has none"):
        _adapter(tmp_path, tiny, {"draft_quantization": Q8})


def test_quantize_head_modules_and_bytes(tiny):
    from mlx2.runtime.drafters.hy4_dfly import QUANTIZATION_KEPT, quantize_head

    config, W, _, _ = tiny
    _, head = build(config, W)
    dtypes = {k: v.dtype for k, v in tree_flatten(head.parameters())}
    ok, skipped = _expected_split(head, 32)
    assert ok and skipped  # the tiny geometry exercises both branches
    receipt = quantize_head(head, Q8)
    assert receipt["modules"] == ok and receipt["module_count"] == len(ok)
    assert receipt["skipped_group_width"] == skipped
    assert receipt["bits"] == 8 and receipt["group_size"] == 32 and receipt["mode"] == "affine"
    assert receipt["bytes_after"] < receipt["bytes_before"]
    assert receipt["kept_checkpoint_dtype"] == sorted(QUANTIZATION_KEPT)
    modules = dict(head.named_modules())
    for path in ok:
        q = modules[path]
        assert isinstance(q, nn.QuantizedLinear) and q.bits == 8 and q.group_size == 32
    for path in skipped:
        assert type(modules[path]) is nn.Linear
    after = {k: v.dtype for k, v in tree_flatten(head.parameters())}
    # Everything that is not a quantized projection keeps its dtype.
    kept = [k for k in dtypes if not any(k == f"{p}.weight" for p in ok)]
    assert kept and all(after[k] == dtypes[k] for k in kept)
    for i in range(len(head.layers)):
        attn = f"layers.{i}.self_attn"
        assert type(modules[f"{attn}.embed_q"]).__name__ == "MultiLinear"
        assert type(modules[f"{attn}.unembed_out"]).__name__ == "MultiLinear"
        assert after[f"{attn}.learnable_sink_param"] == mx.float32
        assert type(modules[f"layers.{i}.input_layernorm"]) is nn.RMSNorm
    assert head.runtime_quantization is receipt and head.quantization_receipt is receipt
    # A plain attribute, never a child tree of the module.
    assert "runtime_quantization" not in head and "quantization_receipt" not in head


def test_default_is_q8_and_null_keeps_bf16(tmp_path, tiny):
    from mlx2.adapters import hy_v4_flash as A

    assert A.DFLY_DRAFT_QUANTIZATION_DEFAULT == {"bits": 8, "group_size": 64}
    _, external, kind = A._parse_policy({"draft_model": "embedded_dfly"})
    assert kind == "dfly" and external["draft_quantization"] == {"bits": 8, "group_size": 64}
    _, external, _ = A._parse_policy({"draft_model": "embedded_dfly", "draft_quantization": None})
    assert "draft_quantization" not in external
    _, external, kind = A._parse_policy({"draft_model": "prompt_lookup"})
    assert "draft_quantization" not in external


def test_explicit_bf16_receipt_is_unchanged(tmp_path, tiny):
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly", "num_draft": 3,
                                        "draft_quantization": None})
    try:
        q = adapter.diagnostics()["dfly2"]["quantization"]
        assert q["selected"] is False and q["observed_used"] is False and not q["qualified"]
        assert q["bits"] is None and q["bytes_before"] == q["bytes_after"] > 0
        assert "runtime_quantization" not in adapter.draft_model.receipt_settings
        assert not any(isinstance(m, nn.QuantizedLinear) for _, m in adapter.dfly.named_modules())
    finally:
        adapter.close()


@pytest.mark.parametrize("width", [1, 2])
def test_q8_drafter_commits_ordinary_greedy_and_records_receipt(tmp_path, tiny, width):
    from mlx2.adapters import hy_v4_flash as A

    # One artifact for both arms, so the layouts compare like for like.
    path = str(_tiny_artifact(tmp_path, tiny[0], tiny[1]))
    base = A.HYV4FlashAdapter(path, execution_policy={"draft_model": "embedded_dfly", "num_draft": 3})
    base_layout, base_bytes = base.layout, base.diagnostics()["dfly2"]["quantization"]["bytes_before"]
    base_target = {k: v for k, v in tree_flatten(base.model.parameters())}
    base.close()
    adapter = A.HYV4FlashAdapter(path, execution_policy={"draft_model": "embedded_dfly", "num_draft": 3,
                                                         "draft_quantization": Q8})
    try:
        q = adapter.diagnostics()["dfly2"]["quantization"]
        assert q["selected"] and not q["observed_used"]  # loaded, not yet drafted
        n = 12
        batch = adapter.create_external_batch(completion_batch_size=width, prefill_step_size=4)
        prompts = PROMPTS[:width]
        uids = batch.insert(prompts, max_tokens=[n] * width,
                            sampling_configs=[{"sampling_temp": 0}] * width)
        out = _drain(batch, uids, n)
        for uid, prompt in zip(uids, prompts):
            assert out[uid][:n] == _greedy(adapter.model, prompt, n), prompt
        d = adapter.diagnostics()
        q = d["dfly2"]["quantization"]
        assert d["dfly2"]["observed_used"] and d["dfly2"]["engine_stats"]["block_drafts"] > 0
        assert q["selected"] and q["observed_used"] and not q["qualified"]
        assert (q["bits"], q["group_size"], q["mode"]) == (8, 32, "affine")
        assert q["bytes_before"] == base_bytes and q["bytes_after"] < q["bytes_before"]
        assert q["module_count"] > 0 and q["skipped_group_width"]
        assert adapter.draft_model.receipt_settings["runtime_quantization"]["bits"] == 8
        # Target untouched; APCv2 namespace shared with bf16 (cached drafter
        # state is the target's own feature rows).
        target = {k: v for k, v in tree_flatten(adapter.model.parameters())}
        assert target.keys() == base_target.keys()
        assert all(mx.array_equal(target[k], base_target[k]).item() for k in target)
        assert adapter.layout == base_layout
    finally:
        adapter.close()
