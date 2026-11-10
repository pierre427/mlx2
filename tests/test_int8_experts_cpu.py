"""int8 prefill ``experts`` (grouped W4A8 routed experts): policy, CLI,
binding, refusal and the below-threshold path (CPU only).

The Metal kernel itself never runs here; GPU correctness lives in the
probe receipts recorded with the commit.  A small qwen4_exp model with the
served layout (split gate/up ``FusedDownSwitchGLU`` experts at 4-bit gs64,
every other projection 8-bit gs64) stands in for Flash-Next.
"""

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.flash_next import FlashNextAdapter
from mlx2.runtime import int8_experts as ix
from mlx2.runtime import int8_prefill as ip


def _model(*, split=True):
    from mlx2.runtime.models import qwen3_next
    from mlx2.runtime.models.qwen4_exp import Model, ModelArgs

    text_config = dict(
        model_type="qwen4_exp_text", hidden_size=256, intermediate_size=0,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        head_dim=64, vocab_size=256, linear_num_value_heads=4,
        linear_num_key_heads=4, linear_key_head_dim=64, linear_value_head_dim=64,
        linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention"],
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=256,
        shared_expert_intermediate_size=256, hc_count=2, hc_lowrank=64,
        ple_layer_ids=[1], ple_embed_dim=256, ple_conv_kernel_size=4, ngram_size=3,
        heads_per_ngram=2, ngram_vocab_size_base=256,
        make_ngram_vocab_size_divisible_by=256, split_ngram_parts=1,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=64,
        indexer_budget=8, indexer_compress_ratio=2, mtp_num_hidden_layers=1,
        rope_parameters={"type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 0.25},
    )
    saved = qwen3_next._MOE_FUSED_GATE_UP
    qwen3_next._MOE_FUSED_GATE_UP = not split
    try:
        mx.random.seed(11)
        model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    finally:
        qwen3_next._MOE_FUSED_GATE_UP = saved
    model.set_dtype(mx.bfloat16)

    def bits(path, module):
        if not hasattr(module, "to_quantized") or module.weight.shape[-1] % 64:
            return False
        b = 4 if "switch_mlp" in path.split(".") else 8
        return {"group_size": 64, "bits": b, "mode": "affine"}

    nn.quantize(model, group_size=64, bits=8, class_predicate=bits)
    model.eval()
    return model


def _adapter(**kw):
    adapter = object.__new__(FlashNextAdapter)
    adapter.model = _model(**kw)
    return adapter


POLICY = ip.Int8PrefillPolicy(
    enabled=True, scope="all", q8_inplace=True, act_scale="group64", q8_only=True,
    experts=True)


@pytest.fixture
def device_ok(monkeypatch):
    # The Metal kernel probe runs on the GPU check, not here.
    monkeypatch.setattr(ip, "require_supported_device", lambda: "test M5")
    monkeypatch.setattr(ix, "probe", lambda module: (True, ""))


def _blocks(adapter):
    return [layer.mlp.switch_mlp for layer in adapter.model.language_model.model.layers]


# ------------------------------------------------------------------ policy


def test_policy_fields_validate_and_enter_revision_only_when_on():
    base = ip.Int8PrefillPolicy(enabled=True, scope="all", q8_inplace=True, q8_only=True)
    assert "experts" not in base.as_dict()
    assert POLICY.as_dict()["experts"] is True
    assert POLICY.as_dict()["experts_min_tokens"] == 8192
    assert POLICY.revision != base.revision
    assert POLICY.revision != ip.Int8PrefillPolicy(
        enabled=True, scope="all", q8_inplace=True, q8_only=True, experts=True,
        experts_min_tokens=16384).revision
    with pytest.raises(ValueError, match="requires an enabled"):
        ip.Int8PrefillPolicy(experts=True)
    with pytest.raises(ValueError, match="experts_min_tokens"):
        ip.Int8PrefillPolicy(enabled=True, experts=True, experts_min_tokens=256)
    # Off, the expert threshold constrains nothing (a raised row threshold
    # stays valid, as before the field existed).
    assert ip.Int8PrefillPolicy(enabled=True, row_threshold=16384).row_threshold == 16384
    assert ip.Int8PrefillPolicy.from_value(POLICY.as_dict()) == POLICY


def test_cli_value_adds_experts_only_with_an_explicit_scope():
    assert ip.cli_value("all", "group64", "off") == {
        "enabled": True, "scope": "all", "q8_inplace": True, "act_scale": "group64"}
    assert ip.cli_value("all", "group64", "q4")["experts"] is True
    assert ip.cli_value("mlp", "off", "q4") == {"enabled": True, "scope": "mlp", "experts": True}
    for scope in ("auto", "off"):
        with pytest.raises(ValueError, match="requires --int8-prefill mlp or all"):
            ip.cli_value(scope, "off", "q4")
    with pytest.raises(ValueError):
        ip.cli_value("all", "off", "q8")


# ----------------------------------------------------------------- binding


def test_flash_next_binds_trunk_expert_blocks_and_remove_restores(device_ok):
    adapter = _adapter()
    handle = ip.apply_for_adapter(adapter, POLICY)
    try:
        assert handle.experts_module_count() == 2   # trunk layers, not MTP
        for block in _blocks(adapter):
            assert block.__dict__[ix.STATE_ATTR]() is handle
        assert ix.STATE_ATTR not in adapter.model.mtp.layers[0].mlp.switch_mlp.__dict__
        assert {"experts_calls", "experts_rows", "experts_fallback_rows"} <= set(handle.counts)
    finally:
        ip.remove(handle)
    for block in _blocks(adapter):
        assert ix.STATE_ATTR not in block.__dict__
    # A second install after remove() is allowed.
    ip.remove(ip.apply_for_adapter(adapter, POLICY))


def test_refuses_without_an_adapter_declaration(device_ok):
    adapter = _adapter()
    with pytest.raises(ip.Int8PrefillError, match="int8_prefill_experts_select"):
        ip.apply(adapter.model, POLICY, select=adapter.int8_prefill_select("all"))
    assert not any(ix.STATE_ATTR in b.__dict__ for b in _blocks(adapter))
    assert not any(hasattr(type(m), "_mlx2_int8_prefill_base")
                   for _, m in adapter.model.named_modules())


def test_refuses_fused_gate_up_blocks_without_touching_the_model(device_ok):
    adapter = _adapter(split=False)
    assert ix.module_refusal(_blocks(adapter)[0]).startswith("FusedGateUpSwitchGLU")
    with pytest.raises(ip.Int8PrefillError, match="selected no 4-bit gs64 routed expert"):
        ip.apply_for_adapter(adapter, POLICY)
    # Fail closed before any dense projection was class-swapped.
    assert not any(hasattr(type(m), "_mlx2_int8_prefill_base")
                   for _, m in adapter.model.named_modules())


def test_refuses_non_q4_expert_tables():
    adapter = _adapter()
    block = _blocks(adapter)[0]
    assert ix.module_refusal(block) is None
    block.down_proj.bits = 8
    assert "not affine q4 gs64" in ix.module_refusal(block)


def test_below_min_tokens_runs_the_stock_body_exactly(device_ok):
    adapter = _adapter()
    block = _blocks(adapter)[0]
    mx.random.seed(3)
    x = mx.random.normal((1, 16, 256), dtype=mx.bfloat16)
    inds = mx.random.randint(0, 4, (1, 16, 2)).astype(mx.uint32)
    scores = mx.softmax(mx.random.normal((1, 16, 2)), axis=-1).astype(mx.bfloat16)
    ref = block(x, inds, scores=scores)
    handle = ip.apply_for_adapter(adapter, POLICY)
    try:
        got = block(x, inds, scores=scores)
        assert mx.array_equal(got, ref).item()
        assert handle.counts["experts_fallback_rows"] == 1
        assert handle.counts["experts_calls"] == 0
    finally:
        ip.remove(handle)


def test_unbound_or_dead_handle_declines():
    adapter = _adapter()
    block = _blocks(adapter)[0]
    handle = ip.Int8PrefillHandle(POLICY)
    x = mx.zeros((8192, 256), dtype=mx.bfloat16)
    inds = mx.zeros((8192, 2), dtype=mx.uint32)
    assert handle.experts_forward(block, x, inds) is None
    assert handle.counts["experts_fallback_unbound"] == 1


def test_default_off_everywhere():
    assert ip.Int8PrefillPolicy().experts is False
    assert ip.auto_policy(_adapter(), {"q8": 3, "q45": 0}).experts is False


def _exact_block(x_tok, idx, rmap, gate, up, down, activation=None):
    """Stand-in for the W4A8 kernel: the stock sorted gathers on the same
    sorted rows, so the hook's sort / row map / unsort / weighting must
    reproduce the stock block."""
    def gq(x, t):
        return mx.gather_qmm(x, t[0], t[1], t[2], rhs_indices=idx, transpose=True,
                             group_size=64, bits=4, sorted_indices=True)
    xs = x_tok[rmap][:, None, :]
    return gq(activation(gq(xs, up), gq(xs, gate)), down)[:, 0, :]


@pytest.mark.parametrize("shape,weighted", [((1, 512), True), ((2, 300), True), ((1, 512), False)])
def test_taken_path_reproduces_the_stock_block(shape, weighted, device_ok, monkeypatch):
    adapter = _adapter()
    block = _blocks(adapter)[1]
    mx.random.seed(7)
    x = mx.random.normal(shape + (256,), dtype=mx.bfloat16)
    inds = mx.argpartition(-mx.random.normal(shape + (4,)), kth=1, axis=-1)[..., :2]
    inds = inds.astype(mx.uint32)
    scores = (mx.softmax(mx.random.normal(shape + (2,)), axis=-1).astype(mx.bfloat16)
              if weighted else None)
    ref = block(x, inds, scores=scores)
    monkeypatch.setattr(ix, "expert_block", _exact_block)
    policy = ip.Int8PrefillPolicy(enabled=True, scope="all", q8_inplace=True,
                                  act_scale="group64", q8_only=True, experts=True,
                                  experts_min_tokens=512)
    handle = ip.apply_for_adapter(adapter, policy)
    try:
        got = block(x, inds, scores=scores)
        assert handle.counts["experts_calls"] == 1
        assert handle.counts["experts_rows"] == inds.size
        assert got.shape == ref.shape
        assert mx.allclose(got.astype(mx.float32), ref.astype(mx.float32),
                           atol=1e-2, rtol=1e-2).item()
    finally:
        ip.remove(handle)


def test_status_reports_the_expert_route(device_ok):
    adapter = _adapter()
    handle = ip.apply_for_adapter(adapter, POLICY)
    try:
        status = handle.status()
        assert status["experts"]["modules"] == 2
        assert status["experts"]["observed_used"] is False
        assert status["experts"]["kernel_revision"] == ix.KERNEL_REVISION
        handle.counts["experts_calls"] = 1
        engine = type("E", (), {"int8_prefill_handle": handle})()
        assert ip.engine_status(engine)["observed_used"] is True
    finally:
        ip.remove(handle)


def test_server_flag_reaches_the_engine_policy():
    from mlx2.server import build_parser, serving_engine_kwargs

    args = build_parser().parse_args([
        "--model", "/m", "--int8-prefill", "all", "--int8-prefill-q8", "group64",
        "--int8-prefill-experts", "q4"])
    kwargs = serving_engine_kwargs(args, None, native_mtp=False, approximate_kv=None,
                                   max_request_bytes=1)
    policy = ip.Int8PrefillPolicy.from_value(kwargs["int8_prefill"])
    assert policy.experts and policy.q8_inplace and policy.scope == "all"
    args = build_parser().parse_args(["--model", "/m"])
    assert getattr(args, "int8_prefill_experts") == "off"
