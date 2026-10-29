"""Complete profile-builder CPU scope controls; no real checkpoint or forward.

Checkpoint: not_applicable_no_real_checkpoint. Synthetic weights exercise
byte priors, not model qualification, device performance, or generation.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters.ast_execution_profile import build_execution_profile
from mlx2.runtime.models.switch_layers import SwitchLinear


def profile(target, draft):
    return build_execution_profile(
        target, draft, binding="synthetic-scope-prior", tile_cap=3,
        max_lanes=1, num_draft=7,
    )


@pytest.mark.parametrize("embedding_first", [True, False])
def test_tied_embedding_does_not_hide_output_projection(embedding_first):
    embedding = nn.Embedding(8, 4)
    head = nn.Linear(4, 8, bias=False)
    head.weight = embedding.weight
    projection = nn.Linear(4, 4, bias=False)
    modules = [("embed_tokens", embedding), ("lm_head", head)]
    if not embedding_first:
        modules.reverse()
    target = SimpleNamespace(named_modules=lambda: modules)
    draft = SimpleNamespace(named_modules=lambda: [("projection", projection)])
    result = profile(target, draft)
    # Indexed embedding work is excluded, but the same weight's real dense
    # output projection must still contribute one full pass, in either order.
    assert result.draft_cost == projection.weight.nbytes / head.weight.nbytes
    assert result.cost_authority.startswith("device_build_kernel_geometry_prior+")


def test_wrapper_uses_backend_geometry_modules_and_revision():
    target_bank = SwitchLinear(4, 4, 8, bias=False)
    draft_bank = SwitchLinear(4, 4, 8, bias=False)
    target = SimpleNamespace(
        args=SimpleNamespace(num_experts_per_tok=2),
        named_modules=lambda: [("experts", target_bank)],
    )
    backend = SimpleNamespace(
        args=SimpleNamespace(num_experts_per_tok=6), revision="bound-backend",
        named_modules=lambda: [("experts", draft_bank)],
    )
    wrapper = SimpleNamespace(
        backend=backend, args=SimpleNamespace(num_experts_per_tok=1),
        revision="unrelated-wrapper",
        named_modules=lambda: pytest.fail("wrapper modules are not draft execution"),
    )
    direct, wrapped = profile(target, backend), profile(target, wrapper)
    assert direct.draft_cost == wrapped.draft_cost == 3
    assert direct.binding == wrapped.binding
    backend.revision = "different-backend"
    assert profile(target, wrapper).binding != wrapped.binding


def test_independent_target_and_draft_roots_ignore_dormant_modules_and_dedup_roots():
    target_body, draft_body = nn.Module(), nn.Module()
    target_body.experts = SwitchLinear(4, 4, 8, bias=False)
    draft_body.experts = SwitchLinear(4, 4, 4, bias=False)
    dormant_target = nn.Linear(8, 16, bias=False)
    target = SimpleNamespace(
        args=SimpleNamespace(num_experts_per_tok=2),
        progressive_external_target_modules=(target_body,),
        named_modules=lambda: [
            *(('body.' + name, module) for name, module in target_body.named_modules()),
            ("resident_head", dormant_target),
        ],
    )
    draft = SimpleNamespace(
        args=SimpleNamespace(num_experts_per_tok=3),
        ast_execution_modules=(draft_body, draft_body),
        named_modules=lambda: pytest.fail("declared draft roots must win"),
    )
    first = profile(target, draft)
    expected = (draft_body.experts.weight.nbytes * 3 / 4) / (
        target_body.experts.weight.nbytes * 2 / 8
    )
    assert first.draft_cost == expected == 1.5
    dormant_target = nn.Linear(16, 32, bias=False)
    assert profile(target, draft).binding == first.binding


@pytest.mark.parametrize("which", ["target", "draft", "both"])
def test_weight_aliases_and_unrelated_embeddings_do_not_change_byte_ratio(which):
    target_layer, draft_layer = nn.Linear(4, 8, bias=False), nn.Linear(4, 4, bias=False)
    target_rows = [("projection", target_layer)]
    draft_rows = [("projection", draft_layer)]
    target = SimpleNamespace(named_modules=lambda: target_rows)
    draft = SimpleNamespace(named_modules=lambda: draft_rows)
    before = profile(target, draft)
    if which in ("target", "both"):
        target_rows.extend([
            ("projection_alias", target_layer),
            ("embed_tokens", nn.Embedding(64, 4)),
        ])
    if which in ("draft", "both"):
        draft_rows.extend([
            ("projection_alias", draft_layer),
            ("embedding", nn.Embedding(64, 4)),
        ])
    after = profile(target, draft)
    assert before.draft_cost == after.draft_cost == 0.5
    assert before.row_costs == after.row_costs


@pytest.mark.parametrize("target_k", [1, 2, 8, 20])
@pytest.mark.parametrize("draft_k", [1, 3, None])
def test_mixed_dense_expert_bytes_obey_independent_geometry_ratios(target_k, draft_k):
    target_dense, draft_dense = nn.Linear(4, 4, bias=False), nn.Linear(4, 8, bias=False)
    target_bank, draft_bank = SwitchLinear(4, 4, 8, bias=False), SwitchLinear(4, 4, 4, bias=False)
    target = SimpleNamespace(
        args=SimpleNamespace(num_experts_per_tok=target_k),
        named_modules=lambda: [("dense", target_dense), ("experts", target_bank)],
        progressive_external_attention_geometry={"kv_bytes_per_token": 32, "windows": {64: 16}},
    )
    draft = SimpleNamespace(
        config={"num_experts_per_tok": draft_k},
        named_modules=lambda: [("dense", draft_dense), ("experts", draft_bank)],
    )
    result = profile(target, draft)
    # Independent arithmetic control from physical CPU tensor payloads:
    # dense projections read their entire matrix; a routed expert projection
    # reads one expert payload per selected expert, capped by bank count.
    target_bytes = target_dense.weight.nbytes + (target_bank.weight.nbytes // 8) * min(target_k, 8)
    draft_bytes = draft_dense.weight.nbytes + (draft_bank.weight.nbytes // 4) * (
        4 if draft_k is None else min(draft_k, 4)
    )
    assert result.draft_cost == draft_bytes / target_bytes
    assert result.context_row_cost == 32 / target_bytes
    assert result.context_windows == ((64, 16 / target_bytes),)
    assert result.row_costs == (1, 2, 3, 4)


@pytest.mark.parametrize(
    "declarations,expected_k,authority",
    [
        ({"speculative_args": None, "args": {"num_experts_per_tok": 3}}, 3, "args"),
        ({"speculative_args": {}, "args": {"num_experts_per_tok": 3}}, 3, "args"),
        ({"speculative_args": {"num_experts_per_tok": None}, "config": {"num_experts_per_tok": 3}}, 3, "config"),
        ({"speculative_args": {"num_experts_per_tok": 1}, "args": {"num_experts_per_tok": 3}}, 1, "speculative_args"),
        ({"args": {"num_experts_per_tok": 1}, "config": {"num_experts_per_tok": 3}}, 1, "args"),
        ({"speculative_args": {"num_experts_per_tok": 0}, "args": {"num_experts_per_tok": 3}}, None, "speculative_args_unknown"),
    ],
)
def test_missing_and_conflicting_draft_declarations_have_explicit_precedence(
    declarations, expected_k, authority
):
    target_bank, draft_bank = SwitchLinear(4, 4, 8, bias=False), SwitchLinear(4, 4, 8, bias=False)
    target = SimpleNamespace(
        args=SimpleNamespace(num_experts_per_tok=2),
        named_modules=lambda: [("experts", target_bank)],
    )
    draft = SimpleNamespace(**declarations, named_modules=lambda: [("experts", draft_bank)])
    result = profile(target, draft)
    assert result.draft_cost == (8 if expected_k is None else expected_k) / 2
    coverage = "full_bank_unknown" if expected_k is None else f"top_k={expected_k}"
    assert f"draft_experts=independent:{authority}:{coverage}" in result.cost_authority


def test_wrapped_native_mtp_uses_head_scope_and_intentional_shared_geometry():
    from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel

    body, head = nn.Module(), nn.Module()
    body.experts = SwitchLinear(4, 4, 8, bias=False)
    head.experts = SwitchLinear(4, 4, 8, bias=False)
    target = SimpleNamespace(
        args=SimpleNamespace(num_experts_per_tok=2), mtp=head,
        progressive_external_target_modules=(body,),
        named_modules=lambda: [
            *(('body.' + name, module) for name, module in body.named_modules()),
            *(('mtp.' + name, module) for name, module in head.named_modules()),
        ],
    )
    native = NativeMTPDraftModel.__new__(NativeMTPDraftModel)
    native.target, native.revision = target, "scoped-native-head"
    wrapped = SimpleNamespace(backend=native, args={"num_experts_per_tok": 7})
    first = profile(target, wrapped)
    assert first.draft_cost == 1
    target.args.num_experts_per_tok = 6
    second = profile(target, wrapped)
    assert second.draft_cost == 1
    assert second.binding != first.binding
    assert "draft_experts=shared_target:args:top_k=6" in second.cost_authority
    assert first.sequential_draft and second.reusable_draft_lattice


def test_test_fixture_device_is_cpu():
    assert mx.default_device() == mx.cpu
