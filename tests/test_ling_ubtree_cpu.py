import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters.external_drafter_formats import format_for_config
from mlx2.runtime.drafters.ubtree_config import UBTreeConfig
from mlx2.runtime.drafters.ubtree_tree import build_ubtree


def _config():
    return {
        "architectures": ["Ling3UBTreeModel"],
        "model_type": "qwen3",
        "hidden_size": 2560,
        "intermediate_size": 6144,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "vocab_size": 157184,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 6000000},
        "block_size": 16,
        "mask_token_id": 156892,
        "layer_types": ["full_attention"] * 5,
        "dflash_config": {
            "block_size": 16,
            "contract": "flashflow_anchor_inclusive_v1",
            "mask_token_id": 156892,
            "num_target_layers": 42,
            "target_layer_ids": [1, 11, 23, 29, 35],
        },
        "ubtree_config": {
            "type": "residual_bigram",
            "version": 1,
            "architecture": "cp",
            "candidate_source": "frozen_dflash_unary_topk_ids_only",
            "depth_conditioning": "unary_correction_two_scale",
            "depth_scale_parameterization": "exp_log_scale",
            "token_factorizer": "target_embedding_mlp",
            "hidden_projection_type": "shared",
            "inference_hidden_projection": "materialized_per_depth_affine",
            "tree_builder": "pruned",
            "adds_unary_logits": True,
            "trained": True,
            "rank": 256,
            "factorizer_hidden_width": 512,
            "candidate_k": 64,
            "tree_supertree_width": 12,
            "tree_budget": 64,
            "depth_condition_count": 15,
        },
    }


def test_released_config_is_distinct_ubtree_format():
    args = UBTreeConfig.from_dict(_config())
    assert args.target_layer_ids == [1, 11, 23, 29, 35]
    assert args.selector_top_k == 64
    assert args.tree_budget == 64
    assert args.maximum_proposal_length == 15
    assert format_for_config(_config()).name == "ubtree"


def test_config_refuses_changed_selector_contract():
    config = copy.deepcopy(_config())
    config["ubtree_config"]["tree_builder"] = "greedy"
    with pytest.raises(ValueError, match="selector contract"):
        UBTreeConfig.from_dict(config)


def test_graft_keeps_history_and_copy_topology_out_of_real_ubtree_protocol():
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
    from mlx2.runtime.drafters.ubtree import UBTreeDraftModel

    class ReachedLattice(Exception):
        pass

    def lattice(*args):
        raise ReachedLattice

    class Base:
        config = SimpleNamespace(block_size=16)
        _lattice = staticmethod(lattice)
        propose_tree = UBTreeDraftModel.propose_tree

    draft = Base()
    target = SimpleNamespace(speculative_args=SimpleNamespace(vocab_size=100, hidden_size=8), layers=[object()])
    graft = PromptLookupDraftModel(target, num_draft=15, base=draft)
    with pytest.raises(ReachedLattice):
        graft.propose_tree(
            [1], None, [], 15, processor_histories=[[2, 3]], topology="3x5",
        )
    with pytest.raises(TypeError, match="unsupported UBTree tree options"):
        UBTreeDraftModel.propose_tree(draft, [1], None, [], 15, topology="3x5")
    with pytest.raises(TypeError, match="unsupported UBTree tree options"):
        UBTreeDraftModel.propose_tree(draft, [1], None, [], 15, unknown_option=True)


def test_tree_is_bounded_parent_first_and_prefix_closed():
    candidates = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    unary = np.array([[3.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
    projected = np.ones((3, 1))
    ids = np.arange(7, dtype=np.int32)
    predecessor = np.ones((7, 1))
    successor = np.zeros((7, 1))
    tokens, parents = build_ubtree(
        candidates,
        unary,
        projected,
        0,
        predecessor,
        successor,
        np.zeros(3),
        np.zeros(3),
        codebook_ids=ids,
        frontier_width=2,
        max_nodes=5,
    )
    assert len(tokens) == len(parents) == 5
    assert parents[0] == -1
    assert all(parent < row for row, parent in enumerate(parents))
    assert max(parents) < len(tokens)


def test_tree_prefix_preserves_smaller_budget_acceptance():
    # The likely chain outranks the second root. Depth-ordered publication
    # would insert that root ahead of the chain and mislead AST's budget
    # counterfactual: B3 accepts three drafts, not just two from B4[:3].
    candidates = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    unary = np.log([[0.6, 0.4], [0.99, 0.01], [0.99, 0.01]])
    zeros = np.zeros((7, 1))

    def tree(budget):
        return build_ubtree(
            candidates, unary, np.zeros((3, 1)), 0, zeros, zeros,
            np.zeros(3), np.zeros(3), frontier_width=2, max_nodes=budget,
        )

    tokens, parents = tree(4)
    assert tokens == [1, 3, 5, 2]
    assert parents == [-1, 0, 1, -1]
    for budget in (1, 2, 3):
        assert tree(budget) == (tokens[:budget], parents[:budget])


def test_depth_calibration_changes_the_selected_root():
    candidates = np.array([[1, 2]], dtype=np.int32)
    unary = np.array([[1.0, 0.0]])
    projected = np.ones((1, 1))
    ids = np.array([0, 1, 2], dtype=np.int32)
    predecessor = np.array([[1.0], [0.0], [0.0]])
    successor = np.array([[0.0], [-2.0], [2.0]])
    tokens, _ = build_ubtree(
        candidates,
        unary,
        projected,
        0,
        predecessor,
        successor,
        np.array([-6.0]),
        np.array([2.0]),
        codebook_ids=ids,
        frontier_width=1,
        max_nodes=1,
    )
    assert tokens == [2]


def test_provenance_records_both_checkpoint_sources():
    root = Path(__file__).resolve().parents[1]
    path = root / "provenance/ling3-ubtree.json"
    if not path.is_file():
        pytest.skip("private material absent: Ling 3 UBTree provenance (not exported to the public mirror)")
    document = json.loads(path.read_text())
    assert document["target"]["revision"] == "c00a8c3074236b713f15e12db59f0158d29f3b13"
    assert document["drafter"]["revision"] == "59ced6e1e8fa434570be038e262c3e62552cf9a4"
    assert document["license"] == "MIT"


def test_codebooks_build_once_per_binding_and_match_direct_factorizers():
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx import nn

    from mlx2.runtime.drafters.ubtree import UBTreeDraftModel

    config = UBTreeConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=1, num_key_value_heads=1, head_dim=8,
        vocab_size=32, target_layer_ids=[0], num_target_layers=1,
        selector_rank=2, factorizer_hidden_width=4,
    )
    draft = UBTreeDraftModel(config)
    target = SimpleNamespace(
        args=SimpleNamespace(hidden_size=8, vocab_size=32, num_hidden_layers=1),
        model=SimpleNamespace(word_embeddings=nn.Embedding(32, 8)), lm_head=nn.Linear(8, 32),
    )
    draft.bind(target)
    first = draft._precompute_codebooks()
    second = draft._precompute_codebooks()
    assert first is second
    assert draft.receipt_settings["codebook_builds"] == 1
    embeddings = target.model.word_embeddings(mx.arange(32)).astype(mx.float32)
    np.testing.assert_array_equal(first[0], np.asarray(draft.ubtree_selector.predecessor_factorizer(embeddings)))
    np.testing.assert_array_equal(first[1], np.asarray(draft.ubtree_selector.successor_factorizer(embeddings)))
    assert not first[0].flags.writeable
    draft.bind(target)
    assert draft._codebook_tables is None
    draft._precompute_codebooks()
    assert draft.receipt_settings["codebook_builds"] == 2


def test_hybrid_tree_executor_keeps_live_state_private_until_commit():
    import mlx.core as mx

    from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor, tree_forward
    from mlx2.runtime.models.bailing_hybrid import Model, ModelArgs
    from mlx2.runtime.target_tree_executor import TargetTreeExecutorProvider

    args = ModelArgs(
        model_type="bailing_hybrid",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=None,
        num_experts_per_tok=1,
        num_shared_experts=0,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=2,
        layer_group_size=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        routed_scaling_factor=1.0,
        head_dim=8,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        qk_nope_head_dim=4,
        v_head_dim=8,
        no_kda_lora=True,
        gated_attention_proj_granularity_type="head_wise",
    )
    model = Model(args)
    assert isinstance(HybridTreeExecutor.create(), TargetTreeExecutorProvider)
    cache = model.make_cache()
    logits, taps, transaction = tree_forward(
        model, [1, 2, 3], [-1, 0, 0], cache, [0, 1], "test-token"
    )
    mx.eval(logits, taps)
    assert logits.shape == (1, 3, 32)
    assert taps.shape == (1, 3, 32)
    assert cache[0].lengths is None
    assert cache[1].offset == 0
    transaction.commit_paths([[0, 2]])
    assert all(value is not None for value in cache[0].cache)
    assert cache[1].offset == 2


def test_ast_invocation_selects_ling_ubtree_tiles(tmp_path, monkeypatch):
    from mlx2.adapters import tensorfold_dtype_gate
    from mlx2.adapters.ling3_flash import Ling3FlashAdapter
    from mlx2.server import apply_ast_invocation

    monkeypatch.setattr(
        tensorfold_dtype_gate,
        "artifact_tree_tile_dtype_issue",
        lambda _path: (_ for _ in ()).throw(AssertionError("Qwen-only dtype gate")),
    )

    args = SimpleNamespace(
        ast=True,
        ast_drafter="/models/Ling-3.0-flash-UBTree",
        ast_topology=None,
        ast_graft_pld=False,
        ast_tiles=None,
        ast_tile_rows=None,
        draft_model=None,
        model=str(tmp_path),
        external_draft=False,
        native_mtp=False,
        prompt_lookup=False,
    )
    resolution = SimpleNamespace(
        adapter_type=Ling3FlashAdapter, artifact={}, default_route="ordinary"
    )
    policy = apply_ast_invocation(args, None, resolution)
    assert policy == {
        "draft_model": "/models/Ling-3.0-flash-UBTree",
        "batch_size_route": "tree15_bounded_cohorts_v1",
    }
    adapter = object.__new__(Ling3FlashAdapter)
    assert Ling3FlashAdapter.ast_tree_tile_dtype_gate is False
    assert adapter._parse_external_policy(policy, family="Ling 3 Flash") is True
    assert args.external_draft is True
    assert args.ast_route["tree_tiles"] is True
