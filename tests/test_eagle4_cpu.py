"""CPU contracts for the dense EAGLE-4 checkpoint and AST drafter."""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

mx.set_default_device(mx.cpu)

from mlx2.runtime.drafters.eagle4 import (
    Eagle4Config,
    Eagle4DraftModel,
    expected_weight_schema,
)
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.standard_decoder import Model, ModelArgs

VOCAB, LAYERS, HIDDEN = 40, 8, 16


def target_config():
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": 24,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "eos_token_id": VOCAB - 1,
    }


def draft_config():
    return {
        "_torchspec_version": "0.1.0",
        "architectures": ["LlamaForCausalLMEagle4"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "draft_vocab_size": VOCAB,
        "dtype": "float32",
        "eagle_config": {
            "eagle_aux_hidden_state_layer_ids": [1, 3, 6],
            "use_aux_hidden_state": True,
        },
        "efficient_lm_head_config": {
            "num_candidates": 8,
            "rank": 4,
            "type": "low_rank",
        },
        "fc_norm": True,
        "head_dim": 4,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": 24,
        "layer_types": ["sliding_attention"],
        "max_position_embeddings": 256,
        "mlp_bias": False,
        "norm_output": True,
        "num_attention_heads": 4,
        "num_aux_hidden_states": 3,
        "num_hidden_layers": 1,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "sliding_window": 32,
        "target_hidden_size": HIDDEN,
        "tie_word_embeddings": False,
        "use_sliding_window": True,
        "vocab_size": VOCAB,
    }


def tiny(seed=7):
    mx.random.seed(seed)
    target = Model(
        ModelArgs(
            **{
                k: v
                for k, v in target_config().items()
                if k not in {"architectures", "eos_token_id"}
            }
        )
    )
    args = Eagle4Config.from_hf(draft_config(), num_target_layers=LAYERS)
    draft = Eagle4DraftModel(args).bind(target)
    mx.eval(target.parameters(), draft.parameters())
    return target, draft


def greedy_reference(target, prompt, count):
    cache = target.make_cache()
    tokens, output = list(prompt), []
    for step in range(count):
        inputs = tokens if step == 0 else [tokens[-1]]
        logits = target(mx.array([inputs]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        output.append(token)
    return output


def drain(batch):
    output, final = {}, {}
    for _ in range(300):
        _, responses = batch.next()
        for response in responses:
            output.setdefault(response.uid, []).append(response.token)
            if response.finish_reason:
                final[response.uid] = response
        if not batch.lanes:
            return output, final
    raise AssertionError("scheduler stalled")


def test_config_schema_and_candidate_head_are_strict():
    args = Eagle4Config.from_hf(draft_config(), num_target_layers=LAYERS)
    assert args.target_layer_ids == [1, 3, 6]
    schema = expected_weight_schema(args)
    assert schema["efficient_lm_head.weight"] == [VOCAB, 4]
    assert schema["layers.0.self_attn.q_proj.weight"] == [HIDDEN, 2 * HIDDEN]
    with pytest.raises(ValueError, match="fc_norm"):
        Eagle4Config.from_hf(
            {**draft_config(), "fc_norm": False}, num_target_layers=LAYERS
        )


def test_candidate_law_rescores_only_low_rank_candidates():
    target, draft = tiny()
    assert draft.embed_tokens is target.model.embed_tokens
    hidden = mx.random.normal((2, HIDDEN))
    ids, _, probs = draft._candidate_rows(hidden)
    mx.eval(ids, probs)
    assert ids.shape == probs.shape == (2, 8)
    np.testing.assert_allclose(np.asarray(probs).sum(axis=-1), 1, atol=1e-5)
    assert len(set(np.asarray(ids[0]).tolist())) == 8


def test_external_chain_preserves_greedy_target_output():
    target, draft = tiny()
    batch = ExternalDraftBatchGenerator(
        target,
        draft_model=draft,
        binding="eagle4-test",
        num_draft=3,
        prefill_step_size=3,
    )
    prompt = [1, 2, 3, 4]
    identifiers = batch.insert(
        [prompt], max_tokens=[8], sampling_configs=[{"sampling_temp": 0}]
    )
    output, final = drain(batch)
    assert output[identifiers[0]] == greedy_reference(target, prompt, 8)
    assert final[identifiers[0]].speculative_receipt["kind"] == "external_eagle4"
    batch.close()


def test_tree_is_parent_first_and_honors_forbidden_tokens():
    _, draft = tiny()
    cache = draft.make_cache()
    hidden = mx.random.normal((1, 1, 3 * HIDDEN))
    ((tokens, parents),) = draft.propose_tree(
        [5],
        hidden,
        draft.batch_caches([cache]),
        11,
        context_tokens=[[5]],
        forbidden_token_ids=[(7,)],
    )
    assert len(tokens) == len(parents) == 11
    assert all(parent < row for row, parent in enumerate(parents))
    assert 7 not in tokens
    assert cache[0].offset == 1
    with pytest.raises(ValueError, match="positive"):
        draft.propose_tree([5], hidden, draft.batch_caches([draft.make_cache()]), 0)


def test_artifact_inspection_load_and_qwen_format_dispatch(tmp_path):
    from mlx2.adapters.eagle4 import content_revision, inspect_drafter, load_drafter
    from mlx2.adapters.external_drafter_formats import drafter_format

    draft_dir, target_dir = tmp_path / "draft", tmp_path / "target"
    draft_dir.mkdir()
    target_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps(draft_config()))
    (target_dir / "config.json").write_text(json.dumps(target_config()))
    model = Eagle4DraftModel(
        Eagle4Config.from_hf(draft_config(), num_target_layers=LAYERS)
    )
    mx.save_safetensors(
        str(draft_dir / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    record = inspect_drafter(draft_dir, target_dir)
    assert len(record["fingerprint"]) == len(content_revision(record)) == 64
    assert drafter_format(draft_dir).name == "eagle4"
    target = Model(
        ModelArgs(
            **{
                k: v
                for k, v in target_config().items()
                if k not in {"architectures", "eos_token_id"}
            }
        )
    )
    loaded = load_drafter(record, target)
    assert loaded.receipt_kind == "external_eagle4"


def test_eagle4_inspection_imports_no_model_module(tmp_path):
    """The Qwen3.8 27B adapter inspects the drafter before pinning its environment.

    Live 2026-10-21: ``--ast --ast-drafter <EAGLE-4>`` refused to load the
    target (ImportOrderError), first through ``ast_drafter_policy_defaults``
    at CLI time, then through ``inspect_external_policy`` at adapter init,
    because the EAGLE-4 format module imported ``runtime.models``.
    """
    import subprocess
    import sys

    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLMEagle4"]}))
    code = (
        "import sys\n"
        "from mlx2.adapters.external_drafter_formats import drafter_format, drafter_format_name\n"
        f"assert drafter_format_name({str(tmp_path)!r}) == 'eagle4'\n"
        f"assert drafter_format({str(tmp_path)!r}).name == 'eagle4'\n"
        "from mlx2.adapters.qwen38_27b import Qwen3827BAdapter\n"
        f"assert Qwen3827BAdapter.ast_drafter_policy_defaults({str(tmp_path)!r}) == {{}}\n"
        "bad = sorted(m for m in sys.modules if m.startswith('mlx2.runtime.models'))\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _tree_depths(parents):
    depths = []
    for parent in parents:
        depths.append(0 if parent < 0 else depths[parent] + 1)
    return depths


def test_eagle4_tree_policy_is_validated():
    from mlx2.runtime.drafters.eagle4_config import EAGLE4_TREE_DEFAULT, validate_eagle4_tree

    assert validate_eagle4_tree(None) == EAGLE4_TREE_DEFAULT == {
        "children": 4, "depth": None, "max_nodes": None,
    }
    assert validate_eagle4_tree({"children": 1, "depth": 7}) == {
        "children": 1, "depth": 7, "max_nodes": None,
    }
    for bad, match in (
        ([1], "object"),
        ({"width": 2}, "unknown"),
        ({"children": 0}, "children"),
        ({"children": 9}, "children"),
        ({"children": True}, "children"),
        ({"depth": 0}, "depth"),
        ({"depth": 32}, "depth"),
        ({"max_nodes": 65}, "max_nodes"),
        ({"max_nodes": 2.0}, "max_nodes"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_eagle4_tree(bad, num_candidates=8)


@pytest.mark.parametrize(
    ("tree", "budget", "length", "max_depth", "max_children"),
    [
        ({"children": 1, "depth": 4}, 11, 4, 3, 1),       # MTP-like chain
        ({"children": 1, "depth": 10}, 15, 10, 9, 1),
        ({"children": 2, "depth": 2}, 11, 6, 1, 2),       # full binary, two levels
        ({"children": 8, "depth": 2, "max_nodes": 5}, 15, 5, 0, 8),  # wide: all roots
    ],
)
def test_eagle4_tree_shape_follows_policy(tree, budget, length, max_depth, max_children):
    from mlx2.runtime.drafters.eagle4_config import validate_eagle4_tree

    _, draft = tiny()
    draft.tree = validate_eagle4_tree(tree, num_candidates=draft.config.num_candidates)
    cache = draft.make_cache()
    hidden = mx.random.normal((1, 1, 3 * HIDDEN))
    ((tokens, parents),) = draft.propose_tree(
        [5], hidden, draft.batch_caches([cache]), budget, context_tokens=[[5]],
    )
    assert len(tokens) == len(parents) == length
    assert all(parent < row for row, parent in enumerate(parents))
    assert max(_tree_depths(parents)) == max_depth
    fanout = {}
    for parent in parents:
        fanout[parent] = fanout.get(parent, 0) + 1
    assert max(fanout.values()) <= max_children
    settings = draft.receipt_settings
    assert settings["tree_children"] == draft.tree["children"]
    assert settings["tree_depth"] == draft.tree["depth"]
    assert settings["tree_max_nodes"] == draft.tree["max_nodes"]


def test_eagle4_default_tree_is_unchanged():
    """The policy default reproduces the pre-policy shape (children 4, budget-bound)."""
    _, draft = tiny()
    assert draft.tree == {"children": 4, "depth": None, "max_nodes": None}
    hidden = mx.random.normal((1, 1, 3 * HIDDEN))
    ((tokens, parents),) = draft.propose_tree(
        [5], hidden, draft.batch_caches([draft.make_cache()]), 15, context_tokens=[[5]],
    )
    fanout = {}
    for parent in parents:
        fanout[parent] = fanout.get(parent, 0) + 1
    assert len(tokens) == 15 and fanout[-1] == 4
