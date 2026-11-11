"""CPU exactness of the Nemotron-H parent-indexed tree verify.

Every node's logits and taps must be bit-equal to ordinary one-token decode
along its root-to-node path (the serial oracle), and a committed path must
leave the same cache planes as an exact chain verify of that path.
"""

import copy
import json

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.hybrid_tree_executor import HybridTreeExecutor
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models.cache import ArraysCache, KVCache
from mlx2.runtime.models.nemotron_tree import NemotronTree
from mlx2.runtime.nemotron_tree_executor import (
    PROTOCOL,
    NemotronTreeExecutor,
    NemotronTreeTransaction,
)
from tests.test_nemotron_external_taps_cpu import tiny_model

CAPTURES = (0, 1, 4)
# A branched 16-row tile: chain prefix, siblings, windows crossing the
# committed conv state, and a late branch off the deepest chain node.
PARENTS_16 = [-1, 0, 1, 2, 3, 4, 5, 6, 0, 8, 1, 10, 2, 12, 12, 7]


def _model(pattern=("M", "*", "E", "M", "M")):
    mx.set_default_device(mx.cpu)
    seed = tiny_model()
    args = copy.copy(seed.args)
    args.hybrid_override_pattern = list(pattern)
    args.num_hidden_layers = len(pattern)
    args.moe_latent_size = 4
    args.moe_intermediate_size = 6
    args.moe_shared_expert_intermediate_size = 7
    args.n_group = 1
    args.n_routed_experts = 4
    args.n_shared_experts = 1
    args.topk_group = 1
    args.num_experts_per_tok = 2
    args.norm_topk_prob = True
    args.routed_scaling_factor = 1.0
    mx.random.seed(11)
    model = type(seed)(args)
    model.set_short_segment_policy("layerwise_short_exact_routed_batch")
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.set_conv_policy("fused_short")
            layer.mixer.set_ssm_policy("fused_short_gate")
            layer.mixer.set_projection_policy("short_row_exact")
        elif layer.block_type == "E":
            object.__setattr__(
                layer.mixer, "_routed_batch_geometry_refusal", lambda _rows: None
            )
    return model


def _prefilled(model, prompt=(1, 2, 3, 4, 5)):
    cache = model.make_cache()
    model(mx.array([list(prompt)]), cache)
    mx.eval([entry.state for entry in cache])
    return cache


def _tokens(count):
    return [(3 * index + 5) % 16 for index in range(count)]


def _bits(value):
    value = np.asarray(value)
    return value.view(np.uint32) if value.dtype == np.float32 else value


def _assert_bits(actual, expected):
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def _assert_planes_equal(actual, expected):
    for left, right in zip(actual, expected, strict=True):
        assert type(left) is type(right)
        if isinstance(left, ArraysCache):
            for a, e in zip(left.cache, right.cache, strict=True):
                assert a.dtype == e.dtype and a.shape == e.shape
                _assert_bits(a, e)
            assert not left.speculating and not left._rollbacks
        else:
            assert left.offset == right.offset
            for a, e in zip(left.state, right.state, strict=True):
                _assert_bits(a[..., : left.offset, :], e[..., : right.offset, :])


def _paths(parents):
    paths = []
    for row, parent in enumerate(parents):
        paths.append([row] if parent < 0 else paths[parent] + [row])
    return paths


def _oracle(model, cache, tokens, parents):
    """Ordinary one-token decode from a snapshot of each node's parent."""
    logits, features, transaction = HybridTreeExecutor().forward(
        model, tokens, parents, cache, CAPTURES, cached=False
    )
    mx.eval(logits, features)
    transaction.abort()
    return np.asarray(logits), np.asarray(features)


def _ordinary_commit(model, cache, tokens, parents, path):
    """The serial oracle's committed planes: one-token decode of the path."""
    live = copy.deepcopy(cache)
    logits, features, transaction = HybridTreeExecutor().forward(
        model, tokens, parents, live, CAPTURES, cached=False
    )
    mx.eval(logits, features)
    transaction.commit_paths([path])
    return live


def _assert_planes_close(actual, expected, tolerance=2e-6):
    for left, right in zip(actual, expected, strict=True):
        states = (
            zip(left.cache, right.cache)
            if isinstance(left, ArraysCache)
            else zip(
                (value[..., : left.offset, :] for value in left.state),
                (value[..., : right.offset, :] for value in right.state),
            )
        )
        for a, e in states:
            np.testing.assert_allclose(np.asarray(a), np.asarray(e), rtol=1e-5, atol=tolerance)


def _chain_commit(model, cache, tokens):
    """Exact chain verify of one path; returns its committed planes."""
    owner = HybridVerifyRows([copy.deepcopy(cache)])
    transaction = owner.begin([len(tokens)])
    logits, taps = model.forward_with_taps(
        mx.array([tokens]), transaction.caches, CAPTURES
    )
    mx.eval(logits, taps)
    return transaction.commit([len(tokens)])[0]


def _continue(model, cache, token=7):
    logits = model(mx.array([[token]]), cache)
    mx.eval(logits)
    return np.asarray(logits)


@pytest.mark.parametrize(
    "parents",
    [
        [-1],
        [-1, 0],
        [-1, 0, 0, 0],
        [-1, 0, 1, 1, 2, 3],
        PARENTS_16,
    ],
)
def test_tree_nodes_match_one_token_decode_per_path(parents):
    model = _model()
    cache = _prefilled(model)
    tokens = _tokens(len(parents))
    expected_logits, expected_features = _oracle(model, cache, tokens, parents)

    executor = NemotronTreeExecutor()
    live = copy.deepcopy(cache)
    logits, features, transaction = executor.forward(
        model, tokens, parents, live, CAPTURES, cached=False
    )
    mx.eval(logits, features)
    assert isinstance(transaction, NemotronTreeTransaction)
    assert transaction.provider_protocol == PROTOCOL
    _assert_bits(logits, expected_logits)
    _assert_bits(features, expected_features)
    transaction.abort()
    receipt = executor.receipt()
    assert receipt["implemented"] and not receipt["qualified"]
    assert receipt["observed_used"] and receipt["dispatches"] == 1
    # CPU float32 refuses the bf16 Metal tree kernels: the serial backend.
    assert receipt["mamba_backend"] == "serial_or_mixed"


@pytest.mark.parametrize("terminal", [0, 3, 7, 9, 11, 13, 14, 15])
def test_committed_path_equals_ordinary_decode_and_continues(terminal):
    model = _model()
    cache = _prefilled(model)
    tokens = _tokens(len(PARENTS_16))
    path = _paths(PARENTS_16)[terminal]
    expected = _ordinary_commit(model, cache, tokens, PARENTS_16, path)

    live = copy.deepcopy(cache)
    _logits, _features, transaction = NemotronTreeExecutor().forward(
        model, tokens, PARENTS_16, live, CAPTURES, cached=False
    )
    mx.eval(_logits, _features)
    committed = transaction.commit_paths([path])[0]
    assert committed is live and transaction.closed
    mx.eval([entry.state for entry in live])
    _assert_planes_equal(live, expected)
    # The exact chain verify of the same path commits the same planes up to
    # its own CPU strip rounding (its gated norm and S=p replay are not
    # row-invariant on MLX CPU; on Metal both are, see the Metal tests).
    _assert_planes_close(live, _chain_commit(model, cache, [tokens[row] for row in path]))
    _assert_bits(_continue(model, live), _continue(model, expected))


def test_abort_leaves_live_cache_at_its_committed_boundary():
    model = _model()
    cache = _prefilled(model)
    reference = copy.deepcopy(cache)
    live = copy.deepcopy(cache)
    _logits, _features, transaction = NemotronTreeExecutor().forward(
        model, _tokens(16), PARENTS_16, live, CAPTURES, cached=False
    )
    mx.eval(_logits, _features)
    transaction.abort()
    assert transaction.closed
    _assert_planes_equal(live, reference)
    _assert_bits(_continue(model, live), _continue(model, reference))
    with pytest.raises(RuntimeError):
        transaction.commit_paths([[0]])


def test_shallow_windows_read_committed_conv_rows():
    # Depth-0..2 nodes take 3 - depth conv taps from the committed state.
    tree = NemotronTree([-1, 0, 1, 2, 0], _prefilled(_model()))
    assert tree.window_rows(4) == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [1, 2, 3]]
    assert tree.window_rows(3) == [[0, 1], [1, 2], [2, 3], [3, 4], [1, 2]]
    assert tree.chain_prefix == [True, True, True, True, False]


def test_tree_bounds_and_commit_paths_are_validated():
    model = _model()
    cache = _prefilled(model)
    with pytest.raises(ValueError):
        NemotronTree(list(range(-1, 16)), cache)  # 17 rows
    with pytest.raises(ValueError):
        NemotronTree([-1, 1], cache)
    with pytest.raises(ValueError):
        NemotronTree([0, 0], cache)
    tree = NemotronTree(PARENTS_16, cache)
    with pytest.raises(ValueError):
        tree.validate_path([0, 8, 1])
    with pytest.raises(ValueError):
        tree.validate_path([1])
    assert tree.validate_path([0, 8, 9]) == [0, 8, 9]


def test_refusals_fail_closed_before_any_state_changes():
    model = _model()
    cache = _prefilled(model)
    model.layers[0].mixer.set_projection_policy("auto")
    live = copy.deepcopy(cache)
    with pytest.raises(ValueError, match="mamba projection auto"):
        NemotronTreeExecutor().forward(
            model, _tokens(4), [-1, 0, 0, 1], live, CAPTURES, cached=False
        )
    _assert_planes_equal(live, cache)
    with pytest.raises(ValueError, match="committed"):
        speculating = copy.deepcopy(cache)
        speculating[0].speculating = True
        NemotronTree([-1, 0], speculating)


def test_receipt_binds_sources_and_reports_states_distinctly():
    executor = NemotronTreeExecutor()
    receipt = executor.receipt()
    assert receipt["provider"] == PROTOCOL
    assert receipt["dispatches"] == 0 and not receipt["observed_used"]
    assert receipt["implemented"] is True and receipt["qualified"] is False
    assert "models/nemotron_mamba_tree.py" in receipt["source_identity"]
    executor.validate_cohort(1, 1)
    with pytest.raises(ValueError):
        executor.validate_cohort(2, 4)


# -- LiLiCorr tree proposals --------------------------------------------------


def _lilicorr_with_lattice(scores, logits):
    from types import SimpleNamespace

    from mlx import nn

    from mlx2.runtime.drafters.lilicorr import LiLiCorrDraftModel
    from tests.test_lilicorr_cpu import config

    model = LiLiCorrDraftModel(config(block_size=4, lilicorr_candidate_topk=4))
    model.bind(SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(9, 4))))
    model._hidden = lambda inputs, h, c: mx.zeros((1, 4, 4))
    model._logits = lambda h: mx.array([logits] * 3)[None]
    model.lilicorr = lambda *a: mx.array(scores)
    return model


# Candidates per position are the top-4 logits: tokens 0..3 (columns 0..3).
_LOGITS = [10.0, 9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_SCORES = [[
    [[3.0, 2.0, -1.0, -2.0]] * 4,  # start: broadcast over the predecessor column
    [[0.0, 4.0, 1.0, 0.0], [5.0, 0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0] * 4],
    [[4.0, 0.0, 3.5, 0.0], [0.0, 4.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0] * 4],
]]


def test_lilicorr_tree_contains_the_greedy_chain_as_first_children():
    mx.set_default_device(mx.cpu)
    model = _lilicorr_with_lattice(_SCORES, _LOGITS)
    chain, _laws = model.draft_distributions(
        [2], mx.zeros((1, 0, 8)), model.make_cache(), 3, [object()], [0.0]
    )
    trees = model.propose_tree([2], mx.zeros((1, 0, 8)), model.make_cache(), 7)
    tokens, parents = trees[0]
    assert len(tokens) == 7 and parents[0] == -1
    assert all(parent < row for row, parent in enumerate(parents))
    # Follow the first child at each depth: that is the greedy chain.
    row, walked = 0, [tokens[0]]
    while True:
        children = [child for child, parent in enumerate(parents) if parent == row]
        if not children:
            break
        row = children[0]
        walked.append(tokens[row])
    assert walked == chain[0] == [0, 1, 1]
    # Second-best root child is a sibling of the greedy root.
    assert [tokens[child] for child, p in enumerate(parents) if p == -1][:2] == [0, 1]


def test_lilicorr_tree_respects_budget_depth_and_forbidden_tokens():
    mx.set_default_device(mx.cpu)
    model = _lilicorr_with_lattice(_SCORES, _LOGITS)
    tokens, parents = model.propose_tree(
        [2], mx.zeros((1, 0, 8)), model.make_cache(), 15
    )[0]
    depths = []
    for parent in parents:
        depths.append(0 if parent < 0 else depths[parent] + 1)
    assert max(depths) <= 2  # block_size 4: three draft positions
    tokens, parents = model.propose_tree(
        [2], mx.zeros((1, 0, 8)), model.make_cache(), 5, forbidden_token_ids=[[0]]
    )[0]
    assert 0 not in tokens and len(tokens) <= 5
    assert model.propose_tree([2], mx.zeros((1, 0, 8)), model.make_cache(), 2)[0][0][:1] == [0]


# -- end to end through the external-draft scheduler ------------------------


def _tree_drafter(model, *, siblings=6):
    from mlx2.adapters.xpress import XPressConfig
    from mlx2.runtime.drafters.xpress import XPressDraftModel

    draft = XPressDraftModel(XPressConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=16, mask_token_id=15, num_target_layers=5,
        target_layer_ids=list(CAPTURES), block_size=8,
        layer_types=["full_attention"],
        xpress_rank=3, xpress_mlp_hidden=5, xpress_num_passes=3,
    )).bind(model)
    calls = []

    def propose_tree(anchors, hidden, cache, max_nodes, *, forbidden_token_ids=None, **_):
        chains, _laws = draft.draft_distributions(
            anchors, hidden, cache, 4, [None] * len(anchors), [0.0] * len(anchors),
            predecessor_ids=[0] * len(anchors),
        )
        draft.adaptive_confidence_features = None  # tree rows carry none
        trees = []
        for row, chain in enumerate(chains):
            banned = set(forbidden_token_ids[row]) if forbidden_token_ids else set()
            tokens, parents = [], []
            for depth, token in enumerate(chain):
                if token in banned:
                    break
                tokens.append(int(token))
                parents.append(-1 if depth == 0 else len(tokens) - 2)
            # Many root siblings, so the target's token is usually present at
            # depth one even when the drafter is wrong; then a second level.
            for token in range(16):
                if len(tokens) >= min(max_nodes, 4 + siblings) or token in banned:
                    continue
                if token not in [tokens[i] for i, p in enumerate(parents) if p == -1]:
                    tokens.append(token)
                    parents.append(-1)
            first_sibling = next((i for i, p in enumerate(parents) if p == -1 and i), None)
            for token in range(16):
                if len(tokens) >= max_nodes or first_sibling is None or token in banned:
                    break
                tokens.append(token)
                parents.append(first_sibling)
            calls.append(len(tokens))
            trees.append((tokens, parents))
        return trees

    draft.propose_tree = propose_tree
    return draft, calls


def test_tree_route_serves_ordinary_greedy_tokens_end_to_end():
    from contextlib import closing

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    model = _model()
    prompts = [[1, 2, 3, 4, 5, 6]]
    configs = [{"sampling_temp": 0.0, "emit_logprobs": False}]
    cache = _prefilled(model, prompts[0][:-1])
    expected, token = [], prompts[0][-1]
    for _ in range(24):
        token = int(mx.argmax(model(mx.array([[token]]), cache)[0, -1]).item())
        expected.append(token)

    draft, calls = _tree_drafter(model)
    executor = NemotronTreeExecutor()
    with closing(ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="nemotron-tree-cpu", num_draft=4,
        completion_batch_size=1, prefill_step_size=8,
        dynamic_singleton_tree=True, dynamic_tree_max_width=1,
        dynamic_tree_overflow="chain", tree_node_budget_by_lanes={1: 15},
        target_tree_executor_provider=executor, cost_per_commit_handoff=False,
        ast_adaptive_rows=False,
    )) as batch:
        batch.insert(prompts, max_tokens=[24], sampling_configs=configs)
        actual = []
        for _ in range(200):
            _, responses = batch.next()
            actual.extend(response.token for response in responses)
            if not batch.lanes:
                break
    assert actual == expected
    receipt = executor.receipt()
    assert receipt["observed_used"] and receipt["dispatches"] >= 2
    assert calls and max(calls) >= 8
    # Fewer target dispatches than tokens: trees accepted multiple rows.
    assert receipt["dispatches"] < len(expected)


# -- adapter wiring (opt-in, default off) -------------------------------------


def test_super_tree_policy_is_opt_in_and_validated():
    from mlx2.adapters.nemotron35_super import (
        NEMOTRON_TREE_ROUTE,
        validate_super_tree_policy,
    )

    assert validate_super_tree_policy({"draft_model": "x"}) is False
    assert validate_super_tree_policy(
        {"draft_model": "x", "batch_size_route": NEMOTRON_TREE_ROUTE}
    ) is True
    for bad in (
        {"batch_size_route": "tree15_b1_b4_chain_b5plus_v1"},
        {"ast_tile_rows": 8},
        {"batch_size_route": NEMOTRON_TREE_ROUTE, "ast_tile_rows": 17},
        {"batch_size_route": NEMOTRON_TREE_ROUTE, "ast_target_executor": "fast"},
        {"batch_size_route": NEMOTRON_TREE_ROUTE, "continuation_pool": {}},
        {"batch_size_route": NEMOTRON_TREE_ROUTE, "progressive_verification_tile": 4},
    ):
        with pytest.raises(ValueError):
            validate_super_tree_policy(bad)


@pytest.mark.parametrize("executor", ["packed", "serial"])
def test_super_adapter_binds_the_tree_provider(monkeypatch, executor):
    from types import SimpleNamespace

    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
    from mlx2.adapters.nemotron35_super import (
        NEMOTRON_TREE_ROUTE,
        Nemotron35SuperAdapter,
    )

    adapter = object.__new__(Nemotron35SuperAdapter)
    adapter.external_policy = {
        "draft_model": "x", "batch_size_route": NEMOTRON_TREE_ROUTE,
        "ast_target_executor": executor, "ast_tile_rows": 12,
    }
    adapter.tree_route = True
    adapter.draft_model = SimpleNamespace(propose_tree=lambda *a, **k: None)
    monkeypatch.setattr(
        ExternalDraftAdapterMixin, "create_external_batch", lambda self, **kw: kw
    )
    options = adapter.create_external_batch(completion_batch_size=1)
    assert options["dynamic_singleton_tree"] is True
    assert options["cost_per_commit_handoff"] is False  # no plain-decode probes
    assert options["dynamic_tree_max_width"] == 1
    assert options["tree_node_budget_by_lanes"] == {1: 11}
    provider = options["target_tree_executor_provider"]
    assert type(provider) is (
        NemotronTreeExecutor if executor == "packed" else HybridTreeExecutor
    )
    adapter.draft_model = SimpleNamespace()
    with pytest.raises(ValueError, match="propose_tree"):
        adapter.create_external_batch(completion_batch_size=1)


def test_kv_capacity_growth_inside_a_tile_stays_exact():
    # 250 committed rows + 16 tile rows cross the 256-row KV allocation step.
    model = _model()
    cache = _prefilled(model, [(index * 5 + 1) % 16 for index in range(250)])
    tokens = _tokens(16)
    expected_logits, _ = _oracle(model, cache, tokens, PARENTS_16)
    for terminal in (7, 14):
        live = copy.deepcopy(cache)
        logits, features, transaction = NemotronTreeExecutor().forward(
            model, tokens, PARENTS_16, live, CAPTURES, cached=False
        )
        mx.eval(logits, features)
        _assert_bits(logits, expected_logits)
        path = _paths(PARENTS_16)[terminal]
        transaction.commit_paths([path])
        _assert_planes_equal(live, _ordinary_commit(model, cache, tokens, PARENTS_16, path))


def test_tree_policy_refusal_is_available_before_any_cache():
    model = _model()
    assert model.tree_policy_refusal(16) is None
    model.set_short_segment_policy("auto")
    assert model.tree_policy_refusal(16) == "mamba segment auto"
    model = _model()
    model.layers[3].mixer.set_ssm_policy("auto")
    assert model.tree_policy_refusal(16) == "mamba ssm auto"


def test_cached_fp32_router_weight_is_bit_identical_to_the_per_call_cast():
    from mlx2.runtime.models.nemotron_h import MoEGate, group_expert_select

    model = _model(("E", "M", "*", "E", "M"))
    gate = next(layer.mixer.gate for layer in model.layers if layer.block_type == "E")
    gate.config.router_compute_dtype = "float32"
    gate.weight = gate.weight.astype(mx.bfloat16)
    x = mx.random.normal((1, 3, gate.weight.shape[1])).astype(mx.bfloat16)
    cached = gate(x)
    logits = x.astype(mx.float32) @ gate.weight.astype(mx.float32).T
    reference = group_expert_select(
        logits, gate.e_score_correction_bias, gate.top_k, gate.n_group,
        gate.topk_group, gate.routed_scaling_factor, gate.norm_topk_prob,
    )
    for a, e in zip(cached, reference, strict=True):
        _assert_bits(a, e)
    first = gate.float32_weight()
    assert gate.float32_weight() is first
    gate.weight = gate.weight * 1  # a new weight array re-converts
    assert gate.float32_weight() is not first
    assert isinstance(gate, MoEGate)


def _tree_default_adapter(tmp_path, policy, architectures=("LiLiCorrDraftModel",)):
    from mlx2.adapters.nemotron35_super import Nemotron35SuperAdapter

    (tmp_path / "config.json").write_text(json.dumps({"architectures": list(architectures)}))
    adapter = object.__new__(Nemotron35SuperAdapter)
    adapter.external_policy = {"draft_model": str(tmp_path), **policy}
    adapter.tree_route_authority = None
    adapter._default_tree_route()
    return adapter


def test_super_lilicorr_defaults_to_the_tree8_route(tmp_path):
    from mlx2.adapters.nemotron35_super import (
        NEMOTRON_TREE_ROUTE,
        Nemotron35SuperAdapter,
        validate_super_tree_policy,
    )

    assert Nemotron35SuperAdapter.EXTERNAL_LILICORR_TREE_ROWS == 8
    adapter = _tree_default_adapter(tmp_path, {})
    assert adapter.external_policy["batch_size_route"] == NEMOTRON_TREE_ROUTE
    assert adapter.external_policy["ast_tile_rows"] == 8
    assert adapter.tree_route_authority == "adapter_default_lilicorr_tree8"
    assert validate_super_tree_policy(adapter.external_policy) is True
    # Operator route, tree controls or chain-only controls: no default.
    for policy in (
        {"batch_size_route": "chain"},
        {"batch_size_route": NEMOTRON_TREE_ROUTE, "ast_tile_rows": 12},
        {"proposal_composition": {}},
        {"cost_per_commit_handoff": True},
    ):
        adapter = _tree_default_adapter(tmp_path, policy)
        assert adapter.tree_route_authority is None
        assert adapter.external_policy.get("ast_tile_rows") == policy.get("ast_tile_rows")
    assert validate_super_tree_policy({"batch_size_route": "chain"}) is False
    with pytest.raises(ValueError):
        validate_super_tree_policy({"batch_size_route": "chain", "ast_tile_rows": 8})
    # Not LiLiCorr (DSpark): the chain.
    adapter = _tree_default_adapter(tmp_path, {}, architectures=("DSparkDraftModel",))
    assert "batch_size_route" not in adapter.external_policy


def test_bolt_keeps_the_chain():
    from mlx2.adapters.nemotron35_bolt import Nemotron35BoltAdapter

    assert Nemotron35BoltAdapter.EXTERNAL_LILICORR_TREE_ROWS is None


def test_a_refused_default_tree_yields_to_the_chain(tmp_path):
    adapter = _tree_default_adapter(tmp_path, {})
    adapter.tree_route = True
    adapter._resolve_tree_refusal("mamba segment auto")
    assert adapter.tree_route is False
    assert "batch_size_route" not in adapter.external_policy
    assert "ast_tile_rows" not in adapter.external_policy
    assert adapter.tree_route_authority == "adapter_default_refused:mamba segment auto"
    operator = _tree_default_adapter(tmp_path, {"batch_size_route": "tree15_b1_chain_b2plus_v1"})
    operator.tree_route_authority = "operator"
    with pytest.raises(ValueError, match="refused at load"):
        operator._resolve_tree_refusal("mamba segment auto")


def test_feedback_and_physical_target_keep_the_chain(tmp_path):
    for key in ("lilicorr_feedback", "ast_physical_target"):
        adapter = _tree_default_adapter(tmp_path, {key: True})
        assert "batch_size_route" not in adapter.external_policy
