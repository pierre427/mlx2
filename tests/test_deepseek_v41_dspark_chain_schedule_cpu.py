"""Scheduled DSpark chains commit exactly the ordinary greedy tokens; CPU only."""

import math

import mlx.core as mx
import pytest
from test_deepseek_v41_embedded_dspark_cpu import pair

from mlx2.adapters import deepseek_v41_dspark_schedule as schedule
from mlx2.adapters.deepseek_v41_dspark_external import (
    DeepSeekV41DSparkDraftModel,
    create_dspark_batch,
)
from mlx2.adapters.deepseek_v41_dspark_policy import (
    split_chain_policy,
    validate_chain_policy,
)

pytest.importorskip("greenlet")


def _ordinary(target, prompt, count):
    cache = target.make_cache()
    logits = target(mx.array([prompt], mx.int32), cache)
    out = []
    for _ in range(count):
        token = int(mx.argmax(logits[0, -1]).item())
        out.append(token)
        logits = target(mx.array([[token]], mx.int32), cache)
    return out


def _speculative(target, draft, prompt, count):
    generator = create_dspark_batch(target, draft, num_draft=5, stop_tokens=set(), prefill_step_size=3,
                                    cost_per_commit_handoff=False)
    generator.insert([prompt], max_tokens=[count])
    output = []
    for _ in range(80):
        _, rows = generator.next()
        output.extend(int(r.token) for r in rows)
        if not generator.lanes:
            break
    assert generator.lanes == {}
    return output


@pytest.mark.parametrize("rows,schedule_policy", [
    ("serial", "fixed"), ("lockstep_multirow_v1", "fixed"),
    ("lockstep_multirow_v1", "survival_v1"), ("serial", "survival_v1")])
def test_scheduled_local_frontier_chain_matches_ordinary_greedy(rows, schedule_policy):
    target, backbone, _ = pair()
    prompt = [1, 2, 3, 4, 5, 6, 7]
    expected = _ordinary(target, prompt, 14)
    target.verification_commit_policy = "local_frontier"
    target.verify_rows_policy = rows
    target.verified_commit_stats = {}
    draft = DeepSeekV41DSparkDraftModel(backbone, verify_schedule=schedule_policy).bind(target)
    assert _speculative(target, draft, prompt, 14) == expected
    settings = draft.receipt_settings
    assert settings["verification"].endswith("local_frontier")
    assert settings["verify_schedule"] == schedule_policy
    if rows != "serial":
        assert target.verified_commit_stats["lockstep_calls"] > 0
    if schedule_policy == "survival_v1":
        assert draft.schedule_stats["rounds"] > 0
        assert settings["verify_costs"] == list(schedule.DEFAULT_COSTS[rows])


def test_schedule_binding_and_identity_are_distinct():
    target, backbone, fixed = pair()
    scheduled = DeepSeekV41DSparkDraftModel(backbone, verify_schedule="survival_v1").bind(target)
    custom = DeepSeekV41DSparkDraftModel(backbone, verify_schedule="survival_v1",
                                         verify_costs=[1, 1.2, 1.4, 1.6, 1.8, 2.0], draft_cost=0.1).bind(target)
    assert len({fixed.execution_binding, scheduled.execution_binding, custom.execution_binding}) == 3
    with pytest.raises(ValueError):
        DeepSeekV41DSparkDraftModel(backbone, verify_costs=[1] * 6)
    with pytest.raises(ValueError):
        DeepSeekV41DSparkDraftModel(backbone, verify_schedule="survival_v1", verify_costs=[1, 0.5, 1, 1, 1, 1])


def test_survival_choice_follows_expected_rate():
    serial = schedule.DEFAULT_COSTS["serial"]
    # Once the draft is paid, a confident chain amortizes it even serially.
    assert schedule.choose([10.0] * 5, maximum=5, costs=serial)[0] == 5
    k, receipt = schedule.choose([-10.0] * 5, maximum=5, costs=serial)
    assert k == 0 and receipt["expected_tokens"] == 1.0
    cheap = (1.0, 1.05, 1.1, 1.15, 1.2, 1.25)
    k, receipt = schedule.choose([6.0] * 5, maximum=5, costs=cheap, draft_cost=0.2)
    assert k == 5 and math.isclose(receipt["survival"][0], 1 / (1 + math.exp(-6)))
    # A confident prefix followed by a collapse stops at the collapse.
    k, _ = schedule.choose([6.0, 6.0, -8.0, 6.0, 6.0], maximum=5, costs=cheap, draft_cost=0.2)
    assert k == 2
    assert schedule.choose([6.0] * 5, maximum=3, costs=cheap)[0] == 3
    with pytest.raises(ValueError):
        schedule.choose([float("nan")], maximum=1, costs=cheap)


def test_policy_admission_fails_closed():
    value = {"draft_model": "embedded_dspark", "deepseek_dspark_verify_schedule": "survival_v1",
             "deepseek_verify_rows": "lockstep_multirow_v1"}
    chain = split_chain_policy(value, embedded=True)
    assert chain == {"deepseek_dspark_verify_schedule": "survival_v1", "deepseek_verify_rows": "lockstep_multirow_v1"}
    assert "deepseek_verify_rows" not in value
    with pytest.raises(ValueError, match="embedded_dspark"):
        split_chain_policy({"deepseek_dspark_verify_schedule": "survival_v1"}, embedded=False)
    with pytest.raises(ValueError, match="survival_v1"):
        split_chain_policy({"deepseek_dspark_draft_cost": 0.2}, embedded=True)
    base = {"draft_model": "embedded_dspark", "deepseek_verify_rows": "lockstep_multirow_v1"}
    with pytest.raises(ValueError, match="local_frontier"):
        validate_chain_policy(dict(base))
    validate_chain_policy({**base, "deepseek_verify_commit": "local_frontier"})
    with pytest.raises(ValueError, match="draft route"):
        validate_chain_policy({"deepseek_verify_rows": "lockstep_multirow_v1", "deepseek_verify_commit": "local_frontier"})
    for key, value in (("deepseek_kv_storage", "native_packed"), ("deepseek_source_dspark_ast", "tile3x5_v1"),
                       ("deepseek_physical_batch", "b2")):
        with pytest.raises(ValueError, match=key):
            validate_chain_policy({**base, "deepseek_verify_commit": "local_frontier", key: value})


def test_source_policy_round_trip_keeps_defaults_unchanged():
    from mlx2.adapters.deepseek_v41_source import source_policy
    plain = source_policy({"draft_model": "embedded_dspark", "num_draft": 3})
    assert plain["deepseek_verify_rows"] == "serial" and "deepseek_dspark_verify_schedule" not in plain
    chained = source_policy({"draft_model": "embedded_dspark", "num_draft": 5,
                             "deepseek_verify_commit": "local_frontier",
                             "deepseek_verify_rows": "lockstep_multirow_v1",
                             "deepseek_dspark_verify_schedule": "survival_v1",
                             "deepseek_dspark_verify_costs": [1, 1.4, 1.8, 2.2, 2.6, 3.0]})
    assert chained["deepseek_verify_rows"] == "lockstep_multirow_v1"
    assert chained["deepseek_dspark_verify_costs"] == [1.0, 1.4, 1.8, 2.2, 2.6, 3.0]
    with pytest.raises(ValueError):
        source_policy({"deepseek_verify_rows": "lockstep_multirow_v1", "deepseek_verify_commit": "local_frontier"})
