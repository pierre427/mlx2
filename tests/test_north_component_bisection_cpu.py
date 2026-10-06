from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_harness():
    path = ROOT / "scripts/north_component_bisection.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


H = load_harness()


def fixture():
    rng = np.random.default_rng(17)
    routes = np.array(
        [[0, 2], [1, 3], [2, 0], [3, 1]], dtype=np.int32
    )
    route_weights = np.array(
        [[0.75, 0.25], [0.6, 0.4], [0.8, 0.2], [0.55, 0.45]],
        dtype=np.float32,
    )
    cache_k = rng.normal(size=(4, 3, 4)).astype(np.float32)
    cache_v = rng.normal(size=(4, 3, 4)).astype(np.float32)
    return {
        "x": rng.normal(size=(4, 4)).astype(np.float32),
        "q4_weight": rng.normal(size=(4, 6)).astype(np.float32),
        "q8_router_weight": rng.normal(size=(4, 4)).astype(np.float32),
        "expert_weight": rng.normal(size=(4, 4, 5)).astype(np.float32),
        "routes": routes,
        "route_weights": route_weights,
        "q": rng.normal(size=(4, 1, 4)).astype(np.float32),
        "cache_k": cache_k,
        "cache_v": cache_v,
        "new_v": rng.normal(size=(4, 1, 4)).astype(np.float32),
        "head_weight": rng.normal(size=(9, 4)).astype(np.float32),
    }


def evidence(stage, arm, payload):
    result = {
        "batched_rows": 4 if arm == "batched" else 0,
        "rowwise_calls": 0 if arm == "batched" else 4,
    }
    if stage == "q4_projection":
        result.update(bits=4, group_size=64)
    elif stage == "q8_router":
        result.update(bits=8, group_size=64)
    elif stage == "expert_gather_frozen":
        result.update(
            routes_frozen=True,
            route_ids_digest=H.tree_digest(payload["routes"]),
            route_weights_digest=H.tree_digest(payload["route_weights"]),
        )
    elif stage == "sdpa":
        result.update(
            cache_inputs_frozen=True,
            preappend_cache_digest=H.tree_digest(
                {"k": payload["cache_k"], "v": payload["cache_v"]}
            ),
        )
    elif stage == "tied_head":
        result.update(tied=True, bits=4, group_size=64)
    return result


def expert(payload):
    rows = []
    for lane in range(4):
        values = []
        for slot, expert_id in enumerate(payload["routes"][lane]):
            projected = payload["x"][lane] @ payload["expert_weight"][expert_id]
            values.append(projected * payload["route_weights"][lane, slot])
        rows.append(np.sum(values, axis=0))
    return np.stack(rows)


def sdpa(payload):
    rows = []
    for lane in range(4):
        scores = payload["q"][lane] @ payload["cache_k"][lane].T / 2.0
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        rows.append(probs @ payload["cache_v"][lane])
    state = {
        "k": np.concatenate([payload["cache_k"], payload["q"]], axis=1),
        "v": np.concatenate([payload["cache_v"], payload["new_v"]], axis=1),
    }
    return np.concatenate(rows, axis=0), state


def calculate(stage, payload):
    if stage == "q4_projection":
        return payload["x"] @ payload["q4_weight"]
    if stage == "q8_router":
        return payload["x"] @ payload["q8_router_weight"]
    if stage == "expert_gather_frozen":
        return expert(payload)
    if stage == "sdpa":
        return sdpa(payload)
    if stage == "tied_head":
        return payload["x"] @ payload["head_weight"].T
    raise AssertionError(stage)


def observe(stage, arm, *, mutate=None, evidence_mutation=None, input_mutation=False):
    def run(payload):
        value = calculate(stage, payload)
        if stage == "sdpa":
            output, state = value
        else:
            output, state = value, None
        if arm == "batched" and mutate == stage:
            output = output.copy()
            output.flat[0] += np.float32(0.25)
        if input_mutation:
            payload["x"][0, 0] += np.float32(1.0)
        detail = evidence(stage, arm, payload)
        if evidence_mutation:
            detail.update(evidence_mutation)
        return H.Observation(output=output, state=state, evidence=detail)

    return run


def complete_oracle(*, mutate=None):
    oracle = H.NorthComponentBisection()
    source = fixture()
    for stage in H.STAGES:
        oracle.observe(
            stage,
            source,
            observe(stage, "batched", mutate=mutate),
            observe(stage, "rowwise", mutate=mutate),
        )
    return oracle.receipt()


def test_complete_exact_oracle_is_width_invariant_and_preserves_source():
    receipt = complete_oracle()
    assert receipt["complete"] is True
    assert receipt["verdict"] == "width_invariant"
    assert receipt["width_sensitive_stages"] == []
    assert all(row["input_unchanged"] for row in receipt["stages"].values())
    assert all(row["independent_inputs"] for row in receipt["stages"].values())
    assert receipt["stages"]["sdpa"]["state"]["exact"] is True


@pytest.mark.parametrize("stage", H.STAGES)
def test_each_width_mutation_is_classified_at_its_component(stage):
    receipt = complete_oracle(mutate=stage)
    assert receipt["complete"] is True
    assert receipt["verdict"] == "width_sensitive"
    assert receipt["width_sensitive_stages"] == [stage]
    assert receipt["first_width_sensitive_stage"] == stage
    assert receipt["stages"][stage]["output"]["max_abs"] == pytest.approx(0.25)


def test_frozen_expert_routes_require_matching_ids_and_weights():
    oracle = H.NorthComponentBisection()
    source = fixture()
    oracle.observe(
        "expert_gather_frozen",
        source,
        observe("expert_gather_frozen", "batched"),
        observe(
            "expert_gather_frozen",
            "rowwise",
            evidence_mutation={"route_ids_digest": "not-the-same"},
        ),
    )
    stage = oracle.receipt()["stages"]["expert_gather_frozen"]
    assert stage["complete"] is False
    assert stage["exact"] is False
    assert stage["evidence_failures"] == ["matching.route_ids_digest"]


def test_sdpa_requires_matching_preappend_cache_and_poststate():
    source = fixture()
    oracle = H.NorthComponentBisection()
    oracle.observe(
        "sdpa",
        source,
        observe("sdpa", "batched"),
        observe(
            "sdpa",
            "rowwise",
            evidence_mutation={"preappend_cache_digest": "different"},
        ),
    )
    receipt = oracle.receipt()
    assert receipt["verdict"] == "incomplete"
    assert receipt["stages"]["sdpa"]["state"]["exact"] is True
    assert receipt["stages"]["sdpa"]["evidence_failures"] == [
        "matching.preappend_cache_digest"
    ]

    missing_state = H.NorthComponentBisection()

    def no_state(payload):
        output, _ = sdpa(payload)
        return H.Observation(
            output,
            evidence=evidence("sdpa", "batched", payload),
        )

    missing_state.observe(
        "sdpa",
        source,
        no_state,
        observe("sdpa", "rowwise"),
    )
    assert missing_state.receipt()["stages"]["sdpa"]["complete"] is False


def test_incomplete_evidence_never_publishes_a_sensitivity_claim():
    oracle = H.NorthComponentBisection()
    source = fixture()
    oracle.observe(
        "q4_projection",
        source,
        observe("q4_projection", "batched", mutate="q4_projection"),
        observe(
            "q4_projection",
            "rowwise",
            evidence_mutation={"group_size": 32},
        ),
    )
    receipt = oracle.receipt()
    assert receipt["verdict"] == "incomplete"
    assert receipt["width_sensitive_stages"] == []
    assert receipt["stages"]["q4_projection"]["output"]["exact"] is False
    assert "rowwise.group_size" in receipt["stages"]["q4_projection"][
        "evidence_failures"
    ]


def test_source_mutation_and_shared_clone_fail_closed():
    source = fixture()
    mutated = H.NorthComponentBisection()

    def mutates_source(payload):
        # Deliberately reaches the source rather than its private arm copy.
        source["x"][0, 0] += np.float32(1.0)
        return observe("q4_projection", "batched")(payload)

    mutated.observe(
        "q4_projection",
        source,
        mutates_source,
        observe("q4_projection", "rowwise"),
    )
    assert mutated.receipt()["stages"]["q4_projection"]["input_unchanged"] is False

    aliased = H.NorthComponentBisection(clone=lambda value: value)
    aliased.observe(
        "q4_projection",
        fixture(),
        observe("q4_projection", "batched"),
        observe("q4_projection", "rowwise"),
    )
    assert aliased.receipt()["stages"]["q4_projection"]["independent_inputs"] is False
    assert aliased.receipt()["verdict"] == "incomplete"


def test_errors_duplicates_and_missing_stages_fail_closed():
    oracle = H.NorthComponentBisection()
    source = fixture()

    def broken(_payload):
        raise RuntimeError("deliberate")

    stage = oracle.observe(
        "q4_projection",
        source,
        broken,
        observe("q4_projection", "rowwise"),
    )
    assert stage["error"] == "RuntimeError: deliberate"
    assert oracle.receipt()["verdict"] == "incomplete"
    with pytest.raises(RuntimeError, match="already observed"):
        oracle.observe(
            "q4_projection",
            source,
            broken,
            observe("q4_projection", "rowwise"),
        )
    with pytest.raises(ValueError, match="unknown"):
        H.NorthComponentBisection().observe("other", source, broken, broken)


def test_describe_is_directly_runnable_and_imports_no_mlx():
    script = ROOT / "scripts/north_component_bisection.py"
    result = subprocess.run(
        [sys.executable, str(script), "--describe"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["required_stages"] == list(H.STAGES)
    assert receipt["sdpa_requires_state_comparison"] is True
    assert receipt["imports_mlx"] is False
