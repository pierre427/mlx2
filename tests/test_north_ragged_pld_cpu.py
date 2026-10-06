"""CPU regression for North's ragged ordinary/PLD final-cache boundary.

This uses a small random ``cohere2_moe`` body, not the North artifact, and
therefore qualifies nothing.  It pins two host-safe facts needed by the real
qualification run:

* the fixed tiny North fixture has the same greedy tokens at ordinary B1 and
  ordinary BN geometry; and
* prompt lookup deliberately returns a cache/history ending immediately
  before its final emitted token (the next anchor).  Resumption must feed that
  uncovered token and remain token-exact with ordinary decode.

The latter is a route convention, not an off-by-one defect: ``pld.py`` commits
verified inputs, while the last sampled token has not yet been forwarded.
"""

from __future__ import annotations

import pytest

from scripts import qualify_ragged_pld as Q


def _args():
    return Q.resolve_args(
        Q.build_parser(),
        [
            "--tiny",
            "--lanes",
            "2",
            "--logprob-rows",
            "2",
            "--continuation-tokens",
            "4",
            "--out",
            "/dev/null",
        ],
    )


def _tiny_north_model():
    import mlx.core as mx

    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    mx.random.seed(8)
    model = Model(
        ModelArgs(
            hidden_size=32,
            head_dim=8,
            num_hidden_layers=4,
            intermediate_size=32,
            prefix_dense_intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=48,
            sliding_window=8,
            max_position_embeddings=1024,
            num_experts=4,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            rms_norm_eps=1e-6,
        )
    )
    model.eval()
    mx.eval(model.parameters())
    return model


def _finalize(driver, lane, record):
    final = record["_final"]
    snapshot = {
        "tokens": list(record["tokens"]),
        "covered_tokens": record["covered_tokens"],
        "all_tokens": list(final.all_tokens),
        "finish_reason": record["finish_reason"],
    }
    snapshot["continuation"] = driver.continuation(lane, record)
    return snapshot


@pytest.fixture(scope="module")
def tiny_north_routes():
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Q, "tiny_model", _tiny_north_model)
        driver = Q.Driver(_args())
        driver.check_geometry()
        lanes = list(range(len(driver.prompts)))
        routes = {}

        ordinary_b1 = {}
        for lane in lanes:
            out, _stats, failures, _removed = driver.run("ordinary_b1", [lane])
            assert failures == []
            ordinary_b1[lane] = _finalize(driver, lane, out[lane])
        routes["ordinary_b1"] = ordinary_b1

        for arm in ("ordinary_bN", "pld_per_lane", "pld_batched"):
            out, _stats, failures, _removed = driver.run(arm, lanes)
            assert failures == []
            routes[arm] = {
                lane: _finalize(driver, lane, out[lane]) for lane in lanes
            }

    return {"prompts": driver.prompts, "routes": routes}


def test_tiny_north_ordinary_b1_and_bn_greedy_tokens_match(tiny_north_routes):
    routes = tiny_north_routes["routes"]
    for lane, reference in routes["ordinary_b1"].items():
        batched = routes["ordinary_bN"][lane]
        assert batched["tokens"] == reference["tokens"]
        assert batched["finish_reason"] == reference["finish_reason"] == "length"
        assert batched["continuation"]["status"] == "complete"
        assert batched["continuation"]["tokens"] == reference["continuation"]["tokens"]


@pytest.mark.parametrize("arm", ["pld_per_lane", "pld_batched"])
def test_pld_final_cache_stops_before_pending_anchor_and_resumes_exactly(
    tiny_north_routes, arm
):
    prompts, routes = tiny_north_routes["prompts"], tiny_north_routes["routes"]
    for lane, reference in routes["ordinary_b1"].items():
        speculative = routes[arm][lane]
        full = list(prompts[lane]) + speculative["tokens"]

        assert speculative["tokens"] == reference["tokens"]
        assert reference["all_tokens"] == full
        assert reference["covered_tokens"] == len(full)

        # PLD has forwarded and committed every token except the final sample,
        # which is the pending anchor for the next target forward.
        assert speculative["all_tokens"] == full[:-1]
        assert speculative["covered_tokens"] == len(full) - 1
        assert speculative["covered_tokens"] == reference["covered_tokens"] - 1

        resumed = speculative["continuation"]
        assert resumed["status"] == "complete"
        assert resumed["tokens"] == reference["continuation"]["tokens"]
