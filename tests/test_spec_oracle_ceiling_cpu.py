import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spec_oracle_ceiling", ROOT / "scripts/spec_oracle_ceiling.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def test_flat_cost_oracle_equals_actual_and_deepest_fixed_is_best():
    # Flat cost: drafting deeper is free, so neither the oracle nor a retune gains.
    accepted, depths = {0: 2, 3: 1, 4: 1}, {4: 4}
    r = oracle.ceiling(accepted, depths, [1.0] * 5)
    assert r["tokens_per_cost"]["oracle"] == pytest.approx(r["tokens_per_cost"]["actual"])
    assert r["best_fixed_depth"] == 4
    assert r["opportunity_vs_actual"] == pytest.approx(0)
    assert r["full_accept_censored_rounds"] == 1 and r["oracle_is_lower_bound"]


def test_hand_computed_linear_case():
    # c(d) = 1 + d.  Rounds: a=0 (x3), a=2 (x1) at depth 2.
    accepted, depths = {0: 3, 2: 1}, {2: 4}
    costs = [1.0, 2.0, 3.0]
    r = oracle.ceiling(accepted, depths, costs)
    tokens = 3 * 1 + 1 * 3
    assert r["tokens_per_cost"]["actual"] == pytest.approx(tokens / (4 * 3.0))
    assert r["tokens_per_cost"]["oracle"] == pytest.approx(tokens / (3 * 1.0 + 1 * 3.0))
    # fixed d=0: 4 tokens / 4; d=1: (3*1 + 2) / 8; d=2: 6 / 12
    assert r["fixed_depth_tokens_per_cost"] == {"0": 1.0, "1": pytest.approx(5 / 8), "2": 0.5}
    assert r["best_fixed_depth"] == 0
    assert r["opportunity_vs_actual"] == pytest.approx(1.0)


def test_validation():
    with pytest.raises(ValueError, match="no speculation"):
        oracle.ceiling({}, {}, [1.0])
    with pytest.raises(ValueError, match="cover depths"):
        oracle.ceiling({1: 1}, {3: 1}, [1.0, 1.0])
    with pytest.raises(ValueError, match="different rounds"):
        oracle.ceiling({1: 2}, {2: 1}, [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        oracle.linear_costs(0.0, 1.0, 3)


def test_cli_linear_model_from_receipt(tmp_path, capsys):
    rows = [{"accept_hist": {"0": 2, "3": 2}, "depth_hist": {"3": 4}, "completion_tokens": 11,
             "decode_tps": 10.0}]
    ordinary = [{"completion_tokens": 21, "decode_tps": 20.0}]
    receipt = {"checkpoint": "m@sha256:1", "arms": [{"name": "spec", "rows": rows},
                                                     {"name": "ord", "rows": ordinary}]}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    assert oracle.main([str(path), "--arm", "spec", "--ordinary-arm", "ord"]) == 0
    out = json.loads(capsys.readouterr().out)["arms"]["spec"]
    # c0 = 1/20 s, cK = (10 tokens / 10 tok/s) / 4 rounds = 0.25 s
    assert out["measured_tok_s"] == pytest.approx(10.0)
    flat, linear = out["models"]["flat"], out["models"]["linear"]
    assert flat["oracle_is_lower_bound"] is True
    # Rounds: a=0 x2, a=3 x2.  Flat: oracle saves only the a=0 rounds (c0 not cK).
    assert flat["tokens_per_cost"]["oracle"] == pytest.approx(10 / (2 * 0.05 + 2 * 0.25))
    assert linear["tokens_per_cost"]["oracle"] >= flat["tokens_per_cost"]["oracle"] * (1 - 1e-12)


def test_flat_model_handles_mixed_depths(tmp_path, capsys):
    rows = [{"accept_hist": {"0": 1, "2": 1, "4": 2}, "depth_hist": {"2": 2, "5": 2},
             "completion_tokens": 13, "decode_tps": 12.0}]
    receipt = {"arms": [{"name": "adaptive", "rows": rows},
                        {"name": "ord", "rows": [{"completion_tokens": 11, "decode_tps": 20.0}]}]}
    path = tmp_path / "r.json"
    path.write_text(json.dumps(receipt))
    oracle.main([str(path), "--arm", "adaptive", "--ordinary-arm", "ord"])
    out = json.loads(capsys.readouterr().out)["arms"]["adaptive"]["models"]
    assert "tokens_per_cost" in out["flat"] and "error" in out["linear"]
